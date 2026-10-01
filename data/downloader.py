from datetime import datetime, timezone
from pathlib import Path
import time
import logging
from typing import List, Optional
import pandas as pd

from data.config import AppConfig, load_config
from data.mexc_client import MexcClient, StandardKline
from data.cache import ParquetCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downloader")


class HistoricalDownloader:
    """
    Incremental, resilient historical data downloader.
    - Never re-downloads existing data.
    - Paginates backward on initial ingestion until start_date / API limit.
    - Incrementally fetches only new closed bars on subsequent runs.
    - Safely resumes after interruptions.
    """

    def __init__(self, config: AppConfig, client: Optional[MexcClient] = None, cache: Optional[ParquetCache] = None):
        self.config = config
        self.client = client or MexcClient(config)
        self.cache = cache or ParquetCache(config.data.cache_dir)

    def _timeframe_to_seconds(self, tf: str) -> int:
        if tf == "15m":
            return 15 * 60
        elif tf == "1h":
            return 60 * 60
        elif tf == "4h":
            return 4 * 60 * 60
        elif tf == "1d":
            return 24 * 60 * 60
        raise ValueError(f"Unknown timeframe: {tf}")

    def download_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        start_time_sec: int,
        end_time_sec: int
    ) -> int:
        """
        Incrementally downloads candles for a symbol and timeframe.
        Returns the number of new candles fetched and written.
        """
        tf_secs = self._timeframe_to_seconds(timeframe)
        earliest_cached_ms, latest_cached_ms = self.cache.get_kline_time_bounds(symbol, timeframe)

        # Case 1: Cache is already up to date with target end_time
        if latest_cached_ms is not None:
            latest_cached_sec = latest_cached_ms // 1000
            # If the latest cached bar is within one candle period of the end time, no-op
            if latest_cached_sec >= (end_time_sec - tf_secs):
                logger.debug(f"{symbol} ({timeframe}): Cache is already up to date ({latest_cached_sec} >= {end_time_sec - tf_secs}). Skipping.")
                return 0

        collected_klines: List[StandardKline] = []

        if latest_cached_ms is None:
            # Cold initial download: Paginate backward from end_time_sec down to start_time_sec
            curr_end = end_time_sec
            logger.info(f"{symbol} ({timeframe}): Starting cold initial download from {datetime.fromtimestamp(curr_end, tz=timezone.utc)} backward...")
            
            last_earliest = None
            while curr_end > start_time_sec:
                batch = self.client.get_klines_batch(
                    symbol=symbol,
                    timeframe=timeframe,
                    end_time_sec=curr_end,
                    limit=2000
                )
                if not batch:
                    logger.debug(f"{symbol} ({timeframe}): Empty batch received at end={curr_end}. Reached boundary.")
                    break

                batch_earliest_sec = batch[0].open_time_ms // 1000
                batch_latest_sec = batch[-1].open_time_ms // 1000

                if last_earliest is not None and batch_earliest_sec >= last_earliest:
                    # No backward progress
                    break
                last_earliest = batch_earliest_sec

                collected_klines.extend(batch)
                curr_end = batch_earliest_sec - 1

                if batch_earliest_sec <= start_time_sec:
                    break
        else:
            # Incremental forward update: Fetch from latest_cached_sec + tf_secs to end_time_sec
            fetch_start_sec = (latest_cached_ms // 1000) + tf_secs
            logger.info(f"{symbol} ({timeframe}): Cache exists up to {datetime.fromtimestamp(latest_cached_ms//1000, tz=timezone.utc)}. Fetching incremental bars to {datetime.fromtimestamp(end_time_sec, tz=timezone.utc)}...")
            
            # For forward fetch, futures API returns latest candles when end is passed
            # Or pass start=fetch_start_sec
            curr_end = end_time_sec
            while curr_end >= fetch_start_sec:
                batch = self.client.get_klines_batch(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time_sec=fetch_start_sec,
                    end_time_sec=curr_end,
                    limit=2000
                )
                if not batch:
                    break

                new_bars = [k for k in batch if (k.open_time_ms // 1000) >= fetch_start_sec]
                collected_klines.extend(new_bars)
                break # Forward batch generally captures up to 2000 bars (more than enough for daily incremental updates)

        if not collected_klines:
            return 0

        # Convert collected klines to DataFrame and save to cache
        records = [{
            "open_time": pd.to_datetime(k.open_time_ms, unit="ms", utc=True),
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume_base": k.volume_base,
            "quote_volume_usdt": k.quote_volume_usdt,
            "source_market": k.source_market
        } for k in collected_klines]

        df = pd.DataFrame(records)
        self.cache.save_klines(symbol, timeframe, df)
        logger.info(f"{symbol} ({timeframe}): Saved {len(df)} bars to cache.")
        return len(df)

    def download_funding_history(self, symbol: str) -> int:
        """Downloads funding rate history for futures symbol."""
        if self.config.market.venue.lower() != "futures":
            return 0

        cached_df = self.cache.load_funding(symbol)
        latest_settle_ms = int(cached_df["settle_time"].max().timestamp() * 1000) if not cached_df.empty else 0

        all_records = []
        page = 1
        while page <= 10:  # 10 pages * 100 = 1000 funding intervals (over 330 days of 8h funding)
            records = self.client.get_funding_history(symbol=symbol, page_num=page, page_size=100)
            if not records:
                break
            
            new_records = [r for r in records if r.settle_time_ms > latest_settle_ms]
            all_records.extend(new_records)
            
            # If some records are older than latest_settle_ms, we've caught up
            if len(new_records) < len(records) or records[-1].settle_time_ms <= latest_settle_ms:
                break
            page += 1

        if not all_records:
            return 0

        df_records = [{
            "settle_time": pd.to_datetime(r.settle_time_ms, unit="ms", utc=True),
            "symbol": r.symbol,
            "funding_rate": r.funding_rate,
            "collect_cycle_hours": r.collect_cycle_hours
        } for r in all_records]

        df = pd.DataFrame(df_records)
        self.cache.save_funding(symbol, df)
        logger.info(f"{symbol}: Saved {len(df)} funding history records.")
        return len(df)

    def get_eligible_symbols(self, max_count: Optional[int] = None) -> List[str]:
        """
        Gets list of active symbols sorted by volume, excluding non-tradables and stablecoins,
        and unconditionally including BTC and ETH.
        """
        tickers = self.client.get_active_tickers()
        always_include = self.config.universe.always_include
        excluded = set(self.config.universe.excluded_symbols)

        eligible = []
        # Always include primary benchmarks first
        for btc_eth in always_include:
            eligible.append(btc_eth)

        for t in tickers:
            sym = t.symbol
            if sym in eligible or sym in excluded:
                continue
            if t.volume_24h_usdt < self.config.universe.min_volume_usdt:
                continue
            eligible.append(sym)

        if max_count:
            return eligible[:max_count]
        return eligible[:self.config.universe.target_size]

    def run(self, sample_size: Optional[int] = None) -> dict:
        """
        Main runner: downloads all timeframes and funding for selected symbols.
        """
        start_dt = datetime.fromisoformat(self.config.data.start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(self.config.data.end_date.replace("Z", "+00:00"))
        start_sec = int(start_dt.timestamp())
        end_sec = int(end_dt.timestamp())

        symbols = self.get_eligible_symbols(max_count=sample_size)
        logger.info(f"Downloader running for {len(symbols)} symbols. Target date range: {start_dt} to {end_dt}")

        total_new_bars = 0
        symbol_stats = {}

        for idx, sym in enumerate(symbols):
            logger.info(f"[{idx+1}/{len(symbols)}] Processing {sym}...")
            sym_bars = 0
            for tf in self.config.data.timeframes:
                new_bars = self.download_symbol_timeframe(
                    symbol=sym,
                    timeframe=tf,
                    start_time_sec=start_sec,
                    end_time_sec=end_sec
                )
                sym_bars += new_bars

            # Funding history
            if self.config.market.venue.lower() == "futures":
                self.download_funding_history(sym)

            symbol_stats[sym] = sym_bars
            total_new_bars += sym_bars

        stats = self.cache.get_cache_stats()
        stats["total_new_bars_downloaded"] = total_new_bars
        stats["symbols_processed"] = len(symbols)
        logger.info(f"Download complete. Total new bars: {total_new_bars}. Cache size: {stats['total_size_mb']:.2f} MB")
        return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MEXC Historical Data Downloader")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--sample", type=int, default=None, help="Number of symbols to sample (e.g. 20 for test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    downloader = HistoricalDownloader(cfg)
    downloader.run(sample_size=args.sample)


if __name__ == "__main__":
    main()
