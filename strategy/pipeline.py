from pathlib import Path
import logging
from typing import List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.config import AppConfig, load_config
from data.cache import ParquetCache
from strategy.indicators import compute_wilder_atr
from strategy.alignment import align_1h_to_15m
from strategy.trend_detector import TrendDetector
from strategy.swing_detector import SwingDetector, ConfirmedSwingHigh
from strategy.pullback_detector import PullbackDetector, ThresholdCrossingEvent
from strategy.universe import PointInTimeUniverse
from strategy.btc_regime import BTCRegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


class StrategyPipeline:
    """
    High-performance pipeline for Phase 2:
    - Pre-groups universe rankings by symbol for instant lookup
    - Vectorized 1H trends & 15m alignment
    - Vectorized running low accumulation
    - Partitioned streaming disk writes to eliminate memory overhead
    """

    def __init__(self, config: AppConfig, cache: Optional[ParquetCache] = None):
        self.config = config
        self.cache = cache or ParquetCache(config.data.cache_dir)
        self.trend_detector = TrendDetector()
        self.pullback_detector = PullbackDetector()
        self.universe_engine = PointInTimeUniverse(config, self.cache)
        self.btc_engine = BTCRegimeDetector(self.cache)
        self.results_dir = Path(config.data.results_dir)
        self.setups_partition_dir = self.results_dir / "setups_partitions"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.setups_partition_dir.mkdir(parents=True, exist_ok=True)

    def process_symbol(
        self,
        symbol: str,
        sym_univ_df: Optional[pd.DataFrame],
        fractal_n: int = 3
    ) -> List[ThresholdCrossingEvent]:
        df_15m = self.cache.load_klines(symbol, "15m")
        df_1h = self.cache.load_klines(symbol, "1h")

        if df_15m.empty or df_1h.empty:
            return []

        # 1. 1H Trends
        df_1h_trends = self.trend_detector.compute_1h_trends(df_1h)

        # 2. Causal 1H -> 15m Alignment
        trend_cols = ["ema50", "ema200", "trend_a", "trend_b", "trend_c", "trend_d", "trend_invalidated"]
        df_15m_aligned = align_1h_to_15m(df_15m, df_1h_trends, trend_cols)

        # 3. 15m Causal ATR
        atr_15m = compute_wilder_atr(
            df_15m_aligned["high"],
            df_15m_aligned["low"],
            df_15m_aligned["close"],
            period=14
        )

        # 4. Attach Universe Rank
        if sym_univ_df is not None and not sym_univ_df.empty:
            univ_copy = sym_univ_df[["timestamp", "rank", "eligible", "in_universe"]].copy()
            univ_copy["timestamp"] = univ_copy["timestamp"].astype(df_15m_aligned["open_time"].dtype)
            df_15m_aligned = pd.merge_asof(
                df_15m_aligned.sort_values(by="open_time"),
                univ_copy.sort_values(by="timestamp"),
                left_on="open_time",
                right_on="timestamp",
                direction="backward"
            )
        else:
            df_15m_aligned["rank"] = None

        # 5. Detect Confirmed Swing Highs
        swing_detector = SwingDetector(fractal_n=fractal_n, lookback_l=32, min_impulse_atr=2.0)
        swings = swing_detector.find_swing_highs(symbol, df_15m_aligned, atr_15m)

        if not swings:
            return []

        # 6. Detect Pullback Events
        trend_flags = df_15m_aligned[["trend_a", "trend_b", "trend_c", "trend_d", "trend_invalidated"]]
        ranks = df_15m_aligned["rank"] if "rank" in df_15m_aligned.columns else None
        ctx = self.pullback_detector.extract_context(
            df_15m=df_15m_aligned,
            atr_15m=atr_15m,
            trend_flags=trend_flags,
            universe_ranks=ranks
        )

        all_events = []
        for swing in swings:
            evs = self.pullback_detector.process_swing_high(
                swing=swing,
                df_15m=df_15m_aligned,
                atr_15m=atr_15m,
                trend_flags=trend_flags,
                universe_ranks=ranks,
                ctx=ctx
            )
            all_events.extend(evs)

        return all_events

    def run(
        self,
        symbols: Optional[List[str]] = None,
        fractal_n_list: Optional[List[int]] = None
    ) -> pd.DataFrame:
        if fractal_n_list is None:
            fractal_n_list = [3]  # Default fractal

        if symbols is None:
            symbols = self.cache.list_cached_symbols("15m")

        logger.info(f"Running pipeline for {len(symbols)} symbols across fractals {fractal_n_list}...")

        # 1. Build / Load Universe Table & Pre-group by symbol
        universe_path = Path(self.config.data.cache_dir) / "universe.parquet"
        if universe_path.is_file():
            universe_df = pd.read_parquet(universe_path)
        else:
            universe_df = self.universe_engine.save_universe_table(universe_path)

        univ_by_symbol = dict(tuple(universe_df.groupby("symbol"))) if not universe_df.empty else {}

        # Stream write each symbol's events directly to partitioned parquet files
        total_events = 0
        written_files = []
        dt_cols = ["swing_high_time", "confirmation_time", "swing_low_time", "event_time", "event_close_time", "action_open_time"]

        for n in fractal_n_list:
            n_dir = self.setups_partition_dir / f"fractal_n={n}"
            n_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting fractal size N={n}...")

            for idx, sym in enumerate(symbols):
                part_file = n_dir / f"{sym}.parquet"
                if part_file.is_file() and part_file.stat().st_size > 0:
                    written_files.append(part_file)
                    continue

                sym_univ = univ_by_symbol.get(sym)
                evs = self.process_symbol(sym, sym_univ, fractal_n=n)
                if evs:
                    records = [e.__dict__ for e in evs]
                    df_sym = pd.DataFrame(records)
                    df_sym["fractal_n"] = n
                    df_sym["config_hash"] = self.config.config_hash
                    df_sym["universe_rank"] = pd.to_numeric(df_sym["universe_rank"], errors="coerce")
                    for col in dt_cols:
                        if col in df_sym.columns:
                            df_sym[col] = pd.to_datetime(df_sym[col], utc=True).astype("datetime64[ms, UTC]")

                    df_sym.to_parquet(part_file, index=False)
                    written_files.append(part_file)
                    total_events += len(df_sym)

                if (idx + 1) % 50 == 0 or (idx + 1) == len(symbols):
                    logger.info(f"  [N={n}] Processed {idx + 1}/{len(symbols)} symbols")

        logger.info(f"Combining partition files into unified results/setups.parquet...")
        import pyarrow.dataset as ds
        import pyarrow as pa
        part = ds.partitioning(pa.schema([pa.field("fractal_n", pa.int64())]), flavor="hive")
        dataset = ds.dataset(self.setups_partition_dir, partitioning=part)
        
        out_path = self.results_dir / "setups.parquet"
        total_rows = dataset.count_rows()
        logger.info(f"Streaming {total_rows:,} setup events to {out_path}...")
        with pq.ParquetWriter(out_path, schema=dataset.schema, compression="snappy") as writer:
            for batch in dataset.to_batches(batch_size=500_000):
                writer.write_batch(batch)
        logger.info(f"Successfully saved {total_rows:,} setup events to {out_path}.")

        # Convert sample/summary to pandas for reporting
        summary_cols = ["symbol", "swing_high_time", "event_time", "fractal_n", "threshold_type", "threshold_value", "trend_a", "trend_b", "trend_c", "trend_d", "is_controlled_pullback"]
        df_summary = dataset.to_table(columns=summary_cols).to_pandas()
        self.print_summary_breakdown(df_summary)
        return df_summary

    def print_summary_breakdown(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("PHASE 2 SETUP EVENTS BREAKDOWN")
        print("=" * 60)
        print(f"Total Events Found: {len(df):,}")
        print(f"Unique Symbols: {df['symbol'].nunique():,}")
        print(f"Unique Swing Highs: {df['swing_high_time'].nunique():,}")
        print(f"Controlled Pullback Events (Trend C): {df['is_controlled_pullback'].sum():,} ({df['is_controlled_pullback'].mean()*100:.1f}%)")

        print("\n--- By Fractal Size (N) ---")
        print(df["fractal_n"].value_counts().to_string())

        print("\n--- By Threshold Type & Value ---")
        print(df.groupby(["threshold_type", "threshold_value"]).size().to_string())

        print("\n--- By 1H Trend Definition ---")
        print(f"Trend A (close > EMA50, rising): {df['trend_a'].sum():,}")
        print(f"Trend B (close > EMA50 > EMA200): {df['trend_b'].sum():,}")
        print(f"Trend C (DEFAULT - close > EMA50 > EMA200, rising): {df['trend_c'].sum():,}")
        print(f"Trend D (Supertrend Bull): {df['trend_d'].sum():,}")

        print("\n--- By Month ---")
        df["month"] = pd.to_datetime(df["event_time"]).dt.to_period("M")
        print(df["month"].value_counts().sort_index().to_string())

        print("\n--- Top 15 Symbols by Event Count ---")
        print(df["symbol"].value_counts().head(15).to_string())
        print("=" * 60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Feature & Setup Extractor")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fractals", nargs="+", type=int, default=[2, 3, 4, 5], help="Fractal N values (e.g. 2 3 4 5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = StrategyPipeline(cfg)
    pipeline.run(fractal_n_list=args.fractals)


if __name__ == "__main__":
    main()
