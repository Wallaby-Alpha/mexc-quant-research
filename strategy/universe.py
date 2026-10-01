from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
from typing import List, Optional, Set
import pandas as pd
import numpy as np

from data.config import AppConfig, load_config
from data.cache import ParquetCache

logger = logging.getLogger("universe")


class PointInTimeUniverse:
    """
    Point-in-Time Universe Ranker & Filter (DEFINITIONS.md §12).
    
    Rules:
    - Evaluated each hour on 1H close.
    - For each symbol, computes trailing 24h quote volume in USDT using ONLY
      candles closed at or before that hour.
    - Ranks eligible symbols descending by volume.
    - Ranks apply to the following hour: for an hour ending at T (e.g. 11:00 UTC),
      its rank governs all 15m bars in [11:00, 12:00).
    - Eligibility:
      1. Has >= 250 closed 1H bars of history at decision time.
      2. Trailing 24h quote volume >= min_volume_usdt (default $1,000,000).
      3. Not in excluded_symbols list.
    """

    def __init__(self, config: AppConfig, cache: Optional[ParquetCache] = None):
        self.config = config
        self.cache = cache or ParquetCache(config.data.cache_dir)
        self.min_volume_usdt = config.universe.min_volume_usdt
        self.min_history_bars = config.universe.min_history_1h_bars
        self.target_size = config.universe.target_size
        self.always_include = set(config.universe.always_include)
        self.excluded_symbols = set(config.universe.excluded_symbols)

    def build_universe_table(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Builds the hourly point-in-time universe table across all cached symbols.
        Returns DataFrame with columns:
        [timestamp, symbol, rank, volume_24h_usdt, eligible, in_universe]
        """
        if symbols is None:
            symbols = self.cache.list_cached_symbols("1h")

        if not symbols:
            logger.warning("No 1H cached symbols available to build universe.")
            return pd.DataFrame()

        logger.info(f"Building point-in-time universe across {len(symbols)} symbols...")

        # Load 1H quote_volume_usdt for each symbol
        symbol_series = {}
        for sym in symbols:
            df = self.cache.load_klines(sym, "1h")
            if df.empty:
                continue
            # Ensure open_time is datetime UTC
            df = df.set_index("open_time").sort_index()
            # 1H candle open_time + 1 hour = close_time (known time)
            # Trailing 24h volume on closed bars: rolling sum of 24 bars
            vol_24h = df["quote_volume_usdt"].rolling(24, min_periods=24).sum()
            # Cumulative bar count at each closed bar
            cum_bars = pd.Series(np.arange(1, len(df) + 1), index=df.index)
            symbol_series[sym] = pd.DataFrame({
                "quote_vol_24h": vol_24h,
                "cum_bars": cum_bars
            })

        if not symbol_series:
            return pd.DataFrame()

        # Combine into multi-symbol panel aligned by closed timestamp
        # In MEXC 1H candles, open_time = 10:00 means the bar closes at 11:00.
        # Decision time = open_time + 1 hour.
        all_dfs = []
        for sym, df_sym in symbol_series.items():
            df_sym["symbol"] = sym
            # Availability timestamp: when the 1H candle closed
            df_sym["decision_time"] = df_sym.index + pd.Timedelta(hours=1)
            all_dfs.append(df_sym)

        combined = pd.concat(all_dfs, ignore_index=True)

        # Filter out NaN volumes (first 24 hours of history)
        combined = combined.dropna(subset=["quote_vol_24h"])

        # Determine eligibility at decision_time
        # 1. Has >= min_history_bars closed bars
        # 2. Volume >= min_volume_usdt OR in always_include
        # 3. Not in excluded_symbols
        is_not_excluded = ~combined["symbol"].isin(self.excluded_symbols)
        has_min_bars = combined["cum_bars"] >= self.min_history_bars
        has_min_vol = combined["quote_vol_24h"] >= self.min_volume_usdt
        is_always_include = combined["symbol"].isin(self.always_include)

        combined["eligible"] = is_not_excluded & has_min_bars & (has_min_vol | is_always_include)

        # Rank hourly among eligible symbols by quote_vol_24h descending
        # Non-eligible symbols receive rank = NaN
        eligible_subset = combined[combined["eligible"]].copy()
        eligible_subset["rank"] = eligible_subset.groupby("decision_time")["quote_vol_24h"].rank(
            ascending=False, method="min"
        ).astype(int)

        # Merge rank back
        combined = combined.merge(
            eligible_subset[["decision_time", "symbol", "rank"]],
            on=["decision_time", "symbol"],
            how="left"
        )

        # in_universe: rank <= target_size (e.g. top 200)
        combined["in_universe"] = combined["rank"] <= self.target_size

        # Format output
        out = combined[[
            "decision_time", "symbol", "rank", "quote_vol_24h", "eligible", "in_universe"
        ]].rename(columns={"decision_time": "timestamp", "quote_vol_24h": "volume_24h_usdt"})

        out = out.sort_values(by=["timestamp", "rank"]).reset_index(drop=True)
        return out

    def save_universe_table(self, out_path: str | Path = "data_cache/universe.parquet") -> pd.DataFrame:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.build_universe_table()
        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved point-in-time universe table to {out_path} ({len(df):,} rows).")
        return df

    def get_rank_at_time(self, universe_df: pd.DataFrame, symbol: str, dt: pd.Timestamp) -> Optional[int]:
        """
        Retrieves the effective universe rank for a symbol at decision time `dt`.
        Causal rule: Uses the most recent hourly ranking closed at or before `dt`.
        """
        if universe_df.empty:
            return None
        sub = universe_df[(universe_df["symbol"] == symbol) & (universe_df["timestamp"] <= dt)]
        if sub.empty:
            return None
        last_row = sub.iloc[-1]
        rank_val = last_row["rank"]
        return int(rank_val) if pd.notna(rank_val) else None
