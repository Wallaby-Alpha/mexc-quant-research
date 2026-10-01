"""
data/resampler.py
Causal Timeframe Resampler.
Aggregates strictly closed 1H klines into 4H and 1D bars with exact UTC alignment.
Anti-lookahead rule: A 4H bar with open_time T is fully closed and known only at T + 4h.
"""

from typing import Optional
import pandas as pd
import numpy as np


class KlineResampler:
    """
    Resamples 1H Kline dataframes into higher timeframes (4H, 1D).
    """

    @staticmethod
    def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples 1H klines into 4H bars (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).
        Only completed 4-bar clusters are emitted.
        """
        if df_1h.empty:
            return pd.DataFrame()

        df = df_1h.sort_values("open_time").copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume_base": "sum",
            "quote_volume_usdt": "sum"
        }
        # Keep any rank or metadata if present
        for col in ["rank", "source_market"]:
            if col in df.columns:
                agg_dict[col] = "last"

        # Resample with 4h binning, label left, closed left
        df_4h = df.resample("4h", closed="left", label="left").agg(agg_dict)

        # Count 1H bars per 4H group to ensure completeness (must have 4 1H bars)
        counts = df["close"].resample("4h", closed="left", label="left").count()
        df_4h = df_4h[counts == 4].dropna(subset=["close"]).reset_index()

        return df_4h

    @staticmethod
    def resample_1h_to_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples 1H klines into Daily (1D) bars (00:00 UTC).
        Requires all 24 1H bars to be present for a completed day.
        """
        if df_1h.empty:
            return pd.DataFrame()

        df = df_1h.sort_values("open_time").copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume_base": "sum",
            "quote_volume_usdt": "sum"
        }

        df_1d = df.resample("1D", closed="left", label="left").agg(agg_dict)
        counts = df["close"].resample("1D", closed="left", label="left").count()
        df_1d = df_1d[counts >= 23].dropna(subset=["close"]).reset_index()

        return df_1d
