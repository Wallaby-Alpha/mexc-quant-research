"""
strategy/absorption.py
Implements Volume Spread Absorption (VSA) detectors on 4H candles.
Identifies institutional absorption at deep pullback zones:
1. Volume Surge: Volume >= mult * SMA20(Volume)
2. Lower Wick Rejection: (min(open, close) - low) / (high - low) >= wick_ratio
3. Constructive Close: close in top 50% of the bar.
Strictly causal, trailing closed data only.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AbsorptionCandle:
    bar_idx: int
    open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    vol_sma: float
    vol_ratio: float
    lower_wick_ratio: float
    is_valid_absorption: bool


class VolumeAbsorptionDetector:
    """
    Detects candlestick volume absorption signatures at support/pullback zones.
    """

    def __init__(
        self,
        vol_sma_period: int = 20,
        min_vol_mult: float = 1.5,
        min_wick_ratio: float = 0.40,
        require_green_or_top_close: bool = True
    ):
        self.vol_sma_period = vol_sma_period
        self.min_vol_mult = min_vol_mult
        self.min_wick_ratio = min_wick_ratio
        self.require_top_close = require_green_or_top_close

    def detect_absorption_bars(self, df_4h: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean series indicating whether each bar is a confirmed absorption candle.
        """
        if len(df_4h) < self.vol_sma_period + 5:
            return pd.Series(False, index=df_4h.index)

        opens = df_4h["open"].to_numpy(dtype=float)
        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        volumes = df_4h["quote_volume_usdt"].to_numpy(dtype=float)

        vol_series = df_4h["quote_volume_usdt"]
        vol_sma = vol_series.rolling(self.vol_sma_period, min_periods=self.vol_sma_period).mean().to_numpy(dtype=float)

        n = len(df_4h)
        is_absorption = np.zeros(n, dtype=bool)

        for i in range(self.vol_sma_period, n):
            bar_range = highs[i] - lows[i]
            if bar_range <= 0 or np.isnan(vol_sma[i]) or vol_sma[i] <= 0:
                continue

            vol_mult = volumes[i] / vol_sma[i]
            if vol_mult < self.min_vol_mult:
                continue

            # Lower wick = distance from low to candle body bottom
            body_bottom = min(opens[i], closes[i])
            lower_wick = body_bottom - lows[i]
            wick_ratio = lower_wick / bar_range

            if wick_ratio < self.min_wick_ratio:
                continue

            # Close in upper 50% of the bar
            close_position = (closes[i] - lows[i]) / bar_range
            if self.require_top_close and close_position < 0.45:
                continue

            is_absorption[i] = True

        return pd.Series(is_absorption, index=df_4h.index, name="is_absorption")
