from typing import Optional
import pandas as pd
import numpy as np

from strategy.indicators import compute_ema, compute_supertrend


class TrendDetector:
    """
    1H Trend Detector (DEFINITIONS.md §3 & §9).
    
    Evaluates:
    - Trend A: close > EMA50 AND EMA50 rising
    - Trend B: close > EMA50 > EMA200
    - Trend C (DEFAULT): close > EMA50 AND EMA50 > EMA200 AND EMA50 rising
    - Trend D: Supertrend(period=10, mult=3.0) direction == +1 (up)
    
    Parameters:
    - k: lookback for EMA50 rising (default k=3 closed 1H bars)
    - min_warmup_bars: minimum closed 1H bars (default 200)
    """

    def __init__(self, k: int = 3, min_warmup_bars: int = 200):
        self.k = k
        self.min_warmup_bars = min_warmup_bars

    def compute_1h_trends(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Computes 1H trend features on closed 1H bars.
        Input df_1h must have columns: open_time, open, high, low, close.
        Returns copy of df_1h with added trend columns.
        """
        if df_1h.empty:
            return df_1h.copy()

        df = df_1h.sort_values(by="open_time").copy()

        # Indicators
        df["ema50"] = compute_ema(df["close"], span=50)
        df["ema200"] = compute_ema(df["close"], span=200)
        _, st_dir = compute_supertrend(df["high"], df["low"], df["close"], period=10, multiplier=3.0)
        df["supertrend_dir"] = st_dir

        # EMA50 rising condition: EMA50[t] > EMA50[t-k]
        ema50_lagged = df["ema50"].shift(self.k)
        df["ema50_rising"] = df["ema50"] > ema50_lagged

        # Cumulative bar count for warm-up check
        bar_count = np.arange(1, len(df) + 1)
        valid_warmup = bar_count >= self.min_warmup_bars

        # Trend evaluations
        close = df["close"]
        ema50 = df["ema50"]
        ema200 = df["ema200"]
        rising = df["ema50_rising"]
        st_up = df["supertrend_dir"] == 1

        trend_a = valid_warmup & (close > ema50) & rising
        trend_b = valid_warmup & (close > ema50) & (ema50 > ema200)
        trend_c = valid_warmup & (close > ema50) & (ema50 > ema200) & rising
        trend_d = valid_warmup & st_up

        df["trend_a"] = trend_a
        df["trend_b"] = trend_b
        df["trend_c"] = trend_c  # DEFAULT
        df["trend_d"] = trend_d
        df["trend_valid_default"] = trend_c

        # Invalidation flag: closed 1H bar closes below EMA50
        df["trend_invalidated"] = close < ema50

        return df
