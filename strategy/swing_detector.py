from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ConfirmedSwingHigh:
    symbol: str
    swing_high_bar_idx: int          # Index T in 15m series
    swing_high_time: pd.Timestamp    # Open time of candle T
    swing_high_price: float          # High price at candle T
    confirmation_bar_idx: int        # Index T+N in 15m series
    confirmation_time: pd.Timestamp  # Open time of candle T+N (confirmed at close of T+N)
    earliest_action_time: pd.Timestamp # Open time of candle T+N+1
    fractal_n: int
    swing_low_bar_idx: int           # Index of lowest low in [T-L, T]
    swing_low_time: pd.Timestamp     # Open time of swing low candle
    swing_low_price: float           # Lowest low price
    impulse: float                   # swing_high_price - swing_low_price
    impulse_pct: float               # impulse / swing_low_price
    impulse_atr: float               # impulse / atr_ref
    atr_ref: float                   # Wilder ATR(14) frozen at confirmation time


class SwingDetector:
    """
    N/N Fractal Swing High Detector (DEFINITIONS.md §4 & §5).
    
    Rules:
    - Candle T is a swing high if high[T] > max(high[T-N..T-1]) AND high[T] > max(high[T+1..T+N]).
    - Strictly greater than both sides.
    - Confirmation: at the CLOSE of candle T+N. Earliest action: open of T+N+1.
    - Setup cannot exist or use ATR before close of T+N.
    - Preceding impulse: lowest low in [T-L, T], default L=32 bars (8h).
    - Impulse filter: impulse / ATR_ref >= min_impulse_atr (default 2.0).
    """

    def __init__(
        self,
        fractal_n: int = 3,
        lookback_l: int = 32,
        min_impulse_atr: float = 2.0
    ):
        self.fractal_n = fractal_n
        self.lookback_l = lookback_l
        self.min_impulse_atr = min_impulse_atr

    def find_swing_highs(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_15m: pd.Series
    ) -> List[ConfirmedSwingHigh]:
        n_bars = len(df_15m)
        N = self.fractal_n
        L = self.lookback_l

        if n_bars < (L + N + 1):
            return []

        # Vectorized N/N fractal detection
        high_series = df_15m["high"]
        low_series = df_15m["low"]

        max_before = high_series.shift(1).rolling(N).max()
        max_after = high_series.iloc[::-1].shift(1).rolling(N).max().iloc[::-1]
        is_swing = (high_series > max_before) & (high_series > max_after)

        # Preceding impulse: lowest low in [T - L, T]
        # Rolling min on low with window L + 1 gives min over [T-L..T]
        rolling_min_low = low_series.rolling(L + 1).min()

        candidate_indices = np.where(is_swing.values)[0]
        # Exclude candidates too close to boundaries
        candidate_indices = candidate_indices[(candidate_indices >= max(N, L)) & (candidate_indices < n_bars - N)]

        if len(candidate_indices) == 0:
            return []

        atrs_arr = atr_15m.values
        lows_arr = low_series.values
        highs_arr = high_series.values

        swings = []
        for T in candidate_indices:
            conf_idx = T + N
            atr_ref = atrs_arr[conf_idx]

            if np.isnan(atr_ref) or atr_ref <= 0:
                continue

            h_T = highs_arr[T]
            # Find swing low in [T-L..T]
            window_lows = lows_arr[T - L: T + 1]
            min_offset = int(np.argmin(window_lows))
            swing_low_idx = (T - L) + min_offset
            swing_low_px = float(window_lows[min_offset])

            impulse = h_T - swing_low_px
            if impulse <= 0:
                continue

            impulse_atr = impulse / atr_ref
            if impulse_atr < self.min_impulse_atr:
                continue

            impulse_pct = impulse / swing_low_px

            t_high_time = df_15m["open_time"].iloc[T]
            t_conf_time = df_15m["open_time"].iloc[conf_idx]
            t_action_time = df_15m["open_time"].iloc[conf_idx + 1] if conf_idx + 1 < n_bars else t_conf_time + pd.Timedelta(minutes=15)
            swing_low_time = df_15m["open_time"].iloc[swing_low_idx]

            swings.append(ConfirmedSwingHigh(
                symbol=symbol,
                swing_high_bar_idx=int(T),
                swing_high_time=t_high_time,
                swing_high_price=float(h_T),
                confirmation_bar_idx=int(conf_idx),
                confirmation_time=t_conf_time,
                earliest_action_time=t_action_time,
                fractal_n=N,
                swing_low_bar_idx=int(swing_low_idx),
                swing_low_time=swing_low_time,
                swing_low_price=swing_low_px,
                impulse=float(impulse),
                impulse_pct=float(impulse_pct),
                impulse_atr=float(impulse_atr),
                atr_ref=float(atr_ref)
            ))

        return swings
