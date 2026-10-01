from typing import Tuple
import pandas as pd
import numpy as np


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Computes causal True Range (TR).
    TR = max(H - L, |H - C_prev|, |L - C_prev|)
    First bar has TR = H - L.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def compute_wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder ATR(14) using exact causal recursive smoothing:
    ATR[t] = (ATR[t-1] * (period - 1) + TR[t]) / period
    Initial seed at index (period - 1) is the simple moving average of the first 'period' TR bars.
    Prior bars are NaN.
    """
    tr = compute_true_range(high, low, close)
    n = len(tr)
    atr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return pd.Series(atr, index=high.index, name=f"atr_{period}")

    # Seed with SMA of first 'period' bars
    initial_atr = tr.iloc[:period].mean()
    atr[period - 1] = initial_atr

    # Recursive Wilder smoothing
    tr_vals = tr.values
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr_vals[i] * alpha

    return pd.Series(atr, index=high.index, name=f"atr_{period}")


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Causal Exponential Moving Average via recursive form (adjust=False).
    EMA[t] = alpha * series[t] + (1 - alpha) * EMA[t-1]
    where alpha = 2 / (span + 1).
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Causal Supertrend indicator.
    Returns:
    (supertrend_line, direction)
    where direction is +1 (bullish/up) or -1 (bearish/down).
    """
    atr = compute_wilder_atr(high, low, close, period=period)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(close)
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, 1, dtype=np.int32)
    st = np.full(n, np.nan, dtype=np.float64)

    # Find first valid ATR index
    first_valid = period - 1
    if n <= first_valid:
        return (
            pd.Series(st, index=close.index, name="supertrend"),
            pd.Series(direction, index=close.index, name="supertrend_dir")
        )

    final_upper[first_valid] = basic_upper.iloc[first_valid]
    final_lower[first_valid] = basic_lower.iloc[first_valid]
    direction[first_valid] = 1 if close.iloc[first_valid] >= final_lower[first_valid] else -1
    st[first_valid] = final_lower[first_valid] if direction[first_valid] == 1 else final_upper[first_valid]

    c = close.values
    bu = basic_upper.values
    bl = basic_lower.values

    for i in range(first_valid + 1, n):
        # Final Upper Band
        if bu[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band
        if bl[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Trend Direction
        if direction[i - 1] == 1:
            if c[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    return (
        pd.Series(st, index=close.index, name="supertrend"),
        pd.Series(direction, index=close.index, name="supertrend_dir")
    )
