"""
strategy/asymmetric_sniper.py
-----------------------------
4-Hour Short Asymmetric Sniper Strategy:
1. Detects strong downward displacement (Impulse wave: drop >= 3.0x ATR in <= 6 bars).
2. Sets staggered limit orders in the premium retracement zone:
   - Tier 1: 33% at 50.0% retracement
   - Tier 2: 33% at 61.8% retracement
   - Tier 3: 34% at 70.5% retracement
3. Risk Management (Style A: Tight Asymmetric Invalidation):
   - Stop Loss: Placed at the 78.6% Fibonacci retracement level.
   - Average Stop Distance: ~2.6% (high leverage capacity).
   - Take Profit 1 (50% scale): Previous Impulse Trough (1.0R baseline, ~2.5R from 78.6% SL).
   - Take Profit 2 (50% runner): 1.272 Fibonacci Extension (~3.7R to 4.5R from 78.6% SL).
4. Strictly zero-lookahead, typed, and modular.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class ImpulseWave:
    symbol: str
    origin_bar_idx: int
    trough_bar_idx: int
    origin_time: pd.Timestamp
    trough_time: pd.Timestamp
    origin_high: float
    trough_low: float
    atr_at_origin: float
    delta_price: float


@dataclass
class SniperLevels:
    entry_tier1_50: float
    entry_tier2_62: float
    entry_tier3_70: float
    stop_loss_786: float
    take_profit_trough: float
    take_profit_ext1272: float
    blended_entry: float
    blended_risk_pct: float
    design_rr_to_trough: float
    design_rr_to_ext: float


@dataclass
class SniperSignal:
    symbol: str
    signal_time: pd.Timestamp
    wave: ImpulseWave
    levels: SniperLevels
    is_active: bool


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Computes standard Average True Range via numpy."""
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def calculate_sniper_levels(origin_high: float, trough_low: float) -> SniperLevels:
    """Calculates Style A Asymmetric Sniper price levels for a downward impulse wave."""
    delta = origin_high - trough_low
    p50 = trough_low + 0.50 * delta
    p62 = trough_low + 0.618 * delta
    p70 = trough_low + 0.705 * delta
    sl786 = trough_low + 0.786 * delta
    tp_trough = trough_low
    tp_ext = trough_low - 0.272 * delta
    
    blended_entry = (0.33 * p50) + (0.33 * p62) + (0.34 * p70)
    risk_dist = sl786 - blended_entry
    risk_pct = risk_dist / blended_entry if blended_entry > 0 else 0.0
    
    rr_trough = (blended_entry - tp_trough) / risk_dist if risk_dist > 0 else 0.0
    rr_ext = (blended_entry - tp_ext) / risk_dist if risk_dist > 0 else 0.0
    
    return SniperLevels(
        entry_tier1_50=p50,
        entry_tier2_62=p62,
        entry_tier3_70=p70,
        stop_loss_786=sl786,
        take_profit_trough=tp_trough,
        take_profit_ext1272=tp_ext,
        blended_entry=blended_entry,
        blended_risk_pct=risk_pct,
        design_rr_to_trough=rr_trough,
        design_rr_to_ext=rr_ext
    )


def scan_current_sniper_setups(
    df_4h: pd.DataFrame,
    symbol: str,
    min_atr_expansion: float = 3.0,
    max_impulse_bars: int = 6,
    max_wait_bars: int = 24
) -> Optional[SniperSignal]:
    """
    Scans the latest candles of a 4-Hour dataframe to find active, unfilled
    Style A Sniper retracement opportunities right now.
    """
    if len(df_4h) < 30:
        return None
        
    opens = df_4h["open"].values.astype(float)
    highs = df_4h["high"].values.astype(float)
    lows = df_4h["low"].values.astype(float)
    closes = df_4h["close"].values.astype(float)
    times = pd.to_datetime(df_4h["open_time"], utc=True).values
    atrs = compute_atr(highs, lows, closes, 14)
    
    n_bars = len(closes)
    latest_idx = n_bars - 1
    
    # Check backwards up to max_wait_bars to find recent confirmed downward impulse
    for i in range(latest_idx, max(14, latest_idx - max_wait_bars), -1):
        cur_atr = atrs[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            continue
            
        for w in range(3, max_impulse_bars + 1):
            if i - w < 0:
                continue
            sub_h = np.max(highs[i - w:i + 1])
            sub_l = np.min(lows[i - w:i + 1])
            if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= min_atr_expansion * cur_atr:
                origin_high = sub_h
                trough_low = sub_l
                origin_idx = i - w + int(np.argmax(highs[i - w:i + 1]))
                trough_idx = i
                
                # Check if price has subsequently broken below trough or above SL since bar i
                subsequent_lows = lows[i + 1:latest_idx + 1] if latest_idx > i else []
                subsequent_highs = highs[i + 1:latest_idx + 1] if latest_idx > i else []
                
                levels = calculate_sniper_levels(origin_high, trough_low)
                
                # Invalidation checks:
                if len(subsequent_lows) > 0 and np.min(subsequent_lows) < trough_low * 0.998:
                    break  # Continued dumping, impulse extended
                if len(subsequent_highs) > 0 and np.max(subsequent_highs) >= levels.stop_loss_786:
                    break  # Already breached SL
                    
                wave = ImpulseWave(
                    symbol=symbol,
                    origin_bar_idx=origin_idx,
                    trough_bar_idx=trough_idx,
                    origin_time=pd.Timestamp(times[origin_idx]),
                    trough_time=pd.Timestamp(times[trough_idx]),
                    origin_high=origin_high,
                    trough_low=trough_low,
                    atr_at_origin=cur_atr,
                    delta_price=origin_high - trough_low
                )
                
                return SniperSignal(
                    symbol=symbol,
                    signal_time=pd.Timestamp(times[latest_idx]),
                    wave=wave,
                    levels=levels,
                    is_active=True
                )
    return None
