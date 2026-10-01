from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class TradeExitReason(str, Enum):
    TARGET = "target"
    STOP = "stop"
    TIME_STOP = "time_stop"
    TREND_INVALIDATION = "trend_invalidation"


class EntryMode(str, Enum):
    ENTRY_A = "Entry_A"  # Immediate at next open
    ENTRY_B = "Entry_B"  # 15m candle reversal
    ENTRY_C = "Entry_C"  # 15m lower-high break
    ENTRY_D = "Entry_D"  # Plugin / extension


class StopType(str, Enum):
    PULLBACK_BUFFER = "pullback_buffer"      # pullback_low - buffer * ATR
    PRIOR_SWING_LOW = "prior_swing_low"      # below prior 15m swing low
    FIXED_ATR = "fixed_atr"                  # entry - k * ATR
    PERCENTAGE = "percentage"                # entry * (1 - pct)


@dataclass
class Trade:
    # Identifiers & Link to Phase 3 Setup
    trade_id: str
    event_id: str
    symbol: str
    entry_mode: str

    # Timestamps
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    # Universe & Market Condition at Decision Time
    universe_rank: Optional[int]
    volume_24h_usdt: Optional[float]
    trend_state: str
    trend_a: bool
    trend_b: bool
    trend_c: bool
    trend_d: bool

    # Swing & Pullback Context
    swing_high_time: pd.Timestamp
    swing_high_price: float
    swing_low_time: pd.Timestamp
    swing_low_price: float
    atr_ref: float
    atr_entry: float
    impulse: float
    impulse_atr: float
    running_pullback_low: float
    pullback_depth_atr: float
    retracement_pct: float

    # Trade Levels & Pricing
    entry_price_raw: float
    entry_price: float
    stop_price: float
    target_price: float
    potential_rr: float

    # Exit Execution
    exit_price_raw: float
    exit_price: float
    exit_reason: str
    bars_held: int
    holding_time_minutes: float
    time_to_target_bars: Optional[int]
    time_to_target_minutes: Optional[float]

    # Financial Returns & Costs
    gross_pnl_pct: float
    gross_pnl_r: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    slippage_cost: float
    funding_cost: float
    net_pnl_pct: float
    net_pnl_r: float

    # Excursions & Diagnostics
    mfe_price: float
    mfe_atr: float
    mfe_r: float
    mae_price: float
    mae_atr: float
    mae_r: float
    retested_high: bool
    broke_high: bool
    eventually_broke_pullback_low: bool
    ambiguous_same_bar: bool

    # BTC Context / Regime
    btc_regime: str
    btc_above_ema50: bool
    btc_above_ema200: bool


@dataclass
class Position:
    trade_id: str
    event_id: str
    symbol: str
    entry_mode: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_bar_idx: int
    entry_price_raw: float
    entry_price: float
    stop_price: float
    target_price: float
    potential_rr: float
    atr_ref: float
    atr_entry: float

    # Snapshot context
    universe_rank: Optional[int]
    volume_24h_usdt: Optional[float]
    trend_state: str
    trend_a: bool
    trend_b: bool
    trend_c: bool
    trend_d: bool
    swing_high_time: pd.Timestamp
    swing_high_price: float
    swing_low_time: pd.Timestamp
    swing_low_price: float
    impulse: float
    impulse_atr: float
    running_pullback_low: float
    pullback_depth_atr: float
    retracement_pct: float
    entry_fee: float
    entry_slippage: float

    # Regime snapshot
    btc_regime: str
    btc_above_ema50: bool
    btc_above_ema200: bool

    # Running tracking during position life
    mfe_price: float = 0.0
    mae_price: float = float("inf")
    bars_held: int = 0
    funding_paid: float = 0.0
