from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd
import numpy as np

from backtest.trade import TradeExitReason, EntryMode, StopType
from backtest.execution_model import ExecutionModel
from backtest.position_manager import PositionManager
from backtest.engine import BacktestEngine
from strategy.entry_rules import EntryA_Immediate, EntryB_15mReversal, EntryC_LowerHighBreak
from strategy.exit_rules import ExitRules, PullbackBufferStop
from strategy.pullback_detector import ThresholdCrossingEvent


def make_test_bar(
    time_str: str,
    o: float,
    h: float,
    l: float,
    c: float,
    vol: float = 1_000_000.0,
    rank: int = 10,
    trend_inv: bool = False
) -> dict:
    return {
        "open_time": pd.Timestamp(time_str, tz="UTC"),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "quote_volume_usdt": vol,
        "rank": rank,
        "trend_invalidated": trend_inv
    }


def make_dummy_event(
    event_id: str,
    symbol: str,
    event_bar_idx: int,
    event_time: pd.Timestamp,
    swing_high: float,
    running_low: float,
    atr_ref: float = 2.0
) -> ThresholdCrossingEvent:
    return ThresholdCrossingEvent(
        event_id=event_id,
        symbol=symbol,
        swing_high_time=event_time - pd.Timedelta(hours=2),
        swing_high_price=swing_high,
        confirmation_time=event_time - pd.Timedelta(hours=1),
        swing_low_time=event_time - pd.Timedelta(hours=4),
        swing_low_price=swing_high - 10.0,
        impulse=10.0,
        impulse_atr=5.0,
        atr_ref=atr_ref,
        threshold_type="atr",
        threshold_value=0.50,
        event_bar_idx=event_bar_idx,
        event_time=event_time,
        event_close_time=event_time + pd.Timedelta(minutes=15),
        action_open_time=event_time + pd.Timedelta(minutes=15),
        reference_price=swing_high - 1.0,
        running_pullback_low=running_low,
        pullback_depth_atr=(swing_high - running_low) / atr_ref,
        retracement_pct=30.0,
        atr_entry=atr_ref,
        stop_buffer_atr=0.25,
        event_stop_level=running_low - 0.25 * atr_ref,
        trend_valid_default=True,
        trend_a=True,
        trend_b=True,
        trend_c=True,
        trend_d=True,
        universe_rank=10,
        is_controlled_pullback=True
    )


# ==============================================================================
# SCENARIO (a): Stop and Target in the Same Bar (Conservative Rule)
# ==============================================================================
def test_scenario_a_same_bar_ambiguity():
    """
    If candle high reaches target AND low reaches stop in the same bar,
    conservative rule must trigger STOP first and flag ambiguous_same_bar=True.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Event
        make_test_bar("2026-01-01 00:15:00", 100.0, 102.0, 98.0, 101.0),  # Bar 1: Entry
        make_test_bar("2026-01-01 00:30:00", 101.0, 115.0, 85.0, 100.0),  # Bar 2: Both hit! (Target=110, Stop=90)
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0, 2.0, 2.0])

    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=110.0, running_low=92.0, atr_ref=8.0)
    # Stop level = 92.0 - 0.25 * 8.0 = 90.0

    exec_model = ExecutionModel(taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop"
    assert t.ambiguous_same_bar is True
    assert t.exit_price_raw == 90.0  # Stop price


# ==============================================================================
# SCENARIO (b): Gap Through Stop
# ==============================================================================
def test_scenario_b_gap_through_stop():
    """
    If a bar gaps open below the stop price, exit fills at the bar open minus slippage,
    NEVER at the higher stop price.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Event
        make_test_bar("2026-01-01 00:15:00", 100.0, 101.0, 99.0, 100.0),  # Bar 1: Entry at open 100.0
        make_test_bar("2026-01-01 00:30:00", 85.0, 86.0, 80.0, 82.0),     # Bar 2: Gaps down to 85.0 (Stop is 90.0)
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0, 2.0, 2.0])

    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=110.0, running_low=92.0, atr_ref=8.0)
    # Stop is 90.0. Bar 2 opens at 85.0 < 90.0.

    exec_model = ExecutionModel(taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price_raw == 85.0  # Gapped bar open
    expected_fill = 85.0 * (1.0 - 0.0005)
    assert pytest.approx(t.exit_price, rel=1e-5) == expected_fill


# ==============================================================================
# SCENARIO (c): Fees, Slippage, and Funding Arithmetic
# ==============================================================================
def test_scenario_c_fees_slippage_funding_arithmetic():
    """
    Verify exact accounting of separate entry/exit fees, rank slippage, and 8h funding.
    """
    # Spanning across 08:00 UTC funding timestamp
    bars = [
        make_test_bar("2026-01-01 07:30:00", 100.0, 101.0, 99.0, 100.0, rank=10),  # Bar 0: Event
        make_test_bar("2026-01-01 07:45:00", 100.0, 101.0, 99.0, 100.0, rank=10),  # Bar 1: Entry at 100.0
        make_test_bar("2026-01-01 08:00:00", 100.0, 105.0, 99.0, 102.0, rank=10),  # Bar 2: Funding event 08:00
        make_test_bar("2026-01-01 08:15:00", 102.0, 112.0, 101.0, 111.0, rank=10), # Bar 3: Hits Target 110.0
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0, 2.0, 2.0, 2.0])

    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=110.0, running_low=95.0, atr_ref=4.0)
    # Target = 110.0, Stop = 94.0

    exec_model = ExecutionModel(
        maker_fee_rate=0.0000,
        taker_fee_rate=0.0002,
        funding_rate_8h=0.0001,
        slippage_by_rank={"1-25": 5.0},
        target_as_limit=True
    )
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades) == 1
    t = trades[0]

    # Entry: open = 100.0, slippage = 5 bps = 0.0005 -> fill = 100.05
    assert pytest.approx(t.entry_price, rel=1e-5) == 100.05
    assert pytest.approx(t.entry_fee, rel=1e-5) == 0.0002

    # Exit at target limit order: fill = 110.0, maker fee = 0.0, slippage = 0.0
    assert pytest.approx(t.exit_price, rel=1e-5) == 110.0
    assert pytest.approx(t.exit_fee, rel=1e-5) == 0.0000
    assert pytest.approx(t.total_fees, rel=1e-5) == 0.0002

    # Funding across 08:00 UTC -> 1 event * 0.0001 = 0.0001
    assert pytest.approx(t.funding_cost, rel=1e-5) == 0.0001

    # Net Return % = (110.0 - 100.05) / 100.05 - 0.0002 - 0.0001
    expected_price_ret = (110.0 - 100.05) / 100.05
    expected_net_ret = expected_price_ret - 0.0002 - 0.0001
    assert pytest.approx(t.net_pnl_pct, rel=1e-5) == expected_net_ret

    # Net R: net_ret / risk_pct
    risk_pct = (100.05 - 94.0) / 100.05
    expected_net_r = expected_net_ret / risk_pct
    assert pytest.approx(t.net_pnl_r, rel=1e-4) == expected_net_r


# ==============================================================================
# ==============================================================================
# SCENARIO (d): Time-Stop
# ==============================================================================
def test_scenario_d_time_stop():
    """
    Positions exceeding max holding period exit at the OPEN of the next candle.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Event
        make_test_bar("2026-01-01 00:15:00", 100.0, 101.0, 99.0, 100.0),  # Bar 1: Entry
        make_test_bar("2026-01-01 00:30:00", 100.0, 101.0, 99.0, 100.0),  # Bar 2: Held bar 1
        make_test_bar("2026-01-01 00:45:00", 100.5, 101.0, 99.0, 100.0),  # Bar 3: Held bar 2 -> Exits at open of Bar 3!
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0] * len(bars))

    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=110.0, running_low=95.0, atr_ref=4.0)

    exec_model = ExecutionModel(taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=0.5) # 2 bars
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "time_stop"
    assert t.exit_time == bars[3]["open_time"]
    assert t.exit_price_raw == 100.5


# ==============================================================================
# SCENARIO (e): Overlapping Signals
# ==============================================================================
def test_scenario_e_overlapping_signals():
    """
    Max ONE open position per symbol. A new signal while in a position is skipped
    and logged as 'skipped_overlap'.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Event 1
        make_test_bar("2026-01-01 00:15:00", 100.0, 101.0, 99.0, 100.0),  # Bar 1: Entry for Event 1
        make_test_bar("2026-01-01 00:30:00", 100.0, 101.0, 99.0, 100.0),  # Bar 2: Event 2 arrives here!
        make_test_bar("2026-01-01 00:45:00", 100.0, 101.0, 99.0, 100.0),  # Bar 3: Entry for Event 2 would be here
        make_test_bar("2026-01-01 01:00:00", 100.0, 115.0, 99.0, 114.0),  # Bar 4: Event 1 hits target
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0] * len(bars))

    ev1 = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=110.0, running_low=95.0)
    ev2 = make_dummy_event("ev2", "BTC_USDT", event_bar_idx=2, event_time=bars[2]["open_time"], swing_high=112.0, running_low=96.0)

    exec_model = ExecutionModel(taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev1, ev2])
    assert len(trades) == 1
    assert trades[0].event_id == "ev1"

    # Verify skipped_overlap logging
    skipped = engine.position_manager.skipped_overlaps
    assert len(skipped) == 1
    assert skipped[0]["candidate_id"] == "ev2"
    assert skipped[0]["reason"] == "skipped_overlap"


# ==============================================================================
# SCENARIO (f): Entry Fill at Next Open (Never at Signal Close)
# ==============================================================================
def test_scenario_f_entry_fill_at_next_open():
    """
    Entry MUST fill at the open of candle t+1, NEVER at the close of signal candle t.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Signal close = 100.0
        make_test_bar("2026-01-01 00:15:00", 105.0, 115.0, 104.0, 114.0), # Bar 1: Next open = 105.0 (gapped up)
        make_test_bar("2026-01-01 00:30:00", 114.0, 125.0, 113.0, 122.0), # Bar 2: Hits Target 120.0
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0, 2.0, 2.0])

    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=120.0, running_low=95.0)

    exec_model = ExecutionModel(taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )

    trades = engine.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price_raw == 105.0  # Open of bar 1, NOT close 100.0 of bar 0
    assert pytest.approx(t.entry_price, rel=1e-5) == 105.0 * (1.0 + 0.0005)


# ==============================================================================
# SCENARIO (g): Rejected by Min R:R Filter
# ==============================================================================
def test_scenario_g_rejected_by_min_rr():
    """
    Candidates with potential R:R < min_rr must be rejected and logged.
    """
    bars = [
        make_test_bar("2026-01-01 00:00:00", 100.0, 101.0, 99.0, 100.0),  # Bar 0: Event
        make_test_bar("2026-01-01 00:15:00", 100.0, 101.0, 99.0, 100.0),  # Bar 1: Entry candidate
        make_test_bar("2026-01-01 00:30:00", 100.0, 110.0, 99.0, 108.0),  # Bar 2: Hits Target 105.0
    ]
    df_15m = pd.DataFrame(bars)
    atr_series = pd.Series([2.0, 2.0, 2.0])

    # Target = 105.0, Stop = 95.0. Potential R:R = (105 - 100) / (100 - 95) = 5.0 / 5.0 = 1.0
    ev = make_dummy_event("ev1", "BTC_USDT", event_bar_idx=0, event_time=bars[0]["open_time"], swing_high=105.0, running_low=95.5, atr_ref=2.0)
    # Stop level = 95.5 - 0.25*2.0 = 95.0

    exec_model = ExecutionModel()
    
    # 1. Require min_rr = 2.0 -> Rejected
    engine_strict = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=2.0),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25))
    )
    trades_strict = engine_strict.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades_strict) == 0
    assert len(engine_strict.rejected_min_rr) == 1
    assert engine_strict.rejected_min_rr[0]["event_id"] == "ev1"

    # 2. Require min_rr = 0.8 -> Accepted
    engine_lenient = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.8),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25))
    )
    trades_lenient = engine_lenient.simulate_symbol("BTC_USDT", df_15m, atr_series, [ev])
    assert len(trades_lenient) == 1

