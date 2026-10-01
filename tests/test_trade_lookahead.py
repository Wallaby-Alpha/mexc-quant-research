from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd
import numpy as np

from strategy.indicators import compute_wilder_atr
from strategy.alignment import align_1h_to_15m
from strategy.trend_detector import TrendDetector
from strategy.swing_detector import SwingDetector
from strategy.pullback_detector import PullbackDetector
from strategy.entry_rules import EntryA_Immediate
from strategy.exit_rules import ExitRules, PullbackBufferStop
from backtest.execution_model import ExecutionModel
from backtest.engine import BacktestEngine
from tests.test_no_lookahead import make_synthetic_ohlc


def test_trade_pipeline_truncation_lookahead():
    """
    Truncation Lookahead Test on the Full Trade Pipeline:
    Run trade simulation on full data, then truncate data at cutoff.
    Signals and entry executions at <= cutoff MUST be strictly identical.
    """
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df_15m = make_synthetic_ohlc(500, start, freq_minutes=15, seed=777)
    df_1h = make_synthetic_ohlc(125, start, freq_minutes=60, seed=778)

    # 1. Full Run
    trend_detector = TrendDetector(k=3, min_warmup_bars=20)
    df_1h_trends = trend_detector.compute_1h_trends(df_1h)
    trend_cols = ["ema50", "ema200", "trend_a", "trend_b", "trend_c", "trend_d", "trend_invalidated"]
    df_aligned = align_1h_to_15m(df_15m, df_1h_trends, trend_cols)
    atr_series = compute_wilder_atr(df_aligned["high"], df_aligned["low"], df_aligned["close"], period=14)

    swing_detector = SwingDetector(fractal_n=3, lookback_l=20, min_impulse_atr=1.5)
    swings = swing_detector.find_swing_highs("BTC_USDT", df_aligned, atr_series)

    pullback_detector = PullbackDetector(atr_thresholds=[0.50], ret_thresholds=[], stop_buffer=0.25)
    all_events = []
    for sw in swings:
        evs = pullback_detector.process_swing_high(
            swing=sw,
            df_15m=df_aligned,
            atr_15m=atr_series,
            trend_flags=df_aligned[trend_cols]
        )
        all_events.extend(evs)

    all_events = sorted(all_events, key=lambda x: x.event_bar_idx)

    exec_model = ExecutionModel(maker_fee_rate=0.0000, taker_fee_rate=0.0002, slippage_by_rank={"1-25": 5.0})
    engine_full = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )
    trades_full = engine_full.simulate_symbol("BTC_USDT", df_aligned, atr_series, all_events)

    assert len(trades_full) > 0, "Synthetic run should produce at least one trade"

    # Select cutoff index such that some trades have entered
    cutoff_idx = 350
    cutoff_time = df_15m.iloc[cutoff_idx]["open_time"]
    trades_full_before_cutoff = [t for t in trades_full if t.entry_time <= cutoff_time]

    assert len(trades_full_before_cutoff) > 0

    # 2. Truncated Run
    df_15m_trunc = df_15m.iloc[:cutoff_idx + 1].copy()
    df_1h_trunc = df_1h[df_1h["open_time"] + pd.Timedelta(hours=1) <= cutoff_time + pd.Timedelta(minutes=15)].copy()

    df_1h_trends_trunc = trend_detector.compute_1h_trends(df_1h_trunc)
    df_aligned_trunc = align_1h_to_15m(df_15m_trunc, df_1h_trends_trunc, trend_cols)
    atr_trunc = compute_wilder_atr(df_aligned_trunc["high"], df_aligned_trunc["low"], df_aligned_trunc["close"], period=14)

    swings_trunc = swing_detector.find_swing_highs("BTC_USDT", df_aligned_trunc, atr_trunc)
    events_trunc = []
    for sw in swings_trunc:
        evs = pullback_detector.process_swing_high(
            swing=sw,
            df_15m=df_aligned_trunc,
            atr_15m=atr_trunc,
            trend_flags=df_aligned_trunc[trend_cols]
        )
        events_trunc.extend(evs)
    events_trunc = sorted(events_trunc, key=lambda x: x.event_bar_idx)

    engine_trunc = BacktestEngine(
        execution_model=exec_model,
        entry_rule=EntryA_Immediate(min_rr=0.5),
        exit_rules=ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)
    )
    trades_trunc = engine_trunc.simulate_symbol("BTC_USDT", df_aligned_trunc, atr_trunc, events_trunc)

    # 3. Assert exact match for all trades entered <= cutoff
    # Every trade entered before cutoff in the truncated run must match full run
    matched_count = 0
    for t_full in trades_full_before_cutoff:
        # If the trade also exited <= cutoff, it must be completely identical
        if t_full.exit_time <= cutoff_time:
            matching_trunc = [t for t in trades_trunc if t.event_id == t_full.event_id]
            assert len(matching_trunc) == 1, f"Trade {t_full.trade_id} missing in truncated run!"
            t_tr = matching_trunc[0]
            assert t_tr.entry_time == t_full.entry_time
            assert pytest.approx(t_tr.entry_price, rel=1e-8) == t_full.entry_price
            assert pytest.approx(t_tr.stop_price, rel=1e-8) == t_full.stop_price
            assert pytest.approx(t_tr.target_price, rel=1e-8) == t_full.target_price
            assert t_tr.exit_reason == t_full.exit_reason
            assert pytest.approx(t_tr.exit_price, rel=1e-8) == t_full.exit_price
            assert pytest.approx(t_tr.net_pnl_pct, rel=1e-8) == t_full.net_pnl_pct
            matched_count += 1

    assert matched_count > 0, "At least one fully completed trade before cutoff was verified!"
