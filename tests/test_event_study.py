"""
tests/test_event_study.py
Unit tests for Phase 3 event outcome calculation, holdout guard, and classification.
"""

import pytest
import pandas as pd
import numpy as np

from analysis.outcomes import EventOutcomeCalculator


def test_holdout_guard_raises():
    calc = EventOutcomeCalculator(holdout_start="2026-07-01T00:00:00Z")
    
    # In-sample event
    in_sample_times = pd.Series([pd.Timestamp("2026-03-01T12:00:00Z")])
    calc.validate_holdout_guard(in_sample_times, allow_holdout=False)

    # Holdout event
    holdout_times = pd.Series([pd.Timestamp("2026-07-02T12:00:00Z")])
    with pytest.raises(PermissionError, match="HOLDOUT VIOLATION"):
        calc.validate_holdout_guard(holdout_times, allow_holdout=False)

    # Allowed when explicitly set (Phase 5 only)
    calc.validate_holdout_guard(holdout_times, allow_holdout=True)


def test_outcome_retest_and_types():
    calc = EventOutcomeCalculator(horizon_bars=10, holdout_start="2026-07-01T00:00:00Z")

    # Construct synthetic 15m OHLC bars (20 bars)
    # Event fires at bar 5 (close of bar 5, entry open at bar 6)
    # Reference price = open[6] = 100
    # Swing high = 105
    # Stop level = 95
    # ATR ref = 2.0
    dates = pd.date_range("2026-01-01", periods=25, freq="15min", tz="UTC")
    
    # Case 1: Type A Continuation (Retest at bar 7, broke high at bar 8, no stop hit)
    highs = [100.0] * 25
    lows = [98.0] * 25
    closes = [99.0] * 25
    opens = [99.0] * 25

    # Bar 7 hits retest (high >= 105)
    highs[7] = 106.0
    # Bar 8 breaks high (close > 105 + 0.25 * 2.0 = 105.5)
    closes[8] = 107.0

    df_bars = pd.DataFrame({
        "open_time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes
    })

    events = pd.DataFrame([{
        "event_id": "test_ev_1",
        "event_bar_idx": 5,
        "event_time": dates[5],
        "reference_price": 99.0,
        "swing_high_price": 105.0,
        "event_stop_level": 95.0,
        "atr_ref": 2.0
    }])

    res = calc.compute_symbol_outcomes(events, df_bars)
    assert res.iloc[0]["retest_primary"] == True
    assert res.iloc[0]["retest_before_failure_pessimistic"] == True
    assert res.iloc[0]["retest_before_failure_optimistic"] == True
    assert res.iloc[0]["ambiguous_same_bar"] == False
    assert res.iloc[0]["time_to_retest_bars"] == 2  # bar 7 is index 2 relative to start_bar 6
    assert res.iloc[0]["broke_high"] == True
    assert res.iloc[0]["outcome_type"] == "A"


def test_outcome_type_c_failure():
    calc = EventOutcomeCalculator(horizon_bars=10, holdout_start="2026-07-01T00:00:00Z")
    dates = pd.date_range("2026-01-01", periods=25, freq="15min", tz="UTC")
    
    highs = [100.0] * 25
    lows = [98.0] * 25
    closes = [99.0] * 25
    opens = [99.0] * 25

    # Bar 7 hits stop (low <= 95)
    lows[7] = 94.0

    df_bars = pd.DataFrame({
        "open_time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes
    })

    events = pd.DataFrame([{
        "event_id": "test_ev_2",
        "event_bar_idx": 5,
        "event_time": dates[5],
        "reference_price": 99.0,
        "swing_high_price": 105.0,
        "event_stop_level": 95.0,
        "atr_ref": 2.0
    }])

    res = calc.compute_symbol_outcomes(events, df_bars)
    assert res.iloc[0]["retest_primary"] == False
    assert res.iloc[0]["failure"] == True
    assert res.iloc[0]["retest_before_failure_pessimistic"] == False
    assert res.iloc[0]["outcome_type"] == "C"


def test_same_bar_ambiguity():
    calc = EventOutcomeCalculator(horizon_bars=10, holdout_start="2026-07-01T00:00:00Z")
    dates = pd.date_range("2026-01-01", periods=25, freq="15min", tz="UTC")
    
    highs = [100.0] * 25
    lows = [98.0] * 25
    closes = [99.0] * 25
    opens = [99.0] * 25

    # Bar 6 (first bar) has high 106 (hits target 105) AND low 94 (hits stop 95)
    highs[6] = 106.0
    lows[6] = 94.0

    df_bars = pd.DataFrame({
        "open_time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes
    })

    events = pd.DataFrame([{
        "event_id": "test_ev_3",
        "event_bar_idx": 5,
        "event_time": dates[5],
        "reference_price": 99.0,
        "swing_high_price": 105.0,
        "event_stop_level": 95.0,
        "atr_ref": 2.0
    }])

    res = calc.compute_symbol_outcomes(events, df_bars)
    assert res.iloc[0]["ambiguous_same_bar"] == True
    assert res.iloc[0]["retest_before_failure_pessimistic"] == False  # Stop-first
    assert res.iloc[0]["retest_before_failure_optimistic"] == True   # Target-first
    assert res.iloc[0]["outcome_type"] == "C"  # Conservative default


def test_wilson_and_bootstrap_cis():
    from analysis.baselines import (
        compute_wilson_ci,
        compute_cluster_bootstrap_ci,
        compute_difference_bootstrap_ci
    )

    # Wilson CI on 60 successes out of 100
    p, lo, hi = compute_wilson_ci(60, 100)
    assert p == 0.60
    assert 0.50 < lo < 0.60
    assert 0.60 < hi < 0.70

    # Cluster bootstrap
    df = pd.DataFrame({
        "swing_high_time": ["2026-01-01"] * 50 + ["2026-01-02"] * 50,
        "retest": [1] * 40 + [0] * 10 + [1] * 35 + [0] * 15
    })
    p_boot, b_lo, b_hi = compute_cluster_bootstrap_ci(df, "retest", "swing_high_time", n_bootstrap=100)
    assert 0.70 <= p_boot <= 0.80
    assert 0.60 <= b_lo <= b_hi <= 0.90

    # Difference CI
    t_succ = np.array([1] * 70 + [0] * 30)
    b_succ = np.array([1] * 50 + [0] * 50)
    diff, d_lo, d_hi = compute_difference_bootstrap_ci(t_succ, b_succ, n_bootstrap=100)
    assert np.isclose(diff, 0.20)
    assert d_lo < diff < d_hi

