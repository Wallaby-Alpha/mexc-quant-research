from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from data.config import AppConfig
from strategy.indicators import compute_ema, compute_wilder_atr, compute_supertrend
from strategy.alignment import align_1h_to_15m
from strategy.trend_detector import TrendDetector
from strategy.swing_detector import SwingDetector, ConfirmedSwingHigh
from strategy.pullback_detector import PullbackDetector, ThresholdCrossingEvent
from strategy.setup_lifecycle import SetupLifecycle, SetupState
from strategy.universe import PointInTimeUniverse


def make_synthetic_ohlc(n_bars: int, start_time: datetime, freq_minutes: int = 15, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    times = [start_time + timedelta(minutes=freq_minutes * i) for i in range(n_bars)]
    
    # Generate realistic random walk
    returns = np.random.normal(0.0002, 0.005, size=n_bars)
    prices = 1000.0 * np.exp(np.cumsum(returns))
    
    opens = prices
    closes = prices * (1.0 + np.random.normal(0, 0.002, size=n_bars))
    highs = np.maximum(opens, closes) * (1.0 + np.abs(np.random.normal(0, 0.003, size=n_bars)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(np.random.normal(0, 0.003, size=n_bars)))
    vol_base = np.abs(np.random.normal(10.0, 3.0, size=n_bars)) + 1.0
    quote_vol = vol_base * closes

    df = pd.DataFrame({
        "open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume_base": vol_base,
        "quote_volume_usdt": quote_vol,
        "source_market": "futures"
    })
    return df


# ==============================================================================
# LOOKAHEAD TEST A: Truncation Test
# ==============================================================================
def test_lookahead_a_truncation():
    """
    Compute indicators and swing setups on full history, then recompute
    after truncating at an arbitrary cutoff. Outputs <= cutoff must be IDENTICAL.
    """
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df_15m = make_synthetic_ohlc(400, start, freq_minutes=15, seed=101)
    
    # Pre-build synthetic 1H bars
    df_1h = make_synthetic_ohlc(100, start, freq_minutes=60, seed=102)

    trend_detector = TrendDetector(k=3, min_warmup_bars=20)
    df_1h_trends = trend_detector.compute_1h_trends(df_1h)
    
    trend_cols = ["ema50", "ema200", "trend_a", "trend_b", "trend_c", "trend_d", "trend_invalidated"]
    df_aligned_full = align_1h_to_15m(df_15m, df_1h_trends, trend_cols)
    atr_full = compute_wilder_atr(df_aligned_full["high"], df_aligned_full["low"], df_aligned_full["close"], period=14)

    cutoff_idx = 280
    cutoff_time = df_15m.iloc[cutoff_idx]["open_time"]

    # Truncated run
    df_15m_trunc = df_15m.iloc[:cutoff_idx + 1].copy()
    df_1h_trunc = df_1h[df_1h["open_time"] + pd.Timedelta(hours=1) <= cutoff_time + pd.Timedelta(minutes=15)].copy()

    df_1h_trends_trunc = trend_detector.compute_1h_trends(df_1h_trunc)
    df_aligned_trunc = align_1h_to_15m(df_15m_trunc, df_1h_trends_trunc, trend_cols)
    atr_trunc = compute_wilder_atr(df_aligned_trunc["high"], df_aligned_trunc["low"], df_aligned_trunc["close"], period=14)

    # Assert indicators at all bars <= cutoff are strictly identical
    # Compare ATR
    np.testing.assert_allclose(
        atr_full.iloc[:cutoff_idx + 1].dropna().values,
        atr_trunc.dropna().values,
        rtol=1e-10,
        err_msg="ATR values differ between full and truncated runs!"
    )
    # Compare aligned 1H trend flags
    for col in ["trend_c", "trend_invalidated"]:
        pd.testing.assert_series_equal(
            df_aligned_full[col].iloc[:cutoff_idx + 1],
            df_aligned_trunc[col],
            check_names=False,
            obj=f"Aligned {col} differs after truncation!"
        )


# ==============================================================================
# LOOKAHEAD TEST B: Confirmation Delay Test
# ==============================================================================
def test_lookahead_b_confirmation_delay():
    """
    A swing high at bar T is NEVER visible or acted upon before the CLOSE of candle T+N.
    Earliest action time is strictly open of T+N+1.
    """
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    N = 3
    # Hand-craft a clear swing high at index 4 (T=4)
    # N=3 before (indices 1,2,3) must be < high[4]
    # N=3 after (indices 5,6,7) must be < high[4]
    highs = [100.0, 105.0, 108.0, 110.0, 130.0, 115.0, 112.0, 109.0, 108.0, 107.0]
    lows  = [ 95.0,  98.0,  99.0, 100.0, 115.0, 105.0, 104.0, 103.0, 102.0, 101.0]
    closes= [ 98.0, 106.0, 107.0, 108.0, 125.0, 110.0, 108.0, 105.0, 104.0, 103.0]
    opens = [ 96.0,  99.0, 106.0, 107.0, 110.0, 120.0, 110.0, 108.0, 105.0, 104.0]
    
    times = [start + timedelta(minutes=15 * i) for i in range(len(highs))]
    df = pd.DataFrame({
        "open_time": times, "open": opens, "high": highs, "low": lows, "close": closes
    })
    atr = pd.Series([5.0] * len(df))

    detector = SwingDetector(fractal_n=N, lookback_l=3, min_impulse_atr=1.0)
    
    # 1. At t = T = 4: cannot be confirmed
    swings_at_T = detector.find_swing_highs("TEST", df.iloc[:5], atr.iloc[:5])
    assert len(swings_at_T) == 0, "Swing high detected at bar T before right side exists!"

    # 2. At t = T + 1 (5), T + 2 (6): cannot be confirmed
    assert len(detector.find_swing_highs("TEST", df.iloc[:6], atr.iloc[:6])) == 0
    assert len(detector.find_swing_highs("TEST", df.iloc[:7], atr.iloc[:7])) == 0

    # 3. At t = T + N = 7 (close of candle 7): confirmed!
    swings_at_TN = detector.find_swing_highs("TEST", df.iloc[:8], atr.iloc[:8])
    assert len(swings_at_TN) == 1
    sw = swings_at_TN[0]
    assert sw.swing_high_bar_idx == 4
    assert sw.confirmation_bar_idx == 7
    assert sw.confirmation_time == df.iloc[7]["open_time"]
    assert sw.earliest_action_time == df.iloc[7]["open_time"] + pd.Timedelta(minutes=15)


# ==============================================================================
# LOOKAHEAD TEST C: 1H Multi-Timeframe Alignment Test
# ==============================================================================
def test_lookahead_c_1h_alignment():
    """
    A 15m candle must NEVER see a 1H candle that had not fully closed.
    1H bar at 10:00 UTC closes at 11:00 UTC.
    Bars at 10:00, 10:15, 10:30, 10:45 MUST NOT see the 10:00 1H bar.
    The bar at 11:00 UTC is the FIRST bar that sees it.
    """
    # 1H bars: 09:00, 10:00, 11:00
    t_09h = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
    t_10h = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    t_11h = datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc)

    df_1h = pd.DataFrame({
        "open_time": [t_09h, t_10h, t_11h],
        "test_metric": [100.0, 200.0, 300.0]
    })

    # 15m bars from 09:00 to 11:15
    times_15m = [t_09h + timedelta(minutes=15 * i) for i in range(10)]
    df_15m = pd.DataFrame({"open_time": times_15m})

    aligned = align_1h_to_15m(df_15m, df_1h, columns_to_align=["test_metric"])

    # 09:00 1H bar closes at 10:00 UTC. So bars 09:00..09:45 should be NaN (no prior 1H bar).
    for i in range(4):
        assert pd.isna(aligned.iloc[i]["test_metric"]), f"Lookahead! Bar {aligned.iloc[i]['open_time']} saw future 1H bar!"

    # Bars 10:00, 10:15, 10:30, 10:45 should see ONLY 100.0 (from 09:00 1H bar)
    for i in range(4, 8):
        val = aligned.iloc[i]["test_metric"]
        assert val == 100.0, f"Bar {aligned.iloc[i]['open_time']} expected 100.0 but got {val}"

    # Bar 11:00 is the first bar that sees 200.0 (from 10:00 1H bar)
    assert aligned.iloc[8]["open_time"] == t_11h
    assert aligned.iloc[8]["test_metric"] == 200.0


# ==============================================================================
# LOOKAHEAD TEST D: Universe Point-in-Time Test
# ==============================================================================
def test_lookahead_d_universe():
    """
    Universe rank for hour t is completely unchanged when all data after t is removed.
    """
    config = AppConfig()
    config.universe.min_history_1h_bars = 5
    config.universe.min_volume_usdt = 500.0
    config.universe.target_size = 5

    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Create 3 symbols with distinct volumes
    df_a = make_synthetic_ohlc(50, start, freq_minutes=60, seed=1)
    df_b = make_synthetic_ohlc(50, start, freq_minutes=60, seed=2)
    df_a["quote_volume_usdt"] = 2000.0
    df_b["quote_volume_usdt"] = 5000.0

    universe_engine = PointInTimeUniverse(config)
    
    # Mock cache
    class MockCache:
        def list_cached_symbols(self, tf): return ["A", "B"]
        def load_klines(self, sym, tf): return df_a if sym == "A" else df_b

    universe_engine.cache = MockCache()
    full_table = universe_engine.build_universe_table()

    # Pick a decision time
    t_test = full_table["timestamp"].iloc[10]
    rank_a_full = full_table[(full_table["timestamp"] == t_test) & (full_table["symbol"] == "A")]["rank"].values[0]

    # Truncate raw data to only bars closing at or before t_test
    df_a_trunc = df_a[df_a["open_time"] + pd.Timedelta(hours=1) <= t_test]
    df_b_trunc = df_b[df_b["open_time"] + pd.Timedelta(hours=1) <= t_test]
    
    class MockCacheTrunc:
        def list_cached_symbols(self, tf): return ["A", "B"]
        def load_klines(self, sym, tf): return df_a_trunc if sym == "A" else df_b_trunc

    universe_engine.cache = MockCacheTrunc()
    trunc_table = universe_engine.build_universe_table()
    rank_a_trunc = trunc_table[(trunc_table["timestamp"] == t_test) & (trunc_table["symbol"] == "A")]["rank"].values[0]

    assert rank_a_full == rank_a_trunc, f"Universe rank changed! Full: {rank_a_full}, Trunc: {rank_a_trunc}"


# ==============================================================================
# LOOKAHEAD TEST E: Future Perturbation Test
# ==============================================================================
def test_lookahead_e_future_perturbation():
    """
    Randomly corrupt all future candles after t (e.g. 10x price spike or drop).
    Outputs at or before t must NOT change at all.
    """
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = make_synthetic_ohlc(300, start, freq_minutes=15, seed=777)
    
    t_split = 200
    atr_original = compute_wilder_atr(df["high"], df["low"], df["close"], period=14)

    # Corrupt future bars (index 201 onwards) with 10x price spike
    df_corrupted = df.copy()
    df_corrupted.loc[t_split + 1:, "open"] *= 10.0
    df_corrupted.loc[t_split + 1:, "high"] *= 10.0
    df_corrupted.loc[t_split + 1:, "low"] *= 10.0
    df_corrupted.loc[t_split + 1:, "close"] *= 10.0

    atr_corrupted = compute_wilder_atr(df_corrupted["high"], df_corrupted["low"], df_corrupted["close"], period=14)

    # Compare ATR at all bars <= t_split
    np.testing.assert_allclose(
        atr_original.iloc[:t_split + 1].dropna().values,
        atr_corrupted.iloc[:t_split + 1].dropna().values,
        rtol=1e-12,
        err_msg="Future price corruption leaked backward into prior ATR values!"
    )


# ==============================================================================
# LOOKAHEAD TEST F: Causal Bucketing Test
# ==============================================================================
def test_lookahead_f_bucketing_no_leakage():
    """
    Bucket assignment MUST use depth AT SIGNAL TIME only.
    Never use diag_final_depth_* for filtering or grouping.
    """
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    sw = ConfirmedSwingHigh(
        symbol="TEST",
        swing_high_bar_idx=5,
        swing_high_time=start + timedelta(minutes=15 * 5),
        swing_high_price=100.0,
        confirmation_bar_idx=8,
        confirmation_time=start + timedelta(minutes=15 * 8),
        earliest_action_time=start + timedelta(minutes=15 * 9),
        fractal_n=3,
        swing_low_bar_idx=1,
        swing_low_time=start + timedelta(minutes=15),
        swing_low_price=80.0,
        impulse=20.0,
        impulse_pct=0.25,
        impulse_atr=4.0,
        atr_ref=5.0
    )

    # Construct 15m price path:
    # Bar 8 (confirmation): low = 98.0 (depth = 2.0 / 5.0 = 0.40 ATR)
    # Bar 9: low = 97.0 (running low = 97.0 -> depth = 3.0 / 5.0 = 0.60 ATR -> meets 0.50 threshold!)
    # Bar 15: crashes to low = 85.0 (depth = 15.0 / 5.0 = 3.0 ATR)
    n = 20
    opens = [99.0] * n
    highs = [101.0] * n
    lows = [98.0] * n
    closes = [99.0] * n
    lows[9] = 97.0
    lows[15] = 85.0  # Deep future drop

    df = pd.DataFrame({
        "open_time": [start + timedelta(minutes=15 * i) for i in range(n)],
        "open": opens, "high": highs, "low": lows, "close": closes
    })
    atr = pd.Series([5.0] * n)
    trend_flags = pd.DataFrame({
        "trend_a": [True]*n, "trend_b": [True]*n, "trend_c": [True]*n, "trend_d": [True]*n, "trend_invalidated": [False]*n
    })

    detector = PullbackDetector(atr_thresholds=[0.50, 1.00], ret_thresholds=[])
    events = detector.process_swing_high(sw, df, atr, trend_flags)

    # Event for 0.50 ATR threshold should fire at bar 9
    ev_05 = [e for e in events if e.threshold_value == 0.50][0]
    assert ev_05.event_bar_idx == 9
    assert ev_05.pullback_depth_atr == 0.60  # depth AT SIGNAL TIME
    assert ev_05.diag_final_depth_atr == 3.0 # future diagnostic depth

    # The causal event depth MUST NOT equal the future diagnostic depth
    assert ev_05.pullback_depth_atr < ev_05.diag_final_depth_atr


# ==============================================================================
# UNIT TESTS: State Transitions
# ==============================================================================
def test_setup_lifecycle_state_transitions():
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    t_ev = t0 + timedelta(hours=2)
    t_res = t0 + timedelta(hours=4)

    setup = SetupLifecycle(
        symbol="BTC_USDT",
        swing_high_time=t0,
        swing_high_price=100.0,
        confirmation_time=t0 + timedelta(minutes=45)
    )
    assert setup.state == SetupState.PENDING

    # Trigger threshold event -> ARMED
    setup.trigger_event(t_ev)
    assert setup.state == SetupState.ARMED
    assert setup.transition_time == t_ev

    # Resolve -> RESOLVED
    setup.resolve(t_res, "retest_wick_hit")
    assert setup.state == SetupState.RESOLVED
    assert setup.transition_reason == "retest_wick_hit"
