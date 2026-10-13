import pytest
import numpy as np
import pandas as pd
from strategy.asymmetric_sniper import (
    calculate_sniper_levels,
    compute_atr,
    scan_current_sniper_setups,
    ImpulseWave,
    SniperLevels,
)


def test_calculate_sniper_levels_math():
    origin_high = 100.0
    trough_low = 80.0
    # delta = 20.0
    levels = calculate_sniper_levels(origin_high, trough_low)
    
    # 50% = 80 + 10 = 90.0
    assert abs(levels.entry_tier1_50 - 90.0) < 1e-4
    # 61.8% = 80 + 12.36 = 92.36
    assert abs(levels.entry_tier2_62 - 92.36) < 1e-4
    # 70.5% = 80 + 14.1 = 94.1
    assert abs(levels.entry_tier3_70 - 94.1) < 1e-4
    # 78.6% SL = 80 + 15.72 = 95.72
    assert abs(levels.stop_loss_786 - 95.72) < 1e-4
    # Trough TP = 80.0
    assert abs(levels.take_profit_trough - 80.0) < 1e-4
    # 1.272 Ext TP = 80 - (0.272 * 20) = 80 - 5.44 = 74.56
    assert abs(levels.take_profit_ext1272 - 74.56) < 1e-4
    
    # Blended entry is between 90 and 94.1
    assert 90.0 < levels.blended_entry < 94.1
    # Risk distance is positive
    assert levels.stop_loss_786 > levels.blended_entry
    # Design RR to Ext is > 3.0
    assert levels.design_rr_to_ext > 3.0


def test_compute_atr():
    high = np.array([10.0, 11.0, 12.0, 11.5, 12.5] * 4)
    low = np.array([9.0, 9.5, 10.5, 10.0, 11.0] * 4)
    close = np.array([9.5, 10.5, 11.0, 10.8, 11.8] * 4)
    atr = compute_atr(high, low, close, 5)
    assert len(atr) == len(close)
    assert not np.isnan(atr[-1])
    assert atr[-1] > 0


def test_scan_current_sniper_setups_synthetic():
    # Construct a synthetic 4h dataframe with an impulse down then shallow retrace
    n = 40
    dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    prices = [100.0] * 20
    # Sharp impulse down from 100 to 80 over 4 bars (bars 20 to 24)
    prices += [95.0, 90.0, 85.0, 80.0]
    # Retrace to 89 (around 45% retrace) over next bars
    prices += [82.0, 84.0, 86.0, 88.0, 89.0]
    prices += [89.0] * (n - len(prices))
    
    df = pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1000.0] * n
    })
    
    sig = scan_current_sniper_setups(df, "TEST_USDT", min_atr_expansion=2.0)
    assert sig is not None
    assert sig.symbol == "TEST_USDT"
    assert sig.wave.origin_high > sig.wave.trough_low
    assert sig.levels.stop_loss_786 > sig.levels.blended_entry
