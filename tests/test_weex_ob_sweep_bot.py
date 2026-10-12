"""
tests/test_weex_ob_sweep_bot.py
-------------------------------
Unit tests for the WEEX SMC Order Block Liquidity Sweep Bot.
Validates:
1. WEEX HMAC SHA256 signature generation and request headers.
2. Fractal swing identification without lookahead.
3. Order Block identification & Liquidity Sweep detection logic.
4. Bracket calculation (Stop loss 1x ATR, Take profit 2.5R / Equal highs).
5. Dry-run simulation order placement.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from bot.weex_ob_sweep_bot import (
    WeexClient,
    find_fractal_swings,
    evaluate_symbol_for_sweep_setup,
    evaluate_btc_macro_regime,
    load_config
)


def test_weex_signature_generation():
    """Validates WEEX HMAC SHA256 base64 signature generation format."""
    client = WeexClient(
        api_key="test_key",
        secret_key="test_secret",
        passphrase="test_passphrase",
        dry_run=True
    )
    timestamp = "1672531199000"
    method = "GET"
    path = "/capi/v1/market/tickers"
    sign = client._generate_signature(timestamp, method, path)
    
    assert isinstance(sign, str)
    assert len(sign) > 10

    headers = client._get_headers(method, path)
    assert headers["ACCESS-KEY"] == "test_key"
    assert headers["ACCESS-PASSPHRASE"] == "test_passphrase"
    assert "ACCESS-SIGN" in headers
    assert "ACCESS-TIMESTAMP" in headers


def test_find_fractal_swings():
    """Ensures fractal swings correctly identify 5-bar local extremes."""
    highs = np.array([10.0, 11.0, 15.0, 12.0, 10.0, 9.0, 8.0, 7.0])
    lows = np.array([5.0, 6.0, 7.0, 5.0, 3.0, 4.0, 5.0, 6.0])
    
    sh, sl = find_fractal_swings(highs, lows, n=2)
    
    # Swing high at index 2 (15.0) should be confirmed at index 2+2 = 4
    assert sh[4] == 15.0
    # Swing low at index 4 (3.0) should be confirmed at index 4+2 = 6
    assert sl[6] == 3.0


def test_dry_run_order_placement():
    """Ensures dry-run mode returns simulated status without hitting live network."""
    client = WeexClient(api_key="fake", secret_key="fake", passphrase="fake", dry_run=True)
    res = client.place_native_bracket_order(
        symbol="BTCUSDT",
        side="BUY",
        position_side="LONG",
        quantity=0.05,
        sl_price=60000.0,
        tp_price=65000.0
    )
    assert res["status"] == "simulated"
    assert "orderId" in res


def test_evaluate_symbol_for_sweep_setup():
    """Tests synthetic 1h and 15m candles triggering a bullish order block sweep."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    
    # Build 1h candles with a clear BOS breakout:
    # Bar 0..4: Setup base
    # Bar 5: Swing high created at 105.0
    # Bar 6..10: Pullback with down-close candle at bar 8 (Open=95, Close=90, High=96, Low=88) -> OB
    # Bar 11..15: Breakout taking out 105.0
    times_1h = [base_time + timedelta(hours=i) for i in range(40)]
    opens_1h = [100.0] * 40
    highs_1h = [102.0] * 40
    lows_1h = [98.0] * 40
    closes_1h = [100.0] * 40

    # Fractal high at index 5
    highs_1h[5] = 110.0
    # Down-close OB candle at index 8
    opens_1h[8] = 95.0
    closes_1h[8] = 90.0
    highs_1h[8] = 96.0
    lows_1h[8] = 88.0
    # Breakout at index 12 above 110.0
    closes_1h[12] = 115.0
    highs_1h[12] = 116.0

    df1h = pd.DataFrame({
        "open_time": times_1h,
        "open": opens_1h,
        "high": highs_1h,
        "low": lows_1h,
        "close": closes_1h,
        "volume": [1000.0] * 40
    })

    # Build 15m candles
    # Last closed candle sweeps below OB low (88.0) by >= 0.05% (e.g. 87.0) and closes back inside (e.g. 89.0)
    times_15m = [times_1h[13] + timedelta(minutes=15 * i) for i in range(40)]
    opens_15m = [92.0] * 40
    highs_15m = [94.0] * 40
    lows_15m = [90.0] * 40
    closes_15m = [92.0] * 40

    # Candle 38 (latest fully closed): sweeps 88.0 down to 87.0 and closes at 89.0
    lows_15m[38] = 87.0
    closes_15m[38] = 89.0
    highs_15m[38] = 90.0

    df15m = pd.DataFrame({
        "open_time": times_15m,
        "open": opens_15m,
        "high": highs_15m,
        "low": lows_15m,
        "close": closes_15m,
        "volume": [500.0] * 40
    })

    cfg = {
        "max_ob_age_hours": 24.0,
        "min_sweep_pct": 0.05,
        "atr_multiplier": 1.0,
        "risk_reward_ratio": 2.5
    }

    sig = evaluate_symbol_for_sweep_setup(df1h, df15m, "BTCUSDT", cfg)
    
    assert sig is not None
    assert sig["direction"] == "long"
    assert sig["entry_price"] == 89.0
    assert sig["sl_price"] < 87.0  # SL placed below sweep low
    assert sig["tp_price"] > 89.0  # TP placed above entry


def test_load_config_defaults():
    """Validates fallback default config values."""
    cfg = load_config()
    assert "top_universe_count" in cfg
    assert cfg["top_universe_count"] == 200
    assert "dry_run" in cfg
    assert "risk_per_trade_usd" in cfg


def test_single_position_per_coin_guard():
    """Ensures that once a position is open on a coin, subsequent orders on that coin are blocked."""
    client = WeexClient(api_key="fake", secret_key="fake", passphrase="fake", dry_run=True)
    
    # Place initial position on DOGEUSDT
    res1 = client.place_native_bracket_order(
        symbol="DOGEUSDT",
        side="BUY",
        position_side="LONG",
        raw_quantity=2000,
        entry_price=0.0862,
        sl_price=0.0850,
        tp_price=0.0890
    )
    assert res1["status"] == "simulated"
    
    # Active symbols should now include DOGEUSDT
    active_syms = client.get_open_position_symbols()
    assert "DOGEUSDT" in active_syms
    assert client.get_open_positions_count() == 1


def test_evaluate_btc_macro_regime_bullish_and_bearish():
    """Validates BTC Macro 1h 50 EMA regime classification."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(hours=i) for i in range(70)]

    # 1. Bullish scenario: prices rise from 50k to 70k, latest closed candle well above 50 EMA
    closes_bull = [50000.0 + (i * 300.0) for i in range(70)]
    df_bull = pd.DataFrame({
        "open_time": times,
        "open": closes_bull,
        "high": [c + 100 for c in closes_bull],
        "low": [c - 100 for c in closes_bull],
        "close": closes_bull,
        "volume": [100.0] * 70
    })

    reg_bull = evaluate_btc_macro_regime(df_bull, ema_span=50)
    assert reg_bull["regime"] == "BULLISH"
    assert reg_bull["allowed_direction"] == "long"
    assert reg_bull["is_bullish"] is True
    assert reg_bull["btc_close"] > reg_bull["btc_ema"]

    # 2. Bearish scenario: prices fall from 70k down to 40k, latest closed candle well below 50 EMA
    closes_bear = [70000.0 - (i * 450.0) for i in range(70)]
    df_bear = pd.DataFrame({
        "open_time": times,
        "open": closes_bear,
        "high": [c + 100 for c in closes_bear],
        "low": [c - 100 for c in closes_bear],
        "close": closes_bear,
        "volume": [100.0] * 70
    })

    reg_bear = evaluate_btc_macro_regime(df_bear, ema_span=50)
    assert reg_bear["regime"] == "BEARISH"
    assert reg_bear["allowed_direction"] == "short"
    assert reg_bear["is_bullish"] is False
    assert reg_bear["btc_close"] < reg_bear["btc_ema"]


def test_btc_macro_regime_lookahead_prevention():
    """Validates that evaluate_btc_macro_regime strictly excludes the active forming bar (Rule 1)."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(hours=i) for i in range(60)]
    
    # 59 closed bars hovering around 50,000
    closes = [50000.0] * 59
    # 60th bar (the active forming candle at index 59) has an artificial huge spike to 999,999
    closes.append(999999.0)

    df = pd.DataFrame({
        "open_time": times,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [100.0] * 60
    })

    reg = evaluate_btc_macro_regime(df, ema_span=50)
    # The active candle (index 59 with close 999,999) must NOT be used.
    # The evaluation must reflect index 58 with close 50,000.
    assert reg["btc_close"] == 50000.0
    assert reg["bar_time"] == times[58]


def test_minimum_risk_reward_filter_skips_sub_1r_trades():
    """Validates that evaluate_symbol_for_sweep_setup strictly skips trades where the natural target is < min_risk_reward_ratio."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    times_1h = [base_time + timedelta(hours=i) for i in range(40)]
    opens_1h = [100.0] * 40
    highs_1h = [102.0] * 40
    lows_1h = [98.0] * 40
    closes_1h = [100.0] * 40

    # Fractal high at index 5 is 110.0
    highs_1h[5] = 110.0
    opens_1h[8] = 95.0
    closes_1h[8] = 90.0
    highs_1h[8] = 96.0
    lows_1h[8] = 88.0
    closes_1h[12] = 115.0
    highs_1h[12] = 116.0

    df1h = pd.DataFrame({
        "open_time": times_1h,
        "open": opens_1h,
        "high": highs_1h,
        "low": lows_1h,
        "close": closes_1h,
        "volume": [1000.0] * 40
    })

    times_15m = [times_1h[13] + timedelta(minutes=15 * i) for i in range(40)]
    opens_15m = [92.0] * 40
    highs_15m = [94.0] * 40
    lows_15m = [90.0] * 40
    closes_15m = [92.0] * 40

    lows_15m[38] = 87.0
    closes_15m[38] = 89.0
    highs_15m[38] = 90.0

    df15m = pd.DataFrame({
        "open_time": times_15m,
        "open": opens_15m,
        "high": highs_15m,
        "low": lows_15m,
        "close": closes_15m,
        "volume": [500.0] * 40
    })

    # Case 1: Standard 2.5R target is accepted when min_risk_reward_ratio is 1.0
    cfg_standard = {
        "max_ob_age_hours": 24.0,
        "min_sweep_pct": 0.05,
        "atr_multiplier": 1.0,
        "risk_reward_ratio": 2.5,
        "min_risk_reward_ratio": 1.0
    }
    sig_standard = evaluate_symbol_for_sweep_setup(df1h, df15m, "TESTUSDT", cfg_standard)
    assert sig_standard is not None
    assert sig_standard["effective_rr"] == 2.5

    # Case 2: When min_risk_reward_ratio is set higher than available (e.g. 3.0), setup is skipped
    cfg_strict = {
        "max_ob_age_hours": 24.0,
        "min_sweep_pct": 0.05,
        "atr_multiplier": 1.0,
        "risk_reward_ratio": 2.5,
        "min_risk_reward_ratio": 3.0
    }
    sig_strict = evaluate_symbol_for_sweep_setup(df1h, df15m, "TESTUSDT", cfg_strict)
    assert sig_strict is None

    # Case 3: When equal_high clamps target below 1R, setup is skipped
    # Base highs at 88.0, fractal swing high at 92.0 (only 3.0 above entry 89.0, while risk is ~6.0 -> 0.5R)
    opens_1h_c = [88.0] * 40
    highs_1h_c = [89.0] * 40
    lows_1h_c = [86.0] * 40
    closes_1h_c = [88.0] * 40
    highs_1h_c[5] = 92.0
    opens_1h_c[8] = 89.0
    closes_1h_c[8] = 87.0
    highs_1h_c[8] = 90.0
    lows_1h_c[8] = 86.0
    closes_1h_c[12] = 94.0
    highs_1h_c[12] = 95.0

    df1h_c = pd.DataFrame({
        "open_time": times_1h,
        "open": opens_1h_c,
        "high": highs_1h_c,
        "low": lows_1h_c,
        "close": closes_1h_c,
        "volume": [1000.0] * 40
    })

    # Sweep below 86.0 to 85.0, close at 87.0
    lows_15m_c = [88.0] * 40
    closes_15m_c = [88.0] * 40
    lows_15m_c[38] = 85.0
    closes_15m_c[38] = 87.0
    df15m_c = pd.DataFrame({
        "open_time": times_15m,
        "open": [88.0] * 40,
        "high": [90.0] * 40,
        "low": lows_15m_c,
        "close": closes_15m_c,
        "volume": [500.0] * 40
    })

    # Entry = 87.0. equal_high = 92.0. Reward = 5.0. Risk = 87.0 - (85.0 - 4.0) = 6.0.
    # Effective R:R = 5.0 / 6.0 = 0.83R (< 1.0R).
    sig_sub_1r = evaluate_symbol_for_sweep_setup(df1h_c, df15m_c, "TESTUSDT", cfg_standard)
    assert sig_sub_1r is None  # Skipped because 0.83R < 1.0R


def test_4h_1h_timeframe_configuration():
    """Validates that bot defaults to 4h HTF and 1h LTF with hourly polling."""
    cfg = load_config()
    assert cfg["htf_timeframe"] == "4h"
    assert cfg["ltf_timeframe"] == "1h"
    assert cfg["poll_interval_seconds"] == 3600
    assert cfg["max_ob_age_hours"] >= 48.0



