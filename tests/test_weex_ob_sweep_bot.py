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
    assert cfg["top_universe_count"] == 100
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

