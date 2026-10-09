import pytest
from scanner.kraken_market_neutral_scanner import (
    determine_rebalance,
    calculate_market_neutral_brackets,
    format_market_neutral_telegram_alert,
    DEFAULT_BREAKOUT_UNIVERSE
)

def test_universe_contains_65_coins():
    assert len(DEFAULT_BREAKOUT_UNIVERSE) == 65
    assert "SOL" in DEFAULT_BREAKOUT_UNIVERSE
    assert "SUI" in DEFAULT_BREAKOUT_UNIVERSE
    assert "RENDER" in DEFAULT_BREAKOUT_UNIVERSE

def test_zero_net_delta_math():
    config = {
        "account_size_usd": 100000.0,
        "gross_exposure_pct": 25.0, # 12.5% Long, 12.5% Short
        "stop_loss_pct": 3.5,
        "take_profit_1_pct": 8.0,
        "take_profit_2_pct": 15.0
    }
    long_coins = [
        {"coin": "SOL", "symbol": "PF_SOLUSD", "sharpe": 2.5, "return_7d": 0.15, "mark_price": 200.0},
        {"coin": "SUI", "symbol": "PF_SUIUSD", "sharpe": 2.2, "return_7d": 0.12, "mark_price": 2.50},
        {"coin": "RENDER", "symbol": "PF_RENDERUSD", "sharpe": 1.9, "return_7d": 0.10, "mark_price": 6.0}
    ]
    btc_metrics = {
        "mark_price": 80000.0,
        "ema50": 76000.0,
        "is_bullish": True
    }
    
    brackets = calculate_market_neutral_brackets(long_coins, btc_metrics, config)
    
    assert brackets["total_long_usd"] == 12500.0
    assert brackets["total_short_usd"] == 12500.0
    assert brackets["net_market_delta_usd"] == 0.0
    assert len(brackets["long_brackets"]) == 3
    
    # Each long coin gets exactly 12500 / 3 = 4166.67 USD
    for b in brackets["long_brackets"]:
        assert abs(b["target_usd"] - 4166.67) < 0.1
        assert b["stop_loss_price"] < b["mark_price"]
        assert b["take_profit_1_price"] > b["mark_price"]

    # Short hedge has equal dollar amount
    sh = brackets["short_hedge"]
    assert sh["symbol"] == "PF_XBTUSD"
    assert sh["target_usd"] == 12500.0
    assert sh["target_units"] == 12500.0 / 80000.0
    assert sh["stop_loss_price"] > sh["mark_price"] # Stop loss above mark price for a short!

def test_turnover_smoothing():
    scored_coins = [
        {"coin": "COIN_A", "sharpe": 3.0},
        {"coin": "COIN_B", "sharpe": 2.8},
        {"coin": "COIN_C", "sharpe": 2.5},
        {"coin": "COIN_D", "sharpe": 2.2},
        {"coin": "COIN_E", "sharpe": 2.0},
        {"coin": "COIN_F", "sharpe": 1.8},
        {"coin": "COIN_G", "sharpe": 1.5},
    ]
    current_state = {
        "active_longs": ["COIN_D", "COIN_E", "COIN_G"]
    }
    
    # COIN_D is rank 4 (<= 6), COIN_E is rank 5 (<= 6) -> Retained!
    # COIN_G is rank 7 (> 6) -> Exited!
    # COIN_A is rank 1 -> New Entry!
    selected, new_entries, exits = determine_rebalance(scored_coins, current_state, top_k=3, rank_exit_buffer=6)
    
    selected_names = [c["coin"] for c in selected]
    assert "COIN_D" in selected_names
    assert "COIN_E" in selected_names
    assert "COIN_A" in selected_names
    assert "COIN_G" not in selected_names
    assert new_entries == ["COIN_A"]
    assert exits == ["COIN_G"]

def test_telegram_alert_generation():
    config = {
        "account_size_usd": 100000.0,
        "gross_exposure_pct": 25.0,
        "top_k": 3,
        "stop_loss_pct": 3.5,
        "take_profit_1_pct": 8.0,
        "take_profit_2_pct": 15.0
    }
    brackets = {
        "total_long_usd": 12500.0,
        "total_short_usd": 12500.0,
        "long_brackets": [
            {
                "coin": "SOL", "symbol": "PF_SOLUSD", "sharpe": 2.5, "return_7d": 0.15,
                "mark_price": 200.0, "target_usd": 4166.67, "target_units": 20.83,
                "stop_loss_price": 193.0, "take_profit_1_price": 216.0, "take_profit_2_price": 230.0
            }
        ],
        "short_hedge": {
            "symbol": "PF_XBTUSD", "mark_price": 80000.0, "target_usd": 12500.0,
            "target_units": 0.15625, "stop_loss_price": 82800.0, "take_profit_price": 76000.0
        }
    }
    btc_metrics = {"mark_price": 80000.0, "ema50": 76000.0, "is_bullish": True}
    breadth_metrics = {"breadth_pct": 45.0, "coins_above_ema": 27, "total_coins": 60, "ema_period": 20}
    regime = {
        "state": "STATE_2_HEDGED",
        "name": "🛡️ STATE 2: HEDGED (Selective Market)",
        "action": "Deploy Long Top 3 + 100% BTC Hedge.",
        "execute_short_hedge": True
    }
    
    msg = format_market_neutral_telegram_alert(brackets, btc_metrics, breadth_metrics, regime, ["SOL"], [], config)
    assert "MARKET BREADTH" in msg
    assert "Altcoin Breadth (>20 EMA):</b> <code>45.0%</code>" in msg
    assert "STATE 2: HEDGED" in msg
    assert "PF_SOLUSD" in msg
    assert "PF_XBTUSD" in msg
    assert "NEW BUYS:</b> SOL" in msg


def test_determine_regime_states():
    from scanner.kraken_market_neutral_scanner import determine_regime
    config = {"breadth_expansion_threshold": 50.0, "regime_mode": "auto"}

    # 1. State 1: Flat when BTC < 50 EMA
    btc_bearish = {"mark_price": 70000.0, "ema50": 75000.0, "is_bullish": False}
    breadth_high = {"breadth_pct": 70.0, "coins_above_ema": 42, "total_coins": 60}
    r1 = determine_regime(btc_bearish, breadth_high, config)
    assert r1["state"] == "STATE_1_FLAT"
    assert r1["execute_short_hedge"] is False

    # 2. State 2: Hedged when BTC > 50 EMA and Breadth < 50%
    btc_bullish = {"mark_price": 80000.0, "ema50": 75000.0, "is_bullish": True}
    breadth_low = {"breadth_pct": 40.0, "coins_above_ema": 24, "total_coins": 60}
    r2 = determine_regime(btc_bullish, breadth_low, config)
    assert r2["state"] == "STATE_2_HEDGED"
    assert r2["execute_short_hedge"] is True

    # 3. State 3: Naked Long when BTC > 50 EMA and Breadth >= 50%
    r3 = determine_regime(btc_bullish, breadth_high, config)
    assert r3["state"] == "STATE_3_NAKED_LONG"
    assert r3["execute_short_hedge"] is False

    # 4. Overrides
    cfg_override = {"regime_mode": "force_naked_long"}
    r4 = determine_regime(btc_bearish, breadth_low, cfg_override)
    assert r4["state"] == "STATE_3_NAKED_LONG"

