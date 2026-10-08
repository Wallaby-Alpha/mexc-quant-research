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
    
    msg = format_market_neutral_telegram_alert(brackets, btc_metrics, ["SOL"], [], config)
    assert "Net Market Delta:" in msg
    assert "0.0% ($0.00)" in msg
    assert "PF_SOLUSD" in msg
    assert "PF_XBTUSD" in msg
    assert "NEW BUYS:</b> SOL" in msg
