"""
research_10_strategies_100_coins.py

Systematic evaluation of 10 simple, theoretically sound trading strategies
across 100+ liquid altcoins to uncover coin-level behavioral archetypes.

Strict execution guidelines:
- Zero lookahead: signal generated on bar T close, executed on bar T+1 Open.
- Fixed transaction friction: 0.10% round-trip (taker/maker fee + slippage).
- Risk-adjusted and expectancy metrics tracked per coin x strategy pair.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data_cache/klines_multiyear_1d")
OUT_DIR = Path("results/strategy_coin_matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Filter out equity synthetic tokens and non-crypto symbols
EXCLUDE_SUBSTRINGS = ["ONUSDT", "牛来", "龙虾", "JMDT", "EURUSDT", "USD1USDT", "USUSDT"]

def get_clean_crypto_universe(min_bars=200):
    files = list(DATA_DIR.glob("*.parquet"))
    valid_symbols = []
    
    for f in files:
        sym = f.stem
        if any(sub in sym for sub in EXCLUDE_SUBSTRINGS):
            continue
        # Check file length quickly
        try:
            df = pd.read_parquet(f, columns=["close"])
            if len(df) >= min_bars:
                valid_symbols.append((sym, len(df), f))
        except Exception:
            continue
            
    # Sort by number of bars descending
    valid_symbols.sort(key=lambda x: x[1], reverse=True)
    return valid_symbols

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_backtest_for_symbol(df, btc_df=None):
    """
    Evaluates 10 strategies on a single coin dataframe.
    Returns a dict of metrics for each strategy.
    
    Fee model: 0.10% deduction per round trip.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n < 200:
        return {}
        
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    atr14 = calculate_atr(df, 14)
    rsi14 = calculate_rsi(c, 14)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    
    # 20-period Donchian
    donchian_high_20 = h.shift(1).rolling(20).max()
    donchian_low_10 = l.shift(1).rolling(10).min()
    donchian_low_20 = l.shift(1).rolling(20).min()
    
    results = {}
    
    # -------------------------------------------------------------
    # Helper simulation function for vector/event trades
    # -------------------------------------------------------------
    def evaluate_signals(entry_mask, exit_type="fixed_bars", exit_bars=5, trailing_exit_mask=None, fee=0.0010):
        trades = []
        i = 50
        while i < n - 1:
            if entry_mask.iloc[i]:
                entry_idx = i + 1  # Open of next bar
                entry_price = o.iloc[entry_idx]
                exit_idx = None
                exit_price = None
                
                if exit_type == "fixed_bars":
                    target_exit = min(entry_idx + exit_bars, n - 1)
                    exit_idx = target_exit
                    exit_price = c.iloc[exit_idx]
                elif exit_type == "trailing" and trailing_exit_mask is not None:
                    for j in range(entry_idx, min(entry_idx + 40, n)):
                        if trailing_exit_mask.iloc[j]:
                            exit_idx = j
                            exit_price = c.iloc[j]
                            break
                    if exit_idx is None:
                        exit_idx = min(entry_idx + 40, n - 1)
                        exit_price = c.iloc[exit_idx]
                        
                if entry_price > 0 and exit_price is not None:
                    gross_ret = (exit_price - entry_price) / entry_price
                    net_ret = gross_ret - fee
                    hold_time = exit_idx - entry_idx
                    trades.append(net_ret)
                    i = max(entry_idx, exit_idx) + 1
                    continue
            i += 1
            
        if len(trades) < 5:
            return {"trades": len(trades), "win_rate": 0.0, "total_pnl": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
            
        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        win_rate = len(wins) / len(trades)
        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-6
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 1.0
        expectancy = np.mean(trades)
        total_pnl = np.prod(1 + trades) - 1.0
        
        return {
            "trades": len(trades),
            "win_rate": round(win_rate * 100, 1),
            "total_pnl": round(total_pnl * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy * 100, 2)
        }

    # -------------------------------------------------------------
    # S1: Donchian 20-Day Breakout (Turtle Trend)
    # Entry: Close > 20d High. Exit: Close < 10d Low
    # -------------------------------------------------------------
    s1_entry = (c > donchian_high_20) & (c.shift(1) <= donchian_high_20.shift(1))
    s1_exit = c < donchian_low_10
    results["S1_Donchian_Breakout"] = evaluate_signals(s1_entry, exit_type="trailing", trailing_exit_mask=s1_exit)

    # -------------------------------------------------------------
    # S2: Swing-High Break & Retest
    # Prior swing high (10d local high), breakout, then pullback within 1.5% without breaking below it
    # -------------------------------------------------------------
    swing_high_10 = h.shift(2).rolling(10).max()
    broke_out = (c.shift(1) > swing_high_10.shift(1)) & (c.shift(2) <= swing_high_10.shift(2))
    retest = broke_out & (l <= swing_high_10 * 1.015) & (c >= swing_high_10 * 0.99)
    results["S2_Swing_Retest"] = evaluate_signals(retest, exit_type="fixed_bars", exit_bars=5)

    # -------------------------------------------------------------
    # S3: EMA Trend-Pullback (20/50 Ribbon)
    # 20 EMA > 50 EMA, Low dips to touch or pierce 20 EMA, Close finishes above 20 EMA
    # -------------------------------------------------------------
    ema_uptrend = ema20 > ema50
    ema_touch = (l <= ema20) & (c > ema20) & ema_uptrend
    results["S3_EMA_Pullback"] = evaluate_signals(ema_touch, exit_type="fixed_bars", exit_bars=5)

    # -------------------------------------------------------------
    # S4: Turtle Soup (False Breakdown Liquidity Sweep)
    # Price pierces 20d Low (stop run), but reverses to close back above the 20d Low
    # -------------------------------------------------------------
    s4_sweep = (l < donchian_low_20) & (c > donchian_low_20)
    results["S4_Turtle_Soup"] = evaluate_signals(s4_sweep, exit_type="fixed_bars", exit_bars=5)

    # -------------------------------------------------------------
    # S5: Bollinger Band Mean Reversion
    # Price touches or pierces lower BB (-2 std dev), closes back up with bullish candle
    # -------------------------------------------------------------
    s5_bb = (l <= lower_bb) & (c > o)
    s5_bb_exit = c >= sma20
    results["S5_BB_MeanReversion"] = evaluate_signals(s5_bb, exit_type="trailing", trailing_exit_mask=s5_bb_exit)

    # -------------------------------------------------------------
    # S6: RSI Extreme Compression & Snapback
    # RSI(14) was < 28 yesterday, crosses back above 30 today
    # -------------------------------------------------------------
    s6_rsi = (rsi14.shift(1) < 28) & (rsi14 >= 30)
    results["S6_RSI_Snapback"] = evaluate_signals(s6_rsi, exit_type="fixed_bars", exit_bars=5)

    # -------------------------------------------------------------
    # S7: NR7 Volatility Squeeze Breakout
    # Candle range is smallest of last 7 bars, enter on next day break of NR7 high
    # -------------------------------------------------------------
    candle_range = h - l
    is_nr7 = candle_range == candle_range.rolling(7).min()
    # We trigger when NR7 is confirmed
    results["S7_NR7_Expansion"] = evaluate_signals(is_nr7, exit_type="fixed_bars", exit_bars=3)

    # -------------------------------------------------------------
    # S8: Weekly Open Reclaim (Simplified to 7-day cyclical open)
    # -------------------------------------------------------------
    s8_reclaim = (l < o.shift(7)) & (c > o.shift(7)) & (c > o)
    results["S8_Weekly_Open_Reclaim"] = evaluate_signals(s8_reclaim, exit_type="fixed_bars", exit_bars=4)

    # -------------------------------------------------------------
    # S9: Relative Strength Divergence (vs BTC)
    # Altcoin 5d return is positive while BTC 5d return is negative
    # -------------------------------------------------------------
    if btc_df is not None:
        merged = pd.merge(df[["open_time", "close"]], btc_df[["open_time", "close"]], on="open_time", suffixes=("", "_btc"))
        alt_ret5 = merged["close"].pct_change(5)
        btc_ret5 = merged["close_btc"].pct_change(5)
        rs_divergence = (alt_ret5 > 0.03) & (btc_ret5 < -0.02)
        # Reindex back to df
        rs_mask = pd.Series(False, index=df.index)
        rs_mask.iloc[rs_divergence.index] = rs_divergence.values
        results["S9_RS_vs_BTC"] = evaluate_signals(rs_mask, exit_type="fixed_bars", exit_bars=5)
    else:
        results["S9_RS_vs_BTC"] = {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "profit_factor": 0.0, "expectancy": 0.0}

    # -------------------------------------------------------------
    # S10: High-Volume Exhaustion Wick (Pin Bar Absorption)
    # Lower wick >= 60% of total bar, volume > 2.0x 20d volume MA
    # -------------------------------------------------------------
    lower_wick = np.minimum(o, c) - l
    total_range = np.maximum(h - l, 1e-8)
    vol_ma20 = v.rolling(20).mean()
    s10_hammer = (lower_wick / total_range >= 0.55) & (v > 1.8 * vol_ma20)
    results["S10_Volume_Absorption"] = evaluate_signals(s10_hammer, exit_type="fixed_bars", exit_bars=5)

    return results

def main():
    print("Loading universe of cached crypto tokens...")
    universe = get_clean_crypto_universe(min_bars=250)
    # Limit to top 110 clean coins
    target_coins = universe[:110]
    print(f"Selected {len(target_coins)} clean crypto coins for evaluation.")
    
    # Load BTC for relative strength strategy
    btc_df = None
    btc_path = DATA_DIR / "BTCUSDT.parquet"
    if btc_path.exists():
        btc_df = pd.read_parquet(btc_path)
        
    records = []
    
    for sym, bar_count, fpath in target_coins:
        df = pd.read_parquet(fpath)
        res = run_backtest_for_symbol(df, btc_df=btc_df)
        if not res:
            continue
            
        for strat, metrics in res.items():
            records.append({
                "symbol": sym,
                "bars": bar_count,
                "strategy": strat,
                "trades": metrics["trades"],
                "win_rate": metrics["win_rate"],
                "expectancy_pct": metrics["expectancy"],
                "profit_factor": metrics["profit_factor"],
                "total_pnl_pct": metrics["total_pnl"]
            })
            
    df_results = pd.DataFrame(records)
    df_results.to_parquet(OUT_DIR / "strategy_coin_matrix.parquet", index=False)
    print(f"Saved {len(df_results)} strategy-coin evaluations to {OUT_DIR / 'strategy_coin_matrix.parquet'}")
    
    # Print SUI vs XRP head to head
    print("\n" + "="*80)
    print("HEAD-TO-HEAD: SUI vs XRP Across All 10 Strategies")
    print("="*80)
    sui_xrp = df_results[df_results["symbol"].isin(["SUIUSDT", "XRPUSDT"])].pivot(
        index="strategy", columns="symbol", values=["win_rate", "expectancy_pct", "profit_factor", "trades"]
    )
    print(sui_xrp.to_string())
    
    # Identify Archetypes:
    # 1. Trend/Momentum Score = Avg expectancy of S1 (Donchian), S2 (Retest), S3 (EMA Pullback)
    # 2. Mean Reversion Score = Avg expectancy of S4 (Turtle Soup), S5 (BB Reversion), S6 (RSI Snapback)
    pivot_exp = df_results.pivot(index="symbol", columns="strategy", values="expectancy_pct").fillna(0)
    pivot_trades = df_results.pivot(index="symbol", columns="strategy", values="trades").fillna(0)
    
    pivot_exp["trend_score"] = (pivot_exp["S1_Donchian_Breakout"] + pivot_exp["S2_Swing_Retest"] + pivot_exp["S3_EMA_Pullback"]) / 3.0
    pivot_exp["mean_reversion_score"] = (pivot_exp["S4_Turtle_Soup"] + pivot_exp["S5_BB_MeanReversion"] + pivot_exp["S6_RSI_Snapback"]) / 3.0
    pivot_exp["archetype_bias"] = pivot_exp["trend_score"] - pivot_exp["mean_reversion_score"]
    
    print("\n" + "="*80)
    print("TOP 15 MOMENTUM / TREND-FOLLOWING COINS (High Trend Score)")
    print("="*80)
    top_trend = pivot_exp.sort_values("trend_score", ascending=False)[["trend_score", "mean_reversion_score", "S1_Donchian_Breakout", "S2_Swing_Retest", "S3_EMA_Pullback"]].head(15)
    print(top_trend.to_string())
    
    print("\n" + "="*80)
    print("TOP 15 MEAN-REVERSION / LIQUIDITY SWEEP COINS (High Mean Reversion Score)")
    print("="*80)
    top_mr = pivot_exp.sort_values("mean_reversion_score", ascending=False)[["mean_reversion_score", "trend_score", "S4_Turtle_Soup", "S5_BB_MeanReversion", "S6_RSI_Snapback"]].head(15)
    print(top_mr.to_string())
    
    print("\n" + "="*80)
    print("SPECIFIC SPOTLIGHT: SUI vs XRP")
    print("="*80)
    if "SUIUSDT" in pivot_exp.index and "XRPUSDT" in pivot_exp.index:
        spotlight = pivot_exp.loc[["SUIUSDT", "XRPUSDT"], ["trend_score", "mean_reversion_score", "archetype_bias", "S1_Donchian_Breakout", "S3_EMA_Pullback", "S4_Turtle_Soup", "S6_RSI_Snapback"]]
        print(spotlight.to_string())

if __name__ == "__main__":
    main()
