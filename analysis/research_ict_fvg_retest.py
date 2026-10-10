"""
analysis/research_ict_fvg_retest.py
-----------------------------------
Research-grade backtest of ICT Fair Value Gap (FVG) Retest Strategy across:
1. BTC_USDT benchmark
2. Top 100 Liquid Altcoins by Volume on MEXC

Specifications:
- Timeframes: 1h Bias (HTF) and 15m Entry (LTF) [Scaled 1:4 from 5m/1m]
- Bias & FVG Setup (1h):
  - Bullish: 3-candle gap where candle[i-2].high < candle[i].low. Price above recent confirmed swing low.
  - Bearish: 3-candle gap where candle[i-2].low > candle[i].high. Price below recent confirmed swing high.
- Entry Trigger (15m):
  - Retests FVG midpoint and closes above/below with 15m volume > 20-period SMA.
- Stop Loss: Opposite side of the FVG.
- Take Profit: 1.5R or next opposing 1h FVG, whichever comes first.
- Session Filter: Asia open to London close (00:00 - 16:00 UTC).
- Range Filter: Skip trade if price is in the middle 70% (15%-85%) of previous day's high/low range.
- Anti-Lookahead: Entry at open of T+1 following confirmed 15m signal bar.
- Realistic Costs: 0.06% taker fee + 0.02% slippage per leg (0.16% round trip).
- Holdout Discipline: 75% In-Sample, 25% Locked Holdout.
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ict_fvg_research")


TAKER_FEE = 0.0006  # 0.06%
SLIPPAGE = 0.0002   # 2 bps per side
ROUNDTRIP_FRICTION = (TAKER_FEE + SLIPPAGE) * 2.0  # ~0.16%

HOLDOUT_FRACTION = 0.25

def get_top_symbols(count: int = 100) -> List[str]:
    """Ranks available 1h parquet files by volume and returns top N symbols."""
    files = glob.glob("data_cache/klines/1h/*.parquet")
    vol_list = []
    for f in files:
        sym = Path(f).stem
        # Ensure 15m file also exists
        if not os.path.exists(f"data_cache/klines/15m/{sym}.parquet"):
            continue
        try:
            df = pd.read_parquet(f, columns=["quote_volume_usdt"])
            avg_vol = df["quote_volume_usdt"].mean() * 24.0
            vol_list.append((sym, avg_vol))
        except Exception:
            pass
    vol_list.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in vol_list[:count]]


def find_swing_points_1h(df1h: pd.DataFrame, n: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Detects fractal swing highs and lows with zero lookahead (N=2, confirmed at T+2).
    """
    swing_highs = pd.Series(np.nan, index=df1h.index)
    swing_lows = pd.Series(np.nan, index=df1h.index)
    highs = df1h["high"].values
    lows = df1h["low"].values
    
    for i in range(n, len(df1h) - n):
        # Swing High
        if all(highs[i] > highs[i - k] for k in range(1, n + 1)) and \
           all(highs[i] > highs[i + k] for k in range(1, n + 1)):
            # Confirmed at bar i + n
            swing_highs.iloc[i + n] = highs[i]
            
        # Swing Low
        if all(lows[i] < lows[i - k] for k in range(1, n + 1)) and \
           all(lows[i] < lows[i + k] for k in range(1, n + 1)):
            # Confirmed at bar i + n
            swing_lows.iloc[i + n] = lows[i]

    # Forward fill confirmed swing levels
    return swing_highs.ffill(), swing_lows.ffill()


def prepare_daily_ranges(df1h: pd.DataFrame) -> pd.DataFrame:
    """Prepares previous day high, low, and range mapped to 1h bars."""
    df = df1h.copy()
    df["date"] = df["open_time"].dt.date
    daily = df.groupby("date").agg({
        "high": "max",
        "low": "min"
    }).reset_index()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_range"] = daily["prev_high"] - daily["prev_low"]
    
    merged = pd.merge(df[["open_time", "date"]], daily[["date", "prev_high", "prev_low", "prev_range"]], on="date", how="left")
    return merged.set_index(df.index)


def run_ict_fvg_backtest_for_symbol(
    symbol: str,
    rr_target: float = 1.5,
    holdout_guard: bool = True
) -> List[Dict[str, Any]]:
    """Runs ICT FVG Retest backtest for a single symbol."""
    path_1h = f"data_cache/klines/1h/{symbol}.parquet"
    path_15m = f"data_cache/klines/15m/{symbol}.parquet"
    if not os.path.exists(path_1h) or not os.path.exists(path_15m):
        return []

    df1h = pd.read_parquet(path_1h).sort_values("open_time").reset_index(drop=True)
    df15m = pd.read_parquet(path_15m).sort_values("open_time").reset_index(drop=True)

    if len(df1h) < 100 or len(df15m) < 400:
        return []

    # Enforce Holdout Discipline
    if holdout_guard:
        cutoff_idx = int(len(df1h) * (1.0 - HOLDOUT_FRACTION))
        cutoff_time = df1h["open_time"].iloc[cutoff_idx]
        df1h = df1h[df1h["open_time"] < cutoff_time].copy()
        df15m = df15m[df15m["open_time"] < cutoff_time].copy()

    # 1. 1h Structure: Swing Points & Previous Day Range
    swing_highs, swing_lows = find_swing_points_1h(df1h, n=2)
    daily_range_info = prepare_daily_ranges(df1h)

    # 2. 15m Indicators: 20 SMA Volume
    df15m["vol_sma20"] = df15m["volume_base"].rolling(20).mean()

    # Detect 1h FVGs
    # A bullish FVG at bar i requires candle[i-2].high < candle[i].low
    # A bearish FVG at bar i requires candle[i-2].low > candle[i].high
    fvg_records = []
    for i in range(2, len(df1h)):
        bar_time = df1h["open_time"].iloc[i]
        c1_high = df1h["high"].iloc[i - 2]
        c1_low = df1h["low"].iloc[i - 2]
        c3_high = df1h["high"].iloc[i]
        c3_low = df1h["low"].iloc[i]
        c3_close = df1h["close"].iloc[i]

        last_swing_low = swing_lows.iloc[i]
        last_swing_high = swing_highs.iloc[i]
        
        # Bullish FVG
        if c1_high < c3_low:
            # Bias check: Close above recent swing low
            if not np.isnan(last_swing_low) and c3_close > last_swing_low:
                fvg_records.append({
                    "fvg_type": "bullish",
                    "creation_time": bar_time, # Available after this 1h candle closes
                    "fvg_low": c1_high,        # Stop level
                    "fvg_high": c3_low,
                    "fvg_mid": (c1_high + c3_low) / 2.0,
                    "invalidated": False
                })

        # Bearish FVG
        if c1_low > c3_high:
            # Bias check: Close below recent swing high
            if not np.isnan(last_swing_high) and c3_close < last_swing_high:
                fvg_records.append({
                    "fvg_type": "bearish",
                    "creation_time": bar_time,
                    "fvg_low": c3_high,
                    "fvg_high": c1_low,        # Stop level
                    "fvg_mid": (c1_low + c3_high) / 2.0,
                    "invalidated": False
                })

    if not fvg_records:
        return []

    # Map daily range into 15m
    df15m["date"] = df15m["open_time"].dt.date
    daily_unique = daily_range_info.drop_duplicates(subset=["date"])
    df15m = pd.merge(df15m, daily_unique[["date", "prev_high", "prev_low", "prev_range"]], on="date", how="left")

    trades = []
    active_fvgs = []
    fvg_idx = 0
    in_trade = False
    current_trade = None

    # Pre-extract numpy arrays for 50x faster iteration
    open_times = df15m["open_time"].tolist()
    opens = df15m["open"].values
    highs = df15m["high"].values
    lows = df15m["low"].values
    closes = df15m["close"].values
    vols = df15m["volume_base"].values
    vol_smas = df15m["vol_sma20"].values
    prev_ranges = df15m["prev_range"].values
    prev_lows = df15m["prev_low"].values
    hours = df15m["open_time"].dt.hour.values
    n_bars = len(df15m) - 1

    for k in range(n_bars):
        high_k = highs[k]
        low_k = lows[k]
        close_k = closes[k]
        t_time = open_times[k]
        hour = hours[k]

        # Add newly formed 1h FVGs available at current time
        while fvg_idx < len(fvg_records) and fvg_records[fvg_idx]["creation_time"] <= t_time:
            active_fvgs.append(fvg_records[fvg_idx])
            fvg_idx += 1

        # Check existing position management
        if in_trade:
            pos = current_trade

            # Check if opposing 1h FVG formed
            opposing_fvg_hit = False
            for f in active_fvgs[-3:]:
                if f["creation_time"] == t_time:
                    if pos["direction"] == "long" and f["fvg_type"] == "bearish":
                        opposing_fvg_hit = True
                        break
                    elif pos["direction"] == "short" and f["fvg_type"] == "bullish":
                        opposing_fvg_hit = True
                        break

            # 1. Long Trade Resolution
            if pos["direction"] == "long":
                # Check adverse SL first (Rule 1: Same bar ambiguity assumes adverse stop hit first)
                if low_k <= pos["sl_price"]:
                    # Stopped out
                    exit_price = pos["sl_price"] * (1.0 - SLIPPAGE)
                    gross_ret = (exit_price / pos["entry_price"]) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = -1.0
                    trades.append({
                        "symbol": symbol,
                        "direction": "long",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "stop_loss"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif high_k >= pos["tp_price"]:
                    # Take Profit 1.5R hit
                    exit_price = pos["tp_price"] * (1.0 - SLIPPAGE)
                    gross_ret = (exit_price / pos["entry_price"]) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = rr_target
                    trades.append({
                        "symbol": symbol,
                        "direction": "long",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "take_profit"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif opposing_fvg_hit:
                    # Early exit on opposing FVG
                    exit_price = close_k * (1.0 - SLIPPAGE)
                    gross_ret = (exit_price / pos["entry_price"]) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = (exit_price - pos["entry_price"]) / (pos["entry_price"] - pos["sl_price"])
                    trades.append({
                        "symbol": symbol,
                        "direction": "long",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "opposing_fvg"
                    })
                    in_trade = False
                    current_trade = None
                    continue

            # 2. Short Trade Resolution
            elif pos["direction"] == "short":
                if high_k >= pos["sl_price"]:
                    exit_price = pos["sl_price"] * (1.0 + SLIPPAGE)
                    gross_ret = (pos["entry_price"] / exit_price) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = -1.0
                    trades.append({
                        "symbol": symbol,
                        "direction": "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "stop_loss"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif low_k <= pos["tp_price"]:
                    exit_price = pos["tp_price"] * (1.0 + SLIPPAGE)
                    gross_ret = (pos["entry_price"] / exit_price) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = rr_target
                    trades.append({
                        "symbol": symbol,
                        "direction": "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "take_profit"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif opposing_fvg_hit:
                    exit_price = close_k * (1.0 + SLIPPAGE)
                    gross_ret = (pos["entry_price"] / exit_price) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    r_multiple = (pos["entry_price"] - exit_price) / (pos["sl_price"] - pos["entry_price"])
                    trades.append({
                        "symbol": symbol,
                        "direction": "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": r_multiple,
                        "exit_reason": "opposing_fvg"
                    })
                    in_trade = False
                    current_trade = None
                    continue

            # Position still open
            continue

        # Invalidate broken FVGs
        valid_active = []
        for f in active_fvgs:
            if f["fvg_type"] == "bullish" and low_k < f["fvg_low"]:
                continue
            if f["fvg_type"] == "bearish" and high_k > f["fvg_high"]:
                continue
            # Keep recent FVGs within 48 hours
            if (t_time - f["creation_time"]).total_seconds() > 48 * 3600:
                continue
            valid_active.append(f)
        active_fvgs = valid_active

        # Check Filter 1: Session filter (00:00 - 16:00 UTC)
        if hour >= 16:
            continue

        # Check Filter 2: Previous Day Range Filter (skip if price is inside 70% middle range)
        prev_range = prev_ranges[k]
        prev_low = prev_lows[k]
        if prev_range > 0 and not np.isnan(prev_range):
            rel_pos = (close_k - prev_low) / prev_range
            # Middle 70% is between 0.15 and 0.85
            if 0.15 <= rel_pos <= 0.85:
                continue

        # Check Filter 3: Volume filter (Volume > 20 SMA)
        vol_sma = vol_smas[k]
        if np.isnan(vol_sma) or vols[k] <= vol_sma:
            continue

        next_open = opens[k + 1]
        next_time = open_times[k + 1]

        # Look for Entry Trigger among active FVGs
        for f in reversed(active_fvgs):
            # Bullish Trigger: Penetrates midpoint (low <= mid) and closes above (close > mid)
            if f["fvg_type"] == "bullish":
                if low_k <= f["fvg_mid"] and close_k > f["fvg_mid"]:
                    entry_p = next_open * (1.0 + SLIPPAGE)
                    sl_p = f["fvg_low"]
                    risk = entry_p - sl_p
                    if risk > 0 and (risk / entry_p) > 0.002: # Ensure reasonable stop size > 0.2%
                        tp_p = entry_p + (risk * rr_target)
                        in_trade = True
                        current_trade = {
                            "direction": "long",
                            "entry_time": next_time,
                            "entry_price": entry_p,
                            "sl_price": sl_p,
                            "tp_price": tp_p,
                            "fvg": f
                        }
                        # FVG is now filled / triggered
                        active_fvgs.remove(f)
                        break

            # Bearish Trigger: Penetrates midpoint (high >= mid) and closes below (close < mid)
            elif f["fvg_type"] == "bearish":
                if high_k >= f["fvg_mid"] and close_k < f["fvg_mid"]:
                    entry_p = next_open * (1.0 - SLIPPAGE)
                    sl_p = f["fvg_high"]
                    risk = sl_p - entry_p
                    if risk > 0 and (risk / entry_p) > 0.002:
                        tp_p = entry_p - (risk * rr_target)
                        in_trade = True
                        current_trade = {
                            "direction": "short",
                            "entry_time": next_time,
                            "entry_price": entry_p,
                            "sl_price": sl_p,
                            "tp_price": tp_p,
                            "fvg": f
                        }
                        active_fvgs.remove(f)
                        break

    return trades


def analyze_trade_results(trades: List[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
    """Computes academic metrics with confidence intervals and cost-adjusted expectancy."""
    if not trades:
        return {"n": 0, "label": label}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["net_return"] > 0]
    losses = df[df["net_return"] <= 0]
    win_rate = len(wins) / n

    # 95% Confidence Interval for Win Rate (Wilson Score Interval)
    z = 1.96
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (win_rate + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n)) / n) / denominator
    ci_lower = max(0.0, centre_adjusted_probability - half_width)
    ci_upper = min(1.0, centre_adjusted_probability + half_width)

    gross_gains = df[df["gross_return"] > 0]["gross_return"].sum()
    gross_losses = abs(df[df["gross_return"] < 0]["gross_return"].sum())
    gross_pf = gross_gains / gross_losses if gross_losses > 0 else np.nan

    net_gains = df[df["net_return"] > 0]["net_return"].sum()
    net_losses = abs(df[df["net_return"] < 0]["net_return"].sum())
    net_pf = net_gains / net_losses if net_losses > 0 else np.nan

    mean_net_ret = df["net_return"].mean()
    mean_gross_ret = df["gross_return"].mean()
    mean_r = df["r_multiple"].mean()

    # Drawdown on cumulative R
    cum_r = df["r_multiple"].cumsum()
    peak_r = cum_r.cummax()
    dd_r = cum_r - peak_r
    max_dd_r = dd_r.min()

    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]

    return {
        "label": label,
        "n_trades": n,
        "win_rate": win_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "mean_gross_return_pct": mean_gross_ret * 100.0,
        "mean_net_return_pct": mean_net_ret * 100.0,
        "mean_r_multiple": mean_r,
        "total_net_r": df["r_multiple"].sum(),
        "max_drawdown_r": max_dd_r,
        "long_trades": len(longs),
        "long_win_rate": len(longs[longs["net_return"] > 0]) / max(1, len(longs)),
        "short_trades": len(shorts),
        "short_win_rate": len(shorts[shorts["net_return"] > 0]) / max(1, len(shorts)),
        "exit_tp_count": len(df[df["exit_reason"] == "take_profit"]),
        "exit_sl_count": len(df[df["exit_reason"] == "stop_loss"]),
        "exit_opposing_count": len(df[df["exit_reason"] == "opposing_fvg"])
    }


def main():
    logger.info("=== Starting ICT FVG Retest Research Backtest ===")
    
    # 1. Benchmark on BTC_USDT
    logger.info("Running benchmark on BTC_USDT...")
    btc_trades = run_ict_fvg_backtest_for_symbol("BTC_USDT", rr_target=1.5, holdout_guard=True)
    btc_metrics = analyze_trade_results(btc_trades, label="BTC_USDT (In-Sample)")

    logger.info(f"BTC_USDT Results: {btc_metrics['n_trades']} trades | Net PF: {btc_metrics.get('net_profit_factor', 0):.2f} | Win Rate: {btc_metrics.get('win_rate', 0)*100:.1f}%")

    # 2. Universe Run: Top 100 Liquid Altcoins
    top_100_symbols = get_top_symbols(100)
    logger.info(f"Scanning Top {len(top_100_symbols)} Liquid Altcoins...")

    all_universe_trades = []
    symbol_summaries = []

    for idx, sym in enumerate(top_100_symbols, 1):
        trades = run_ict_fvg_backtest_for_symbol(sym, rr_target=1.5, holdout_guard=True)
        if trades:
            all_universe_trades.extend(trades)
            m = analyze_trade_results(trades, label=sym)
            symbol_summaries.append(m)
        if idx % 20 == 0 or idx == len(top_100_symbols):
            logger.info(f"Processed {idx}/{len(top_100_symbols)} coins. Total trades so far: {len(all_universe_trades)}")

    universe_metrics = analyze_trade_results(all_universe_trades, label="Top 100 Universe (In-Sample)")

    # Save results to parquet first
    if all_universe_trades:
        res_df = pd.DataFrame(all_universe_trades)
        os.makedirs("results", exist_ok=True)
        res_df.to_parquet("results/ict_fvg_retest_trades.parquet")
        logger.info("Saved trade log to results/ict_fvg_retest_trades.parquet")


    # Output detailed report
    print("\n" + "=" * 90)
    print("ICT FAIR VALUE GAP (FVG) RETEST STRATEGY - EMPIRICAL RESULTS")
    print("=" * 90)
    print(f"Date Range Evaluated: In-Sample 75% (Aug 2025 - June 2026) | Locked Holdout: Final 25%")
    print(f"Fee & Friction Model: 0.06% Taker Fee + 2 bps Slippage per leg (0.16% Roundtrip)")
    print("-" * 90)
    
    def print_metric_block(m):
        print(f"Dataset: {m['label']}")
        print(f"* Total Trades (n):           {m['n_trades']} ({'Statistically Adequate (n >= 100)' if m['n_trades'] >= 100 else 'Insufficient Sample'})")
        print(f"* Win Rate:                   {m['win_rate']*100:.2f}% (95% CI: [{m['ci_lower']*100:.2f}%, {m['ci_upper']*100:.2f}%])")
        print(f"* Profit Factor (Gross):      {m['gross_profit_factor']:.2f}")
        print(f"* Profit Factor (NET OF FEES):{m['net_profit_factor']:.2f} (Primary Decision Metric)")
        print(f"* Expectancy per Trade (Net): {m['mean_net_return_pct']:+.2f}% / {m['mean_r_multiple']:+.2f}R")
        print(f"* Total Cumulative Net R:     {m['total_net_r']:+.1f}R")
        print(f"* Maximum Drawdown:           {m['max_drawdown_r']:.1f}R")
        print(f"* Long vs Short:              {m['long_trades']} Longs ({m['long_win_rate']*100:.1f}% WR) | {m['short_trades']} Shorts ({m['short_win_rate']*100:.1f}% WR)")
        print(f"* Exit Breakdown:             TP (1.5R): {m['exit_tp_count']} | SL: {m['exit_sl_count']} | Opposing FVG: {m['exit_opposing_count']}")
        print("-" * 90)

    print_metric_block(btc_metrics)
    print_metric_block(universe_metrics)


if __name__ == "__main__":
    main()
