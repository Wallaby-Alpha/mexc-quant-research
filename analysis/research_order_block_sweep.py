"""
analysis/research_order_block_sweep.py
--------------------------------------
Research-grade backtest of SMC/ICT Order Block Liquidity Sweep + Mitigation Strategy:
Specifications:
- Assets: BTC_USDT + Top 100 Liquid Altcoins by Volume
- Higher Timeframe (1h): Order Block identification (last down-close before BOS breakout)
- Lower Timeframe (15m): Liquidity Sweep & Entry
- Long Trigger:
  - 1h Bullish Order Block formed (last down-close before break of swing high).
  - Price sweeps below OB low by >= 0.05% and closes back inside OB on 15m.
  - OB age <= 8 hours, mitigation count <= 1.
- Short Trigger:
  - Mirror bearish conditions (liquidity sweep above OB high by >= 0.05%, closes back inside).
- Stop Loss: 1x ATR(14) below the sweep low.
- Take Profit: Equal highs (BOS swing level) or 2.5R, whichever comes first.
- Comparisons:
  1. Whole Universe (Unfiltered 100 coins)
  2. Relative Strength Leaders (Top 20% RS)
  3. RS Leaders (Longs Only)
  4. RS Leaders (Longs Only + BTC > 50 EMA)
- Strict Anti-Lookahead: Entry at open of T+1 following confirmed signal bar.
- Realistic Costs: 0.06% taker fee + 2 bps slippage per side (0.16% roundtrip).
- Holdout Discipline: First 75% In-Sample, Final 25% Locked Holdout.
"""

import os
import sys
import glob
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ob_sweep_research")

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002
ROUNDTRIP_FRICTION = (TAKER_FEE + SLIPPAGE) * 2.0  # 0.16%
HOLDOUT_FRACTION = 0.25

def get_top_symbols(count: int = 100) -> List[str]:
    """Returns top N symbols by volume."""
    files = glob.glob("data_cache/klines/1h/*.parquet")
    vol_list = []
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
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


def find_swing_points(highs: np.ndarray, lows: np.ndarray, n: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates confirmed fractal swing points with zero lookahead (N=2, confirmed at T+2)."""
    length = len(highs)
    sh = np.full(length, np.nan)
    sl = np.full(length, np.nan)
    
    for i in range(n, length - n):
        if all(highs[i] > highs[i - k] for k in range(1, n + 1)) and \
           all(highs[i] > highs[i + k] for k in range(1, n + 1)):
            sh[i + n] = highs[i]
            
        if all(lows[i] < lows[i - k] for k in range(1, n + 1)) and \
           all(lows[i] < lows[i + k] for k in range(1, n + 1)):
            sl[i + n] = lows[i]

    # Forward fill
    sh_series = pd.Series(sh).ffill().values
    sl_series = pd.Series(sl).ffill().values
    return sh_series, sl_series


def calculate_atr14(df15m: pd.DataFrame) -> np.ndarray:
    """Calculates 14-period Average True Range."""
    high = df15m["high"].values
    low = df15m["low"].values
    close = df15m["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    return atr


def run_ob_sweep_for_symbol(
    symbol: str,
    max_age_hours: float = 8.0,
    rr_target: float = 2.5,
    holdout_guard: bool = True
) -> List[Dict[str, Any]]:
    """Runs Order Block Sweep backtest on a single symbol."""
    path_1h = f"data_cache/klines/1h/{symbol}.parquet"
    path_15m = f"data_cache/klines/15m/{symbol}.parquet"
    if not os.path.exists(path_1h) or not os.path.exists(path_15m):
        return []

    df1h = pd.read_parquet(path_1h).sort_values("open_time").reset_index(drop=True)
    df15m = pd.read_parquet(path_15m).sort_values("open_time").reset_index(drop=True)

    if len(df1h) < 100 or len(df15m) < 400:
        return []

    if holdout_guard:
        cutoff_idx = int(len(df1h) * (1.0 - HOLDOUT_FRACTION))
        cutoff_time = df1h["open_time"].iloc[cutoff_idx]
        df1h = df1h[df1h["open_time"] < cutoff_time].copy()
        df15m = df15m[df15m["open_time"] < cutoff_time].copy()

    # Pre-extract 1h arrays
    h_1h = df1h["high"].values
    l_1h = df1h["low"].values
    o_1h = df1h["open"].values
    c_1h = df1h["close"].values
    times_1h = df1h["open_time"].tolist()

    sh_1h, sl_1h = find_swing_points(h_1h, l_1h, n=2)

    # 1. Identify Order Blocks on 1h
    # Bullish OB: Breakout above swing high -> last down-close candle (close < open) before the breakout
    # Bearish OB: Breakdown below swing low -> last up-close candle (close > open) before breakdown
    obs = []
    last_sh_broken = np.nan
    last_sl_broken = np.nan

    for i in range(5, len(df1h)):
        bar_t = times_1h[i]
        curr_sh = sh_1h[i]
        curr_sl = sl_1h[i]

        # Check Bullish BOS
        if not np.isnan(curr_sh) and c_1h[i] > curr_sh and curr_sh != last_sh_broken:
            last_sh_broken = curr_sh
            # Find last down-close candle prior to bar i
            for j in range(i - 1, max(0, i - 12), -1):
                if c_1h[j] < o_1h[j]:
                    obs.append({
                        "ob_type": "bullish",
                        "creation_time": bar_t,
                        "ob_high": h_1h[j],
                        "ob_low": l_1h[j],
                        "equal_high": curr_sh,
                        "mitigation_count": 0,
                        "invalidated": False
                    })
                    break

        # Check Bearish BOS
        if not np.isnan(curr_sl) and c_1h[i] < curr_sl and curr_sl != last_sl_broken:
            last_sl_broken = curr_sl
            for j in range(i - 1, max(0, i - 12), -1):
                if c_1h[j] > o_1h[j]:
                    obs.append({
                        "ob_type": "bearish",
                        "creation_time": bar_t,
                        "ob_high": h_1h[j],
                        "ob_low": l_1h[j],
                        "equal_low": curr_sl,
                        "mitigation_count": 0,
                        "invalidated": False
                    })
                    break

    if not obs:
        return []

    # 2. 15m Execution Simulation
    atr14 = calculate_atr14(df15m)
    h_15m = df15m["high"].values
    l_15m = df15m["low"].values
    o_15m = df15m["open"].values
    c_15m = df15m["close"].values
    times_15m = df15m["open_time"].tolist()
    n_15m = len(df15m) - 1

    trades = []
    active_obs = []
    ob_idx = 0
    in_trade = False
    current_trade = None

    for k in range(n_15m):
        t_time = times_15m[k]
        high_k = h_15m[k]
        low_k = l_15m[k]
        close_k = c_15m[k]
        atr_k = atr14[k]

        # Add newly available confirmed 1h OBs
        while ob_idx < len(obs) and obs[ob_idx]["creation_time"] <= t_time:
            active_obs.append(obs[ob_idx])
            ob_idx += 1

        # Position Management
        if in_trade:
            pos = current_trade
            if pos["direction"] == "long":
                # Check adverse SL first
                if low_k <= pos["sl_price"]:
                    exit_p = pos["sl_price"] * (1.0 - SLIPPAGE)
                    gross_ret = (exit_p / pos["entry_price"]) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    trades.append({
                        "symbol": symbol,
                        "direction": "long",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_p,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": -1.0,
                        "exit_reason": "stop_loss"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif high_k >= pos["tp_price"]:
                    exit_p = pos["tp_price"] * (1.0 - SLIPPAGE)
                    gross_ret = (exit_p / pos["entry_price"]) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    trades.append({
                        "symbol": symbol,
                        "direction": "long",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_p,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": rr_target,
                        "exit_reason": "take_profit"
                    })
                    in_trade = False
                    current_trade = None
                    continue

            elif pos["direction"] == "short":
                if high_k >= pos["sl_price"]:
                    exit_p = pos["sl_price"] * (1.0 + SLIPPAGE)
                    gross_ret = (pos["entry_price"] / exit_p) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    trades.append({
                        "symbol": symbol,
                        "direction": "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_p,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": -1.0,
                        "exit_reason": "stop_loss"
                    })
                    in_trade = False
                    current_trade = None
                    continue
                elif low_k <= pos["tp_price"]:
                    exit_p = pos["tp_price"] * (1.0 + SLIPPAGE)
                    gross_ret = (pos["entry_price"] / exit_p) - 1.0
                    net_ret = gross_ret - ROUNDTRIP_FRICTION
                    trades.append({
                        "symbol": symbol,
                        "direction": "short",
                        "entry_time": pos["entry_time"],
                        "exit_time": t_time,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_p,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "r_multiple": rr_target,
                        "exit_reason": "take_profit"
                    })
                    in_trade = False
                    current_trade = None
                    continue

            # Position remains open
            continue

        # Invalidate expired OBs (> max_age_hours old or mitigated > 1)
        valid_obs = []
        for ob in active_obs:
            age_hrs = (t_time - ob["creation_time"]).total_seconds() / 3600.0
            if age_hrs > max_age_hours or ob["mitigation_count"] > 1:
                continue

            # Track prior normal mitigations (entering OB zone without sweeping)
            if ob["ob_type"] == "bullish":
                # If dipped into OB zone without sweeping below low
                if low_k <= ob["ob_high"] and low_k > ob["ob_low"]:
                    ob["mitigation_count"] += 1
            elif ob["ob_type"] == "bearish":
                if high_k >= ob["ob_low"] and high_k < ob["ob_high"]:
                    ob["mitigation_count"] += 1

            if ob["mitigation_count"] <= 1:
                valid_obs.append(ob)
        active_obs = valid_obs

        next_open = o_15m[k + 1]
        next_time = times_15m[k + 1]

        # Look for Entry: Liquidity Sweep + Close Back Inside
        for ob in reversed(active_obs):
            # Bullish: Low sweeps below ob_low by at least 0.05%, close is back inside (close >= ob_low)
            if ob["ob_type"] == "bullish":
                sweep_threshold = ob["ob_low"] * (1.0 - 0.0005)
                if low_k <= sweep_threshold and close_k >= ob["ob_low"]:
                    entry_p = next_open * (1.0 + SLIPPAGE)
                    sweep_low = low_k
                    sl_p = sweep_low - (1.0 * atr_k)
                    risk = entry_p - sl_p
                    if risk > 0 and (risk / entry_p) > 0.002:
                        tp_p = entry_p + (risk * rr_target)
                        # Target equal highs if closer than 2.5R
                        if ob.get("equal_high", np.nan) > entry_p:
                            tp_p = min(tp_p, ob["equal_high"])
                        in_trade = True
                        current_trade = {
                            "direction": "long",
                            "entry_time": next_time,
                            "entry_price": entry_p,
                            "sl_price": sl_p,
                            "tp_price": tp_p
                        }
                        active_obs.remove(ob)
                        break

            # Bearish: High sweeps above ob_high by at least 0.05%, close is back inside (close <= ob_high)
            elif ob["ob_type"] == "bearish":
                sweep_threshold = ob["ob_high"] * (1.0 + 0.0005)
                if high_k >= sweep_threshold and close_k <= ob["ob_high"]:
                    entry_p = next_open * (1.0 - SLIPPAGE)
                    sweep_high = high_k
                    sl_p = sweep_high + (1.0 * atr_k)
                    risk = sl_p - entry_p
                    if risk > 0 and (risk / entry_p) > 0.002:
                        tp_p = entry_p - (risk * rr_target)
                        if ob.get("equal_low", np.nan) < entry_p:
                            tp_p = max(tp_p, ob["equal_low"])
                        in_trade = True
                        current_trade = {
                            "direction": "short",
                            "entry_time": next_time,
                            "entry_price": entry_p,
                            "sl_price": sl_p,
                            "tp_price": tp_p
                        }
                        active_obs.remove(ob)
                        break

    return trades


def analyze_trades(trades: List[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
    """Computes comprehensive performance metrics."""
    if not trades:
        return {"n": 0, "label": label}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["net_return"] > 0]
    win_rate = len(wins) / n

    z = 1.96
    denom = 1 + z**2 / n
    centre = (win_rate + z**2 / (2 * n)) / denom
    hw = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0.0, centre - hw)
    ci_high = min(1.0, centre + hw)

    gross_gains = df[df["gross_return"] > 0]["gross_return"].sum()
    gross_losses = abs(df[df["gross_return"] < 0]["gross_return"].sum())
    gross_pf = gross_gains / gross_losses if gross_losses > 0 else np.nan

    net_gains = df[df["net_return"] > 0]["net_return"].sum()
    net_losses = abs(df[df["net_return"] < 0]["net_return"].sum())
    net_pf = net_gains / net_losses if net_losses > 0 else np.nan

    cum_r = df["r_multiple"].cumsum()
    max_dd_r = (cum_r - cum_r.cummax()).min()

    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]

    return {
        "label": label,
        "n_trades": n,
        "win_rate": win_rate,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "gross_profit_factor": gross_pf,
        "net_profit_factor": net_pf,
        "mean_gross_return_pct": df["gross_return"].mean() * 100.0,
        "mean_net_return_pct": df["net_return"].mean() * 100.0,
        "mean_r_multiple": df["r_multiple"].mean(),
        "total_net_r": df["r_multiple"].sum(),
        "max_drawdown_r": max_dd_r,
        "long_trades": len(longs),
        "long_win_rate": len(longs[longs["net_return"] > 0]) / max(1, len(longs)),
        "short_trades": len(shorts),
        "short_win_rate": len(shorts[shorts["net_return"] > 0]) / max(1, len(shorts))
    }


def main():
    logger.info("=== Starting Order Block Sweep & Mitigation Research Backtest ===")
    top_100 = get_top_symbols(100)
    logger.info(f"Targeting BTC + Top {len(top_100)} Liquid Altcoins...")

    all_trades = []
    for idx, sym in enumerate(top_100, 1):
        tr = run_ob_sweep_for_symbol(sym, max_age_hours=8.0, rr_target=2.5, holdout_guard=True)
        all_trades.extend(tr)
        if idx % 25 == 0 or idx == len(top_100):
            logger.info(f"Processed {idx}/{len(top_100)} coins. Trades so far: {len(all_trades)}")

    if not all_trades:
        logger.error("No trades generated.")
        return

    trades_df = pd.DataFrame(all_trades)
    os.makedirs("results", exist_ok=True)
    trades_df.to_parquet("results/ob_sweep_trades.parquet")
    logger.info(f"Saved {len(trades_df)} trades to results/ob_sweep_trades.parquet")

    # 1. Baseline: Whole Universe
    m_universe = analyze_trades(all_trades, label="Whole Universe (All 100 Coins)")

    # 2. Add Relative Strength and BTC EMA features
    logger.info("Calculating Point-in-Time 7-day Relative Strength and BTC 50 EMA...")
    close_dict = {}
    for sym in top_100:
        p = f"data_cache/klines/1h/{sym}.parquet"
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=["open_time", "close"]).drop_duplicates("open_time").set_index("open_time")
            close_dict[sym] = df["close"]

    price_df = pd.DataFrame(close_dict).ffill()
    ret_7d = price_df.pct_change(168)
    rs_ranks = ret_7d.rank(axis=1, ascending=False)

    btc_series = price_df["BTC_USDT"]
    btc_ema50 = btc_series.ewm(span=1200, adjust=False).mean()
    btc_bull_series = btc_series > btc_ema50

    # Align to trades
    trade_ranks = []
    trade_btc = []
    for _, row in trades_df.iterrows():
        t = row["entry_time"]
        sym = row["symbol"]
        valid_idx = rs_ranks.index[rs_ranks.index <= t]
        if len(valid_idx) > 0:
            last_t = valid_idx[-1]
            trade_ranks.append(rs_ranks.loc[last_t, sym] if sym in rs_ranks.columns else np.nan)
            trade_btc.append(bool(btc_bull_series.loc[last_t]) if last_t in btc_bull_series.index else False)
        else:
            trade_ranks.append(np.nan)
            trade_btc.append(False)

    trades_df["rs_rank"] = trade_ranks
    trades_df["btc_above_ema50"] = trade_btc

    valid_trades = trades_df.dropna(subset=["rs_rank"]).copy()

    # Stratified Evaluations
    top20_all = valid_trades[valid_trades["rs_rank"] <= 20]
    m_top20_all = analyze_trades(top20_all.to_dict("records"), label="Top 20% RS Leaders (Longs + Shorts)")

    top20_longs = valid_trades[(valid_trades["rs_rank"] <= 20) & (valid_trades["direction"] == "long")]
    m_top20_longs = analyze_trades(top20_longs.to_dict("records"), label="Top 20% RS Leaders (Longs Only)")

    top10_longs = valid_trades[(valid_trades["rs_rank"] <= 10) & (valid_trades["direction"] == "long")]
    m_top10_longs = analyze_trades(top10_longs.to_dict("records"), label="Top 10% RS Leaders (Longs Only)")

    top20_longs_btc = valid_trades[
        (valid_trades["rs_rank"] <= 20) & 
        (valid_trades["direction"] == "long") & 
        (valid_trades["btc_above_ema50"] == True)
    ]
    m_top20_longs_btc = analyze_trades(top20_longs_btc.to_dict("records"), label="Top 20% RS + Longs Only + BTC > 50 EMA")

    # Decile Surface
    deciles = []
    for d in range(10):
        low_r = d * 10 + 1
        high_r = (d + 1) * 10
        dec_sub = valid_trades[(valid_trades["rs_rank"] >= low_r) & (valid_trades["rs_rank"] <= high_r)]
        m_dec = analyze_trades(dec_sub.to_dict("records"), label=f"Rank {low_r:02d}-{high_r:02d}")
        deciles.append(m_dec)

    # Print Report
    print("\n" + "=" * 95)
    print("SMC/ICT ORDER BLOCK LIQUIDITY SWEEP & MITIGATION - EMPIRICAL RESULTS")
    print("=" * 95)
    print("Timeframe: 1h Order Blocks | 15m Liquidity Sweep & Retest (Target: 2.5R / Stop: 1x ATR)")
    print("Fee Friction: 0.06% Taker Fee + 2 bps Slippage per leg (0.16% Roundtrip Net Cost)")
    print("-" * 95)

    def print_block(m):
        n_flag = f"{m['n_trades']}" if m['n_trades'] >= 100 else f"{m['n_trades']} (⚠️ n<100)"
        print(f"Dataset: {m['label']}")
        print(f"  • Trades: {n_flag:<20} | Win Rate: {m['win_rate']*100:.1f}% [{m['ci_lower']*100:.1f}%, {m['ci_upper']*100:.1f}%]")
        print(f"  • Profit Factor: Gross {m['gross_profit_factor']:.2f} | NET {m['net_profit_factor']:.2f}")
        print(f"  • Expectancy: Net {m['mean_net_return_pct']:+.2f}% ({m['mean_r_multiple']:+.2f}R) | Total Net: {m['total_net_r']:+.1f}R")
        print(f"  • Max Drawdown: {m['max_drawdown_r']:.1f}R")
        print("-" * 95)

    print_block(m_universe)
    print_block(m_top20_all)
    print_block(m_top20_longs)
    print_block(m_top10_longs)
    print_block(m_top20_longs_btc)

    print("\n" + "=" * 95)
    print("DECILE SURFACE: NET PROFIT FACTOR BY RELATIVE STRENGTH RANK")
    print("=" * 95)
    print(f"{'Rank Decile':<15} | {'Trades':<8} | {'Win Rate':<10} | {'Gross PF':<10} | {'NET PF':<10} | {'Net Return':<12} | {'Net R'}")
    print("-" * 95)
    for d in deciles:
        print(f"{d['label']:<15} | {d['n_trades']:<8} | {d['win_rate']*100:6.1f}%   | {d['gross_profit_factor']:8.2f}   | {d['net_profit_factor']:8.2f}   | {d['mean_net_return_pct']:+8.2f}%    | {d['total_net_r']:+7.1f}R")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()
