"""
analysis/research_ict_fvg_relative_strength.py
----------------------------------------------
Empirical study testing the impact of Relative Strength (RS) on the ICT FVG Retest strategy:
Hypothesis: Filtering ICT FVG retest entries to only altcoins with high 7-day relative strength
(and applying BTC > 50-day EMA) converts negative net expectancy into a robust positive edge.

Methodology:
1. Point-in-time cross-sectional 7-day return ranking for all 100 liquid altcoins at each 1h bar.
2. BTC 50-day EMA macro trend indicator calculated point-in-time.
3. Zero lookahead: Trade at entry_time only knows the RS rank from the most recent closed 1h bar.
4. Stratifications evaluated:
   - Stratum 0: Baseline (Unfiltered Top 100)
   - Stratum 1: Top 20% RS (Rank <= 20) - Both Longs and Shorts
   - Stratum 2: Top 20% RS (Rank <= 20) - Longs Only
   - Stratum 3: Top 10% RS (Rank <= 10) - Longs Only
   - Stratum 4: Top 20% RS + BTC > 50 EMA - Longs Only
   - Stratum 5: Full Decile Rank Surface (Rank 1-10, 11-20, ..., 91-100) to test monotonicity.
"""

import os
import sys
import glob
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ict_fvg_rs")

def analyze_trade_subset(df: pd.DataFrame, label: str = "") -> Dict[str, Any]:
    """Calculates statistical metrics for a subset of trades."""
    n = len(df)
    if n == 0:
        return {"n": 0, "label": label}

    wins = df[df["net_return"] > 0]
    win_rate = len(wins) / n

    # Wilson Score 95% Confidence Interval
    z = 1.96
    denominator = 1 + z**2 / n
    centre_adj = (win_rate + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n)) / n) / denominator
    ci_lower = max(0.0, centre_adj - half_width)
    ci_upper = min(1.0, centre_adj + half_width)

    gross_gains = df[df["gross_return"] > 0]["gross_return"].sum()
    gross_losses = abs(df[df["gross_return"] < 0]["gross_return"].sum())
    gross_pf = gross_gains / gross_losses if gross_losses > 0 else np.nan

    net_gains = df[df["net_return"] > 0]["net_return"].sum()
    net_losses = abs(df[df["net_return"] < 0]["net_return"].sum())
    net_pf = net_gains / net_losses if net_losses > 0 else np.nan

    mean_net_ret = df["net_return"].mean()
    mean_gross_ret = df["gross_return"].mean()
    mean_r = df["r_multiple"].mean()
    total_net_r = df["r_multiple"].sum()

    cum_r = df["r_multiple"].cumsum()
    peak_r = cum_r.cummax()
    max_dd_r = (cum_r - peak_r).min()

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
        "total_net_r": total_net_r,
        "max_drawdown_r": max_dd_r,
        "long_trades": len(longs),
        "long_win_rate": len(longs[longs["net_return"] > 0]) / max(1, len(longs)),
        "short_trades": len(shorts),
        "short_win_rate": len(shorts[shorts["net_return"] > 0]) / max(1, len(shorts))
    }

def main():
    logger.info("Loading trades log from results/ict_fvg_retest_trades.parquet...")
    trades_path = "results/ict_fvg_retest_trades.parquet"
    if not os.path.exists(trades_path):
        logger.error(f"{trades_path} not found. Run analysis/research_ict_fvg_retest.py first.")
        return

    trades_df = pd.read_parquet(trades_path)
    logger.info(f"Loaded {len(trades_df)} trades across {trades_df['symbol'].nunique()} symbols.")

    # 1. Build 1h Price Matrix for 7-day Relative Strength Ranking
    logger.info("Building point-in-time 7-day Relative Strength matrix across 100 liquid coins...")
    files = glob.glob("data_cache/klines/1h/*.parquet")
    close_dict = {}
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        try:
            df = pd.read_parquet(f, columns=["open_time", "close"]).drop_duplicates("open_time").set_index("open_time")
            close_dict[sym] = df["close"]
        except Exception:
            pass

    price_df = pd.DataFrame(close_dict).ffill()
    
    # 7-day rolling return (168 1h bars)
    ret_7d_df = price_df.pct_change(168)
    
    # Cross-sectional rank (1 = strongest, 100 = weakest)
    rs_rank_df = ret_7d_df.rank(axis=1, ascending=False)

    # 2. Compute BTC 50-day EMA
    btc_series = price_df["BTC_USDT"].copy()
    # 50 days = 50 * 24 = 1200 1h bars
    btc_ema50 = btc_series.ewm(span=1200, adjust=False).mean()
    btc_bullish_series = btc_series > btc_ema50

    # 3. Align RS Rank and BTC EMA to each Trade at Entry Time (Zero Lookahead)
    logger.info("Aligning point-in-time features to trades...")
    rs_ranks = []
    btc_regimes = []

    for _, row in trades_df.iterrows():
        t = row["entry_time"]
        sym = row["symbol"]
        
        # Most recent closed 1h bar <= entry_time
        valid_idx = rs_rank_df.index[rs_rank_df.index <= t]
        if len(valid_idx) > 0:
            last_t = valid_idx[-1]
            rank_val = rs_rank_df.loc[last_t, sym] if sym in rs_rank_df.columns else np.nan
            btc_bull = bool(btc_bullish_series.loc[last_t]) if last_t in btc_bullish_series.index else False
        else:
            rank_val = np.nan
            btc_bull = False

        rs_ranks.append(rank_val)
        btc_regimes.append(btc_bull)

    trades_df["rs_rank"] = rs_ranks
    trades_df["btc_above_ema50"] = btc_regimes

    # Filter out trades where RS rank was not computable (early history before 7 days)
    valid_trades = trades_df.dropna(subset=["rs_rank"]).copy()
    logger.info(f"Valid trades with confirmed RS ranking: {len(valid_trades)}")

    # 4. Stratified Evaluations
    m_baseline = analyze_trade_subset(valid_trades, label="Baseline (All 100 Coins)")
    
    top20_both = valid_trades[valid_trades["rs_rank"] <= 20]
    m_top20_both = analyze_trade_subset(top20_both, label="Top 20% RS (Longs + Shorts)")

    top20_longs = valid_trades[(valid_trades["rs_rank"] <= 20) & (valid_trades["direction"] == "long")]
    m_top20_longs = analyze_trade_subset(top20_longs, label="Top 20% RS (Longs Only)")

    top10_longs = valid_trades[(valid_trades["rs_rank"] <= 10) & (valid_trades["direction"] == "long")]
    m_top10_longs = analyze_trade_subset(top10_longs, label="Top 10% RS (Longs Only)")

    top20_longs_btc = valid_trades[
        (valid_trades["rs_rank"] <= 20) & 
        (valid_trades["direction"] == "long") & 
        (valid_trades["btc_above_ema50"] == True)
    ]
    m_top20_longs_btc = analyze_trade_subset(top20_longs_btc, label="Top 20% RS + Longs Only + BTC > 50 EMA")

    top10_longs_btc = valid_trades[
        (valid_trades["rs_rank"] <= 10) & 
        (valid_trades["direction"] == "long") & 
        (valid_trades["btc_above_ema50"] == True)
    ]
    m_top10_longs_btc = analyze_trade_subset(top10_longs_btc, label="Top 10% RS + Longs Only + BTC > 50 EMA")

    # Decile Surface
    deciles = []
    for d in range(10):
        low_rank = d * 10 + 1
        high_rank = (d + 1) * 10
        dec_sub = valid_trades[(valid_trades["rs_rank"] >= low_rank) & (valid_trades["rs_rank"] <= high_rank)]
        m_dec = analyze_trade_subset(dec_sub, label=f"Rank {low_rank:02d}-{high_rank:02d}")
        deciles.append(m_dec)

    # Output Detailed Terminal Report
    print("\n" + "=" * 95)
    print("ICT FAIR VALUE GAP (FVG) RETEST - RELATIVE STRENGTH FILTER STUDY")
    print("=" * 95)
    print("Fee Friction: 0.06% Taker Fee + 2 bps Slippage per leg (0.16% Roundtrip Net Cost)")
    print("Zero-Lookahead: Point-in-time 7d Sharpe/Return ranking on fully closed 1h bars")
    print("-" * 95)

    def print_row(m):
        n_flag = f"{m['n_trades']}" if m['n_trades'] >= 100 else f"{m['n_trades']} (⚠️ n<100)"
        print(f"Dataset: {m['label']}")
        print(f"  • Trades: {n_flag:<20} | Win Rate: {m['win_rate']*100:.1f}% [{m['ci_lower']*100:.1f}%, {m['ci_upper']*100:.1f}%]")
        print(f"  • Profit Factor: Gross {m['gross_profit_factor']:.2f} | NET {m['net_profit_factor']:.2f}")
        print(f"  • Expectancy: Net {m['mean_net_return_pct']:+.2f}% ({m['mean_r_multiple']:+.2f}R) | Total Net: {m['total_net_r']:+.1f}R")
        print(f"  • Max Drawdown: {m['max_drawdown_r']:.1f}R")
        print("-" * 95)

    print_row(m_baseline)
    print_row(m_top20_both)
    print_row(m_top20_longs)
    print_row(m_top10_longs)
    print_row(m_top20_longs_btc)
    print_row(m_top10_longs_btc)

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
