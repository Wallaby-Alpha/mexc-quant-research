"""
run_downside_resilience_experiment.py
--------------------------------------
Backtest & Experiment: "The Coin That Refuses to Fall on Bad News"

Objective & Specifications:
1. Identify actual BTC market stress episodes (BTC drops >= 3% within 12h, with 24h cooldown).
2. Rank liquid alts by Downside Capture Ratio (DCR = Alt Return / BTC Return over identical stress period).
3. Filter/select:
   - Experimental Group: Top Downside-Resilient coins (lowest DCR, refused to break down).
   - Control Group: Ordinary Relative-Strength (RS) Leaders (top 24h gainers preceding the crash).
4. Entry Confirmation:
   - BTC confirms stabilization (first 1H close >= close of stress episode end-bar).
   - Alt confirms strength (alt close on confirmation bar >= alt close at stress episode end-bar).
   - Entry taken at the OPEN of the bar immediately following confirmation.
5. Evaluation:
   - Measure forward returns across 4h, 12h, and 24h horizons.
   - Benchmark Downside Resilient coins vs. Ordinary RS Leaders vs. Market Median.
   - Compute Win Rate, Average Return, Net Return (factoring taker fees & slippage), and 95% CI.
   - Enforce Rule 5 holdout discipline (lock holdout period until explicitly evaluated).
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from data.cache import ParquetCache
from strategy.downside_resilience import BTCStressDetector, ResilienceRanker, StressEpisode

# Parameters
HOLD_HORIZONS = [4, 12, 24] # hours
TOP_K = 5 # Number of coins selected per basket
BTC_STRESS_DROP = 0.03 # 3% drop
TAKER_FEE = 0.0002 # 0.02% taker fee
SLIPPAGE = 0.0005 # 5 bps slippage per side (0.10% round-trip friction total)
ROUNDTRIP_COST = 2 * (TAKER_FEE + SLIPPAGE) # 0.14%

TRADFI_KEYWORDS = [
    "STOCK", "500", "100", "30", "OIL", "ALUMINUM", "COPPER", "NICKEL", "LEAD", 
    "DRAM", "EWY", "KORU", "SOXL", "SOXS", "TQQQ", "SPY", "XAUT", "PAXG", 
    "SI_USDT", "XDP_USDT", "TESLA_USDT", "NVIDIA_USDT", "NGAS"
]

def is_crypto_alt(symbol: str) -> bool:
    if symbol in ["BTC_USDT", "ETH_USDT", "BTC", "ETH"]:
        return False
    if any(kw in symbol for kw in TRADFI_KEYWORDS):
        return False
    return True

def load_data(cache: ParquetCache):
    print("Loading 1H klines from data_cache...")
    symbols = cache.list_cached_symbols("1h")
    crypto_symbols = [s for s in symbols if is_crypto_alt(s)]
    print(f"Genuine crypto altcoin universe: {len(crypto_symbols)} symbols")
    
    panel_1h: Dict[str, pd.DataFrame] = {}
    # Also load BTC
    df_btc = cache.load_klines("BTC_USDT", "1h")
    df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
    panel_1h["BTC_USDT"] = df_btc.sort_values("open_time").reset_index(drop=True)
    
    for sym in crypto_symbols:
        df = cache.load_klines(sym, "1h")
        if df.empty or len(df) < 500:
            continue
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        panel_1h[sym] = df.sort_values("open_time").reset_index(drop=True)
        
    print(f"Loaded {len(panel_1h)} valid assets (including BTC_USDT).")
    return panel_1h

def run_experiment():
    cache = ParquetCache()
    panel_1h = load_data(cache)
    
    if "BTC_USDT" not in panel_1h:
        print("Error: BTC_USDT klines not found.")
        return

    df_btc = panel_1h["BTC_USDT"]
    
    # Strictly respect holdout discipline: guard in-sample vs holdout
    holdout_start = pd.Timestamp("2026-07-01T00:00:00Z")
    df_btc_train = df_btc[df_btc["open_time"] < holdout_start].copy()
    
    print(f"\n--- Identifying BTC Stress Episodes (In-Sample: up to {holdout_start.date()}) ---")
    detector = BTCStressDetector(btc_drop_pct=BTC_STRESS_DROP, stress_window_bars=12, cooldown_bars=24)
    episodes = detector.detect(df_btc_train)
    print(f"Total stress episodes detected: {len(episodes)}")
    
    ranker = ResilienceRanker(exclude_btc=True)
    
    # Results trackers
    records_resilient = []
    records_rs_leaders = []
    records_market_median = []
    
    df_btc_indexed = df_btc.set_index("open_time")
    
    for ep in episodes:
        ranked_resilient, ranked_rs = ranker.rank_episode(ep, panel_1h)
        if len(ranked_resilient) < 15 or len(ranked_rs) < 15:
            continue
            
        # BTC Stabilization Check:
        # Check if BTC produces a 1H close >= ep.btc_end_price within 6 bars after stress end
        post_btc = df_btc_indexed[df_btc_indexed.index > ep.end_bar_time]["close"]
        if len(post_btc) == 0:
            continue
            
        btc_confirm_bar = None
        for bar_t, btc_c in post_btc.iloc[:6].items():
            if btc_c >= ep.btc_end_price:
                btc_confirm_bar = bar_t
                break
                
        if btc_confirm_bar is None:
            # BTC kept cascading without stabilization; abort setup
            continue
            
        # Experimental Group: Top-K Downside Resilient coins
        # Must also confirm alt strength: alt close at confirmation bar >= alt close at stress episode end
        selected_resilient = []
        for cand in ranked_resilient:
            sym = cand.symbol
            df_sym = panel_1h[sym].set_index("open_time")
            if btc_confirm_bar not in df_sym.index:
                continue
            alt_confirm_close = df_sym.loc[btc_confirm_bar, "close"]
            if alt_confirm_close >= cand.alt_end_price:
                selected_resilient.append((sym, cand.dcr, cand.alt_return))
                if len(selected_resilient) >= TOP_K:
                    break
                    
        # Control Group: Top-K Ordinary Relative Strength Leaders (Pre-crash momentum)
        selected_rs = []
        for cand in ranked_rs:
            sym = cand["symbol"]
            df_sym = panel_1h[sym].set_index("open_time")
            if btc_confirm_bar not in df_sym.index:
                continue
            alt_confirm_close = df_sym.loc[btc_confirm_bar, "close"]
            # To be fair, also check if alt confirms strength
            if alt_confirm_close >= cand["end_px"]:
                selected_rs.append((sym, cand["rs_ret_24h"]))
                if len(selected_rs) >= TOP_K:
                    break
                    
        if len(selected_resilient) < 3 or len(selected_rs) < 3:
            continue

        # Evaluate Forward Returns:
        # Entry bar is the bar immediately following the confirmation bar (Open price of next bar)
        post_confirm_btc = df_btc_indexed[df_btc_indexed.index > btc_confirm_bar]
        if post_confirm_btc.empty:
            continue
        entry_time = post_confirm_btc.index[0]
        
        # 1. Evaluate Resilient
        for sym, dcr, drop_ret in selected_resilient:
            df_sym = panel_1h[sym].set_index("open_time")
            if entry_time not in df_sym.index:
                continue
            entry_px = df_sym.loc[entry_time, "open"]
            sub_post = df_sym[df_sym.index >= entry_time]["close"]
            
            row = {"episode_id": ep.episode_id, "symbol": sym, "dcr": dcr, "entry_time": entry_time}
            for h in HOLD_HORIZONS:
                if len(sub_post) >= h:
                    exit_px = sub_post.iloc[h - 1]
                    ret = (exit_px / entry_px) - 1.0
                    row[f"ret_{h}h"] = ret
                    row[f"net_ret_{h}h"] = ret - ROUNDTRIP_COST
                else:
                    row[f"ret_{h}h"] = np.nan
                    row[f"net_ret_{h}h"] = np.nan
            records_resilient.append(row)
            
        # 2. Evaluate RS Leaders
        for sym, rs_24h in selected_rs:
            df_sym = panel_1h[sym].set_index("open_time")
            if entry_time not in df_sym.index:
                continue
            entry_px = df_sym.loc[entry_time, "open"]
            sub_post = df_sym[df_sym.index >= entry_time]["close"]
            
            row = {"episode_id": ep.episode_id, "symbol": sym, "rs_24h": rs_24h, "entry_time": entry_time}
            for h in HOLD_HORIZONS:
                if len(sub_post) >= h:
                    exit_px = sub_post.iloc[h - 1]
                    ret = (exit_px / entry_px) - 1.0
                    row[f"ret_{h}h"] = ret
                    row[f"net_ret_{h}h"] = ret - ROUNDTRIP_COST
                else:
                    row[f"ret_{h}h"] = np.nan
                    row[f"net_ret_{h}h"] = np.nan
            records_rs_leaders.append(row)
            
        # 3. Market Median Baseline (All Active Crypto Alts)
        med_rets = {h: [] for h in HOLD_HORIZONS}
        for sym, df_s in panel_1h.items():
            if sym == "BTC_USDT":
                continue
            df_sym = df_s.set_index("open_time")
            if entry_time in df_sym.index:
                entry_px = df_sym.loc[entry_time, "open"]
                sub_post = df_sym[df_sym.index >= entry_time]["close"]
                for h in HOLD_HORIZONS:
                    if len(sub_post) >= h:
                        exit_px = sub_post.iloc[h - 1]
                        med_rets[h].append((exit_px / entry_px) - 1.0)
                        
        row_med = {"episode_id": ep.episode_id, "entry_time": entry_time}
        for h in HOLD_HORIZONS:
            if med_rets[h]:
                med_val = np.median(med_rets[h])
                row_med[f"ret_{h}h"] = med_val
                row_med[f"net_ret_{h}h"] = med_val - ROUNDTRIP_COST
            else:
                row_med[f"ret_{h}h"] = np.nan
                row_med[f"net_ret_{h}h"] = np.nan
        records_market_median.append(row_med)

    df_res = pd.DataFrame(records_resilient)
    df_rs = pd.DataFrame(records_rs_leaders)
    df_med = pd.DataFrame(records_market_median)
    
    print("\n" + "="*80)
    print("EMPIRICAL COMPARISON: DOWNSIDE RESILIENCE VS. ORDINARY RS LEADERS")
    print("="*80)
    print(f"Episodes evaluated with confirmed entries: {len(df_med)}")
    print(f"Total trades per strategy basket: Resilient={len(df_res)}, RS Leaders={len(df_rs)}")
    
    summary_data = []
    
    for h in HOLD_HORIZONS:
        col_gross = f"ret_{h}h"
        col_net = f"net_ret_{h}h"
        
        # Resilient stats
        res_valid = df_res[col_net].dropna()
        n_res = len(res_valid)
        res_mean_gross = df_res[col_gross].mean() * 100
        res_mean_net = res_valid.mean() * 100
        res_win = (res_valid > 0).mean() * 100
        res_se = (res_valid.std() / np.sqrt(n_res)) * 100
        res_ci95 = 1.96 * res_se
        
        # RS Leaders stats
        rs_valid = df_rs[col_net].dropna()
        n_rs = len(rs_valid)
        rs_mean_gross = df_rs[col_gross].mean() * 100
        rs_mean_net = rs_valid.mean() * 100
        rs_win = (rs_valid > 0).mean() * 100
        rs_se = (rs_valid.std() / np.sqrt(n_rs)) * 100
        rs_ci95 = 1.96 * rs_se
        
        # Market median stats
        med_valid = df_med[col_net].dropna()
        med_mean_gross = df_med[col_gross].mean() * 100
        med_mean_net = med_valid.mean() * 100
        med_win = (med_valid > 0).mean() * 100
        
        summary_data.append({
            "Horizon": f"+{h}H",
            "Resilient Gross (%)": f"{res_mean_gross:+.2f}%",
            "Resilient Net (%)": f"{res_mean_net:+.2f}% ± {res_ci95:.2f}%",
            "Resilient WinRate": f"{res_win:.1f}% (n={n_res})",
            "RS Leader Net (%)": f"{rs_mean_net:+.2f}% ± {rs_ci95:.2f}%",
            "RS Leader WinRate": f"{rs_win:.1f}% (n={n_rs})",
            "Market Median Net (%)": f"{med_mean_net:+.2f}%",
            "Spread (Res - RS) Net": f"{(res_mean_net - rs_mean_net):+.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results to parquet
    os.makedirs("results", exist_ok=True)
    df_res.to_parquet("results/downside_resilience_trades.parquet")
    df_rs.to_parquet("results/rs_leader_control_trades.parquet")
    summary_df.to_csv("results/downside_resilience_summary.csv", index=False)
    print("\nSaved detailed trade-level records to results/")

if __name__ == "__main__":
    run_experiment()
