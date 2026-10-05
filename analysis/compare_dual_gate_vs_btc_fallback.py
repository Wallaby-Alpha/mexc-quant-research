"""
analysis/compare_dual_gate_vs_btc_fallback.py

Compares:
1. Baseline: BTC > 50 EMA -> 100% Alts; else 100% USDT Cash.
2. Dual Gate (Cash): BTC > 50 EMA AND Median Alt > Thresh -> 100% Alts; else 100% USDT Cash.
3. BTC Fallback: BTC > 50 EMA:
   - If Median Alt >= Thresh -> 100% Alts (Top 7 Quality + Top 3 Raw).
   - If Median Alt < Thresh -> Hold 100% Bitcoin (BTC is bull, but alts are weak!).
   - If BTC < 50 EMA -> 100% USDT Cash.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from analysis.evaluate_btc_filter_permutations import load_data

def run_strategy(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays, mode='baseline', alt_thresh=0.0):
    eq = 7000.0
    current_q, current_r = [], []
    current_w = {}
    weekly_records = []
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i+1]
        
        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_p = s_btc.loc[t_now]
        btc_ema = s_btc_ema50.loc[t_now]
        btc_next = s_btc.loc[t_next] if t_next in s_btc.index else btc_p
        btc_w_ret = (btc_next - btc_p) / btc_p
        btc_30 = (btc_p - s_btc.loc[t_past_30]) / s_btc.loc[t_past_30]
        
        is_btc_bull = btc_p >= btc_ema
        
        alt_30s = []
        cands = []
        for sym in df_prices.columns:
            s_p = df_prices[sym]
            if t_now in s_p.index and t_past_30 in s_p.index:
                p0 = s_p.loc[t_past_30]
                p1 = s_p.loc[t_now]
                if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                    r = (p1 - p0) / p0
                    alt_30s.append(r)
                    slice_30 = s_p.iloc[max(0, idx_now - 30) : idx_now + 1]
                    pct = slice_30.pct_change().dropna()
                    vol = float(pct.std() * np.sqrt(365.25)) if len(pct) >= 20 else 1.0
                    cands.append({'sym': sym, 'ret': r, 'rs': r - btc_30, 'score': r / max(vol, 0.05)})
                    
        med_alt = float(np.median(alt_30s)) if alt_30s else -1.0
        
        action = 'CASH'
        if is_btc_bull:
            if mode == 'baseline':
                action = 'ALTS'
            elif mode == 'dual_gate_cash':
                action = 'ALTS' if med_alt >= alt_thresh else 'CASH'
            elif mode == 'btc_fallback':
                action = 'ALTS' if med_alt >= alt_thresh else 'BTC'
                
        if action == 'CASH':
            targets = []
            target_w = {}
            turnover = sum(current_w.values())
            fee = turnover * 2.0 * 0.0012
            net = -fee
        elif action == 'BTC':
            targets = ['BTC']
            target_w = {'BTC': 1.0}
            all_s = set(current_w.keys()).union({'BTC'})
            turnover = sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_s) / 2.0
            fee = turnover * 2.0 * 0.0012
            net = btc_w_ret - fee - 0.0021
        else: # ALTS
            df_c = pd.DataFrame(cands)
            df_q = df_c.sort_values('score', ascending=False).reset_index(drop=True)
            hq = [s for s in current_q if s in df_q.head(11)['sym'].tolist()]
            bq = [s for s in df_q.head(7)['sym'].tolist() if s not in hq][:7 - len(hq)]
            target_q = hq + bq
            df_r = df_c[~df_c['sym'].isin(target_q)].sort_values('rs', ascending=False).reset_index(drop=True)
            hr = [s for s in current_r if s in df_r.head(6)['sym'].tolist() and s not in target_q]
            br = [s for s in df_r.head(3)['sym'].tolist() if s not in hr][:3 - len(hr)]
            target_r = hr + br
            targets = target_q + target_r
            target_w = {s: 1.0 / len(targets) for s in targets}
            all_s = set(current_w.keys()).union(set(target_w.keys()))
            turnover = sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_s) / 2.0
            fee = turnover * 2.0 * 0.0012
            
            rets = []
            for s in targets:
                p_e = df_prices[s].loc[t_now]
                p_x = df_prices[s].loc[t_next] if t_next in df_prices[s].index else np.nan
                c_ret = (p_x - p_e) / p_e if (not np.isnan(p_x) and p_e > 0) else 0.0
                rets.append(c_ret)
            gross = float(np.mean(rets)) if rets else 0.0
            net = gross - fee - 0.0021
            
        eq = eq * (1.0 + net)
        weekly_records.append({'monday': t_now, 'year': t_now.year, 'action': action, 'net': net})
        current_w = target_w
        if action == 'ALTS':
            current_q, current_r = target_q, target_r
        else:
            current_q, current_r = [], []
            
    df_w = pd.DataFrame(weekly_records)
    r_arr = df_w['net'].values
    eq_arr = 7000.0 * np.cumprod(1.0 + r_arr)
    tot = (eq_arr[-1] - 7000.0) / 7000.0 * 100.0
    pks = np.maximum.accumulate(eq_arr)
    dd = float(np.max((pks - eq_arr) / pks)) * 100.0
    sh = np.mean(r_arr) / np.std(r_arr) * np.sqrt(52)
    return eq_arr[-1], tot, sh, dd, df_w

def main():
    df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays = load_data()
    
    configs = [
        ("1. Baseline (Alts or Cash)", "baseline", 0.0),
        ("2. Dual Gate (Cash if Alt < 0%)", "dual_gate_cash", 0.0),
        ("3. Dual Gate (Cash if Alt < +5%)", "dual_gate_cash", 0.05),
        ("4. BTC Fallback (Hold BTC if Alt < 0%)", "btc_fallback", 0.0),
        ("5. BTC Fallback (Hold BTC if Alt < +5%)", "btc_fallback", 0.05),
    ]

    print("=" * 110)
    print("COMPARISON: BASELINE vs DUAL GATE (CASH) vs BTC FALLBACK (2021-2026)")
    print("=" * 110)
    print(f"{'Variant':<45} | {'Ending Capital':<15} | {'Total Return':<13} | {'CAGR':<8} | {'Sharpe':<7} | {'Max DD':<8}")
    print("-" * 110)
    
    results = {}
    for name, m, t in configs:
        fin, tot, sh, dd, df_w = run_strategy(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays, m, t)
        results[name] = df_w
        n_years = len(df_w) / 52.0
        cagr = ((fin / 7000.0) ** (1.0 / n_years) - 1.0) * 100.0
        print(f"{name:<45} | ${fin:<14,.2f} | {tot:+11.1f}% | {cagr:+6.1f}% | {sh:6.2f} | {dd:6.1f}%")

    print("\n" + "=" * 110)
    print("YEAR-BY-YEAR COMPARISON (% NET RETURN PER CALENDAR YEAR)")
    print("=" * 110)
    print(f"{'Year':<6} | {'1. Baseline':<14} | {'2. Cash (0%)':<14} | {'3. Cash (+5%)':<14} | {'4. BTC (0%)':<14} | {'5. BTC (+5%)':<14}")
    print("-" * 85)
    
    years = sorted(list(results["1. Baseline (Alts or Cash)"]["year"].unique()))
    for yr in years:
        r1 = (np.prod(1.0 + results["1. Baseline (Alts or Cash)"][results["1. Baseline (Alts or Cash)"]["year"] == yr]["net"].values) - 1.0) * 100.0
        r2 = (np.prod(1.0 + results["2. Dual Gate (Cash if Alt < 0%)"][results["2. Dual Gate (Cash if Alt < 0%)"]["year"] == yr]["net"].values) - 1.0) * 100.0
        r3 = (np.prod(1.0 + results["3. Dual Gate (Cash if Alt < +5%)"][results["3. Dual Gate (Cash if Alt < +5%)"]["year"] == yr]["net"].values) - 1.0) * 100.0
        r4 = (np.prod(1.0 + results["4. BTC Fallback (Hold BTC if Alt < 0%)"][results["4. BTC Fallback (Hold BTC if Alt < 0%)"]["year"] == yr]["net"].values) - 1.0) * 100.0
        r5 = (np.prod(1.0 + results["5. BTC Fallback (Hold BTC if Alt < +5%)"][results["5. BTC Fallback (Hold BTC if Alt < +5%)"]["year"] == yr]["net"].values) - 1.0) * 100.0
        print(f"{yr:<6} | {r1:+12.1f}% | {r2:+12.1f}% | {r3:+12.1f}% | {r4:+12.1f}% | {r5:+12.1f}%")
        
    print("-" * 85)

if __name__ == "__main__":
    main()
