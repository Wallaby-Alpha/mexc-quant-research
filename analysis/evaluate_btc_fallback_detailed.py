"""
analysis/evaluate_btc_fallback_detailed.py

Detailed evaluation of the "BTC Fallback" Macro System:
- If BTC < 50 EMA: 100% USDT Cash
- If BTC >= 50 EMA:
    - If Altcoins Performing (tested at Median Alt 30d Ret >= 0% and >= 5%):
        Hold 10-Coin Momentum Barbell (7 Quality + 3 Raw)
    - If Altcoins Lagging (Median Alt 30d Ret < Thresh):
        Hold 100% Bitcoin!

Evaluated:
1. Full 5.6-Year Period (March 2021 to October 2026)
2. Last 1 Year (October 2025 to October 2026)
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from analysis.evaluate_btc_filter_permutations import load_data

def run_simulation(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays, alt_thresh=0.0):
    initial_capital = 7000.0
    eq = initial_capital
    current_q, current_r = [], []
    current_w = {}
    weekly_records = []
    
    fee_rate = 0.0012
    weekly_funding = 0.0021
    
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
        
        # Check Altcoins
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
        
        # Sizing and Regime
        if not is_btc_bull:
            regime = 'CASH'
        else:
            if med_alt >= alt_thresh:
                regime = 'ALTS'
            else:
                regime = 'BTC'
                
        if regime == 'CASH':
            targets = []
            target_w = {}
            turnover = sum(current_w.values())
            fee = turnover * 2.0 * fee_rate
            net = -fee
        elif regime == 'BTC':
            targets = ['BTC']
            target_w = {'BTC': 1.0}
            all_s = set(current_w.keys()).union({'BTC'})
            turnover = sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_s) / 2.0
            fee = turnover * 2.0 * fee_rate
            net = btc_w_ret - fee - weekly_funding
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
            fee = turnover * 2.0 * fee_rate
            
            rets = []
            for s in targets:
                p_e = df_prices[s].loc[t_now]
                p_x = df_prices[s].loc[t_next] if t_next in df_prices[s].index else np.nan
                c_ret = (p_x - p_e) / p_e if (not np.isnan(p_x) and p_e > 0) else 0.0
                rets.append(c_ret)
            gross = float(np.mean(rets)) if rets else 0.0
            net = gross - fee - weekly_funding
            
        eq = eq * (1.0 + net)
        weekly_records.append({
            'monday': t_now,
            'year': t_now.year,
            'regime': regime,
            'net': net,
            'btc_ret': btc_w_ret,
            'equity': eq
        })
        current_w = target_w
        if regime == 'ALTS':
            current_q, current_r = target_q, target_r
        else:
            current_q, current_r = [], []
            
    return pd.DataFrame(weekly_records)

def calc_stats(df_sub, initial_cap=7000.0):
    col = 'net' if 'net' in df_sub.columns else 'net_ret'
    r_arr = df_sub[col].values
    eq_arr = initial_cap * np.cumprod(1.0 + r_arr)
    fin_eq = eq_arr[-1]
    tot_ret = (fin_eq - initial_cap) / initial_cap * 100.0
    n_years = len(df_sub) / 52.0
    cagr = ((fin_eq / initial_cap) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0
    pks = np.maximum.accumulate(eq_arr)
    dd = float(np.max((pks - eq_arr) / pks)) * 100.0
    sh = (np.mean(r_arr) / np.std(r_arr) * np.sqrt(52)) if np.std(r_arr) > 1e-6 else 0.0
    
    pct_alts = (df_sub['regime'] == 'ALTS').mean() * 100.0 if 'regime' in df_sub.columns else 0.0
    pct_btc = (df_sub['regime'] == 'BTC').mean() * 100.0 if 'regime' in df_sub.columns else 0.0
    pct_cash = (df_sub['regime'] == 'CASH').mean() * 100.0 if 'regime' in df_sub.columns else 0.0
    
    return {
        'fin_eq': fin_eq,
        'tot_ret': tot_ret,
        'cagr': cagr,
        'sharpe': sh,
        'max_dd': dd,
        'pct_alts': pct_alts,
        'pct_btc': pct_btc,
        'pct_cash': pct_cash,
        'weeks': len(df_sub)
    }

def main():
    df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays = load_data()
    
    # 1. Baseline
    from analysis.evaluate_btc_filter_permutations import run_permutation
    df_base = run_permutation('1_baseline', df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays)
    df_base['regime'] = np.where(df_base['is_cash'], 'CASH', 'ALTS')
    
    # 2. BTC Fallback (Thresh 0%)
    df_fb0 = run_simulation(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays, alt_thresh=0.0)
    
    # 3. BTC Fallback (Thresh +5%)
    df_fb5 = run_simulation(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays, alt_thresh=0.05)
    
    # Filter for Last 1 Year: Oct 2025 to Oct 2026
    t_1y_start = pd.Timestamp("2025-10-01", tz="UTC")
    df_base_1y = df_base[df_base['monday'] >= t_1y_start].copy()
    df_fb0_1y = df_fb0[df_fb0['monday'] >= t_1y_start].copy()
    df_fb5_1y = df_fb5[df_fb5['monday'] >= t_1y_start].copy()
    
    print("=" * 125)
    print("BTC FALLBACK SYSTEM EVALUATION: COMPARING LIFETIME (5.6 YEARS) vs LAST 1 YEAR")
    print("Rule: If BTC > 50 EMA: Hold 10 Alts if alts performing; otherwise Hold 100% BTC. If BTC < 50 EMA: 100% Cash.")
    print("=" * 125)
    
    # Table 1: Lifetime
    print("\n--- PART 1: FULL LIFETIME (March 2021 to October 2026: 5.6 Years | 291 Weeks) ---")
    print(f"{'Strategy Variant':<42} | {'Ending Capital':<15} | {'Total Return':<13} | {'CAGR':<8} | {'Sharpe':<7} | {'Max DD':<8} | {'% in Alts':<10} | {'% in BTC':<10} | {'% in Cash':<10}")
    print("-" * 135)
    
    st_b = calc_stats(df_base)
    st_0 = calc_stats(df_fb0)
    st_5 = calc_stats(df_fb5)
    
    print(f"{'1. Baseline (10 Alts or 100% Cash)':<42} | ${st_b['fin_eq']:<14,.2f} | {st_b['tot_ret']:+11.1f}% | {st_b['cagr']:+6.1f}% | {st_b['sharpe']:6.2f} | {st_b['max_dd']:6.1f}% | {st_b['pct_alts']:8.1f}% | {0.0:8.1f}% | {st_b['pct_cash']:8.1f}%")
    print(f"{'2. BTC Fallback (Hold BTC if Alt < 0%)':<42} | ${st_0['fin_eq']:<14,.2f} | {st_0['tot_ret']:+11.1f}% | {st_0['cagr']:+6.1f}% | {st_0['sharpe']:6.2f} | {st_0['max_dd']:6.1f}% | {st_0['pct_alts']:8.1f}% | {st_0['pct_btc']:8.1f}% | {st_0['pct_cash']:8.1f}%")
    print(f"{'3. BTC Fallback (Hold BTC if Alt < +5%)':<42} | ${st_5['fin_eq']:<14,.2f} | {st_5['tot_ret']:+11.1f}% | {st_5['cagr']:+6.1f}% | {st_5['sharpe']:6.2f} | {st_5['max_dd']:6.1f}% | {st_5['pct_alts']:8.1f}% | {st_5['pct_btc']:8.1f}% | {st_5['pct_cash']:8.1f}%")

    # Table 2: Last 1 Year
    print("\n--- PART 2: LAST 1 YEAR (October 2025 to October 2026: 52 Weeks) ---")
    print(f"{'Strategy Variant':<42} | {'Ending Capital':<15} | {'Total Return':<13} | {'CAGR':<8} | {'Sharpe':<7} | {'Max DD':<8} | {'% in Alts':<10} | {'% in BTC':<10} | {'% in Cash':<10}")
    print("-" * 135)
    
    st_b_1y = calc_stats(df_base_1y)
    st_0_1y = calc_stats(df_fb0_1y)
    st_5_1y = calc_stats(df_fb5_1y)
    
    print(f"{'1. Baseline (10 Alts or 100% Cash)':<42} | ${st_b_1y['fin_eq']:<14,.2f} | {st_b_1y['tot_ret']:+11.1f}% | {st_b_1y['cagr']:+6.1f}% | {st_b_1y['sharpe']:6.2f} | {st_b_1y['max_dd']:6.1f}% | {st_b_1y['pct_alts']:8.1f}% | {0.0:8.1f}% | {st_b_1y['pct_cash']:8.1f}%")
    print(f"{'2. BTC Fallback (Hold BTC if Alt < 0%)':<42} | ${st_0_1y['fin_eq']:<14,.2f} | {st_0_1y['tot_ret']:+11.1f}% | {st_0_1y['cagr']:+6.1f}% | {st_0_1y['sharpe']:6.2f} | {st_0_1y['max_dd']:6.1f}% | {st_0_1y['pct_alts']:8.1f}% | {st_0_1y['pct_btc']:8.1f}% | {st_0_1y['pct_cash']:8.1f}%")
    print(f"{'3. BTC Fallback (Hold BTC if Alt < +5%)':<42} | ${st_5_1y['fin_eq']:<14,.2f} | {st_5_1y['tot_ret']:+11.1f}% | {st_5_1y['cagr']:+6.1f}% | {st_5_1y['sharpe']:6.2f} | {st_5_1y['max_dd']:6.1f}% | {st_5_1y['pct_alts']:8.1f}% | {st_5_1y['pct_btc']:8.1f}% | {st_5_1y['pct_cash']:8.1f}%")

if __name__ == "__main__":
    main()
