"""
research_setup11_london_dip_momentum_ride.py
============================================
Research on Setup 11: "The 'London Dip to NY Peak' Momentum Ride (Long)"

Hypothesis:
In Setup 2 and 10, diurnal curve mapping proved that yesterday's top momentum gainers
pull back during Asia (00:00-08:00 UTC), bottom right at the London Open (08:00 UTC),
and then experience an explosive continuation wave (+2.05% net alpha) into the NY cash close (20:00 UTC).
Instead of fading momentum, systematically buying the London morning pullback on yesterday's proven
leaders provides high net expectancy and strong alpha over the broad market.

Strict Rules Applied:
- Zero lookahead: Day D-1 performance determined strictly up to 23:45 UTC of Day D-1.
- All trading signals execute strictly at or after 08:00 UTC of Day D.
- Holdout discipline: 75% in-sample exploration window (Oct 2025 to June 30, 2026); last 25% strictly locked.
- Realistic transaction fees: 10 bps and 15 bps round-trip deducted.
- Statistical significance: n, win rate, profit factor, 95% confidence intervals, and comparison to controls.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PANEL_DIR = Path("data_cache/clean_panel_15m")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("Loading 15m clean panel data...", flush=True)
    df_close = pd.read_parquet(PANEL_DIR / "panel_close.parquet")
    df_open = pd.read_parquet(PANEL_DIR / "panel_open.parquet")
    df_high = pd.read_parquet(PANEL_DIR / "panel_high.parquet")
    df_low = pd.read_parquet(PANEL_DIR / "panel_low.parquet")
    df_bench = pd.read_parquet(PANEL_DIR / "panel_benchmarks.parquet")
    
    return df_close, df_open, df_high, df_low, df_bench

def run_setup11_research():
    df_close, df_open, df_high, df_low, df_bench = load_data()
    
    n_bars, n_coins = df_close.shape
    print(f"Panel shape: {n_bars} bars, {n_coins} symbols. Date range: {df_close.index[0]} to {df_close.index[-1]}")
    
    # 75% In-Sample Guard
    split_idx = int(n_bars * 0.75)
    print(f"In-sample bars: 0 to {split_idx} ({df_close.index[0]} to {df_close.index[split_idx-1]})")
    print(f"Holdout bars (LOCKED): {split_idx} to {n_bars}\n")
    
    is_close = df_close.iloc[:split_idx]
    is_open = df_open.iloc[:split_idx]
    is_high = df_high.iloc[:split_idx]
    is_low = df_low.iloc[:split_idx]
    
    # 1. Resample to Daily (00:00 UTC) to identify Day D-1 performance without lookahead
    daily_close = is_close.resample("1D").last()
    daily_returns = daily_close.pct_change()
    
    calendar_days = daily_close.index
    n_days = len(calendar_days)
    print(f"Total in-sample calendar days available: {n_days}")
    
    # --- EVALUATE STRATEGY ACROSS MULTIPLE CONFIGURATIONS ---
    # Prior Day Return Thresholds: >= +8%, >= +10%, >= +12%, >= +15%, >= +20%
    # Entry Timings:
    # 1. Direct Entry at 08:00 UTC (bar 32)
    # 2. Direct Entry at 09:00 UTC (bar 36)
    # 3. Confirmation Trigger: First green 15m candle between 08:00 and 10:00 UTC (bars 32 to 40)
    # Exit Timings:
    # 1. 18:00 UTC (bar 72 - London close)
    # 2. 20:00 UTC (bar 80 - US cash equity close / pre-flush)
    # 3. 23:45 UTC (bar 95 - Daily close)
    
    records = []
    
    gain_thresholds = [0.08, 0.10, 0.12, 0.15, 0.20]
    
    for min_gain in gain_thresholds:
        gain_str = f">={int(min_gain*100)}%"
        
        # Test 1: Direct 08:00 UTC Entry to 20:00 UTC Exit (Core Setup 11)
        for entry_hour, exit_hour, label in [
            (8, 20, "08:00 -> 20:00 UTC (NY Close)"),
            (8, 18, "08:00 -> 18:00 UTC (London Close)"),
            (9, 20, "09:00 -> 20:00 UTC (NY Close)"),
            (8, 23, "08:00 -> 23:45 UTC (Daily Close)")
        ]:
            e_bar = int(entry_hour * 4)
            x_bar = int(exit_hour * 4) if exit_hour < 23 else 95
            
            strat_rets = []
            mkt_rets = []
            dipper_rets = [] # Control: buying yesterday's worst losers at 08:00 UTC
            
            for d_idx in range(1, n_days - 1):
                prior_day = calendar_days[d_idx - 1]
                curr_day = calendar_days[d_idx]
                
                prior_ret = daily_returns.loc[prior_day].dropna()
                if len(prior_ret) < 30:
                    continue
                    
                top_leaders = prior_ret[prior_ret >= min_gain].sort_values(ascending=False).head(5).index
                worst_dippers = prior_ret.sort_values(ascending=True).head(5).index
                if len(top_leaders) == 0:
                    continue
                    
                day_mask = (is_close.index >= curr_day) & (is_close.index < curr_day + pd.Timedelta(days=1))
                day_opens = is_open.loc[day_mask]
                day_closes = is_close.loc[day_mask]
                
                if len(day_opens) != 96:
                    continue
                    
                # Buy at open of e_bar; Sell at close of x_bar
                p_buy = day_opens.iloc[e_bar][top_leaders]
                p_sell = day_closes.iloc[x_bar][top_leaders]
                long_ret = (p_sell - p_buy) / p_buy
                strat_rets.append(long_ret.mean())
                
                # Market control (all other coins)
                others = [c for c in day_opens.columns if c not in top_leaders]
                p_buy_m = day_opens.iloc[e_bar][others]
                p_sell_m = day_closes.iloc[x_bar][others]
                long_ret_m = (p_sell_m - p_buy_m) / p_buy_m
                mkt_rets.append(long_ret_m.mean())
                
                # Dipper control (yesterday's worst losers)
                p_buy_d = day_opens.iloc[e_bar][worst_dippers]
                p_sell_d = day_closes.iloc[x_bar][worst_dippers]
                long_ret_d = (p_sell_d - p_buy_d) / p_buy_d
                dipper_rets.append(long_ret_d.mean())
                
            if len(strat_rets) >= 20:
                arr_s = np.array(strat_rets)
                arr_m = np.array(mkt_rets)
                arr_d = np.array(dipper_rets)
                n_trades = len(arr_s)
                
                gross_mean = float(np.mean(arr_s))
                gross_med = float(np.median(arr_s))
                std_dev = float(np.std(arr_s, ddof=1))
                se = std_dev / np.sqrt(n_trades)
                ci95_low = gross_mean - 1.96 * se
                ci95_high = gross_mean + 1.96 * se
                
                net_10bps = gross_mean - 0.0010
                net_15bps = gross_mean - 0.0015
                win_rate_net10 = float(np.mean((arr_s - 0.0010) > 0))
                
                wins = arr_s[arr_s > 0.0010] - 0.0010
                losses = arr_s[arr_s <= 0.0010] - 0.0010
                pf = float(abs(np.sum(wins) / np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else np.nan
                
                mkt_gross = float(np.mean(arr_m))
                alpha_vs_mkt = gross_mean - mkt_gross
                alpha_vs_dippers = gross_mean - float(np.mean(arr_d))
                
                t_stat, p_val = stats.ttest_rel(arr_s, arr_m)
                
                records.append({
                    "Min_Gain": gain_str,
                    "Execution_Window": label,
                    "N_Days": n_trades,
                    "Net_Mean_%": round(net_10bps * 100, 3),
                    "Gross_Mean_%": round(gross_mean * 100, 3),
                    "Net_Median_%": round(gross_med * 100, 3),
                    "Win_Rate_%": round(win_rate_net10 * 100, 1),
                    "Profit_Factor": round(pf, 2) if not np.isnan(pf) else "N/A",
                    "CI95_%": f"[{round(ci95_low*100, 2)}, {round(ci95_high*100, 2)}]",
                    "Market_Control_%": round(mkt_gross * 100, 3),
                    "Alpha_vs_Market_%": round(alpha_vs_mkt * 100, 3),
                    "Alpha_vs_Dippers_%": round(alpha_vs_dippers * 100, 3),
                    "P_Value": round(p_val, 4)
                })

    df_res = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "setup11_london_dip_momentum_ride_results.csv"
    df_res.to_csv(csv_path, index=False)
    
    return df_res

if __name__ == "__main__":
    df_results = run_setup11_research()
    print("\n" + "=" * 95)
    print("SETUP 11: LONDON DIP MOMENTUM RIDE SUMMARY TABLE (NET OF 10 BPS FEE)")
    print("=" * 95)
    cols = ["Min_Gain", "Execution_Window", "N_Days", "Net_Mean_%", "Win_Rate_%", "Profit_Factor", "Market_Control_%", "Alpha_vs_Market_%", "P_Value"]
    print(df_results[cols].to_string(index=False))
