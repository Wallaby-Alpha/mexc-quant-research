"""
research_setup11_mae_mfe.py
===========================
Detailed Excursion & Optimization Analysis for Setup 11:
"London Dip to NY Peak Momentum Ride"

Analyzes:
1. Hourly Unrealized Equity Curve from 08:00 UTC through next day 08:00 UTC
   (Checking if 20:00 UTC is truly the optimal exit time).
2. MAE (Maximum Adverse Excursion) Distribution:
   - For all trades, winners, and losers.
   - Identifies the optimal Stop Loss placement (cutting losers without clipping winners).
3. MFE (Maximum Favorable Excursion) Distribution:
   - Distribution of peak gains and the exact hour peak MFE is achieved.
   - Identifies the optimal Take Profit placement.
4. Comprehensive Stop Loss & Take Profit Optimization Grid:
   - Testing combinations of SL (-2%, -3%, -4%, -5%, -6%, None) and
     TP (+3%, +4%, +5%, +6%, +8%, +10%, +12%, None) with 20:00 UTC time backstop.
   - Realistic 10 bps fees deducted on all exits.
   - In-Sample vs Out-of-Sample verification.
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
    print("Loading 15m panel data...", flush=True)
    df_close = pd.read_parquet(PANEL_DIR / "panel_close.parquet")
    df_open = pd.read_parquet(PANEL_DIR / "panel_open.parquet")
    df_high = pd.read_parquet(PANEL_DIR / "panel_high.parquet")
    df_low = pd.read_parquet(PANEL_DIR / "panel_low.parquet")
    return df_close, df_open, df_high, df_low

def extract_trades_and_excursions(df_close, df_open, df_high, df_low, gain_thresh=0.10):
    daily_close = df_close.resample("1D").last()
    daily_rets = daily_close.pct_change()
    calendar_days = daily_close.index
    n_days = len(calendar_days)
    n_bars, n_coins = df_close.shape
    split_idx = int(n_bars * 0.75)
    split_date = df_close.index[split_idx].date()
    
    trades = []
    
    for d_idx in range(1, n_days - 1):
        prior_d = calendar_days[d_idx - 1]
        curr_d = calendar_days[d_idx]
        is_holdout = curr_d.date() >= split_date
        
        prior_ret = daily_rets.loc[prior_d].dropna()
        leaders = prior_ret[prior_ret >= gain_thresh].sort_values(ascending=False).head(5).index
        if len(leaders) == 0:
            continue
            
        day_mask = (df_close.index >= curr_d) & (df_close.index < curr_d + pd.Timedelta(days=2))
        sub_open = df_open.loc[day_mask]
        sub_high = df_high.loc[day_mask]
        sub_low = df_low.loc[day_mask]
        sub_close = df_close.loc[day_mask]
        
        if len(sub_open) < 96:
            continue
            
        # Entry is bar 32 (08:00 UTC on day curr_d)
        entry_idx = 32
        
        for coin in leaders:
            entry_p = sub_open.iloc[entry_idx][coin]
            if np.isnan(entry_p) or entry_p <= 0:
                continue
                
            # Path during day: from bar 32 to 95 (08:00 to 23:45 UTC, 64 bars)
            # and up to 20:00 UTC (bar 80, 48 bars after entry)
            highs_to_20 = sub_high.iloc[entry_idx:81][coin].values
            lows_to_20 = sub_low.iloc[entry_idx:81][coin].values
            closes_to_20 = sub_close.iloc[entry_idx:81][coin].values
            
            # Full 24h path up to next day 08:00 UTC (up to 96 bars)
            max_bars = min(96, len(sub_open) - entry_idx)
            highs_full = sub_high.iloc[entry_idx:entry_idx + max_bars][coin].values
            lows_full = sub_low.iloc[entry_idx:entry_idx + max_bars][coin].values
            closes_full = sub_close.iloc[entry_idx:entry_idx + max_bars][coin].values
            opens_full = sub_open.iloc[entry_idx:entry_idx + max_bars][coin].values
            
            # Excursions to 20:00 UTC (bar 80)
            mae_20 = (np.min(lows_to_20) - entry_p) / entry_p
            mfe_20 = (np.max(highs_to_20) - entry_p) / entry_p
            ret_20 = (sub_close.iloc[80][coin] - entry_p) / entry_p
            
            # Time of peak MFE (bar index offset from entry)
            mfe_bar_offset = int(np.argmax(highs_to_20))
            mfe_hour_utc = 8.0 + (mfe_bar_offset * 15 / 60.0)
            
            # End returns at alternative times
            ret_16 = (sub_close.iloc[64][coin] - entry_p) / entry_p if len(sub_close) > 64 else np.nan # 16:00 UTC
            ret_18 = (sub_close.iloc[72][coin] - entry_p) / entry_p if len(sub_close) > 72 else np.nan # 18:00 UTC
            ret_22 = (sub_close.iloc[88][coin] - entry_p) / entry_p if len(sub_close) > 88 else np.nan # 22:00 UTC
            ret_24 = (sub_close.iloc[95][coin] - entry_p) / entry_p if len(sub_close) > 95 else np.nan # 23:45 UTC
            
            trades.append({
                "date": curr_d,
                "coin": coin,
                "is_holdout": is_holdout,
                "entry_p": entry_p,
                "ret_16": ret_16,
                "ret_18": ret_18,
                "ret_20": ret_20,
                "ret_22": ret_22,
                "ret_24": ret_24,
                "mae_20": mae_20,
                "mfe_20": mfe_20,
                "mfe_hour_utc": mfe_hour_utc,
                "is_winner_20": ret_20 > 0,
                "highs_full": highs_full,
                "lows_full": lows_full,
                "closes_full": closes_full,
                "opens_full": opens_full
            })
            
    df_trades = pd.DataFrame(trades)
    print(f"Extracted {len(df_trades)} individual coin trades across {n_days} days.")
    print(f"In-Sample trades: {len(df_trades[~df_trades['is_holdout']])}, Holdout trades: {len(df_trades[df_trades['is_holdout']])}\n")
    return df_trades

def analyze_mae_mfe(df_trades):
    print("=" * 90)
    print("1. MAXIMUM ADVERSE EXCURSION (MAE) & MAXIMUM FAVORABLE EXCURSION (MFE) ANALYSIS")
    print("=" * 90)
    
    for partition, sub_df in [("IN-SAMPLE (Bars 0-25932)", df_trades[~df_trades["is_holdout"]]),
                              ("HOLDOUT (Bars 25932-34576)", df_trades[df_trades["is_holdout"]]),
                              ("ALL TRADES COMBINED", df_trades)]:
        print(f"\n--- {partition} (N = {len(sub_df)}) ---")
        
        # MFE Distribution
        mfes = sub_df["mfe_20"] * 100
        print(f"MFE (Peak Run-Up) Quantiles:")
        print(f"  Min:   {mfes.min():.2f}%")
        print(f"  25th:  {mfes.quantile(0.25):.2f}%")
        print(f"  50th:  {mfes.median():.2f}%")
        print(f"  75th:  {mfes.quantile(0.75):.2f}%")
        print(f"  90th:  {mfes.quantile(0.90):.2f}%")
        print(f"  Max:   {mfes.max():.2f}%")
        print(f"  Mean:  {mfes.mean():.2f}%")
        
        # MAE Distribution
        maes = sub_df["mae_20"] * 100
        print(f"\nMAE (Max Drawdown) Quantiles:")
        print(f"  Min (Worst): {maes.min():.2f}%")
        print(f"  10th:        {maes.quantile(0.10):.2f}%")
        print(f"  25th:        {maes.quantile(0.25):.2f}%")
        print(f"  50th:        {maes.median():.2f}%")
        print(f"  75th:        {maes.quantile(0.75):.2f}%")
        print(f"  Max (Best):  {maes.max():.2f}%")
        print(f"  Mean:        {maes.mean():.2f}%")
        
        # Breakdown by Winners vs Losers at 20:00 UTC
        win_sub = sub_df[sub_df["is_winner_20"]]
        loss_sub = sub_df[~sub_df["is_winner_20"]]
        
        print(f"\nMAE of Winning Trades (N = {len(win_sub)}):")
        print(f"  Median MAE of winners: {win_sub['mae_20'].median()*100:.2f}% | Mean: {win_sub['mae_20'].mean()*100:.2f}%")
        print(f"  % of winners with MAE <= -2%: {(win_sub['mae_20'] <= -0.02).mean()*100:.1f}%")
        print(f"  % of winners with MAE <= -3%: {(win_sub['mae_20'] <= -0.03).mean()*100:.1f}%")
        print(f"  % of winners with MAE <= -4%: {(win_sub['mae_20'] <= -0.04).mean()*100:.1f}%")
        print(f"  % of winners with MAE <= -5%: {(win_sub['mae_20'] <= -0.05).mean()*100:.1f}%")
        
        print(f"\nMAE of Losing Trades (N = {len(loss_sub)}):")
        print(f"  Median MAE of losers:  {loss_sub['mae_20'].median()*100:.2f}% | Mean: {loss_sub['mae_20'].mean()*100:.2f}%")
        print(f"  % of losers with MAE <= -2%:  {(loss_sub['mae_20'] <= -0.02).mean()*100:.1f}%")
        print(f"  % of losers with MAE <= -3%:  {(loss_sub['mae_20'] <= -0.03).mean()*100:.1f}%")
        print(f"  % of losers with MAE <= -4%:  {(loss_sub['mae_20'] <= -0.04).mean()*100:.1f}%")
        print(f"  % of losers with MAE <= -5%:  {(loss_sub['mae_20'] <= -0.05).mean()*100:.1f}%")

def analyze_time_based_exits(df_trades):
    print("\n" + "=" * 90)
    print("2. HOURLY EXIT TIME EVALUATION: IS 20:00 UTC TRULY OPTIMAL?")
    print("=" * 90)
    
    fee = 0.0010 # 10 bps
    
    for partition, sub_df in [("IN-SAMPLE", df_trades[~df_trades["is_holdout"]]),
                              ("HOLDOUT", df_trades[df_trades["is_holdout"]]),
                              ("ALL TRADES", df_trades)]:
        print(f"\n--- {partition} (N = {len(sub_df)}) ---")
        times = [
            ("16:00 UTC (8h hold - US Mid-Day)", sub_df["ret_16"] - fee),
            ("18:00 UTC (10h hold - London Close)", sub_df["ret_18"] - fee),
            ("20:00 UTC (12h hold - US Cash Close)", sub_df["ret_20"] - fee),
            ("22:00 UTC (14h hold - Late NY)", sub_df["ret_22"] - fee),
            ("23:45 UTC (16h hold - Daily Close)", sub_df["ret_24"] - fee),
        ]
        
        rows = []
        for name, r in times:
            r = r.dropna()
            mean_r = r.mean() * 100
            med_r = r.median() * 100
            wr = (r > 0).mean() * 100
            wins = r[r > 0]
            losses = r[r < 0]
            pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan
            rows.append({
                "Exit Time": name,
                "Net Mean %": round(mean_r, 3),
                "Net Median %": round(med_r, 3),
                "Win Rate %": round(wr, 1),
                "Profit Factor": round(pf, 2)
            })
        print(pd.DataFrame(rows).to_string(index=False))
        
        # When does Peak MFE happen during the day?
        mfe_hours = sub_df["mfe_hour_utc"]
        print(f"\nPeak MFE Hour Distribution:")
        print(f"  Mean Peak Hour:   {mfe_hours.mean():.1f}:00 UTC")
        print(f"  Median Peak Hour: {mfe_hours.median():.1f}:00 UTC")
        print(f"  % Peak MFE reached between 16:00 and 20:00 UTC: {((mfe_hours >= 16) & (mfe_hours <= 20)).mean()*100:.1f}%")
        print(f"  % Peak MFE reached before 14:00 UTC: {(mfe_hours < 14).mean()*100:.1f}%")

def evaluate_tp_sl_grid(df_trades):
    print("\n" + "=" * 90)
    print("3. STOP LOSS & TAKE PROFIT BRACKET OPTIMIZATION GRID")
    print("=" * 90)
    
    # We test combinations of:
    # Stop Loss: None, -2.0%, -2.5%, -3.0%, -4.0%, -5.0%, -6.0%
    # Take Profit: None, +3.0%, +4.0%, +5.0%, +6.0%, +8.0%, +10.0%, +12.0%
    # Backstop exit: 20:00 UTC (bar 48 after entry)
    # Fee: 10 bps round trip deducted on all exits.
    
    sl_levels = [None, -0.02, -0.025, -0.03, -0.04, -0.05, -0.06]
    tp_levels = [None, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
    
    def simulate_bracket(sub_df, sl, tp):
        net_pnls = []
        exit_types = [] # 'TP', 'SL', 'TIME'
        
        for _, row in sub_df.iterrows():
            entry_p = row["entry_p"]
            highs = row["highs_full"][:49] # up to bar 80
            lows = row["lows_full"][:49]
            closes = row["closes_full"][:49]
            
            pnl = 0.0
            exited = False
            
            for b in range(1, len(highs)):
                bar_h = highs[b]
                bar_l = lows[b]
                
                # Check Stop Loss first (conservative intraday assumption)
                if sl is not None:
                    sl_price = entry_p * (1.0 + sl)
                    if bar_l <= sl_price:
                        pnl = sl - 0.0010 # exit at sl price minus fee
                        exit_types.append("SL")
                        exited = True
                        break
                        
                # Check Take Profit
                if tp is not None:
                    tp_price = entry_p * (1.0 + tp)
                    if bar_h >= tp_price:
                        pnl = tp - 0.0010 # exit at tp price minus fee
                        exit_types.append("TP")
                        exited = True
                        break
                        
            if not exited:
                # Time exit at 20:00 UTC (close of bar 48)
                pnl = (closes[-1] - entry_p) / entry_p - 0.0010
                exit_types.append("TIME")
                
            net_pnls.append(pnl)
            
        arr = np.array(net_pnls)
        mean_pnl = float(np.mean(arr))
        wr = float(np.mean(arr > 0))
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        pf = float(abs(wins.sum() / losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.nan
        types_s = pd.Series(exit_types).value_counts(normalize=True).to_dict()
        
        return mean_pnl, wr, pf, types_s
        
    for partition, sub_df in [("IN-SAMPLE", df_trades[~df_trades["is_holdout"]]),
                              ("HOLDOUT", df_trades[df_trades["is_holdout"]]),
                              ("ALL TRADES", df_trades)]:
        print(f"\n--- GRID RESULTS FOR {partition} (N = {len(sub_df)}) ---")
        grid_rows = []
        
        for sl in sl_levels:
            for tp in tp_levels:
                mean_pnl, wr, pf, exit_dist = simulate_bracket(sub_df, sl, tp)
                sl_str = f"{int(sl*100)}%" if sl is not None else "None"
                tp_str = f"+{int(tp*100)}%" if tp is not None else "None"
                
                grid_rows.append({
                    "SL": sl_str,
                    "TP": tp_str,
                    "Net Mean %": round(mean_pnl * 100, 3),
                    "Win Rate %": round(wr * 100, 1),
                    "Profit Factor": round(pf, 2) if not np.isnan(pf) else "N/A",
                    "% TP": round(exit_dist.get("TP", 0.0) * 100, 1),
                    "% SL": round(exit_dist.get("SL", 0.0) * 100, 1),
                    "% TIME": round(exit_dist.get("TIME", 0.0) * 100, 1),
                })
                
        df_grid = pd.DataFrame(grid_rows)
        # Sort by Net Mean % descending
        df_top = df_grid.sort_values("Net Mean %", ascending=False).head(10)
        print("Top 10 Configurations by Net Return:")
        print(df_top.to_string(index=False))
        
        # Save complete grid
        out_csv = RESULTS_DIR / f"setup11_tp_sl_grid_{partition.lower().split()[0]}.csv"
        df_grid.to_csv(out_csv, index=False)
        print(f"Saved full grid to {out_csv}")

if __name__ == "__main__":
    df_close, df_open, df_high, df_low = load_data()
    df_trades = extract_trades_and_excursions(df_close, df_open, df_high, df_low, gain_thresh=0.10)
    
    analyze_mae_mfe(df_trades)
    analyze_time_based_exits(df_trades)
    evaluate_tp_sl_grid(df_trades)
