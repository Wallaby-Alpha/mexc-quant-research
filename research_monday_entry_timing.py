"""
research_monday_entry_timing.py
Comprehensive empirical investigation into Weekly Portfolio Rotation Entry Timing:
- Monday 00:00 UTC vs Delayed Entry (+1h, +2h, +4h, +8h, +12h, +16h, +20h, +24h / Tuesday, +48h / Wednesday)
- Synchronized Weekly Exit vs 168-Hour Rolling Hold
- Intraday Monday Price Action Anatomy (MAE / Dip Depth, Low Timing, Hourly Drift)
- Limit Order / Pullback Entry (Buying Dips vs Missed Fill Opportunity Cost)
- Multi-Year Day-of-Week Rebalance Comparison (2021-2026)
"""

import sys
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from data.config import load_config
from data.cache import ParquetCache
from backtest.execution_model import ExecutionModel

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("entry_timing")

def load_rebalance_weeks_and_holdings(db_path: Path):
    conn = sqlite3.connect(db_path)
    df_weeks = pd.read_sql("SELECT * FROM rebalance_weeks ORDER BY week_number", conn)
    df_rankings = pd.read_sql("SELECT * FROM weekly_rankings", conn)
    conn.close()
    return df_weeks, df_rankings

def main():
    logger.info("=" * 80)
    logger.info("EMPIRICAL RESEARCH: OPTIMAL ENTRY TIMING FOR WEEKLY ROTATION PORTFOLIO")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    db_path = Path("results/weekly_relative_strength.db")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return

    df_weeks, df_rankings = load_rebalance_weeks_and_holdings(db_path)
    logger.info(f"Loaded {len(df_weeks)} weeks from DB. Bullish weeks: {(df_weeks['btc_macro_bullish'] == 1).sum()}")

    # Load 1h Klines for all universe symbols
    logger.info("Loading 1h Kline data for universe...")
    symbols_1h = cache.list_cached_symbols("1h")
    dict_1h = {}
    for sym in symbols_1h:
        df = cache.load_klines(sym, "1h")
        if not df.empty:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.sort_values("open_time").set_index("open_time")
            dict_1h[sym] = df

    logger.info(f"Loaded 1h data for {len(dict_1h)} symbols.")

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )
    total_one_way_cost = exec_model.taker_fee_rate + (exec_model.base_slippage_bps / 10000.0)

    # We evaluate across the active/bullish weeks
    # Let's inspect the 10-Coin Barbell portfolio (7 Quality + 3 Raw)
    # as well as Top 7 Quality and Top 10 Raw portfolios.
    
    delays_hours = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48]
    
    # -------------------------------------------------------------------------
    # PART 1: INTRADAY MONDAY ANATOMY & HOURLY DRIFT
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: INTRADAY MONDAY PRICE PATH (HOURS 0 TO 24 UTC)")
    logger.info("=" * 80)

    # For every bullish week and every selected barbell coin:
    # Measure price at 0h, 1h, 2h, ..., 24h, plus intraday min (dip) and max (rally).
    monday_paths = [] # list of 25-element arrays
    monday_maes = []  # max drawdown within Monday
    monday_mfes = []  # max gain within Monday
    min_hour_dist = [] # which hour did the minimum occur

    for _, w_row in df_weeks.iterrows():
        if w_row["btc_macro_bullish"] != 1:
            continue
        
        rebal_day = w_row["rebalance_day"] # e.g. "2025-09-29"
        t0 = pd.Timestamp(rebal_day, tz="UTC") # Monday 00:00 UTC
        
        # Get selected barbell coins: top 7 sharpe + top 3 raw (excluding sharpe)
        w_ranks = df_rankings[df_rankings["rebalance_day"] == rebal_day]
        q_syms = w_ranks[w_ranks["is_top7_sharpe"] == 1]["symbol"].tolist()
        raw_candidates = w_ranks[w_ranks["is_top10_raw"] == 1].sort_values("rank_raw_mom")
        raw_syms = [s for s in raw_candidates["symbol"] if s not in q_syms][:3]
        portfolio_syms = q_syms + raw_syms

        week_coin_paths = []
        for sym in portfolio_syms:
            if sym not in dict_1h:
                continue
            df_sym = dict_1h[sym]
            
            # Check if 24 hours available
            t_hours = [t0 + pd.Timedelta(hours=h) for h in range(25)]
            if not all(th in df_sym.index for th in t_hours):
                continue
            
            p0 = df_sym.loc[t0, "open"]
            if p0 <= 0:
                continue
            
            # Hourly opens
            h_prices = [df_sym.loc[th, "open"] for th in t_hours]
            h_rets = [(p - p0) / p0 for p in h_prices]
            week_coin_paths.append(h_rets)

            # Highs and lows over the 24 hours
            sub_df = df_sym.loc[t0 : t0 + pd.Timedelta(hours=23)]
            mae = (sub_df["low"].min() - p0) / p0
            mfe = (sub_df["high"].max() - p0) / p0
            monday_maes.append(mae)
            monday_mfes.append(mfe)
            
            # Hour of lowest low
            min_ts = sub_df["low"].idxmin()
            min_h = int((min_ts - t0).total_seconds() / 3600)
            min_hour_dist.append(min_h)

        if week_coin_paths:
            # Portfolio average path for this week
            avg_week_path = np.mean(week_coin_paths, axis=0)
            monday_paths.append(avg_week_path)

    df_monday_paths = pd.DataFrame(monday_paths, columns=[f"H+{h}" for h in range(25)])
    mean_hourly_drift = df_monday_paths.mean() * 100
    median_hourly_drift = df_monday_paths.median() * 100

    print("\n--- INTRADAY MONDAY RETURN DRIFT (%) FROM 00:00 UTC OPEN ---")
    drift_summary = pd.DataFrame({
        "Mean Return (%)": mean_hourly_drift,
        "Median Return (%)": median_hourly_drift,
        "% Weeks Positive": (df_monday_paths > 0).mean() * 100
    })
    print(drift_summary.loc[[f"H+{h}" for h in [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]]].to_string())

    print("\n--- MONDAY EXCURSION DISTRIBUTIONS (ALL ACTIVE PORTFOLIO COINS) ---")
    print(f"Sample size (coin-weeks): n = {len(monday_maes)}")
    print(f"Average Max Intraday Dip (MAE):  {np.mean(monday_maes)*100:+.2f}%  (Median: {np.median(monday_maes)*100:+.2f}%)")
    print(f"Average Max Intraday Peak (MFE): {np.mean(monday_mfes)*100:+.2f}%  (Median: {np.median(monday_mfes)*100:+.2f}%)")
    print(f"% Coins that dipped at least -1% below 00:00 open: {np.mean(np.array(monday_maes) <= -0.01)*100:.1f}%")
    print(f"% Coins that dipped at least -2% below 00:00 open: {np.mean(np.array(monday_maes) <= -0.02)*100:.1f}%")
    print(f"% Coins that dipped at least -3% below 00:00 open: {np.mean(np.array(monday_maes) <= -0.03)*100:.1f}%")
    
    # Low timing distribution
    s_min_h = pd.Series(min_hour_dist).value_counts(normalize=True).sort_index() * 100
    print("\n--- TIMING OF MONDAY INTRADAY LOW (UTC HOUR) ---")
    print("Top hours when the daily low is established:")
    print(s_min_h.nlargest(5).to_string())

    # -------------------------------------------------------------------------
    # PART 2: ENTRY DELAY SWEEP (SYNCHRONIZED EXIT AT NEXT MONDAY 00:00 UTC)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: ENTRY DELAY HORIZONS (SYNCHRONIZED WEEKLY REBALANCE)")
    logger.info("=" * 80)

    # In Model 1:
    # Exit always happens at Next Monday 00:00 UTC.
    # If delay = 0: hold 168h (Mon 00:00 -> Mon 00:00)
    # If delay = 4: hold 164h (Mon 04:00 -> Mon 00:00)
    # If delay = 8: hold 160h (Mon 08:00 -> Mon 00:00)
    # If delay = 24: hold 144h (Tue 00:00 -> Mon 00:00)
    
    results_model1 = []
    
    # Store weekly returns to compute paired t-tests vs delay=0 baseline
    weekly_rets_by_delay = {d: [] for d in delays_hours}
    weekly_dates = []

    # Identify all pairs of consecutive rebalance weeks
    bullish_weeks_count = 0
    for w_idx in range(len(df_weeks) - 1):
        w_curr = df_weeks.iloc[w_idx]
        w_next = df_weeks.iloc[w_idx + 1]

        t_curr_0 = pd.Timestamp(w_curr["rebalance_day"], tz="UTC")
        t_exit_0 = pd.Timestamp(w_next["rebalance_day"], tz="UTC")

        is_bullish = (w_curr["btc_macro_bullish"] == 1)
        if not is_bullish:
            # In cash week, return is 0 across all delays
            for d in delays_hours:
                weekly_rets_by_delay[d].append(0.0)
            weekly_dates.append(t_curr_0)
            continue

        bullish_weeks_count += 1
        weekly_dates.append(t_curr_0)

        # Get barbell portfolio coins
        w_ranks = df_rankings[df_rankings["rebalance_day"] == w_curr["rebalance_day"]]
        q_syms = w_ranks[w_ranks["is_top7_sharpe"] == 1]["symbol"].tolist()
        raw_candidates = w_ranks[w_ranks["is_top10_raw"] == 1].sort_values("rank_raw_mom")
        raw_syms = [s for s in raw_candidates["symbol"] if s not in q_syms][:3]
        portfolio_syms = q_syms + raw_syms

        for d in delays_hours:
            t_entry = t_curr_0 + pd.Timedelta(hours=d)
            coin_rets = []
            for sym in portfolio_syms:
                if sym not in dict_1h:
                    continue
                df_sym = dict_1h[sym]
                if t_entry not in df_sym.index or t_exit_0 not in df_sym.index:
                    continue
                p_in = df_sym.loc[t_entry, "open"]
                p_out = df_sym.loc[t_exit_0, "open"]
                if p_in > 0:
                    ret = (p_out - p_in) / p_in
                    coin_rets.append(ret)

            if coin_rets:
                mean_gross = float(np.mean(coin_rets))
                # Turnover cost (assuming 100% turnover round-trip if new positions)
                # Round-trip cost = 2 * total_one_way_cost
                fee_drag = 2.0 * total_one_way_cost
                funding_drag = 0.0001 * 3.0 * ((168 - d) / 24.0)
                net_ret = mean_gross - fee_drag - funding_drag
            else:
                net_ret = 0.0

            weekly_rets_by_delay[d].append(net_ret)

    # Compute stats for Model 1
    base_rets = np.array(weekly_rets_by_delay[0])
    
    for d in delays_hours:
        r_arr = np.array(weekly_rets_by_delay[d])
        eq = np.cumprod(1.0 + r_arr)
        total_ret = eq[-1] - 1.0
        n_periods = len(r_arr)
        ann_factor = 52.14
        cagr = (eq[-1] ** (ann_factor / n_periods)) - 1.0 if eq[-1] > 0 else -1.0
        
        m_ret = np.mean(r_arr)
        s_ret = np.std(r_arr)
        sharpe = (m_ret / s_ret * np.sqrt(ann_factor)) if s_ret > 1e-6 else 0.0
        
        peaks = np.maximum.accumulate(eq)
        max_dd = np.max((peaks - eq) / peaks)
        
        # Paired t-test vs delay 0
        diff = r_arr - base_rets
        if d == 0:
            p_val = 1.0
            t_stat = 0.0
            mean_diff = 0.0
        else:
            t_stat, p_val = stats.ttest_rel(r_arr, base_rets)
            mean_diff = np.mean(diff)

        # Confidence interval on mean weekly return
        ci95 = 1.96 * (s_ret / np.sqrt(n_periods))

        # Only on active/bullish weeks:
        active_mask = base_rets != 0
        active_diff = r_arr[active_mask] - base_rets[active_mask]
        t_active, p_active = (0.0, 1.0) if d == 0 else stats.ttest_rel(r_arr[active_mask], base_rets[active_mask])

        results_model1.append({
            "Delay_Hours": d,
            "Entry_Time": f"Mon +{d}h" if d < 24 else f"Tue +{d-24}h" if d < 48 else f"Wed +{d-48}h",
            "Total_Net_Ret": total_ret,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max_DD": max_dd,
            "Mean_Weekly_Ret": m_ret,
            "CI95": ci95,
            "Win_Rate": np.mean(r_arr > 0),
            "Mean_Diff_vs_Mon00": mean_diff,
            "t_stat": t_stat,
            "p_value": p_val,
            "p_val_active_weeks": p_active
        })

    df_m1 = pd.DataFrame(results_model1)
    print("\n--- MODEL 1: DELAYED ENTRY WITH SYNCHRONIZED MONDAY 00:00 UTC EXIT ---")
    display_cols = ["Delay_Hours", "Entry_Time", "Total_Net_Ret", "CAGR", "Sharpe", "Max_DD", "Mean_Weekly_Ret", "Mean_Diff_vs_Mon00", "t_stat", "p_value"]
    df_m1_disp = df_m1[display_cols].copy()
    df_m1_disp["Total_Net_Ret"] = df_m1_disp["Total_Net_Ret"].map(lambda x: f"{x*100:+.1f}%")
    df_m1_disp["CAGR"] = df_m1_disp["CAGR"].map(lambda x: f"{x*100:+.1f}%")
    df_m1_disp["Sharpe"] = df_m1_disp["Sharpe"].map(lambda x: f"{x:.2f}")
    df_m1_disp["Max_DD"] = df_m1_disp["Max_DD"].map(lambda x: f"{x*100:.1f}%")
    df_m1_disp["Mean_Weekly_Ret"] = df_m1_disp["Mean_Weekly_Ret"].map(lambda x: f"{x*100:+.2f}%")
    df_m1_disp["Mean_Diff_vs_Mon00"] = df_m1_disp["Mean_Diff_vs_Mon00"].map(lambda x: f"{x*100:+.2f}%")
    df_m1_disp["t_stat"] = df_m1_disp["t_stat"].map(lambda x: f"{x:+.2f}")
    df_m1_disp["p_value"] = df_m1_disp["p_value"].map(lambda x: f"{x:.4f}")
    print(df_m1_disp.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 3: ENTRY DELAY SWEEP (FULL 168-HOUR / 7-DAY ROLLING HOLD)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: ENTRY DELAY HORIZONS (FULL 168-HOUR HOLDING PERIOD)")
    logger.info("=" * 80)

    # In Model 2:
    # If delay = d, enter at Mon +d and exit at Next Mon +d (exactly 168 hours of market exposure)
    results_model2 = []
    weekly_rets_m2 = {d: [] for d in delays_hours}

    for w_idx in range(len(df_weeks) - 1):
        w_curr = df_weeks.iloc[w_idx]
        w_next = df_weeks.iloc[w_idx + 1]

        t_curr_0 = pd.Timestamp(w_curr["rebalance_day"], tz="UTC")
        t_next_0 = pd.Timestamp(w_next["rebalance_day"], tz="UTC")

        is_bullish = (w_curr["btc_macro_bullish"] == 1)
        if not is_bullish:
            for d in delays_hours:
                weekly_rets_m2[d].append(0.0)
            continue

        w_ranks = df_rankings[df_rankings["rebalance_day"] == w_curr["rebalance_day"]]
        q_syms = w_ranks[w_ranks["is_top7_sharpe"] == 1]["symbol"].tolist()
        raw_candidates = w_ranks[w_ranks["is_top10_raw"] == 1].sort_values("rank_raw_mom")
        raw_syms = [s for s in raw_candidates["symbol"] if s not in q_syms][:3]
        portfolio_syms = q_syms + raw_syms

        for d in delays_hours:
            t_entry = t_curr_0 + pd.Timedelta(hours=d)
            t_exit = t_next_0 + pd.Timedelta(hours=d)
            coin_rets = []
            for sym in portfolio_syms:
                if sym not in dict_1h:
                    continue
                df_sym = dict_1h[sym]
                if t_entry not in df_sym.index or t_exit not in df_sym.index:
                    continue
                p_in = df_sym.loc[t_entry, "open"]
                p_out = df_sym.loc[t_exit, "open"]
                if p_in > 0:
                    ret = (p_out - p_in) / p_in
                    coin_rets.append(ret)

            if coin_rets:
                mean_gross = float(np.mean(coin_rets))
                fee_drag = 2.0 * total_one_way_cost
                funding_drag = 0.0001 * 3.0 * 7.0
                net_ret = mean_gross - fee_drag - funding_drag
            else:
                net_ret = 0.0

            weekly_rets_m2[d].append(net_ret)

    base_rets_m2 = np.array(weekly_rets_m2[0])
    for d in delays_hours:
        r_arr = np.array(weekly_rets_m2[d])
        eq = np.cumprod(1.0 + r_arr)
        total_ret = eq[-1] - 1.0
        n_periods = len(r_arr)
        ann_factor = 52.14
        cagr = (eq[-1] ** (ann_factor / n_periods)) - 1.0 if eq[-1] > 0 else -1.0
        
        m_ret = np.mean(r_arr)
        s_ret = np.std(r_arr)
        sharpe = (m_ret / s_ret * np.sqrt(ann_factor)) if s_ret > 1e-6 else 0.0
        
        peaks = np.maximum.accumulate(eq)
        max_dd = np.max((peaks - eq) / peaks)
        
        diff = r_arr - base_rets_m2
        if d == 0:
            p_val = 1.0
            t_stat = 0.0
            mean_diff = 0.0
        else:
            t_stat, p_val = stats.ttest_rel(r_arr, base_rets_m2)
            mean_diff = np.mean(diff)

        results_model2.append({
            "Delay_Hours": d,
            "Entry_Time": f"Mon +{d}h" if d < 24 else f"Tue +{d-24}h" if d < 48 else f"Wed +{d-48}h",
            "Total_Net_Ret": total_ret,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max_DD": max_dd,
            "Mean_Weekly_Ret": m_ret,
            "Mean_Diff_vs_Mon00": mean_diff,
            "t_stat": t_stat,
            "p_value": p_val
        })

    df_m2 = pd.DataFrame(results_model2)
    print("\n--- MODEL 2: DELAYED ENTRY WITH FULL 168-HOUR (7-DAY) ROLLING HOLD ---")
    df_m2_disp = df_m2[display_cols].copy()
    df_m2_disp["Total_Net_Ret"] = df_m2_disp["Total_Net_Ret"].map(lambda x: f"{x*100:+.1f}%")
    df_m2_disp["CAGR"] = df_m2_disp["CAGR"].map(lambda x: f"{x*100:+.1f}%")
    df_m2_disp["Sharpe"] = df_m2_disp["Sharpe"].map(lambda x: f"{x:.2f}")
    df_m2_disp["Max_DD"] = df_m2_disp["Max_DD"].map(lambda x: f"{x*100:.1f}%")
    df_m2_disp["Mean_Weekly_Ret"] = df_m2_disp["Mean_Weekly_Ret"].map(lambda x: f"{x*100:+.2f}%")
    df_m2_disp["Mean_Diff_vs_Mon00"] = df_m2_disp["Mean_Diff_vs_Mon00"].map(lambda x: f"{x*100:+.2f}%")
    df_m2_disp["t_stat"] = df_m2_disp["t_stat"].map(lambda x: f"{x:+.2f}")
    df_m2_disp["p_value"] = df_m2_disp["p_value"].map(lambda x: f"{x:.4f}")
    print(df_m2_disp.to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 4: LIMIT ORDER / PULLBACK DIP-BUYING STRATEGY
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 4: LIMIT ORDER / PULLBACK DIP ENTRY EVALUATION")
    logger.info("=" * 80)

    # What if instead of entering at Monday 00:00 at market open,
    # the trader places limit buy orders at X% discount (e.g. -0.5%, -1.0%, -1.5%, -2.0%, -3.0%)?
    # Window to fill: 24 hours (Monday full day).
    # If unfilled:
    # Option A: Remain in cash for that coin (0 return)
    # Option B: Market buy at Monday 24:00 (Tuesday 00:00) close
    discounts = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    limit_results = []

    for disc in discounts:
        disc_pct_str = f"-{disc*100:.1f}%"
        filled_count = 0
        total_orders = 0
        
        # Policy A weekly returns (unfilled = stay in cash)
        policy_a_rets = []
        # Policy B weekly returns (unfilled = market buy at Mon 24:00 / Tue 00:00)
        policy_b_rets = []

        for w_idx in range(len(df_weeks) - 1):
            w_curr = df_weeks.iloc[w_idx]
            w_next = df_weeks.iloc[w_idx + 1]

            t_curr_0 = pd.Timestamp(w_curr["rebalance_day"], tz="UTC")
            t_exit_0 = pd.Timestamp(w_next["rebalance_day"], tz="UTC")

            if w_curr["btc_macro_bullish"] != 1:
                policy_a_rets.append(0.0)
                policy_b_rets.append(0.0)
                continue

            w_ranks = df_rankings[df_rankings["rebalance_day"] == w_curr["rebalance_day"]]
            q_syms = w_ranks[w_ranks["is_top7_sharpe"] == 1]["symbol"].tolist()
            raw_candidates = w_ranks[w_ranks["is_top10_raw"] == 1].sort_values("rank_raw_mom")
            raw_syms = [s for s in raw_candidates["symbol"] if s not in q_syms][:3]
            portfolio_syms = q_syms + raw_syms

            coin_rets_a = []
            coin_rets_b = []

            for sym in portfolio_syms:
                if sym not in dict_1h:
                    continue
                df_sym = dict_1h[sym]
                if t_curr_0 not in df_sym.index or t_exit_0 not in df_sym.index:
                    continue
                
                p0 = df_sym.loc[t_curr_0, "open"]
                p_exit = df_sym.loc[t_exit_0, "open"]
                limit_target = p0 * (1.0 - disc)

                # Check if hit within first 24 hours
                sub_24h = df_sym.loc[t_curr_0 : t_curr_0 + pd.Timedelta(hours=23)]
                min_low_24h = sub_24h["low"].min()
                total_orders += 1

                fee_drag = 2.0 * total_one_way_cost
                funding_drag = 0.0001 * 3.0 * 7.0

                if min_low_24h <= limit_target:
                    # Filled at limit price!
                    filled_count += 1
                    ret = (p_exit - limit_target) / limit_target - fee_drag - funding_drag
                    coin_rets_a.append(ret)
                    coin_rets_b.append(ret)
                else:
                    # Unfilled!
                    # Policy A: 0 return (remains cash)
                    coin_rets_a.append(0.0)
                    
                    # Policy B: Buy at Mon 24:00 (Tue 00:00)
                    t_tue_0 = t_curr_0 + pd.Timedelta(hours=24)
                    if t_tue_0 in df_sym.index:
                        p_tue = df_sym.loc[t_tue_0, "open"]
                        ret_b = (p_exit - p_tue) / p_tue - fee_drag - (0.0001 * 3.0 * 6.0)
                        coin_rets_b.append(ret_b)
                    else:
                        coin_rets_b.append(0.0)

            policy_a_rets.append(float(np.mean(coin_rets_a)) if coin_rets_a else 0.0)
            policy_b_rets.append(float(np.mean(coin_rets_b)) if coin_rets_b else 0.0)

        fill_rate = filled_count / total_orders if total_orders > 0 else 0.0
        
        # Policy A metrics
        eq_a = np.cumprod(1.0 + np.array(policy_a_rets))
        tot_a = eq_a[-1] - 1.0
        sharpe_a = (np.mean(policy_a_rets) / np.std(policy_a_rets) * np.sqrt(52.14)) if np.std(policy_a_rets) > 1e-6 else 0.0

        # Policy B metrics
        eq_b = np.cumprod(1.0 + np.array(policy_b_rets))
        tot_b = eq_b[-1] - 1.0
        sharpe_b = (np.mean(policy_b_rets) / np.std(policy_b_rets) * np.sqrt(52.14)) if np.std(policy_b_rets) > 1e-6 else 0.0

        limit_results.append({
            "Discount_Limit": disc_pct_str,
            "Fill_Rate": f"{fill_rate*100:.1f}% ({filled_count}/{total_orders})",
            "Policy_A_Tot_Net": f"{tot_a*100:+.1f}%",
            "Policy_A_Sharpe": f"{sharpe_a:.2f}",
            "Policy_B_Tot_Net": f"{tot_b*100:+.1f}%",
            "Policy_B_Sharpe": f"{sharpe_b:.2f}",
        })

    print(pd.DataFrame(limit_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # PART 5: MULTI-YEAR DAY-OF-WEEK REBALANCE COMPARISON (2021-2026)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PART 5: MULTI-YEAR REBALANCE DAY COMPARISON (2021-2026)")
    logger.info("=" * 80)

    # Using klines_multiyear_1d:
    # Test weekly rebalancing on every day of the week:
    # Day 0: Monday, Day 1: Tuesday, Day 2: Wednesday, Day 3: Thursday,
    # Day 4: Friday, Day 5: Saturday, Day 6: Sunday
    multi_dir = Path("data_cache/klines_multiyear_1d")
    btc_multi = pd.read_parquet(multi_dir / "BTCUSDT.parquet")
    btc_multi["open_time"] = pd.to_datetime(btc_multi["open_time"], utc=True)
    btc_multi = btc_multi.sort_values("open_time").set_index("open_time")
    s_btc_multi = btc_multi["close"]
    s_btc_ema50 = s_btc_multi.ewm(span=50, adjust=False).mean()

    # Load altcoins multiyear
    alt_files = [f for f in multi_dir.glob("*.parquet") if f.name != "BTCUSDT.parquet"]
    alt_closes = {}
    for f in alt_files:
        sym = f.stem
        try:
            df_a = pd.read_parquet(f)
            if len(df_a) < 100:
                continue
            df_a["open_time"] = pd.to_datetime(df_a["open_time"], utc=True)
            df_a = df_a.sort_values("open_time").set_index("open_time")
            alt_closes[sym] = df_a["close"]
        except Exception:
            continue

    df_multi_p = pd.DataFrame(alt_closes)
    common_multi_idx = df_multi_p.index.intersection(s_btc_multi.index).sort_values()

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    multi_day_results = []

    for d_idx, day_name in enumerate(day_names):
        # All dates with dayofweek == d_idx, separated by 7 days
        all_d_dates = [t for t in common_multi_idx if t.dayofweek == d_idx]
        valid_dates = [t for t in all_d_dates if (t - common_multi_idx[0]).days >= 60]

        d_rets = []
        for i in range(len(valid_dates) - 1):
            t_curr = valid_dates[i]
            t_next = valid_dates[i + 1]

            btc_curr = s_btc_multi.loc[t_curr]
            ema_curr = s_btc_ema50.loc[t_curr]

            if btc_curr < ema_curr:
                d_rets.append(0.0)
                continue

            # Momentum over past 30 days
            idx_curr = common_multi_idx.get_loc(t_curr)
            t_past_30 = common_multi_idx[max(0, idx_curr - 30)]

            p_curr = df_multi_p.loc[t_curr]
            p_past = df_multi_p.loc[t_past_30]

            mask = (p_curr > 0) & (p_past > 0) & (~p_curr.isna()) & (~p_past.isna())
            ret_30d = (p_curr[mask] - p_past[mask]) / p_past[mask]

            if len(ret_30d) < 10:
                d_rets.append(0.0)
                continue

            top10 = ret_30d.nlargest(10).index.tolist()

            p_next = df_multi_p.loc[t_next]
            fwd_rets = [(p_next[s] - p_curr[s]) / p_curr[s] for s in top10 if s in p_next and not np.isnan(p_next[s])]
            
            if fwd_rets:
                mean_g = float(np.mean(fwd_rets))
                net = mean_g - (2.0 * total_one_way_cost) - (0.0001 * 3.0 * 7.0)
                d_rets.append(net)
            else:
                d_rets.append(0.0)

        r_arr = np.array(d_rets)
        eq = np.cumprod(1.0 + r_arr)
        tot_ret = eq[-1] - 1.0
        n_p = len(r_arr)
        cagr = (eq[-1] ** (52.14 / n_p)) - 1.0 if eq[-1] > 0 else -1.0
        m = np.mean(r_arr)
        s = np.std(r_arr)
        sh = (m / s * np.sqrt(52.14)) if s > 1e-6 else 0.0
        peaks = np.maximum.accumulate(eq)
        mdd = np.max((peaks - eq) / peaks)

        multi_day_results.append({
            "Rebalance_Day": day_name,
            "Total_Periods": n_p,
            "Total_Net_Return": f"{tot_ret*100:+.1f}%",
            "CAGR": f"{cagr*100:+.1f}%",
            "Sharpe_Ratio": f"{sh:.2f}",
            "Max_Drawdown": f"{mdd*100:.1f}%",
            "Mean_Weekly_Return": f"{m*100:+.2f}%",
            "Win_Rate": f"{np.mean(r_arr > 0)*100:.1f}%"
        })

    print(pd.DataFrame(multi_day_results).to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("EMPIRICAL TIMING ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
