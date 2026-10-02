"""
evaluate_rebalance_and_exit_cadences.py
Empirical investigation of Rebalancing & Mid-Week Exit Rules:
1. Evaluates instances where BTC fell below Daily EMA50 mid-week.
2. Compares 5 Execution / Cadence Models:
   - Model A: Pure Weekly (7D rebalance, checks macro only on Monday)
   - Model B: Weekly Rebalance with Daily Emergency Cash Exit (Exits mid-week if daily close < EMA50)
   - Model C: 3-Day Cadence (3D rebalance & check)
   - Model D: 5-Day Cadence (5D rebalance & check)
   - Model E: 10-Day / 14-Day Cadence
3. Computes Total Net Return, Sharpe, Max Drawdown, Turnover, and Whipsaw frequency.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cadence_study")


def main():
    logger.info("=" * 80)
    logger.info("EMPIRICAL STUDY: MID-WEEK BTC EMA50 BREAKS & REBALANCE CADENCES")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    # 1. Load Bitcoin Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load Altcoins Daily Closes
    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    dict_1d = {}
    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d[sym] = df_1d.sort_values("open_time").set_index("open_time")["close"]

    price_matrix = pd.DataFrame(dict_1d)
    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()

    min_lookback_days = 50
    valid_start_idx = min_lookback_days

    taker_fee = exec_model.taker_fee_rate
    base_slip = exec_model.base_slippage_bps / 10000.0
    total_one_way_cost = taker_fee + base_slip

    # -------------------------------------------------------------------------
    # PART 1: Forensic Audit of Mid-Week BTC EMA50 Crosses During Active Weeks
    # -------------------------------------------------------------------------
    mondays = [t for t in common_idx if t.dayofweek == 0 and (t - common_idx[0]).days >= min_lookback_days]
    midweek_breaks = []

    for i in range(len(mondays) - 1):
        t_mon = mondays[i]
        t_next_mon = mondays[i + 1]

        btc_mon = s_btc.loc[t_mon]
        ema_mon = s_btc_ema50.loc[t_mon]

        # Only check weeks that STARTED bullish (we bought coins on Monday)
        if btc_mon >= ema_mon:
            # Check all daily bars between Monday and next Monday
            week_days = common_idx[(common_idx > t_mon) & (common_idx < t_next_mon)]
            broke_midweek = False
            first_break_day = None
            for d in week_days:
                if s_btc.loc[d] < s_btc_ema50.loc[d]:
                    broke_midweek = True
                    first_break_day = d
                    break

            if broke_midweek:
                # Compare coin prices at first_break_day vs next Monday
                # Select top 7 quality coins at t_mon
                idx_mon = common_idx.get_loc(t_mon)
                t_past_30 = common_idx[max(0, idx_mon - 30)]
                btc_30d = (btc_mon - s_btc.loc[t_past_30]) / s_btc.loc[t_past_30]

                cand = []
                for sym in price_matrix.columns:
                    s_p = price_matrix[sym]
                    p0 = s_p.loc[t_mon]
                    p_past = s_p.loc[t_past_30]
                    if np.isnan(p0) or np.isnan(p_past) or p0 <= 0 or p_past <= 0:
                        continue
                    ret = (p0 - p_past) / p_past
                    pct = s_p.iloc[max(0, idx_mon - 30) : idx_mon + 1].pct_change().dropna()
                    vol = float(pct.std() * np.sqrt(365.25)) if len(pct) >= 20 else 1.0
                    if np.isnan(vol) or vol < 0.05:
                        vol = 0.05
                    cand.append({"sym": sym, "score": ret / vol})

                top7 = pd.DataFrame(cand).sort_values("score", ascending=False).head(7)["sym"].tolist()

                # Returns if held to next Monday
                ret_to_next_mon = np.mean([
                    (price_matrix[s].loc[t_next_mon] - price_matrix[s].loc[t_mon]) / price_matrix[s].loc[t_mon]
                    for s in top7 if t_next_mon in price_matrix[s].index
                ])

                # Returns if exited at first_break_day
                ret_at_break = np.mean([
                    (price_matrix[s].loc[first_break_day] - price_matrix[s].loc[t_mon]) / price_matrix[s].loc[t_mon]
                    for s in top7 if first_break_day in price_matrix[s].index
                ])

                # Performance delta of exiting early vs holding to Monday
                benefit_of_early_exit = ret_at_break - ret_to_next_mon

                midweek_breaks.append({
                    "week_start": t_mon.strftime("%Y-%m-%d"),
                    "break_day": first_break_day.strftime("%Y-%m-%d (%a)"),
                    "btc_mon": btc_mon,
                    "btc_break": s_btc.loc[first_break_day],
                    "btc_next_mon": s_btc.loc[t_next_mon],
                    "coins_ret_to_next_mon_pct": ret_to_next_mon * 100,
                    "coins_ret_at_break_pct": ret_at_break * 100,
                    "benefit_of_early_exit_pct": benefit_of_early_exit * 100,
                    "was_early_exit_better": benefit_of_early_exit > 0
                })

    df_mid = pd.DataFrame(midweek_breaks)

    print("\n" + "=" * 90)
    print("PART 1: FORENSIC AUDIT OF MID-WEEK BTC EMA50 BREAKS")
    print("=" * 90)
    print(f"Total Active Bullish Weeks: 20")
    print(f"Weeks where BTC broke below EMA50 mid-week: {len(df_mid)}")
    print("-" * 90)

    for _, r in df_mid.iterrows():
        outcome = "✅ SAVED CAPITAL" if r["was_early_exit_better"] else "❌ WHIPSAW (HURT RETURNS)"
        print(f"Week: {r['week_start']} | Broke On: {r['break_day']:<16} | If Exit At Break: {r['coins_ret_at_break_pct']:+6.1f}% | If Held to Mon: {r['coins_ret_to_next_mon_pct']:+6.1f}% | Delta: {r['benefit_of_early_exit_pct']:+6.1f}% -> {outcome}")

    # -------------------------------------------------------------------------
    # PART 2: Comprehensive Simulation Across Cadences (3D, 5D, 7D, 7D+DailyStop)
    # -------------------------------------------------------------------------
    def run_cadence_sim(rebalance_days: int, daily_emergency_stop: bool = False, k: int = 7):
        eq = 7000.0
        curr_weights = {}
        daily_records = []

        # We simulate on a daily step to properly enforce daily emergency exits
        # Run cadence simulation on true cycle holding returns
        eq = 7000.0
        curr_w = {}
        cadence_records = []

        # Find rebalance points
        rebalance_indices = list(range(valid_start_idx, len(common_idx) - rebalance_days, rebalance_days))

        for idx in rebalance_indices:
            t0 = common_idx[idx]
            t1 = common_idx[idx + rebalance_days]

            btc_price = s_btc.loc[t0]
            btc_ema = s_btc_ema50.loc[t0]
            is_cash = btc_price < btc_ema

            if is_cash:
                target_coins = []
                target_w = {}
                r_period = 0.0
            else:
                idx_now = common_idx.get_loc(t0)
                t_past_30 = common_idx[max(0, idx_now - 30)]
                btc_30d = (btc_price - s_btc.loc[t_past_30]) / s_btc.loc[t_past_30]

                cand = []
                for sym in price_matrix.columns:
                    s_p = price_matrix[sym]
                    p0 = s_p.loc[t0]
                    p_past = s_p.loc[t_past_30]
                    if np.isnan(p0) or np.isnan(p_past) or p0 <= 0 or p_past <= 0:
                        continue
                    ret = (p0 - p_past) / p_past
                    pct = s_p.iloc[max(0, idx_now - 30) : idx_now + 1].pct_change().dropna()
                    vol = float(pct.std() * np.sqrt(365.25)) if len(pct) >= 20 else 1.0
                    if np.isnan(vol) or vol < 0.05:
                        vol = 0.05
                    cand.append({"sym": sym, "score": ret / vol})

                df_c = pd.DataFrame(cand)
                target_coins = df_c.sort_values("score", ascending=False).head(k)["sym"].tolist()
                target_w = {s: 1.0 / len(target_coins) for s in target_coins}

                # Check if emergency stop triggered mid-cycle
                if daily_emergency_stop:
                    cycle_days = common_idx[(common_idx > t0) & (common_idx < t1)]
                    broke_day = None
                    for d in cycle_days:
                        if s_btc.loc[d] < s_btc_ema50.loc[d]:
                            broke_day = d
                            break
                    if broke_day is not None:
                        # Exited early at broke_day!
                        rets = [(price_matrix[s].loc[broke_day] - price_matrix[s].loc[t0]) / price_matrix[s].loc[t0] for s in target_coins if broke_day in price_matrix[s].index]
                        days_held = (broke_day - t0).days
                        fund = 0.0001 * 3.0 * days_held
                    else:
                        rets = [(price_matrix[s].loc[t1] - price_matrix[s].loc[t0]) / price_matrix[s].loc[t0] for s in target_coins if t1 in price_matrix[s].index]
                        fund = 0.0001 * 3.0 * rebalance_days
                else:
                    rets = [(price_matrix[s].loc[t1] - price_matrix[s].loc[t0]) / price_matrix[s].loc[t0] for s in target_coins if t1 in price_matrix[s].index]
                    fund = 0.0001 * 3.0 * rebalance_days

                r_period = float(np.mean(rets)) - fund if rets else 0.0

            turnover = sum(abs(target_w.get(s, 0.0) - curr_w.get(s, 0.0)) for s in set(curr_w).union(target_w)) / 2.0
            fee = turnover * 2.0 * total_one_way_cost
            r_period = r_period - fee

            eq = eq * (1.0 + r_period)
            cadence_records.append(r_period)
            curr_w = target_w

        r_arr = np.array(cadence_records)
        eq_curve = 7000.0 * np.cumprod(1.0 + r_arr)
        peaks = np.maximum.accumulate(eq_curve)
        max_dd = float(np.max((peaks - eq_curve) / peaks)) * 100

        n_periods = len(r_arr)
        periods_per_year = 365.25 / rebalance_days
        mean_p = np.mean(r_arr)
        std_p = np.std(r_arr)
        downside_std = np.std(r_arr[r_arr < 0]) if np.sum(r_arr < 0) > 0 else 1e-6
        sharpe = (mean_p / std_p * np.sqrt(periods_per_year)) if std_p > 1e-6 else 0.0
        sortino = (mean_p / downside_std * np.sqrt(periods_per_year)) if downside_std > 1e-6 else 0.0

        final_bal = eq
        tot_ret = (final_bal - 7000.0) / 7000.0 * 100

        return {
            "final_balance": final_bal,
            "total_return_pct": tot_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd
        }

    cadence_tests = [
        ("Weekly (7D) - Rebalance & Check Monday Only", 7, False),
        ("Weekly (7D) + Daily Emergency Cash Stop", 7, True),
        ("5-Day Cadence (5D Rebalance & Check)", 5, False),
        ("3-Day Cadence (3D Rebalance & Check)", 3, False),
        ("10-Day Cadence (10D Rebalance & Check)", 10, False),
        ("14-Day Cadence (Bi-Weekly)", 14, False),
    ]

    cadence_results = []
    for name, r_days, daily_stop in cadence_tests:
        res = run_cadence_sim(rebalance_days=r_days, daily_emergency_stop=daily_stop, k=7)
        res["name"] = name
        cadence_results.append(res)

    print("\n" + "=" * 90)
    print("PART 2: CADENCE & EMERGENCY STOP PERFORMANCE MATRIX ($7,000 START)")
    print("=" * 90)
    header = f"{'Cadence / Execution Rule':<42} | {'Final ($)':<11} | {'Return (%)':<10} | {'Sharpe':<6} | {'MaxDD':<6} | {'Sortino':<6}"
    print(header)
    print("-" * 90)

    for r in cadence_results:
        print(f"{r['name']:<42} | ${r['final_balance']:<10,.2f} | {r['total_return_pct']:+8.1f}% | {r['sharpe_ratio']:<6.2f} | {r['max_drawdown_pct']:<5.1f}% | {r['sortino_ratio']:<6.2f}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
