"""
analysis/evaluate_critique_advanced_stress_tests.py

Executes the 3 Core Empirical Stress Tests for the 10-Coin Momentum Barbell Strategy:
1. Randomized Null Baseline (Monte Carlo N=1,000):
   - Compares the 10-coin Barbell (7 Quality + 3 Raw) against 1,000 simulations of picking
     10 random liquid altcoins weekly under the identical BTC 50-day EMA gate.
   - Computes empirical p-value and true alpha vs random coin-picking.
2. Severe Outlier Pruning:
   - Removes Top 1%, Top 3%, Top 5%, and Top 10% of individual coin trade returns.
   - Removes the single best 4 weeks of portfolio performance entirely.
   - Evaluates whether the edge survives or collapses into negative expectancy.
3. Volume-Tiered Dynamic Slippage & Bull Funding Stress Test:
   - Tests 4 cost tiers ranging from baseline to harsh tiered slippage (up to 85 bps one-way
     on raw coins) and elevated bull market funding drag (up to 1.05%/week).
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import logging
import time

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("advanced_stress_tests")

def load_data():
    logger.info("Loading cache and preparing data...")
    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    
    # 1. BTC Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Altcoin Data
    all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    dict_1d_close = {}

    for sym in all_symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d_close[sym] = df_1d.sort_values("open_time").set_index("open_time")["close"]

    df_prices = pd.DataFrame(dict_1d_close)
    common_idx = df_prices.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]
    
    logger.info(f"Loaded {len(df_prices.columns)} altcoins across {len(valid_mondays)} valid rebalance Mondays.")
    return df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays

def precompute_weekly_metrics(df_prices, s_btc, common_idx, valid_mondays):
    """
    Precomputes 30d return, 30d volatility, and Sharpe score for all coins on each Monday.
    Returns a dict mapping Monday timestamp to a DataFrame of candidate metrics.
    """
    logger.info("Precomputing weekly cross-sectional metrics...")
    weekly_candidates = {}
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_price = s_btc.loc[t_now]
        btc_past = s_btc.loc[t_past_30]
        btc_30d_ret = (btc_price - btc_past) / btc_past
        
        records = []
        for sym in df_prices.columns:
            s_p = df_prices[sym]
            p_now = s_p.loc[t_now]
            p_past = s_p.loc[t_past_30]
            if np.isnan(p_now) or np.isnan(p_past) or p_now <= 0 or p_past <= 0:
                continue
            ret_30d = (p_now - p_past) / p_past
            rs_spread = ret_30d - btc_30d_ret
            
            slice_30d = s_p.iloc[max(0, idx_now - 30) : idx_now + 1]
            pct_changes = slice_30d.pct_change().dropna()
            vol_30d = float(pct_changes.std() * np.sqrt(365.25)) if len(pct_changes) >= 20 else 1.0
            if np.isnan(vol_30d) or vol_30d < 0.05:
                vol_30d = 0.05
            sharpe_score = ret_30d / vol_30d
            
            records.append({
                "symbol": sym,
                "ret_30d": ret_30d,
                "rs_spread": rs_spread,
                "vol_30d": vol_30d,
                "sharpe_score": sharpe_score
            })
        weekly_candidates[t_now] = pd.DataFrame(records)
        
    return weekly_candidates

def run_barbell_backtest(
    df_prices,
    s_btc,
    s_btc_ema50,
    valid_mondays,
    weekly_candidates,
    cost_regime="baseline",
    excluded_trade_indices=None,
    excluded_best_weeks=None,
    k_quality=7,
    k_raw=3,
    buf_quality=11,
    buf_raw=6,
    initial_capital=7000.0
):
    """
    Executes the 10-Coin Barbell Momentum Strategy with exact buffer logic and customizable cost/pruning models.
    """
    if excluded_trade_indices is None:
        excluded_trade_indices = set()
    if excluded_best_weeks is None:
        excluded_best_weeks = set()
        
    current_quality = []
    current_raw = []
    
    portfolio_equity = initial_capital
    weekly_returns = []
    weekly_records = []
    all_coin_trades = []
    trade_counter = 0
    total_fees_paid = 0.0
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]
        
        btc_price = s_btc.loc[t_now]
        btc_ema = s_btc_ema50.loc[t_now]
        is_cash = btc_price < btc_ema
        
        if is_cash:
            target_quality = []
            target_raw = []
        else:
            df_c = weekly_candidates.get(t_now, pd.DataFrame())
            if df_c.empty:
                target_quality = []
                target_raw = []
            else:
                # 1. Quality Ranking (Sharpe score)
                df_q = df_c.sort_values("sharpe_score", ascending=False).reset_index(drop=True)
                quality_buffer_symbols = df_q.head(buf_quality)["symbol"].tolist()
                quality_top_symbols = df_q.head(k_quality)["symbol"].tolist()
                
                holds_q = [s for s in current_quality if s in quality_buffer_symbols]
                slots_q = k_quality - len(holds_q)
                buys_q = []
                for s in quality_top_symbols:
                    if s not in holds_q and len(buys_q) < slots_q:
                        buys_q.append(s)
                target_quality = holds_q + buys_q
                
                # 2. Raw Ranking (RS Spread vs BTC, excluding Quality holdings)
                df_raw_c = df_c[~df_c["symbol"].isin(target_quality)].sort_values("rs_spread", ascending=False).reset_index(drop=True)
                raw_buffer_symbols = df_raw_c.head(buf_raw)["symbol"].tolist()
                raw_top_symbols = df_raw_c.head(k_raw)["symbol"].tolist()
                
                holds_raw = [s for s in current_raw if s in raw_buffer_symbols and s not in target_quality]
                slots_raw = k_raw - len(holds_raw)
                buys_raw = []
                for s in raw_top_symbols:
                    if s not in holds_raw and len(buys_raw) < slots_raw:
                        buys_raw.append(s)
                target_raw = holds_raw + buys_raw

        # Construct target weights (each coin target 10% = 0.10)
        target_symbols = target_quality + target_raw
        target_weights = {s: 1.0 / len(target_symbols) for s in target_symbols} if target_symbols else {}
        current_symbols = current_quality + current_raw
        current_weights = {s: 1.0 / len(current_symbols) for s in current_symbols} if current_symbols else {}

        # Determine cost parameters
        if cost_regime == "baseline":
            fee_rate_q = 0.0012  # 0.12% one-way
            fee_rate_raw = 0.0012
            funding_drag_weekly = 0.0001 * 3.0 * 7 # 0.21% per week
        elif cost_regime == "moderate_stress":
            fee_rate_q = 0.0020  # 0.20% one-way (40 bps round trip)
            fee_rate_raw = 0.0045 # 0.45% one-way (90 bps round trip)
            funding_drag_weekly = 0.0002 * 3.0 * 7 # 0.42% per week
        elif cost_regime == "harsh_stress":
            fee_rate_q = 0.0040  # 0.40% one-way (80 bps round trip)
            fee_rate_raw = 0.0085 # 0.85% one-way (170 bps round trip)
            funding_drag_weekly = 0.00035 * 3.0 * 7 # 0.735% per week
        elif cost_regime == "nightmare_stress":
            fee_rate_q = 0.0100  # 1.00% one-way (200 bps round trip)
            fee_rate_raw = 0.0100
            funding_drag_weekly = 0.0005 * 3.0 * 7 # 1.05% per week
        else:
            fee_rate_q = 0.0012
            fee_rate_raw = 0.0012
            funding_drag_weekly = 0.0021

        # Calculate Turnover Fee Drag
        all_syms = set(current_weights.keys()).union(set(target_weights.keys()))
        turnover_cost = 0.0
        for s in all_syms:
            w_diff = abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            # If coin was or is in raw, apply raw fee, else quality fee
            is_raw = (s in target_raw) or (s in current_raw)
            f_rate = fee_rate_raw if is_raw else fee_rate_q
            turnover_cost += w_diff * f_rate # entry and exit accounted by w_diff accumulation

        # Compute coin returns
        week_coin_rets = []
        for s in target_symbols:
            p_e = df_prices[s].loc[t_now]
            p_x = df_prices[s].loc[t_next] if t_next in df_prices[s].index else np.nan
            c_ret = (p_x - p_e) / p_e if (not np.isnan(p_x) and p_e > 0) else 0.0
            
            # Record trade
            all_coin_trades.append({
                "trade_id": trade_counter,
                "week_idx": i,
                "t_now": t_now,
                "symbol": s,
                "coin_ret": c_ret,
                "is_quality": s in target_quality
            })
            
            # Apply trade pruning if this trade is flagged
            if trade_counter in excluded_trade_indices:
                c_ret = 0.0
                
            week_coin_rets.append(c_ret)
            trade_counter += 1

        if is_cash or not week_coin_rets:
            gross_ret = 0.0
            weekly_funding = 0.0
        else:
            gross_ret = float(np.mean(week_coin_rets))
            weekly_funding = funding_drag_weekly

        net_ret = gross_ret - turnover_cost - weekly_funding
        
        # If this entire week is in excluded_best_weeks, zero it out
        if i in excluded_best_weeks:
            net_ret = 0.0

        portfolio_equity = portfolio_equity * (1.0 + net_ret)
        total_fees_paid += portfolio_equity * (turnover_cost + weekly_funding)
        
        weekly_returns.append(net_ret)
        weekly_records.append({
            "week_idx": i,
            "monday": t_now,
            "is_cash": is_cash,
            "gross_ret": gross_ret,
            "turnover_cost": turnover_cost,
            "funding_cost": weekly_funding,
            "net_ret": net_ret,
            "equity": portfolio_equity
        })
        
        current_quality = target_quality
        current_raw = target_raw

    w_arr = np.array(weekly_returns)
    eq_curve = np.cumprod(1.0 + w_arr)
    peaks = np.maximum.accumulate(eq_curve)
    max_dd = float(np.max((peaks - eq_curve) / peaks)) * 100.0 if len(peaks) > 0 else 0.0
    tot_ret = (portfolio_equity - initial_capital) / initial_capital * 100.0
    
    # Annualized stats (52 weeks per year)
    n_weeks = len(weekly_returns)
    n_years = n_weeks / 52.0
    cagr = ((portfolio_equity / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 and portfolio_equity > 0 else -100.0
    sharpe = (np.mean(w_arr) / np.std(w_arr) * np.sqrt(52.0)) if np.std(w_arr) > 1e-6 else 0.0
    active_weeks = [r for r in weekly_records if not r["is_cash"]]
    win_rate = (np.array([r["net_ret"] for r in active_weeks]) > 0).mean() * 100.0 if active_weeks else 0.0

    return {
        "final_equity": portfolio_equity,
        "tot_ret": tot_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "total_fees_paid": total_fees_paid,
        "all_coin_trades": pd.DataFrame(all_coin_trades),
        "weekly_records": pd.DataFrame(weekly_records),
        "weekly_returns": w_arr
    }

def run_random_portfolio_sim(
    df_prices,
    s_btc,
    s_btc_ema50,
    valid_mondays,
    weekly_candidates,
    k_coins=10,
    fee_rate=0.0012,
    initial_capital=7000.0,
    seed=None
):
    """
    Runs a single simulation of picking 10 random liquid altcoins weekly under the identical BTC 50-day EMA filter.
    """
    if seed is not None:
        np.random.seed(seed)
        
    portfolio_equity = initial_capital
    weekly_returns = []
    current_holdings = []
    funding_drag_weekly = 0.0001 * 3.0 * 7
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]
        
        btc_price = s_btc.loc[t_now]
        btc_ema = s_btc_ema50.loc[t_now]
        is_cash = btc_price < btc_ema
        
        if is_cash:
            target_holdings = []
        else:
            df_c = weekly_candidates.get(t_now, pd.DataFrame())
            if df_c.empty:
                target_holdings = []
            else:
                available_syms = df_c["symbol"].tolist()
                n_pick = min(k_coins, len(available_syms))
                target_holdings = list(np.random.choice(available_syms, size=n_pick, replace=False))
                
        target_w = {s: 1.0 / len(target_holdings) for s in target_holdings} if target_holdings else {}
        current_w = {s: 1.0 / len(current_holdings) for s in current_holdings} if current_holdings else {}
        
        # Turnover
        all_syms = set(current_w.keys()).union(set(target_w.keys()))
        turnover = sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_syms) / 2.0
        turnover_cost = turnover * 2.0 * fee_rate
        
        # Coin returns
        coin_rets = []
        for s in target_holdings:
            p_e = df_prices[s].loc[t_now]
            p_x = df_prices[s].loc[t_next] if t_next in df_prices[s].index else np.nan
            c_ret = (p_x - p_e) / p_e if (not np.isnan(p_x) and p_e > 0) else 0.0
            coin_rets.append(c_ret)
            
        gross_ret = float(np.mean(coin_rets)) if coin_rets else 0.0
        weekly_funding = funding_drag_weekly if not is_cash and target_holdings else 0.0
        net_ret = gross_ret - turnover_cost - weekly_funding
        
        portfolio_equity = portfolio_equity * (1.0 + net_ret)
        weekly_returns.append(net_ret)
        current_holdings = target_holdings
        
    w_arr = np.array(weekly_returns)
    eq_curve = np.cumprod(1.0 + w_arr)
    peaks = np.maximum.accumulate(eq_curve)
    max_dd = float(np.max((peaks - eq_curve) / peaks)) * 100.0 if len(peaks) > 0 else 0.0
    tot_ret = (portfolio_equity - initial_capital) / initial_capital * 100.0
    sharpe = (np.mean(w_arr) / np.std(w_arr) * np.sqrt(52.0)) if np.std(w_arr) > 1e-6 else 0.0
    
    return {
        "final_equity": portfolio_equity,
        "tot_ret": tot_ret,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

def main():
    print("=" * 100)
    print("MOMENTUM BARBELL SYSTEM: ADVANCED EMPIRICAL STRESS TESTS")
    print("Directly answering the 3 Core Institutional Critique Challenges")
    print("=" * 100)

    df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays = load_data()
    weekly_candidates = precompute_weekly_metrics(df_prices, s_btc, common_idx, valid_mondays)
    
    # 0. Run Baseline Barbell Strategy
    logger.info("Running Baseline 10-Coin Barbell Strategy...")
    baseline = run_barbell_backtest(df_prices, s_btc, s_btc_ema50, valid_mondays, weekly_candidates)
    
    print("\n" + "#" * 100)
    print(f"BASELINE 10-COIN BARBELL SYSTEM (7 Quality + 3 Raw, Buffer 11/6):")
    print(f"  • Ending Capital:      ${baseline['final_equity']:,.2f} (from $7,000.00)")
    print(f"  • Total Net Return:    {baseline['tot_ret']:+.2f}%")
    print(f"  • Net Annualized CAGR: {baseline['cagr']:+.2f}%")
    print(f"  • Annualized Sharpe:   {baseline['sharpe']:.2f}")
    print(f"  • Max Drawdown:        {baseline['max_dd']:.2f}%")
    print(f"  • Active Week Win Rate:{baseline['win_rate']:.1f}%")
    print(f"  • Total Executed Trades: {len(baseline['all_coin_trades'])} coin-weeks")
    print("#" * 100 + "\n")

    # =========================================================================
    # TEST 1: RANDOMIZED NULL BASELINE (Monte Carlo N=1,000)
    # =========================================================================
    print("=" * 100)
    print("TEST 1: RANDOMIZED NULL BASELINE (MONTE CARLO N = 1,000 ITERATIONS)")
    print("Question: Does our 7 Quality + 3 Raw Barbell generate genuine ranking alpha,")
    print("          or does picking ANY 10 liquid coins under the BTC 50-day EMA gate give the same result?")
    print("=" * 100)
    
    N_SIMS = 1000
    random_results = []
    t0 = time.time()
    for sim_i in range(N_SIMS):
        res = run_random_portfolio_sim(
            df_prices, s_btc, s_btc_ema50, valid_mondays, weekly_candidates,
            k_coins=10, fee_rate=0.0012, seed=sim_i
        )
        random_results.append(res)
    elapsed = time.time() - t0
    logger.info(f"Completed {N_SIMS} Monte Carlo iterations in {elapsed:.2f} seconds.")

    df_mc = pd.DataFrame(random_results)
    
    p_val_return = (df_mc["tot_ret"] >= baseline["tot_ret"]).mean()
    p_val_sharpe = (df_mc["sharpe"] >= baseline["sharpe"]).mean()
    
    alpha_return = baseline["tot_ret"] - df_mc["tot_ret"].median()
    alpha_sharpe = baseline["sharpe"] - df_mc["sharpe"].median()
    
    print(f"\nMONTE CARLO NULL DISTRIBUTION (N = {N_SIMS} Random 10-Coin Portfolios under same BTC Gate):")
    print(f"{'Metric':<25} | {'Barbell Strategy':<18} | {'Random Median':<15} | {'Random 5th Pct':<15} | {'Random 95th Pct':<15} | {'Empirical p-value':<18}")
    print("-" * 115)
    print(f"{'Total Net Return (%)':<25} | {baseline['tot_ret']:+17.1f}% | {df_mc['tot_ret'].median():+14.1f}% | {df_mc['tot_ret'].quantile(0.05):+14.1f}% | {df_mc['tot_ret'].quantile(0.95):+14.1f}% | p = {p_val_return:.4f} {'***' if p_val_return < 0.01 else '*' if p_val_return < 0.05 else '(Not Sig)'}")
    print(f"{'Annualized Sharpe':<25} | {baseline['sharpe']:18.2f} | {df_mc['sharpe'].median():15.2f} | {df_mc['sharpe'].quantile(0.05):15.2f} | {df_mc['sharpe'].quantile(0.95):15.2f} | p = {p_val_sharpe:.4f} {'***' if p_val_sharpe < 0.01 else '*' if p_val_sharpe < 0.05 else '(Not Sig)'}")
    print(f"{'Max Drawdown (%)':<25} | {baseline['max_dd']:17.1f}% | {df_mc['max_dd'].median():14.1f}% | {df_mc['max_dd'].quantile(0.05):14.1f}% | {df_mc['max_dd'].quantile(0.95):14.1f}% | (Lower is better)")
    print(f"{'Ending Equity ($)':<25} | ${baseline['final_equity']:<17,.2f} | ${df_mc['final_equity'].median():<14,.2f} | ${df_mc['final_equity'].quantile(0.05):<14,.2f} | ${df_mc['final_equity'].quantile(0.95):<14,.2f} | p = {p_val_return:.4f}")
    print("-" * 115)
    print(f"• Pure Cross-Sectional Alpha above Random Selection: {alpha_return:+.1f}% Total Return | Sharpe Spread: {alpha_sharpe:+.2f}")
    print(f"• Percentile Rank of Barbell Strategy: {(1.0 - p_val_return)*100:.1f}th Percentile of all random portfolios")

    # =========================================================================
    # TEST 2: SEVERE OUTLIER PRUNING (RIGHT-TAIL FRAGILITY TEST)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TEST 2: SEVERE OUTLIER PRUNING (IS THE EDGE JUST A MEME COIN LOTTERY?)")
    print("Question: If we systematically delete the top 1%, 3%, 5%, and 10% highest winning trades,")
    print("          and the single best 4 weeks of market mania, does the strategy stay net positive?")
    print("=" * 100)
    
    df_trades = baseline["all_coin_trades"].copy()
    total_trades = len(df_trades)
    sorted_trades = df_trades.sort_values("coin_ret", ascending=False)
    
    prune_scenarios = [
        ("Full Baseline (0% pruned)", 0),
        ("Prune Top 1% Coin Trades", int(np.ceil(total_trades * 0.01))),
        ("Prune Top 3% Coin Trades", int(np.ceil(total_trades * 0.03))),
        ("Prune Top 5% Coin Trades", int(np.ceil(total_trades * 0.05))),
        ("Prune Top 10% Coin Trades", int(np.ceil(total_trades * 0.10))),
    ]
    
    print(f"\nTotal Coin-Trades in Backtest: {total_trades}")
    print(f"Top 5 Single Best Trades: " + ", ".join([f"{r['symbol']} ({r['coin_ret']*100:+.1f}%)" for _, r in sorted_trades.head(5).iterrows()]))
    print(f"Worst 5 Single Trades:   " + ", ".join([f"{r['symbol']} ({r['coin_ret']*100:+.1f}%)" for _, r in sorted_trades.tail(5).iterrows()]))
    print("\n" + "-" * 115)
    print(f"{'Pruning Scenario':<32} | {'Trades Pruned':<14} | {'Ending Capital':<15} | {'Total Return':<13} | {'CAGR':<9} | {'Sharpe':<8} | {'Max DD':<9} | {'Win Rate':<9}")
    print("-" * 115)
    
    for label, n_prune in prune_scenarios:
        if n_prune == 0:
            prune_set = set()
        else:
            prune_set = set(sorted_trades.head(n_prune)["trade_id"].values)
            
        res_p = run_barbell_backtest(
            df_prices, s_btc, s_btc_ema50, valid_mondays, weekly_candidates,
            cost_regime="baseline", excluded_trade_indices=prune_set
        )
        print(f"{label:<32} | {n_prune:<14} | ${res_p['final_equity']:<14,.2f} | {res_p['tot_ret']:+11.1f}% | {res_p['cagr']:+7.1f}% | {res_p['sharpe']:<8.2f} | {res_p['max_dd']:7.1f}% | {res_p['win_rate']:7.1f}%")

    # Prune Best 4 Weeks entirely
    df_w_rec = baseline["weekly_records"].copy()
    top_4_weeks = set(df_w_rec.sort_values("net_ret", ascending=False).head(4)["week_idx"].values)
    best_weeks_info = df_w_rec.loc[df_w_rec["week_idx"].isin(top_4_weeks)]
    print("\n--- Single Best 4 Weeks in History (Excluded in Next Stress Test) ---")
    for _, w_row in best_weeks_info.iterrows():
        print(f"  • Week {w_row['week_idx']} ({w_row['monday'].strftime('%Y-%m-%d')}): Net Return = {w_row['net_ret']*100:+.2f}%")
        
    res_no_top4w = run_barbell_backtest(
        df_prices, s_btc, s_btc_ema50, valid_mondays, weekly_candidates,
        cost_regime="baseline", excluded_best_weeks=top_4_weeks
    )
    print(f"\n{'Drop Best 4 Weeks Entirely':<32} | {'4 Weeks (0%)':<14} | ${res_no_top4w['final_equity']:<14,.2f} | {res_no_top4w['tot_ret']:+11.1f}% | {res_no_top4w['cagr']:+7.1f}% | {res_no_top4w['sharpe']:<8.2f} | {res_no_top4w['max_dd']:7.1f}% | {res_no_top4w['win_rate']:7.1f}%")
    print("-" * 115)

    # =========================================================================
    # TEST 3: VOLUME-TIERED DYNAMIC SLIPPAGE & HIGH FUNDING STRESS TEST
    # =========================================================================
    print("\n" + "=" * 100)
    print("TEST 3: VOLUME-TIERED DYNAMIC SLIPPAGE & ELEVATED BULL FUNDING STRESS TEST")
    print("Question: How does the strategy perform under pessimistic execution frictions and crowded Monday rebalancing?")
    print("=" * 100)

    cost_scenarios = [
        (
            "Regime 1: Baseline Execution",
            "baseline",
            "12 bps flat one-way (24 bps RT)",
            "0.21% / week"
        ),
        (
            "Regime 2: Moderate Slippage",
            "moderate_stress",
            "Quality: 20 bps | Raw: 45 bps (90 bps RT)",
            "0.42% / week"
        ),
        (
            "Regime 3: Harsh / Low-Liquidity Drag",
            "harsh_stress",
            "Quality: 40 bps | Raw: 85 bps (170 bps RT)",
            "0.735% / week"
        ),
        (
            "Regime 4: Execution Nightmare Penalty",
            "nightmare_stress",
            "100 bps flat one-way (200 bps RT)",
            "1.05% / week"
        )
    ]

    print(f"\n{'Execution Regime':<35} | {'Assumed Slippage (One-Way)':<38} | {'Weekly Funding':<15} | {'Ending Capital':<15} | {'Total Net Return':<17} | {'Net CAGR':<9} | {'Sharpe':<8} | {'Max DD':<8}")
    print("-" * 155)

    for name, regime_key, slip_desc, fund_desc in cost_scenarios:
        res_c = run_barbell_backtest(
            df_prices, s_btc, s_btc_ema50, valid_mondays, weekly_candidates,
            cost_regime=regime_key
        )
        print(f"{name:<35} | {slip_desc:<38} | {fund_desc:<15} | ${res_c['final_equity']:<14,.2f} | {res_c['tot_ret']:+15.1f}% | {res_c['cagr']:+7.1f}% | {res_c['sharpe']:<8.2f} | {res_c['max_dd']:6.1f}%")

    print("=" * 100)
    print("STRESS TESTS COMPLETE. COMPILING SUMMARY REPORT...")
    print("=" * 100)

if __name__ == "__main__":
    main()
