"""
analysis/evaluate_multiyear_momentum_barbell.py

Comprehensive 5.5-Year Historical Lifecycle Evaluation (2021-2026)
of the 10-Coin Momentum Barbell Strategy (7 Quality + 3 Raw).

Evaluates across:
1. 2021: The Grand Bull Run ($69k peak, NFT/DeFi craze)
2. 2022: The Great Crypto Winter (Terra/Luna, 3AC, Celsius, FTX collapse to $15.5k)
3. 2023: The Grinding Recovery (SVB bank run, slow grind $16k -> $30k)
4. 2024: The Spot ETF Euphoria & Pre-Halving ATHs ($73k)
5. 2025: Extended Macro Cycle & Altcoin Dispersion
6. 2026: Mid-Year Capitulation Bottom & Massive Rebound
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import logging
import time

from strategy.indicators import compute_ema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiyear_backtest")

DATA_DIR = Path("data_cache/klines_multiyear_1d")

def load_multiyear_data():
    logger.info("Loading multi-year daily dataset from 2021 to 2026...")
    
    # 1. Load BTC
    btc_file = DATA_DIR / "BTCUSDT.parquet"
    if not btc_file.exists():
        raise FileNotFoundError("BTCUSDT.parquet not found in multiyear data directory!")
        
    df_btc = pd.read_parquet(btc_file)
    df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
    df_btc = df_btc.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load all altcoins
    dict_close = {}
    total_files = list(DATA_DIR.glob("*.parquet"))
    
    for f in total_files:
        sym = f.stem
        if sym == "BTCUSDT":
            continue
        try:
            df = pd.read_parquet(f)
            if len(df) < 50:
                continue
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.sort_values("open_time").reset_index(drop=True)
            dict_close[sym] = df.set_index("open_time")["close"]
        except Exception:
            continue
            
    df_prices = pd.DataFrame(dict_close)
    
    # Align common dates with BTC
    common_idx = df_prices.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    
    # Start from March 2021 to ensure 50-day EMA lookback
    valid_mondays = [m for m in mondays if m >= pd.Timestamp("2021-03-01", tz="UTC")]
    
    logger.info(f"Loaded {len(df_prices.columns)} altcoins across {len(valid_mondays)} rebalance Mondays (2021 to 2026).")
    return df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays

def run_multiyear_simulation(
    df_prices,
    s_btc,
    s_btc_ema50,
    common_idx,
    valid_mondays,
    k_quality=7,
    k_raw=3,
    buf_quality=11,
    buf_raw=6,
    fee_rate=0.0012,
    weekly_funding=0.0021,
    initial_capital=7000.0
):
    portfolio_equity = initial_capital
    current_quality = []
    current_raw = []
    
    weekly_records = []
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]
        
        btc_price = s_btc.loc[t_now]
        btc_ema = s_btc_ema50.loc[t_now]
        is_cash = btc_price < btc_ema
        
        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_past = s_btc.loc[t_past_30]
        btc_30d_ret = (btc_price - btc_past) / btc_past
        
        if is_cash:
            target_quality = []
            target_raw = []
        else:
            candidates = []
            for sym in df_prices.columns:
                s_p = df_prices[sym]
                if t_now not in s_p.index or t_past_30 not in s_p.index:
                    continue
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
                
                candidates.append({
                    "symbol": sym,
                    "ret_30d": ret_30d,
                    "rs_spread": rs_spread,
                    "vol_30d": vol_30d,
                    "sharpe_score": sharpe_score
                })
                
            df_c = pd.DataFrame(candidates)
            if df_c.empty:
                target_quality = []
                target_raw = []
            else:
                # 1. Quality Ranking
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
                
                # 2. Raw Ranking
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

        target_symbols = target_quality + target_raw
        target_w = {s: 1.0 / len(target_symbols) for s in target_symbols} if target_symbols else {}
        current_symbols = current_quality + current_raw
        current_w = {s: 1.0 / len(current_symbols) for s in current_symbols} if current_symbols else {}
        
        all_syms = set(current_w.keys()).union(set(target_w.keys()))
        turnover = sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_syms) / 2.0
        turnover_cost = turnover * 2.0 * fee_rate
        
        week_coin_rets = []
        for s in target_symbols:
            p_e = df_prices[s].loc[t_now]
            p_x = df_prices[s].loc[t_next] if t_next in df_prices[s].index else np.nan
            c_ret = (p_x - p_e) / p_e if (not np.isnan(p_x) and p_e > 0) else 0.0
            week_coin_rets.append(c_ret)
            
        if is_cash or not week_coin_rets:
            gross_ret = 0.0
            funding_drag = 0.0
        else:
            gross_ret = float(np.mean(week_coin_rets))
            funding_drag = weekly_funding
            
        net_ret = gross_ret - turnover_cost - funding_drag
        portfolio_equity = portfolio_equity * (1.0 + net_ret)
        
        btc_next_p = s_btc.loc[t_next] if t_next in s_btc.index else btc_price
        btc_week_ret = (btc_next_p - btc_price) / btc_price
        
        weekly_records.append({
            "week_idx": i,
            "monday": t_now,
            "year": t_now.year,
            "is_cash": is_cash,
            "active_coins": len(target_symbols),
            "gross_ret": gross_ret,
            "turnover_cost": turnover_cost,
            "funding_cost": funding_drag,
            "net_ret": net_ret,
            "btc_ret": btc_week_ret,
            "equity": portfolio_equity
        })
        
        current_quality = target_quality
        current_raw = target_raw
        
    return pd.DataFrame(weekly_records)

def analyze_multiyear_results(df_w, initial_capital=7000.0):
    print("\n" + "=" * 115)
    print("5.5-YEAR MULTI-YEAR PERFORMANCE EVALUATION (2021 - 2026)")
    print("Exact 10-Coin Momentum Barbell Strategy (7 Quality + 3 Raw with BTC 50-day EMA Gate)")
    print("=" * 115)
    
    # 1. Total Lifetime Stats
    w_arr = df_w["net_ret"].values
    btc_w_arr = df_w["btc_ret"].values
    
    eq_curve = initial_capital * np.cumprod(1.0 + w_arr)
    btc_curve = initial_capital * np.cumprod(1.0 + btc_w_arr)
    
    final_eq = eq_curve[-1]
    final_btc = btc_curve[-1]
    
    tot_ret = (final_eq - initial_capital) / initial_capital * 100.0
    btc_tot_ret = (final_btc - initial_capital) / initial_capital * 100.0
    
    n_weeks = len(df_w)
    n_years = n_weeks / 52.0
    cagr = ((final_eq / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0
    btc_cagr = ((final_btc / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0
    
    peaks = np.maximum.accumulate(eq_curve)
    max_dd = float(np.max((peaks - eq_curve) / peaks)) * 100.0
    
    btc_peaks = np.maximum.accumulate(btc_curve)
    btc_max_dd = float(np.max((btc_peaks - btc_curve) / btc_peaks)) * 100.0
    
    sharpe = (np.mean(w_arr) / np.std(w_arr) * np.sqrt(52.0)) if np.std(w_arr) > 1e-6 else 0.0
    btc_sharpe = (np.mean(btc_w_arr) / np.std(btc_w_arr) * np.sqrt(52.0)) if np.std(btc_w_arr) > 1e-6 else 0.0
    
    cash_pct = (df_w["is_cash"].mean()) * 100.0
    active_weeks = df_w[~df_w["is_cash"]]
    active_wr = (active_weeks["net_ret"] > 0).mean() * 100.0
    
    print(f"\nLIFETIME SUMMARY (March 2021 to October 2026: {n_years:.2f} Years | {n_weeks} Weeks):")
    print(f"  • Starting Capital:          ${initial_capital:,.2f}")
    print(f"  • Strategy Ending Capital:   ${final_eq:,.2f} ({tot_ret:+,.1f}%)")
    print(f"  • Bitcoin Buy & Hold End:    ${final_btc:,.2f} ({btc_tot_ret:+,.1f}%)")
    print(f"  • Net Alpha vs Bitcoin:      {tot_ret - btc_tot_ret:+,.1f}% Total Excess Return")
    print(f"  • Strategy Annualized CAGR:  {cagr:+.1f}% vs BTC: {btc_cagr:+.1f}%")
    print(f"  • Strategy Lifetime Sharpe:  {sharpe:.2f} vs BTC: {btc_sharpe:.2f}")
    print(f"  • Strategy Maximum Drawdown: {max_dd:.1f}% vs BTC: {btc_max_dd:.1f}%")
    print(f"  • Cash Shield Exposure:      {cash_pct:.1f}% of all weeks spent in 100% USDT Cash")
    print(f"  • Active Trading Win Rate:   {active_wr:.1f}% across {len(active_weeks)} active rebalance weeks")

    # 2. Year-by-Year Table
    print("\n" + "=" * 125)
    print("YEAR-BY-YEAR REGIME BREAKDOWN (HOW DID IT HANDLE BULLS, CRASHES, AND BEAR WINTERS?)")
    print("=" * 125)
    print(f"{'Year':<6} | {'Market Regime':<32} | {'Portfolio Net':<14} | {'BTC Return':<12} | {'Alpha vs BTC':<13} | {'Sharpe':<7} | {'Max DD':<8} | {'Cash Time':<10} | {'Active WR':<9}")
    print("-" * 125)
    
    regime_names = {
        2021: "Bull Peak & Alt Season (Mar-Dec)",
        2022: "Great Crypto Winter / FTX Crash",
        2023: "Silicon Valley Bank & Recovery",
        2024: "Spot ETF Launch & New ATHs",
        2025: "Macro Cycle & Chop",
        2026: "July Bottom & Alt Rebound"
    }
    
    for yr, grp in df_w.groupby("year"):
        r_w = grp["net_ret"].values
        btc_r_w = grp["btc_ret"].values
        
        y_eq = np.cumprod(1.0 + r_w)
        y_btc_eq = np.cumprod(1.0 + btc_r_w)
        
        y_net = (y_eq[-1] - 1.0) * 100.0
        y_btc = (y_btc_eq[-1] - 1.0) * 100.0
        y_alpha = y_net - y_btc
        
        y_pks = np.maximum.accumulate(y_eq)
        y_dd = float(np.max((y_pks - y_eq) / y_pks)) * 100.0 if len(y_pks) > 0 else 0.0
        
        y_sharpe = (np.mean(r_w) / np.std(r_w) * np.sqrt(52.0)) if np.std(r_w) > 1e-6 else 0.0
        y_cash = grp["is_cash"].mean() * 100.0
        
        act = grp[~grp["is_cash"]]
        y_wr = (act["net_ret"] > 0).mean() * 100.0 if len(act) > 0 else 0.0
        
        name = regime_names.get(yr, "Full Calendar Year")
        print(f"{yr:<6} | {name:<32} | {y_net:+12.1f}% | {y_btc:+10.1f}% | {y_alpha:+11.1f}% | {y_sharpe:6.2f} | {y_dd:6.1f}% | {y_cash:8.1f}% | {y_wr:7.1f}%")
        
    print("-" * 125)

if __name__ == "__main__":
    df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays = load_multiyear_data()
    df_w = run_multiyear_simulation(df_prices, s_btc, s_btc_ema50, common_idx, valid_mondays)
    analyze_multiyear_results(df_w)
