"""
analysis/evaluate_btc_filter_permutations.py

Evaluates 6 distinct permutations of the Macro Trend Filter on the 5.6-Year (2021-2026) dataset:

1. Baseline: Single Monday close > 50-day EMA.
2. User's Asymmetric Reversal Rule (Prolonged Downtrend Guard):
   If BTC was below 50 EMA for >= 3 consecutive weeks, require 2 consecutive weekly closes
   above 50 EMA before re-entering (avoids dead-cat traps). Fast re-entry if < 3 weeks.
3. Universal 2-Close Confirmation:
   Once in cash, ALWAYS require 2 consecutive Monday closes above 50 EMA to re-enter. Exit on 1.
4. Percentage Clearance Buffer (Hysteresis):
   Exit if BTC < 50 EMA. Re-enter only if BTC >= 50 EMA * 1.02 (+2% above EMA).
5. Dual EMA Trend Alignment:
   BTC > 50 EMA AND BTC > 20 EMA AND EMA 20 sloping upward (EMA20_now >= EMA20_past_week).
6. Hybrid Altcoin Market Breadth Gate (The 2024 ETF Divergence Fix):
   BTC > 50 EMA AND >= 40% of liquid altcoins have positive 30-day returns (avoids BTC-only rallies).
"""

import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import logging

from strategy.indicators import compute_ema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("btc_filter_permutations")

DATA_DIR = Path("data_cache/klines_multiyear_1d")

def load_data():
    df_btc = pd.read_parquet(DATA_DIR / "BTCUSDT.parquet")
    df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
    df_btc = df_btc.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)
    s_btc_ema20 = compute_ema(s_btc, span=20)

    dict_close = {}
    for f in DATA_DIR.glob("*.parquet"):
        sym = f.stem
        if sym == "BTCUSDT":
            continue
        try:
            df = pd.read_parquet(f)
            if len(df) < 50:
                continue
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            dict_close[sym] = df.sort_values("open_time").set_index("open_time")["close"]
        except Exception:
            continue

    df_prices = pd.DataFrame(dict_close)
    common_idx = df_prices.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    valid_mondays = [m for m in mondays if m >= pd.Timestamp("2021-03-01", tz="UTC")]
    
    return df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays

def run_permutation(
    perm_name: str,
    df_prices: pd.DataFrame,
    s_btc: pd.Series,
    s_btc_ema50: pd.Series,
    s_btc_ema20: pd.Series,
    common_idx: pd.DatetimeIndex,
    valid_mondays: list,
    initial_capital: float = 7000.0,
    k_quality: int = 7,
    k_raw: int = 3,
    buf_quality: int = 11,
    buf_raw: int = 6,
    fee_rate: float = 0.0012,
    weekly_funding: float = 0.0021
):
    portfolio_equity = initial_capital
    current_quality = []
    current_raw = []
    
    weekly_records = []
    consecutive_weeks_below_ema = 0
    consecutive_weeks_above_ema = 0
    in_cash = True
    
    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]
        
        btc_p = s_btc.loc[t_now]
        btc_ema50 = s_btc_ema50.loc[t_now]
        btc_ema20 = s_btc_ema20.loc[t_now]
        
        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        t_past_7 = common_idx[max(0, idx_now - 7)]
        btc_ema20_prev = s_btc_ema20.loc[t_past_7] if t_past_7 in s_btc_ema20.index else btc_ema20
        btc_past = s_btc.loc[t_past_30]
        btc_30d_ret = (btc_p - btc_past) / btc_past
        
        is_raw_above_50 = btc_p >= btc_ema50
        
        if is_raw_above_50:
            consecutive_weeks_above_ema += 1
            # Reset consecutive below counter
            # consecutive_weeks_below_ema is preserved for logic check before resetting
        else:
            consecutive_weeks_below_ema += 1
            consecutive_weeks_above_ema = 0
            
        # Determine is_cash based on permutation
        is_cash = True
        
        if perm_name == "1_baseline":
            # Direct 1 close > 50 EMA
            is_cash = not is_raw_above_50
            if is_raw_above_50:
                consecutive_weeks_below_ema = 0

        elif perm_name == "2_user_prolonged_downtrend_guard":
            # If below for >= 3 weeks, require 2 weekly closes above 50 EMA to re-enter
            # If below for < 3 weeks, 1 close is sufficient
            if not is_raw_above_50:
                is_cash = True
            else:
                if consecutive_weeks_below_ema >= 3:
                    if consecutive_weeks_above_ema >= 2:
                        is_cash = False
                        consecutive_weeks_below_ema = 0
                    else:
                        is_cash = True  # Still waiting for 2nd close
                else:
                    is_cash = False
                    consecutive_weeks_below_ema = 0

        elif perm_name == "3_universal_2close_confirm":
            # Once in cash, ALWAYS require 2 consecutive closes > 50 EMA to re-enter
            if not is_raw_above_50:
                is_cash = True
            else:
                if in_cash:
                    if consecutive_weeks_above_ema >= 2:
                        is_cash = False
                    else:
                        is_cash = True
                else:
                    is_cash = False

        elif perm_name == "4_hysteresis_buffer_2pct":
            # Exit if BTC < 50 EMA. Re-enter only if BTC >= 50 EMA * 1.02
            if in_cash:
                if btc_p >= btc_ema50 * 1.02:
                    is_cash = False
                else:
                    is_cash = True
            else:
                if btc_p < btc_ema50:
                    is_cash = True
                else:
                    is_cash = False

        elif perm_name == "5_dual_ema_slope":
            # BTC > 50 EMA AND BTC > 20 EMA AND 20 EMA sloping up
            if is_raw_above_50 and (btc_p >= btc_ema20) and (btc_ema20 >= btc_ema20_prev):
                is_cash = False
            else:
                is_cash = True

        elif perm_name == "6_hybrid_alt_breadth_gate":
            # BTC > 50 EMA AND at least 40% of available alts have positive 30d return
            if not is_raw_above_50:
                is_cash = True
            else:
                # Count positive 30d returns among available alts
                alt_rets = []
                for sym in df_prices.columns:
                    s_p = df_prices[sym]
                    if t_now in s_p.index and t_past_30 in s_p.index:
                        p0 = s_p.loc[t_past_30]
                        p1 = s_p.loc[t_now]
                        if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                            alt_rets.append((p1 - p0) / p0)
                if len(alt_rets) >= 20:
                    breadth_pct = (np.array(alt_rets) > 0.0).mean()
                    is_cash = breadth_pct < 0.40  # require >= 40% alts positive
                else:
                    is_cash = False

        in_cash = is_cash

        # Portfolio selection
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
                candidates.append({
                    "symbol": sym,
                    "ret_30d": ret_30d,
                    "rs_spread": rs_spread,
                    "vol_30d": vol_30d,
                    "sharpe_score": ret_30d / vol_30d
                })
            df_c = pd.DataFrame(candidates)
            if df_c.empty:
                target_quality = []
                target_raw = []
            else:
                df_q = df_c.sort_values("sharpe_score", ascending=False).reset_index(drop=True)
                quality_buf = df_q.head(buf_quality)["symbol"].tolist()
                quality_top = df_q.head(k_quality)["symbol"].tolist()
                holds_q = [s for s in current_quality if s in quality_buf]
                slots_q = k_quality - len(holds_q)
                buys_q = [s for s in quality_top if s not in holds_q][:slots_q]
                target_quality = holds_q + buys_q
                
                df_raw_c = df_c[~df_c["symbol"].isin(target_quality)].sort_values("rs_spread", ascending=False).reset_index(drop=True)
                raw_buf = df_raw_c.head(buf_raw)["symbol"].tolist()
                raw_top = df_raw_c.head(k_raw)["symbol"].tolist()
                holds_raw = [s for s in current_raw if s in raw_buf and s not in target_quality]
                slots_raw = k_raw - len(holds_raw)
                buys_raw = [s for s in raw_top if s not in holds_raw][:slots_raw]
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
            
        gross_ret = float(np.mean(week_coin_rets)) if week_coin_rets else 0.0
        funding_drag = weekly_funding if not is_cash and target_symbols else 0.0
        net_ret = gross_ret - turnover_cost - funding_drag
        portfolio_equity = portfolio_equity * (1.0 + net_ret)
        
        btc_next = s_btc.loc[t_next] if t_next in s_btc.index else btc_p
        btc_week_ret = (btc_next - btc_p) / btc_p
        
        weekly_records.append({
            "week_idx": i,
            "monday": t_now,
            "year": t_now.year,
            "is_cash": is_cash,
            "net_ret": net_ret,
            "btc_ret": btc_week_ret,
            "equity": portfolio_equity
        })
        
        current_quality = target_quality
        current_raw = target_raw
        
    return pd.DataFrame(weekly_records)

def main():
    print("=" * 115)
    print("EMPIRICAL COMPARISON: 6 BTC FILTER PERMUTATIONS (5.6 YEARS: 2021 - 2026)")
    print("=" * 115)
    
    df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays = load_data()
    
    permutations = [
        ("1_baseline", "1. Baseline (Single Monday Close > 50 EMA)"),
        ("2_user_prolonged_downtrend_guard", "2. User Rule: 2-Close if >=3w Bear, else 1-Close"),
        ("3_universal_2close_confirm", "3. Universal 2-Close Confirmation (Always 2w)"),
        ("4_hysteresis_buffer_2pct", "4. Hysteresis Buffer (+2.0% above 50 EMA to Enter)"),
        ("5_dual_ema_slope", "5. Dual EMA Alignment (BTC > 50 & 20 EMA Rising)"),
        ("6_hybrid_alt_breadth_gate", "6. Hybrid Alt Breadth (BTC > 50 & >=40% Alts +30d)")
    ]
    
    results = {}
    
    print(f"\n{'Permutation':<45} | {'Ending Capital':<15} | {'Total Return':<13} | {'CAGR':<8} | {'Sharpe':<7} | {'Max DD':<8} | {'Cash Time':<10} | {'Active WR':<9}")
    print("-" * 125)
    
    for key, label in permutations:
        df_w = run_permutation(key, df_prices, s_btc, s_btc_ema50, s_btc_ema20, common_idx, valid_mondays)
        results[key] = df_w
        
        r_arr = df_w["net_ret"].values
        eq = 7000.0 * np.cumprod(1.0 + r_arr)
        fin_eq = eq[-1]
        tot_ret = (fin_eq - 7000.0) / 7000.0 * 100.0
        n_years = len(df_w) / 52.0
        cagr = ((fin_eq / 7000.0) ** (1.0 / n_years) - 1.0) * 100.0
        peaks = np.maximum.accumulate(eq)
        max_dd = float(np.max((peaks - eq) / peaks)) * 100.0
        sharpe = (np.mean(r_arr) / np.std(r_arr) * np.sqrt(52.0)) if np.std(r_arr) > 1e-6 else 0.0
        cash_pct = df_w["is_cash"].mean() * 100.0
        act = df_w[~df_w["is_cash"]]
        wr = (act["net_ret"] > 0).mean() * 100.0 if len(act) > 0 else 0.0
        
        print(f"{label:<45} | ${fin_eq:<14,.2f} | {tot_ret:+11.1f}% | {cagr:+6.1f}% | {sharpe:6.2f} | {max_dd:6.1f}% | {cash_pct:8.1f}% | {wr:7.1f}%")
        
    print("-" * 125)
    
    # Detailed Year-by-Year Comparison for Key Permutations
    print("\n" + "=" * 125)
    print("YEAR-BY-YEAR NET RETURNS COMPARISON (% RETURN PER YEAR)")
    print("=" * 125)
    print(f"{'Year':<6} | {'BTC B&H':<10} | {'1. Baseline':<14} | {'2. User Guard':<14} | {'3. Univ 2-Close':<16} | {'4. +2% Buffer':<14} | {'5. Dual EMA':<14} | {'6. Alt Breadth':<14}")
    print("-" * 125)
    
    years = sorted(list(df_w["year"].unique()))
    for yr in years:
        row_str = f"{yr:<6} | "
        # BTC
        sub_b = results["1_baseline"][results["1_baseline"]["year"] == yr]
        btc_yr = (np.prod(1.0 + sub_b["btc_ret"].values) - 1.0) * 100.0
        row_str += f"{btc_yr:+8.1f}% | "
        
        for key, _ in permutations:
            sub = results[key][results[key]["year"] == yr]
            y_ret = (np.prod(1.0 + sub["net_ret"].values) - 1.0) * 100.0
            row_str += f"{y_ret:+12.1f}% | "
            
        print(row_str)
        
    print("-" * 125)

if __name__ == "__main__":
    main()
