import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler

cfg = load_config()
cache = ParquetCache(cfg.data.cache_dir)
all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]

# Load BTC
df_btc_1h = cache.load_klines("BTC_USDT", "1h")
df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
s_btc = df_btc_1d.sort_values("open_time").set_index("open_time")["close"]
s_btc_ema50 = s_btc.ewm(span=50, adjust=False).mean()

# Load 1D closes for altcoins
dict_1d = {}
for sym in all_symbols:
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

# Evaluate 7-day rebalance on each day of the week (0=Mon, 1=Tue, ..., 6=Sun)
days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
results = []

k_qual = 7
k_raw = 3
buf_qual = 11
buf_raw = 6
lookback = 30

for day_idx, day_name in days_map.items():
    cycle_days = [t for t in common_idx if t.weekday() == day_idx]
    
    held_quality = []
    held_raw = []
    capital = 10000.0
    equity_curve = [capital]
    dates = []
    
    for i in range(len(cycle_days) - 1):
        t0 = cycle_days[i]
        t1 = cycle_days[i+1]
        
        loc0 = price_matrix.index.get_loc(t0)
        if loc0 < lookback:
            continue
            
        btc_c = s_btc.loc[t0]
        btc_ema = s_btc_ema50.loc[t0]
        is_bullish = btc_c >= btc_ema
        
        if not is_bullish:
            # 100% Cash
            held_quality = []
            held_raw = []
            equity_curve.append(capital)
            dates.append(t1)
            continue
            
        t_past = price_matrix.index[loc0 - lookback]
        slice_30d = price_matrix.loc[t_past:t0]
        
        p0_btc = s_btc.loc[t_past]
        p1_btc = s_btc.loc[t0]
        btc_30d_ret = (p1_btc - p0_btc) / p0_btc
        
        metrics = []
        for sym in price_matrix.columns:
            series = slice_30d[sym].dropna()
            if len(series) < 25:
                continue
            p_start = series.iloc[0]
            p_end = series.iloc[-1]
            if p_start <= 0:
                continue
            alt_ret = (p_end - p_start) / p_start
            rs_spread = alt_ret - btc_30d_ret
            daily_rets = series.pct_change().dropna()
            vol = float(daily_rets.std() * np.sqrt(365.25)) if len(daily_rets) >= 20 else 1.0
            if np.isnan(vol) or vol <= 0.05:
                vol = 0.05
            sharpe = alt_ret / vol
            metrics.append({
                "sym": sym,
                "alt_ret": alt_ret,
                "rs_spread": rs_spread,
                "sharpe": sharpe
            })
            
        df_m = pd.DataFrame(metrics)
        df_q = df_m.sort_values(by="sharpe", ascending=False).reset_index(drop=True)
        df_r = df_m.sort_values(by="rs_spread", ascending=False).reset_index(drop=True)
        
        q_buf = df_q.head(buf_qual)["sym"].tolist()
        q_top = df_q.head(k_qual)["sym"].tolist()
        holds_q = [s for s in held_quality if s in q_buf]
        slots_q = k_qual - len(holds_q)
        buys_q = [s for s in q_top if s not in holds_q][:slots_q]
        final_q = holds_q + buys_q
        
        df_r_filt = df_r[~df_r["sym"].isin(final_q)].reset_index(drop=True)
        r_buf = df_r_filt.head(buf_raw)["sym"].tolist()
        r_top = df_r_filt.head(k_raw)["sym"].tolist()
        holds_r = [s for s in held_raw if s in r_buf and s not in final_q]
        slots_r = k_raw - len(holds_r)
        buys_r = [s for s in r_top if s not in holds_r][:slots_r]
        final_r = holds_r + buys_r
        
        held_quality = final_q
        held_raw = final_r
        all_10 = final_q + final_r
        
        # Calculate forward 7-day return to t1
        p_t0 = price_matrix.loc[t0, all_10]
        p_t1 = price_matrix.loc[t1, all_10]
        rets = (p_t1 - p_t0) / p_t0
        avg_ret = rets.mean()
        
        # Apply costs: 0.15% on buys
        turnover_cost = (len(buys_q) + len(buys_r)) * 0.1 * 0.0015
        net_period_ret = avg_ret - turnover_cost
        
        capital *= (1.0 + net_period_ret)
        equity_curve.append(capital)
        dates.append(t1)
        
    total_ret = (capital - 10000.0) / 10000.0 * 100.0
    s_eq = pd.Series(equity_curve)
    peak = s_eq.cummax()
    dd = (s_eq - peak) / peak
    max_dd = dd.min() * 100.0
    
    # Sharpe
    period_rets = s_eq.pct_change().dropna()
    ann_sharpe = float(period_rets.mean() / period_rets.std() * np.sqrt(52)) if period_rets.std() > 0 else 0.0
    
    results.append({
        "Day": day_name,
        "Total Net Return": f"{total_ret:+.1f}%",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Sharpe": f"{ann_sharpe:.2f}",
        "Ending Capital": f"${capital:,.0f}"
    })

df_res = pd.DataFrame(results)
print("\n" + "=" * 70)
print("EMPIRICAL COMPARISON: 7-DAY REBALANCE BY DAY OF WEEK")
print("=" * 70)
print(df_res.to_string(index=False))
print("=" * 70)
