import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from data.config import load_config
from data.cache import ParquetCache

cfg = load_config()
cache = ParquetCache(cfg.data.cache_dir)
all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]

print("Loading 1h data...")
df_btc = cache.load_klines("BTC_USDT", "1h")
df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
s_btc = df_btc.sort_values("open_time").set_index("open_time")["close"]
s_btc_ema50d = s_btc.ewm(span=50 * 24, adjust=False).mean()

dict_1h = {}
for sym in all_symbols:
    df = cache.load_klines(sym, "1h")
    if df.empty or len(df) < 70 * 24:
        continue
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    dict_1h[sym] = df.sort_values("open_time").set_index("open_time")["close"]

price_matrix = pd.DataFrame(dict_1h)
common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
price_matrix = price_matrix.loc[common_idx]
s_btc = s_btc.loc[common_idx]
s_btc_ema50d = s_btc_ema50d.loc[common_idx]

print(f"Matrix shape: {price_matrix.shape}, Dates: {common_idx[0]} to {common_idx[-1]}")

# Precompute 30-day (720h) momentum and volatility
lookback_h = 30 * 24
ret_30d = price_matrix.pct_change(lookback_h)
btc_ret_30d = s_btc.pct_change(lookback_h)

# Rolling annual volatility (720h window)
# Daily returns std * sqrt(365.25) -> hourly returns std * sqrt(365.25 * 24)
hourly_rets = price_matrix.pct_change()
vol_30d = hourly_rets.rolling(lookback_h, min_periods=400).std() * np.sqrt(365.25 * 24)
vol_30d = vol_30d.clip(lower=0.05)
sharpe_30d = ret_30d / vol_30d

# Let's test a simulation function
# scan_time = T, execution_time = T + 1 hour, next rebalance = T + 168 hours (execution at T + 169)
def run_weekly_rebalance(weekday, hour_utc, exec_delay_hours=1):
    # Find all timestamps matching (weekday, hour_utc)
    # in common_idx starting from lookback_h
    valid_times = [
        t for t in common_idx[lookback_h:]
        if t.weekday() == weekday and t.hour == hour_utc
    ]
    
    capital = 10000.0
    fee_bps = 15.0 / 10000.0
    held_quality = []
    held_raw = []
    
    k_qual = 7
    k_raw = 3
    buf_qual = 11
    buf_raw = 6
    
    curve = [capital]
    
    for i in range(len(valid_times) - 1):
        t_scan = valid_times[i]
        t_next_scan = valid_times[i+1]
        
        # Execution happens exec_delay_hours later
        # Find index of t_scan + exec_delay
        t_exec = t_scan + pd.Timedelta(hours=exec_delay_hours)
        t_next_exec = t_next_scan + pd.Timedelta(hours=exec_delay_hours)
        
        if t_exec not in common_idx or t_next_exec not in common_idx:
            continue
            
        # Check BTC regime at t_scan
        is_bullish = s_btc.loc[t_scan] >= s_btc_ema50d.loc[t_scan]
        
        if not is_bullish:
            # Cash
            if held_quality or held_raw:
                # Liquidate
                capital *= (1.0 - fee_bps)
                held_quality = []
                held_raw = []
            curve.append(capital)
            continue
            
        # Get metrics at t_scan
        s_sharpe = sharpe_30d.loc[t_scan].dropna()
        s_ret = ret_30d.loc[t_scan].dropna()
        
        if len(s_sharpe) < 15:
            curve.append(capital)
            continue
            
        qual_ranked = s_sharpe.sort_values(ascending=False)
        raw_ranked = s_ret.sort_values(ascending=False)
        
        # Select target coins
        new_qual = []
        for sym in held_quality:
            if sym in qual_ranked.index:
                r = qual_ranked.index.get_loc(sym) + 1
                if r <= buf_qual:
                    new_qual.append(sym)
        for sym in qual_ranked.index:
            if len(new_qual) >= k_qual:
                break
            if sym not in new_qual:
                new_qual.append(sym)
                
        new_raw = []
        for sym in held_raw:
            if sym in raw_ranked.index:
                r = raw_ranked.index.get_loc(sym) + 1
                if r <= buf_raw:
                    new_raw.append(sym)
        for sym in raw_ranked.index:
            if len(new_raw) >= k_raw:
                break
            if sym not in new_raw:
                new_raw.append(sym)
                
        target_coins = list(set(new_qual + new_raw))
        if not target_coins:
            curve.append(capital)
            continue
            
        # Turnover calculation for fees
        current_holdings = set(held_quality + held_raw)
        target_set = set(target_coins)
        exits = len(current_holdings - target_set)
        entries = len(target_set - current_holdings)
        turnover_fraction = (exits + entries) / (2.0 * max(len(target_set), 1))
        
        # Execute at t_exec, hold until t_next_exec
        p_exec = price_matrix.loc[t_exec, target_coins]
        p_next = price_matrix.loc[t_next_exec, target_coins]
        
        # Calculate return across target_coins
        coin_rets = (p_next - p_exec) / p_exec
        valid_rets = coin_rets.dropna()
        if len(valid_rets) == 0:
            avg_ret = 0.0
        else:
            avg_ret = float(valid_rets.mean())
            
        capital *= (1.0 - turnover_fraction * fee_bps * 2.0)
        capital *= (1.0 + avg_ret)
        held_quality = new_qual
        held_raw = new_raw
        curve.append(capital)
        
    tot_ret = (capital - 10000.0) / 10000.0
    s_curve = pd.Series(curve)
    rets = s_curve.pct_change().dropna()
    mean_r = rets.mean()
    std_r = rets.std()
    sharpe = float((mean_r / std_r) * np.sqrt(52.14)) if std_r > 0 else 0.0
    cummax = s_curve.cummax()
    dd = (s_curve - cummax) / cummax
    max_dd = float(dd.min())
    
    return tot_ret, sharpe, max_dd, capital

# Compare specific interesting scenarios:
# 1. Monday 00:00 scan -> 00:00 exec (0 delay)
# 2. Monday 00:00 scan -> 01:00 exec (1h delay)
# 3. Monday at various hours: 00:00, 01:00, 08:00, 12:00, 16:00, 20:00
# 4. Wednesday at various hours: 00:00, 12:00, 16:00, 20:00
# 5. Full grid across all 7 days for peak hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)

print("\nEvaluating Timing Scenarios...")
scenarios = [
    ("Monday 00:00 UTC (immediate)", 0, 0, 0),
    ("Monday 00:00 scan -> 01:00 exec (1h delay)", 0, 0, 1),
    ("Monday 01:00 scan -> 01:00 exec", 0, 1, 0),
    ("Monday 12:00 scan -> 13:00 exec (noon UTC)", 0, 12, 1),
    ("Monday 16:00 scan -> 17:00 exec (US morning)", 0, 16, 1),
    ("Monday 20:00 scan -> 21:00 exec (US afternoon)", 0, 20, 1),
    ("Tuesday 12:00 scan -> 13:00 exec", 1, 12, 1),
    ("Wednesday 00:00 scan -> 00:00 exec", 2, 0, 0),
    ("Wednesday 00:00 scan -> 01:00 exec (1h delay)", 2, 0, 1),
    ("Wednesday 12:00 scan -> 13:00 exec (noon UTC)", 2, 12, 1),
    ("Wednesday 16:00 scan -> 17:00 exec (US morning)", 2, 16, 1),
    ("Wednesday 20:00 scan -> 21:00 exec (US afternoon)", 2, 20, 1),
    ("Thursday 16:00 scan -> 17:00 exec", 3, 16, 1),
    ("Friday 16:00 scan -> 17:00 exec", 4, 16, 1),
    ("Saturday 16:00 scan -> 17:00 exec", 5, 16, 1),
    ("Sunday 16:00 scan -> 17:00 exec", 6, 16, 1),
]

results = []
for label, w, h, delay in scenarios:
    r, sh, dd, cap = run_weekly_rebalance(w, h, delay)
    results.append({
        "Scenario": label,
        "Net Return": f"{r*100:+.1f}%",
        "Sharpe": f"{sh:.2f}",
        "Max DD": f"{dd*100:.1f}%",
        "Ending ($10k)": f"${cap:,.0f}"
    })

df_res = pd.DataFrame(results)
print("\n" + df_res.to_string(index=False))

# Now let's scan all 24 hours of Monday and Wednesday with 1h delay to see the full intraday curve!
print("\n--- Intraday Curve (1h exec delay): Monday vs Wednesday across 24 Hours ---")
intraday = []
for h in range(24):
    r_mon, sh_mon, dd_mon, cap_mon = run_weekly_rebalance(0, h, 1)
    r_wed, sh_wed, dd_wed, cap_wed = run_weekly_rebalance(2, h, 1)
    intraday.append({
        "Hour UTC": f"{h:02d}:00",
        "Mon Ret": f"{r_mon*100:+.1f}%",
        "Mon Sharpe": f"{sh_mon:.2f}",
        "Wed Ret": f"{r_wed*100:+.1f}%",
        "Wed Sharpe": f"{sh_wed:.2f}"
    })

df_intra = pd.DataFrame(intraday)
print(df_intra.to_string(index=False))
