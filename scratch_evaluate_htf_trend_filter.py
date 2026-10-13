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

print("Loading BTC klines...", flush=True)
df_1h = cache.load_klines("BTC_USDT", "1h").sort_values("open_time").reset_index(drop=True)
df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)

df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)
df_4h["open_time"] = pd.to_datetime(df_4h["open_time"], utc=True)

df_1d = KlineResampler.resample_1h_to_1d(df_1h).sort_values("open_time").reset_index(drop=True)
df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)

print(f"Loaded: 1h ({len(df_1h)}), 4h ({len(df_4h)}), 1d ({len(df_1d)}) bars", flush=True)

# Compute 1D EMAs: 20, 50, 200
df_1d["ema20"] = df_1d["close"].ewm(span=20, adjust=False).mean()
df_1d["ema50"] = df_1d["close"].ewm(span=50, adjust=False).mean()
df_1d["ema200"] = df_1d["close"].ewm(span=200, adjust=False).mean()

# Daily candle at open_time D closes at D + 1 day (00:00 UTC).
# To prevent lookahead, shift daily indicators by 1 day so that at D+1 00:00, the closed D values become effective.
df_1d_shifted = df_1d.copy()
df_1d_shifted["effective_time"] = pd.to_datetime(df_1d_shifted["open_time"] + pd.Timedelta(days=1), utc=True).astype("datetime64[ns, UTC]")
df_1d_shifted = df_1d_shifted.sort_values("effective_time")
df_4h["open_time"] = pd.to_datetime(df_4h["open_time"], utc=True).astype("datetime64[ns, UTC]")
df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True).astype("datetime64[ns, UTC]")

cols_to_merge = ["effective_time", "close", "ema20", "ema50", "ema200"]
df_1d_sub = df_1d_shifted[cols_to_merge].rename(columns={
    "close": "htf_1d_close",
    "ema20": "htf_1d_ema20",
    "ema50": "htf_1d_ema50",
    "ema200": "htf_1d_ema200"
})

# Merge_asof onto 4h and 1h
df_4h = pd.merge_asof(df_4h.sort_values("open_time"), df_1d_sub, left_on="open_time", right_on="effective_time", direction="backward")
df_1h = pd.merge_asof(df_1h.sort_values("open_time"), df_1d_sub, left_on="open_time", right_on="effective_time", direction="backward")
print("Aligned Daily HTF indicators onto 4h and 1h instantly.", flush=True)

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

def run_htf_simulation(
    df: pd.DataFrame,
    tf_name: str,
    htf_filter: str,        # "NONE", "1D_EMA20", "1D_EMA50", "1D_EMA200"
    retrace_level: str,     # "50.0%", "61.8%", "70.5%", "STAGGERED_50_62_70"
    target_mode: str,       # "PEAK_1R", "EXT_1.272"
    fee_bps: float = 15.0
):
    # 75% In-sample
    split_idx = int(len(df) * 0.75)
    df_is = df.iloc[:split_idx]
    
    opens = df_is["open"].values.astype(float)
    highs = df_is["high"].values.astype(float)
    lows = df_is["low"].values.astype(float)
    closes = df_is["close"].values.astype(float)
    atrs = compute_atr_np(highs, lows, closes, 14)
    
    d_closes = df_is["htf_1d_close"].values.astype(float)
    d_ema20 = df_is["htf_1d_ema20"].values.astype(float) if "htf_1d_ema20" in df_is else None
    d_ema50 = df_is["htf_1d_ema50"].values.astype(float)
    d_ema200 = df_is["htf_1d_ema200"].values.astype(float) if "htf_1d_ema200" in df_is else None
    
    n_bars = len(closes)
    trades = []
    i = 30
    in_trade_until = -1
    max_wait_bars = 24
    max_hold_bars = 40
    
    while i < n_bars - 50:
        if i <= in_trade_until:
            i += 1
            continue
            
        cur_atr = atrs[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue
            
        # 1. HTF Trend Gate Check (Evaluated at signal time i)
        if htf_filter == "1D_EMA20":
            if np.isnan(d_ema20[i]) or d_closes[i] < d_ema20[i]:
                i += 1
                continue
        elif htf_filter == "1D_EMA50":
            if np.isnan(d_ema50[i]) or d_closes[i] < d_ema50[i]:
                i += 1
                continue
        elif htf_filter == "1D_EMA200":
            if np.isnan(d_ema200[i]) or d_closes[i] < d_ema200[i]:
                i += 1
                continue
                
        # 2. Impulse Detection (>= 3.0x ATR expansion in <= 6 bars)
        is_impulse = False
        origin_low, peak_high = 0.0, 0.0
        for w in (3, 4, 5, 6):
            sub_l = np.min(lows[i-w:i+1])
            sub_h = np.max(highs[i-w:i+1])
            if highs[i] >= sub_h * 0.999 and (sub_h - sub_l) >= 3.0 * cur_atr:
                is_impulse = True
                origin_low, peak_high = sub_l, sub_h
                break
                
        if not is_impulse or peak_high <= origin_low:
            i += 1
            continue
            
        delta = peak_high - origin_low
        sl_price = origin_low - 0.2 * cur_atr
        tp_price = peak_high if target_mode == "PEAK_1R" else (peak_high + 0.272 * delta)
        
        # 3. Retracement Entry
        filled = False
        entry_price = 0.0
        entry_bar = 0
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            if highs[f] > peak_high * 1.002 or lows[f] <= sl_price:
                break
                
            if retrace_level == "50.0%":
                t_price = peak_high - 0.50 * delta
                if lows[f] <= t_price:
                    filled = True
                    entry_price = min(opens[f], t_price)
                    entry_bar = f
                    break
            elif retrace_level == "61.8%":
                t_price = peak_high - 0.618 * delta
                if lows[f] <= t_price:
                    filled = True
                    entry_price = min(opens[f], t_price)
                    entry_bar = f
                    break
            elif retrace_level == "70.5%":
                t_price = peak_high - 0.705 * delta
                if lows[f] <= t_price:
                    filled = True
                    entry_price = min(opens[f], t_price)
                    entry_bar = f
                    break
            elif retrace_level == "STAGGERED_50_62_70":
                p50 = peak_high - 0.50 * delta
                p62 = peak_high - 0.618 * delta
                p70 = peak_high - 0.705 * delta
                if lows[f] <= p50:
                    fills = [min(opens[f], p50)]
                    wts = [0.33]
                    if lows[f] <= p62:
                        fills.append(min(opens[f], p62))
                        wts.append(0.33)
                    if lows[f] <= p70:
                        fills.append(min(opens[f], p70))
                        wts.append(0.34)
                    entry_price = sum(p * w for p, w in zip(fills, wts)) / sum(wts)
                    filled = True
                    entry_bar = f
                    break
                    
        if not filled:
            i += 1
            continue
            
        risk = entry_price - sl_price
        reward = tp_price - entry_price
        if risk <= 0 or reward <= 0:
            i = entry_bar + 1
            continue
            
        exit_bar = entry_bar
        r_mult = 0.0
        
        for e in range(entry_bar + 1, min(entry_bar + 1 + max_hold_bars, n_bars)):
            if lows[e] <= sl_price:
                r_mult = -1.0
                exit_bar = e
                break
            if highs[e] >= tp_price:
                r_mult = reward / risk
                exit_bar = e
                break
        else:
            exit_bar = min(entry_bar + max_hold_bars, n_bars - 1)
            pnl = closes[exit_bar] - entry_price
            r_mult = pnl / risk
            
        fee_in_r = (entry_price * (fee_bps / 10000.0) * 2.0) / risk
        trades.append(r_mult - fee_in_r)
        
        in_trade_until = exit_bar
        i = exit_bar + 1
        
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": "0.0%", "ci_95": "N/A", "avg_net_r": "0.00R", "profit_factor": "0.00", "total_r": "0.0R"}
        
    arr = np.array(trades)
    wins = np.sum(arr > 0)
    wr = wins / n
    ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n)
    avg_r = float(np.mean(arr))
    gains = np.sum(arr[arr > 0])
    losses = abs(np.sum(arr[arr <= 0]))
    pf = (gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
    
    return {
        "n": n,
        "win_rate": f"{wr*100:.1f}%",
        "ci_95": f"±{ci*100:.1f}%",
        "avg_net_r": f"{avg_r:+.2f}R",
        "profit_factor": f"{pf:.2f}",
        "total_r": f"{np.sum(arr):+.1f}R"
    }

if __name__ == "__main__":
    print("\nEvaluating HTF Trend Filters on 4h Execution...", flush=True)
    filters_4h = ["NONE", "1D_EMA20", "1D_EMA50", "1D_EMA200"]
    retraces = ["50.0%", "61.8%", "70.5%", "STAGGERED_50_62_70"]
    targets = ["PEAK_1R", "EXT_1.272"]
    
    records = []
    for htf in filters_4h:
        for ret in retraces:
            for tgt in targets:
                res = run_htf_simulation(df_4h, "4h", htf, ret, tgt)
                records.append({
                    "TF": "4h",
                    "HTF Filter": htf,
                    "Retrace Level": ret,
                    "Target": tgt,
                    "N": res["n"],
                    "Win Rate": res["win_rate"],
                    "95% CI": res["ci_95"],
                    "Avg Net R": res["avg_net_r"],
                    "Profit Factor": res["profit_factor"],
                    "Total R": res["total_r"]
                })
                
    df_out = pd.DataFrame(records)
    print("\n" + "=" * 95, flush=True)
    print("4H RETRACE STRATEGY WITH HIGHER TIMEFRAME (1D) TREND FILTERS", flush=True)
    print("=" * 95, flush=True)
    print(df_out.to_string(index=False), flush=True)
    
    # Also evaluate 1h with 1D EMA50 to see if 1h becomes profitable
    print("\nEvaluating 1H Execution with 1D EMA50 Trend Filter...", flush=True)
    rec_1h = []
    for htf in ["NONE", "1D_EMA50"]:
        for ret in retraces:
            for tgt in targets:
                res = run_htf_simulation(df_1h, "1h", htf, ret, tgt)
                rec_1h.append({
                    "TF": "1h",
                    "HTF Filter": htf,
                    "Retrace Level": ret,
                    "Target": tgt,
                    "N": res["n"],
                    "Win Rate": res["win_rate"],
                    "95% CI": res["ci_95"],
                    "Avg Net R": res["avg_net_r"],
                    "Profit Factor": res["profit_factor"],
                    "Total R": res["total_r"]
                })
    df_1h_out = pd.DataFrame(rec_1h)
    print(df_1h_out.to_string(index=False), flush=True)
