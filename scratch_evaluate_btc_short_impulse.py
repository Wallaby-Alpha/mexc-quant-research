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

print("Loading BTC klines for Short-Side Analysis...", flush=True)
df_15m = cache.load_klines("BTC_USDT", "15m").sort_values("open_time").reset_index(drop=True)
df_1h = cache.load_klines("BTC_USDT", "1h").sort_values("open_time").reset_index(drop=True)
df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)

print(f"Loaded: 15m ({len(df_15m)}), 1h ({len(df_1h)}), 4h ({len(df_4h)}) bars", flush=True)

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

def run_short_simulation(
    df: pd.DataFrame,
    tf_name: str,
    impulse_def: str,      # "A_ATR", "B_BODIES", "C_BOS"
    retrace_style: str,    # "50.0%", "61.8%", "70.5%", "STAGGERED_50_62_70", "SWEEP_RECLAIM_50"
    target_mode: str,      # "TROUGH_1R", "EXT_1.272"
    fee_bps: float = 15.0
):
    # 75% in-sample guard
    split_idx = int(len(df) * 0.75)
    df_is = df.iloc[:split_idx]
    
    opens = df_is["open"].values.astype(float)
    highs = df_is["high"].values.astype(float)
    lows = df_is["low"].values.astype(float)
    closes = df_is["close"].values.astype(float)
    atrs = compute_atr_np(highs, lows, closes, 14)
    
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
            
        is_impulse = False
        origin_high, trough_low = 0.0, 0.0
        
        # --- Def A: Downward ATR Expansion (drop >= 3.0x ATR in 3 to 6 bars) ---
        if impulse_def == "A_ATR":
            for w in (3, 4, 5, 6):
                sub_h = np.max(highs[i-w:i+1])
                sub_l = np.min(lows[i-w:i+1])
                # Trough must be at bar i or i-1, and drop >= 3x ATR
                if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_high = sub_h
                    trough_low = sub_l
                    break
                    
        # --- Def B: Consecutive Red Bodies (>= 3 bars with body/range >= 65% and drop >= 1.2%) ---
        elif impulse_def == "B_BODIES":
            for w in (3, 4):
                all_red = True
                for idx in range(i-w+1, i+1):
                    rng = highs[idx] - lows[idx]
                    body = opens[idx] - closes[idx]
                    if rng <= 0 or (body / rng) < 0.65 or body <= 0:
                        all_red = False
                        break
                if all_red:
                    sub_h = np.max(highs[i-w+1:i+1])
                    sub_l = np.min(lows[i-w+1:i+1])
                    if (sub_h - sub_l) / sub_h >= 0.012:
                        is_impulse = True
                        origin_high = sub_h
                        trough_low = sub_l
                        break
                        
        # --- Def C: Downward Break of Structure (closes below 20-bar prior low with expansion candle) ---
        elif impulse_def == "C_BOS":
            prior_l = np.min(lows[i-21:i])
            c_rng = highs[i] - lows[i]
            if closes[i] < prior_l and c_rng >= 1.8 * cur_atr and closes[i] < opens[i]:
                is_impulse = True
                origin_high = np.max(highs[i-10:i])
                trough_low = lows[i]
                
        if not is_impulse or origin_high <= trough_low:
            i += 1
            continue
            
        delta = origin_high - trough_low
        sl_price = origin_high + 0.2 * cur_atr
        tp_price = trough_low if target_mode == "TROUGH_1R" else (trough_low - 0.272 * delta)
        
        filled = False
        entry_price = 0.0
        entry_bar = 0
        pump_above_50 = False
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            # If price drops below previous trough before filling retrace, impulse extended
            if lows[f] < trough_low * 0.998 or highs[f] >= sl_price:
                break
                
            if retrace_style == "50.0%":
                p50 = trough_low + 0.50 * delta
                if highs[f] >= p50:
                    filled = True
                    entry_price = max(opens[f], p50)
                    entry_bar = f
                    break
            elif retrace_style == "61.8%":
                p62 = trough_low + 0.618 * delta
                if highs[f] >= p62:
                    filled = True
                    entry_price = max(opens[f], p62)
                    entry_bar = f
                    break
            elif retrace_style == "70.5%":
                p70 = trough_low + 0.705 * delta
                if highs[f] >= p70:
                    filled = True
                    entry_price = max(opens[f], p70)
                    entry_bar = f
                    break
            elif retrace_style == "STAGGERED_50_62_70":
                p50 = trough_low + 0.50 * delta
                p62 = trough_low + 0.618 * delta
                p70 = trough_low + 0.705 * delta
                if highs[f] >= p50:
                    fills = [max(opens[f], p50)]
                    wts = [0.33]
                    if highs[f] >= p62:
                        fills.append(max(opens[f], p62))
                        wts.append(0.33)
                    if highs[f] >= p70:
                        fills.append(max(opens[f], p70))
                        wts.append(0.34)
                    entry_price = sum(p * w for p, w in zip(fills, wts)) / sum(wts)
                    filled = True
                    entry_bar = f
                    break
            elif retrace_style == "SWEEP_RECLAIM_50":
                p50 = trough_low + 0.50 * delta
                if highs[f] > p50:
                    pump_above_50 = True
                if pump_above_50 and closes[f] < p50 and f + 1 < n_bars:
                    # Close back below 50%, short next open
                    filled = True
                    entry_price = opens[f + 1]
                    entry_bar = f + 1
                    break
                    
        if not filled:
            i += 1
            continue
            
        risk = sl_price - entry_price
        reward = entry_price - tp_price
        if risk <= 0 or reward <= 0:
            i = entry_bar + 1
            continue
            
        exit_bar = entry_bar
        r_mult = 0.0
        
        for e in range(entry_bar + 1, min(entry_bar + 1 + max_hold_bars, n_bars)):
            # For short: SL triggered if high >= sl_price
            if highs[e] >= sl_price:
                r_mult = -1.0
                exit_bar = e
                break
            # TP triggered if low <= tp_price
            if lows[e] <= tp_price:
                r_mult = reward / risk
                exit_bar = e
                break
        else:
            exit_bar = min(entry_bar + max_hold_bars, n_bars - 1)
            pnl = entry_price - closes[exit_bar]  # Profit when close is lower
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
    timeframes = [("4h", df_4h), ("1h", df_1h), ("15m", df_15m)]
    impulse_defs = ["A_ATR", "C_BOS"]
    retraces = ["50.0%", "61.8%", "70.5%", "STAGGERED_50_62_70", "SWEEP_RECLAIM_50"]
    targets = ["TROUGH_1R", "EXT_1.272"]
    
    results = []
    print("\nRunning Short-Side Strategy Matrix on BTC...", flush=True)
    for tf_name, df_data in timeframes:
        for imp in impulse_defs:
            for ret in retraces:
                for tgt in targets:
                    res = run_short_simulation(df_data, tf_name, imp, ret, tgt)
                    results.append({
                        "TF": tf_name,
                        "Impulse Def": imp,
                        "Retrace Level": ret,
                        "Target": tgt,
                        "N": res["n"],
                        "Win Rate": res["win_rate"],
                        "95% CI": res["ci_95"],
                        "Avg Net R": res["avg_net_r"],
                        "Profit Factor": res["profit_factor"],
                        "Total R": res["total_r"]
                    })
                    
    df_res = pd.DataFrame(results)
    
    # Filter for N >= 20 and sort by Profit Factor
    valid_res = df_res[df_res["N"] >= 15].copy()
    valid_res["PF_num"] = valid_res["Profit Factor"].astype(float)
    valid_res = valid_res.sort_values("PF_num", ascending=False)
    
    print("\n" + "=" * 95, flush=True)
    print("TOP SHORT-SIDE CONFIGURATIONS ON BTC (N >= 15, Ranked by Profit Factor)", flush=True)
    print("=" * 95, flush=True)
    print(valid_res.drop(columns=["PF_num"]).to_string(index=False), flush=True)
    
    print("\n" + "=" * 95, flush=True)
    print("COMPLETE SHORT-SIDE MATRIX ON BTC (IN-SAMPLE 75%)", flush=True)
    print("=" * 95, flush=True)
    print(df_res.to_string(index=False), flush=True)
