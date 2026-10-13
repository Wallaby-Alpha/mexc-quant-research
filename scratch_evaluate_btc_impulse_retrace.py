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

print("Loading BTC data...", flush=True)
df_15m = cache.load_klines("BTC_USDT", "15m").sort_values("open_time").reset_index(drop=True)
df_1h = cache.load_klines("BTC_USDT", "1h").sort_values("open_time").reset_index(drop=True)
df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)

print(f"BTC 15m: {len(df_15m)} bars | 1h: {len(df_1h)} bars | 4h: {len(df_4h)} bars", flush=True)

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

def run_simulation_fast(
    df: pd.DataFrame,
    tf_name: str,
    impulse_def: str,      # "A_ATR", "B_BODIES", "C_BOS"
    entry_style: str,      # "LIMIT_50", "LIMIT_45", "STAGGERED_45_50_55", "SWEEP_RECLAIM"
    target_mode: str,      # "PEAK_1R", "EXT_1.5R"
    max_hold_bars: int = 40,
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
    trades_r = []
    
    i = 30
    in_trade_until = -1
    max_wait_bars = 24
    
    while i < n_bars - 50:
        if i <= in_trade_until:
            i += 1
            continue
            
        cur_atr = atrs[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue
            
        is_impulse = False
        origin_low = 0.0
        peak_high = 0.0
        
        # --- Def A: ATR Expansion (>= 3.0x ATR within 3-6 bars) ---
        if impulse_def == "A_ATR":
            for w in (3, 4, 5, 6):
                sub_l = np.min(lows[i-w:i+1])
                sub_h = np.max(highs[i-w:i+1])
                if highs[i] >= sub_h * 0.999 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_low = sub_l
                    peak_high = sub_h
                    break
                    
        # --- Def B: Consecutive Strong Green Bodies ---
        elif impulse_def == "B_BODIES":
            for w in (3, 4):
                all_green = True
                for idx in range(i-w+1, i+1):
                    rng = highs[idx] - lows[idx]
                    body = closes[idx] - opens[idx]
                    if rng <= 0 or (body / rng) < 0.65 or body <= 0:
                        all_green = False
                        break
                if all_green:
                    sub_l = np.min(lows[i-w+1:i+1])
                    sub_h = np.max(highs[i-w+1:i+1])
                    if (sub_h - sub_l) / sub_l >= 0.012:
                        is_impulse = True
                        origin_low = sub_l
                        peak_high = sub_h
                        break
                        
        # --- Def C: Break of Structure with Expansion ---
        elif impulse_def == "C_BOS":
            prior_h = np.max(highs[i-21:i])
            c_rng = highs[i] - lows[i]
            if closes[i] > prior_h and c_rng >= 1.8 * cur_atr and closes[i] > opens[i]:
                is_impulse = True
                origin_low = np.min(lows[i-10:i])
                peak_high = highs[i]
                
        if not is_impulse or peak_high <= origin_low:
            i += 1
            continue
            
        delta = peak_high - origin_low
        p_45 = peak_high - 0.45 * delta
        p_50 = peak_high - 0.50 * delta
        p_55 = peak_high - 0.55 * delta
        sl_price = origin_low - 0.2 * cur_atr
        tp_price = peak_high if target_mode == "PEAK_1R" else (peak_high + 0.272 * delta)
        
        filled = False
        entry_price = 0.0
        entry_bar = 0
        dip_below_50 = False
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            b_o = opens[f]
            b_h = highs[f]
            b_l = lows[f]
            b_c = closes[f]
            
            if b_h > peak_high * 1.002 or b_l <= sl_price:
                break
                
            if entry_style == "LIMIT_50":
                if b_l <= p_50:
                    filled = True
                    entry_price = min(b_o, p_50)
                    entry_bar = f
                    break
            elif entry_style == "LIMIT_45":
                if b_l <= p_45:
                    filled = True
                    entry_price = min(b_o, p_45)
                    entry_bar = f
                    break
            elif entry_style == "STAGGERED_45_50_55":
                if b_l <= p_45:
                    fills = [min(b_o, p_45)]
                    wts = [0.33]
                    if b_l <= p_50:
                        fills.append(min(b_o, p_50))
                        wts.append(0.33)
                    if b_l <= p_55:
                        fills.append(min(b_o, p_55))
                        wts.append(0.34)
                    entry_price = sum(p * w for p, w in zip(fills, wts)) / sum(wts)
                    filled = True
                    entry_bar = f
                    break
            elif entry_style == "SWEEP_RECLAIM":
                if b_l < p_50:
                    dip_below_50 = True
                if dip_below_50 and b_c > p_50 and f + 1 < n_bars:
                    filled = True
                    entry_price = opens[f + 1]
                    entry_bar = f + 1
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
        trades_r.append(r_mult - fee_in_r)
        
        in_trade_until = exit_bar
        i = exit_bar + 1
        
    n = len(trades_r)
    if n == 0:
        return {"n": 0, "win_rate": "0.0%", "ci_95": "N/A", "avg_net_r": "0.00R", "profit_factor": "0.00", "total_r": "0.0R"}
        
    arr = np.array(trades_r)
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
    # Run full matrix
    timeframes = [("15m", df_15m), ("1h", df_1h), ("4h", df_4h)]
    impulse_defs = ["A_ATR", "B_BODIES", "C_BOS"]
    entry_styles = ["LIMIT_50", "LIMIT_45", "STAGGERED_45_50_55", "SWEEP_RECLAIM"]
    target_modes = ["PEAK_1R", "EXT_1.5R"]

    results = []
    print("Evaluating all 72 combinations...", flush=True)

    for tf_name, df_data in timeframes:
        for imp in impulse_defs:
            for entry in entry_styles:
                for tgt in target_modes:
                    res = run_simulation_fast(df_data, tf_name, imp, entry, tgt)
                    results.append({
                        "TF": tf_name,
                        "Impulse Def": imp,
                        "Entry Style": entry,
                        "Target": tgt,
                        "N": res["n"],
                        "Win Rate": res["win_rate"],
                        "95% CI": res["ci_95"],
                        "Avg Net R": res["avg_net_r"],
                        "Profit Factor": res["profit_factor"],
                        "Total R": res["total_r"]
                    })

    df_res = pd.DataFrame(results)

    # Filter for setups with N >= 25 and sort by Profit Factor
    valid_res = df_res[df_res["N"] >= 25].copy()
    valid_res["PF_num"] = valid_res["Profit Factor"].astype(float)
    valid_res = valid_res.sort_values("PF_num", ascending=False)

    print("\n" + "=" * 95, flush=True)
    print("TOP PERFORMING CONFIGURATIONS (N >= 25, Ranked by Profit Factor)", flush=True)
    print("=" * 95, flush=True)
    print(valid_res.drop(columns=["PF_num"]).to_string(index=False), flush=True)

    print("\n" + "=" * 95, flush=True)
    print("COMPLETE 72-ROW STRATEGY MATRIX (BTC IN-SAMPLE 75%)", flush=True)
    print("=" * 95, flush=True)
    print(df_res.to_string(index=False), flush=True)

