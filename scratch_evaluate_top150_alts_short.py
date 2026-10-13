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

# Filter out non-crypto synthetics
crypto_symbols = [
    s for s in all_symbols
    if not any(bad in s for bad in ["STOCK", "OIL", "XAUT", "GOLD", "ETF", "NVDA", "AAPL", "TSLA", "MSFT"])
]

# Rank by volume to select Top 150
vol_dict = {}
for s in crypto_symbols:
    df = cache.load_klines(s, "1h")
    if not df.empty and len(df) > 1000:
        vol = df["quote_volume_usdt"].astype(float).mean() if "quote_volume_usdt" in df.columns else df["close"].astype(float).mean()
        vol_dict[s] = vol

s_vol = pd.Series(vol_dict).sort_values(ascending=False)
top_150 = list(s_vol.head(150).index)
print(f"Total crypto altcoins available: {len(crypto_symbols)} | Selected Top {len(top_150)} by volume.", flush=True)

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

# Pre-process 4h numpy data for all 150 coins
preprocessed = {}
for s in top_150:
    df_1h = cache.load_klines(s, "1h")
    if df_1h.empty:
        continue
    df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)
    split_idx = int(len(df_4h) * 0.75)
    df_is = df_4h.iloc[:split_idx]
    if len(df_is) < 100:
        continue
    opens = df_is["open"].values.astype(float)
    highs = df_is["high"].values.astype(float)
    lows = df_is["low"].values.astype(float)
    closes = df_is["close"].values.astype(float)
    atrs = compute_atr_np(highs, lows, closes, 14)
    preprocessed[s] = (opens, highs, lows, closes, atrs)

print(f"Preprocessed 4h numpy data for {len(preprocessed)} valid coins.", flush=True)

def evaluate_short_arrays(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    retrace_style: str,
    target_mode: str,
    fee_bps: float = 15.0
):
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
        for w in (3, 4, 5, 6):
            sub_h = np.max(highs[i-w:i+1])
            sub_l = np.min(lows[i-w:i+1])
            if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= 3.0 * cur_atr:
                is_impulse = True
                origin_high, trough_low = sub_h, sub_l
                break
                
        if not is_impulse or origin_high <= trough_low:
            i += 1
            continue
            
        delta = origin_high - trough_low
        sl_price = origin_high + 0.2 * cur_atr
        tp_price = trough_low if target_mode == "TROUGH_1R" else (trough_low - 0.272 * delta)
        
        filled = False
        entry_price, entry_bar = 0.0, 0
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
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
            if highs[e] >= sl_price:
                r_mult = -1.0
                exit_bar = e
                break
            if lows[e] <= tp_price:
                r_mult = reward / risk
                exit_bar = e
                break
        else:
            exit_bar = min(entry_bar + max_hold_bars, n_bars - 1)
            pnl = entry_price - closes[exit_bar]
            r_mult = pnl / risk
            
        fee_in_r = (entry_price * (fee_bps / 10000.0) * 2.0) / risk
        trades.append(r_mult - fee_in_r)
        in_trade_until = exit_bar
        i = exit_bar + 1

    return trades

if __name__ == "__main__":
    test_combos = [
        ("STAGGERED_50_62_70", "TROUGH_1R"),
        ("STAGGERED_50_62_70", "EXT_1.272"),
        ("61.8%", "EXT_1.272"),
        ("70.5%", "EXT_1.272"),
        ("50.0%", "TROUGH_1R"),
        ("61.8%", "TROUGH_1R"),
        ("70.5%", "TROUGH_1R"),
    ]
    
    summary_records = []
    
    print("\nRunning Top 150 Altcoins Short-Side Matrix...", flush=True)
    
    for ret, tgt in test_combos:
        all_trades = []
        coin_pfs = []
        profitable_coins = 0
        total_coins_with_trades = 0
        
        for s, arrays in preprocessed.items():
            t_list = evaluate_short_arrays(*arrays, ret, tgt)
            if t_list:
                all_trades.extend(t_list)
                arr_c = np.array(t_list)
                g_c = np.sum(arr_c[arr_c > 0])
                l_c = abs(np.sum(arr_c[arr_c <= 0]))
                pf_c = (g_c / l_c) if l_c > 0 else (99.0 if g_c > 0 else 0.0)
                if len(t_list) >= 5:
                    total_coins_with_trades += 1
                    if pf_c > 1.0:
                        profitable_coins += 1
                    coin_pfs.append(pf_c)
                    
        arr = np.array(all_trades)
        n = len(arr)
        if n == 0:
            continue
            
        wr = np.sum(arr > 0) / n
        ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n)
        avg_r = float(np.mean(arr))
        gains = np.sum(arr[arr > 0])
        losses = abs(np.sum(arr[arr <= 0]))
        pf = (gains / losses) if losses > 0 else 0.0
        
        pct_profitable_coins = (profitable_coins / total_coins_with_trades * 100) if total_coins_with_trades > 0 else 0
        
        summary_records.append({
            "Retrace Style": ret,
            "Target": tgt,
            "Total Trades (N)": n,
            "Avg Trades/Coin": f"{n/len(preprocessed):.1f}",
            "Win Rate": f"{wr*100:.1f}%",
            "95% CI": f"±{ci*100:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Avg Net R": f"{avg_r:+.2f}R",
            "Total Portfolio R": f"{np.sum(arr):+.1f}R",
            "% Profitable Coins": f"{pct_profitable_coins:.1f}% ({profitable_coins}/{total_coins_with_trades})"
        })
        
    df_res = pd.DataFrame(summary_records)
    print("\n" + "=" * 115, flush=True)
    print(f"PORTFOLIO PERFORMANCE ACROSS ALL {len(preprocessed)} COINS (SHORT 4H IMPULSE RETRACEMENT)", flush=True)
    print("=" * 115, flush=True)
    print(df_res.to_string(index=False), flush=True)
    
    # Detailed per-coin distribution stats for STAGGERED + EXT_1.272
    print("\n" + "=" * 115, flush=True)
    print("DISTRIBUTION OF COIN PERFORMANCE (SHORT STAGGERED 50/62/70 + EXT_1.272)", flush=True)
    print("=" * 115, flush=True)
    
    coin_data = []
    for s, arrays in preprocessed.items():
        t_list = evaluate_short_arrays(*arrays, "STAGGERED_50_62_70", "EXT_1.272")
        arr_c = np.array(t_list)
        if len(arr_c) >= 5:
            g_c = np.sum(arr_c[arr_c > 0])
            l_c = abs(np.sum(arr_c[arr_c <= 0]))
            pf_c = (g_c / l_c) if l_c > 0 else 0.0
            coin_data.append({
                "Coin": s,
                "Trades": len(arr_c),
                "Win Rate": f"{np.sum(arr_c > 0)/len(arr_c)*100:.1f}%",
                "Profit Factor": pf_c,
                "Avg Net R": float(np.mean(arr_c)),
                "Total Net R": float(np.sum(arr_c))
            })
            
    df_c = pd.DataFrame(coin_data).sort_values("Profit Factor", ascending=False)
    pfs = df_c["Profit Factor"]
    print(f"Median Coin Profit Factor: {pfs.median():.2f}")
    print(f"Mean Coin Profit Factor: {pfs.mean():.2f}")
    print(f"Coins with PF >= 1.0: {(pfs >= 1.0).sum()}/{len(pfs)} ({(pfs >= 1.0).mean()*100:.1f}%)")
    print(f"Coins with PF >= 1.3: {(pfs >= 1.3).sum()}/{len(pfs)} ({(pfs >= 1.3).mean()*100:.1f}%)")
    
    print("\n--- Top 15 Best Performing Coins ---")
    print(df_c.head(15).to_string(index=False))
    print("\n--- Bottom 10 Worst Performing Coins ---")
    print(df_c.tail(10).to_string(index=False))
