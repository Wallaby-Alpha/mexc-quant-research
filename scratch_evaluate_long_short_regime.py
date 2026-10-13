"""
scratch_evaluate_long_short_regime.py
-------------------------------------
Evaluates Style A (78.6% Fib Stop Loss, 50-70.5% Staggered Entry, 1.272 Ext Target)
across Longs and Shorts, both WITH and WITHOUT the BTC HTF Trend Regime Filter.

Metrics evaluated:
1. Longs Ungated vs Longs Regime-Gated (BTC 1D > 50 EMA)
2. Shorts Ungated vs Shorts Regime-Gated (BTC 1D < 50 EMA)
3. Combined (Longs + Shorts) Ungated vs Regime-Gated

Evaluated on:
- In-Sample (75%)
- Out-of-Sample Holdout (25%)
- Full Sample
Net of 15 bps roundtrip fees.
"""

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

print("Preparing BTC Daily HTF indicators (Zero-Lookahead)...", flush=True)
df_btc_1h = cache.load_klines("BTC_USDT", "1h").sort_values("open_time").reset_index(drop=True)
df_btc_1h["open_time"] = pd.to_datetime(df_btc_1h["open_time"], utc=True)

df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h).sort_values("open_time").reset_index(drop=True)
df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
df_btc_1d["ema50"] = df_btc_1d["close"].ewm(span=50, adjust=False).mean()
df_btc_1d["ema20"] = df_btc_1d["close"].ewm(span=20, adjust=False).mean()
df_btc_1d["ema200"] = df_btc_1d["close"].ewm(span=200, adjust=False).mean()

# Daily bar at day D closes at D + 1 day (00:00 UTC).
# To strictly prevent lookahead, shift by 1 day so that at D+1 00:00, the closed D values become effective.
df_btc_1d_shifted = df_btc_1d.copy()
df_btc_1d_shifted["effective_time"] = pd.to_datetime(df_btc_1d_shifted["open_time"] + pd.Timedelta(days=1), utc=True).astype("datetime64[ns, UTC]")
df_btc_1d_shifted = df_btc_1d_shifted.sort_values("effective_time")

btc_htf_cols = ["effective_time", "close", "ema50", "ema20", "ema200"]
df_btc_sub = df_btc_1d_shifted[btc_htf_cols].rename(columns={
    "close": "btc_1d_close",
    "ema50": "btc_1d_ema50",
    "ema20": "btc_1d_ema20",
    "ema200": "btc_1d_ema200"
})

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

# Load symbols: Top 30 Liquid + BTC
all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
crypto_symbols = [
    s for s in all_symbols
    if not any(bad in s for bad in ["STOCK", "OIL", "XAUT", "GOLD", "ETF", "NVDA", "AAPL", "TSLA", "MSFT"])
]

vol_dict = {}
for s in crypto_symbols:
    df = cache.load_klines(s, "1h")
    if not df.empty and len(df) > 1000:
        vol = df["quote_volume_usdt"].astype(float).mean() if "quote_volume_usdt" in df.columns else df["close"].astype(float).mean()
        vol_dict[s] = vol

s_vol = pd.Series(vol_dict).sort_values(ascending=False)
top30_coins = ["BTC_USDT"] + list(s_vol.head(30).index)
all_crypto_coins = ["BTC_USDT"] + list(s_vol.index)

print(f"Loaded {len(top30_coins)} Top Coins, and {len(all_crypto_coins)} Total Crypto Coins.", flush=True)

def preprocess_symbol_data(symbol: str):
    df_1h = cache.load_klines(symbol, "1h")
    if df_1h.empty:
        return None
    df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
    df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)
    df_4h["open_time"] = pd.to_datetime(df_4h["open_time"], utc=True).astype("datetime64[ns, UTC]")
    
    # Merge BTC daily regime indicators
    df_merged = pd.merge_asof(
        df_4h.sort_values("open_time"),
        df_btc_sub,
        left_on="open_time",
        right_on="effective_time",
        direction="backward"
    )
    return df_merged

def simulate_trades(
    df_4h: pd.DataFrame,
    symbol: str,
    side: str,              # "LONG" or "SHORT"
    regime_mode: str,       # "NONE" or "REGIME_FILTERED"
    sample_split: str,      # "IS" (0-75%), "OOS" (75-100%), "FULL" (0-100%)
    fee_bps: float = 15.0,
    max_wait_bars: int = 24,
    max_hold_bars: int = 40
):
    n_total = len(df_4h)
    split_idx = int(n_total * 0.75)
    
    if sample_split == "IS":
        sub_df = df_4h.iloc[:split_idx].copy().reset_index(drop=True)
    elif sample_split == "OOS":
        sub_df = df_4h.iloc[split_idx:].copy().reset_index(drop=True)
    else:
        sub_df = df_4h.copy().reset_index(drop=True)
        
    if len(sub_df) < 50:
        return []
        
    opens = sub_df["open"].values.astype(float)
    highs = sub_df["high"].values.astype(float)
    lows = sub_df["low"].values.astype(float)
    closes = sub_df["close"].values.astype(float)
    times = pd.to_datetime(sub_df["open_time"], utc=True).values
    atrs = compute_atr_np(highs, lows, closes, 14)
    
    btc_closes = sub_df["btc_1d_close"].values.astype(float)
    btc_ema50 = sub_df["btc_1d_ema50"].values.astype(float)
    
    n_bars = len(closes)
    trades = []
    i = 30
    in_trade_until = -1
    
    while i < n_bars - 50:
        if i <= in_trade_until:
            i += 1
            continue
            
        cur_atr = atrs[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            i += 1
            continue
            
        # 1. Regime Filter Check (if enabled)
        if regime_mode == "REGIME_FILTERED":
            if np.isnan(btc_ema50[i]) or np.isnan(btc_closes[i]):
                i += 1
                continue
            if side == "LONG" and btc_closes[i] < btc_ema50[i]:
                # In bear regime, skip longs
                i += 1
                continue
            elif side == "SHORT" and btc_closes[i] >= btc_ema50[i]:
                # In bull regime, skip shorts
                i += 1
                continue
                
        # 2. Impulse Detection
        is_impulse = False
        origin_price = 0.0
        extreme_price = 0.0
        
        if side == "SHORT":
            # Downward impulse (drop >= 3.0x ATR in 3 to 6 bars)
            for w in (3, 4, 5, 6):
                if i - w < 0:
                    continue
                sub_h = np.max(highs[i - w:i + 1])
                sub_l = np.min(lows[i - w:i + 1])
                if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_price = sub_h   # Origin High
                    extreme_price = sub_l  # Trough Low
                    break
        elif side == "LONG":
            # Upward impulse (rise >= 3.0x ATR in 3 to 6 bars)
            for w in (3, 4, 5, 6):
                if i - w < 0:
                    continue
                sub_l = np.min(lows[i - w:i + 1])
                sub_h = np.max(highs[i - w:i + 1])
                if highs[i] >= sub_h * 0.999 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_price = sub_l   # Origin Low
                    extreme_price = sub_h  # Peak High
                    break
                    
        if not is_impulse or origin_price == extreme_price:
            i += 1
            continue
            
        delta = abs(origin_price - extreme_price)
        
        # 3. Style A Setup Levels (Staggered 50, 61.8, 70.5; SL at 78.6% Fib; TP at 1.272 Ext)
        if side == "SHORT":
            p50 = extreme_price + 0.50 * delta
            p62 = extreme_price + 0.618 * delta
            p70 = extreme_price + 0.705 * delta
            sl_price = extreme_price + 0.786 * delta
            tp_price = extreme_price - 0.272 * delta
            inval_extreme = extreme_price * 0.998
        else: # LONG
            p50 = extreme_price - 0.50 * delta
            p62 = extreme_price - 0.618 * delta
            p70 = extreme_price - 0.705 * delta
            sl_price = extreme_price - 0.786 * delta
            tp_price = extreme_price + 0.272 * delta
            inval_extreme = extreme_price * 1.002
            
        filled = False
        entry_price = 0.0
        entry_bar = 0
        
        # Retracement fill check
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            if side == "SHORT":
                if lows[f] < inval_extreme or highs[f] >= sl_price:
                    break
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
            else: # LONG
                if highs[f] > inval_extreme or lows[f] <= sl_price:
                    break
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
            
        risk = (sl_price - entry_price) if side == "SHORT" else (entry_price - sl_price)
        reward = (entry_price - tp_price) if side == "SHORT" else (tp_price - entry_price)
        if risk <= 0 or reward <= 0:
            i = entry_bar + 1
            continue
            
        exit_bar = entry_bar
        r_mult = 0.0
        
        # Trade execution
        for e in range(entry_bar + 1, min(entry_bar + 1 + max_hold_bars, n_bars)):
            if side == "SHORT":
                if highs[e] >= sl_price:
                    r_mult = -1.0
                    exit_bar = e
                    break
                if lows[e] <= tp_price:
                    r_mult = reward / risk
                    exit_bar = e
                    break
            else: # LONG
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
            pnl = (entry_price - closes[exit_bar]) if side == "SHORT" else (closes[exit_bar] - entry_price)
            r_mult = pnl / risk
            
        fee_in_r = (entry_price * (fee_bps / 10000.0) * 2.0) / risk
        trades.append({
            "symbol": symbol,
            "side": side,
            "entry_time": pd.Timestamp(times[entry_bar]),
            "net_r": r_mult - fee_in_r,
            "risk_pct": risk / entry_price
        })
        in_trade_until = exit_bar
        i = exit_bar + 1

    return trades

def summarize_trades(trades_list):
    if not trades_list:
        return {
            "n": 0, "wr": 0.0, "ci": 0.0, "pf": 0.0, "avg_r": 0.0, "total_r": 0.0, "avg_risk": 0.0
        }
    df = pd.DataFrame(trades_list)
    n = len(df)
    wins = df[df["net_r"] > 0]
    wr = len(wins) / n
    ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n)
    avg_r = df["net_r"].mean()
    total_r = df["net_r"].sum()
    gains = df[df["net_r"] > 0]["net_r"].sum()
    losses = abs(df[df["net_r"] <= 0]["net_r"].sum())
    pf = (gains / losses) if losses > 0 else 0.0
    avg_risk = df["risk_pct"].mean() * 100.0
    return {
        "n": n, "wr": wr, "ci": ci, "pf": pf, "avg_r": avg_r, "total_r": total_r, "avg_risk": avg_risk
    }

def run_full_suite(universe_coins, universe_name):
    print(f"\n=========================================================================================", flush=True)
    print(f"ANALYZING: {universe_name} ({len(universe_coins)} symbols)", flush=True)
    print(f"=========================================================================================", flush=True)
    
    # Pre-cache preprocessed dataframes
    preprocessed_dfs = {}
    for s in universe_coins:
        df = preprocess_symbol_data(s)
        if df is not None and len(df) >= 100:
            preprocessed_dfs[s] = df
            
    print(f"Preprocessed {len(preprocessed_dfs)} valid datasets for {universe_name}.", flush=True)
    
    splits = ["IS", "OOS", "FULL"]
    split_labels = {
        "IS": "In-Sample (75% Early History)",
        "OOS": "Holdout (25% Locked Final Range)",
        "FULL": "Full Historical Period (100%)"
    }
    
    for split in splits:
        print(f"\n--- {split_labels[split]} ---", flush=True)
        print(f"{'Strategy Configuration':<36} | {'Trades':<7} | {'Win Rate (95% CI)':<19} | {'PF':<6} | {'Avg R':<8} | {'Total R':<10}", flush=True)
        print(f"{'-'*36}-+-{'-'*7}-+-{'-'*19}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}", flush=True)
        
        # 1. Longs Ungated
        t_long_ungated = []
        for s, df in preprocessed_dfs.items():
            t_long_ungated.extend(simulate_trades(df, s, "LONG", "NONE", split))
        res = summarize_trades(t_long_ungated)
        print(f"{'Longs (No Regime Filter)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)
        
        # 2. Longs Regime-Gated
        t_long_gated = []
        for s, df in preprocessed_dfs.items():
            t_long_gated.extend(simulate_trades(df, s, "LONG", "REGIME_FILTERED", split))
        res = summarize_trades(t_long_gated)
        print(f"{'Longs (BTC > 50 EMA Gated)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)
        
        # 3. Shorts Ungated
        t_short_ungated = []
        for s, df in preprocessed_dfs.items():
            t_short_ungated.extend(simulate_trades(df, s, "SHORT", "NONE", split))
        res = summarize_trades(t_short_ungated)
        print(f"{'Shorts (No Regime Filter)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)
        
        # 4. Shorts Regime-Gated
        t_short_gated = []
        for s, df in preprocessed_dfs.items():
            t_short_gated.extend(simulate_trades(df, s, "SHORT", "REGIME_FILTERED", split))
        res = summarize_trades(t_short_gated)
        print(f"{'Shorts (BTC < 50 EMA Gated)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)
        
        # 5. Dual Engine Ungated (Longs + Shorts together without filter)
        t_dual_ungated = t_long_ungated + t_short_ungated
        res = summarize_trades(t_dual_ungated)
        print(f"{'Dual (Long+Short, Ungated)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)
        
        # 6. Dual Engine Regime-Gated (Longs in Bull, Shorts in Bear)
        t_dual_gated = t_long_gated + t_short_gated
        res = summarize_trades(t_dual_gated)
        print(f"{'Dual Engine (Regime Gated)':<36} | {res['n']:<7} | {res['wr']*100:4.1f}% (+-{res['ci']*100:3.1f}%)     | {res['pf']:<6.2f} | {res['avg_r']:+6.2f}R | {res['total_r']:+9.1f}R", flush=True)

if __name__ == "__main__":
    # 1. Run BTC Alone
    run_full_suite(["BTC_USDT"], "BTC ONLY")
    
    # 2. Run Top 30 Liquid Universe
    run_full_suite(top30_coins, "TOP 30 LIQUID CRYPTO ASSETS + BTC")
    
    # 3. Run All Crypto Assets (~136 coins)
    run_full_suite(all_crypto_coins, "ALL 136 CRYPTO ASSETS")
