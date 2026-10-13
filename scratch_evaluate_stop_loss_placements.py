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
crypto_symbols = [
    s for s in all_symbols
    if not any(bad in s for bad in ["STOCK", "OIL", "XAUT", "GOLD", "ETF", "NVDA", "AAPL", "TSLA", "MSFT"])
]

# Pick Top 30 liquid alts + BTC for stop loss analysis
vol_dict = {}
for s in crypto_symbols:
    df = cache.load_klines(s, "1h")
    if not df.empty and len(df) > 1000:
        vol = df["quote_volume_usdt"].astype(float).mean() if "quote_volume_usdt" in df.columns else df["close"].astype(float).mean()
        vol_dict[s] = vol

s_vol = pd.Series(vol_dict).sort_values(ascending=False)
test_coins = ["BTC_USDT"] + list(s_vol.head(30).index)
print(f"Testing Stop Loss Placements across {len(test_coins)} assets (BTC + Top 30 Alts)...", flush=True)

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

preprocessed = {}
for s in test_coins:
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

print(f"Preprocessed 4h data for {len(preprocessed)} assets.", flush=True)

def evaluate_sl_option(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    sl_type: str,          # "ORIGIN_EXACT", "ORIGIN_0.2ATR", "ORIGIN_0.5ATR", "ORIGIN_1.0ATR", "FIB_786", "FIB_886", "ENTRY_1.5ATR"
    retrace_style: str = "STAGGERED_50_62_70",
    target_mode: str = "EXT_1.272",
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
            
        # Detect Short Impulse (drop >= 3.0x ATR in 3 to 6 bars)
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
        tp_price = trough_low - 0.272 * delta if target_mode == "EXT_1.272" else trough_low
        
        # Check Retracement Fills
        p50 = trough_low + 0.50 * delta
        p62 = trough_low + 0.618 * delta
        p70 = trough_low + 0.705 * delta
        
        filled = False
        entry_price = 0.0
        entry_bar = 0
        
        # Tentative maximum invalidation price during wait phase
        max_inval = origin_high + 1.0 * cur_atr
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            if lows[f] < trough_low * 0.998 or highs[f] >= max_inval:
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
                
        if not filled:
            i += 1
            continue
            
        # Determine Stop Loss Price based on sl_type
        if sl_type == "ORIGIN_EXACT":
            sl_price = origin_high
        elif sl_type == "ORIGIN_0.2ATR":
            sl_price = origin_high + 0.2 * cur_atr
        elif sl_type == "ORIGIN_0.5ATR":
            sl_price = origin_high + 0.5 * cur_atr
        elif sl_type == "ORIGIN_1.0ATR":
            sl_price = origin_high + 1.0 * cur_atr
        elif sl_type == "FIB_786":
            sl_price = trough_low + 0.786 * delta
        elif sl_type == "FIB_886":
            sl_price = trough_low + 0.886 * delta
        elif sl_type == "ENTRY_1.5ATR":
            sl_price = entry_price + 1.5 * cur_atr
        else:
            sl_price = origin_high + 0.2 * cur_atr
            
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
        trades.append({
            "net_r": r_mult - fee_in_r,
            "risk_pct": risk / entry_price,
            "reward_risk_ratio": reward / risk
        })
        in_trade_until = exit_bar
        i = exit_bar + 1

    return trades

if __name__ == "__main__":
    sl_candidates = [
        ("FIB_786", "Tight Fibonacci (78.6% Retracement Level)"),
        ("FIB_886", "Deep Fibonacci (88.6% Retracement Level)"),
        ("ORIGIN_EXACT", "Exact Structural Origin (0.0x ATR buffer)"),
        ("ORIGIN_0.2ATR", "Origin + 0.2x ATR buffer (Baseline)"),
        ("ORIGIN_0.5ATR", "Origin + 0.5x ATR buffer (Wide Buffer)"),
        ("ORIGIN_1.0ATR", "Origin + 1.0x ATR buffer (Loose Anti-Wick)"),
        ("ENTRY_1.5ATR", "Entry + 1.5x ATR Trailing Distance"),
    ]
    
    results = []
    
    for sl_key, sl_desc in sl_candidates:
        all_trades = []
        for s, arrays in preprocessed.items():
            t_list = evaluate_sl_option(*arrays, sl_key)
            all_trades.extend(t_list)
            
        if not all_trades:
            continue
            
        df_t = pd.DataFrame(all_trades)
        n = len(df_t)
        wins = df_t[df_t["net_r"] > 0]
        wr = len(wins) / n
        ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n)
        avg_r = df_t["net_r"].mean()
        
        gains = df_t[df_t["net_r"] > 0]["net_r"].sum()
        losses = abs(df_t[df_t["net_r"] <= 0]["net_r"].sum())
        pf = (gains / losses) if losses > 0 else 0.0
        
        avg_risk_pct = df_t["risk_pct"].mean() * 100
        avg_rr = df_t["reward_risk_ratio"].mean()
        
        results.append({
            "SL Placement Concept": sl_desc,
            "Code": sl_key,
            "Trades (N)": n,
            "Win Rate": f"{wr*100:.1f}%",
            "95% CI": f"±{ci*100:.1f}%",
            "Avg Stop %": f"{avg_risk_pct:.2f}%",
            "Design R:R": f"{avg_rr:.2f}:1",
            "Profit Factor": f"{pf:.2f}",
            "Avg Net R": f"{avg_r:+.2f}R",
            "Total Portfolio R": f"{df_t['net_r'].sum():+.1f}R"
        })
        
    df_res = pd.DataFrame(results).sort_values("Profit Factor", ascending=False)
    print("\n" + "=" * 120, flush=True)
    print("COMPARATIVE EVALUATION: LOGICAL STOP LOSS PLACEMENTS (SHORT 4H IMPULSE RETRACEMENT)", flush=True)
    print("=" * 120, flush=True)
    print(df_res.to_string(index=False), flush=True)
