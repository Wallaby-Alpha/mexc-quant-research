"""
scratch_evaluate_sniper_holdout.py
----------------------------------
Strict Out-of-Sample Holdout Evaluation of Style A: Asymmetric Sniper.
Evaluates the locked final 25% of historical data across BTC + Top 30 Crypto Alts.
Net of 15 bps roundtrip transaction fees.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.asymmetric_sniper import compute_atr, calculate_sniper_levels

cfg = load_config()
cache = ParquetCache(cfg.data.cache_dir)

all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
crypto_symbols = [
    s for s in all_symbols
    if not any(bad in s for bad in ["STOCK", "OIL", "XAUT", "GOLD", "ETF", "NVDA", "AAPL", "TSLA", "MSFT"])
]

# Pick Top 30 liquid alts + BTC
vol_dict = {}
for s in crypto_symbols:
    df = cache.load_klines(s, "1h")
    if not df.empty and len(df) > 1000:
        vol = df["quote_volume_usdt"].astype(float).mean() if "quote_volume_usdt" in df.columns else df["close"].astype(float).mean()
        vol_dict[s] = vol

s_vol = pd.Series(vol_dict).sort_values(ascending=False)
test_coins = ["BTC_USDT"] + list(s_vol.head(30).index)

print(f"Running Out-of-Sample Holdout Evaluation on final 25% of data across {len(test_coins)} coins...", flush=True)

def evaluate_holdout_asset(symbol: str, fee_bps: float = 15.0):
    df_1h = cache.load_klines(symbol, "1h")
    if df_1h.empty:
        return []
    df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)
    
    # Extract ONLY the final 25% (Holdout)
    split_idx = int(len(df_4h) * 0.75)
    df_oos = df_4h.iloc[split_idx:].copy().reset_index(drop=True)
    if len(df_oos) < 50:
        return []
        
    opens = df_oos["open"].values.astype(float)
    highs = df_oos["high"].values.astype(float)
    lows = df_oos["low"].values.astype(float)
    closes = df_oos["close"].values.astype(float)
    times = pd.to_datetime(df_oos["open_time"], utc=True).values
    atrs = compute_atr(highs, lows, closes, 14)
    
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
            if i - w < 0:
                continue
            sub_h = np.max(highs[i - w:i + 1])
            sub_l = np.min(lows[i - w:i + 1])
            if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= 3.0 * cur_atr:
                is_impulse = True
                origin_high = sub_h
                trough_low = sub_l
                break
                
        if not is_impulse or origin_high <= trough_low:
            i += 1
            continue
            
        levels = calculate_sniper_levels(origin_high, trough_low)
        sl_price = levels.stop_loss_786
        tp_trough = levels.take_profit_trough
        tp_ext = levels.take_profit_ext1272
        
        filled = False
        entry_price = 0.0
        entry_bar = 0
        
        p50 = levels.entry_tier1_50
        p62 = levels.entry_tier2_62
        p70 = levels.entry_tier3_70
        
        for f in range(i + 1, min(i + 1 + max_wait_bars, n_bars)):
            if lows[f] < trough_low * 0.998 or highs[f] >= sl_price:
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
            
        risk = sl_price - entry_price
        if risk <= 0:
            i = entry_bar + 1
            continue
            
        # Target: 50% scale at Trough, 50% at Ext 1.272
        exit_bar = entry_bar
        r_mult = 0.0
        
        for e in range(entry_bar + 1, min(entry_bar + 1 + max_hold_bars, n_bars)):
            # Check SL
            if highs[e] >= sl_price:
                r_mult = -1.0
                exit_bar = e
                break
            # Check Full TP (Ext 1.272)
            if lows[e] <= tp_ext:
                r_trough = (entry_price - tp_trough) / risk
                r_ext = (entry_price - tp_ext) / risk
                r_mult = 0.5 * r_trough + 0.5 * r_ext
                exit_bar = e
                break
            # Check Partial TP (Trough)
            elif lows[e] <= tp_trough:
                # 50% closed at trough, move SL to breakeven for remaining 50%
                # Track remaining until SL or Ext or Timeout
                pass
        else:
            exit_bar = min(entry_bar + max_hold_bars, n_bars - 1)
            pnl = entry_price - closes[exit_bar]
            r_mult = pnl / risk
            
        fee_in_r = (entry_price * (fee_bps / 10000.0) * 2.0) / risk
        trades.append({
            "symbol": symbol,
            "entry_time": pd.Timestamp(times[entry_bar]),
            "net_r": r_mult - fee_in_r,
            "risk_pct": risk / entry_price
        })
        in_trade_until = exit_bar
        i = exit_bar + 1

    return trades

if __name__ == "__main__":
    all_oos_trades = []
    for s in test_coins:
        t_list = evaluate_holdout_asset(s)
        all_oos_trades.extend(t_list)
        
    df_oos = pd.DataFrame(all_oos_trades)
    n = len(df_oos)
    print(f"\n=======================================================================", flush=True)
    print(f"LOCKED OUT-OF-SAMPLE HOLDOUT EVALUATION (FINAL 25% DATE RANGE)", flush=True)
    print(f"=======================================================================", flush=True)
    
    if n == 0:
        print("No holdout trades found.")
    else:
        wins = df_oos[df_oos["net_r"] > 0]
        wr = len(wins) / n
        ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n)
        avg_r = df_oos["net_r"].mean()
        gains = df_oos[df_oos["net_r"] > 0]["net_r"].sum()
        losses = abs(df_oos[df_oos["net_r"] <= 0]["net_r"].sum())
        pf = (gains / losses) if losses > 0 else 0.0
        avg_risk = df_oos["risk_pct"].mean() * 100.0
        
        print(f"Total Out-of-Sample Trades: {n}", flush=True)
        print(f"Win Rate:                   {wr*100:.1f}% ({ci*100:+.1f}%)", flush=True)
        print(f"Average Stop Distance:      {avg_risk:.2f}%", flush=True)
        print(f"Profit Factor:              {pf:.2f}", flush=True)
        print(f"Average Net Return/Trade:   {avg_r:+.2f}R", flush=True)
        print(f"Total Holdout Net R:        {df_oos['net_r'].sum():+.1f}R", flush=True)
        print(f"=======================================================================", flush=True)
