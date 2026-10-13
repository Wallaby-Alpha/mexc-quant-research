"""
scratch_evaluate_simultaneous_trades.py
---------------------------------------
Evaluates the impact of Maximum Simultaneous Open Positions (K) on portfolio performance,
drawdown, Sharpe ratio, and tail-risk under the Dual Engine Regime-Gated 4h Sniper strategy.

Simulates chronologically bar-by-bar (4-hour intervals) with:
- Zero lookahead
- Net of 15 bps roundtrip fees
- Setup ranking when slots are contested (e.g. rank by Impulse Strength: Delta / ATR)
- Values tested: K in [1, 2, 3, 5, 8, 10, 15, 20, 30, Uncapped]
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

print("1. Preparing BTC Daily HTF indicators (Zero-Lookahead)...", flush=True)
df_btc_1h = cache.load_klines("BTC_USDT", "1h").sort_values("open_time").reset_index(drop=True)
df_btc_1h["open_time"] = pd.to_datetime(df_btc_1h["open_time"], utc=True)

df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h).sort_values("open_time").reset_index(drop=True)
df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
df_btc_1d["ema50"] = df_btc_1d["close"].ewm(span=50, adjust=False).mean()

# Zero lookahead daily shift
df_btc_1d_shifted = df_btc_1d.copy()
df_btc_1d_shifted["effective_time"] = pd.to_datetime(df_btc_1d_shifted["open_time"] + pd.Timedelta(days=1), utc=True).astype("datetime64[ns, UTC]")
df_btc_1d_shifted = df_btc_1d_shifted.sort_values("effective_time")

btc_htf_cols = ["effective_time", "close", "ema50"]
df_btc_sub = df_btc_1d_shifted[btc_htf_cols].rename(columns={
    "close": "btc_1d_close",
    "ema50": "btc_1d_ema50"
})

def compute_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

print("2. Loading candidate symbols and 4h data...", flush=True)
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

def generate_asset_trade_candidates(symbol: str, fee_bps: float = 15.0):
    """
    Generates all potential Style A trades for a single asset with entry_time, exit_time, net_r, and priority score.
    Zero-lookahead.
    """
    df_1h = cache.load_klines(symbol, "1h")
    if df_1h.empty:
        return []
    df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
    df_4h = KlineResampler.resample_1h_to_4h(df_1h).sort_values("open_time").reset_index(drop=True)
    df_4h["open_time"] = pd.to_datetime(df_4h["open_time"], utc=True).astype("datetime64[ns, UTC]")
    
    df_merged = pd.merge_asof(
        df_4h.sort_values("open_time"),
        df_btc_sub,
        left_on="open_time",
        right_on="effective_time",
        direction="backward"
    )
    
    opens = df_merged["open"].values.astype(float)
    highs = df_merged["high"].values.astype(float)
    lows = df_merged["low"].values.astype(float)
    closes = df_merged["close"].values.astype(float)
    times = pd.to_datetime(df_merged["open_time"], utc=True).values
    atrs = compute_atr_np(highs, lows, closes, 14)
    btc_closes = df_merged["btc_1d_close"].values.astype(float)
    btc_ema50 = df_merged["btc_1d_ema50"].values.astype(float)
    
    n_bars = len(closes)
    candidates = []
    i = 30
    in_trade_until = -1
    max_wait_bars = 24
    max_hold_bars = 40
    
    while i < n_bars - 50:
        if i <= in_trade_until:
            i += 1
            continue
            
        cur_atr = atrs[i]
        if np.isnan(cur_atr) or cur_atr <= 0 or np.isnan(btc_ema50[i]) or np.isnan(btc_closes[i]):
            i += 1
            continue
            
        # Regime determination
        is_bull = btc_closes[i] >= btc_ema50[i]
        side = "LONG" if is_bull else "SHORT"
        
        # Detect impulse
        is_impulse = False
        origin_price = 0.0
        extreme_price = 0.0
        
        if side == "SHORT":
            for w in (3, 4, 5, 6):
                if i - w < 0:
                    continue
                sub_h = np.max(highs[i - w:i + 1])
                sub_l = np.min(lows[i - w:i + 1])
                if lows[i] <= sub_l * 1.001 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_price = sub_h
                    extreme_price = sub_l
                    break
        else:
            for w in (3, 4, 5, 6):
                if i - w < 0:
                    continue
                sub_l = np.min(lows[i - w:i + 1])
                sub_h = np.max(highs[i - w:i + 1])
                if highs[i] >= sub_h * 0.999 and (sub_h - sub_l) >= 3.0 * cur_atr:
                    is_impulse = True
                    origin_price = sub_l
                    extreme_price = sub_h
                    break
                    
        if not is_impulse or origin_price == extreme_price:
            i += 1
            continue
            
        delta = abs(origin_price - extreme_price)
        impulse_strength = delta / cur_atr # Priority metric
        
        if side == "SHORT":
            p50 = extreme_price + 0.50 * delta
            p62 = extreme_price + 0.618 * delta
            p70 = extreme_price + 0.705 * delta
            sl_price = extreme_price + 0.786 * delta
            tp_price = extreme_price - 0.272 * delta
            inval_extreme = extreme_price * 0.998
        else:
            p50 = extreme_price - 0.50 * delta
            p62 = extreme_price - 0.618 * delta
            p70 = extreme_price - 0.705 * delta
            sl_price = extreme_price - 0.786 * delta
            tp_price = extreme_price + 0.272 * delta
            inval_extreme = extreme_price * 1.002
            
        filled = False
        entry_price = 0.0
        entry_bar = 0
        
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
            else:
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
            else:
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
        
        candidates.append({
            "symbol": symbol,
            "side": side,
            "entry_time": pd.Timestamp(times[entry_bar]),
            "exit_time": pd.Timestamp(times[exit_bar]),
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "net_r": r_mult - fee_in_r,
            "risk_pct": risk / entry_price,
            "impulse_strength": impulse_strength
        })
        in_trade_until = exit_bar
        i = exit_bar + 1

    return candidates

def run_portfolio_simulation(all_candidates, max_simultaneous: int, holdout_start_time: pd.Timestamp = None, split_mode: str = "FULL"):
    """
    Chronological portfolio simulator enforcing max simultaneous positions constraint.
    """
    if not all_candidates:
        return {}
        
    df_cand = pd.DataFrame(all_candidates).sort_values(by=["entry_time", "impulse_strength"], ascending=[True, False]).reset_index(drop=True)
    
    df_cand["entry_time"] = pd.to_datetime(df_cand["entry_time"], utc=True)
    df_cand["exit_time"] = pd.to_datetime(df_cand["exit_time"], utc=True)
    
    if holdout_start_time is not None:
        holdout_start_utc = pd.to_datetime(holdout_start_time, utc=True)
        if split_mode == "IS":
            df_cand = df_cand[df_cand["entry_time"] < holdout_start_utc].reset_index(drop=True)
        elif split_mode == "OOS":
            df_cand = df_cand[df_cand["entry_time"] >= holdout_start_utc].reset_index(drop=True)
        
    if len(df_cand) == 0:
        return {}
        
    active_positions = [] # list of dicts with 'exit_time', 'symbol', 'net_r'
    executed_trades = []
    concurrent_counts = []
    
    # Process trades chronologically
    for _, row in df_cand.iterrows():
        t_entry = row["entry_time"]
        
        # 1. Clean up positions that have already exited prior to or at t_entry
        active_positions = [pos for pos in active_positions if pos["exit_time"] > t_entry]
        
        # 2. Check if we have capacity
        if max_simultaneous is None or len(active_positions) < max_simultaneous:
            # Enter trade
            active_positions.append({
                "symbol": row["symbol"],
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "net_r": row["net_r"]
            })
            executed_trades.append(row)
            concurrent_counts.append(len(active_positions))
        else:
            # Slot full, skip trade
            pass
            
    if not executed_trades:
        return {}
        
    df_exec = pd.DataFrame(executed_trades).sort_values("exit_time").reset_index(drop=True)
    
    # Compute cumulative equity curve & max drawdown
    df_exec["cum_r"] = df_exec["net_r"].cumsum()
    df_exec["peak_r"] = df_exec["cum_r"].cummax()
    df_exec["drawdown_r"] = df_exec["peak_r"] - df_exec["cum_r"]
    max_dd_r = df_exec["drawdown_r"].max()
    
    n = len(df_exec)
    total_r = df_exec["net_r"].sum()
    wins = df_exec[df_exec["net_r"] > 0]
    wr = len(wins) / n if n > 0 else 0
    ci = 1.96 * np.sqrt((wr * (1.0 - wr)) / n) if n > 0 else 0
    gains = wins["net_r"].sum()
    losses = abs(df_exec[df_exec["net_r"] <= 0]["net_r"].sum())
    pf = (gains / losses) if losses > 0 else 0.0
    avg_r = df_exec["net_r"].mean()
    
    # Return to Drawdown ratio (Calmar-equivalent in R terms)
    r_dd_ratio = (total_r / max_dd_r) if max_dd_r > 0 else 0.0
    
    # Capital Efficiency: Return per allocated position slot
    r_per_slot = (total_r / max_simultaneous) if max_simultaneous else (total_r / max(concurrent_counts))
    
    # Calculate daily Sharpe of R returns
    df_exec["exit_date"] = pd.to_datetime(df_exec["exit_time"]).dt.date
    daily_r = df_exec.groupby("exit_date")["net_r"].sum()
    daily_mean = daily_r.mean()
    daily_std = daily_r.std()
    annualized_sharpe = (daily_mean / daily_std * np.sqrt(365)) if daily_std > 0 else 0.0
    
    avg_concurrent = np.mean(concurrent_counts) if concurrent_counts else 0.0
    max_concurrent = np.max(concurrent_counts) if concurrent_counts else 0.0
    
    # Max simultaneous loss in a single 4h exit bar (tail risk)
    df_exec["exit_4h"] = pd.to_datetime(df_exec["exit_time"]).dt.floor("4h")
    worst_4h_loss = df_exec.groupby("exit_4h")["net_r"].sum().min()
    
    return {
        "max_slots": max_simultaneous if max_simultaneous is not None else "Uncapped",
        "trades": n,
        "win_rate": wr,
        "ci": ci,
        "profit_factor": pf,
        "total_r": total_r,
        "avg_r": avg_r,
        "max_dd_r": max_dd_r,
        "r_dd_ratio": r_dd_ratio,
        "r_per_slot": r_per_slot,
        "sharpe": annualized_sharpe,
        "avg_concurrent": avg_concurrent,
        "max_concurrent": max_concurrent,
        "worst_bar_loss_r": worst_4h_loss
    }

def evaluate_portfolio_matrix(symbols_list, universe_name):
    print(f"\n=========================================================================================", flush=True)
    print(f"COLLECTING TRADE CANDIDATES FOR: {universe_name} ({len(symbols_list)} symbols)", flush=True)
    print(f"=========================================================================================", flush=True)
    
    all_candidates = []
    for s in symbols_list:
        cands = generate_asset_trade_candidates(s)
        all_candidates.extend(cands)
        
    print(f"Generated {len(all_candidates)} raw candidate setups.", flush=True)
    
    # Determine holdout threshold date (75% point from BTC data)
    df_btc = cache.load_klines("BTC_USDT", "1h")
    df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
    split_idx = int(len(df_btc) * 0.75)
    holdout_start = df_btc.iloc[split_idx]["open_time"]
    print(f"Holdout Start Date: {holdout_start.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    
    slot_options = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, None]
    
    for split in ["FULL", "IS", "OOS"]:
        split_names = {
            "FULL": "FULL 1-YEAR PERIOD (100%)",
            "IS": "IN-SAMPLE 75% PERIOD (Bear/Range)",
            "OOS": "LOCKED HOLDOUT 25% PERIOD (Macro Bull Expansion)"
        }
        print(f"\n-----------------------------------------------------------------------------------------", flush=True)
        print(f"PORTFOLIO SIMULATION: {split_names[split]}", flush=True)
        print(f"-----------------------------------------------------------------------------------------", flush=True)
        print(f"{'Max Slots':<10} | {'Trades':<7} | {'Win Rate':<10} | {'PF':<5} | {'Total Net R':<12} | {'Max DD':<8} | {'R / DD':<7} | {'R / Slot':<9} | {'Worst Bar':<10} | {'Sharpe':<6}", flush=True)
        print(f"{'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}", flush=True)
        
        for k in slot_options:
            res = run_portfolio_simulation(all_candidates, k, holdout_start, split)
            if not res:
                continue
            k_str = str(res["max_slots"])
            print(f"{k_str:<10} | {res['trades']:<7} | {res['win_rate']*100:4.1f}%     | {res['profit_factor']:<5.2f} | {res['total_r']:+9.1f}R  | {res['max_dd_r']:6.1f}R | {res['r_dd_ratio']:<7.2f} | {res['r_per_slot']:+7.1f}R  | {res['worst_bar_loss_r']:+7.1f}R  | {res['sharpe']:<6.2f}", flush=True)

if __name__ == "__main__":
    # Test on Top 30 first
    evaluate_portfolio_matrix(top30_coins, "TOP 30 LIQUID CRYPTO ASSETS")
    
    # Test on All 136 Assets
    evaluate_portfolio_matrix(all_crypto_coins, "ALL 136 CRYPTO ASSETS")
