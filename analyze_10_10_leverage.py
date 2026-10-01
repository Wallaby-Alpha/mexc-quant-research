"""
analyze_10_10_leverage.py
Investigates the impact of October 10, 2025 (10/10/2025) on:
1. Leverage liquidations (>20%, >25%, >33%, >50%, >75% intra-week drops).
2. Comparison: Including 10/10/2025 vs. Excluding 10/10/2025.
3. Detailed breakdown of what occurred on 10/10/2025 across all held coins.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("10_10_analysis")


def main():
    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)

    # 1. Load Bitcoin Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load 1H and Daily Altcoin Data
    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    dict_1h = {}
    dict_1d_close = {}

    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
        dict_1h[sym] = df_1h.sort_values("open_time").set_index("open_time")

        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d_close[sym] = df_1d.set_index("open_time")["close"]

    price_matrix = pd.DataFrame(dict_1d_close)

    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]

    # Target date: 2025-10-10
    target_dt_start = pd.Timestamp("2025-10-10 00:00:00+00:00")
    target_dt_end = pd.Timestamp("2025-10-10 23:59:59+00:00")

    # Evaluate Top 7 Quality Momentum and Top 7 Raw Momentum
    for mode in ["sharpe_mom", "raw_mom"]:
        strat_title = "Top 7 Quality (Sharpe) Momentum" if mode == "sharpe_mom" else "Top 7 Raw 30d Momentum"
        logger.info("=" * 80)
        logger.info(f"ANALYZING: {strat_title}")
        logger.info("=" * 80)

        all_trades = []

        for i in range(len(valid_mondays) - 1):
            t_now = valid_mondays[i]
            t_next = valid_mondays[i + 1]

            btc_price = s_btc.loc[t_now]
            btc_ema = s_btc_ema50.loc[t_now]
            is_cash = btc_price < btc_ema

            if is_cash:
                continue

            idx_now = common_idx.get_loc(t_now)
            t_past_30 = common_idx[max(0, idx_now - 30)]
            btc_past = s_btc.loc[t_past_30]
            btc_30d_ret = (btc_price - btc_past) / btc_past

            candidates = []
            for sym in price_matrix.columns:
                s_p = price_matrix[sym]
                p_now = s_p.loc[t_now]
                p_past = s_p.loc[t_past_30]
                if np.isnan(p_now) or np.isnan(p_past) or p_now <= 0 or p_past <= 0:
                    continue
                ret_30d = (p_now - p_past) / p_past
                rs_spread = ret_30d - btc_30d_ret

                slice_30d = s_p.iloc[max(0, idx_now - 30) : idx_now + 1]
                pct_changes = slice_30d.pct_change().dropna()
                vol_30d = float(pct_changes.std() * np.sqrt(365.25)) if len(pct_changes) >= 20 else 1.0
                if np.isnan(vol_30d) or vol_30d < 0.05:
                    vol_30d = 0.05
                sharpe_score = ret_30d / vol_30d

                candidates.append({
                    "symbol": sym,
                    "ret_30d": ret_30d,
                    "rs_spread": rs_spread,
                    "sharpe_score": sharpe_score
                })

            df_cand = pd.DataFrame(candidates)
            if mode == "sharpe_mom":
                selected_coins = df_cand.sort_values("sharpe_score", ascending=False).head(7)["symbol"].tolist()
            else:
                selected_coins = df_cand.sort_values("rs_spread", ascending=False).head(7)["symbol"].tolist()

            is_10_10_week = (t_now <= target_dt_start <= t_next)

            for sym in selected_coins:
                s_p = price_matrix[sym]
                p_entry = s_p.loc[t_now]
                p_exit = s_p.loc[t_next] if t_next in s_p.index else np.nan
                coin_ret = (p_exit - p_entry) / p_entry if (not np.isnan(p_exit) and p_entry > 0) else 0.0

                df_sym_1h = dict_1h.get(sym, pd.DataFrame())
                if not df_sym_1h.empty:
                    # Entire week
                    mask_full = (df_sym_1h.index >= t_now) & (df_sym_1h.index <= t_next)
                    slice_full = df_sym_1h.loc[mask_full]
                    min_price_full = float(slice_full["low"].min()) if not slice_full.empty else p_entry
                    min_time_full = slice_full["low"].idxmin() if not slice_full.empty else t_now
                    drop_full = (min_price_full - p_entry) / p_entry

                    # Exclude 10/10/2025 specifically (exclude all bars from 2025-10-10 00:00 to 23:59 UTC)
                    mask_no_1010 = mask_full & ~((df_sym_1h.index >= target_dt_start) & (df_sym_1h.index <= target_dt_end))
                    slice_no_1010 = df_sym_1h.loc[mask_no_1010]
                    min_price_no_1010 = float(slice_no_1010["low"].min()) if not slice_no_1010.empty else p_entry
                    drop_no_1010 = (min_price_no_1010 - p_entry) / p_entry
                    
                    # Exactly on 10/10/2025
                    mask_1010_only = mask_full & ((df_sym_1h.index >= target_dt_start) & (df_sym_1h.index <= target_dt_end))
                    slice_1010 = df_sym_1h.loc[mask_1010_only]
                    min_price_1010 = float(slice_1010["low"].min()) if not slice_1010.empty else np.nan
                    drop_1010 = (min_price_1010 - p_entry) / p_entry if not np.isnan(min_price_1010) else np.nan
                else:
                    drop_full = coin_ret
                    drop_no_1010 = coin_ret
                    drop_1010 = np.nan
                    min_time_full = t_now

                all_trades.append({
                    "week_start": t_now.strftime("%Y-%m-%d"),
                    "week_end": t_next.strftime("%Y-%m-%d"),
                    "is_10_10_week": is_10_10_week,
                    "symbol": sym,
                    "entry_price": p_entry,
                    "exit_price": p_exit,
                    "coin_ret": coin_ret,
                    "drop_full": drop_full,
                    "min_time_full": min_time_full,
                    "drop_no_1010": drop_no_1010,
                    "drop_1010": drop_1010,
                })

        df_t = pd.DataFrame(all_trades)

        # Print 10/10/2025 specifically
        df_week2 = df_t[df_t["is_10_10_week"]]
        print("\n" + "=" * 75)
        print(f"COINS HELD DURING THE WEEK OF 10/10/2025 (Week of {df_week2['week_start'].iloc[0]}) - {strat_title}")
        print("=" * 75)
        for _, r in df_week2.iterrows():
            print(f"• {r['symbol']:12s} | Entry: {r['entry_price']:.4f} | End-of-Week Ret: {r['coin_ret']*100:+6.2f}% | Max Intra-Week Drop: {r['drop_full']*100:6.2f}% (Trough: {r['min_time_full']}) | Drop on 10/10: {r['drop_1010']*100:6.2f}%")

        # Compare stats: With 10/10 vs Without 10/10 Date vs Without 10/10 Entire Week
        total_pos = len(df_t)

        def calc_thresholds(series):
            return {
                "drop_20": (series <= -0.20).sum(),
                "drop_25": (series <= -0.25).sum(),
                "drop_33": (series <= -0.3333).sum(),
                "drop_50": (series <= -0.50).sum(),
                "drop_75": (series <= -0.75).sum(),
                "worst": series.min() * 100
            }

        stats_with = calc_thresholds(df_t["drop_full"])
        stats_without_date = calc_thresholds(df_t["drop_no_1010"])
        
        # Also compute if entire week of 10/10 is omitted completely
        df_no_week = df_t[~df_t["is_10_10_week"]]
        stats_without_week = calc_thresholds(df_no_week["drop_full"])

        print("\n" + "-" * 75)
        print(f"LEVERAGE LIQUIDATION COMPARISON MATRIX: {strat_title}")
        print("-" * 75)
        print(f"{'Liquidation Level':<30} | {'With 10/10/2025':<16} | {'Excl 10/10 Day':<16} | {'Excl Entire Week':<16}")
        print("-" * 75)
        print(f"{'5x Lev (Drop > 20%)':<30} | {stats_with['drop_20']:2d} ({stats_with['drop_20']/total_pos*100:4.1f}%)        | {stats_without_date['drop_20']:2d} ({stats_without_date['drop_20']/total_pos*100:4.1f}%)        | {stats_without_week['drop_20']:2d} ({stats_without_week['drop_20']/len(df_no_week)*100:4.1f}%)")
        print(f"{'4x Lev (Drop > 25%)':<30} | {stats_with['drop_25']:2d} ({stats_with['drop_25']/total_pos*100:4.1f}%)        | {stats_without_date['drop_25']:2d} ({stats_without_date['drop_25']/total_pos*100:4.1f}%)        | {stats_without_week['drop_25']:2d} ({stats_without_week['drop_25']/len(df_no_week)*100:4.1f}%)")
        print(f"{'3x Lev (Drop > 33.3%)':<30} | {stats_with['drop_33']:2d} ({stats_with['drop_33']/total_pos*100:4.1f}%)        | {stats_without_date['drop_33']:2d} ({stats_without_date['drop_33']/total_pos*100:4.1f}%)        | {stats_without_week['drop_33']:2d} ({stats_without_week['drop_33']/len(df_no_week)*100:4.1f}%)")
        print(f"{'2x Lev (Drop > 50%)':<30} | {stats_with['drop_50']:2d} ({stats_with['drop_50']/total_pos*100:4.1f}%)        | {stats_without_date['drop_50']:2d} ({stats_without_date['drop_50']/total_pos*100:4.1f}%)        | {stats_without_week['drop_50']:2d} ({stats_without_week['drop_50']/len(df_no_week)*100:4.1f}%)")
        print(f"{'1.33x Lev (Drop > 75%)':<30} | {stats_with['drop_75']:2d} ({stats_with['drop_75']/total_pos*100:4.1f}%)        | {stats_without_date['drop_75']:2d} ({stats_without_date['drop_75']/total_pos*100:4.1f}%)        | {stats_without_week['drop_75']:2d} ({stats_without_week['drop_75']/len(df_no_week)*100:4.1f}%)")
        print(f"{'Worst Single Drop':<30} | {stats_with['worst']:6.2f}%          | {stats_without_date['worst']:6.2f}%          | {stats_without_week['worst']:6.2f}%")
        print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
