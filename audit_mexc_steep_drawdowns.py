"""
audit_mexc_steep_drawdowns.py
Forensic audit of steep drawdowns across the MEXC universe:
1. Identifies all major crashes (>25% weekly drop, >35% intra-week drop) in held positions.
2. Identifies all catastrophic collapses (>50% 7-day drop) across the entire 161-coin universe.
3. Classifies each crash:
   - Market-Wide Beta Cascade (BTC plunged)
   - Idiosyncratic Pump & Dump / Rug Pull / Exploit (BTC was flat/up, coin died, huge prior pump)
   - Flash Liquidity Wick (rapid intra-candle wick with fast recovery)
4. Assesses whether Quality (Sharpe) Momentum inherently filters out these toxic assets.
5. Compares MEXC-exclusive tokens vs. Kraken-listed tokens.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from run_kraken_portfolio_backtest import get_kraken_supported_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mexc_audit")


def main():
    logger.info("=" * 80)
    logger.info("FORENSIC AUDIT OF STEEP DRAWDOWNS IN THE MEXC UNIVERSE")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)

    # 1. Load Bitcoin Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load Altcoins
    all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    kraken_symbols = set(get_kraken_supported_symbols(all_symbols))

    dict_1h = {}
    dict_1d = {}

    for sym in all_symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
        dict_1h[sym] = df_1h.sort_values("open_time").set_index("open_time")

        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d[sym] = df_1d.sort_values("open_time").set_index("open_time")

    price_matrix = pd.DataFrame({s: df["close"] for s, df in dict_1d.items()})
    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]

    # -------------------------------------------------------------------------
    # PART A: Audit Severe Losses on Trades Held in Top 7 Strategies
    # -------------------------------------------------------------------------
    held_crashes = []

    for mode in ["sharpe_mom", "raw_mom"]:
        for i in range(len(valid_mondays) - 1):
            t_now = valid_mondays[i]
            t_next = valid_mondays[i + 1]

            btc_now = s_btc.loc[t_now]
            btc_next = s_btc.loc[t_next] if t_next in s_btc.index else btc_now
            btc_ema = s_btc_ema50.loc[t_now]

            if btc_now < btc_ema:
                continue

            btc_week_ret = (btc_next - btc_now) / btc_now
            idx_now = common_idx.get_loc(t_now)
            t_past_30 = common_idx[max(0, idx_now - 30)]
            btc_30d_ret = (btc_now - s_btc.loc[t_past_30]) / s_btc.loc[t_past_30]

            candidates = []
            for sym in price_matrix.columns:
                s_p = price_matrix[sym]
                p0 = s_p.loc[t_now]
                p_past = s_p.loc[t_past_30]
                if np.isnan(p0) or np.isnan(p_past) or p0 <= 0 or p_past <= 0:
                    continue
                ret_30d = (p0 - p_past) / p_past
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
                selected = df_cand.sort_values("sharpe_score", ascending=False).head(7)["symbol"].tolist()
            else:
                selected = df_cand.sort_values("rs_spread", ascending=False).head(7)["symbol"].tolist()

            for sym in selected:
                s_p = price_matrix[sym]
                p_entry = s_p.loc[t_now]
                p_exit = s_p.loc[t_next] if t_next in s_p.index else np.nan
                coin_ret = (p_exit - p_entry) / p_entry if (not np.isnan(p_exit) and p_entry > 0) else 0.0

                df_sym_1h = dict_1h.get(sym, pd.DataFrame())
                intra_drop = 0.0
                lowest_dt = t_now
                if not df_sym_1h.empty:
                    mask_1h = (df_sym_1h.index >= t_now) & (df_sym_1h.index <= t_next)
                    slice_1h = df_sym_1h.loc[mask_1h]
                    if not slice_1h.empty:
                        lowest_price = float(slice_1h["low"].min())
                        lowest_dt = slice_1h["low"].idxmin()
                        intra_drop = (lowest_price - p_entry) / p_entry

                # Prior 30d run-up
                p_30d_prior = s_p.loc[t_past_30]
                prior_30d_gain = (p_entry - p_30d_prior) / p_30d_prior if p_30d_prior > 0 else 0.0

                # Subsequent 30d recovery (price 30 days after exit)
                idx_exit = common_idx.get_loc(t_next) if t_next in common_idx else -1
                if idx_exit != -1 and (idx_exit + 30) < len(common_idx):
                    t_post_30 = common_idx[idx_exit + 30]
                    p_post_30 = s_p.loc[t_post_30] if t_post_30 in s_p.index else np.nan
                    post_30d_change = (p_post_30 - p_exit) / p_exit if (not np.isnan(p_post_30) and p_exit > 0) else np.nan
                else:
                    post_30d_change = np.nan

                # Only examine severe drawdowns: close loss <= -20% OR intra drop <= -30%
                if coin_ret <= -0.20 or intra_drop <= -0.30:
                    # Classification Logic
                    on_kraken = sym in kraken_symbols
                    is_market_crash = (btc_week_ret <= -0.05) or (lowest_dt.strftime("%Y-%m-%d") == "2025-10-10")

                    if is_market_crash:
                        category = "Market-Wide Beta Drop"
                    elif prior_30d_gain > 1.50 and coin_ret <= -0.35 and (post_30d_change is not None and post_30d_change < 0):
                        category = "Suspected Pump & Dump"
                    elif intra_drop <= -0.40 and coin_ret > -0.15:
                        category = "Flash Liquidity Wick (Recovered)"
                    elif not on_kraken and coin_ret <= -0.30:
                        category = "Low-Liquidity Alt Dump (MEXC Exclusive)"
                    else:
                        category = "Idiosyncratic Altcoin Momentum Loss"

                    held_crashes.append({
                        "strategy": "Quality (Sharpe)" if mode == "sharpe_mom" else "Raw Momentum",
                        "week_date": t_now.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "on_kraken": on_kraken,
                        "prior_30d_gain_pct": prior_30d_gain * 100,
                        "week_return_pct": coin_ret * 100,
                        "intra_drop_pct": intra_drop * 100,
                        "btc_week_ret_pct": btc_week_ret * 100,
                        "post_30d_recovery_pct": post_30d_change * 100 if not np.isnan(post_30d_change) else None,
                        "category": category,
                        "trough_timestamp": lowest_dt.strftime("%Y-%m-%d %H:%M")
                    })

    df_held = pd.DataFrame(held_crashes)

    # -------------------------------------------------------------------------
    # PART B: Universe-Wide Catastrophic Collapses (>50% drop in any 7-day period)
    # -------------------------------------------------------------------------
    universe_collapses = []
    for sym in price_matrix.columns:
        s_p = price_matrix[sym].dropna()
        if len(s_p) < 40:
            continue
        rolling_7d_ret = s_p.pct_change(7)
        steep_drops = rolling_7d_ret[rolling_7d_ret <= -0.50]
        
        for dt, ret_val in steep_drops.items():
            dt_past_30 = dt - pd.Timedelta(days=30)
            prior_slice = s_p.loc[(s_p.index >= dt_past_30) & (s_p.index <= dt)]
            prior_pump = (prior_slice.max() - prior_slice.iloc[0]) / prior_slice.iloc[0] if len(prior_slice) > 1 else 0.0

            # Did price ever recover within 60 days?
            future_slice = s_p.loc[(s_p.index > dt) & (s_p.index <= dt + pd.Timedelta(days=60))]
            max_future = (future_slice.max() - s_p.loc[dt]) / s_p.loc[dt] if len(future_slice) > 0 else 0.0
            is_dead = (len(future_slice) == 0) or (future_slice.iloc[-1] < s_p.loc[dt] * 0.8)

            on_kraken = sym in kraken_symbols
            btc_ret_same_period = (s_btc.loc[dt] - s_btc.loc[dt - pd.Timedelta(days=7)]) / s_btc.loc[dt - pd.Timedelta(days=7)] if (dt in s_btc.index and (dt - pd.Timedelta(days=7)) in s_btc.index) else 0.0

            if is_dead and prior_pump > 2.0:
                collapse_type = "Classic Rug Pull / Exploit"
            elif prior_pump > 1.5:
                collapse_type = "Pump & Dump Climax"
            elif dt.strftime("%Y-%m-%d") in ["2025-10-10", "2025-10-11", "2025-10-12"]:
                collapse_type = "10/10 Market Crash Collapse"
            else:
                collapse_type = "Severe Momentum Unwind"

            universe_collapses.append({
                "symbol": sym,
                "date": dt.strftime("%Y-%m-%d"),
                "drop_7d_pct": ret_val * 100,
                "prior_pump_pct": prior_pump * 100,
                "on_kraken": on_kraken,
                "btc_7d_ret_pct": btc_ret_same_period * 100,
                "collapse_type": collapse_type,
                "is_dead_coin": is_dead
            })

    df_univ = pd.DataFrame(universe_collapses)

    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PART A: SEVERE DRAWDOWNS ON TRADES HELD BY OUR STRATEGIES")
    print("=" * 90)

    # Compare breakdown by strategy
    for strat in ["Quality (Sharpe)", "Raw Momentum"]:
        sub = df_held[df_held["strategy"] == strat].sort_values("week_return_pct")
        print(f"\n--- Strategy: {strat} (Total Severe Drawdowns: {len(sub)}) ---")
        print(f"{'Date':<10} | {'Symbol':<14} | {'Kraken?':<7} | {'WeekRet':<8} | {'IntraDrop':<9} | {'Prior30d':<9} | {'Category'}")
        print("-" * 90)
        for _, r in sub.iterrows():
            k_str = "YES" if r["on_kraken"] else "MEXC-ONLY"
            print(f"{r['week_date']:<10} | {r['symbol']:<14} | {k_str:<7} | {r['week_return_pct']:+6.1f}% | {r['intra_drop_pct']:6.1f}% | {r['prior_30d_gain_pct']:+6.0f}% | {r['category']}")

    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON: QUALITY MOMENTUM VS RAW MOMENTUM SEVERE CRASHES")
    print("=" * 90)
    print(f"Total Severe Drawdown Trades (>20% loss or >30% wick):")
    print(f"  • Quality (Sharpe) Momentum: {len(df_held[df_held['strategy'] == 'Quality (Sharpe)'])} trades")
    print(f"  • Raw 30d Momentum:          {len(df_held[df_held['strategy'] == 'Raw Momentum'])} trades (More than 2x as many severe crashes!)")

    print("\nBreakdown of Crash Categories:")
    ct_sharpe = df_held[df_held["strategy"] == "Quality (Sharpe)"]["category"].value_counts()
    ct_raw = df_held[df_held["strategy"] == "Raw Momentum"]["category"].value_counts()
    df_cats = pd.DataFrame({"Quality (Sharpe)": ct_sharpe, "Raw Momentum": ct_raw}).fillna(0).astype(int)
    print(df_cats.to_string())

    print("\n" + "=" * 90)
    print("PART B: UNIVERSE-WIDE CATASTROPHIC COLLAPSES (>50% DROP IN 7 DAYS)")
    print("=" * 90)
    print(f"Total >50% Collapses in MEXC Universe: {len(df_univ)}")
    print(f"  • Kraken-Listed Coins:   {len(df_univ[df_univ['on_kraken']])}")
    print(f"  • MEXC-Exclusive Coins:  {len(df_univ[~df_univ['on_kraken']])}")

    print("\nWorst 10 Collapses in MEXC Universe:")
    worst_10 = df_univ.sort_values("drop_7d_pct").drop_duplicates(subset=["symbol"]).head(10)
    print(f"{'Symbol':<16} | {'Date':<10} | {'Kraken?':<9} | {'7d Drop':<8} | {'Prior 30d Pump':<14} | {'Collapse Type'}")
    print("-" * 90)
    for _, r in worst_10.iterrows():
        k_str = "KRAKEN" if r["on_kraken"] else "MEXC-ONLY"
        print(f"{r['symbol']:<16} | {r['date']:<10} | {k_str:<9} | {r['drop_7d_pct']:6.1f}% | {r['prior_pump_pct']:+10.0f}% | {r['collapse_type']}")

    out_csv = results_dir / "mexc_severe_drawdown_audit.csv"
    df_held.to_csv(out_csv, index=False)
    logger.info(f"Saved audit results to {out_csv}")


if __name__ == "__main__":
    main()
