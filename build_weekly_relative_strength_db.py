"""
build_weekly_relative_strength_db.py
Constructs a comprehensive, point-in-time relational SQLite database, Parquet, and CSV
recording the Top Relative Strength and Quality Momentum coins for every calendar week.

Features:
- Monday 00:00 UTC rebalance cadence (aligning with live scanner cron).
- Zero lookahead: signal metrics strictly computed on past 30 days.
- Macro regime tracking (BTC price, EMA50, bullish/bearish cash shield flag).
- Both Raw 30d Momentum and Quality (Sharpe) Momentum rankings.
- Forward 7d, 14d, and 30d performance tracking for causal strategy backtesting.
- SQL views for Top 10 Raw, Top 10 Sharpe, and Top 7 Sharpe portfolios.
"""

from pathlib import Path
import sqlite3
import logging
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_weekly_db")


def main():
    logger.info("=" * 80)
    logger.info("BUILDING WEEKLY RELATIVE STRENGTH DATABASE")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    holdout_start = cfg.data.holdout_start
    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")

    # 1. Load Bitcoin 1D Resampled Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)

    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load Universe Daily Close & Volume Data
    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    daily_close_dict = {}
    daily_volume_dict = {}

    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        df_1d = df_1d.sort_values("open_time").reset_index(drop=True)
        daily_close_dict[sym] = df_1d.set_index("open_time")["close"]
        if "quote_volume_usdt" in df_1d.columns:
            daily_volume_dict[sym] = df_1d.set_index("open_time")["quote_volume_usdt"]

    price_matrix = pd.DataFrame(daily_close_dict)
    logger.info(f"Loaded price matrix with {price_matrix.shape[1]} altcoins.")

    # 3. Identify Monday Rebalance Timestamps
    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]

    # Require at least 50 days of lookback for EMA50 and 30 days for momentum
    min_lookback_days = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback_days]
    logger.info(f"Identified {len(valid_mondays)} rebalance Mondays (from {valid_mondays[0].strftime('%Y-%m-%d')} to {valid_mondays[-1].strftime('%Y-%m-%d')}).")

    rebalance_records = []
    ranking_records = []

    for week_num, t_now in enumerate(valid_mondays, start=1):
        idx_now = common_idx.get_loc(t_now)
        btc_now = float(s_btc.loc[t_now])
        btc_ema = float(s_btc_ema50.loc[t_now])
        btc_bullish = int(btc_now >= btc_ema)

        # BTC 30d return
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_past_30 = float(s_btc.loc[t_past_30])
        btc_ret_30d = (btc_now - btc_past_30) / btc_past_30 if btc_past_30 > 0 else 0.0

        # BTC Forward 7d return
        has_fwd_7d = (idx_now + 7) < len(common_idx)
        if has_fwd_7d:
            t_fwd_7d = common_idx[idx_now + 7]
            btc_fwd_7d = float((s_btc.loc[t_fwd_7d] - btc_now) / btc_now)
        else:
            t_fwd_7d = None
            btc_fwd_7d = np.nan

        is_holdout = int(t_now >= holdout_ts)

        # Evaluate all altcoins at t_now (strictly causal)
        coin_metrics = []
        for sym in price_matrix.columns:
            s_p = price_matrix[sym]
            p_now = s_p.loc[t_now]
            p_past_30 = s_p.loc[t_past_30]

            if np.isnan(p_now) or np.isnan(p_past_30) or p_now <= 0 or p_past_30 <= 0:
                continue

            alt_ret_30d = (p_now - p_past_30) / p_past_30
            rs_spread = alt_ret_30d - btc_ret_30d

            # Trailing 30-day volatility
            slice_30d = s_p.iloc[max(0, idx_now - 30) : idx_now + 1]
            pct_changes = slice_30d.pct_change().dropna()
            if len(pct_changes) >= 20:
                vol_30d = float(pct_changes.std() * np.sqrt(365.25))
            else:
                vol_30d = 1.0
            if np.isnan(vol_30d) or vol_30d < 0.05:
                vol_30d = 0.05

            sharpe_score = alt_ret_30d / vol_30d

            # Forward returns
            p_fwd_7d = s_p.loc[t_fwd_7d] if (has_fwd_7d and t_fwd_7d in s_p.index) else np.nan
            if not np.isnan(p_fwd_7d) and p_fwd_7d > 0:
                fwd_ret_7d = (p_fwd_7d - p_now) / p_now
                fwd_alpha_7d = fwd_ret_7d - btc_fwd_7d if not np.isnan(btc_fwd_7d) else np.nan
            else:
                p_fwd_7d = np.nan
                fwd_ret_7d = np.nan
                fwd_alpha_7d = np.nan

            # Forward 14d return
            if (idx_now + 14) < len(common_idx):
                t_fwd_14d = common_idx[idx_now + 14]
                p_fwd_14d = s_p.loc[t_fwd_14d] if t_fwd_14d in s_p.index else np.nan
                fwd_ret_14d = (p_fwd_14d - p_now) / p_now if (not np.isnan(p_fwd_14d) and p_fwd_14d > 0) else np.nan
            else:
                fwd_ret_14d = np.nan

            # Forward 30d return
            if (idx_now + 30) < len(common_idx):
                t_fwd_30d = common_idx[idx_now + 30]
                p_fwd_30d = s_p.loc[t_fwd_30d] if t_fwd_30d in s_p.index else np.nan
                fwd_ret_30d = (p_fwd_30d - p_now) / p_now if (not np.isnan(p_fwd_30d) and p_fwd_30d > 0) else np.nan
            else:
                fwd_ret_30d = np.nan

            coin_metrics.append({
                "symbol": sym,
                "price_at_rebalance": p_now,
                "alt_return_30d": alt_ret_30d,
                "rs_spread_vs_btc": rs_spread,
                "volatility_30d": vol_30d,
                "sharpe_momentum_score": sharpe_score,
                "fwd_7d_price": p_fwd_7d,
                "fwd_7d_return": fwd_ret_7d,
                "fwd_7d_alpha": fwd_alpha_7d,
                "fwd_14d_return": fwd_ret_14d,
                "fwd_30d_return": fwd_ret_30d,
            })

        if not coin_metrics:
            continue

        df_week_coins = pd.DataFrame(coin_metrics)

        # Compute Ranks
        df_week_coins["rank_raw_mom"] = df_week_coins["rs_spread_vs_btc"].rank(ascending=False, method="min").astype(int)
        df_week_coins["rank_sharpe_mom"] = df_week_coins["sharpe_momentum_score"].rank(ascending=False, method="min").astype(int)

        df_week_coins["is_top10_raw"] = (df_week_coins["rank_raw_mom"] <= 10).astype(int)
        df_week_coins["is_top10_sharpe"] = (df_week_coins["rank_sharpe_mom"] <= 10).astype(int)
        df_week_coins["is_top7_sharpe"] = (df_week_coins["rank_sharpe_mom"] <= 7).astype(int)
        df_week_coins["is_top5_sharpe"] = (df_week_coins["rank_sharpe_mom"] <= 5).astype(int)

        # Filter to keep at least top 20 of both factors or top 25 overall to save database space while keeping full breadth
        is_relevant = (df_week_coins["rank_raw_mom"] <= 20) | (df_week_coins["rank_sharpe_mom"] <= 20)
        df_week_filtered = df_week_coins[is_relevant].copy()

        df_week_filtered["rebalance_date"] = t_now.strftime("%Y-%m-%d %H:%M:%S")
        df_week_filtered["rebalance_day"] = t_now.strftime("%Y-%m-%d")
        df_week_filtered["week_number"] = week_num

        ranking_records.extend(df_week_filtered.to_dict("records"))

        rebalance_records.append({
            "week_number": week_num,
            "rebalance_date": t_now.strftime("%Y-%m-%d %H:%M:%S"),
            "rebalance_day": t_now.strftime("%Y-%m-%d"),
            "btc_price": btc_now,
            "btc_ema50": btc_ema,
            "btc_macro_bullish": btc_bullish,
            "btc_return_30d": btc_ret_30d,
            "btc_fwd_7d_return": btc_fwd_7d,
            "coins_evaluated": len(df_week_coins),
            "is_holdout_period": is_holdout
        })

    df_rebalance = pd.DataFrame(rebalance_records)
    df_rankings = pd.DataFrame(ranking_records)

    # 4. Save to SQLite Database
    db_path = results_dir / "weekly_relative_strength.db"
    conn = sqlite3.connect(db_path)

    df_rebalance.to_sql("rebalance_weeks", conn, if_exists="replace", index=False)
    df_rankings.to_sql("weekly_rankings", conn, if_exists="replace", index=False)

    cursor = conn.cursor()

    # Create Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_date ON weekly_rankings (rebalance_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_day ON weekly_rankings (rebalance_day);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON weekly_rankings (symbol);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rank_raw ON weekly_rankings (rank_raw_mom);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rank_sharpe ON weekly_rankings (rank_sharpe_mom);")

    # Create Convenient Views
    cursor.execute("DROP VIEW IF EXISTS v_top10_raw;")
    cursor.execute("""
    CREATE VIEW v_top10_raw AS
    SELECT 
        r.week_number,
        r.rebalance_day,
        r.rebalance_date,
        w.btc_macro_bullish,
        w.btc_price,
        r.rank_raw_mom,
        r.symbol,
        r.price_at_rebalance,
        ROUND(r.alt_return_30d * 100, 2) AS alt_ret_30d_pct,
        ROUND(r.rs_spread_vs_btc * 100, 2) AS rs_spread_pct,
        ROUND(r.fwd_7d_return * 100, 2) AS fwd_7d_return_pct,
        ROUND(r.fwd_7d_alpha * 100, 2) AS fwd_7d_alpha_pct
    FROM weekly_rankings r
    JOIN rebalance_weeks w ON r.rebalance_day = w.rebalance_day
    WHERE r.is_top10_raw = 1
    ORDER BY r.week_number, r.rank_raw_mom;
    """)

    # 2. Top 10 Sharpe Momentum View
    cursor.execute("DROP VIEW IF EXISTS v_top10_sharpe;")
    cursor.execute("""
    CREATE VIEW v_top10_sharpe AS
    SELECT 
        r.week_number,
        r.rebalance_day,
        r.rebalance_date,
        w.btc_macro_bullish,
        w.btc_price,
        r.rank_sharpe_mom,
        r.symbol,
        r.price_at_rebalance,
        ROUND(r.alt_return_30d * 100, 2) AS alt_ret_30d_pct,
        ROUND(r.volatility_30d * 100, 2) AS vol_30d_pct,
        ROUND(r.sharpe_momentum_score, 2) AS sharpe_score,
        ROUND(r.fwd_7d_return * 100, 2) AS fwd_7d_return_pct,
        ROUND(r.fwd_7d_alpha * 100, 2) AS fwd_7d_alpha_pct
    FROM weekly_rankings r
    JOIN rebalance_weeks w ON r.rebalance_day = w.rebalance_day
    WHERE r.is_top10_sharpe = 1
    ORDER BY r.week_number, r.rank_sharpe_mom;
    """)

    # 3. Top 7 Sharpe Momentum View (Our New Default)
    cursor.execute("DROP VIEW IF EXISTS v_top7_sharpe;")
    cursor.execute("""
    CREATE VIEW v_top7_sharpe AS
    SELECT 
        r.week_number,
        r.rebalance_day,
        r.rebalance_date,
        w.btc_macro_bullish,
        w.btc_price,
        r.rank_sharpe_mom,
        r.symbol,
        r.price_at_rebalance,
        ROUND(r.alt_return_30d * 100, 2) AS alt_ret_30d_pct,
        ROUND(r.volatility_30d * 100, 2) AS vol_30d_pct,
        ROUND(r.sharpe_momentum_score, 2) AS sharpe_score,
        ROUND(r.fwd_7d_return * 100, 2) AS fwd_7d_return_pct,
        ROUND(r.fwd_7d_alpha * 100, 2) AS fwd_7d_alpha_pct
    FROM weekly_rankings r
    JOIN rebalance_weeks w ON r.rebalance_day = w.rebalance_day
    WHERE r.is_top7_sharpe = 1
    ORDER BY r.week_number, r.rank_sharpe_mom;
    """)

    # 4. Weekly Macro & Forward Alpha Summary View
    cursor.execute("DROP VIEW IF EXISTS v_weekly_performance_summary;")
    cursor.execute("""
    CREATE VIEW v_weekly_performance_summary AS
    SELECT 
        w.week_number,
        w.rebalance_day AS date,
        w.btc_macro_bullish,
        ROUND(w.btc_fwd_7d_return * 100, 2) AS btc_fwd_7d_pct,
        ROUND(AVG(CASE WHEN r.is_top10_raw = 1 THEN r.fwd_7d_return END) * 100, 2) AS top10_raw_fwd_7d_pct,
        ROUND(AVG(CASE WHEN r.is_top10_sharpe = 1 THEN r.fwd_7d_return END) * 100, 2) AS top10_sharpe_fwd_7d_pct,
        ROUND(AVG(CASE WHEN r.is_top7_sharpe = 1 THEN r.fwd_7d_return END) * 100, 2) AS top7_sharpe_fwd_7d_pct
    FROM rebalance_weeks w
    LEFT JOIN weekly_rankings r ON w.rebalance_day = r.rebalance_day
    GROUP BY w.week_number
    ORDER BY w.week_number;
    """)

    conn.commit()
    conn.close()
    logger.info(f"Successfully constructed SQLite database at: {db_path}")

    # 5. Export to Parquet and CSV
    parquet_path = results_dir / "weekly_top_rankings.parquet"
    df_rankings.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet export to: {parquet_path}")

    # Top 10 CSV export
    top10_csv_path = results_dir / "weekly_top10_relative_strength.csv"
    df_top10 = df_rankings[df_rankings["is_top10_raw"] == 1].sort_values(["week_number", "rank_raw_mom"])
    df_top10.to_csv(top10_csv_path, index=False)
    logger.info(f"Saved Top 10 CSV export to: {top10_csv_path}")

    # Top 7 Sharpe CSV export
    top7_csv_path = results_dir / "weekly_top7_sharpe_momentum.csv"
    df_top7 = df_rankings[df_rankings["is_top7_sharpe"] == 1].sort_values(["week_number", "rank_sharpe_mom"])
    df_top7.to_csv(top7_csv_path, index=False)
    logger.info(f"Saved Top 7 Sharpe CSV export to: {top7_csv_path}")

    logger.info(f"Total rebalance weeks recorded: {len(df_rebalance)}")
    logger.info(f"Total ranking rows recorded: {len(df_rankings)}")


if __name__ == "__main__":
    main()
