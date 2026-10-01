"""
query_weekly_db.py
Convenient CLI utility to query the Weekly Relative Strength & Sharpe Momentum Database.

Usage Examples:
    python query_weekly_db.py --latest
    python query_weekly_db.py --week 1
    python query_weekly_db.py --date 2026-03-02
    python query_weekly_db.py --symbol ZEC_USDT
    python query_weekly_db.py --summary
    python query_weekly_db.py --sql "SELECT week_number, rebalance_day, symbol, fwd_7d_return_pct FROM v_top7_sharpe WHERE rank_sharpe_mom = 1 LIMIT 10"
"""

import sys
import argparse
import sqlite3
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output on all consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_PATH = Path("results/weekly_relative_strength.db")


def get_connection():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}. Run 'python build_weekly_relative_strength_db.py' first.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def show_week(conn, week_num: int = None, date_str: str = None):
    if week_num is not None:
        where_clause = f"w.week_number = {week_num}"
    elif date_str is not None:
        where_clause = f"w.rebalance_day LIKE '{date_str}%'"
    else:
        where_clause = "w.week_number = (SELECT MAX(week_number) FROM rebalance_weeks)"

    query_meta = f"""
    SELECT week_number, rebalance_day, btc_price, btc_ema50, btc_macro_bullish, 
           ROUND(btc_return_30d * 100, 2) AS btc_30d_pct,
           ROUND(btc_fwd_7d_return * 100, 2) AS btc_fwd_7d_pct
    FROM rebalance_weeks w
    WHERE {where_clause};
    """
    df_meta = pd.read_sql(query_meta, conn)
    if df_meta.empty:
        print("No matching rebalance week found.")
        return

    row = df_meta.iloc[0]
    wn = int(row["week_number"])
    day = row["rebalance_day"]
    bullish = "✅ BULLISH" if row["btc_macro_bullish"] == 1 else "❌ BEARISH (CASH)"

    print("=" * 80)
    print(f"WEEK #{wn:2d} REBALANCE: {day} | BTC Regime: {bullish}")
    print(f"BTC Price: ${row['btc_price']:,.2f} | EMA50: ${row['btc_ema50']:,.2f} | 30d Return: {row['btc_30d_pct']:+.1f}% | Next 7d: {row['btc_fwd_7d_pct']:+.1f}%")
    print("=" * 80)

    # Top 10 Raw Momentum
    print("\n🏆 TOP 10 RELATIVE STRENGTH (30-Day Return vs BTC):")
    query_raw = f"""
    SELECT rank_raw_mom AS rank, symbol, price_at_rebalance, alt_ret_30d_pct, rs_spread_pct, fwd_7d_return_pct, fwd_7d_alpha_pct
    FROM v_top10_raw
    WHERE week_number = {wn}
    ORDER BY rank_raw_mom;
    """
    df_raw = pd.read_sql(query_raw, conn)
    print(df_raw.to_string(index=False))

    # Top 7 Quality / Sharpe Momentum
    print("\n🛡️ TOP 7 QUALITY / SHARPE MOMENTUM (Return / Volatility):")
    query_sharpe = f"""
    SELECT rank_sharpe_mom AS rank, symbol, price_at_rebalance, alt_ret_30d_pct, vol_30d_pct, sharpe_score, fwd_7d_return_pct, fwd_7d_alpha_pct
    FROM v_top7_sharpe
    WHERE week_number = {wn}
    ORDER BY rank_sharpe_mom;
    """
    df_sharpe = pd.read_sql(query_sharpe, conn)
    print(df_sharpe.to_string(index=False))
    print()


def show_symbol(conn, symbol: str):
    sym = symbol.upper()
    if not sym.endswith("_USDT"):
        sym = f"{sym}_USDT"

    print("=" * 80)
    print(f"HISTORICAL APPEARANCES IN TOP 10 / TOP 7 FOR: {sym}")
    print("=" * 80)

    query = f"""
    SELECT 
        w.week_number,
        w.rebalance_day,
        w.btc_macro_bullish,
        r.rank_raw_mom,
        r.rank_sharpe_mom,
        ROUND(r.alt_return_30d * 100, 2) AS alt_30d_pct,
        ROUND(r.rs_spread_vs_btc * 100, 2) AS rs_spread_pct,
        ROUND(r.volatility_30d * 100, 2) AS vol_30d_pct,
        ROUND(r.sharpe_momentum_score, 2) AS sharpe_score,
        ROUND(r.fwd_7d_return * 100, 2) AS fwd_7d_return_pct
    FROM weekly_rankings r
    JOIN rebalance_weeks w ON r.rebalance_day = w.rebalance_day
    WHERE r.symbol = '{sym}'
    ORDER BY w.week_number;
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        print(f"No records found for {sym}.")
    else:
        print(df.to_string(index=False))
        print(f"\nTotal weekly appearances: {len(df)}")
        avg_fwd = df["fwd_7d_return_pct"].mean()
        win_rate = (df["fwd_7d_return_pct"] > 0).mean() * 100
        print(f"Average forward 7-day return: {avg_fwd:+.2f}% | Win rate: {win_rate:.1f}%")
    print()


def show_summary(conn):
    print("=" * 80)
    print("ALL 53 WEEKS PERFORMANCE & REGIME SUMMARY")
    print("=" * 80)
    query = "SELECT * FROM v_weekly_performance_summary ORDER BY week_number;"
    df = pd.read_sql(query, conn)
    print(df.to_string(index=False))
    print()


def run_custom_sql(conn, sql: str):
    print("=" * 80)
    print("CUSTOM SQL QUERY EXECUTION")
    print(f"Query: {sql}")
    print("=" * 80)
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    print(f"\n({len(df)} rows returned)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the Weekly Relative Strength Database.")
    parser.add_argument("--latest", action="store_true", help="Show the most recent rebalance week.")
    parser.add_argument("--week", type=int, help="Show a specific week number (1 to 53).")
    parser.add_argument("--date", type=str, help="Show week matching date (e.g. 2026-03-02).")
    parser.add_argument("--symbol", type=str, help="Show historical record for a symbol (e.g. SOL or SOL_USDT).")
    parser.add_argument("--summary", action="store_true", help="Show 53-week performance summary table.")
    parser.add_argument("--sql", type=str, help="Execute custom SQL statement.")

    args = parser.parse_args()
    conn = get_connection()

    if args.latest:
        show_week(conn)
    elif args.week is not None:
        show_week(conn, week_num=args.week)
    elif args.date:
        show_week(conn, date_str=args.date)
    elif args.symbol:
        show_symbol(conn, symbol=args.symbol)
    elif args.summary:
        show_summary(conn)
    elif args.sql:
        run_custom_sql(conn, args.sql)
    else:
        # Default: show latest week
        show_week(conn)

    conn.close()


if __name__ == "__main__":
    main()
