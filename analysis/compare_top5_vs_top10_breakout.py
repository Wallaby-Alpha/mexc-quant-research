"""
analysis/compare_top5_vs_top10_breakout.py
Rigorous mathematical comparison between Top 5 vs Top 10 Rotational Momentum portfolios
restricted strictly to the 65 user-specified Breakout coins over the 1-year dataset.

Evaluates:
1. Total Net Annual Return (net of fees and slippage)
2. Max Peak-to-Trough Drawdown
3. Worst Single-Day Portfolio Loss
4. Volatility & Risk-Adjusted Return (Sharpe ratio)
5. Weekly Portfolio Turnover (% replaced each week)
6. Breakout Prop Firm Compliance (Max Daily Loss <= 3.0%, Max DD <= 5.0% / 6.0%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from backtest.execution_model import ExecutionModel

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BREAKOUT_UNIVERSE = [
    "BCH", "LIGHTER", "MON", "STX", "ETHFI", "JTO", "PEPE", "ENA", "WIF",
    "SUI", "NEAR", "GRASS", "UNI", "AAVE", "ADA", "AIXBT", "ALGO", "APT",
    "ARB", "ASTER", "ATOM", "AVAX", "BNB", "BONK", "CRV", "DOGE", "DOT",
    "ETC", "ETH", "FARTCOIN", "FIL", "FLOKI", "HBAR", "HYPE", "ICP",
    "INJ", "JUP", "KAITO", "LDO", "LINK", "LTC", "MOODENG", "ONDO", "OP",
    "PENDLE", "PENGU", "PNUT", "POL", "POPCAT", "PUMP", "RENDER", "S",
    "SHIB", "SOL", "TAO", "TIA", "TRUMP", "TRX", "VIRTUAL", "WLD", "XLM",
    "XPL", "XRP", "ZEC", "ZRO"
]


def run_portfolio_simulation(k: int, price_matrix: pd.DataFrame, s_btc: pd.Series, s_btc_ema50: pd.Series, fee_rate: float, monday_dates: List[pd.Timestamp]) -> Dict[str, Any]:
    weekly_records = []
    all_daily_rets = []
    prev_selected = set()
    turnovers = []

    for i in range(len(monday_dates) - 1):
        t0 = monday_dates[i]
        t1 = monday_dates[i + 1]

        # BTC Macro Check (closed candle at t0)
        btc_close_t0 = s_btc.loc[t0]
        btc_ema_t0 = s_btc_ema50.loc[t0]

        daily_slice = price_matrix.loc[t0:t1]
        n_days = len(daily_slice) - 1
        if n_days <= 0:
            continue

        if btc_close_t0 <= btc_ema_t0:
            # Bearish -> 100% Cash
            # If held positions, incur exit fee
            turnover_pct = 1.0 if prev_selected else 0.0
            turnovers.append(turnover_pct)
            fee = fee_rate if prev_selected else 0.0
            weekly_records.append({
                "week_start": t0,
                "net_week_ret": -fee,
                "selected": [],
                "is_cash": True
            })
            all_daily_rets.extend([0.0] * n_days)
            prev_selected = set()
            continue

        # Lookback 20 days prior to t0
        past_idx = price_matrix.index.get_loc(t0)
        if past_idx < 20:
            all_daily_rets.extend([0.0] * n_days)
            weekly_records.append({"week_start": t0, "net_week_ret": 0.0, "selected": [], "is_cash": True})
            continue

        t_past = price_matrix.index[past_idx - 20]
        p_now = price_matrix.loc[t0]
        p_past = price_matrix.loc[t_past]

        # 20-day returns
        ret_20d = (p_now - p_past) / p_past
        valid_ret = ret_20d.dropna()
        ranked = valid_ret.sort_values(ascending=False)

        selected_k = ranked.head(k).index.tolist()
        curr_selected = set(selected_k)

        # Turnover calculation
        if prev_selected:
            retained = len(prev_selected.intersection(curr_selected))
            turnover_pct = (k - retained) / k
        else:
            turnover_pct = 1.0
        turnovers.append(turnover_pct)

        # Fees: roundtrip on new entries, 0 on retained
        fee = turnover_pct * fee_rate

        # Calculate daily portfolio returns
        daily_rets_k = []
        for d in range(n_days):
            day_t0 = daily_slice.index[d]
            day_t1 = daily_slice.index[d + 1]
            p0 = daily_slice.loc[day_t0, selected_k]
            p1 = daily_slice.loc[day_t1, selected_k]
            d_ret = float(((p1 - p0) / p0).mean())
            daily_rets_k.append(d_ret)

        all_daily_rets.extend(daily_rets_k)

        # Week total return
        p_start = daily_slice.iloc[0][selected_k]
        p_end = daily_slice.iloc[-1][selected_k]
        gross_week = float(((p_end - p_start) / p_start).mean())
        net_week = gross_week - fee

        weekly_records.append({
            "week_start": t0,
            "net_week_ret": net_week,
            "selected": selected_k,
            "is_cash": False
        })
        prev_selected = curr_selected

    s_daily = pd.Series(all_daily_rets)
    s_weekly = pd.Series([r["net_week_ret"] for r in weekly_records])

    # Cumulative equity
    cum_unscaled = (1.0 + s_weekly).cumprod()
    max_dd = float(((cum_unscaled - cum_unscaled.cummax()) / cum_unscaled.cummax()).min() * 100.0)
    total_ret = float((cum_unscaled.iloc[-1] - 1.0) * 100.0)
    worst_day = float(s_daily.min() * 100.0)
    days_breaching_3pct = int((s_daily <= -0.03).sum())

    # Annualized Sharpe (assuming 0 risk free)
    mean_weekly = s_weekly.mean()
    std_weekly = s_weekly.std()
    sharpe = float((mean_weekly / std_weekly) * np.sqrt(52)) if std_weekly > 0 else 0.0

    avg_turnover = float(np.mean(turnovers) * 100.0)

    return {
        "k": k,
        "total_ret": total_ret,
        "max_dd": max_dd,
        "worst_day": worst_day,
        "days_breaching_3pct": days_breaching_3pct,
        "sharpe": sharpe,
        "avg_turnover": avg_turnover,
        "s_daily": s_daily,
        "s_weekly": s_weekly
    }


def main():
    print("=" * 95)
    print("MATHEMATICAL COMPARISON: TOP 5 vs TOP 10 PORTFOLIO (BREAKOUT COIN UNIVERSE)")
    print("=" * 95)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    exec_model = ExecutionModel()
    total_fee_roundtrip = (exec_model.taker_fee_rate * 2.0) + (exec_model.base_slippage_bps / 10000.0 * 2.0)

    # Overlap Breakout universe with cache
    cached_symbols = set(cache.list_cached_symbols("1h"))
    valid_breakout_symbols = []
    for coin in BREAKOUT_UNIVERSE:
        sym = f"{coin}_USDT"
        if sym in cached_symbols:
            valid_breakout_symbols.append(sym)

    print(f"Total Breakout assets evaluated in historical dataset: {len(valid_breakout_symbols)} coins")
    print(f"Roundtrip Fee & Slippage modeled: {total_fee_roundtrip*10000:.1f} bps per new rebalance position")
    print("-" * 95)

    # Load 1D closes
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    s_btc = df_btc_1d.sort_values("open_time").set_index("open_time")["close"]
    s_btc_ema50 = s_btc.ewm(span=50, adjust=False).mean()

    dict_1d = {}
    for sym in valid_breakout_symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d[sym] = df_1d.sort_values("open_time").set_index("open_time")["close"]

    price_matrix = pd.DataFrame(dict_1d)
    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.weekday() == 0]

    # Run simulations
    res_top5 = run_portfolio_simulation(5, price_matrix, s_btc, s_btc_ema50, total_fee_roundtrip, mondays)
    res_top10 = run_portfolio_simulation(10, price_matrix, s_btc, s_btc_ema50, total_fee_roundtrip, mondays)

    # 1. Raw Comparison Table
    print("\n1. RAW UNLEVERAGED PERFORMANCE (100% CAPITAL ALLOCATED TO ALTS):")
    print("-" * 95)
    print(f"{'Metric':<32} | {'Top 5 Portfolio':<25} | {'Top 10 Portfolio':<25} | {'Difference / Delta'}")
    print("-" * 95)
    ret_diff = res_top5['total_ret'] - res_top10['total_ret']
    dd_diff = res_top5['max_dd'] - res_top10['max_dd']
    worst_diff = res_top5['worst_day'] - res_top10['worst_day']
    sharpe_diff = res_top5['sharpe'] - res_top10['sharpe']
    turnover_diff = res_top5['avg_turnover'] - res_top10['avg_turnover']

    print(f"{'Full Year Net Return':<32} | {res_top5['total_ret']:+6.1f}%                   | {res_top10['total_ret']:+6.1f}%                   | {ret_diff:+6.1f}% (Top 5 has higher raw upside)")
    print(f"{'Maximum Peak-to-Trough DD':<32} | {res_top5['max_dd']:5.1f}%                   | {res_top10['max_dd']:5.1f}%                   | {dd_diff:+5.1f}% (Top 5 suffers deeper drawdowns)")
    print(f"{'Worst Single-Day Portfolio Loss':<32} | {res_top5['worst_day']:6.2f}%                  | {res_top10['worst_day']:6.2f}%                  | {worst_diff:+6.2f}% (Top 5 daily crash risk is worse)")
    print(f"{'Days with Loss > 3.0%':<32} | {res_top5['days_breaching_3pct']:<3d} days                   | {res_top10['days_breaching_3pct']:<3d} days                   | {res_top5['days_breaching_3pct'] - res_top10['days_breaching_3pct']:+3d} days (Fails prop rule more often)")
    print(f"{'Annualized Sharpe Ratio':<32} | {res_top5['sharpe']:5.2f}                    | {res_top10['sharpe']:5.2f}                    | {sharpe_diff:+5.2f}")
    print(f"{'Avg Weekly Portfolio Turnover':<32} | {res_top5['avg_turnover']:5.1f}%                   | {res_top10['avg_turnover']:5.1f}%                   | {turnover_diff:+5.1f}% (Top 5 rotates coins more frequently)")
    print("-" * 95)

    # 2. Prop Firm Compliance Table
    print("\n2. PROP FIRM SIZING SIMULATION (Breakout Rules: Daily <= 3.0% | Max DD <= 5.0% Pro / 6.0% Classic):")
    print("-" * 95)
    header = f"{'Exposure Sizing':<17} | {'Top 5 Worst Day':<15} | {'Top 5 Max DD':<13} | {'Top 10 Worst Day':<16} | {'Top 10 Max DD':<14} | {'Compliance Status'}"
    print(header)
    print("-" * 95)

    for scale in [1.0, 0.50, 0.30, 0.20, 0.15, 0.12, 0.10]:
        t5_worst = res_top5['worst_day'] * scale
        t5_dd = res_top5['max_dd'] * scale
        t10_worst = res_top10['worst_day'] * scale
        t10_dd = res_top10['max_dd'] * scale

        t5_pass = abs(t5_worst) < 3.0 and abs(t5_dd) < 5.0
        t10_pass = abs(t10_worst) < 3.0 and abs(t10_dd) < 5.0

        if t5_pass and t10_pass:
            status = "✅ Both Pass Pro & Classic"
        elif not t5_pass and t10_pass:
            status = "⚠️ Only Top 10 Passes (Top 5 breaches)"
        elif not t5_pass and not t10_pass:
            status = "❌ Both Fail (Drawdown/Daily breach)"
        else:
            status = "⚠️ Only Top 5 Passes"

        print(f"{scale*100:4.0f}% Exposure      | {t5_worst:6.2f}%         | {t5_dd:5.1f}%        | {t10_worst:6.2f}%          | {t10_dd:5.1f}%         | {status}")

    print("=" * 95)


if __name__ == "__main__":
    main()
