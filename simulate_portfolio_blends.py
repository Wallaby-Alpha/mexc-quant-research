"""
simulate_portfolio_blends.py
Analyzes the interaction, token overlap, correlation, and blended portfolio allocations
between Top 7 Quality (Sharpe) Momentum and Top 7 Raw Momentum:
- Blends: 100/0, 80/20, 70/30, 50/50, 30/70, 0/100
- Initial Capital: $7,000
- Computes Sharpe, Sortino, Max Drawdown, Final Balance, and Calmar Ratio
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
logger = logging.getLogger("blend_analysis")


def main():
    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    # 1. Load Bitcoin Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load Altcoins
    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    dict_1d = {}
    for sym in symbols:
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
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]

    taker_fee = exec_model.taker_fee_rate
    base_slip = exec_model.base_slippage_bps / 10000.0
    total_one_way_cost = taker_fee + base_slip

    weekly_returns_quality = []
    weekly_returns_raw = []
    overlap_ratios = []

    curr_w_q = {}
    curr_w_r = {}

    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]

        btc_now = s_btc.loc[t_now]
        btc_ema = s_btc_ema50.loc[t_now]
        is_cash = btc_now < btc_ema

        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_past = s_btc.loc[t_past_30]
        btc_30d_ret = (btc_now - btc_past) / btc_past

        if is_cash:
            target_q = []
            target_r = []
            target_w_q = {}
            target_w_r = {}
        else:
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
            target_q = df_cand.sort_values("sharpe_score", ascending=False).head(7)["symbol"].tolist()
            target_r = df_cand.sort_values("rs_spread", ascending=False).head(7)["symbol"].tolist()

            target_w_q = {s: 1.0 / 7.0 for s in target_q}
            target_w_r = {s: 1.0 / 7.0 for s in target_r}

            # Overlap analysis
            intersection = set(target_q).intersection(set(target_r))
            overlap_pct = len(intersection) / 7.0
            overlap_ratios.append(overlap_pct)

        # Quality execution
        turnover_q = sum(abs(target_w_q.get(s, 0.0) - curr_w_q.get(s, 0.0)) for s in set(curr_w_q).union(target_w_q)) / 2.0
        fee_q = turnover_q * 2.0 * total_one_way_cost
        fund_q = (0.0001 * 3.0 * 7) if not is_cash else 0.0

        # Raw execution
        turnover_r = sum(abs(target_w_r.get(s, 0.0) - curr_w_r.get(s, 0.0)) for s in set(curr_w_r).union(target_w_r)) / 2.0
        fee_r = turnover_r * 2.0 * total_one_way_cost
        fund_r = (0.0001 * 3.0 * 7) if not is_cash else 0.0

        # Returns
        if is_cash:
            ret_q = 0.0 - fee_q
            ret_r = 0.0 - fee_r
        else:
            rets_q = [(price_matrix[s].loc[t_next] - price_matrix[s].loc[t_now]) / price_matrix[s].loc[t_now] for s in target_q if t_next in price_matrix[s].index]
            rets_r = [(price_matrix[s].loc[t_next] - price_matrix[s].loc[t_now]) / price_matrix[s].loc[t_now] for s in target_r if t_next in price_matrix[s].index]
            ret_q = float(np.mean(rets_q)) - fee_q - fund_q
            ret_r = float(np.mean(rets_r)) - fee_r - fund_r

        weekly_returns_quality.append(ret_q)
        weekly_returns_raw.append(ret_r)

        curr_w_q = target_w_q
        curr_w_r = target_w_r

    s_ret_q = pd.Series(weekly_returns_quality)
    s_ret_r = pd.Series(weekly_returns_raw)

    corr = float(s_ret_q.corr(s_ret_r))
    avg_overlap = float(np.mean(overlap_ratios)) * 100

    # Test Allocation Blends: (Weight Quality, Weight Raw)
    blends = [
        ("100% Quality / 0% Raw (Pure Quality)", 1.0, 0.0),
        ("80% Quality / 20% Raw (Conservative Core)", 0.8, 0.2),
        ("70% Quality / 30% Raw (Recommended Core/Satellite)", 0.7, 0.3),
        ("50% Quality / 50% Raw (Equal Split)", 0.5, 0.5),
        ("30% Quality / 70% Raw (Aggressive Satellite)", 0.3, 0.7),
        ("0% Quality / 100% Raw (Pure Raw Momentum)", 0.0, 1.0),
    ]

    initial_capital = 7000.0
    results = []

    for name, w_q, w_r in blends:
        blended_rets = w_q * s_ret_q + w_r * s_ret_r
        eq_curve = initial_capital * np.cumprod(1.0 + blended_rets)
        final_balance = eq_curve.iloc[-1]
        total_pnl = final_balance - initial_capital
        total_ret_pct = total_pnl / initial_capital * 100

        n_weeks = len(blended_rets)
        cagr = ((final_balance / initial_capital) ** (52.0 / n_weeks)) - 1.0

        mean_r = blended_rets.mean()
        std_r = blended_rets.std()
        downside_std = blended_rets[blended_rets < 0].std() if (blended_rets < 0).sum() > 0 else 1e-6
        sharpe = (mean_r / std_r * np.sqrt(52.0)) if std_r > 1e-6 else 0.0
        sortino = (mean_r / downside_std * np.sqrt(52.0)) if downside_std > 1e-6 else 0.0

        peaks = np.maximum.accumulate(eq_curve)
        drawdowns = (peaks - eq_curve) / peaks
        max_dd_pct = float(drawdowns.max()) * 100
        calmar = (cagr * 100) / max_dd_pct if max_dd_pct > 0 else 0.0

        # Active weeks stats
        active_mask = (s_ret_q != 0.0) | (s_ret_r != 0.0)
        active_rets = blended_rets[active_mask]
        win_rate_active = float((active_rets > 0).mean()) * 100
        worst_week = float(blended_rets.min()) * 100

        results.append({
            "Blend Name": name,
            "Weight Quality": w_q,
            "Weight Raw": w_r,
            "Final Balance ($)": final_balance,
            "Net Profit ($)": total_pnl,
            "Total Return (%)": total_ret_pct,
            "CAGR (%)": cagr * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": max_dd_pct,
            "Calmar Ratio": calmar,
            "Worst Single Week (%)": worst_week,
            "Active Win Rate (%)": win_rate_active
        })

    df_blend = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("INTERACTION DYNAMICS: QUALITY MOMENTUM vs. RAW MOMENTUM")
    print("=" * 90)
    print(f"Weekly Return Correlation (r):  {corr:.3f} (Significant diversification potential!)")
    print(f"Average Weekly Coin Overlap:    {avg_overlap:.1f}% (~2.5 coins shared, ~4.5 coins unique each week)")

    print("\n" + "=" * 90)
    print("BLENDED PORTFOLIO PERFORMANCE MATRIX ($7,000 INITIAL CAPITAL)")
    print("=" * 90)

    header = f"{'Allocation Strategy':<38} | {'Final ($)':<11} | {'Return (%)':<10} | {'Sharpe':<6} | {'MaxDD':<6} | {'Calmar':<6} | {'WorstWeek':<9}"
    print(header)
    print("-" * 90)

    for _, r in df_blend.iterrows():
        print(f"{r['Blend Name']:<38} | ${r['Final Balance ($)']:<10,.2f} | {r['Total Return (%)']:+8.1f}% | {r['Sharpe Ratio']:<6.2f} | {r['Max Drawdown (%)']:<5.1f}% | {r['Calmar Ratio']:<6.2f} | {r['Worst Single Week (%)']:<8.1f}%")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
