"""
run_portfolio_size_comparison.py
Empirical investigation of Portfolio Concentration / Breadth:
Compares Top 3, Top 5, Top 7, Top 10, Top 15, Top 20
across both:
1. Baseline 30-Day Momentum (Raw Return)
2. Quality/Sharpe Momentum (Return / Volatility)
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("size_comparison")


def run_portfolio_simulation(
    price_matrix: pd.DataFrame,
    btc_series: pd.Series,
    selection_fn: Callable[[pd.DataFrame, pd.Series, int, pd.Timestamp, int], List[str]],
    portfolio_size: int,
    rebalance_days: int = 7,
    btc_filter: bool = True,
    exec_model: Optional[ExecutionModel] = None,
    holdout_ts: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    common_idx = price_matrix.index.intersection(btc_series.index).sort_values()
    if holdout_ts is not None:
        common_idx = common_idx[common_idx < holdout_ts]

    min_lookback = 65
    if len(common_idx) < min_lookback + rebalance_days + 5:
        return {}

    df_prices = price_matrix.loc[common_idx]
    s_btc = btc_series.loc[common_idx]

    btc_ema50 = compute_ema(s_btc, span=50)

    current_weights: Dict[str, float] = {}
    rebalance_indices = range(min_lookback, len(common_idx) - rebalance_days, rebalance_days)

    portfolio_equity = 1.0
    equity_curve = [portfolio_equity]
    equity_dates = [common_idx[min_lookback]]

    taker_fee = exec_model.taker_fee_rate if exec_model else 0.0002
    base_slip = (exec_model.base_slippage_bps / 10000.0) if exec_model else 0.0010
    total_one_way_cost = taker_fee + base_slip

    rebalance_records = []

    for idx in rebalance_indices:
        t_decision = common_idx[idx]
        t_next = common_idx[idx + rebalance_days]

        # Macro trend check
        is_cash = False
        if btc_filter:
            if s_btc.loc[t_decision] < btc_ema50.loc[t_decision]:
                is_cash = True

        # Run coin selection function with portfolio_size
        if is_cash:
            selected = []
            target_weights = {}
        else:
            selected = selection_fn(df_prices, s_btc, idx, t_decision, portfolio_size)
            if not selected:
                target_weights = {}
            else:
                w = 1.0 / len(selected)
                target_weights = {s: w for s in selected}

        # Turnover & Cost
        all_syms = set(current_weights.keys()).union(set(target_weights.keys()))
        turnover = sum(abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_syms) / 2.0
        fee_drag = turnover * 2.0 * total_one_way_cost
        funding_drag = (0.0001 * 3.0 * rebalance_days) if not is_cash else 0.0

        # Forward return
        p_start = df_prices.loc[t_decision]
        p_end = df_prices.loc[t_next]
        fwd_returns = (p_end - p_start) / p_start

        if is_cash or not selected:
            fwd_gross_ret = 0.0
        else:
            rets = [fwd_returns[s] for s in selected if s in fwd_returns and not np.isnan(fwd_returns[s])]
            fwd_gross_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0

        btc_fwd_ret = (s_btc.loc[t_next] - s_btc.loc[t_decision]) / s_btc.loc[t_decision]
        fwd_net_ret = fwd_gross_ret - fee_drag - funding_drag

        portfolio_equity = portfolio_equity * (1.0 + fwd_net_ret)
        equity_curve.append(portfolio_equity)
        equity_dates.append(t_next)

        rebalance_records.append({
            "timestamp": t_decision,
            "portfolio_return_net": fwd_net_ret,
            "portfolio_return_gross": fwd_gross_ret,
            "btc_return": btc_fwd_ret,
            "turnover": turnover,
            "is_cash": is_cash,
            "n_selected": len(selected)
        })
        current_weights = target_weights

    df_rebs = pd.DataFrame(rebalance_records)
    net_rets = df_rebs["portfolio_return_net"].to_numpy(dtype=float)
    btc_rets = df_rebs["btc_return"].to_numpy(dtype=float)

    total_net_ret = portfolio_equity - 1.0
    n_periods = len(net_rets)
    periods_per_year = 365.25 / rebalance_days
    cagr = ((portfolio_equity) ** (periods_per_year / n_periods)) - 1.0 if (n_periods > 0 and portfolio_equity > 0) else -1.0

    mean_ret = np.mean(net_rets)
    std_ret = np.std(net_rets)
    downside_std = np.std(net_rets[net_rets < 0]) if np.sum(net_rets < 0) > 0 else 1e-6
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 1e-6 else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 1e-6 else 0.0

    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks))

    btc_total_ret = np.prod(1.0 + btc_rets) - 1.0

    return {
        "portfolio_size": portfolio_size,
        "total_net_return": total_net_ret,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "win_rate_periods": float(np.mean(net_rets > 0)),
        "outperformed_btc_pct": float(np.mean(net_rets > btc_rets)),
        "btc_total_return": btc_total_ret,
        "alpha_vs_btc": total_net_ret - btc_total_ret,
        "avg_turnover": float(np.mean(df_rebs["turnover"])),
        "equity_curve": equity_curve,
        "equity_dates": equity_dates
    }


def main():
    logger.info("=" * 80)
    logger.info("PORTFOLIO CONCENTRATION STUDY: TOP N COMPARISON (TOP 3 TO TOP 20)")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    reports_dir = Path(cfg.data.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    holdout_start = cfg.data.holdout_start
    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    # Load Daily Price Matrix
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    btc_series = df_btc_1d.set_index("open_time")["close"]

    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    daily_close_dict = {}
    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        daily_close_dict[sym] = df_1d.set_index("open_time")["close"]

    price_matrix = pd.DataFrame(daily_close_dict)
    logger.info(f"Loaded price matrix with {price_matrix.shape[1]} altcoins.")

    # Selection Functions
    def get_valid_series(df_p, idx, lookback):
        t_now = df_p.index[idx]
        t_past = df_p.index[idx - lookback]
        p_now = df_p.loc[t_now]
        p_past = df_p.loc[t_past]
        mask = (p_now > 0) & (p_past > 0) & (~p_now.isna()) & (~p_past.isna())
        return (p_now[mask] - p_past[mask]) / p_past[mask]

    # 1. Raw 30d Momentum
    def sel_raw_momentum(df_p, s_btc, idx, t_now, n):
        ret_30d = get_valid_series(df_p, idx, 30)
        return ret_30d.nlargest(n).index.tolist()

    # 2. Quality/Sharpe Momentum (Return / Vol)
    def sel_sharpe_momentum(df_p, s_btc, idx, t_now, n):
        ret_30d = get_valid_series(df_p, idx, 30)
        daily_rets = df_p.iloc[idx - 30 : idx + 1].pct_change().dropna(how="all")
        vol_30d = daily_rets.std() * np.sqrt(365.25)
        valid_coins = ret_30d.index.intersection(vol_30d[vol_30d > 0.05].index)
        sharpe_mom = ret_30d.loc[valid_coins] / vol_30d.loc[valid_coins]
        return sharpe_mom.nlargest(n).index.tolist()

    sizes_to_test = [3, 5, 7, 10, 12, 15, 20]
    results_list = []
    equity_curves = {}

    for size in sizes_to_test:
        # Raw Momentum
        res_raw = run_portfolio_simulation(
            price_matrix, btc_series, sel_raw_momentum, portfolio_size=size,
            rebalance_days=7, btc_filter=True, exec_model=exec_model, holdout_ts=holdout_ts
        )
        res_raw["factor"] = "Raw 30d Momentum"
        res_raw["name"] = f"Raw 30d Mom (Top {size})"
        results_list.append(res_raw)
        equity_curves[f"Raw Top {size}"] = (res_raw["equity_dates"], res_raw["equity_curve"])

        # Sharpe Momentum
        res_sharpe = run_portfolio_simulation(
            price_matrix, btc_series, sel_sharpe_momentum, portfolio_size=size,
            rebalance_days=7, btc_filter=True, exec_model=exec_model, holdout_ts=holdout_ts
        )
        res_sharpe["factor"] = "Quality (Sharpe) Mom"
        res_sharpe["name"] = f"Quality Mom (Top {size})"
        results_list.append(res_sharpe)
        equity_curves[f"Quality Top {size}"] = (res_sharpe["equity_dates"], res_sharpe["equity_curve"])

        logger.info(
            f"Size {size:2d} | "
            f"Raw Mom: Net={res_raw['total_net_return']*100:+6.1f}%, Sharpe={res_raw['sharpe_ratio']:4.2f}, MaxDD={res_raw['max_drawdown']*100:4.1f}% | "
            f"Quality Mom: Net={res_sharpe['total_net_return']*100:+6.1f}%, Sharpe={res_sharpe['sharpe_ratio']:4.2f}, MaxDD={res_sharpe['max_drawdown']*100:4.1f}%"
        )

    # Format Table
    df_results = pd.DataFrame([
        {
            "Factor": r["factor"],
            "Portfolio Size": r["portfolio_size"],
            "Net Return (%)": r["total_net_return"] * 100,
            "CAGR (%)": r["cagr"] * 100,
            "Sharpe": r["sharpe_ratio"],
            "Sortino": r["sortino_ratio"],
            "Max Drawdown (%)": r["max_drawdown"] * 100,
            "Win Rate (%)": r["win_rate_periods"] * 100,
            "Avg Turnover (%)": r["avg_turnover"] * 100,
            "Alpha vs BTC (%)": r["alpha_vs_btc"] * 100,
        }
        for r in results_list
    ])

    out_csv = results_dir / "portfolio_size_comparison.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved size comparison results to {out_csv}")

    # Plot Comparison Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Net Return vs Portfolio Size
    raw_df = df_results[df_results["Factor"] == "Raw 30d Momentum"]
    quality_df = df_results[df_results["Factor"] == "Quality (Sharpe) Mom"]

    ax1 = axes[0, 0]
    ax1.plot(raw_df["Portfolio Size"], raw_df["Net Return (%)"], marker="o", color="#3b82f6", linewidth=2.5, label="Raw 30d Momentum")
    ax1.plot(quality_df["Portfolio Size"], quality_df["Net Return (%)"], marker="s", color="#10b981", linewidth=2.5, label="Quality (Sharpe) Momentum")
    ax1.set_title("Total Net Return vs. Portfolio Size (Concentration)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Coins (Portfolio Size)")
    ax1.set_ylabel("Net Return (%)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True)

    # 2. Sharpe Ratio vs Portfolio Size
    ax2 = axes[0, 1]
    ax2.plot(raw_df["Portfolio Size"], raw_df["Sharpe"], marker="o", color="#3b82f6", linewidth=2.5, label="Raw 30d Momentum")
    ax2.plot(quality_df["Portfolio Size"], quality_df["Sharpe"], marker="s", color="#10b981", linewidth=2.5, label="Quality (Sharpe) Momentum")
    ax2.set_title("Sharpe Ratio vs. Portfolio Size", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Coins (Portfolio Size)")
    ax2.set_ylabel("Annualized Sharpe Ratio")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True)

    # 3. Max Drawdown vs Portfolio Size
    ax3 = axes[1, 0]
    ax3.plot(raw_df["Portfolio Size"], raw_df["Max Drawdown (%)"], marker="o", color="#ef4444", linewidth=2.5, label="Raw 30d Momentum")
    ax3.plot(quality_df["Portfolio Size"], quality_df["Max Drawdown (%)"], marker="s", color="#f59e0b", linewidth=2.5, label="Quality (Sharpe) Momentum")
    ax3.set_title("Maximum Drawdown vs. Portfolio Size", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Number of Coins (Portfolio Size)")
    ax3.set_ylabel("Max Drawdown (%)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(frameon=True)

    # 4. Equity Curves: Top 5 vs Top 10
    ax4 = axes[1, 1]
    d_raw5, eq_raw5 = equity_curves["Raw Top 5"]
    d_raw10, eq_raw10 = equity_curves["Raw Top 10"]
    d_q5, eq_q5 = equity_curves["Quality Top 5"]
    d_q10, eq_q10 = equity_curves["Quality Top 10"]

    ax4.plot(d_raw5, eq_raw5, label="Raw Top 5", color="#93c5fd", linestyle="--", linewidth=1.8)
    ax4.plot(d_raw10, eq_raw10, label="Raw Top 10 (Baseline)", color="#1d4ed8", linewidth=2.0)
    ax4.plot(d_q5, eq_q5, label="Quality Top 5", color="#6ee7b7", linestyle="--", linewidth=1.8)
    ax4.plot(d_q10, eq_q10, label="Quality Top 10", color="#047857", linewidth=2.2)
    ax4.set_title("Equity Curves: Top 5 vs. Top 10 Head-to-Head", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Portfolio Value (Base = 1.0)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(frameon=True)

    plt.tight_layout()
    chart_path = figures_dir / "portfolio_size_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    logger.info(f"Saved size comparison chart to {chart_path}")


if __name__ == "__main__":
    main()
