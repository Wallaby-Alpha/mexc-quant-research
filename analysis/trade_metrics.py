from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from analysis.baselines import compute_wilson_ci


class TradeMetricsCalculator:
    """
    Computes performance metrics with NET-OF-COST as headline, gross secondary.
    Includes statistical confidence intervals, drawdown, Sharpe/Sortino,
    and segmented breakdowns per DEFINITIONS.md and AGENTS.md.
    """

    @staticmethod
    def compute_summary_metrics(df_trades: pd.DataFrame, initial_capital: float = 10_000.0, risk_fraction: float = 0.01) -> Dict[str, Any]:
        if df_trades.empty:
            return {
                "n_trades": 0,
                "insufficient_sample": True,
                "win_rate_net": 0.0,
                "expectancy_net_r": 0.0,
                "profit_factor_net": 0.0
            }

        n = len(df_trades)
        insufficient_sample = n < 100

        # Sort chronologically by entry_time
        df = df_trades.sort_values(by="entry_time").reset_index(drop=True)

        net_r = df["net_pnl_r"].to_numpy(dtype=float)
        gross_r = df["gross_pnl_r"].to_numpy(dtype=float)
        net_pct = df["net_pnl_pct"].to_numpy(dtype=float)
        gross_pct = df["gross_pnl_pct"].to_numpy(dtype=float)

        # Net wins and losses
        net_wins = net_r[net_r > 0]
        net_losses = net_r[net_r <= 0]
        n_wins = len(net_wins)
        n_losses = len(net_losses)

        win_rate_net = n_wins / n if n > 0 else 0.0
        _, ci_low, ci_high = compute_wilson_ci(n_wins, n)

        avg_win_r = float(np.mean(net_wins)) if n_wins > 0 else 0.0
        avg_loss_r = float(np.mean(net_losses)) if n_losses > 0 else 0.0
        avg_win_pct = float(np.mean(net_pct[net_r > 0])) if n_wins > 0 else 0.0
        avg_loss_pct = float(np.mean(net_pct[net_r <= 0])) if n_losses > 0 else 0.0

        expectancy_net_r = float(np.mean(net_r))
        expectancy_net_pct = float(np.mean(net_pct))

        # Profit Factor
        sum_gains = float(np.sum(net_wins)) if n_wins > 0 else 0.0
        sum_losses = float(np.abs(np.sum(net_losses))) if n_losses > 0 else 0.0
        profit_factor_net = (sum_gains / sum_losses) if sum_losses > 0 else float("inf")

        # Gross metrics (secondary)
        gross_wins = gross_r[gross_r > 0]
        gross_losses = gross_r[gross_r <= 0]
        win_rate_gross = len(gross_wins) / n if n > 0 else 0.0
        expectancy_gross_r = float(np.mean(gross_r))
        sum_gross_gains = float(np.sum(gross_wins)) if len(gross_wins) > 0 else 0.0
        sum_gross_losses = float(np.abs(np.sum(gross_losses))) if len(gross_losses) > 0 else 0.0
        profit_factor_gross = (sum_gross_gains / sum_gross_losses) if sum_gross_losses > 0 else float("inf")

        # Equity Curve Simulation (Fixed Fractional Risk)
        # Starting equity $10,000, risking 1% per trade
        equity = [initial_capital]
        for r_mult in net_r:
            curr_eq = equity[-1]
            trade_pnl = curr_eq * risk_fraction * r_mult
            equity.append(max(0.0, curr_eq + trade_pnl))

        equity_curve = np.array(equity)
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_drawdown_pct = float(np.min(drawdowns)) * 100.0  # as negative percentage

        # Holding times
        holding_mins = df["holding_time_minutes"].to_numpy(dtype=float)
        avg_holding_hours = float(np.mean(holding_mins)) / 60.0
        median_holding_hours = float(np.median(holding_mins)) / 60.0

        # Cost totals
        total_fees = float(df["total_fees"].sum())
        total_slippage = float(df["slippage_cost"].sum())
        total_funding = float(df["funding_cost"].sum())

        # Sharpe & Sortino (State assumptions clearly)
        # Assumed annualized based on average trading days span
        t_min = df["entry_time"].min()
        t_max = df["exit_time"].max()
        days_span = max(1.0, (t_max - t_min).total_seconds() / 86400.0)
        annualization_factor = np.sqrt(365.25 * (n / days_span)) if days_span > 0 else 1.0

        std_pct = float(np.std(net_pct))
        sharpe_ratio = (float(np.mean(net_pct)) / std_pct * annualization_factor) if std_pct > 0 else 0.0

        downside_returns = net_pct[net_pct < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (float(np.mean(net_pct)) / downside_std * annualization_factor) if downside_std > 0 else 0.0

        return {
            "n_trades": n,
            "insufficient_sample": insufficient_sample,
            # Net Headline Metrics
            "win_rate_net": win_rate_net,
            "win_rate_net_ci_low": ci_low,
            "win_rate_net_ci_high": ci_high,
            "expectancy_net_r": expectancy_net_r,
            "expectancy_net_pct": expectancy_net_pct,
            "profit_factor_net": profit_factor_net,
            "avg_win_net_r": avg_win_r,
            "avg_loss_net_r": avg_loss_r,
            "avg_win_net_pct": avg_win_pct,
            "avg_loss_net_pct": avg_loss_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "final_equity": float(equity_curve[-1]),
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "sharpe_assumptions": f"Annualized using active trade frequency ({n} trades across {days_span:.1f} days)",
            # Holding Times
            "avg_holding_hours": avg_holding_hours,
            "median_holding_hours": median_holding_hours,
            # Frictions
            "total_fees_pct": total_fees,
            "total_slippage_pct": total_slippage,
            "total_funding_pct": total_funding,
            # Gross Secondary Metrics
            "win_rate_gross": win_rate_gross,
            "expectancy_gross_r": expectancy_gross_r,
            "profit_factor_gross": profit_factor_gross,
            # Exit Reason Breakdown
            "exit_counts": df["exit_reason"].value_counts().to_dict(),
            "ambiguous_same_bar_count": int(df["ambiguous_same_bar"].sum())
        }

    @classmethod
    def compute_segmentations(cls, df_trades: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Computes performance breakdowns across required dimensions:
        - Entry Mode
        - Volume Rank Bucket
        - BTC Regime
        """
        if df_trades.empty:
            return {}

        results = {}

        # 1. By Entry Mode
        results["entry_mode"] = {}
        for mode, group in df_trades.groupby("entry_mode"):
            results["entry_mode"][str(mode)] = cls.compute_summary_metrics(group)

        # 2. By Volume Rank Bucket
        results["volume_rank"] = {}
        def map_rank_bucket(r):
            if pd.isna(r) or r is None or r <= 0:
                return "Unranked"
            elif r <= 25:
                return "Rank 1-25"
            elif r <= 50:
                return "Rank 26-50"
            elif r <= 100:
                return "Rank 51-100"
            elif r <= 200:
                return "Rank 101-200"
            else:
                return "Rank 201-300"

        df_rank = df_trades.copy()
        df_rank["rank_bucket"] = df_rank["universe_rank"].apply(map_rank_bucket)
        for bucket, group in df_rank.groupby("rank_bucket"):
            results["volume_rank"][str(bucket)] = cls.compute_summary_metrics(group)

        # 3. By BTC Regime
        results["btc_regime"] = {}
        if "btc_regime" in df_trades.columns:
            for reg, group in df_trades.groupby("btc_regime"):
                results["btc_regime"][str(reg)] = cls.compute_summary_metrics(group)

        return results
