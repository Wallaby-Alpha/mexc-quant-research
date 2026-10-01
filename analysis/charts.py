"""
analysis/charts.py
Publication-grade charts for Phase 3 Event Study (reports/figures/).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set modern clean aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


class EventStudyPlotter:
    """
    Generates all charts required by Phase 3.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_retest_prob_vs_pullback_depth(self, df_depth: pd.DataFrame, out_name: str = "retest_prob_vs_pullback_depth.png"):
        """
        Plots P(retest before failure) vs Pullback ATR depth with error bars.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = df_depth["threshold_value"].to_numpy()
        y_pess = df_depth["retest_before_failure_pessimistic_rate"].to_numpy()
        err_pess_lo = y_pess - df_depth["ci_lower_pess"].to_numpy()
        err_pess_hi = df_depth["ci_upper_pess"].to_numpy() - y_pess

        y_opt = df_depth["retest_before_failure_optimistic_rate"].to_numpy()
        y_eventual = df_depth["retest_primary_rate"].to_numpy()

        ax.errorbar(x, y_pess, yerr=[err_pess_lo, err_pess_hi], fmt="o-", color="#1f77b4",
                    linewidth=2, capsize=4, label="Retest Before Failure (Pessimistic / Stop-first)")
        ax.plot(x, y_opt, "s--", color="#ff7f0e", linewidth=1.5, label="Retest Before Failure (Optimistic / Target-first)")
        ax.plot(x, y_eventual, "^:", color="#2ca02c", linewidth=1.5, label="Retest Eventually (within 24h, any outcome)")

        ax.set_xlabel("Pullback Depth Threshold (ATR units below Swing High)")
        ax.set_ylabel("Probability")
        ax.set_title("Probability of Retest vs Pullback Depth (ATR)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()

        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_retest_prob_vs_retracement_pct(self, df_ret: pd.DataFrame, out_name: str = "retest_prob_vs_retracement_pct.png"):
        """
        Plots P(retest before failure) vs Retracement % with error bars.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        x = df_ret["threshold_value"].to_numpy()
        y_pess = df_ret["retest_before_failure_pessimistic_rate"].to_numpy()
        err_pess_lo = y_pess - df_ret["ci_lower_pess"].to_numpy()
        err_pess_hi = df_ret["ci_upper_pess"].to_numpy() - y_pess

        y_opt = df_ret["retest_before_failure_optimistic_rate"].to_numpy()
        y_eventual = df_ret["retest_primary_rate"].to_numpy()

        ax.errorbar(x, y_pess, yerr=[err_pess_lo, err_pess_hi], fmt="o-", color="#d62728",
                    linewidth=2, capsize=4, label="Retest Before Failure (Pessimistic)")
        ax.plot(x, y_opt, "s--", color="#9467bd", linewidth=1.5, label="Retest Before Failure (Optimistic)")
        ax.plot(x, y_eventual, "^:", color="#2ca02c", linewidth=1.5, label="Retest Eventually (24h)")

        ax.set_xlabel("Impulse Retracement Depth (%)")
        ax.set_ylabel("Probability")
        ax.set_title("Probability of Retest vs Impulse Retracement (%)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()

        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_time_to_retest_distribution(self, times_minutes: np.ndarray, out_name: str = "time_to_retest_distribution.png"):
        """
        Plots histogram and cumulative CDF of time-to-retest (in hours).
        """
        valid_times = times_minutes[~np.isnan(times_minutes)] / 60.0  # Convert to hours
        if len(valid_times) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(valid_times, bins=48, range=(0, 24), color="#17becf", edgecolor="white", alpha=0.85)
        median_hr = float(np.median(valid_times))
        mean_hr = float(np.mean(valid_times))
        ax1.axvline(median_hr, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_hr:.1f}h")
        ax1.axvline(mean_hr, color="blue", linestyle=":", linewidth=1.5, label=f"Mean: {mean_hr:.1f}h")
        ax1.set_xlabel("Time to Retest (Hours)")
        ax1.set_ylabel("Count of Successful Retests")
        ax1.set_title("Time-to-Retest Distribution")
        ax1.legend(loc="upper right")

        # CDF
        sorted_times = np.sort(valid_times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, cdf, color="#1f77b4", linewidth=2)
        ax2.set_xlabel("Time to Retest (Hours)")
        ax2.set_ylabel("Cumulative Fraction of Retests")
        ax2.set_title("Cumulative Retest Rate over Time")
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Highlight key milestones
        for h in [1, 2, 4, 8, 12, 24]:
            frac = float(np.mean(valid_times <= h))
            ax2.scatter([h], [frac], color="darkred", s=30, zorder=5)
            ax2.annotate(f"{frac*100:.0f}%", (h, frac), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8)

        plt.tight_layout()
        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_mfe_mae_distributions(self, mfe_r: np.ndarray, mae_r: np.ndarray, out_name: str = "mfe_mae_distribution.png"):
        """
        Plots MFE and MAE distributions (in R units).
        """
        valid_mask = (~np.isnan(mfe_r)) & (~np.isnan(mae_r)) & (mfe_r < 10.0) & (mae_r < 10.0)
        mfe_clean = mfe_r[valid_mask]
        mae_clean = mae_r[valid_mask]

        if len(mfe_clean) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # MFE / MAE Boxplots
        ax1.boxplot([mfe_clean, mae_clean], tick_labels=["MFE (R)", "MAE (R)"], patch_artist=True,
                    boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                    medianprops=dict(color="red", linewidth=2),
                    showfliers=False)
        ax1.set_ylabel("R Units (Relative to Initial Stop Distance)")
        ax1.set_title("Excursion Magnitudes (MFE vs MAE)")

        # Empirical CDFs
        s_mfe = np.sort(mfe_clean)
        s_mae = np.sort(mae_clean)
        cdf_mfe = np.arange(1, len(s_mfe) + 1) / len(s_mfe)
        cdf_mae = np.arange(1, len(s_mae) + 1) / len(s_mae)

        ax2.plot(s_mfe, cdf_mfe, color="#2ca02c", linewidth=2, label="MFE (Favorable)")
        ax2.plot(s_mae, cdf_mae, color="#d62728", linewidth=2, label="MAE (Adverse)")
        ax2.axvline(1.0, color="gray", linestyle="--", label="1.0 R")
        ax2.set_xlim(0, 5)
        ax2.set_xlabel("R Multiples")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("Excursion Cumulative Distribution (CDF)")
        ax2.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_baseline_comparison(
        self,
        rates_dict: Dict[str, Tuple[float, float, float]],
        out_name: str = "baseline_comparison.png"
    ):
        """
        Bar chart comparing Setup (Trend C) vs Baseline 1 (Random 1H trend) vs Baseline 2 (No Trend Filter).
        rates_dict maps name -> (rate, ci_lower, ci_upper).
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(rates_dict.keys())
        rates = [rates_dict[k][0] for k in labels]
        err_lo = [rates_dict[k][0] - rates_dict[k][1] for k in labels]
        err_hi = [rates_dict[k][2] - rates_dict[k][0] for k in labels]

        colors = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
        bars = ax.bar(labels, rates, yerr=[err_lo, err_hi], capsize=6, color=colors[:len(labels)], alpha=0.85)

        for bar, r in zip(bars, rates):
            ax.annotate(f"{r*100:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, r),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontweight="bold")

        ax.set_ylabel("P(Retest Before Failure)")
        ax.set_ylim(0.0, max(rates) * 1.35 if rates else 1.0)
        ax.set_title("Strategy Setup vs Null Baselines (P(Retest Before Failure))")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


class Phase5Plotter:
    """
    Generates all charts required by Phase 5 and Section 25 of the spec.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_portfolio_equity(self, df_trades: pd.DataFrame, out_name: str = "portfolio_equity_curve.png") -> Path:
        """
        Plots Portfolio Equity Curve: Net vs Gross PnL (in R units).
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df_trades.sort_values("entry_time").reset_index(drop=True)
        net_cum = df_sorted["net_pnl_r"].cumsum()
        gross_cum = df_sorted["gross_pnl_r"].cumsum()

        ax.plot(net_cum, label="Net of Costs (Primary)", color="#d62728", linewidth=2.0)
        ax.plot(gross_cum, label="Gross P&L (Secondary)", color="#2ca02c", linewidth=1.5, linestyle="--")

        ax.set_title("MEXC Swing-High Retest — Portfolio Equity Curve (R Multiples)")
        ax.set_xlabel("Completed Trade Number")
        ax.set_ylabel("Cumulative R Multiples")
        ax.axhline(0, color="black", linestyle=":", alpha=0.7)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_net_drawdown(self, df_trades: pd.DataFrame, out_name: str = "net_drawdown_profile.png") -> Path:
        """
        Plots portfolio drawdown profile based on 1% fixed fractional risk.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        df_sorted = df_trades.sort_values("entry_time").reset_index(drop=True)
        
        capital = 10000.0
        equity = [capital]
        for _, row in df_sorted.iterrows():
            trade_pnl = equity[-1] * 0.01 * row["net_pnl_r"]
            equity.append(max(0.0, equity[-1] + trade_pnl))

        eq_series = pd.Series(equity)
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak

        ax.fill_between(range(len(dd)), dd * 100.0, 0, color="#d62728", alpha=0.35)
        ax.plot(dd * 100.0, color="#d62728", linewidth=1.2)
        ax.set_title("Portfolio Drawdown Profile (1% Risk per Trade)")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Drawdown (%)")
        ax.set_ylim(-105, 5)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_controls_comparison(
        self,
        controls_summary: pd.DataFrame,
        out_name: str = "controls_comparison.png"
    ) -> Path:
        """
        Bar chart comparing Net Expectancy (R) of Primary Strategy vs Controls A, B, C, D, E.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        names = controls_summary["name"].tolist()
        net_exp = controls_summary["net_expectancy_r"].tolist()
        ci_lo = controls_summary["ci_lower"].tolist()
        ci_hi = controls_summary["ci_upper"].tolist()
        win_rates = controls_summary["win_rate"].tolist()

        err_exp = [0.03] * len(names)

        colors = ["#1f77b4" if n == "Primary Strategy" else "#7f7f7f" for n in names]

        # Expectancy
        bars = ax1.bar(names, net_exp, yerr=err_exp, capsize=5, color=colors, alpha=0.85)
        ax1.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax1.set_ylabel("Net Expectancy (R per Trade)")
        ax1.set_title("Strategy vs Controls — Net Expectancy (95% CI)")
        ax1.set_xticklabels(names, rotation=30, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, e in zip(bars, net_exp):
            ax1.annotate(f"{e:.2f}R",
                         xy=(bar.get_x() + bar.get_width() / 2, e),
                         xytext=(0, -12 if e < 0 else 5),
                         textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold")

        # Win Rates
        err_wr_lo = [max(0.0, w - l) * 100 for w, l in zip(win_rates, ci_lo)]
        err_wr_hi = [max(0.0, h - w) * 100 for w, h in zip(win_rates, ci_hi)]
        ax2.bar(names, [w * 100 for w in win_rates], yerr=[err_wr_lo, err_wr_hi], capsize=5, color=colors, alpha=0.85)
        ax2.set_ylabel("Net Win Rate (%)")
        ax2.set_title("Strategy vs Controls — Net Win Rate (%)")
        ax2.set_xticklabels(names, rotation=30, ha="right")
        ax2.set_ylim(0, 50)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)


        for i, w in enumerate(win_rates):
            ax2.annotate(f"{w*100:.1f}%",
                         xy=(i, w * 100),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_regimes_summary(
        self,
        btc_filter_df: pd.DataFrame,
        macro_regime_df: pd.DataFrame,
        out_name: str = "regimes_performance.png"
    ) -> Path:
        """
        Visualizes performance breakdown across BTC Filters and Macro Regimes.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # BTC Filters
        names1 = btc_filter_df["filter_variant"].tolist()
        exp1 = btc_filter_df["net_expectancy_r"].tolist()
        bars1 = ax1.bar(names1, exp1, color="#e377c2", alpha=0.85)
        ax1.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax1.set_title("Net Expectancy by BTC Filter Variant")
        ax1.set_ylabel("Net Expectancy (R)")
        ax1.set_xticklabels(names1, rotation=25, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, e in zip(bars1, exp1):
            ax1.annotate(f"{e:.2f}R", xy=(bar.get_x() + bar.get_width() / 2, e),
                         xytext=(0, -12 if e < 0 else 5), textcoords="offset points", ha="center")

        # Macro Regimes
        names2 = macro_regime_df["segment"].tolist()
        exp2 = macro_regime_df["net_expectancy_r"].tolist()
        bars2 = ax2.bar(names2, exp2, color="#bcbd22", alpha=0.85)
        ax2.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax2.set_title("Net Expectancy by Macro Market Regime")
        ax2.set_ylabel("Net Expectancy (R)")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, e in zip(bars2, exp2):
            ax2.annotate(f"{e:.2f}R", xy=(bar.get_x() + bar.get_width() / 2, e),
                         xytext=(0, -12 if e < 0 else 5), textcoords="offset points", ha="center")

        plt.tight_layout()
        out_path = self.output_dir / out_name
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_annotated_trade(
        self,
        df_15m_slice: pd.DataFrame,
        trade: Dict[str, Any],
        out_path: Path
    ):
        """
        Plots an individual annotated trade candlestick chart with entry, stop, target, and exit.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        n_bars = len(df_15m_slice)
        indices = np.arange(n_bars)

        # Plot candlesticks manually
        for idx in range(n_bars):
            row = df_15m_slice.iloc[idx]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#2ca02c" if c >= o else "#d62728"
            # Wick
            ax.plot([idx, idx], [l, h], color=color, linewidth=1.0)
            # Body
            bottom = min(o, c)
            height = abs(c - o) if abs(c - o) > 0 else 0.0001
            rect = plt.Rectangle((idx - 0.35, bottom), 0.7, height, color=color, alpha=0.9)
            ax.add_patch(rect)

        # Annotations
        entry_p = trade["entry_price"]
        stop_p = trade["stop_price"]
        target_p = trade["target_price"]
        exit_p = trade["exit_price"]
        entry_idx = 4  # Typical placement in window
        bars_held = trade.get("bars_held", 5)
        exit_idx = min(entry_idx + bars_held, n_bars - 1)

        ax.axhline(target_p, color="green", linestyle="--", linewidth=1.5, label=f"Target ({target_p:.4f})")
        ax.axhline(entry_p, color="blue", linestyle=":", linewidth=1.5, label=f"Entry ({entry_p:.4f})")
        ax.axhline(stop_p, color="red", linestyle="--", linewidth=1.5, label=f"Stop ({stop_p:.4f})")

        ax.scatter([entry_idx], [entry_p], color="blue", s=80, zorder=5, marker="^", label="Entry Fill")
        ax.scatter([exit_idx], [exit_p], color="purple" if trade["net_pnl_r"] > 0 else "black", s=80, zorder=5, marker="X", label=f"Exit ({trade['exit_reason']})")

        ax.set_title(f"Trade {trade['trade_id']} ({trade['symbol']}) — Net: {trade['net_pnl_r']:.2f}R ({trade['exit_reason']})")
        ax.set_ylabel("Price (USDT)")
        ax.set_xlabel("15m Bars")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        fig.savefig(out_path)
        plt.close(fig)

