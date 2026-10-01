"""
analysis/parameter_sweep.py
Conducts multi-parameter sweeps, records trial log (results/trials.parquet per Rule 6),
evaluates parameter surfaces/heatmaps, neighbor robustness, and multiple-testing adjustments (DSR).
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    var_trials_sr: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Computes Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to adjust
    for selection bias under multiple testing.
    Returns the p-value probability that true Sharpe > 0 after accounting for M trials.
    """
    if n_trials <= 1 or var_trials_sr <= 1e-9 or sample_length <= 1:
        return 0.5

    euler_mascheroni = 0.5772156649
    # Expected maximum Sharpe under null hypothesis of M independent trials
    sr_std = np.sqrt(var_trials_sr)
    z_m = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials) + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_sr = sr_std * z_m

    # Adjusted variance for non-normality
    denom_var = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr**2)
    if denom_var <= 0:
        denom_var = 1.0

    test_stat = (observed_sr - expected_max_sr) * np.sqrt(sample_length - 1) / np.sqrt(denom_var)
    dsr_prob = float(stats.norm.cdf(test_stat))
    return dsr_prob


class ParameterSweepAnalyzer:
    """
    Analyzes parameter sweep surfaces, neighbor robustness, and multiple testing penalties.
    """

    def __init__(self, trials_path: str = "results/trials.parquet"):
        self.trials_path = Path(trials_path)

    def load_trials(self) -> pd.DataFrame:
        if not self.trials_path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(self.trials_path)

    @staticmethod
    def compute_neighbor_robustness(
        df_trials: pd.DataFrame,
        param_col: str,
        ordered_values: List[Any],
        metric_col: str = "expectancy_net_r"
    ) -> pd.DataFrame:
        """
        Computes neighbor robustness: for each parameter value, computes the mean
        performance of its immediate adjacent neighbors in the parameter ordering.
        Broad plateaus have neighbor performance close to parameter performance;
        isolated peaks show high drop-off between point and neighbors.
        """
        if df_trials.empty or param_col not in df_trials.columns:
            return pd.DataFrame()

        if metric_col not in df_trials.columns:
            if "net_expectancy_r" in df_trials.columns:
                metric_col = "net_expectancy_r"
            elif "expectancy_net_r" in df_trials.columns:
                metric_col = "expectancy_net_r"
            else:
                return pd.DataFrame()

        grouped = df_trials.groupby(param_col)[metric_col].agg(["mean", "std", "count"]).reset_index()
        val_map = {row[param_col]: row["mean"] for _, row in grouped.iterrows()}

        robustness_rows = []
        n_vals = len(ordered_values)
        for idx, val in enumerate(ordered_values):
            if val not in val_map:
                continue
            point_perf = val_map[val]
            neighbors = []
            if idx > 0 and ordered_values[idx - 1] in val_map:
                neighbors.append(val_map[ordered_values[idx - 1]])
            if idx < n_vals - 1 and ordered_values[idx + 1] in val_map:
                neighbors.append(val_map[ordered_values[idx + 1]])

            neighbor_mean = float(np.mean(neighbors)) if neighbors else point_perf
            dropoff = point_perf - neighbor_mean
            is_suspect_peak = (point_perf > 0) and (neighbor_mean < -0.2)

            robustness_rows.append({
                param_col: val,
                "point_mean_net_r": point_perf,
                "neighbor_mean_net_r": neighbor_mean,
                "neighbor_dropoff": dropoff,
                "is_isolated_peak": is_suspect_peak,
                "sample_count": int(grouped.loc[grouped[param_col] == val, "count"].values[0])
            })

        return pd.DataFrame(robustness_rows)

    @staticmethod
    def evaluate_multiple_testing_penalty(df_trials: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates trial count, multiple-testing inflation, and Deflated Sharpe Ratio.
        """
        if df_trials.empty:
            return {"total_trials": 0, "dsr_p_value": 0.0, "status": "No trials"}

        m = len(df_trials)
        metric_col = "expectancy_net_r" if "expectancy_net_r" in df_trials.columns else "net_expectancy_r"
        best_trial = df_trials.loc[df_trials[metric_col].idxmax()]
        best_net_r = float(best_trial[metric_col])
        best_sharpe = float(best_trial.get("sharpe_ratio", 0.0))


        # Variance of Sharpe ratios across all trials
        sharpes = df_trials["sharpe_ratio"].dropna().to_numpy(dtype=float) if "sharpe_ratio" in df_trials.columns else np.array([0.0])
        var_sr = float(np.var(sharpes)) if len(sharpes) > 1 else 0.01

        # Deflated Sharpe
        sample_n = int(best_trial.get("total_trades", 1000))
        dsr = compute_deflated_sharpe_ratio(
            observed_sr=best_sharpe,
            n_trials=m,
            var_trials_sr=var_sr,
            sample_length=sample_n
        )

        return {
            "total_trials_evaluated": m,
            "best_trial_id": best_trial.get("trial_id", "unknown"),
            "best_net_expectancy_r": best_net_r,
            "best_sharpe": best_sharpe,
            "sharpe_variance_across_trials": var_sr,
            "deflated_sharpe_ratio": dsr,
            "statistically_significant_after_trials": (dsr >= 0.95) and (best_net_r > 0.0),
            "multiple_testing_warning": f"Total {m} parameter configurations were tested. Due to selection bias, any unadjusted peak is suspect."
        }
