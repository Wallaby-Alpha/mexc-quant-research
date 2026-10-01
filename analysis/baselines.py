"""
analysis/baselines.py
Computes benchmark baselines to evaluate edge honesty (DEFINITIONS & Phase 3 rules).

Baseline 1: Random 15m bars in a valid 1H uptrend (Trend C), matched on distance-to-high and stop distance.
Baseline 2: Same pullback setups with NO 1H trend filter (trend_c == False).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score 95% confidence interval for proportion k/n.
    Returns (p_hat, ci_lower, ci_upper).
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    
    p = k / n
    z = 1.95996  # 95% confidence
    denom = 1.0 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denom
    spread = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    
    return float(p), float(max(0.0, center - spread)), float(min(1.0, center + spread))


def compute_cluster_bootstrap_ci(
    df: pd.DataFrame,
    success_col: str,
    cluster_col: str = "swing_high_time",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Cluster-robust bootstrap confidence interval (clustered by swing_high or day).
    """
    if len(df) == 0:
        return 0.0, 0.0, 0.0

    p_hat = float(df[success_col].mean())
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    if n_clusters < 5:
        # Fallback to Wilson CI if too few clusters
        k = int(df[success_col].sum())
        return compute_wilson_ci(k, len(df), confidence)

    # Pre-aggregate success sum and total count per cluster
    grouped = df.groupby(cluster_col)[success_col].agg(["sum", "count"]).reset_index()
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()

    rng = np.random.default_rng(random_seed)
    boot_rates = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        b_sum = np.sum(sums[idx])
        b_count = np.sum(counts[idx])
        boot_rates[b] = b_sum / b_count if b_count > 0 else 0.0

    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_rates, alpha * 100))
    ci_upper = float(np.percentile(boot_rates, (1.0 - alpha) * 100))

    return p_hat, ci_lower, ci_upper


def compute_difference_bootstrap_ci(
    treatment_success: np.ndarray,
    baseline_success: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the difference (p_treatment - p_baseline).
    """
    n_t = len(treatment_success)
    n_b = len(baseline_success)
    if n_t == 0 or n_b == 0:
        return 0.0, 0.0, 0.0

    p_t = float(np.mean(treatment_success))
    p_b = float(np.mean(baseline_success))
    diff = p_t - p_b

    rng = np.random.default_rng(random_seed)
    boot_diffs = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample_t = rng.choice(treatment_success, size=min(n_t, 10000), replace=True)
        sample_b = rng.choice(baseline_success, size=min(n_b, 10000), replace=True)
        boot_diffs[b] = np.mean(sample_t) - np.mean(sample_b)

    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    return diff, ci_lower, ci_upper


class BaselineSimulator:
    """
    Simulates Baseline 1 (Random 15m bars in valid 1H uptrend matched on distance-to-high and stop)
    and Baseline 2 (Same setups with NO 1H trend filter).
    """

    def __init__(self, horizon_bars: int = 96, lookback_l: int = 32):
        self.horizon_bars = horizon_bars
        self.lookback_l = lookback_l

    def simulate_baseline_1_symbol(
        self,
        df_15m_aligned: pd.DataFrame,
        atr_15m: pd.Series,
        treatment_dist_high_atr: np.ndarray,
        treatment_dist_stop_atr: np.ndarray,
        sample_size: int = 5000,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Simulates Baseline 1 on a single symbol:
        Picks random bars where trend_c == True.
        Evaluates whether matched distance-to-high is reached before matched distance-to-stop.
        """
        n_bars = len(df_15m_aligned)
        H = self.horizon_bars
        L = self.lookback_l

        if n_bars < (L + H + 1):
            return pd.DataFrame()

        # Find eligible bars: trend_c is True, and at least L bars before and H bars after
        trend_c = df_15m_aligned["trend_c"].to_numpy(dtype=bool) if "trend_c" in df_15m_aligned.columns else np.zeros(n_bars, dtype=bool)
        eligible_bars = np.where(trend_c)[0]
        eligible_bars = eligible_bars[(eligible_bars >= L) & (eligible_bars < n_bars - H - 1)]

        if len(eligible_bars) == 0:
            return pd.DataFrame()

        rng = np.random.default_rng(random_seed)
        chosen_bars = rng.choice(eligible_bars, size=min(len(eligible_bars), sample_size), replace=False)

        high_arr = df_15m_aligned["high"].to_numpy(dtype=float)
        low_arr = df_15m_aligned["low"].to_numpy(dtype=float)
        open_arr = df_15m_aligned["open"].to_numpy(dtype=float)
        atr_arr = atr_15m.to_numpy(dtype=float)

        results = []
        n_treat = len(treatment_dist_high_atr)

        for bar_idx in chosen_bars:
            atr_ref = atr_arr[bar_idx]
            if np.isnan(atr_ref) or atr_ref <= 0:
                continue

            p0 = open_arr[bar_idx + 1]

            # Pick a matched distance from treatment distribution
            t_idx = rng.integers(0, n_treat)
            d_high_atr = treatment_dist_high_atr[t_idx]
            d_stop_atr = treatment_dist_stop_atr[t_idx]

            target = p0 + (d_high_atr * atr_ref)
            stop = p0 - (d_stop_atr * atr_ref)

            # Horizon forward slice
            highs_fwd = high_arr[bar_idx + 1 : bar_idx + 1 + H]
            lows_fwd = low_arr[bar_idx + 1 : bar_idx + 1 + H]

            hit_target = highs_fwd >= target
            hit_stop = lows_fwd <= stop

            idx_target = int(np.argmax(hit_target)) + 1 if np.any(hit_target) else 9999
            idx_stop = int(np.argmax(hit_stop)) + 1 if np.any(hit_stop) else 9999

            is_retest = idx_target <= len(highs_fwd)
            is_stop = idx_stop <= len(highs_fwd)
            retest_before_pess = is_retest and (idx_target < idx_stop)
            retest_before_opt = is_retest and (idx_target <= idx_stop)

            results.append({
                "bar_idx": bar_idx,
                "retest_primary": is_retest,
                "failure": is_stop,
                "retest_before_failure_pessimistic": retest_before_pess,
                "retest_before_failure_optimistic": retest_before_opt,
                "ambiguous_same_bar": (idx_target == idx_stop) and is_retest
            })

        return pd.DataFrame(results)
