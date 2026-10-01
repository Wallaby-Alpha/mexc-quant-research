"""
analysis/event_study.py
Master Phase 3 Event Study Engine.
Computes outcomes, statistical confidence intervals (Wilson & Cluster Bootstrap),
baseline comparisons, figures, setup viewer, and summary report.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa

from data.config import AppConfig, load_config
from data.cache import ParquetCache
from strategy.btc_regime import BTCRegimeDetector
from analysis.outcomes import EventOutcomeCalculator
from analysis.baselines import (
    compute_wilson_ci,
    compute_cluster_bootstrap_ci,
    compute_difference_bootstrap_ci,
    BaselineSimulator
)
from analysis.charts import EventStudyPlotter
from reports.setup_viewer import SetupViewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EventStudyEngine:
    """
    Executes the Phase 3 Event Study on non-holdout data.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.cache = ParquetCache(config.data.cache_dir)
        self.btc_detector = BTCRegimeDetector(self.cache)
        self.outcome_calc = EventOutcomeCalculator(
            horizon_bars=96,
            holdout_start=config.data.holdout_start
        )
        self.baseline_sim = BaselineSimulator(horizon_bars=96)
        self.plotter = EventStudyPlotter(output_dir=Path(config.data.reports_dir) / "figures")
        self.viewer = SetupViewer(output_html_path=Path(config.data.reports_dir) / "setup_viewer.html")
        self.results_dir = Path(config.data.results_dir)
        self.reports_dir = Path(config.data.reports_dir)

    def load_non_holdout_setups(self, fractal_n: Optional[int] = None) -> pd.DataFrame:
        """
        Loads non-holdout setup events (strictly event_time < holdout_start).
        """
        holdout_ts = pd.Timestamp(self.config.data.holdout_start, tz="UTC")
        setups_path = self.results_dir / "setups.parquet"
        
        logger.info(f"Loading non-holdout setups from {setups_path} (< {holdout_ts})...")
        import pyarrow.compute as pc
        cutoff = pa.scalar(holdout_ts, type=pa.timestamp("ms", tz="UTC"))
        
        filter_expr = pc.field("event_time") < cutoff
        if fractal_n is not None:
            filter_expr = filter_expr & (pc.field("fractal_n") == fractal_n)

        d = ds.dataset(setups_path)
        table = d.to_table(filter=filter_expr)
        df = table.to_pandas()
        logger.info(f"Loaded {len(df):,} non-holdout setup events.")
        return df

    def attach_btc_regime(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches BTC regime features causally based on event_time.
        """
        logger.info("Computing and attaching BTC 1H regime series...")
        btc_1h = self.btc_detector.compute_btc_1h_regime()
        if btc_1h.empty:
            df_out = df_events.copy()
            for col in ["btc_trend_bull", "btc_above_ema50", "btc_above_ema200"]:
                df_out[col] = False
            return df_out

        btc_regime = btc_1h.sort_values(by="open_time").copy()
        btc_regime["open_time"] = pd.to_datetime(btc_regime["open_time"], utc=True).astype("datetime64[ms, UTC]")
        
        df_sorted = df_events.sort_values(by="event_time").copy()
        df_sorted["event_time"] = pd.to_datetime(df_sorted["event_time"], utc=True).astype("datetime64[ms, UTC]")

        # Merge backwards so event only sees closed BTC 1H bars
        merged = pd.merge_asof(
            df_sorted,
            btc_regime[["open_time", "btc_trend_bull", "btc_above_ema50", "btc_above_ema200"]],
            left_on="event_time",
            right_on="open_time",
            direction="backward"
        )
        if "open_time_y" in merged.columns:
            merged = merged.drop(columns=["open_time_y"])
        return merged

    def compute_all_outcomes(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Computes forward trajectories and outcomes across all symbols.
        """
        symbols = df_events["symbol"].unique()
        logger.info(f"Computing outcomes for {len(df_events):,} events across {len(symbols)} symbols...")

        outcomes_list = []
        for idx, sym in enumerate(symbols):
            df_sym_events = df_events[df_events["symbol"] == sym].copy()
            df_15m = self.cache.load_klines(sym, "15m")
            if df_15m.empty:
                continue

            sym_outcomes = self.outcome_calc.compute_symbol_outcomes(
                events_df=df_sym_events,
                df_15m=df_15m,
                allow_holdout=False
            )
            if not sym_outcomes.empty:
                outcomes_list.append(sym_outcomes)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(symbols):
                logger.info(f"  Processed {idx + 1}/{len(symbols)} symbols...")

        if not outcomes_list:
            return pd.DataFrame()

        df_all_outcomes = pd.concat(outcomes_list, ignore_index=True)
        out_path = self.results_dir / "event_outcomes.parquet"
        df_all_outcomes.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df_all_outcomes):,} evaluated outcomes to {out_path}.")
        return df_all_outcomes

    def compute_metric_table(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        success_col: str = "retest_before_failure_pessimistic"
    ) -> pd.DataFrame:
        """
        Computes n, rate, 95% Wilson CI, and cluster-robust bootstrap CI for each group.
        Flags buckets with n < 100 independent setups.
        """
        records = []
        for name, group in df.groupby(group_cols):
            n = len(group)
            k = int(group[success_col].sum())
            p_hat, wilson_lo, wilson_hi = compute_wilson_ci(k, n)

            # Independent setups count (unique swing highs)
            n_independent = group["swing_high_time"].nunique()
            is_insufficient = n_independent < 100

            # Cluster bootstrap by swing_high_time
            _, boot_lo, boot_hi = compute_cluster_bootstrap_ci(
                group, success_col=success_col, cluster_col="swing_high_time", n_bootstrap=500
            )

            rec = {}
            if isinstance(name, tuple):
                for col_name, val in zip(group_cols, name):
                    rec[col_name] = val
            else:
                rec[group_cols[0]] = name

            rec["n_events"] = n
            rec["n_independent_setups"] = n_independent
            rec["insufficient_sample"] = is_insufficient
            rec["success_count"] = k
            rec["rate"] = p_hat
            rec["wilson_ci_lower"] = wilson_lo
            rec["wilson_ci_upper"] = wilson_hi
            rec["cluster_boot_ci_lower"] = boot_lo
            rec["cluster_boot_ci_upper"] = boot_hi

            # Optimistic and eventual rates
            if "retest_before_failure_optimistic" in group.columns:
                rec["rate_optimistic"] = float(group["retest_before_failure_optimistic"].mean())
            if "retest_primary" in group.columns:
                rec["rate_eventual_24h"] = float(group["retest_primary"].mean())

            records.append(rec)

        return pd.DataFrame(records)

    def run_baselines(self, df_controlled_c: pd.DataFrame, df_all_setups: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes Baseline 1 (Random 15m in 1H uptrend) and Baseline 2 (Same setups with no 1H trend).
        """
        logger.info("Computing Baseline 1 (Random bars in 1H uptrend) and Baseline 2 (No trend filter)...")

        # Baseline 2: Same setups where trend_c == False
        df_base_2 = df_all_setups[~df_all_setups["trend_c"]].copy()
        b2_success = df_base_2["retest_before_failure_pessimistic"].to_numpy(dtype=float)
        t_success = df_controlled_c["retest_before_failure_pessimistic"].to_numpy(dtype=float)

        p_t = float(np.mean(t_success))
        p_b2 = float(np.mean(b2_success))
        diff_b2, diff_b2_lo, diff_b2_hi = compute_difference_bootstrap_ci(t_success, b2_success, n_bootstrap=1000)

        # Baseline 1: Simulate matched random bars in Trend C across top symbols
        sample_symbols = df_controlled_c["symbol"].value_counts().head(20).index.tolist()
        b1_results = []
        d_high_dist = df_controlled_c["distance_to_high_atr"].dropna().to_numpy()
        d_stop_dist = df_controlled_c["distance_to_stop_atr"].dropna().to_numpy()

        for sym in sample_symbols:
            df_15m = self.cache.load_klines(sym, "15m")
            df_1h = self.cache.load_klines(sym, "1h")
            if df_15m.empty or df_1h.empty:
                continue

            from strategy.trend_detector import TrendDetector
            from strategy.alignment import align_1h_to_15m
            from strategy.indicators import compute_wilder_atr

            td = TrendDetector()
            df_1h_trends = td.compute_1h_trends(df_1h)
            df_aligned = align_1h_to_15m(df_15m, df_1h_trends, ["trend_c"])
            atr_15m = compute_wilder_atr(df_aligned["high"], df_aligned["low"], df_aligned["close"], period=14)

            # Simulate
            df_b1_sym = self.baseline_sim.simulate_baseline_1_symbol(
                df_15m_aligned=df_aligned,
                atr_15m=atr_15m,
                treatment_dist_high_atr=d_high_dist,
                treatment_dist_stop_atr=d_stop_dist,
                sample_size=1000
            )
            if not df_b1_sym.empty:
                b1_results.append(df_b1_sym)

        df_b1 = pd.concat(b1_results, ignore_index=True) if b1_results else pd.DataFrame()
        b1_success = df_b1["retest_before_failure_pessimistic"].to_numpy(dtype=float) if not df_b1.empty else np.array([])
        p_b1 = float(np.mean(b1_success)) if len(b1_success) > 0 else 0.0
        diff_b1, diff_b1_lo, diff_b1_hi = compute_difference_bootstrap_ci(t_success, b1_success, n_bootstrap=1000) if len(b1_success) > 0 else (0,0,0)

        return {
            "p_treatment": p_t,
            "p_baseline_1": p_b1,
            "diff_baseline_1": diff_b1,
            "diff_b1_ci": (diff_b1_lo, diff_b1_hi),
            "p_baseline_2": p_b2,
            "diff_baseline_2": diff_b2,
            "diff_b2_ci": (diff_b2_lo, diff_b2_hi)
        }
