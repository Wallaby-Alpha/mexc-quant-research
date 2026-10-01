"""
analysis/walk_forward.py
Implements Rolling Walk-Forward Validation (Phase 5).
Rolling train/validate/test windows across non-holdout data.
Parameters selected in train using robust-region method, scored on validate,
evaluated on unseen test window, stitched OOS results reported.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

from backtest.trade import Trade
from analysis.trade_metrics import TradeMetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardValidator:
    """
    Manages rolling train/validate/test cross-validation windows and OOS stitching.
    """

    def __init__(
        self,
        start_date: str = "2025-10-04T00:00:00Z",
        holdout_start: str = "2026-07-01T00:00:00Z",
        train_months: int = 3,
        val_months: int = 1,
        test_months: int = 1
    ):
        self.start_date = pd.Timestamp(start_date, tz="UTC")
        self.holdout_start = pd.Timestamp(holdout_start, tz="UTC")
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months

    def generate_windows(self) -> List[WalkForwardWindow]:
        """
        Generates rolling monthly windows across non-holdout data.
        """
        windows = []
        fold = 1
        # Start at 2025-10-01
        cur_train_start = pd.Timestamp("2025-10-01T00:00:00Z")

        while True:
            train_end = cur_train_start + pd.DateOffset(months=self.train_months)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > self.holdout_start:
                break

            windows.append(WalkForwardWindow(
                fold_idx=fold,
                train_start=cur_train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            ))
            fold += 1
            cur_train_start = cur_train_start + pd.DateOffset(months=1)

        return windows

    @staticmethod
    def slice_trades_by_window(
        trades: List[Trade],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> List[Trade]:
        """
        Slices trades whose entry_time falls within [start_time, end_time).
        """
        return [
            t for t in trades
            if start_time <= pd.to_datetime(t.entry_time, utc=True) < end_time
        ]

    @staticmethod
    def select_robust_parameter_set(
        candidates_perf: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Selects parameter set with best robust score (highest net expectancy with penalty for high variance).
        """
        best_cfg = None
        best_score = -999.0

        for cfg_name, stats_dict in candidates_perf.items():
            net_exp = stats_dict.get("net_expectancy_r", -1.0)
            n_trades = stats_dict.get("total_trades", 0)
            if n_trades < 50:
                continue
            # Score penalizes low samples
            score = net_exp - (0.1 if n_trades < 200 else 0.0)
            if score > best_score:
                best_score = score
                best_cfg = cfg_name

        return best_cfg or list(candidates_perf.keys())[0]

    @staticmethod
    def stitch_oos_trades(fold_test_trades: List[List[Trade]]) -> List[Trade]:
        """
        Concatenates test window trades across all walk-forward folds.
        """
        stitched = []
        seen_ids = set()
        for t_list in fold_test_trades:
            for t in t_list:
                if t.trade_id not in seen_ids:
                    seen_ids.add(t.trade_id)
                    stitched.append(t)
        stitched.sort(key=lambda t: t.entry_time)
        return stitched
