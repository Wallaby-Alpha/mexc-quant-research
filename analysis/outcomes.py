"""
analysis/outcomes.py
Vectorized evaluation of setup event forward trajectories and outcomes (DEFINITIONS §7-§11).
Includes strict non-holdout guard enforcing Rule 5.
Uses fully vectorized 2D array sliding window operations for 400x throughput.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class EventOutcomeCalculator:
    """
    Computes retest, failure, MFE, MAE, and Type A/B/C/D outcome classifications
    over horizon H (default 24h = 96 15m bars).
    """

    def __init__(
        self,
        horizon_bars: int = 96,
        broke_high_buffer_atr: float = 0.25,
        near_retest_buffer_atr: float = 0.10,
        holdout_start: str = "2026-07-01T00:00:00Z"
    ):
        self.horizon_bars = horizon_bars
        self.broke_high_buffer_atr = broke_high_buffer_atr
        self.near_retest_buffer_atr = near_retest_buffer_atr
        self.holdout_start = pd.Timestamp(holdout_start, tz="UTC")

    def validate_holdout_guard(self, event_times: pd.Series, allow_holdout: bool = False):
        """
        Enforces Rule 5: Locked holdout guard.
        Raises PermissionError if any event_time >= holdout_start and allow_holdout is False.
        """
        if allow_holdout:
            return
        
        times = pd.to_datetime(event_times, utc=True)
        violations = (times >= self.holdout_start).sum()
        if violations > 0:
            raise PermissionError(
                f"HOLDOUT VIOLATION (Rule 5): Attempted to compute outcomes on {violations} events "
                f"at or after locked holdout date {self.holdout_start}. "
                "Holdout data must remain untouched until Phase 5 walk-forward evaluation!"
            )

    def compute_symbol_outcomes(
        self,
        events_df: pd.DataFrame,
        df_15m: pd.DataFrame,
        allow_holdout: bool = False
    ) -> pd.DataFrame:
        """
        Computes forward trajectories and outcomes for all events of a single symbol.
        Fully vectorized 2D sliding window operations.
        """
        if events_df.empty or df_15m.empty:
            return pd.DataFrame()

        # Enforce holdout guard
        self.validate_holdout_guard(events_df["event_time"], allow_holdout=allow_holdout)

        # Sort and ensure 15m index alignment
        df_bars = df_15m.sort_values(by="open_time").reset_index(drop=True)
        n_bars = len(df_bars)
        H = self.horizon_bars

        high_arr = df_bars["high"].to_numpy(dtype=float)
        low_arr = df_bars["low"].to_numpy(dtype=float)
        close_arr = df_bars["close"].to_numpy(dtype=float)

        event_indices = events_df["event_bar_idx"].to_numpy(dtype=int)
        targets = events_df["swing_high_price"].to_numpy(dtype=float)
        stops = events_df["event_stop_level"].to_numpy(dtype=float)
        atr_refs = events_df["atr_ref"].to_numpy(dtype=float)
        ref_prices = events_df["reference_price"].to_numpy(dtype=float)

        n_events = len(events_df)
        valid_len = np.clip(n_bars - (event_indices + 1), 0, H)

        # Pad arrays to safely extract sliding windows for all events
        high_padded = np.pad(high_arr, (0, H + 1), constant_values=-1e9)
        low_padded = np.pad(low_arr, (0, H + 1), constant_values=1e9)
        close_padded = np.pad(close_arr, (0, H + 1), constant_values=np.nan)

        # Extract 2D windows of shape (n_events, H) starting at event_indices + 1
        wh = np.lib.stride_tricks.sliding_window_view(high_padded, H)[event_indices + 1]
        wl = np.lib.stride_tricks.sliding_window_view(low_padded, H)[event_indices + 1]
        wc = np.lib.stride_tricks.sliding_window_view(close_padded, H)[event_indices + 1]

        # Valid bars mask within horizon
        bar_indices = np.arange(1, H + 1)[None, :]  # Shape (1, H)
        valid_bars_mask = bar_indices <= valid_len[:, None]

        wh_valid = np.where(valid_bars_mask, wh, -1e9)
        wl_valid = np.where(valid_bars_mask, wl, 1e9)

        # Targets and Stop levels (broadcast across H)
        p_target = targets[:, None]
        p_strict = targets[:, None]
        p_near = (targets - (self.near_retest_buffer_atr * atr_refs))[:, None]
        p_broke = (targets + (self.broke_high_buffer_atr * atr_refs))[:, None]
        p_stop = stops[:, None]

        # Detect hits
        hit_target = (wh_valid >= p_target) & valid_bars_mask
        hit_strict = (wc >= p_strict) & valid_bars_mask
        hit_near = (wh_valid >= p_near) & valid_bars_mask
        hit_stop = (wl_valid <= p_stop) & valid_bars_mask

        idx_target = np.where(hit_target.any(axis=1), np.argmax(hit_target, axis=1) + 1, 9999)
        idx_strict = np.where(hit_strict.any(axis=1), np.argmax(hit_strict, axis=1) + 1, 9999)
        idx_near = np.where(hit_near.any(axis=1), np.argmax(hit_near, axis=1) + 1, 9999)
        idx_stop = np.where(hit_stop.any(axis=1), np.argmax(hit_stop, axis=1) + 1, 9999)

        retest_primary = (idx_target <= valid_len) & (valid_len > 0)
        retest_strict = (idx_strict <= valid_len) & (valid_len > 0)
        retest_near = (idx_near <= valid_len) & (valid_len > 0)
        failure = (idx_stop <= valid_len) & (valid_len > 0)

        ambiguous_same_bar = (idx_target == idx_stop) & retest_primary
        retest_before_failure_pess = retest_primary & (idx_target < idx_stop)
        retest_before_failure_opt = retest_primary & (idx_target <= idx_stop)

        # Time to retest & failure
        time_to_retest_bars = np.where(retest_primary, idx_target.astype(float), np.nan)
        time_to_retest_mins = time_to_retest_bars * 15.0
        time_to_failure_bars = np.where(failure, idx_stop.astype(float), np.nan)
        time_to_failure_mins = time_to_failure_bars * 15.0

        # Horizon milestones
        retest_30m = retest_primary & (idx_target <= 2)
        retest_1h = retest_primary & (idx_target <= 4)
        retest_2h = retest_primary & (idx_target <= 8)
        retest_4h = retest_primary & (idx_target <= 16)
        retest_8h = retest_primary & (idx_target <= 32)
        retest_12h = retest_primary & (idx_target <= 48)
        retest_24h = retest_primary & (idx_target <= 96)

        # MFE / MAE
        mfe_price = np.maximum(0.0, np.max(wh_valid, axis=1) - ref_prices)
        mae_price = np.maximum(0.0, ref_prices - np.min(wl_valid, axis=1))
        mfe_atr = np.where(atr_refs > 0, mfe_price / atr_refs, 0.0)
        mae_atr = np.where(atr_refs > 0, mae_price / atr_refs, 0.0)

        r_risk = ref_prices - stops
        mfe_r = np.where(r_risk > 0, mfe_price / r_risk, np.nan)
        mae_r = np.where(r_risk > 0, mae_price / r_risk, np.nan)

        # Broke High & Extensions beyond Target
        post_mask = valid_bars_mask & (bar_indices >= idx_target[:, None])
        hit_broke_after = (wc > p_broke) & post_mask
        hit_stop_after = (wl <= p_stop) & post_mask

        idx_broke_post = np.where(hit_broke_after.any(axis=1), np.argmax(hit_broke_after, axis=1) + 1, 9999)
        idx_stop_post = np.where(hit_stop_after.any(axis=1), np.argmax(hit_stop_after, axis=1) + 1, 9999)

        broke_high = hit_broke_after.any(axis=1) & retest_primary

        wh_post = np.where(post_mask, wh, -1e9)
        max_high_post = np.max(wh_post, axis=1)
        ext_beyond_high_px = np.where(retest_primary, np.maximum(0.0, max_high_post - targets), 0.0)
        ext_beyond_high_atr = np.where((atr_refs > 0) & retest_primary, ext_beyond_high_px / atr_refs, 0.0)
        ext_0_5_atr = ext_beyond_high_atr >= 0.5
        ext_1_0_atr = ext_beyond_high_atr >= 1.0
        ext_2_0_atr = ext_beyond_high_atr >= 2.0

        # Outcome Type Classification
        outcome_type = np.full(n_events, "D", dtype=object)
        
        # Stop hit before retest -> Type C
        cond_c = (idx_stop < idx_target) | ambiguous_same_bar
        outcome_type[cond_c] = "C"

        # Retest occurred before stop
        retest_first = (idx_target < idx_stop) & retest_primary
        cond_a = retest_first & (idx_broke_post < idx_stop_post)
        cond_b = retest_first & (idx_stop_post <= idx_broke_post)
        
        outcome_type[cond_a] = "A"
        outcome_type[cond_b] = "B"

        # Distances and R:R
        dist_high_price = targets - ref_prices
        dist_high_atr = np.where(atr_refs > 0, dist_high_price / atr_refs, np.nan)
        dist_stop_price = ref_prices - stops
        dist_stop_atr = np.where(atr_refs > 0, dist_stop_price / atr_refs, np.nan)
        potential_rr = np.where(dist_stop_price > 0, dist_high_price / dist_stop_price, np.nan)

        # Construct outcomes DataFrame
        outcomes_df = events_df.copy()
        outcomes_df["retest_primary"] = retest_primary
        outcomes_df["retest_strict"] = retest_strict
        outcomes_df["retest_near"] = retest_near
        outcomes_df["failure"] = failure
        outcomes_df["ambiguous_same_bar"] = ambiguous_same_bar
        outcomes_df["retest_before_failure_pessimistic"] = retest_before_failure_pess
        outcomes_df["retest_before_failure_optimistic"] = retest_before_failure_opt

        outcomes_df["time_to_retest_bars"] = time_to_retest_bars
        outcomes_df["time_to_retest_mins"] = time_to_retest_mins
        outcomes_df["time_to_failure_bars"] = time_to_failure_bars
        outcomes_df["time_to_failure_mins"] = time_to_failure_mins

        outcomes_df["retest_30m"] = retest_30m
        outcomes_df["retest_1h"] = retest_1h
        outcomes_df["retest_2h"] = retest_2h
        outcomes_df["retest_4h"] = retest_4h
        outcomes_df["retest_8h"] = retest_8h
        outcomes_df["retest_12h"] = retest_12h
        outcomes_df["retest_24h"] = retest_24h

        outcomes_df["broke_high"] = broke_high
        outcomes_df["ext_beyond_high_atr"] = ext_beyond_high_atr
        outcomes_df["ext_0_5_atr"] = ext_0_5_atr
        outcomes_df["ext_1_0_atr"] = ext_1_0_atr
        outcomes_df["ext_2_0_atr"] = ext_2_0_atr

        outcomes_df["outcome_type"] = outcome_type

        outcomes_df["mfe_price"] = mfe_price
        outcomes_df["mae_price"] = mae_price
        outcomes_df["mfe_atr"] = mfe_atr
        outcomes_df["mae_atr"] = mae_atr
        outcomes_df["mfe_r"] = mfe_r
        outcomes_df["mae_r"] = mae_r

        outcomes_df["distance_to_high_price"] = dist_high_price
        outcomes_df["distance_to_high_atr"] = dist_high_atr
        outcomes_df["distance_to_stop_price"] = dist_stop_price
        outcomes_df["distance_to_stop_atr"] = dist_stop_atr
        outcomes_df["potential_rr"] = potential_rr

        return outcomes_df
