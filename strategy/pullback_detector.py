from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from strategy.swing_detector import ConfirmedSwingHigh


PULLBACK_ATR_THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
RETRACEMENT_PCT_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


@dataclass(frozen=True)
class ThresholdCrossingEvent:
    # Identifiers
    event_id: str                      # symbol_T_threshold
    symbol: str
    swing_high_time: pd.Timestamp
    swing_high_price: float
    confirmation_time: pd.Timestamp
    
    # Preceding impulse
    swing_low_time: pd.Timestamp
    swing_low_price: float
    impulse: float
    impulse_atr: float
    atr_ref: float                     # Frozen at swing confirmation

    # Event trigger details
    threshold_type: str                # "atr" or "retracement"
    threshold_value: float             # e.g. 0.50 ATR or 30.0%
    event_bar_idx: int                 # 15m bar index t
    event_time: pd.Timestamp           # 15m bar open_time
    event_close_time: pd.Timestamp     # 15m bar close_time (decision time = open + 15m)
    action_open_time: pd.Timestamp     # Open of t+1
    reference_price: float             # Open price of t+1 bar

    # Causal state at decision time
    running_pullback_low: float        # lowest low in [T+1..t]
    pullback_depth_atr: float          # (swing_high - running_low) / atr_ref
    retracement_pct: float             # (swing_high - running_low) / impulse * 100
    atr_entry: float                   # ATR15 known at event close
    stop_buffer_atr: float             # default 0.25
    event_stop_level: float            # running_low - stop_buffer * atr_ref
    
    # 1H trend status at decision time
    trend_valid_default: bool          # Trend C
    trend_a: bool
    trend_b: bool
    trend_c: bool
    trend_d: bool
    
    # Universe rank at decision time
    universe_rank: Optional[int]
    is_controlled_pullback: bool       # trend valid AND not stopped out prior to event

    # Diagnostics (stamped for analysis, NEVER used as causal filters)
    diag_final_depth_atr: Optional[float] = None
    diag_final_depth_pct: Optional[float] = None


class PullbackDetector:
    """
    Causal Pullback & Threshold-Crossing Event Detector (DEFINITIONS.md §6 & §7).
    Vectorized with np.minimum.accumulate for extreme performance.
    """

    def __init__(
        self,
        atr_thresholds: List[float] = PULLBACK_ATR_THRESHOLDS,
        ret_thresholds: List[float] = RETRACEMENT_PCT_THRESHOLDS,
        stop_buffer: float = 0.25,
        max_horizon_bars: int = 96  # 24 hours of 15m bars
    ):
        self.atr_thresholds = sorted(atr_thresholds)
        self.ret_thresholds = sorted(ret_thresholds)
        self.stop_buffer = stop_buffer
        self.max_horizon_bars = max_horizon_bars

    def extract_context(
        self,
        df_15m: pd.DataFrame,
        atr_15m: pd.Series,
        trend_flags: pd.DataFrame,
        universe_ranks: Optional[pd.Series] = None
    ) -> dict:
        n_bars = len(df_15m)
        open_times = pd.to_datetime(df_15m["open_time"], utc=True).tolist()
        return {
            "n_bars": n_bars,
            "lows": df_15m["low"].to_numpy(dtype=float),
            "opens": df_15m["open"].to_numpy(dtype=float),
            "closes": df_15m["close"].to_numpy(dtype=float),
            "open_times": open_times,
            "atrs": atr_15m.to_numpy(dtype=float),
            "trend_c": trend_flags["trend_c"].to_numpy(dtype=bool) if "trend_c" in trend_flags.columns else np.zeros(n_bars, dtype=bool),
            "trend_a": trend_flags["trend_a"].to_numpy(dtype=bool) if "trend_a" in trend_flags.columns else np.zeros(n_bars, dtype=bool),
            "trend_b": trend_flags["trend_b"].to_numpy(dtype=bool) if "trend_b" in trend_flags.columns else np.zeros(n_bars, dtype=bool),
            "trend_d": trend_flags["trend_d"].to_numpy(dtype=bool) if "trend_d" in trend_flags.columns else np.zeros(n_bars, dtype=bool),
            "trend_inv": trend_flags["trend_invalidated"].to_numpy(dtype=bool) if "trend_invalidated" in trend_flags.columns else np.zeros(n_bars, dtype=bool),
            "ranks": universe_ranks.to_numpy() if universe_ranks is not None else None,
        }

    def process_swing_high(
        self,
        swing: ConfirmedSwingHigh,
        df_15m: pd.DataFrame,
        atr_15m: pd.Series,
        trend_flags: pd.DataFrame,
        universe_ranks: Optional[pd.Series] = None,
        ctx: Optional[dict] = None
    ) -> List[ThresholdCrossingEvent]:
        if ctx is None:
            ctx = self.extract_context(df_15m, atr_15m, trend_flags, universe_ranks)

        n_bars = ctx["n_bars"]
        T = swing.swing_high_bar_idx
        conf_idx = swing.confirmation_bar_idx  # T + N
        swing_high_px = swing.swing_high_price
        impulse = swing.impulse
        atr_ref = swing.atr_ref

        if conf_idx >= n_bars:
            return []

        end_horizon = min(n_bars, conf_idx + self.max_horizon_bars + 1)
        lows = ctx["lows"]
        opens = ctx["opens"]
        closes = ctx["closes"]
        atrs = ctx["atrs"]
        open_times = ctx["open_times"]
        trend_c_arr = ctx["trend_c"]
        trend_a_arr = ctx["trend_a"]
        trend_b_arr = ctx["trend_b"]
        trend_d_arr = ctx["trend_d"]
        trend_inv_arr = ctx["trend_inv"]
        ranks_arr = ctx["ranks"]

        # Vectorized running low across horizon [conf_idx .. end_horizon]
        pre_min = np.min(lows[T + 1: conf_idx + 1])
        post_accum = np.minimum.accumulate(lows[conf_idx: end_horizon])
        running_lows = np.minimum(pre_min, post_accum)

        depths_atr = (swing_high_px - running_lows) / atr_ref
        depths_ret = (swing_high_px - running_lows) / impulse * 100.0

        diag_final_low = float(running_lows[-1])
        diag_atr = float((swing_high_px - diag_final_low) / atr_ref)
        diag_pct = float((swing_high_px - diag_final_low) / impulse * 100.0)

        events: List[ThresholdCrossingEvent] = []

        # Find first crossing for each ATR threshold
        for th in self.atr_thresholds:
            cross_mask = depths_atr >= th
            if not np.any(cross_mask):
                continue
            offset = int(np.argmax(cross_mask))
            t = conf_idx + offset
            running_low = float(running_lows[offset])
            depth_atr = float(depths_atr[offset])
            depth_ret = float(depths_ret[offset])

            stop_level = running_low - (self.stop_buffer * atr_ref)
            atr_entry = float(atrs[t]) if not np.isnan(atrs[t]) else atr_ref
            ev_time = open_times[t]
            action_time = open_times[t + 1] if t + 1 < n_bars else ev_time + pd.Timedelta(minutes=15)
            ref_price = float(opens[t + 1]) if t + 1 < n_bars else float(closes[t])
            u_rank = int(ranks_arr[t]) if ranks_arr is not None and not pd.isna(ranks_arr[t]) else None

            current_bar_low = float(lows[t])
            t_trend_c = bool(trend_c_arr[t])
            t_invalid = bool(trend_inv_arr[t])
            is_controlled = t_trend_c and (not t_invalid) and (current_bar_low > stop_level)

            event_id = f"{swing.symbol}_{swing.swing_high_time.strftime('%Y%m%d%H%M')}_ATR_{th:.2f}"
            events.append(ThresholdCrossingEvent(
                event_id=event_id,
                symbol=swing.symbol,
                swing_high_time=swing.swing_high_time,
                swing_high_price=swing_high_px,
                confirmation_time=swing.confirmation_time,
                swing_low_time=swing.swing_low_time,
                swing_low_price=swing.swing_low_price,
                impulse=impulse,
                impulse_atr=swing.impulse_atr,
                atr_ref=atr_ref,
                threshold_type="atr",
                threshold_value=th,
                event_bar_idx=t,
                event_time=ev_time,
                event_close_time=ev_time + pd.Timedelta(minutes=15),
                action_open_time=action_time,
                reference_price=ref_price,
                running_pullback_low=running_low,
                pullback_depth_atr=depth_atr,
                retracement_pct=depth_ret,
                atr_entry=atr_entry,
                stop_buffer_atr=self.stop_buffer,
                event_stop_level=float(stop_level),
                trend_valid_default=t_trend_c,
                trend_a=bool(trend_a_arr[t]),
                trend_b=bool(trend_b_arr[t]),
                trend_c=t_trend_c,
                trend_d=bool(trend_d_arr[t]),
                universe_rank=u_rank,
                is_controlled_pullback=is_controlled,
                diag_final_depth_atr=diag_atr,
                diag_final_depth_pct=diag_pct
            ))

        # Find first crossing for each Retracement % threshold
        for r_th in self.ret_thresholds:
            cross_mask = depths_ret >= r_th
            if not np.any(cross_mask):
                continue
            offset = int(np.argmax(cross_mask))
            t = conf_idx + offset
            running_low = float(running_lows[offset])
            depth_atr = float(depths_atr[offset])
            depth_ret = float(depths_ret[offset])

            stop_level = running_low - (self.stop_buffer * atr_ref)
            atr_entry = float(atrs[t]) if not np.isnan(atrs[t]) else atr_ref
            ev_time = open_times[t]
            action_time = open_times[t + 1] if t + 1 < n_bars else ev_time + pd.Timedelta(minutes=15)
            ref_price = float(opens[t + 1]) if t + 1 < n_bars else float(closes[t])
            u_rank = int(ranks_arr[t]) if ranks_arr is not None and not pd.isna(ranks_arr[t]) else None

            current_bar_low = float(lows[t])
            t_trend_c = bool(trend_c_arr[t])
            t_invalid = bool(trend_inv_arr[t])
            is_controlled = t_trend_c and (not t_invalid) and (current_bar_low > stop_level)

            event_id = f"{swing.symbol}_{swing.swing_high_time.strftime('%Y%m%d%H%M')}_RET_{int(r_th)}"
            events.append(ThresholdCrossingEvent(
                event_id=event_id,
                symbol=swing.symbol,
                swing_high_time=swing.swing_high_time,
                swing_high_price=swing_high_px,
                confirmation_time=swing.confirmation_time,
                swing_low_time=swing.swing_low_time,
                swing_low_price=swing.swing_low_price,
                impulse=impulse,
                impulse_atr=swing.impulse_atr,
                atr_ref=atr_ref,
                threshold_type="retracement",
                threshold_value=r_th,
                event_bar_idx=t,
                event_time=ev_time,
                event_close_time=ev_time + pd.Timedelta(minutes=15),
                action_open_time=action_time,
                reference_price=ref_price,
                running_pullback_low=running_low,
                pullback_depth_atr=depth_atr,
                retracement_pct=depth_ret,
                atr_entry=atr_entry,
                stop_buffer_atr=self.stop_buffer,
                event_stop_level=float(stop_level),
                trend_valid_default=t_trend_c,
                trend_a=bool(trend_a_arr[t]),
                trend_b=bool(trend_b_arr[t]),
                trend_c=t_trend_c,
                trend_d=bool(trend_d_arr[t]),
                universe_rank=u_rank,
                is_controlled_pullback=is_controlled,
                diag_final_depth_atr=diag_atr,
                diag_final_depth_pct=diag_pct
            ))

        return events
