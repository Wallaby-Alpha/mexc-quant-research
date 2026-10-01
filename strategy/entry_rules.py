from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from strategy.pullback_detector import ThresholdCrossingEvent


@dataclass
class EntrySignal:
    event_id: str
    symbol: str
    entry_mode: str
    signal_bar_idx: int
    signal_time: pd.Timestamp       # 15m candle close time
    entry_bar_idx: int
    entry_time: pd.Timestamp        # 15m candle open time (action time)
    candidate_entry_price: float    # Expected fill price (open of entry bar)
    stop_price: float
    target_price: float
    potential_rr: float
    atr_entry: float
    is_valid_rr: bool


class BaseEntryRule(ABC):
    """
    Abstract Base Class for Entry Rules (DEFINITIONS.md §14).
    Pluggable for Entry A, B, C, D and future 5m refinement.
    Supports high-speed pre-extracted numpy arrays for extreme performance.
    """

    def __init__(self, name: str, min_rr: float = 1.0):
        self.name = name
        self.min_rr = min_rr

    @abstractmethod
    def evaluate(
        self,
        event: ThresholdCrossingEvent,
        df_15m: pd.DataFrame,
        stop_level: float,
        target_price: float,
        atr_series: pd.Series,
        max_search_bars: int = 96,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ) -> Optional[EntrySignal]:
        pass

    def check_min_rr(self, target_price: float, entry_price: float, stop_price: float) -> Tuple[float, bool]:
        risk = entry_price - stop_price
        reward = target_price - entry_price
        if risk <= 0:
            return 0.0, False
        rr = reward / risk
        return rr, rr >= self.min_rr

    @staticmethod
    def extract_arrays(
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ):
        if arrays is not None:
            return arrays
        opens = df_15m["open"].to_numpy(dtype=float)
        highs = df_15m["high"].to_numpy(dtype=float)
        lows = df_15m["low"].to_numpy(dtype=float)
        closes = df_15m["close"].to_numpy(dtype=float)
        open_times = pd.to_datetime(df_15m["open_time"], utc=True).tolist()
        atrs = atr_series.to_numpy(dtype=float)
        return opens, highs, lows, closes, open_times, atrs


class EntryA_Immediate(BaseEntryRule):
    """
    Entry A (Immediate):
    Signal occurs at the close of event bar t.
    Entry fill occurs at open of bar t+1.
    """

    def __init__(self, min_rr: float = 1.0):
        super().__init__("Entry_A", min_rr)

    def evaluate(
        self,
        event: ThresholdCrossingEvent,
        df_15m: pd.DataFrame,
        stop_level: float,
        target_price: float,
        atr_series: pd.Series,
        max_search_bars: int = 96,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ) -> Optional[EntrySignal]:
        opens, highs, lows, closes, open_times, atrs = self.extract_arrays(df_15m, atr_series, arrays)
        t = event.event_bar_idx
        if t + 1 >= len(opens):
            return None

        entry_bar_idx = t + 1
        entry_time = open_times[entry_bar_idx]
        signal_time = open_times[t] + pd.Timedelta(minutes=15)
        candidate_price = float(opens[entry_bar_idx])
        atr_entry = float(atrs[t]) if not np.isnan(atrs[t]) else event.atr_ref

        rr, valid_rr = self.check_min_rr(target_price, candidate_price, stop_level)

        return EntrySignal(
            event_id=event.event_id,
            symbol=event.symbol,
            entry_mode=self.name,
            signal_bar_idx=t,
            signal_time=signal_time,
            entry_bar_idx=entry_bar_idx,
            entry_time=entry_time,
            candidate_entry_price=candidate_price,
            stop_price=stop_level,
            target_price=target_price,
            potential_rr=rr,
            atr_entry=atr_entry,
            is_valid_rr=valid_rr
        )


class EntryB_15mReversal(BaseEntryRule):
    """
    Entry B (15m Reversal):
    After event bar t, scans for the first 15m candle closing above the prior 15m candle's high.
    Fill occurs at the open of the next candle.
    Invalidated if bar low <= stop_level before reversal.
    """

    def __init__(self, min_rr: float = 1.0):
        super().__init__("Entry_B", min_rr)

    def evaluate(
        self,
        event: ThresholdCrossingEvent,
        df_15m: pd.DataFrame,
        stop_level: float,
        target_price: float,
        atr_series: pd.Series,
        max_search_bars: int = 96,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ) -> Optional[EntrySignal]:
        opens, highs, lows, closes, open_times, atrs = self.extract_arrays(df_15m, atr_series, arrays)
        t = event.event_bar_idx
        n_bars = len(opens)
        end_idx = min(n_bars, t + 1 + max_search_bars)

        # Start checking from bar t+1
        for k in range(t + 1, end_idx):
            bar_low = float(lows[k])
            bar_high = float(highs[k])
            bar_close = float(closes[k])
            prev_high = float(highs[k - 1])

            # Invalidation: low breaks stop before reversal
            if bar_low <= stop_level:
                return None

            # Retest reached before entry: setup resolved without trade entry
            if bar_high >= target_price:
                return None

            # Reversal condition: close above previous bar's high
            if bar_close > prev_high:
                entry_bar_idx = k + 1
                if entry_bar_idx >= n_bars:
                    return None

                signal_time = open_times[k] + pd.Timedelta(minutes=15)
                entry_time = open_times[entry_bar_idx]
                candidate_price = float(opens[entry_bar_idx])
                atr_entry = float(atrs[k]) if not np.isnan(atrs[k]) else event.atr_ref

                rr, valid_rr = self.check_min_rr(target_price, candidate_price, stop_level)

                return EntrySignal(
                    event_id=event.event_id,
                    symbol=event.symbol,
                    entry_mode=self.name,
                    signal_bar_idx=k,
                    signal_time=signal_time,
                    entry_bar_idx=entry_bar_idx,
                    entry_time=entry_time,
                    candidate_entry_price=candidate_price,
                    stop_price=stop_level,
                    target_price=target_price,
                    potential_rr=rr,
                    atr_entry=atr_entry,
                    is_valid_rr=valid_rr
                )

        return None


class EntryC_LowerHighBreak(BaseEntryRule):
    """
    Entry C (Lower-High Break):
    Tracks confirmed 15m lower highs (1/1 fractal: high[m] > high[m-1] and high[m] > high[m+1])
    below the swing high during the pullback.
    Signals on the first close above this lower high; fills at next open.
    Invalidated if bar low <= stop_level before breakout.
    """

    def __init__(self, min_rr: float = 1.0):
        super().__init__("Entry_C", min_rr)

    def evaluate(
        self,
        event: ThresholdCrossingEvent,
        df_15m: pd.DataFrame,
        stop_level: float,
        target_price: float,
        atr_series: pd.Series,
        max_search_bars: int = 96,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ) -> Optional[EntrySignal]:
        opens, highs, lows, closes, open_times, atrs = self.extract_arrays(df_15m, atr_series, arrays)
        t = event.event_bar_idx
        n_bars = len(opens)
        end_idx = min(n_bars, t + 1 + max_search_bars)

        recent_lower_high: Optional[float] = None

        # Track bars starting from t+1
        for k in range(t + 1, end_idx):
            bar_low = float(lows[k])
            bar_high = float(highs[k])
            bar_close = float(closes[k])

            # Invalidation: low breaks stop before entry
            if bar_low <= stop_level:
                return None

            # Retest reached before entry
            if bar_high >= target_price:
                return None

            # Check if bar k-1 was a 1/1 fractal lower high (confirmed at close of bar k)
            if k >= t + 2:
                prev_high = float(highs[k - 1])
                prev2_high = float(highs[k - 2])
                curr_high = bar_high
                if prev_high > prev2_high and prev_high > curr_high and prev_high < target_price:
                    recent_lower_high = prev_high

            # If a lower high exists, check for causal breakout on close of candle k
            if recent_lower_high is not None and bar_close > recent_lower_high:
                entry_bar_idx = k + 1
                if entry_bar_idx >= n_bars:
                    return None

                signal_time = open_times[k] + pd.Timedelta(minutes=15)
                entry_time = open_times[entry_bar_idx]
                candidate_price = float(opens[entry_bar_idx])
                atr_entry = float(atrs[k]) if not np.isnan(atrs[k]) else event.atr_ref

                rr, valid_rr = self.check_min_rr(target_price, candidate_price, stop_level)

                return EntrySignal(
                    event_id=event.event_id,
                    symbol=event.symbol,
                    entry_mode=self.name,
                    signal_bar_idx=k,
                    signal_time=signal_time,
                    entry_bar_idx=entry_bar_idx,
                    entry_time=entry_time,
                    candidate_entry_price=candidate_price,
                    stop_price=stop_level,
                    target_price=target_price,
                    potential_rr=rr,
                    atr_entry=atr_entry,
                    is_valid_rr=valid_rr
                )

        return None


class EntryD_Plugin(BaseEntryRule):
    """
    Plugin interface for Entry D and future 5m refinement (DEFINITIONS.md §14).
    Kept ready per scope discipline.
    """

    def __init__(self, name: str = "Entry_D", min_rr: float = 1.0):
        super().__init__(name, min_rr)

    def evaluate(
        self,
        event: ThresholdCrossingEvent,
        df_15m: pd.DataFrame,
        stop_level: float,
        target_price: float,
        atr_series: pd.Series,
        max_search_bars: int = 96,
        arrays: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray]] = None
    ) -> Optional[EntrySignal]:
        return None
