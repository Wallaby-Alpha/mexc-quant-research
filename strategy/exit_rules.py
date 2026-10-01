from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple
import pandas as pd


class StopType(str, Enum):
    PULLBACK_BUFFER = "pullback_buffer"      # pullback_low - buffer * ATR
    PRIOR_SWING_LOW = "prior_swing_low"      # below prior 15m swing low
    FIXED_ATR = "fixed_atr"                  # entry - k * ATR
    PERCENTAGE = "percentage"                # entry * (1 - pct)


class BaseStopRule(ABC):
    """
    Abstract Base Class for Stop Level Calculation.
    """

    def __init__(self, stop_type: StopType):
        self.stop_type = stop_type

    @abstractmethod
    def calculate_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        pass


class PullbackBufferStop(BaseStopRule):
    """
    Default Stop: pullback_low - buffer * ATR (DEFINITIONS.md §9)
    """

    def __init__(self, buffer_atr: float = 0.25):
        super().__init__(StopType.PULLBACK_BUFFER)
        self.buffer_atr = buffer_atr

    def calculate_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        return running_pullback_low - (self.buffer_atr * atr)


class PriorSwingLowStop(BaseStopRule):
    """
    Stop placed below the prior 15m swing low (impulse origin).
    """

    def __init__(self, buffer_atr: float = 0.0):
        super().__init__(StopType.PRIOR_SWING_LOW)
        self.buffer_atr = buffer_atr

    def calculate_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        return swing_low_price - (self.buffer_atr * atr)


class FixedATRStop(BaseStopRule):
    """
    Stop placed at fixed ATR distance below entry price.
    """

    def __init__(self, mult_atr: float = 1.5):
        super().__init__(StopType.FIXED_ATR)
        self.mult_atr = mult_atr

    def calculate_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        return entry_price - (self.mult_atr * atr)


class PercentageStop(BaseStopRule):
    """
    Stop placed at a fixed percentage below entry price.
    """

    def __init__(self, pct: float = 0.02):
        super().__init__(StopType.PERCENTAGE)
        self.pct = pct

    def calculate_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        return entry_price * (1.0 - self.pct)


class ExitRules:
    """
    Evaluator for exit conditions: Target, Stops, Time Stop, and Optional Trend Exit.
    """

    def __init__(
        self,
        stop_rule: BaseStopRule,
        time_stop_hours: Optional[float] = 24.0,
        enable_trend_exit: bool = False
    ):
        self.stop_rule = stop_rule
        self.time_stop_hours = time_stop_hours
        self.max_holding_bars = int(time_stop_hours * 4) if time_stop_hours is not None else None
        self.enable_trend_exit = enable_trend_exit

    def get_initial_stop(
        self,
        running_pullback_low: float,
        swing_low_price: float,
        entry_price: float,
        atr: float
    ) -> float:
        return self.stop_rule.calculate_stop(
            running_pullback_low=running_pullback_low,
            swing_low_price=swing_low_price,
            entry_price=entry_price,
            atr=atr
        )

    def is_time_stop_reached(self, bars_held: int) -> bool:
        if self.max_holding_bars is None:
            return False
        return bars_held >= self.max_holding_bars

    def is_trend_invalidated(self, trend_invalidated_flag: bool) -> bool:
        if not self.enable_trend_exit:
            return False
        return trend_invalidated_flag
