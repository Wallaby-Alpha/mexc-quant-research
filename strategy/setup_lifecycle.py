from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd


class SetupState(str, Enum):
    PENDING = "PENDING"          # Confirmed swing high, tracking pullback
    ARMED = "ARMED"              # Threshold-crossing event has fired
    RESOLVED = "RESOLVED"        # Retest hit or stop failure occurred
    CANCELLED = "CANCELLED"      # 1H trend invalidated before event fired
    SUPERSEDED = "SUPERSEDED"    # New higher confirmed swing high appeared
    EXPIRED = "EXPIRED"          # Max horizon exceeded without retest or stop


@dataclass
class SetupLifecycle:
    symbol: str
    swing_high_time: pd.Timestamp
    swing_high_price: float
    confirmation_time: pd.Timestamp
    state: SetupState = SetupState.PENDING
    transition_time: Optional[pd.Timestamp] = None
    transition_reason: Optional[str] = None

    def trigger_event(self, event_time: pd.Timestamp):
        if self.state == SetupState.PENDING:
            self.state = SetupState.ARMED
            self.transition_time = event_time
            self.transition_reason = "threshold_crossed"

    def resolve(self, resolve_time: pd.Timestamp, reason: str):
        if self.state in (SetupState.PENDING, SetupState.ARMED):
            self.state = SetupState.RESOLVED
            self.transition_time = resolve_time
            self.transition_reason = reason

    def cancel(self, cancel_time: pd.Timestamp, reason: str = "trend_invalidated_before_event"):
        if self.state == SetupState.PENDING:
            self.state = SetupState.CANCELLED
            self.transition_time = cancel_time
            self.transition_reason = reason

    def supersede(self, new_swing_time: pd.Timestamp):
        if self.state in (SetupState.PENDING, SetupState.ARMED):
            self.state = SetupState.SUPERSEDED
            self.transition_time = new_swing_time
            self.transition_reason = "higher_swing_high_confirmed"

    def expire(self, expire_time: pd.Timestamp):
        if self.state in (SetupState.PENDING, SetupState.ARMED):
            self.state = SetupState.EXPIRED
            self.transition_time = expire_time
            self.transition_reason = "max_horizon_elapsed"
