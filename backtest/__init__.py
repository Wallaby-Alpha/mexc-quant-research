"""
MEXC Swing-High Retest - Backtest Package (Phase 4)
"""

from backtest.trade import Trade, Position, TradeExitReason, EntryMode, StopType
from backtest.execution_model import ExecutionModel
from backtest.position_manager import PositionManager
from backtest.engine import BacktestEngine, HoldoutProtectionError

__all__ = [
    "Trade",
    "Position",
    "TradeExitReason",
    "EntryMode",
    "StopType",
    "ExecutionModel",
    "PositionManager",
    "BacktestEngine",
    "HoldoutProtectionError"
]
