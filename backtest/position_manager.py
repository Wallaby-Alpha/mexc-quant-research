from typing import Dict, List, Optional, Any
import pandas as pd

from backtest.trade import Position


class PositionManager:
    """
    Position Manager (DEFINITIONS.md §10):
    - Strict constraint: Max ONE open position per symbol
    - Logs skipped overlapping signals to `skipped_overlaps`
    - Interfaces designed so portfolio-level limits and partial exits can be added later (per scope discipline).
    """

    def __init__(self):
        self.active_positions: Dict[str, Position] = {}
        self.skipped_overlaps: List[Dict[str, Any]] = []

    def can_open_position(self, symbol: str, timestamp: pd.Timestamp, candidate_id: str) -> bool:
        """
        Checks if a position can be opened.
        If symbol already has an open position, logs skipped_overlap and returns False.
        """
        if symbol in self.active_positions:
            active = self.active_positions[symbol]
            self.skipped_overlaps.append({
                "candidate_id": candidate_id,
                "symbol": symbol,
                "timestamp": timestamp,
                "active_trade_id": active.trade_id,
                "reason": "skipped_overlap"
            })
            return False

        # Portfolio-level capacity hook (ready for future extension, not implemented yet)
        if not self._check_portfolio_capacity_hook(symbol, timestamp):
            return False

        return True

    def open_position(self, position: Position):
        """
        Registers an opened position.
        """
        if position.symbol in self.active_positions:
            raise ValueError(f"Cannot open multiple positions for symbol {position.symbol}!")
        self.active_positions[position.symbol] = position

    def close_position(self, symbol: str) -> Optional[Position]:
        """
        Removes and returns active position upon trade closure.
        """
        return self.active_positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.active_positions.get(symbol)

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.active_positions

    # --------------------------------------------------------------------------
    # Scope Discipline Hooks for Future Extensions (do not implement logic now)
    # --------------------------------------------------------------------------
    def _check_portfolio_capacity_hook(self, symbol: str, timestamp: pd.Timestamp) -> bool:
        """
        Placeholder hook for portfolio-wide simultaneous position limits.
        """
        return True

    def _partial_exit_hook(self, symbol: str, fraction: float, exit_price: float):
        """
        Placeholder hook for partial take-profit / scaling out.
        """
        pass
