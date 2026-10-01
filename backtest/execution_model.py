from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np


class ExecutionModel:
    """
    Execution and Cost Model (DEFINITIONS.md §13):
    - Separate entry & exit fees (maker/taker)
    - Volume rank-scaled slippage per side
    - 8H funding fee accrual (00:00, 08:00, 16:00 UTC)
    - Gap handling on stops
    - Conservative same-bar ambiguity resolution (stop hit first)
    """

    def __init__(
        self,
        maker_fee_rate: float = 0.0000,
        taker_fee_rate: float = 0.0002,
        base_slippage_bps: float = 5.0,
        funding_rate_8h: float = 0.0001,
        slippage_by_rank: Optional[Dict[str, float]] = None,
        target_as_limit: bool = True
    ):
        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate
        self.base_slippage_bps = base_slippage_bps
        self.funding_rate_8h = funding_rate_8h
        self.target_as_limit = target_as_limit
        self.slippage_by_rank = slippage_by_rank or {
            "1-25": 5.0,
            "26-50": 7.5,
            "51-100": 10.0,
            "101-200": 15.0,
            "201-300": 20.0,
            "default": 25.0
        }

    def get_slippage_rate(self, universe_rank: Optional[int]) -> float:
        """
        Returns fractional slippage (e.g. 5 bps -> 0.0005) based on universe volume rank.
        """
        if universe_rank is None or universe_rank <= 0:
            bps = self.slippage_by_rank.get("default", 25.0)
        elif universe_rank <= 25:
            bps = self.slippage_by_rank.get("1-25", 5.0)
        elif universe_rank <= 50:
            bps = self.slippage_by_rank.get("26-50", 7.5)
        elif universe_rank <= 100:
            bps = self.slippage_by_rank.get("51-100", 10.0)
        elif universe_rank <= 200:
            bps = self.slippage_by_rank.get("101-200", 15.0)
        elif universe_rank <= 300:
            bps = self.slippage_by_rank.get("201-300", 20.0)
        else:
            bps = self.slippage_by_rank.get("default", 25.0)

        return float(bps) / 10_000.0

    def calculate_entry(
        self,
        bar_open: float,
        universe_rank: Optional[int]
    ) -> Tuple[float, float, float]:
        """
        Calculates entry fill at the open of the bar immediately following signal close.
        Returns: (fill_price, slippage_pct, fee_pct)
        """
        slippage_rate = self.get_slippage_rate(universe_rank)
        fill_price = bar_open * (1.0 + slippage_rate)
        fee_pct = self.taker_fee_rate
        return fill_price, slippage_rate, fee_pct

    def evaluate_bar_exit(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        stop_price: float,
        target_price: float,
        universe_rank: Optional[int]
    ) -> Optional[Tuple[str, float, float, float, float, bool]]:
        """
        Evaluates potential exit within a 15m bar.
        Returns None if neither target nor stop is hit.
        If hit, returns:
          (exit_reason, exit_price_raw, exit_price_filled, slippage_rate, fee_rate, ambiguous_same_bar)
        """
        target_hit = bar_high >= target_price
        stop_hit = bar_low <= stop_price
        slippage_rate = self.get_slippage_rate(universe_rank)

        # Same-bar ambiguity: conservative rule assumes stop hit first
        if target_hit and stop_hit:
            raw_price, fill_price, slip = self._fill_stop(bar_open, stop_price, slippage_rate)
            return ("stop", raw_price, fill_price, slip, self.taker_fee_rate, True)

        if stop_hit:
            raw_price, fill_price, slip = self._fill_stop(bar_open, stop_price, slippage_rate)
            return ("stop", raw_price, fill_price, slip, self.taker_fee_rate, False)

        if target_hit:
            if self.target_as_limit:
                # Resting limit order filled at target price (maker)
                return ("target", target_price, target_price, 0.0, self.maker_fee_rate, False)
            else:
                # Market order exit at target
                fill_price = target_price * (1.0 - slippage_rate)
                return ("target", target_price, fill_price, slippage_rate, self.taker_fee_rate, False)

        return None

    def calculate_market_exit(
        self,
        bar_open: float,
        reason: str,
        universe_rank: Optional[int]
    ) -> Tuple[str, float, float, float, float, bool]:
        """
        Calculates market exit at open of next bar (e.g. for time stop or trend invalidation).
        """
        slippage_rate = self.get_slippage_rate(universe_rank)
        fill_price = bar_open * (1.0 - slippage_rate)
        return (reason, bar_open, fill_price, slippage_rate, self.taker_fee_rate, False)

    def _fill_stop(
        self,
        bar_open: float,
        stop_price: float,
        slippage_rate: float
    ) -> Tuple[float, float, float]:
        """
        Handles regular stop vs gap down stop.
        If bar_open <= stop_price, trade gapped through stop: filled at bar_open - slippage.
        Otherwise filled at stop_price - slippage.
        """
        if bar_open <= stop_price:
            # Gapped down through stop
            raw_price = bar_open
            fill_price = bar_open * (1.0 - slippage_rate)
        else:
            raw_price = stop_price
            fill_price = stop_price * (1.0 - slippage_rate)
        return raw_price, fill_price, slippage_rate

    def calculate_funding(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp
    ) -> float:
        """
        Calculates cumulative funding rate paid by long position.
        Funding timestamps occur at 00:00, 08:00, 16:00 UTC (every 28,800 seconds).
        Counts funding timestamps strictly inside (entry_time, exit_time].
        """
        if exit_time <= entry_time:
            return 0.0

        t_entry_sec = int(entry_time.timestamp())
        t_exit_sec = int(exit_time.timestamp())
        count = (t_exit_sec // 28800) - (t_entry_sec // 28800)
        return float(max(0, count) * self.funding_rate_8h)
