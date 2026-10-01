"""
strategy/turtle_soup_4h.py
4-Hour Liquidity Sweep / False Breakout ("Turtle Soup") Short Strategy Engine with Relative Weakness.
Fades false breakouts of 4H swing highs on altcoins that are structurally underperforming Bitcoin.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np

from strategy.indicators import compute_wilder_atr, compute_ema
from backtest.execution_model import ExecutionModel


@dataclass
class ActiveShortTurtlePosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_bar_idx: int
    entry_price_raw: float
    entry_price: float
    stop_price: float
    target_price: float
    initial_risk_dist: float
    entry_fee: float
    entry_slippage: float
    bars_held: int = 0


class TurtleSoup4HEngine:
    """
    Simulates 4H Turtle Soup short setups with Relative Weakness (RW) conditioning.
    """

    @staticmethod
    def simulate_symbol_short_trades(
        symbol: str,
        df_4h: pd.DataFrame,
        lookback_window: int = 18, # 18 4H bars = 3 days
        min_wick_ratio: float = 0.35, # Upper wick must be >= 35% of candle range
        target_rr: float = 2.0, # Target = Entry - target_rr * Risk
        execution_type: str = "maker_limit_swept_level", # "taker_next_open" or "maker_limit_swept_level"
        max_holding_bars: int = 30, # 30 4H bars = 5 days time stop
        stop_buffer_atr: float = 0.25, # Buffer above the sweep wick
        rw_filter_fn: Optional[Callable[[pd.Series], bool]] = None,
        exec_model: Optional[ExecutionModel] = None,
        holdout_ts: Optional[pd.Timestamp] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a causal bar-by-bar simulation for shorting 4H false breakouts.
        """
        n_bars = len(df_4h)
        min_required = max(lookback_window + 10, 60)
        if n_bars < min_required:
            return []

        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        opens = df_4h["open"].to_numpy(dtype=float)
        open_times = pd.to_datetime(df_4h["open_time"], utc=True).tolist()
        ranks = df_4h["rank"].to_numpy() if "rank" in df_4h.columns else [50] * n_bars

        # Precompute causal indicators
        atr_series = compute_wilder_atr(df_4h["high"], df_4h["low"], df_4h["close"], period=14).to_numpy(dtype=float)

        # Causal rolling highest high of previous N bars (excluding current bar)
        roll_highs = pd.Series(highs).rolling(window=lookback_window).max().shift(1).to_numpy(dtype=float)

        active_pos: Optional[ActiveShortTurtlePosition] = None
        completed_trades: List[Dict[str, Any]] = []
        trade_seq = 0

        for i in range(lookback_window + 5, n_bars - 1):
            curr_t = open_times[i]
            if holdout_ts is not None and curr_t >= holdout_ts:
                break

            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_atr = atr_series[i]
            rank_val = ranks[i] if ranks[i] is not None else 50

            # -------------------------------------------------------------
            # 1. Manage Active Short Position (Check Stop / Target / Time Stop)
            # -------------------------------------------------------------
            if active_pos is not None:
                active_pos.bars_held += 1
                hit_target = bar_low <= active_pos.target_price
                hit_stop = bar_high >= active_pos.stop_price
                exit_reason = None
                exit_raw = bar_close

                if hit_target and hit_stop:
                    # Conservative tie-breaker: stop wins
                    hit_stop = True
                    hit_target = False

                if hit_target:
                    exit_raw = active_pos.target_price
                    exit_reason = "target"
                elif hit_stop:
                    exit_raw = max(bar_open, active_pos.stop_price) if bar_open > active_pos.stop_price else active_pos.stop_price
                    exit_reason = "stop"
                elif active_pos.bars_held >= max_holding_bars:
                    exit_raw = bar_close
                    exit_reason = "time_stop"

                if exit_reason is not None:
                    slip = exec_model.get_slippage_rate(rank_val) if (exec_model and exit_reason == "stop") else 0.0
                    fee = exec_model.taker_fee_rate if exec_model else 0.0002

                    fill_exit = exit_raw * (1.0 + slip)
                    gross_pnl_pct = (active_pos.entry_price_raw - exit_raw) / active_pos.entry_price_raw
                    net_price_ret = (active_pos.entry_price - fill_exit) / active_pos.entry_price

                    risk_pct = active_pos.initial_risk_dist / active_pos.entry_price_raw
                    if risk_pct <= 0:
                        risk_pct = 0.01

                    gross_pnl_r = gross_pnl_pct / risk_pct
                    funding_cost = exec_model.calculate_funding(active_pos.entry_time, curr_t) if exec_model else 0.0
                    total_fees = active_pos.entry_fee + fee
                    slippage_cost = active_pos.entry_slippage + slip

                    net_pnl_pct = net_price_ret - total_fees - funding_cost
                    net_pnl_r = net_pnl_pct / risk_pct

                    trade_dict = {
                        "trade_id": f"TS4H_{symbol}_{trade_seq}",
                        "symbol": symbol,
                        "side": "short",
                        "entry_time": active_pos.entry_time,
                        "exit_time": curr_t,
                        "bars_held": active_pos.bars_held,
                        "holding_time_minutes": active_pos.bars_held * 240.0,
                        "entry_price": active_pos.entry_price,
                        "exit_price": fill_exit,
                        "stop_price": active_pos.stop_price,
                        "target_price": active_pos.target_price,
                        "gross_pnl_pct": gross_pnl_pct,
                        "gross_pnl_r": gross_pnl_r,
                        "net_pnl_pct": net_pnl_pct,
                        "net_pnl_r": net_pnl_r,
                        "total_fees": total_fees,
                        "slippage_cost": slippage_cost,
                        "funding_cost": funding_cost,
                        "exit_reason": exit_reason,
                        "ambiguous_same_bar": False
                    }
                    completed_trades.append(trade_dict)
                    active_pos = None
                    continue

            # -------------------------------------------------------------
            # 2. Check 4H Bearish Sweep Signal on Previous Bar (i - 1)
            # -------------------------------------------------------------
            if active_pos is None:
                p_open = opens[i - 1]
                p_high = highs[i - 1]
                p_low = lows[i - 1]
                p_close = closes[i - 1]
                p_range = p_high - p_low
                p_atr = atr_series[i - 1]

                if p_range <= 0 or p_atr <= 0:
                    continue

                ref_h = roll_highs[i - 1]

                # Sweep Condition: penetrated above prior swing high, closed back below
                if p_high > ref_h and p_close < ref_h:
                    upper_wick = p_high - max(p_open, p_close)
                    if (upper_wick / p_range) >= min_wick_ratio:
                        
                        # Relative Weakness filter check
                        if rw_filter_fn is not None:
                            prev_row = df_4h.iloc[i - 1]
                            if not rw_filter_fn(prev_row):
                                continue

                        trade_seq += 1
                        init_stop = p_high + (stop_buffer_atr * p_atr)

                        if execution_type == "maker_limit_swept_level":
                            if bar_high >= ref_h:
                                entry_raw = ref_h
                                entry_fee = exec_model.maker_fee_rate if exec_model else 0.0
                                entry_slip = 0.0
                                fill_entry = entry_raw
                            else:
                                continue # Limit not filled
                        else:
                            entry_raw = bar_open
                            entry_slip = exec_model.get_slippage_rate(rank_val) if exec_model else 0.0005
                            entry_fee = exec_model.taker_fee_rate if exec_model else 0.0002
                            fill_entry = entry_raw * (1.0 - entry_slip)

                        risk_dist = init_stop - fill_entry
                        if risk_dist <= 0:
                            continue

                        target_p = fill_entry - (target_rr * risk_dist)

                        active_pos = ActiveShortTurtlePosition(
                            symbol=symbol,
                            entry_time=curr_t,
                            entry_bar_idx=i,
                            entry_price_raw=entry_raw,
                            entry_price=fill_entry,
                            stop_price=init_stop,
                            target_price=target_p,
                            initial_risk_dist=risk_dist,
                            entry_fee=entry_fee,
                            entry_slippage=entry_slip,
                            bars_held=0
                        )

        return completed_trades
