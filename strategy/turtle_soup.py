"""
strategy/turtle_soup.py
1-Hour Liquidity Sweep / False Breakout ("Turtle Soup") Strategy Engine.
Detects when price sweeps beyond a key 1H swing high or low to trigger stops/breakouts,
then rejects and closes back inside the range.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from strategy.indicators import compute_wilder_atr, compute_ema
from backtest.execution_model import ExecutionModel


@dataclass
class ActiveTurtlePosition:
    symbol: str
    side: str # "long" or "short"
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


class TurtleSoupEngine:
    """
    Simulates 1H Turtle Soup liquidity sweep reversals.
    """

    @staticmethod
    def simulate_symbol_trades(
        symbol: str,
        df_1h: pd.DataFrame,
        lookback_window: int = 24, # 24 bars = 24 hours
        trade_side: str = "both", # "long", "short", "both"
        min_wick_ratio: float = 0.30, # Wick must be >= 30% of total candle range
        target_rr: float = 2.0, # Target = Entry +/- target_rr * Risk
        execution_type: str = "taker_next_open", # "taker_next_open" or "maker_limit_swept_level"
        max_holding_bars: int = 24, # 24 hours time stop
        stop_buffer_atr: float = 0.25, # Buffer beyond the sweep wick
        exec_model: Optional[ExecutionModel] = None,
        holdout_ts: Optional[pd.Timestamp] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a causal bar-by-bar simulation for a single symbol on 1H klines.
        """
        n_bars = len(df_1h)
        min_required = max(lookback_window + 10, 60)
        if n_bars < min_required:
            return []

        highs = df_1h["high"].to_numpy(dtype=float)
        lows = df_1h["low"].to_numpy(dtype=float)
        closes = df_1h["close"].to_numpy(dtype=float)
        opens = df_1h["open"].to_numpy(dtype=float)
        open_times = pd.to_datetime(df_1h["open_time"], utc=True).tolist()
        ranks = df_1h["rank"].to_numpy() if "rank" in df_1h.columns else [50] * n_bars

        # Precompute causal indicators
        atr_series = compute_wilder_atr(df_1h["high"], df_1h["low"], df_1h["close"], period=14).to_numpy(dtype=float)

        # Causal Donchian / Swing extremes: strictly from previous bars (excluding current bar)
        roll_highs = pd.Series(highs).rolling(window=lookback_window).max().shift(1).to_numpy(dtype=float)
        roll_lows = pd.Series(lows).rolling(window=lookback_window).min().shift(1).to_numpy(dtype=float)

        active_pos: Optional[ActiveTurtlePosition] = None
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
            # 1. Manage Active Position (Check Stop / Target / Time Stop)
            # -------------------------------------------------------------
            if active_pos is not None:
                active_pos.bars_held += 1
                hit_target = False
                hit_stop = False
                exit_raw = bar_close
                exit_reason = None

                if active_pos.side == "long":
                    if bar_high >= active_pos.target_price:
                        hit_target = True
                    if bar_low <= active_pos.stop_price:
                        hit_stop = True

                    if hit_target and hit_stop:
                        # Conservative same-bar tie-breaker: stop wins
                        hit_stop = True
                        hit_target = False

                    if hit_target:
                        exit_raw = active_pos.target_price
                        exit_reason = "target"
                    elif hit_stop:
                        exit_raw = min(bar_open, active_pos.stop_price) if bar_open < active_pos.stop_price else active_pos.stop_price
                        exit_reason = "stop"
                    elif active_pos.bars_held >= max_holding_bars:
                        exit_raw = bar_close
                        exit_reason = "time_stop"

                else: # Short position
                    if bar_low <= active_pos.target_price:
                        hit_target = True
                    if bar_high >= active_pos.stop_price:
                        hit_stop = True

                    if hit_target and hit_stop:
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

                    if active_pos.side == "long":
                        fill_exit = exit_raw * (1.0 - slip)
                        gross_pnl_pct = (exit_raw - active_pos.entry_price_raw) / active_pos.entry_price_raw
                        net_price_ret = (fill_exit - active_pos.entry_price) / active_pos.entry_price
                    else: # short
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
                        "trade_id": f"TS_{symbol}_{trade_seq}",
                        "symbol": symbol,
                        "side": active_pos.side,
                        "entry_time": active_pos.entry_time,
                        "exit_time": curr_t,
                        "bars_held": active_pos.bars_held,
                        "holding_time_minutes": active_pos.bars_held * 60.0,
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
            # 2. Check Liquidity Sweep Signal on Previous Bar (i - 1)
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
                ref_l = roll_lows[i - 1]

                # Check Bullish Sweep (Swept below prior low, closed above)
                bull_sweep = False
                if trade_side in ("long", "both"):
                    if p_low < ref_l and p_close > ref_l:
                        # Check lower wick ratio
                        lower_wick = min(p_open, p_close) - p_low
                        if (lower_wick / p_range) >= min_wick_ratio:
                            bull_sweep = True

                # Check Bearish Sweep (Swept above prior high, closed below)
                bear_sweep = False
                if trade_side in ("short", "both"):
                    if p_high > ref_h and p_close < ref_h:
                        # Check upper wick ratio
                        upper_wick = p_high - max(p_open, p_close)
                        if (upper_wick / p_range) >= min_wick_ratio:
                            bear_sweep = True

                # Execute Setup on Bar i
                if bull_sweep and not bear_sweep:
                    trade_seq += 1
                    init_stop = p_low - (stop_buffer_atr * p_atr)
                    
                    if execution_type == "maker_limit_swept_level":
                        # Limit resting at swept level (ref_l)
                        if bar_low <= ref_l:
                            entry_raw = ref_l
                            entry_fee = exec_model.maker_fee_rate if exec_model else 0.0
                            entry_slip = 0.0
                            fill_entry = entry_raw
                        else:
                            continue # Limit order not filled
                    else:
                        # Taker market order at bar open
                        entry_raw = bar_open
                        entry_slip = exec_model.get_slippage_rate(rank_val) if exec_model else 0.0005
                        entry_fee = exec_model.taker_fee_rate if exec_model else 0.0002
                        fill_entry = entry_raw * (1.0 + entry_slip)

                    risk_dist = fill_entry - init_stop
                    if risk_dist <= 0:
                        continue

                    target_p = fill_entry + (target_rr * risk_dist)

                    active_pos = ActiveTurtlePosition(
                        symbol=symbol,
                        side="long",
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

                elif bear_sweep and not bull_sweep:
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

                    active_pos = ActiveTurtlePosition(
                        symbol=symbol,
                        side="short",
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
