"""
strategy/trend_follower.py
Uncapped Fat-Tail Trend Following Strategy (Donchian / ATR Chandelier Trailing).
Captures large multi-R asymmetric runners in crypto altcoins.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from strategy.indicators import compute_wilder_atr, compute_ema
from backtest.execution_model import ExecutionModel


@dataclass
class ActiveTrendPosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_bar_idx: int
    entry_price_raw: float
    entry_price: float
    initial_stop_price: float
    current_stop_price: float
    initial_risk_dist: float
    highest_since_entry: float
    entry_fee: float
    entry_slippage: float
    bars_held: int = 0


class TrendFollowerEngine:
    """
    Simulates Donchian breakout entries with dynamic ATR/Donchian trailing stop exits.
    No take-profit cap: allows winning runners to extend to maximum right-tail R-multiples.
    """

    @staticmethod
    def simulate_symbol_trades(
        symbol: str,
        df_4h: pd.DataFrame,
        breakout_window: int = 20,
        trailing_stop_type: str = "chandelier_atr_3.0", # 'chandelier_atr_2.5', 'chandelier_atr_3.0', 'donchian_low_10', 'ema20'
        initial_stop_atr_mult: float = 2.0,
        volume_filter_mult: float = 0.0, # 0.0 = none, 1.2 = 1.2x SMA20
        rs_filter_series: Optional[pd.Series] = None,
        min_rs_spread: float = -999.0, # e.g. 0.0 or 0.10
        exec_model: Optional[ExecutionModel] = None,
        holdout_ts: Optional[pd.Timestamp] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a causal bar-by-bar simulation for a single symbol.
        Returns a list of trade dictionaries.
        """
        n_bars = len(df_4h)
        min_required = max(breakout_window + 10, 60)
        if n_bars < min_required:
            return []

        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        opens = df_4h["open"].to_numpy(dtype=float)
        volumes = df_4h["volume"].to_numpy(dtype=float) if "volume" in df_4h.columns else np.ones(n_bars)
        open_times = pd.to_datetime(df_4h["open_time"], utc=True).tolist()
        ranks = df_4h["rank"].to_numpy() if "rank" in df_4h.columns else [50] * n_bars

        # Precompute causal indicators
        atr_series = compute_wilder_atr(df_4h["high"], df_4h["low"], df_4h["close"], period=14).to_numpy(dtype=float)
        ema50 = compute_ema(df_4h["close"], span=50).to_numpy(dtype=float)
        ema20 = compute_ema(df_4h["close"], span=20).to_numpy(dtype=float)
        vol_sma20 = pd.Series(volumes).rolling(window=20).mean().shift(1).to_numpy(dtype=float)

        # Causal Donchian channel: highest high of previous N bars (excluding current bar)
        donchian_high = pd.Series(highs).rolling(window=breakout_window).max().shift(1).to_numpy(dtype=float)
        donchian_low_10 = pd.Series(lows).rolling(window=10).min().shift(1).to_numpy(dtype=float)

        rs_arr = rs_filter_series.to_numpy(dtype=float) if rs_filter_series is not None else np.zeros(n_bars)

        active_pos: Optional[ActiveTrendPosition] = None
        completed_trades: List[Dict[str, Any]] = []
        trade_seq = 0

        # Trailing multiplier parser
        if trailing_stop_type.startswith("chandelier_atr_"):
            atr_trail_mult = float(trailing_stop_type.split("_")[-1])
        else:
            atr_trail_mult = 3.0

        for i in range(breakout_window + 5, n_bars - 1):
            curr_t = open_times[i]
            if holdout_ts is not None and curr_t >= holdout_ts:
                break

            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_atr = atr_series[i]
            bar_vol = volumes[i]
            rank_val = ranks[i] if ranks[i] is not None else 50

            # -------------------------------------------------------------
            # 1. Manage Active Position (Check Trailing Stop Exit)
            # -------------------------------------------------------------
            if active_pos is not None:
                active_pos.bars_held += 1
                # Update highest high achieved
                if bar_high > active_pos.highest_since_entry:
                    active_pos.highest_since_entry = bar_high

                # Update trailing stop level based on type
                if trailing_stop_type.startswith("chandelier_atr_"):
                    new_trail = active_pos.highest_since_entry - (atr_trail_mult * bar_atr)
                    if new_trail > active_pos.current_stop_price:
                        active_pos.current_stop_price = new_trail
                elif trailing_stop_type == "donchian_low_10":
                    new_trail = donchian_low_10[i]
                    if new_trail > active_pos.current_stop_price:
                        active_pos.current_stop_price = new_trail
                elif trailing_stop_type == "ema20":
                    new_trail = ema20[i]
                    if new_trail > active_pos.current_stop_price:
                        active_pos.current_stop_price = new_trail

                # Check if stop was hit on this bar
                if bar_low <= active_pos.current_stop_price:
                    exit_raw = min(bar_open, active_pos.current_stop_price) if bar_open < active_pos.current_stop_price else active_pos.current_stop_price
                    exit_t = curr_t

                    slip = exec_model.get_slippage_rate(rank_val) if exec_model else 0.001
                    fee = exec_model.taker_fee_rate if exec_model else 0.0002
                    fill_exit = exit_raw * (1.0 - slip)

                    risk_pct = active_pos.initial_risk_dist / active_pos.entry_price_raw
                    if risk_pct <= 0:
                        risk_pct = 0.01

                    gross_pnl_pct = (exit_raw - active_pos.entry_price_raw) / active_pos.entry_price_raw
                    gross_pnl_r = gross_pnl_pct / risk_pct

                    funding_cost = exec_model.calculate_funding(active_pos.entry_time, exit_t) if exec_model else 0.0
                    total_fees = active_pos.entry_fee + fee
                    slippage_cost = active_pos.entry_slippage + slip

                    net_price_ret = (fill_exit - active_pos.entry_price) / active_pos.entry_price
                    net_pnl_pct = net_price_ret - total_fees - funding_cost
                    net_pnl_r = net_pnl_pct / risk_pct

                    trade_dict = {
                        "trade_id": f"TF_{symbol}_{trade_seq}",
                        "symbol": symbol,
                        "entry_time": active_pos.entry_time,
                        "exit_time": exit_t,
                        "bars_held": active_pos.bars_held,
                        "holding_time_minutes": active_pos.bars_held * 240.0,
                        "entry_price": active_pos.entry_price,
                        "exit_price": fill_exit,
                        "stop_price": active_pos.initial_stop_price,
                        "gross_pnl_pct": gross_pnl_pct,
                        "gross_pnl_r": gross_pnl_r,
                        "net_pnl_pct": net_pnl_pct,
                        "net_pnl_r": net_pnl_r,
                        "total_fees": total_fees,
                        "slippage_cost": slippage_cost,
                        "funding_cost": funding_cost,
                        "exit_reason": "trailing_stop_hit",
                        "ambiguous_same_bar": False
                    }
                    completed_trades.append(trade_dict)
                    active_pos = None
                    continue

            # -------------------------------------------------------------
            # 2. Check Breakout Entry Signal
            # -------------------------------------------------------------
            if active_pos is None:
                prev_close = closes[i - 1]
                prev_high_ref = donchian_high[i - 1]
                prev_vol = volumes[i - 1]
                prev_sma_vol = vol_sma20[i - 1]
                prev_ema50 = ema50[i - 1]

                breakout_confirmed = (prev_close > prev_high_ref) and (prev_close > prev_ema50)

                if breakout_confirmed:
                    if volume_filter_mult > 0.0:
                        if prev_vol < (volume_filter_mult * prev_sma_vol):
                            breakout_confirmed = False

                    if rs_filter_series is not None and min_rs_spread > -900.0:
                        if rs_arr[i - 1] < min_rs_spread:
                            breakout_confirmed = False

                if breakout_confirmed:
                    # Enter on open of bar i (taker market order at bar open)
                    entry_raw = bar_open
                    entry_t = curr_t

                    slip = exec_model.get_slippage_rate(rank_val) if exec_model else 0.001
                    fee = exec_model.taker_fee_rate if exec_model else 0.0002
                    fill_entry = entry_raw * (1.0 + slip)

                    init_stop_dist = initial_stop_atr_mult * bar_atr
                    init_stop_p = entry_raw - init_stop_dist

                    trade_seq += 1

                    active_pos = ActiveTrendPosition(
                        symbol=symbol,
                        entry_time=entry_t,
                        entry_bar_idx=i,
                        entry_price_raw=entry_raw,
                        entry_price=fill_entry,
                        initial_stop_price=init_stop_p,
                        current_stop_price=init_stop_p,
                        initial_risk_dist=init_stop_dist,
                        highest_since_entry=max(entry_raw, bar_high),
                        entry_fee=fee,
                        entry_slippage=slip,
                        bars_held=0
                    )

        # Close position if still open at end of sample
        if active_pos is not None:
            last_p = closes[n_bars - 1]
            last_t = open_times[n_bars - 1]
            slip = exec_model.get_slippage_rate(50) if exec_model else 0.001
            fee = exec_model.taker_fee_rate if exec_model else 0.0002
            fill_exit = last_p * (1.0 - slip)

            risk_pct = active_pos.initial_risk_dist / active_pos.entry_price_raw
            if risk_pct <= 0:
                risk_pct = 0.01

            gross_pnl_pct = (last_p - active_pos.entry_price_raw) / active_pos.entry_price_raw
            gross_pnl_r = gross_pnl_pct / risk_pct
            total_fees = active_pos.entry_fee + fee
            slippage_cost = active_pos.entry_slippage + slip
            funding_cost = exec_model.calculate_funding(active_pos.entry_time, last_t) if exec_model else 0.0

            net_price_ret = (fill_exit - active_pos.entry_price) / active_pos.entry_price
            net_pnl_pct = net_price_ret - total_fees - funding_cost
            net_pnl_r = net_pnl_pct / risk_pct

            trade_dict = {
                "trade_id": f"TF_{symbol}_{trade_seq}",
                "symbol": symbol,
                "entry_time": active_pos.entry_time,
                "exit_time": last_t,
                "bars_held": active_pos.bars_held,
                "holding_time_minutes": active_pos.bars_held * 240.0,
                "entry_price": active_pos.entry_price,
                "exit_price": fill_exit,
                "stop_price": active_pos.initial_stop_price,
                "gross_pnl_pct": gross_pnl_pct,
                "gross_pnl_r": gross_pnl_r,
                "net_pnl_pct": net_pnl_pct,
                "net_pnl_r": net_pnl_r,
                "total_fees": total_fees,
                "slippage_cost": slippage_cost,
                "funding_cost": funding_cost,
                "exit_reason": "end_of_data",
                "ambiguous_same_bar": False
            }
            completed_trades.append(trade_dict)

        return completed_trades
