"""
strategy/htf_swing_strategy.py
Higher Timeframe (4H / Daily) Swing-Retest Strategy Engine.
Implements Deep Discount Retests (61.8% - 78.6% Fib / 1.5 - 3.0 ATR)
with wide structural stops on 4-hour bars.
Strictly causal, zero-lookahead.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from strategy.indicators import compute_ema, compute_wilder_atr
from backtest.trade import Trade, Position
from backtest.execution_model import ExecutionModel
from backtest.position_manager import PositionManager


@dataclass
class HTFSetup:
    symbol: str
    swing_high_time: pd.Timestamp
    swing_high_price: float
    swing_low_time: pd.Timestamp
    swing_low_price: float
    impulse: float
    impulse_atr: float
    atr_4h: float
    trigger_bar_idx: int
    trigger_time: pd.Timestamp
    pullback_depth_atr: float
    retracement_pct: float
    reversal_bar_idx: Optional[int] = None
    reversal_time: Optional[pd.Timestamp] = None


class HTFSwingEngine:
    """
    Detects 4H swing structures and simulates deep discount retest trades.
    """

    def __init__(
        self,
        execution_model: ExecutionModel,
        fractal_n: int = 2,
        impulse_lookback: int = 30,
        max_holding_bars_4h: int = 42, # 7 days (42 * 4h = 168h)
        holdout_start: str = "2026-07-01T00:00:00Z"
    ):
        self.execution_model = execution_model
        self.fractal_n = fractal_n
        self.impulse_lookback = impulse_lookback
        self.max_holding_bars = max_holding_bars_4h
        self.holdout_start = pd.Timestamp(holdout_start, tz="UTC")

    def detect_deep_setups(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        atr_4h: pd.Series,
        retrace_target_pct: float = 61.8, # 61.8% Golden Ratio
        retrace_type: str = "fib",        # "fib" or "atr"
        depth_atr_target: float = 2.0
    ) -> List[HTFSetup]:
        """
        Scans 4H bars for confirmed swing highs and causal deep pullbacks in an uptrend.
        """
        if len(df_4h) < 60:
            return []

        # 4H Trend Filter: Close > 50 EMA on 4H, and 50 EMA rising
        ema50 = compute_ema(df_4h["close"], span=50).to_numpy(dtype=float)
        trend_up = (df_4h["close"].to_numpy(dtype=float) > ema50)

        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        opens = df_4h["open"].to_numpy(dtype=float)
        open_times = pd.to_datetime(df_4h["open_time"], utc=True).tolist()
        atrs = atr_4h.to_numpy(dtype=float)

        n_bars = len(df_4h)
        N = self.fractal_n
        L = self.impulse_lookback
        setups: List[HTFSetup] = []

        active_swing_high = None
        active_swing_high_p = 0.0
        active_swing_low_p = 0.0
        active_swing_low_t = None
        active_impulse = 0.0
        active_atr = 0.0
        active_sh_time = None
        armed = False
        running_low = float("inf")

        for i in range(L + N, n_bars - 1):
            curr_t = open_times[i]
            if curr_t >= self.holdout_start:
                break

            # 1. Check if bar i - N is a new confirmed Swing High
            sh_candidate_idx = i - N
            is_sh = True
            cand_high = highs[sh_candidate_idx]
            for offset in range(1, N + 1):
                if highs[sh_candidate_idx - offset] >= cand_high or highs[sh_candidate_idx + offset] >= cand_high:
                    is_sh = False
                    break

            if is_sh and trend_up[i]:
                # Measure preceding impulse from lowest low in [sh_candidate_idx - L, sh_candidate_idx]
                impulse_window_lows = lows[max(0, sh_candidate_idx - L) : sh_candidate_idx]
                if len(impulse_window_lows) > 0:
                    sl_p = float(np.min(impulse_window_lows))
                    sl_idx = max(0, sh_candidate_idx - L) + int(np.argmin(impulse_window_lows))
                    sl_t = open_times[sl_idx]
                    imp = cand_high - sl_p
                    atr_ref = atrs[sh_candidate_idx]

                    if imp > 0 and not np.isnan(atr_ref) and (imp / atr_ref >= 2.0):
                        active_swing_high = sh_candidate_idx
                        active_swing_high_p = cand_high
                        active_swing_low_p = sl_p
                        active_swing_low_t = sl_t
                        active_impulse = imp
                        active_atr = atr_ref
                        active_sh_time = open_times[sh_candidate_idx]
                        running_low = float("inf")
                        armed = True

            # 2. Check pullback into deep discount zone
            if armed:
                running_low = min(running_low, lows[i])

                # Check invalidation: break below impulse origin
                if running_low <= active_swing_low_p:
                    armed = False
                    continue

                # Measure depth
                depth_price = active_swing_high_p - running_low
                depth_pct = (depth_price / active_impulse) * 100.0 if active_impulse > 0 else 0.0
                depth_atr = depth_price / active_atr if active_atr > 0 else 0.0

                trigger_fired = False
                if retrace_type == "fib" and depth_pct >= retrace_target_pct:
                    trigger_fired = True
                elif retrace_type == "atr" and depth_atr >= depth_atr_target:
                    trigger_fired = True

                if trigger_fired and trend_up[i]:
                    # Find subsequent reversal confirmation candle if available
                    reversal_idx = None
                    reversal_time = None
                    # Scan forward up to 6 bars (24 hours) for a reversal candle
                    for fwd in range(i, min(i + 6, n_bars - 1)):
                        if closes[fwd] > opens[fwd] and closes[fwd] > highs[fwd - 1]:
                            reversal_idx = fwd
                            reversal_time = open_times[fwd]
                            break

                    setups.append(HTFSetup(
                        symbol=symbol,
                        swing_high_time=active_sh_time,
                        swing_high_price=active_swing_high_p,
                        swing_low_time=active_swing_low_t,
                        swing_low_price=active_swing_low_p,
                        impulse=active_impulse,
                        impulse_atr=active_impulse / active_atr if active_atr > 0 else 2.0,
                        atr_4h=active_atr,
                        trigger_bar_idx=i,
                        trigger_time=curr_t,
                        pullback_depth_atr=depth_atr,
                        retracement_pct=depth_pct,
                        reversal_bar_idx=reversal_idx,
                        reversal_time=reversal_time
                    ))
                    armed = False  # Reset after event

        return setups

    def simulate_htf_trades(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        atr_4h: pd.Series,
        setups: List[HTFSetup],
        entry_mode: str = "Entry_A",  # "Entry_A" (immediate) or "Entry_B" (4H reversal)
        stop_type: str = "origin",     # "origin", "wide_atr_2.5", "wide_atr_3.5", "pct_5.0"
        min_rr: float = 1.0
    ) -> List[Trade]:
        """
        Executes 4H trades bar-by-bar with position lock and realistic costs.
        """
        if not setups or len(df_4h) < 10:
            return []

        n_bars = len(df_4h)
        opens = df_4h["open"].to_numpy(dtype=float)
        highs = df_4h["high"].to_numpy(dtype=float)
        lows = df_4h["low"].to_numpy(dtype=float)
        closes = df_4h["close"].to_numpy(dtype=float)
        open_times = pd.to_datetime(df_4h["open_time"], utc=True).tolist()
        ranks = df_4h["rank"].to_numpy() if "rank" in df_4h.columns else [None] * n_bars
        atrs = atr_4h.to_numpy(dtype=float)

        # Map setups to execution signal bars
        pending_orders: Dict[int, HTFSetup] = {}
        for s in setups:
            if entry_mode == "Entry_B":
                if s.reversal_bar_idx is not None and s.reversal_bar_idx + 1 < n_bars:
                    pending_orders[s.reversal_bar_idx + 1] = s
            else:
                if s.trigger_bar_idx + 1 < n_bars:
                    pending_orders[s.trigger_bar_idx + 1] = s

        pos_mgr = PositionManager()
        active_pos: Optional[Position] = None
        completed_trades: List[Trade] = []
        trade_seq = 0

        for i in range(n_bars):
            curr_t = open_times[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_rank = int(ranks[i]) if (ranks[i] is not None and not pd.isna(ranks[i])) else None

            # 1. Process pending entry at bar open
            if i in pending_orders and active_pos is None:
                setup = pending_orders[i]
                if pos_mgr.can_open_position(symbol, curr_t, f"htf_{setup.trigger_time}"):
                    fill_p, slip_rate, fee_rate = self.execution_model.calculate_entry(bar_open, bar_rank)
                    atr_val = atrs[i - 1] if not np.isnan(atrs[i - 1]) and atrs[i - 1] > 0 else bar_open * 0.03

                    # Calculate Stop Loss
                    if stop_type == "origin":
                        stop_p = setup.swing_low_price - (0.25 * atr_val)
                    elif stop_type == "wide_atr_2.5":
                        stop_p = bar_open - (2.5 * atr_val)
                    elif stop_type == "wide_atr_3.5":
                        stop_p = bar_open - (3.5 * atr_val)
                    elif stop_type == "pct_5.0":
                        stop_p = bar_open * 0.95
                    else:
                        stop_p = setup.swing_low_price

                    target_p = setup.swing_high_price
                    potential_rr = (target_p - bar_open) / (bar_open - stop_p) if (bar_open > stop_p) else 0.0

                    if potential_rr >= min_rr and stop_p < bar_open and target_p > bar_open:
                        trade_seq += 1
                        active_pos = Position(
                            trade_id=f"HTF_{symbol}_{curr_t.strftime('%Y%m%d%H%M')}_{trade_seq}",
                            event_id=f"htf_{trade_seq}",
                            symbol=symbol,
                            entry_mode=f"HTF_{entry_mode}",
                            signal_time=open_times[i - 1],
                            entry_time=curr_t,
                            entry_bar_idx=i,
                            entry_price_raw=bar_open,
                            entry_price=fill_p,
                            stop_price=stop_p,
                            target_price=target_p,
                            potential_rr=potential_rr,
                            atr_ref=atr_val,
                            atr_entry=atr_val,
                            universe_rank=bar_rank,
                            volume_24h_usdt=float(df_4h["quote_volume_usdt"].iloc[i]) if "quote_volume_usdt" in df_4h.columns else 0.0,
                            trend_state="HTF_Uptrend",
                            trend_a=False, trend_b=False, trend_c=True, trend_d=False,
                            swing_high_time=setup.swing_high_time,
                            swing_high_price=setup.swing_high_price,
                            swing_low_time=setup.swing_low_time,
                            swing_low_price=setup.swing_low_price,
                            impulse=setup.impulse,
                            impulse_atr=setup.impulse_atr,
                            running_pullback_low=setup.swing_low_price,
                            pullback_depth_atr=setup.pullback_depth_atr,
                            retracement_pct=setup.retracement_pct,
                            entry_fee=fee_rate,
                            entry_slippage=slip_rate,
                            btc_regime="Unknown",
                            btc_above_ema50=False,
                            btc_above_ema200=False,
                            mfe_price=bar_high,
                            mae_price=bar_low,
                            bars_held=0
                        )
                        pos_mgr.open_position(active_pos)

            # 2. Check active trade lifecycle intra-bar
            if active_pos is not None:
                active_pos.bars_held += 1
                active_pos.mfe_price = max(active_pos.mfe_price, bar_high)
                active_pos.mae_price = min(active_pos.mae_price, bar_low)

                # Check target or stop
                hit_target = bar_high >= active_pos.target_price
                hit_stop = bar_low <= active_pos.stop_price

                exit_reason = None
                exit_raw_p = bar_close
                ambig = False

                if hit_target and hit_stop:
                    # Same-bar: conservative stop-first
                    ambig = True
                    exit_reason = "stop"
                    exit_raw_p = active_pos.stop_price
                elif hit_target:
                    exit_reason = "target"
                    exit_raw_p = active_pos.target_price
                elif hit_stop:
                    exit_reason = "stop"
                    exit_raw_p = active_pos.stop_price
                elif active_pos.bars_held >= self.max_holding_bars:
                    exit_reason = "time_stop"
                    exit_raw_p = bar_close

                if exit_reason is not None:
                    slip = self.execution_model.get_slippage_rate(bar_rank)
                    fee = self.execution_model.taker_fee_rate
                    fill_exit = exit_raw_p * (1.0 - slip) if exit_reason == "stop" else exit_raw_p * (1.0 - slip)

                    trade = self._finalize_htf_trade(
                        pos=active_pos,
                        exit_time=curr_t + pd.Timedelta(hours=4),
                        exit_reason=exit_reason,
                        exit_price_raw=exit_raw_p,
                        exit_price=fill_exit,
                        exit_fee=fee,
                        exit_slip=slip,
                        ambig=ambig
                    )
                    completed_trades.append(trade)
                    pos_mgr.close_position(symbol)
                    active_pos = None

        return completed_trades

    def _finalize_htf_trade(
        self,
        pos: Position,
        exit_time: pd.Timestamp,
        exit_reason: str,
        exit_price_raw: float,
        exit_price: float,
        exit_fee: float,
        exit_slip: float,
        ambig: bool
    ) -> Trade:
        risk_pct = (pos.entry_price - pos.stop_price) / pos.entry_price
        if risk_pct <= 0:
            risk_pct = 0.01

        gross_pnl_pct = (exit_price_raw - pos.entry_price_raw) / pos.entry_price_raw
        gross_pnl_r = gross_pnl_pct / risk_pct

        total_fees = pos.entry_fee + exit_fee
        slippage_cost = pos.entry_slippage + exit_slip
        funding_cost = self.execution_model.calculate_funding(pos.entry_time, exit_time)

        net_price_ret = (exit_price - pos.entry_price) / pos.entry_price
        net_pnl_pct = net_price_ret - total_fees - funding_cost
        net_pnl_r = net_pnl_pct / risk_pct

        holding_time_minutes = pos.bars_held * 240.0 # 4 hours per bar

        return Trade(
            trade_id=pos.trade_id,
            event_id=pos.event_id,
            symbol=pos.symbol,
            entry_mode=pos.entry_mode,
            signal_time=pos.signal_time,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            universe_rank=pos.universe_rank,
            volume_24h_usdt=pos.volume_24h_usdt,
            trend_state=pos.trend_state,
            trend_a=False, trend_b=False, trend_c=True, trend_d=False,
            swing_high_time=pos.swing_high_time,
            swing_high_price=pos.swing_high_price,
            swing_low_time=pos.swing_low_time,
            swing_low_price=pos.swing_low_price,
            atr_ref=pos.atr_ref,
            atr_entry=pos.atr_entry,
            impulse=pos.impulse,
            impulse_atr=pos.impulse_atr,
            running_pullback_low=pos.running_pullback_low,
            pullback_depth_atr=pos.pullback_depth_atr,
            retracement_pct=pos.retracement_pct,
            entry_price_raw=pos.entry_price_raw,
            entry_price=pos.entry_price,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            potential_rr=pos.potential_rr,
            exit_price_raw=exit_price_raw,
            exit_price=exit_price,
            exit_reason=exit_reason,
            bars_held=pos.bars_held,
            holding_time_minutes=holding_time_minutes,
            time_to_target_bars=pos.bars_held if exit_reason == "target" else None,
            time_to_target_minutes=holding_time_minutes if exit_reason == "target" else None,
            gross_pnl_pct=gross_pnl_pct,
            gross_pnl_r=gross_pnl_r,
            entry_fee=pos.entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            slippage_cost=slippage_cost,
            funding_cost=funding_cost,
            net_pnl_pct=net_pnl_pct,
            net_pnl_r=net_pnl_r,
            mfe_price=pos.mfe_price,
            mfe_atr=(pos.mfe_price - pos.entry_price) / pos.atr_entry if pos.atr_entry > 0 else 0.0,
            mfe_r=(pos.mfe_price - pos.entry_price) / (pos.entry_price - pos.stop_price) if pos.entry_price > pos.stop_price else 0.0,
            mae_price=pos.mae_price,
            mae_atr=(pos.entry_price - pos.mae_price) / pos.atr_entry if pos.atr_entry > 0 else 0.0,
            mae_r=(pos.entry_price - pos.mae_price) / (pos.entry_price - pos.stop_price) if pos.entry_price > pos.stop_price else 0.0,
            retested_high=pos.mfe_price >= pos.target_price,
            broke_high=pos.mfe_price >= (pos.target_price + 0.25 * pos.atr_ref),
            eventually_broke_pullback_low=pos.mae_price <= pos.stop_price,
            ambiguous_same_bar=ambig,
            btc_regime="Unknown",
            btc_above_ema50=False,
            btc_above_ema200=False
        )
