from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backtest.trade import Trade, Position, TradeExitReason, EntryMode, StopType
from backtest.execution_model import ExecutionModel
from backtest.position_manager import PositionManager
from strategy.entry_rules import BaseEntryRule, EntrySignal, EntryA_Immediate
from strategy.exit_rules import BaseStopRule, PullbackBufferStop, ExitRules
from strategy.pullback_detector import ThresholdCrossingEvent

logger = logging.getLogger(__name__)


class HoldoutProtectionError(Exception):
    """Raised when data past the holdout boundary is accessed outside Phase 5."""
    pass


class BacktestEngine:
    """
    Event-driven, bar-by-bar, deterministic trade simulation engine (DEFINITIONS.md §13 & §14).
    Enforces non-lookahead, strict holdout discipline, and 1-position-per-symbol constraint.
    """

    def __init__(
        self,
        execution_model: ExecutionModel,
        entry_rule: BaseEntryRule,
        exit_rules: ExitRules,
        holdout_start: str = "2026-07-01T00:00:00Z"
    ):
        self.execution_model = execution_model
        self.entry_rule = entry_rule
        self.exit_rules = exit_rules
        self.holdout_start = pd.Timestamp(holdout_start, tz="UTC")
        self.position_manager = PositionManager()
        self.rejected_min_rr: List[Dict[str, Any]] = []

    def simulate_symbol(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        events: List[ThresholdCrossingEvent],
        btc_regime_df: Optional[pd.DataFrame] = None,
        allow_holdout: bool = False
    ) -> List[Trade]:
        """
        Runs bar-by-bar deterministic simulation for a single symbol.
        """
        if df_15m.empty or not events:
            return []

        # Holdout verification
        max_time = pd.to_datetime(df_15m["open_time"].max(), utc=True)
        if not allow_holdout and max_time >= self.holdout_start:
            # Filter non-holdout slice
            df_15m = df_15m[pd.to_datetime(df_15m["open_time"], utc=True) < self.holdout_start].copy()
            events = [e for e in events if e.event_time < self.holdout_start]
            if df_15m.empty or not events:
                return []

        # Map events by event_bar_idx and event_id for O(1) instant lookup
        events_by_bar: Dict[int, List[ThresholdCrossingEvent]] = {}
        events_by_id: Dict[str, ThresholdCrossingEvent] = {}
        for ev in events:
            events_by_bar.setdefault(ev.event_bar_idx, []).append(ev)
            events_by_id[ev.event_id] = ev

        # Pre-extract BTC regime numpy arrays for O(1) binary search
        if btc_regime_df is not None and not btc_regime_df.empty:
            btc_times = pd.to_datetime(btc_regime_df["open_time"], utc=True).to_numpy(dtype="datetime64[ns]")
            btc_bull = btc_regime_df["btc_bull_uptrend"].to_numpy(dtype=bool) if "btc_bull_uptrend" in btc_regime_df.columns else np.zeros(len(btc_regime_df), dtype=bool)
            btc_ema50 = btc_regime_df["btc_above_ema50"].to_numpy(dtype=bool) if "btc_above_ema50" in btc_regime_df.columns else np.zeros(len(btc_regime_df), dtype=bool)
            btc_ema200 = btc_regime_df["btc_above_ema200"].to_numpy(dtype=bool) if "btc_above_ema200" in btc_regime_df.columns else np.zeros(len(btc_regime_df), dtype=bool)
        else:
            btc_times = None

        # Pre-extract arrays for speed
        n_bars = len(df_15m)
        open_times = pd.to_datetime(df_15m["open_time"], utc=True).tolist()
        opens = df_15m["open"].to_numpy(dtype=float)
        highs = df_15m["high"].to_numpy(dtype=float)
        lows = df_15m["low"].to_numpy(dtype=float)
        closes = df_15m["close"].to_numpy(dtype=float)
        volumes = df_15m["quote_volume_usdt"].to_numpy(dtype=float) if "quote_volume_usdt" in df_15m.columns else np.zeros(n_bars)
        ranks = df_15m["rank"].to_numpy() if "rank" in df_15m.columns else [None] * n_bars
        trend_invs = df_15m["trend_invalidated"].to_numpy(dtype=bool) if "trend_invalidated" in df_15m.columns else np.zeros(n_bars, dtype=bool)
        atrs = atr_series.to_numpy(dtype=float)
        arrays = (opens, highs, lows, closes, open_times, atrs)

        completed_trades: List[Trade] = []
        active_pos: Optional[Position] = None
        pending_exit: Optional[Tuple[str, str]] = None  # (reason, exit_time)

        # Track candidate entry signals generated from events
        pending_signals: List[EntrySignal] = []

        trade_seq = 0

        for i in range(n_bars):
            curr_time = open_times[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_rank = int(ranks[i]) if (ranks[i] is not None and not pd.isna(ranks[i])) else None
            bar_vol = float(volumes[i])

            # ------------------------------------------------------------------
            # 1. Process Pending Market Exits at Bar Open (Time Stop / Trend Exit)
            # ------------------------------------------------------------------
            if active_pos is not None and pending_exit is not None:
                reason, _ = pending_exit
                trade = self._finalize_trade(
                    pos=active_pos,
                    exit_time=curr_time,
                    exit_reason=reason,
                    exit_price_raw=bar_open,
                    exit_price=bar_open * (1.0 - self.execution_model.get_slippage_rate(bar_rank)),
                    exit_fee_rate=self.execution_model.taker_fee_rate,
                    exit_slippage_rate=self.execution_model.get_slippage_rate(bar_rank),
                    ambiguous_same_bar=False,
                    last_bar_high=bar_high,
                    last_bar_low=bar_low
                )
                completed_trades.append(trade)
                self.position_manager.close_position(symbol)
                active_pos = None
                pending_exit = None

            # ------------------------------------------------------------------
            # 2. Process Pending Entry Orders at Bar Open
            # ------------------------------------------------------------------
            if pending_signals:
                # Find valid candidate for this bar
                sig_to_execute = None
                remaining_sigs = []
                for sig in pending_signals:
                    if sig.entry_bar_idx == i:
                        if not sig.is_valid_rr:
                            self.rejected_min_rr.append({
                                "event_id": sig.event_id,
                                "symbol": symbol,
                                "time": curr_time,
                                "rr": sig.potential_rr,
                                "min_rr": self.entry_rule.min_rr
                            })
                        elif self.position_manager.can_open_position(symbol, curr_time, sig.event_id):
                            if active_pos is None and sig_to_execute is None:
                                sig_to_execute = sig
                    elif sig.entry_bar_idx > i:
                        remaining_sigs.append(sig)

                pending_signals = remaining_sigs

                if sig_to_execute is not None:
                    # Execute entry at open of bar i
                    fill_price, slip_rate, fee_rate = self.execution_model.calculate_entry(bar_open, bar_rank)
                    trade_seq += 1
                    trade_id = f"{symbol}_{curr_time.strftime('%Y%m%d%H%M')}_{trade_seq}"

                    # Fast O(1) parent event lookup
                    ev = events_by_id.get(sig_to_execute.event_id)

                    # Fast O(1) binary search for BTC regime
                    btc_reg = "BTC_Unknown"
                    btc_e50 = False
                    btc_e200 = False
                    if btc_times is not None:
                        curr_np = curr_time.asm8
                        b_idx = np.searchsorted(btc_times, curr_np, side="right") - 1
                        if b_idx >= 0:
                            btc_reg = "BTC_Bull_Uptrend" if btc_bull[b_idx] else "BTC_Non_Uptrend"
                            btc_e50 = bool(btc_ema50[b_idx])
                            btc_e200 = bool(btc_ema200[b_idx])

                    active_pos = Position(
                        trade_id=trade_id,
                        event_id=sig_to_execute.event_id,
                        symbol=symbol,
                        entry_mode=sig_to_execute.entry_mode,
                        signal_time=sig_to_execute.signal_time,
                        entry_time=curr_time,
                        entry_bar_idx=i,
                        entry_price_raw=bar_open,
                        entry_price=fill_price,
                        stop_price=sig_to_execute.stop_price,
                        target_price=sig_to_execute.target_price,
                        potential_rr=sig_to_execute.potential_rr,
                        atr_ref=ev.atr_ref if ev else sig_to_execute.atr_entry,
                        atr_entry=sig_to_execute.atr_entry,
                        universe_rank=bar_rank,
                        volume_24h_usdt=bar_vol,
                        trend_state="Trend_C" if (ev and ev.trend_c) else "Trend_Other",
                        trend_a=ev.trend_a if ev else False,
                        trend_b=ev.trend_b if ev else False,
                        trend_c=ev.trend_c if ev else False,
                        trend_d=ev.trend_d if ev else False,
                        swing_high_time=ev.swing_high_time if ev else curr_time,
                        swing_high_price=ev.swing_high_price if ev else sig_to_execute.target_price,
                        swing_low_time=ev.swing_low_time if ev else curr_time,
                        swing_low_price=ev.swing_low_price if ev else sig_to_execute.stop_price,
                        impulse=ev.impulse if ev else (sig_to_execute.target_price - sig_to_execute.stop_price),
                        impulse_atr=ev.impulse_atr if ev else 2.0,
                        running_pullback_low=ev.running_pullback_low if ev else bar_low,
                        pullback_depth_atr=ev.pullback_depth_atr if ev else 0.5,
                        retracement_pct=ev.retracement_pct if ev else 30.0,
                        entry_fee=fee_rate,
                        entry_slippage=slip_rate,
                        btc_regime=btc_reg,
                        btc_above_ema50=btc_e50,
                        btc_above_ema200=btc_e200,
                        mfe_price=bar_high,
                        mae_price=bar_low,
                        bars_held=0
                    )
                    self.position_manager.open_position(active_pos)


            # ------------------------------------------------------------------
            # 3. Active Position Intra-Bar Lifecycle Check
            # ------------------------------------------------------------------
            if active_pos is not None:
                active_pos.bars_held += 1
                active_pos.mfe_price = max(active_pos.mfe_price, bar_high)
                active_pos.mae_price = min(active_pos.mae_price, bar_low)

                # Check intra-bar target or stop exit
                bar_exit = self.execution_model.evaluate_bar_exit(
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    stop_price=active_pos.stop_price,
                    target_price=active_pos.target_price,
                    universe_rank=bar_rank
                )

                if bar_exit is not None:
                    reason, raw_price, fill_price, slip, fee, ambig = bar_exit
                    trade = self._finalize_trade(
                        pos=active_pos,
                        exit_time=curr_time + pd.Timedelta(minutes=15),
                        exit_reason=reason,
                        exit_price_raw=raw_price,
                        exit_price=fill_price,
                        exit_fee_rate=fee,
                        exit_slippage_rate=slip,
                        ambiguous_same_bar=ambig,
                        last_bar_high=bar_high,
                        last_bar_low=bar_low
                    )
                    completed_trades.append(trade)
                    self.position_manager.close_position(symbol)
                    active_pos = None
                else:
                    # Check Time Stop or Trend Invalidation exit to take effect next open
                    if self.exit_rules.is_time_stop_reached(active_pos.bars_held):
                        pending_exit = (TradeExitReason.TIME_STOP.value, curr_time)
                    elif self.exit_rules.is_trend_invalidated(trend_invs[i]):
                        pending_exit = (TradeExitReason.TREND_INVALIDATION.value, curr_time)

            # ------------------------------------------------------------------
            # 4. End-of-Bar Decision Time: Scan Events & Generate Entry Signals
            # ------------------------------------------------------------------
            if i in events_by_bar:
                for ev in events_by_bar[i]:
                    stop_lvl = self.exit_rules.get_initial_stop(
                        running_pullback_low=ev.running_pullback_low,
                        swing_low_price=ev.swing_low_price,
                        entry_price=bar_close,
                        atr=ev.atr_entry
                    )
                    target_lvl = ev.swing_high_price

                    sig = self.entry_rule.evaluate(
                        event=ev,
                        df_15m=df_15m,
                        stop_level=stop_lvl,
                        target_price=target_lvl,
                        atr_series=atr_series,
                        arrays=arrays
                    )
                    if sig is not None:
                        pending_signals.append(sig)

        return completed_trades

    def _finalize_trade(
        self,
        pos: Position,
        exit_time: pd.Timestamp,
        exit_reason: str,
        exit_price_raw: float,
        exit_price: float,
        exit_fee_rate: float,
        exit_slippage_rate: float,
        ambiguous_same_bar: bool,
        last_bar_high: float,
        last_bar_low: float
    ) -> Trade:
        pos.mfe_price = max(pos.mfe_price, last_bar_high)
        pos.mae_price = min(pos.mae_price, last_bar_low)

        # Financial PnL arithmetic (Gross & Net)
        risk_pct = (pos.entry_price - pos.stop_price) / pos.entry_price
        if risk_pct <= 0:
            risk_pct = 0.001  # Guard against zero-division

        gross_pnl_pct = (exit_price_raw - pos.entry_price_raw) / pos.entry_price_raw
        gross_pnl_r = (exit_price_raw - pos.entry_price_raw) / (pos.entry_price_raw - pos.stop_price) if (pos.entry_price_raw > pos.stop_price) else 0.0

        entry_fee = pos.entry_fee
        exit_fee = exit_fee_rate
        total_fees = entry_fee + exit_fee
        slippage_cost = pos.entry_slippage + exit_slippage_rate
        funding_cost = self.execution_model.calculate_funding(pos.entry_time, exit_time)

        # Net Return: price spread with slippages minus total fees minus funding
        net_price_return = (exit_price - pos.entry_price) / pos.entry_price
        net_pnl_pct = net_price_return - total_fees - funding_cost
        net_pnl_r = net_pnl_pct / risk_pct

        # Excursion calculations
        mfe_diff = max(0.0, pos.mfe_price - pos.entry_price)
        mae_diff = max(0.0, pos.entry_price - pos.mae_price)

        mfe_atr = mfe_diff / pos.atr_entry if pos.atr_entry > 0 else 0.0
        mfe_r = mfe_diff / (pos.entry_price - pos.stop_price) if (pos.entry_price > pos.stop_price) else 0.0

        mae_atr = mae_diff / pos.atr_entry if pos.atr_entry > 0 else 0.0
        mae_r = mae_diff / (pos.entry_price - pos.stop_price) if (pos.entry_price > pos.stop_price) else 0.0

        retested_high = pos.mfe_price >= pos.target_price
        broke_high = pos.mfe_price >= (pos.target_price + 0.25 * pos.atr_ref)
        eventually_broke_low = pos.mae_price <= pos.stop_price

        bars_held = pos.bars_held
        holding_time_minutes = bars_held * 15.0

        time_to_target_bars = bars_held if exit_reason == "target" else None
        time_to_target_minutes = holding_time_minutes if exit_reason == "target" else None

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
            trend_a=pos.trend_a,
            trend_b=pos.trend_b,
            trend_c=pos.trend_c,
            trend_d=pos.trend_d,
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
            bars_held=bars_held,
            holding_time_minutes=holding_time_minutes,
            time_to_target_bars=time_to_target_bars,
            time_to_target_minutes=time_to_target_minutes,
            gross_pnl_pct=gross_pnl_pct,
            gross_pnl_r=gross_pnl_r,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            slippage_cost=slippage_cost,
            funding_cost=funding_cost,
            net_pnl_pct=net_pnl_pct,
            net_pnl_r=net_pnl_r,
            mfe_price=pos.mfe_price,
            mfe_atr=mfe_atr,
            mfe_r=mfe_r,
            mae_price=pos.mae_price,
            mae_atr=mae_atr,
            mae_r=mae_r,
            retested_high=retested_high,
            broke_high=broke_high,
            eventually_broke_pullback_low=eventually_broke_low,
            ambiguous_same_bar=ambiguous_same_bar,
            btc_regime=pos.btc_regime,
            btc_above_ema50=pos.btc_above_ema50,
            btc_above_ema200=pos.btc_above_ema200
        )

    def _find_event(self, events: List[ThresholdCrossingEvent], event_id: str) -> Optional[ThresholdCrossingEvent]:
        for e in events:
            if e.event_id == event_id:
                return e
        return None

    def _get_btc_snapshot(self, btc_regime_df: Optional[pd.DataFrame], t: pd.Timestamp) -> Dict[str, Any]:
        if btc_regime_df is None or btc_regime_df.empty:
            return {"regime": "BTC_Unknown", "above_ema50": False, "above_ema200": False}

        matched = btc_regime_df[btc_regime_df["open_time"] <= t]
        if matched.empty:
            return {"regime": "BTC_Unknown", "above_ema50": False, "above_ema200": False}

        last = matched.iloc[-1]
        regime = "BTC_Bull_Uptrend" if last.get("btc_bull_uptrend", False) else "BTC_Non_Uptrend"
        return {
            "regime": regime,
            "above_ema50": bool(last.get("btc_above_ema50", False)),
            "above_ema200": bool(last.get("btc_above_ema200", False))
        }

    @staticmethod
    def log_trial(
        trial_record: Dict[str, Any],
        results_dir: str | Path = "results"
    ):
        """
        Records parameter run in results/trials.parquet per Rule 6 (Multiple-Testing Honesty).
        """
        p = Path(results_dir) / "trials.parquet"
        df_new = pd.DataFrame([trial_record])
        if p.is_file():
            df_old = pd.read_parquet(p)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(p, index=False)
