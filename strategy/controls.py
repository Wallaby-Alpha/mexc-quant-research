"""
strategy/controls.py
Implements Control experiments (Phase 5) evaluated through the exact same
ExecutionModel and cost structure as the primary strategy.

Controls:
A: Random entries during valid 1H uptrends (matched frequency, hold time, and risk profile)
B: Buy every X% pullback in an uptrend (X configurable, default 2%, 3%, 5%)
C: Buy when price crosses above 15m EMA in 1H uptrend (period configurable, default 20)
D: Buy after a generic bullish 15m candle in 1H uptrend
E: Swing-high strategy WITHOUT the 1H trend filter (trend_c == False)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

from backtest.trade import Trade, Position, TradeExitReason
from backtest.execution_model import ExecutionModel
from backtest.position_manager import PositionManager
from strategy.indicators import compute_ema, compute_wilder_atr

logger = logging.getLogger(__name__)


class ControlExperimentRunner:
    """
    Executes benchmark control strategies through the identical execution model,
    slippage, fees, funding, and position constraints.
    """

    def __init__(
        self,
        execution_model: ExecutionModel,
        holdout_start: str = "2026-07-01T00:00:00Z",
        max_hold_bars: int = 96
    ):
        self.execution_model = execution_model
        self.holdout_start = pd.Timestamp(holdout_start, tz="UTC")
        self.max_hold_bars = max_hold_bars

    def _prepare_df(self, df_15m: pd.DataFrame, allow_holdout: bool = False) -> pd.DataFrame:
        df = df_15m.copy()
        if not allow_holdout:
            df = df[pd.to_datetime(df["open_time"], utc=True) < self.holdout_start]
        return df

    def run_control_a_random(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        target_trades_count: int,
        mean_target_atr: float = 1.75,
        mean_stop_atr: float = 1.25,
        random_seed: int = 42,
        allow_holdout: bool = False
    ) -> List[Trade]:
        """
        Control A: Random entries during valid 1H uptrends (Trend C).
        Matched on frequency, stop/target distance distribution, and hold time.
        """
        df = self._prepare_df(df_15m, allow_holdout)
        if df.empty or len(df) < self.max_hold_bars + 10:
            return []

        trend_c = df["trend_c"].to_numpy(dtype=bool) if "trend_c" in df.columns else np.zeros(len(df), dtype=bool)
        eligible_bars = np.where(trend_c)[0]
        eligible_bars = eligible_bars[(eligible_bars >= 10) & (eligible_bars < len(df) - self.max_hold_bars)]

        if len(eligible_bars) == 0:
            return []

        rng = np.random.default_rng(random_seed)
        n_samples = min(len(eligible_bars), max(1, target_trades_count))
        # Stochastically select candidate signal bars
        candidate_bars = np.sort(rng.choice(eligible_bars, size=min(len(eligible_bars), n_samples * 2), replace=False))

        return self._simulate_custom_signals(
            symbol=symbol,
            df_15m=df,
            atr_series=atr_series,
            signal_bars=candidate_bars,
            target_atr_dist=mean_target_atr,
            stop_atr_dist=mean_stop_atr,
            control_name="Control_A_Random"
        )

    def run_control_b_pct_pullback(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        pullback_pct: float = 0.03,  # 3% pullback
        lookback_bars: int = 32,
        allow_holdout: bool = False
    ) -> List[Trade]:
        """
        Control B: Buy every X% pullback from rolling high during 1H uptrend.
        """
        df = self._prepare_df(df_15m, allow_holdout)
        if len(df) < lookback_bars + 10:
            return []

        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        trend_c = df["trend_c"].to_numpy(dtype=bool) if "trend_c" in df.columns else np.zeros(len(df), dtype=bool)

        signal_bars = []
        for i in range(lookback_bars, len(df) - 1):
            if not trend_c[i]:
                continue
            recent_high = np.max(highs[i - lookback_bars : i])
            if recent_high > 0:
                depth = (recent_high - closes[i]) / recent_high
                if depth >= pullback_pct:
                    signal_bars.append(i)

        return self._simulate_custom_signals(
            symbol=symbol,
            df_15m=df,
            atr_series=atr_series,
            signal_bars=np.array(signal_bars, dtype=int),
            target_atr_dist=1.75,
            stop_atr_dist=1.25,
            control_name=f"Control_B_Pullback_{int(pullback_pct*100)}pct"
        )

    def run_control_c_ema_cross(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        ema_period: int = 20,
        allow_holdout: bool = False
    ) -> List[Trade]:
        """
        Control C: Buy when 15m price crosses above 15m EMA in 1H uptrend.
        """
        df = self._prepare_df(df_15m, allow_holdout)
        if len(df) < ema_period + 10:
            return []

        ema_15m = compute_ema(df["close"], span=ema_period).to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        trend_c = df["trend_c"].to_numpy(dtype=bool) if "trend_c" in df.columns else np.zeros(len(df), dtype=bool)

        signal_bars = []
        for i in range(1, len(df) - 1):
            if not trend_c[i]:
                continue
            # Causal cross: previous close <= EMA, current close > EMA
            if closes[i - 1] <= ema_15m[i - 1] and closes[i] > ema_15m[i]:
                signal_bars.append(i)

        return self._simulate_custom_signals(
            symbol=symbol,
            df_15m=df,
            atr_series=atr_series,
            signal_bars=np.array(signal_bars, dtype=int),
            target_atr_dist=1.75,
            stop_atr_dist=1.25,
            control_name="Control_C_EMA20_Cross"
        )

    def run_control_d_bullish_candle(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        allow_holdout: bool = False
    ) -> List[Trade]:
        """
        Control D: Buy after generic bullish 15m candle (close > open, strong body) in 1H uptrend.
        """
        df = self._prepare_df(df_15m, allow_holdout)
        if len(df) < 10:
            return []

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        trend_c = df["trend_c"].to_numpy(dtype=bool) if "trend_c" in df.columns else np.zeros(len(df), dtype=bool)

        signal_bars = []
        for i in range(len(df) - 1):
            if not trend_c[i]:
                continue
            bar_range = highs[i] - lows[i]
            if bar_range <= 0:
                continue
            body = closes[i] - opens[i]
            # Strong bullish candle: green body >= 50% of range and close in top 30%
            if body > 0.5 * bar_range and (highs[i] - closes[i]) < 0.3 * bar_range:
                signal_bars.append(i)

        return self._simulate_custom_signals(
            symbol=symbol,
            df_15m=df,
            atr_series=atr_series,
            signal_bars=np.array(signal_bars, dtype=int),
            target_atr_dist=1.75,
            stop_atr_dist=1.25,
            control_name="Control_D_Bullish_Candle"
        )

    def _simulate_custom_signals(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        atr_series: pd.Series,
        signal_bars: np.ndarray,
        target_atr_dist: float,
        stop_atr_dist: float,
        control_name: str
    ) -> List[Trade]:
        """
        Simulates custom signals with position lock (1 open trade per symbol),
        conservative same-bar resolution, realistic fees, slippage, and funding.
        """
        if len(signal_bars) == 0:
            return []

        n_bars = len(df_15m)
        open_times = pd.to_datetime(df_15m["open_time"], utc=True).tolist()
        opens = df_15m["open"].to_numpy(dtype=float)
        highs = df_15m["high"].to_numpy(dtype=float)
        lows = df_15m["low"].to_numpy(dtype=float)
        closes = df_15m["close"].to_numpy(dtype=float)
        volumes = df_15m["quote_volume_usdt"].to_numpy(dtype=float) if "quote_volume_usdt" in df_15m.columns else np.zeros(n_bars)
        ranks = df_15m["rank"].to_numpy() if "rank" in df_15m.columns else [None] * n_bars
        atrs = atr_series.to_numpy(dtype=float)

        sig_set = set(signal_bars)
        completed_trades: List[Trade] = []
        pos_mgr = PositionManager()
        active_pos: Optional[Position] = None
        pending_exit: Optional[Tuple[str, str]] = None
        pending_signal_bar: Optional[int] = None

        trade_seq = 0

        for i in range(n_bars):
            curr_time = open_times[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_rank = int(ranks[i]) if (ranks[i] is not None and not pd.isna(ranks[i])) else None
            bar_vol = float(volumes[i])

            # 1. Process pending exit at open (e.g. time stop)
            if active_pos is not None and pending_exit is not None:
                reason, _ = pending_exit
                trade = self._build_trade(
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
                pos_mgr.close_position(symbol)
                active_pos = None
                pending_exit = None

            # 2. Process pending entry at open
            if pending_signal_bar is not None and pending_signal_bar == i - 1:
                if active_pos is None and pos_mgr.can_open_position(symbol, curr_time, f"ctrl_{i}"):
                    fill_price, slip_rate, fee_rate = self.execution_model.calculate_entry(bar_open, bar_rank)
                    atr_val = atrs[i - 1] if not np.isnan(atrs[i - 1]) and atrs[i - 1] > 0 else bar_open * 0.01

                    target_price = bar_open + (target_atr_dist * atr_val)
                    stop_price = bar_open - (stop_atr_dist * atr_val)
                    potential_rr = (target_price - bar_open) / (bar_open - stop_price) if (bar_open > stop_price) else 1.0

                    trade_seq += 1
                    active_pos = Position(
                        trade_id=f"{symbol}_{curr_time.strftime('%Y%m%d%H%M')}_{trade_seq}",
                        event_id=f"ctrl_{trade_seq}",
                        symbol=symbol,
                        entry_mode=control_name,
                        signal_time=open_times[i - 1],
                        entry_time=curr_time,
                        entry_bar_idx=i,
                        entry_price_raw=bar_open,
                        entry_price=fill_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        potential_rr=potential_rr,
                        atr_ref=atr_val,
                        atr_entry=atr_val,
                        universe_rank=bar_rank,
                        volume_24h_usdt=bar_vol,
                        trend_state="Control",
                        trend_a=False, trend_b=False, trend_c=True, trend_d=False,
                        swing_high_time=curr_time,
                        swing_high_price=target_price,
                        swing_low_time=curr_time,
                        swing_low_price=stop_price,
                        impulse=target_price - stop_price,
                        impulse_atr=target_atr_dist + stop_atr_dist,
                        running_pullback_low=bar_low,
                        pullback_depth_atr=stop_atr_dist,
                        retracement_pct=50.0,
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
                pending_signal_bar = None

            # 3. Active position intra-bar check
            if active_pos is not None:
                active_pos.bars_held += 1
                active_pos.mfe_price = max(active_pos.mfe_price, bar_high)
                active_pos.mae_price = min(active_pos.mae_price, bar_low)

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
                    trade = self._build_trade(
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
                    pos_mgr.close_position(symbol)
                    active_pos = None
                elif active_pos.bars_held >= self.max_hold_bars:
                    pending_exit = (TradeExitReason.TIME_STOP.value, curr_time)

            # 4. Signal check at bar close
            if i in sig_set:
                pending_signal_bar = i

        return completed_trades

    def _build_trade(
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

        risk_pct = (pos.entry_price - pos.stop_price) / pos.entry_price
        if risk_pct <= 0:
            risk_pct = 0.001

        gross_pnl_pct = (exit_price_raw - pos.entry_price_raw) / pos.entry_price_raw
        gross_pnl_r = (exit_price_raw - pos.entry_price_raw) / (pos.entry_price_raw - pos.stop_price) if (pos.entry_price_raw > pos.stop_price) else 0.0

        total_fees = pos.entry_fee + exit_fee_rate
        slippage_cost = pos.entry_slippage + exit_slippage_rate
        funding_cost = self.execution_model.calculate_funding(pos.entry_time, exit_time)

        net_price_return = (exit_price - pos.entry_price) / pos.entry_price
        net_pnl_pct = net_price_return - total_fees - funding_cost
        net_pnl_r = net_pnl_pct / risk_pct

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
            bars_held=bars_held,
            holding_time_minutes=holding_time_minutes,
            time_to_target_bars=bars_held if exit_reason == "target" else None,
            time_to_target_minutes=holding_time_minutes if exit_reason == "target" else None,
            gross_pnl_pct=gross_pnl_pct,
            gross_pnl_r=gross_pnl_r,
            entry_fee=pos.entry_fee,
            exit_fee=exit_fee_rate,
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
