"""
strategy/rotational_momentum.py
Cross-Sectional Momentum & Relative Strength Rotational Portfolio Engine.
Systematically ranks the universe of altcoins by trailing performance vs. Bitcoin,
holds the Top K leaders, and rebalances at regular cadences with causal cost modeling.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel


@dataclass
class RebalanceSnapshot:
    timestamp: pd.Timestamp
    portfolio_return_gross: float
    portfolio_return_net: float
    turnover: float
    fee_drag: float
    selected_symbols: List[str]
    is_cash: bool
    btc_return: float
    market_equal_weight_return: float


class RotationalMomentumEngine:
    """
    Simulates multi-asset cross-sectional momentum portfolios across daily klines.
    """

    @staticmethod
    def run_backtest(
        price_matrix: pd.DataFrame, # Columns: symbols, Index: daily UTC Timestamp, Values: close price
        btc_series: pd.Series, # Index: daily UTC Timestamp, Values: BTC close price
        lookback_days: int = 14, # Momentum lookback window
        top_k: int = 10, # Number of coins to hold
        rebalance_days: int = 7, # Rebalance frequency in days
        btc_filter: str = "none", # "none", "btc_above_ema50", "btc_above_ema200"
        negative_control_bottom_k: bool = False, # If True, holds the WORST coins
        exec_model: Optional[ExecutionModel] = None,
        holdout_ts: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Executes causal portfolio rebalancing simulation.
        """
        # Ensure aligned dates
        common_idx = price_matrix.index.intersection(btc_series.index).sort_values()
        if holdout_ts is not None:
            common_idx = common_idx[common_idx < holdout_ts]

        if len(common_idx) < lookback_days + rebalance_days + 10:
            return {}

        df_prices = price_matrix.loc[common_idx]
        s_btc = btc_series.loc[common_idx]

        # Precompute BTC EMA
        btc_ema50 = compute_ema(s_btc, span=50)
        btc_ema200 = compute_ema(s_btc, span=200)

        # Precompute daily return matrix
        daily_returns = df_prices.pct_change(fill_method=None)
        btc_daily_returns = s_btc.pct_change()

        # Track portfolio state
        current_weights: Dict[str, float] = {}
        snapshots: List[RebalanceSnapshot] = []

        # Rebalance dates: start after lookback
        rebalance_indices = range(lookback_days, len(common_idx) - rebalance_days, rebalance_days)

        portfolio_equity = 1.0
        equity_curve = [portfolio_equity]
        equity_dates = [common_idx[lookback_days]]

        taker_fee = exec_model.taker_fee_rate if exec_model else 0.0002
        base_slip = (exec_model.base_slippage_bps / 10000.0) if exec_model else 0.0010
        total_one_way_cost = taker_fee + base_slip

        for idx in rebalance_indices:
            t_decision = common_idx[idx]
            t_next = common_idx[idx + rebalance_days]

            # 1. Causal Trailing Returns (strictly up to t_decision)
            t_lookback_start = common_idx[idx - lookback_days]
            p_start = df_prices.loc[t_lookback_start]
            p_end = df_prices.loc[t_decision]

            # Available symbols on this date (must have valid prices at both start and end)
            valid_mask = (p_start > 0) & (p_end > 0) & (~p_start.isna()) & (~p_end.isna())
            valid_symbols = valid_mask[valid_mask].index.tolist()

            if len(valid_symbols) < top_k:
                continue

            trailing_returns = (p_end[valid_symbols] - p_start[valid_symbols]) / p_start[valid_symbols]
            btc_trailing = (s_btc.loc[t_decision] - s_btc.loc[t_lookback_start]) / s_btc.loc[t_lookback_start]

            # Relative Strength spread vs BTC
            rs_spread = trailing_returns - btc_trailing

            # 2. BTC Macro Filter Check
            is_cash = False
            if btc_filter == "btc_above_ema50":
                if s_btc.loc[t_decision] < btc_ema50.loc[t_decision]:
                    is_cash = True
            elif btc_filter == "btc_above_ema200":
                if s_btc.loc[t_decision] < btc_ema200.loc[t_decision]:
                    is_cash = True

            # 3. Selection: Top K (or Bottom K if negative control)
            if negative_control_bottom_k:
                selected = rs_spread.nsmallest(top_k).index.tolist()
            else:
                selected = rs_spread.nlargest(top_k).index.tolist()

            target_weights: Dict[str, float] = {}
            if is_cash:
                # 100% Cash / USDT
                target_weights = {}
            else:
                weight_per_coin = 1.0 / len(selected)
                for sym in selected:
                    target_weights[sym] = weight_per_coin

            # 4. Calculate Portfolio Turnover & Transaction Costs
            all_symbols = set(current_weights.keys()).union(set(target_weights.keys()))
            turnover = 0.0
            for sym in all_symbols:
                w_old = current_weights.get(sym, 0.0)
                w_new = target_weights.get(sym, 0.0)
                turnover += abs(w_new - w_old)
            turnover = turnover / 2.0 # Standard one-way portfolio turnover

            fee_drag = turnover * 2.0 * total_one_way_cost # Cost to rotate out of old and into new

            # 5. Measure Forward Return from t_decision to t_next
            # Forward asset returns over the holding period
            p_fwd_start = df_prices.loc[t_decision]
            p_fwd_end = df_prices.loc[t_next]
            fwd_asset_returns = (p_fwd_end - p_fwd_start) / p_fwd_start

            if is_cash:
                fwd_gross_ret = 0.0
            else:
                # Equal-weighted return of selected coins
                fwd_rets = [fwd_asset_returns[s] for s in selected if s in fwd_asset_returns and not np.isnan(fwd_asset_returns[s])]
                fwd_gross_ret = float(np.mean(fwd_rets)) if len(fwd_rets) > 0 else 0.0

            # Forward BTC return
            btc_fwd_ret = (s_btc.loc[t_next] - s_btc.loc[t_decision]) / s_btc.loc[t_decision]

            # Forward equal-weight universe return (market benchmark)
            mkt_fwd_rets = fwd_asset_returns[valid_symbols].dropna()
            mkt_fwd_ret = float(np.mean(mkt_fwd_rets)) if len(mkt_fwd_rets) > 0 else 0.0

            # Forward funding cost deduction (1 bps per 8 hours = 3 bps per day)
            funding_drag = (0.0001 * 3.0 * rebalance_days) if not is_cash else 0.0

            fwd_net_ret = fwd_gross_ret - fee_drag - funding_drag

            portfolio_equity = portfolio_equity * (1.0 + fwd_net_ret)
            equity_curve.append(portfolio_equity)
            equity_dates.append(t_next)

            snap = RebalanceSnapshot(
                timestamp=t_decision,
                portfolio_return_gross=fwd_gross_ret,
                portfolio_return_net=fwd_net_ret,
                turnover=turnover,
                fee_drag=fee_drag,
                selected_symbols=selected,
                is_cash=is_cash,
                btc_return=btc_fwd_ret,
                market_equal_weight_return=mkt_fwd_ret
            )
            snapshots.append(snap)
            current_weights = target_weights

        # -------------------------------------------------------------
        # Compute Portfolio Summary Performance Metrics
        # -------------------------------------------------------------
        if len(snapshots) == 0:
            return {}

        df_snaps = pd.DataFrame([s.__dict__ for s in snapshots])
        net_rets = df_snaps["portfolio_return_net"].to_numpy(dtype=float)
        gross_rets = df_snaps["portfolio_return_gross"].to_numpy(dtype=float)
        btc_rets = df_snaps["btc_return"].to_numpy(dtype=float)
        mkt_rets = df_snaps["market_equal_weight_return"].to_numpy(dtype=float)

        total_net_return = portfolio_equity - 1.0
        n_periods = len(net_rets)
        periods_per_year = 365.25 / rebalance_days

        cagr = ((portfolio_equity) ** (periods_per_year / n_periods)) - 1.0 if (n_periods > 0 and portfolio_equity > 0) else -1.0

        mean_period_net = np.mean(net_rets)
        std_period_net = np.std(net_rets)
        downside_std = np.std(net_rets[net_rets < 0]) if np.sum(net_rets < 0) > 0 else 1e-6

        sharpe = (mean_period_net / std_period_net * np.sqrt(periods_per_year)) if std_period_net > 1e-6 else 0.0
        sortino = (mean_period_net / downside_std * np.sqrt(periods_per_year)) if downside_std > 1e-6 else 0.0

        # Maximum Drawdown
        eq_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = float(np.max(dds))

        # Benchmark comparisons
        btc_equity = np.prod(1.0 + btc_rets) - 1.0
        mkt_equity = np.prod(1.0 + mkt_rets) - 1.0
        win_rate_periods = float(np.mean(net_rets > 0))
        outperformed_btc_periods = float(np.mean(net_rets > btc_rets))
        outperformed_mkt_periods = float(np.mean(net_rets > mkt_rets))
        avg_turnover = float(np.mean(df_snaps["turnover"]))

        return {
            "total_net_return": total_net_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate_periods": win_rate_periods,
            "outperformed_btc_pct": outperformed_btc_periods,
            "outperformed_market_pct": outperformed_mkt_periods,
            "btc_total_return": btc_equity,
            "market_equal_weight_return": mkt_equity,
            "alpha_vs_btc": total_net_return - btc_equity,
            "alpha_vs_market": total_net_return - mkt_equity,
            "avg_turnover": avg_turnover,
            "n_rebalances": n_periods,
            "equity_curve": equity_curve,
            "equity_dates": equity_dates,
            "snapshots_df": df_snaps
        }
