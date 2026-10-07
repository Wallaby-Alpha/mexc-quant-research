"""
research_short_term_rotation.py
Vectorized, empirical investigation of Short-Term Portfolio Rotation:
Tests rotational cadences:
- 4 Hours (6x Daily)
- 8 Hours (3x Daily)
- 1 Day (24 Hours)
- 2 Days (48 Hours)
- 3 Days (72 Hours)
- 5 Days (120 Hours)
- 7 Days (168 Hours / Weekly Baseline)

Evaluates:
- Trailing lookbacks: 24h, 3d, 7d, 14d, 30d
- Gross Return vs Net Return (net of taker fee + slippage + funding)
- Annualized Turnover & Total Fee Drag
- Sharpe Ratio, Max Drawdown, Win Rate
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("short_term_rotation")


def main():
    logger.info("=" * 80)
    logger.info("STARTING SHORT-TERM ROTATION CADENCE STUDY (4H, 8H, 1D, 2D, 3D, 5D, 7D)")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )
    total_one_way_cost = exec_model.taker_fee_rate + (exec_model.base_slippage_bps / 10000.0) # 0.0007 (7 bps)

    # Load 1h Klines
    logger.info("Loading 1h Kline data for universe...")
    symbols_1h = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    
    # Load BTC
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1h["open_time"] = pd.to_datetime(df_btc_1h["open_time"], utc=True)
    df_btc_1h = df_btc_1h.sort_values("open_time").set_index("open_time")
    s_btc_close = df_btc_1h["close"]

    # Daily resampled BTC for EMA50 macro trend
    df_btc_1d = KlineResampler.resample_1h_to_1d(cache.load_klines("BTC_USDT", "1h"))
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").set_index("open_time")
    s_btc_1d_close = df_btc_1d["close"]
    s_btc_1d_ema50 = compute_ema(s_btc_1d_close, span=50)

    # Build 1h Close and Open price matrices for altcoins
    dict_close = {}
    dict_open = {}
    for sym in symbols_1h:
        df = cache.load_klines(sym, "1h")
        if len(df) < 2000:
            continue
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.sort_values("open_time").set_index("open_time")
        dict_close[sym] = df["close"]
        dict_open[sym] = df["open"]

    df_close = pd.DataFrame(dict_close)
    df_open = pd.DataFrame(dict_open)
    logger.info(f"Loaded price matrices with {df_close.shape[1]} altcoins and {len(df_close)} hourly bars.")

    common_idx = df_close.index.intersection(s_btc_close.index).sort_values()
    df_close = df_close.loc[common_idx]
    df_open = df_open.loc[common_idx]
    s_btc_close = s_btc_close.loc[common_idx]

    # Pre-map macro regime
    btc_macro_bullish_1h = pd.Series(index=common_idx, dtype=bool)
    for t in common_idx:
        past_daily_dates = s_btc_1d_close.index[s_btc_1d_close.index <= t]
        if len(past_daily_dates) > 0:
            t_last_d = past_daily_dates[-1]
            btc_macro_bullish_1h.loc[t] = (s_btc_1d_close.loc[t_last_d] >= s_btc_1d_ema50.loc[t_last_d])
        else:
            btc_macro_bullish_1h.loc[t] = False

    logger.info("Macro regime mapped.")

    # -------------------------------------------------------------------------
    # Pre-compute Vectorized Momentum & Sharpe Scores
    # -------------------------------------------------------------------------
    lookback_list = [24, 72, 168, 336, 720] # 24h, 3d, 7d, 14d, 30d
    logger.info("Pre-computing vectorized rolling scores...")
    
    pct_changes = df_close.pct_change()
    periods_yr = 8766.0

    scores_sharpe: Dict[int, pd.DataFrame] = {}
    scores_raw: Dict[int, pd.DataFrame] = {}

    for lb in lookback_list:
        ret_lb = (df_close - df_close.shift(lb)) / df_close.shift(lb)
        vol_lb = pct_changes.rolling(lb).std() * np.sqrt(periods_yr)
        vol_lb = vol_lb.replace(0, np.nan).fillna(1.0)
        vol_lb = np.maximum(vol_lb, 0.05)
        
        scores_sharpe[lb] = ret_lb / vol_lb
        scores_raw[lb] = ret_lb

    logger.info("Vectorized scoring matrices ready.")

    # -------------------------------------------------------------------------
    # FAST SIMULATION ENGINE
    # -------------------------------------------------------------------------
    def run_cadence_experiment(
        cadence_hours: int,
        lookback_hours: int,
        use_sharpe: bool = True,
        k: int = 10,
        rank_exit_buffer: int = 0
    ) -> Dict[str, Any]:
        score_df = scores_sharpe[lookback_hours] if use_sharpe else scores_raw[lookback_hours]

        min_start = max(lookback_hours + 48, 24 * 50)
        rebal_indices = list(range(min_start, len(common_idx) - cadence_hours, cadence_hours))
        if len(rebal_indices) < 10:
            return {}

        current_holdings = []
        portfolio_equity = 1.0
        equity_curve = [1.0]
        rebal_records = []

        total_turnover = 0.0
        total_fee_drag = 0.0

        for idx in rebal_indices:
            t_now = common_idx[idx]
            t_next = common_idx[idx + cadence_hours]

            is_macro_bullish = btc_macro_bullish_1h.iloc[idx]

            if not is_macro_bullish:
                target_holdings = []
            else:
                s_row = score_df.iloc[idx].dropna()
                if len(s_row) < k:
                    target_holdings = []
                else:
                    if rank_exit_buffer > k:
                        buf_top = s_row.nlargest(rank_exit_buffer).index.tolist()
                        top_k = s_row.nlargest(k).index.tolist()
                        kept = [s for s in current_holdings if s in buf_top]
                        needed = k - len(kept)
                        new_buys = [s for s in top_k if s not in kept][:needed]
                        target_holdings = kept + new_buys
                    else:
                        target_holdings = s_row.nlargest(k).index.tolist()

            # Turnover
            old_set = set(current_holdings)
            new_set = set(target_holdings)
            w_old = {s: 1.0 / len(current_holdings) for s in current_holdings} if current_holdings else {}
            w_new = {s: 1.0 / len(target_holdings) for s in target_holdings} if target_holdings else {}
            
            all_s = old_set.union(new_set)
            turnover = sum(abs(w_new.get(s, 0.0) - w_old.get(s, 0.0)) for s in all_s) / 2.0
            total_turnover += turnover

            fee_cost = turnover * 2.0 * total_one_way_cost
            funding_cost = (0.0001 * (cadence_hours / 8.0)) if target_holdings else 0.0
            period_drag = fee_cost + funding_cost
            total_fee_drag += period_drag

            # Forward return from t_now to t_next (open to open)
            if not target_holdings:
                fwd_gross = 0.0
            else:
                p_in = df_open.iloc[idx][target_holdings]
                p_out = df_open.iloc[idx + cadence_hours][target_holdings]
                rets = (p_out - p_in) / p_in
                rets = rets.dropna()
                fwd_gross = float(rets.mean()) if len(rets) > 0 else 0.0

            fwd_net = fwd_gross - period_drag
            portfolio_equity = portfolio_equity * (1.0 + fwd_net)
            equity_curve.append(portfolio_equity)

            rebal_records.append({
                "t": t_now,
                "gross_ret": fwd_gross,
                "net_ret": fwd_net,
                "turnover": turnover,
                "fee_drag": period_drag
            })

            current_holdings = target_holdings

        df_rec = pd.DataFrame(rebal_records)
        net_rets = df_rec["net_ret"].to_numpy()
        gross_rets = df_rec["gross_ret"].to_numpy()

        n_periods = len(net_rets)
        periods_per_year = 8766.0 / cadence_hours
        tot_net = portfolio_equity - 1.0
        tot_gross = np.prod(1.0 + gross_rets) - 1.0
        
        cagr = (portfolio_equity ** (periods_per_year / n_periods)) - 1.0 if (n_periods > 0 and portfolio_equity > 0) else -1.0

        m_net = np.mean(net_rets)
        s_net = np.std(net_rets)
        sharpe = (m_net / s_net * np.sqrt(periods_per_year)) if s_net > 1e-6 else 0.0

        eq_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq_arr)
        max_dd = float(np.max((peaks - eq_arr) / peaks))

        avg_turnover = float(df_rec["turnover"].mean())
        annualized_fee_drag = avg_turnover * 2.0 * total_one_way_cost * periods_per_year * 100

        return {
            "cadence_hours": cadence_hours,
            "lookback_hours": lookback_hours,
            "tot_net": tot_net,
            "tot_gross": tot_gross,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "avg_turnover": avg_turnover,
            "annualized_fee_drag_pct": annualized_fee_drag,
            "win_rate": float(np.mean(net_rets > 0)),
            "n_rebalances": n_periods
        }

    # -------------------------------------------------------------------------
    # TEST SUITE 1: 30-Day Lookback across Cadences (4h to 7d)
    # -------------------------------------------------------------------------
    cadence_grid = [
        ("4 Hours (6x Daily)", 4, 720),
        ("8 Hours (3x Daily)", 8, 720),
        ("1 Day (Daily)", 24, 720),
        ("2 Days (48 Hours)", 48, 720),
        ("3 Days (72 Hours)", 72, 720),
        ("5 Days (120 Hours)", 120, 720),
        ("7 Days (Weekly Baseline)", 168, 720),
    ]

    results_standard = []
    for label, c_hrs, lb_hrs in cadence_grid:
        r = run_cadence_experiment(cadence_hours=c_hrs, lookback_hours=lb_hrs, use_sharpe=True, k=10, rank_exit_buffer=0)
        r["label"] = label
        results_standard.append(r)

    print("\n" + "=" * 105)
    print("TEST SUITE 1: ROTATION CADENCE SPECTRUM (30-DAY SHARPE MOMENTUM, TOP 10)")
    print("=" * 105)
    header = f"{'Cadence':<25} | {'Gross Return':<12} | {'Net Return':<11} | {'Sharpe':<6} | {'MaxDD':<6} | {'Avg Turnover':<12} | {'Annual Fee Drag':<15}"
    print(header)
    print("-" * 105)
    for r in results_standard:
        print(f"{r['label']:<25} | {r['tot_gross']*100:+10.1f}% | {r['tot_net']*100:+9.1f}% | {r['sharpe']:<6.2f} | {r['max_dd']*100:<5.1f}% | {r['avg_turnover']*100:<11.1f}% | {r['annualized_fee_drag_pct']:<14.1f}%")

    # -------------------------------------------------------------------------
    # TEST SUITE 2: Short Lookbacks (Fast Cadence + Fast Lookback)
    # -------------------------------------------------------------------------
    short_experiments = [
        ("4H Cadence / 24H Lookback", 4, 24),
        ("4H Cadence / 3D Lookback", 4, 72),
        ("4H Cadence / 7D Lookback", 4, 168),
        ("8H Cadence / 3D Lookback", 8, 72),
        ("1D Cadence / 24H Lookback", 24, 24),
        ("1D Cadence / 3D Lookback", 24, 72),
        ("1D Cadence / 7D Lookback", 24, 168),
        ("1D Cadence / 14D Lookback", 24, 336),
        ("2D Cadence / 3D Lookback", 48, 72),
        ("2D Cadence / 7D Lookback", 48, 168),
        ("3D Cadence / 7D Lookback", 72, 168),
        ("3D Cadence / 14D Lookback", 72, 336),
    ]

    results_short = []
    for label, c_hrs, lb_hrs in short_experiments:
        r = run_cadence_experiment(cadence_hours=c_hrs, lookback_hours=lb_hrs, use_sharpe=True, k=10, rank_exit_buffer=0)
        r["label"] = label
        results_short.append(r)

    print("\n" + "=" * 105)
    print("TEST SUITE 2: SHORT-TERM ROTATIONS (FAST CADENCE + SHORT LOOKBACKS)")
    print("=" * 105)
    print(header)
    print("-" * 105)
    for r in results_short:
        print(f"{r['label']:<25} | {r['tot_gross']*100:+10.1f}% | {r['tot_net']*100:+9.1f}% | {r['sharpe']:<6.2f} | {r['max_dd']*100:<5.1f}% | {r['avg_turnover']*100:<11.1f}% | {r['annualized_fee_drag_pct']:<14.1f}%")

    # -------------------------------------------------------------------------
    # TEST SUITE 3: Buffer Smoothing on Daily and 3-Day Cadences
    # -------------------------------------------------------------------------
    buffer_experiments = [
        ("1D Cadence (No Buffer)", 24, 720, 0),
        ("1D Cadence (Buffer Top 15)", 24, 720, 15),
        ("1D Cadence (Buffer Top 20)", 24, 720, 20),
        ("2D Cadence (Buffer Top 15)", 48, 720, 15),
        ("3D Cadence (No Buffer)", 72, 720, 0),
        ("3D Cadence (Buffer Top 15)", 72, 720, 15),
        ("7D Cadence (Weekly Baseline)", 168, 720, 0),
    ]

    results_buffer = []
    for label, c_hrs, lb_hrs, buf in buffer_experiments:
        r = run_cadence_experiment(cadence_hours=c_hrs, lookback_hours=lb_hrs, use_sharpe=True, k=10, rank_exit_buffer=buf)
        r["label"] = label
        results_buffer.append(r)

    print("\n" + "=" * 105)
    print("TEST SUITE 3: TURNOVER SMOOTHING WITH RANK EXIT BUFFERS")
    print("=" * 105)
    print(header)
    print("-" * 105)
    for r in results_buffer:
        print(f"{r['label']:<25} | {r['tot_gross']*100:+10.1f}% | {r['tot_net']*100:+9.1f}% | {r['sharpe']:<6.2f} | {r['max_dd']*100:<5.1f}% | {r['avg_turnover']*100:<11.1f}% | {r['annualized_fee_drag_pct']:<14.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("SHORT-TERM ROTATION RESEARCH COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
