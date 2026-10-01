"""
run_detailed_portfolio_risk_analysis.py
Answers the 4 user questions with exact trade-by-trade and intra-week 1H bar analysis:
1. $7,000 initial capital ($1k/coin) progression and final dollar balance.
2. Individual coin-week win/loss counts, loss rates, and loss distribution.
3. Overall portfolio weekly win/loss counts and worst weekly portfolio drop.
4. Leverage liquidation analysis: intra-week max adverse excursion (MAE) across
   20% (5x), 25% (4x), 33.3% (3x), 50% (2x), and 75% liquidation thresholds.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import logging

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("risk_analysis")


def main():
    logger.info("=" * 80)
    logger.info("DETAILED PORTFOLIO RISK, DOLLAR GROWTH, AND LEVERAGE ANALYSIS")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    holdout_start = cfg.data.holdout_start
    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    # 1. Load Bitcoin Data
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # 2. Load 1H and Daily Altcoin Data
    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    dict_1h = {}
    dict_1d_close = {}

    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
        df_1h = df_1h.sort_values("open_time").reset_index(drop=True)
        dict_1h[sym] = df_1h.set_index("open_time")

        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d_close[sym] = df_1d.set_index("open_time")["close"]

    price_matrix = pd.DataFrame(dict_1d_close)
    logger.info(f"Loaded price matrix with {price_matrix.shape[1]} altcoins and 1H intraday klines.")

    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]

    # Evaluate Strategies: Top 7 Quality/Sharpe Momentum vs Top 7 Raw Momentum
    strategies_to_test = [
        ("Top 7 Quality (Sharpe) Momentum", "sharpe_mom", 7),
        ("Top 7 Raw 30d Momentum", "raw_mom", 7),
        ("Top 10 Quality (Sharpe) Momentum", "sharpe_mom", 10),
    ]

    taker_fee = exec_model.taker_fee_rate
    base_slip = exec_model.base_slippage_bps / 10000.0
    total_one_way_cost = taker_fee + base_slip

    for strat_name, mode, k in strategies_to_test:
        logger.info("-" * 80)
        logger.info(f"ANALYZING STRATEGY: {strat_name} (Top {k})")
        logger.info("-" * 80)

        initial_capital = 7000.0 if k == 7 else 10000.0
        portfolio_equity = initial_capital
        current_weights = {}

        coin_trades = []
        weekly_portfolio_records = []

        for i in range(len(valid_mondays) - 1):
            t_now = valid_mondays[i]
            t_next = valid_mondays[i + 1]

            btc_price = s_btc.loc[t_now]
            btc_ema = s_btc_ema50.loc[t_now]
            is_cash = btc_price < btc_ema

            # Signal selection strictly up to t_now
            idx_now = common_idx.get_loc(t_now)
            t_past_30 = common_idx[max(0, idx_now - 30)]
            btc_past = s_btc.loc[t_past_30]
            btc_30d_ret = (btc_price - btc_past) / btc_past

            if is_cash:
                selected_coins = []
                target_weights = {}
            else:
                candidates = []
                for sym in price_matrix.columns:
                    s_p = price_matrix[sym]
                    p_now = s_p.loc[t_now]
                    p_past = s_p.loc[t_past_30]
                    if np.isnan(p_now) or np.isnan(p_past) or p_now <= 0 or p_past <= 0:
                        continue
                    ret_30d = (p_now - p_past) / p_past
                    rs_spread = ret_30d - btc_30d_ret

                    slice_30d = s_p.iloc[max(0, idx_now - 30) : idx_now + 1]
                    pct_changes = slice_30d.pct_change().dropna()
                    vol_30d = float(pct_changes.std() * np.sqrt(365.25)) if len(pct_changes) >= 20 else 1.0
                    if np.isnan(vol_30d) or vol_30d < 0.05:
                        vol_30d = 0.05
                    sharpe_score = ret_30d / vol_30d

                    candidates.append({
                        "symbol": sym,
                        "ret_30d": ret_30d,
                        "rs_spread": rs_spread,
                        "vol_30d": vol_30d,
                        "sharpe_score": sharpe_score
                    })

                df_cand = pd.DataFrame(candidates)
                if mode == "sharpe_mom":
                    selected_coins = df_cand.sort_values("sharpe_score", ascending=False).head(k)["symbol"].tolist()
                else:
                    selected_coins = df_cand.sort_values("rs_spread", ascending=False).head(k)["symbol"].tolist()

                target_weights = {s: 1.0 / len(selected_coins) for s in selected_coins}

            # Transaction costs
            all_syms = set(current_weights.keys()).union(set(target_weights.keys()))
            turnover = sum(abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_syms) / 2.0
            fee_drag = turnover * 2.0 * total_one_way_cost
            funding_drag = (0.0001 * 3.0 * 7) if not is_cash else 0.0

            # Evaluate each coin performance and intra-week MAE (liquidation risk)
            week_coin_returns = []
            allocation_per_coin = (portfolio_equity / len(selected_coins)) if selected_coins else 0.0

            for sym in selected_coins:
                s_p = price_matrix[sym]
                p_entry = s_p.loc[t_now]
                p_exit = s_p.loc[t_next] if t_next in s_p.index else np.nan
                coin_ret = (p_exit - p_entry) / p_entry if (not np.isnan(p_exit) and p_entry > 0) else 0.0
                dollar_pnl = allocation_per_coin * coin_ret
                week_coin_returns.append(coin_ret)

                # Intra-week 1H analysis for leverage liquidation
                intra_min_drop = 0.0
                if sym in dict_1h:
                    df_sym_1h = dict_1h[sym]
                    # Get 1h bars strictly within the week holding period [t_now, t_next]
                    mask_1h = (df_sym_1h.index >= t_now) & (df_sym_1h.index <= t_next)
                    slice_1h = df_sym_1h.loc[mask_1h]
                    if not slice_1h.empty:
                        lowest_price = float(slice_1h["low"].min())
                        intra_min_drop = (lowest_price - p_entry) / p_entry

                coin_trades.append({
                    "week_index": i + 1,
                    "rebalance_date": t_now.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "entry_price": p_entry,
                    "exit_price": p_exit,
                    "coin_return": coin_ret,
                    "dollar_allocation": allocation_per_coin,
                    "dollar_pnl": dollar_pnl,
                    "is_profitable": coin_ret > 0,
                    "is_loss": coin_ret < 0,
                    "intra_min_drop": intra_min_drop,
                    "hit_20pct_drop": intra_min_drop <= -0.20,
                    "hit_25pct_drop": intra_min_drop <= -0.25,
                    "hit_33pct_drop": intra_min_drop <= -0.3333,
                    "hit_50pct_drop": intra_min_drop <= -0.50,
                    "hit_75pct_drop": intra_min_drop <= -0.75,
                })

            if is_cash:
                gross_portfolio_ret = 0.0
            else:
                gross_portfolio_ret = float(np.mean(week_coin_returns)) if week_coin_returns else 0.0

            net_portfolio_ret = gross_portfolio_ret - fee_drag - funding_drag
            start_equity = portfolio_equity
            portfolio_equity = portfolio_equity * (1.0 + net_portfolio_ret)
            end_equity = portfolio_equity
            net_dollar_change = end_equity - start_equity

            weekly_portfolio_records.append({
                "week_index": i + 1,
                "rebalance_date": t_now.strftime("%Y-%m-%d"),
                "is_cash": is_cash,
                "start_equity": start_equity,
                "end_equity": end_equity,
                "net_dollar_change": net_dollar_change,
                "net_portfolio_ret": net_portfolio_ret,
                "gross_portfolio_ret": gross_portfolio_ret,
                "is_profitable_week": net_portfolio_ret > 0,
                "is_losing_week": net_portfolio_ret < 0,
                "is_cash_week": is_cash,
            })
            current_weights = target_weights

        df_coins = pd.DataFrame(coin_trades)
        df_port = pd.DataFrame(weekly_portfolio_records)

        # -------------------------------------------------------------
        # 1. Ending Capital & Progression
        # -------------------------------------------------------------
        final_equity = portfolio_equity
        total_pnl = final_equity - initial_capital
        total_ret_pct = (final_equity - initial_capital) / initial_capital * 100
        peak_equity = df_port["end_equity"].max()
        trough_equity = df_port["end_equity"].min()

        # -------------------------------------------------------------
        # 2. Individual Coin Performance
        # -------------------------------------------------------------
        total_coin_weeks = len(df_coins)
        winning_coin_weeks = int(df_coins["is_profitable"].sum())
        losing_coin_weeks = int(df_coins["is_loss"].sum())
        scratch_coin_weeks = total_coin_weeks - winning_coin_weeks - losing_coin_weeks
        coin_win_rate = (winning_coin_weeks / total_coin_weeks * 100) if total_coin_weeks > 0 else 0
        coin_loss_rate = (losing_coin_weeks / total_coin_weeks * 100) if total_coin_weeks > 0 else 0
        avg_coin_gain = df_coins[df_coins["coin_return"] > 0]["coin_return"].mean() * 100
        avg_coin_loss = df_coins[df_coins["coin_return"] < 0]["coin_return"].mean() * 100
        worst_single_coin_loss = df_coins["coin_return"].min() * 100
        best_single_coin_gain = df_coins["coin_return"].max() * 100

        # -------------------------------------------------------------
        # 3. Overall Portfolio Performance
        # -------------------------------------------------------------
        total_weeks = len(df_port)
        cash_weeks = int(df_port["is_cash_week"].sum())
        active_weeks = total_weeks - cash_weeks
        winning_weeks = int(df_port["is_profitable_week"].sum())
        losing_weeks = int(df_port["is_losing_week"].sum())
        flat_weeks = total_weeks - winning_weeks - losing_weeks
        port_win_rate_all = (winning_weeks / total_weeks) * 100
        port_loss_rate_all = (losing_weeks / total_weeks) * 100
        port_win_rate_active = (winning_weeks / active_weeks * 100) if active_weeks > 0 else 0
        port_loss_rate_active = (losing_weeks / active_weeks * 100) if active_weeks > 0 else 0
        worst_week_ret = df_port["net_portfolio_ret"].min() * 100
        best_week_ret = df_port["net_portfolio_ret"].max() * 100
        avg_win_week = df_port[df_port["net_portfolio_ret"] > 0]["net_portfolio_ret"].mean() * 100
        avg_loss_week = df_port[df_port["net_portfolio_ret"] < 0]["net_portfolio_ret"].mean() * 100

        # -------------------------------------------------------------
        # 4. Leverage Liquidation Analysis (Intra-week Drops)
        # -------------------------------------------------------------
        n_drop_20 = int(df_coins["hit_20pct_drop"].sum())
        n_drop_25 = int(df_coins["hit_25pct_drop"].sum())
        n_drop_33 = int(df_coins["hit_33pct_drop"].sum())
        n_drop_50 = int(df_coins["hit_50pct_drop"].sum())
        n_drop_75 = int(df_coins["hit_75pct_drop"].sum())

        pct_drop_20 = (n_drop_20 / total_coin_weeks) * 100 if total_coin_weeks > 0 else 0
        pct_drop_25 = (n_drop_25 / total_coin_weeks) * 100 if total_coin_weeks > 0 else 0
        pct_drop_33 = (n_drop_33 / total_coin_weeks) * 100 if total_coin_weeks > 0 else 0
        pct_drop_50 = (n_drop_50 / total_coin_weeks) * 100 if total_coin_weeks > 0 else 0
        pct_drop_75 = (n_drop_75 / total_coin_weeks) * 100 if total_coin_weeks > 0 else 0
        worst_intra_drop = df_coins["intra_min_drop"].min() * 100

        print("\n" + "=" * 70)
        print(f"RESULTS FOR: {strat_name}")
        print("=" * 70)
        print(f"QUESTION 1: DOLLAR PROGRESSION (Initial = ${initial_capital:,.2f})")
        print(f"  • Final Account Balance: ${final_equity:,.2f}")
        print(f"  • Net Dollar Profit:     ${total_pnl:+,.2f} ({total_ret_pct:+.2f}%)")
        print(f"  • Peak Account Value:    ${peak_equity:,.2f}")
        print(f"  • Lowest Account Value:  ${trough_equity:,.2f}")

        print(f"\nQUESTION 2: INDIVIDUAL COIN LOSS FREQUENCY")
        print(f"  • Total Coin-Week Positions: {total_coin_weeks}")
        print(f"  • Winning Coin Positions:    {winning_coin_weeks} ({coin_win_rate:.1f}%)")
        print(f"  • Losing Coin Positions:     {losing_coin_weeks} ({coin_loss_rate:.1f}%)")
        print(f"  • Average Winner Return:     +{avg_coin_gain:.2f}%")
        print(f"  • Average Loser Return:      {avg_coin_loss:.2f}%")
        print(f"  • Best Single Coin Week:     +{best_single_coin_gain:.2f}%")
        print(f"  • Worst Single Coin Week:    {worst_single_coin_loss:.2f}%")

        print(f"\nQUESTION 3: OVERALL PORTFOLIO WEEKLY PERFORMANCE")
        print(f"  • Total Weeks Evaluated:     {total_weeks}")
        print(f"  • Active Trading Weeks:      {active_weeks} (BTC Bullish)")
        print(f"  • Cash Shield Weeks:         {cash_weeks} (BTC Bearish / 0% return)")
        print(f"  • Winning Portfolio Weeks:   {winning_weeks} ({port_win_rate_all:.1f}% of all weeks, {port_win_rate_active:.1f}% of active weeks)")
        print(f"  • Losing Portfolio Weeks:    {losing_weeks} ({port_loss_rate_all:.1f}% of all weeks, {port_loss_rate_active:.1f}% of active weeks)")
        print(f"  • Average Winning Week:      +{avg_win_week:.2f}%")
        print(f"  • Average Losing Week:       {avg_loss_week:.2f}%")
        print(f"  • Best Week Return:          +{best_week_ret:.2f}%")
        print(f"  • Worst Week Return:         {worst_week_ret:.2f}%")

        print(f"\nQUESTION 4: LEVERAGE & LIQUIDATION RISK (Intra-Week Drops from Entry)")
        print(f"  • Drops > 20% (Liquidates 5x):  {n_drop_20} times ({pct_drop_20:.1f}% of coin positions)")
        print(f"  • Drops > 25% (Liquidates 4x):  {n_drop_25} times ({pct_drop_25:.1f}% of coin positions)")
        print(f"  • Drops > 33.3% (Liquidates 3x):{n_drop_33} times ({pct_drop_33:.1f}% of coin positions)")
        print(f"  • Drops > 50% (Liquidates 2x):  {n_drop_50} times ({pct_drop_50:.1f}% of coin positions)")
        print(f"  • Drops > 75% (Liquidates 1.33x):{n_drop_75} times ({pct_drop_75:.1f}% of coin positions)")
        print(f"  • Worst Intra-Week Flash Drop:   {worst_intra_drop:.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
