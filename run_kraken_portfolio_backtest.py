"""
run_kraken_portfolio_backtest.py
Empirical backtest and risk analysis filtering strictly to Kraken-listed coins.
Compares:
1. Full Universe (157 altcoins) vs. Kraken-Only Universe (104 altcoins)
2. Ending Capital ($7,000 start)
3. Sharpe Ratio, Sortino, Max Drawdown
4. Individual Coin Win Rate & Portfolio Win Rate
5. Leverage & Liquidation Risk (Intra-week 1H MAE drops)
"""

from pathlib import Path
import requests
import json
import logging
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from strategy.indicators import compute_ema
from backtest.execution_model import ExecutionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kraken_backtest")


def get_kraken_supported_symbols(all_cached_symbols):
    """Queries Kraken public API to find which cached symbols are listed on Kraken."""
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("result", {})
    except Exception as e:
        logger.error(f"Failed to fetch Kraken pairs: {e}")
        return []

    kraken_assets = set()
    for k, v in data.items():
        wsname = v.get("wsname", "")
        if wsname and "/" in wsname:
            kraken_assets.add(wsname.split("/")[0].upper())
        altname = v.get("altname", "")
        for quote in ["USD", "USDT", "EUR"]:
            if altname.endswith(quote):
                kraken_assets.add(altname[:-len(quote)].upper())
        base = v.get("base", "")
        if base.startswith("X") and len(base) == 4:
            kraken_assets.add(base[1:].upper())

    # Add standard base aliases
    kraken_assets.update(["BTC", "ETH", "SOL", "DOGE", "ZEC", "XMR", "LTC"])

    matched = []
    for s in all_cached_symbols:
        base = s.replace("_USDT", "").upper()
        if base in kraken_assets:
            matched.append(s)

    logger.info(f"Identified {len(matched)} Kraken-supported coins out of {len(all_cached_symbols)} universe coins.")
    return matched


def run_portfolio_eval(
    price_matrix: pd.DataFrame,
    dict_1h: dict,
    btc_series: pd.Series,
    btc_ema50: pd.Series,
    valid_mondays: list,
    common_idx: pd.DatetimeIndex,
    mode: str = "sharpe_mom",
    k: int = 7,
    initial_capital: float = 7000.0,
    exec_model: ExecutionModel = None
):
    portfolio_equity = initial_capital
    current_weights = {}
    coin_trades = []
    weekly_records = []

    taker_fee = exec_model.taker_fee_rate if exec_model else 0.0002
    base_slip = (exec_model.base_slippage_bps / 10000.0) if exec_model else 0.0010
    total_one_way_cost = taker_fee + base_slip

    for i in range(len(valid_mondays) - 1):
        t_now = valid_mondays[i]
        t_next = valid_mondays[i + 1]

        btc_price = btc_series.loc[t_now]
        btc_ema = btc_ema50.loc[t_now]
        is_cash = btc_price < btc_ema

        idx_now = common_idx.get_loc(t_now)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_past = btc_series.loc[t_past_30]
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
            if df_cand.empty:
                selected_coins = []
            elif mode == "sharpe_mom":
                selected_coins = df_cand.sort_values("sharpe_score", ascending=False).head(k)["symbol"].tolist()
            else:
                selected_coins = df_cand.sort_values("rs_spread", ascending=False).head(k)["symbol"].tolist()

            target_weights = {s: 1.0 / len(selected_coins) for s in selected_coins}

        # Turnover & Costs
        all_syms = set(current_weights.keys()).union(set(target_weights.keys()))
        turnover = sum(abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_syms) / 2.0
        fee_drag = turnover * 2.0 * total_one_way_cost
        funding_drag = (0.0001 * 3.0 * 7) if not is_cash else 0.0

        week_coin_returns = []
        allocation_per_coin = (portfolio_equity / len(selected_coins)) if selected_coins else 0.0

        for sym in selected_coins:
            s_p = price_matrix[sym]
            p_entry = s_p.loc[t_now]
            p_exit = s_p.loc[t_next] if t_next in s_p.index else np.nan
            coin_ret = (p_exit - p_entry) / p_entry if (not np.isnan(p_exit) and p_entry > 0) else 0.0
            week_coin_returns.append(coin_ret)

            # Intra-week 1H analysis for leverage liquidation
            intra_min_drop = 0.0
            if sym in dict_1h:
                df_sym_1h = dict_1h[sym]
                mask_1h = (df_sym_1h.index >= t_now) & (df_sym_1h.index <= t_next)
                slice_1h = df_sym_1h.loc[mask_1h]
                if not slice_1h.empty:
                    lowest_price = float(slice_1h["low"].min())
                    intra_min_drop = (lowest_price - p_entry) / p_entry

            coin_trades.append({
                "week_index": i + 1,
                "symbol": sym,
                "coin_return": coin_ret,
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
            gross_ret = 0.0
        else:
            gross_ret = float(np.mean(week_coin_returns)) if week_coin_returns else 0.0

        net_ret = gross_ret - fee_drag - funding_drag
        portfolio_equity = portfolio_equity * (1.0 + net_ret)

        weekly_records.append({
            "week_index": i + 1,
            "is_cash": is_cash,
            "net_ret": net_ret,
            "is_profitable_week": net_ret > 0,
            "is_losing_week": net_ret < 0,
        })
        current_weights = target_weights

    df_c = pd.DataFrame(coin_trades)
    df_w = pd.DataFrame(weekly_records)

    total_net_ret = (portfolio_equity - initial_capital) / initial_capital
    net_rets = df_w["net_ret"].to_numpy(dtype=float)
    active_rets = df_w[~df_w["is_cash"]]["net_ret"].to_numpy(dtype=float)

    n_active = len(active_rets)
    win_active = np.sum(active_rets > 0)
    loss_active = np.sum(active_rets < 0)

    periods_per_year = 52.0
    mean_ret = np.mean(net_rets)
    std_ret = np.std(net_rets)
    downside_std = np.std(net_rets[net_rets < 0]) if np.sum(net_rets < 0) > 0 else 1e-6
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 1e-6 else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 1e-6 else 0.0

    eq_arr = np.cumprod(1.0 + net_rets)
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / peaks))

    return {
        "final_equity": portfolio_equity,
        "net_dollar_profit": portfolio_equity - initial_capital,
        "total_net_return_pct": total_net_ret * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "total_coin_positions": len(df_c),
        "coin_win_rate": (df_c["is_profitable"].sum() / len(df_c) * 100) if len(df_c) > 0 else 0,
        "coin_loss_rate": (df_c["is_loss"].sum() / len(df_c) * 100) if len(df_c) > 0 else 0,
        "avg_coin_gain": df_c[df_c["coin_return"] > 0]["coin_return"].mean() * 100 if len(df_c) > 0 else 0,
        "avg_coin_loss": df_c[df_c["coin_return"] < 0]["coin_return"].mean() * 100 if len(df_c) > 0 else 0,
        "active_weeks": n_active,
        "active_win_weeks": int(win_active),
        "active_loss_weeks": int(loss_active),
        "active_win_rate": (win_active / n_active * 100) if n_active > 0 else 0,
        "drop_20": int(df_c["hit_20pct_drop"].sum()) if len(df_c) > 0 else 0,
        "drop_25": int(df_c["hit_25pct_drop"].sum()) if len(df_c) > 0 else 0,
        "drop_33": int(df_c["hit_33pct_drop"].sum()) if len(df_c) > 0 else 0,
        "drop_50": int(df_c["hit_50pct_drop"].sum()) if len(df_c) > 0 else 0,
        "drop_75": int(df_c["hit_75pct_drop"].sum()) if len(df_c) > 0 else 0,
        "worst_intra_drop": df_c["intra_min_drop"].min() * 100 if len(df_c) > 0 else 0,
    }


def main():
    logger.info("=" * 80)
    logger.info("KRAKEN-ONLY UNIVERSE BACKTEST & LEVERAGE STUDY")
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

    # Load BTC
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    df_btc_1d = df_btc_1d.sort_values("open_time").reset_index(drop=True)
    s_btc = df_btc_1d.set_index("open_time")["close"]
    s_btc_ema50 = compute_ema(s_btc, span=50)

    # Load all altcoins
    all_symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    kraken_symbols = get_kraken_supported_symbols(all_symbols)

    dict_1h = {}
    dict_1d_close = {}

    for sym in all_symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"], utc=True)
        dict_1h[sym] = df_1h.sort_values("open_time").set_index("open_time")

        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d_close[sym] = df_1d.set_index("open_time")["close"]

    df_prices_all = pd.DataFrame(dict_1d_close)
    df_prices_kraken = df_prices_all[[c for c in kraken_symbols if c in df_prices_all.columns]]

    common_idx = df_prices_all.index.intersection(s_btc.index).sort_values()
    mondays = [t for t in common_idx if t.dayofweek == 0]
    min_lookback = 50
    valid_mondays = [m for m in mondays if (m - common_idx[0]).days >= min_lookback]

    # Run Comparisons
    results = {}

    # 1. Top 7 Quality Momentum (Full Universe vs Kraken-Only)
    results["Top 7 Quality (Full MEXC Universe)"] = run_portfolio_eval(
        df_prices_all, dict_1h, s_btc, s_btc_ema50, valid_mondays, common_idx, mode="sharpe_mom", k=7, exec_model=exec_model
    )
    results["Top 7 Quality (Kraken-Only Universe)"] = run_portfolio_eval(
        df_prices_kraken, dict_1h, s_btc, s_btc_ema50, valid_mondays, common_idx, mode="sharpe_mom", k=7, exec_model=exec_model
    )

    # 2. Top 7 Raw Momentum (Full Universe vs Kraken-Only)
    results["Top 7 Raw Mom (Full MEXC Universe)"] = run_portfolio_eval(
        df_prices_all, dict_1h, s_btc, s_btc_ema50, valid_mondays, common_idx, mode="raw_mom", k=7, exec_model=exec_model
    )
    results["Top 7 Raw Mom (Kraken-Only Universe)"] = run_portfolio_eval(
        df_prices_kraken, dict_1h, s_btc, s_btc_ema50, valid_mondays, common_idx, mode="raw_mom", k=7, exec_model=exec_model
    )

    # Print Results Comparison Table
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD RESULTS: FULL UNIVERSE vs. KRAKEN-ONLY UNIVERSE ($7,000 START)")
    print("=" * 90)

    header = f"{'Configuration':<38} | {'Final ($)':<11} | {'Return (%)':<10} | {'Sharpe':<6} | {'MaxDD':<6} | {'ActiveWin%':<10} | {'WorstDrop':<9}"
    print(header)
    print("-" * 90)

    for name, r in results.items():
        print(f"{name:<38} | ${r['final_equity']:<10,.2f} | {r['total_net_return_pct']:+8.1f}% | {r['sharpe_ratio']:<6.2f} | {r['max_drawdown_pct']:<5.1f}% | {r['active_win_rate']:<9.1f}% | {r['worst_intra_drop']:<8.1f}%")

    print("\n" + "=" * 90)
    print("LEVERAGE LIQUIDATION OCCURRENCES: FULL UNIVERSE vs. KRAKEN-ONLY")
    print("=" * 90)

    lev_header = f"{'Configuration':<38} | {'5x (>20%)':<11} | {'4x (>25%)':<11} | {'3x (>33%)':<11} | {'2x (>50%)':<11} | {'>75%':<8}"
    print(lev_header)
    print("-" * 90)

    for name, r in results.items():
        tot = r['total_coin_positions']
        print(f"{name:<38} | {r['drop_20']:2d} ({r['drop_20']/tot*100:4.1f}%) | {r['drop_25']:2d} ({r['drop_25']/tot*100:4.1f}%) | {r['drop_33']:2d} ({r['drop_33']/tot*100:4.1f}%) | {r['drop_50']:2d} ({r['drop_50']/tot*100:4.1f}%) | {r['drop_75']:2d} ({r['drop_75']/tot*100:4.1f}%)")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
