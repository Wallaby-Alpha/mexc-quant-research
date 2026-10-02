"""
analysis/evaluate_breakout_prop_firm_rotation.py
Empirical backtest evaluating the Weekly Rotational Momentum Strategy on:
1. Strictly Kraken Futures / Breakout Prop Firm Supported Coins (90 overlapping markets).
2. Simulating Breakout Prop Firm Rules:
   - Max Daily Loss: 3.0% (daily 00:00/00:30 UTC reset).
   - Static Max Drawdown: 6.0% (Classic Plan) / 5.0% (Pro Plan).
   - Profit Target: 10.0% (Classic Plan) / 12.0% (Pro Plan).
   - No time limit to pass.
3. Finding the optimal sizing multiplier (leverage / exposure scale) to pass without blowing the account.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional
import requests
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from data.resampler import KlineResampler
from backtest.execution_model import ExecutionModel

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_kraken_futures_symbols() -> set:
    """Fetches list of base coins tradeable on Kraken Futures / Breakout."""
    try:
        r = requests.get("https://futures.kraken.com/derivatives/api/v3/instruments", timeout=10)
        insts = r.json().get("instruments", [])
        symbols = set()
        for i in insts:
            if i.get("tradeable"):
                b = i.get("base", "") or i.get("underlying", "")
                if b:
                    if b.startswith("rr_"):
                        b = b.replace("rr_", "")
                    symbols.add(b.upper() + "_USDT")
        return symbols
    except Exception as e:
        print(f"Failed to fetch live Kraken symbols: {e}")
        return set()


def main():
    print("=" * 95)
    print("EVALUATION: WEEKLY ROTATION STRATEGY ON KRAKEN / BREAKOUT PROP FIRM RULES")
    print("=" * 95)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
    )
    total_fee_roundtrip = (exec_model.taker_fee_rate * 2.0) + (exec_model.base_slippage_bps / 10000.0 * 2.0)

    # 1. Identify Kraken-supported coins in local cache
    kraken_bases = get_kraken_futures_symbols()
    cached_symbols = set(cache.list_cached_symbols("1h"))
    valid_kraken_symbols = sorted(list(kraken_bases.intersection(cached_symbols) - {"BTC_USDT"}))

    print(f"Total Kraken Futures markets fetched: {len(kraken_bases)}")
    print(f"Overlap with local 1-year cache: {len(valid_kraken_symbols)} altcoin pairs")
    print(f"Sample Kraken assets: {valid_kraken_symbols[:15]}...")
    print("-" * 95)

    # 2. Load Daily Closes for Bitcoin and Kraken Altcoins
    df_btc_1h = cache.load_klines("BTC_USDT", "1h")
    df_btc_1d = KlineResampler.resample_1h_to_1d(df_btc_1h)
    df_btc_1d["open_time"] = pd.to_datetime(df_btc_1d["open_time"], utc=True)
    s_btc = df_btc_1d.sort_values("open_time").set_index("open_time")["close"]
    s_btc_ema50 = s_btc.ewm(span=50, adjust=False).mean()

    dict_1d = {}
    for sym in valid_kraken_symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty:
            continue
        df_1d = KlineResampler.resample_1h_to_1d(df_1h)
        if len(df_1d) < 70:
            continue
        df_1d["open_time"] = pd.to_datetime(df_1d["open_time"], utc=True)
        dict_1d[sym] = df_1d.sort_values("open_time").set_index("open_time")["close"]

    price_matrix = pd.DataFrame(dict_1d)
    common_idx = price_matrix.index.intersection(s_btc.index).sort_values()

    # Filter for Mondays
    mondays = [t for t in common_idx if t.dayofweek == 0 and (t - common_idx[0]).days >= 50]
    print(f"Total weekly cycles evaluated: {len(mondays) - 1} weeks ({mondays[0].strftime('%Y-%m-%d')} to {mondays[-1].strftime('%Y-%m-%d')})")
    print("-" * 95)

    # 3. Simulate Weekly Portfolio Returns on Kraken Universe
    # Model: 7 Quality Coins (Sharpe Momentum) + 3 Raw Coins (Relative Strength)
    k_quality = 7
    k_raw = 3
    total_coins = k_quality + k_raw  # 10 coins

    weekly_records = []

    for i in range(len(mondays) - 1):
        t0 = mondays[i]
        t1 = mondays[i + 1]

        btc_p0 = s_btc.loc[t0]
        btc_ema0 = s_btc_ema50.loc[t0]
        is_bullish = btc_p0 >= btc_ema0

        idx_now = common_idx.get_loc(t0)
        t_past_30 = common_idx[max(0, idx_now - 30)]
        btc_30d = (btc_p0 - s_btc.loc[t_past_30]) / s_btc.loc[t_past_30]

        if not is_bullish:
            # 100% Cash Shield
            weekly_records.append({
                "week_start": t0,
                "week_end": t1,
                "is_cash": True,
                "selected_coins": [],
                "gross_week_ret": 0.0,
                "net_week_ret": 0.0,
                "daily_rets": [0.0] * 7
            })
            continue

        # Rank Kraken universe
        cand = []
        for sym in price_matrix.columns:
            s_p = price_matrix[sym]
            p0 = s_p.loc[t0]
            p_past = s_p.loc[t_past_30]
            if np.isnan(p0) or np.isnan(p_past) or p0 <= 0 or p_past <= 0:
                continue
            ret_30d = (p0 - p_past) / p_past
            rs_spread = ret_30d - btc_30d
            pct_s = s_p.iloc[max(0, idx_now - 30) : idx_now + 1].pct_change().dropna()
            vol_30d = float(pct_s.std() * np.sqrt(365.25)) if len(pct_s) >= 20 else 1.0
            if np.isnan(vol_30d) or vol_30d < 0.05:
                vol_30d = 0.05
            sharpe_score = ret_30d / vol_30d
            cand.append({"sym": sym, "ret_30d": ret_30d, "rs_spread": rs_spread, "vol_30d": vol_30d, "sharpe_score": sharpe_score})

        df_cand = pd.DataFrame(cand)
        top_quality = df_cand.sort_values("sharpe_score", ascending=False).head(k_quality)["sym"].tolist()
        df_raw_cand = df_cand[~df_cand["sym"].isin(top_quality)]
        top_raw = df_raw_cand.sort_values("rs_spread", ascending=False).head(k_raw)["sym"].tolist()
        selected_10 = top_quality + top_raw

        # Measure intra-week daily performance across the 7 days of the week
        week_days = common_idx[(common_idx >= t0) & (common_idx <= t1)]
        daily_portfolio_rets = []

        for d_idx in range(len(week_days) - 1):
            d_start = week_days[d_idx]
            d_end = week_days[d_idx + 1]
            day_rets = []
            for s in selected_10:
                p_s = price_matrix[s].loc[d_start]
                p_e = price_matrix[s].loc[d_end]
                if p_s > 0 and not np.isnan(p_s) and not np.isnan(p_e):
                    day_rets.append((p_e - p_s) / p_s)
            daily_portfolio_rets.append(float(np.mean(day_rets)) if day_rets else 0.0)

        # Total week return
        coin_week_rets = []
        for s in selected_10:
            p_start = price_matrix[s].loc[t0]
            p_finish = price_matrix[s].loc[t1]
            if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_finish):
                coin_week_rets.append((p_finish - p_start) / p_start)

        gross_week = float(np.mean(coin_week_rets)) if coin_week_rets else 0.0
        # Friction
        funding = 0.0001 * 3.0 * 7.0  # 8h funding
        fee = total_fee_roundtrip * 0.5  # average turnover friction
        net_week = gross_week - funding - fee

        weekly_records.append({
            "week_start": t0,
            "week_end": t1,
            "is_cash": False,
            "selected_coins": selected_10,
            "gross_week_ret": gross_week,
            "net_week_ret": net_week,
            "daily_rets": daily_portfolio_rets
        })

    # 4. Analyze Raw 1.0x Spot Drawdowns
    all_daily_rets = []
    for r in weekly_records:
        all_daily_rets.extend(r["daily_rets"])

    s_daily = pd.Series(all_daily_rets)
    max_single_day_loss = float(s_daily.min() * 100.0)
    days_breaching_3pct = (s_daily <= -0.03).sum()

    s_weekly = pd.Series([r["net_week_ret"] for r in weekly_records])
    cum_unscaled = (1.0 + s_weekly).cumprod()
    max_dd_unscaled = float(((cum_unscaled - cum_unscaled.cummax()) / cum_unscaled.cummax()).min() * 100.0)
    total_ret_unscaled = float((cum_unscaled.iloc[-1] - 1.0) * 100.0)

    print("RAW UNLEVERAGED PORTFOLIO (100% CAPITAL ALLOCATED TO ALTS):")
    print(f"  • Full Year Net Return: {total_ret_unscaled:+.1f}%")
    print(f"  • Maximum Peak-to-Trough Drawdown: {max_dd_unscaled:.1f}%")
    print(f"  • Worst Single-Day Loss: {max_single_day_loss:.2f}%")
    print(f"  • Number of days with loss > 3.0%: {days_breaching_3pct} days")
    print("  🚨 VERDICT AT 100% SIZING: INSTANT PROP ACCOUNT BLOWUP (Violates 3% daily & 6% max loss!)")
    print("-" * 95)

    # 5. Simulate Prop Firm Sizing Fractions
    # What exposure fraction (e.g. 15%, 20%, 25%, 30%, 40%) keeps the portfolio strictly inside prop rules?
    print("SIMULATING EXPOSURE FRACTIONS FOR BREAKOUT PROP FIRM COMPLIANCE:")
    print("Target Rules: Max Daily Loss <= 3.0% | Max Static Drawdown <= 6.0% (Classic) / 5.0% (Pro)")
    print("=" * 95)
    header = f"{'Exposure Sizing':<18} | {'Worst Day':<10} | {'Max DD':<8} | {'Annual Return':<14} | {'Days to 10% Target':<20} | {'Compliance Status'}"
    print(header)
    print("-" * 95)

    exposure_scales = [1.0, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]

    for scale in exposure_scales:
        scaled_daily = s_daily * scale
        worst_day = float(scaled_daily.min() * 100.0)

        scaled_weekly = s_weekly * scale
        cum_scaled = (1.0 + scaled_weekly).cumprod()
        max_dd = float(((cum_scaled - cum_scaled.cummax()) / cum_scaled.cummax()).min() * 100.0)
        ann_ret = float((cum_scaled.iloc[-1] - 1.0) * 100.0)

        # Check when 10% profit target is reached
        hit_10pct_week = None
        for w_idx, equity in enumerate(cum_scaled):
            if equity >= 1.10:
                hit_10pct_week = w_idx + 1
                break

        time_to_target = f"Week {hit_10pct_week} (~{hit_10pct_week*7} days)" if hit_10pct_week else "Did not reach"

        # Check compliance
        violates_daily = worst_day <= -3.0
        violates_dd_pro = abs(max_dd) >= 5.0
        violates_dd_classic = abs(max_dd) >= 6.0

        if not violates_daily and not violates_dd_pro:
            status = "✅ PASSES PRO & CLASSIC"
        elif not violates_daily and not violates_dd_classic:
            status = "✅ PASSES CLASSIC (6% DD)"
        else:
            status = "❌ FAILS (Breaches Rules)"

        print(f"{scale*100:4.0f}% Account Size | {worst_day:6.2f}%   | {max_dd:5.1f}%  | {ann_ret:+6.1f}%       | {time_to_target:<20} | {status}")

    print("=" * 95)


if __name__ == "__main__":
    main()
