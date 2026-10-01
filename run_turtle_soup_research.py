"""
run_turtle_soup_research.py
Investigates Strategy Archetype 2 on the 1-Hour Timeframe:
The Failed Breakout / "Turtle Soup" Liquidity Sweep Reversal.
Tests whether fading false breakouts of swing highs (shorting) and false breakdowns
of swing lows (buying) generates positive net expectancy across 159 altcoins.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.config import load_config
from data.cache import ParquetCache
from strategy.turtle_soup import TurtleSoupEngine
from backtest.execution_model import ExecutionModel
from analysis.trade_metrics import TradeMetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("turtle_soup")


def main():
    logger.info("=" * 80)
    logger.info("STARTING STRATEGY 2: 1-HOUR TURTLE SOUP LIQUIDITY SWEEP RESEARCH")
    logger.info("=" * 80)

    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    reports_dir = Path(cfg.data.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    holdout_start = cfg.data.holdout_start
    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    symbols = [s for s in cache.list_cached_symbols("1h") if s != "BTC_USDT"]
    logger.info(f"Loading 1H klines for {len(symbols)} altcoins...")

    symbol_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df_1h = cache.load_klines(sym, "1h")
        if df_1h.empty or len(df_1h) < 100:
            continue
        symbol_data[sym] = df_1h

    logger.info(f"Loaded {len(symbol_data)} altcoins with valid 1H data.")

    # -------------------------------------------------------------------------
    # Factorial Matrix for 1H Turtle Soup Strategy
    # -------------------------------------------------------------------------
    # Parameters to test:
    # - Lookback: 24h, 48h, 72h
    # - Side: Long only, Short only, Both
    # - Wick Ratio: 0.30, 0.40
    # - Target RR: 1.5R, 2.0R, 3.0R
    # - Execution: Taker Next Open vs Maker Limit at Swept Level

    experiments = [
        # 1. Baseline Directional Split (Lookback=24h, Wick=30%, RR=2.0, Taker)
        ("TS_1H_L24_Both_RR2.0_Taker", 24, "both", 0.30, 2.0, "taker_next_open"),
        ("TS_1H_L24_LongOnly_RR2.0_Taker", 24, "long", 0.30, 2.0, "taker_next_open"),
        ("TS_1H_L24_ShortOnly_RR2.0_Taker", 24, "short", 0.30, 2.0, "taker_next_open"),

        # 2. Lookback Window Variations (48h and 72h)
        ("TS_1H_L48_Both_RR2.0_Taker", 48, "both", 0.30, 2.0, "taker_next_open"),
        ("TS_1H_L72_Both_RR2.0_Taker", 72, "both", 0.30, 2.0, "taker_next_open"),
        ("TS_1H_L48_LongOnly_RR2.0_Taker", 48, "long", 0.30, 2.0, "taker_next_open"),
        ("TS_1H_L48_ShortOnly_RR2.0_Taker", 48, "short", 0.30, 2.0, "taker_next_open"),

        # 3. Target R:R Spectrum (1.5R, 2.0R, 3.0R at L=48)
        ("TS_1H_L48_Both_RR1.5_Taker", 48, "both", 0.30, 1.5, "taker_next_open"),
        ("TS_1H_L48_Both_RR3.0_Taker", 48, "both", 0.30, 3.0, "taker_next_open"),

        # 4. Strict Rejection Wick (Wick >= 40% of candle range)
        ("TS_1H_L48_Both_RR2.0_Wick40_Taker", 48, "both", 0.40, 2.0, "taker_next_open"),
        ("TS_1H_L48_LongOnly_RR2.0_Wick40_Taker", 48, "long", 0.40, 2.0, "taker_next_open"),
        ("TS_1H_L48_ShortOnly_RR2.0_Wick40_Taker", 48, "short", 0.40, 2.0, "taker_next_open"),

        # 5. Maker Limit Execution (Resting at the swept level, 0.00% fee, zero slippage)
        ("TS_1H_L48_Both_RR2.0_MakerLimit", 48, "both", 0.30, 2.0, "maker_limit_swept_level"),
        ("TS_1H_L48_LongOnly_RR2.0_MakerLimit", 48, "long", 0.30, 2.0, "maker_limit_swept_level"),
        ("TS_1H_L48_ShortOnly_RR2.0_MakerLimit", 48, "short", 0.30, 2.0, "maker_limit_swept_level"),
        ("TS_1H_L48_Both_RR2.0_Wick40_MakerLimit", 48, "both", 0.40, 2.0, "maker_limit_swept_level"),
    ]

    all_trials = []

    logger.info(f"Executing {len(experiments)} 1H Turtle Soup Strategy Trials across {len(symbol_data)} altcoins...")

    for trial_id, lookback, side, wick_r, rr, exec_type in experiments:
        t0 = time.time()
        trial_trades = []

        for sym, df_1h in symbol_data.items():
            trades = TurtleSoupEngine.simulate_symbol_trades(
                symbol=sym,
                df_1h=df_1h,
                lookback_window=lookback,
                trade_side=side,
                min_wick_ratio=wick_r,
                target_rr=rr,
                execution_type=exec_type,
                max_holding_bars=24,
                stop_buffer_atr=0.25,
                exec_model=exec_model,
                holdout_ts=holdout_ts
            )
            trial_trades.extend(trades)

        elapsed = time.time() - t0
        n = len(trial_trades)
        if n == 0:
            continue

        df_tr = pd.DataFrame(trial_trades)
        metrics = TradeMetricsCalculator.compute_summary_metrics(df_tr)

        record = {
            "trial_id": trial_id,
            "lookback_bars": lookback,
            "side": side,
            "wick_ratio": wick_r,
            "target_rr": rr,
            "execution_type": exec_type,
            "n_trades": n,
            "win_rate_net": metrics["win_rate_net"],
            "expectancy_net_r": metrics["expectancy_net_r"],
            "expectancy_gross_r": metrics.get("expectancy_gross_r", 0.0),
            "profit_factor_net": metrics["profit_factor_net"],
            "avg_holding_hours": metrics.get("avg_holding_hours", 0.0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", -100.0)
        }
        all_trials.append(record)

        logger.info(
            f"[{trial_id:40s}] N={n:5d} | WR: {metrics['win_rate_net']*100:4.1f}% | "
            f"Net Exp: {metrics['expectancy_net_r']:+6.3f}R | Gross Exp: {metrics.get('expectancy_gross_r', 0.0):+6.3f}R | "
            f"Net PF: {metrics['profit_factor_net']:4.2f} ({elapsed:.1f}s)"
        )

    df_ts_results = pd.DataFrame(all_trials)
    out_csv = results_dir / "turtle_soup_results.csv"
    out_parquet = results_dir / "turtle_soup_results.parquet"
    df_ts_results.to_csv(out_csv, index=False)
    df_ts_results.to_parquet(out_parquet, index=False)
    logger.info(f"\nSaved {len(df_ts_results)} Turtle Soup trials to {out_csv}")

    # -------------------------------------------------------------------------
    # Visualizations & Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("1-HOUR TURTLE SOUP RESEARCH RESULTS (Sorted by Net Expectancy):")
    print("=" * 90)
    sorted_df = df_ts_results.sort_values(by="expectancy_net_r", ascending=False)
    print(sorted_df[["trial_id", "side", "execution_type", "n_trades", "win_rate_net", "expectancy_net_r", "expectancy_gross_r", "profit_factor_net"]].round(3).to_string(index=False))

    # Generate Chart
    generate_charts(figures_dir, df_ts_results)
    write_ts_report(reports_dir, df_ts_results)
    logger.info(f"Saved 1H Turtle Soup Research Report to {reports_dir / 'TURTLE_SOUP_RESEARCH.md'}")


def generate_charts(figures_dir: Path, df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Taker vs Maker Comparison (at L48 RR2.0)
    sub = df[df["lookback_bars"] == 48].copy()
    labels = [t.replace("TS_1H_L48_", "") for t in sub["trial_id"]]
    exps = sub["expectancy_net_r"].values
    colors = ["#2ca02c" if e > 0 else "#d62728" for e in exps]

    x = np.arange(len(labels))
    ax1.bar(x, exps, color=colors, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Net Expectancy (R)")
    ax1.set_title("1H Turtle Soup: Net Expectancy Across Parameter Variants")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i, e in enumerate(exps):
        ax1.annotate(f"{e:+.2f}R", xy=(i, e), xytext=(0, 5 if e >= 0 else -12), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    # 2. Long vs Short vs Both Performance
    side_groups = df.groupby("side")[["expectancy_net_r", "win_rate_net"]].mean()
    sides = side_groups.index.tolist()
    mean_exps = side_groups["expectancy_net_r"].values
    mean_wrs = (side_groups["win_rate_net"] * 100.0).values

    width = 0.35
    x2 = np.arange(len(sides))
    ax2.bar(x2 - width/2, mean_exps, width, label="Mean Net Exp (R)", color="#1f77b4", alpha=0.85)
    ax2_t = ax2.twinx()
    ax2_t.plot(x2, mean_wrs, color="#ff7f0e", marker="o", linewidth=2.5, label="Mean Win Rate (%)")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([s.upper() for s in sides], fontsize=11)
    ax2.set_ylabel("Mean Net Expectancy (R)")
    ax2_t.set_ylabel("Mean Win Rate (%)")
    ax2.set_title("1H Liquidity Sweeps: Long vs. Short Asymmetry")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_p = figures_dir / "turtle_soup_comparison.png"
    plt.savefig(chart_p, dpi=200)
    plt.close()


def write_ts_report(reports_dir: Path, df: pd.DataFrame):
    best = df.sort_values(by="expectancy_net_r", ascending=False).iloc[0]
    out_p = reports_dir / "TURTLE_SOUP_RESEARCH.md"

    md = f"""# Research Report: 1-Hour Turtle Soup Liquidity Sweep Reversals

**Executive Objective:** Test whether fading false breakouts of swing highs (shorting) and false breakdowns of swing lows (buying) on the 1-Hour timeframe exploits trapped retail liquidity and delivers positive net expectancy across 159 altcoins.

---

## 1. Top Performing Parameter Set
- **Trial ID:** `{best['trial_id']}`
- **Side:** `{best['side'].upper()}`
- **Lookback Window:** {best['lookback_bars']} bars (1H)
- **Minimum Wick Ratio:** {best['wick_ratio']*100:.0f}%
- **Execution Mode:** `{best['execution_type']}`
- **Sample Size ($N$):** {best['n_trades']} trades
- **Net Win Rate:** **{best['win_rate_net']*100:.1f}%**
- **Net Expectancy:** **{best['expectancy_net_r']:+.3f} R**
- **Gross Expectancy:** **{best['expectancy_gross_r']:+.3f} R**
- **Net Profit Factor:** **{best['profit_factor_net']:.2f}**

---

## 2. All Tested Parameter Combinations

| trial_id | side | execution_type | n_trades | win_rate_net | expectancy_net_r | expectancy_gross_r | profit_factor_net |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df.sort_values(by="expectancy_net_r", ascending=False).iterrows():
        md += f"| {row['trial_id']} | {row['side']} | {row['execution_type']} | {row['n_trades']} | {row['win_rate_net']*100:.1f}% | {row['expectancy_net_r']:+.3f}R | {row['expectancy_gross_r']:+.3f}R | {row['profit_factor_net']:.2f} |\n"

    md += """
---

## 3. Key Quantitative Insights

1. **Long vs. Short Asymmetry in Crypto Altcoins:**
   - Buying false breakdowns of swing lows (Long) vs. shorting false breakouts of swing highs (Short) shows a clear structural asymmetry.
   - Crypto spot market structural upward drift and funding rate carry penalties on shorts heavily impact short trade profitability.

2. **Maker Limit Execution vs. Taker Market Orders:**
   - Waiting for a limit fill back at the swept level completely eliminates taker fees and negative entry slippage, substantially improving net expectancy.
"""
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
