"""
run_phase4.py
Executes Phase 4: Execution Model and Trade Simulation.
Bar-by-bar deterministic simulation, non-holdout guard (< 2026-07-01),
net-of-cost as primary headline, multi-testing trial log, reconciliation with Phase 3,
and comprehensive reporting.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.config import load_config, AppConfig
from data.cache import ParquetCache
from strategy.btc_regime import BTCRegimeDetector
from strategy.indicators import compute_wilder_atr
from strategy.pullback_detector import ThresholdCrossingEvent
from strategy.entry_rules import EntryA_Immediate, EntryB_15mReversal, EntryC_LowerHighBreak, BaseEntryRule
from strategy.exit_rules import ExitRules, PullbackBufferStop
from backtest.trade import Trade
from backtest.execution_model import ExecutionModel
from backtest.engine import BacktestEngine
from analysis.trade_metrics import TradeMetricsCalculator
from analysis.reconciliation import PhaseReconciler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase4")


def preload_single_symbol(
    sym: str,
    cache_dir: str,
    results_dir: str,
    holdout_start: str,
    pullback_depth: float = 0.50
) -> Optional[Tuple[str, pd.DataFrame, pd.Series, List[ThresholdCrossingEvent]]]:
    cache = ParquetCache(cache_dir)
    df_15m = cache.load_klines(sym, "15m")
    if df_15m.empty:
        return None

    part_file = Path(results_dir) / "setups_partitions" / "fractal_n=3" / f"{sym}.parquet"
    if not part_file.is_file():
        return None

    df_part = pd.read_parquet(part_file)
    if df_part.empty:
        return None

    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")
    mask = (
        (df_part["is_controlled_pullback"] == True) &
        (df_part["threshold_type"] == "atr") &
        (df_part["threshold_value"] == pullback_depth) &
        (pd.to_datetime(df_part["event_time"], utc=True) < holdout_ts)
    )
    df_filtered = df_part[mask]
    if df_filtered.empty:
        return None

    fields = set(ThresholdCrossingEvent.__dataclass_fields__.keys())
    records = df_filtered.to_dict("records")
    events = [ThresholdCrossingEvent(**{k: r[k] for k in fields if k in r}) for r in records]

    # Pre-calculate causal ATR
    atr_series = compute_wilder_atr(df_15m["high"], df_15m["low"], df_15m["close"], 14)
    return (sym, df_15m, atr_series, events)


def run_simulation_in_memory(
    symbol_data: Dict[str, Tuple[pd.DataFrame, pd.Series, List[ThresholdCrossingEvent]]],
    exec_model: ExecutionModel,
    entry_rule: BaseEntryRule,
    exit_rules: ExitRules,
    holdout_start: str,
    btc_regime_df: Optional[pd.DataFrame] = None
) -> Tuple[List[Trade], int, int]:
    all_trades: List[Trade] = []
    total_rejected_rr = 0
    total_skipped_overlap = 0

    def _sim_worker(sym: str, data: Tuple[pd.DataFrame, pd.Series, List[ThresholdCrossingEvent]]):
        df_15m, atr_series, events = data
        engine = BacktestEngine(
            execution_model=exec_model,
            entry_rule=entry_rule,
            exit_rules=exit_rules,
            holdout_start=holdout_start
        )
        tr = engine.simulate_symbol(
            symbol=sym,
            df_15m=df_15m,
            atr_series=atr_series,
            events=events,
            btc_regime_df=btc_regime_df,
            allow_holdout=False
        )
        return tr, len(engine.rejected_min_rr), len(engine.position_manager.skipped_overlaps)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_sim_worker, sym, data) for sym, data in symbol_data.items()]
        for f in concurrent.futures.as_completed(futures):
            tr, rej, skip = f.result()
            all_trades.extend(tr)
            total_rejected_rr += rej
            total_skipped_overlap += skip

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades, total_rejected_rr, total_skipped_overlap


def plot_equity_curve(df_trades: pd.DataFrame, out_path: Path):
    if df_trades.empty:
        return

    df = df_trades.sort_values(by="entry_time").reset_index(drop=True)
    initial_cap = 10_000.0
    risk_pct = 0.01

    equity_net = [initial_cap]
    equity_gross = [initial_cap]

    for _, row in df.iterrows():
        eq_n = equity_net[-1]
        eq_g = equity_gross[-1]
        pnl_n = eq_n * risk_pct * row["net_pnl_r"]
        pnl_g = eq_g * risk_pct * row["gross_pnl_r"]
        equity_net.append(max(0.0, eq_n + pnl_n))
        equity_gross.append(max(0.0, eq_g + pnl_g))

    plt.figure(figsize=(11, 6))
    plt.plot(equity_net, label="Net of All Costs (Headline)", color="#0052FF", lw=2)
    plt.plot(equity_gross, label="Gross (Before Fees & Slippage)", color="#707070", lw=1.5, ls="--")
    plt.axhline(initial_cap, color="black", lw=0.8, ls=":")
    plt.title("Portfolio Equity Curve: MEXC Swing-High Retest (Non-Holdout)", fontsize=13, fontweight="bold")
    plt.xlabel("Trade Sequence Number", fontsize=11)
    plt.ylabel("Equity ($USD, $10,000 Base, 1% Risk)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_drawdown(df_trades: pd.DataFrame, out_path: Path):
    if df_trades.empty:
        return

    df = df_trades.sort_values(by="entry_time").reset_index(drop=True)
    initial_cap = 10_000.0
    risk_pct = 0.01

    equity = [initial_cap]
    for _, row in df.iterrows():
        curr = equity[-1]
        equity.append(max(0.0, curr + curr * risk_pct * row["net_pnl_r"]))

    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks * 100.0

    plt.figure(figsize=(11, 4))
    plt.fill_between(range(len(dd)), dd, 0, color="#FF3B30", alpha=0.4)
    plt.plot(dd, color="#D70015", lw=1.5)
    plt.title("Portfolio Drawdown Profile (Net of Costs)", fontsize=12, fontweight="bold")
    plt.xlabel("Trade Number", fontsize=11)
    plt.ylabel("Drawdown (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_entry_mode_comparison(trials: List[Dict[str, Any]], out_path: Path):
    modes = []
    win_rates = []
    exp_r = []
    for t in trials:
        if t.get("min_rr") == 1.0 and t.get("time_stop_hours") == 24.0 and t.get("stop_buffer") == 0.25:
            modes.append(t["entry_mode"])
            win_rates.append(t["win_rate_net"] * 100.0)
            exp_r.append(t["expectancy_net_r"])

    if not modes:
        return

    x = np.arange(len(modes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#0052FF"
    ax1.set_xlabel("Entry Mode", fontsize=11)
    ax1.set_ylabel("Net Win Rate (%)", color=color, fontsize=11)
    ax1.bar(x - width/2, win_rates, width, label="Net Win Rate (%)", color=color, alpha=0.8)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#FF9500"
    ax2.set_ylabel("Net Expectancy (R)", color=color, fontsize=11)
    ax2.bar(x + width/2, exp_r, width, label="Net Expectancy (R)", color=color, alpha=0.8)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.xticks(x, modes)
    plt.title("Comparison by Entry Mode (Net of All Costs)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    cfg = load_config("config/default.yaml")
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    reports_dir = Path(cfg.data.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 4: TRADE SIMULATION & EXECUTION MODEL (NON-HOLDOUT ONLY)")
    logger.info("=" * 60)

    # 1. Load BTC 1H Regime
    logger.info("Loading BTC 1H regime series...")
    btc_detector = BTCRegimeDetector(cache)
    btc_regime_df = btc_detector.compute_btc_1h_regime()

    # 2. Identify cached symbols and preload in memory
    symbols = cache.list_cached_symbols("15m")
    logger.info(f"Preloading {len(symbols)} symbols into memory...")
    t0_preload = time.time()
    symbol_data: Dict[str, Tuple[pd.DataFrame, pd.Series, List[ThresholdCrossingEvent]]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                preload_single_symbol,
                sym,
                cfg.data.cache_dir,
                cfg.data.results_dir,
                cfg.data.holdout_start,
                0.50
            ): sym for sym in symbols
        }
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                sym, df_15m, atr_series, events = res
                symbol_data[sym] = (df_15m, atr_series, events)

    n_loaded_events = sum(len(evs) for _, _, evs in symbol_data.values())
    logger.info(f"Preloaded {len(symbol_data)} valid symbols ({n_loaded_events:,} controlled events) in {time.time()-t0_preload:.2f}s.")

    # 3. Execution Model setup
    exec_defaults = cfg.execution_defaults
    exec_model = ExecutionModel(
        maker_fee_rate=exec_defaults.maker_fee_rate,
        taker_fee_rate=exec_defaults.taker_fee_rate,
        base_slippage_bps=exec_defaults.base_slippage_bps,
        funding_rate_8h=exec_defaults.funding_rate_8h,
        slippage_by_rank=exec_defaults.slippage_by_rank,
        target_as_limit=True
    )

    # --------------------------------------------------------------------------
    # 4. PRIMARY BASELINE SIMULATION:
    # Entry A (Immediate), Min R:R = 1.0, Stop Buffer = 0.25 ATR, Time Stop = 24h
    # --------------------------------------------------------------------------
    logger.info("Running Primary Baseline Trade Simulation...")
    t0 = time.time()
    primary_entry_rule = EntryA_Immediate(min_rr=1.0)
    primary_exit_rules = ExitRules(stop_rule=PullbackBufferStop(buffer_atr=0.25), time_stop_hours=24.0)

    trades_baseline, rej_baseline, skip_baseline = run_simulation_in_memory(
        symbol_data=symbol_data,
        exec_model=exec_model,
        entry_rule=primary_entry_rule,
        exit_rules=primary_exit_rules,
        holdout_start=cfg.data.holdout_start,
        btc_regime_df=btc_regime_df
    )
    logger.info(f"Baseline simulation finished in {time.time()-t0:.2f}s: {len(trades_baseline):,} trades executed.")

    # Convert baseline trades to DataFrame
    trade_dicts = [t.__dict__ for t in trades_baseline]
    df_trades_baseline = pd.DataFrame(trade_dicts)

    # Save Trade Database (Parquet + CSV per original spec section 26)
    trades_parquet_path = results_dir / "trades.parquet"
    trades_csv_path = results_dir / "trades.csv"
    logger.info(f"Saving Trade Database to {trades_parquet_path} and {trades_csv_path}...")
    df_trades_baseline.to_parquet(trades_parquet_path, index=False)
    df_trades_baseline.to_csv(trades_csv_path, index=False)

    # --------------------------------------------------------------------------
    # 5. MULTIPLE-TESTING PARAMETER SWEEP
    # Evaluates permutations and logs each to results/trials.parquet per Rule 6
    # --------------------------------------------------------------------------
    logger.info("Executing Parameter Variations for Multiple-Testing Discipline...")
    trials_records: List[Dict[str, Any]] = []

    sweep_configs = [
        # Baseline
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 24.0, "buffer": 0.25},
        # Entry modes
        {"mode": "Entry_B", "min_rr": 1.0, "time_stop": 24.0, "buffer": 0.25},
        {"mode": "Entry_C", "min_rr": 1.0, "time_stop": 24.0, "buffer": 0.25},
        # Min R:R sweeps
        {"mode": "Entry_A", "min_rr": 1.5, "time_stop": 24.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 2.0, "time_stop": 24.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 2.5, "time_stop": 24.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 3.0, "time_stop": 24.0, "buffer": 0.25},
        # Time stop sweeps
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 1.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 2.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 4.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 8.0, "buffer": 0.25},
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 12.0, "buffer": 0.25},
        # Stop buffer sweeps
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 24.0, "buffer": 0.10},
        {"mode": "Entry_A", "min_rr": 1.0, "time_stop": 24.0, "buffer": 0.50},
    ]

    for idx, sc in enumerate(sweep_configs):
        t_start = time.time()
        # Instantiate entry rule
        if sc["mode"] == "Entry_A":
            e_rule = EntryA_Immediate(min_rr=sc["min_rr"])
        elif sc["mode"] == "Entry_B":
            e_rule = EntryB_15mReversal(min_rr=sc["min_rr"])
        elif sc["mode"] == "Entry_C":
            e_rule = EntryC_LowerHighBreak(min_rr=sc["min_rr"])
        else:
            e_rule = EntryA_Immediate(min_rr=sc["min_rr"])

        x_rules = ExitRules(stop_rule=PullbackBufferStop(buffer_atr=sc["buffer"]), time_stop_hours=sc["time_stop"])

        if idx == 0:
            tr_list = trades_baseline
            rej_c = rej_baseline
            skip_c = skip_baseline
        else:
            tr_list, rej_c, skip_c = run_simulation_in_memory(
                symbol_data=symbol_data,
                exec_model=exec_model,
                entry_rule=e_rule,
                exit_rules=x_rules,
                holdout_start=cfg.data.holdout_start,
                btc_regime_df=btc_regime_df
            )

        df_t = pd.DataFrame([t.__dict__ for t in tr_list])
        metrics = TradeMetricsCalculator.compute_summary_metrics(df_t)

        trial_record = {
            "trial_id": f"TRIAL_{idx+1:03d}_{sc['mode']}_RR{sc['min_rr']}_H{sc['time_stop']}_B{sc['buffer']}",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "config_hash": cfg.config_hash,
            "entry_mode": sc["mode"],
            "min_rr": sc["min_rr"],
            "time_stop_hours": sc["time_stop"],
            "stop_buffer": sc["buffer"],
            "n_trades": metrics["n_trades"],
            "win_rate_net": metrics["win_rate_net"],
            "win_rate_net_ci_low": metrics.get("win_rate_net_ci_low", 0.0),
            "win_rate_net_ci_high": metrics.get("win_rate_net_ci_high", 0.0),
            "expectancy_net_r": metrics["expectancy_net_r"],
            "expectancy_net_pct": metrics["expectancy_net_pct"],
            "profit_factor_net": metrics["profit_factor_net"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "avg_holding_hours": metrics["avg_holding_hours"],
            "median_holding_hours": metrics["median_holding_hours"],
            "win_rate_gross": metrics["win_rate_gross"],
            "expectancy_gross_r": metrics["expectancy_gross_r"],
            "profit_factor_gross": metrics["profit_factor_gross"],
            "rejected_min_rr": rej_c,
            "skipped_overlap": skip_c
        }
        trials_records.append(trial_record)
        BacktestEngine.log_trial(trial_record, results_dir=results_dir)
        logger.info(f"Trial {idx+1}/{len(sweep_configs)} finished in {time.time()-t_start:.2f}s: {metrics['n_trades']:,} trades, Net Exp: {metrics['expectancy_net_r']:+.3f} R, PF: {metrics['profit_factor_net']:.2f}.")

    # --------------------------------------------------------------------------
    # 6. RECONCILIATION WITH PHASE 3
    # --------------------------------------------------------------------------
    logger.info("Reconciling Phase 4 Trade Universe against Phase 3 Event Setups...")
    phase3_event_ids = set()
    for _, _, evs in symbol_data.values():
        for e in evs:
            phase3_event_ids.add(e.event_id)

    reconciliation = PhaseReconciler.reconcile(
        df_trades=df_trades_baseline,
        n_events_total=n_loaded_events,
        rejected_rr_count=rej_baseline,
        skipped_overlap_count=skip_baseline,
        phase3_event_ids=phase3_event_ids
    )
    logger.info(reconciliation["reconciliation_summary"])

    # --------------------------------------------------------------------------
    # 7. METRIC COMPUTATION & BREAKDOWNS
    # --------------------------------------------------------------------------
    logger.info("Computing Segmented Performance Metrics...")
    baseline_summary = TradeMetricsCalculator.compute_summary_metrics(df_trades_baseline)
    segmentations = TradeMetricsCalculator.compute_segmentations(df_trades_baseline)

    # --------------------------------------------------------------------------
    # 8. VISUALIZATIONS
    # --------------------------------------------------------------------------
    logger.info("Generating Visual Charts...")
    plot_equity_curve(df_trades_baseline, figures_dir / "phase4_equity_curve.png")
    plot_drawdown(df_trades_baseline, figures_dir / "phase4_drawdown.png")
    plot_entry_mode_comparison(trials_records, figures_dir / "phase4_entry_modes.png")

    # --------------------------------------------------------------------------
    # 9. GENERATE RESEARCH REPORT (reports/TRADE_SIMULATION.md)
    # --------------------------------------------------------------------------
    logger.info("Writing Phase 4 Research Report...")
    write_trade_report(
        reports_dir / "TRADE_SIMULATION.md",
        baseline_summary=baseline_summary,
        segmentations=segmentations,
        trials=trials_records,
        reconciliation=reconciliation,
        cfg=cfg
    )

    logger.info("Phase 4 Execution and Trade Simulation Completed Successfully.")


def write_trade_report(
    report_path: Path,
    baseline_summary: Dict[str, Any],
    segmentations: Dict[str, Any],
    trials: List[Dict[str, Any]],
    reconciliation: Dict[str, Any],
    cfg: AppConfig
):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4 Research Report: Execution and Trade Simulation\n\n")
        f.write("## Executive Summary\n")
        f.write("> **Hypothesis:** Buying a controlled 15m pullback toward a prior swing high inside an established 1H uptrend yields a tradable edge on MEXC crypto markets.\n\n")

        net_exp_r = baseline_summary["expectancy_net_r"]
        gross_exp_r = baseline_summary["expectancy_gross_r"]
        net_wr = baseline_summary["win_rate_net"] * 100.0
        pf_net = baseline_summary["profit_factor_net"]
        n_trades = baseline_summary["n_trades"]
        ci_l = baseline_summary["win_rate_net_ci_low"] * 100.0
        ci_h = baseline_summary["win_rate_net_ci_high"] * 100.0

        f.write("### Plain Language Verdict\n")
        f.write(f"- Across **{n_trades:,}** simulated trades under realistic execution frictions (maker target fills, taker stop fills, volume-rank slippage, and 8h funding fees) from October 2025 to June 2026 (non-holdout):\n")
        f.write(f"- **Primary Net Expectancy:** **{net_exp_r:+.4f} R per trade** (Net Return: **{baseline_summary['expectancy_net_pct']*100:+.2f}%** per trade).\n")
        f.write(f"- **Gross Expectancy (Secondary):** **{gross_exp_r:+.4f} R per trade**.\n")
        f.write(f"- **Net Win Rate:** **{net_wr:.2f}%** (95% Wilson CI: [{ci_l:.2f}%, {ci_h:.2f}%]).\n")
        f.write(f"- **Net Profit Factor:** **{pf_net:.2f}** (Gross Profit Factor: **{baseline_summary['profit_factor_gross']:.2f}**).\n")
        f.write(f"- **Max Drawdown (Fixed 1% Risk):** **{baseline_summary['max_drawdown_pct']:.2f}%**.\n")
        f.write(f"- **Annualized Sharpe / Sortino:** **{baseline_summary['sharpe_ratio']:.2f}** / **{baseline_summary['sortino_ratio']:.2f}** (*{baseline_summary['sharpe_assumptions']}*).\n\n")

        if net_exp_r > 0 and pf_net > 1.05:
            f.write("**VERDICT: MEASURABLE EDGE CONFIRMED AFTER COSTS.** The strategy maintains positive net expectancy and robust profit factor after accounting for taker fees, rank-scaled slippage, gap penalties, and funding rates.\n\n")
        else:
            f.write("**VERDICT: NO ECONOMIC EDGE AFTER COSTS.** While gross expectancy is mildly positive or neutral, trading frictions (spread, taker fees on stops, stop slippage, and gap execution) consume the margin, resulting in negative or sub-threshold net expectancy.\n\n")

        f.write("---\n\n")
        f.write("## 1. Primary Strategy Performance (Headline: Net of Costs)\n\n")
        f.write("| Metric | Net-of-Cost (Primary) | Gross (Secondary) |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| **Total Completed Trades** | **{n_trades:,}** | {n_trades:,} |\n")
        f.write(f"| **Win Rate** | **{net_wr:.2f}%** [{ci_l:.2f}%, {ci_h:.2f}%] | {baseline_summary['win_rate_gross']*100:.2f}% |\n")
        f.write(f"| **Expectancy (R-Multiple)** | **{net_exp_r:+.4f} R** | {gross_exp_r:+.4f} R |\n")
        f.write(f"| **Expectancy (%)** | **{baseline_summary['expectancy_net_pct']*100:+.3f}%** | {(baseline_summary['expectancy_net_pct'] + baseline_summary['total_fees_pct']/n_trades)*100:+.3f}% |\n")
        f.write(f"| **Profit Factor** | **{pf_net:.2f}** | {baseline_summary['profit_factor_gross']:.2f} |\n")
        f.write(f"| **Avg Win / Avg Loss (R)** | **+{baseline_summary['avg_win_net_r']:.2f} R / -{abs(baseline_summary['avg_loss_net_r']):.2f} R** | - |\n")
        f.write(f"| **Max Drawdown (1% Risk)** | **{baseline_summary['max_drawdown_pct']:.2f}%** | - |\n")
        f.write(f"| **Sharpe Ratio (Annualized)** | **{baseline_summary['sharpe_ratio']:.2f}** | - |\n")
        f.write(f"| **Sortino Ratio (Annualized)** | **{baseline_summary['sortino_ratio']:.2f}** | - |\n")
        f.write(f"| **Avg Holding Time** | **{baseline_summary['avg_holding_hours']:.1f} hours** ({baseline_summary['median_holding_hours']:.1f}h median) | - |\n")
        f.write(f"| **Total Cost Impact** | **Fees: {baseline_summary['total_fees_pct']*100:.1f}%, Slip: {baseline_summary['total_slippage_pct']*100:.1f}%, Funding: {baseline_summary['total_funding_pct']*100:.2f}%** | - |\n\n")

        f.write("### Exit Reason Distribution\n")
        for reason, count in baseline_summary.get("exit_counts", {}).items():
            pct = count / n_trades * 100.0 if n_trades > 0 else 0.0
            f.write(f"- **{reason.upper()}**: {count:,} ({pct:.1f}%)\n")
        f.write(f"- **Ambiguous Same-Bar Events** (Conservative Stop-First Triggered): {baseline_summary.get('ambiguous_same_bar_count', 0):,}\n\n")

        f.write("---\n\n")
        f.write("## 2. Reconciliation with Phase 3 Event Study\n\n")
        f.write(f"> {reconciliation['reconciliation_summary']}\n\n")
        f.write("| Funnel Stage | Count | Share of Phase 3 Setups |\n")
        f.write("|---|---:|---:|\n")
        tot = reconciliation['total_phase3_events']
        f.write(f"| **Phase 3 Controlled Pullback Setups (0.50 ATR, Trend C)** | {tot:,} | 100.0% |\n")
        f.write(f"| **Rejected by Min R:R Filter (< 1.0)** | {reconciliation['rejected_min_rr']:,} | {reconciliation['rejected_min_rr']/tot*100:.2f}% |\n")
        f.write(f"| **Skipped Due to Open Position Overlap (Max 1 / Symbol)** | {reconciliation['skipped_overlap']:,} | {reconciliation['skipped_overlap']/tot*100:.2f}% |\n")
        f.write(f"| **Executed Trades** | **{reconciliation['executed_trades']:,}** | **{reconciliation['executed_trades']/tot*100:.2f}%** |\n")
        f.write(f"| **Discrepancies / Orphaned Trades** | **{reconciliation['discrepancies_count']}** | **0.00%** |\n\n")

        f.write("---\n\n")
        f.write("## 3. Multiple-Testing Parameter Surface (Trial Log)\n\n")
        f.write("All parameter variations are logged in `results/trials.parquet` per Rule 6:\n\n")
        f.write("| Trial ID | Entry Mode | Min R:R | Time Stop | Stop Buffer | Trades ($n$) | Win Rate (Net) | Net Exp (R) | Profit Factor | Max DD |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for t in trials:
            f.write(f"| `{t['trial_id']}` | {t['entry_mode']} | {t['min_rr']} | {t['time_stop_hours']}h | {t['stop_buffer']} ATR | {t['n_trades']:,} | {t['win_rate_net']*100:.1f}% | **{t['expectancy_net_r']:+.3f} R** | {t['profit_factor_net']:.2f} | {t['max_drawdown_pct']:.1f}% |\n")
        f.write("\n---\n\n")

        f.write("## 4. Segmentations & Sub-Regime Analysis\n\n")
        # Volume Rank
        f.write("### By Volume Rank Bucket (MEXC Universe)\n")
        f.write("| Volume Rank | Trades ($n$) | Win Rate (Net) | 95% Wilson CI | Net Exp (R) | Profit Factor |\n")
        f.write("|---|---:|---:|---|---:|---:|\n")
        for bucket, m in segmentations.get("volume_rank", {}).items():
            wr = m["win_rate_net"] * 100.0
            ci_str = f"[{m['win_rate_net_ci_low']*100:.1f}%, {m['win_rate_net_ci_high']*100:.1f}%]"
            f.write(f"| **{bucket}** | {m['n_trades']:,} | {wr:.2f}% | {ci_str} | **{m['expectancy_net_r']:+.3f} R** | {m['profit_factor_net']:.2f} |\n")

        # BTC Regime
        f.write("\n### By BTC Macro Regime\n")
        f.write("| BTC Regime | Trades ($n$) | Win Rate (Net) | Net Exp (R) | Profit Factor |\n")
        f.write("|---|---:|---:|---|---:|---:|\n")
        for reg, m in segmentations.get("btc_regime", {}).items():
            wr = m["win_rate_net"] * 100.0
            f.write(f"| **{reg}** | {m['n_trades']:,} | {wr:.2f}% | **{m['expectancy_net_r']:+.3f} R** | {m['profit_factor_net']:.2f} |\n")

        f.write("\n---\n\n")
        f.write("## 5. Artifacts and Visualization Links\n")
        f.write("- **Primary Trade Database (Parquet):** `results/trades.parquet`\n")
        f.write("- **Primary Trade Database (CSV):** `results/trades.csv`\n")
        f.write("- **Trial Log (Multiple-Testing):** `results/trials.parquet`\n")
        f.write("- **Equity Curve Chart:** `reports/figures/phase4_equity_curve.png`\n")
        f.write("- **Drawdown Profile Chart:** `reports/figures/phase4_drawdown.png`\n")
        f.write("- **Entry Mode Comparison Chart:** `reports/figures/phase4_entry_modes.png`\n\n")

        f.write("---\n\n")
        f.write("## 6. Phase 4 Conclusion & Transition to Phase 5\n")
        f.write("All simulations strictly observed the non-holdout guard (< 2026-07-01). The holdout dataset remains pristine and locked for Phase 5 Walk-Forward Validation.\n")


if __name__ == "__main__":
    main()
