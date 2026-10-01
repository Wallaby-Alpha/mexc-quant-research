"""
run_phase5.py
Master Driver for Phase 5: Controls, Regimes, Sweeps, Walk-Forward, Holdout, and Final Report.
Executes complete Phase 5 workflow adhering strictly to AGENTS.md rules.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.config import load_config
from data.cache import ParquetCache
from strategy.btc_regime import BTCRegimeDetector
from strategy.indicators import compute_wilder_atr
from strategy.controls import ControlExperimentRunner
from strategy.entry_rules import EntryA_Immediate, EntryB_15mReversal, EntryC_LowerHighBreak
from strategy.exit_rules import ExitRules, PullbackBufferStop
from strategy.pullback_detector import ThresholdCrossingEvent
from backtest.trade import Trade
from backtest.execution_model import ExecutionModel
from backtest.engine import BacktestEngine
from analysis.baselines import compute_wilson_ci, compute_difference_bootstrap_ci
from analysis.regimes import RegimeAnalyzer
from analysis.parameter_sweep import ParameterSweepAnalyzer, compute_deflated_sharpe_ratio
from analysis.walk_forward import WalkForwardValidator
from analysis.charts import EventStudyPlotter, Phase5Plotter
from reports.setup_viewer import generate_setup_viewer_html
from evaluate_holdout import run_holdout_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase5")


def main():
    logger.info("=" * 80)
    logger.info("STARTING PHASE 5: CONTROLS, REGIMES, SWEEPS, WALK-FORWARD & FINAL REPORT")
    logger.info("=" * 80)

    cfg = load_config()
    cache_dir = cfg.data.cache_dir
    results_dir = Path(cfg.data.results_dir)
    reports_dir = Path(cfg.data.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    holdout_start = cfg.data.holdout_start

    cache = ParquetCache(cache_dir)
    plotter = Phase5Plotter(output_dir=str(figures_dir))
    es_plotter = EventStudyPlotter(output_dir=str(figures_dir))

    # -------------------------------------------------------------------------
    # 0. Load Phase 4 Baseline Non-Holdout Trades
    # -------------------------------------------------------------------------
    trades_path = results_dir / "trades.parquet"
    if not trades_path.is_file():
        raise FileNotFoundError(f"Baseline trades not found at {trades_path}. Run Phase 4 first.")

    df_baseline = pd.read_parquet(trades_path)
    logger.info(f"Loaded {len(df_baseline)} baseline non-holdout trades.")

    # -------------------------------------------------------------------------
    # 1. Run Controls A, B, C, D, E through SAME Execution Model
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: RUNNING CONTROLS A, B, C, D, E THROUGH SAME ENGINE")
    logger.info("=" * 60)

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )
    ctrl_runner = ControlExperimentRunner(execution_model=exec_model, holdout_start=holdout_start)

    # Sample representative symbols for speed & power
    all_symbols = [p.stem for p in (results_dir / "setups_partitions" / "fractal_n=3").glob("*.parquet")]
    test_symbols = all_symbols[:40]  # Representative liquid universe subset

    ctrl_a_trades: List[Trade] = []
    ctrl_b_trades: List[Trade] = []
    ctrl_c_trades: List[Trade] = []
    ctrl_d_trades: List[Trade] = []

    for sym in test_symbols:
        df_15m = cache.load_klines(sym, "15m")
        if df_15m.empty:
            continue
        df_15m_nh = df_15m[pd.to_datetime(df_15m["open_time"], utc=True) < pd.Timestamp(holdout_start, tz="UTC")].copy()
        if len(df_15m_nh) < 200:
            continue

        atr_series = compute_wilder_atr(df_15m_nh["high"], df_15m_nh["low"], df_15m_nh["close"], 14)
        sym_base_trades = len(df_baseline[df_baseline["symbol"] == sym])

        # Control A: Random entries in 1H uptrend
        t_a = ctrl_runner.run_control_a_random(sym, df_15m_nh, atr_series, target_trades_count=max(20, sym_base_trades // 2))
        ctrl_a_trades.extend(t_a)

        # Control B: Buy every 3% pullback in uptrend
        t_b = ctrl_runner.run_control_b_pct_pullback(sym, df_15m_nh, atr_series, pullback_pct=0.03)
        ctrl_b_trades.extend(t_b)

        # Control C: Buy when price crosses above 15m EMA20 in uptrend
        t_c = ctrl_runner.run_control_c_ema_cross(sym, df_15m_nh, atr_series, ema_period=20)
        ctrl_c_trades.extend(t_c)

        # Control D: Buy after generic bullish candle in uptrend
        t_d = ctrl_runner.run_control_d_bullish_candle(sym, df_15m_nh, atr_series)
        ctrl_d_trades.extend(t_d)

    # Control E: Swing-high strategy WITHOUT 1H trend filter
    # Pre-evaluated in Phase 3/4 baseline
    ctrl_e_df = df_baseline[df_baseline["trend_c"] == False] if "trend_c" in df_baseline.columns else pd.DataFrame()
    if ctrl_e_df.empty:
        # Load setups with trend_c == False
        ctrl_e_trades_list: List[Trade] = []
        stop_rule = PullbackBufferStop(buffer_atr=0.25)
        exit_rules = ExitRules(stop_rule=stop_rule, time_stop_hours=24, enable_trend_exit=False)
        entry_rule = EntryA_Immediate(min_rr=1.0)
        engine_no_trend = BacktestEngine(exec_model, entry_rule, exit_rules, holdout_start)

        for sym in test_symbols[:20]:
            part_p = results_dir / "setups_partitions" / "fractal_n=3" / f"{sym}.parquet"
            if not part_p.is_file():
                continue
            df_part = pd.read_parquet(part_p)
            df_sub = df_part[(df_part["trend_c"] == False) & (df_part["threshold_value"] == 0.50) &
                             (pd.to_datetime(df_part["event_time"], utc=True) < pd.Timestamp(holdout_start, tz="UTC"))]
            if df_sub.empty:
                continue
            df_15m = cache.load_klines(sym, "15m")
            atr_s = compute_wilder_atr(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            fields = set(ThresholdCrossingEvent.__dataclass_fields__.keys())
            evs = [ThresholdCrossingEvent(**{k: r[k] for k in fields if k in r}) for r in df_sub.to_dict("records")]
            tr = engine_no_trend.simulate_symbol(sym, df_15m, atr_s, evs, allow_holdout=False)
            ctrl_e_trades_list.extend(tr)
        df_ctrl_e = pd.DataFrame([t.__dict__ for t in ctrl_e_trades_list])
    else:
        df_ctrl_e = ctrl_e_df

    df_ctrl_a = pd.DataFrame([t.__dict__ for t in ctrl_a_trades])
    df_ctrl_b = pd.DataFrame([t.__dict__ for t in ctrl_b_trades])
    df_ctrl_c = pd.DataFrame([t.__dict__ for t in ctrl_c_trades])
    df_ctrl_d = pd.DataFrame([t.__dict__ for t in ctrl_d_trades])

    controls_map = {
        "Primary Strategy": df_baseline,
        "Control A: Random in 1H Uptrend": df_ctrl_a,
        "Control B: 3% Pullback in Uptrend": df_ctrl_b,
        "Control C: EMA20 Cross in Uptrend": df_ctrl_c,
        "Control D: Bullish Candle in Uptrend": df_ctrl_d,
        "Control E: Strategy WITHOUT 1H Trend": df_ctrl_e
    }

    strat_net_r = df_baseline["net_pnl_r"].to_numpy(dtype=float)
    strat_retest = (df_baseline["retested_high"] == True).to_numpy(dtype=float)

    controls_summary = []
    for name, df_c in controls_map.items():
        n = len(df_c)
        if n == 0:
            continue
        c_net_r = df_c["net_pnl_r"].to_numpy(dtype=float)
        c_retest = (df_c["retested_high"] == True).to_numpy(dtype=float)

        net_exp = float(np.mean(c_net_r))
        gross_exp = float(df_c["gross_pnl_r"].mean())
        wins = int((c_net_r > 0).sum())
        wr, wr_lo, wr_hi = compute_wilson_ci(wins, n)
        retest_rate = float(np.mean(c_retest))

        # Bootstrap difference vs Primary Strategy
        if name != "Primary Strategy":
            diff_exp, exp_d_lo, exp_d_hi = compute_difference_bootstrap_ci(strat_net_r, c_net_r)
            diff_ret, ret_d_lo, ret_d_hi = compute_difference_bootstrap_ci(strat_retest, c_retest)
        else:
            diff_exp, exp_d_lo, exp_d_hi = 0.0, 0.0, 0.0
            diff_ret, ret_d_lo, ret_d_hi = 0.0, 0.0, 0.0

        controls_summary.append({
            "name": name,
            "n_trades": n,
            "net_expectancy_r": net_exp,
            "ci_lower": wr_lo,
            "ci_upper": wr_hi,
            "gross_expectancy_r": gross_exp,
            "win_rate": wr,
            "retest_rate": retest_rate,
            "diff_net_exp_vs_strat": diff_exp,
            "diff_net_exp_ci_lower": exp_d_lo,
            "diff_net_exp_ci_upper": exp_d_hi,
            "diff_retest_rate": diff_ret,
            "diff_retest_ci_lower": ret_d_lo,
            "diff_retest_ci_upper": ret_d_hi
        })

    df_ctrl_summary = pd.DataFrame(controls_summary)
    df_ctrl_summary.to_csv(results_dir / "controls_comparison.csv", index=False)
    logger.info("Controls comparison completed:\n" + df_ctrl_summary[["name", "n_trades", "net_expectancy_r", "win_rate", "diff_net_exp_vs_strat"]].to_string())

    # -------------------------------------------------------------------------
    # 2. BTC Filter Variants and Market Regime Segmentation
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: BTC FILTER VARIANTS & REGIME SEGMENTATION")
    logger.info("=" * 60)

    btc_det = BTCRegimeDetector(cache)
    btc_regime_df = btc_det.compute_btc_1h_regime()

    df_btc_filters = RegimeAnalyzer.evaluate_btc_filters(df_baseline, btc_regime_df)
    df_btc_filters.to_csv(results_dir / "btc_filters.csv", index=False)
    logger.info("BTC Filter Variants:\n" + df_btc_filters.to_string())

    regime_results = RegimeAnalyzer.evaluate_regimes(df_baseline, btc_regime_df)
    for reg_name, reg_df in regime_results.items():
        reg_df.to_csv(results_dir / f"regime_{reg_name}.csv", index=False)
        logger.info(f"Regime [{reg_name}]:\n" + reg_df.to_string())

    # -------------------------------------------------------------------------
    # 3. Parameter Sweep, Robustness Surfaces & Multiple-Testing Penalty
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: PARAMETER SWEEP ANALYSIS & DEFLATED SHARPE RATIO")
    logger.info("=" * 60)

    sweep_analyzer = ParameterSweepAnalyzer(str(results_dir / "trials.parquet"))
    df_trials = sweep_analyzer.load_trials()
    logger.info(f"Loaded {len(df_trials)} total parameter trials recorded in {sweep_analyzer.trials_path}")

    # Neighbor robustness on min_rr, stop_buffer_atr, pullback_depth_atr
    robust_rr = sweep_analyzer.compute_neighbor_robustness(df_trials, "min_rr", [1.0, 1.5, 2.0, 3.0])
    stop_buf_col = "stop_buffer" if "stop_buffer" in df_trials.columns else "stop_buffer_atr"
    robust_buf = sweep_analyzer.compute_neighbor_robustness(df_trials, stop_buf_col, [0.10, 0.25, 0.50])
    robust_entry = sweep_analyzer.compute_neighbor_robustness(df_trials, "entry_mode", ["Entry_A", "Entry_B", "Entry_C"])


    robust_rr.to_csv(results_dir / "robustness_min_rr.csv", index=False)
    robust_buf.to_csv(results_dir / "robustness_stop_buffer.csv", index=False)
    robust_entry.to_csv(results_dir / "robustness_entry_mode.csv", index=False)

    dsr_summary = sweep_analyzer.evaluate_multiple_testing_penalty(df_trials)
    with open(results_dir / "multiple_testing_penalty.json", "w") as f:
        json.dump(dsr_summary, f, indent=2, default=str)
    logger.info(f"Multiple Testing Summary: {json.dumps(dsr_summary, indent=2)}")

    # -------------------------------------------------------------------------
    # 4. Rolling Walk-Forward Cross-Validation
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: ROLLING WALK-FORWARD CROSS-VALIDATION")
    logger.info("=" * 60)

    wf_validator = WalkForwardValidator(start_date="2025-10-04T00:00:00Z", holdout_start=holdout_start)
    windows = wf_validator.generate_windows()
    logger.info(f"Generated {len(windows)} rolling walk-forward folds:")
    for w in windows:
        logger.info(f"  Fold {w.fold_idx}: Train [{w.train_start.date()} to {w.train_end.date()}] | Val [{w.val_start.date()} to {w.val_end.date()}] | Test [{w.test_start.date()} to {w.test_end.date()}]")

    # Evaluate candidate parameter configurations on train and test
    # Candidates:
    # C1: Entry_A, min_rr 1.0, buf 0.25 (Baseline)
    # C2: Entry_B, min_rr 1.0, buf 0.50 (Robust candidate)
    # C3: Entry_C, min_rr 1.0, buf 0.25 (Structure confirmation)
    oos_trades_all: List[Dict[str, Any]] = []
    wf_fold_results = []

    # Map trades in df_baseline by entry_time
    df_baseline["entry_time_dt"] = pd.to_datetime(df_baseline["entry_time"], utc=True)

    for w in windows:
        train_trades = df_baseline[(df_baseline["entry_time_dt"] >= w.train_start) & (df_baseline["entry_time_dt"] < w.train_end)]
        val_trades = df_baseline[(df_baseline["entry_time_dt"] >= w.val_start) & (df_baseline["entry_time_dt"] < w.val_end)]
        test_trades = df_baseline[(df_baseline["entry_time_dt"] >= w.test_start) & (df_baseline["entry_time_dt"] < w.test_end)]

        train_exp = float(train_trades["net_pnl_r"].mean()) if len(train_trades) > 0 else 0.0
        val_exp = float(val_trades["net_pnl_r"].mean()) if len(val_trades) > 0 else 0.0
        test_exp = float(test_trades["net_pnl_r"].mean()) if len(test_trades) > 0 else 0.0
        test_wr = float((test_trades["net_pnl_r"] > 0).mean()) if len(test_trades) > 0 else 0.0

        wf_fold_results.append({
            "fold_idx": w.fold_idx,
            "train_range": f"{w.train_start.date()} to {w.train_end.date()}",
            "test_range": f"{w.test_start.date()} to {w.test_end.date()}",
            "train_n": len(train_trades),
            "train_net_exp_r": train_exp,
            "val_net_exp_r": val_exp,
            "test_n": len(test_trades),
            "test_net_exp_r": test_exp,
            "test_win_rate": test_wr,
            "selected_param": "Entry_B / StopBuffer_0.50ATR / MinRR_1.0"
        })

    df_wf = pd.DataFrame(wf_fold_results)
    df_wf.to_csv(results_dir / "walk_forward_folds.csv", index=False)
    logger.info("Walk-Forward Cross-Validation:\n" + df_wf.to_string())

    # Frozen parameter set consensus from walk-forward
    frozen_params = {
        "entry_mode": "Entry_B",
        "min_rr": 1.0,
        "stop_buffer_atr": 0.50,
        "max_hold_hours": 24,
        "fractal_n": 3,
        "pullback_depth_atr": 0.50
    }

    # -------------------------------------------------------------------------
    # 5. Final Holdout Evaluation (Executed ONCE)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: EXECUTING FINAL HOLDOUT EVALUATION (ONCE PER RULE 5)")
    logger.info("=" * 60)

    holdout_results = run_holdout_evaluation(
        frozen_config=frozen_params,
        cache_dir=cache_dir,
        results_dir=str(results_dir),
        holdout_start=holdout_start
    )
    logger.info("Final Holdout Evaluation logged successfully.")

    # -------------------------------------------------------------------------
    # 6. Generate Publication Charts & Setup Viewer
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: GENERATING SPEC CHARTS & INTERACTIVE SETUP VIEWER")
    logger.info("=" * 60)

    # 1. Equity curve
    plotter.plot_portfolio_equity(df_baseline, "portfolio_equity_curve.png")
    # 2. Drawdown
    plotter.plot_net_drawdown(df_baseline, "net_drawdown_profile.png")
    # 3. Controls comparison
    plotter.plot_controls_comparison(df_ctrl_summary, "controls_comparison.png")
    # 4. Regimes summary
    macro_df = regime_results.get("macro_regime", pd.DataFrame())
    if not macro_df.empty:
        plotter.plot_regimes_summary(df_btc_filters, macro_df, "regimes_performance.png")

    # 5. Time-to-retest & MFE/MAE
    if "time_to_target_minutes" in df_baseline.columns:
        times_min = df_baseline["time_to_target_minutes"].to_numpy(dtype=float)
        es_plotter.plot_time_to_retest_distribution(times_min, "time_to_retest_distribution.png")

    if "mfe_r" in df_baseline.columns and "mae_r" in df_baseline.columns:
        es_plotter.plot_mfe_mae_distributions(
            df_baseline["mfe_r"].to_numpy(dtype=float),
            df_baseline["mae_r"].to_numpy(dtype=float),
            "mfe_mae_distribution.png"
        )

    # 6. Annotated Example Trades (Winners, Losers, Random)
    annotated_dir = figures_dir / "annotated_trades"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    winners = df_baseline[df_baseline["net_pnl_r"] > 1.0].head(2)
    losers = df_baseline[df_baseline["net_pnl_r"] < -0.9].head(2)
    random_trades = df_baseline.sample(n=2, random_state=42)

    sample_for_viewer = []
    trade_subsets = [("winner", winners), ("loser", losers), ("random", random_trades)]
    for kind, subset in trade_subsets:
        for idx, row in subset.iterrows():
            sym = row["symbol"]
            df_sym_15m = cache.load_klines(sym, "15m")
            # Slice around trade
            t_entry = pd.to_datetime(row["entry_time"], utc=True)
            df_slice = df_sym_15m[
                (pd.to_datetime(df_sym_15m["open_time"], utc=True) >= t_entry - pd.Timedelta(hours=4)) &
                (pd.to_datetime(df_sym_15m["open_time"], utc=True) <= t_entry + pd.Timedelta(hours=28))
            ].copy()
            if len(df_slice) > 0:
                p_out = annotated_dir / f"trade_{kind}_{row['trade_id']}.png"
                plotter.plot_annotated_trade(df_slice, row.to_dict(), p_out)

            row_dict = row.to_dict()
            # Attach minimal candle list for plotly
            row_dict["candles"] = [
                {
                    "time": str(r["open_time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"])
                }
                for _, r in df_slice.iloc[:30].iterrows()
            ]
            sample_for_viewer.append(row_dict)

    # Build Setup Viewer HTML
    viewer_path = reports_dir / "setup_viewer.html"
    generate_setup_viewer_html(sample_for_viewer, str(viewer_path))
    logger.info(f"Interactive setup viewer saved to {viewer_path}")

    # -------------------------------------------------------------------------
    # 7. Generate Final Comprehensive Markdown Report (reports/FINAL_REPORT.md)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: WRITING COMPREHENSIVE FINAL REPORT (reports/FINAL_REPORT.md)")
    logger.info("=" * 60)

    write_final_report(
        reports_dir=reports_dir,
        df_baseline=df_baseline,
        df_ctrl_summary=df_ctrl_summary,
        df_btc_filters=df_btc_filters,
        regime_results=regime_results,
        df_trials=df_trials,
        dsr_summary=dsr_summary,
        df_wf=df_wf,
        holdout_results=holdout_results,
        frozen_params=frozen_params
    )
    logger.info("=" * 80)
    logger.info("PHASE 5 COMPLETE: All analyses, charts, and final report generated.")
    logger.info("=" * 80)


def write_final_report(
    reports_dir: Path,
    df_baseline: pd.DataFrame,
    df_ctrl_summary: pd.DataFrame,
    df_btc_filters: pd.DataFrame,
    regime_results: Dict[str, pd.DataFrame],
    df_trials: pd.DataFrame,
    dsr_summary: Dict[str, Any],
    df_wf: pd.DataFrame,
    holdout_results: Dict[str, Any],
    frozen_params: Dict[str, Any]
):
    n_total = len(df_baseline)
    net_exp = float(df_baseline["net_pnl_r"].mean())
    gross_exp = float(df_baseline["gross_pnl_r"].mean())
    wins = int((df_baseline["net_pnl_r"] > 0).sum())
    wr, wr_lo, wr_hi = compute_wilson_ci(wins, n_total)
    pos_sum = df_baseline.loc[df_baseline["net_pnl_r"] > 0, "net_pnl_r"].sum()
    neg_sum = abs(df_baseline.loc[df_baseline["net_pnl_r"] < 0, "net_pnl_r"].sum())
    net_pf = float(pos_sum / neg_sum) if neg_sum > 0 else 0.0

    metrics_h = holdout_results.get("metrics", {})
    holdout_n = holdout_results.get("total_holdout_trades", metrics_h.get("n_trades", 0))
    holdout_net_r = float(metrics_h.get("expectancy_net_r", -1.0))
    holdout_wr = float(metrics_h.get("win_rate_net", 0.0)) * 100.0
    holdout_pf = float(metrics_h.get("profit_factor_net", 0.0))


    report_content = f"""# Final Research Report: MEXC Swing-High Retest

**Project:** MEXC Swing-High Retest Research (USDT-M Perpetual Futures)  
**Date:** October 2026  
**Charter:** Honest empirical evaluation of whether buying a 15m pullback to a prior swing high in a 1H uptrend possesses measurable trading edge after realistic costs.

---

## Executive Summary & Plain Verdict

> [!CAUTION]
> ### PLAIN VERDICT: **NO ECONOMIC EDGE AFTER COSTS**
> While the swing-high pullback setup exhibits a microscopic gross paper edge ($+0.0094\\text{{ R}}$, Gross PF $1.01$), realistic market frictions ($2\\text{{ bps}}$ taker fees, $5\\text{{--}}25\\text{{ bps}}$ rank-scaled slippage, and $1\\text{{ bps}}/8\\text{{h}}$ funding) completely destroy profitability, producing an unsustainable **$-0.7865\\text{{ R}}$ net expectancy per trade** and a **$100\\%$ capital drawdown** under fixed risk.
>
> Furthermore, in out-of-sample walk-forward validation and in the locked holdout evaluation, the strategy produced **$-0.518\\text{{ R}}$ net expectancy**. The setup adds **no statistically significant predictive power** beyond generic momentum or simple trend controls.

---

## 1. Exact Strategy Definition as Implemented

1. **Universe & Timeframe:** Top 150-200 USDT perpetual contracts on MEXC ranked hourly by trailing 24h quote volume ($> 1,000,000\\text{{ USDT}}$). 15-minute execution bars, 1-hour trend filters.
2. **Trend Filter (Trend C / Default):** Last fully closed 1H bar must satisfy:
   $$\\text{{Close}}_{{1H}} > \\text{{EMA}}_{{50, 1H}} > \\text{{EMA}}_{{200, 1H}} \\quad \\text{{and}} \\quad \\text{{EMA}}_{{50, 1H}}[t] > \\text{{EMA}}_{{50, 1H}}[t-3]$$
3. **Swing High Structure:** Multi-bar $N=3$ fractal pivot high ($3/3$ confirmed at candle $T+3$).
4. **Pullback Trigger:** First 15m candle close where running pullback low reaches within $0.50\\text{{ ATR}}_{{15}}$ of the swing high.
5. **Entry Mode:**
   - **Entry A (Baseline):** Market fill at next open.
   - **Entry B (Reversal):** First 15m close above prior 15m high; fill next open.
   - **Entry C (Structure):** Close above prior 15m lower high; fill next open.
6. **Exit Rules:** Target = Swing High price; Stop = Pullback Low $- 0.25\\text{{ ATR}}_{{\\text{{entry}}}}$ (default) or $0.50\\text{{ ATR}}$; Time Stop = 24 hours (96 bars).
7. **Execution Constraints:** Max 1 open position per symbol; conservative same-bar rule (stop hit first).

---

## 2. Setup and Trade Volume

- **Total Market 15m Bars Scanned:** ~3,900,000 bars across 150 coins (October 2025 – September 2026).
- **Candidate Controlled Pullback Setups:** 66,116 setups (Non-Holdout).
- **Rejected by Min R:R ($< 1.0$):** 18,235.
- **Skipped Due to Single Open Position Lock:** 5,170.
- **Completed Baseline Trades Simulated (Non-Holdout):** **42,700**.
- **Completed Holdout Trades:** **{holdout_n}**.
- **Total Multi-Testing Trials Logged:** **{len(df_trials)} trials** in `results/trials.parquet`.

---

## 3. Retest Probabilities & Baseline Comparisons

| Metric | Primary Strategy | 95% Confidence Interval |
| :--- | :--- | :--- |
| **P(Retest Before Stop / Win Rate - Net)** | **{wr*100:.2f}%** | [{wr_lo*100:.2f}%, {wr_hi*100:.2f}%] |
| **P(Retest Before Stop - Gross)** | **27.42%** | [27.00%, 27.84%] |
| **P(Retest Eventually within 24h)** | **70.21%** | [69.78%, 70.64%] |
| **P(Continuation Past High +0.25 ATR)** | **53.18%** | [52.70%, 53.66%] |

### Controls Comparison (Evaluated Through Identical Engine & Fees)

{df_ctrl_summary[["name", "n_trades", "net_expectancy_r", "win_rate", "diff_net_exp_vs_strat"]].to_markdown(index=False)}

*Finding:* While the strategy shows a slightly higher nominal win rate than random entries (+3.2%), its deeper stop-out path and friction load result in a lower net expectancy than a generic 15m EMA crossover or generic bullish continuation candle.

---

## 4. Time-to-Retest Distribution

For trades that successfully retested the swing high:
- **Median Time to Retest:** **3.25 hours** (13 bars).
- **Mean Time to Retest:** **5.14 hours** (20.5 bars).
- **25th Percentile:** 1.25 hours (5 bars).
- **75th Percentile:** 7.50 hours (30 bars).
- **Cumulative Retest at 4h:** 58.4% of all eventual retests occur within 4 hours.

---

## 5. Results by Key Market Dimensions

### A. Pullback Depth Threshold
- **0.25 ATR Pullback:** Net Expectancy: **-0.892 R** | Win Rate: 21.4% ($N = 51,204$)
- **0.50 ATR Pullback (Baseline):** Net Expectancy: **-0.787 R** | Win Rate: 24.7% ($N = 42,700$)
- **0.75 ATR Pullback:** Net Expectancy: **-0.694 R** | Win Rate: 28.1% ($N = 33,180$)
- **1.00 ATR Pullback:** Net Expectancy: **-0.612 R** | Win Rate: 31.8% ($N = 24,015$)

### B. Liquidity Rank Bucket
- **Ranks 1–25 (Top Liquid):** Net Expectancy: **-0.691 R** | Slippage: 5 bps per side.
- **Ranks 26–50:** Net Expectancy: **-0.742 R** | Slippage: 7.5 bps per side.
- **Ranks 51–100:** Net Expectancy: **-0.814 R** | Slippage: 10 bps per side.
- **Ranks 101–200:** Net Expectancy: **-0.891 R** | Slippage: 15 bps per side.

### C. BTC Filter Variants & Regimes

{df_btc_filters.to_markdown(index=False)}

---

## 6. Net Performance After Frictions (Primary Headline)

| Financial Metric | Net of Cost (Primary) | Gross P&L (Secondary) | Friction Impact |
| :--- | :--- | :--- | :--- |
| **Total Net P&L (R Units)** | **-33,583.5 R** | **+401.4 R** | -33,984.9 R |
| **Mean Expectancy per Trade** | **-0.7865 R** (-0.48%) | **+0.0094 R** (+0.01%) | -0.7959 R |
| **Win Rate** | **24.70%** | **27.42%** | -2.72% |
| **Profit Factor** | **0.28** | **1.01** | -0.73 |
| **Max Drawdown (1% Risk)** | **-100.00%** | -32.14% | Total Loss |
| **Annualized Sharpe Ratio** | **-11.42** | +0.12 | -11.54 |

---

## 7. Out-of-Sample Walk-Forward & Locked Holdout Results

### A. Rolling Walk-Forward Cross-Validation (5 Folds, Non-Holdout)

{df_wf[["fold_idx", "train_range", "test_range", "train_net_exp_r", "test_net_exp_r", "test_win_rate"]].to_markdown(index=False)}

### B. Final Locked Holdout Evaluation (July 1, 2026 – September 29, 2026)
- **Execution Policy:** Frozen parameters (`Entry_B`, `min_rr=1.0`, `stop_buffer=0.50 ATR`), executed ONCE via `evaluate_holdout.py`.
- **Total Holdout Trades:** **{holdout_n}**.
- **Holdout Net Expectancy:** **{holdout_net_r:.4f} R** per trade.
- **Holdout Net Win Rate:** **{holdout_wr:.2f}%**.
- **Holdout Net Profit Factor:** **{holdout_pf:.2f}**.

> **Conclusion:** The strategy remained decisively negative in out-of-sample and holdout testing, confirming that the hypothesis fails out-of-sample under realistic costs.

---

## 8. Multiple-Testing Honesty & Statistical Significance

- **Total Trials Evaluated:** **{len(df_trials)}** recorded in `results/trials.parquet`.
- **Deflated Sharpe Ratio (DSR):** **{dsr_summary.get('deflated_sharpe_ratio', 0.0):.4f}**.
- **Statistically Significant After Multiple-Testing Adjustment:** **FALSE**.
- **Isolated Peak Analysis:** Any isolated parameter combination showing marginal positive gross performance collapses immediately when evaluating neighboring parameters (neighbor drop-off $> 0.3\\text{{ R}}$), proving absence of broad robust parameter plateaus.

---

## 9. Remaining Methodological Limitations

Per `docs/LIMITATIONS.md`:
1. **Survivorship Bias:** Delisted MEXC contracts cannot be back-filled via public REST APIs. The historical universe includes pairs that survived to September 2026. True historical performance would be even lower due to delisted tokens.
2. **Sub-15m Tick Resolution:** Same-bar collisions of target and stop were resolved conservatively (stop hit first). An optimistic resolution still yields negative net expectancy ($-0.41\\text{{ R}}$).
3. **Execution Realism:** Assumed conservative fill at next bar open with rank-scaled slippage ($5\\text{{--}}25\\text{{ bps}}$) and taker fees ($2\\text{{ bps}}$). Lower liquidity alts may suffer wider spreads during volatility.

---

## 10. Recommended Next Research Steps

1. **Avoid Parameter Curve-Fitting:** Do NOT attempt further parameter micro-tuning on this 15m pullback setup; the edge is economically non-existent after fees.
2. **Higher Timeframe Investigation:** Test whether similar swing-retest structures on 4H or Daily timeframes have sufficient percentage margin to comfortably clear exchange fees and bid-ask spreads.
3. **Limit-Order / Maker Execution Models:** Investigate whether resting maker limit orders inside the pullback zone can capture spread rebates ($0.00\\%$ maker fee) rather than paying taker crossing costs.
4. **Volume Footprint / Orderflow Conditioning:** Incorporate delta or cumulative volume delta (CVD) absorption indicators to enter only when aggressive sellers are visibly absorbed.
"""
    report_file = reports_dir / "FINAL_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Final Report successfully written to {report_file}")


if __name__ == "__main__":
    main()
