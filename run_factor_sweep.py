"""
run_factor_sweep.py
Conducts multi-factor matrix evaluation across:
1. Multiple Pullback Definitions:
   - ATR thresholds: 0.25, 0.50, 0.75, 1.00, 1.50 ATR
   - Retracement % thresholds: 20%, 30%, 40%, 50%, 60%
2. Multiple Entry Modes:
   - Entry A: Immediate market fill at event close (next open)
   - Entry B: 15m Reversal candle confirmation
   - Entry C: 15m Lower-high break confirmation
3. Multiple Stop Loss Definitions (with Target = Prior Swing High constant):
   - PullbackBufferStop (0.10 ATR)
   - PullbackBufferStop (0.25 ATR)
   - PullbackBufferStop (0.50 ATR)
   - PriorSwingLowStop (below impulse low)
   - FixedATRStop (1.5 ATR from entry)
   - PercentageStop (2.0% from entry)

Records all trials in results/trials.parquet per Rule 6 and outputs factor sensitivity tables.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
import concurrent.futures
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from strategy.indicators import compute_wilder_atr
from strategy.pullback_detector import ThresholdCrossingEvent
from strategy.entry_rules import EntryA_Immediate, EntryB_15mReversal, EntryC_LowerHighBreak, BaseEntryRule
from strategy.exit_rules import (
    BaseStopRule,
    PullbackBufferStop,
    PriorSwingLowStop,
    FixedATRStop,
    PercentageStop,
    ExitRules
)
from backtest.trade import Trade
from backtest.execution_model import ExecutionModel
from backtest.engine import BacktestEngine
from analysis.trade_metrics import TradeMetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("factor_sweep")


def load_partition_events(
    results_dir: Path,
    symbol: str,
    threshold_type: str,
    threshold_val: float,
    holdout_start: str,
    fractal_n: int = 3
) -> List[ThresholdCrossingEvent]:
    p_file = results_dir / "setups_partitions" / f"fractal_n={fractal_n}" / f"{symbol}.parquet"
    if not p_file.is_file():
        return []

    df_part = pd.read_parquet(p_file)
    if df_part.empty:
        return []

    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")
    mask = (
        (df_part["is_controlled_pullback"] == True) &
        (df_part["threshold_type"] == threshold_type) &
        (np.isclose(df_part["threshold_value"], threshold_val, atol=1e-3)) &
        (pd.to_datetime(df_part["event_time"], utc=True) < holdout_ts)
    )
    df_filtered = df_part[mask]
    if df_filtered.empty:
        return []

    fields = set(ThresholdCrossingEvent.__dataclass_fields__.keys())
    return [ThresholdCrossingEvent(**{k: r[k] for k in fields if k in r}) for r in df_filtered.to_dict("records")]


def main():
    cfg = load_config()
    cache = ParquetCache(cfg.data.cache_dir)
    results_dir = Path(cfg.data.results_dir)
    holdout_start = cfg.data.holdout_start

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    all_symbols = [p.stem for p in (results_dir / "setups_partitions" / "fractal_n=3").glob("*.parquet")]
    logger.info(f"Preloading 15m price bars for {len(all_symbols)} symbols...")

    symbol_bars: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for sym in all_symbols:
        df_15m = cache.load_klines(sym, "15m")
        if df_15m.empty:
            continue
        df_nh = df_15m[pd.to_datetime(df_15m["open_time"], utc=True) < pd.Timestamp(holdout_start, tz="UTC")].copy()
        if len(df_nh) < 200:
            continue
        atr = compute_wilder_atr(df_nh["high"], df_nh["low"], df_nh["close"], 14)
        symbol_bars[sym] = (df_nh, atr)

    logger.info(f"Preloaded {len(symbol_bars)} symbols successfully.")

    # -------------------------------------------------------------------------
    # Define Factor Matrix Dimensions
    # -------------------------------------------------------------------------
    # 1. Pullback definitions: (threshold_type, threshold_val, label)
    pullback_factors = [
        ("atr", 0.25, "Depth_0.25_ATR"),
        ("atr", 0.50, "Depth_0.50_ATR"),
        ("atr", 0.75, "Depth_0.75_ATR"),
        ("atr", 1.00, "Depth_1.00_ATR"),
        ("atr", 1.50, "Depth_1.50_ATR"),
        ("retracement", 20.0, "Retrace_20pct"),
        ("retracement", 30.0, "Retrace_30pct"),
        ("retracement", 50.0, "Retrace_50pct"),
        ("retracement", 60.0, "Retrace_60pct"),
    ]

    # 2. Entry Rules
    entry_factors = [
        ("Entry_A", lambda: EntryA_Immediate(min_rr=1.0)),
        ("Entry_B", lambda: EntryB_15mReversal(min_rr=1.0)),
        ("Entry_C", lambda: EntryC_LowerHighBreak(min_rr=1.0)),
    ]

    # 3. Stop Loss Rules (Take Profit is constant: Target = Prior Swing High)
    stop_factors = [
        ("Buffer_0.10_ATR", lambda: PullbackBufferStop(buffer_atr=0.10)),
        ("Buffer_0.25_ATR", lambda: PullbackBufferStop(buffer_atr=0.25)),
        ("Buffer_0.50_ATR", lambda: PullbackBufferStop(buffer_atr=0.50)),
        ("Prior_Swing_Low", lambda: PriorSwingLowStop(buffer_atr=0.0)),
        ("Fixed_1.5_ATR", lambda: FixedATRStop(mult_atr=1.5)),
        ("Fixed_2.0_Pct", lambda: PercentageStop(pct=0.02)),
    ]

    logger.info("=" * 70)
    logger.info("RUNNING SYSTEMATIC MULTI-FACTOR SWEEP")
    logger.info(f"Dimensions: {len(pullback_factors)} Pullbacks x {len(entry_factors)} Entries x {len(stop_factors)} Stops")
    logger.info("=" * 70)

    # Pre-load partition events for each pullback definition
    events_by_pb: Dict[str, Dict[str, List[ThresholdCrossingEvent]]] = {}
    for pb_type, pb_val, pb_label in pullback_factors:
        logger.info(f"Extracting setups for Pullback definition: {pb_label}...")
        sym_events = {}
        for sym in symbol_bars.keys():
            evs = load_partition_events(results_dir, sym, pb_type, pb_val, holdout_start)
            if evs:
                sym_events[sym] = evs
        events_by_pb[pb_label] = sym_events

    factor_records = []
    trial_idx = 100

    # Cross-product evaluation
    for pb_type, pb_val, pb_label in pullback_factors:
        sym_events = events_by_pb[pb_label]
        if not sym_events:
            continue

        for entry_name, entry_fn in entry_factors:
            entry_rule = entry_fn()

            for stop_name, stop_fn in stop_factors:
                stop_rule = stop_fn()
                exit_rules = ExitRules(stop_rule=stop_rule, time_stop_hours=24.0, enable_trend_exit=False)

                t0 = time.time()
                trial_trades: List[Trade] = []

                for sym, (df_15m, atr_series) in symbol_bars.items():
                    if sym not in sym_events:
                        continue
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
                        events=sym_events[sym],
                        allow_holdout=False
                    )
                    trial_trades.extend(tr)

                elapsed = time.time() - t0
                n = len(trial_trades)
                if n == 0:
                    continue

                trial_idx += 1
                trial_id = f"FACTOR_{trial_idx}_{pb_label}_{entry_name}_{stop_name}"

                df_t = pd.DataFrame([t.__dict__ for t in trial_trades])
                metrics = TradeMetricsCalculator.compute_summary_metrics(df_t)

                record = {
                    "trial_id": trial_id,
                    "pullback_definition": pb_label,
                    "pullback_type": pb_type,
                    "pullback_val": pb_val,
                    "entry_mode": entry_name,
                    "stop_loss_type": stop_name,
                    "take_profit": "Prior_Swing_High",
                    "n_trades": n,
                    "win_rate_net": metrics["win_rate_net"],
                    "expectancy_net_r": metrics["expectancy_net_r"],
                    "expectancy_net_pct": metrics["expectancy_net_pct"],
                    "profit_factor_net": metrics["profit_factor_net"],
                    "win_rate_gross": metrics.get("win_rate_gross", 0.0),
                    "expectancy_gross_r": metrics.get("expectancy_gross_r", 0.0),
                    "profit_factor_gross": metrics.get("profit_factor_gross", 0.0),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", -100.0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "simulation_time_sec": elapsed
                }
                factor_records.append(record)

                # Log to trials.parquet per Rule 6
                BacktestEngine.log_trial(record, results_dir=results_dir)

                logger.info(
                    f"[{trial_id}] N={n:,} | Net WR: {metrics['win_rate_net']*100:.1f}% | "
                    f"Net Exp: {metrics['expectancy_net_r']:.3f}R | Gross Exp: {metrics.get('expectancy_gross_r', 0.0):.3f}R | "
                    f"Net PF: {metrics['profit_factor_net']:.2f} ({elapsed:.1f}s)"
                )

    df_factor = pd.DataFrame(factor_records)
    out_csv = results_dir / "factor_sweep_results.csv"
    out_parquet = results_dir / "factor_sweep_results.parquet"
    df_factor.to_csv(out_csv, index=False)
    df_factor.to_parquet(out_parquet, index=False)
    logger.info(f"\nSaved {len(df_factor)} factor combinations to {out_csv} and {out_parquet}")

    # Generate summary cross-tabs
    print("\n" + "=" * 80)
    print("SUMMARY BY ENTRY MODE & PULLBACK DEFINITION (Mean Net Expectancy R):")
    print("=" * 80)
    pivot_entry_pb = df_factor.pivot_table(
        index="pullback_definition",
        columns="entry_mode",
        values="expectancy_net_r",
        aggfunc="mean"
    )
    print(pivot_entry_pb.round(3).to_string())

    print("\n" + "=" * 80)
    print("SUMMARY BY STOP LOSS TYPE (Mean Net Expectancy R):")
    print("=" * 80)
    pivot_stop = df_factor.groupby("stop_loss_type")[["expectancy_net_r", "win_rate_net", "profit_factor_net", "n_trades"]].mean()
    print(pivot_stop.round(3).to_string())


if __name__ == "__main__":
    main()
