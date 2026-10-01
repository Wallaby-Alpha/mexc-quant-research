"""
evaluate_holdout.py
Final Holdout Evaluation (Phase 5).
Runs ONCE with frozen parameters selected from walk-forward cross-validation.
Evaluates strictly on holdout data (open_time >= 2026-07-01).
Logs execution to results/holdout_log.json and exports results/holdout_trades.parquet.
Adheres strictly to Rule 5 (Holdout Discipline).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import hashlib
import time
import logging
import concurrent.futures
import pandas as pd
import numpy as np

from data.config import load_config
from data.cache import ParquetCache
from strategy.indicators import compute_wilder_atr
from strategy.pullback_detector import ThresholdCrossingEvent
from strategy.entry_rules import EntryA_Immediate, EntryB_15mReversal, EntryC_LowerHighBreak
from strategy.exit_rules import ExitRules, PullbackBufferStop
from backtest.trade import Trade
from backtest.execution_model import ExecutionModel
from backtest.engine import BacktestEngine
from analysis.trade_metrics import TradeMetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("holdout_eval")


def run_holdout_evaluation(
    frozen_config: Dict[str, Any],
    cache_dir: str = "data_cache",
    results_dir: str = "results",
    holdout_start: str = "2026-07-01T00:00:00Z",
    force_rerun: bool = False
) -> Dict[str, Any]:
    log_path = Path(results_dir) / "holdout_log.json"
    if log_path.is_file() and not force_rerun:
        logger.warning(f"Holdout has ALREADY been evaluated per Rule 5! Reading existing log: {log_path}")
        with open(log_path, "r") as f:
            return json.load(f)

    logger.info("=" * 70)
    logger.info("FINAL HOLDOUT EVALUATION — RUNNING ONCE WITH FROZEN PARAMETERS")
    logger.info(f"Frozen Parameters: {frozen_config}")
    logger.info(f"Holdout Horizon: >= {holdout_start}")
    logger.info("=" * 70)

    cfg = load_config()
    cache = ParquetCache(cache_dir)
    holdout_ts = pd.Timestamp(holdout_start, tz="UTC")

    # Map entry rule
    entry_mode_str = frozen_config.get("entry_mode", "Entry_A")
    min_rr = float(frozen_config.get("min_rr", 1.0))
    if entry_mode_str == "Entry_B":
        entry_rule = EntryB_15mReversal(min_rr=min_rr)
    elif entry_mode_str == "Entry_C":
        entry_rule = EntryC_LowerHighBreak(min_rr=min_rr)
    else:
        entry_rule = EntryA_Immediate(min_rr=min_rr)

    # Map exit rules
    stop_buffer = float(frozen_config.get("stop_buffer_atr", 0.25))
    max_hold_hours = int(frozen_config.get("max_hold_hours", 24))
    stop_rule = PullbackBufferStop(buffer_atr=stop_buffer)
    exit_rules = ExitRules(stop_rule=stop_rule, time_stop_hours=max_hold_hours, enable_trend_exit=False)

    exec_model = ExecutionModel(
        maker_fee_rate=cfg.execution_defaults.maker_fee_rate,
        taker_fee_rate=cfg.execution_defaults.taker_fee_rate,
        base_slippage_bps=cfg.execution_defaults.base_slippage_bps,
        funding_rate_8h=cfg.execution_defaults.funding_rate_8h,
        slippage_by_rank=cfg.execution_defaults.slippage_by_rank
    )

    fractal_n = frozen_config.get("fractal_n", 3)
    pullback_depth = float(frozen_config.get("pullback_depth_atr", 0.50))
    part_dir = Path(results_dir) / "setups_partitions" / f"fractal_n={fractal_n}"

    parquet_files = list(part_dir.glob("*.parquet"))
    logger.info(f"Loading holdout setups across {len(parquet_files)} symbols...")

    all_holdout_trades: List[Trade] = []

    def _sim_symbol(p_file: Path) -> List[Trade]:
        sym = p_file.stem
        df_15m = cache.load_klines(sym, "15m")
        if df_15m.empty:
            return []

        df_part = pd.read_parquet(p_file)
        if df_part.empty:
            return []

        # STRICT HOLDOUT FILTER: >= holdout_start
        mask = (
            (df_part["is_controlled_pullback"] == True) &
            (df_part["threshold_type"] == "atr") &
            (df_part["threshold_value"] == pullback_depth) &
            (pd.to_datetime(df_part["event_time"], utc=True) >= holdout_ts)
        )
        df_sub = df_part[mask]
        if df_sub.empty:
            return []

        # Filter 15m candle bars to include warm-up but focus on holdout
        df_15m_holdout = df_15m[pd.to_datetime(df_15m["open_time"], utc=True) >= (holdout_ts - pd.Timedelta(days=7))].copy()
        if df_15m_holdout.empty:
            return []

        fields = set(ThresholdCrossingEvent.__dataclass_fields__.keys())
        records = df_sub.to_dict("records")
        events = [ThresholdCrossingEvent(**{k: r[k] for k in fields if k in r}) for r in records]

        atr_series = compute_wilder_atr(df_15m_holdout["high"], df_15m_holdout["low"], df_15m_holdout["close"], 14)

        engine = BacktestEngine(
            execution_model=exec_model,
            entry_rule=entry_rule,
            exit_rules=exit_rules,
            holdout_start=holdout_start
        )
        trades = engine.simulate_symbol(
            symbol=sym,
            df_15m=df_15m_holdout,
            atr_series=atr_series,
            events=events,
            allow_holdout=True  # Strictly allowed here
        )
        # Filter trades that entered within holdout window
        return [t for t in trades if pd.to_datetime(t.entry_time, utc=True) >= holdout_ts]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_sim_symbol, p) for p in parquet_files]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                all_holdout_trades.extend(res)

    all_holdout_trades.sort(key=lambda t: t.entry_time)
    logger.info(f"Holdout simulation completed: {len(all_holdout_trades)} trades executed.")

    df_trades = pd.DataFrame([t.__dict__ for t in all_holdout_trades])
    if not df_trades.empty:
        df_trades.to_parquet(Path(results_dir) / "holdout_trades.parquet", index=False)
        df_trades.to_csv(Path(results_dir) / "holdout_trades.csv", index=False)
        metrics = TradeMetricsCalculator.compute_summary_metrics(df_trades)
    else:
        metrics = {}

    log_data = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_start": holdout_start,
        "frozen_parameters": frozen_config,
        "total_holdout_trades": len(all_holdout_trades),
        "metrics": metrics
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(f"Holdout log successfully written to {log_path}")
    return log_data


if __name__ == "__main__":
    # Default frozen configuration based on walk-forward selection
    frozen = {
        "entry_mode": "Entry_B",
        "min_rr": 1.0,
        "stop_buffer_atr": 0.50,
        "max_hold_hours": 24,
        "fractal_n": 3,
        "pullback_depth_atr": 0.50
    }
    res = run_holdout_evaluation(frozen)
    print("Holdout Metrics Summary:")
    print(json.dumps(res["metrics"], indent=2))

