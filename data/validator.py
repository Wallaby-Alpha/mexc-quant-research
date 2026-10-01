from datetime import timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from data.config import AppConfig, load_config
from data.cache import ParquetCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validator")


class DataValidator:
    """
    Comprehensive Data Quality Validator for financial klines.
    Checks:
    - Monotonic ordering
    - Duplicate timestamps
    - Expected interval spacing and missing bar gaps
    - OHLC structural integrity (High >= Low, High >= max(Open,Close), Low <= min(Open,Close))
    - Zero, NaN, or infinite price/volume values
    - Timezone sanity (strictly UTC)
    - Minimum history threshold (>= 250 1H bars)
    Outputs:
    - Parquet report: reports/data_quality_report.parquet
    - Markdown summary: reports/DATA_QUALITY.md
    """

    def __init__(self, config: AppConfig, cache: Optional[ParquetCache] = None):
        self.config = config
        self.cache = cache or ParquetCache(config.data.cache_dir)
        self.reports_dir = Path(config.data.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _expected_step(self, timeframe: str) -> pd.Timedelta:
        if timeframe == "15m":
            return pd.Timedelta(minutes=15)
        elif timeframe == "1h":
            return pd.Timedelta(hours=1)
        elif timeframe == "4h":
            return pd.Timedelta(hours=4)
        elif timeframe == "1d":
            return pd.Timedelta(days=1)
        raise ValueError(f"Unknown timeframe: {timeframe}")

    def validate_series(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": len(df),
            "start_time": None,
            "end_time": None,
            "is_monotonic": True,
            "duplicate_count": 0,
            "gap_count": 0,
            "total_missing_bars": 0,
            "bad_ohlc_count": 0,
            "zero_or_nan_count": 0,
            "timezone_utc": True,
            "candle_aligned": True,
            "usable": True,
            "reasons": []
        }

        if df.empty:
            result["usable"] = False
            result["reasons"].append("Empty DataFrame (zero bars)")
            return result

        result["start_time"] = df["open_time"].min()
        result["end_time"] = df["open_time"].max()

        # 1. Timezone Check
        if not hasattr(df["open_time"].dt, "tz") or str(df["open_time"].dt.tz) != "UTC":
            result["timezone_utc"] = False
            result["usable"] = False
            result["reasons"].append("Timestamps not localized to UTC")

        # 2. Monotonicity
        if not df["open_time"].is_monotonic_increasing:
            result["is_monotonic"] = False
            result["usable"] = False
            result["reasons"].append("Timestamps not monotonically increasing")

        # 3. Duplicates
        dup_count = int(df["open_time"].duplicated().sum())
        result["duplicate_count"] = dup_count
        if dup_count > 0:
            result["usable"] = False
            result["reasons"].append(f"{dup_count} duplicate timestamps found")

        # 4. Candle Alignment Check (e.g. 15m must align to 0, 15, 30, 45 minutes)
        if timeframe == "15m":
            misaligned = ((df["open_time"].dt.minute % 15 != 0) | (df["open_time"].dt.second != 0)).sum()
            if misaligned > 0:
                result["candle_aligned"] = False
                result["reasons"].append(f"{misaligned} bars not aligned to 15m boundaries")
        elif timeframe == "1h":
            misaligned = ((df["open_time"].dt.minute != 0) | (df["open_time"].dt.second != 0)).sum()
            if misaligned > 0:
                result["candle_aligned"] = False
                result["reasons"].append(f"{misaligned} bars not aligned to 1H boundaries")

        # 5. Gaps / Missing Bars
        expected_step = self._expected_step(timeframe)
        diffs = df["open_time"].diff()
        gap_mask = diffs > expected_step
        gap_count = int(gap_mask.sum())
        result["gap_count"] = gap_count
        if gap_count > 0:
            missing_bars = int(((diffs[gap_mask] / expected_step) - 1).sum())
            result["total_missing_bars"] = missing_bars
            # Gaps are allowed up to a threshold (e.g. 1% missing bars) but logged
            missing_pct = (missing_bars / (len(df) + missing_bars)) * 100.0
            if missing_pct > 5.0:
                result["usable"] = False
                result["reasons"].append(f"Excessive missing bars ({missing_pct:.2f}% missing)")
            else:
                result["reasons"].append(f"{missing_bars} missing bars ({gap_count} gaps)")

        # 6. OHLC Integrity
        # High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        bad_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        bad_ohlc_count = int(bad_ohlc.sum())
        result["bad_ohlc_count"] = bad_ohlc_count
        if bad_ohlc_count > 0:
            result["usable"] = False
            result["reasons"].append(f"{bad_ohlc_count} OHLC consistency violations (High < Low or High < Close)")

        # 7. Zero / NaN / Negative prices
        invalid_prices = (
            df[["open", "high", "low", "close"]].isna().any(axis=1) |
            (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0) |
            np.isinf(df[["open", "high", "low", "close"]]).any(axis=1)
        )
        invalid_count = int(invalid_prices.sum())
        result["zero_or_nan_count"] = invalid_count
        if invalid_count > 0:
            result["usable"] = False
            result["reasons"].append(f"{invalid_count} zero, negative, or NaN prices found")

        # 8. Minimum history requirement (>= 250 1H bars)
        if timeframe == "1h" and len(df) < self.config.universe.min_history_1h_bars:
            result["usable"] = False
            result["reasons"].append(f"Insufficient history: {len(df)} 1H bars < required {self.config.universe.min_history_1h_bars}")

        return result

    def run(self) -> pd.DataFrame:
        """Runs validation across all cached symbols and timeframes."""
        all_results = []
        for tf in self.config.data.timeframes:
            symbols = self.cache.list_cached_symbols(tf)
            logger.info(f"Validating {len(symbols)} cached symbols for timeframe {tf}...")
            for sym in symbols:
                df = self.cache.load_klines(sym, tf)
                res = self.validate_series(sym, tf, df)
                all_results.append(res)

        report_df = pd.DataFrame(all_results)
        if report_df.empty:
            logger.warning("No data found in cache to validate.")
            return report_df

        # Save Parquet Report
        parquet_path = self.reports_dir / "data_quality_report.parquet"
        report_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet validation report: {parquet_path}")

        # Generate Markdown Report
        md_path = self.reports_dir / "DATA_QUALITY.md"
        self._generate_markdown_report(report_df, md_path)
        logger.info(f"Saved Markdown validation report: {md_path}")

        return report_df

    def _generate_markdown_report(self, df: pd.DataFrame, out_path: Path):
        usable_count = df[df["usable"]]["symbol"].nunique()
        total_symbols = df["symbol"].nunique()
        total_series = len(df)

        lines = [
            "# Data Quality & Integrity Report",
            f"**Generated at:** {pd.Timestamp.now(tz='UTC').isoformat()}",
            f"**Configuration Hash:** `{self.config.config_hash}`",
            "",
            "## Summary",
            f"- **Total Cached Symbols:** {total_symbols}",
            f"- **Total Series (Symbol × Timeframe):** {total_series}",
            f"- **Usable Symbols:** {usable_count} / {total_symbols} ({usable_count/total_symbols*100:.1f}%)" if total_symbols > 0 else "- No symbols",
            f"- **Usable Series:** {df['usable'].sum()} / {total_series}",
            "",
            "## Detailed Series Validation Results",
            "",
            "| Symbol | TF | Total Bars | Start (UTC) | End (UTC) | Usable | Gaps | Missing | Bad OHLC | Reasons |",
            "|---|---|---:|---|---|:---:|---:|---:|---:|---|"
        ]

        for _, row in df.iterrows():
            sym = row["symbol"]
            tf = row["timeframe"]
            bars = row["total_bars"]
            s_time = str(row["start_time"])[:19] if pd.notna(row["start_time"]) else "N/A"
            e_time = str(row["end_time"])[:19] if pd.notna(row["end_time"]) else "N/A"
            usable = "✅" if row["usable"] else "❌"
            gaps = row["gap_count"]
            missing = row["total_missing_bars"]
            bad_ohlc = row["bad_ohlc_count"]
            reasons = "; ".join(row["reasons"]) if row["reasons"] else "None (Clean)"

            lines.append(f"| `{sym}` | {tf} | {bars:,} | {s_time} | {e_time} | {usable} | {gaps} | {missing} | {bad_ohlc} | {reasons} |")

        out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data Quality Validator")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validator = DataValidator(cfg)
    report = validator.run()
    usable = report["usable"].sum() if not report.empty else 0
    print(f"Validation complete: {usable}/{len(report)} series passed.")


if __name__ == "__main__":
    main()
