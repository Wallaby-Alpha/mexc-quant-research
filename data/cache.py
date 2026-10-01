from pathlib import Path
from typing import List, Optional
import os
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0.0"

KLINE_SCHEMA = pa.schema([
    ("open_time", pa.timestamp("ms", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume_base", pa.float64()),
    ("quote_volume_usdt", pa.float64()),
    ("source_market", pa.string()),
    ("schema_version", pa.string())
])

FUNDING_SCHEMA = pa.schema([
    ("settle_time", pa.timestamp("ms", tz="UTC")),
    ("symbol", pa.string()),
    ("funding_rate", pa.float64()),
    ("collect_cycle_hours", pa.int32()),
    ("schema_version", pa.string())
])


class ParquetCache:
    """
    Schema-versioned, atomic Parquet cache for historical market data.
    Ensures safe concurrent/interrupted writes via tempfile + atomic rename.
    """

    def __init__(self, base_dir: str | Path = "data_cache"):
        self.base_dir = Path(base_dir)
        self.klines_dir = self.base_dir / "klines"
        self.funding_dir = self.base_dir / "funding"
        self.klines_dir.mkdir(parents=True, exist_ok=True)
        self.funding_dir.mkdir(parents=True, exist_ok=True)

    def _kline_file_path(self, symbol: str, timeframe: str) -> Path:
        tf_dir = self.klines_dir / timeframe
        tf_dir.mkdir(parents=True, exist_ok=True)
        # Normalize symbol name for file systems (replace / with _)
        safe_sym = symbol.replace("/", "_")
        return tf_dir / f"{safe_sym}.parquet"

    def _funding_file_path(self, symbol: str) -> Path:
        safe_sym = symbol.replace("/", "_")
        return self.funding_dir / f"{safe_sym}.parquet"

    def get_kline_time_bounds(self, symbol: str, timeframe: str) -> tuple[Optional[int], Optional[int]]:
        """
        Returns (earliest_open_time_ms, latest_open_time_ms) from cache without loading entire dataset.
        """
        file_path = self._kline_file_path(symbol, timeframe)
        if not file_path.is_file():
            return None, None

        try:
            # Read only open_time column for fast boundary lookup
            table = pq.read_table(file_path, columns=["open_time"])
            if table.num_rows == 0:
                return None, None
            col = table.column("open_time")
            # Convert pyarrow timestamp array to int ms
            t_min = col[0].as_py()
            t_max = col[-1].as_py()
            min_ms = int(t_min.timestamp() * 1000)
            max_ms = int(t_max.timestamp() * 1000)
            return min_ms, max_ms
        except Exception:
            return None, None

    def save_klines(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Atomically saves or updates the kline DataFrame to Parquet.
        Deduplicates by open_time, sorts monotonically ascending.
        """
        if df.empty:
            return

        file_path = self._kline_file_path(symbol, timeframe)

        # Merge with existing data if present
        if file_path.is_file():
            try:
                existing_df = pd.read_parquet(file_path)
                combined = pd.concat([existing_df, df], ignore_index=True)
            except Exception:
                combined = df
        else:
            combined = df

        # Ensure datetime UTC open_time
        if not pd.api.types.is_datetime64_any_dtype(combined["open_time"]):
            combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
        elif combined["open_time"].dt.tz is None:
            combined["open_time"] = combined["open_time"].dt.tz_localize("UTC")

        # Deduplicate and sort
        combined = combined.drop_duplicates(subset=["open_time"], keep="last")
        combined = combined.sort_values(by="open_time").reset_index(drop=True)
        combined["schema_version"] = SCHEMA_VERSION

        # Ensure correct column ordering and dtypes
        cols = [
            "open_time", "open", "high", "low", "close",
            "volume_base", "quote_volume_usdt", "source_market", "schema_version"
        ]
        for col in cols:
            if col not in combined.columns:
                raise ValueError(f"Missing required column '{col}' for cache write")
        out_df = combined[cols]

        table = pa.Table.from_pandas(out_df, schema=KLINE_SCHEMA, preserve_index=False)

        # Atomic write via temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.stem}_",
            suffix=".tmp"
        )
        os.close(temp_fd)

        try:
            pq.write_table(table, temp_path, compression="snappy")
            os.replace(temp_path, file_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Loads sorted, deduplicated klines from Parquet."""
        file_path = self._kline_file_path(symbol, timeframe)
        if not file_path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        """Atomically saves funding history records."""
        if df.empty:
            return

        file_path = self._funding_file_path(symbol)
        if file_path.is_file():
            try:
                existing_df = pd.read_parquet(file_path)
                combined = pd.concat([existing_df, df], ignore_index=True)
            except Exception:
                combined = df
        else:
            combined = df

        if not pd.api.types.is_datetime64_any_dtype(combined["settle_time"]):
            combined["settle_time"] = pd.to_datetime(combined["settle_time"], utc=True)
        elif combined["settle_time"].dt.tz is None:
            combined["settle_time"] = combined["settle_time"].dt.tz_localize("UTC")

        combined = combined.drop_duplicates(subset=["settle_time"], keep="last")
        combined = combined.sort_values(by="settle_time").reset_index(drop=True)
        combined["schema_version"] = SCHEMA_VERSION

        cols = ["settle_time", "symbol", "funding_rate", "collect_cycle_hours", "schema_version"]
        out_df = combined[cols]

        table = pa.Table.from_pandas(out_df, schema=FUNDING_SCHEMA, preserve_index=False)

        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.stem}_",
            suffix=".tmp"
        )
        os.close(temp_fd)

        try:
            pq.write_table(table, temp_path, compression="snappy")
            os.replace(temp_path, file_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load_funding(self, symbol: str) -> pd.DataFrame:
        file_path = self._funding_file_path(symbol)
        if not file_path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    def list_cached_symbols(self, timeframe: str) -> List[str]:
        tf_dir = self.klines_dir / timeframe
        if not tf_dir.is_dir():
            return []
        return [f.stem for f in tf_dir.glob("*.parquet")]

    def get_cache_stats(self) -> dict:
        """Returns total size, symbol count per timeframe, file count."""
        total_bytes = 0
        file_count = 0
        symbols_by_tf = {}

        for p in self.base_dir.rglob("*.parquet"):
            total_bytes += p.stat().st_size
            file_count += 1

        for tf_dir in self.klines_dir.iterdir():
            if tf_dir.is_dir():
                syms = [f.stem for f in tf_dir.glob("*.parquet")]
                symbols_by_tf[tf_dir.name] = len(syms)

        return {
            "total_bytes": total_bytes,
            "total_size_mb": total_bytes / (1024 * 1024),
            "file_count": file_count,
            "symbols_by_timeframe": symbols_by_tf
        }
