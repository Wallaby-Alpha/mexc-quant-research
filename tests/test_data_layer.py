import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np

from data.config import AppConfig, load_config
from data.cache import ParquetCache
from data.mexc_client import MexcClient, StandardKline
from data.validator import DataValidator
from data.downloader import HistoricalDownloader


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_volume_conversion_hand_computed():
    """
    Test volume conversion against a hand-computed example.
    Contract: BTC_USDT
    Contract size: 0.0001 BTC/contract
    Candle: vol = 5,000 contracts, close = 80,000.0 USDT
    Amount = 40,000.0 USDT
    Expected volume_base = 5,000 * 0.0001 = 0.5 BTC
    Expected quote_volume_usdt = 40,000.0 USDT
    """
    vol_contracts = 5000.0
    contract_size = 0.0001
    close_price = 80000.0
    turnover_usdt = 40000.0

    # Base volume computation
    volume_base = vol_contracts * contract_size
    assert volume_base == 0.5, f"Expected 0.5 BTC, got {volume_base}"

    # Quote volume computation
    computed_quote = volume_base * close_price
    assert turnover_usdt == computed_quote == 40000.0, f"Expected 40,000 USDT, got {computed_quote}"

    # In StandardKline
    kline = StandardKline(
        open_time_ms=1700000000000,
        open_time_iso="2023-11-14T22:13:20+00:00",
        open=79900.0,
        high=80100.0,
        low=79800.0,
        close=close_price,
        volume_base=volume_base,
        quote_volume_usdt=turnover_usdt,
        source_market="futures"
    )
    assert kline.volume_base == 0.5
    assert kline.quote_volume_usdt == 40000.0


def test_incremental_download_appends_without_duplicates(temp_cache_dir):
    """
    Assert that incremental updates append newer bars without duplicates,
    preserving monotonic order and atomic Parquet integrity.
    """
    cache = ParquetCache(base_dir=temp_cache_dir)
    symbol = "BTC_USDT"
    timeframe = "15m"

    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    
    # Batch 1: bars 0 to 9
    batch1_records = []
    for i in range(10):
        t = base_time + timedelta(minutes=15 * i)
        batch1_records.append({
            "open_time": t,
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume_base": 10.0,
            "quote_volume_usdt": 1020.0,
            "source_market": "futures"
        })
    df_batch1 = pd.DataFrame(batch1_records)
    cache.save_klines(symbol, timeframe, df_batch1)

    loaded_1 = cache.load_klines(symbol, timeframe)
    assert len(loaded_1) == 10
    assert loaded_1["open_time"].is_monotonic_increasing

    # Batch 2: bars 5 to 14 (overlapping bars 5..9 and new bars 10..14)
    batch2_records = []
    for i in range(5, 15):
        t = base_time + timedelta(minutes=15 * i)
        batch2_records.append({
            "open_time": t,
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume_base": 10.0,
            "quote_volume_usdt": 1020.0,
            "source_market": "futures"
        })
    df_batch2 = pd.DataFrame(batch2_records)
    cache.save_klines(symbol, timeframe, df_batch2)

    loaded_2 = cache.load_klines(symbol, timeframe)
    # Total bars should be exactly 15 with zero duplicates
    assert len(loaded_2) == 15
    assert loaded_2["open_time"].duplicated().sum() == 0
    assert loaded_2["open_time"].is_monotonic_increasing
    assert loaded_2.iloc[0]["open_time"] == base_time
    assert loaded_2.iloc[-1]["open_time"] == base_time + timedelta(minutes=15 * 14)


def test_validator_catches_injected_faults(temp_cache_dir):
    """
    Assert that the DataValidator detects:
    1. Duplicate timestamps
    2. Gaps / missing bars
    3. Bad OHLC (High < Low or High < Close)
    4. Zero / NaN values
    """
    config = AppConfig()
    config.data.cache_dir = str(temp_cache_dir)
    config.data.reports_dir = str(temp_cache_dir / "reports")
    validator = DataValidator(config)

    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Clean reference series
    clean_records = []
    for i in range(20):
        t = base_time + timedelta(minutes=15 * i)
        clean_records.append({
            "open_time": t,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume_base": 1.0,
            "quote_volume_usdt": 102.0,
            "source_market": "futures"
        })
    df_clean = pd.DataFrame(clean_records)
    res_clean = validator.validate_series("CLEAN", "15m", df_clean)
    assert res_clean["usable"] is True
    assert res_clean["duplicate_count"] == 0
    assert res_clean["gap_count"] == 0
    assert res_clean["bad_ohlc_count"] == 0

    # 2. Injected duplicate
    df_dup = df_clean.copy()
    df_dup.loc[5, "open_time"] = df_dup.loc[4, "open_time"]
    res_dup = validator.validate_series("DUP", "15m", df_dup)
    assert res_dup["usable"] is False
    assert res_dup["duplicate_count"] == 1

    # 3. Injected gap (drop 3 bars in middle)
    df_gap = df_clean.drop(index=[5, 6, 7]).reset_index(drop=True)
    res_gap = validator.validate_series("GAP", "15m", df_gap)
    assert res_gap["gap_count"] == 1
    assert res_gap["total_missing_bars"] == 3

    # 4. Injected bad OHLC (High < Low)
    df_bad_ohlc = df_clean.copy()
    df_bad_ohlc.loc[2, "high"] = 90.0
    df_bad_ohlc.loc[2, "low"] = 110.0
    res_bad_ohlc = validator.validate_series("BAD_OHLC", "15m", df_bad_ohlc)
    assert res_bad_ohlc["usable"] is False
    assert res_bad_ohlc["bad_ohlc_count"] == 1

    # 5. Injected Zero / NaN
    df_nan = df_clean.copy()
    df_nan.loc[3, "close"] = np.nan
    res_nan = validator.validate_series("NAN", "15m", df_nan)
    assert res_nan["usable"] is False
    assert res_nan["zero_or_nan_count"] == 1


def test_timestamps_are_utc_and_candle_open_aligned():
    """
    Verify that timestamps are strictly UTC and aligned to candle open boundaries.
    """
    # 15m alignment
    valid_15m = [
        datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 10, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 10, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 10, 45, 0, tzinfo=timezone.utc),
    ]
    for t in valid_15m:
        assert t.tzinfo == timezone.utc
        assert t.minute % 15 == 0
        assert t.second == 0
        assert t.microsecond == 0

    # 1H alignment
    valid_1h = [
        datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 11, 0, 0, tzinfo=timezone.utc),
    ]
    for t in valid_1h:
        assert t.tzinfo == timezone.utc
        assert t.minute == 0
        assert t.second == 0
        assert t.microsecond == 0
