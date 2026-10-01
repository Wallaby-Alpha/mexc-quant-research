from pathlib import Path
from typing import List, Optional
import hashlib
import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "mexc_swing_high_retest"
    version: str = "1.0.0"


class MarketConfig(BaseModel):
    venue: str = "futures"  # "futures" or "spot"
    quote_currency: str = "USDT"


class DataConfig(BaseModel):
    cache_dir: str = "data_cache"
    reports_dir: str = "reports"
    results_dir: str = "results"
    timeframes: List[str] = Field(default_factory=lambda: ["15m", "1h"])
    history_days: int = 360
    start_date: str = "2025-10-04T00:00:00Z"
    end_date: str = "2026-09-29T00:00:00Z"
    holdout_start: str = "2026-07-01T00:00:00Z"
    request_rate_limit: int = 10
    max_retries: int = 3
    backoff_seconds: float = 1.0
    timeout_seconds: float = 15.0


class UniverseConfig(BaseModel):
    target_size: int = 200
    sample_size_smoke: int = 20
    min_volume_usdt: float = 1_000_000.0
    min_history_1h_bars: int = 250
    always_include: List[str] = Field(default_factory=lambda: ["BTC_USDT", "ETH_USDT"])
    excluded_symbols: List[str] = Field(default_factory=list)


class ExecutionDefaultsConfig(BaseModel):
    maker_fee_rate: float = 0.0000
    taker_fee_rate: float = 0.0002
    base_slippage_bps: float = 5.0
    funding_rate_8h: float = 0.0001
    slippage_by_rank: dict = Field(
        default_factory=lambda: {
            "1-25": 5.0,
            "26-50": 7.5,
            "51-100": 10.0,
            "101-200": 15.0,
            "201-300": 20.0,
            "default": 25.0,
        }
    )


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    execution_defaults: ExecutionDefaultsConfig = Field(default_factory=ExecutionDefaultsConfig)
    config_hash: Optional[str] = None


def load_config(path: str | Path = "config/default.yaml") -> AppConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")

    content = p.read_bytes()
    raw = yaml.safe_load(content) or {}
    config_hash = hashlib.sha256(content).hexdigest()[:16]

    app_config = AppConfig(**raw)
    app_config.config_hash = config_hash
    return app_config
