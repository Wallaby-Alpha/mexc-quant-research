from dataclasses import dataclass
from datetime import datetime, timezone
import time
import random
import logging
from typing import List, Dict, Optional, Any
import httpx

from data.config import AppConfig

logger = logging.getLogger("mexc_client")


@dataclass(frozen=True)
class StandardKline:
    open_time_ms: int       # Milliseconds UTC
    open_time_iso: str      # ISO8601 UTC string
    open: float
    high: float
    low: float
    close: float
    volume_base: float      # In base currency units
    quote_volume_usdt: float# In USDT
    source_market: str      # "futures" or "spot"


@dataclass(frozen=True)
class FundingRecord:
    symbol: str
    funding_rate: float
    settle_time_ms: int
    settle_time_iso: str
    collect_cycle_hours: int


@dataclass(frozen=True)
class TickerSummary:
    symbol: str
    volume_24h_usdt: float
    last_price: float
    contract_size: float = 1.0


class RateLimiter:
    """Thread-safe / deterministic token bucket rate limiter."""
    def __init__(self, max_per_second: float):
        self.capacity = max_per_second
        self.tokens = max_per_second
        self.fill_rate = max_per_second
        self.last_update = time.time()

    def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now

        if self.tokens < 1.0:
            sleep_time = (1.0 - self.tokens) / self.fill_rate
            time.sleep(sleep_time)
            self.tokens = 0.0
            self.last_update = time.time()
        else:
            self.tokens -= 1.0


class MexcClient:
    """
    Thin, robust client for MEXC REST API (Spot and USDT-M Perpetual Futures).
    Features:
    - Point-in-time adherence
    - Automatic retries with exponential backoff & jitter
    - Rate-limiting (default 10 req/s)
    - Normalized, typed schema output
    """

    SPOT_BASE_URL = "https://api.mexc.com/api/v3"
    FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"

    INTERVAL_MAP_FUTURES = {
        "15m": "Min15",
        "1h": "Min60",
        "4h": "Hour4",
        "1d": "Day1",
    }

    INTERVAL_MAP_SPOT = {
        "15m": "15m",
        "1h": "60m",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.venue = config.market.venue.lower()
        self.rate_limiter = RateLimiter(max_per_second=config.data.request_rate_limit)
        self.client = httpx.Client(
            timeout=config.data.timeout_seconds,
            headers={"User-Agent": "MEXC-Research-Client/1.0"}
        )
        self._contract_details_cache: Optional[Dict[str, float]] = None

    def _request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        max_retries = self.config.data.max_retries
        backoff = self.config.data.backoff_seconds

        for attempt in range(max_retries + 1):
            self.rate_limiter.acquire()
            try:
                resp = self.client.request(method, url, params=params)
                if resp.status_code == 429:
                    wait = backoff * (2 ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(f"Rate limited (429) on {url}. Retrying in {wait:.2f}s...")
                    time.sleep(wait)
                    continue
                elif resp.status_code >= 500:
                    wait = backoff * (2 ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(f"Server error ({resp.status_code}) on {url}. Retrying in {wait:.2f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == max_retries:
                    logger.error(f"Request failed permanently after {max_retries} retries: {url} {params} -> {e}")
                    raise e
                wait = backoff * (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(f"Network error: {e}. Retrying {url} in {wait:.2f}s...")
                time.sleep(wait)

    def get_contract_sizes(self) -> Dict[str, float]:
        """Fetch and cache futures contract sizes (base units per contract)."""
        if self._contract_details_cache is not None:
            return self._contract_details_cache

        url = f"{self.FUTURES_BASE_URL}/detail"
        data = self._request("GET", url)
        mapping: Dict[str, float] = {}
        for item in data.get("data", []):
            sym = item.get("symbol")
            cs = float(item.get("contractSize", 1.0) or 1.0)
            if sym:
                mapping[sym] = cs

        self._contract_details_cache = mapping
        return mapping

    def get_active_tickers(self) -> List[TickerSummary]:
        """Fetch all active USDT pairs sorted by 24h quote volume descending."""
        if self.venue == "futures":
            contract_sizes = self.get_contract_sizes()
            url = f"{self.FUTURES_BASE_URL}/ticker"
            data = self._request("GET", url)
            tickers = []
            for item in data.get("data", []):
                sym = item.get("symbol", "")
                if not sym.endswith("_USDT"):
                    continue
                vol_usdt = float(item.get("amount24", 0.0) or 0.0)
                last_px = float(item.get("lastPrice", 0.0) or 0.0)
                cs = contract_sizes.get(sym, 1.0)
                tickers.append(TickerSummary(
                    symbol=sym,
                    volume_24h_usdt=vol_usdt,
                    last_price=last_px,
                    contract_size=cs
                ))
            tickers.sort(key=lambda x: x.volume_24h_usdt, reverse=True)
            return tickers
        else:
            url = f"{self.SPOT_BASE_URL}/ticker/24hr"
            data = self._request("GET", url)
            tickers = []
            for item in data:
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                vol_usdt = float(item.get("quoteVolume", 0.0) or 0.0)
                last_px = float(item.get("lastPrice", 0.0) or 0.0)
                tickers.append(TickerSummary(
                    symbol=sym,
                    volume_24h_usdt=vol_usdt,
                    last_price=last_px,
                    contract_size=1.0
                ))
            tickers.sort(key=lambda x: x.volume_24h_usdt, reverse=True)
            return tickers

    def get_klines_batch(
        self,
        symbol: str,
        timeframe: str,
        start_time_sec: Optional[int] = None,
        end_time_sec: Optional[int] = None,
        limit: int = 2000
    ) -> List[StandardKline]:
        """
        Fetch a batch of klines for a given symbol and interval.
        Normalizes outputs from Spot or Futures into StandardKline.
        """
        if self.venue == "futures":
            iv = self.INTERVAL_MAP_FUTURES.get(timeframe)
            if not iv:
                raise ValueError(f"Unsupported futures timeframe: {timeframe}")

            url = f"{self.FUTURES_BASE_URL}/kline/{symbol}"
            params: Dict[str, Any] = {"interval": iv}
            if start_time_sec is not None:
                params["start"] = start_time_sec
            if end_time_sec is not None:
                params["end"] = end_time_sec

            data = self._request("GET", url, params=params)
            if not data.get("success") or not data.get("data"):
                return []

            d = data["data"]
            times = d.get("time", [])
            opens = d.get("open", [])
            highs = d.get("high", [])
            lows = d.get("low", [])
            closes = d.get("close", [])
            vols = d.get("vol", [])
            amounts = d.get("amount", [])

            contract_size = self.get_contract_sizes().get(symbol, 1.0)
            klines = []
            for i in range(len(times)):
                t_sec = int(times[i])
                t_ms = t_sec * 1000
                iso_str = datetime.fromtimestamp(t_sec, tz=timezone.utc).isoformat()
                o = float(opens[i])
                h = float(highs[i])
                l = float(lows[i])
                c = float(closes[i])
                v_contracts = float(vols[i])
                v_base = v_contracts * contract_size
                amt_usdt = float(amounts[i]) if amounts and i < len(amounts) else (v_base * c)

                klines.append(StandardKline(
                    open_time_ms=t_ms,
                    open_time_iso=iso_str,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume_base=v_base,
                    quote_volume_usdt=amt_usdt,
                    source_market="futures"
                ))
            return klines
        else:
            # Spot venue
            iv = self.INTERVAL_MAP_SPOT.get(timeframe)
            if not iv:
                raise ValueError(f"Unsupported spot timeframe: {timeframe}")

            url = f"{self.SPOT_BASE_URL}/klines"
            params = {"symbol": symbol, "interval": iv, "limit": min(limit, 1000)}
            if start_time_sec is not None:
                params["startTime"] = start_time_sec * 1000
            if end_time_sec is not None:
                params["endTime"] = end_time_sec * 1000

            data = self._request("GET", url, params=params)
            if not isinstance(data, list):
                return []

            klines = []
            for row in data:
                t_ms = int(row[0])
                t_sec = t_ms // 1000
                iso_str = datetime.fromtimestamp(t_sec, tz=timezone.utc).isoformat()
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
                v_base = float(row[5])
                amt_usdt = float(row[7])

                klines.append(StandardKline(
                    open_time_ms=t_ms,
                    open_time_iso=iso_str,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume_base=v_base,
                    quote_volume_usdt=amt_usdt,
                    source_market="spot"
                ))
            return klines

    def get_funding_history(
        self,
        symbol: str,
        page_num: int = 1,
        page_size: int = 100
    ) -> List[FundingRecord]:
        """Fetch funding rate history for futures."""
        if self.venue != "futures":
            return []

        url = f"{self.FUTURES_BASE_URL}/funding_rate/history"
        params = {"symbol": symbol, "page_num": page_num, "page_size": page_size}
        data = self._request("GET", url, params=params)
        if not data.get("success") or not data.get("data"):
            return []

        res_list = data["data"].get("resultList", [])
        records = []
        for r in res_list:
            settle_ms = int(r.get("settleTime", 0))
            records.append(FundingRecord(
                symbol=symbol,
                funding_rate=float(r.get("fundingRate", 0.0)),
                settle_time_ms=settle_ms,
                settle_time_iso=datetime.fromtimestamp(settle_ms / 1000, tz=timezone.utc).isoformat(),
                collect_cycle_hours=int(r.get("collectCycle", 8))
            ))
        return records

    def close(self):
        self.client.close()
