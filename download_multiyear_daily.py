"""
download_multiyear_daily.py

Downloads daily (1D) klines from MEXC public REST API from 2021-01-01 to present
for BTC, ETH, and ~150-180 liquid altcoins covering both historical cycles (2021-2024)
and modern leaders (2024-2026).
Saves to data_cache/klines_multiyear_1d/<symbol>.parquet
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiyear_downloader")

DATA_DIR = Path("data_cache/klines_multiyear_1d")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MEXC_BASE_URL = "https://api.mexc.com"

# Core historical coins from 2021-2024 cycles + modern ecosystem leaders
HISTORICAL_CORE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT",
    "XLMUSDT", "ALGOUSDT", "FILUSDT", "TRXUSDT", "EOSUSDT", "THETAUSDT", "AAVEUSDT",
    "AXSUSDT", "FTMUSDT", "MANAUSDT", "SANDUSDT", "CRVUSDT", "RUNEUSDT", "ICPUSDT",
    "CAKEUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT", "DYDXUSDT", "INJUSDT", "LDOUSDT",
    "OPUSDT", "ARBUSDT", "SUIUSDT", "APTUSDT", "TIAUSDT", "SEIUSDT", "KASUSDT",
    "RENDERUSDT", "FETUSDT", "TAOUSDT", "WLDUSDT", "JTOUSDT", "PYTHUSDT", "STXUSDT",
    "PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT", "PENDLEUSDT", "ORDIUSDT",
    "RBNUSDT", "BLURUSDT", "CFXUSDT", "GMXUSDT", "SNXUSDT", "MKRUSDT", "COMPUSDT",
    "UNIUSDT", "SUSHIUSDT", "1INCHUSDT", "KSMUSDT", "ZECUSDT", "DASHUSDT", "IOTAUSDT",
    "KLAYUSDT", "EGLDUSDT", "HBARUSDT", "FLOWUSDT", "ONEUSDT", "HOTUSDT", "ZILUSDT",
    "BATUSDT", "QTUMUSDT", "OMGUSDT", "ICXUSDT", "ONTUSDT", "WAVESUSDT", "SCUSDT"
]

def get_target_symbols():
    # 1. Start with historical core
    sym_set = set(HISTORICAL_CORE)
    
    # 2. Add existing cached symbols from 1h cache
    cache_1h = Path("data_cache/klines/1h")
    if cache_1h.exists():
        for p in cache_1h.glob("*.parquet"):
            s = p.stem.replace("_USDT", "USDT")
            sym_set.add(s)
            
    # 3. Add top 150 current liquid MEXC pairs (>100k volume)
    try:
        resp = requests.get(f"{MEXC_BASE_URL}/api/v3/ticker/24hr", timeout=10)
        if resp.status_code == 200:
            tickers = resp.json()
            liquid = [
                x["symbol"] for x in tickers 
                if x["symbol"].endswith("USDT") and float(x.get("quoteVolume", 0)) >= 100000.0
            ]
            sym_set.update(liquid[:150])
    except Exception as e:
        logger.warning(f"Could not fetch current 24hr tickers: {e}")
        
    symbols = sorted(list(sym_set))
    logger.info(f"Targeting {len(symbols)} total candidate symbols for multi-year daily download.")
    return symbols

def fetch_symbol_daily(symbol: str, start_date: str = "2021-01-01", end_date: str = "2026-10-05"):
    out_file = DATA_DIR / f"{symbol}.parquet"
    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            if len(existing_df) >= 300 and existing_df["open_time"].max() >= pd.Timestamp("2026-09-01", tz="UTC"):
                return symbol, len(existing_df), "already_cached"
        except Exception:
            pass

    cur_end = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
    min_time = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    
    all_rows = []
    url = f"{MEXC_BASE_URL}/api/v3/klines"
    
    while cur_end > min_time:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "endTime": cur_end,
            "limit": 500
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                time.sleep(0.5)
                break
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            all_rows.extend(data)
            first_t = data[0][0]
            if first_t >= cur_end:
                break
            cur_end = first_t - 1
            if len(data) < 500:
                break
            time.sleep(0.04)
        except Exception as e:
            time.sleep(0.5)
            break
            
    if not all_rows:
        return symbol, 0, "no_data"
        
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
        
    if len(df) < 50:
        return symbol, len(df), "too_few_rows"
        
    df.to_parquet(out_file, index=False)
    return symbol, len(df), "success"

def main():
    symbols = get_target_symbols()
    logger.info(f"Starting parallel download with 6 workers...")
    
    t0 = time.time()
    success_count = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_symbol_daily, s): s for s in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                s, count, status = future.result()
                if status in ("success", "already_cached"):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error fetching {sym}: {e}")
                
    elapsed = time.time() - t0
    logger.info(f"Downloaded multi-year daily data for {success_count}/{len(symbols)} symbols in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()
