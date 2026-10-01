"""
scanner/weekly_rotation_scanner.py
Production-ready scanner for the 30-Day Rotational Momentum Strategy.
Runs weekly (e.g., Monday 00:00 UTC), evaluates BTC 50-day EMA, ranks the top 150 MEXC altcoins,
applies the Rank-15 buffer rule against current holdings, and sends actionable buy/sell orders via Telegram.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np

# Ensure UTF-8 output on all platforms (including Windows consoles)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scanner.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("rotation_scanner")

MEXC_BASE_URL = "https://api.mexc.com"
STATE_FILE = Path("portfolio_state.json")
CONFIG_FILE = Path("scanner_config.json")


def load_config() -> Dict[str, Any]:
    """Loads Telegram credentials and settings from env or config file."""
    config = {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "top_k": int(os.environ.get("TOP_K", "10")),
        "rank_exit_buffer": int(os.environ.get("RANK_EXIT_BUFFER", "15")),
        "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "30")),
        "min_volume_24h_usdt": float(os.environ.get("MIN_VOLUME_USDT", "500000.0")),
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                config.update(file_cfg)
        except Exception as e:
            logger.warning(f"Failed to read {CONFIG_FILE}: {e}")
    return config


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Dispatches formatted message to Telegram Bot."""
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not configured. Printing message to stdout only.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Successfully sent Telegram alert.")
            return True
        else:
            logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def get_mexc_klines(symbol: str, interval: str = "1d", limit: int = 100) -> pd.DataFrame:
    """Fetches daily klines from MEXC public REST API with automatic retries."""
    url = f"{MEXC_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
            data = resp.json()
            if not data or not isinstance(data, list):
                return pd.DataFrame()

            # Format: [open_time, open, high, low, close, volume, close_time, quote_volume]
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)
            return df.sort_values("open_time").reset_index(drop=True)
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Error fetching klines for {symbol} after 3 attempts: {e}")
            time.sleep(1.0)
    return pd.DataFrame()


def get_top_mexc_pairs(min_volume: float = 500000.0) -> List[str]:
    """Retrieves liquid USDT spot trading pairs on MEXC."""
    url = f"{MEXC_BASE_URL}/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()

        pairs = []
        for item in data:
            sym = item.get("symbol", "")
            # Only consider standard USDT spot pairs (exclude leveraged ETF tokens, stablecoins)
            if not sym.endswith("USDT"):
                continue
            if any(x in sym for x in ["3L", "3S", "4L", "4S", "5L", "5S", "USDC", "BUSD", "TUSD", "FDUSD", "EUR"]):
                continue

            quote_vol = float(item.get("quoteVolume", 0.0))
            if quote_vol >= min_volume:
                pairs.append((sym, quote_vol))

        # Sort by 24h volume descending and return top 150
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [p[0] for p in pairs[:150]]
        logger.info(f"Retrieved {len(top_symbols)} liquid USDT pairs above ${min_volume:,.0f} 24h volume.")
        return top_symbols
    except Exception as e:
        logger.error(f"Error fetching 24hr tickers: {e}")
        return []


def check_btc_macro_trend() -> Tuple[bool, float, float]:
    """Checks if BTC daily close is above its 50-day EMA."""
    df_btc = get_mexc_klines("BTCUSDT", interval="1d", limit=100)
    if len(df_btc) < 55:
        logger.error("Insufficient BTC daily bars.")
        return False, 0.0, 0.0

    # Calculate 50-day EMA
    df_btc["ema50"] = df_btc["close"].ewm(span=50, adjust=False).mean()
    latest = df_btc.iloc[-1]
    btc_close = latest["close"]
    btc_ema50 = latest["ema50"]
    is_bullish = btc_close >= btc_ema50

    return is_bullish, btc_close, btc_ema50


def load_portfolio_state() -> List[str]:
    """Loads currently held symbols from local state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("current_holdings", [])
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")
    return []


def save_portfolio_state(holdings: List[str]):
    """Persists updated portfolio holdings to local state."""
    data = {
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "current_holdings": holdings
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_scanner():
    logger.info("=" * 60)
    logger.info("RUNNING WEEKLY ROTATIONAL MOMENTUM SCANNER")
    logger.info("=" * 60)

    cfg = load_config()
    current_holdings = load_portfolio_state()

    # 1. Evaluate Bitcoin Macro Health
    is_btc_bullish, btc_close, btc_ema50 = check_btc_macro_trend()
    logger.info(f"BTC Close: ${btc_close:,.2f} | 50-day EMA: ${btc_ema50:,.2f} | Bullish: {is_btc_bullish}")

    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    if not is_btc_bullish:
        # Cash Protection Mode triggered
        msg = (
            f"🚨 <b>WEEKLY ROTATION SCANNER: CASH SHIELD TRIGGERED</b>\n"
            f"📅 <i>{now_utc}</i>\n\n"
            f"<b>BTC Macro Trend:</b> ❌ BEARISH\n"
            f"• BTC Price: <code>${btc_close:,.2f}</code>\n"
            f"• 50-Day EMA: <code>${btc_ema50:,.2f}</code>\n\n"
            f"⚠️ <b>ACTION REQUIRED:</b>\n"
            f"<b>MOVE 100% TO USDT CASH</b>\n\n"
        )
        if current_holdings:
            msg += "<b>SELL ORDERS:</b>\n"
            for sym in current_holdings:
                msg += f"🔴 SELL <code>{sym}</code> to USDT\n"
            save_portfolio_state([])
        else:
            msg += "✅ Already 100% Cash. No action needed.\n"

        print(msg)
        send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
        return

    # 2. BTC is Bullish: Rank Universe by 30-Day Return vs. BTC
    symbols = get_top_mexc_pairs(min_volume=cfg["min_volume_24h_usdt"])
    if not symbols:
        logger.error("No valid pairs returned.")
        return

    df_btc = get_mexc_klines("BTCUSDT", interval="1d", limit=40)
    btc_p0 = df_btc.iloc[-31]["close"]
    btc_p1 = df_btc.iloc[-1]["close"]
    btc_30d_return = (btc_p1 - btc_p0) / btc_p0

    logger.info(f"Bitcoin 30-day Return: {btc_30d_return*100:+.2f}%")

    rankings = []
    for i, sym in enumerate(symbols):
        if sym == "BTCUSDT":
            continue
        df_k = get_mexc_klines(sym, interval="1d", limit=40)
        if len(df_k) < 32:
            continue
        p0 = df_k.iloc[-31]["close"]
        p1 = df_k.iloc[-1]["close"]
        if p0 <= 0:
            continue

        alt_ret = (p1 - p0) / p0
        rs_spread = alt_ret - btc_30d_return

        rankings.append({
            "symbol": sym,
            "alt_return_30d": alt_ret,
            "rs_spread_vs_btc": rs_spread,
            "latest_close": p1
        })
        time.sleep(0.04) # API rate limit etiquette

    df_ranks = pd.DataFrame(rankings).sort_values(by="rs_spread_vs_btc", ascending=False).reset_index(drop=True)
    df_ranks["rank"] = df_ranks.index + 1

    top_10 = df_ranks.head(cfg["top_k"])
    top_10_symbols = top_10["symbol"].tolist()
    top_15_symbols = df_ranks.head(cfg["rank_exit_buffer"])["symbol"].tolist()

    # 3. Determine Portfolio Actions (Applying Rank 15 Buffer Rule)
    sells = []
    holds = []
    buys = []

    # Check existing holdings
    for sym in current_holdings:
        if sym not in top_15_symbols:
            sells.append(sym)
        else:
            holds.append(sym)

    # Determine which new leaders need to be added to reach Top 10 capacity
    slots_needed = cfg["top_k"] - len(holds)
    for sym in top_10_symbols:
        if sym not in holds and len(buys) < slots_needed:
            buys.append(sym)

    new_holdings = holds + buys

    # 4. Format Actionable Telegram Notification
    msg = (
        f"🚀 <b>WEEKLY ROTATION SCANNER: PORTFOLIO ORDERS</b>\n"
        f"📅 <i>{now_utc}</i>\n\n"
        f"<b>BTC Macro Trend:</b> ✅ BULLISH\n"
        f"• BTC Price: <code>${btc_close:,.2f}</code>\n"
        f"• 50-Day EMA: <code>${btc_ema50:,.2f}</code>\n"
        f"• BTC 30d Return: <code>{btc_30d_return*100:+.1f}%</code>\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>ACTION PLAN FOR THIS WEEK:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
    )

    if sells:
        msg += "🔴 <b>SELL ORDERS (Dropped out of Top 15):</b>\n"
        for sym in sells:
            msg += f"• SELL <code>{sym}</code> to 100% USDT\n"
        msg += "\n"

    if buys:
        msg += f"🟢 <b>BUY ORDERS (Target 10% allocation each):</b>\n"
        for sym in buys:
            row = df_ranks[df_ranks["symbol"] == sym].iloc[0]
            msg += f"• BUY <code>{sym}</code> (Rank #{row['rank']} | RS: {row['rs_spread_vs_btc']*100:+.1f}%)\n"
        msg += "\n"

    if holds:
        msg += "🟡 <b>HOLD (Maintained in Top 15):</b>\n"
        for sym in holds:
            row = df_ranks[df_ranks["symbol"] == sym].iloc[0]
            msg += f"• HOLD <code>{sym}</code> (Rank #{row['rank']})\n"
        msg += "\n"

    if not sells and not buys:
        msg += "✨ <b>NO CHANGES REQUIRED!</b> Current holdings remain firmly inside the Top 10/15.\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 <b>TOP 10 RELATIVE STRENGTH LEADERS:</b>\n"
    for _, row in top_10.iterrows():
        msg += f"<b>#{row['rank']:2d}</b> <code>{row['symbol']:10s}</code>: {row['alt_return_30d']*100:+5.1f}% (RS: {row['rs_spread_vs_btc']*100:+5.1f}%)\n"

    print(msg)
    save_portfolio_state(new_holdings)
    send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)


if __name__ == "__main__":
    run_scanner()
