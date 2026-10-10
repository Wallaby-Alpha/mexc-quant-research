"""
bot/weex_ob_sweep_bot.py
------------------------
Production Automated SMC/ICT Order Block Liquidity Sweep Trading Bot for WEEX.
Targets: Top 100 Crypto Contracts by Volume.

Core Strategy Logic:
1. Higher Timeframe (1h): Detects confirmed Order Blocks (last down-close before BOS breakout).
2. Lower Timeframe (15m): Detects Liquidity Sweeps (penetrates OB by >= 0.05% and closes back inside).
3. Risk Management:
   - Stop Loss: 1x ATR(14) below the sweep low.
   - Take Profit: Equal highs or 2.5R.
   - Native Exchange Brackets: Submits preset SL & TP directly to WEEX matching engine.
4. Operational Features:
   - Dynamic Top 100 Universe Scanner (refreshes daily).
   - Dry-Run Simulation Mode (enabled by default for safety).
   - Instant Telegram alerts with clickable bracket levels.
   - Automatic 15-minute synchronization loop (:00, :15, :30, :45 UTC).
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np

# Ensure UTF-8 output on all platforms
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("weex_ob_bot.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("weex_ob_bot")

DEFAULT_CONFIG_PATH = Path("weex_bot_config.json")
SCRIPT_DIR = Path(__file__).resolve().parent


def load_config() -> Dict[str, Any]:
    """Loads configuration with fallback candidates."""
    candidates = [
        DEFAULT_CONFIG_PATH,
        SCRIPT_DIR / "weex_bot_config.json",
        SCRIPT_DIR / "weex_bot_config.example.json",
    ]
    cfg = {
        "api_key": os.environ.get("WEEX_API_KEY", ""),
        "secret_key": os.environ.get("WEEX_SECRET_KEY", ""),
        "passphrase": os.environ.get("WEEX_PASSPHRASE", ""),
        "dry_run": True,
        "top_universe_count": 100,
        "risk_per_trade_usd": 100.0,
        "account_size_usd": 10000.0,
        "max_concurrent_positions": 5,
        "risk_reward_ratio": 2.5,
        "atr_multiplier": 1.0,
        "max_ob_age_hours": 8.0,
        "min_sweep_pct": 0.05,
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "poll_interval_seconds": 900
    }

    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                    cfg.update(file_cfg)
                    logger.info(f"Loaded config from {c}")
                    break
            except Exception as e:
                logger.warning(f"Failed to read {c}: {e}")

    return cfg


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    """Dispatches Telegram notifications with HTML formatting."""
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


class WeexClient:
    """Production client for WEEX V3 Contract API."""
    def __init__(self, api_key: str, secret_key: str, passphrase: str, dry_run: bool = True):
        self.base_url = "https://api-contract.weex.com"
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.dry_run = dry_run

    def _generate_signature(self, timestamp: str, method: str, path: str, query: str = "", body: str = "") -> str:
        """Constructs HMAC SHA256 base64 signature per WEEX documentation."""
        message = f"{timestamp}{method.upper()}{path}{query}{body}"
        mac = hmac.new(self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _get_headers(self, method: str, path: str, query: str = "", body: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sign = self._generate_signature(ts, method, path, query, body)
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }

    def get_top_contracts_by_volume(self, top_n: int = 100) -> List[str]:
        """Fetches all active USDT perpetuals and ranks by 24h quote turnover."""
        url = f"{self.base_url}/capi/v1/market/tickers"
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                logger.error(f"Failed to fetch tickers: {r.status_code} {r.text}")
                return []
            data = r.json()
            tickers = data.get("data", [])
            usdt_tickers = []
            for t in tickers:
                sym = t.get("symbol", "")
                if sym.endswith("USDT") or sym.endswith("_USDT") or "USDT" in sym:
                    vol = float(t.get("quoteVolume", 0.0) or t.get("turnover", 0.0) or t.get("amount24", 0.0))
                    usdt_tickers.append((sym, vol))

            usdt_tickers.sort(key=lambda x: x[1], reverse=True)
            top_syms = [x[0] for x in usdt_tickers[:top_n]]
            logger.info(f"Retrieved Top {len(top_syms)} WEEX symbols by volume.")
            return top_syms
        except Exception as e:
            logger.error(f"Error getting top contracts: {e}")
            return []

    def get_klines(self, symbol: str, granularity: str = "15m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches historical klines for analysis."""
        url = f"{self.base_url}/capi/v1/market/candles"
        gran_map = {"15m": "15m", "1h": "1h", "1d": "1d"}
        params = {
            "symbol": symbol,
            "granularity": gran_map.get(granularity, "15m"),
            "limit": limit
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json().get("data", [])
            if not data or len(data) < 20:
                return None
            # Format: [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
            df = df.sort_values("open_time").reset_index(drop=True)
            return df
        except Exception:
            return None

    def get_open_positions_count(self) -> int:
        """Fetches currently open positions count."""
        if self.dry_run or not self.api_key:
            return 0
        path = "/capi/v3/position/allPosition"
        try:
            headers = self._get_headers("GET", path)
            r = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            if r.status_code == 200:
                pos_list = r.json().get("data", [])
                active = [p for p in pos_list if float(p.get("total", 0.0)) > 0]
                return len(active)
            return 0
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")
            return 0

    def place_native_bracket_order(
        self,
        symbol: str,
        side: str,
        position_side: str,
        quantity: float,
        sl_price: float,
        tp_price: float
    ) -> Dict[str, Any]:
        """
        Submits market order with native preset TP & SL attached directly to WEEX engine.
        """
        if self.dry_run:
            logger.info(f"[DRY-RUN SIMULATION] {side} {position_side} {quantity} {symbol} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
            return {"status": "simulated", "orderId": "sim_123456"}

        path = "/capi/v3/order"
        payload = {
            "symbol": symbol,
            "side": side.upper(),                 # "BUY" or "SELL"
            "positionSide": position_side.upper(), # "LONG" or "SHORT"
            "type": "MARKET",
            "quantity": str(round(quantity, 4)),
            "slTriggerPrice": str(round(sl_price, 4)),
            "tpTriggerPrice": str(round(tp_price, 4)),
            "SlWorkingType": "MARK_PRICE",
            "TpWorkingType": "MARK_PRICE"
        }
        body_str = json.dumps(payload)
        headers = self._get_headers("POST", path, body=body_str)

        try:
            r = requests.post(f"{self.base_url}{path}", headers=headers, data=body_str, timeout=10)
            res = r.json()
            if r.status_code == 200 and res.get("code") == "00000":
                logger.info(f"✅ Live WEEX Order Placed: {symbol} {side} | Response: {res}")
                return {"status": "success", "data": res.get("data")}
            else:
                logger.error(f"❌ WEEX Order Failed: {r.status_code} {res}")
                return {"status": "error", "error": res}
        except Exception as e:
            logger.error(f"Exception submitting WEEX order: {e}")
            return {"status": "error", "error": str(e)}


# -----------------------------------------------------------------------------
# SMC/ICT SIGNAL DETECTION ENGINE
# -----------------------------------------------------------------------------

def find_fractal_swings(highs: np.ndarray, lows: np.ndarray, n: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Computes confirmed swing highs and lows with zero lookahead."""
    length = len(highs)
    sh = np.full(length, np.nan)
    sl = np.full(length, np.nan)
    for i in range(n, length - n):
        if all(highs[i] > highs[i - k] for k in range(1, n + 1)) and \
           all(highs[i] > highs[i + k] for k in range(1, n + 1)):
            sh[i + n] = highs[i]
        if all(lows[i] < lows[i - k] for k in range(1, n + 1)) and \
           all(lows[i] < lows[i + k] for k in range(1, n + 1)):
            sl[i + n] = lows[i]
    return pd.Series(sh).ffill().values, pd.Series(sl).ffill().values


def evaluate_symbol_for_sweep_setup(
    df1h: pd.DataFrame,
    df15m: pd.DataFrame,
    symbol: str,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Evaluates whether the symbol currently has a fresh 15m Order Block Liquidity Sweep."""
    if len(df1h) < 30 or len(df15m) < 30:
        return None

    # Exclude open forming candle; analyze fully closed historical candles
    closed_1h = df1h.iloc[:-1].copy()
    closed_15m = df15m.iloc[:-1].copy()

    h_1h = closed_1h["high"].values
    l_1h = closed_1h["low"].values
    o_1h = closed_1h["open"].values
    c_1h = closed_1h["close"].values
    times_1h = closed_1h["open_time"].tolist()

    sh_1h, sl_1h = find_fractal_swings(h_1h, l_1h, n=2)

    # 1. Identify Most Recent Confirmed 1h Order Block
    obs = []
    last_sh_broken = np.nan
    last_sl_broken = np.nan

    for i in range(5, len(closed_1h)):
        bar_t = times_1h[i]
        curr_sh = sh_1h[i]
        curr_sl = sl_1h[i]

        # Bullish BOS Breakout
        if not np.isnan(curr_sh) and c_1h[i] > curr_sh and curr_sh != last_sh_broken:
            last_sh_broken = curr_sh
            for j in range(i - 1, max(0, i - 10), -1):
                if c_1h[j] < o_1h[j]:
                    obs.append({
                        "ob_type": "bullish",
                        "creation_time": bar_t,
                        "ob_high": h_1h[j],
                        "ob_low": l_1h[j],
                        "equal_high": curr_sh
                    })
                    break

        # Bearish BOS Breakdown
        if not np.isnan(curr_sl) and c_1h[i] < curr_sl and curr_sl != last_sl_broken:
            last_sl_broken = curr_sl
            for j in range(i - 1, max(0, i - 10), -1):
                if c_1h[j] > o_1h[j]:
                    obs.append({
                        "ob_type": "bearish",
                        "creation_time": bar_t,
                        "ob_high": h_1h[j],
                        "ob_low": l_1h[j],
                        "equal_low": curr_sl
                    })
                    break

    if not obs:
        return None

    latest_ob = obs[-1]
    now_utc = closed_15m["open_time"].iloc[-1]
    age_hours = (now_utc - latest_ob["creation_time"]).total_seconds() / 3600.0
    if age_hours > config["max_ob_age_hours"]:
        return None

    # Calculate 15m ATR(14)
    h_15 = closed_15m["high"].values
    l_15 = closed_15m["low"].values
    c_15 = closed_15m["close"].values
    prev_c = np.roll(c_15, 1)
    prev_c[0] = c_15[0]
    tr = np.maximum(h_15 - l_15, np.maximum(np.abs(h_15 - prev_c), np.abs(l_15 - prev_c)))
    atr14 = float(pd.Series(tr).rolling(14, min_periods=1).mean().iloc[-1])

    latest_15m = closed_15m.iloc[-1]
    last_low = float(latest_15m["low"])
    last_high = float(latest_15m["high"])
    last_close = float(latest_15m["close"])
    sweep_pct = config["min_sweep_pct"] / 100.0

    # Bullish Liquidity Sweep Check
    if latest_ob["ob_type"] == "bullish":
        sweep_threshold = latest_ob["ob_low"] * (1.0 - sweep_pct)
        if last_low <= sweep_threshold and last_close >= latest_ob["ob_low"]:
            entry_p = last_close
            sl_p = last_low - (config["atr_multiplier"] * atr14)
            risk = entry_p - sl_p
            if risk > 0:
                tp_p = entry_p + (config["risk_reward_ratio"] * risk)
                if latest_ob.get("equal_high", np.nan) > entry_p:
                    tp_p = min(tp_p, latest_ob["equal_high"])
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "side": "BUY",
                    "position_side": "LONG",
                    "entry_price": entry_p,
                    "sl_price": sl_p,
                    "tp_price": tp_p,
                    "risk_usd_dist": risk,
                    "atr14": atr14,
                    "ob_level": latest_ob["ob_low"]
                }

    # Bearish Liquidity Sweep Check
    elif latest_ob["ob_type"] == "bearish":
        sweep_threshold = latest_ob["ob_high"] * (1.0 + sweep_pct)
        if last_high >= sweep_threshold and last_close <= latest_ob["ob_high"]:
            entry_p = last_close
            sl_p = last_high + (config["atr_multiplier"] * atr14)
            risk = sl_p - entry_p
            if risk > 0:
                tp_p = entry_p - (config["risk_reward_ratio"] * risk)
                if latest_ob.get("equal_low", np.nan) < entry_p:
                    tp_p = max(tp_p, latest_ob["equal_low"])
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "side": "SELL",
                    "position_side": "SHORT",
                    "entry_price": entry_p,
                    "sl_price": sl_p,
                    "tp_price": tp_p,
                    "risk_usd_dist": risk,
                    "atr14": atr14,
                    "ob_level": latest_ob["ob_high"]
                }

    return None


# -----------------------------------------------------------------------------
# MAIN BOT RUNNER
# -----------------------------------------------------------------------------

def run_scan_and_execute(client: WeexClient, config: Dict[str, Any], symbols: List[str]):
    """Scans all symbols in the universe and executes confirmed sweeps."""
    logger.info(f"Scanning {len(symbols)} Top Volume symbols for 15m Order Block Sweeps...")
    active_positions = client.get_open_positions_count()
    max_pos = config["max_concurrent_positions"]

    if active_positions >= max_pos:
        logger.info(f"Max concurrent positions reached ({active_positions}/{max_pos}). Skipping scan cycle.")
        return

    confirmed_signals = []

    for sym in symbols:
        df1h = client.get_klines(sym, granularity="1h", limit=50)
        df15m = client.get_klines(sym, granularity="15m", limit=50)
        if df1h is None or df15m is None:
            continue

        sig = evaluate_symbol_for_sweep_setup(df1h, df15m, sym, config)
        if sig:
            confirmed_signals.append(sig)

    logger.info(f"Scan finished. Found {len(confirmed_signals)} sweep signals.")

    for sig in confirmed_signals:
        if active_positions >= max_pos:
            logger.info("Maximum positions filled. Halting execution.")
            break

        sym = sig["symbol"]
        entry = sig["entry_price"]
        sl = sig["sl_price"]
        tp = sig["tp_price"]
        risk_dist = sig["risk_usd_dist"]

        # Calculate position size based on risk_per_trade_usd
        risk_amount = config["risk_per_trade_usd"]
        quantity = risk_amount / risk_dist if risk_dist > 0 else 0.0

        if quantity <= 0:
            continue

        # Place Native Bracket Order on WEEX
        result = client.place_native_bracket_order(
            symbol=sym,
            side=sig["side"],
            position_side=sig["position_side"],
            quantity=quantity,
            sl_price=sl,
            tp_price=tp
        )

        active_positions += 1

        # Dispatch Telegram Alert
        prefix = "🚨 [DRY-RUN SIMULATION]" if config["dry_run"] else "⚡ [LIVE ORDER EXECUTED]"
        msg = (
            f"<b>{prefix} SMC ORDER BLOCK SWEEP!</b>\n"
            f"• Contract: <code>{sym}</code> ({sig['direction'].upper()})\n"
            f"• Entry: <code>${entry:,.4f}</code>\n"
            f"• <b>Native Exchange SL:</b> <code>${sl:,.4f}</code> (-1x ATR)\n"
            f"• <b>Native Exchange TP:</b> <code>${tp:,.4f}</code> (+2.5R)\n"
            f"• Position Size: <code>{quantity:,.2f}</code> units (${risk_amount:,.0f} risk)\n"
            f"• Active Positions: <code>{active_positions}/{max_pos}</code>"
        )
        send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], msg)
        time.sleep(0.5)


def start_bot_daemon():
    """Runs the 15-minute continuous scheduler."""
    logger.info("Starting WEEX Order Block Liquidity Sweep Bot Daemon...")
    config = load_config()
    client = WeexClient(
        api_key=config["api_key"],
        secret_key=config["secret_key"],
        passphrase=config["passphrase"],
        dry_run=config["dry_run"]
    )

    logger.info(f"Bot Mode: {'DRY RUN (Simulated)' if config['dry_run'] else 'LIVE TRADING'}")
    logger.info(f"Target Universe: Top {config['top_universe_count']} Volume Coins")

    # Initial Universe Discovery
    universe = client.get_top_contracts_by_volume(top_n=config["top_universe_count"])
    last_universe_refresh = time.time()

    # Send Startup Message
    startup_msg = (
        f"🤖 <b>WEEX SMC ORDER BLOCK BOT INITIALIZED</b>\n"
        f"• Mode: <code>{'DRY-RUN SIMULATION' if config['dry_run'] else 'LIVE EXECUTION'}</code>\n"
        f"• Universe: <b>Top {len(universe)} Volume Coins</b>\n"
        f"• Risk per Trade: <code>${config['risk_per_trade_usd']:,.0f}</code>\n"
        f"• Native Bracket Target: <b>2.5R</b> | Stop: <b>1x ATR(14)</b>\n"
        f"• Cadence: <b>Every 15 Minutes at Candle Close</b>"
    )
    send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], startup_msg)

    while True:
        try:
            # Refresh universe daily
            if time.time() - last_universe_refresh > 86400:
                universe = client.get_top_contracts_by_volume(top_n=config["top_universe_count"])
                last_universe_refresh = time.time()

            # Execute Scan Cycle
            run_scan_and_execute(client, config, universe)

            # Sleep until next 15m candle close + 3 seconds buffer
            now = time.time()
            remainder = now % 900 # 900s = 15m
            sleep_time = (900 - remainder) + 3.0
            logger.info(f"Sleeping {sleep_time:.1f}s until next 15-minute candle close...")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in bot execution loop: {e}")
            time.sleep(15)


if __name__ == "__main__":
    start_bot_daemon()
