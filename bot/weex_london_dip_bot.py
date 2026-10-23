"""
bot/weex_london_dip_bot.py
==========================
Production 24/7 Automated Trading Bot for Setup 11:
"The London Dip to NY Peak Momentum Ride" on WEEX Contract Exchange.

Strategy Specification:
1. Every day at 07:55 UTC:
   - Scans WEEX USDT perpetuals for yesterday's top momentum runners (gain >= +10%).
   - Selects up to top K leaders (default: 5).
   - Computes strict $1.00 risk position sizing with a -8.0% hard stop loss.
   - Automatically REJECTS any coin where the minimum order size would risk > $1.00.
2. Every day at 08:00 UTC (London Open):
   - Enters Market BUY orders on all eligible candidates.
   - Attaches native Hard Stop Loss (-8.0%) directly to the WEEX matching engine.
   - Dispatches Telegram notifications with entry price, size, and exact dollar risk.
3. Every day at 20:00 UTC (US Cash Close):
   - Market closes all active positions.
   - Computes realized P&L and dispatches Telegram summary.
4. Intraday Protection & Persistence:
   - Tracks active positions in a local state file to survive process reboots.
   - Supports dry-run paper trading mode (`"dry_run": true`).
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np

# Force UTF-8 console output
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
        logging.FileHandler("weex_london_dip.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("weex_london_dip")

CONFIG_CANDIDATES = [
    Path("weex_london_dip_config.json"),
    Path(__file__).resolve().parent / "weex_london_dip_config.json",
    Path(__file__).resolve().parent / "weex_london_dip_config.example.json",
]
STATE_FILE = Path("weex_london_dip_positions.json")


def load_config() -> Dict[str, Any]:
    """Loads configuration with fallback candidates and environment variable overrides."""
    cfg = {
        "api_key": os.environ.get("WEEX_API_KEY", ""),
        "secret_key": os.environ.get("WEEX_SECRET_KEY", ""),
        "passphrase": os.environ.get("WEEX_PASSPHRASE", ""),
        "dry_run": True,
        "max_risk_usd": 1.0,
        "min_prior_day_gain_pct": 10.0,
        "max_candidates": 5,
        "stop_loss_pct": 8.0,
        "entry_time_utc": "08:00",
        "scan_time_utc": "07:55",
        "exit_time_utc": "20:00",
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "send_heartbeat_daily": True
    }

    for path in CONFIG_CANDIDATES:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    cfg.update(file_data)
                    logger.info(f"Loaded config from {path}")
                    break
            except Exception as e:
                logger.warning(f"Could not load config from {path}: {e}")

    return cfg


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Dispatches Telegram notifications with HTML formatting."""
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"Failed to send Telegram message: {e}")
        return False


class WeexClient:
    """Client for WEEX V3 Contract API."""
    def __init__(self, api_key: str, secret_key: str, passphrase: str, dry_run: bool = True):
        self.base_url = "https://api-contract.weex.com"
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.dry_run = dry_run
        self.exchange_info: Dict[str, Dict[str, Any]] = {}
        self.refresh_exchange_info()

    def _generate_signature(self, timestamp: str, method: str, path: str, query: str = "", body: str = "") -> str:
        """Constructs HMAC SHA256 base64 signature per WEEX V3 API specification."""
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

    def refresh_exchange_info(self):
        """Fetches and caches symbol precision and lot size limits."""
        url = f"{self.base_url}/capi/v3/market/exchangeInfo"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json().get("symbols", [])
                meta = {}
                for s in data:
                    sym = s.get("symbol", "")
                    meta[sym] = {
                        "pricePrecision": int(s.get("pricePrecision", 4)),
                        "quantityPrecision": int(s.get("quantityPrecision", 2)),
                        "contractVal": float(s.get("contractVal", 1.0)),
                        "minOrderSize": float(s.get("minOrderSize", 1.0)),
                        "maxOrderSize": float(s.get("maxOrderSize", 100000000.0)),
                    }
                self.exchange_info = meta
                logger.info(f"Loaded exchange trading rules for {len(self.exchange_info)} symbols.")
        except Exception as e:
            logger.warning(f"Failed to fetch exchangeInfo: {e}")

    def get_ticker_24hr(self) -> List[Dict[str, Any]]:
        """Fetches 24hr ticker data for all perpetual contracts."""
        url = f"{self.base_url}/capi/v3/market/ticker/24hr"
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error fetching 24hr tickers: {e}")
        return []

    def get_yesterday_gain(self, symbol: str) -> Optional[float]:
        """
        Fetches daily klines and calculates exact prior-calendar-day return (00:00 to 23:45 UTC).
        Chronologically sorts candles so candle [-2] is yesterday's completed day.
        """
        url = f"{self.base_url}/capi/v3/market/klines"
        params = {"symbol": symbol, "interval": "1d", "limit": 5}
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                klines = r.json()
                if isinstance(klines, list) and len(klines) >= 2:
                    # Sort chronologically by start timestamp (index 0)
                    sorted_klines = sorted(klines, key=lambda x: int(x[0]))
                    # Index -1 is current in-progress day; Index -2 is yesterday's completed day
                    y_open = float(sorted_klines[-2][1])
                    y_close = float(sorted_klines[-2][4])
                    if y_open > 0:
                        return (y_close - y_open) / y_open
        except Exception as e:
            logger.debug(f"Could not fetch 1d klines for {symbol}: {e}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fetches the latest market price for a symbol."""
        url = f"{self.base_url}/capi/v3/market/ticker/24hr"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                for t in r.json():
                    if t.get("symbol") == symbol:
                        p = float(t.get("lastPrice", 0.0) or t.get("last", 0.0) or 0.0)
                        if p > 0:
                            return p
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None

    def calculate_risk_position_size(
        self,
        symbol: str,
        entry_price: float,
        sl_pct: float = 0.08,
        max_risk_usd: float = 1.0
    ) -> Tuple[float, float, bool, str]:
        """
        Calculates position size strictly risking max_risk_usd ($1.00).
        If the minimum contract size requires more than max_risk_usd, REJECTS trade.

        Returns:
            (actual_quantity, actual_risk_usd, is_eligible, reason)
        """
        info = self.exchange_info.get(symbol, {
            "pricePrecision": 4,
            "quantityPrecision": 2,
            "contractVal": 1.0,
            "minOrderSize": 1.0
        })

        min_size = info.get("minOrderSize", 1.0)
        c_val = info.get("contractVal", 1.0)
        step = c_val if c_val > 0 else min_size

        if entry_price <= 0 or sl_pct <= 0:
            return 0.0, 0.0, False, "Invalid price or SL percentage"

        # Check: What is the risk of buying the absolute minimum order?
        min_order_risk = min_size * entry_price * sl_pct
        if min_order_risk > max_risk_usd:
            return (
                0.0,
                min_order_risk,
                False,
                f"Minimum order ({min_size} lots) risks ${min_order_risk:.2f}, exceeding max allowed ${max_risk_usd:.2f}"
            )

        # Desired quantity to risk exactly max_risk_usd
        desired_qty = max_risk_usd / (entry_price * sl_pct)

        # Round down to the nearest lot step size to strictly keep risk <= max_risk_usd
        lots = int(desired_qty / step)
        actual_qty = lots * step

        if actual_qty < min_size:
            return (
                0.0,
                0.0,
                False,
                f"Calculated quantity {actual_qty:.4f} is below minimum allowed order size {min_size}"
            )

        actual_risk_usd = actual_qty * entry_price * sl_pct
        if actual_risk_usd > max_risk_usd:
            # Safety boundary check
            actual_qty = (lots - 1) * step
            actual_risk_usd = actual_qty * entry_price * sl_pct

        return actual_qty, actual_risk_usd, True, "Eligible"

    def format_values(self, symbol: str, qty: float, entry_p: float, sl_p: float) -> Tuple[str, str, str]:
        """Formats order numbers to symbol precision rules."""
        info = self.exchange_info.get(symbol, {"pricePrecision": 4, "quantityPrecision": 2})
        q_prec = info.get("quantityPrecision", 2)
        p_prec = info.get("pricePrecision", 4)

        qty_str = f"{qty:.{q_prec}f}" if q_prec > 0 else str(int(qty))
        entry_str = f"{entry_p:.{p_prec}f}"
        sl_str = f"{sl_p:.{p_prec}f}"
        return qty_str, entry_str, sl_str

    def place_market_buy_with_sl(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        sl_price: float
    ) -> Dict[str, Any]:
        """Submits Market Buy with native Stop Loss attached to the matching engine."""
        qty_str, entry_str, sl_str = self.format_values(symbol, qty, entry_price, sl_price)

        if self.dry_run:
            logger.info(f"[DRY RUN] BUY {qty_str} {symbol} @ ~{entry_str} | Native SL @ {sl_str}")
            return {
                "status": "simulated",
                "symbol": symbol,
                "qty": float(qty_str),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "orderId": f"sim_{int(time.time()*1000)}"
            }

        path = "/capi/v3/order"
        payload = {
            "symbol": symbol,
            "side": "BUY",
            "positionSide": "LONG",
            "type": "MARKET",
            "quantity": qty_str,
            "slTriggerPrice": sl_str,
            "SlWorkingType": "MARK_PRICE",
            "newClientOrderId": f"ld_{int(time.time()*1000)}"
        }
        body_str = json.dumps(payload)
        headers = self._get_headers("POST", path, body=body_str)

        try:
            r = requests.post(f"{self.base_url}{path}", headers=headers, data=body_str, timeout=12)
            res = r.json()
            if r.status_code == 200 and res.get("code") == "00000":
                logger.info(f"✅ Live WEEX Order Placed: {symbol} BUY {qty_str} | SL: {sl_str}")
                return {
                    "status": "success",
                    "symbol": symbol,
                    "qty": float(qty_str),
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "orderId": res.get("data", {}).get("orderId", "N/A")
                }
            # Auto-fallback to positionSide="BOTH" if One-Way Mode (-1191)
            elif str(res.get("code")) in ("-1191", "1191"):
                logger.warning(f"Got code -1191 for {symbol}. Retrying with positionSide='BOTH'...")
                payload["positionSide"] = "BOTH"
                body_both = json.dumps(payload)
                headers_both = self._get_headers("POST", path, body=body_both)
                r_both = requests.post(f"{self.base_url}{path}", headers=headers_both, data=body_both, timeout=12)
                res_both = r_both.json()
                if r_both.status_code == 200 and res_both.get("code") == "00000":
                    logger.info(f"✅ Live WEEX Order Placed (BOTH): {symbol} BUY {qty_str} | SL: {sl_str}")
                    return {
                        "status": "success",
                        "symbol": symbol,
                        "qty": float(qty_str),
                        "entry_price": entry_price,
                        "sl_price": sl_price,
                        "orderId": res_both.get("data", {}).get("orderId", "N/A")
                    }
                else:
                    logger.error(f"❌ WEEX Order Rejected (BOTH): {res_both}")
                    return {"status": "error", "error": res_both}
            else:
                logger.error(f"❌ WEEX Order Rejected: HTTP {r.status_code} | Body: {res}")
                return {"status": "error", "error": res}
        except Exception as e:
            logger.error(f"Exception placing order for {symbol}: {e}")
            return {"status": "error", "error": str(e)}

    def close_position_market(self, symbol: str, qty: float) -> Dict[str, Any]:
        """Submits a market SELL order to close an open long position."""
        info = self.exchange_info.get(symbol, {"quantityPrecision": 2})
        q_prec = info.get("quantityPrecision", 2)
        qty_str = f"{qty:.{q_prec}f}" if q_prec > 0 else str(int(qty))

        if self.dry_run:
            logger.info(f"[DRY RUN] CLOSE POSITION {qty_str} {symbol}")
            return {"status": "simulated", "symbol": symbol, "qty": float(qty_str)}

        path = "/capi/v3/order"
        payload = {
            "symbol": symbol,
            "side": "SELL",
            "positionSide": "LONG",
            "type": "MARKET",
            "quantity": qty_str,
            "newClientOrderId": f"close_{int(time.time()*1000)}"
        }
        body_str = json.dumps(payload)
        headers = self._get_headers("POST", path, body=body_str)

        try:
            r = requests.post(f"{self.base_url}{path}", headers=headers, data=body_str, timeout=12)
            res = r.json()
            if r.status_code == 200 and res.get("code") == "00000":
                logger.info(f"✅ Closed Position: {symbol} SELL {qty_str}")
                return {"status": "success", "symbol": symbol, "data": res.get("data")}
            elif str(res.get("code")) in ("-1191", "1191"):
                payload["positionSide"] = "BOTH"
                body_b = json.dumps(payload)
                headers_b = self._get_headers("POST", path, body=body_b)
                r_b = requests.post(f"{self.base_url}{path}", headers=headers_b, data=body_b, timeout=12)
                res_b = r_b.json()
                if r_b.status_code == 200 and res_b.get("code") == "00000":
                    logger.info(f"✅ Closed Position (BOTH): {symbol} SELL {qty_str}")
                    return {"status": "success", "symbol": symbol, "data": res_b.get("data")}
            logger.error(f"❌ Failed to close {symbol}: {res}")
            return {"status": "error", "error": res}
        except Exception as e:
            logger.error(f"Exception closing position {symbol}: {e}")
            return {"status": "error", "error": str(e)}


# -----------------------------------------------------------------------------
# BOT COORDINATOR & LIFECYCLE
# -----------------------------------------------------------------------------

class LondonDipBot:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.client = WeexClient(
            api_key=self.cfg.get("api_key", ""),
            secret_key=self.cfg.get("secret_key", ""),
            passphrase=self.cfg.get("passphrase", ""),
            dry_run=self.cfg.get("dry_run", True)
        )
        self.active_positions: Dict[str, Dict[str, Any]] = self.load_state()

    def load_state(self) -> Dict[str, Dict[str, Any]]:
        """Loads active positions from disk if process restarted."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} active positions from {STATE_FILE}")
                    return data
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return {}

    def save_state(self):
        """Persists active positions to disk."""
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def notify(self, message: str):
        """Dispatches notification to Telegram and logger."""
        logger.info(message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
        token = self.cfg.get("telegram_bot_token", "")
        chat_id = self.cfg.get("telegram_chat_id", "")
        if token and chat_id:
            send_telegram(token, chat_id, message)

    def scan_morning_candidates(self) -> List[Dict[str, Any]]:
        """
        Scans all WEEX USDT perpetuals at 07:55 UTC:
        1. Filters symbols where yesterday's gain >= min_prior_day_gain_pct (e.g. +10%).
        2. Ranks by gain and selects top candidates (up to max_candidates).
        3. Runs strict $1 risk qualification check.
        """
        min_gain = self.cfg.get("min_prior_day_gain_pct", 10.0) / 100.0
        max_cands = self.cfg.get("max_candidates", 5)
        max_risk = self.cfg.get("max_risk_usd", 1.0)
        sl_pct = self.cfg.get("stop_loss_pct", 8.0) / 100.0

        logger.info("🌅 Running 07:55 UTC Pre-London Momentum Scan...")
        tickers = self.client.get_ticker_24hr()
        if not tickers:
            self.notify("⚠️ <b>WEEX SCAN WARNING:</b> Failed to retrieve 24hr tickers.")
            return []

        # Filter active USDT contracts with sufficient volume
        usdt_tickers = []
        for t in tickers:
            sym = t.get("symbol", "")
            if sym.endswith("USDT") or sym.endswith("_USDT"):
                # Broad initial filter using 24h change before querying completed klines
                change_24h = float(t.get("priceChangePercent", 0.0) or 0.0)
                vol = float(t.get("quoteVolume", 0.0) or t.get("volume", 0.0) or 0.0)
                if change_24h >= (min_gain * 0.5):
                    usdt_tickers.append((sym, change_24h, vol))

        # Check precise completed prior-day candle return
        qualified = []
        for sym, chg, vol in usdt_tickers:
            y_ret = self.client.get_yesterday_gain(sym)
            if y_ret is not None and y_ret >= min_gain:
                curr_price = self.client.get_current_price(sym)
                if curr_price and curr_price > 0:
                    qty, risk_usd, eligible, reason = self.client.calculate_risk_position_size(
                        symbol=sym,
                        entry_price=curr_price,
                        sl_pct=sl_pct,
                        max_risk_usd=max_risk
                    )
                    qualified.append({
                        "symbol": sym,
                        "yesterday_gain": y_ret,
                        "current_price": curr_price,
                        "qty": qty,
                        "risk_usd": risk_usd,
                        "eligible": eligible,
                        "reason": reason
                    })

        # Sort by yesterday's gain descending
        qualified.sort(key=lambda x: x["yesterday_gain"], reverse=True)
        selected = qualified[:max_cands]

        # Dispatch Morning Scan Telegram Report
        mode_str = "🧪 DRY RUN" if self.cfg.get("dry_run", True) else "⚡ LIVE"
        if not selected:
            msg = (
                f"🌅 <b>SETUP 11 MORNING SCAN [{mode_str}]</b>\n"
                f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>\n\n"
                f"No coins met the +{min_gain*100:.0f}% prior-day momentum threshold today.\n"
                f"Bot standing aside."
            )
            self.notify(msg)
            return []

        lines = [
            f"🌅 <b>SETUP 11 MORNING SCAN [{mode_str}]</b>",
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>\n",
            f"Found <b>{len(selected)}</b> top momentum leader(s) (>= +{min_gain*100:.0f}%):\n"
        ]

        for s in selected:
            gain_pct = s["yesterday_gain"] * 100
            price = s["current_price"]
            if s["eligible"]:
                notional = s["qty"] * price
                lines.append(
                    f"✅ <b>{s['symbol']}</b>: +{gain_pct:.1f}% yesterday\n"
                    f"   • Price: ${price:.4f} | Size: {s['qty']} lots (${notional:.2f} notional)\n"
                    f"   • Dollar Risk: <b>${s['risk_usd']:.2f}</b> (<= ${max_risk:.2f})\n"
                )
            else:
                lines.append(
                    f"⚠️ <b>{s['symbol']}</b>: +{gain_pct:.1f}% yesterday\n"
                    f"   • <b>REJECTED:</b> {s['reason']}\n"
                )

        lines.append(f"Execution scheduled for <b>08:00 UTC</b> (London Open).")
        self.notify("\n".join(lines))
        return selected

    def execute_morning_entries(self, candidates: List[Dict[str, Any]]):
        """Executes Market Buy orders on all eligible candidates at 08:00 UTC."""
        sl_pct = self.cfg.get("stop_loss_pct", 8.0) / 100.0
        mode_str = "🧪 DRY RUN" if self.cfg.get("dry_run", True) else "⚡ LIVE"
        executed = []

        for cand in candidates:
            if not cand.get("eligible", False):
                continue

            sym = cand["symbol"]
            curr_p = self.client.get_current_price(sym) or cand["current_price"]
            sl_price = curr_p * (1.0 - sl_pct)
            qty = cand["qty"]

            res = self.client.place_market_buy_with_sl(
                symbol=sym,
                qty=qty,
                entry_price=curr_p,
                sl_price=sl_price
            )

            if res.get("status") in ("success", "simulated"):
                self.active_positions[sym] = {
                    "symbol": sym,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "entry_price": curr_p,
                    "qty": qty,
                    "sl_price": sl_price,
                    "risk_usd": cand["risk_usd"],
                    "orderId": res.get("orderId", "")
                }
                executed.append(cand)

        self.save_state()

        if executed:
            lines = [
                f"🚀 <b>LONDON OPEN EXECUTION [{mode_str}]</b>",
                f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>\n",
                f"Entered <b>{len(executed)}</b> position(s) with native -{sl_pct*100:.1f}% stop loss:\n"
            ]
            for e in executed:
                p = e["current_price"]
                sl = p * (1.0 - sl_pct)
                notional = e["qty"] * p
                lines.append(
                    f"🔹 <b>{e['symbol']}</b>: Bought {e['qty']} @ ${p:.4f}\n"
                    f"   • Notional: ${notional:.2f} | Risk: <b>${e['risk_usd']:.2f}</b>\n"
                    f"   • Native Stop Loss: ${sl:.4f} (-{sl_pct*100:.1f}%)\n"
                )
            lines.append("Positions will be held until <b>20:00 UTC</b> (NY Cash Close).")
            self.notify("\n".join(lines))

    def close_evening_positions(self):
        """Closes all active Setup 11 positions at 20:00 UTC and computes P&L."""
        if not self.active_positions:
            logger.info("20:00 UTC reached: No active positions to close.")
            return

        mode_str = "🧪 DRY RUN" if self.cfg.get("dry_run", True) else "⚡ LIVE"
        lines = [
            f"🏁 <b>NY CLOSE TIME EXIT [{mode_str}]</b>",
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>\n",
            f"Closing <b>{len(self.active_positions)}</b> active position(s):\n"
        ]

        total_pnl_usd = 0.0

        for sym, pos in list(self.active_positions.items()):
            qty = pos.get("qty", 0.0)
            entry_p = pos.get("entry_price", 0.0)
            curr_p = self.client.get_current_price(sym) or entry_p

            res = self.client.close_position_market(sym, qty)

            # Compute gross P&L minus realistic 10 bps fee
            raw_pnl = (curr_p - entry_p) * qty
            fee = (curr_p + entry_p) * qty * 0.0010
            net_pnl = raw_pnl - fee
            pct_ret = ((curr_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0
            total_pnl_usd += net_pnl

            icon = "🟢" if net_pnl >= 0 else "🔴"
            lines.append(
                f"{icon} <b>{sym}</b>: Sold {qty} @ ${curr_p:.4f} (Entry: ${entry_p:.4f})\n"
                f"   • Return: <b>{pct_ret:+.2f}%</b> | Net P&L: <b>${net_pnl:+.2f} USD</b>\n"
            )

        lines.append(f"━━━━━━━━━━━━━━━━━━")
        total_icon = "🎉" if total_pnl_usd >= 0 else "📉"
        lines.append(f"{total_icon} <b>Total Day Net P&L: ${total_pnl_usd:+.2f} USD</b>")

        self.active_positions.clear()
        self.save_state()
        self.notify("\n".join(lines))

    def run_daemon(self):
        """Main 24/7 background loop."""
        mode_str = "DRY RUN (Paper Trading)" if self.cfg.get("dry_run", True) else "LIVE CAPITAL"
        self.notify(f"🤖 <b>WEEX SETUP 11 BOT STARTED</b>\nMode: <b>{mode_str}</b>\nMax Risk: <b>${self.cfg.get('max_risk_usd', 1.0):.2f}/trade</b>")

        last_scan_day = None
        last_entry_day = None
        last_exit_day = None
        staged_candidates = []

        scan_t = self.cfg.get("scan_time_utc", "07:55")
        entry_t = self.cfg.get("entry_time_utc", "08:00")
        exit_t = self.cfg.get("exit_time_utc", "20:00")

        while True:
            now_utc = datetime.now(timezone.utc)
            curr_day = now_utc.date()
            curr_time_str = now_utc.strftime("%H:%M")

            # 1. 07:55 UTC: Morning Scan
            if curr_time_str == scan_t and last_scan_day != curr_day:
                staged_candidates = self.scan_morning_candidates()
                last_scan_day = curr_day

            # 2. 08:00 UTC: Morning Entry
            if curr_time_str == entry_t and last_entry_day != curr_day:
                if staged_candidates:
                    self.execute_morning_entries(staged_candidates)
                    staged_candidates = []
                last_entry_day = curr_day

            # 3. 20:00 UTC: NY Close Exit
            if curr_time_str == exit_t and last_exit_day != curr_day:
                self.close_evening_positions()
                last_exit_day = curr_day

            # Sleep 15 seconds before checking next minute
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description="WEEX Setup 11 Automated Bot")
    parser.add_argument("--scan-now", action="store_true", help="Run morning scan immediately")
    parser.add_argument("--dry-run-now", action="store_true", help="Execute immediate dry-run scan and entry")
    parser.add_argument("--close-now", action="store_true", help="Close all open active positions immediately")
    args = parser.parse_args()

    cfg = load_config()
    bot = LondonDipBot(cfg)

    if args.scan_now:
        cands = bot.scan_morning_candidates()
        print(f"\nDiscovered {len(cands)} candidates.")
    elif args.dry_run_now:
        cands = bot.scan_morning_candidates()
        if cands:
            bot.execute_morning_entries(cands)
    elif args.close_now:
        bot.close_evening_positions()
    else:
        # Default: 24/7 background daemon
        bot.run_daemon()


if __name__ == "__main__":
    main()
