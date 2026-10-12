"""
bot/weex_ob_sweep_bot.py
------------------------
Production Automated SMC/ICT Order Block Liquidity Sweep Trading Bot for WEEX.
Targets: Top 100 Crypto Contracts by Volume (WEEX Contract V3 API).

Core Strategy Logic:
1. Higher Timeframe (4h Default): Detects confirmed Order Blocks (last down-close before BOS breakout).
2. Lower Timeframe (1h Default): Detects Liquidity Sweeps (penetrates OB by >= 0.05% and closes back inside).
3. Risk Management & Execution:
   - Dynamic Precision Engine: Formats quantity and prices strictly to WEEX exchangeInfo specs
     (contractVal, minOrderSize, quantityPrecision, pricePrecision).
   - Stop Loss: 1x ATR(14) below/above the sweep low/high.
   - Take Profit: Structural range target (equal highs/lows) or up to 2.5R.
   - Minimum R:R Filter: Skips setups where natural target offers < 1.0R.
   - BTC Macro Regime Filter: Longs only when BTC >= 1h 50 EMA, shorts only when BTC < 1h 50 EMA.
   - Native Exchange Brackets: Submits preset SL & TP directly to WEEX matching engine.
4. Operational Features:
   - Dynamic Top 100 Universe Scanner via WEEX V3 Contract API (/capi/v3/market/ticker/24hr).
   - Real-time V3 K-line fetching with 4h and 1h intervals.
   - Single Position per Coin Guard (prevents re-entering the same coin).
   - Hourly synchronization loop (:00 UTC).
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
        "top_universe_count": 250,
        "risk_per_trade_usd": 100.0,
        "account_size_usd": 10000.0,
        "max_concurrent_positions": 5,
        "htf_timeframe": "4h",
        "ltf_timeframe": "1h",
        "risk_reward_ratio": 2.5,
        "min_risk_reward_ratio": 1.0,
        "atr_multiplier": 1.0,
        "max_ob_age_hours": 72.0,
        "min_sweep_pct": 0.05,
        "btc_regime_filter": True,
        "btc_symbol": "BTCUSDT",
        "btc_ema_span": 50,
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "poll_interval_seconds": 3600
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
        self.simulated_positions: set = set()
        self.exchange_info: Dict[str, Dict[str, Any]] = {}
        self.refresh_exchange_info()

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

    def refresh_exchange_info(self):
        """Fetches and caches symbol trading rules (precisions, lot sizes, min quantities)."""
        url = f"{self.base_url}/capi/v3/market/exchangeInfo"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                symbols = r.json().get("symbols", [])
                meta = {}
                for s in symbols:
                    sym = s.get("symbol", "")
                    meta[sym] = {
                        "pricePrecision": int(s.get("pricePrecision", 4)),
                        "quantityPrecision": int(s.get("quantityPrecision", 2)),
                        "contractVal": float(s.get("contractVal", 1.0)),
                        "minOrderSize": float(s.get("minOrderSize", 1.0)),
                        "maxOrderSize": float(s.get("maxOrderSize", 100000000.0)),
                    }
                self.exchange_info = meta
                logger.info(f"Loaded exchange trading specifications for {len(self.exchange_info)} symbols.")
        except Exception as e:
            logger.warning(f"Failed to fetch exchangeInfo: {e}")

    def format_order_values(
        self, symbol: str, raw_qty: float, entry_p: float, sl_p: float, tp_p: float
    ) -> Tuple[str, str, str, str]:
        """
        Formats quantity and price strictly to WEEX contract specifications:
        - Quantities are rounded down to nearest contractVal / minOrderSize lot multiple.
        - Prices are formatted to the exact pricePrecision decimals.
        """
        info = self.exchange_info.get(symbol, {
            "pricePrecision": 4,
            "quantityPrecision": 2,
            "contractVal": 1.0,
            "minOrderSize": 0.0001
        })

        c_val = info.get("contractVal", 1.0)
        min_size = info.get("minOrderSize", 0.0001)
        step = c_val if c_val > 0 else min_size

        if step > 0:
            lots = int(raw_qty / step)
            qty = lots * step
        else:
            qty = raw_qty

        if qty < min_size:
            formatted_qty = "0"
        elif info.get("quantityPrecision", 2) > 0:
            formatted_qty = f"{qty:.{info['quantityPrecision']}f}"
        else:
            formatted_qty = str(int(qty))

        p_prec = info.get("pricePrecision", 4)
        formatted_entry = f"{entry_p:.{p_prec}f}"
        formatted_sl = f"{sl_p:.{p_prec}f}"
        formatted_tp = f"{tp_p:.{p_prec}f}"

        return formatted_qty, formatted_entry, formatted_sl, formatted_tp

    def get_account_balance(self) -> Dict[str, Any]:
        """Fetches account USDT available margin and equity."""
        if not self.api_key:
            return {}
        path = "/capi/v3/account/balance"
        try:
            headers = self._get_headers("GET", path)
            r = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            if r.status_code == 200:
                res = r.json()
                return res.get("data", {}) if isinstance(res, dict) else {}
        except Exception as e:
            logger.warning(f"Could not fetch account balance: {e}")
        return {}

    def get_top_contracts_by_volume(self, top_n: int = 100) -> List[str]:
        """Fetches all active USDT perpetuals and ranks by 24h quote turnover via V3 API."""
        url = f"{self.base_url}/capi/v3/market/ticker/24hr"
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                logger.error(f"Failed to fetch tickers: {r.status_code} {r.text}")
                return []
            tickers = r.json()
            if not isinstance(tickers, list):
                logger.error(f"Unexpected ticker response format: {tickers}")
                return []
            usdt_tickers = []
            for t in tickers:
                sym = t.get("symbol", "")
                if sym.endswith("USDT") or sym.endswith("_USDT"):
                    vol = float(t.get("quoteVolume", 0.0) or t.get("volume", 0.0) or 0.0)
                    usdt_tickers.append((sym, vol))

            usdt_tickers.sort(key=lambda x: x[1], reverse=True)
            top_syms = [x[0] for x in usdt_tickers[:top_n]]
            logger.info(f"Retrieved Top {len(top_syms)} WEEX symbols by volume.")
            return top_syms
        except Exception as e:
            logger.error(f"Error getting top contracts: {e}")
            return []

    def get_klines(self, symbol: str, granularity: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches historical klines for analysis via V3 API."""
        url = f"{self.base_url}/capi/v3/market/klines"
        gran_map = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
        params = {
            "symbol": symbol,
            "interval": gran_map.get(granularity, "1h"),
            "limit": limit
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            if not isinstance(data, list) or len(data) < 20:
                return None
            # Format: [openTime, open, high, low, close, volume, ...]
            rows = []
            for d in data:
                rows.append([d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])])
            df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
            df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
            df = df.sort_values("open_time").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches list of currently open positions from exchange."""
        if not self.api_key:
            return []
        path = "/capi/v3/account/position/allPosition"
        try:
            headers = self._get_headers("GET", path)
            r = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            if r.status_code == 200:
                res_data = r.json()
                pos_list = res_data.get("data", []) if isinstance(res_data, dict) else res_data
                active = [p for p in pos_list if float(p.get("total", 0.0) or p.get("size", 0.0) or 0.0) > 0]
                return active
            return []
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")
            return []

    def get_open_position_symbols(self) -> set:
        """Returns set of symbols that currently have an open position."""
        if self.dry_run:
            return set(self.simulated_positions)
        positions = self.get_open_positions()
        return {p.get("symbol", "") for p in positions if p.get("symbol")}

    def get_open_positions_count(self) -> int:
        """Fetches currently open positions count."""
        return len(self.get_open_position_symbols())

    def place_native_bracket_order(
        self,
        symbol: str,
        side: str,
        position_side: str,
        raw_quantity: float = 0.0,
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submits market order with native preset TP & SL attached directly to WEEX engine.
        Quantity and prices are formatted to match symbol exchange rules.
        """
        if quantity is not None and raw_quantity == 0.0:
            raw_quantity = quantity

        qty_str, entry_str, sl_str, tp_str = self.format_order_values(
            symbol, raw_quantity, entry_price, sl_price, tp_price
        )

        if qty_str == "0":
            err_msg = f"Quantity {raw_quantity:.4f} is below minimum allowed size for {symbol}"
            logger.error(f"❌ Order Aborted: {err_msg}")
            return {"status": "error", "error": {"msg": err_msg, "code": "MIN_SIZE_VIOLATION"}}

        if self.dry_run:
            self.simulated_positions.add(symbol)
            logger.info(f"[DRY-RUN SIMULATION] {side} {position_side} {qty_str} {symbol} | Entry: {entry_str} | SL: {sl_str} | TP: {tp_str}")
            return {
                "status": "simulated",
                "orderId": f"sim_{int(time.time()*1000)}",
                "formatted_qty": qty_str,
                "formatted_sl": sl_str,
                "formatted_tp": tp_str
            }

        path = "/capi/v3/order"
        payload = {
            "symbol": symbol,
            "side": side.upper(),                 # "BUY" or "SELL"
            "positionSide": position_side.upper(), # "LONG" or "SHORT"
            "type": "MARKET",
            "quantity": qty_str,
            "slTriggerPrice": sl_str,
            "tpTriggerPrice": tp_str,
            "SlWorkingType": "MARK_PRICE",
            "TpWorkingType": "MARK_PRICE",
            "newClientOrderId": f"ob_{int(time.time()*1000)}"
        }
        body_str = json.dumps(payload)
        headers = self._get_headers("POST", path, body=body_str)

        logger.info(f"Submitting WEEX V3 Contract Order: {payload}")

        try:
            r = requests.post(f"{self.base_url}{path}", headers=headers, data=body_str, timeout=12)
            res = r.json()
            if r.status_code == 200 and res.get("code") == "00000":
                logger.info(f"✅ Live WEEX Order Placed Successfully: {symbol} {side} | Response: {res}")
                return {
                    "status": "success",
                    "orderId": res.get("data", {}).get("orderId", "N/A"),
                    "data": res.get("data"),
                    "formatted_qty": qty_str,
                    "formatted_sl": sl_str,
                    "formatted_tp": tp_str
                }
            # Auto-retry with positionSide="BOTH" if account is in One-Way Mode (triggers -1191)
            elif str(res.get("code")) in ("-1191", "1191") and position_side.upper() in ("LONG", "SHORT"):
                logger.warning(f"Got -1191 for {symbol} with positionSide={position_side}. Retrying with positionSide=BOTH (One-Way Mode)...")
                payload_both = dict(payload)
                payload_both["positionSide"] = "BOTH"
                body_both = json.dumps(payload_both)
                headers_both = self._get_headers("POST", path, body=body_both)
                r_both = requests.post(f"{self.base_url}{path}", headers=headers_both, data=body_both, timeout=12)
                res_both = r_both.json()
                if r_both.status_code == 200 and res_both.get("code") == "00000":
                    logger.info(f"✅ Live WEEX Order Placed with positionSide=BOTH: {symbol} | Response: {res_both}")
                    return {
                        "status": "success",
                        "orderId": res_both.get("data", {}).get("orderId", "N/A"),
                        "data": res_both.get("data"),
                        "formatted_qty": qty_str,
                        "formatted_sl": sl_str,
                        "formatted_tp": tp_str
                    }
                else:
                    logger.error(f"❌ WEEX Order Rejected with positionSide=BOTH: {res_both}")
                    return {
                        "status": "error",
                        "error": res_both,
                        "formatted_qty": qty_str,
                        "formatted_sl": sl_str,
                        "formatted_tp": tp_str
                    }
            else:
                logger.error(f"❌ WEEX Order Rejected by Exchange: HTTP {r.status_code} | Body: {res}")
                return {
                    "status": "error",
                    "error": res,
                    "formatted_qty": qty_str,
                    "formatted_sl": sl_str,
                    "formatted_tp": tp_str
                }
        except Exception as e:
            logger.error(f"Exception submitting WEEX order: {e}")
            return {
                "status": "error",
                "error": str(e),
                "formatted_qty": qty_str,
                "formatted_sl": sl_str,
                "formatted_tp": tp_str
            }


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
    df_htf: pd.DataFrame,
    df_ltf: pd.DataFrame,
    symbol: str,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether the symbol currently has a fresh Lower Timeframe Liquidity Sweep
    of a confirmed Higher Timeframe Order Block.
    Checks all valid active order blocks formed within max_ob_age_hours.
    """
    if len(df_htf) < 20 or len(df_ltf) < 20:
        return None

    # Exclude open forming candle; analyze fully closed historical candles
    closed_htf = df_htf.iloc[:-1].copy()
    closed_ltf = df_ltf.iloc[:-1].copy()

    h_htf = closed_htf["high"].values
    l_htf = closed_htf["low"].values
    o_htf = closed_htf["open"].values
    c_htf = closed_htf["close"].values
    times_htf = closed_htf["open_time"].tolist()

    sh_htf, sl_htf = find_fractal_swings(h_htf, l_htf, n=2)

    # 1. Identify Confirmed HTF Order Blocks
    obs = []
    last_sh_broken = np.nan
    last_sl_broken = np.nan

    for i in range(5, len(closed_htf)):
        bar_t = times_htf[i]
        curr_sh = sh_htf[i]
        curr_sl = sl_htf[i]

        # Bullish BOS Breakout
        if not np.isnan(curr_sh) and c_htf[i] > curr_sh and curr_sh != last_sh_broken:
            last_sh_broken = curr_sh
            for j in range(i - 1, max(0, i - 12), -1):
                if c_htf[j] < o_htf[j]:
                    obs.append({
                        "ob_type": "bullish",
                        "creation_time": bar_t,
                        "ob_high": h_htf[j],
                        "ob_low": l_htf[j],
                        "equal_high": curr_sh
                    })
                    break

        # Bearish BOS Breakdown
        if not np.isnan(curr_sl) and c_htf[i] < curr_sl and curr_sl != last_sl_broken:
            last_sl_broken = curr_sl
            for j in range(i - 1, max(0, i - 12), -1):
                if c_htf[j] > o_htf[j]:
                    obs.append({
                        "ob_type": "bearish",
                        "creation_time": bar_t,
                        "ob_high": h_htf[j],
                        "ob_low": l_htf[j],
                        "equal_low": curr_sl
                    })
                    break

    if not obs:
        return None

    now_utc = closed_ltf["open_time"].iloc[-1]
    max_age = config.get("max_ob_age_hours", 72.0)

    # Filter OBs within acceptable age
    active_obs = [
        ob for ob in obs
        if (now_utc - ob["creation_time"]).total_seconds() / 3600.0 <= max_age
    ]

    if not active_obs:
        return None

    # Calculate LTF ATR(14)
    h_ltf = closed_ltf["high"].values
    l_ltf = closed_ltf["low"].values
    c_ltf = closed_ltf["close"].values
    prev_c = np.roll(c_ltf, 1)
    prev_c[0] = c_ltf[0]
    tr = np.maximum(h_ltf - l_ltf, np.maximum(np.abs(h_ltf - prev_c), np.abs(l_ltf - prev_c)))
    atr14 = float(pd.Series(tr).rolling(14, min_periods=1).mean().iloc[-1])

    latest_ltf = closed_ltf.iloc[-1]
    last_low = float(latest_ltf["low"])
    last_high = float(latest_ltf["high"])
    last_close = float(latest_ltf["close"])
    sweep_pct = config.get("min_sweep_pct", 0.05) / 100.0

    # Check active OBs in reverse order (most recent first)
    for ob in reversed(active_obs):
        # Bullish Liquidity Sweep Check
        if ob["ob_type"] == "bullish":
            sweep_threshold = ob["ob_low"] * (1.0 - sweep_pct)
            if last_low <= sweep_threshold and last_close >= ob["ob_low"]:
                entry_p = last_close
                sl_p = last_low - (config.get("atr_multiplier", 1.0) * atr14)
                risk = entry_p - sl_p
                if risk > 0 and (risk / entry_p) > 0.002:
                    tp_p = entry_p + (config.get("risk_reward_ratio", 2.5) * risk)
                    if ob.get("equal_high", np.nan) > entry_p:
                        tp_p = min(tp_p, ob["equal_high"])
                    reward = tp_p - entry_p
                    effective_rr = reward / risk if risk > 0 else 0.0
                    min_rr = config.get("min_risk_reward_ratio", 1.0)
                    if effective_rr < min_rr:
                        continue  # Skip setup: natural structural target offers less than minimum R:R
                    return {
                        "symbol": symbol,
                        "direction": "long",
                        "side": "BUY",
                        "position_side": "LONG",
                        "entry_price": entry_p,
                        "sl_price": sl_p,
                        "tp_price": tp_p,
                        "risk_usd_dist": risk,
                        "effective_rr": round(effective_rr, 2),
                        "atr14": atr14,
                        "ob_level": ob["ob_low"],
                        "ob_type": "bullish"
                    }

        # Bearish Liquidity Sweep Check
        elif ob["ob_type"] == "bearish":
            sweep_threshold = ob["ob_high"] * (1.0 + sweep_pct)
            if last_high >= sweep_threshold and last_close <= ob["ob_high"]:
                entry_p = last_close
                sl_p = last_high + (config.get("atr_multiplier", 1.0) * atr14)
                risk = sl_p - entry_p
                if risk > 0 and (risk / entry_p) > 0.002:
                    tp_p = entry_p - (config.get("risk_reward_ratio", 2.5) * risk)
                    if ob.get("equal_low", np.nan) < entry_p:
                        tp_p = max(tp_p, ob["equal_low"])
                    reward = entry_p - tp_p
                    effective_rr = reward / risk if risk > 0 else 0.0
                    min_rr = config.get("min_risk_reward_ratio", 1.0)
                    if effective_rr < min_rr:
                        continue  # Skip setup: natural structural target offers less than minimum R:R
                    return {
                        "symbol": symbol,
                        "direction": "short",
                        "side": "SELL",
                        "position_side": "SHORT",
                        "entry_price": entry_p,
                        "sl_price": sl_p,
                        "tp_price": tp_p,
                        "risk_usd_dist": risk,
                        "effective_rr": round(effective_rr, 2),
                        "atr14": atr14,
                        "ob_level": ob["ob_high"],
                        "ob_type": "bearish"
                    }

    return None


def evaluate_btc_macro_regime(
    df_btc_1h: Optional[pd.DataFrame],
    ema_span: int = 50
) -> Dict[str, Any]:
    """
    Evaluates BTC macro market trend on 1h timeframe using strictly closed candles.
    Rule 1 (No lookahead): Excludes the active, currently forming 1h candle.
    Information availability: uses df_btc_1h.iloc[:-1] only.

    Returns:
        dict with:
            - regime: "BULLISH" | "BEARISH" | "UNKNOWN"
            - allowed_direction: "long" | "short" | "both"
            - btc_close: latest fully closed 1h price
            - btc_ema: latest 1h EMA value
            - is_bullish: bool
            - bar_time: timestamp of latest closed candle
    """
    if df_btc_1h is None or len(df_btc_1h) < max(10, ema_span // 2):
        return {
            "regime": "UNKNOWN",
            "allowed_direction": "both",
            "btc_close": 0.0,
            "btc_ema": 0.0,
            "is_bullish": False,
            "bar_time": None
        }

    # Strict lookahead prevention: exclude active forming candle
    closed_1h = df_btc_1h.iloc[:-1].copy() if len(df_btc_1h) > 1 else df_btc_1h.copy()
    if closed_1h.empty or len(closed_1h) < 5:
        return {
            "regime": "UNKNOWN",
            "allowed_direction": "both",
            "btc_close": 0.0,
            "btc_ema": 0.0,
            "is_bullish": False,
            "bar_time": None
        }

    s_close = closed_1h["close"].astype(float)
    ema = s_close.ewm(span=ema_span, adjust=False).mean()
    latest_close = float(s_close.iloc[-1])
    latest_ema = float(ema.iloc[-1])
    is_bullish = latest_close >= latest_ema

    regime = "BULLISH" if is_bullish else "BEARISH"
    allowed_dir = "long" if is_bullish else "short"

    return {
        "regime": regime,
        "allowed_direction": allowed_dir,
        "btc_close": latest_close,
        "btc_ema": latest_ema,
        "is_bullish": is_bullish,
        "bar_time": closed_1h["open_time"].iloc[-1]
    }


# -----------------------------------------------------------------------------
# MAIN BOT RUNNER & DIAGNOSTICS
# -----------------------------------------------------------------------------

def run_diagnostics(client: WeexClient, config: Dict[str, Any]):
    """Executes a full interactive health check and market status report."""
    print("\n" + "=" * 70)
    print("🔍 WEEX ORDER BLOCK SWEEP BOT — DIAGNOSTIC HEALTH CHECK")
    print("=" * 70)

    # 1. Market Data Connectivity
    print("\n[1/5] Testing WEEX V3 Market Data API...")
    universe = client.get_top_contracts_by_volume(top_n=10)
    if universe:
        print(f"  ✅ SUCCESS: Retrieved Top Volume contracts! Sample: {universe[:5]}")
    else:
        print("  ❌ ERROR: Could not fetch top contracts from WEEX V3.")
        return

    # 2. Kline Data Retrieval & BTC Regime Check
    print("\n[2/5] Testing Historical K-Line Retrieval & BTC Macro Regime...")
    df1h = client.get_klines("BTCUSDT", granularity="1h", limit=100)
    df15m = client.get_klines("BTCUSDT", granularity="15m", limit=50)
    if df1h is not None and df15m is not None:
        print(f"  ✅ SUCCESS: 1h bars={len(df1h)} (Latest: {df1h['open_time'].iloc[-1]} UTC, Close=${df1h['close'].iloc[-1]:,.1f})")
        print(f"  ✅ SUCCESS: 15m bars={len(df15m)} (Latest: {df15m['open_time'].iloc[-1]} UTC, Close=${df15m['close'].iloc[-1]:,.1f})")
        btc_reg = evaluate_btc_macro_regime(df1h, ema_span=config.get("btc_ema_span", 50))
        status_icon = "🟢" if btc_reg["is_bullish"] else "🔴"
        print(f"  🌐 BTC Macro Regime Filter: {status_icon} {btc_reg['regime']} (BTC Close=${btc_reg['btc_close']:,.1f} vs EMA{config.get('btc_ema_span', 50)}=${btc_reg['btc_ema']:,.1f})")
        print(f"     Permitted Execution Direction: {btc_reg['allowed_direction'].upper()} ONLY")
    else:
        print("  ❌ ERROR: Failed to retrieve 1h/15m klines.")

    # 3. Authenticated Credentials & Balance Check
    print("\n[3/5] Testing Private WEEX API Credentials & Account Balance...")
    if not client.api_key or not client.secret_key or not client.passphrase:
        print("  ⚠️ INFO: API credentials not provided (Dry-Run Only mode).")
    else:
        pos_cnt = client.get_open_positions_count()
        bal = client.get_account_balance()
        print(f"  ✅ SUCCESS: Private API authenticated! Active Open Positions: {pos_cnt}")
        if bal:
            avail = float(bal.get("available", 0.0) or bal.get("crossMargin", 0.0) or 0.0)
            equity = float(bal.get("equity", 0.0) or bal.get("total", 0.0) or 0.0)
            print(f"  💰 Account Balance: Available Margin=${avail:,.2f} | Total Equity=${equity:,.2f}")

    # 4. Telegram Alert Test
    print("\n[4/5] Testing Telegram Notification Connection...")
    if config["telegram_bot_token"] and config["telegram_chat_id"]:
        test_msg = "🔔 <b>WEEX Bot Diagnostics Ping</b>: Communication channel verified and active."
        ok = send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], test_msg)
        if ok:
            print("  ✅ SUCCESS: Test alert delivered to Telegram!")
        else:
            print("  ❌ WARNING: Telegram message failed. Check bot token and chat ID.")
    else:
        print("  ⚠️ INFO: Telegram not configured in config. Alerts will print to terminal/log only.")

    # 5. Universe Scan Preview
    htf = config.get("htf_timeframe", "4h")
    ltf = config.get("ltf_timeframe", "1h")
    print(f"\n[5/5] Scanning Top {config['top_universe_count']} Volume Universe for Live {htf.upper()}/{ltf.upper()} Setups...")
    full_universe = client.get_top_contracts_by_volume(top_n=config["top_universe_count"])
    signals_found = 0
    scanned_count = 0

    for sym in full_universe:
        d_htf = client.get_klines(sym, granularity=htf, limit=100)
        d_ltf = client.get_klines(sym, granularity=ltf, limit=100)
        if d_htf is None or d_ltf is None:
            continue
        scanned_count += 1
        sig = evaluate_symbol_for_sweep_setup(d_htf, d_ltf, sym, config)
        if sig:
            signals_found += 1
            print(f"  ⚡ ACTIVE SWEEP FOUND: {sym} ({sig['direction'].upper()}) | Entry: ${sig['entry_price']:.4f} | SL: ${sig['sl_price']:.4f} | TP: ${sig['tp_price']:.4f} (R:R={sig.get('effective_rr', 0.0):.2f})")

    print(f"\nScan Completed: Analyzed {scanned_count} symbols. Found {signals_found} active sweep signals right now.")
    print("=" * 70 + "\n")


def run_scan_and_execute(client: WeexClient, config: Dict[str, Any], symbols: List[str]):
    """Scans all symbols in the universe and executes confirmed sweeps without duplicate coin positions."""
    active_symbols = client.get_open_position_symbols()
    active_positions = len(active_symbols)
    max_pos = config["max_concurrent_positions"]

    if active_symbols:
        logger.info(f"Existing active position(s) on {len(active_symbols)} coin(s): {sorted(list(active_symbols))}")

    if active_positions >= max_pos:
        logger.info(f"Max concurrent positions reached ({active_positions}/{max_pos}). Skipping scan cycle.")
        return

    # 1. Evaluate BTC Macro Regime Filter
    btc_regime = {"regime": "UNKNOWN", "allowed_direction": "both", "is_bullish": False, "btc_close": 0.0, "btc_ema": 0.0}
    if config.get("btc_regime_filter", True):
        btc_sym = config.get("btc_symbol", "BTCUSDT")
        df_btc = client.get_klines(btc_sym, granularity="1h", limit=100)
        btc_regime = evaluate_btc_macro_regime(df_btc, ema_span=config.get("btc_ema_span", 50))
        if btc_regime["regime"] != "UNKNOWN":
            logger.info(
                f"🌐 BTC Macro Regime: {btc_regime['regime']} "
                f"(BTC ${btc_regime['btc_close']:,.1f} vs EMA{config.get('btc_ema_span', 50)} ${btc_regime['btc_ema']:,.1f}) "
                f"| Permitted Direction: {btc_regime['allowed_direction'].upper()} ONLY"
            )
        else:
            logger.warning("Could not resolve BTC Macro Regime; proceeding without regime gate.")

    htf = config.get("htf_timeframe", "4h")
    ltf = config.get("ltf_timeframe", "1h")
    logger.info(f"Scanning {len(symbols)} Top Volume symbols for {ltf.upper()} Sweeps of {htf.upper()} Order Blocks...")
    confirmed_signals = []

    for sym in symbols:
        # Never enter or scan a second position on the same coin
        if sym in active_symbols:
            continue

        df_htf = client.get_klines(sym, granularity=htf, limit=100)
        df_ltf = client.get_klines(sym, granularity=ltf, limit=100)
        if df_htf is None or df_ltf is None:
            continue

        sig = evaluate_symbol_for_sweep_setup(df_htf, df_ltf, sym, config)
        if sig:
            # Apply BTC Macro Regime gate
            if btc_regime["regime"] != "UNKNOWN" and sig["direction"] != btc_regime["allowed_direction"]:
                logger.info(
                    f"🛡️ Filtered {sym} ({sig['direction'].upper()}): Conflict with BTC Macro Regime "
                    f"({btc_regime['regime']} permits {btc_regime['allowed_direction'].upper()} only)"
                )
                continue
            confirmed_signals.append(sig)

    logger.info(f"Scan finished. Found {len(confirmed_signals)} regime-compliant sweep signals.")

    regime_desc = (
        f"• Market Regime: {'🟢 BULLISH' if btc_regime.get('is_bullish') else '🔴 BEARISH'} "
        f"(BTC ${btc_regime.get('btc_close', 0):,.1f} vs EMA{config.get('btc_ema_span', 50)})\n"
    ) if btc_regime.get("regime") != "UNKNOWN" else ""

    for sig in confirmed_signals:
        sym = sig["symbol"]
        # Double check to prevent duplicate position on the same coin
        if sym in active_symbols:
            logger.info(f"Skipping {sym}: Position already active on this coin.")
            continue

        if len(active_symbols) >= max_pos:
            logger.info(f"Maximum positions filled ({len(active_symbols)}/{max_pos}). Halting execution.")
            break

        entry = sig["entry_price"]
        sl = sig["sl_price"]
        tp = sig["tp_price"]
        risk_dist = sig["risk_usd_dist"]

        # Calculate position size based on risk_per_trade_usd
        risk_amount = config["risk_per_trade_usd"]
        raw_quantity = risk_amount / risk_dist if risk_dist > 0 else 0.0

        if raw_quantity <= 0:
            continue

        # Place Native Bracket Order on WEEX with lot-precision formatting
        result = client.place_native_bracket_order(
            symbol=sym,
            side=sig["side"],
            position_side=sig["position_side"],
            raw_quantity=raw_quantity,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp
        )

        status = result.get("status")
        formatted_qty = result.get("formatted_qty", str(raw_quantity))
        formatted_sl = result.get("formatted_sl", str(sl))
        formatted_tp = result.get("formatted_tp", str(tp))

        # 1. LIVE ORDER FILLED
        if status == "success":
            active_symbols.add(sym)
            order_id = result.get("orderId", "N/A")
            effective_rr = sig.get("effective_rr", config.get("risk_reward_ratio", 2.5))
            msg = (
                f"⚡ <b>[LIVE ORDER EXECUTED] {htf.upper()}/{ltf.upper()} ORDER BLOCK SWEEP!</b>\n"
                f"• Contract: <code>{sym}</code> ({sig['direction'].upper()})\n"
                f"{regime_desc}"
                f"• Entry: <code>${entry:,.4f}</code>\n"
                f"• <b>Native Exchange SL:</b> <code>${formatted_sl}</code> (-1x ATR)\n"
                f"• <b>Native Exchange TP:</b> <code>${formatted_tp}</code> (+{effective_rr:.1f}R)\n"
                f"• Position Size: <code>{formatted_qty}</code> units (${risk_amount:,.0f} risk)\n"
                f"• Order ID: <code>{order_id}</code>\n"
                f"• Active Positions: <code>{len(active_symbols)}/{max_pos}</code>"
            )
            send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], msg)

        # 2. DRY-RUN SIMULATION
        elif status == "simulated":
            active_symbols.add(sym)
            effective_rr = sig.get("effective_rr", config.get("risk_reward_ratio", 2.5))
            msg = (
                f"🚨 <b>[DRY-RUN SIMULATION] {htf.upper()}/{ltf.upper()} ORDER BLOCK SWEEP!</b>\n"
                f"• Contract: <code>{sym}</code> ({sig['direction'].upper()})\n"
                f"{regime_desc}"
                f"• Entry: <code>${entry:,.4f}</code>\n"
                f"• <b>Simulated SL:</b> <code>${formatted_sl}</code> (-1x ATR)\n"
                f"• <b>Simulated TP:</b> <code>${formatted_tp}</code> (+{effective_rr:.1f}R)\n"
                f"• Position Size: <code>{formatted_qty}</code> units (${risk_amount:,.0f} risk)\n"
                f"• Active Positions: <code>{len(active_symbols)}/{max_pos}</code>"
            )
            send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], msg)

        # 3. ORDER REJECTED BY EXCHANGE
        else:
            err = result.get("error", "Unknown error")
            err_msg = err.get("msg", str(err)) if isinstance(err, dict) else str(err)
            err_code = err.get("code", "") if isinstance(err, dict) else ""
            logger.error(f"Order rejected for {sym}: Code {err_code} - {err_msg}")
            fail_msg = (
                f"⚠️ <b>[WEEX ORDER REJECTED] {sym} ({sig['direction'].upper()})</b>\n"
                f"• Error Code: <code>{err_code}</code>\n"
                f"• Reason: <code>{err_msg}</code>\n"
                f"• Attempted Size: <code>{formatted_qty}</code> units\n"
                f"• SL: <code>${formatted_sl}</code> | TP: <code>${formatted_tp}</code>\n"
                f"• <i>No position opened on exchange.</i>"
            )
            send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], fail_msg)

        time.sleep(0.5)


def start_bot_daemon():
    """Runs the 15-minute continuous scheduler."""
    config = load_config()
    client = WeexClient(
        api_key=config["api_key"],
        secret_key=config["secret_key"],
        passphrase=config["passphrase"],
        dry_run=config["dry_run"]
    )

    if "--diagnostics" in sys.argv or "--test" in sys.argv:
        run_diagnostics(client, config)
        return

    logger.info("Starting WEEX Order Block Liquidity Sweep Bot Daemon...")
    logger.info(f"Bot Mode: {'DRY RUN (Simulated)' if config['dry_run'] else 'LIVE TRADING'}")
    logger.info(f"Target Universe: Top {config['top_universe_count']} Volume Coins")

    # Initial Universe Discovery
    universe = client.get_top_contracts_by_volume(top_n=config["top_universe_count"])
    last_universe_refresh = time.time()

    # Send Startup Message
    regime_mode_desc = f"1h {config.get('btc_ema_span', 50)} EMA" if config.get("btc_regime_filter", True) else "DISABLED"
    htf = config.get("htf_timeframe", "4h")
    ltf = config.get("ltf_timeframe", "1h")
    startup_msg = (
        f"🤖 <b>WEEX SMC ORDER BLOCK BOT INITIALIZED</b>\n"
        f"• Strategy: <b>{htf.upper()} Structure + {ltf.upper()} Sweeps</b>\n"
        f"• Mode: <code>{'DRY-RUN SIMULATION' if config['dry_run'] else 'LIVE EXECUTION'}</code>\n"
        f"• Universe: <b>Top {len(universe)} Volume Coins</b>\n"
        f"• BTC Macro Regime Filter: <b>{regime_mode_desc}</b>\n"
        f"• Minimum Target Filter: <b>{config.get('min_risk_reward_ratio', 1.0):.1f}R Floor</b>\n"
        f"• Risk per Trade: <code>${config['risk_per_trade_usd']:,.0f}</code>\n"
        f"• Native Bracket Target: <b>Up to {config.get('risk_reward_ratio', 2.5)}R</b> | Stop: <b>1x ATR(14)</b>\n"
        f"• Cadence: <b>Hourly at Candle Close (:00 UTC)</b>"
    )
    send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], startup_msg)

    while True:
        try:
            # Refresh universe & trading specifications daily
            if time.time() - last_universe_refresh > 86400:
                universe = client.get_top_contracts_by_volume(top_n=config["top_universe_count"])
                client.refresh_exchange_info()
                last_universe_refresh = time.time()

            # Execute Scan Cycle
            run_scan_and_execute(client, config, universe)

            # Sleep until next candle close + 3 seconds buffer
            now = time.time()
            interval = config.get("poll_interval_seconds", 3600)
            remainder = now % interval
            sleep_time = (interval - remainder) + 3.0
            logger.info(f"Sleeping {sleep_time:.1f}s until next {ltf} candle close...")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in bot execution loop: {e}")
            time.sleep(15)


if __name__ == "__main__":
    start_bot_daemon()
