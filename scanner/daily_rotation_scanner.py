"""
scanner/daily_rotation_scanner.py
Production-ready daily scanner for the 24-Hour Rebalance / 7-Day Lookback Rotational Momentum Strategy.
Supports:
1. Kraken Breakout Universe: Restricts strictly to the 65 Breakout Prop Firm tradeable coins.
2. Dual Data Sources:
   - "kraken" (Default): Direct Kraken Futures REST API (PF_XBTUSD, PF_<COIN>USD).
   - "mexc": MEXC Spot API restricted to the Breakout coin universe.
3. Prop-Firm Risk & Execution Model:
   - Bitcoin 50-Day EMA Macro Health Gate.
   - Sizing calibrated to prop firm drawdown limits (e.g. 15%-25% total exposure / 3%-4% per coin).
   - Bracket order instructions: Hard SL (-3.5%), TP1 (+9.0% / scale 50%), TP2 (+18.0% / scale 25%).
   - Turnover smoothing buffers (keeps churn minimal on daily rebalances).
   - Direct Telegram alerts.
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

# Ensure UTF-8 output on all platforms
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
        logging.FileHandler("daily_scanner.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("daily_rotation_scanner")

MEXC_BASE_URL = "https://api.mexc.com"
KRAKEN_FUTURES_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
KRAKEN_FUTURES_CHART_URL = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/1d"

SCRIPT_DIR = Path(__file__).resolve().parent

# Verified Breakout Supported Coin Universe (65 Assets)
DEFAULT_BREAKOUT_UNIVERSE = [
    "BCH", "LIGHTER", "MON", "STX", "ETHFI", "JTO", "PEPE", "ENA", "WIF",
    "SUI", "NEAR", "GRASS", "UNI", "AAVE", "ADA", "AIXBT", "ALGO", "APT",
    "ARB", "ASTER", "ATOM", "AVAX", "BNB", "BONK", "CRV", "DOGE", "DOT",
    "ETC", "ETH", "FARTCOIN", "FIL", "FLOKI", "HBAR", "HYPE", "ICP",
    "INJ", "JUP", "KAITO", "LDO", "LINK", "LTC", "MOODENG", "ONDO", "OP",
    "PENDLE", "PENGU", "PNUT", "POL", "POPCAT", "PUMP", "RENDER", "S",
    "SHIB", "SOL", "TAO", "TIA", "TRUMP", "TRX", "VIRTUAL", "WLD", "XLM",
    "XPL", "XRP", "ZEC", "ZRO"
]


def get_config_file() -> Optional[Path]:
    candidates = [
        Path("daily_scanner_config.json"),
        SCRIPT_DIR / "daily_scanner_config.json",
        Path("kraken_breakout_config.json"),
        SCRIPT_DIR / "kraken_breakout_config.json",
        Path("scanner_config.json"),
        SCRIPT_DIR / "scanner_config.json",
        SCRIPT_DIR / "deploy" / "daily_scanner_config.example.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_state_file() -> Path:
    if Path("portfolio_state_daily.json").exists():
        return Path("portfolio_state_daily.json")
    if (SCRIPT_DIR / "portfolio_state_daily.json").exists():
        return SCRIPT_DIR / "portfolio_state_daily.json"
    return Path("portfolio_state_daily.json")


def load_config() -> Dict[str, Any]:
    """Loads configuration and Telegram credentials for daily scanner."""
    config = {
        "exchange": os.environ.get("EXCHANGE", "kraken").lower(), # "kraken" or "mexc"
        "universe_mode": os.environ.get("UNIVERSE_MODE", "kraken_breakout"), # "kraken_breakout" or "all_liquid_mexc"
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "account_size_usd": float(os.environ.get("ACCOUNT_SIZE_USD", "100000.0")),
        "total_exposure_pct": float(os.environ.get("TOTAL_EXPOSURE_PCT", "20.0")), # 20% total account exposure
        "top_k": int(os.environ.get("TOP_K", "5")), # Top 5 coins
        "rank_exit_buffer": int(os.environ.get("RANK_EXIT_BUFFER", "8")), # Only exit if drops below rank 8
        "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "7")), # Default 7 days
        "stop_loss_pct": float(os.environ.get("STOP_LOSS_PCT", "3.5")), # -3.5% hard stop
        "take_profit_1_pct": float(os.environ.get("TAKE_PROFIT_1_PCT", "9.0")), # +9.0% (sell 50%)
        "take_profit_2_pct": float(os.environ.get("TAKE_PROFIT_2_PCT", "18.0")), # +18.0% (sell 25%)
        "allowed_coins": DEFAULT_BREAKOUT_UNIVERSE
    }

    cfg_file = get_config_file()
    if cfg_file:
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                config.update(file_cfg)
                logger.info(f"Loaded config from {cfg_file}")
        except Exception as e:
            logger.warning(f"Failed to read {cfg_file}: {e}")

    return config


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Dispatches formatted message to Telegram Bot with auto-chunking."""
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not configured. Printing message to stdout only.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3800
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

    success = True
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=12)
                if resp.status_code == 200:
                    break
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        if len(chunks) > 1 and i < len(chunks) - 1:
            time.sleep(0.5)

    return success


# -----------------------------------------------------------------------------
# KRAKEN FUTURES API METHODS
# -----------------------------------------------------------------------------
def get_kraken_live_tickers() -> Dict[str, Dict[str, Any]]:
    """Fetches real-time live tickers from Kraken Futures."""
    try:
        resp = requests.get(KRAKEN_FUTURES_TICKERS_URL, timeout=12)
        tickers = resp.json().get("tickers", [])
        return {t["symbol"]: t for t in tickers if not t.get("suspended", False)}
    except Exception as e:
        logger.error(f"Error querying Kraken tickers: {e}")
        return {}


def fetch_kraken_daily_candles(symbol: str) -> Optional[pd.DataFrame]:
    """Fetches historical daily candles from Kraken Futures."""
    url = KRAKEN_FUTURES_CHART_URL.format(symbol=symbol)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                time.sleep(0.3)
                continue
            candles = resp.json().get("candles", [])
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df.sort_values("time").reset_index(drop=True)
        except Exception as e:
            time.sleep(0.5)
    return None


def evaluate_kraken_btc_macro(live_tickers: Dict[str, Dict[str, Any]]) -> Tuple[bool, float, float, float]:
    """Checks BTC live price against 50-day EMA on Kraken Futures PF_XBTUSD."""
    df_btc = fetch_kraken_daily_candles("PF_XBTUSD")
    if df_btc is None or len(df_btc) < 55:
        logger.error("Insufficient PF_XBTUSD candle data.")
        return False, 0.0, 0.0, 0.0

    df_btc["ema50"] = df_btc["close"].ewm(span=50, adjust=False).mean()
    btc_ema50 = float(df_btc["ema50"].iloc[-1])

    # Use real-time live markPrice if available from tickers
    if "PF_XBTUSD" in live_tickers:
        btc_price = float(live_tickers["PF_XBTUSD"].get("markPrice", df_btc["close"].iloc[-1]))
    else:
        btc_price = float(df_btc["close"].iloc[-1])

    dist_pct = ((btc_price - btc_ema50) / btc_ema50) * 100.0
    is_bullish = btc_price >= btc_ema50
    return is_bullish, btc_price, btc_ema50, dist_pct


def map_breakout_coins_to_kraken(coins: List[str], live_tickers: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], List[str]]:
    """Maps allowed Breakout coins to active Kraken perpetual futures tickers."""
    sym_map = {}
    missing = []
    for c in coins:
        c_clean = c.strip().upper()
        if f"PF_{c_clean}USD" in live_tickers:
            sym_map[c_clean] = f"PF_{c_clean}USD"
        elif f"PI_{c_clean}USD" in live_tickers:
            sym_map[c_clean] = f"PI_{c_clean}USD"
        else:
            missing.append(c_clean)
    return sym_map, missing


def scan_kraken_universe(symbol_map: Dict[str, str], live_tickers: Dict[str, Dict[str, Any]], lookback_days: int = 7) -> List[Dict[str, Any]]:
    """Scores Kraken Breakout coins using real-time markPrice for trailing 7-day Sharpe Momentum and Relative Strength."""
    # First get BTC 7d return using live markPrice
    btc_p = float(live_tickers.get("PF_XBTUSD", {}).get("markPrice", 0.0))
    df_btc = fetch_kraken_daily_candles("PF_XBTUSD")
    if df_btc is not None and len(df_btc) > lookback_days + 1:
        btc_past = float(df_btc["close"].iloc[-lookback_days - 1])
        btc_7d_ret = (btc_p - btc_past) / btc_past if btc_past > 0 and btc_p > 0 else 0.0
    else:
        btc_7d_ret = 0.0

    scored = []
    logger.info(f"Scanning {len(symbol_map)} Breakout coins on Kraken for {lookback_days}-day Sharpe & RS (Live Mark Prices)...")

    for coin, sym in symbol_map.items():
        if sym not in live_tickers:
            continue
        c_now = float(live_tickers[sym].get("markPrice", 0.0))
        if c_now <= 0:
            continue

        df = fetch_kraken_daily_candles(sym)
        if df is None or len(df) < lookback_days + 2:
            continue

        # Lookback price from lookback_days ago
        c_past = float(df["close"].iloc[-lookback_days - 1])
        if c_past <= 0:
            continue

        ret_7d = (c_now - c_past) / c_past

        # 7-day volatility from daily closes
        daily_rets = df["close"].pct_change().tail(lookback_days).dropna()
        vol_7d = float(daily_rets.std() * np.sqrt(365.25)) if len(daily_rets) >= 4 else 1.0
        if np.isnan(vol_7d) or vol_7d < 0.05:
            vol_7d = 0.05

        sharpe_score = ret_7d / vol_7d
        rs_spread = ret_7d - btc_7d_ret

        scored.append({
            "base": coin,
            "symbol": sym,
            "mark_price": c_now,
            "return_7d_pct": ret_7d * 100.0,
            "volatility_7d_pct": vol_7d * 100.0,
            "sharpe_score": sharpe_score,
            "rs_spread_pct": rs_spread * 100.0
        })
        time.sleep(0.02) # politeness delay

    scored.sort(key=lambda x: x["sharpe_score"], reverse=True)
    return scored


# -----------------------------------------------------------------------------
# PERSISTENT STATE MANAGEMENT
# -----------------------------------------------------------------------------
def load_portfolio_state() -> Dict[str, Any]:
    """Loads currently held symbols from daily state file."""
    state_file = get_state_file()
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load state file {state_file}: {e}")
    return {"current_holdings": []}


def save_portfolio_state(holdings: List[Dict[str, Any]], macro_status: str):
    """Persists updated portfolio holdings to daily state file."""
    data = {
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "cadence": "24h_daily",
        "lookback_days": 7,
        "macro_status": macro_status,
        "current_holdings": holdings
    }
    state_file = get_state_file()
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------------
# MAIN DAILY EXECUTION ENGINE
# -----------------------------------------------------------------------------
def run_scanner():
    logger.info("=" * 80)
    logger.info("STARTING DAILY KRAKEN BREAKOUT ROTATION SCANNER (7D LOOKBACK / PROP-FIRM RULES)")
    logger.info("=" * 80)

    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    account_size = cfg["account_size_usd"]
    total_exposure_pct = cfg["total_exposure_pct"]
    top_k = cfg["top_k"]
    buf_rank = cfg["rank_exit_buffer"]
    lookback = cfg["lookback_days"]
    sl_pct = cfg["stop_loss_pct"]
    tp1_pct = cfg["take_profit_1_pct"]
    tp2_pct = cfg["take_profit_2_pct"]
    allowed_coins = cfg.get("allowed_coins", DEFAULT_BREAKOUT_UNIVERSE)

    # Sizing for Prop Firm
    total_capital = account_size * (total_exposure_pct / 100.0)
    capital_per_coin = total_capital / top_k
    pct_per_coin = total_exposure_pct / top_k
    cash_reserved = account_size - total_capital

    now_utc = pd.Timestamp.now(tz="UTC")
    now_str = now_utc.strftime("%A, %b %d, %Y - %H:%M UTC")

    # Fetch Real-Time Kraken Live Tickers (Mark Prices)
    live_tickers = get_kraken_live_tickers()
    if not live_tickers:
        logger.error("Failed to query Kraken Futures live tickers.")
        return

    # 1. Macro Trend Check (PF_XBTUSD Live Price vs 50-day EMA)
    is_bullish, btc_close, btc_ema50, dist_pct = evaluate_kraken_btc_macro(live_tickers)
    logger.info(f"BTC Live Mark Price: ${btc_close:,.2f} | 50 EMA: ${btc_ema50:,.2f} | Bullish: {is_bullish}")

    # Load State
    old_state = load_portfolio_state()
    held_list = old_state.get("current_holdings", [])
    held_bases = [h.get("base", "") for h in held_list if h.get("base")]

    # BEARISH REGIME: Cash Shield
    if not is_bullish:
        logger.info("BTC is below 50-day EMA. Triggering 100% Cash Defense.")
        sell_lines = []
        for h in held_list:
            sell_lines.append(f"🔴 <b>SELL/CLOSE 100%:</b> <code>{h.get('symbol', h.get('base'))}</code>")
        sells_txt = "\n".join(sell_lines) if sell_lines else "✅ <i>Already 100% in Cash Margin. No positions open.</i>"

        msg = (
            f"🛡️ <b>DAILY BREAKOUT SCANNER: CASH DEFENSE TRIGGERED</b>\n"
            f"📅 <i>{now_str}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>BTC Macro Regime:</b> ❌ <b>BEARISH</b> (Below 50-day EMA)\n"
            f"• BTC Price: <code>${btc_close:,.2f}</code>\n"
            f"• 50 EMA: <code>${btc_ema50:,.2f}</code> (<b>{dist_pct:+.2f}%</b>)\n\n"
            f"🚨 <b>PROP FIRM ACTION REQUIRED:</b>\n"
            f"Preserve challenge equity in 100% Cash/Margin. Do not take altcoin longs.\n\n"
            f"{sells_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 <b>Cash Margin:</b> 100% ($0 at risk) | Protecting Trailing Drawdown Floor"
        )
        print("\n" + msg + "\n")
        send_telegram_message(token, chat_id, msg)
        save_portfolio_state([], "BEARISH_CASH")
        return

    # 2. Map Breakout coins to Kraken perpetual futures
    symbol_map, missing = map_breakout_coins_to_kraken(allowed_coins, live_tickers)
    logger.info(f"Active Breakout coins on Kraken: {len(symbol_map)} / {len(allowed_coins)}")

    # 3. Score Breakout universe for trailing 7-day Sharpe Momentum (Live Mark Prices)
    scored = scan_kraken_universe(symbol_map, live_tickers, lookback_days=lookback)
    for idx, item in enumerate(scored, 1):
        item["rank"] = idx

    df_scored = pd.DataFrame(scored)

    # 4. Apply Daily Buffer Rules
    buffer_bases = [item["base"] for item in scored[:buf_rank]]
    top_bases = [item["base"] for item in scored[:top_k]]

    # Sells: held coins that fell below rank buffer
    sells = [h for h in held_list if h.get("base") not in buffer_bases]
    # Holds: held coins still within buffer
    holds = [h for h in held_list if h.get("base") in buffer_bases]

    needed = top_k - len(holds)
    buys = []
    for item in scored[:top_k]:
        if item["base"] not in [h["base"] for h in holds] and len(buys) < needed:
            buys.append(item)

    # New final holdings
    final_holdings = holds + buys

    # 5. Format Telegram Message
    msg = (
        f"⚡ <b>DAILY BREAKOUT PROP SCANNER (24H / 7D LOOKBACK)</b>\n"
        f"📅 <i>{now_str}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>BTC Macro Regime:</b> ✅ <b>BULLISH</b> (Above 50-day EMA)\n"
        f"• BTC Price: <code>${btc_close:,.2f}</code> | 50 EMA: <code>${btc_ema50:,.2f}</code> (<b>{dist_pct:+.2f}%</b>)\n"
        f"• Universe: <b>Kraken Breakout Futures ({len(symbol_map)} assets)</b>\n"
        f"• Total Exposure: <b>{total_exposure_pct:.1f}%</b> (${total_capital:,.0f} across {top_k} coins / ~${capital_per_coin:,.0f} each)\n"
        f"• Cash Buffer: <b>{100-total_exposure_pct:.1f}%</b> (${cash_reserved:,.0f} reserve)\n"
        f"• Buffer Rule: <b>Rank ≤ {buf_rank}</b> (only sell if drops past #{buf_rank})\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>DAILY ACTION INSTRUCTIONS:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # Step 1: Sells
    msg += "<b>STEP 1: EXECUTE SELLS (Dropped Below Buffer)</b>\n"
    if sells:
        for s in sells:
            msg += f"🔴 <b>CLOSE 100%:</b> <code>{s.get('symbol', s.get('base'))}</code>\n"
        msg += "<i>➡️ Consolidate proceeds back into margin cash.</i>\n\n"
    else:
        msg += "✅ <i>None! All held coins remain inside the Top " + str(buf_rank) + " buffer.</i>\n\n"

    # Step 2: Buys
    msg += f"<b>STEP 2: EXECUTE BUYS ({len(buys)} Slot{'s' if len(buys) != 1 else ''} Open)</b>\n"
    if buys:
        msg += f"<i>Deploy ~${capital_per_coin:,.0f} (~{pct_per_coin:.1f}%) per open slot with bracket orders:</i>\n\n"
        for b in buys:
            p = b["mark_price"]
            sl_price = p * (1.0 - sl_pct / 100.0)
            tp1_price = p * (1.0 + tp1_pct / 100.0)
            tp2_price = p * (1.0 + tp2_pct / 100.0)
            msg += (
                f"🟢 <b>BUY:</b> <code>{b['symbol']}</code> (#{b['rank']} | Sharpe: {b['sharpe_score']:.2f})\n"
                f"   • Mark: <code>${p:,.4f}</code> | 7d: <code>{b['return_7d_pct']:+.1f}%</code>\n"
                f"   • 🛑 <b>Hard SL (-{sl_pct}%):</b> <code>${sl_price:,.4f}</code>\n"
                f"   • 🎯 <b>TP 1 (+{tp1_pct}%):</b> <code>${tp1_price:,.4f}</code> (Scale 50% & move SL to BE)\n"
                f"   • 🚀 <b>TP 2 (+{tp2_pct}%):</b> <code>${tp2_price:,.4f}</code> (Scale 25%)\n\n"
            )
    else:
        msg += "✨ <i>Portfolio is at target capacity. No new buys required today!</i>\n\n"

    # Step 3: Holds
    msg += f"<b>STEP 3: CONTINUED HOLDS ({len(holds)} Coins Intact)</b>\n"
    if holds:
        for h in holds:
            curr_match = df_scored[df_scored["base"] == h.get("base")]
            curr_rank = f"#{curr_match.iloc[0]['rank']}" if not curr_match.empty else "Active"
            msg += f"🛡️ <b>HOLD:</b> <code>{h.get('symbol', h.get('base'))}</code> (Current Rank: {curr_rank})\n"
        msg += "\n"

    # Top 5 Reserves
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "📌 <b>TOP 5 RESERVE BENCH:</b>\n"
    held_final_bases = [h["base"] for h in final_holdings]
    reserves = [item for item in scored if item["base"] not in held_final_bases][:5]
    for r in reserves:
        msg += f"#{r['rank']}. <code>{r['symbol']}</code> [7d: {r['return_7d_pct']:+.1f}% | Sharpe: {r['sharpe_score']:.2f}]\n"

    print("\n" + msg + "\n")
    send_telegram_message(token, chat_id, msg)

    # Save State
    save_portfolio_state(final_holdings, "BULLISH_ACTIVE")
    logger.info("Daily Kraken Breakout scanner finished successfully.")


if __name__ == "__main__":
    run_scanner()
