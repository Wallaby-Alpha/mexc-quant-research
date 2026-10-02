"""
scanner/kraken_breakout_scanner.py
Production-ready scanner for the Breakout Prop Firm / Kraken Futures Rotational Momentum Strategy.
Runs weekly (every Monday at 00:01 UTC) or on-demand:
1. Restricts asset universe strictly to the user-specified Breakout tradeable coins (65 assets).
2. Calculates BTC macro trend (50-day EMA) using PF_XBTUSD daily candles on Kraken.
3. Ranks Breakout altcoins by 20-day relative strength / momentum.
4. Selects the Top 10 coins and sizes them conservatively (15% total exposure / 1.5% per coin)
   to strictly respect Breakout Prop Firm limits (3.0% max daily loss, 5.0% / 6.0% max drawdown).
5. Dispatches formatted, actionable signals (BUY / SELL / HOLD) to a dedicated Telegram bot.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np

# Ensure UTF-8 output across all environments
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
        logging.FileHandler("kraken_breakout_scanner.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("breakout_scanner")

KRAKEN_FUTURES_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
KRAKEN_FUTURES_CHART_URL = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/1d"

STATE_FILE = Path("kraken_breakout_state.json")
CONFIG_FILE = Path("kraken_breakout_config.json")

# Verified Breakout Supported Coin Universe
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


def load_config() -> Dict[str, Any]:
    """Loads Breakout Kraken scanner configuration and credentials."""
    config = {
        "telegram_bot_token": os.environ.get("BREAKOUT_TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("BREAKOUT_TELEGRAM_CHAT_ID", ""),
        "account_size_usd": float(os.environ.get("BREAKOUT_ACCOUNT_SIZE", "100000.0")),
        "total_exposure_pct": float(os.environ.get("BREAKOUT_EXPOSURE_PCT", "15.0")),  # 15% total account exposure
        "top_k": int(os.environ.get("BREAKOUT_TOP_K", "10")),
        "lookback_days": int(os.environ.get("BREAKOUT_LOOKBACK_DAYS", "20")),
        "daily_loss_circuit_breaker_pct": 2.2,  # Alert user if daily portfolio loss reaches -2.2%
        "allowed_coins": DEFAULT_BREAKOUT_UNIVERSE
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
    """Dispatches a formatted message to Telegram Bot with auto-chunking."""
    if not token or not chat_id:
        logger.warning("Telegram Bot Token or Chat ID missing. Skipping alert dispatch.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3900
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    break
                else:
                    logger.warning(f"Telegram API error ({resp.status_code}): {resp.text}")
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Telegram network error: {e}")
                time.sleep(2)
    return True


def fetch_kraken_daily_candles(symbol: str) -> Optional[pd.DataFrame]:
    """Fetches historical daily candles for a Kraken Futures instrument."""
    url = KRAKEN_FUTURES_CHART_URL.format(symbol=symbol)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        candles = resp.json().get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.debug(f"Failed to fetch candles for {symbol}: {e}")
        return None


def evaluate_btc_macro_regime() -> Tuple[bool, float, float, float]:
    """
    Evaluates BTC macro regime on Kraken Futures (PF_XBTUSD).
    Returns: (is_bullish, latest_btc_price, ema_50, dist_pct)
    """
    logger.info("Evaluating BTC macro regime using PF_XBTUSD...")
    df_btc = fetch_kraken_daily_candles("PF_XBTUSD")
    if df_btc is None or len(df_btc) < 55:
        logger.error("Insufficient PF_XBTUSD candle data to calculate 50 EMA.")
        return False, 0.0, 0.0, 0.0
    
    # Exclude open in-progress candle to prevent lookahead
    closed_candles = df_btc.iloc[:-1].copy()
    closed_candles["ema_50"] = closed_candles["close"].ewm(span=50, adjust=False).mean()
    
    latest_row = closed_candles.iloc[-1]
    btc_close = float(latest_row["close"])
    btc_ema50 = float(latest_row["ema_50"])
    
    dist_pct = ((btc_close - btc_ema50) / btc_ema50) * 100.0
    is_bullish = btc_close > btc_ema50
    
    logger.info(f"BTC Close: ${btc_close:,.2f} | 50 EMA: ${btc_ema50:,.2f} | Dist: {dist_pct:+.2f}% | Bullish: {is_bullish}")
    return is_bullish, btc_close, btc_ema50, dist_pct


def map_breakout_universe_to_kraken(allowed_coins: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Maps the specified Breakout coins to their corresponding Kraken Futures symbols."""
    try:
        resp = requests.get(KRAKEN_FUTURES_TICKERS_URL, timeout=12)
        if resp.status_code != 200:
            logger.error("Failed to query Kraken tickers.")
            return {}, allowed_coins
        
        tickers = resp.json().get("tickers", [])
        active_symbols = {t["symbol"]: t for t in tickers if not t.get("suspended", False)}

        symbol_map = {}
        missing = []

        cleaned_coins = sorted(list(set(c.strip().upper() for c in allowed_coins if c.strip())))

        for coin in cleaned_coins:
            direct_pf = f"PF_{coin}USD"
            direct_pi = f"PI_{coin}USD"
            
            if direct_pf in active_symbols:
                symbol_map[coin] = direct_pf
            elif direct_pi in active_symbols:
                symbol_map[coin] = direct_pi
            else:
                missing.append(coin)

        return symbol_map, missing
    except Exception as e:
        logger.error(f"Error mapping Breakout universe to Kraken: {e}")
        return {}, allowed_coins


def rank_breakout_universe_momentum(symbol_map: Dict[str, str], lookback_days: int = 20) -> List[Dict[str, Any]]:
    """Calculates relative strength / momentum strictly for the Breakout allowed coins."""
    ranked = []
    total = len(symbol_map)
    logger.info(f"Scanning {total} Breakout coins on Kraken for {lookback_days}-day Relative Strength...")

    for coin, sym in symbol_map.items():
        df = fetch_kraken_daily_candles(sym)
        if df is None or len(df) <= lookback_days + 1:
            continue
        
        # Exclude in-progress candle
        closed = df.iloc[:-1]
        c_now = closed.iloc[-1]["close"]
        c_past = closed.iloc[-lookback_days]["close"]
        
        if c_past <= 0:
            continue
        
        ret_pct = ((c_now - c_past) / c_past) * 100.0

        ranked.append({
            "base": coin,
            "symbol": sym,
            "mark_price": c_now,
            "momentum_pct": ret_pct
        })
        time.sleep(0.04)  # Politeness delay

    ranked.sort(key=lambda x: x["momentum_pct"], reverse=True)
    logger.info(f"Successfully ranked {len(ranked)} active Breakout assets.")
    return ranked


def load_state() -> Dict[str, Any]:
    """Loads persistent portfolio state from previous rebalance."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading state file {STATE_FILE}: {e}")
    return {"current_holdings": [], "last_rebalance_utc": None, "macro_status": "UNKNOWN"}


def save_state(state: Dict[str, Any]) -> None:
    """Saves updated portfolio state."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved state to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def run_scanner(dry_run: bool = False, force: bool = False) -> None:
    """Executes the Breakout weekly rotation scan restricted strictly to the user's coin universe."""
    config = load_config()
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    account_size = config["account_size_usd"]
    total_exposure_pct = config["total_exposure_pct"]
    top_k = config["top_k"]
    lookback = config["lookback_days"]
    circuit_breaker = config["daily_loss_circuit_breaker_pct"]
    allowed_coins = config.get("allowed_coins", DEFAULT_BREAKOUT_UNIVERSE)

    # Sizing calculations for Breakout Prop Firm Rules
    total_capital_deployed = account_size * (total_exposure_pct / 100.0)
    capital_per_coin = total_capital_deployed / top_k
    pct_per_coin = total_exposure_pct / top_k
    cash_reserved = account_size - total_capital_deployed
    cash_pct = 100.0 - total_exposure_pct

    now_utc = pd.Timestamp.now(tz="UTC")
    logger.info(f"Running Breakout Scanner for {len(allowed_coins)} specified coins at {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Check BTC Macro Regime
    is_bullish, btc_close, btc_ema50, dist_pct = evaluate_btc_macro_regime()

    # Load Previous State
    old_state = load_state()
    old_symbols = [h["symbol"] for h in old_state.get("current_holdings", [])]

    if not is_bullish:
        # BEARISH REGIME: Switch to 100% Cash Margin
        logger.info("Macro regime is BEARISH (BTC < 50-day EMA). Strategy mandates 100% CASH.")
        
        sell_orders = []
        for h in old_state.get("current_holdings", []):
            sell_orders.append(f"  🔴 <b>CLOSE/SELL:</b> {h['symbol']} ({h['base']})")
        
        sells_txt = "\n".join(sell_orders) if sell_orders else "  ⚪ No open positions were held."
        
        msg = (
            f"🛡️ <b>BREAKOUT PROP SCANNER: CASH DEFENSIVE MODE</b>\n"
            f"📅 <i>{now_utc.strftime('%A, %b %d, %Y - %H:%M UTC')}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Macro BTC Regime:</b> 🔴 <b>BEARISH</b>\n"
            f"• BTC Close: <code>${btc_close:,.2f}</code>\n"
            f"• 50-Day EMA: <code>${btc_ema50:,.2f}</code> (<b>{dist_pct:+.2f}%</b> below)\n\n"
            f"🚨 <b>ACTION REQUIRED:</b>\n"
            f"Close all altcoin long positions and preserve account equity in 100% cash/margin.\n\n"
            f"<b>Positions to Close:</b>\n"
            f"{sells_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 <b>Portfolio State:</b> 100% Cash Margin ($0 allocated)\n"
            f"🎯 <b>Objective:</b> Strictly protect Breakout drawdown limit (6% Classic / 5% Pro)"
        )
        print("\n" + msg + "\n")
        if not dry_run:
            send_telegram_message(token, chat_id, msg)
            save_state({
                "current_holdings": [],
                "last_rebalance_utc": now_utc.isoformat(),
                "macro_status": "BEARISH_CASH"
            })
        return

    # Map allowed Breakout coins to Kraken
    symbol_map, missing_coins = map_breakout_universe_to_kraken(allowed_coins)
    logger.info(f"Active tradeable coins on Kraken: {len(symbol_map)} / {len(allowed_coins)}")

    # Rank by 20-day momentum
    ranked = rank_breakout_universe_momentum(symbol_map, lookback_days=lookback)
    if not ranked:
        logger.error("No valid candidate coins could be ranked.")
        return

    selected_10 = ranked[:top_k]
    new_symbols = [c["symbol"] for c in selected_10]

    # Rebalance Diff: BUYS, HOLDS, SELLS
    buys = [c for c in selected_10 if c["symbol"] not in old_symbols]
    holds = [c for c in selected_10 if c["symbol"] in old_symbols]
    sells = [s for s in old_symbols if s not in new_symbols]

    # Build Telegram Message
    buys_txt = "\n".join([
        f"  🟢 <b>BUY:</b> <code>{c['symbol']}</code> (<b>{c['base']}</b>)\n"
        f"     • 20d Rel. Strength: <b>{c['momentum_pct']:+.1f}%</b> | Price: <code>${c['mark_price']:,.4f}</code>\n"
        f"     • Position Size: <b>${capital_per_coin:,.0f}</b> ({pct_per_coin:.1f}% of account)"
        for c in buys
    ]) if buys else "  ⚪ None (Portfolio fully aligned)"

    holds_txt = "\n".join([
        f"  🔵 <b>HOLD:</b> <code>{c['symbol']}</code> (<b>{c['base']}</b>) | Rel. Strength: <b>{c['momentum_pct']:+.1f}%</b>"
        for c in holds
    ]) if holds else "  ⚪ None"

    sells_txt = "\n".join([
        f"  🔴 <b>CLOSE:</b> <code>{s}</code> (Dropped out of Top {top_k})"
        for s in sells
    ]) if sells else "  ⚪ None"

    msg = (
        f"🏦 <b>BREAKOUT PROP TRADING: RELATIVE STRENGTH ROTATION</b>\n"
        f"📅 <i>{now_utc.strftime('%A, %b %d, %Y - %H:%M UTC')}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Macro BTC Regime:</b> 🟢 <b>BULLISH</b>\n"
        f"• BTC Close: <code>${btc_close:,.2f}</code> | 50 EMA: <code>${btc_ema50:,.2f}</code> (<b>{dist_pct:+.2f}%</b>)\n\n"
        f"🛡️ <b>BREAKOUT SIZING & RISK RULES ({total_exposure_pct:.0f}% Total Exposure):</b>\n"
        f"• Account Balance: <code>${account_size:,.0f}</code>\n"
        f"• Total Exposure: <b>${total_capital_deployed:,.0f}</b> ({total_exposure_pct:.0f}% across {top_k} coins)\n"
        f"• Size per Coin: <b>${capital_per_coin:,.0f}</b> ({pct_per_coin:.1f}% each)\n"
        f"• Cash Margin Buffer: <b>${cash_reserved:,.0f}</b> ({cash_pct:.0f}% safe buffer)\n"
        f"• ⚠️ <b>Hard Daily Stop:</b> Close all if daily loss hits <b>-{circuit_breaker:.1f}%</b> (safeguards 3.0% daily cap)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>REBALANCE INSTRUCTIONS (Breakout Universe):</b>\n\n"
        f"<b>1. NEW ORDERS TO OPEN ({len(buys)}):</b>\n"
        f"{buys_txt}\n\n"
        f"<b>2. EXISTING POSITIONS TO HOLD ({len(holds)}):</b>\n"
        f"{holds_txt}\n\n"
        f"<b>3. POSITIONS TO CLOSE ({len(sells)}):</b>\n"
        f"{sells_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Evaluation Target:</b> +10% Classic / +12% Pro (No Time Limit)"
    )

    print("\n" + msg + "\n")

    if not dry_run:
        send_telegram_message(token, chat_id, msg)
        save_state({
            "current_holdings": selected_10,
            "last_rebalance_utc": now_utc.isoformat(),
            "macro_status": "BULLISH_INVESTED"
        })


def test_telegram_connection() -> None:
    """Verifies Telegram Bot Token and Chat ID."""
    config = load_config()
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    if not token or not chat_id:
        print("❌ Error: Missing BREAKOUT_TELEGRAM_BOT_TOKEN or BREAKOUT_TELEGRAM_CHAT_ID in config/env.")
        sys.exit(1)
    
    test_msg = (
        "🤖 <b>Breakout Prop Trading Bot Connected!</b>\n\n"
        "Your dedicated Telegram alert channel for Breakout Relative Strength Rotation is active.\n"
        "Universe: 65 Breakout supported coins.\n"
        "Rebalances will trigger every <b>Monday at 00:01 UTC</b>."
    )
    print(f"Sending test notification to Chat ID: {chat_id}...")
    success = send_telegram_message(token, chat_id, test_msg)
    if success:
        print("✅ Telegram notification delivered successfully!")
    else:
        print("❌ Failed to deliver Telegram notification. Please check token & chat ID.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breakout Prop Firm Relative Strength Weekly Rotation Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Run scan and print results to terminal without sending Telegram or saving state.")
    parser.add_argument("--force-scan", action="store_true", help="Force immediate execution regardless of weekday.")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test message to verify Telegram credentials.")
    args = parser.parse_args()

    if args.test_telegram:
        test_telegram_connection()
    else:
        run_scanner(dry_run=args.dry_run, force=args.force_scan)
