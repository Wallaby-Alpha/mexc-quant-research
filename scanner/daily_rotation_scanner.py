"""
scanner/daily_rotation_scanner.py
Production-ready daily scanner for the 24-Hour Rebalance / 7-Day Lookback Rotational Momentum Strategy.
Evaluates BTC 50-day EMA macro health daily at 00:00 UTC, ranks liquid MEXC altcoins by trailing 7-day
Sharpe momentum and Relative Strength vs BTC, applies buffer rules to smooth daily turnover, resolves
blockchain networks (Solana, Base, BSC, L1s), constructs the active 10-coin portfolio (7 Quality + 3 Raw),
and sends actionable step-by-step instructions via Telegram.
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
STATE_FILE = Path("portfolio_state_daily.json")
CONFIG_FILE = Path("daily_scanner_config.json")
FALLBACK_CONFIG_FILE = Path("scanner_config.json")

# Known Native Layer 1s / Layer 2s
NATIVE_L1_CHAINS = {
    "BTC": "Bitcoin L1",
    "ETH": "Ethereum L1",
    "SOL": "Solana L1",
    "NEAR": "NEAR L1",
    "RUNE": "THORChain L1",
    "MOVR": "Moonriver EVM",
    "GLMR": "Moonbeam EVM",
    "SUI": "Sui L1",
    "APT": "Aptos L1",
    "SEI": "Sei L1",
    "AVAX": "Avalanche C-Chain",
    "DOT": "Polkadot L1",
    "ATOM": "Cosmos L1",
    "ADA": "Cardano L1",
    "TON": "TON L1",
    "KAS": "Kaspa L1",
    "TAO": "Bittensor L1",
    "XRP": "XRP Ledger",
    "DOGE": "Dogecoin L1",
    "LTC": "Litecoin L1",
    "TRX": "Tron L1",
    "INJ": "Injective L1",
    "FTM": "Fantom / Sonic",
    "ICP": "Internet Computer L1",
    "HBAR": "Hedera L1",
    "ALGO": "Algorand L1",
}

CHAIN_CACHE: Dict[str, str] = {}


def format_chain_name(raw_chain: str) -> str:
    """Formats chain identifiers nicely for display."""
    mapping = {
        "solana": "Solana",
        "base": "Base",
        "bsc": "BSC",
        "ethereum": "Ethereum",
        "arbitrum": "Arbitrum",
        "optimism": "Optimism",
        "polygon": "Polygon",
        "avalanche": "Avalanche",
        "sui": "Sui",
        "aptos": "Aptos",
        "ton": "TON",
        "fantom": "Fantom",
        "blast": "Blast",
        "mantle": "Mantle",
        "linea": "Linea",
        "scroll": "Scroll",
        "ronin": "Ronin"
    }
    return mapping.get(raw_chain.lower(), raw_chain.title())


def get_mexc_exchange_info() -> Dict[str, Any]:
    """Fetches full symbol metadata from MEXC exchangeInfo."""
    url = f"{MEXC_BASE_URL}/api/v3/exchangeInfo"
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            return {s["symbol"]: s for s in data.get("symbols", [])}
    except Exception as e:
        logger.warning(f"Failed to fetch exchangeInfo: {e}")
    return {}


def resolve_chain(symbol: str, exchange_info: Dict[str, Any]) -> str:
    """Resolves blockchain network for a given symbol."""
    if symbol in CHAIN_CACHE:
        return CHAIN_CACHE[symbol]

    base = symbol.replace("USDT", "")
    if base in NATIVE_L1_CHAINS:
        chain_name = NATIVE_L1_CHAINS[base]
        CHAIN_CACHE[symbol] = chain_name
        return chain_name

    info = exchange_info.get(symbol, {})
    contract = info.get("contractAddress", "").strip()

    chain_name = "Unknown"
    if contract:
        try:
            r = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{contract}", timeout=3)
            if r.status_code == 200:
                pairs = r.json().get("pairs", [])
                if pairs:
                    cid = pairs[0].get("chainId", "").lower()
                    chain_name = format_chain_name(cid)
        except Exception:
            pass

        if chain_name == "Unknown":
            if not contract.startswith("0x") and len(contract) >= 32:
                chain_name = "Solana"
            elif contract.startswith("0x"):
                chain_name = "EVM"

    if chain_name == "Unknown":
        plates = info.get("conceptPlates", [])
        if plates:
            chain_name = plates[0]
        else:
            chain_name = "MEXC Spot"

    CHAIN_CACHE[symbol] = chain_name
    return chain_name


def load_config() -> Dict[str, Any]:
    """Loads configuration and Telegram credentials for daily scanner."""
    config = {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "top_k_quality": int(os.environ.get("TOP_K_QUALITY", "7")),
        "top_k_raw": int(os.environ.get("TOP_K_RAW", "3")),
        "rank_exit_buffer_quality": int(os.environ.get("BUFFER_QUALITY", "12")),
        "rank_exit_buffer_raw": int(os.environ.get("BUFFER_RAW", "6")),
        "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "7")), # Default 7 days for daily cadence
        "min_volume_24h_usdt": float(os.environ.get("MIN_VOLUME_USDT", "500000.0")),
    }
    
    # Try daily config first, fallback to standard scanner config
    cfg_to_read = CONFIG_FILE if CONFIG_FILE.exists() else (FALLBACK_CONFIG_FILE if FALLBACK_CONFIG_FILE.exists() else None)
    if cfg_to_read:
        try:
            with open(cfg_to_read, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                config.update(file_cfg)
        except Exception as e:
            logger.warning(f"Failed to read {cfg_to_read}: {e}")

    return config


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Dispatches formatted message to Telegram Bot, auto-chunking if length > 3800."""
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not configured. Printing message to stdout only.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    max_len = 3800
    chunks = []
    if len(text) <= max_len:
        chunks = [text]
    else:
        current_chunk = ""
        for line in text.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    success = True
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                logger.info(f"Successfully sent Telegram alert part {i+1}/{len(chunks)}.")
            else:
                logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
                success = False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            success = False
        if len(chunks) > 1 and i < len(chunks) - 1:
            time.sleep(0.5)

    return success


def get_mexc_klines(symbol: str, interval: str = "1d", limit: int = 100) -> pd.DataFrame:
    """Fetches daily klines from MEXC public REST API with retries."""
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

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = df[col].astype(float)
            return df.sort_values("open_time").reset_index(drop=True)
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Error fetching klines for {symbol}: {e}")
            time.sleep(0.8)
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
            if not sym.endswith("USDT"):
                continue
            if any(x in sym for x in ["3L", "3S", "4L", "4S", "5L", "5S", "USDC", "BUSD", "TUSD", "FDUSD", "EUR", "DAI"]):
                continue

            quote_vol = float(item.get("quoteVolume", 0.0))
            if quote_vol >= min_volume:
                pairs.append((sym, quote_vol))

        pairs.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [p[0] for p in pairs[:160]]
        logger.info(f"Retrieved {len(top_symbols)} liquid USDT pairs above ${min_volume:,.0f} 24h volume.")
        return top_symbols
    except Exception as e:
        logger.error(f"Error fetching 24hr tickers: {e}")
        return []


def check_btc_macro_trend() -> Tuple[bool, float, float]:
    """Evaluates whether Bitcoin daily close is above its 50-day EMA."""
    df_btc = get_mexc_klines("BTCUSDT", interval="1d", limit=100)
    if len(df_btc) < 55:
        logger.error("Insufficient BTC daily bars.")
        return False, 0.0, 0.0

    df_btc["ema50"] = df_btc["close"].ewm(span=50, adjust=False).mean()
    latest = df_btc.iloc[-1]
    btc_close = latest["close"]
    btc_ema50 = latest["ema50"]
    is_bullish = btc_close >= btc_ema50

    return is_bullish, btc_close, btc_ema50


def load_portfolio_state() -> Dict[str, Any]:
    """Loads currently held symbols from daily state file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load daily state file: {e}")
    return {"current_quality_holdings": [], "current_raw_holdings": []}


def save_portfolio_state(quality_holdings: List[str], raw_holdings: List[str]):
    """Persists updated portfolio holdings to daily state file."""
    data = {
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "cadence": "24h_daily",
        "lookback_days": 7,
        "current_quality_holdings": quality_holdings,
        "current_raw_holdings": raw_holdings,
        "total_active_holdings": quality_holdings + raw_holdings
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_scanner():
    logger.info("=" * 70)
    logger.info("RUNNING DAILY 24H ROTATIONAL MOMENTUM SCANNER (7D LOOKBACK)")
    logger.info("=" * 70)

    cfg = load_config()
    state = load_portfolio_state()
    held_quality = state.get("current_quality_holdings", [])
    held_raw = state.get("current_raw_holdings", [])
    all_held = held_quality + held_raw

    # 1. Evaluate Bitcoin Macro Health
    is_btc_bullish, btc_close, btc_ema50 = check_btc_macro_trend()
    logger.info(f"BTC Price: ${btc_close:,.2f} | 50-day EMA: ${btc_ema50:,.2f} | Bullish: {is_btc_bullish}")

    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    # If BTC is Bearish -> Cash Protection Mode
    if not is_btc_bullish:
        msg = (
            f"🚨 <b>DAILY ROTATION SCANNER: CASH SHIELD TRIGGERED</b>\n"
            f"📅 <i>{now_utc}</i>\n\n"
            f"<b>BTC Macro Trend:</b> ❌ <b>BEARISH</b> (Below 50 EMA)\n"
            f"• BTC Price: <code>${btc_close:,.2f}</code>\n"
            f"• 50-Day EMA: <code>${btc_ema50:,.2f}</code>\n"
            f"• Trend Deficit: <code>{(btc_close - btc_ema50) / btc_ema50 * 100:+.2f}%</code>\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 <b>ACTION REQUIRED: PROTECT CAPITAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        if all_held:
            msg += "<b>SELL 100% OF EXISTING HOLDINGS TO USDT:</b>\n"
            for s in all_held:
                msg += f"🔴 <b>SELL</b> <code>{s}</code>\n"
            save_portfolio_state([], [])
        else:
            msg += "✅ <b>Portfolio is already 100% in USDT Cash.</b> No actions needed.\n"

        print(msg)
        send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
        return

    # 2. Retrieve liquid universe
    universe = get_top_mexc_pairs(min_volume=cfg["min_volume_24h_usdt"])
    exchange_info = get_mexc_exchange_info()

    # 3. Trailing BTC Return over lookback_days (7 days)
    lookback = cfg["lookback_days"] # 7
    df_btc = get_mexc_klines("BTCUSDT", interval="1d", limit=lookback + 5)
    if len(df_btc) > lookback:
        btc_past = df_btc.iloc[-lookback - 1]["close"]
        btc_now = df_btc.iloc[-1]["close"]
        btc_7d_return = (btc_now - btc_past) / btc_past
    else:
        btc_7d_return = 0.0

    # 4. Compute 7-day Relative Strength & Sharpe Scores across Universe
    scored_coins = []
    logger.info(f"Computing 7-day momentum and Sharpe features across {len(universe)} pairs...")

    for sym in universe:
        df_k = get_mexc_klines(sym, interval="1d", limit=lookback + 15)
        if len(df_k) < lookback + 1:
            continue

        alt_close_now = df_k.iloc[-1]["close"]
        alt_close_past = df_k.iloc[-lookback - 1]["close"]
        if alt_close_past <= 0:
            continue

        alt_ret = (alt_close_now - alt_close_past) / alt_close_past

        # Trailing 7-day Volatility
        daily_rets = df_k.iloc[-lookback - 1:]["close"].pct_change().dropna()
        vol = float(daily_rets.std() * np.sqrt(365.25)) if len(daily_rets) >= 5 else 1.0
        if np.isnan(vol) or vol < 0.05:
            vol = 0.05

        sharpe_score = alt_ret / vol
        rs_spread = alt_ret - btc_7d_return

        scored_coins.append({
            "symbol": sym,
            "price": alt_close_now,
            "alt_return_7d": alt_ret,
            "volatility_7d": vol,
            "sharpe_score": sharpe_score,
            "rs_spread_vs_btc": rs_spread
        })

    if not scored_coins:
        logger.error("No valid scored coins available.")
        return

    df_all = pd.DataFrame(scored_coins)

    # A. Quality Ranking (Sharpe Momentum: 7d Return / 7d Vol)
    df_quality = df_all.sort_values(by="sharpe_score", ascending=False).reset_index(drop=True)
    df_quality["rank_quality"] = df_quality.index + 1

    # B. Raw Ranking (Relative Strength Spread vs BTC)
    df_raw = df_all.sort_values(by="rs_spread_vs_btc", ascending=False).reset_index(drop=True)
    df_raw["rank_raw"] = df_raw.index + 1

    k_quality = cfg["top_k_quality"] # 7
    k_raw = cfg["top_k_raw"]         # 3
    buf_quality = cfg["rank_exit_buffer_quality"] # 12
    buf_raw = cfg["rank_exit_buffer_raw"]         # 6

    # 5. Apply Daily Buffer Rules
    quality_buffer_symbols = df_quality.head(buf_quality)["symbol"].tolist()
    quality_top_symbols = df_quality.head(k_quality)["symbol"].tolist()

    holds_quality = [s for s in held_quality if s in quality_buffer_symbols]
    sells_quality = [s for s in held_quality if s not in quality_buffer_symbols]
    slots_needed_quality = k_quality - len(holds_quality)

    buys_quality = []
    for s in quality_top_symbols:
        if s not in holds_quality and len(buys_quality) < slots_needed_quality:
            buys_quality.append(s)

    final_quality_holdings = holds_quality + buys_quality

    # Raw Selection (prevent overlap with quality)
    df_raw_filtered = df_raw[~df_raw["symbol"].isin(final_quality_holdings)].reset_index(drop=True)
    raw_buffer_symbols = df_raw_filtered.head(buf_raw)["symbol"].tolist()
    raw_top_symbols = df_raw_filtered.head(k_raw)["symbol"].tolist()

    holds_raw = [s for s in held_raw if s in raw_buffer_symbols and s not in final_quality_holdings]
    sells_raw = [s for s in held_raw if s not in holds_raw]
    slots_needed_raw = k_raw - len(holds_raw)

    buys_raw = []
    for s in raw_top_symbols:
        if s not in holds_raw and len(buys_raw) < slots_needed_raw:
            buys_raw.append(s)

    final_raw_holdings = holds_raw + buys_raw
    total_new_holdings = final_quality_holdings + final_raw_holdings

    all_sells = list(set(sells_quality + sells_raw))
    all_buys = buys_quality + buys_raw
    all_holds = holds_quality + holds_raw

    # Resolve chains
    coin_chains = {s: resolve_chain(s, exchange_info) for s in total_new_holdings}
    for s in all_sells + all_buys:
        if s not in coin_chains:
            coin_chains[s] = resolve_chain(s, exchange_info)

    chain_counts: Dict[str, int] = {}
    for s in total_new_holdings:
        ch = coin_chains.get(s, "Unknown")
        chain_counts[ch] = chain_counts.get(ch, 0) + 1
    chain_dist_str = " | ".join([f"{ch}: {cnt}" for ch, cnt in sorted(chain_counts.items(), key=lambda x: x[1], reverse=True)])

    # 6. Format Telegram Message
    msg = (
        f"⚡ <b>DAILY 24H ROTATION SCANNER (7D LOOKBACK)</b>\n"
        f"📅 <i>{now_utc}</i>\n\n"
        f"<b>BTC Macro Trend:</b> ✅ <b>BULLISH</b> (Above 50 EMA)\n"
        f"• BTC Price: <code>${btc_close:,.2f}</code> (EMA50: <code>${btc_ema50:,.2f}</code>)\n"
        f"• BTC 7d Return: <code>{btc_7d_return*100:+.1f}%</code>\n"
        f"• Allocation: <b>7 Quality (Sharpe) + 3 Raw (High-Beta)</b>\n"
        f"• Buffer Rules: <b>Quality Rank ≤ {buf_quality} | Raw Rank ≤ {buf_raw}</b>\n"
        f"• ⛓️ <b>Chain Exposure:</b> <i>{chain_dist_str}</i>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>DAILY ACTION INSTRUCTIONS:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # Step 1: Sells
    msg += "<b>STEP 1: EXECUTE SELLS (Dropped Below Buffer)</b>\n"
    if all_sells:
        for s in all_sells:
            ch = coin_chains.get(s, "Unknown")
            msg += f"🔴 <b>SELL 100%</b> of <code>{s}</code> [{ch}] to USDT\n"
        msg += "<i>➡️ Consolidate proceeds into USDT cash.</i>\n\n"
    else:
        msg += "✅ <i>None! No held coins dropped out of buffer.</i>\n\n"

    # Step 2: Buys
    msg += f"<b>STEP 2: EXECUTE BUYS ({len(all_buys)} Slot{'s' if len(all_buys) != 1 else ''} Open)</b>\n"
    if all_buys:
        msg += f"<i>Allocate ~10% per open slot:</i>\n"
        for s in buys_quality:
            row = df_quality[df_quality["symbol"] == s].iloc[0]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🟢 <b>BUY</b> <code>{s}</code> [{ch} | Quality #Q{row['rank_quality']} | 7d: {row['alt_return_7d']*100:+.1f}% | Sharpe: {row['sharpe_score']:.2f}]\n"
        for s in buys_raw:
            row = df_raw[df_raw["symbol"] == s].iloc[0]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🚀 <b>BUY</b> <code>{s}</code> [{ch} | Raw #R{row['rank_raw']} | 7d: {row['alt_return_7d']*100:+.1f}% | RS: {row['rs_spread_vs_btc']*100:+.1f}%]\n"
        msg += "\n"
    else:
        msg += "✨ <i>Portfolio is at target capacity. No new buys required today!</i>\n\n"

    # Step 3: Holds
    msg += f"<b>STEP 3: CONTINUED HOLDS ({len(all_holds)} Coins Intact)</b>\n"
    if all_holds:
        for s in holds_quality:
            row = df_quality[df_quality["symbol"] == s].iloc[0] if s in df_quality["symbol"].values else None
            q_rk = f"#Q{row['rank_quality']}" if row is not None else "Rank OK"
            ch = coin_chains.get(s, "Unknown")
            msg += f"🛡️ <b>HOLD</b> <code>{s}</code> [{ch} | Quality {q_rk}]\n"
        for s in holds_raw:
            row = df_raw[df_raw["symbol"] == s].iloc[0] if s in df_raw["symbol"].values else None
            r_rk = f"#R{row['rank_raw']}" if row is not None else "Rank OK"
            ch = coin_chains.get(s, "Unknown")
            msg += f"⚡ <b>HOLD</b> <code>{s}</code> [{ch} | Raw {r_rk}]\n"
        msg += "\n"

    # Top 5 Alternates
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "📌 <b>TOP 5 RESERVE BENCH (If a coin is unavailable):</b>\n"
    top_reserves = [s for s in df_quality["symbol"] if s not in total_new_holdings][:5]
    for idx, sym in enumerate(top_reserves, 1):
        row = df_quality[df_quality["symbol"] == sym].iloc[0]
        ch = resolve_chain(sym, exchange_info)
        msg += f"{idx}. <code>{sym}</code> [{ch} | 7d: {row['alt_return_7d']*100:+.1f}% | Sharpe: {row['sharpe_score']:.2f}]\n"

    print(msg)
    send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)

    # Save State
    save_portfolio_state(final_quality_holdings, final_raw_holdings)
    logger.info("Daily scanner cycle completed successfully. State saved.")


if __name__ == "__main__":
    run_scanner()
