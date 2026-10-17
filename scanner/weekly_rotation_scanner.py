"""
scanner/weekly_rotation_scanner.py
Production-ready scanner for the 10-Coin Barbell Momentum Strategy (7 Quality + 3 Raw).
Runs weekly (Monday 00:00 UTC), evaluates BTC 50-day EMA macro health, ranks liquid MEXC altcoins,
resolves blockchain networks (Solana, Base, BSC, L1s, etc.), constructs the combined 10-coin portfolio,
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
        logging.FileHandler("scanner.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("rotation_scanner")

MEXC_BASE_URL = "https://api.mexc.com"
STATE_FILE = Path("portfolio_state.json")
CONFIG_FILE = Path("scanner_config.json")

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
        # Check DexScreener token endpoint
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
    """Loads configuration and Telegram credentials."""
    config = {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "top_k_quality": int(os.environ.get("TOP_K_QUALITY", "7")),
        "top_k_raw": int(os.environ.get("TOP_K_RAW", "3")),
        "rank_exit_buffer_quality": int(os.environ.get("BUFFER_QUALITY", "11")),
        "rank_exit_buffer_raw": int(os.environ.get("BUFFER_RAW", "6")),
        "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "30")),
        "min_volume_24h_usdt": float(os.environ.get("MIN_VOLUME_USDT", "500000.0")),
        "max_3d_pump_pct": float(os.environ.get("MAX_3D_PUMP_PCT", "0.25")),  # Solution A: reject new buys with >25% in last 3d
        "max_position_weight_pct": float(os.environ.get("MAX_POS_WEIGHT", "0.20")),  # Rebalance trim trigger at 20%
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
    """Dispatches formatted message to Telegram Bot, auto-chunking if length > 3800."""
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not configured. Printing message to stdout only.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Split text if it exceeds Telegram's 4096 character limit
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
            # Exclude leveraged ETF tokens and stablecoins
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
    """Loads currently held symbols and metadata from local state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")
    return {"current_quality_holdings": [], "current_raw_holdings": []}


def save_portfolio_state(quality_holdings: List[str], raw_holdings: List[str]):
    """Persists updated portfolio holdings to local state."""
    data = {
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "current_quality_holdings": quality_holdings,
        "current_raw_holdings": raw_holdings,
        "total_active_holdings": quality_holdings + raw_holdings
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_scanner(cfg_override: Optional[Dict[str, Any]] = None):
    logger.info("=" * 70)
    logger.info("RUNNING WEEKLY 10-COIN BARBELL ROTATIONAL MOMENTUM SCANNER")
    logger.info("=" * 70)

    cfg = load_config()
    if cfg_override:
        cfg.update(cfg_override)
    state = load_portfolio_state()
    held_quality = state.get("current_quality_holdings", [])
    held_raw = state.get("current_raw_holdings", [])
    all_held = held_quality + held_raw

    # 1. Evaluate Bitcoin Macro Health
    is_btc_bullish, btc_close, btc_ema50 = check_btc_macro_trend()
    logger.info(f"BTC Price: ${btc_close:,.2f} | 50-day EMA: ${btc_ema50:,.2f} | Bullish: {is_btc_bullish}")

    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    bypass_btc = cfg.get("bypass_btc_gate", False)
    if not is_btc_bullish and bypass_btc:
        logger.warning("Bypassing BTC Cash Gate as requested via --bypass-btc-gate.")

    # If BTC is Bearish -> Cash Protection Mode (unless bypassed)
    if not is_btc_bullish and not bypass_btc:
        msg = (
            f"🚨 <b>WEEKLY ROTATION SCANNER: CASH SHIELD TRIGGERED</b>\n"
            f"📅 <i>{now_utc}</i>\n\n"
            f"<b>BTC Macro Trend:</b> ❌ <b>BEARISH</b> (Below 50 EMA)\n"
            f"• BTC Price: <code>${btc_close:,.2f}</code>\n"
            f"• 50-Day EMA: <code>${btc_ema50:,.2f}</code>\n"
            f"• Trend Deficit: <code>{(btc_close - btc_ema50) / btc_ema50 * 100:+.2f}%</code>\n\n"
            f"🛑 <b>MANDATORY ACTION: MOVE 100% TO USDT CASH</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        if all_held:
            msg += "<b>EXECUTE IMMEDIATE SELL ORDERS:</b>\n"
            for sym in all_held:
                msg += f"🔴 SELL <code>{sym}</code> to 100% USDT\n"
            save_portfolio_state([], [])
            msg += "\n✅ Hold 100% USDT cash until next Monday's 00:00 UTC check."
        else:
            msg += "✅ <b>Portfolio is already 100% USDT Cash.</b>\nNo action required. Wait for BTC to reclaim 50-day EMA."

        print(msg)
        send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
        return

    # 2. Fetch MEXC exchangeInfo for chain metadata
    exchange_info = get_mexc_exchange_info()
    logger.info(f"Loaded exchangeInfo metadata for {len(exchange_info)} symbols.")

    # 3. Scan & Rank Universe
    lookback = cfg.get("lookback_days", 30)
    logger.info(f"Using {lookback}-day lookback period for Relative Strength & Momentum.")

    symbols = get_top_mexc_pairs(min_volume=cfg["min_volume_24h_usdt"])
    if not symbols:
        logger.error("No valid pairs returned.")
        return

    limit_bars = max(lookback + 15, 60)
    df_btc = get_mexc_klines("BTCUSDT", interval="1d", limit=limit_bars)
    if len(df_btc) < lookback + 1:
        logger.error("Insufficient BTC daily bars for lookback.")
        return

    btc_p0 = df_btc.iloc[-(lookback + 1)]["close"]
    btc_p1 = df_btc.iloc[-1]["close"]
    btc_lookback_return = (btc_p1 - btc_p0) / btc_p0
    logger.info(f"Bitcoin {lookback}-day Return: {btc_lookback_return*100:+.2f}%")

    metrics_list = []
    for sym in symbols:
        if sym == "BTCUSDT":
            continue
        df_k = get_mexc_klines(sym, interval="1d", limit=limit_bars)
        if len(df_k) < lookback + 1:
            continue
        p0 = df_k.iloc[-(lookback + 1)]["close"]
        p1 = df_k.iloc[-1]["close"]
        if p0 <= 0 or np.isnan(p0) or np.isnan(p1):
            continue

        alt_ret = (p1 - p0) / p0
        rs_spread = alt_ret - btc_lookback_return

        # Trailing 3-day return (acute recent pump)
        p_3d_ago = df_k.iloc[-4]["close"] if len(df_k) >= 4 else p0
        ret_3d = ((p1 - p_3d_ago) / p_3d_ago) if p_3d_ago > 0 else 0.0

        # Trailing annualized daily volatility over lookback
        daily_rets = df_k["close"].pct_change().dropna().tail(lookback)
        vol_lookback = float(daily_rets.std() * np.sqrt(365.25)) if len(daily_rets) >= max(10, int(lookback * 0.6)) else 1.0
        if np.isnan(vol_lookback) or vol_lookback <= 0.05:
            vol_lookback = 0.05

        sharpe_score = alt_ret / vol_lookback

        metrics_list.append({
            "symbol": sym,
            "latest_close": p1,
            "alt_return_lookback": alt_ret,
            "ret_3d": ret_3d,
            "rs_spread_vs_btc": rs_spread,
            "volatility_lookback": vol_lookback,
            "sharpe_score": sharpe_score
        })
        time.sleep(0.04)

    df_all = pd.DataFrame(metrics_list)
    if df_all.empty:
        logger.error("No valid metrics computed.")
        return

    # 4. Compute Rankings
    # A. Quality Ranking (Sharpe Score = Ret / Vol)
    df_quality = df_all.sort_values(by="sharpe_score", ascending=False).reset_index(drop=True)
    df_quality["rank_quality"] = df_quality.index + 1

    # B. Raw Ranking (Relative Strength Spread vs BTC)
    df_raw = df_all.sort_values(by="rs_spread_vs_btc", ascending=False).reset_index(drop=True)
    df_raw["rank_raw"] = df_raw.index + 1

    # C. Laggard Ranking (Weakest Altcoins vs BTC for Long/Short Barbell)
    df_laggards = df_all.sort_values(by="rs_spread_vs_btc", ascending=True).reset_index(drop=True)
    df_laggards["rank_laggard"] = df_laggards.index + 1

    # Portfolio Sizing
    k_quality = cfg["top_k_quality"]  # 7
    k_raw = cfg["top_k_raw"]          # 3
    buf_quality = cfg["rank_exit_buffer_quality"]  # 11
    buf_raw = cfg["rank_exit_buffer_raw"]          # 6
    max_3d_pump = cfg.get("max_3d_pump_pct", 0.25)  # 25% max 3d pump for new buys

    # 5. Construct Selections and Apply Buffer Rules
    # Quality Buffer & Target
    # Existing holdings: evaluated strictly on rank (Buffer #11), NOT penalized by 3d pump
    quality_buffer_symbols = df_quality.head(buf_quality)["symbol"].tolist()
    holds_quality = [s for s in held_quality if s in quality_buffer_symbols]
    sells_quality = [s for s in held_quality if s not in quality_buffer_symbols]
    slots_needed_quality = k_quality - len(holds_quality)

    # New Buys: apply Solution A1 (anti-exhaustion filter: reject if 3d pump > max_3d_pump)
    buys_quality = []
    skipped_quality_exhaustion = []
    for _, row in df_quality.iterrows():
        s = row["symbol"]
        if s in holds_quality:
            continue
        if len(buys_quality) >= slots_needed_quality:
            break
        if row["ret_3d"] > max_3d_pump:
            skipped_quality_exhaustion.append((s, row["ret_3d"]))
            continue
        buys_quality.append(s)

    final_quality_holdings = holds_quality + buys_quality

    # Raw Buffer & Target (Exclude any symbols already in Quality to prevent overlap!)
    df_raw_filtered = df_raw[~df_raw["symbol"].isin(final_quality_holdings)].reset_index(drop=True)
    raw_buffer_symbols = df_raw_filtered.head(buf_raw)["symbol"].tolist()

    holds_raw = [s for s in held_raw if s in raw_buffer_symbols and s not in final_quality_holdings]
    sells_raw = [s for s in held_raw if s not in holds_raw]
    slots_needed_raw = k_raw - len(holds_raw)

    buys_raw = []
    skipped_raw_exhaustion = []
    for _, row in df_raw_filtered.iterrows():
        s = row["symbol"]
        if s in holds_raw:
            continue
        if len(buys_raw) >= slots_needed_raw:
            break
        if row["ret_3d"] > max_3d_pump:
            skipped_raw_exhaustion.append((s, row["ret_3d"]))
            continue
        buys_raw.append(s)

    final_raw_holdings = holds_raw + buys_raw
    total_new_holdings = final_quality_holdings + final_raw_holdings

    all_sells = list(set(sells_quality + sells_raw))
    all_buys = buys_quality + buys_raw

    # Resolve chains for portfolio holdings and top coins
    coin_chains = {s: resolve_chain(s, exchange_info) for s in total_new_holdings}
    for s in all_sells + all_buys:
        if s not in coin_chains:
            coin_chains[s] = resolve_chain(s, exchange_info)

    # Calculate chain distribution
    chain_counts: Dict[str, int] = {}
    for s in total_new_holdings:
        ch = coin_chains.get(s, "Unknown")
        chain_counts[ch] = chain_counts.get(ch, 0) + 1
    chain_dist_str = " | ".join([f"{ch}: {cnt}" for ch, cnt in sorted(chain_counts.items(), key=lambda x: x[1], reverse=True)])

    # 6. Format Action-Oriented Telegram Alert
    msg = (
        f"🎯 <b>10-COIN BARBELL MOMENTUM SCANNER</b>\n"
        f"📅 <i>{now_utc}</i>\n\n"
        f"<b>BTC Macro Trend:</b> ✅ <b>BULLISH</b> (Above 50 EMA)\n"
        f"• BTC Price: <code>${btc_close:,.2f}</code> (EMA50: <code>${btc_ema50:,.2f}</code>)\n"
        f"• BTC {lookback}d Return: <code>{btc_lookback_return*100:+.1f}%</code>\n"
        f"• Strategy: <b>70% Quality (Sharpe) + 30% Raw (High-Beta)</b>\n"
        f"• ⛓️ <b>Chain Exposure:</b> <i>{chain_dist_str}</i>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>YOUR STEP-BY-STEP ACTION INSTRUCTIONS:</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # Step 1: Sells
    msg += "<b>STEP 1: EXECUTE SELLS (Dropped Below Buffer)</b>\n"
    if all_sells:
        for s in all_sells:
            ch = coin_chains.get(s, "Unknown")
            msg += f"🔴 <b>SELL 100%</b> of <code>{s}</code> [{ch}] to USDT\n"
        msg += "<i>➡️ Consolidate all proceeds into USDT cash.</i>\n\n"
    else:
        msg += "✅ <i>None! No existing coins fell out of buffer.</i>\n\n"

    # Step 2: Buys
    msg += f"<b>STEP 2: EXECUTE BUYS ({len(all_buys)} Slot{'s' if len(all_buys) != 1 else ''} Open)</b>\n"
    if all_buys:
        msg += f"<i>Divide your total available USDT from sells (or fresh capital) equally into {len(all_buys)} parts (~10% per coin):</i>\n"
        for s in buys_quality:
            row = df_quality[df_quality["symbol"] == s].iloc[0]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🟢 <b>BUY</b> <code>{s}</code> [{ch} | Quality #Q{row['rank_quality']} | Ret: {row['alt_return_lookback']*100:+.1f}% | Sharpe: {row['sharpe_score']:.2f}]\n"
        for s in buys_raw:
            row = df_raw[df_raw["symbol"] == s].iloc[0]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🚀 <b>BUY</b> <code>{s}</code> [{ch} | Raw Momentum #R{row['rank_raw']} | Ret: {row['alt_return_lookback']*100:+.1f}% | RS: {row['rs_spread_vs_btc']*100:+.1f}%]\n"
        msg += "<i>💡 If a coin is unavailable or you prefer not to trade its chain, choose the next coin from the <b>Reserve Alternates</b> list below!</i>\n\n"
    else:
        msg += "✨ <i>Portfolio is at full capacity. No new buys required this week!</i>\n\n"

    # Step 3: Holds
    all_holds = holds_quality + holds_raw
    if all_holds:
        msg += f"<b>STEP 3: HOLD SURVIVING COINS</b>\n"
        msg += "<i>• Let winners run! Do NOT trim winning coins unless a single coin grows to <b>>20% of your total portfolio</b>.</i>\n"
        msg += "<i>• If a coin exceeds >20% NAV, trim only the excess back to 10%–12% and redistribute proceeds into open buy slots.</i>\n\n"
        for s in holds_quality:
            r = df_quality[df_quality["symbol"] == s].iloc[0]["rank_quality"]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🟡 <b>HOLD</b> <code>{s}</code> [{ch} | Quality #Q{r}]\n"
        for s in holds_raw:
            r = df_raw[df_raw["symbol"] == s].iloc[0]["rank_raw"]
            ch = coin_chains.get(s, "Unknown")
            msg += f"🟡 <b>HOLD</b> <code>{s}</code> [{ch} | Raw #R{r}]\n"
        msg += "\n"

    # Rules & Reminders
    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <b>STRATEGY EXECUTION RULES:</b>\n"
        f"1. <b>Weekly Cadence:</b> Hold positions until next <b>Monday 00:00 UTC</b>.\n"
        f"2. <b>Anti-Exhaustion Protection:</b> New buys strictly reject coins that surged >25% in the last 72 hours (prevents buying blow-off tops).\n"
        f"3. <b>20% Position Cap:</b> If any single coin reaches >20% of total portfolio, harvest profits back down to 10%–12%.\n"
        f"4. <b>No Mid-Week Panic Exits:</b> Do NOT sell on Tuesday/Wednesday dips. Wait for Sunday night 50-EMA check.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # Leaderboard Summaries
    msg += f"🏆 <b>ACTIVE 10-COIN PORTFOLIO ALLOCATION:</b>\n"
    msg += "<b>Core Quality Momentum (70%):</b>\n"
    for i, s in enumerate(final_quality_holdings, 1):
        row = df_quality[df_quality["symbol"] == s].iloc[0]
        ch = coin_chains.get(s, "Unknown")
        msg += f" {i}. <code>{s:<10s}</code> [{ch:<10s}] +{row['alt_return_lookback']*100:5.1f}% | Sharpe: {row['sharpe_score']:4.2f}\n"

    msg += "\n<b>Speculative Raw Momentum (30%):</b>\n"
    for i, s in enumerate(final_raw_holdings, 1):
        row = df_raw[df_raw["symbol"] == s].iloc[0]
        ch = coin_chains.get(s, "Unknown")
        msg += f" {i}. <code>{s:<10s}</code> [{ch:<10s}] +{row['alt_return_lookback']*100:5.1f}% | RS: {row['rs_spread_vs_btc']*100:+5.1f}%\n"

    # Part 2: Reserves and Depth Leaderboards
    msg_reserves = (
        f"🔄 <b>RESERVE ALTERNATES & DEPTH LEADERBOARDS ({lookback}d Lookback)</b>\n"
        f"📅 <i>{now_utc}</i>\n\n"
        f"<i>Use these replacements if you cannot or prefer not to buy any of the top picks:</i>\n\n"
        f"<b>Quality Alternates (Ranks #8 to #15):</b>\n"
    )
    for _, row in df_quality.iloc[7:15].iterrows():
        s = row["symbol"]
        ch = resolve_chain(s, exchange_info)
        msg_reserves += f"• <code>{s:<10s}</code> [{ch:<10s}] #Q{row['rank_quality']:2d} | +{row['alt_return_lookback']*100:5.1f}% (Sharpe: {row['sharpe_score']:4.2f})\n"

    msg_reserves += "\n<b>Raw Momentum Alternates (Next In Line):</b>\n"
    raw_reserves = df_raw_filtered.iloc[3:10]
    for _, row in raw_reserves.iterrows():
        s = row["symbol"]
        ch = resolve_chain(s, exchange_info)
        msg_reserves += f"• <code>{s:<10s}</code> [{ch:<10s}] #R{row['rank_raw']:2d} | +{row['alt_return_lookback']*100:5.1f}% (RS: {row['rs_spread_vs_btc']*100:+5.1f}%)\n"

    msg_reserves += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg_reserves += f"📊 <b>TOP 15 QUALITY LEADERBOARD ({lookback}d):</b>\n"
    for _, row in df_quality.head(15).iterrows():
        s = row["symbol"]
        ch = resolve_chain(s, exchange_info)
        msg_reserves += f"<b>#Q{row['rank_quality']:2d}</b> <code>{s:<10s}</code> [{ch:<10s}]: {row['alt_return_lookback']*100:+5.1f}% (Sharpe: {row['sharpe_score']:4.2f})\n"

    msg_reserves += f"\n📊 <b>TOP 15 RAW RELATIVE STRENGTH LEADERBOARD ({lookback}d):</b>\n"
    for _, row in df_raw.head(15).iterrows():
        s = row["symbol"]
        ch = resolve_chain(s, exchange_info)
        msg_reserves += f"<b>#R{row['rank_raw']:2d}</b> <code>{s:<10s}</code> [{ch:<10s}]: {row['alt_return_lookback']*100:+5.1f}% (RS: {row['rs_spread_vs_btc']*100:+5.1f}%)\n"

    msg_reserves += f"\n📉 <b>BOTTOM 15 WEAKEST LAGGARDS ({lookback}d SHORT BASKET):</b>\n"
    for _, row in df_laggards.head(15).iterrows():
        s = row["symbol"]
        ch = resolve_chain(s, exchange_info)
        msg_reserves += f"<b>#S{row['rank_laggard']:2d}</b> <code>{s:<10s}</code> [{ch:<10s}]: {row['alt_return_lookback']*100:+5.1f}% (RS: {row['rs_spread_vs_btc']*100:+5.1f}%)\n"

    print("=" * 60)
    print(msg)
    print("=" * 60)
    print(msg_reserves)

    save_portfolio_state(final_quality_holdings, final_raw_holdings)

    # Dispatch Part 1 (Main Action Orders) and Part 2 (Reserves/Leaderboards)
    if not cfg.get("no_telegram", False):
        send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
        time.sleep(1.0)
        send_telegram_message(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg_reserves)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weekly Relative Strength & Momentum Rotation Scanner")
    parser.add_argument("--lookback", type=int, default=None, help="Lookback period in days (e.g. 30, 21, 14, 7)")
    parser.add_argument("--min-volume", type=float, default=None, help="Minimum 24h volume in USDT (e.g. 500000)")
    parser.add_argument("--bypass-btc-gate", action="store_true", help="Show rankings even if BTC is below 50-day EMA")
    parser.add_argument("--no-telegram", action="store_true", help="Print to console only without sending Telegram alerts")
    args = parser.parse_args()

    cfg = load_config()
    if args.lookback is not None:
        cfg["lookback_days"] = args.lookback
    if args.min_volume is not None:
        cfg["min_volume_24h_usdt"] = args.min_volume
    if args.bypass_btc_gate:
        cfg["bypass_btc_gate"] = True
    if args.no_telegram:
        cfg["no_telegram"] = True

    run_scanner(cfg_override=cfg)


if __name__ == "__main__":
    main()

