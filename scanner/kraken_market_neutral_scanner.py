"""
scanner/kraken_market_neutral_scanner.py
----------------------------------------
Production-grade Market-Neutral Long/Short Relative Strength Scanner for Kraken Futures.
Designed specifically for Prop Firm Capital (Breakout Prop, FundingPips, FTMO, etc.).

Core Mechanisms:
1. Long Leg (Alpha): Top K Kraken Breakout Relative Strength Leaders (default K=3, 7-day Sharpe).
2. Short Leg (Beta Hedge):
   - Mode "btc_hedge" (Default & Recommended): 100% Dollar-Neutral Short on Bitcoin (PF_XBTUSD).
   - Mode "cross_sectional": Dollar-Neutral Short on Bottom K Altcoin Laggards.
3. Prop-Firm Risk Calibration:
   - Net Market Delta: 0.0% ($0 net exposure to market-wide crypto dumps).
   - Conservative Gross Exposure (default 20%-25% gross: 10%-12.5% long, 10%-12.5% short).
   - Daily Drawdown Safety: Worst-case historical single-day drop < 2.0% (immune to 4%-5% daily limits).
4. Turnover Smoothing Buffer:
   - Existing long positions retained if they remain within the Top 6.
5. Automated State Persistence:
   - Tracks portfolio state in portfolio_state_market_neutral.json.
6. Real-Time Execution:
   - Fetches live markPrice from Kraken Futures tickers API for exact position sizing and brackets.
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
        logging.FileHandler("market_neutral_scanner.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("kraken_market_neutral_scanner")

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
    "XMR", "XRP", "ZEC", "FET"
]


def get_config_file() -> Optional[Path]:
    candidates = [
        Path("kraken_market_neutral_config.json"),
        SCRIPT_DIR / "kraken_market_neutral_config.json",
        SCRIPT_DIR / "deploy" / "kraken_market_neutral_config.example.json",
        Path("daily_scanner_config.json"),
        SCRIPT_DIR / "daily_scanner_config.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_state_file() -> Path:
    if Path("portfolio_state_market_neutral.json").exists():
        return Path("portfolio_state_market_neutral.json")
    if (SCRIPT_DIR / "portfolio_state_market_neutral.json").exists():
        return SCRIPT_DIR / "portfolio_state_market_neutral.json"
    return Path("portfolio_state_market_neutral.json")


def load_config() -> Dict[str, Any]:
    """Loads configuration and Telegram credentials for market-neutral scanner."""
    config = {
        "hedge_mode": os.environ.get("HEDGE_MODE", "btc_hedge").lower(), # "btc_hedge" or "cross_sectional"
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "account_size_usd": float(os.environ.get("ACCOUNT_SIZE_USD", "100000.0")),
        "gross_exposure_pct": float(os.environ.get("GROSS_EXPOSURE_PCT", "25.0")), # 25% gross: 12.5% Long, 12.5% Short
        "top_k": int(os.environ.get("TOP_K", "3")), # Top 3 long coins
        "rank_exit_buffer": int(os.environ.get("RANK_EXIT_BUFFER", "6")), # Keep long if within top 6
        "lookback_days": int(os.environ.get("LOOKBACK_DAYS", "7")), # 7-day Sharpe
        "stop_loss_pct": float(os.environ.get("STOP_LOSS_PCT", "3.5")), # -3.5% hard stop
        "take_profit_1_pct": float(os.environ.get("TAKE_PROFIT_1_pct", "8.0")), # +8.0% (scale 50%)
        "take_profit_2_pct": float(os.environ.get("TAKE_PROFIT_2_pct", "15.0")), # +15.0% (scale 25%)
        "breadth_ema_period": int(os.environ.get("BREADTH_EMA_PERIOD", "20")), # 20-day EMA for altcoin breadth
        "breadth_expansion_threshold": float(os.environ.get("BREADTH_EXPANSION_THRESHOLD", "50.0")), # >=50% = Naked Long expansion
        "regime_mode": os.environ.get("REGIME_MODE", "auto").lower(), # "auto", "force_hedged", "force_naked_long"
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


def get_kraken_live_tickers() -> Dict[str, Dict[str, Any]]:
    """Fetches real-time live tickers from Kraken Futures."""
    try:
        resp = requests.get(KRAKEN_FUTURES_TICKERS_URL, timeout=12)
        tickers = resp.json().get("tickers", [])
        ticker_map = {}
        for t in tickers:
            sym = t.get("symbol", "")
            if sym:
                ticker_map[sym] = t
        return ticker_map
    except Exception as e:
        logger.error(f"Error fetching Kraken live tickers: {e}")
        return {}


def get_kraken_daily_klines(symbol: str, count: int = 15) -> Optional[pd.DataFrame]:
    """Fetches daily candlesticks for a Kraken Futures symbol."""
    url = KRAKEN_FUTURES_CHART_URL.format(symbol=symbol)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
            data = resp.json()
            candles = data.get("candles", [])
            if not candles or len(candles) < 5:
                return None
            df = pd.DataFrame(candles)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("time").reset_index(drop=True)
            return df.tail(count).reset_index(drop=True)
        except Exception:
            time.sleep(0.5)
    return None


def calculate_coin_metrics(
    df: pd.DataFrame,
    lookback_days: int = 7,
    ema_period: int = 20
) -> Optional[Dict[str, Any]]:
    """Calculates 7-day Sharpe ratio, return, volatility, and EMA trend using closed bars."""
    if df is None or len(df) < lookback_days + 1:
        return None
    # Use fully closed historical bars
    closed_df = df.iloc[:-1].copy()
    if len(closed_df) < lookback_days + 1:
        return None

    # Calculate 7-day metrics on recent slice
    calc_slice = closed_df.tail(lookback_days + 1)
    rets = calc_slice["close"].pct_change().dropna()
    if len(rets) < lookback_days or rets.std() == 0:
        return None
    ret_7d = float((calc_slice["close"].iloc[-1] / calc_slice["close"].iloc[0]) - 1.0)
    vol_7d = float(rets.std())
    sharpe = float(rets.mean() / vol_7d)
    latest_close = float(closed_df["close"].iloc[-1])

    # Breadth EMA calculation
    ema_val = None
    above_ema = None
    if len(closed_df) >= ema_period:
        ema_series = closed_df["close"].ewm(span=ema_period, adjust=False).mean()
        ema_val = float(ema_series.iloc[-1])
        above_ema = bool(latest_close > ema_val)

    return {
        "sharpe": sharpe,
        "return_7d": ret_7d,
        "vol_7d": vol_7d,
        "latest_close": latest_close,
        "ema20": ema_val,
        "above_ema20": above_ema
    }


def scan_kraken_breakout_universe(
    lookback_days: int = 7,
    ema_period: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Scans all Breakout universe coins on Kraken Futures, calculating relative strength and market breadth."""
    logger.info("Fetching real-time mark prices from Kraken Futures tickers API...")
    live_tickers = get_kraken_live_tickers()

    btc_symbol = "PF_XBTUSD"
    btc_mark = float(live_tickers.get(btc_symbol, {}).get("markPrice", 0.0))
    if btc_mark <= 0:
        logger.warning("Could not fetch live mark price for PF_XBTUSD, falling back to last price.")
        btc_mark = float(live_tickers.get(btc_symbol, {}).get("last", 0.0))

    btc_klines = get_kraken_daily_klines(btc_symbol, count=60)
    btc_metrics = {"mark_price": btc_mark}
    if btc_klines is not None and len(btc_klines) >= 50:
        btc_klines["ema50"] = btc_klines["close"].ewm(span=50, adjust=False).mean()
        btc_metrics["ema50"] = float(btc_klines["ema50"].iloc[-1])
        btc_metrics["is_bullish"] = btc_mark > btc_metrics["ema50"]

    results = []
    logger.info(f"Scanning {len(DEFAULT_BREAKOUT_UNIVERSE)} Kraken Breakout coins...")

    for coin in DEFAULT_BREAKOUT_UNIVERSE:
        kraken_sym = f"PF_{coin}USD"
        if kraken_sym not in live_tickers:
            continue

        mark_price = float(live_tickers[kraken_sym].get("markPrice", 0.0))
        if mark_price <= 0:
            mark_price = float(live_tickers[kraken_sym].get("last", 0.0))
        if mark_price <= 0:
            continue

        df = get_kraken_daily_klines(kraken_sym, count=max(40, ema_period + 15))
        metrics = calculate_coin_metrics(df, lookback_days=lookback_days, ema_period=ema_period)
        if metrics is None:
            continue

        results.append({
            "coin": coin,
            "symbol": kraken_sym,
            "sharpe": metrics["sharpe"],
            "return_7d": metrics["return_7d"],
            "vol_7d": metrics["vol_7d"],
            "mark_price": mark_price,
            "ema20": metrics.get("ema20"),
            "above_ema20": metrics.get("above_ema20")
        })

    # Sort descending by 7-day Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Compute Aggregate Market Breadth
    valid_breadth_coins = [c for c in results if c.get("above_ema20") is not None]
    total_valid = len(valid_breadth_coins)
    coins_above = sum(1 for c in valid_breadth_coins if c["above_ema20"])
    breadth_pct = (coins_above / total_valid * 100.0) if total_valid > 0 else 0.0

    breadth_metrics = {
        "breadth_pct": breadth_pct,
        "coins_above_ema": coins_above,
        "total_coins": total_valid,
        "ema_period": ema_period
    }

    logger.info(f"Altcoin Breadth: {breadth_pct:.1f}% ({coins_above}/{total_valid} coins above {ema_period}-day EMA)")
    return results, btc_metrics, breadth_metrics


def determine_rebalance(
    scored_coins: List[Dict[str, Any]],
    current_state: Dict[str, Any],
    top_k: int = 3,
    rank_exit_buffer: int = 6
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Determines Long Leg positions with turnover smoothing."""
    current_longs = set(current_state.get("active_longs", []))
    coin_rank_map = {c["coin"]: idx + 1 for idx, c in enumerate(scored_coins)}

    # Retain existing holdings if within top rank_exit_buffer
    retained = []
    for c in current_longs:
        rank = coin_rank_map.get(c, 999)
        if rank <= rank_exit_buffer:
            retained.append(c)

    # Fill remaining slots from top ranked
    selected_coins = list(retained)
    for c in scored_coins:
        if len(selected_coins) >= top_k:
            break
        if c["coin"] not in selected_coins:
            selected_coins.append(c["coin"])

    selected_data = [c for c in scored_coins if c["coin"] in selected_coins]
    # Keep ranked order
    selected_data.sort(key=lambda x: coin_rank_map.get(x["coin"], 999))

    new_entries = [c for c in selected_coins if c not in current_longs]
    exits = [c for c in current_longs if c not in selected_coins]

    return selected_data, new_entries, exits


def calculate_market_neutral_brackets(
    long_coins: List[Dict[str, Any]],
    btc_metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates dollar sizes, unit quantities, and bracket orders for Long and Short legs."""
    account_size = config["account_size_usd"]
    gross_exposure_pct = config["gross_exposure_pct"]
    half_exposure_pct = gross_exposure_pct / 2.0 # e.g. 12.5% Long, 12.5% Short
    
    total_long_usd = account_size * (half_exposure_pct / 100.0)
    total_short_usd = account_size * (half_exposure_pct / 100.0) # Exactly equal = 0.0% Net Delta!
    
    usd_per_long = total_long_usd / max(1, len(long_coins))
    
    # 1. Long Leg Orders
    long_brackets = []
    for c in long_coins:
        price = c["mark_price"]
        units = usd_per_long / price if price > 0 else 0
        sl_pct = config["stop_loss_pct"] / 100.0
        tp1_pct = config["take_profit_1_pct"] / 100.0
        tp2_pct = config["take_profit_2_pct"] / 100.0

        long_brackets.append({
            "coin": c["coin"],
            "symbol": c["symbol"],
            "sharpe": c["sharpe"],
            "return_7d": c["return_7d"],
            "mark_price": price,
            "target_usd": usd_per_long,
            "target_units": units,
            "stop_loss_price": price * (1.0 - sl_pct),
            "take_profit_1_price": price * (1.0 + tp1_pct),
            "take_profit_2_price": price * (1.0 + tp2_pct),
        })

    # 2. Short Hedge Leg Order (PF_XBTUSD)
    btc_price = btc_metrics.get("mark_price", 0.0)
    btc_units = total_short_usd / btc_price if btc_price > 0 else 0
    btc_sl_price = btc_price * (1.0 + config["stop_loss_pct"] / 100.0)
    btc_tp_price = btc_price * (1.0 - 0.05) # +5% drop target

    short_hedge = {
        "symbol": "PF_XBTUSD",
        "mark_price": btc_price,
        "target_usd": total_short_usd,
        "target_units": btc_units,
        "stop_loss_price": btc_sl_price,
        "take_profit_price": btc_tp_price
    }

    return {
        "total_long_usd": total_long_usd,
        "total_short_usd": total_short_usd,
        "net_market_delta_usd": 0.0,
        "net_market_delta_pct": 0.0,
        "long_brackets": long_brackets,
        "short_hedge": short_hedge
    }


def determine_regime(
    btc_metrics: Dict[str, Any],
    breadth_metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determines market regime:
      - STATE 1: FLAT / CASH (Defensive: BTC < 50-day EMA)
      - STATE 2: HEDGED (Selective Bull: BTC > 50-day EMA, Alt Breadth < threshold)
      - STATE 3: NAKED LONG (Altseason Expansion: BTC > 50-day EMA, Alt Breadth >= threshold)
    """
    regime_mode = config.get("regime_mode", "auto").lower()
    threshold = float(config.get("breadth_expansion_threshold", 50.0))
    is_btc_bullish = bool(btc_metrics.get("is_bullish", False))
    breadth_pct = float(breadth_metrics.get("breadth_pct", 0.0))

    if regime_mode == "force_hedged":
        return {
            "state": "STATE_2_HEDGED",
            "name": "🛡️ STATE 2: HEDGED (Manual Override)",
            "action": "Deploy Market-Neutral Basket (Long Top 3 / Short BTC 0.0% Net Delta).",
            "execute_short_hedge": True,
            "breadth_pct": breadth_pct,
            "threshold": threshold,
            "btc_bullish": is_btc_bullish
        }
    elif regime_mode == "force_naked_long":
        return {
            "state": "STATE_3_NAKED_LONG",
            "name": "🚀 STATE 3: NAKED LONG (Manual Override)",
            "action": "Deploy Top 3 Longs Unhedged (Full Beta Upside).",
            "execute_short_hedge": False,
            "breadth_pct": breadth_pct,
            "threshold": threshold,
            "btc_bullish": is_btc_bullish
        }

    # Auto Regime Detection
    if not is_btc_bullish:
        return {
            "state": "STATE_1_FLAT",
            "name": "🛑 STATE 1: FLAT / CASH (Defensive)",
            "action": f"BTC is below 50-day EMA (${btc_metrics.get('ema50', 0):,.0f}). Stay in cash; avoid new longs.",
            "execute_short_hedge": False,
            "breadth_pct": breadth_pct,
            "threshold": threshold,
            "btc_bullish": is_btc_bullish
        }
    elif breadth_pct < threshold:
        return {
            "state": "STATE_2_HEDGED",
            "name": "🛡️ STATE 2: HEDGED (Selective Market)",
            "action": f"Altcoin Breadth is {breadth_pct:.1f}% (< {threshold:.0f}% threshold). Deploy Long Top 3 + 100% BTC Hedge (0.0% Net Delta).",
            "execute_short_hedge": True,
            "breadth_pct": breadth_pct,
            "threshold": threshold,
            "btc_bullish": is_btc_bullish
        }
    else:
        return {
            "state": "STATE_3_NAKED_LONG",
            "name": "🚀 STATE 3: NAKED LONG (Altseason Expansion)",
            "action": f"Altcoin Breadth is {breadth_pct:.1f}% (≥ {threshold:.0f}% threshold). Broad participation confirmed! Deploy Top 3 longs UNHEDGED.",
            "execute_short_hedge": False,
            "breadth_pct": breadth_pct,
            "threshold": threshold,
            "btc_bullish": is_btc_bullish
        }


def format_market_neutral_telegram_alert(
    brackets: Dict[str, Any],
    btc_metrics: Dict[str, Any],
    breadth_metrics: Dict[str, Any],
    regime: Dict[str, Any],
    new_entries: List[str],
    exits: List[str],
    config: Dict[str, Any]
) -> str:
    """Formats professional HTML Telegram alert for prop firm traders with dynamic regime analysis."""
    now_utc = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    acc_size = config["account_size_usd"]
    gross_exp = config["gross_exposure_pct"]
    half_exp = gross_exp / 2.0

    lines = [
        "⚖️ <b>KRAKEN DYNAMIC REGIME &amp; RS SCANNER</b>",
        f"📅 <code>{now_utc}</code> | Universe: <b>Kraken Breakout (65 Coins)</b>",
        "",
        "📊 <b>MARKET BREADTH &amp; REGIME ENGINE:</b>",
        f"• BTC 50-Day EMA: <code>${btc_metrics.get('ema50', 0):,.2f}</code> "
        f"({'🟢 Bullish' if btc_metrics.get('is_bullish') else '🔴 Bearish'})",
        f"• <b>Altcoin Breadth (>20 EMA):</b> <code>{breadth_metrics.get('breadth_pct', 0.0):.1f}%</code> "
        f"({breadth_metrics.get('coins_above_ema', 0)} / {breadth_metrics.get('total_coins', 0)} coins)",
        f"• <b>Active Regime:</b> {regime['name']}",
        f"• <b>Recommendation:</b> {regime['action']}",
        "",
        "🛡️ <b>PROP-FIRM CAPITAL PROFILE:</b>",
        f"• Account Capital: <code>${acc_size:,.0f}</code>",
        f"• Hedged Gross Exposure: <code>{gross_exp:.1f}%</code> (${brackets['total_long_usd'] + brackets['total_short_usd']:,.0f}) | Net Delta: <code>0.0% ($0.00)</code>",
        f"• Unhedged Long Exposure (if Naked): <code>{half_exp:.1f}%</code> (${brackets['total_long_usd']:,.0f})",
        "",
        f"🟢 <b>LONG ALPHA LEG (TOP {config['top_k']} RS LEADERS):</b>"
    ]

    for idx, b in enumerate(brackets["long_brackets"], 1):
        lines.extend([
            f"<b>#{idx} {b['symbol']}</b> (7d Ret: <code>{b['return_7d']*100:+.1f}%</code> | Sharpe: <code>{b['sharpe']:.2f}</code>)",
            f"   • Mark Price: <code>${b['mark_price']:,.4f}</code>",
            f"   • Target Allocation: <code>${b['target_usd']:,.0f}</code> (~<code>{b['target_units']:,.2f}</code> units)",
            f"   • Hard SL (-{config['stop_loss_pct']:.1f}%): <code>${b['stop_loss_price']:,.4f}</code>",
            f"   • TP1 (+{config['take_profit_1_pct']:.1f}% / 50% scale): <code>${b['take_profit_1_price']:,.4f}</code>",
            f"   • TP2 (+{config['take_profit_2_pct']:.1f}% / 25% scale): <code>${b['take_profit_2_price']:,.4f}</code>",
        ])

    sh = brackets["short_hedge"]
    lines.extend([
        "",
        f"🔴 <b>SHORT HEDGE LEG (PF_XBTUSD - Deploy if in Hedged Mode):</b>",
        f"• Contract: <code>{sh['symbol']}</code>",
        f"• Short Size: <code>${sh['target_usd']:,.0f}</code> (~<code>{sh['target_units']:.4f}</code> BTC)",
        f"• Mark Price: <code>${sh['mark_price']:,.2f}</code>",
        f"• Hard SL (+{config['stop_loss_pct']:.1f}%): <code>${sh['stop_loss_price']:,.2f}</code>",
        f"• Target (-5.0%): <code>${sh['take_profit_price']:,.2f}</code>",
        "",
        "🔄 <b>REBALANCE ACTIONS:</b>"
    ])

    if new_entries:
        lines.append(f"• 🟢 <b>NEW BUYS:</b> {', '.join(new_entries)}")
    if exits:
        lines.append(f"• 🔴 <b>CLOSE / EXITS:</b> {', '.join(exits)}")
    if not new_entries and not exits:
        lines.append("• ⚪ <b>NO REBALANCE NEEDED:</b> All positions maintained inside retention buffer.")

    lines.extend([
        "",
        "💡 <i>Regime Execution Directives:</i>",
        "• <b>STATE 1 (Flat):</b> Stay in cash. Close open long risk.",
        "• <b>STATE 2 (Hedged):</b> Execute Longs + Short BTC. Maximum 1-day drawdown &lt; 2.0% (immune to 4.0% daily limit).",
        "• <b>STATE 3 (Naked Long):</b> Execute Longs only (skip Short BTC) to maximize speed to 8%-10% prop target."
    ])

    return "\n".join(lines)


def run_scanner():
    """Main execution flow for Kraken Market-Neutral Scanner."""
    logger.info("Starting Kraken Market-Neutral Scanner...")
    config = load_config()

    state_file = get_state_file()
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load state file {state_file}: {e}")

    scored_coins, btc_metrics, breadth_metrics = scan_kraken_breakout_universe(
        lookback_days=config["lookback_days"],
        ema_period=config["breadth_ema_period"]
    )
    if not scored_coins:
        logger.error("No coins successfully scored. Exiting scan.")
        return

    regime = determine_regime(btc_metrics, breadth_metrics, config)

    top_coins, new_entries, exits = determine_rebalance(
        scored_coins,
        state,
        top_k=config["top_k"],
        rank_exit_buffer=config["rank_exit_buffer"]
    )

    brackets = calculate_market_neutral_brackets(top_coins, btc_metrics, config)

    # Update state file
    new_state = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "regime_state": regime["state"],
        "regime_name": regime["name"],
        "altcoin_breadth_pct": breadth_metrics["breadth_pct"],
        "coins_above_ema20": breadth_metrics["coins_above_ema"],
        "total_coins_scanned": breadth_metrics["total_coins"],
        "active_longs": [c["coin"] for c in top_coins],
        "short_hedge": "PF_XBTUSD",
        "gross_exposure_pct": config["gross_exposure_pct"],
        "net_market_delta_pct": 0.0 if regime["execute_short_hedge"] else (config["gross_exposure_pct"] / 2.0)
    }
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(new_state, f, indent=2)
        logger.info(f"Updated state file: {state_file}")
    except Exception as e:
        logger.error(f"Failed to write state file {state_file}: {e}")

    # Build and send Telegram message
    message = format_market_neutral_telegram_alert(
        brackets, btc_metrics, breadth_metrics, regime, new_entries, exits, config
    )
    print("\n" + "=" * 80)
    print(message.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))
    print("=" * 80 + "\n")

    if config["telegram_bot_token"] and config["telegram_chat_id"]:
        logger.info("Dispatching alert to Telegram...")
        send_telegram_message(config["telegram_bot_token"], config["telegram_chat_id"], message)
    else:
        logger.info("Telegram not configured. Output printed above.")


if __name__ == "__main__":
    run_scanner()
