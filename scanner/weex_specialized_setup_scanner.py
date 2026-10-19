"""
scanner/weex_specialized_setup_scanner.py
-----------------------------------------
Production Real-Time Scanner using live WEEX Contract API data.
Geared specifically to the empirically proven best coins for each foundational setup:

1. SOL & SUI: Weekly Open Reclaim & Swing-High Retest (Institutional Momentum Continuation)
2. DOGE & FLOKI: 20-Day Donchian Breakout (Meme Momentum Expansion)
3. XRP, LTC & DASH: Lower Bollinger Band Reversal (Old-Guard Mean Reversion)
4. ONDO & GMX: Turtle Soup 20-Day Liquidity Sweeps (RWA & DeFi Stop Runs)
5. ZIL & ZEC: High-Volume Exhaustion Absorption Pin Bars (Climax Wicks)

Outputs:
- Rich formatted console summary
- Optional Telegram alerts
- Saved JSON snapshot to results/weex_live_signals.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import numpy as np

# Set UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("weex_scanner")

WEEX_BASE_URL = "https://api-contract.weex.com"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = ROOT_DIR / "weex_scanner_config.json"
OUT_DIR = ROOT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_FILE = OUT_DIR / "weex_live_signals.json"

def load_scanner_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "send_daily_digest": True
    }

def send_telegram_alert(text: str, token: str, chat_id: str) -> bool:
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
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")
        return False


# Coin-to-strategy specialized mapping
SPECIALIZED_UNIVERSE = {
    # 1. Weekly Open Reclaim / Swing Retest Leaders
    "SUIUSDT": {
        "coin": "SUI",
        "archetype": "Momentum / Ecosystem Leader",
        "strategies": ["Weekly_Open_Reclaim", "Swing_Retest"],
        "historical_stats": "65% Win Rate | 2.81 PF on Weekly Reclaim; +2.96% Exp on Retest"
    },
    "SOLUSDT": {
        "coin": "SOL",
        "archetype": "Momentum / Ecosystem Leader",
        "strategies": ["Weekly_Open_Reclaim", "Swing_Retest", "Donchian_Breakout"],
        "historical_stats": "53.3% Win Rate | 2.48 PF on Retest; 4.33 PF on Donchian"
    },
    # 2. Donchian Trend Expansion Memes
    "DOGEUSDT": {
        "coin": "DOGE",
        "archetype": "High-Beta Meme Expansion",
        "strategies": ["Donchian_Breakout", "Swing_Retest"],
        "historical_stats": "+17.7% Exp | 3.62 PF on Donchian Breakout"
    },
    "1000FLOKIUSDT": {
        "coin": "FLOKI",
        "archetype": "High-Beta Meme Expansion",
        "strategies": ["Donchian_Breakout", "Swing_Retest"],
        "historical_stats": "+26.4% Exp | 4.51 PF on Donchian Breakout"
    },
    # 3. Bollinger Band & Mean-Reversion Legacy Dinosaurs
    "XRPUSDT": {
        "coin": "XRP",
        "archetype": "Range-Bound Mean Reverter",
        "strategies": ["BB_Mean_Reversion"],
        "historical_stats": "73.6% Win Rate | 1.64 PF on Lower BB Bounces"
    },
    "LTCUSDT": {
        "coin": "LTC",
        "archetype": "Range-Bound Mean Reverter",
        "strategies": ["BB_Mean_Reversion", "Turtle_Soup"],
        "historical_stats": "66.7% Win Rate | 1.47 PF on Lower BB Bounces"
    },
    "DASHUSDT": {
        "coin": "DASH",
        "archetype": "Range-Bound Mean Reverter",
        "strategies": ["BB_Mean_Reversion"],
        "historical_stats": "75.5% Win Rate | 2.16 PF on Lower BB Bounces"
    },
    # 4. Turtle Soup Liquidity Sweeps (False Breakdowns)
    "ONDOUSDT": {
        "coin": "ONDO",
        "archetype": "DeFi / RWA Liquidity Hunter",
        "strategies": ["Turtle_Soup"],
        "historical_stats": "66.7% Win Rate | 2.40 PF on False Breakdowns"
    },
    "GMXUSDT": {
        "coin": "GMX",
        "archetype": "DeFi / RWA Liquidity Hunter",
        "strategies": ["Turtle_Soup"],
        "historical_stats": "52.4% Win Rate | 2.32 PF on False Breakdowns"
    },
    # 5. Climax Absorption Wicks
    "ZILUSDT": {
        "coin": "ZIL",
        "archetype": "Capitulation Absorption",
        "strategies": ["Volume_Absorption"],
        "historical_stats": "68.2% Win Rate | 3.38 PF on Climax Lower Wicks"
    },
    "ZECUSDT": {
        "coin": "ZEC",
        "archetype": "Capitulation Absorption",
        "strategies": ["Volume_Absorption", "Donchian_Breakout"],
        "historical_stats": "47.1% Win Rate | 2.45 PF on Climax Lower Wicks"
    }
}

def fetch_weex_klines(symbol: str, interval: str = "1d", limit: int = 40) -> Optional[pd.DataFrame]:
    url = f"{WEEX_BASE_URL}/capi/v3/market/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            logger.warning(f"Failed to fetch {symbol} from WEEX: HTTP {r.status_code}")
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 25:
            return None
        
        # WEEX returns newest first: [open_time, open, high, low, close, volume, ...]
        rows = []
        for item in data:
            rows.append({
                "open_time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            })
        df = pd.DataFrame(rows)
        # Sort chronologically (oldest to newest)
        df = df.sort_values("open_time").reset_index(drop=True)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return None

def check_setups_for_coin(symbol: str, df: pd.DataFrame, strat_list: List[str]) -> List[Dict[str, Any]]:
    signals = []
    if len(df) < 25:
        return signals
        
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Weekly Open Reclaim
    if "Weekly_Open_Reclaim" in strat_list and len(df) >= 8:
        # 7 bars ago open serves as cyclical weekly open reference
        weekly_open = df["open"].iloc[-7]
        # Condition: traded below weekly open during the week, but current price reclaimed above it
        dipped_below = df["low"].iloc[-6:].min() < weekly_open
        reclaimed = (curr["close"] > weekly_open) and (curr["close"] >= curr["open"])
        dist_from_wo = (curr["close"] - weekly_open) / weekly_open
        if dipped_below and reclaimed and (0.0 <= dist_from_wo <= 0.045):
            signals.append({
                "strategy": "Weekly Open Reclaim",
                "bias": "BULLISH",
                "trigger_price": curr["close"],
                "key_level": round(weekly_open, 4),
                "action": f"Buy reclaim above Weekly Open ({weekly_open:.4f}). SL below weekly low.",
                "status": "ACTIVE_TRIGGER"
            })
            
    # 2. Swing High Retest
    if "Swing_Retest" in strat_list and len(df) >= 20:
        # Prior 10-period swing high before current bar
        prior_swing_high = df["high"].iloc[-15:-2].max()
        broke_out = df["close"].iloc[-2] > prior_swing_high or df["high"].iloc[-2] > prior_swing_high
        retesting = (curr["low"] <= prior_swing_high * 1.015) and (curr["close"] >= prior_swing_high * 0.99)
        if broke_out and retesting:
            signals.append({
                "strategy": "Swing High Retest",
                "bias": "BULLISH",
                "trigger_price": curr["close"],
                "key_level": round(prior_swing_high, 4),
                "action": f"Dip into broken swing high ({prior_swing_high:.4f}) holding. Long with 2R target.",
                "status": "ACTIVE_TRIGGER"
            })

    # 3. Donchian 20D Breakout
    if "Donchian_Breakout" in strat_list and len(df) >= 21:
        donchian_20_high = df["high"].iloc[-21:-1].max()
        donchian_10_low = df["low"].iloc[-11:-1].min()
        if curr["close"] > donchian_20_high:
            signals.append({
                "strategy": "Donchian 20D Breakout",
                "bias": "BULLISH MOMENTUM",
                "trigger_price": curr["close"],
                "key_level": round(donchian_20_high, 4),
                "action": f"New 20-day high printed ({donchian_20_high:.4f} broken). Trail SL via 10D low ({donchian_10_low:.4f}).",
                "status": "ACTIVE_TRIGGER"
            })
        elif (donchian_20_high - curr["close"]) / curr["close"] <= 0.025:
            # Approaching breakout
            signals.append({
                "strategy": "Donchian 20D Breakout",
                "bias": "WATCHLIST",
                "trigger_price": curr["close"],
                "key_level": round(donchian_20_high, 4),
                "action": f"Within 2.5% of 20-day high ({donchian_20_high:.4f}). Prepare buy-stop on daily breakout.",
                "status": "PENDING_SETUP"
            })

    # 4. Bollinger Band Mean Reversion
    if "BB_Mean_Reversion" in strat_list and len(df) >= 21:
        sma20 = df["close"].iloc[-21:-1].mean()
        std20 = df["close"].iloc[-21:-1].std()
        lower_bb = sma20 - 2.0 * std20
        touched_lower = curr["low"] <= lower_bb or prev["low"] <= lower_bb
        bullish_reaction = curr["close"] > curr["open"]
        if touched_lower and bullish_reaction:
            signals.append({
                "strategy": "Bollinger Band Reversion",
                "bias": "BULLISH MEAN REVERSION",
                "trigger_price": curr["close"],
                "key_level": round(lower_bb, 4),
                "action": f"Lower BB pierced ({lower_bb:.4f}) and closing green. Target 20 SMA mean ({sma20:.4f}).",
                "status": "ACTIVE_TRIGGER"
            })
        elif curr["low"] <= lower_bb * 1.02:
            signals.append({
                "strategy": "Bollinger Band Reversion",
                "bias": "WATCHLIST",
                "trigger_price": curr["close"],
                "key_level": round(lower_bb, 4),
                "action": f"Approaching lower BB ({lower_bb:.4f}). Await bullish reversal bar.",
                "status": "PENDING_SETUP"
            })

    # 5. Turtle Soup 20D Liquidity Sweep
    if "Turtle_Soup" in strat_list and len(df) >= 21:
        donchian_20_low = df["low"].iloc[-21:-1].min()
        swept_low = curr["low"] < donchian_20_low
        closed_inside = curr["close"] > donchian_20_low
        if swept_low and closed_inside:
            signals.append({
                "strategy": "Turtle Soup Liquidity Sweep",
                "bias": "BULLISH REVERSAL",
                "trigger_price": curr["close"],
                "key_level": round(donchian_20_low, 4),
                "action": f"20D low ({donchian_20_low:.4f}) was swept and rejected. Long false breakdown, SL below wick.",
                "status": "ACTIVE_TRIGGER"
            })

    # 6. High-Volume Exhaustion Wick (Pin Bar Absorption)
    if "Volume_Absorption" in strat_list and len(df) >= 21:
        body = abs(curr["close"] - curr["open"])
        total_range = max(curr["high"] - curr["low"], 1e-8)
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        vol_avg = df["volume"].iloc[-21:-1].mean()
        is_pinbar = (lower_wick / total_range) >= 0.55
        high_vol = curr["volume"] >= 1.8 * vol_avg
        if is_pinbar and high_vol:
            signals.append({
                "strategy": "Climax Volume Absorption",
                "bias": "BULLISH ABSORPTION",
                "trigger_price": curr["close"],
                "key_level": round(curr["low"], 4),
                "action": f"Heavy volume absorption wick (vol {curr['volume']/vol_avg:.1f}x avg). Long with SL below {curr['low']:.4f}.",
                "status": "ACTIVE_TRIGGER"
            })

    return signals

def run_scan() -> Dict[str, Any]:
    print("=" * 90)
    print("WEEX SPECIALIZED SETUP SCANNER — REAL-TIME CONTRACT MARKET DATA")
    print(f"Time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    print("=" * 90)
    
    active_triggers = []
    pending_watchlists = []
    
    for symbol, meta in SPECIALIZED_UNIVERSE.items():
        time.sleep(0.05)  # respectful rate limit
        df = fetch_weex_klines(symbol, interval="1d", limit=35)
        if df is None:
            continue
            
        coin_signals = check_setups_for_coin(symbol, df, meta["strategies"])
        last_price = df["close"].iloc[-1]
        
        for sig in coin_signals:
            record = {
                "symbol": symbol,
                "coin": meta["coin"],
                "archetype": meta["archetype"],
                "historical_edge": meta["historical_stats"],
                "current_price": last_price,
                **sig
            }
            if sig["status"] == "ACTIVE_TRIGGER":
                active_triggers.append(record)
            else:
                pending_watchlists.append(record)
                
    # Display Active Triggers
    print("\n>>> ACTIVE READY-TO-TRADE TRIGGERS (Live on WEEX) <<<")
    if not active_triggers:
        print("  No setups are currently in active trigger bar status. Check watchlist below.")
    else:
        for t in active_triggers:
            print(f"\n[ALERT] {t['coin']} ({t['symbol']}) | Strategy: {t['strategy']}")
            print(f"  Archetype: {t['archetype']}")
            print(f"  Empirical Edge: {t['historical_edge']}")
            print(f"  Live Price: {t['current_price']} | Key Level: {t['key_level']}")
            print(f"  Action Plan: {t['action']}")

    # Display Pending Watchlist Setups
    print("\n" + "-" * 90)
    print(">>> NEARBY / PENDING HIGH-EDGE SETUPS (On Watch) <<<")
    if not pending_watchlists:
        print("  No pending watchlist setups within trigger threshold.")
    else:
        for w in pending_watchlists:
            print(f"  * {w['coin']} ({w['symbol']}) -> {w['strategy']} | Price: {w['current_price']} | Level: {w['key_level']}")
            print(f"    Notice: {w['action']}")

    # Print Live Coin Distance Table
    print("\n" + "-" * 90)
    print(">>> LIVE DISTANCE TO HIGH-EDGE TRIGGERS (WEEX Real-Time) <<<")
    print(f"{'Coin':<10} {'Live Price':<12} {'Key Strategy':<24} {'Key Trigger Level':<18} {'Distance to Trigger'}")
    print("-" * 90)
    for sym, m in SPECIALIZED_UNIVERSE.items():
        df_c = fetch_weex_klines(sym, interval='1d', limit=25)
        if df_c is not None:
            c = df_c['close'].iloc[-1]
            strat = m['strategies'][0]
            if strat == 'Weekly_Open_Reclaim':
                ref = df_c['open'].iloc[-7] if len(df_c) >= 7 else c
                dist = ((c - ref) / ref) * 100
                lvl_str = f"WO: {ref:.4f}"
                dist_str = f"{dist:+.2f}%"
            elif strat == 'Donchian_Breakout':
                ref = df_c['high'].iloc[-21:-1].max()
                dist = ((ref - c) / c) * 100
                lvl_str = f"20D High: {ref:.4f}"
                dist_str = f"{dist:+.2f}%"
            elif strat == 'BB_Mean_Reversion':
                sma20 = df_c['close'].iloc[-21:-1].mean()
                std20 = df_c['close'].iloc[-21:-1].std()
                ref = sma20 - 2.0 * std20
                dist = ((c - ref) / c) * 100
                lvl_str = f"Lower BB: {ref:.4f}"
                dist_str = f"{dist:+.2f}%"
            elif strat == 'Turtle_Soup':
                ref = df_c['low'].iloc[-21:-1].min()
                dist = ((c - ref) / c) * 100
                lvl_str = f"20D Low: {ref:.4f}"
                dist_str = f"{dist:+.2f}%"
            elif strat == 'Volume_Absorption':
                curr = df_c.iloc[-1]
                tot_rng = max(curr['high'] - curr['low'], 1e-8)
                lw = min(curr['open'], curr['close']) - curr['low']
                wick_pct = (lw / tot_rng) * 100
                vol_avg = df_c['volume'].iloc[-21:-1].mean()
                vol_mult = curr['volume'] / vol_avg if vol_avg > 0 else 0
                lvl_str = f"Wick:{wick_pct:.0f}%/55%"
                dist_str = f"Vol:{vol_mult:.1f}x/1.8x"
            else:
                lvl_str = f"{c:.4f}"
                dist_str = "N/A"
            print(f"{m['coin']:<10} {c:<12.4f} {strat:<24} {lvl_str:<18} {dist_str}")

    print("\n" + "=" * 90)
    
    scan_output = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "active_triggers": active_triggers,
        "pending_watchlists": pending_watchlists
    }
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(scan_output, f, indent=2)

    # Telegram Notification Integration
    cfg = load_scanner_config()
    token = cfg.get("telegram_bot_token", "")
    chat_id = cfg.get("telegram_chat_id", "")
    
    if token and chat_id:
        if active_triggers:
            lines = [f"🚨 <b>WEEX SPECIALIZED SETUP ALERTS ({len(active_triggers)})</b>", f"<i>{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}</i>\n"]
            for t in active_triggers:
                lines.append(f"<b>{t['coin']} ({t['strategy']})</b>")
                lines.append(f"• Price: <code>{t['current_price']}</code> | Level: <code>{t['key_level']}</code>")
                lines.append(f"• Edge: <i>{t['historical_edge']}</i>")
                lines.append(f"• Plan: {t['action']}\n")
            send_telegram_alert("\n".join(lines), token, chat_id)
        elif cfg.get("send_daily_digest", False):
            # Send status update
            lines = ["📡 <b>WEEX SPECIALIZED SCANNER — HEARTBEAT</b>", f"<i>{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}</i>\n", "No active triggers fired. Proximity to triggers:"]
            for sym, m in SPECIALIZED_UNIVERSE.items():
                df_c = fetch_weex_klines(sym, interval='1d', limit=25)
                if df_c is not None:
                    c = df_c['close'].iloc[-1]
                    strat = m['strategies'][0]
                    if strat == 'Weekly_Open_Reclaim':
                        ref = df_c['open'].iloc[-7] if len(df_c) >= 7 else c
                        dist = ((c - ref) / ref) * 100
                    elif strat == 'Donchian_Breakout':
                        ref = df_c['high'].iloc[-21:-1].max()
                        dist = ((ref - c) / c) * 100
                    elif strat == 'BB_Mean_Reversion':
                        sma20 = df_c['close'].iloc[-21:-1].mean()
                        std20 = df_c['close'].iloc[-21:-1].std()
                        ref = sma20 - 2.0 * std20
                        dist = ((c - ref) / c) * 100
                    elif strat == 'Turtle_Soup':
                        ref = df_c['low'].iloc[-21:-1].min()
                        dist = ((c - ref) / c) * 100
                    else:
                        dist = 0.0
                    lines.append(f"• <b>{m['coin']}</b> ({strat[:12]}): <code>{dist:+.1f}%</code>")
            send_telegram_alert("\n".join(lines), token, chat_id)
        
    return scan_output

if __name__ == "__main__":
    run_scan()
