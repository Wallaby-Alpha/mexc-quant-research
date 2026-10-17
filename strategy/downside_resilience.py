"""
strategy/downside_resilience.py
--------------------------------
"The Coin That Refuses to Fall" — Downside Resilience Strategy

Hypothesis:
    When BTC falls sharply, some alts absorb the selling without breaking down.
    This is evidence of unusually strong marginal demand. If sellers failed to
    push the coin lower, the next market-wide bounce may produce an outsized
    recovery. We long the MOST resilient coins AFTER BTC stabilises and the
    alt confirms strength.

Anti-Lookahead:
    - BTC stress episodes are strictly detected on closed bars.
    - Alt Downside Capture Ratio (DCR) is measured on identical closed bars.
    - Stabilization and strength confirmation are verified on closed bars.
    - Entry is executed at the OPEN of the bar immediately following confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StressEpisode:
    episode_id: str
    start_bar_time: pd.Timestamp
    end_bar_time: pd.Timestamp
    btc_return: float
    btc_start_price: float
    btc_end_price: float
    n_bars: int


@dataclass
class AltResilience:
    symbol: str
    episode_id: str
    alt_return: float
    dcr: float                     # Downside Capture Ratio = alt_return / btc_return
    alt_start_price: float
    alt_end_price: float
    resilience_rank: int          # 1 = most resilient
    resilience_pct: float         # Percentile [0, 100]


@dataclass
class ResilienceSetup:
    episode_id: str
    symbol: str
    entry_bar_time: pd.Timestamp
    confirm_bar_time: pd.Timestamp
    btc_return: float
    alt_return: float
    dcr: float
    resilience_rank: int
    resilience_pct: float
    entry_price: float
    alt_confirm_close: float


class BTCStressDetector:
    """
    Detects BTC stress episodes from 1H closed klines.
    """

    def __init__(
        self,
        btc_drop_pct: float = 0.03,      # 3% drop
        stress_window_bars: int = 12,    # Within 12h
        cooldown_bars: int = 24          # 24h cooldown between episodes
    ):
        self.btc_drop_pct = btc_drop_pct
        self.stress_window_bars = stress_window_bars
        self.cooldown_bars = cooldown_bars

    def detect(self, df_btc_1h: pd.DataFrame) -> List[StressEpisode]:
        if df_btc_1h.empty or len(df_btc_1h) < self.stress_window_bars + 5:
            return []

        df = df_btc_1h.sort_values("open_time").reset_index(drop=True).copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

        closes = df["close"].to_numpy(dtype=float)
        times = df["open_time"].to_numpy()
        n = len(df)

        episodes: List[StressEpisode] = []
        last_episode_end_idx = -self.cooldown_bars

        for i in range(self.stress_window_bars, n):
            if i - last_episode_end_idx < self.cooldown_bars:
                continue

            best_start_idx: Optional[int] = None
            best_return: float = 0.0
            best_lookback: int = 0

            for lookback in range(1, self.stress_window_bars + 1):
                j = i - lookback
                if j < 0:
                    break
                ret = (closes[i] / closes[j]) - 1.0
                if ret <= -self.btc_drop_pct:
                    best_start_idx = j
                    best_return = ret
                    best_lookback = lookback
                    break

            if best_start_idx is None:
                continue

            ep_id = f"EP-{pd.Timestamp(times[i]).strftime('%Y%m%d-%H%M')}"
            episodes.append(StressEpisode(
                episode_id=ep_id,
                start_bar_time=pd.Timestamp(times[best_start_idx]),
                end_bar_time=pd.Timestamp(times[i]),
                btc_return=best_return,
                btc_start_price=float(closes[best_start_idx]),
                btc_end_price=float(closes[i]),
                n_bars=best_lookback
            ))
            last_episode_end_idx = i

        return episodes


class ResilienceRanker:
    """
    Computes Downside Capture Ratio (DCR) for each alt during stress episodes:
        DCR = alt_return / btc_return (both over identical stress window)
    Lower/less negative DCR or positive DCR indicates strong absorption.
    Also computes ordinary relative-strength leaders (highest trailing returns prior to stress)
    for control group benchmarking.
    """

    def __init__(self, exclude_btc: bool = True):
        self.exclude_btc = exclude_btc

    def rank_episode(
        self,
        episode: StressEpisode,
        panel_1h: Dict[str, pd.DataFrame]
    ) -> Tuple[List[AltResilience], List[Dict[str, any]]]:
        if episode.btc_return >= 0:
            return [], []

        start_t = episode.start_bar_time
        end_t = episode.end_bar_time

        resilience_rows = []
        rs_leader_rows = []

        for symbol, df in panel_1h.items():
            if self.exclude_btc and symbol in ["BTC_USDT", "BTC"]:
                continue
            if df.empty:
                continue

            # Need fast lookup by open_time
            if "open_time" not in df.columns or "close" not in df.columns:
                continue

            df_sub = df[(df["open_time"] >= start_t - pd.Timedelta(hours=48)) & (df["open_time"] <= end_t)].copy()
            if df_sub.empty:
                continue

            df_sub = df_sub.sort_values("open_time").set_index("open_time")
            try:
                start_px = df_sub["close"].asof(start_t)
                end_px = df_sub["close"].asof(end_t)
            except Exception:
                continue

            if pd.isna(start_px) or pd.isna(end_px) or start_px <= 0 or end_px <= 0:
                continue

            alt_ret = (end_px / start_px) - 1.0
            dcr = alt_ret / episode.btc_return

            resilience_rows.append({
                "symbol": symbol,
                "alt_ret": alt_ret,
                "dcr": dcr,
                "start_px": float(start_px),
                "end_px": float(end_px)
            })

            # Calculate ordinary RS leader: 24h return prior to the stress event start
            try:
                pre_t = start_t - pd.Timedelta(hours=24)
                pre_px = df_sub["close"].asof(pre_t)
                if not pd.isna(pre_px) and pre_px > 0:
                    rs_ret_24h = (start_px / pre_px) - 1.0
                    rs_leader_rows.append({
                        "symbol": symbol,
                        "rs_ret_24h": rs_ret_24h,
                        "start_px": float(start_px),
                        "end_px": float(end_px)
                    })
            except Exception:
                pass

        # Sort resilience: lowest DCR first (most resilient)
        # Note: DCR = alt_ret / btc_ret.
        # If BTC drops 5% (btc_ret = -0.05):
        # Alt A gains 2%: alt_ret = +0.02 -> DCR = -0.40
        # Alt B drops 1%: alt_ret = -0.01 -> DCR = +0.20
        # Alt C drops 5%: alt_ret = -0.05 -> DCR = +1.00
        # Alt D drops 10%: alt_ret = -0.10 -> DCR = +2.00
        # Ordering by DCR ascending cleanly orders from best upside/absorption to worst breakdown!
        resilience_rows.sort(key=lambda x: x["dcr"])

        ranked_resilience: List[AltResilience] = []
        total = len(resilience_rows)
        for idx, row in enumerate(resilience_rows):
            pct = 100.0 * (1.0 - idx / max(total - 1, 1))
            ranked_resilience.append(AltResilience(
                symbol=row["symbol"],
                episode_id=episode.episode_id,
                alt_return=row["alt_ret"],
                dcr=row["dcr"],
                alt_start_price=row["start_px"],
                alt_end_price=row["end_px"],
                resilience_rank=idx + 1,
                resilience_pct=round(pct, 2)
            ))

        # Sort standard RS leaders: highest 24h return prior to stress
        rs_leader_rows.sort(key=lambda x: x["rs_ret_24h"], reverse=True)

        return ranked_resilience, rs_leader_rows
