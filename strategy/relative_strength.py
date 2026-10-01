"""
strategy/relative_strength.py
Computes Causal Relative Strength (RS) of Altcoins vs Bitcoin.
Anti-lookahead rule: RS at decision time t uses only bars closed at or before t.

Features:
- rs_spread(tau): R_alt(tau) - R_btc(tau) over trailing lookback tau
- rs_ratio_trend: alt_price / btc_price > EMA50(alt_price / btc_price)
- cross_sectional_rank: percentile rank of altcoin RS across active universe at time t.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class RelativeStrengthEngine:
    """
    Computes trailing Relative Strength metrics of an altcoin vs BTC_USDT.
    """

    def __init__(self, df_btc_4h: pd.DataFrame):
        df_btc = df_btc_4h.sort_values("open_time").copy()
        df_btc["open_time"] = pd.to_datetime(df_btc["open_time"], utc=True)
        self.df_btc = df_btc.set_index("open_time")
        self.btc_closes = self.df_btc["close"]

    def compute_symbol_rs_features(
        self,
        df_sym_4h: pd.DataFrame,
        lookbacks: Dict[str, int] = {
            "3d": 18,   # 18 * 4h = 72 hours
            "7d": 42,   # 42 * 4h = 168 hours
            "14d": 84,  # 84 * 4h = 336 hours
            "30d": 180  # 180 * 4h = 720 hours
        }
    ) -> pd.DataFrame:
        """
        Aligns altcoin 4H bars with BTC 4H bars and computes trailing outperformance features.
        """
        if df_sym_4h.empty or self.df_btc.empty:
            return pd.DataFrame()

        df = df_sym_4h.sort_values("open_time").copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")

        # Align with BTC close
        df["btc_close"] = self.btc_closes.reindex(df.index, method="ffill")

        # Relative Strength Price Ratio (ALT / BTC)
        df["alt_btc_ratio"] = df["close"] / df["btc_close"]
        df["ratio_ema50"] = df["alt_btc_ratio"].ewm(span=50, adjust=False).mean()
        df["ratio_trend_bull"] = df["alt_btc_ratio"] > df["ratio_ema50"]

        for label, bars in lookbacks.items():
            sym_ret = (df["close"] / df["close"].shift(bars)) - 1.0
            btc_ret = (df["btc_close"] / df["btc_close"].shift(bars)) - 1.0
            rs_spread = sym_ret - btc_ret
            df[f"rs_{label}_spread"] = rs_spread
            df[f"rs_{label}_outperforming"] = rs_spread > 0.0

        return df.reset_index()
