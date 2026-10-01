"""
strategy/btc_regime.py
Computes BTC Market Regime and Macro Features causally aligned to 15m decision time.
Follows docs/DEFINITIONS.md §15.
"""

from typing import Optional
import numpy as np
import pandas as pd

from data.cache import ParquetCache
from strategy.indicators import compute_ema, compute_wilder_atr
from strategy.trend_detector import TrendDetector
from strategy.alignment import align_1h_to_15m


class BTCRegimeDetector:
    """
    Computes BTC Market Regime Features (strictly causal, trailing closed data only).
    """

    def __init__(self, cache: Optional[ParquetCache] = None):
        self.cache = cache or ParquetCache()
        self.trend_detector = TrendDetector()

    def compute_btc_1h_regime(self) -> pd.DataFrame:
        df_btc = self.cache.load_klines("BTC_USDT", "1h")
        if df_btc.empty:
            return pd.DataFrame()

        df = df_btc.sort_values(by="open_time").copy()
        df = self.trend_detector.compute_1h_trends(df)

        # 24h trailing closed return
        df["btc_return_24h"] = (df["close"] / df["close"].shift(24)) - 1.0
        df["btc_up_24h"] = df["btc_return_24h"] >= 0.0

        # Wilder ATR14 and 30-day (720h) rolling median
        atr_1h = compute_wilder_atr(df["high"], df["low"], df["close"], period=14)
        df["btc_atr_1h"] = atr_1h
        rolling_median_atr = atr_1h.rolling(720, min_periods=168).median()
        df["btc_high_volatility"] = atr_1h > rolling_median_atr

        # EMA relations
        df["btc_above_ema50"] = df["close"] > df["ema50"]
        df["btc_above_ema200"] = df["close"] > df["ema200"]
        
        # EMA50 falling over k=3 bars
        ema50_falling = df["ema50"] < df["ema50"].shift(3)
        df["btc_declining"] = (~df["btc_above_ema50"]) & ema50_falling

        # Macro Trend State: Bull, Bear, Chop
        is_bull = df["trend_c"] == True
        is_bear = (~df["btc_above_ema50"]) & (df["ema50"] < df["ema200"]) & ema50_falling
        
        macro_state = np.where(is_bull, "Bull", np.where(is_bear, "Bear", "Chop"))
        df["macro_regime"] = macro_state

        regime_cols = [
            "open_time", "trend_c", "btc_above_ema50", "btc_above_ema200",
            "btc_declining", "btc_return_24h", "btc_up_24h",
            "btc_atr_1h", "btc_high_volatility", "macro_regime"
        ]
        return df[regime_cols].rename(columns={"trend_c": "btc_trend_bull"})

    def align_to_15m(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        btc_1h = self.compute_btc_1h_regime()
        regime_cols = [
            "btc_trend_bull", "btc_above_ema50", "btc_above_ema200",
            "btc_declining", "btc_return_24h", "btc_up_24h",
            "btc_atr_1h", "btc_high_volatility", "macro_regime"
        ]
        if btc_1h.empty:
            df_out = df_15m.copy()
            for col in regime_cols:
                df_out[col] = None
            return df_out

        return align_1h_to_15m(
            df_15m=df_15m,
            df_1h=btc_1h,
            columns_to_align=regime_cols
        )
