"""
analysis/regimes.py
Analyzes strategy performance across BTC filter variants and market regime segmentations
per docs/DEFINITIONS.md §15 and Phase 5 prompt.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from analysis.baselines import compute_wilson_ci


class RegimeAnalyzer:
    """
    Evaluates trades across BTC filters and multi-dimensional regime segmentations.
    """

    @staticmethod
    def evaluate_btc_filters(df_trades: pd.DataFrame, btc_regime_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tests BTC filter variants:
        1. none (baseline)
        2. btc_trend_c (BTC in 1H uptrend)
        3. btc_above_ema50
        4. btc_above_ema200
        5. btc_declining
        """
        if df_trades.empty or btc_regime_df.empty:
            return pd.DataFrame()

        # Merge trades with btc regime by entry_time using merge_asof (causal trailing)
        trades = df_trades.sort_values("entry_time").copy()
        trades["entry_time_dt"] = pd.to_datetime(trades["entry_time"], utc=True)

        btc = btc_regime_df.sort_values("open_time").copy()
        btc["open_time_dt"] = pd.to_datetime(btc["open_time"], utc=True)

        overlap = [c for c in btc.columns if c in trades.columns and c != "entry_time_dt"]
        if overlap:
            trades = trades.drop(columns=overlap)

        merged = pd.merge_asof(
            trades,
            btc,
            left_on="entry_time_dt",
            right_on="open_time_dt",
            direction="backward"
        )


        variants = [
            ("none", np.ones(len(merged), dtype=bool)),
            ("btc_trend_c", merged["btc_trend_bull"] == True),
            ("btc_above_ema50", merged["btc_above_ema50"] == True),
            ("btc_above_ema200", merged["btc_above_ema200"] == True),
            ("btc_declining", merged["btc_declining"] == True)
        ]

        rows = []
        for name, mask in variants:
            sub = merged[mask]
            n = len(sub)
            if n == 0:
                continue

            wins = int((sub["net_pnl_r"] > 0).sum())
            wr, wr_lo, wr_hi = compute_wilson_ci(wins, n)

            net_exp = float(sub["net_pnl_r"].mean())
            gross_exp = float(sub["gross_pnl_r"].mean())

            pos_sum = sub.loc[sub["net_pnl_r"] > 0, "net_pnl_r"].sum()
            neg_sum = abs(sub.loc[sub["net_pnl_r"] < 0, "net_pnl_r"].sum())
            net_pf = float(pos_sum / neg_sum) if neg_sum > 0 else (10.0 if pos_sum > 0 else 0.0)

            rows.append({
                "filter_variant": name,
                "n_trades": n,
                "win_rate": wr,
                "win_rate_ci_lower": wr_lo,
                "win_rate_ci_upper": wr_hi,
                "net_expectancy_r": net_exp,
                "gross_expectancy_r": gross_exp,
                "net_profit_factor": net_pf
            })

        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_regimes(
        df_trades: pd.DataFrame,
        btc_regime_df: pd.DataFrame,
        market_vol_median: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Segments trades by:
        - BTC 24h Return: Up vs Down
        - BTC Volatility: High vs Low
        - Alt Volatility: High vs Low (trade-level relative to sample)
        - Market Volume: High vs Low
        - Macro State: Bull vs Bear vs Chop
        """
        if df_trades.empty or btc_regime_df.empty:
            return {}

        trades = df_trades.sort_values("entry_time").copy()
        trades["entry_time_dt"] = pd.to_datetime(trades["entry_time"], utc=True)

        btc = btc_regime_df.sort_values("open_time").copy()
        btc["open_time_dt"] = pd.to_datetime(btc["open_time"], utc=True)

        overlap = [c for c in btc.columns if c in trades.columns and c != "entry_time_dt"]
        if overlap:
            trades = trades.drop(columns=overlap)

        merged = pd.merge_asof(
            trades,
            btc,
            left_on="entry_time_dt",
            right_on="open_time_dt",
            direction="backward"
        )


        results = {}

        # 1. BTC Direction
        btc_dir = np.where(merged["btc_up_24h"] == True, "BTC_Up", "BTC_Down")
        results["btc_direction"] = RegimeAnalyzer._summarize_segments(merged, btc_dir)

        # 2. BTC Volatility
        btc_vol = np.where(merged["btc_high_volatility"] == True, "BTC_Vol_High", "BTC_Vol_Low")
        results["btc_volatility"] = RegimeAnalyzer._summarize_segments(merged, btc_vol)

        # 3. Alt Volatility (ATR relative to entry price)
        vol_pct = merged["atr_entry"] / merged["entry_price"]
        median_alt_vol = vol_pct.median()
        alt_vol = np.where(vol_pct > median_alt_vol, "Alt_Vol_High", "Alt_Vol_Low")
        results["alt_volatility"] = RegimeAnalyzer._summarize_segments(merged, alt_vol)

        # 4. Market Volume
        vol_med = market_vol_median if market_vol_median is not None else merged["volume_24h_usdt"].median()
        mkt_vol = np.where(merged["volume_24h_usdt"] > vol_med, "Mkt_Vol_High", "Mkt_Vol_Low")
        results["market_volume"] = RegimeAnalyzer._summarize_segments(merged, mkt_vol)

        # 5. Macro State
        macro = merged["macro_regime"].fillna("Chop").to_numpy(dtype=str)
        results["macro_regime"] = RegimeAnalyzer._summarize_segments(merged, macro)

        return results

    @staticmethod
    def _summarize_segments(df: pd.DataFrame, segment_labels: np.ndarray) -> pd.DataFrame:
        df_seg = df.copy()
        df_seg["segment"] = segment_labels

        rows = []
        for seg_name, group in df_seg.groupby("segment"):
            n = len(group)
            if n == 0:
                continue

            wins = int((group["net_pnl_r"] > 0).sum())
            wr, wr_lo, wr_hi = compute_wilson_ci(wins, n)
            net_exp = float(group["net_pnl_r"].mean())
            gross_exp = float(group["gross_pnl_r"].mean())

            pos_sum = group.loc[group["net_pnl_r"] > 0, "net_pnl_r"].sum()
            neg_sum = abs(group.loc[group["net_pnl_r"] < 0, "net_pnl_r"].sum())
            net_pf = float(pos_sum / neg_sum) if neg_sum > 0 else (10.0 if pos_sum > 0 else 0.0)

            rows.append({
                "segment": seg_name,
                "n_trades": n,
                "win_rate": wr,
                "win_rate_ci_lower": wr_lo,
                "win_rate_ci_upper": wr_hi,
                "net_expectancy_r": net_exp,
                "gross_expectancy_r": gross_exp,
                "net_profit_factor": net_pf
            })

        return pd.DataFrame(rows)
