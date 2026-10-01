from typing import List, Optional
import pandas as pd


def align_1h_to_15m(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    columns_to_align: List[str]
) -> pd.DataFrame:
    """
    Causally aligns 1H closed indicators onto 15m candles (DEFINITIONS.md §1).

    Time Convention Rule:
    - A 1H candle with open_time = T closes at T + 1 hour.
    - It is only 'known' and available at or after T + 1 hour.
    - A 15m candle opening at t <= T + 45m MUST NOT see the 1H candle opening at T.
    - A 15m candle opening at t = T + 1 hour is the first 15m bar that sees it.

    Implementation:
    - Uses pd.merge_asof on sorted timestamps with direction='backward'
      where left_on is 15m open_time and right_on is 1H available_time (open_time + 1h).
    """
    if df_15m.empty:
        return df_15m.copy()
    if df_1h.empty:
        df_out = df_15m.copy()
        for col in columns_to_align:
            df_out[col] = None
        return df_out

    # Ensure sorted by open_time
    df_15m_sorted = df_15m.sort_values(by="open_time").copy()
    df_1h_sorted = df_1h.sort_values(by="open_time").copy()

    # Define 1H available_time as open_time + 1 hour (candle close)
    df_1h_sorted["available_time"] = (df_1h_sorted["open_time"] + pd.Timedelta(hours=1)).astype(df_15m_sorted["open_time"].dtype)

    # Select columns to merge
    cols = ["available_time"] + [c for c in columns_to_align if c in df_1h_sorted.columns]
    right_df = df_1h_sorted[cols]

    # Causal backward merge
    merged = pd.merge_asof(
        df_15m_sorted,
        right_df,
        left_on="open_time",
        right_on="available_time",
        direction="backward"
    )

    merged = merged.drop(columns=["available_time"], errors="ignore")
    return merged
