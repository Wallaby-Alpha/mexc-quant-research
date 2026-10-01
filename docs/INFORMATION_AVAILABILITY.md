# Information Availability Specification

Per Non-Negotiable Rule 1 and Phase 2 requirements:
This document defines for every field produced by the Phase 2 feature and setup extraction pipeline the exact timestamp at which the value becomes known and tradable.

---

## 1. Multi-Timeframe Availability Rules

| Timeframe | Timestamp Format | Closed / Known Timestamp | Earliest Decision Time | Earliest Fill / Action Time |
|---|---|---|---|---|
| **15m Candle** | $t$ (Open Time UTC) | $t + \text{15m}$ | $t + \text{15m}$ (at bar close) | $t + \text{15m}$ (Open of $t+1$) |
| **1H Candle** | $T$ (Open Time UTC) | $T + \text{1h}$ | $T + \text{1h}$ (at bar close) | $T + \text{1h}$ (Open of next 15m bar) |

### Strict Multi-Timeframe Alignment
- A 1H candle opening at `10:00:00 UTC` closes at `11:00:00 UTC`.
- 15m bars opening at `10:00, 10:15, 10:30, 10:45` belong to decision times up to `11:00:00 UTC`. They **MUST NOT** see the 1H candle opening at `10:00:00 UTC`.
- The first 15m bar permitted to access the `10:00:00 UTC` 1H indicators is the 15m bar opening at `11:00:00 UTC`.

---

## 2. Setup & Event Feature Field Matrix

| Field Name | Type | Information Availability Time | Rationale / Reference |
|---|---|---|---|
| `symbol` | `str` | Static ($t_0$) | Symbol identifier. |
| `fractal_n` | `int` | Static ($t_0$) | Fractal parameter $N \in \{2, 3, 4, 5\}$ (default $N=3$). |
| `swing_high_time` | `datetime (UTC)` | $T + N \times \text{15m} + \text{15m}$ | Open time of swing high candle $T$. Only confirmed after candle $T+N$ closes! |
| `swing_high_price` | `float` | $T + N \times \text{15m} + \text{15m}$ | High price of candle $T$. Confirmed at $T+N$ close. |
| `confirmation_time` | `datetime (UTC)` | $T + N \times \text{15m} + \text{15m}$ | Open time of confirmation bar $T+N$. Known strictly at its close. |
| `earliest_action_time` | `datetime (UTC)` | $T + (N+1) \times \text{15m}$ | Open of candle $T+N+1$. Earliest possible time any order could execute. |
| `swing_low_time` | `datetime (UTC)` | $T + N \times \text{15m} + \text{15m}$ | Lowest low candle in $[T-L, T]$ ($L=32$ bars). Known at confirmation. |
| `swing_low_price` | `float` | $T + N \times \text{15m} + \text{15m}$ | Price of swing low. Known at confirmation. |
| `impulse` | `float` | $T + N \times \text{15m} + \text{15m}$ | $High_T - Low_{swing}$. Known at confirmation. |
| `atr_ref` | `float` | $T + N \times \text{15m} + \text{15m}$ | Wilder ATR(14) on 15m at close of $T+N$. **Frozen** for that setup. |
| `impulse_atr` | `float` | $T + N \times \text{15m} + \text{15m}$ | $Impulse / ATR_{ref}$. Must be $\ge 2.0$ to qualify. |
| `threshold_type` | `str` | Decision time $t + \text{15m}$ | "atr" or "retracement". |
| `threshold_value` | `float` | Decision time $t + \text{15m}$ | Threshold level (e.g. 0.50 ATR or 30%). |
| `event_bar_idx` | `int` | Decision time $t + \text{15m}$ | 15m bar index $t$ where running depth first meets threshold. |
| `event_time` | `datetime (UTC)` | Decision time $t + \text{15m}$ | Open time of event bar $t$. |
| `event_close_time` | `datetime (UTC)` | Decision time $t + \text{15m}$ | Decision time = $event\_time + 15m$. When event is known. |
| `action_open_time` | `datetime (UTC)` | $t + \text{15m}$ (Fills at $t+1$) | Open time of bar $t+1$. Order execution occurs here. |
| `reference_price` | `float` | $t + \text{15m}$ | Open price of candle $t+1$. Fill price before slippage. |
| `running_pullback_low`| `float` | Decision time $t + \text{15m}$ | $\min(low[T+1 \dots t])$. Causal running minimum. |
| `pullback_depth_atr` | `float` | Decision time $t + \text{15m}$ | $(High_T - running\_low) / ATR_{ref}$ at event close. |
| `retracement_pct` | `float` | Decision time $t + \text{15m}$ | $(High_T - running\_low) / Impulse \times 100\%$ at event close. |
| `atr_entry` | `float` | Decision time $t + \text{15m}$ | Wilder ATR(14) at event bar $t$ close. |
| `event_stop_level` | `float` | Decision time $t + \text{15m}$ | $running\_low - 0.25 \times ATR_{ref}$. Fixed at signal time. |
| `trend_valid_default` | `bool` | Decision time $t + \text{15m}$ | Evaluated using last closed 1H bar before $t + \text{15m}$. |
| `trend_a` | `bool` | Decision time $t + \text{15m}$ | Trend A on last closed 1H bar. |
| `trend_b` | `bool` | Decision time $t + \text{15m}$ | Trend B on last closed 1H bar. |
| `trend_c` | `bool` | Decision time $t + \text{15m}$ | Trend C on last closed 1H bar. |
| `trend_d` | `bool` | Decision time $t + \text{15m}$ | Supertrend on last closed 1H bar. |
| `universe_rank` | `int` | Decision time $t + \text{15m}$ | Trailing 24h volume rank from last closed 1H bar. |
| `is_controlled_pullback`| `bool` | Decision time $t + \text{15m}$ | True if Trend C valid, not invalidated, and stop not breached. |
| `diag_final_depth_atr` | `float` | **$t + \text{24h}$ (Future Diagnostic)** | Final deepest pullback depth. **FORBIDDEN as a signal filter!** |
| `diag_final_depth_pct` | `float` | **$t + \text{24h}$ (Future Diagnostic)** | Final deepest retracement. **FORBIDDEN as a signal filter!** |
