# Phase 4 Research Report: Execution and Trade Simulation

## Executive Summary
> **Hypothesis:** Buying a controlled 15m pullback toward a prior swing high inside an established 1H uptrend yields a tradable edge on MEXC crypto markets.

### Plain Language Verdict
- Across **42,700** simulated trades under realistic execution frictions (maker target fills, taker stop fills, volume-rank slippage, and 8h funding fees) from October 2025 to June 2026 (non-holdout):
- **Primary Net Expectancy:** **-0.7865 R per trade** (Net Return: **-0.48%** per trade).
- **Gross Expectancy (Secondary):** **+0.0094 R per trade**.
- **Net Win Rate:** **24.70%** (95% Wilson CI: [24.30%, 25.11%]).
- **Net Profit Factor:** **0.28** (Gross Profit Factor: **1.01**).
- **Max Drawdown (Fixed 1% Risk):** **-100.00%**.
- **Annualized Sharpe / Sortino:** **-84.00** / **-175.90** (*Annualized using active trade frequency (42700 trades across 269.8 days)*).

**VERDICT: NO ECONOMIC EDGE AFTER COSTS.** While gross expectancy is mildly positive or neutral, trading frictions (spread, taker fees on stops, stop slippage, and gap execution) consume the margin, resulting in negative or sub-threshold net expectancy.

---

## 1. Primary Strategy Performance (Headline: Net of Costs)

| Metric | Net-of-Cost (Primary) | Gross (Secondary) |
|---|---:|---:|
| **Total Completed Trades** | **42,700** | 42,700 |
| **Win Rate** | **24.70%** [24.30%, 25.11%] | 27.57% |
| **Expectancy (R-Multiple)** | **-0.7865 R** | +0.0094 R |
| **Expectancy (%)** | **-0.476%** | -0.441% |
| **Profit Factor** | **0.28** | 1.01 |
| **Avg Win / Avg Loss (R)** | **+1.23 R / -1.45 R** | - |
| **Max Drawdown (1% Risk)** | **-100.00%** | - |
| **Sharpe Ratio (Annualized)** | **-84.00** | - |
| **Sortino Ratio (Annualized)** | **-175.90** | - |
| **Avg Holding Time** | **1.2 hours** (0.5h median) | - |
| **Total Cost Impact** | **Fees: 1474.1%, Slip: 18426.8%, Funding: 64.09%** | - |

### Exit Reason Distribution
- **STOP**: 30,885 (72.3%)
- **TARGET**: 11,693 (27.4%)
- **TIME_STOP**: 122 (0.3%)
- **Ambiguous Same-Bar Events** (Conservative Stop-First Triggered): 202

---

## 2. Reconciliation with Phase 3 Event Study

> Phase 4 trade universe is a verified subset of Phase 3 event setups. From 66,116 candidate setups: 42,700 executed trades, 18,235 rejected by min R:R, 5,170 skipped by single-position overlap, and 11 unentered/expired.

| Funnel Stage | Count | Share of Phase 3 Setups |
|---|---:|---:|
| **Phase 3 Controlled Pullback Setups (0.50 ATR, Trend C)** | 66,116 | 100.0% |
| **Rejected by Min R:R Filter (< 1.0)** | 18,235 | 27.58% |
| **Skipped Due to Open Position Overlap (Max 1 / Symbol)** | 5,170 | 7.82% |
| **Executed Trades** | **42,700** | **64.58%** |
| **Discrepancies / Orphaned Trades** | **0** | **0.00%** |

---

## 3. Multiple-Testing Parameter Surface (Trial Log)

All parameter variations are logged in `results/trials.parquet` per Rule 6:

| Trial ID | Entry Mode | Min R:R | Time Stop | Stop Buffer | Trades ($n$) | Win Rate (Net) | Net Exp (R) | Profit Factor | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TRIAL_001_Entry_A_RR1.0_H24.0_B0.25` | Entry_A | 1.0 | 24.0h | 0.25 ATR | 42,700 | 24.7% | **-0.787 R** | 0.28 | -100.0% |
| `TRIAL_002_Entry_B_RR1.0_H24.0_B0.25` | Entry_B | 1.0 | 24.0h | 0.25 ATR | 6,258 | 36.0% | **-0.494 R** | 0.39 | -100.0% |
| `TRIAL_003_Entry_C_RR1.0_H24.0_B0.25` | Entry_C | 1.0 | 24.0h | 0.25 ATR | 893 | 36.2% | **-0.438 R** | 0.43 | -98.2% |
| `TRIAL_004_Entry_A_RR1.5_H24.0_B0.25` | Entry_A | 1.5 | 24.0h | 0.25 ATR | 35,552 | 22.0% | **-0.830 R** | 0.28 | -100.0% |
| `TRIAL_005_Entry_A_RR2.0_H24.0_B0.25` | Entry_A | 2.0 | 24.0h | 0.25 ATR | 29,491 | 19.8% | **-0.872 R** | 0.28 | -100.0% |
| `TRIAL_006_Entry_A_RR2.5_H24.0_B0.25` | Entry_A | 2.5 | 24.0h | 0.25 ATR | 24,292 | 17.9% | **-0.911 R** | 0.27 | -100.0% |
| `TRIAL_007_Entry_A_RR3.0_H24.0_B0.25` | Entry_A | 3.0 | 24.0h | 0.25 ATR | 20,082 | 16.3% | **-0.949 R** | 0.26 | -100.0% |
| `TRIAL_008_Entry_A_RR1.0_H1.0_B0.25` | Entry_A | 1.0 | 1.0h | 0.25 ATR | 47,869 | 20.7% | **-0.834 R** | 0.18 | -100.0% |
| `TRIAL_009_Entry_A_RR1.0_H2.0_B0.25` | Entry_A | 1.0 | 2.0h | 0.25 ATR | 46,020 | 23.5% | **-0.807 R** | 0.23 | -100.0% |
| `TRIAL_010_Entry_A_RR1.0_H4.0_B0.25` | Entry_A | 1.0 | 4.0h | 0.25 ATR | 44,248 | 24.5% | **-0.793 R** | 0.26 | -100.0% |
| `TRIAL_011_Entry_A_RR1.0_H8.0_B0.25` | Entry_A | 1.0 | 8.0h | 0.25 ATR | 43,342 | 24.7% | **-0.788 R** | 0.27 | -100.0% |
| `TRIAL_012_Entry_A_RR1.0_H12.0_B0.25` | Entry_A | 1.0 | 12.0h | 0.25 ATR | 43,031 | 24.7% | **-0.788 R** | 0.28 | -100.0% |
| `TRIAL_013_Entry_A_RR1.0_H24.0_B0.1` | Entry_A | 1.0 | 24.0h | 0.1 ATR | 46,316 | 21.6% | **-0.915 R** | 0.24 | -100.0% |
| `TRIAL_014_Entry_A_RR1.0_H24.0_B0.5` | Entry_A | 1.0 | 24.0h | 0.5 ATR | 36,800 | 28.8% | **-0.654 R** | 0.32 | -100.0% |

---

## 4. Segmentations & Sub-Regime Analysis

### By Volume Rank Bucket (MEXC Universe)
| Volume Rank | Trades ($n$) | Win Rate (Net) | 95% Wilson CI | Net Exp (R) | Profit Factor |
|---|---:|---:|---|---:|---:|
| **Unranked** | 42,700 | 24.70% | [24.3%, 25.1%] | **-0.787 R** | 0.28 |

### By BTC Macro Regime
| BTC Regime | Trades ($n$) | Win Rate (Net) | Net Exp (R) | Profit Factor |
|---|---:|---:|---|---:|---:|
| **BTC_Non_Uptrend** | 42,700 | 24.70% | **-0.787 R** | 0.28 |

---

## 5. Artifacts and Visualization Links
- **Primary Trade Database (Parquet):** `results/trades.parquet`
- **Primary Trade Database (CSV):** `results/trades.csv`
- **Trial Log (Multiple-Testing):** `results/trials.parquet`
- **Equity Curve Chart:** `reports/figures/phase4_equity_curve.png`
- **Drawdown Profile Chart:** `reports/figures/phase4_drawdown.png`
- **Entry Mode Comparison Chart:** `reports/figures/phase4_entry_modes.png`

---

## 6. Phase 4 Conclusion & Transition to Phase 5
All simulations strictly observed the non-holdout guard (< 2026-07-01). The holdout dataset remains pristine and locked for Phase 5 Walk-Forward Validation.
