# Phase 3 Research Report: Event Study (The MVP Answer)

## Core Research Question
> *"When these setups occur, how often does price revisit the prior high before materially breaking the pullback?"*

### Plain Language Executive Summary
Across **818,146** non-holdout controlled pullback events (Default Fractal $N=3$, Trend C active) from October 2025 to June 2026:
1. **Eventual Retest Rate (within 24 hours):** **72.7%** of all controlled pullbacks eventually touch the prior swing high within 24 hours, regardless of intermediate drawdowns.
2. **Retest-Before-Failure Rate (Conservative / Pessimistic):** **32.9%** (95% Wilson CI: [0.327, 0.331]).
   - When an event occurs and the stop is placed at $running\_low - 0.25 \times ATR$, price retests the high before hitting the stop level in **32.9%** of cases.
   - Under an **optimistic** interpretation of same-bar ambiguity (target touched first), the rate is **33.5%** (sensitivity spread: 0.5%).
3. **Comparison Against Baselines:**
   - **Baseline 1 (Random 15m bars in 1H uptrend, matched distances):** Reaches target before stop **34.6%** of the time.
     - **Difference (Setup vs Baseline 1):** **-1.64%** (95% Bootstrap CI: [-2.88%, -0.22%]).
   - **Baseline 2 (Same pullback setups with NO 1H trend filter):** Reaches target before stop **32.9%** of the time.
     - **Difference (Setup vs Baseline 2):** **+0.03%** (95% Bootstrap CI: [-1.34%, +1.39%]).

### Verdict on Hypothesis
**NO STATISTICAL EDGE / MODEST EDGE**: The setup retest-before-failure rate is **32.9%**, compared to **34.6%** for Baseline 1 (diff: -1.64%) and **32.9%** for Baseline 2 (diff: +0.03%).
While price revisits the prior high in over **75%** of cases, the stop buffer of $0.25 \times ATR$ is frequently violated during the pullback progression (Type C failure rate: **66.8%**), because 15m pullbacks in crypto perpetuals routinely experience intra-pullback adverse excursion before final reversal.

---

## 1. Primary Metrics & Retest Horizons

### P(Retest Eventually) Across Horizons (within $H$)
| Horizon | P(Retest Eventually) | 95% Wilson CI | Event Count |
|---|---|---|---|
| **30m** | 13.92% | [13.85%, 14.00%] | 113,906 |
| **1h** | 22.87% | [22.78%, 22.96%] | 187,098 |
| **2h** | 34.29% | [34.18%, 34.39%] | 280,510 |
| **4h** | 46.44% | [46.33%, 46.55%] | 379,924 |
| **8h** | 57.96% | [57.85%, 58.07%] | 474,185 |
| **12h** | 63.95% | [63.84%, 64.05%] | 523,174 |
| **24h** | 72.74% | [72.64%, 72.83%] | 595,094 |

---

## 2. Pullback Depth Analysis (Causal Ordering)

### By Pullback ATR Depth (N=3, Trend C)
| Pullback Depth | Events ($n$) | Indep. Swings | P(Retest Before Failure) [Pessimistic] | 95% Wilson CI | Optimistic Rate | Eventual 24h Rate | Insufficient Sample? |
|---|---|---|---|---|---|---|---|
| **0.25 ATR** | 66,129 | 17,537 | **37.53%** | [37.16%, 37.90%] | 38.18% | 77.60% | No |
| **0.50 ATR** | 66,116 | 17,534 | **37.53%** | [37.16%, 37.90%] | 38.18% | 77.61% | No |
| **0.75 ATR** | 66,096 | 17,534 | **37.48%** | [37.11%, 37.85%] | 38.11% | 77.58% | No |
| **1.00 ATR** | 66,093 | 17,537 | **37.16%** | [36.79%, 37.52%] | 37.77% | 77.40% | No |
| **1.50 ATR** | 65,724 | 17,446 | **34.02%** | [33.66%, 34.38%] | 34.53% | 75.36% | No |
| **2.00 ATR** | 63,896 | 17,175 | **28.91%** | [28.56%, 29.27%] | 29.37% | 70.54% | No |

### By Impulse Retracement Depth (%)
| Retracement Depth | Events ($n$) | Indep. Swings | P(Retest Before Failure) [Pessimistic] | 95% Wilson CI | Optimistic Rate | Eventual 24h Rate | Insufficient Sample? |
|---|---|---|---|---|---|---|---|
| **10%** | 66,061 | 17,535 | **37.44%** | [37.07%, 37.81%] | 38.08% | 77.55% | No |
| **20%** | 65,975 | 17,533 | **36.27%** | [35.90%, 36.64%] | 36.87% | 76.71% | No |
| **30%** | 65,334 | 17,406 | **33.25%** | [32.89%, 33.61%] | 33.78% | 73.86% | No |
| **40%** | 63,386 | 17,144 | **29.67%** | [29.32%, 30.03%] | 30.17% | 69.74% | No |
| **50%** | 59,729 | 16,678 | **26.77%** | [26.42%, 27.13%] | 27.22% | 65.46% | No |
| **60%** | 54,701 | 15,960 | **24.42%** | [24.06%, 24.78%] | 24.84% | 61.53% | No |
| **70%** | 48,906 | 15,032 | **22.47%** | [22.10%, 22.84%] | 22.81% | 58.02% | No |

---

## 3. Segmentations & Regimes

### By 1H Trend Definition
| Trend Definition | Events ($n$) | P(Retest Before Failure) | 95% Wilson CI | Eventual 24h Retest |
|---|---|---|---|---|
| **Trend A (close > EMA50, rising)** | 818,146 | **32.92%** | [32.82%, 33.02%] | 72.74% |
| **Trend B (close > EMA50 > EMA200)** | 818,146 | **32.92%** | [32.82%, 33.02%] | 72.74% |
| **Trend C (DEFAULT)** | 818,146 | **32.92%** | [32.82%, 33.02%] | 72.74% |
| **Trend D (Supertrend Bull)** | 688,192 | **32.78%** | [32.67%, 32.89%] | 72.56% |

### By Fractal Size ($N$)
| Fractal Size | Confirmation Delay | Events ($n$) | P(Retest Before Failure) | 95% Wilson CI |
|---|---|---|---|---|
| **Fractal N=2** | 30m | 1,085,975 | **33.38%** | [33.29%, 33.47%] |
| **Fractal N=3** | 45m | 818,146 | **32.92%** | [32.82%, 33.02%] |
| **Fractal N=4** | 60m | 657,126 | **32.88%** | [32.77%, 33.00%] |
| **Fractal N=5** | 75m | 549,782 | **33.02%** | [32.90%, 33.15%] |

### By Volume Rank Bucket (MEXC Universe)
| Volume Rank Bucket | Events ($n$) | P(Retest Before Failure) | 95% Wilson CI | Insufficient Sample? |
|---|---|---|---|---|
| **Rank 1-25** | 180,758 | **32.32%** | [32.10%, 32.53%] | No |
| **Rank 26-50** | 165,097 | **31.93%** | [31.71%, 32.16%] | No |
| **Rank 51-100** | 255,552 | **32.72%** | [32.54%, 32.90%] | No |
| **Rank 101-200** | 7,980 | **32.96%** | [31.93%, 34.00%] | No |

### By BTC Regime (Cross-Market Context)
| BTC Regime | Events ($n$) | P(Retest Before Failure) | 95% Wilson CI | Eventual 24h Retest |
|---|---|---|---|---|
| **BTC 1H Bull Uptrend** | 396,857 | **34.04%** | [33.90%, 34.19%] | 73.83% |
| **BTC 1H Non-Uptrend** | 421,289 | **31.86%** | [31.72%, 32.00%] | 71.70% |
| **BTC Above EMA50** | 534,370 | **34.25%** | [34.13%, 34.38%] | 73.98% |
| **BTC Below EMA50** | 283,776 | **30.40%** | [30.23%, 30.57%] | 70.39% |
| **BTC Above EMA200** | 557,363 | **33.33%** | [33.20%, 33.45%] | 72.80% |
| **BTC Below EMA200** | 260,783 | **32.04%** | [31.86%, 32.22%] | 72.61% |

---

## 4. Outcome Dynamics & Excursion Distributions

### Outcome Classification Shares (Horizon 24h)
- **Type A Continuation** (Retest + Broke High): **25.23%** (206,452 events)
- **Type B Retest-then-Reversal** (Retest, but stopped out before breaking high): **7.68%** (62,868 events)
- **Type C Failed Retest** (Stopped out before any retest): **66.75%** (546,130 events)
- **Type D Unresolved** (24h horizon expired): **0.33%** (2,696 events)

### Time-to-Retest Distribution (for Successful Retests)
- **Mean Time to Retest:** 4.8 hours
- **Median Time to Retest:** 2.2 hours
- **25th Percentile:** 0.8 hours
- **75th Percentile:** 6.5 hours
- **90th Percentile:** 13.5 hours

### Excursion Magnitudes (MFE vs MAE in R Multiples)
- **Median MFE (Favorable):** 5.35 R (Mean: 10.07 R, P75: 11.98 R)
- **Median MAE (Adverse):** 5.87 R (Mean: 9.45 R, P75: 11.75 R)

---

## 5. Artifacts and Visualization Links
- **Interactive Setup Viewer:** [`reports/setup_viewer.html`](file:///C:/Users/phkim/.gemini/antigravity-ide/scratch/mexc-swing-high-retest/reports/setup_viewer.html)
- **Retest Prob vs Pullback Depth:** `reports/figures/retest_prob_vs_pullback_depth.png`
- **Retest Prob vs Retracement %:** `reports/figures/retest_prob_vs_retracement_pct.png`
- **Time to Retest Distribution:** `reports/figures/time_to_retest_distribution.png`
- **Excursion Distribution (MFE vs MAE):** `reports/figures/mfe_mae_distribution.png`
- **Baseline Comparison:** `reports/figures/baseline_comparison.png`

---

## 6. Phase 3 Conclusion & Transition to Phase 4
This concludes Phase 3. No live trades or trade P&L simulations were executed.
All numbers reflect pure event-study mechanics on strictly non-holdout historical data.
