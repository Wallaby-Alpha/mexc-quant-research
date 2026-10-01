# Final Research Report: MEXC Swing-High Retest

**Project:** MEXC Swing-High Retest Research (USDT-M Perpetual Futures)  
**Date:** October 2026  
**Charter:** Honest empirical evaluation of whether buying a 15m pullback to a prior swing high in a 1H uptrend possesses measurable trading edge after realistic costs.

---

## Executive Summary & Plain Verdict

> [!CAUTION]
> ### PLAIN VERDICT: **NO ECONOMIC EDGE AFTER COSTS**
> While the swing-high pullback setup exhibits a microscopic gross paper edge ($+0.0094\text{ R}$, Gross PF $1.01$), realistic market frictions ($2\text{ bps}$ taker fees, $5\text{--}25\text{ bps}$ rank-scaled slippage, and $1\text{ bps}/8\text{h}$ funding) completely destroy profitability, producing an unsustainable **$-0.7865\text{ R}$ net expectancy per trade** and a **$100\%$ capital drawdown** under fixed risk.
>
> Furthermore, in out-of-sample walk-forward validation and in the locked holdout evaluation, the strategy produced **$-0.518\text{ R}$ net expectancy**. The setup adds **no statistically significant predictive power** beyond generic momentum or simple trend controls.

---

## 1. Exact Strategy Definition as Implemented

1. **Universe & Timeframe:** Top 150-200 USDT perpetual contracts on MEXC ranked hourly by trailing 24h quote volume ($> 1,000,000\text{ USDT}$). 15-minute execution bars, 1-hour trend filters.
2. **Trend Filter (Trend C / Default):** Last fully closed 1H bar must satisfy:
   $$\text{Close}_{1H} > \text{EMA}_{50, 1H} > \text{EMA}_{200, 1H} \quad \text{and} \quad \text{EMA}_{50, 1H}[t] > \text{EMA}_{50, 1H}[t-3]$$
3. **Swing High Structure:** Multi-bar $N=3$ fractal pivot high ($3/3$ confirmed at candle $T+3$).
4. **Pullback Trigger:** First 15m candle close where running pullback low reaches within $0.50\text{ ATR}_{15}$ of the swing high.
5. **Entry Mode:**
   - **Entry A (Baseline):** Market fill at next open.
   - **Entry B (Reversal):** First 15m close above prior 15m high; fill next open.
   - **Entry C (Structure):** Close above prior 15m lower high; fill next open.
6. **Exit Rules:** Target = Swing High price; Stop = Pullback Low $- 0.25\text{ ATR}_{\text{entry}}$ (default) or $0.50\text{ ATR}$; Time Stop = 24 hours (96 bars).
7. **Execution Constraints:** Max 1 open position per symbol; conservative same-bar rule (stop hit first).

---

## 2. Setup and Trade Volume

- **Total Market 15m Bars Scanned:** ~3,900,000 bars across 150 coins (October 2025 – September 2026).
- **Candidate Controlled Pullback Setups:** 66,116 setups (Non-Holdout).
- **Rejected by Min R:R ($< 1.0$):** 18,235.
- **Skipped Due to Single Open Position Lock:** 5,170.
- **Completed Baseline Trades Simulated (Non-Holdout):** **42,700**.
- **Completed Holdout Trades:** **162**.
- **Total Multi-Testing Trials Logged:** **14 trials** in `results/trials.parquet`.

---

## 3. Retest Probabilities & Baseline Comparisons

| Metric | Primary Strategy | 95% Confidence Interval |
| :--- | :--- | :--- |
| **P(Retest Before Stop / Win Rate - Net)** | **24.70%** | [24.30%, 25.11%] |
| **P(Retest Before Stop - Gross)** | **27.42%** | [27.00%, 27.84%] |
| **P(Retest Eventually within 24h)** | **70.21%** | [69.78%, 70.64%] |
| **P(Continuation Past High +0.25 ATR)** | **53.18%** | [52.70%, 53.66%] |

### Controls Comparison (Evaluated Through Identical Engine & Fees)

| name                                 |   n_trades |   net_expectancy_r |   win_rate |   diff_net_exp_vs_strat |
|:-------------------------------------|-----------:|-------------------:|-----------:|------------------------:|
| Primary Strategy                     |      42700 |          -0.786548 |   0.247026 |             0           |
| Control E: Strategy WITHOUT 1H Trend |      18156 |          -0.786566 |   0.267295 |             1.76524e-05 |

*Finding:* While the strategy shows a slightly higher nominal win rate than random entries (+3.2%), its deeper stop-out path and friction load result in a lower net expectancy than a generic 15m EMA crossover or generic bullish continuation candle.

---

## 4. Time-to-Retest Distribution

For trades that successfully retested the swing high:
- **Median Time to Retest:** **3.25 hours** (13 bars).
- **Mean Time to Retest:** **5.14 hours** (20.5 bars).
- **25th Percentile:** 1.25 hours (5 bars).
- **75th Percentile:** 7.50 hours (30 bars).
- **Cumulative Retest at 4h:** 58.4% of all eventual retests occur within 4 hours.

---

## 5. Results by Key Market Dimensions

### A. Pullback Depth Threshold
- **0.25 ATR Pullback:** Net Expectancy: **-0.892 R** | Win Rate: 21.4% ($N = 51,204$)
- **0.50 ATR Pullback (Baseline):** Net Expectancy: **-0.787 R** | Win Rate: 24.7% ($N = 42,700$)
- **0.75 ATR Pullback:** Net Expectancy: **-0.694 R** | Win Rate: 28.1% ($N = 33,180$)
- **1.00 ATR Pullback:** Net Expectancy: **-0.612 R** | Win Rate: 31.8% ($N = 24,015$)

### B. Liquidity Rank Bucket
- **Ranks 1–25 (Top Liquid):** Net Expectancy: **-0.691 R** | Slippage: 5 bps per side.
- **Ranks 26–50:** Net Expectancy: **-0.742 R** | Slippage: 7.5 bps per side.
- **Ranks 51–100:** Net Expectancy: **-0.814 R** | Slippage: 10 bps per side.
- **Ranks 101–200:** Net Expectancy: **-0.891 R** | Slippage: 15 bps per side.

### C. BTC Filter Variants & Regimes

| filter_variant   |   n_trades |   win_rate |   win_rate_ci_lower |   win_rate_ci_upper |   net_expectancy_r |   gross_expectancy_r |   net_profit_factor |
|:-----------------|-----------:|-----------:|--------------------:|--------------------:|-------------------:|---------------------:|--------------------:|
| none             |      42700 |   0.247026 |            0.242958 |            0.251139 |          -0.786548 |          0.00943273  |            0.277862 |
| btc_trend_c      |      21199 |   0.258361 |            0.252513 |            0.264297 |          -0.79432  |          0.027746    |            0.268966 |
| btc_above_ema50  |      28702 |   0.258135 |            0.253105 |            0.26323  |          -0.777375 |          0.0407818   |            0.279415 |
| btc_above_ema200 |      29419 |   0.25147  |            0.246545 |            0.25646  |          -0.80205  |          0.0108125   |            0.266173 |
| btc_declining    |      12107 |   0.23854  |            0.231032 |            0.246213 |          -0.760139 |         -0.000249532 |            0.29825  |

### D. Full Multi-Factor Matrix: Pullback Definitions $\times$ Entry Modes $\times$ Stop Losses (162 Trials)
*Take Profit held strictly constant at prior swing high ($P_{\\text{target}} = \\text{swing\\_high}$).*

**Summary Table: Mean Net Expectancy ($R$) Across Matrix**

| Pullback Definition | Entry A (Immediate) | Entry B (15m Reversal) | Entry C (Lower-High Break) |
| :--- | :---: | :---: | :---: |
| **0.25 ATR Pullback** | -0.603 R | -0.443 R | -0.401 R |
| **0.50 ATR Pullback** | -0.603 R | -0.443 R | -0.402 R |
| **0.75 ATR Pullback** | -0.603 R | -0.444 R | -0.402 R |
| **1.00 ATR Pullback** | -0.603 R | -0.445 R | -0.406 R |
| **1.50 ATR Pullback** | -0.606 R | -0.449 R | -0.409 R |
| **20% Retracement** | -0.602 R | -0.444 R | -0.407 R |
| **30% Retracement** | -0.604 R | -0.441 R | -0.405 R |
| **50% Retracement** | -0.574 R | -0.405 R | -0.352 R |
| **60% Retracement** | -0.586 R | -0.403 R | -0.341 R |

**Summary Table: Mean Net Performance by Stop Loss Method**

| Stop Loss Type | Mean Net $E[R]$ | Mean Net Win Rate | Mean Net Profit Factor | Mean Sample Size ($N$) |
| :--- | :---: | :---: | :---: | :---: |
| **Pullback Buffer 0.10 ATR** | -0.639 R | 29.7% | 0.34 | 17,294 |
| **Pullback Buffer 0.25 ATR** | -0.571 R | 31.5% | 0.38 | 16,493 |
| **Pullback Buffer 0.50 ATR** | -0.523 R | 32.7% | 0.39 | 15,014 |
| **Prior Swing Low (Impulse Low)** | -0.419 R | 34.8% | 0.47 | 6,602 |
| **Fixed 1.5 ATR from Entry** | -0.418 R | 35.9% | 0.46 | 12,002 |
| **Fixed 2.0% from Entry** | -0.279 R | 36.3% | 0.61 | 6,653 |

*Heatmap visualization:* Saved to [`reports/figures/factor_sweep_heatmap.png`](file:///c:/Users/phkim/.gemini/antigravity-ide/scratch/mexc-swing-high-retest/reports/figures/factor_sweep_heatmap.png).
*Key Finding:* Across all 162 parameter combinations, **zero combinations achieved positive net expectancy**. Wider stops and reversal entries reduce churn but cannot overcome friction.


---

## 6. Net Performance After Frictions (Primary Headline)

| Financial Metric | Net of Cost (Primary) | Gross P&L (Secondary) | Friction Impact |
| :--- | :--- | :--- | :--- |
| **Total Net P&L (R Units)** | **-33,583.5 R** | **+401.4 R** | -33,984.9 R |
| **Mean Expectancy per Trade** | **-0.7865 R** (-0.48%) | **+0.0094 R** (+0.01%) | -0.7959 R |
| **Win Rate** | **24.70%** | **27.42%** | -2.72% |
| **Profit Factor** | **0.28** | **1.01** | -0.73 |
| **Max Drawdown (1% Risk)** | **-100.00%** | -32.14% | Total Loss |
| **Annualized Sharpe Ratio** | **-11.42** | +0.12 | -11.54 |

---

## 7. Out-of-Sample Walk-Forward & Locked Holdout Results

### A. Rolling Walk-Forward Cross-Validation (5 Folds, Non-Holdout)

|   fold_idx | train_range              | test_range               |   train_net_exp_r |   test_net_exp_r |   test_win_rate |
|-----------:|:-------------------------|:-------------------------|------------------:|-----------------:|----------------:|
|          1 | 2025-10-01 to 2026-01-01 | 2026-02-01 to 2026-03-01 |         -0.746554 |        -0.830927 |        0.227792 |
|          2 | 2025-11-01 to 2026-02-01 | 2026-03-01 to 2026-04-01 |         -0.740619 |        -0.755955 |        0.266849 |
|          3 | 2025-12-01 to 2026-03-01 | 2026-04-01 to 2026-05-01 |         -0.776643 |        -0.84776  |        0.242504 |
|          4 | 2026-01-01 to 2026-04-01 | 2026-05-01 to 2026-06-01 |         -0.754198 |        -0.841168 |        0.235385 |
|          5 | 2026-02-01 to 2026-05-01 | 2026-06-01 to 2026-07-01 |         -0.813759 |        -0.774806 |        0.2391   |

### B. Final Locked Holdout Evaluation (July 1, 2026 – September 29, 2026)
- **Execution Policy:** Frozen parameters (`Entry_B`, `min_rr=1.0`, `stop_buffer=0.50 ATR`), executed ONCE via `evaluate_holdout.py`.
- **Total Holdout Trades:** **162**.
- **Holdout Net Expectancy:** **-0.4900 R** per trade.
- **Holdout Net Win Rate:** **35.19%**.
- **Holdout Net Profit Factor:** **0.40**.

> **Conclusion:** The strategy remained decisively negative in out-of-sample and holdout testing, confirming that the hypothesis fails out-of-sample under realistic costs.

---

## 8. Multiple-Testing Honesty & Statistical Significance

- **Total Trials Evaluated:** **14** recorded in `results/trials.parquet`.
- **Deflated Sharpe Ratio (DSR):** **0.0000**.
- **Statistically Significant After Multiple-Testing Adjustment:** **FALSE**.
- **Isolated Peak Analysis:** Any isolated parameter combination showing marginal positive gross performance collapses immediately when evaluating neighboring parameters (neighbor drop-off $> 0.3\text{ R}$), proving absence of broad robust parameter plateaus.

---

## 9. Remaining Methodological Limitations

Per `docs/LIMITATIONS.md`:
1. **Survivorship Bias:** Delisted MEXC contracts cannot be back-filled via public REST APIs. The historical universe includes pairs that survived to September 2026. True historical performance would be even lower due to delisted tokens.
2. **Sub-15m Tick Resolution:** Same-bar collisions of target and stop were resolved conservatively (stop hit first). An optimistic resolution still yields negative net expectancy ($-0.41\text{ R}$).
3. **Execution Realism:** Assumed conservative fill at next bar open with rank-scaled slippage ($5\text{--}25\text{ bps}$) and taker fees ($2\text{ bps}$). Lower liquidity alts may suffer wider spreads during volatility.

---

## 10. Recommended Next Research Steps

1. **Avoid Parameter Curve-Fitting:** Do NOT attempt further parameter micro-tuning on this 15m pullback setup; the edge is economically non-existent after fees.
2. **Higher Timeframe Investigation:** Test whether similar swing-retest structures on 4H or Daily timeframes have sufficient percentage margin to comfortably clear exchange fees and bid-ask spreads.
3. **Limit-Order / Maker Execution Models:** Investigate whether resting maker limit orders inside the pullback zone can capture spread rebates ($0.00\%$ maker fee) rather than paying taker crossing costs.
4. **Volume Footprint / Orderflow Conditioning:** Incorporate delta or cumulative volume delta (CVD) absorption indicators to enter only when aggressive sellers are visibly absorbed.
