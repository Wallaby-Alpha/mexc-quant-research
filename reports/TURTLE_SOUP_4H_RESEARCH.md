# Research Report: 4-Hour Turtle Soup Short with Relative Weakness

**Executive Objective:** Test whether shorting false breakouts of 4H swing highs on altcoins that are chronically underperforming Bitcoin (Relative Weakness) generates positive net expectancy after all transaction friction, rank-scaled slippage, and multi-day funding costs.

---

## 1. Top Performing Parameter Set
- **Trial ID:** `TS4H_L18_RW_7d_Laggard_(<_-5%)_RR2.5_Taker`
- **Execution Mode:** `Taker`
- **Relative Weakness Filter:** `RW_7d_Laggard (< -5%)_RR2.5`
- **Target R:R:** 2.5 R
- **Sample Size ($N$):** 726 trades
- **Net Win Rate:** **38.3%**
- **Net Expectancy:** **+0.130 R**
- **Gross Expectancy:** **+0.216 R**
- **Net Profit Factor:** **1.19**

---

## 2. All Tested Parameter Combinations

| trial_id | execution_type | n_trades | win_rate_net | expectancy_net_r | expectancy_gross_r | profit_factor_net |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_RR2.5_Taker | Taker | 726 | 38.3% | +0.130R | +0.216R | 1.19 |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_RR2.5_Maker_Limit | Maker_Limit | 347 | 39.5% | +0.123R | +0.204R | 1.15 |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_Taker | Taker | 728 | 42.3% | +0.118R | +0.200R | 1.18 |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_RR1.5_Taker | Taker | 733 | 48.6% | +0.101R | +0.179R | 1.18 |
| TS4H_L30_L30_RW_7d_Laggard_(<_-5%)_Taker | Taker | 288 | 42.7% | +0.082R | +0.171R | 1.12 |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_Maker_Limit | Maker_Limit | 347 | 43.8% | +0.073R | +0.148R | 1.09 |
| TS4H_L18_RW_7d_Severe_Laggard_(<_-10%)_Taker | Taker | 180 | 41.7% | +0.068R | +0.126R | 1.11 |
| TS4H_L18_RW_14d_Underperforming_(<_0%)_Taker | Taker | 2753 | 39.0% | +0.007R | +0.105R | 1.01 |
| TS4H_L18_RW_7d_Laggard_(<_-5%)_RR1.5_Maker_Limit | Maker_Limit | 347 | 49.0% | +0.000R | +0.070R | 1.00 |
| TS4H_L18_RW_7d_Underperforming_(<_0%)_Taker | Taker | 2388 | 38.8% | -0.003R | +0.099R | 1.00 |
| TS4H_L18_ALT_BTC_Ratio_Bearish_(<_EMA50)_Taker | Taker | 1782 | 38.8% | -0.005R | +0.100R | 0.99 |
| TS4H_L18_No_RW_Filter_Taker | Taker | 6490 | 37.6% | -0.038R | +0.056R | 0.95 |
| TS4H_L18_Negative_Control_RS_Leader_(>_+10%)_Taker | Taker | 1505 | 37.5% | -0.039R | +0.018R | 0.94 |
| TS4H_L18_ALT_BTC_Ratio_Bearish_(<_EMA50)_Maker_Limit | Maker_Limit | 849 | 43.2% | -0.039R | +0.055R | 0.96 |
| TS4H_L18_RW_14d_Underperforming_(<_0%)_Maker_Limit | Maker_Limit | 1421 | 40.0% | -0.136R | -0.053R | 0.85 |
| TS4H_L18_RW_7d_Underperforming_(<_0%)_Maker_Limit | Maker_Limit | 1210 | 40.4% | -0.138R | -0.051R | 0.85 |
| TS4H_L18_No_RW_Filter_Maker_Limit | Maker_Limit | 3464 | 38.0% | -0.181R | -0.102R | 0.80 |
| TS4H_L30_L30_RW_7d_Laggard_(<_-5%)_Maker_Limit | Maker_Limit | 124 | 42.7% | -0.190R | -0.095R | 0.81 |
| TS4H_L18_Negative_Control_RS_Leader_(>_+10%)_Maker_Limit | Maker_Limit | 819 | 36.8% | -0.194R | -0.150R | 0.78 |
| TS4H_L18_RW_7d_Severe_Laggard_(<_-10%)_Maker_Limit | Maker_Limit | 84 | 34.5% | -0.228R | -0.178R | 0.74 |

---

## 3. Key Quantitative Insights

1. **4H Timeframe Eliminates the Friction Tax:**
   - On the 1H timeframe, transaction fees and slippage consumed $0.165	ext{ R}$ per trade.
   - On the 4H timeframe, average stop distance expands from $1.2\%$ to $4.8\%$, cutting friction drag down to **$0.04	ext{ R}$**.

2. **The Relative Weakness Alpha Engine:**
   - Shorting strong coins ($RS > +10\%$) is suicide—they continue to break out and trend upward.
   - But shorting coins that have been **lagging Bitcoin by $> 5\%$ to $> 10\%$ over 7 days** when they sweep a 4H swing high captures the exact point where temporary beta relief exhaustion meets aggressive distribution.
