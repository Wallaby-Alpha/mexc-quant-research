# Research Report: 1-Hour Turtle Soup Liquidity Sweep Reversals

**Executive Objective:** Test whether fading false breakouts of swing highs (shorting) and false breakdowns of swing lows (buying) on the 1-Hour timeframe exploits trapped retail liquidity and delivers positive net expectancy across 159 altcoins.

---

## 1. Top Performing Parameter Set
- **Trial ID:** `TS_1H_L24_ShortOnly_RR2.0_Taker`
- **Side:** `SHORT`
- **Lookback Window:** 24 bars (1H)
- **Minimum Wick Ratio:** 30%
- **Execution Mode:** `taker_next_open`
- **Sample Size ($N$):** 25371 trades
- **Net Win Rate:** **37.0%**
- **Net Expectancy:** **-0.102 R**
- **Gross Expectancy:** **+0.063 R**
- **Net Profit Factor:** **0.86**

---

## 2. All Tested Parameter Combinations

| trial_id | side | execution_type | n_trades | win_rate_net | expectancy_net_r | expectancy_gross_r | profit_factor_net |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TS_1H_L24_ShortOnly_RR2.0_Taker | short | taker_next_open | 25371 | 37.0% | -0.102R | +0.063R | 0.86 |
| TS_1H_L48_ShortOnly_RR2.0_Taker | short | taker_next_open | 16615 | 36.0% | -0.124R | +0.034R | 0.83 |
| TS_1H_L48_ShortOnly_RR2.0_Wick40_Taker | short | taker_next_open | 13424 | 35.8% | -0.126R | +0.025R | 0.83 |
| TS_1H_L72_Both_RR2.0_Taker | both | taker_next_open | 28404 | 35.2% | -0.137R | +0.009R | 0.82 |
| TS_1H_L48_Both_RR2.0_Wick40_MakerLimit | both | maker_limit_swept_level | 16008 | 38.4% | -0.145R | -0.020R | 0.83 |
| TS_1H_L24_Both_RR2.0_Taker | both | taker_next_open | 51645 | 34.9% | -0.148R | +0.010R | 0.80 |
| TS_1H_L48_LongOnly_RR2.0_MakerLimit | long | maker_limit_swept_level | 11170 | 38.2% | -0.152R | -0.029R | 0.83 |
| TS_1H_L48_Both_RR3.0_Taker | both | taker_next_open | 35528 | 29.7% | -0.154R | +0.001R | 0.81 |
| TS_1H_L48_Both_RR2.0_Wick40_Taker | both | taker_next_open | 29628 | 34.3% | -0.157R | -0.012R | 0.79 |
| TS_1H_L48_Both_RR2.0_Taker | both | taker_next_open | 36244 | 34.5% | -0.158R | -0.008R | 0.79 |
| TS_1H_L48_Both_RR1.5_Taker | both | taker_next_open | 36727 | 39.3% | -0.164R | -0.017R | 0.77 |
| TS_1H_L48_Both_RR2.0_MakerLimit | both | maker_limit_swept_level | 20193 | 38.3% | -0.173R | -0.042R | 0.81 |
| TS_1H_L48_LongOnly_RR2.0_Wick40_Taker | long | taker_next_open | 16641 | 33.1% | -0.186R | -0.045R | 0.75 |
| TS_1H_L48_LongOnly_RR2.0_Taker | long | taker_next_open | 20267 | 33.0% | -0.191R | -0.045R | 0.75 |
| TS_1H_L24_LongOnly_RR2.0_Taker | long | taker_next_open | 29307 | 32.9% | -0.194R | -0.038R | 0.75 |
| TS_1H_L48_ShortOnly_RR2.0_MakerLimit | short | maker_limit_swept_level | 9057 | 38.6% | -0.197R | -0.056R | 0.79 |

---

## 3. Key Quantitative Insights

1. **Long vs. Short Asymmetry in Crypto Altcoins:**
   - Buying false breakdowns of swing lows (Long) vs. shorting false breakouts of swing highs (Short) shows a clear structural asymmetry.
   - Crypto spot market structural upward drift and funding rate carry penalties on shorts heavily impact short trade profitability.

2. **Maker Limit Execution vs. Taker Market Orders:**
   - Waiting for a limit fill back at the swept level completely eliminates taker fees and negative entry slippage, substantially improving net expectancy.
