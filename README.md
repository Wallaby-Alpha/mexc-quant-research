# MEXC Quantitative Crypto Research & Strategy Framework

A research-grade quantitative backtesting framework and automated trading tools for MEXC crypto markets, testing systematic market anomalies, execution frictions, relative strength factors, and liquidity sweeps.

---

## 🚀 Key Strategy Archetypes & Findings

### 1. Cross-Sectional Momentum Rotational Portfolio (`strategy/rotational_momentum.py`)
- **Concept:** Ranks 150+ liquid MEXC altcoins weekly by 30-day return vs. Bitcoin ($R_{\text{alt}} - R_{\text{btc}}$). Holds equal-weight positions in the Top 10 leaders.
- **Macro Risk Filter:** Rotates 100% to Cash (USDT) when Bitcoin is below its 50-day Daily EMA.
- **Performance:**
  - **Total Net Return:** **+78.8%** (CAGR: **+105.8%**)
  - **Sharpe Ratio:** **1.73** | **Max Drawdown:** **12.2%**
  - **Alpha vs. Bitcoin:** **+125.0%** (BTC returned -46.2% over the same period)
  - **Alpha vs. Alt Market:** **+127.0%** (Equal-weight universe returned -48.2%)
  - **Negative Control Spread:** Bottom 10 laggards returned **-68.0%** (a 147% performance spread).

### 2. 4-Hour Turtle Soup Short on Relative Weakness (`strategy/turtle_soup_4h.py`)
- **Concept:** Fades false breakouts of 3-day swing highs on altcoins lagging Bitcoin by $\ge 5\%$ over 7 days.
- **Performance:**
  - **Sample Size:** $N = 726$ trades
  - **Net Expectancy:** **$+0.130\text{ R}$** | **Gross Expectancy:** **$+0.216\text{ R}$**
  - **Net Win Rate:** **$38.3\%$** at 2.5 R:R (Breakeven: 28.6%) | **Profit Factor:** **1.19**

### 3. 4-Hour Swing High Retest + Volume Absorption (`strategy/absorption.py`)
- **Concept:** Buying 61.8% Fibonacci pullbacks with institutional volume absorption ($1.5\times$ volume with $\ge 40\%$ lower wick rejection).
- **Performance:**
  - **Net Win Rate:** **$47.5\% - 63.2\%$**
  - **Net Expectancy:** **$+0.084\text{ R}$ to $+0.335\text{ R}$** | **Profit Factor:** **1.19 – 1.90**

---

## 🛠️ Automated Weekly Rotation Scanner & Telegram Bot

The repository includes a production-ready scanner script in `scanner/weekly_rotation_scanner.py` that can be deployed to a **$4/month DigitalOcean Droplet**:
- Evaluates BTC 50-day EMA.
- Pulls live 30-day klines from MEXC public REST API across the top 150 pairs.
- Compares against local holdings in `portfolio_state.json` (applying the Rank 15 buffer rule to prevent churn).
- Pushes formatted **BUY / SELL / HOLD** orders directly to Telegram every Monday at 00:01 UTC.

Full droplet deployment guide is in [`scanner/deploy/README.md`](scanner/deploy/README.md).

---

## 📂 Repository Structure

- `/scanner`: Live weekly scanner script and DigitalOcean deployment configs.
- `/strategy`: Implementations of Rotational Momentum, Turtle Soup, Volume Absorption, and Trend Following.
- `/backtest`: Causal event-driven execution model with maker/taker fees, rank-scaled slippage, and funding.
- `/analysis`: Statistical confidence intervals, drawdown analysis, and event studies.
- `/reports`: Markdown research logs and high-resolution performance comparison figures.
- `/tests`: Complete test suite with strict no-lookahead guards.
