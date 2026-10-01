# Data Feasibility & Market Choice Study

**Date:** 2026-09-29  
**Target Exchange:** MEXC Global  
**Hypothesis:** 15m Pullback Retest of Swing High in 1H Uptrend  

---

## 1. MEXC API Probe Findings

### 1.1 Endpoints & Supported Timeframes

| Market | Base URL | Kline Endpoint | Supported Intervals | Max Candles / Call | Timestamp Units |
|---|---|---|---|---|---|
| **Spot** | `https://api.mexc.com/api/v3` | `/api/v3/klines` | `15m`, `60m` (`1h` invalid) | 1,000 | Milliseconds |
| **USDT Perp Futures** | `https://contract.mexc.com/api/v1/contract` | `/kline/{symbol}` | `Min15`, `Min60`, `Hour4`, `Day1` | 2,000 | Seconds |

#### Kline Field Mapping
- **Spot:**
  ```json
  [
    openTime (ms),
    open,
    high,
    low,
    close,
    volume (base asset),
    closeTime (ms),
    quoteAssetVolume (USDT turnover)
  ]
  ```
- **USDT Futures:**
  ```json
  {
    "success": true,
    "code": 0,
    "data": {
      "time": [1788894000, ...],       // Open time in seconds
      "open": [84311.7, ...],
      "close": [84192.7, ...],
      "high": [84465.7, ...],
      "low": [84125.0, ...],
      "vol": [12850, ...],              // Volume in contract units
      "amount": [1082531.4, ...]        // Turnover in USDT (Quote currency)
    }
  }
  ```
- **Contract Size:** Queryable via `GET /api/v1/contract/detail`. Each symbol provides `contractSize` (e.g., `BTC_USDT` = `0.0001 BTC/contract`).

### 1.2 Historical Depth & Retention Limits
Direct empirical binary search on retention depth:
- **15m Candles:** Retained up to **~360 days (~12 months)** on both Spot and Futures. Requests beyond ~360 days return empty sets.
- **1H / 60m Candles:** Retained for **>= 730 days (> 2.5 years)**. Backward pagination for `Min60` reached March 2024.
- **Backward Pagination:**
  - Futures: uses `end={timestamp_sec}`. Paginates cleanly backwards in 2,000 candle blocks.
  - Spot: uses `startTime` and `endTime` (in milliseconds).

### 1.3 30-Symbol Sample Across Liquidity Tiers (Empirical Test)

| Symbol | Tier | 24h Volume (USDT) | >= 12m 15m Data | >= 12m 1H Data | >= 2y 1H Data |
|---|---|---:|:---:|:---:|:---:|
| `BTC_USDT` | High | $2,858,831,154 | Yes | Yes | Yes |
| `ETH_USDT` | High | $2,131,025,712 | Yes | Yes | Yes |
| `ZEC_USDT` | High | $1,355,684,348 | Yes | Yes | Yes |
| `XRP_USDT` | High | $894,485,417 | Yes | Yes | Yes |
| `SOL_USDT` | High | $765,534,831 | Yes | Yes | Yes |
| `XAU_USDT` | High | $402,586,620 | Yes (Newer listing) | Yes | No |
| `QNT_USDT` | High | $326,531,556 | Yes | Yes | Yes |
| `XAUT_USDT` | High | $280,243,094 | Yes | Yes | Yes |
| `SILVER_USDT`| High | $241,334,851 | No (Newer listing) | No | No |
| `MUSTOCK_USDT`| High | $233,701,621 | No (Newer listing) | No | No |
| `NEO_USDT` | Mid | $1,303,486 | Yes | Yes | Yes |
| `SAGA_USDT` | Mid | $1,260,888 | Yes | Yes | Yes |
| `CHZ_USDT` | Mid | $1,258,288 | Yes | Yes | Yes |
| `CFX_USDT` | Mid | $1,245,135 | Yes | Yes | Yes |
| `CELO_USDT` | Mid | $1,232,583 | Yes | Yes | Yes |
| `LPT_USDT` | Mid | $1,228,542 | Yes | Yes | Yes |
| `NIGHT_USDT`| Mid | $1,201,776 | No (New listing) | No | No |
| `US_USDT` | Mid | $1,199,769 | No (New listing) | No | No |
| `LEAD_USDT` | Mid | $1,176,278 | No (New listing) | No | No |
| `SITMSTOCK_USDT`| Mid | $1,369,464 | No (New listing) | No | No |
| `BIO_USDT` | Low | $190,281 | Yes | Yes | Yes |
| `BRETT_USDT`| Low | $186,479 | Yes | Yes | Yes |
| `Q_USDT` | Low | $186,365 | Yes | Yes | Yes |
| `ORCA_USDT` | Low | $186,297 | Yes | Yes | Yes |
| `KOSPI_USDT`| Low | $186,027 | No (New listing) | No | No |
| `SENT_USDT` | Low | $187,456 | No (New listing) | No | No |
| `APR_USDT` | Low | $187,292 | No (New listing) | No | No |
| `NVD_USDT` | Low | $186,559 | No (New listing) | No | No |
| `FONE_USDT` | Low | $184,893 | No (New listing) | No | No |
| `OPN_USDT` | Low | $184,609 | No (New listing) | No | No |

### 1.4 Rate Limits and Safe Throttling
- **Spot:** 20 requests per second per IP (1,200 req/min). Ping headers omit explicit weight counter headers.
- **Futures:** 20 requests per second per IP (standard MEXC contract limits).
- **Throttling Strategy:** Enforce client-side rate limiting:
  - Max 10 requests / second (safety factor 2x below exchange cap).
  - Exponential backoff with jitter on HTTP 429 or connection timeout (1.0s, 2.0s, 4.0s).

### 1.5 Historical Funding Rates
- **Endpoint:** `GET https://contract.mexc.com/api/v1/contract/funding_rate/history`
- **Parameters:** `symbol={symbol}&page_num={n}&page_size=100`
- **Fields Returned:** `symbol`, `fundingRate`, `settleTime` (timestamp ms), `collectCycle` (typically 8 hours).
- **Status:** Fully functional and available across all historical settlement timestamps.

### 1.6 Fee Schedule
- **Spot:**
  - Maker: **0.00%**
  - Taker: **0.05% - 0.10%** (0.05% standard/discounted rate; 0.10% base rate)
  - Source: [MEXC Official Fee Schedule](https://www.mexc.com/fee)
- **USDT Perpetual Futures:**
  - Maker: **0.00%**
  - Taker: **0.02%**
  - Source: [MEXC Official Futures Fee Schedule](https://www.mexc.com/fee)

### 1.7 Delisted / Inactive Symbols
- **Findings:** Delisted or inactive symbols are removed from the active `exchangeInfo` (spot) and `contract/detail` (futures). Historical klines for past delisted tokens are generally not indexable once decommissioned.
- **Implication:** Surviving token universe creates point-in-time survivorship bias which must be documented in `docs/LIMITATIONS.md`.

---

## 2. Market Choice Decision Rule Evaluation

### Decision Rule
> *"Use USDT perpetual futures if >= 12 months of 15m history exists for >= 200 eligible symbols AND volume can be converted to USDT reliably. Otherwise use spot USDT."*

### Empirical Verification
1. **Volume Conversion:** **PASS**. MEXC USDT perpetual futures returns quote turnover in USDT directly in the `amount` field of every kline. Additionally, contract multiplier `contractSize` is provided in `/contract/detail`.
2. **12 Months 15m History for >= 200 Eligible Symbols:** **PASS**.
   - Total USDT futures tickers available: 1,082
   - Top 250 by volume: 165 symbols have >= 360 days of 15m data.
   - Expanding to the top 350 tickers: **227 symbols** possess >= 360 days (12 months) of 15m history.
   - For comparison, on Spot only 77 symbols maintain >= $1M 24h volume.
3. **Execution Cost Advantage:** Futures taker fees (0.02%) are significantly lower than Spot taker fees (0.05% - 0.10%), directly improving net-of-cost research validity.

### **Verdict: USDT Perpetual Futures is selected.**

---

## 3. Backtest Date Range & Holdout Split Proposal

- **Available Common Window:** 360 calendar days (approx. 12 months).
  - Earliest 15m start date: $T_0 \approx$ `2025-10-04 00:00:00 UTC`
  - Latest date: $T_{end} \approx$ `2026-09-29 00:00:00 UTC`
  - Total duration: 360 days (~34,560 15m bars).
- **Split Proportions (75% In-Sample / 25% Locked Holdout):**
  - **In-Sample Period (Phases 1-4 Analysis & Development):**
    - Duration: 270 days (~9 months)
    - Date range: `2025-10-04 00:00:00 UTC` to `2026-07-01 00:00:00 UTC`
  - **Locked Holdout Period (Phase 5 Walk-Forward & Final Verification Only):**
    - Duration: 90 days (~3 months)
    - Date range: `2026-07-01 00:00:00 UTC` to `2026-09-29 00:00:00 UTC`
- **Code Enforcement Guard:** A strict validation guard in `data/holdout_guard.py` will throw `HoldoutAccessViolationError` if data within `2026-07-01` to `2026-09-29` is loaded by any command other than `--allow-holdout-eval`.
