# Limitations and Known Biases

Per Non-Negotiable Rule 2: *"Where bias cannot be removed, document it in docs/LIMITATIONS.md instead of hiding it."*

---

## 1. Survivorship Bias & Universe Reconstruction
- **Delisted Contract Purging:** MEXC public endpoints (`/contract/detail`, `/ticker`) only return active, currently listed contracts. When a perpetual contract or spot pair is delisted or halted, MEXC removes it from the catalog, making true historical point-in-time exchange listings inaccessible via public endpoints without external historical snapshot archives.
- **Impact:** Backtesting across symbols listed today introduces survivorship bias, as poor performers that went to zero and were delisted within the past 12 months cannot be reconstructed from public endpoints.
- **Mitigation:**
  - Strictly enforce point-in-time liquidity ranking on the available dataset: a symbol only enters the active universe at hour $t$ if its closed trailing 24h quote volume at $t$ met the threshold and had >= 250 closed 1H bars.
  - Transparently state the survivorship limitation in all headline tables and reports.

## 2. API Data Retention Limits
- **15m Candle Horizon:** Empirical testing reveals MEXC public REST endpoints restrict 15m candle lookback to **~360 days (~12 months)**. Calls requesting timestamps prior to this window return empty results.
- **1H Horizon:** 1H candles are retained for >= 730 days (> 2 years).
- **Consequence:** The maximum backtest date range for 15m execution is bounded to 360 days. This provides sufficient statistical sample size (~34,560 bars per symbol across 200+ symbols) but spans one full market regime cycle rather than multiple multi-year cycles.

## 3. Contract Units vs. Quote Turnover
- **Contract vs USDT Volume:** Perpetual futures klines report `vol` in number of contracts, while `amount` reports turnover in USDT. If volume were mistakenly computed as `vol * close` without applying `contractSize`, turnover would be distorted by orders of magnitude for symbols where `contractSize != 1.0` (e.g. `BTC_USDT` has `contractSize = 0.0001`).
- **Mitigation:** All universe liquidity rankings use the native `amount` (USDT turnover) field directly.

## 4. Intra-Bar Execution & Same-Bar Ambiguity
- **15m Granularity Limit:** When both the event stop level and the target (swing high) fall within the high-low range of a single 15m candle, the exact tick sequence is unknown without tick-level order book trades.
- **Resolution Policy:**
  - All primary/headline metrics treat same-bar collisions conservatively as **stop-first** (failure).
  - An optimistic bound (target-first) is logged alongside to measure parameter sensitivity.

## 5. Funding Rate & Friction Realism
- **Funding Rates:** Realized funding payments in crypto perpetuals can degrade returns on long positions during extreme bull runs or provide income during negative funding regimes. Historical 8-hour funding rates are downloaded and matched strictly to the holding period timestamps.
- **Slippage Scale:** Conservative slippage (5 bps default base, scaled upwards by liquidity rank bucket) is charged on market order entries and stop fills.
