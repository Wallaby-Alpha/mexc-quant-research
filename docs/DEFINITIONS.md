# DEFINITIONS (single source of truth)

## 1. Time conventions (anti-lookahead)
- Candle timestamp = candle OPEN time (UTC). A candle is "known" only at open_time + timeframe.
- Decisions are made at the CLOSE of a 15m candle t (known time = t_open + 15m).
- Orders triggered by a close-based signal are filled at the OPEN of candle t+1 (plus slippage).
  Never fill at the close of the signal candle.
- 1H data at decision time uses ONLY the last fully closed 1H candle (open_time + 1h <= decision time).
  Never join a still-forming 1H candle to 15m bars. 1H values are forward-filled onto 15m bars
  only after the 1H candle has closed.
- All indicators are causal: EMA via recursive form (adjust=False), ATR via Wilder smoothing.
  No centered windows, no full-sample normalization, no future-informed warm-up.
- Universe rank at decision time uses only data closed before that time.

## 2. Indicators
- ATR15 = Wilder ATR(14) on 15m candles.
- ATR_ref = ATR15 as known at swing-high confirmation time (frozen for that setup).
  Used for bucket classification and impulse sizing.
- ATR_entry = ATR15 known at the entry decision bar. Used for stop buffers. Record both.
- EMA50 / EMA200 on 1H close.
- "EMA50 rising" = EMA50[t] > EMA50[t-k] using closed 1H bars, k = 3 (configurable).

## 3. Trend definitions (evaluated on last closed 1H bar)
- DEFAULT / Trend C: close > EMA50 AND EMA50 > EMA200 AND EMA50 rising
- Trend A: close > EMA50 AND EMA50 rising
- Trend B: close > EMA50 > EMA200
- Trend D: Supertrend(period=10, mult=3.0) direction is up (configurable)
- Symbols need >= 200 closed 1H bars + warm-up before trend is defined. Otherwise trend = UNDEFINED (not tradable).

## 4. Swing high (N/N fractal, default N=3)
- Candle T is a swing high if high[T] is STRICTLY greater than highs of the N candles before AND the N after.
- Confirmation: at close of candle T+N. Earliest possible action: open of T+N+1.
- The setup may not exist, be logged, or use ATR/EMA values before that confirmation time.

## 5. Preceding impulse
- swing_low = lowest low in the window [T-L, T], L = 32 bars (8h, configurable).
- impulse = swing_high - swing_low.
- impulse_pct = impulse / swing_low. impulse_ATR = impulse / ATR_ref.
- "Meaningful impulse" = impulse_ATR >= 2.0 (configurable). Swing highs below this are not setups.
- Record swing_low timestamp and price.

## 6. Pullback
- pullback_low(t) = lowest low over bars T+1 .. t (running minimum; causal).
- pullback_ATR(t) = (swing_high - pullback_low(t)) / ATR_ref
- retracement_pct(t) = (swing_high - pullback_low(t)) / (swing_high - swing_low)
- CAUSAL BUCKETING RULE: bucket assignment uses depth AT SIGNAL TIME only. Never label a setup by its
  final/deepest pullback (that leaks the future). Final depth may be stored as a diagnostic column
  named `diag_final_depth_*` and must never be used as a filter or grouping key.

## 7. Threshold-crossing events (event-study unit)
- For each swing high, and for each pullback_ATR threshold in {0.25, 0.50, 0.75, 1.00, 1.50, 2.00}
  (and separately each retracement threshold in {10,20,30,40,50,60,70}%), an EVENT fires at the
  first 15m CLOSE (t >= T+N) at which the running depth first meets or exceeds that threshold.
- An event's bucket = [threshold, next threshold) based on depth at that close.
- "Controlled pullback" = event fires while the 1H trend is valid AND the stop level (see 9) has not been hit.
- Outcomes are measured strictly AFTER the event bar's close. Reference price = open of next bar.
- Events from the same swing high are correlated: report stats both per-event and per-swing-high
  (cluster-robust CIs, bootstrap by swing high and by day).

## 8. Retest
- PRIMARY: a later bar's high >= swing_high (wick touch counts).
- STRICT variant: a later 15m close >= swing_high.
- NEAR variant: a later bar's high >= swing_high - 0.10 * ATR_ref.
- Record all three; primary is used in headlines.
- "Broke high" = a 15m close > swing_high + 0.25 * ATR_ref (configurable).
- Extensions recorded after retest: max extension beyond high in ATR, and whether +0.5 / +1 / +2 ATR reached.

## 9. Failure / invalidation ("materially breaking the pullback low")
- Event stop level = pullback_low_at_signal - stop_buffer * ATR_ref, stop_buffer default 0.25 (0, 0.10, 0.25, 0.50 tested).
- Failure = a later bar's LOW <= stop level.
- Trend invalidation (default): a closed 1H candle closes below EMA50. Alternatives: below prior 1H swing low;
  below EMA50 by X ATR. Trend invalidation BEFORE the event cancels the setup. AFTER an event, record it
  as a separate flag but do not alter the outcome unless a trend-exit rule is being tested.

## 10. Setup lifecycle
- One setup per (symbol, swing high). States: PENDING -> ARMED (event fired) -> RESOLVED
  (retest / failure) | CANCELLED (trend invalid before event) | SUPERSEDED (a new higher confirmed
  swing high on same symbol) | EXPIRED (horizon exceeded, default 24h) .
- Event study records ALL setups regardless of overlap.
- Trade simulation (Phase 4): max ONE open position per symbol; a new signal on a symbol with an
  open position is skipped and logged as "skipped_overlap".

## 11. Outcome classification (measured from event bar, horizon H = 24h default)
- Type A Continuation: retest, then "broke high" occurs before failure, within H.
- Type B Retest-then-reversal: retest occurs, then failure (stop level hit) occurs before "broke high".
- Type C Failed retest: failure occurs before any retest.
- Type D Unresolved: none of the above within H.
- Same-bar ambiguity: if ONE candle's range contains both the stop level and the swing high, record
  `ambiguous_same_bar = True`. Headline stats treat it as stop-first (conservative). Also report
  an optimistic bound (target-first) so the sensitivity is visible.
- Also record: MFE, MAE (in ATR and R), time_to_retest, time_to_failure, distance_to_high (ATR),
  distance_to_stop (ATR), potential R:R at event, all three retest variants.

## 12. Point-in-time universe
- Each hour (on 1H close), for each symbol compute trailing 24h quote volume in USDT from candles
  closed at or before that time. Rank all ELIGIBLE symbols. Ranks apply to the following hour.
- Eligible = has >= 250 closed 1H bars of history at that time, trailing 24h quote volume >= min_volume_usdt
  (default 1,000,000 USDT, configurable), not excluded.
- Excluded by default (all configurable): BTC, ETH, stablecoins, leveraged/UP/DOWN/BULL/BEAR tokens,
  wrapped/staked derivatives of BTC/ETH, symbols with > X% missing candles in trailing 30 days.
- Universe size N in {50, 100, 200, 300, 500}, default 300.
- Rank buckets: 1-25, 26-50, 51-100, 101-200, 201-300, 301-500 (only those within universe size).
- FUTURES CAVEAT: volume fields may be in contracts, not USDT. Convert using contract size or use
  the turnover field if provided. Verify in Phase 0 and record in ASSUMPTIONS.md.

## 13. Execution model (Phase 4)
- Entry/exit fees: taker on market fills, maker on resting limit fills (limit fills require price to trade
  THROUGH the level, not just touch it). Verify current MEXC fee schedule in Phase 0 and put the source
  in config comments. Use conservative defaults.
- Slippage per side default 5 bps, scaled up for lower volume ranks (config: rank-bucket -> bps).
- Funding (futures): apply the historical funding rate for each funding timestamp held. If funding
  history is unavailable, apply a conservative constant and flag it.
- Stops fill at stop price minus slippage; if a bar gaps beyond the stop, fill at the bar open.
- Same-bar rule for trades: stop assumed hit first when both stop and target are inside one bar.
- Time stop exits at the OPEN of the next bar after the max holding period elapses.

## 14. Entry modes (Phase 4)
- Entry A Immediate: signal at the close of the event bar; fill at next open.
- Entry B 15m reversal: after the event, first 15m bar that CLOSES above the prior 15m bar's high;
  fill at next open.
- Entry C Lower-high break: after the event, track the most recent confirmed 15m lower high
  (1/1 fractal default, configurable) that is below the swing high; signal on first close above it;
  fill at next open.
- Entry D: plugin interface only.
- Trade rejected if R:R = (swing_high - entry) / (entry - stop) < min_RR (test 1.0 ... 3.0).
- Default target = swing_high. Default stop = pullback_low_at_signal - 0.25 * ATR_entry.
- Time stop tested: 1, 2, 4, 8, 12, 24 hours.

## 15. Market Regimes and BTC Filters (Phase 5)
All regime indicators use strictly trailing, closed data (anti-lookahead).
- **BTC Filter Variants** (evaluated on last closed 1H BTC bar at decision time):
  1. `none`: No BTC conditioning.
  2. `btc_trend_c`: BTC 1H is in valid Trend C (close > EMA50 > EMA200, EMA50 rising).
  3. `btc_above_ema50`: BTC 1H close > EMA50.
  4. `btc_above_ema200`: BTC 1H close > EMA200.
  5. `btc_declining`: BTC 1H close < EMA50 AND EMA50 falling (EMA50[t] < EMA50[t-3]).
- **BTC Return Regime** (trailing 24h return on closed 1H bars):
  - `BTC Up`: btc_return_24h >= 0.0
  - `BTC Down`: btc_return_24h < 0.0
- **BTC Volatility Regime**:
  - `BTC Vol High`: trailing 1H ATR14 > trailing 30-day (720 bars) rolling median ATR.
  - `BTC Vol Low`: trailing 1H ATR14 <= trailing 30-day rolling median ATR.
- **Alt Volatility Regime**:
  - `Alt Vol High`: symbol 15m ATR14 > trailing 30-day (2880 bars) rolling median ATR.
  - `Alt Vol Low`: symbol 15m ATR14 <= trailing 30-day rolling median ATR.
- **Market Volume Regime**:
  - `Market Vol High`: total trailing 24h market quote volume > trailing 30-day rolling median.
  - `Market Vol Low`: total trailing 24h market quote volume <= trailing 30-day rolling median.
- **Macro Trend State**:
  - `Bull`: BTC 1H Trend C is True.
  - `Bear`: BTC 1H close < EMA50 AND EMA50 < EMA200 AND EMA50 falling.
  - `Chop`: Neither Bull nor Bear (EMAs tangled or oscillating).

