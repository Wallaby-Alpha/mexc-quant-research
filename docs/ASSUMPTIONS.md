# Project Assumptions Log

Per Non-Negotiable Rule 4: *"Log every assumption you do make in docs/ASSUMPTIONS.md."*

| ID | Date | Module | Assumption Details | Rationale | Decision / Approved By |
|---|---|---|---|---|---|
| A-001 | 2026-09-29 | Setup | Initial project repository scaffolding created | Scaffold according to Part A requirements | User Prompt Set |
| A-002 | 2026-09-29 | Market Choice | USDT Perpetual Futures chosen as primary research venue | 227 symbols have >= 12m 15m data; volume in USDT is natively provided; lower taker fees (0.02% vs 0.05%+) | Decision Rule in Phase 0 Prompt |
| A-003 | 2026-09-29 | Data | 15m Kline `amount` field represents quote turnover in USDT | MEXC contract API returns turnover in USDT under `amount`, consistent with ticker `amount24` | Verified via empirical API probe |
| A-004 | 2026-09-29 | Date Range | Backtest date range set to 360 calendar days (Oct 2025 – Sep 2026) | MEXC REST API restricts 15m candle history to ~360 days | Empirical binary search findings |
| A-005 | 2026-09-29 | Holdout | First 75% (270 days) is In-Sample; final 25% (90 days) is Locked Holdout | Mandatory Holdout Discipline (Rule 5) | Proposed for Phase 0 approval |
| A-006 | 2026-09-29 | Execution | Base taker fee 0.02%, maker fee 0.00% | Current published MEXC futures fee schedule | MEXC official fee documentation |
| A-007 | 2026-09-29 | Ambiguity | Same-bar collisions of stop and target default to stop-first | Conservative evaluation policy (DEFINITIONS.md §11) | Defined in project specifications |
| A-008 | 2026-10-09 | Strategy / ICT | Scaled 5m/1m to 1h/15m on existing dataset; prior day 70% range filter operationalized as middle equilibrium zone (15%-85%); roundtrip costs set to 0.16% (0.06% taker + 2 bps slippage per leg). | Enables zero-lookahead testing across 100 liquid coins; aligns with realistic prop firm fee friction. | User choice Option 1 |
| A-009 | 2026-10-09 | Strategy / SMC | Order Block sweep tested on 1h OB / 15m entry; sweep low defined as penetration of OB boundary by >= 0.05% with immediate close back inside; stop at 1x ATR(14) below sweep low; target at equal highs (BOS swing level) or 2.5R. | Maps SMC liquidity grab mechanics to available multi-asset dataset with strict zero-lookahead. | User prompt |
| A-010 | 2026-10-14 | L/S Universe | Excluded 43 TradFi synthetics (stocks, commodities, indices). Universe set to 117 genuine crypto altcoins. | TradFi synthetics close on weekends/nights, causing artificial 0-return gaps and spurious cross-sectional momentum/reversal signals. | User decision |
| A-011 | 2026-10-14 | Frequency | 15-Minute panel selected as primary resolution for baseline and reversal study (covering 15m, 1h, 4h, 1d lookbacks and 15m, 1h, 4h holding periods). | Enables high-frequency intraday momentum & reversal microstructure testing on 34,576 bars per coin. | User decision |
| A-012 | 2026-10-14 | Holdout Split | 60% Train/Explore, 20% Validation/Selection, 20% Locked Holdout (untouched until Step 5). | Complies with Rule 5 holdout discipline and Step 5 robustness requirements. | Project Specification |
| A-013 | 2026-10-17 | Downside Resilience | BTC stress defined as cumulative drop >= 3% within <= 12 hours (24h cooldown). Downside Capture Ratio (DCR) defined as Alt Return / BTC Return over identical window. Stabilization confirmed on first 1H BTC close >= stress end price; entry at open of bar following confirmation. Control group uses 24h pre-stress momentum leaders. | Strict zero-lookahead testing of the 'Coin that refuses to fall' hypothesis against ordinary relative strength leaders on 116 genuine crypto altcoins. | User prompt |
