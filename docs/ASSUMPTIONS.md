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
