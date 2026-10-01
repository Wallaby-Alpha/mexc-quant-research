# PROJECT: MEXC Swing-High Retest Research

## Standing Rules

PROJECT: Research-grade backtester testing one hypothesis on MEXC crypto markets.
This is research, NOT a trading bot. Never add live trading, order placement, or API keys with trade permission.

PURPOSE: Answer honestly whether there is a measurable edge in buying a controlled 15m pullback
toward a prior swing high inside an established 1H uptrend. A "no edge" or "edge too small after fees"
result is a valid, valuable outcome. Do not tune anything to make results look good.

### NON-NEGOTIABLE RULES
1. **No lookahead.** Every signal must use only data that was fully closed at decision time.
   Follow `docs/DEFINITIONS.md` section 1 (time conventions) exactly.
2. **No survivorship shortcuts.** Never apply today's top-N list to history. Universe is point-in-time.
   Where bias cannot be removed, document it in `docs/LIMITATIONS.md` instead of hiding it.
3. **Net-of-cost results are the primary result.** Gross P&L is shown only as a secondary column.
4. **Never silently choose.** If a definition is ambiguous, data is missing/inconsistent, or the API
   behaves unexpectedly: STOP, describe the issue, propose 2-3 options, and wait for my decision.
   Log every assumption you do make in `docs/ASSUMPTIONS.md`.
5. **Holdout discipline.** The final 25% of the available date range is a locked holdout. No exploration,
   plotting, parameter selection or summary stats on it until Phase 5 walk-forward/final evaluation.
   Enforce this in code (a guard that raises if holdout data is loaded outside the final-eval command).
6. **Multiple-testing honesty.** Every parameter combination evaluated is counted in a trial log
   (`results/trials.parquet`). Reports state the total number of trials.
7. **Statistical significance.** Every reported probability includes $n$ and a 95% confidence interval. Buckets with $n < 100$
   independent setups are flagged "insufficient sample" and excluded from conclusions.
8. **Robustness.** Never report only the best parameter set. Report the surface/heatmap and the robust region.
9. **Reproducibility:** deterministic runs, pinned dependencies, all parameters in YAML config,
   config hash saved with every output.
10. **Scope discipline:** Do not build anything listed under "Future extensions" until asked. Keep interfaces ready for them.

### TECH STACK
Python 3.11+, pandas + numpy, pyarrow/Parquet, pydantic (config), httpx/requests, pytest, matplotlib/plotly, Streamlit (or static Plotly HTML) for the setup viewer.

### CODE QUALITY
Typed, modular, small functions, docstrings stating the information-availability time of every signal. Every module has unit tests. Lookahead tests are mandatory.

### LAYOUT
- `/data`: mexc_client, downloader, cache, validator
- `/strategy`: trend_detector, swing_detector, pullback_detector, setup_lifecycle, entry_rules, exit_rules
- `/backtest`: engine, execution_model, position_manager
- `/analysis`: event_study, performance, controls, parameter_sweep, regime_analysis, walk_forward
- `/reports`: trade_export, summary, charts, setup_viewer
- `/config`: default.yaml
- `/docs`: DEFINITIONS.md, ASSUMPTIONS.md, LIMITATIONS.md, FEASIBILITY.md
- `/tests`: unit, integration, lookahead tests
