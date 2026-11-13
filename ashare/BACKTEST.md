# P3.1 A-share exploratory transaction simulation

First implementation for NEG(ma20_dev) baseline. This is **NOT a validated
A-share historical PnL backtest and NEVER places live orders**.

Run after the existing quality checks from the AlphaGPT repo root:

    git switch ashare-dev
    git pull --ff-only origin ashare-dev
    python -m unittest discover -s tests -p 'test_*.py' -v
    python -u -m ashare.backtest_run --start 2025-03-01 --end 2026-09-30 --exploratory

Provide the same environment variables as ashare.research. To restrict
sample stocks append --codes 000001,600000.

Output directory (excluded from Git by .gitignore):

    reports/ashare_backtest/2025-03-01_2026-09-30/
      daily_equity.csv
      simulated_trades.csv
      rejected_attempts.csv
      summary.json

## Assumptions and guardrails

- The query selects **only** market.not_billing=1 with unadjusted OHLC.
  Intraday not_billing=2 snapshots are *not used*.
- Start reporting and trading in March 2025. Earlier dates may be loaded
  strictly for indicator warmup. No trades occur before the start.
- A T-close signal (NEG(ma20_dev) = 1-close/MA20) can buy at T+1 open,
  sell no earlier than T+2 open. Positions unable to sell remain held
  until a later observed bar. Positions outstanding at end are marked,
  not artificially liquidated.
- Buys in 100-share lots. 1,000,000 simulated starting cash, maximum
  10 simultaneous holdings, 0.03% commission (minimum 5), sell-side
  stamp 0.05%, transfer 0.001%, 5 bps price slippage per side:
  **research assumptions**, not confirmed historical broker fees.
- Opening fills based on daily open are a **simplification**:
  an observed daily opening price does not establish actual auction
  order execution. Intraday supply, queue priority and depth are
  unknown, so no claiming fills as fact.
- The optional run_backtest(tradeability=DataFrame) requires verified
  date/stock can_buy and can_sell boolean flags. Missing rows fail
  closed. The CLI currently has no history feed for these rules;
  hence **--exploratory must be explicitly supplied** or CLI refuses.
- --exploratory bypasses historical ST/price-limit availability checks.
  It does NOT mean all opening prices were actually tradable, and
  results should not be used as production strategy evidence.
- Corporate actions (cash dividends, bonus/split shares, rights issues)
  are NOT modelled. With unadjusted OHLC, daily-equity returns across
  company actions may be badly biased. Historical adjusted reference
  prices and corporate-action events are still required.
- Signal generation uses existing stored "MA_20" after close. It must
  be audited for causal reconstruction and date availability.
- No liquidity/volume hindsight screening is used at the opening,
  because daily total volume/high/low are unknowable at open. The
  simulation does not yet model actual minute auction liquidity.
- No current stock_info.st/status used for past dates; no trading
  commissions, slippage and price regimes are claimed as historical
  fact. Position valuations may be stale on a missing market day.
- Training/testing split, benchmark, corporate action handling, actual
  historical stock status and date-scoped price-limit fills remain
  P3.2/P3.3 work. Do not train Transformer rewards from P3.1 outputs.

## Data needed next

For audited P3, supply history of: ex-dividend/corporate actions with
cash/share treatment, historical ST status and price-limit levels, true
suspension/nontradable dates, and if possible opening auction tradability
/ minute bars. The stock_scripts daily OHLC source does not provide
reliable opening auction fill probabilities.

## P3.2 audit hardening — 2026-11-13

The uploaded first run (Google Drive `agu/alphagpt/2025-03-01_2026-09-30`)
showed 4,819 fill rows, 326,561 rejected attempts, 9 unclosed holdings and
54 consecutive final trading days where ALL holdings had stale marks.
Of the 326,561 attempts, 325,060 were INSUFFICIENT_CASH. The indicative
-44.26% return was NOT a validated PnL measurement.

This version:
- fixes the full-universe scan on cash exhaustion: T-date's ranked candidate
  shortlist is fixed to `--candidate-pool 20` (tunable), and the first
  unaffordable buy stops further buy attempts that morning;
- reports stale position age in observed trading sessions, total days with
  stale marks, and whether period-end values are fresh;
- adds `open_positions.csv` with each held symbol, entry, last marked
  price/date and estimated stale session count;
- adds `candidate_pool_size`, `last_simulated_trade_date`,
  `fees_assumed_total`, stale-valuation metrics and mandatory
  `audited_total_return_available: false` to summary.json;
- does not claim that proxy fills are actual tradable fills; no changes
  to PostgreSQL or Solana execution logic.

Run:

    python -u -m ashare.backtest_run --start 2025-03-01 \
      --end 2026-09-30 --exploratory --candidate-pool 20 --log-every-days 20

The old and new indicative outcomes will differ because the historical
selection/execution policy has changed; comparisons of PnL between versions
are not meaningful strategy-performance improvements.

Remaining P3.3 gates: corporate-action history, historical ST and daily
limit prices, explicit suspension eligibility, and price-at-open queue/depth.
Do not train GPU reward models from unverified daily-open proxy fills.
