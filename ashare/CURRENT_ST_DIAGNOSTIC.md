# P3.3-B — Current-ST blacklist diagnostic (NON-PIT ONLY)

This mode is **not a historical point-in-time filter**. It reads the
present-day `public.stock_info` table at run time and applies today's
risk-warning status to ALL T-date signals from 2025 onward.
It deliberately introduces future-information/selection bias. Treat it
only as a controlled sensitivity test for the stale-position problem.
Do not use its return, turnover, drawdown or holdings as audited PnL,
as proof of strategy quality, or as Transformer training rewards.

## Rules

1. `stock_info.st` is 1 (ST) or 2 (*ST) => BLOCK new entries.
2. `stock_info.stock_name` contains "ST" (case-insensitive)
   => BLOCK even when current st is 0.
3. A missing stock_info row, unknown st code, blank or missing name
   with st=0 => UNKNOWN => BLOCK new entries (fail closed).
4. Only a recognized st=0 AND a nonblank name without "ST" can enter.
5. The filter applies to candidate signals BEFORE ranking the top N.
   It does NOT liquidate positions or change the price data.
6. This mode cannot be combined with --st-daily / --st-coverage.

The use of "%ST%" is deliberately conservative and can flag company
names containing that substring for other reasons; the audit CSV
exposes those decisions for review.

## Run

    cd ~/gcdev/alphagpt/AlphaGPT
    git switch ashare-dev
    git pull --ff-only origin ashare-dev
    python -m unittest discover -s tests -p 'test_*.py' -v

    python -u -m ashare.backtest_run \
      --start 2025-03-01 --end 2026-09-30 \
      --exploratory --exclude-current-st \
      --candidate-pool 20 --log-every-days 20 \
      --out reports/ashare_backtest_current_st

The explicit --out keeps previous P3.1 and P3.2 results unchanged.
If --out is omitted, --exclude-current-st defaults to the separate
reports/ashare_backtest_current_st directory.

Outputs in reports/ashare_backtest_current_st/2025-03-01_2026-09-30:

- summary.json, daily_equity.csv, simulated_trades.csv
- rejected_attempts.csv, open_positions.csv
- current_st_universe_audit.csv

The audit contains current-stock status, reason and number of blocked
historical signal rows per stock. summary.json records:

- stock_pool_mode: NON_PIT_CURRENT_ST_FILTER
- is_point_in_time_stock_pool: false
- current_st_filter: { status, source, snapshot_at_utc,
  is_point_in_time: false, lookahead_risk: true, before/after counts,
  blocked current-ST and UNKNOWN counts, warnings }
- audited_total_return_available: false

## Compare after uploading to Google Drive

Upload all six files to a new folder named
agu/alphagpt/ashare_backtest_current_st/2025-03-01_2026-09-30.

Compare simulated trades and open_positions to previous P3.2.
Especially audit 000638, 002231, 002898, 300344, 300391, 600200,
600355, 603388 and 920680. The objective is to test whether those
symbols drive the unchanged/stale NAV pathology. A cleaner outcome
under this CURRENT-state blacklist is NOT an estimate of what a
historically executable ST-excluded strategy would have achieved.

## Next for historically valid strategy research

Acquire date-stamped ST/risk-warning status, trading suspension and
delisting events with effective date and announcement date, limits at
opening, and corporate-action handling for unadjusted OHLC.
ashare/st_history.py is the time-safe interface when verified dated
snapshots become available. Current-ST exclusion must NEVER replace it.
