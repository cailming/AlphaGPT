# A-share factor quality and main-force history audit — P1/P2

This is **read-only** analysis against the existing six-table PostgreSQL schema
in agu/public_1111.sql. No DB table or stored procedure is changed.

## Ubuntu commands

Run in the repository root with the same DB_* or ASHARE_DB_URL environment
variables used for existing research:

    git switch ashare-dev
    git pull --ff-only origin ashare-dev
    python -m unittest discover -s tests -p 'test_*.py' -v
    python -u -m ashare.quality_report --start 2025-01-01 --end 2026-09-30 --db-chunk-size 50000

Optionally append --codes 000001,600000 for a small test or
--out /path/to/report-root for a custom LOCAL output directory.

This reports progress as PostgreSQL returns rows. It reads earlier warmup
history for rolling factors but only reports the requested date range.

## Four output files

- factor_summary.csv: all 12 factors, raw non-null counts, numeric-valid
  and timestamp-valid coverage, out-of-range counts, count of stocks and
  dates with eligible factor values.
- factor_monthly.csv: monthly factor coverage on observed market rows.
- main_force_created_batches.csv: created_at calendar-day batches,
  earliest/latest source trade_date, number and proportion of late records.
- main_force_audit.json: market/grid coverage, join rate, missing and
  late created_at timestamps and other provenance screening summaries.

## Important denominator distinction

market_rows = actual daily market records in the requested interval.
grid_rows = number of distinct stocks * number of distinct trading dates
in that SAME requested interval, including missing stock/date pairs.
The prior 44.6% full tensor coverage included warmup history, so it is
not directly comparable unless periods and stock universes agree.

tensor_mask_pct_market / tensor_mask_pct_grid match the current
build_daily_tensor numerical-finite and created_at cutoff rules.
quality_eligible_pct_* further enforce documented factor value ranges.
The bounds checks are DIAGNOSTIC; the original model tensor does not yet
enforce these checks, so a range anomaly should be investigated before
training rather than automatically corrected or zero-filled.

## Timestamp interpretation

For a T-day after-close signal, the current model treats score timestamps
earlier than T+1 midnight as eligible. The main-force table's created_at
describes creation OR last generation time, not an immutable history of
when each score was first available. A later created_at could mean a
historical backfill; an earlier timestamp alone does not prove
point-in-time correctness of the underlying input data.

This report flags missing/late timestamps and clusters records by
created date. Do not use a diagnostic bypass to claim a valid historical
backtest. To recover historical eligibility, reproduce score calculations
with each T-date cutoff, record code/data versions and date of generation,
and validate against data available at T.

No authenticated connection from this chat to the user's own PostgreSQL
instance is available; the actual percentages must come from running this
command on the Ubuntu host.
