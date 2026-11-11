# AlphaGPT A-share adapter (stage 1)

This is **read-only research infrastructure**, NOT a trading bot. The existing
model_core/, Solana execution modules, and six database tables are unchanged.
It is grounded in the public_1111.sql schema dated 2026-11-11.

## Tables used

- market: daily OHLCV; uppercase quoted "MA_20" identifier.
- stock_main_force_daily: daily main-force scores, joined by stock/date.
- billing_bak: historical minute-level **cumulative** snapshots.
- billing: current cumulative snapshots (only when --include-today).
- stock_info: reserved for later survivorship-safe universe construction.
- stock_signal_event: deliberately **not** a historical feature: status and
  other values can be revised after signal creation; needs point-in-time audit.

## Install, configure, test

Run from repository root, Python 3.10+:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_ashare.py'
```

Configure a **SELECT-only** PostgreSQL user; never commit passwords. Either
set ASHARE_DB_URL (SQLAlchemy URL), or set DB_USER, DB_PASSWORD,
DB_HOST, DB_PORT, DB_NAME. Then run:

```bash
python -m ashare.inspect daily 2026-08-01 2026-08-31 --codes 000001,600000
python -m ashare.inspect minute 2026-11-09 2026-11-10 --codes 000001
# Today's live snapshots (optional):
python -m ashare.inspect minute 2026-11-11 2026-11-11 --codes 000001 --include-today
```

No DB data is created or modified. AShareDailyLoader.load uses a 150-calendar-
day warmup for 20-bar lookbacks, also fetches 30 future calendar days
for labels (never as features), and returns only requested dates. If fewer
than 20 valid historical trading rows exist, some factors remain NaN. It joins
main-force rows even if their historic availability is uncertain: **DO NOT**
claim point-in-time backtest validity until score-generation timestamps and
backfill lineage are audited. Current stock_info.status/st must not be used
to retrospectively filter historical universes.

make_daily_features returns 12 candidates (DAILY_COLUMNS) plus a separate
target_next_open_1d label. This is *T+1 open to T+2 open* return, only a
research label; trading fills, adjustment, suspension, auction limit, board
rules, fees, delistings, and market impact are NOT yet modeled. Ensure price
adjustment conventions are consistent before computing returns.

build_minute_intervals uses ticktime as a **last-trade clock** rather than
a guaranteed collection timestamp. It retains the final record per minute,
uses the current billing row when sources tie, and only produces interval
volume/amount/returns when adjacent **observed minute buckets** are one minute
apart and cumulative totals do not decrease. It does NOT reconstruct genuine
one-minute OHLCV bars or fill missed minutes. Noon is a gap; the first 13:00
interval is intentionally unknown. Unit consistency between tables should
also be independently checked before computing ratios.

## Next milestone

Audit actual DB samples, adjust price/corporate-action conventions, establish
point-in-time universe + snapshot timestamps, build factor tensor with a
separate missing-value mask, then implement a causality-safe formula VM and
A-share transaction-cost-aware walk-forward backtest. **Do not run existing
strategy_manager.runner for A-share trading.**
