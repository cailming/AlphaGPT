# P3.3-A — Historical ST snapshot ingestion

The only schema match in the uploaded historical_st_tables.csv is
stock_info (stock_name, status, st, change_date, change_reason).
Current stock_info.st is NOT a historical status series. Do not backfill
2025 signals from today's stock_info state.

## Two input files are required

st_daily.csv: one row per risk-warning stock and trading date, columns:
  stock_code,trade_date
  (ts_code,trade_date is also accepted; .SH/.SZ/.BJ suffixes are stripped)
Optional extra columns: stock_name, risk_type.

st_coverage.csv: exactly one completeness record per trading day:
  trade_date,complete,st_count,source
Optional: fetched_at. Illustrative format only (NOT real dates/counts):
  2025-03-03,true,180,provider-name
  2025-03-04,false,0,provider-name

An absent stock on a CERTIFIED COMPLETE day means not present in that
day's ST list. An absent/incomplete DAY means UNKNOWN: every signal on
that day is BLOCKED, not assumed to be non-ST. The reported st_count
must equal the number of rows on each complete day. Dates must be in
YYYY-MM-DD or YYYYMMDD format. Duplicate dates/stock-days are rejected.

This only screens T-close signals using T-date historical ST risk lists.
It does NOT yet guarantee eligibility at the T+1 opening auction. Even
vendor historical data may have retrospective revisions. Preserve
provenance and verify suspicious histories against exchange notices.

## Optional export from Tushare Pro

Official documentation:
  https://tushare.pro/document/2?doc_id=397 (stock_st, historical daily ST)
  https://tushare.pro/document/2?doc_id=423 (st, warning events)

Tushare's stock_st historical list currently has a 3000-point access
threshold. Do not install packages or purchase permissions automatically.
In your AlphaGPT py312 environment, check whether tushare is installed
and that your account has access. Set the token ONLY in the environment
variable TUSHARE_TOKEN. Never upload the credential to Drive or GitHub.

Run (only if you already have permission):

  python -u -m ashare.st_export_tushare \
    --start 2025-03-01 --end 2026-09-30 \
    --out reports/ashare_st_history

The script SELECTs trading dates from market.not_billing=1, calls the
Tushare daily historical ST API, outputs st_daily.csv and st_coverage.csv
and resumes dates already marked complete. API failures or unexpected
response layouts become incomplete days; they are never converted into
'no ST' observations. It writes no data to PostgreSQL.

If Tushare is unavailable, you can use audited exchange disclosures or
another historical provider to create the SAME two CSV schemas.

## Optional P3.3-A daily historical ST filter in backtest

  python -u -m ashare.backtest_run \
    --start 2025-03-01 --end 2026-09-30 \
    --exploratory --candidate-pool 20 \
    --st-daily reports/ashare_st_history/st_daily.csv \
    --st-coverage reports/ashare_st_history/st_coverage.csv \
    --out reports/ashare_backtest_p33

By default the CLI refuses a run if any signal date lacks complete ST
coverage. The --allow-partial-st-coverage flag permits an incomplete
DIAGNOSTIC ONLY; unknown dates still block all signals. The output adds
st_filter_by_day.csv and a historical_st_filter section in summary.json.

WITHOUT the --st-daily and --st-coverage pair, the CLI leaves the
historical ST filter OFF and says so explicitly. That is not a valid
historical ST-excluded strategy.

## Limits

This P3.3-A work is not a validated A-share PnL system. Historical ST
daily screening does not establish:
- T+1 opening-day warning announcements and full trade eligibility
- suspension, liquidation and delisting cash flows or last trading prices
- historical price limits, opening auction execution and order queues
- dividends, splits and other corporate actions for unadjusted OHLC.

Do not use such exploratory performance as Transformer training rewards.
