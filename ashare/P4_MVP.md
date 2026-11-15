# P4 MVP — Dynamic usable-stock pool + Ridge ranking baseline

**Goal: usable, trainable, repeatable on a subset — not whole-market coverage.**

This release uses read-only public.market OHLCV and a NumPy Ridge baseline.
It validates end-to-end TRAIN -> OUT-OF-SAMPLE EVALUATE -> INFER before
spending RTX 3090 GPU time on Transformer.

## Engineering contract

- We do NOT wait for a full historical ST dataset. Current stock_info.st
  is NEVER backfilled into past training dates.
- Only market.not_billing=1 (formal daily OHLCV). The source uses
  UNADJUSTED prices: corporate actions may distort prices/targets.
- The risk-event CSV is PARTIAL. Public ST/halt/delisted events already
  disclosed by T and effective at the intended next-session open veto
  entries. Absence of a known event is NOT proof of normal status.
- Incomplete rows, gaps and low-price/liquidity stocks are skipped.
  No interpolation, no invented bars, no full-universe requirement.
- Every stock/date is screened using T or older observations only.
  Labels use future opens only and require consecutive real market bars
  from T+1 through T+(H+1), so suspension gaps yield no label.
- Daily top-N is ranked by trailing 20-session median amount. Default
  pool cap 800, price >= 3, >=40 consecutive valid sessions. An
  optional minimum amount is in the DB's native units; default 0
  avoids guessing yuan vs ten-thousand yuan conversion.
- Target = T+1 open to T+1+H open, H=5 trading sessions by default.
- Excludes historically unverified main-force values in this MVP.
  Features: ret_1, ret_5, ret_10, close_ma20_dev, volume_ratio_20,
  amount_ratio_20, range_intraday, return_intraday, volatility_20,
  log_liquidity_20.
- Time split by T date: 60% train, 20% validation, 20% untouched test.
  Purge samples whose forward-label end crosses the next split.
  Select Ridge alpha by validation rank-IC only; refit on past
  train+validation labels; evaluate held-out test.
- Save scaler, coefficients and feature order. Prediction loads no
  post-asof OHLCV and rejects asof dates preceding the final
  training-label date.
- Ranking statistics are NOT audited profits. Forward 5-session
  returns overlap, and simple fee proxy is not an executable NAV.

## On Ubuntu (no new pip dependencies)

    cd ~/gcdev/alphagpt/AlphaGPT
    git switch ashare-dev
    git pull --ff-only origin ashare-dev

    python -m unittest discover -s tests -p 'test_*.py' -v

    python -u -m ashare.p4_train \
      --start 2025-03-01 --end 2026-09-30 \
      --top-n 800 --min-price 3 \
      --consecutive 40 --hold-sessions 5 \
      --top-k 20 --roundtrip-cost-bps 25 \
      --out reports/ashare_p4_mvp

No DB changes. Existing env vars ASHARE_DB_URL (or DB_USER, DB_PASSWORD,
DB_HOST, DB_PORT and DB_NAME) supply the DB credentials.

Optional --risk-events path overrides the default audited sample at
ashare/data/risk_events_verified_examples.csv. Use --no-risk-events
to disable even the small known-event overlay.

The pool can be much smaller than 800. If insufficient samples for
chronological splits, widen dates or adjust selection thresholds;
do not tune the model based on the held-out test.

## Files

reports/ashare_p4_mvp/2025-03-01_2026-09-30/:

- ridge_model.npz: model, scaler and ordered features.
- model_metadata.json: chosen alpha, split boundaries, train-last-label
  date, validation diagnostics and untouched test rank metrics.
- test_predictions.csv: T date, code, predicted and actual H-day
  open-to-open return; NOT actual trade PnL.
- latest_predictions.csv: last-date 50 research rank scores.
- pool_daily_coverage.csv: daily eligible count and valid future labels.

For model inference with the same feature/universe logic:

    python -u -m ashare.p4_predict \
      --model-dir reports/ashare_p4_mvp/2025-03-01_2026-09-30 \
      --asof 2026-09-30 --top-k 50 \
      --min-predicted-return 0 \
      --out reports/ashare_p4_predict

The optional abstention floor drops lower-score candidates. A blank
output is valid ("no forecast today"). A predicted positive return is
not a calibrated success probability.

## Validation checklist

1. The full local unit test suite passes.
2. model_metadata.json: train_last_label_date is strictly earlier
   than test_start.
3. Re-run inference on asof=last test date and compare saved scores
   with latest_predictions.csv for an unchanged data snapshot.
4. Inspect sample coverage, rank IC, selected-top-20 positive fraction,
   gross return relative to pool average, and naive fee proxy.
5. 50% directional hit rate alone may be no better than chance. The
   portfolio should improve above fees with lower turnover before
   any real-money claims.
6. No future market data is read in the inference command; do not
   promote exploratory ranking diagnostics to a backtest NAV.

Next: train small Transformer/MLP on the SAME contracts when this Ridge
baseline and inference are verified. Do not require full-stock coverage.


## P4.3-B: Post-hoc Ridge test diagnostics (2026-11-14)

The 2026-06-12..2026-09-30 holdout has **already been inspected**. It is no
longer untouched for designing or selecting a NEW model; reserve later dates
for genuinely out-of-sample confirmation. These metrics are neither a
realistic backtest, PnL, trading fills nor model-calibrated probabilities.

Run this independently of PostgreSQL:

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev && git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_diagnostics.py' -v
python -u -m ashare.p4_diagnostics \
  --model-dir reports/ashare_p4_mvp/2025-03-01_2026-09-30 \
  --top-k 20 --out reports/ashare_p4_diagnostics
```

Inputs are saved `test_predictions.csv` and `model_metadata.json` only;
date boundary, row count, stock/date uniqueness and finite values are checked.
Outputs: `diagnostics.json`, `daily_diagnostics.csv`,
`monthly_diagnostics.csv` and `score_deciles.csv`
(decile 0 denotes highest predicted scores).
Day-over-day Top-K name replacement is NOT executable portfolio turnover;
five-session labels overlap, and all prices are unadjusted.

**Observed from uploaded model outputs (61,521 rows, 77 dates):**

| Month | Ridge Rank IC | Top-20 minus pool, mean forward label |
|---|---:|---:|
| 2026-06 | -0.02261 | -0.00127 |
| 2026-07 | -0.01650 | -0.00762 |
| 2026-08 | -0.01220 | -0.00130 |
| 2026-09 | -0.03519 | -0.00846 |

Across all rows, 94.66% of Ridge predicted returns were positive while
only 40.13% of realized five-session labels were positive.
Mean predicted was +0.3526% vs actual label -1.6861%.
The daily Top-20 list replaced 88.62% of prior-day names on average;
this is an indication of score/list instability, not verified trading costs.
Predicted-score decile 0's positive-label share was 36.79%, vs 43.81%
for decile 7. Ridge mean Rank IC -0.0214, simple `NEG(close_ma20_dev)`
factor mean Rank IC +0.0820 (previously computed on the same test window).

The assistant ran the new diagnostic CLI on those saved rows and its four
new synthetic unit tests locally. The user Ubuntu full suite and live DB
were NOT run by the assistant during this update.

**Next:** audit corporate actions/target outliers and training-validation
factor behavior, then compare conservative light baselines using ONLY
training/validation folds. A future-date untouched evaluation is required
before promoting an improved model or proceeding to RTX 3090 Transformer.


## P4.3-C: Train/validation factor and label audit (2026-11-15)

This stage inspects **only labels from the original, chronologically purged
training and validation ranges**, without fitting a new model, writing to the
production database or looking at any 2026-06-12..09-30 test label.
This analysis is intended to diagnose Ridge's negative performance but
must NOT use the now-inspected test data to select factors/hyperparameters.

**Why a fresh SELECT is required:** the earlier P4.1 exported
`pool_selected.csv` contains only dates, stocks, liquidity, close and
target_return. It does **not** contain the ten model features or
`label_end_session`. It is insufficient for rigorous IC/embargo analysis.
The new CLI calls the existing `load_formal_ohlcv` read-only loader and
`build_pool`, reconstructing the same causal market sample and label metadata.

**Ubuntu commands (repository root, existing read-only market credentials):**

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev

python -m unittest discover -s tests -p 'test_p4_factor_audit.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

python -u -m ashare.p4_factor_audit \
  --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --out reports/ashare_p4_factor_audit
```

The optional `--reference-metadata` is strongly recommended. It
**fails closed** if the rebuilt P4 pool's validation start/test start,
refit sample count or pool configuration does not match the earlier Ridge
training metadata. By default the prior four verified public risk event
examples are read at `ashare/data/risk_events_verified_examples.csv`;
no present-day ST backfill.

Output directory is
`reports/ashare_p4_factor_audit/2025-03-01_2026-09-30/`,
containing:

- `factor_audit.json`: fixed date cuts, train/validation sample counts,
  purged/missing labels, reference metadata check and explicit holdout exclusion.
- `factor_ic_daily.csv`: each of ten features' daily cross-sectional
  Spearman IC computed separately for training and validation dates.
- `factor_ic_summary.csv`: by-feature train and validation IC mean,
  median, std, valid-day counts and daily positive share. Factor signs are
  determined **only from the train mean IC**, then carried to validation
  unchanged (column `signed_mean_ic_train_direction`).
- `label_quality.csv`: label quantiles, positive share and extreme
  tail counts by train/validation split. All counts exclude the old test set.
- `label_outliers.csv`: up to 200 largest absolute forward-label
  changes **per split** over 20% by default, with signal date, stock code,
  price factor context and label end-session index. These require independent
  corporate-action/price verification; a large return is NOT proof of any
  company event.

**No metrics from the new real database run are claimed yet.**
The new unit test file covers chronological purging, preventing even
corrupt test labels from entering the report, train-only factor sign
selection, label-tail accounting and input guards. Execute the above
on Ubuntu, review the outputs and upload the five files to Drive under
`agu/alphagpt/ashare_p4_factor_audit/` for cross-checking.

**Next gate**: after comparing train vs validation feature ICs, inspect
candidate unadjusted-price extremes against date-stamped corporate action
evidence. Then implement a conservative baseline trained/tuned within
train/validation only. Final performance still requires a genuinely
untouched later period or predeclared walk-forward folds.
