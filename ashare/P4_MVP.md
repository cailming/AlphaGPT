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


## P4.3-D: Robust lightweight baselines and extreme-label price tracing

Code: `ashare/p4_light_baselines.py`, `ashare/p4_price_trace.py`;
synthetic tests: `tests/test_p4_light_baselines.py`,
`tests/test_p4_price_trace.py`.

These are **RESEARCH DIAGNOSTICS**, not portfolio backtests or approved
live models. The old June-September 2026 holdout has already been inspected.
The earlier February-June validation was also examined. Do NOT use the
old validation to select a winner and then call its score a newly
independent out-of-sample result. Obtain a genuinely later untouched date
range, or predeclare walk-forward folds, before promotion.

### D1: Explain possible price anomalies before assuming corporate actions

The previously exported `label_outliers.csv` is a capped
list (top 200 from train and validation each), not a full action registry.
Run read-only price tracing for the 30 most extreme rows in each split:

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev && git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_price_trace.py' -v

python -u -m ashare.p4_price_trace \
  --outliers reports/ashare_p4_factor_audit/2025-03-01_2026-09-30/label_outliers.csv \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --limit-per-split 30 --jump-threshold 0.15 \
  --out reports/ashare_p4_price_trace
```

The module issues exactly two read-only queries against formal
`public.market WHERE not_billing=1`: the market-date calendar and
OHLC for only the requested stocks. Both queries end before the old test
start. It reconstructs T+1...T+6 **on the actual market calendar**, not
by jumping over a halted or missing stock row. It compares unadjusted
T+1/T+6 opens against the original label, checks intermediate OHLC
presence/validity and flags open-price jumps above the stated 15%
review threshold. Price changes do NOT prove corporate actions.

Outputs: `price_trace_summary.json`,
`price_trace_cases.csv`, `price_trace_bars.csv`.
Missing bars, source-price mismatches and jumps must be reported, **not
filled, silently dropped, adjusted without evidence or treated as profit**.

### D2: Seven apples-to-apples lightweight research candidates

Use the same original dynamic Top-800, 40-session history, T+1->T+6
labels, date fractions and purged training/validation split. Validate
those dates, selected-row count, validation-label sample count and refit
sample count against the old `P4_RIDGE_V1 model_metadata.json`.
**The CLI also **recomputes the fixed-alpha Ridge's validation mean Rank IC**
and fails closed if it does not match the archived reference metric within
1e-6, when that alpha appears in saved tuning metadata. This improves
reproducibility but still does NOT prove a byte-identical market snapshot**.

```bash
python -m unittest discover -s tests -p 'test_p4_light_baselines.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

python -u -m ashare.p4_light_baselines \
  --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --alpha 1.0 --top-k 20 \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --out reports/ashare_p4_light_baselines
```

The seven candidates (all compared over exactly the same validation
rows) are:

1. `neg_ma20_fixed`: predeclared negative MA20 deviation, no fitting.
2. `train_best_single_factor`: factor and sign chosen strictly from
   **training-period** mean daily rank IC, not validation or test.
3. `train_signed_rank_ensemble`: equal-weight same-day factor ranks
   with signs chosen on training dates only (score is NOT expected return).
4. `ridge_original`: alpha fixed at 1.0, trained on train only.
5. `ridge_clip_abs_train_q95`: cap train labels at the 95th
   percentile of train-only absolute label magnitudes.
6. `ridge_clip_abs_train_q99`: analogous train-only 99th percentile.
7. `ridge_daily_rank_target`: Ridge trained on within-train-date
   target percent-ranks, not numeric expected returns.

No candidate is automatically selected or deployed. Training-only signs,
caps and fitted model weights do not incorporate future validation/test
targets. The model diagnostics are date-wise cross-sectional IC, Top-20
positive label share, mean five-session open-to-open forward **label**
and excess vs the same day's pool. These are overlapping research labels,
**not fills, trading costs or realized returns**. Even the known validation
period is exploratory and cannot be reused to establish independent
significance or select a strategy for real deployment.

Outputs in `reports/ashare_p4_light_baselines/2025-03-01_2026-09-30/`:

- `baseline_comparison.json`: train-only factor directions,
  clipping limits, experiment protocol and explicit holdout-exclusion warning.
- `baseline_validation_summary.csv`: seven candidates' date-mean IC,
  positive forward labels, top-20 vs pool difference and equal sample counts.
- `baseline_validation_daily.csv`: per-date metrics for all candidates,
  for diagnosing sign/market-regime instability.

**Verification workflow:** run new unit tests and real SELECT commands on
Ubuntu; upload the six resulting files to Drive under
`agu/alphagpt/ashare_p4_price_trace/` and
`agu/alphagpt/ashare_p4_light_baselines/2025-03-01_2026-09-30/`.
Inspect discrepancies before designing a new model. Do NOT automatically
enable Transformer training based on these retrospective comparisons.


## P4.3-D: Real-data audit acceptance and corporate-action blocker (2026-11-15)

The six user-uploaded artifacts were inspected independently and reconciled
with the prior P4.3-C `label_outliers.csv` and archived Ridge metadata:

- Google Drive [price trace](https://drive.google.com/drive/folders/17x-joE0RFwItS-reCt9fgdwcRm7m7znz):
  **60** selected top absolute-label cases, **360** open-price bars;
  all requested labels recompute exactly from the exported opens, no missing
  bars or mismatches. **58** cases cross the 15% consecutive-open jump alert.
  But the alerts represent **41 unique stock/jump-date pairs** in **38 stocks**,
  not 58 independent corporate action incidents.
- Google Drive [light baselines](https://drive.google.com/drive/folders/1bYJ0yWjK7txXK2IJRczQFGuG-aolA9aZ):
  **7 candidates x 71 dates = 497** daily diagnostic rows; each evaluates
  **56,686** purged validation samples. All candidate summary metrics
  independently reconcile with their daily file. The original fixed-alpha
  Ridge validation IC -0.011167283407209223 exactly matches archived metadata.

### Confirmed false economic-return label

The 002594 BYD case with signal 2025-07-25 enters (price proxy) at 2025-07-28
open 337.93 and exits at 2025-08-04 open 104.00, giving an unadjusted
research label -69.2244%. The 2025-07-29 open fell 66.8541% vs the prior
open. BYD's **2025-047 actual implementation announcement**, issued
2025-07-21, specifies **ex-dividend/rights date 2025-07-29,
10 old shares receive 8 stock dividend shares + 12 capital reserve shares
and 39.74 CNY gross cash dividend**:
https://epaper.stcn.com/pic/202507/22/37e227d2c542dd4204dafd9319859879.pdf

One pre-event share therefore corresponds to three post-event shares,
plus 3.974 gross cash dividend. A simplified, not fee/tax-adjusted holding
value example is `(3*104.00+3.974)/337.93-1 = -6.497%`, rather than
`104.00/337.93-1 = -69.224%`. It is NOT audited realized PnL;
it proves that an unadjusted price ratio **is not a correct economic
return target for this corporate-action-crossing window**. The same
ex-date appears in five adjacent BYD signals among the trace cases.
This invalidates at least those historical labels for return-based
supervision; it does not establish that all other large price changes
were corporate actions.

### Seven validation comparisons: no positive Top-20 excess

| Candidate | Daily mean Rank IC | Daily mean Top-20 minus pool label |
|---|---:|---:|
| neg_ma20_fixed | -0.007538 | -0.006864 |
| train_best_single_factor (negative volatility_20) | +0.012886 | -0.002155 |
| train_signed_rank_ensemble | -0.005561 | -0.007102 |
| ridge_original | -0.011167 | -0.003623 |
| ridge_clip_abs_train_q95 | -0.005809 | -0.005577 |
| ridge_clip_abs_train_q99 | -0.008328 | -0.003252 |
| ridge_daily_rank_target | +0.005287 | -0.007340 |

All values are **overlapping five-session forward-label diagnostics**,
not trade profits. Some weakly positive ICs do not overcome negative
Top-20 relative labels. Neither the old validation nor inspected 2026
test can now serve as independent new model selection evidence.

**P4.3-E blocking prerequisite**: assemble historically dated share
split/bonus/capitalization, cash dividend and trading-suspension event
records. Verify coverage and source publication/effective dates, compute
corporate-action-aware forward *economic* returns, and audit T-as-of
features around events without backfilling information from after T.
Do not simply discard every >20% move: genuine high moves may exist.
Retrain and report after these contracts are checked; reserve a genuinely
unseen later period. No automatic Transformer GPU promotion.

This review is an artifact reconciliation plus one independently
verified published corporate action. It did **not** run SQL against
production DB itself, verify all corporate actions, prove exact market
snapshot identity or verify that Ubuntu's entire regression suite passed.


## P4.3-E: verified corporate actions, economic label reconstruction and fail-closed coverage

**Implementation status (2026-11-15)**: `ashare/p4_corporate_actions.py` and `tests/test_p4_corporate_actions.py` are committed to `ashare-dev`. Seven focused synthetic unit tests passed in the development runtime; the Ubuntu full-suite and live PostgreSQL audit have **not** been run during this implementation. Do not confuse unit tests with complete historical data certification.

See **[P4.3-E data contract and Ubuntu commands](P4_3_E_CORPORATE_ACTIONS.md)**. Key invariants:

1. Event ledger `ashare/data/corporate_actions_verified_examples.csv` includes exactly one official source-backed example, BYD 002594 ex-date 2025-07-29, 3 post-event shares per old share plus 3.974 gross cash per old share; source is [SZSE publication 2025-047](https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-07-22/f30e6169-4779-487e-acff-8c1eff230fa4.PDF).
2. Coverage ledger `ashare/data/corporate_action_coverage_examples.csv` is **intentionally header-only**. No full-market or BYD interval completeness claim has been verified. An observed event is never proof that all other events are absent.
3. Model-training label release requires explicit verified full stock/date coverage over the **entire 40-market-session feature lookback and forward label window**, accurate source-price reconciliation, and absence of known ex-dates inside the legacy unadjusted feature window. Otherwise report `UNKNOWN_ACTION_COVERAGE` or `UNADJUSTED_FEATURE_ACTION_WINDOW`, and emit **NaN** for the model-ready economic target.
4. Known corporate actions can produce an **illustrative gross economic return** independent of training status. If the ex-date is exactly T+1 purchase day, the buyer has missed the registration date and receives no entitlement. Cash is not reinvested; fractional lots, taxes, split-specific settlement and brokerage costs are not modeled.
5. The CLI only performs `SELECT` from `public.market` for dates ending **2026-06-11** (old test begins 2026-06-12), rebuilds the same P4 pool and fails before writing files if pre-test valid label count or recomputed raw labels disagrees with the old `refit_samples=241095`. It does not edit the original Ridge model.
6. Output files: `corporate_action_gate.json`, `corporate_action_daily.csv`, `corporate_action_impacted_cases.csv`. Under the currently empty complete coverage ledger, `training_label_allowed_rows=0` is a **correct fail-closed outcome**, not evidence that corporate actions are absent.

**Outstanding blocker**: obtain, source-check and version complete market-wide historical split/bonus/cash-dividend records, including ex-dates and publication timestamps; separately resolve rights issues, halts, adjusted input features and coverage attestations. Only after that rebuild labels and train a new model with an actually unseen future evaluation period. Do not resume the Transformer track based on the existing contaminated labels.
