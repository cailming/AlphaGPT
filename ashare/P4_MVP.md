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
