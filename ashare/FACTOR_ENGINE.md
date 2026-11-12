# A-share factor engine — stage 2

**Research only.** No trading instructions, no order placement, no DB schema writes, no claims of proven profitable PnL.

Reuses the original `model_core.vm.StackVM` with an optional mask-aware path. The original zero-argument `StackVM()` and crypto strategy calls remain compatible.

## Inputs

Based on the verified 2026-11-11 `agu/public_1111.sql` six-table schema:

- `market`: daily OHLCV and quoted `"MA_20"`.
- `stock_main_force_daily`: daily behavior scores, ratios, `created_at`.
- `billing` and `billing_bak`: daily-cumulative minute snapshots; NOT consumed by this daily factor engine yet.
- `stock_info`, `stock_signal_event`: deliberately excluded from factor input at this stage to avoid point-in-time universe/status leakage.

`AShareDailyLoader.load(..., include_warmup=True)` keeps pre-start historical bars for rolling windows, while fetching up to 30 future calendar days solely to construct forward labels. Returned rows never extend past requested end date.

`build_daily_tensor` returns float32 features [stock, factor, date], boolean availability mask, and separately stored target [stock, date]. It never fills missing observations by borrowing future data. The strict default only treats main-force values as available if `main_force_created_at` is on or before the source date at 23:59:59; later backfills become invalid. Timestamp alone does not fully verify all source data lineage.

## Operators and formula execution

Whitelist of 17 operators: `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `ABS`, `SIGN`, `GATE`, `DELAY1`, `TS_DELTA5`, `TS_MEAN5`, `TS_MEAN20`, `TS_STD20`, `TS_MAX20`, `TS_MIN20`, `TS_RANK20`, `CS_RANK`.

- `TS_*` lookbacks require a full sequence of valid observations.
- `CS_RANK` ranks stocks within the same date, with average ranks for ties; its result depends on the research universe.
- Division by zero is invalid rather than silently forced to zero.
- Random search samples only syntactically valid **postfix** expressions.
- Lowercase feature IDs are exactly `DAILY_COLUMNS` from `ashare/daily.py`.

Example Python usage:

~~~python
from ashare.tensor import build_daily_tensor
from ashare.formula import AShareFormulaEngine
bundle = build_daily_tensor(daily_frame)
engine = AShareFormulaEngine()
tokens = engine.parse("main_force holding ADD")
values, valid = engine.execute(tokens, bundle)
print(engine.pretty(tokens))  # ADD(main_force, holding)
~~~

## Run

~~~bash
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
export ASHARE_DB_URL='postgresql+psycopg2://research_user:<PASSWORD>@localhost:5432/agu'
python -m ashare.research --start 2025-01-01 --end 2026-09-30 --samples 200 --min-stocks 15
~~~

Use a SELECT-only database role. Do not commit credentials. `--allow-unverified-main-force` is an explicit diagnostic bypass, not permission to claim a causal historical backtest.

The CLI runs a seeded random-expression **baseline**: select highest positive training-period mean daily Spearman Rank IC, purge two trade dates across the chronological split, and report separate holdout IC. It does not train a Transformer yet, nor model actual orders or portfolio PnL. Use it to verify whether the factor/VM/evaluation plumbing works before training AlphaGPT itself.

## Limitations and follow-up

1. Actual PostgreSQL query execution and source price adjustments have not been audited in this phase; validate real data quality/coverage.
2. `created_at` is not necessarily immutable; if historical scores were retrospectively recomputed or backfilled, stronger event-time/version provenance is needed.
3. Future label T+1 open -> T+2 open is a theoretical outcome; no T+1, board-level price limits, suspension, fees, impact, or fills are modeled yet.
4. Past-universe construction must use historical listing/delisting and tradeability metadata, not today's `stock_info.status/st`.
5. Next: transaction-aware A-share walk-forward backtesting and Transformer generator integration, separated from live execution.


## 实时进度输出

研究命令默认打印四个阶段的即时进度：数据库分批读取的实际行数、
日线因子计算、张量构建、候选公式搜索进度，以及最终 JSON。
数据库每批默认读取 50,000 行；其行数是累计读取值，不是百分比
（没有为此额外执行高成本的 COUNT 查询）。首次查询返回前，会显示
“等待 PostgreSQL 返回首批数据”。公式进度包含完成比例、不同公式数、
有效评分公式数、当前最佳训练期 IC、已用时间及粗略剩余时间。
预计剩余时间仅按目前平均速度外推，不能保证准确。

~~~bash
python -u -m ashare.research --start 2025-01-01 --end 2026-09-30 \
  --samples 200 --min-stocks 15 --log-every 10 --db-chunk-size 50000
~~~

`--quiet` 可关闭进度，恢复只输出最终 JSON；增量查询和计算使用
只读 SQL，不会修改已有的六张表。如果遇到 PostgreSQL 查询超时，
先缩小日期范围或通过 `--codes` 限定股票排查。