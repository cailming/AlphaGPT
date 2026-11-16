# P4.3-E2 — 无 Token 免费公司行为接入（AkShare）

**默认推荐路线：本地正式行情库 `public.market` 的历史股票范围 + AkShare 公开接口（新浪历史分红/配股、巨潮历史分红）+ 严格的候选事件交叉核验。** 不使用或索取 Tushare Pro Token，不读取/写入生产交易数据库的任何数据修改，不重训模型。

> 数据接口免费与开源抓取库，不等于上游数据有无限请求额度、无使用限制或已通过完整性认证。公开网站改版、限流、覆盖缺口、误标历史公告仍然可能发生。两个公开接口一致也**不是完整公司行为覆盖证明**。

## 核心差异

- `p4_action_feed.py`：之前的 Tushare 可选路线，保留用于已有 Token 的用户；不是必需。
- **`p4_action_feed_free.py`**：当前的无 Token 路线，三个子命令 `snapshot -> fetch -> audit`，不依赖 Tushare。运行时 `pip install akshare`。所有供应商结果只作为 `UNVERIFIED` 研究候选。
- 以已保存 Ridge 元数据 `test_start=2026-06-12` 作为历史冻结边界。公开接口可能一次返回全历史记录，程序**过滤掉**旧测试边界及之后的除权/公告日记录再持久化，不使用这些行作实验评价。
- 股票全集来自已有 `public.market WHERE not_billing=1` 在指定历史范围内出现过的六位股票代码；**不会宣称这是交易所全部历史上市股票**，可能遗漏从未进入数据库的股票。需要时补充独立的退市及历史上市名录核对。
- 每股送转增量为 `(送股+转增)/10`，每旧股税前现金为 `派息/10`。两公开接口在有效除权日、股权登记日和权益数值不一致时，将该股票/除权日事件标记为冲突且不输出为可用候选；仅单源成功时明确标注 `SINGLE_SOURCE_*_UNVERIFIED`。
- 配股数据单独保存为 **`rights_issues_for_review.csv`**，不擅自将配股的缴款/认购权益折算成分红。
- 不会产出 `COMPLETE_VERIFIED`，亦**不能把候选 CSV 直接传入 P4.3-E1 `--events` 作为已证实事件**。仍须逐项核对发行人实施公告及完整数据源覆盖，并签发独立的覆盖证明。

## Ubuntu 命令：先试点 10 只

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev
python -m pip install akshare

python -m unittest discover -s tests -p 'test_p4_action_feed_free.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

META=reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json
ROOT=reports/ashare_p4_action_feed_free
SNAP=$ROOT/snapshot_2024-12-01_2026-06-11

# (1) 仅 SELECT 本地正式历史股票代码；不调用 Tushare。
python -u -m ashare.p4_action_feed_free snapshot \
  --start 2024-12-01 \
  --reference-metadata "$META" \
  --out "$SNAP"

# (2) 从快照头部选 10 只股票，每只三个免费接口（新浪分红、巨潮分红、新浪配股）。
# 对公开来源限流较敏感，试点请保留间隔；批次目录必须是新目录。
python -u -m ashare.p4_action_feed_free fetch \
  --reference-metadata "$META" \
  --snapshot "$SNAP" \
  --offset 0 --limit 10 --wait-seconds 1.5 \
  --out "$ROOT/batches/batch_000000"

# (3) 审计抓取完整性、不同公开源不一致、历史日期及比亚迪官方控制样本。
python -u -m ashare.p4_action_feed_free audit \
  --reference-metadata "$META" \
  --snapshot "$SNAP" \
  --batch-root "$ROOT/batches" \
  --out "$ROOT/audit_first_batch"
```

### 分批扩展

审计试点确认接口字段、失败率和限流情况后，可以使用 `--offset 10 --limit 10 --out "$ROOT/batches/batch_000010"` 继续抓取，并用新路径重新执行审计（审计目录不可覆盖）。**不要同时启动多个高并发公开网页请求。** 每次返回上限/接口异常均写在不可变批次 `request_ledger.csv`，不能把 HTTP 成功且记录为空当作覆盖证明。

### 主要输出

- `snapshot.../universe.csv`、`snapshot.json`：来自正式行情库的历史交易股票代码、快照 SHA256、固定测试边界。
- `batches/batch_*/raw_sina_dividend.csv`、`raw_cninfo_dividend.csv`、`raw_sina_rights.csv`、`request_ledger.csv`、`batch.json`：原始但已经过旧测试边界过滤的供应商记录，校验 SHA256 及请求成功/失败/截断状态。
- `audit_first_batch/feed_coverage_audit.json`、`coverage_by_stock.csv`：按股票汇总 `NOT_FETCHED` / `FETCH_FAILED_OR_TRUNCATED` / `FETCHED_NOT_VERIFIED` 三类状态，**没有任何 `COMPLETE_VERIFIED`**。
- `UNVERIFIED_corporate_actions_candidates.csv`：按每十股折算为每旧股的分红送转候选，记录单源/双源对账情况，不可直接作为已验证训练标签。
- `audit_issues.csv`：源数据字段缺失、同日多项矛盾、不同源送转和税前现金冲突。
- `rights_issues_for_review.csv`：配股仍需处理认购比例、缴款和股份到账等复杂影响。
- `issuer_control_reconciliation.csv`：区分 `NOT_FETCHED_FOR_CONTROL`（样本未请求）、`INCOMPLETE_FETCH_FOR_CONTROL`（该股有子接口失败）、`MATCHED_ISSUER_CONTROL`（已核公告与候选条款匹配）和 `MISSING_OR_CONFLICTING`（该股接口完整返回、但未找到一致条款）。没有抓到比亚迪不应误算为公告冲突。

## 安全与最终门槛

这只是**免费数据候选流水线**，不是已验证权益事件数据库。不得自动生成覆盖证明、自动解锁 `ashare.p4_corporate_actions` 的训练闸门，不得用旧已查看测试期挑选信号。未取得完整历史公司行为来源、历史复权特征和公司公告核验前，Transformer / 新 Ridge 训练继续暂停。

免费备选：BaoStock `query_dividend_data(code, year, yearType="operate")` 也可免费获取每股分红、送转、除权日，适合后续第三源交叉验证；但在引入其历史可靠性与时点审计前也不能自动签发完整覆盖。


## 2026-11-16：首批真实运行诊断及兼容修复

用户的首批快照包含 **5,449 只在正式行情数据库中出现过的股票**，10 只测试股票共发起 30 次免费接口调用，其中 20 次成功、10 次失败或截断。因为每只股票必须同时通过新浪分红、巨潮分红和新浪配股三个子接口，十只股票全部为 `FETCH_FAILED_OR_TRUNCATED`；`5439 NOT_FETCHED` 是未轮询的剩余股票，`0 FETCHED_NOT_VERIFIED` 并不代表所有信息源完全无数据。

已从 GitHub 源码发现**一处实际兼容性错误**：巨潮方法错误调用了旧拼写 `stock_dividents_cninfo`。现在优先用当前 `stock_dividend_cninfo`，再兼容旧拼写，并新增对应回归测试。另对配股历史只统计本审计区间内除权日，原始历史条数另列 `rights_records_raw_all_history`；目前共有 **11 项**本地回归测试通过，尚待 Ubuntu 复测。注意：只有读取旧批次的 `request_ledger.csv` 才能确认其十次失败是否全部来自巨潮，不能只由聚合数字反推错误原因。

原审计还错误地将“比亚迪尚未包含在前十只股票”计入已知公告匹配失败。现在将 `NOT_FETCHED_FOR_CONTROL`、`INCOMPLETE_FETCH_FOR_CONTROL` 与真正 `MISSING_OR_CONFLICTING` 分开；`issuer_controls_mismatch` 只统计已完整抓取后仍不匹配的控制事件，另外新增未抓和未完整抓取计数。**新版审计的 `feed_coverage_audit.json`** 会输出 `endpoint_results` 和 `endpoint_failure_examples`，分解各子接口的成功/API 错误/返回上限数量；原始 `batch.json` 保留聚合计数，逐请求详情以 `request_ledger.csv` 为准；未修改旧批次，也不自动签发完整覆盖证明。

### 保留旧快照，使用全新路径重测

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_action_feed_free.py' -v

META=reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json
ROOT=reports/ashare_p4_action_feed_free
SNAP=$ROOT/snapshot_2024-12-01_2026-06-11

# 先查看旧批次哪一种子接口实际失败
python - <<'PY'
import pandas as pd
p = "reports/ashare_p4_action_feed_free/batches/batch_000000/request_ledger.csv"
df = pd.read_csv(p)
print(pd.crosstab(df["endpoint"], df["status"]))
print(df.loc[df.status.ne("SUCCESS"), ["stock_code", "endpoint", "status", "error"]].to_string(index=False))
PY

# 不重建 snapshot，不覆盖原始批次，用新批次根目录重测
python -u -m ashare.p4_action_feed_free fetch \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --offset 0 --limit 10 --wait-seconds 1.5 \
  --out "$ROOT/batches_retry_20261116/batch_000000"

python -u -m ashare.p4_action_feed_free audit \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --batch-root "$ROOT/batches_retry_20261116" \
  --out "$ROOT/audit_retry_20261116"
```

只有当新 `audit_retry_20261116/feed_coverage_audit.json` 的 `endpoint_results` 显示三个数据源都成功，相关股票的 `coverage_by_stock.csv` 才从 `FETCH_FAILED_OR_TRUNCATED` 转为 `FETCHED_NOT_VERIFIED`；**即使三源均成功，也不会成为 `COMPLETE_VERIFIED`**。配股历史的行数不是已确认配股事件数量；原有 4 条 `audit_issues` 必须看具体 `reason` 再判定是源字段无效还是跨源矛盾。不要直接对 5,449 只股票发起 16,347 次请求，先确认免费接口稳定性和服务条款。

## 2026-11-16：AkShare 1.20.65 单股巨潮日期字段 KeyError

用户的 Ubuntu 真实运行（`ashare-dev@7dac7c8464497ff1509ac1ec1165e12654bd6029`，AkShare `1.20.65`）显示：11 项原有单测通过，头 10 只股票的三源接口 **29/30** 成功，其中新浪分红/配股各 10/10，巨潮分红 9/10；股票级 `9 FETCHED_NOT_VERIFIED`、`1 FETCH_FAILED_OR_TRUNCATED`。另有四条未核实候选、五条无效或冲突记录，`COMPLETE_VERIFIED=0`，质量闸门应维持关闭。

剩余巨潮错误为 `KeyError: '实施方案公告日期'`。查阅 [AkShare 开源实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_dividend_cninfo.py)：该函数把 JSON `records` 变成 DataFrame 后直接读取 `实施方案公告日期`。所以报错可能源于 **`records=[]` 空响应或提供方字段变化**；单靠 KeyError **无法证明股票历史上确无分红**。不得把异常转换为 `SUCCESS`、空已验证名录或 `COMPLETE_VERIFIED`。

本次兼容补丁：

- 对 **仅巨潮** 的该字段 KeyError 单独记录 `UPSTREAM_EMPTY_OR_SCHEMA`，明确是“上游数据不确定”，继续按非成功接口计入股票级 `FETCH_FAILED_OR_TRUNCATED`；其他 KeyError 仍记 `API_ERROR`，严禁静默放行。
- 审计 JSON 增加 `cninfo_upstream_ambiguous_calls` 及每数据源至多 5 条 `endpoint_failure_stocks`（股票代码、状态、脱敏/截断后的错误），让单股异常可以直接定位。原有 `endpoint_failure_examples` 仍保留。
- 单元测试新增两项：同名字段 KeyError 必须隔离并保留其他来源的 UNVERIFIED 候选；其他字段 KeyError 不应误归因成“空或结构变动”。之前已成功的 29 次接口调用不受修改影响。

**执行顺序**：先检查旧的不可变批次，确认具体失败股票；完成 `git pull` 后只用新输出目录重试，绝不覆盖先前批次，也不能将三个重叠的 `batch_000000` 混在一个审计根目录。

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev && git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_action_feed_free.py' -v

python - <<'PY'
import pandas as pd
p = "reports/ashare_p4_action_feed_free/batches_retry_fixed/batch_000000/request_ledger.csv"
ledger = pd.read_csv(p, dtype={"stock_code": "string"}, keep_default_na=False)
print(ledger.loc[ledger.status.ne("SUCCESS"),
                 ["stock_code", "endpoint", "status", "error"]].to_string(index=False))
PY

META=reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json
ROOT=reports/ashare_p4_action_feed_free
SNAP=$ROOT/snapshot_2024-12-01_2026-06-11
python -u -m ashare.p4_action_feed_free fetch \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --offset 0 --limit 10 --wait-seconds 1.5 \
  --out "$ROOT/batches_diagnostic_v2/batch_000000"
python -u -m ashare.p4_action_feed_free audit \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --batch-root "$ROOT/batches_diagnostic_v2" \
  --out "$ROOT/audit_diagnostic_v2"
```

**验收**：即使这次返回 `UPSTREAM_EMPTY_OR_SCHEMA`，这也只是对供应商数据质量的正确表达，不等于修复上游数据或提升到 30/30；应暂不扩大至全市场批量。进一步检查该股的官方历史实施公告或选择第三个独立免费源交叉核对。不要修改 `stock_scripts` 的生产行情任务。
