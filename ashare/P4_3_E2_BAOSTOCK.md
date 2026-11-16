# P4.3-E2 · BaoStock 免费第三数据源：历史公司行为交叉核验

> 状态：代码与独立合成单测已完成；**尚未获得用户 Ubuntu 实际 BaoStock 登录/查询产物，不允许声明免费接口覆盖完整，更不准模型重训**。

## 背景及来源

此前 AkShare + 新浪/巨潮首批 10 股，29/30 接口请求成功；`000004` 的巨潮接口在 Ubuntu 以只读观测明确返回 `records=[]`，但这不能证明它历史无分红。`stock_scripts` 已负责生产实时/分钟/日线行情，此任务**不重复采集行情，不改任何行情表或 cron**。

免费第三源采用 [BaoStock 官方 API 文档](https://github.com/zxygithub/baostock/blob/master/docs/pythonAPI.md) 中 `query_dividend_data(code, year, yearType="operate")`，**没有 Tushare Token，也无需 BaoStock API Key**。它返回每股权益而非 AkShare 新浪/巨潮“每 10 股”口径：`dividStocksPs` + `dividReserveToStockPs` 构成每旧股新增股份，`dividCashPsBeforeTax` 为每旧股税前现金。

**重要限制**：BaoStock 未证明包含全部事件，可能存在停服/退市遗漏、字段空白、资料修订、实施公告缺失及配股等复杂公司行为。三源一致也不是发行人官方完整覆盖证明。首批只检查 `000004` 与已有官方单笔证据 `002594 BYD-2025-047`；后续逐批扩大到此前冻结的 5,449 股票样本范围，但该范围只代表现有正式行情库包含过的股票，并不等于交易所上市全体历史股票。

## 代码

- `ashare/p4_action_feed_baostock.py`：`fetch` + `audit` 两个 CLI。只读冻结股票范围快照和 Ridge 元数据；标准化数据按实施年份查询，按旧测试期边界过滤后**才持久化**。任何请求失败、缺失除权日、接口截断和非沪深映射均记为待核，不会当作无公司行为。
- `tests/test_p4_action_feed_baostock.py`：15 项不触网单测，覆盖样本缺失、空结果、年限、截断/失败、不含配股数据的隔离声明、股份权益单位、重复除权事件、跨源冲突、比亚迪公告控制、SHA256 防篡改、北京证券交易所代码隔离等边界。
- 同时保留现有 `ashare/p4_action_feed_free.py` 及用户已经生成的原有 AkShare 抓取批次。**禁止**覆盖它们；新第三源批次在独立根目录存储。

## Ubuntu 执行：先核查 000004，再检查已知 BYD 事件

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev && git pull --ff-only origin ashare-dev
python -m pip install baostock
python -m unittest discover -s tests -p 'test_p4_action_feed_baostock.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

META=reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json
AKROOT=reports/ashare_p4_action_feed_free
SNAP=$AKROOT/snapshot_2024-12-01_2026-06-11
ROOT=reports/ashare_p4_action_feed_baostock

# 首批 2 只：000004 排查巨潮空 records，002594 核对已核实的公司送转事件。
# 仅执行 BaoStock 2024/2025/2026 各一查询，预期共 6 次查询（免费、匿名）。
python -u -m ashare.p4_action_feed_baostock fetch \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --code 000004 --code 002594 --wait-seconds 1.5 \
  --out "$ROOT/batches_pilot/batch_000000"

# 与已有 AkShare 的“未验证候选”对照。注意之前的10只股票批次
# 没抓 BYD，所以比亚迪可能仅匹配发行人已核实公告而非 AkShare 候选。
python -u -m ashare.p4_action_feed_baostock audit \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --batch-root "$ROOT/batches_pilot" \
  --akshare-events "$AKROOT/audit_retry_fixed/UNVERIFIED_corporate_actions_candidates.csv" \
  --akshare-coverage "$AKROOT/audit_retry_fixed/coverage_by_stock.csv" \
  --issuer-controls ashare/data/corporate_actions_verified_examples.csv \
  --out "$ROOT/audit_pilot"
```

若你的 AkShare 审计目录名称有所不同，只修改 `--akshare-events`、`--akshare-coverage` 路径，**不要移动或改写原 AkShare 批次文件**。BaoStock 结果中的 000004 无事件仅表示“第三源本次未发现记录”，不是历史无事件的认证。若 BYD 条款与官方样例不一致，必须作为数据差异保留，不可改写官方证据来迎合来源。

## 独立产物与验收

| 文件 | 核验重点 |
|---|---|
| `batches_pilot/batch_000000/batch.json` | 历史冻结边界、请求数、SHA256、不可变批次 |
| `.../request_ledger.csv` | 每股×每年是否 SUCCESS、无效日期、接口错误 |
| `.../raw_baostock_dividends.csv` | 实施日期在冻结边界前的原始权益字段（**按每股**） |
| `audit_pilot/baostock_corrob_audit.json` | 候选数量、未核实状态、差异数量 |
| `.../UNVERIFIED_baostock_candidates.csv` | 单源送转与现金权益，仍非可训练标签 |
| `.../baostock_cross_source_issues.csv` | AkShare / BaoStock 来源冲突、缺失、格式问题 |
| `.../baostock_issuer_controls.csv` | 官方已知 BYD 样本与 BaoStock 条款对照 |
| `.../baostock_coverage_by_stock.csv` | 股票成功/失败、仅部分年份/未抓、空结果标签 |

### 状态语义

- `NOT_FETCHED`：本批次未查询该股票。
- `FAILED_OR_UNSUPPORTED`：存在接口错误、无效日期或交易所不支持。
- `PARTIAL_YEAR_RANGE`：只查询了部分年份。**2000+股票的历史验证不能根据一年空结果断言无分红。**
- `FETCHED_NOT_VERIFIED`：已经完成此审计区间内所有三个实施年份的请求，但从未认证事件完整。
- `BAOSTOCK_AKSHARE_MATCH_UNVERIFIED`：候选条款比对一致，**仍非历史完整证明**。
- `AKSHARE_NOT_COMPARABLE_YET` / `BAOSTOCK_NOT_COMPARABLE_YET`：某一来源尚未完成对应股票的抓取，**不能记作双方条款冲突**。
- `BAOSTOCK_PARTIAL_AMOUNT_FIELDS_UNVERIFIED`：BaoStock 送/转/派字段部分为空且其余存在正值；按数值零暂供核验对照，禁止直接用作经济收益标签。

所有报告永远输出 `complete_verified_assertions_created=0`、`can_emit_training_labels=false`、`full_historical_action_coverage_certified=false`。年度查询中的“空返回”只表示此接口没有返回信息。异常出现在原数据源、代码版本变化、特殊证券或查询权限时均单独记录而非放行。

## 扩大样本：有节制地分批，不与初批覆盖冲突

试点成功且没有明显响应限流后，可把新的 offset 区间单独采集并审核，先 **10～30 只**，不要立刻对 5,449 只并发请求。

```bash
python -u -m ashare.p4_action_feed_baostock fetch \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --offset 10 --limit 20 --wait-seconds 1.5 \
  --out "$ROOT/batches_10_29/batch_000010"

python -u -m ashare.p4_action_feed_baostock audit \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --batch-root "$ROOT/batches_10_29" \
  --out "$ROOT/audit_10_29"
```

成功完成这一试点并评估请求失败与实际数据差异后，再制定节流、断点续跑、交换所/退市等额外来源和完整覆盖证明方案。新的前瞻实验标签仍然不能启动；保持旧测试开始日 `2026-06-12` 的冻结边界。
