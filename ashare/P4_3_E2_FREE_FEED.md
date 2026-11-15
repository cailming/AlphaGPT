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
- `issuer_control_reconciliation.csv`：若试点批次未覆盖 `002594`，已知比亚迪控制事件仍显示 `MISSING_OR_CONFLICTING`，**这是未抓到该股票导致的预期缺口，不意味着已查到错误**。后续扩展覆盖后再核对。

## 安全与最终门槛

这只是**免费数据候选流水线**，不是已验证权益事件数据库。不得自动生成覆盖证明、自动解锁 `ashare.p4_corporate_actions` 的训练闸门，不得用旧已查看测试期挑选信号。未取得完整历史公司行为来源、历史复权特征和公司公告核验前，Transformer / 新 Ridge 训练继续暂停。

免费备选：BaoStock `query_dividend_data(code, year, yearType="operate")` 也可免费获取每股分红、送转、除权日，适合后续第三源交叉验证；但在引入其历史可靠性与时点审计前也不能自动签发完整覆盖。
