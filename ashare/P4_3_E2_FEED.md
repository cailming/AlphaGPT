# P4.3-E 第二阶段：全市场历史公司行为候选数据接入与覆盖率审计

> **默认方案已改为免费、无 Token 的 [AkShare 双源抓取与覆盖审计](P4_3_E2_FREE_FEED.md)。** 本文件说明的 Tushare Pro 流程仅保留给已有 Token 且自行选择使用的用户；不是项目的必选依赖。两种方案的供应商记录都不能自动签发完整公司行为覆盖证明。

**实施范围（2026-11-15）**：已交付可运行的 Tushare Pro 批量接入、全市场范围快照、供应商字段映射、去重/冲突隔离、按股票覆盖审计与发行人已核公告对账。**未取得全市场完整实施公告档案；不会签发 `COMPLETE_VERIFIED`，旧 P4 经济收益标签质量闸门仍然拒训。**

## 为什么采用两层证据

1. **供应商全市场候选数据**：Tushare Pro `stock_basic` 查询 L（上市）、D（退市）、P（暂停上市），并与本项目 `public.market WHERE not_billing=1` 的历史股票代码取并集，暴露供应商无法映射、漏抓以及退市股票遗漏。Tushare `dividend` 按股票读取历次实施分红、送股、转增及其实施公告、除权、登记日期。官方字段说明：[stock_basic](https://tushare.pro/document/2?doc_id=25)，[dividend](https://tushare.pro/document/2?doc_id=103)。**API 需要用户自己的 Tushare Pro 凭证及相应权限**，没有凭证不能虚构实际数据。
2. **发行人/交易所完整性与审核证明**：单个股票请求成功、空返回，或分红事件能对上既有公告，都**不是**该股票在研究区间的全公司行为完整性的证明。`dividend` 不含配股，且仍有复杂权益、停复牌等问题；外部供应商数据只输出 `UNVERIFIED_VENDOR`，不生成自动的 `COMPLETE_VERIFIED` 覆盖清单。已有比亚迪 002594 实施公告作为差异检查的硬性对照，不代替发行人公告批量验证。

## 数据安全和严格时间边界

- 原 Ridge 模型元数据中的 `test_start=2026-06-12` 是不可更改的历史截止点；三条 CLI 都用 `--reference-metadata` 核验它。市场代码只通过 PostgreSQL `SELECT DISTINCT` 读取，范围为 `start <= trade_date < test_start`。
- 批量供应商请求按股票取回历史记录；写盘前丢弃 `ex_date` / `ann_date` / `imp_ann_date` 落在旧测试期及其后的行。**不加载旧测试期任何收益标签**。取得更多供应商历史记录不是独立新 OOS 测试。
- 快照及批次均为不可覆盖输出，带 SHA-256 校验。API 错误、供应商映射缺失、达到单次返回 2000 行上限等都标为不完整；多批次重复股票拒绝自动择优。历史退市公司必须纳入候选股票列表，不得使用今天的上市名单来覆盖旧历史。
- 实施数据标准化时只接受 `div_proc=实施`、有效的 `imp_ann_date`（实施公告日）与除权日、股权登记日、送转比例及**每股税前**现金额。`ann_date` 可能仅是预案日期，不能代替实施日期。按 `1 + stk_bo_rate + stk_co_rate` 计算股份倍数；`stk_div` 不一致、同股同除权日条款冲突均隔离。
- 此工具**不向数据库写表**，不变更 P4 原始 Ridge/已核标签，更不直接将候选 CSV 送进模型训练。

## Ubuntu 执行（先抓取 50 只验证凭证与限流，再批量）

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_action_feed.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

# 可选的供应商依赖；不要把 Token 保存到仓库、命令记录或输出文件
python -m pip install tushare
# 使用隐藏输入，避免把密钥明文写进 shell 历史或项目文件
read -r -s -p "Tushare Pro Token: " TUSHARE_TOKEN
echo
export TUSHARE_TOKEN
META=reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json
ROOT=reports/ashare_p4_action_feed
SNAP=$ROOT/snapshot_2024-12-01_2026-06-11

# 1. 只读 PostgreSQL 取历史交易股票；Tushare 取 L/D/P 股票名录
python -u -m ashare.p4_action_feed snapshot \
  --start 2024-12-01 --reference-metadata "$META" --out "$SNAP"

# 2. 先抓取前 50 只股票，独立批次不可覆盖，失败也有 ledger
python -u -m ashare.p4_action_feed fetch \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --offset 0 --limit 50 --wait-seconds 1.5 \
  --out "$ROOT/batches/batch_000000"

# 3. 检查首次输出、可用的供应商权限和 API 限流后，再完成剩余批次
N=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["expected_symbols"])' "$SNAP/snapshot.json")
for ((off=50; off<N; off+=50)); do
  printf -v suffix '%06d' "$off"
  output="$ROOT/batches/batch_$suffix"
  if [ -f "$output/batch.json" ]; then echo "Existing immutable batch: $output"; continue; fi
  python -u -m ashare.p4_action_feed fetch \
    --reference-metadata "$META" --snapshot "$SNAP" \
    --offset "$off" --limit 50 --wait-seconds 1.5 --out "$output" || exit 1
done

# 4. 不改写任何原始供应商批次；按股汇总、与比亚迪公告核验
python -u -m ashare.p4_action_feed audit \
  --reference-metadata "$META" --snapshot "$SNAP" \
  --batch-root "$ROOT/batches" \
  --official-controls ashare/data/corporate_actions_verified_examples.csv \
  --out "$ROOT/audit_2024-12-01_2026-06-11"
```

若 `snapshot` 仅用于离线验证，可提供 `--market-codes-csv`（含 `stock_code`）及 `--stock-basic-csv`（含 Tushare `ts_code,list_status,list_date,delist_date`）以跳过数据库和在线基础股票列表调用；`fetch` 仍需真实 Token。先确认 `pip` 有对应 Tushare 权限和接口速率，脚本只是保守延时 1.5 秒，并不保证适用于任何套餐。

## 输入/输出和验收口径

`$SNAP/`: `universe.csv`、`stock_basic.csv`、`snapshot.json`（股票全集、来源标志、退市/映射异常、快照 SHA-256）。**研究范围并集不等于历史数据商保证完整的所有证券**。

`$ROOT/batches/batch_<offset>/`: `request_ledger.csv`、`raw_dividends.csv`、`batch.json`（请求结果、错误或截断、单次响应量、旧测试期剔除数量、文件校验和）。批次内没有旧测试期的有效除权/公告日期记录。

`$ROOT/audit_2024-12-01_2026-06-11/`:

- `feed_coverage_audit.json`：预期股票、成功/失败/缺失抓取、未映射股票、供应商候选及拒绝原因、旧发行人公告对账差异；包含 `full_market_actions_certified=false` 与 `training_gate_must_remain_closed=true`。
- `coverage_by_stock.csv`：每个股票 `NOT_FETCHED` / `FETCH_FAILED_OR_TRUNCATED` / `FETCHED_NOT_VERIFIED`。从不输出 `COMPLETE_VERIFIED`。
- `UNVERIFIED_corporate_actions_candidates.csv`：已规范化但**尚未核验发行人公告的供应商候选事件**。特别注意 `source_url` 仅指供应商接口文档，绝非原始公告凭证。
- `rejected_vendor_rows.csv`：未实施、缺关键字段、条款互相矛盾、同日冲突等记录（不会自动挑一条）。
- `issuer_control_reconciliation.csv`：已有 `BYD-2025-047` 等官方硬核事件与候选事件的匹配/缺失告警；匹配不能证明其余事件完整。
- `COMPLETE_VERIFIED_NOT_ISSUED.txt`：明确本阶段不发训练放行证明。

**不要把 `UNVERIFIED...csv` 重命名为现有 `corporate_actions_verified_examples.csv`，也不要因为成功抓取所有股票就人工填写空白的 `corporate_action_coverage_examples.csv`。** 正确升级路径是获取发行人/交易所证据与独立完整性对账，针对每只股票/日期范围核查配股、复权、停复牌等历史行为，只有独立验证完的具体股票区间才能人工签发 `COMPLETE_VERIFIED`。

建议先将快照、第一批原始供应商记录及审计结果上传到 Google Drive `agu/alphagpt/ashare_p4_action_feed/`，核验逻辑与权限后再分批覆盖所有股票。整个供应商抓取可能需要大量 API 调用且受账户权限、服务频率和请求失败影响；不能据此宣称实际全市场覆盖已经完成。

## 已实现与待解决

已实现：只读股票全集并集、退市记录纳入、Tushare API 批处理、严格旧测试边界、供应商字段/实施时间校验、冲突和上限拒绝、不可覆盖的 SHA-256 快照、逐股缺口审计和原公告对账；新增 11 项合成测试在开发环境通过。

待解决：实际 Ubuntu Tushare 凭证授权、全量 API 拉取、请求失败复跑的人工批次归档与去重、全面的发行人公告交叉验证、配股及特殊公司行为覆盖证明、历史价格与量价特征调整。**模型训练继续被 P4.3-E 第一阶段质量闸门阻断**。
