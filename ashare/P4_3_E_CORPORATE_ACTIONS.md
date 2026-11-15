# P4.3-E — 公司行为经济收益标签的严格质量闸门

阶段状态：**只读审计工具、验证过的单笔公告案例和单元测试已实现；全市场公司行为完整覆盖数据尚未取得，不能重训或晋级模型。**

## 为何必须先做

旧 P4 使用未复权日线的 T+1 开盘至 T+6 开盘之比作为收益标签。比亚迪 002594 的 2025-07-29 送转除权提供了明确反例：持有 1 股在送转后对应 3 股，同时取得税前 3.974 元现金红利；直接比较两日每股未复权价格会制造虚假的大幅亏损。依据深交所披露的 [比亚迪 2025-047 实施公告](https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-07-22/f30e6169-4779-487e-acff-8c1eff230fa4.PDF)。

## 三条严格规则

1. **标签**：只将 (T+1 开盘, T+6 开盘] 区间内按除权日实施的已核事件计入权益。事件 `share_multiplier` 表示每持有 1 股变为多少股，`cash_per_old_share` 表示实施前每股的税前现金权益。按时间顺序累计 `shares = shares × multiplier`、`cash += shares_before × cash_per_old_share`，最终粗略经济收益为 `(shares × exit_open + cash − entry_open) / entry_open`。现金不复投，暂不计分红税、红股税、零碎股、交易费和成交限制。这是研究标签而非可交易 PnL。
2. **覆盖**：有某一条公告不等于一个股票某时段的所有公司行为都已枚举。`corporate_action_coverage_examples.csv` 故意只包含表头，不推断不存在其他事件。只有人工核验数据源达到完整覆盖并显式填写 `COMPLETE_VERIFIED`，且覆盖整个**过去连续 40 个交易日因子回看窗口到标签退出日**，才允许生成训练目标。
3. **特征**：任何已知除权日落在 T 日或此前 40 个市场交易日回看窗口中，均隔离样本，因为旧量价因子未经复权与份额尺度校正；未来实施的公司行为可以用于未来监督标签，但绝不可输入 T 时点的特征或把后来的公开内容倒灌。

## CSV 输入格式

`ashare/data/corporate_actions_verified_examples.csv`：

```csv
stock_code,ex_date,published_on,share_multiplier,cash_per_old_share,source_url,source_id
002594,2025-07-29,2025-07-22,3,3.974,https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-07-22/f30e6169-4779-487e-acff-8c1eff230fa4.PDF,BYD-2025-047
```

按股票、除权日唯一；多项同日动作应先按实施权益汇总到一行，出现歧义时拒绝。公告可见日期取审慎口径 2025-07-22。只做 A 股已核实的税前送转与现金权益估算；不能涵盖配股缴款、增发、并购换股、退市清算等复杂事件。

`ashare/data/corporate_action_coverage_examples.csv`：

```csv
stock_code,start_date,end_date,verified_on,source_url,source_id,status
```

**不应擅自添加 BYD 的 `COMPLETE_VERIFIED` 记录**：已核对一份公告，不能证明该期间其余公司行为都已覆盖。只有经来源审计的完整事件名录才可以提交覆盖清单；清单是人工签发声明，不是自动证明工具。

## Ubuntu 执行

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_corporate_actions.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

python -u -m ashare.p4_corporate_actions \
  --start 2025-03-01 \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --events ashare/data/corporate_actions_verified_examples.csv \
  --coverage ashare/data/corporate_action_coverage_examples.csv \
  --max-impact-rows 500 \
  --out reports/ashare_p4_corporate_actions
```

该程序沿用原 P4 的动态 Top-800 和风险事件局部拦截，但**只 SELECT 旧测试集开始前的数据**，由旧模型元数据读取固定 `test_start=2026-06-12`，只查询到 `2026-06-11`，不开启未来标签尾部加载。复算历史旧标签并强制对账已保存的 `refit_samples=241095`，不匹配则失败退出且不输出结果，不重新查看已见的 2026 年旧测试收益。

输出目录 `reports/ashare_p4_corporate_actions/2025-03-01_2026-06-11/`：

- `corporate_action_gate.json`：旧标签复算一致数、已知公司行为影响数、显式完整覆盖样本数、训练允许样本数、逐原因隔离数。
- `corporate_action_daily.csv`：按 T 日聚合的覆盖与隔离统计。
- `corporate_action_impacted_cases.csv`：有已核公司行为影响标签或因子窗口的病例，限前 500 条（可调）；包含未授权训练的**示例性经济收益**及原因，不能当成训练集。

由于当前完整覆盖证明 CSV 为空，预期 `training_label_allowed_rows=0`。这表示**质量闸门正常拒绝未知数据**，不是程序故障，更不代表股市只有比亚迪发生公司行为。

## 阶段验收及后续数据源

- 已完成：事件格式校验、除权股数和现金权益累计、多事件计算、T+1 除权日无受益边界、源价格与原标签重算、40 日因子污染隔离、覆盖缺口强制拒绝、测试期不可读取。新增 7 项合成单测在开发环境通过。
- 未完成：全市场历史公司行为完整名录及其可信覆盖证明；此前的 Ridge 仍基于旧标签，不可作为修正后模型。未运行 Ubuntu 全仓测试与真实数据库大规模审计。
- 下一道闸：引入可审计、含版本/发布/生效日期的权威公司行为数据，验证完整覆盖并处理复杂配股/税费、停复牌和未复权因子；之后才可构造新训练集，再在预先封存的新日期区间评价模型。

## 真实数据库核验（2026-11-15）

真实数据库只读运行的三份输出已上传并交叉核验：[Google Drive 目录](https://drive.google.com/drive/folders/1RSqOOb5hPsJJXBzKsgcyyDZXOK58eN05)。

- 308 个选股日 × 800 条 = **246,400** 条原股票池样本；**241,095** 条有效旧标签重算一致，**5,305** 条原标签为空。旧 Ridge `model_metadata.json` 的模型版本、配置、2026-06-12 测试开始日与 `refit_samples=241095` 均一致。
- 对旧 `pool_daily_coverage.csv` 逐日期检查：308 日股票池规模 0 差异，302 日标签计数一致。最后 6 日（2026-06-04～06-11）旧产物共有 4,795 条标签，本次因旧测试期边界不读取未来行情，全部舍弃，符合 embargo；旧截至 06-11 的有效标签 245,890 − 4,795 = 241,095。
- 受 002594 比亚迪送转事件影响的明细 **45** 条：5 条标签跨除权、40 条因子回看窗口污染；独立复算 45 条原收益、5 条税前示例经济收益，数值一致、没有重复。其余大幅涨跌尚未归因。
- **完整覆盖证明数量 0；可训练标签 0**。全部 241,095 条有效旧标签按设计标记 `UNKNOWN_ACTION_COVERAGE`，缺失原标签 5,305 条标记 `MISSING_ORIGINAL_LABEL`；经济训练目标仍为 NaN。工程闸门工作正确，但历史权益事件覆盖的数据质量关仍未通过。
- 核验仅涉及已上传文件、保存模型元数据与旧股票池日度产物；不代表独立直连数据库、证明价格快照完全相同或完成全市场送转/分红/配股等事件审计。**继续禁止新模型重训与盈利宣传**。

下一阶段的阻断条件不在算法本身，而在取得并核实可证明完整覆盖的历史公司行为数据源（包含原始记录来源、披露日、实施日、权益规则及覆盖状态），同时落实历史因子窗口复权/隔离契约。数据治理完成后才允许经济收益目标进入训练。
