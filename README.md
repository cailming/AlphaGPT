# AlphaGPT

![](showcase.png)

*在 AlphaGPT 开源后，社区曾使用开源版本管理了 1.66 亿人民币规模资金。*

[时隔数月后，项目作者再次受到了人身安全威胁。](https://github.com/imbue-bit/fuck_all_tian)

也许金融真的是一个意外死亡率很高的行业🤔

---

如果您觉得 AlphaGPT 很有用，可以通过以下渠道捐助作者：

![](helpme.jpg)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=imbue-bit/AlphaGPT&type=date&legend=top-left)](https://www.star-history.com/#imbue-bit/AlphaGPT&type=date&legend=top-left)

---

## AlphaGPT-Ashare 开发进度

> **研究状态（2026-11-14）**：当前工作分支为 [`ashare-dev`](https://github.com/cailming/AlphaGPT/tree/ashare-dev)，`main` 不用于本阶段开发。A 股部分属于**研究与模拟系统**，不是实盘交易程序；以下仅描述现有代码、已上传的研究输出和待验证事项。不要将模拟收益视为可交易利润或模型训练奖励。

### 目标、环境与数据

- **目标**：基于真实 PostgreSQL 行情，完成「按历史时点构建股票池 → 因子与监督标签 → 时间隔离训练/验证/测试 → 模型保存 → 批量推理 → 交易成本评价」闭环；优先覆盖有质量、有流动性的股票，允许主动放弃不适合预测的股票。
- **运行环境**：Ubuntu、Conda `py312`（Python 3.12）、PostgreSQL；计划使用 NVIDIA RTX 3090 24GB 与 PyTorch 训练小型 Transformer。当前 P4 MVP 的 Ridge **不依赖 GPU**。
- **核心数据**：`public.market`（正式日线标志 `not_billing=1`，OHLC 为**不复权**价格）；`public.billing`/`public.billing_bak`（分钟级**累计**成交量/额，需差分）；`public.stock_info`（`stock_name`、当前 `st` 状态）；`public.stock_main_force_daily`（有历史回填时点风险）；`public.stock_signal_event`（不能直接当作可信历史标签）。
- **数据字典**：Google Drive `agu/public_1111.sql`；回测与研究产出位于 `agu/alphagpt/`。数据库连接仅通过既有 `ASHARE_DB_URL` 或 `DB_*` 环境变量，**不提交密码、Token 或真实连接串**；研究模块仅做 SELECT，不自动修改生产 schema。
- **正式回测研究窗口**：从 2025-03-01 起，之前日线仅用于因子预热。既有 P1/P2 因子研究窗口为 2025-01-01 至 2026-09-30。

### 阶段进度（以仓库代码及已有结果为准）

| 阶段 | 实际内容与关键文件 | 状态 |
| --- | --- | --- |
| P1 | A 股 PostgreSQL 日线读取、12 因子、张量/mask、公式 VM、Rank IC：`ashare/daily.py`、`tensor.py`、`formula.py`、`research.py` | **已实现**；历史样本外因子 Rank IC 约 0.01856，非收益率 |
| P2 | 因子覆盖与主力资金时间戳质量审计：`ashare/quality.py`、`quality_report.py` | **已实现**；常规价量因子覆盖较好，历史主力评分覆盖/时点可信度不足 |
| P3.1 | 基础回测、次日开盘价格代理买入、A 股 T+1 卖出约束、100 股单位、费用/滑点：`ashare/backtest.py`、`backtest_run.py` | **已实现探索性模拟**；未证明开盘可成交 |
| P3.2 | 现金/持仓账本、拒单、过期估值、开放持仓诊断：`ashare/backtest.py`、`tests/test_backtest.py` | **已实现诊断** |
| P3.3-B | 当前 `stock_info.st` 黑名单诊断：`ashare/current_st.py`、`tests/test_current_st.py` | **已实现，但 NON-PIT**；不可用于历史真实收益或训练奖励 |
| P3.3-C | 已公告风险事件按披露日/生效日拦截：`ashare/risk_events.py`、`tests/test_risk_events.py`、`ashare/data/risk_events_verified_examples.csv` | **已实现部分历史事件覆盖**；已核实 002731、688496，非全市场历史 ST |
| P4.1 | 当日及此前信息的动态流动性 Top-N 股票池、连续有效交易日检查、风险事件局部拦截：`ashare/p4_mvp.py` | **真实 PostgreSQL 运行产物已交叉核验通过**（数量守恒、日/月汇总、唯一键、排名）；仍有风险/除权和时点可用性限制 |
| P4.2 | 过去窗口量价特征、未来开盘价标签、停牌缺口校验、按日期切分及跨集合标签净化：`ashare/p4_mvp.py` | **已有 MVP 代码与单测**；不复权导致的除权偏差仍需审计 |
| P4.3 | Ridge/轻量基线与因子审计、P4.3-D 原始价格核验、P4.3-E 公司行为经济收益标签质量闸门 | **P4.3-B/C/D 真实产物已核验；P4.3-E 只读治理模块与 7 项新增本地单测已完成**。全市场权益事件覆盖尚不存在，质量闸门默认拒绝训练；需 Ubuntu 实测和完整数据源 |
| P4.4 | PyTorch/CUDA Transformer、训练日志、checkpoint/恢复及与轻量模型公平对比 | **待开发** |

详细设计参见 [ashare/README.md](ashare/README.md)、[ashare/FACTOR_ENGINE.md](ashare/FACTOR_ENGINE.md)、[ashare/QUALITY_REPORT.md](ashare/QUALITY_REPORT.md)、[ashare/BACKTEST.md](ashare/BACKTEST.md)、[ashare/RISK_EVENTS.md](ashare/RISK_EVENTS.md)、[ashare/P4_MVP.md](ashare/P4_MVP.md)。

### 最近一次已上传的 P3.3-C 研究结果（不是已审计收益）

Google Drive：`agu/alphagpt/ashare_backtest_current_st_plus_events/2025-03-01_2026-09-30/summary.json`，本次进度更新时已核对其记录：

| 2025-03-01～2026-09-30 指标 | 探索性模拟结果 |
| --- | ---: |
| 初始模拟现金 | ¥1,000,000 |
| 期末账面估值 | ¥773,446.48 |
| 简单模拟收益率 | -22.66% |
| 最大账面回撤 | -62.93% |
| 模拟成交记录 | 7,546 |
| 假设交易费用合计 | ¥520,799.91 |
| 已知公告阻止的信号 | 127 |
| 期末估值过期持仓数 | 0 |
| 期末仍持有仓位数 | 10 |

`summary.json` 明确标注 `stock_pool_mode=NON_PIT_CURRENT_ST_PLUS_PARTIAL_EVENT_OVERLAY`、`audited_total_return_available=false`、`corporate_action_adjustment=NOT_MODELED`。**当前 ST 状态倒灌存在未来信息偏差；日线开盘价不等于可成交价；不复权价格尚未处理分红/拆股。** 这些数据只能用于工程诊断，不能宣称稳定盈利，也不能作为 Transformer 的真实奖励。当前主要发现为**换手频繁、费用过高、缺乏稳定净收益优势**。

### P4.1 独立股票池审计：真实数据库产物交叉核验（2026-11-14）

用户在 Ubuntu 运行 `python -u -m ashare.p4_pool_report --start 2025-03-01 --end 2026-09-30 --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5`，将以下 4 份产物上传至 Google Drive `agu/alphaGPT/ashare_p4_pool/2025-03-01_2026-09-30/`：

- `pool_summary.json`、`pool_daily_coverage.csv`、`pool_monthly_coverage.csv`、`pool_selected.csv`。
- **跨文件独立对账通过**：窗口内正式日线 2,100,437 条；依次首因拒绝：连续历史/无效 55,938、低价 62,717、流动性阈值 0、非有限特征 0、已知风险事件 0、超过每日 Top-800 1,673,782；最终入池 308,000，合计恰为 2,100,437。
- **股票池规模**：388 个观察到的研究日、385 个有效选股日、2,699 只曾入池股票；2025-03-03、03-04、03-05 没有候选，从 2025-03-06 起每个有效交易日恰为 800 只。19 个月内月度唯一股票数 967–1,132。
- **标签覆盖**：307,411 条有效 5 日未来收益标签、589 条缺失（99.81% 有效率）。没有股票+日期重复行、无效六位代码、价格低于 3 元、入池排名缺口或流动性顺序倒置；日度与月度样本/标签合计、日度流动性中位数均可从 308,000 行 `pool_selected.csv` 完整复算。
- 选股名单相邻两个有效日平均约 **16.54** 只新增成员（占 800 只的 2.07%）；这只是**股票池成员更替率**，不是投资组合实际换手率或成本。
- **重要限制**：`--min-amount=0` 未加独立成交额门槛，仅靠正成交额数据有效性 + 20 日中位额 Top-N 排名；`blocked_known_risk_event_rows=0` 只反映已审核的极少数公告样本未拦截入池行，绝不代表全市场风险为零。002731 在该股票池仅出现于 2025-06-09 至 2025-07-04，而样本风险公告在 2026 年；688496 未入该股票池。未复原全市场 ST、除权、停牌与历史交易可执行性。
- **核验边界**：该结论基于用户上传的产物与源码审计，**不是**重新运行 PostgreSQL SQL / 完整单测，也不等于证明未来泄漏完全不存在或模型有可执行盈利能力。

### 已找到的 P4 Ridge 真实研究输出（旧版 MVP，非交易收益）

Google Drive `agu/alphagpt/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json` 已有运行记录（2026-11-14 上传，与新增的 P4.1 独立审计代码无关）。历史股票池覆盖 **385 个信号日期、2,699 只曾入池的不同股票、308,000 条入池样本、307,411 条有效标签**；最后重训使用 **241,095** 条有标签样本，独立测试集 **61,521** 条、**77** 个交易日期。测试起始日期为 **2026-06-12**，模型拟合所用的最后标签日期为 **2026-06-11**，元数据记录了时间净化规则。

| 样本外重叠五日开盘到开盘标签指标 | Ridge V1 | 简单反转因子 `NEG(close_ma20_dev)` |
| --- | ---: | ---: |
| 平均 Rank IC | -0.02138 | +0.08202 |
| Top-20 正收益比例 | 36.04% | 40.06% |
| Top-20 平均未来收益标签 | -2.20% | -0.88% |
| Top-20 超出同期股票池均值 | -0.51 个百分点 | +0.81 个百分点 |

**解释**：以上是重叠的未来收益标签上的排序诊断，不是可复利/可成交的策略净值。元数据本身也无法证明因果逻辑与统计显著性完全可靠。**Ridge 样本外不优于简单反转因子**；在完成因子、标签及时间稳定性诊断之前，不应把它当作强势 Transformer 基线，更不能用测试集反复调参。该次运行已知公告拦截样本数为 0，并不表示历史 ST 风险完整核查。

### P4.3-B 已完成：Ridge 样本外事后诊断（2026-11-14）

已读取 Google Drive `agu/alphagpt/ashare_p4_mvp/2025-03-01_2026-09-30/` 下的
`test_predictions.csv` 和 `model_metadata.json`，对 61,521 条预测、77 个交易日执行
`ashare/p4_diagnostics.py` 诊断，且新增 4 项合成单测在本地通过：

- **预测显著偏乐观**：94.66% 的模型预测收益为正，但实际五交易日标签为正的仅 40.13%；全样本预测均值 +0.3526%，实际标签均值 -1.6861%。
- **排序结果不稳定**：测试集平均 Rank IC -0.0214，6～9 月各月均为负；每日 Top-20 名单相对上一日平均更换 88.62% 的股票。**这是名单变化率，不是实际成交换手率**。
- **高分组不占优**：当日预测得分最高十分位（decile 0）未来标签为正的比例 36.79%，而 decile 7 为 43.81%，提示模型排序/校准存在问题。
- **信息隔离**：这是对已经看过的 2026 年测试集的**事后诊断**。该窗口不能再为新模型选择特征、参数或门槛并声称“全新样本外”；下一模型必须预留更晚未见时期或预先约定 walk-forward 的评价规则。

命令仅读取本地保存的模型预测和元数据，不连接数据库、不训练模型，也不下单。
输出 `diagnostics.json`、`daily_diagnostics.csv`、
`monthly_diagnostics.csv` 和 `score_deciles.csv`。
详见 [ashare/P4_MVP.md](ashare/P4_MVP.md)。这些重叠标签是研究诊断，非策略净值。

### P4.3-B 上传诊断产物交叉核验（2026-11-15）

用户已上传 Ubuntu 生成的四份结果到 Google Drive `agu/alphagpt/ashare_p4_diagnostics/`：
`diagnostics.json`、`daily_diagnostics.csv`、`monthly_diagnostics.csv`、`score_deciles.csv`。
核验时再次读取旧版 Ridge 保存的 **61,521 条** `test_predictions.csv` 和 `model_metadata.json`；从原始保存预测**独立重新计算**日度指标、分数十分位、月度聚合以及总体指标：

- 文件齐全：77 条日度统计、4 条月度统计、10 个十分位；每日 Top-20 为 20 只，十分位行数累计 61,521。
- 日度逐字段核对 **847 项、0 项不一致**；全部月度均值与分组统计吻合，十分位总数及各组均值吻合；JSON 内旧测试模型与简单基线指标同原始 `model_metadata.json` 一致。
- 保存的预测无空值、非有限值、重复的股票/信号日期；测试窗口 2026-06-12 ～ 2026-09-30。
- Ridge 日均 Rank IC **-0.0213755**；Top-20 五日未来标签收益均值 **-2.2019%**，同期池内日均 **-1.6884%**，超额 **-0.5134 个百分点**，**不是实盘净收益**。
- Ridge 预测为正的样本占 **94.6636%**，真实未来五交易日标签为正占 **40.1326%**；预测均值 **+0.3526%** 对比实际标签均值 **-1.6861%**，平均预测偏差 **+2.0387 个百分点**。
- 前一交易日到下一交易日的 Top-20 名单平均更替 **88.6184%**，只是信号名单变更，**不是实际成交换手**。
- **负 IC 的不确定性**：77 日中 40 日 IC 为负、37 日为正，日度标准差约 **0.2644**；由于五日标签相互重叠，不能据此直接宣称统计显著，也不应据此直接反向交易。

**验收边界**：这是“保存预测 → 诊断代码 → 四份文件”的输出一致性验证；并未从生产数据库重构真实行情、证明全部历史时点因果正确或核验佣金/公司行为。2026-06 至 2026-09 测试窗口已被多次查看，后续改模型不得继续用它选参数后称为新样本外验证。优先对训练/验证时期的因子、目标分布和不复权公司行为进行诊断，并为新模型保留新的未来留出期。

### P4.3-C 训练/验证期因子及标签审计（代码已提交，2026-11-15）

新增 `ashare/p4_factor_audit.py`、`tests/test_p4_factor_audit.py`：复用 P4 原有只读 `public.market WHERE not_billing=1` 加载、过去 20 日流动性 Top-800、至少 40 个连续有效交易日及未来开盘收益标签构造，严格沿用 **60%/20%/20% 日期切分与标签结束交易日的跨区间净化**。审计程序仅针对**训练期和验证期**计算十个量价因子的每日 Spearman Rank IC、均值/中位数/波动、训练集决定的因子方向在验证集的表现；另统计标签分位数、正收益标签比例、大幅异常未来收益候选。**不在审计统计中读取 2026-06-12 起已经看过的测试集标签**；不能把原先 Ridge 的测试集再作为“全新样本外”进行因子筛选。

**重要数据约束**：先前 P4.1 的 `pool_selected.csv` 只含股票、日期、流动性、价格和标签，缺少十个特征及 `label_end_session`；因此本轮审计入口需要重新只读查询行情、重建相同股票池，不能直接拿该 CSV 计算因子 IC。支持用原 Ridge `model_metadata.json` 做一致性锁定，若验证开始日、测试开始日、重训样本数或股票池配置不一致则报错停止。默认只加载已验证的少量历史风险公告；不把当前 ST 倒灌到历史。

Ubuntu 命令：

```bash
python -m unittest discover -s tests -p 'test_p4_factor_audit.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
python -u -m ashare.p4_factor_audit \
  --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --out reports/ashare_p4_factor_audit
```

输出 `reports/ashare_p4_factor_audit/2025-03-01_2026-09-30/` 下的 `factor_audit.json`、`factor_ic_daily.csv`、`factor_ic_summary.csv`、`label_quality.csv`、`label_outliers.csv`。上榜异常只是**需要复核的样本**，不是已确认除权或公司行为；五日收益标签也不是可执行交易净利润。详见 [P4 研究说明](ashare/P4_MVP.md)。

**验收状态**：新增源码及防泄漏单测已提交；本轮尚未在用户 Ubuntu 执行测试或 PostgreSQL 实际审计，**不得声称训练/验证期 Rank IC 已核验**。后续先检验这些结果，再决定是否开发 Logistic/LightGBM，最后预留全新未来时期评价。

**2026-11-15 单测回归修复**：用户 Ubuntu 首轮运行 `tests/test_p4_factor_audit.py` 共 6 项，5 项通过、`test_holdout_targets_never_read` 失败；原因是测试期注入 `NaN` 后 pandas 将全列 `label_end_session` 从 `int64` 提升到 `float64`，即使训练/验证记录没有变化，也导致 `assert_frame_equal` 的 dtype 检查不通过。现已在训练/验证数据切片通过合法性校验后统一转换为可空整数 `Int64`，并强化对应测试对输出 dtype 的断言。此修复不读取或改变测试期收益标签，不修改数据库；尚待用户 Ubuntu 重新运行确认通过。

### P4.3-C 真实数据库输出核验（2026-11-15）

用户已在 Ubuntu 上完成 `python -u -m ashare.p4_factor_audit` 只读审计，并将 **5 份**输出上传到 Google Drive：
[`agu/alphagpt/ashare_p4_factor_audit/2025-03-01_2026-09-30/`](https://drive.google.com/drive/folders/1G5gFhuyMqh3TUZ0UNKuQQ-KUFbBpSzvK)。
已直接读取 `factor_audit.json`、`factor_ic_daily.csv`、`factor_ic_summary.csv`、`label_quality.csv`、`label_outliers.csv`，并与旧 Ridge 的 `model_metadata.json` 对照，用独立 pandas 代码复算汇总指标。

**工程/文件一致性通过**：训练 179,621 行、225 个有效信号日期；验证 56,686 行、71 个有效日期；10 个因子累计 2,960 条逐日 IC、20 行汇总；每日 10 因子无重复、全部 IC 有限且落在 [-1,1]。IC 均值/中位数/标准差、有效天数、正 IC 天数占比及训练集选定方向与日度文件逐项吻合，最大浮点差约 `9.72e-17`。股票池 308,000 行、385 个有效选股日、2,699 只不同股票；日期切分/配置与 Ridge 元数据一致，`reference_metadata_match=true`、`test_labels_used_in_audit=false`。训练标签截至 2026-02-10，验证始于 2026-02-11；验证标签截至 2026-06-11，旧测试始于 2026-06-12。已排除 61,600 条测试**信号行**；旧测试 61,521 条有标签行，两者差 79 条属于标签有效性筛选，并非漏测。原 Ridge 最终重训样本 241,095 条也与此处对账一致。最初 train+val 独立指标样本合计 236,307 条，少于重训样本 4,788 条，因为训练/验证边界两侧的标签净化在重训时不再需要保留该内部隔离；**不能误报为样本丢失**。

**十因子逐期 Rank IC**（先列训练，后列验证；未经多重比较或重叠标签时间序列显著性检验）：

| 因子 | 训练平均 IC | 验证平均 IC |
| --- | ---: | ---: |
| `volatility_20` | -0.09758 | -0.01289 |
| `range_intraday` | -0.09081 | -0.00376 |
| `close_ma20_dev` | -0.07936 | +0.00754 |
| `ret_10` | -0.06409 | +0.00264 |
| `ret_5` | -0.04277 | +0.00174 |
| `amount_ratio_20` | -0.03787 | -0.00275 |
| `return_intraday` | -0.03352 | -0.00410 |
| `volume_ratio_20` | -0.03035 | -0.00809 |
| `ret_1` | -0.02488 | +0.01622 |
| `log_liquidity_20` | -0.00569 | +0.02574 |

**关键结论**：训练期 10/10 原始因子 IC 均为负，验证期 **5/10 反向**，其余五项的负 IC 幅度也显著衰减。训练最强的波动率、日内振幅、MA20 偏离在验证期几乎丧失排序效应；2026 年 4 月部分因子月均 IC 从负转正，表明**时变性/市场状态改变是重要风险**，但不能由本次审计直接断言其因果机制。不要仅凭已经查看过的测试期简单反转因子表现重新筛选因子。

**标签分布**：训练期标签均值 +0.4072%、中位数 -0.2956%、正收益率 47.40%；验证期均值 +0.0120%、中位数 -0.9205%、正收益率 44.85%；绝对值超过 10% 的比例从 15.07% 升至 23.36%，超过 20% 从 3.13% 升至 4.50%（原始记录 **5,624 / 2,552**）。五日收益标签明显偏态、长尾且期间发生分布变化，值得比较仅使用训练期确定阈值的稳健目标函数；不复权公司行为仍需按历史公告/除权日期查证。

`label_outliers.csv` 只保存**每个集合绝对涨跌最大的前 200 条**，共 400 条，已核对排序、阈值、分区日期、`label_end_session-session_id=6` 和唯一性。不是全部 8,176 条绝对变动超过 20% 的记录；反复出现的同一股票常涉及重叠的五日标签，**不是 400 个独立公司行为案例**，也不是已确认错价/除权。不能用不复权未来收益当作可执行交易利润。

**验收边界与下一步**：这是审计产物之间及与旧模型元数据的**数据一致性核验通过**，并非重新访问 PostgreSQL 原始数据、验证完整时点风险历史、排除全部公司行为或证明模型有正向预测能力。建议先核查大幅标签的公司行为/交易日价格源，再仅在训练/验证数据上设计稳健收益标签与保守轻量排序基线；保留未来全新时期或事先约定的 walk-forward 做独立评价。不应立即投入 Transformer GPU 训练。

### P4.3-D 异常标签源价格复核与稳健轻量基线（2026-11-15，已提交待真实运行）

新增源码 `ashare/p4_price_trace.py` 与 `ashare/p4_light_baselines.py`，相应单元测试为 `tests/test_p4_price_trace.py`、`tests/test_p4_light_baselines.py`。该阶段是基于 P4.3-C 已核验的训练/验证期因子时变性与标签长尾问题开发的**研究诊断工具**，不是已训练验证完成的新交易策略，未对旧测试期选参，也未启动 Transformer。

**D1：异常标签价格复核**。从 P4.3-C 的 `label_outliers.csv` 分别选择训练、验证期绝对收益最大的最多 30 条（均可调），只读查询 `public.market WHERE not_billing=1` 的对应股票及正式市场日历。逐一从 T+1 至 T+6 的真实交易日**重新计算开盘到开盘收益**，检查缺失/无效日线、源价格与原标签是否一致，并记录连续开盘价跳变（默认 15% 阈值）。**历史行情只读取已查看测试期开始前的日期**；价格跳变仅表示需要进一步查实，**不自动判定送转/分红/复权错误**。保存 `price_trace_summary.json`、`price_trace_cases.csv`、`price_trace_bars.csv`。

**D2：同一验证样本上的七种轻量比较**。保持旧 P4 动态 Top-800、时间切分、标签净化及相同的验证集，比较固定 MA20 反转、训练期最强单因子、训练期选定方向的等权因子秩组合、原始 Ridge（固定 alpha 1.0）、仅训练期绝对标签 95%/99% 分位截尾的 Ridge、按训练期当日收益百分位排名拟合的 Ridge。**所有截尾阈值、方向、模型权重只从训练集产生**；验证表现仅用于探索诊断、不会自动决定新模型优胜者，旧测试集不参与。原 Ridge 元数据仅锁定股票池配置、训练/验证分界及样本数，**不是每行行情快照相同的密码学证明**。保存 `baseline_comparison.json`、`baseline_validation_summary.csv`、`baseline_validation_daily.csv`。

Ubuntu 测试与运行命令如下（仅本地 CSV 输出、只读数据库）：

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git switch ashare-dev
git pull --ff-only origin ashare-dev
python -m unittest discover -s tests -p 'test_p4_price_trace.py' -v
python -m unittest discover -s tests -p 'test_p4_light_baselines.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v

python -u -m ashare.p4_price_trace \
  --outliers reports/ashare_p4_factor_audit/2025-03-01_2026-09-30/label_outliers.csv \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --limit-per-split 30 --jump-threshold 0.15 \
  --out reports/ashare_p4_price_trace

python -u -m ashare.p4_light_baselines \
  --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --alpha 1.0 --top-k 20 \
  --reference-metadata reports/ashare_p4_mvp/2025-03-01_2026-09-30/model_metadata.json \
  --out reports/ashare_p4_light_baselines
```

**验收状态**：已向 `ashare-dev` 提交源码、测试与文档，已做本地 SQLAlchemy/SQLite 与 pandas 联合日期索引的兼容性冒烟检查；本轮**尚未在用户 Ubuntu 上执行完整新增测试或生产 PostgreSQL 真实数据运行**，因此不得声称新基线改善了 IC、标签跳变已归因或价格差异不存在。详细限制与六份输出见 [P4_MVP.md](ashare/P4_MVP.md)。

**单测修复（2026-11-15）**：用户 Ubuntu 执行完整单测共 86 项，仅 `test_p4_price_trace.test_detects_mismatch_and_large_open_jump_without_classifying_action` 因警告文案缺少 `NOT proof` 导致断言失败，算法及数量断言未报错。已在 `ashare/p4_price_trace.py` 明确写出“开盘价跳变不是公司行为证据”，使源码告警与既有单测契约一致；需用户 `git pull` 后重测，不可提前宣称 86 项全通过。SQLite 的 Python 3.12 `DeprecationWarning` 非致命。

### P4.3-D 真实产物核验：复现通过，但发现公司行为标签污染（2026-11-15）

用户在 Ubuntu 运行 `ashare.p4_price_trace` / `ashare.p4_light_baselines`，并把 6 份输出上传至 Google Drive：
- [`agu/alphagpt/ashare_p4_price_trace/`](https://drive.google.com/drive/folders/17x-joE0RFwItS-reCt9fgdwcRm7m7znz)：`price_trace_summary.json`、`price_trace_cases.csv`、`price_trace_bars.csv`；
- [`agu/alphagpt/ashare_p4_light_baselines/2025-03-01_2026-09-30/`](https://drive.google.com/drive/folders/1bYJ0yWjK7txXK2IJRczQFGuG-aolA9aZ)：`baseline_comparison.json`、`baseline_validation_summary.csv`、`baseline_validation_daily.csv`。

**文件间与保存产物对账通过，尚非模型/行情源质量验收**：
- 价格追溯抽样为训练、验证期各 30 条，共 **60 条研究标签、360 条正式日线路径**。与 P4.3-C 的 `label_outliers.csv` 排名前 30 条/分组全部匹配，开盘到开盘收益、最大相邻开盘跳变、跳变日期与 JSON 状态能逐条独立复算，**60/60 标签与数据库读回值一致**，无缺失路径、无重算不符。**58/60** 条信号窗口出现 >15% 的相邻开盘价变化，涉及 **38 只股票、41 个股票/跳变日期组合**，有重复覆盖的事件，不能认为是 58 个独立公司行为案例。
- **已由公开发行人公告确认的真实公司行为反例：002594 比亚迪**。`trade_date=2025-07-25` 的信号在 2025-07-28 以开盘价 ¥337.93 为入场参考，2025-08-04 以 ¥104.00 为退出参考，不复权标签 **-69.2244%**；2025-07-29 相邻开盘价 **-66.8541%**。比亚迪 2025-07-21 发布、2025-07-22 登载的《2024年度利润分配及资本公积金转增股本实施公告》（证券代码 002594、公告编号 2025-047）明确规定 **2025-07-29 除权除息，每 10 股送 8、转增 12、派发现金 ¥39.74**：[公开公告原文](https://epaper.stcn.com/pic/202507/22/37e227d2c542dd4204dafd9319859879.pdf)。每 1 股送转后对应 3 股；以期末股价和税前现金红利粗算，`(3×104.00 + 3.974)/337.93 − 1 ≈ −6.50%`，与原始 `104.00/337.93 − 1 ≈ −69.22%` 差异巨大。该简化经济收益并非交易 PnL；**但已经证实当前不复权的监督目标至少在此类样本中错误**。同一除权事件在抽样中影响 002594 的 **5 条相邻信号**，不得误算为 5 次独立大跌。
- 七种轻量基线在相同的 **71 个验证交易日、56,686 条有效标签样本**上输出 497 条日度结果；逐候选复算 IC/Top20/股票池标签指标与 CSV 一致。固定 `alpha=1.0` 的原 Ridge 验证 Rank IC **-0.0111672834**，与旧 `model_metadata.json` 完全一致（差 0）。元数据仍明确 `per_row_market_snapshot_identity_verified=false`，所以不是数据库逐行快照一致性的证明。
- **七个方案的 Top-20 五日未来标签相对股票池均值全部为负**（单位：百分点）：`neg_ma20_fixed -0.6864`、`train_best_single_factor -0.2155`、`train_signed_rank_ensemble -0.7102`、`ridge_original -0.3623`、`ridge_clip_abs_train_q95 -0.5577`、`ridge_clip_abs_train_q99 -0.3252`、`ridge_daily_rank_target -0.7340`。后两类中的部分方案 IC 小幅为正：最强单因子 **+0.01289**、日内排名目标 Ridge **+0.00529**，但没有稳定正的 Top-20 超额证据，更不能据旧验证集挑选模型后宣称新的样本外胜出。标签重叠、缺失公司行为、手续费与实际成交尚未处理。

**工程判断**：P4.3-D 的**产物一致性核验通过**，但发现**正式标签语义存在已确认的重大缺陷**。应把“全市场历史送转/分红/拆并股/停复牌的时点数据与总回报标签重建”设为 **P4.3-E 前置阻断任务**，处理与公司行为跨越有关的历史因子窗口及标签，并追溯源价格。不能直接把 `abs(label) > 0.2` 的交易全部删除——真实行情可能确实大涨大跌；应依据可核实公告、交易日和有效时间校正或标记。T 日特征只能用截至 T 已经可见的信息，未来公司行为只能用于**监督标签**不能偷渡到特征。完成全市场再训练/新窗口留出之前，不启动 Transformer，也不发表可执行盈利结论。

**验收边界**：此轮是六份文件与前次输出、历史公司公告之间的核对；**没有重新连接生产数据库取数，也没有取得完整全市场复权/历史风险记录**。用户上传输出意味着命令已执行，但没有据此宣称最新完整 86 项单测均已通过。

### P4.3-E 已提交：公司行为经济收益标签与严格覆盖闸门（2026-11-15）

针对 P4.3-D 已证实的 002594 比亚迪送转除权标签失真，新增：

- `ashare/p4_corporate_actions.py`：严格的只读审计 CLI，基于实际交易日复算旧未复权收益标签，对有经核实权益分派的持有窗口计算**税前简化经济收益**，并给出完整覆盖与特征污染标记。支持多事件、股份扩张/缩并、现金权益不复投；只使用旧测试边界前的 `public.market WHERE not_billing=1`。
- `tests/test_p4_corporate_actions.py`：7 项合成测试，覆盖 T+1 除权入场不能享受此前红利、跨多次公司行为累计、无覆盖拒训、源价格不一致隔离、除权污染过去因子与禁止读取旧测试期。
- `ashare/data/corporate_actions_verified_examples.csv`：唯一已列证据是比亚迪 2025-07-29 送转（旧每股变为 3 股）及税前每旧股现金 3.974 元；来源为[深交所披露的比亚迪 2025-047 实施公告](https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-07-22/f30e6169-4779-487e-acff-8c1eff230fa4.PDF)。
- `ashare/data/corporate_action_coverage_examples.csv`：**有意只有表头，没有任何 `COMPLETE_VERIFIED` 覆盖证明**。单份公告不能证实整个时间段不存在其他公司行为；本轮因此**预期训练允许样本为 0**，不允许静默把未发现事件当成无事件。

更多算法定义、风险边界、命令及验收文件见 [P4.3-E 独立说明](ashare/P4_3_E_CORPORATE_ACTIONS.md)。

**实施状态**：源码和单测提交到 `ashare-dev`；**本地新增 7 项单测通过，且 GitHub 源码与本地通过版本的 Git blob 哈希完全一致**。尚未由用户 Ubuntu 执行全仓回归或 P4.3-E PostgreSQL 实测。本阶段只解决“如何记录、估算和隔离”，**尚未获取全市场公司行为完整来源**，不能据此开始复权后模型训练或宣称盈利。

Ubuntu 命令：

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
  --out reports/ashare_p4_corporate_actions
```

输出 `reports/ashare_p4_corporate_actions/2025-03-01_2026-06-11/`：`corporate_action_gate.json`、`corporate_action_daily.csv`、`corporate_action_impacted_cases.csv`。程序只查询早于旧测试期 2026-06-12 的行情、重新构建其时点动态 Top-800，并严格对账当时重训有效样本数 **241,095**；发生不一致则停止、不产生伪造结果。报告不是训练数据集。

### P4 当前已实现的研究契约

当前 `970151c` 提交已经包含 P4 MVP，而非需要从零开发：

1. 从 `market WHERE not_billing=1` 只读加载日线。以交易日 T 及之前的 20 日成交额中位数排序，默认动态 Top 800、最低收盘价 3、至少 40 个连续有效交易日；不使用**当前** `stock_info.st` 回填历史。
2. 训练特征为 10 个过去窗口量价统计（`ret_1`、`ret_5`、`ret_10`、`close_ma20_dev`、`volume_ratio_20`、`amount_ratio_20`、`range_intraday`、`return_intraday`、`volatility_20`、`log_liquidity_20`）；暂不纳入不可验证的主力资金历史值。
3. 标签默认 `T+1 open → T+6 open` 的 5 个持有交易日开盘到开盘变化；未来行情**仅用于标签**，不用于 T 日特征/股票池；无连续真实交易日数据时标签缺失。
4. 依据 T 日期做 60%/20%/20% 训练、验证、测试；净化跨集合的未来标签；Ridge 仅用验证集选 `alpha`，最终报告独立测试集 Rank IC、Top-K 与费用粗代理，**不是可执行 NAV**。
5. 训练输出 `ridge_model.npz`、`model_metadata.json`、`test_predictions.csv`、`latest_predictions.csv`、`pool_daily_coverage.csv`；独立 `ashare.p4_predict` 按同一特征/股票池口径读取模型并给出候选 CSV，可设置放弃预测阈值。

### Ubuntu 运行与测试

从仓库根目录运行，使用已有**只读** PostgreSQL 凭证：

```bash
cd ~/gcdev/alphagpt/AlphaGPT
conda activate py312
git fetch origin
git switch ashare-dev
git pull --ff-only origin ashare-dev
python --version
python -m unittest discover -s tests -p 'test_*.py' -v
```

P1/P2：因子探索与数据质量（仅读取数据库，结果写本地文件）：

```bash
python -u -m ashare.research --start 2025-01-01 --end 2026-09-30 \
  --samples 200 --min-stocks 15 --log-every 10 --db-chunk-size 50000
python -u -m ashare.quality_report --start 2025-01-01 --end 2026-09-30 \
  --db-chunk-size 50000
```

P3 诊断回测（包含 NON-PIT 黑名单，不可作为正式绩效）：

```bash
python -u -m ashare.backtest_run --start 2025-03-01 --end 2026-09-30 \
  --exploratory --exclude-current-st \
  --risk-events ashare/data/risk_events_verified_examples.csv \
  --candidate-pool 20 --log-every-days 20 \
  --out reports/ashare_backtest_current_st_plus_events
```

P4.1 股票池独立审计（不训练模型、不写数据库；输出 `pool_summary.json`、`pool_daily_coverage.csv`、`pool_monthly_coverage.csv`、`pool_selected.csv`）：

```bash
python -u -m ashare.p4_pool_report --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --out reports/ashare_p4_pool
```

拒绝原因在 `rejected_by_reason` 中按顺序分配**第一项不合格原因**（互斥、总量守恒）；筛选后因尚无未来连续完整标签而不可训练的记录另以 `selected_without_valid_label` 统计。日/月覆盖会保留无候选股票的交易日，但不推断缺失的全市场交易日历。

**2026-11-14 回归修复**：修复 `outside_daily_top_n` 在 Top-N 截断后错误计算（恒为 0）的问题，改为在截断前记录候选条数，再计算差值。该修复针对 Ubuntu 用户报告的 `test_rejections_partition_requested_window_and_monthly_totals` 断言失败；**提交已完成，须在用户 Ubuntu 上重新执行单测确认结果**。

P4 Ridge MVP：用**同一**股票池规则先训练/测试，再单独推理：

```bash
python -u -m ashare.p4_train --start 2025-03-01 --end 2026-09-30 \
  --top-n 800 --min-price 3 --consecutive 40 --hold-sessions 5 \
  --top-k 20 --roundtrip-cost-bps 25 --out reports/ashare_p4_mvp

python -u -m ashare.p4_predict \
  --model-dir reports/ashare_p4_mvp/2025-03-01_2026-09-30 \
  --asof 2026-09-30 --top-k 50 \
  --out reports/ashare_p4_predict
```

P4.3-B 已保存测试预测的事后诊断（不需要数据库）：

```bash
python -m unittest discover -s tests -p 'test_p4_diagnostics.py' -v
python -u -m ashare.p4_diagnostics \
  --model-dir reports/ashare_p4_mvp/2025-03-01_2026-09-30 \
  --top-k 20 --out reports/ashare_p4_diagnostics
```

上述是仓库**现有入口与建议运行命令**。Google Drive 已有先前 P4 Ridge 的真实数据库训练输出，且新增 P4.1 审计 CLI 已由用户在 Ubuntu 对真实 PostgreSQL 执行，上传结果经交叉核验；**新增单元测试修复后是否全部通过，仍需依据 Ubuntu 测试日志确认**。重跑后应以最新 `model_metadata.json` 和审计文件为准，不得把旧指标当作新实现的验证结果。

### 当前问题与后续优先顺序

- **P4.1 继续验收**：真实审计的数量对账和日/月分布已通过；下一步独立审计行情源的除权/异常变动、流动性金额单位、样本时点可用性与部分风险事件未拦截的原因。不要使用“回测期末仍存活”作为历史入池条件。
- **P4.2/P4.3 验收**：已有 Ridge 真实输出，样本外 Rank IC 为负且明显弱于简单因子；优先审计数据/标签和模型目标，实跑新增单元测试与独立推理，核验边界日期及训练/推理一致性，再补充模型覆盖率、高置信度准确率、方向准确率、Top-K 相对表现及换手/费用。不要在看过测试集后反复选参；新的模型必须使用新时间窗口或独立留出样本验证。
- **成交/数据限制**：主力资金分数历史时点不可靠；当前 ST 历史倒灌不可用于训练；不复权价格中的公司行为、停复牌与开盘成交概率仍未完整建模。相关治理不阻塞低风险的 P4 MVP，但限制可宣称结论。
- **P4.4**：在 Ridge 真正跑通之后，以**相同样本/时序切分/评价口径**实现 PyTorch 小型 Transformer，CUDA 3090、训练过程日志、checkpoint/恢复、批量 CSV 推理。先做可运行 MVP，再考虑复杂架构。
- **开发纪律**：仅向 `ashare-dev` 提交，每个阶段配单测，并同步更新此章节、最近结果与提交号；禁止自动改动 `main` 或写入生产行情表。

**最近核查日期**：2026-11-15。P4.3-D 的异常价格追溯和七候选轻量比较**真实运行输出已在此前完成对账**，证实原不复权标签存在已公告的比亚迪送转事件污染；没有可宣称的新模型胜者。P4.3-E 新增严格公司行为质量闸门，7 项本地合成单测通过、等待 Ubuntu 实测；当前没有全市场已验证的完整公司行为覆盖证明。本轮新增 `ashare/p4_diagnostics.py` / `tests/test_p4_diagnostics.py`，已在本地运行 4 项新增测试与真实旧测试预测的事后诊断；完整 Ubuntu/真实数据库重跑仍须用户执行。P4.1 增加 `ashare/p4_pool_report.py` 与 `tests/test_p4_pool_report.py`，并对 `ashare/p4_mvp.py` 增加拒绝原因统计。**代码已提交，用户已在 Ubuntu 执行真实市场池审计并上传四份文件；本轮独立核验文件内容，但没有直接连接生产 PostgreSQL 重跑 SQL 或重新运行单元测试**。启动本次工作时 `ashare-dev` HEAD 为 [`970151c0d947984a1cb6919d01bfe59b59b7c61d`](https://github.com/cailming/AlphaGPT/commit/970151c0d947984a1cb6919d01bfe59b59b7c61d)（P4 股票池、缺口安全标签、Ridge OOS MVP）；前一个风险事件核查提交为 [`9febe7fc2a3282108096091c5de11d58e4a8c547`](https://github.com/cailming/AlphaGPT/commit/9febe7fc2a3282108096091c5de11d58e4a8c547)。本轮 P4.1 审计相关提交：[`cb9c059`](https://github.com/cailming/AlphaGPT/commit/cb9c059020342a4115ae6df4663b8b9ba67928a7)、[`59b7c01`](https://github.com/cailming/AlphaGPT/commit/59b7c012db1828985e9ac640c7fb645f16f5b6f6)、[`7511ba6`](https://github.com/cailming/AlphaGPT/commit/7511ba6b148054c8d83ed2ed22f1cd8e129dee99)。后续以分支最新 HEAD 为准。
