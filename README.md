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

<!-- p4-e2-board:start -->
### P4.3-E2 自动审计任务看板

| 任务 ID | 优先级 | 任务与验收条件 | 状态 | 下一步 |
|---|---|---|---|---|
| E2-AUTO-001 | P0 | 验证冻结快照、旧批次 SHA256 和测试基线 | DONE | 复用 239 个成功请求 |
| E2-AUTO-002 | P0 | 实现调度、重试、恢复、去重、审计与离线测试 | DONE | 已复核续跑与证据完整性 |
| E2-AUTO-003 | P1 | 完成索引 10–109 的 100 股试点，队列有明确结论且无 P0 和在途熔断 | DONE | 试点已验收，允许扩大范围 |
| E2-AUTO-004 | P1 | 5,449 股全市场采集及断点恢复 | DONE | 32,694 项均有结论，待重试和未抓取均为 0 |
| E2-AUTO-005 | P1 | 全市场异常画像、定向补采和最终报告 | RUNNING | 76 个北交所事项已补公告，首批剩余 8 个精度差 |
| E2-AUTO-006 | P0 | 公司行为覆盖与训练闸门独立复核 | BLOCKED | 等待可复核的证据 |

**Codex 自动执行进度**

- **2026-11-24 09:34 北京时间。** 76 个北交所待补证事项（67 股）的实施公告全部找到并逐项核对。登记日、除权日、税前现金及适用的送转条款与巨潮账本一致；76 项新浪现金字段为零均与公告不符。46 项通过同一组织编号和公告抬头核验了历史证券代码。
- [北交所实施公告索引](ashare/data/p4_action_bse_issuer_notice_index.csv)含原文链接、分派条款及原 PDF SHA256。原文和核验材料保存在服务器 `reports/ashare_p4_action_pipeline/followup/bse-notices.VlYHW4`；合并分流输出见 `reports/ashare_p4_action_pipeline/followup/p1-conflicts-zawync_a`。
- 首批 107 条冲突归并的 103 个事项中，95 个已有发行人公告核对，另 8 个仅分流为低优先级精度差。71 项离线测试通过。单项公告证据不等于历史覆盖认证，训练闸门仍为 `CLOSED`。

- **2026-11-24 09:20 北京时间。** 107 条日期或金额冲突按股票与除权日归并为 103 个事项，涉及 90 股。首轮核对 19 个发行人公告事项，包括全部 10 个同日多笔和全部 4 个其他条款差异；另有 8 项仅因数值与显示精度相容而降低优先级，未认证条款。
- 余下 76 个事项涉及 67 股，均属北交所新浪现金字段为零、巨潮有正值且股本权益一致的模式。已通过两家发行人的三次公告确认样本；其余逐项保留在补证队列。没有用来源多数或模式推测替代公告。
- 本轮输出 `reports/ashare_p4_action_pipeline/followup/p1-conflicts-z506t6v5`，工件 SHA256 已回验。67 项离线测试通过，未修改原审计、账本、覆盖认证或训练闸门。

- **2026-11-24 08:58 北京时间复核。** 全市场采集已于 2026-11-19 12:07:36 北京时间收尾，checkpoint 为 `DONE`；巨潮和最终审计均以退出码 0 结束。08:58 检查无采集进程、写锁空闲，无熔断或暂停来源。
- 冻结范围 5,449 股全部扫描，4,775 股所有请求成功。32,694 个任务中 31,353 成功、1,341 个终态异常，未抓取和待重试均为 0。巨潮 5,058 成功、391 个终态异常，其中 45 项为明确空响应、覆盖未知。
- 最终审计及异常画像已生成：`reports/ashare_p4_action_pipeline/FINAL_FULL_AUDIT_REPORT.md`，治理队列目录 `reports/ashare_p4_action_pipeline/governance/phase3-q7ipkcru`。P0/P1/P2 为 0/587/14,129；日期或金额冲突类异常 107 条。定向复核和补证尚待处理。
- 已通过旧测试期隔离校验。历史公司行为覆盖未认证，训练闸门 `CLOSED`。

- **2026-11-19 11:27 北京时间核验。** 修复 `61d07d3` 已部署到 `ubuntu92`。巨潮仅在 HTTP/业务状态均为 200、明确空列表与两个零计数一致时记为 `CNINFO_EMPTY_UNVERIFIED`：终态覆盖缺口，不视为无分红、成功或熔断错误。限流、超时和结构异常仍按原规则处理。
- 本地及 Ubuntu 各 64 项离线测试通过。`688173`、`688176`、`688177` 的原始空响应完成离线回放和 SHA256 核验，追加到原账本，保留历史异常。按证据恢复巨潮，启动时只剩 719 个未抓取任务。
- 巨潮单进程运行，首批已核验 3 个成功请求、1 个明确空表，空表未触发熔断。日志为 `reports/ashare_p4_action_pipeline/phase3-cninfo-empty-verified.plveiD.log`。全量达到验收条件后自动生成最终审计，训练闸门 `CLOSED`。

- **2026-11-19 10:01 北京时间核验。** `ubuntu92` 无采集或恢复进程，写锁空闲，只读 `status` 校验通过。昨晚 21:10:54 的检查点为 `BLOCKED`：巨潮在近 5 个不同任务中有 3 个返回 `UPSTREAM_EMPTY_OR_SCHEMA`，最后一个为 `688177`。空响应与解析问题尚未区分。
- 5,449 股均已有结果。全市场 32,694 个任务中，30,676 成功、1,298 个终态异常、1 个待重试、719 个未抓取。剩余 720 个均为巨潮；巨潮累计 4,381 成功、348 个终态异常。BaoStock 和新浪的待重试与未抓取均为 0。
- 阶段审计已生成，P0/P1/P2 为 0/508/13,187。最终审计待全量采集结束。训练闸门 `CLOSED`。累计审计为 `reports/ashare_p4_action_pipeline/audits/audit-otwwbm1w`。

- **2026-11-18 18:03 北京时间核验。** BaoStock、新浪分红和新浪配股均已清空未抓取和重试队列。5,449 股均已有至少一项结果。新浪 17:28 完成，因当时巨潮仍暂停，全流程为 `WAITING_FOR_SOURCES`，非零退出码由 Conda 显示为 `ERROR`。
- 巨潮按验证结果恢复单进程采集。`002568`、`002569`、`300750` 和 `600519` 的成功诊断证据校验后追加至原账本，无需重复请求。巨潮原熔断和诊断证据保留，恢复命令只重新启用该来源。
- 18:03:10 的累计报告显示，巨潮首批新增 25 个实际请求成功，巨潮累计 918 成功、164 个终态异常、0 个待重试、4,367 个未抓取。全市场累计 27,213 成功、1,114 个终态异常。P0 为 0，训练闸门 `CLOSED`。
- 当前日志为 `reports/ashare_p4_action_pipeline/phase3-cninfo.WETDE4.log`，对应累计审计为 `reports/ashare_p4_action_pipeline/audits/audit-bx64rkk5`。采集完成才生成全市场收尾审计。

- **2026-11-17 14:41 北京时间核验。** `ubuntu92` 上仍只有一个 BaoStock 采集进程，checkpoint 为 `RUNNING`，未发现单独的恢复进程。14:41:10 的累计审计显示已扫描 4,780/5,449 股，其中 887 股全部请求成功，669 股尚未扫描。
- 全市场 32,694 个任务中，17,303 成功、198 个终态异常、0 个待重试、15,193 个未抓取。BaoStock 已成功 14,338 个，2 个终态异常、2,007 个未抓取；新浪分红和配股均未续跑，巨潮 `002568` 仍保留 `SCHEMA_CHANGE` 熔断。
- 巨潮 `002568` 独立复检三次，前两次超时，第三次返回 13 行，其中 2 行的除权日在冻结区间内。复检证据独立归档，正式账本和熔断状态未变。
- 固定 checkpoint 后的只读 `status` 与阶段审计均通过。P0/P1/P2 为 0/224/6,994，旧测试期隔离通过，训练闸门 `CLOSED`。对应累计审计为 `reports/ashare_p4_action_pipeline/audits/audit-f1gdteuh`，阶段报告为 `reports/ashare_p4_action_pipeline/PHASE3_PROGRESS.md`。

- **2026-11-17 09:27 北京时间核验。** 运行代码 `a997b86`，同一个检查点仅启动一个 BaoStock 续跑进程。09:02 的 `000001 / 2026` 实际查询成功，返回两行均在冻结审计区间之外。续跑进程已另有 183 次成功请求证据。
- 09:27:42 的累计报告显示全范围 3,446 个任务成功、198 个终态异常、0 个待重试、29,050 个未抓取。巨潮 `002568` 的 `SCHEMA_CHANGE` 熔断保留，新浪随前次整体停机，训练闸门为 `CLOSED`。Ubuntu 61 项离线测试通过。
- BaoStock 续跑日志为 `reports/ashare_p4_action_pipeline/phase3-baostock.ZJGaP7.log`，累计审计仍见 `reports/ashare_p4_action_pipeline/latest_report.json`。

- **2026-11-16 22:13 北京时间（14:13 UTC）核验。** 运行代码 `0d8c4ae`。检查点为 `BLOCKED`，采集和恢复进程均已退出，写锁空闲。只读 `status` 与重算的阶段审计一致，最后一份累计报告时间为 21:17:18 北京时间。
- 已扫描 1,054/5,449 股，72 股全部来源任务成功。32,694 个任务中 3,271 成功、198 终态异常、1 待重试、29,224 未抓取。全量阶段新增 2,843 次尝试，其中 12 次为同键重试。
- **新浪**没有单独熔断，随采集进程停止。新浪分红 1,023 成功、31 终态异常、4,395 未抓取；新浪配股 1,053 成功、4,396 未抓取。
- **巨潮**在 21:17:02 北京时间处理 `002568` 时触发 `SCHEMA_CHANGE`。889 成功、165 终态异常、4,395 未抓取。
- **BaoStock** 因 20:22 左右的登录超时保持隔离，306 成功、2 终态异常、1 待重试、16,038 未抓取。20:55 的独立匿名登录复检仍不可用。21:17 的后台恢复程序遇到巨潮熔断后退出，恢复日志未记录新的登录探测。
- P0/P1/P2 为 0/192/1322，旧测试期隔离通过，训练闸门 `CLOSED`。累计审计目录为 `reports/ashare_p4_action_pipeline/audits/audit-4n5nvut9`，阶段报告为 `reports/ashare_p4_action_pipeline/PHASE3_PROGRESS.md`，采集和恢复日志为 `reports/ashare_p4_action_pipeline/phase3-isolated.YxUK77.log`。

- Phase 3 使用 `3f8a559` 启动全量采集，首次 BaoStock 超时证据保存在 `reports/ashare_p4_action_pipeline/phase3-collect.hYiSo2.log`。20:53 的上次进度快照见提交 `d229e32`。

- Phase 3 前置检查于 2026-11-16 20:13 北京时间（2026-11-16 12:13 UTC）通过。工作基线 `22437d0`，历史证据、快照与测试边界校验通过，磁盘可用 655 GiB。
- 全量 32,694 个任务，既有 603 个成功、36 个终态异常、0 个待重试；剩余 32,055 个任务。846 个 BaoStock 任务可本地判定为不支持，首轮预计最多 31,209 个查询任务，另计重试。
- P0 为 0，当前熔断为空。历史试点 P1 和 P2 分别为 36 和 189。以下保留已验收的 100 股试点记录。
- 2026-11-16 15:23（UTC+08:00）。Phase 0–2 完成，当前运行与审核代码 `8d2dff6`，阶段报告等待用户验收。
- 冻结范围 5,449 股。试点索引 10–109 已扫描 100/100 股，其中 71 股所有请求成功。
- 600 个有效请求任务中 565 个成功、35 个终态异常。新尝试 363 次，重启复核时新增 0 次。未抓取和待重试任务均为 0。
- 96 条合并候选。BaoStock 逐项完整匹配 2 条、部分字段未决 92 条。新浪与巨潮完整匹配 0 条、部分字段未决 94 条。
- P0 为 0，P1 为 36，无当前熔断。与旧测试期的隔离检查通过。Ubuntu `py312` 的 54 项离线测试通过，其中原有免费来源回归 31 项。

**ChatGPT / 用户任务指令区**

- E2-AUTO-003 的 100 股试点已通过技术验收。已授权扩大至冻结快照内 5,449 股，异常按证据留档和治理。训练闸门保持关闭，等待全量审计及独立核验。

**Codex 交付记录**

- Phase 0 核验冻结快照和 13 份旧批次清单，识别 5 份字节级副本。提交 `d138b6b`。
- Phase 1 交付统一调度、恢复、审计与离线测试。提交 `5fd35d8` 和 `8d2dff6`。
- Phase 2 实际完成 100 股试点。按接口与异常明细、恢复命令及报告路径见 [P4.3-E2 试点记录](ashare/P4_3_E2_AUTOMATION.md#100-股试点实测)。
- Phase 3 前置检查完成，新增日期诊断分项及全量对账索引。详见 [Phase 3 运行与审计](ashare/P4_3_E2_AUTOMATION.md#phase-3-运行与审计)。
<!-- p4-e2-board:end -->

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
| P4.3 | Ridge/轻量基线与因子审计、P4.3-D 价格核验、P4.3-E 经济收益质量闸门，以及第二阶段**免费 AkShare 默认接入**/可选 Tushare 数据覆盖审计 | **P4.3-B/C/D/E 第一阶段真实产物已核验；AkShare 无 Token 适配器与 7 项本地测试通过**；真实公开接口批量采集及历史事件覆盖仍需核验，禁止重训 |
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

### P4.3-E 真实数据库产物交叉核验（2026-11-15）

用户在 Ubuntu 完成 `ashare.p4_corporate_actions` 只读运行，并将三份产物上传到 Google Drive：
[`agu/alphagpt/ashare_p4_corporate_actions/2025-03-01_2026-06-11/`](https://drive.google.com/drive/folders/1RSqOOb5hPsJJXBzKsgcyyDZXOK58eN05)：
`corporate_action_gate.json`、`corporate_action_daily.csv`、`corporate_action_impacted_cases.csv`。

**产物及时间切分验收通过**：日度 308 个研究日（2025-03-06 至 2026-06-11），每日 Top-800，合计 **246,400** 行；有效原始标签 **241,095**、缺失 **5,305**，其中 241,095/241,095 原始标签全部由正式行情的 T+1 / T+6 开盘价复算一致。旧 `model_metadata.json` 中 `model_version=P4_RIDGE_V1`、`test_start=2026-06-12`、`refit_samples=241095` 和股票池配置与本次产物核对完全一致；并非验证了源数据库逐行快照同一性。

**与旧版股票池日度产物直接复核**：本轮全部 308 天的入池规模与旧版 `pool_daily_coverage.csv` 一致；其中 **302 天**的有效标签数完全一致；末尾 **2026-06-04 至 2026-06-11 六个交易日**因本次不读取 2026-06-12 及以后行情而全部被净化。原旧报表这六日有 **4,795** 条标签，因此旧报表截至 2026-06-11 的 245,890 条标签减去这 4,795 条，正好是本次 241,095 条。该差异是正确的跨测试边界标签隔离，**不是数据丢失**。

**公司行为事件核对**：只读取已核实的比亚迪 `BYD-2025-047` 单个事件。三份产物统计相符：**5** 条跨越 2025-07-29 除权日的标签、**40** 条此后因子 40 交易日回看窗口仍含除权跳变的记录，总计 `corporate_action_impacted_cases.csv` **45** 行（均为 002594、无股票日期重复、无两种事件计数重叠）。逐行重算原收益比值及五条按 `(3×退出开盘价+3.974)/入场开盘价−1` 的税前示例收益，浮点误差约 `1e-16`；其中信号日 `2025-07-25` 的不复权标签 **-69.2244%**，修正示例为 **-6.4972%**。这只是已经核实的公告案例，并非全市场公司行为覆盖。

**严格质量闸门行为正确**：`explicit_complete_coverage_assertions=0`、`verified_complete_coverage_rows=0`、`training_label_allowed_rows=0`，且全部 `economic_target_return` 没有发放；有效原始标签 241,095 条全部标记 `UNKNOWN_ACTION_COVERAGE`，另有 5,305 条为 `MISSING_ORIGINAL_LABEL`。当前 `universe_coverage_complete=false`、`training_is_blocked_by_incomplete_coverage=true`。这代表当前没有凭空认可其他股票的历史公司行为完整性，**应继续阻止模型重训**。

**本轮验收边界**：是 **三份真实运行产物 vs 旧 Ridge 模型元数据/旧股票池 CSV 的独立对账**；没有再次直连 PostgreSQL 核实原始条目、完成全市场复权/现金权益事件搜集，或者验证用户 Ubuntu 全量单测日志。下一步需建立、版本化并核验全市场权益事件数据和完整覆盖证明，明确研究中使用可用作监督标签的经济收益 vs T 日不能泄漏未来信息的历史因子；数据补齐后才能开启新的模型训练与真正未来未见时期验证。

### P4.3-E 第二阶段：全市场公司行为候选数据接入与覆盖审计（2026-11-15）

**源码已提交（等待用户 Ubuntu 真实 Tushare 权限及数据拉取）**：

- [`ashare/p4_action_feed.py`](ashare/p4_action_feed.py)：`snapshot → fetch → audit` 三段式流程。只读 `public.market WHERE not_billing=1` 中 `2026-06-12` 以前的历史股票代码，再与 Tushare Pro `stock_basic` 的上市/退市/暂停上市名单取并集；按股票调用 `dividend`（默认每批 50 个、请求间延迟 1.5 秒），保存原始供应商行、逐股请求状态和 SHA-256 校验的不可覆盖批次。
- [`tests/test_p4_action_feed.py`](tests/test_p4_action_feed.py)：新增 **11 项本地测试通过**，覆盖上市/退市并集、无映射股票、供应商限额与失败、旧测试期公告排除、同日条款冲突拒绝、文件篡改/重复批次拒绝及比亚迪官方事件对照。全仓 Ubuntu 测试与供应商 API 真实调用尚未由本轮执行。
- 供应商映射只认已实施的 `imp_ann_date`、`record_date`、`ex_date`、`stk_bo_rate + stk_co_rate`、`cash_div_tax`（税前）。预案、字段不全、数量冲突或达到 API 行数上限的股票均不自动通过。供应商候选严格命名 `UNVERIFIED_corporate_actions_candidates.csv`；逐股状态仅为 `NOT_FETCHED`、`FETCH_FAILED_OR_TRUNCATED` 或 `FETCHED_NOT_VERIFIED`，**不会生成 `COMPLETE_VERIFIED`**。
- 供应商 `dividend` 不覆盖配股等复杂权益变化，也不能仅靠返回空行证明“公司没有分红送转”。完整历史事件的发行人公告对账、数据源覆盖证明和历史复权因子仍需后续完成；**P4.3-E 第一阶段质量闸门保持关闭，不允许发布新训练标签或启动 Transformer**。

完整的 Ubuntu 三阶段执行示例（包括用户自己的 `TUSHARE_TOKEN`、批次进度和核验产物）见 [P4.3-E 第二阶段操作说明](ashare/P4_3_E2_FEED.md)。供应商字段以 [Tushare 官方 dividend 文档](https://tushare.pro/document/2?doc_id=103) 和 [stock_basic 文档](https://tushare.pro/document/2?doc_id=25) 为准。**这次交付的是可运行的数据接入与覆盖诊断，不是已经获取的全市场供应商数据或完整公司行为历史。**

### P4.3-E2 免费、开源优先：AkShare 无 Token 公司行为数据接入（2026-11-15）

**Tushare Pro Token 不是必须条件。** 旧的 `ashare/p4_action_feed.py` 是有 Token 时才可使用的可选方案；默认请使用新增的 `ashare/p4_action_feed_free.py`（无需 Token）。它从现有 `public.market` 的**历史出现过的股票代码**生成只读快照，通过 AkShare 的新浪财经（历史分红/配股）及巨潮资讯（历史分红）公开接口取数，按**每十股**换算送转和税前派息并进行双源对账；冲突与失败全部保留为未解决状态，配股另列人工核验清单。

新增 `tests/test_p4_action_feed_free.py` **7 项本地回归测试已通过**，覆盖公司行为数量换算、双源冲突隔离、旧测试边界过滤、SHA256 防篡改、接口失败/配股隔离及股票范围未抓取。**截至本次提交，尚未在用户 Ubuntu 执行真实 AkShare 接口采集或全部仓库回归。** 获取免费数据并不代表其历史完整性或再利用条款得到认证；程序仍不签发任何 `COMPLETE_VERIFIED`，不释放训练标签。

推荐试点及完整命令：[**P4.3-E2 无 Token AkShare 操作说明**](ashare/P4_3_E2_FREE_FEED.md)。试点前只需 `python -m pip install akshare`，无须配置任何 Token；此前 [Tushare 方案](ashare/P4_3_E2_FEED.md)仅作为可选替代。

### P4.3-E2 免费 AkShare 首批真实运行兼容修复（2026-11-16）

用户在 Ubuntu 已通过免费数据源模块原有 7 项单测，并运行基于 `public.market` 的 5,449 股票历史快照、前 10 股票 30 次接口调用（成功 20、失败/截断 10）以及首批覆盖审计。初审 `not_fetched=5439`、`failed_or_truncated=10`、`fetched_not_verified=0` 属于**三个子接口中任一失败就整只股票未完成**的设计；`4` 个事件候选、`4` 条冲突或无效行、`20` 条配股待复核原始记录均未取得完整证明，不能进入模型训练。旧日志中比亚迪官方控制案例 1 项未匹配，因其未被前 10 股票批次抓取而可能被误计为冲突，不能当作已确认数据错误。

**代码修复**：`ashare/p4_action_feed_free.py` 优先使用当期 AkShare `stock_dividend_cninfo`（单数），并兼容历史 `stock_dividents_cninfo`（旧拼写）；原实现仅使用旧拼写，可能导致全部巨潮接口失败。新增每个子接口独立的成功、API 错误、行数限制统计 `endpoint_results`；公告对照结果现在区分 `NOT_FETCHED_FOR_CONTROL`、`INCOMPLETE_FETCH_FOR_CONTROL` 与 `MISSING_OR_CONFLICTING`，不再将“没抓比亚迪”计为条款冲突。新增回归测试，现该模块共 11 项测试（仍须在用户 Ubuntu 重测）。

**复核流程**：先用旧 `request_ledger.csv` 中 `endpoint`/`status`/`error` 确定十次实际报错的来源，然后保留旧快照，在全新的 `batches_retry_20261116` 根目录重新抓前 10 只并审计。不可直接重复写入原 `batches/batch_000000`（不可变快照），也不可让新旧重试批次混入同一根目录使同股票/接口重复。命令、注意事项及解释见 [P4.3-E2 免费接入说明](ashare/P4_3_E2_FREE_FEED.md)。当前未重新访问用户 Ubuntu，**不声称新接口已在真实网络中验证成功**；继续不自动发放 `COMPLETE_VERIFIED` 或训练标签。

### P4.3-E2 免费 AkShare 首批运行诊断与兼容修复（2026-11-16）

Ubuntu 真实运行：`test_p4_action_feed_free.py` 旧版 7/7 通过；正式行情历史股票范围 5,449 只（只表示 DB 内有记录，不是交易所全集）；前 10 只股票 × 新浪分红/巨潮分红/新浪配股各 1 次 = 30 请求，**20 SUCCESS、10 失败或达到返回上限**，因此 10 只均标 `FETCH_FAILED_OR_TRUNCATED`，其余 5,439 只为 `NOT_FETCHED`。4 条未核候选权益、4 条源冲突/无效记录，不能直接并入已核公司行为清单。**仅凭总数还无法确认哪一接口失败**，须查看原批次 `request_ledger.csv` 分组以及每条 `error`。

发现并修复：原免费适配器巨潮方法使用了旧拼写 `stock_dividents_cninfo`；优先调用 AkShare 当前 `stock_dividend_cninfo`，兼容旧包装。官方比亚迪公告控制样本尚未抓取时不再误报为真正 mismatch；区分 `NOT_FETCHED_FOR_CONTROL`、`INCOMPLETE_FETCH_FOR_CONTROL`、`MISSING_OR_CONFLICTING`。配股危险事件只统计请求范围内的除权日，保留全历史原始条数供核对；新版审计 JSON 新增 `endpoint_results`（每接口成功/失败分布）、`endpoint_failure_examples`。**11 项本地合成回归测试通过**，GitHub 上源码与测试的 blob SHA 与本地测试版本完全一致，但这不能替代 Ubuntu 真实接口复测。

旧 `snapshot` / `batch_000000` / `audit_first_batch` 都不可覆盖。先读取 `batches/batch_000000/request_ledger.csv` 确认错误原因，再用 `batch_root=batches_retry_20261116` 的新路径重抓 10 只，并输出到 `audit_retry_20261116`。重测方法与完整命令见 [P4.3-E2 免费接入说明](ashare/P4_3_E2_FREE_FEED.md)。**未产生 `COMPLETE_VERIFIED` 或任何可训练经济收益标签，模型训练闸门继续关闭**。

### P4.3-E2 第三次真实网络诊断：29/30，巨潮单股 KeyError（2026-11-16）

Ubuntu `ashare-dev@7dac7c8` 在 AkShare 1.20.65 下已通过 **11 项**免费源单元测试，并且明确从本仓库加载到修正后的巨潮接口名（`stock_dividend_cninfo`）。头 10 只股票第二轮网络数据抓取成功 **29/30**：新浪分红 10/10、配股 10/10，巨潮分红 9/10，最后一次返回 `KeyError: '实施方案公告日期'`；股票级 `9 FETCHED_NOT_VERIFIED`、`1 FETCH_FAILED_OR_TRUNCATED`、`5439 NOT_FETCHED`。原有分红候选 4 条、无效/冲突行 5 条、配股 20 条为全历史而指定审计期间 0 条；`issuer_controls_not_fetched=1`（前 10 只未请求比亚迪）而 `mismatch=0`，属于合理口径。仍没有任何完整覆盖证明或可训练标签。

[AkShare 的实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_dividend_cninfo.py) 对 CNINFO 返回的 `records` 构造 DataFrame 后直接访问公告日期列，**当上游返回空数组或字段名称变化时都可能抛 KeyError，无法仅据错误归因为“无分红”**。本次在 `ashare/p4_action_feed_free.py` 新增专门的 `UPSTREAM_EMPTY_OR_SCHEMA` 非成功状态，审计结果记录 `cninfo_upstream_ambiguous_calls` 和 `endpoint_failure_stocks`。维持模型训练质量闸门；不要因剩余一只股票而放行数据或影响 `stock_scripts` 的生产行情采集。新增两项合成回归测试，现共 **13 项**，须 Ubuntu 重测。

诊断及新目录重试命令见 [P4.3-E2 免费方案说明](ashare/P4_3_E2_FREE_FEED.md)；旧数据仍保留且不应覆盖。**当前修改尚未在用户 Ubuntu 或真实巨潮接口重新执行。**

### P4.3-E2 第三数据源 BaoStock 已接入（2026-11-16）

沿用用户原有 [`stock_scripts`](https://github.com/cailming/stock_scripts) 的分钟/日线生产体系，不重复抓行情、不修改生产数据或交易任务。P4.3-E2 目前免费 AkShare（新浪+巨潮）首批真实网络结果 29/30：唯一巨潮 000004 原始 JSON `records=[]`，AkShare 1.20.65 因未妥善处理空表抛 `KeyError('实施方案公告日期')`。空响应不能代表无分红；此前程序已将其作为 `UPSTREAM_EMPTY_OR_SCHEMA` 非成功状态，旧测试期继续冻结。

**本轮新增** [BaoStock 无 Token 公司行为独立交叉核对程序](ashare/p4_action_feed_baostock.py) 及 [完整运行手册](ashare/P4_3_E2_BAOSTOCK.md)。在冻结的正式行情 5,449 股票代码快照上，根据 `query_dividend_data(...,yearType="operate")` 按实施年份抓历史送转和税前现金分红，以**每股**权益单位（不同于新浪/巨潮每10股）生成 `UNVERIFIED` 候选，支持与已有 AkShare 候选与发行人比亚迪实施公告样例交叉核对。显式记录 API 异常、无效除权日、年份截断、重复除权日、来源条款冲突、空结果与未支持的交易所；文件带 SHA256，不覆盖原批次，拒绝写入旧测试边界 2026-06-12 及以后的事件。首批限制 000004 与 002594，后续待真实运行后再扩大到 10–30 只批次。

**验证边界**：新增 **15 项独立本地模拟单测通过**，GitHub 回读源码/测试/文档的文件 blob SHA 与通过测试的本地快照一致。BaoStock 的 Ubuntu 登录、实际查询、跨源生产数据审核 **尚未执行**；不能宣称数据覆盖或发放 `COMPLETE_VERIFIED`。即使三源候选条款一致，仍缺少全市场官方事件完整证明、配股等复杂权益的核验；`P4.3-E` 的训练闸门继续关闭，新 Ridge/Transformer 不得依据当前标签重训。

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
