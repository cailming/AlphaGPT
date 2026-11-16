# P4.3-E2 公司行为采集与审计

在 `ubuntu92:/home/gc/gcdev/alphagpt/AlphaGPT` 的 `ashare-dev` 分支运行。默认范围为冻结快照索引 10–109（含两端），每批最多 20 股。支持 `--offset` 与 `--limit` 选择其他范围。`--full` 用于试点验收之后的扩量。已有冻结快照和 Ridge 元数据是全部输入，采集和审计无需数据库连接。

```bash
cd /home/gc/gcdev/alphagpt/AlphaGPT
conda run -n py312 python -m ashare.p4_action_pipeline plan
conda run --no-capture-output -n py312 python -u -m ashare.p4_action_pipeline run
conda run -n py312 python -m ashare.p4_action_pipeline status
conda run --no-capture-output -n py312 python -u -m ashare.p4_action_pipeline resume
conda run -n py312 python -m ashare.p4_action_pipeline audit
conda run -n py312 python -m unittest discover -s tests -p 'test_p4_action_*.py'
```

`plan`、`status`、`run --dry-run` 仅读取本地证据。调度器依次处理每批，记录每次请求结论和每批累计审计。新浪分红和配股共享节流间隔，巨潮和 BaoStock 使用各自的间隔。网络类异常按请求键指数退避，默认每键最多执行三次新尝试。限流、确定的模式变更或滚动窗口内持续失败会停止采集。先查阅 `checkpoint.json` 的 `circuit` 和报告中具体错误，经诊断恢复后执行 `resume --reset-circuit`。

`SIGINT` 或 `SIGTERM` 会让当前请求结束或到达超时上限，写入结果后退出。重新运行会从持久化结果继续。强制终止时，网络请求已经发出但结果尚未写盘的唯一在途请求可能再次发出。单一输出目录只允许一个写入者。已完成的试点重复执行 `resume` 只重算审计。

## 证据与报告

- 原有批次在原位逐一核对 SHA256。字节完全相同的清单及其数据文件按一次尝试计数。请求键为股票、接口、适用的年份。历史清单按路径排序，新请求按序号追加。同一键选第一份通过校验的成功证据，否则使用最近一次失败。历史批次和新尝试均保留。
- 新尝试保存在默认输出根目录 `reports/ashare_p4_action_pipeline/attempts/`。单文件含请求账本、预留测试边界前的原始权益字段及 SHA256。检查点记录已落盘尝试和历史清单校验值。缺失或损坏会中止审计并写入 `p0_blocker.json`。
- `latest_report.json` 指向最近一份完整审计。审计目录含 `pipeline_summary.json`、`coverage_summary.csv`、`source_health.csv`、`cross_source_reconciliation.csv`、`exceptions.csv`、`retry_queue.csv`、`manual_review_queue.csv`、`request_status.csv` 和 `issuer_controls.csv`。长期研究记录保存在本文件。GitHub README 发布阶段进度。
- 100 股试点要求应查任务都有成功或明确的终态例外，且无 P0、待重试任务或当前熔断。个别缺字段、持续空响应及条款冲突保留证据，等待人工核查。试点通过后停在用户验收关口。
- 数据和候选仅用于核对 `2026-06-12` 之前的公司行为。任一金额字段未知时保留未决状态。报告保持 `complete_verified_assertions_created=0`、`can_emit_training_labels=false`、`full_historical_action_coverage_certified=false`。经济收益标签仍需要独立证明发行人事件覆盖、权益参数与实施日期，以及配股等复杂事件的处理口径。

## 100 股试点实测

2026-11-16 在 `ubuntu92` 完成冻结索引 10–109，共 100 股，代码范围 `000014` 到 `000528`。自动处理索引 10–29 的遗留任务和三个新增 20 股批次。原采集退出码为 0。复用 239 个既有成功任务，本次发送 363 次请求。用 `8d2dff6` 执行 `resume` 重新审计后，363 份新尝试的文件清单和 SHA256 完全一致。Ubuntu `py312` 的 54 项相关离线测试通过，包含原有 31 项免费来源回归。

| 接口 | 应查任务 | 有效成功 | 终态异常 | 成功返回空表 |
|---|---:|---:|---:|---:|
| 新浪分红 | 100 | 93 | 7 | 0 |
| 巨潮分红 | 100 | 72 | 28 | 0 |
| 新浪配股 | 100 | 100 | 0 | 33 |
| BaoStock 年度查询 | 300 | 300 | 0 | 122 |

- 600 个请求任务都有明确结论。71 股的全部请求成功，29 股有至少一个终态异常。34 个任务存在无法确认的除权或公告日期，已有可用行仍参与对账。`000016` 巨潮接口 3 次新尝试后仍为 `UPSTREAM_EMPTY_OR_SCHEMA`。
- 96 条去重候选，2 条仅见单一来源。新浪与巨潮逐项完整匹配 0 条、部分字段未决 94 条。BaoStock 与 AkShare 送股、转增、税前现金均有值并匹配 2 条，部分字段未决 92 条。
- 期内配股风险记录为 0。原始证据中另保留 109 条期外配股记录。比亚迪 `002594` 的既有发行人控制样本分别通过 AkShare 与 BaoStock 核对，未把该股票加入试点采集。
- P0 为 0，P1 共 36 项，P2 共 189 项。日期来源问题之外的 P1 包括 `000423`（2025-09-03，新浪与巨潮权益条款冲突）、`000430`（2025-12-29，AkShare 单边候选）。没有待重试请求，当前熔断为空。旧测试期隔离检查通过。
- 报告核验时间为 2026-11-16 15:23（UTC+08:00）。`ubuntu92` 的本次完整报告目录为 `/home/gc/gcdev/alphagpt/AlphaGPT/reports/ashare_p4_action_pipeline/audits/audit-vvs9po3e`。首轮采集日志为 `reports/ashare_p4_action_pipeline/collect.4A72H1.log`。
- 试点停在用户验收关口。继续进入其他股票范围需等待验收。

## Phase 3 运行与审计

试点通过技术验收后，已授权在冻结快照的 5,449 股内运行全量采集。快照 SHA256 为 `c9f61ef189688d86e7183544527737d41a568342b1699fbf242c47c219012ce8`，旧测试起始日 `2026-06-12` 排除。生产任务和数据库无需参与。

前置账本去重后，32,694 个任务中有 603 个成功可复用（其中 38 个来自试点范围之外）、36 个历史终态异常、32,055 个尚未处理。其中 846 个 BaoStock 任务的股票代码缺乏已验证的交易所映射，按本地终态归档。其余首轮最多 31,209 个查询任务，重试按实际响应追加。全部回归通过后使用 `run --full` 写入全市场检查点，断线后通过持久化日志查看，确需进程恢复时沿用 `resume`。

新账本分别记录缺失或无效的除权日和公告日计数。旧试点中只有合并状态的日期异常单列为无法拆分，历史原始证据照常保留。绝对完整性异常触发停机，普通字段问题、明确不支持、空返回和单源记录保留各自结论。整个阶段 `training_gate=CLOSED`。

`resume --isolate-circuit` 保留已诊断的来源熔断，允许其他来源继续。被隔离任务保留未抓取或待重试状态。全量采集会保持未完成，直到能独立验证服务恢复并执行 `resume --restore-source <source>`。股票身份不一致、SHA256 等 P0 先处理完整性问题。

```bash
conda run -n py312 python -m ashare.p4_action_phase3_audit
conda run -n py312 python -m ashare.p4_action_phase3_audit --final
```

进度报告复算截至最近一份累计审计的任务、股票、空表和优先级。异常画像分开计量请求、来源记录和权益字段，并产出发行人公告核验及来源排查队列。最终报告要求全量检查点完成、无在途问题且训练闸门关闭。

### BaoStock 登录超时及隔离续跑

2026-11-16 20:22 北京时间，BaoStock 登录等待超时。TCP 已建立，独立的 12 秒匿名登录探测在 `login → send_msg` 等待阶段仍超时。故障归入来源会话，影响期间未获取的股票年度任务保留未抓取或待重试状态。早期已达最大尝试次数的日期任务保留终态证据。

在同一个检查点执行 `resume --isolate-circuit` 后，新浪分红、巨潮分红和新浪配股继续落盘。守候进程日志为 `reports/ashare_p4_action_pipeline/phase3-isolated.YxUK77.log`。健康来源处理完成后，恢复程序仅针对已知的 BaoStock 登录超时，每 45 分钟至多试一次匿名登录，最多 8 次。探测成功才恢复该来源。新熔断、P0 和其他来源阻断继续按原规则保留，供诊断。全部收尾条件满足后，自动写入 `reports/ashare_p4_action_pipeline/FINAL_FULL_AUDIT_REPORT.md`。
