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
