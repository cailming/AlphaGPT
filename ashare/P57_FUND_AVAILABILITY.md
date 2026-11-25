# P5.7 主力资金与事件历史可见性

审计只读查询正式行情、主力资金日评分、策略事件，以及目录中已有的回填前备份和影子表。使用同一个 PostgreSQL `REPEATABLE READ` 快照。输出每个正式交易日的行情覆盖、主力评分字段覆盖、写入时点，以及按日和信号代码汇总的事件覆盖。

```bash
PGOPTIONS='-c default_transaction_read_only=on' conda run --no-capture-output -n py312 \
  python -m ashare.p57_fund_audit --start 2025-01-01 --end 2026-11-24 --out "$new_audit"
```

日期和时间统一以北京时间解释。18:00 是既有日线研究的信号边界，下一正式交易日 09:30 用作单独的时间分布诊断。评分有同日时间戳，只表示当前存储的值带有该时点。数据库程序能覆盖评分和生成时间，事件状态与评分也能随更新变化。审计同时核对备份和现表的同股同日记录，区分字段内容变化与时间戳变化。

资金活跃度、持仓程度、出货风险与每日总分必须有按时点保存的来源。评分变化需要当前及前一正式交易日的版本双方都有来源。突破和缩量回踩的策略事件需要能确认当时的评分、状态和完整事件覆盖。回填前的单次副本用于查证版本变化，影子表用于核对覆盖与写入时间。可信样本的准入仍以可验证的当日版本为依据。

`fund_availability_audit.json` 绑定审计文件散列及写入函数的版本指纹。`formal_market_daily.csv`、`money_daily_availability.csv`、`signal_daily_availability.csv` 和 `column_catalog.csv` 可用于逐日复核。若没有可核验的历史版本，保留审计结果，资金增强训练保持未启动。原量价模型及每日前瞻任务仍运行。
