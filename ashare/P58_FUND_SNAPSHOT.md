# P5.8-A 资金与事件快照

ubuntu92 研究任务使用 `ashare.p58_fund_snapshot --out <fund-root> --phase close`，在上海时间收盘后、18:00 前记录一次真实观察。已有收盘后研究流程可调用 `--phase next-open` 保存当晚可供后续开盘研究的数据。手动 `--phase observe` 只记录观测，没有预测资格。上述三种调用只读取运行当日的生产记录，命令没有指定历史日期或模拟当前时间的参数。

生产调度在 ubuntu77 用户的 crontab 中，工作日 17:00 运行 `/home/ubuntu1/cailm/stocksrc/main.py`。日线及多项派生任务执行后，`utils.monitor.detect_stock_main_force_daily()` 先后调用资金 V1 与 V2 函数，随后还有颈线突破和平台候选生成。生产日志记录 2026-11-23、24、25 两版资金生成分别在 22:06:01、22:04:43、21:55:15 结束。11 月 25 日整条日任务在 21:55:18 结束。日内事件也由独立函数生成或变更。这些时间用于安排重复观察，每日实际发布可能更晚。

每次读取使用一个 PostgreSQL REPEATABLE READ、READ ONLY 事务。三个来源各自保存所有原字段、来源标签、写函数与列结构指纹、采集时钟、源记录时间和 PostgreSQL 完整行 JSON 文本。日期时间列按源文本保存，数值列另提供可计算的数值类型，需要核对精确小数时读取原始 JSON。V2 影子表单独存放。当前三个来源均无真实 updated_at 时保留空字段。文件以数据源命名，SHA256、字段非空率、有效股票数、创建时间范围、相对 18:00 的生成延迟和上一正式交易日同阶段的变化保存于 manifest。

发布目录为 `YYYY-MM-DD/{close,next-open,observe}/rNNNNN`。只有本次实读内容或来源指纹发生变化才产生修订，保留旧版本并记录前一版清单散列；采集锁和同文件系统目录重命名保护重试及并发。选取某一时点的数据必须校验文件散列、phase、availability、snapshot_at 和 files_ready_at。close 快照可能只包含信号，空行数保留；晚间资金评分只能用于实际采集之后的研究。formal_stocks_at_observation 是读取瞬间已有正式行情数，mutable_source_completion_certified 始终为 false，整日完成情况须结合当日生产任务和后续版本判断。若前一正式交易日无同阶段快照，则变化指标缺失。

预开盘应急观察使用 `--phase preopen`。程序以服务器真实时钟确定本次观察日，按上交所已公告的年度休市表推导相邻两个计划开市日，同时在同一只读事务中核对生产最新正式日线就是前一日。只支持已收录的 2026 年日历，09:15 开盘集合竞价开始即停止。仅这一入口可取前一个正式交易日的源行，另置 `preopen` 目录，manifest 记录 target_session、实际早间观察时间及日历来源。[上交所 2026 年休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml)

ubuntu92 用 `/bin/bash p58_research_schedule.sh close <研究根目录>` 在工作日 17:55 单独采集，原夜间入口仍先调用完整 P55 程序，再运行 `post`。旧入口和 crontab 保存在 `schedule-backup`。新程序分别记录资金、监控、C组和组合的状态，运行日志位于 `schedule-logs/YYYY-MM-DD`。早于18:00调用 post 直接返回。若六方法的当日清单仍缺，常规重试先保留观察，到最后的23:35仍尝试缺口报告和账户结算。组合使用已冻结的上一日名单，当日C组缺失不会阻断就绪账本的估值。
