# P3.3-C — 有公告依据的历史风险事件拦截

**当前只实现按历史公开公告拦截已知风险，不代表全市场历史非 ST 状态已完整重建。**

## 1. 两只核实的历史风险样本

- 002731 萃华珠宝：2026-07-06 公开公告 2026-067；2026-07-07 起 *ST。
  原始公告：https://disc.static.szse.cn/disc/disk03/finalpage/2026-07-06/2ff408da-fb79-4c86-a18a-bd07ad1be299.PDF
- 688496 清越科技：2026-05-09 公告 2026-033；2026-05-12 起 *ST。
  原始公告：https://static.cninfo.com.cn/finalpage/2026-05-09/1225286765.PDF
- 两者后续停牌事件也列在示例 CSV，但若来源公告晚于停牌日期，不能在
  公告披露前的交易决策中倒灌。002731 的 9 月 1 日停牌证据取自
  9 月 7 日公告，在 9 月 1 日之前**不可**用于当时的策略判断。
- 688496 的 9 月 11 日公告同日披露，仅凭日期无法证明其在当日开盘前
  已知；日期粒度采用保守的 published_on < order_day 判定。

示例来源文件：`ashare/data/risk_events_verified_examples.csv`。
严格说这是两只股票的**部分**历史事件，不能推断其它股票或早期日期正常。

## 2. 数据规范

CSV 必填字段：

    stock_code,event_type,effective_date,published_on,source_url,source_id

event_type 可选：
- RISK_ST：实施 ST
- RISK_STAR_ST：实施 *ST
- RISK_NORMAL：撤销 ST 风险警示
- HALT：停牌开始
- RESUME：恢复交易
- DELISTED：终止上市

日期：YYYY-MM-DD；6 位股票代码；保留公告来源 URL 和唯一编号。
同一股票同一类型范围在相同生效日，不允许冲突变更。

**时间安全规则：**T 日收盘产生信号；模拟订单日为下一个市场交易日
（T+1）。只有 `published_on < T+1` 且
`effective_date <= T+1` 的公告才能拦截买入。
`published_on` 为日期，没有精确时分秒；此规则排除在 T+1
当天才公布的事件，宁可遗漏即时信息，也不凭日后公告“预知”风险。
即使预先知道停牌，现有执行引擎也没有真实队列/涨跌停成交验证。

同一股票范围内后来生效的风险撤销或复牌事件可结束先前限制。
未知股票/无事件**不能被认为历史状态正常**。风险警示过滤只作用于
新开仓信号；已经买入的股票不被模型强行平仓。

## 3. 如何进行混合诊断

不要覆盖 P3.2/P3.3-B 原有结果。在 AlphaGPT 仓库根目录执行：

    git switch ashare-dev
    git pull --ff-only origin ashare-dev
    python -m unittest discover -s tests -p 'test_*.py' -v

    python -u -m ashare.backtest_run \
        --start 2025-03-01 --end 2026-09-30 \
        --exploratory \
        --exclude-current-st \
        --risk-events ashare/data/risk_events_verified_examples.csv \
        --candidate-pool 20 --log-every-days 20 \
        --out reports/ashare_backtest_current_st_plus_events

这个实验 = 旧版当前 ST 黑名单（NON-PIT）+ 部分历史公开公告拦截。
**整体仍为 NON-PIT，并不代表无未来信息偏差**。

如只想看局部历史事件的拦截量，可仅加 `--risk-events` 而不加
`--exclude-current-st`；但全市场其他股票状态无法确认，
仍然只是诊断运行，不能输出合格的策略总收益。

生成的结果在对应日期子目录，除既有成交/净值/持仓等文件外，新增：
- risk_event_source_audit.csv：每个事件源、公开日、生效日及阻止信号数量
- risk_event_blocked_by_day.csv：各信号日/股票阻止买入及其公告编号
- summary.json 中的 risk_event_overlay、stock_pool_mode、
  full_market_risk_history_verified、audited_total_return_available

上传上述输出以核对：002731 与 688496 是否不再模拟买入、旧价格估值是否
缩减，以及新出现的低价/停牌风险股票。不能仅凭风险过滤结果预测净收益。

## 4. 可选的 PostgreSQL 结构

`ashare/sql/stock_risk_event.sql` 提供只读研究之外可用的
事件留存表结构和中文 COMMENT；**不会自动执行**。本次程序直接使用 CSV，
因此不需要用户先改变生产数据库。

## 5. 尚未解决

- 全市场风险警示历史状态完整性及风险撤销的全部历史
- 证券交易日历与停复牌交易可行性的精确时点认证
- 不复权行情下现金分红、拆股等真实收益调整
- 低价退市、涨跌停队列、无法卖出和清算的正确模型
- 当前数据库 stock_info.st 与部分公司官方公告不一致的采集质量排查

不能用于 Transformer 模型的真实收益标签、回测业绩宣传或实盘下单。
