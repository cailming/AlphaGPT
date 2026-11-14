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
| P4.1 | 当日及此前信息的动态流动性 Top-N 股票池、连续有效交易日检查、风险事件局部拦截：`ashare/p4_mvp.py` | **已有 MVP 代码与单测**；真实 PostgreSQL 样本规模/拒绝原因/月度覆盖仍需验收与完善 |
| P4.2 | 过去窗口量价特征、未来开盘价标签、停牌缺口校验、按日期切分及跨集合标签净化：`ashare/p4_mvp.py` | **已有 MVP 代码与单测**；不复权导致的除权偏差仍需审计 |
| P4.3 | NumPy Ridge 基线、验证集选参、保存/加载、样本外预测、CSV：`ashare/p4_train.py`、`p4_predict.py`、`tests/test_p4_mvp.py` | **已有端到端实现，待真实数据库运行验证**；Logistic / LightGBM 尚未实现 |
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

上述是仓库**已存在的入口与建议运行命令**；本次 README 更新不等于已在用户 Ubuntu/真实 PostgreSQL 上执行成功。产出的任何样本外指标需查看实际生成的 `model_metadata.json`，**不得预填准确率或 IC**。

### 当前问题与后续优先顺序

- **P4.1 验收**：在真实 `market` 上核对日期覆盖、活跃股票数、有效标签数、逐月覆盖；补充各阶段拒绝原因与防止异常行情/除权干扰的审计。不要使用“回测期末仍存活”作为历史入池条件。
- **P4.2/P4.3 验收**：实跑全部单元测试、Ridge 训练与独立推理，核验输出、边界日期及训练/推理一致性；补充模型覆盖率、高置信度准确率、方向准确率、Top-K 相对表现及与费用/换手关联的可解释指标。不要在看过测试集表现后反复选参。
- **成交/数据限制**：主力资金分数历史时点不可靠；当前 ST 历史倒灌不可用于训练；不复权价格中的公司行为、停复牌与开盘成交概率仍未完整建模。相关治理不阻塞低风险的 P4 MVP，但限制可宣称结论。
- **P4.4**：在 Ridge 真正跑通之后，以**相同样本/时序切分/评价口径**实现 PyTorch 小型 Transformer，CUDA 3090、训练过程日志、checkpoint/恢复、批量 CSV 推理。先做可运行 MVP，再考虑复杂架构。
- **开发纪律**：仅向 `ashare-dev` 提交，每个阶段配单测，并同步更新此章节、最近结果与提交号；禁止自动改动 `main` 或写入生产行情表。

**最近核查日期**：2026-11-14。核查时 `ashare-dev` HEAD 为 [`970151c0d947984a1cb6919d01bfe59b59b7c61d`](https://github.com/cailming/AlphaGPT/commit/970151c0d947984a1cb6919d01bfe59b59b7c61d)（P4 股票池、缺口安全标签、Ridge OOS MVP）；前一个风险事件核查提交为 [`9febe7fc2a3282108096091c5de11d58e4a8c547`](https://github.com/cailming/AlphaGPT/commit/9febe7fc2a3282108096091c5de11d58e4a8c547)。后续会随新代码和真实输出更新这里，不假定历史 SHA 永远为最新。
