# P4.4 Transformer 与 P4.5 正式日线研究推理

## 冻结输入与训练

该流程复用 P4.3-F1/F2 的 Parquet、冻结序列索引和仅由训练序列计算的归一化参数。训练、验证、历史测试分别有 70,724、17,758、18,498 条序列。模型使用每股 16 个连续市场交易日的 10 个 P4 特征。本轮训练的经济收益标签从 T+1 开盘到 T+6 开盘。原严格模式保持关闭，原 2026-06-12 起的隔离区仍排除于建模及历史比较。

训练在 Ubuntu 的 py312 环境运行，需要现有的 CUDA PyTorch、Polars、NumPy、pandas、SQLAlchemy 和 PostgreSQL 驱动。从 AlphaGPT 根目录执行，使用新的输出路径。

```bash
src=/home/gc/gcdev/alphagpt/AlphaGPT/reports/ashare_p4_subset/f1-f2.MBzBT1
run=/home/gc/gcdev/alphagpt/AlphaGPT/reports/ashare_p4_transformer/new_run

conda run --no-capture-output -n py312 python -u -m ashare.p4_transformer_train \
  --source "$src" --out "$run/model"
```

网络为 48 维、4 个注意力头、2 个编码层、前馈宽度 96 的因果 Transformer。批量 64，AdamW 学习率 0.0003，固定种子 0，训练最多 30 轮。默认在 CUDA 支持时使用 BF16，另可请求 FP32。验证指标按日横截面 Spearman Rank IC 汇总，连续 5 轮未改善则停止，使用验证 IC 最高一轮。训练损失是 Smooth L1。逐轮输出训练与验证指标、学习率、耗时和 CUDA 峰值显存。

训练每轮原子保存当前权重、优化器状态、训练记录和迄今最佳权重。中断后对同一目录执行原命令并加 `--resume`，从最近完整 epoch 继续。输入散列、精度、配置和硬件身份必须一致。训练进程只读取训练和验证标签。`best_model.pt` 保存最佳权重，`model_metadata.json` 记录特征顺序、归一化参数、网络配置、时间边界和数据集散列。保存前与重载后的验证预测逐条比较。

## 一次性历史比较与独立回放

冻结模型后按 `evaluation_plan.json` 在同一组历史测试序列上比较 Transformer、原 Ridge、同序列训练的 Ridge、MA20 反转和训练收益均值。匹配 Ridge 在相同训练序列上拟合，在相同验证序列上定正则。各方法用验证集绝对评分的 80% 分位数确定高置信阈值。MA20 评分是原始负偏离比率，均值基线和同分时以股票代码顺序确定名单；常数 Rank IC 留空。

Top-10/20 按日等权，股票池基准为当日共同可评价样本的均值。名单更替率按相邻观测日的前一名单被替换比例计算。成本诊断按 5、10、25 bp 单边、每笔往返扣费，分开记录组合的费用后收益及相对于费用前等权池的超额。五日持有批次有重叠，报告是收益诊断。此历史区间曾用于旧研究，结果只用于一次比较。

```bash
conda run --no-capture-output -n py312 python -m ashare.p4_transformer_evaluate \
  --source "$src" --model "$run/model" --out "$run/evaluation"
conda run --no-capture-output -n py312 python -m ashare.p4_transformer_infer \
  --source "$src" --model "$run/model" --asof 2026-06-11 --out "$run/replay"
```

## 最新正式日线与前瞻验收

`--latest` 使用 PostgreSQL 的只读一致性快照。沿用项目既有连接配置。北京时间同日 18:00 起允许查询该日正式记录，之前取较早的最新正式交易日。只使用截至信号日的行情和已知风险事件。按原 P4 流动性和特征规则选池，要求每只股票至少有 20 条已冻结的训练序列，并实际具备 16 个连续合格特征日。公司行为和历史 ST 的覆盖限制写入行级标记。可用序列不足时，保留有固定字段的空 Parquet 和弃权审计。

```bash
PGOPTIONS='-c default_transaction_read_only=on' conda run --no-capture-output -n py312 \
  python -m ashare.p4_transformer_infer --source "$src" --model "$run/model" \
  --latest --out "$run/latest"
```

每日产物记录股票、信号日期、预测经济收益、模型版本、散列、数据质量和对应的冻结基线分数。每个预测日使用独立输出目录。从模型与规则冻结后的首个正式交易日起，按时记录当日收盘后预测。回放和迟到补跑留有标记。至少积累 20 个持有期已结束的交易日，且可评价标签覆盖率达到 80%，再按同一批股票比较冻结模型与基线。关注日均 Rank IC、10 bp 单边成本下 Top-20 净超额及标签覆盖，并报告不确定性。前瞻标签要求公司行为窗口经过独立质量核验。研究期间保持版本与阈值冻结。
