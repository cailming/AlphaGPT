# P4.3-F1/F2 A 股子集质量验收

P4.3-E2 继续负责严格的公司行为覆盖审计。P4.3-F1/F2 使用独立的研究资格，在需要的股票和交易日窗口核验证据。全市场严格训练闸门仍为 `CLOSED`，子集流程不创建 `COMPLETE_VERIFIED`，也不将标签或模型用于实盘。

## 股票池、标签和样本状态

- 对每个信号日 T，仅以 T 收盘时可见的正式量价数据排序。沿用 P4 的 40 个连续有效交易日要求、流动性与价格门槛和 10 个特征，按近 20 个交易日成交额中位数取前 800。已核验的风险事件按披露可见时点生效。历史 ST 覆盖有限，禁止用当前 `stock_info` 回填历史。
- 冻结候选及其排序之后再做质量验收。剔除未来路径或标签异常的样本时，保留当时的原排序，空出的名额不补位。股票级报告汇总逐日结果，历史某日能否入池始终按当日判断。
- 复用 P4.3-E2 请求账本、带日期的异常、发行人已核验行为和 76 项北交所实施公告索引。独立的 BaoStock 调整因子接口只用于核对行为日期。计划由各股票首次入池日期及当日排名确定，最多 1,200 只，覆盖区间结束于 2026-06-11。同一批已存证的调整因子请求在恢复时跳过，连续三次来源错误停止并保留现场。
- `TRUSTED_RESEARCH` 需要连续有效行情、各涉及年份至少两家股息/送转来源和配股来源的成功证据，以及可用的因子观察。普通窗口以多源一致、未见有效行为为条件。碰到已实施行为，仅在发行人证据包含所需经济条款且与调整因子日期一致时处理。该等级是有条件的窗口资格。
- `EXPLORATORY_ONLY` 包括缺少独立因子观察或互证来源不足的窗口。`UNAVAILABLE` 包括未解决的冲突、没有可用经济条款的行为、穿越切分的标签、缺失行情及不可解释的跳价。两类均不导出训练标签，原始价格收益可在质量表中作诊断。
- 标签以 T+1 开盘买入，持有 5 个交易日，在 T+6 开盘计价。遇到入场之后至退出当日的已核验除权，以除权前持股数记税前现金权益并更新股数。入场当日按除权后价格成交，因此没有旧股权益。特征仍按原 P4 的原始行情计算，观察窗口内有行为的样本隔离。标签未计税、费、滑点或交易成交限制。
- 训练信号从 2025-03-03 开始，验证与历史测试的分界日分别为 2026-01-05 和 2026-03-23。较早一段的标签必须在下一段开始前结束。读取行情和因子均在 2026-06-11 截止，保留原 2026-06-12 起的隔离区。这里的历史测试用于本版流程的时间检验，相关历史行情可能曾用于旧研究。

## ubuntu92 运行

在 AlphaGPT 根目录执行。数据库配置沿用项目现有配置，行情只读。在网络可用时先完成 `capture`，随后运行数据集构建。发生中途失败时保留现有产物并查日志，构建目录使用新路径。`notice_index` 指向审计归档中原始的 `verification-final/official_notice_index.csv`。

```bash
mkdir -p "$PWD/reports/ashare_p4_subset"
run_dir=$(mktemp -d "$PWD/reports/ashare_p4_subset/f1-f2.XXXXXX")
PGOPTIONS='-c default_transaction_read_only=on' conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_inputs prepare --out "$run_dir/inputs"
conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_inputs capture --out "$run_dir/inputs"
conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_dataset --inputs "$run_dir/inputs" \
  --notice-index "$notice_index" --out "$run_dir/dataset"
```

`training_dataset.parquet` 只含独立可信样本，`sample_quality.parquet` 记录逐样本等级、原因和价格诊断。`stock_quality.csv`、`daily_quality.csv` 和 `subset_audit.json` 记录分布、时间边界、输入及产物摘要。`causal_inference_features.parquet` 单独保存用于历史回放的 T 时点量价候选，不依据未来是否有可用标签筛选。行情快照、来源证据、数据集和模型均保留在运行主机。

## P4.3-F3 Ridge 和 RTX 3090

Ridge 在训练段拟合特征变换和模型，在验证段按日均横截面 Rank IC 选正则，随后对历史测试段评分。历史推理从特征专用 Parquet 读取，时间必须晚于用于选模型的最后一条验证标签，支持独立加载模型和预测回放。

```bash
conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_model train --dataset "$run_dir/dataset" --out "$run_dir/model"
conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_model infer --dataset "$run_dir/dataset" \
  --model "$run_dir/model" --asof 2026-06-11 --out "$run_dir/inference_replay.parquet"
conda run --no-capture-output -n py312 \
  python -m ashare.p4_subset_transformer --dataset "$run_dir/dataset" --out "$run_dir/transformer"
```

Transformer 的预检在 CUDA 上用合成输入检查形状、因果掩码和反向传播。市场数据准备使用同一股票截至 T 的 16 段连续特征，每个序列校验扩展后的公司行为观察窗口。分段 `.npz` 分别指向已校验 Parquet 的特征行和标签行。`train_only_scaler.npz` 保存仅在训练序列用到的特征行上计算的归一化参数。`transformer_preflight.json` 绑定数据集散列、三段样本数和首轮训练配置。后续正式训练按训练段拟合网络权重，按验证段选择模型及早停点，保存归一化参数、模型、特征顺序和数据版本，最后单独评估历史测试段，与同口径 Ridge 对比。
