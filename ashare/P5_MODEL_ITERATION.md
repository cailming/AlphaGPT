# P5 股票池、滚动数据和训练目标

## P5.1 股票池与历史序列

信号日 T 保留原 P4 因果特征、风险过滤和当日流动性排序。历史特征按同一股票连续交易日构造，序列中的股票只须在 T 入选 Top-N。16 个特征日分别使用自己当时可见的数据。停牌或缺少正式 OHLCV、异常跳价、公司行为观察不足的窗口仍排除。当天的可用股票数由证据决定。

V2 允许在供应商的辅助昨收字段缺失或非正数时，用同一股票前一连续正式交易日的有效收盘价完成跳价检查。保留该来源标记和原始价格。本规则适用于训练和推理。价格检查通过后仍要求限定日期范围内的独立公司行为证据。历史 ST 沿用已知事件，覆盖仍有限。

```bash
conda run --no-capture-output -n py312 python -m ashare.p5_sequences \
  --v1-model "$v1_model" --asof 2026-11-23 --out "$run/price_diagnostic"
conda run --no-capture-output -n py312 python -m ashare.p5_infer validation-v1 \
  --source "$v1_data" --model "$v1_model" --out "$run/support_v1"
```

逐股 CSV 记录失败原因、异常日期和辅助字段回退。验证门槛实验使用同一组冻结预测，对最少 0、5、10、20 条每股训练序列分别报告覆盖率、日均 Rank IC、方向和 Top-K。所有门槛都维持同样的价格和证据约束。

## P5.2 滚动构建与增量证据

在项目根目录使用已有 py312 环境和数据库连接配置。新月份取实际已结算的正式日线截止日，使用独立输出目录。默认从截止日往前保留约一个月开发诊断、两个月验证和十二个月训练。可显式指定日期边界。训练标签需在验证开始前已成熟，验证标签需在诊断开始前已成熟，诊断的最后标签退出日期不可超过正式行情截止日。用交易日历计算 T+1 至 T+6。

```bash
PGOPTIONS='-c default_transaction_read_only=on' conda run --no-capture-output -n py312 \
  python -m ashare.p5_dataset prepare --out "$run/rolling" --v1 "$v1_data" \
  --price-end 2026-11-23 --evidence-limit 160
conda run --no-capture-output -n py312 python -m ashare.p5_dataset capture --out "$run/rolling"
conda run --no-capture-output -n py312 python -m ashare.p5_dataset build \
  --out "$run/rolling" --v1 "$v1_data"
```

`prepare` 用训练截止前的流动性候选确定近期证据股票范围。`capture` 只对限定名单观察 BaoStock 调整因子和新浪实施分红；恢复时核验并跳过已保存观察。P4 归档覆盖仍用于之前的窗口。任一来源失败、证据不连续、日期存在冲突或行为无法定价时，窗口继续隔离。原全市场严格闸门保持 `CLOSED`。

`dataset/subset_audit.json` 记录分段、日度横截面宽度、排除原因和散列。特征 Parquet 与锚点质量表分开，特征的生成与未来标签是否有效无关。冻结索引用 `transformer/*_sequences.npz`，归一化只使用训练索引涉及的价格特征。

默认诊断样本过窄时，可以冻结一份独立开发分段，保持原训练和证据计划不变。以下边界是本次根据已验收的样本数量确定的开发诊断例外。原窗口保留。

```bash
conda run --no-capture-output -n py312 python -m ashare.p5_dataset reframe \
  --prior "$run/rolling" --out "$run/development" --diagnostic-start 2026-09-23
conda run --no-capture-output -n py312 python -m ashare.p5_dataset build \
  --out "$run/development" --v1 "$v1_data"
```

近期推理证据单独追加，范围优先为已通过价格检查且历史计划未涉及的当日股票。未来可以通过 `extend-evidence` 为已核验股票延伸连续区间，再执行 `capture` 和 `export-live-evidence`。原观察证据和模型不覆盖。

```bash
conda run --no-capture-output -n py312 python -m ashare.p5_dataset prepare-live-evidence \
  --out "$run/live_evidence" --prior "$run/rolling" \
  --candidate-diagnostics "$run/price_diagnostic/candidate_diagnostics.csv" \
  --price-end 2026-11-23 --evidence-limit 120
conda run --no-capture-output -n py312 python -m ashare.p5_dataset capture --out "$run/live_evidence"
conda run --no-capture-output -n py312 python -m ashare.p5_dataset export-live-evidence \
  --out "$run/live_evidence" --prior "$run/rolling"
conda run --no-capture-output -n py312 python -m ashare.p5_infer qualification \
  --source "$run/development" --evidence-source "$run/live_evidence/dataset" \
  --out "$run/qualification"
```

## P5.3 配对对照和冻结推理

三个模型共用特征、序列和网络结构。A 拟合绝对经济收益，B 拟合同一分段内当日合格股票等权收益的超额，C 拟合中心化百分位排名。同收益取平均名次。只有 A 的预测可以直接解读为绝对收益预测。每个模型只按自己的验证集日均 Rank IC 选轮次。支持度扫描也只用验证集。本轮优先报告门槛 0 的质量合格覆盖，逐股保留训练序列数。

```bash
for objective in absolute excess rank; do
  conda run --no-capture-output -n py312 python -u -m ashare.p5_train train \
    --source "$run/development" --objective "$objective" --out "$run/models/$objective"
done
conda run --no-capture-output -n py312 python -m ashare.p5_infer validation \
  --model "$run/models/absolute" --out "$run/support_v2"
conda run --no-capture-output -n py312 python -m ashare.p5_train compare \
  --source "$run/development" --models "$run/models" --out "$run/comparison"
conda run --no-capture-output -n py312 python -m ashare.p5_infer latest \
  --source "$run/development" --model "$run/models/absolute" \
  --evidence-source "$run/live_evidence/dataset" --out "$run/latest"
```

开发诊断保留同一可评价股票池，要求每个共同诊断日至少 20 个样本且至少有 5 个达标日。输出同样本日均 Rank IC、原收益符号诊断、各目标的方向度量、高置信覆盖、Top-10/20、名单更替和交易成本诊断。5、10、25 bp 为单边假设，每个持有批次扣一次往返成本。五日持有批次可能重叠。

前瞻评价从模型及协议冻结后的首个及时捕获日开始。至少积累 20 个持有期已结束的交易日且标签覆盖率达 80%，才检查排序、成本后 Top-20 超额及标签覆盖。本次历史比较只作为开发诊断。README 和服务器产物记录实际数据、模型散列和本次测试结果。
