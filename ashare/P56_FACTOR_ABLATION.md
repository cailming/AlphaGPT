# P5.6 市场环境因子消融

六组实验使用固定十因子与十八因子窗口的共同样本。两侧的标签、交易日边界、序列索引、原十个特征和归一化数值须一致。十八因子输入投影保留同一形状与随机初始化。基线和分组消融在训练集标准化后将未启用的输入置零。各组沿用同一网络、随机种子、批量大小、AdamW、学习率和早停规则。最佳轮次只依据验证集 Rank IC。

| 实验 | 新增特征 |
|---|---|
| base10 | 无 |
| group_a | 市场 5 日和 20 日趋势、上涨家数占比 |
| group_b | 市场成交活跃度 |
| group_c | 个股相对市场强弱、突破 |
| group_d | 缩量、缩量回踩 |
| full18 | 全部 8 个新增因子 |

```bash
conda run --no-capture-output -n py312 python -m ashare.p56_ablation verify \
  --base "$previous/paired10" --source "$previous/paired18"
for arm in base10 group_a group_b group_c group_d full18; do
  conda run --no-capture-output -n py312 python -m ashare.p56_ablation train \
    --base "$previous/paired10" --source "$previous/paired18" \
    --out "$run/ablation/models/$arm" --arm "$arm"
done
conda run --no-capture-output -n py312 python -m ashare.p56_ablation compare \
  --base "$previous/paired10" --source "$previous/paired18" --universe "$stable/expanded" \
  --models "$run/ablation/models" --out "$run/ablation/diagnostics"
```

`ablation_comparison.json` 记录六组覆盖、Rank IC、Top-10／20、成本、更替率及权重散列。输入包括此前见过的历史时期，结果用于开发诊断。原冻结 V2 和 18 因子每日模型继续使用原权重。

八个因子另有独立诊断。前四个市场状态因子同日对所有股票取值一致，记录其日度时间序列 Rank IC 和按日期分组的等权收益。其余四个个股因子记录同日横截面 IC 和分位组收益。覆盖表分别统计信号锚点和完整序列，各月稳定性、因子 Spearman 相关性、成对相关样本数和分组收益保存在 CSV。覆盖不足或并列导致的未定义指标留空。
