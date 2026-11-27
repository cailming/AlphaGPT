# C 组独立前瞻研究

`python -m ashare.p58_group_c freeze --out C_LEDGER --panel SIX_PANEL --model GROUP_C_MODEL --source PAIRED18` 在独立目录记录真实规则冻结时间、P5.6 权重与元数据散列、特征顺序、训练集归一化参数及 12 个有效输入的掩码。重复冻结校验依赖一致性。训练时间由原权重的元数据给出。

同一组参数下运行 `capture --legacy V2_LEDGER`。当天已有六方法原始记录、已过 18 点时，读取经过散列校验的当日候选和公司行为证据，并从 PostgreSQL 只读复核行情。沿用消融实验完整 16×18 序列资格，对完整输入按训练集参数归一化后，将掩码外的 6 列置零，再使用已训练的 C 组权重评分。弃权仍在候选文件中。每次运行先校验已有成果，按日期原子发布。指定 `--replay YYYY-MM-DD` 始终在 historical 下保存独立记录。

`python -m ashare.p58_group_c report --out C_LEDGER --panel SIX_PANEL --legacy V2_LEDGER --asof YYYY-MM-DD` 读取已完成的六方法标签评价。新报告通过原预测 ID 匹配既有标签，C 组通过同一原始候选的股票和交易日匹配。报告同时列出七种方法自身覆盖和七方共同股票的评价。共同股票集合在查看收益之前确定。配对报告的日期以 C 组采集通道为准，历史补跑始终列为开发诊断。逐日输出存于 `reports/YYYY-MM-DD/daily_comparison.csv`，累计结果及不确定性存于 `comparison.json`。

真实前瞻从晚于模型和实验规则冻结日期、且成功及时采集的正式交易日起算。报告列出空名单和漏记交易日；评审沿用 P5.5 的标签覆盖、完整 Top-20、有效 IC 日期与重叠收益统计协议。配对评价不会更改原六种方法的报告。
