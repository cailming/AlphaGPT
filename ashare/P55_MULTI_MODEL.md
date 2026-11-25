# P5.5-B/C 六方法研究账本

原三套 V2 模型的权重、计划和逐日预测文件维持原有冻结目录。六方法报告引用原账本的散列，新增账本只发布 18 因子 Transformer、MA20 反转和等权基准。通过同一套夜间任务读取正式行情，一次增量公司行为观察同时服务原模型和新增方法。

## 冻结与运行

在项目根目录及 py312 环境下填入已核验的服务器路径。首先冻结新增账本规则和模型。现有 cron 调用夜间入口时传入相同目录，使用者应沿用原研究任务和 PostgreSQL 只读连接。

```bash
conda run --no-capture-output -n py312 python -m ashare.p55_panel freeze \
  --panel "$panel" --factor-model "$previous/stable_models/factors18" \
  --factor-source "$previous/paired18"
conda run --no-capture-output -n py312 python -m ashare.p55_nightly \
  --source "$p5/development" --models "$p5/models" --evidence-seed "$p5/live_evidence" \
  --ledger "$previous/ledger" --panel-ledger "$panel" \
  --factor-model "$previous/stable_models/factors18" --factor-source "$previous/paired18"
```

18 因子的因果计算、列序、归一化与训练一致。新增方法以已通过原研究资格的股票为基础，18 因子再逐条检查完整的历史特征序列。MA20 反转为原 P4 的负收盘价均线偏离。等权记录为全篮子基准，等权本身没有排序。各方法独立保留预测和弃权，交集比较在查看成熟标签前依据日账本冻结。

新增方法只接纳各自规则和权重冻结之后、同日盘后完成的记录为前瞻。原 V2 自己的前瞻起点保持原协议。盘后首次运行前可显式补跑历史账本。补跑数据始终与及时数据分开。

```bash
conda run --no-capture-output -n py312 python -m ashare.p55_panel run \
  --ledger "$previous/ledger" --panel "$panel" --factor-model "$previous/stable_models/factors18" \
  --factor-source "$previous/paired18" --asof 2026-09-23
conda run --no-capture-output -n py312 python -m ashare.p55_reports \
  --ledger "$panel" --legacy-ledger "$previous/ledger" --evidence-seed "$p5/live_evidence/dataset" \
  --out "$new_report"
```

## 读报告

归档里的 `daily.md` 汇总当日正式行情、候选股、各方法评分和逐股弃权原因，并展示各方法保存的研究名单。`cumulative.csv` 保存每个日期、方法、支持度分组与评价口径的计数和指标。`prediction_labels.parquet` 保留预测及成熟标签状态。`panel_report_audit.json` 是完成标记，包含报告文件散列。

在信号的 T+6 开盘路径可计算时，按 T+1 至 T+6 开盘计算经济收益。持有窗口内需有独立行为证据，异常、证据缺口、未成熟状态和零预测日期都保留。Top-10／20 按发布时的分数固定。原名单不足或部分标签缺失时保留不完整计数。成本按单边假设扣除，等权收益基于各日共同可评价股票。报告分别给出独立覆盖、共同覆盖、日均 Rank IC、Top-20 净超额、更替率，以及 0–4、5–19、20+ 支持度分组。

初步前瞻检查要求至少 20 个真实成熟交易日、80% 标签覆盖，并显示能计算 Rank IC 与完整 Top-20 的有效日期数。五日持有窗口可能重叠，报告用日期顺序的自相关稳健区间估计不确定性，并单独统计互不重叠的窗口数。通过时间和覆盖门槛意味着可以人工复核。模型与预测均限研究用途。
