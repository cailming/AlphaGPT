# P5.3-D/E、P5.4、P5.5 运行说明

## 月度研究股票池与样本覆盖

本流程从已有正式日线和公司行为档案起步。研究池默认 400 只。每月用月初之前最多 63 个正式交易日的历史资格和流动性排名选股。最多 300 个名额优先留给仍达标的上月成员，其余名额按排名补充，上月成员也可通过排名入选。每个月的入池依据与观测截止日独立归档。成员仍须通过信号当日的风险、价格、特征和连续性检查。

在仓库根目录的 py312 环境运行。P5 数据根、历史归档和输出目录均应为独立路径。下例用已有 P5 冻结行情，保留最近验证与开发诊断边界。

```bash
conda run --no-capture-output -n py312 python -m ashare.p53_dataset plan \
  --source "$p5/rolling" --supplement "$p5/live_evidence" --out "$stable" \
  --validation-start 2026-06-12 --diagnostic-start 2026-08-17 --capture-limit 120
conda run --no-capture-output -n py312 python -m ashare.p53_dataset build \
  --plan "$stable" --out "$stable/archive_only" --v1 "$p4_subset"
conda run --no-capture-output -n py312 python -m ashare.p53_dataset capture --plan "$stable"
conda run --no-capture-output -n py312 python -m ashare.p53_dataset build \
  --plan "$stable" --out "$stable/expanded" --v1 "$p4_subset"
```

增量计划只针对月度研究池里缺少双源覆盖的区间，并保留已签名的原始观察。某个区间已有一个来源成功、另一个失败时，双源补查仍可能向已成功的来源请求重叠日期。若证据缺口持续，样本保留原拒绝原因。输出包括逐月成员、每日可推理与可评价股票数、逐股每日拒绝原因、分段成熟边界、训练专用归一化和审计散列。训练及标签诊断评估均属于研究用途。

## 市场环境因子与精确配对

新增 8 项特征涵盖市场 5／20 日趋势、上涨家数占比、成交变化、个股相对强弱、突破、缩量和缩量回踩。市场聚合按同日和前一正式交易日有效成对记录计算，覆盖不足的日期对应窗口会缺失。特征只使用信号收盘及以前已可见的正式日线。两组同时剔除受缺失因子影响的序列，标签字节和索引字节保持一致，归一化参数分别在配对训练序列上计算。

```bash
conda run --no-capture-output -n py312 python -m ashare.p54_factors \
  --source "$stable/expanded" --out "$run/paired18" --paired-control "$run/paired10"
for arm in absolute excess rank factors18; do
  conda run --no-capture-output -n py312 python -m ashare.p53_experiments train \
    --source "$run/paired10" --expanded "$run/paired18" --out "$run/stable_models/$arm" --arm "$arm"
done
conda run --no-capture-output -n py312 python -m ashare.p53_experiments compare \
  --source "$run/paired10" --expanded "$run/paired18" --models "$run/stable_models" --out "$run/stable_compare"
```

十因子组用绝对收益、当日合格股票池超额和中心化百分位三个目标。十八因子组用绝对收益。隐藏层结构、优化配置和验证选轮规则沿用 P5，仅调整输入投影维度。同一样本另跑验证定正则的 Ridge、MA20 反转和等权池。历史诊断只用于开发，比较报告既保留过滤前的样本数，也记录真实比较的交易日和覆盖。

完整稳定池十因子对照可使用同一 `train` 和 `compare` 命令。将 `--source` 指向 `"$stable/expanded"`，省略 `--expanded`，使用新的模型与比较输出目录。这样保留因十八因子缺失而无法纳入配对实验的合格十因子序列。

## 冻结模型每日记录与经济标签成熟

日常记录使用原三套 V2 冻结权重及其资格协议。收盘后读取 PostgreSQL 只读正式日线。增量请求限于已核验研究名单和仍需标签结算的及时预测股票。单源错误保留原覆盖截止，不延长观察区间。每个正式交易日保存一份不可覆盖的清单，逐模型保存股票、预测 ID、分数、训练支持度、权重散列及逐股弃权原因。

```bash
PGOPTIONS='-c default_transaction_read_only=on' conda run --no-capture-output -n py312 \
  python -m ashare.p55_nightly --source "$p5/development" --models "$p5/models" \
  --evidence-seed "$p5/live_evidence" --ledger "$run/ledger"
```

ubuntu92 的研究任务在工作日北京时间 18:10 开始检查正式日线，未完成时后续在 19:10 至 22:10 及 23:35 检查。无行情的早期尝试只写探测记录。若收盘后当天正式日线仍不可见，最终记录原状态。成功提交后同日重复执行校验旧清单。服务从进程环境或受限访问的 PostgreSQL 凭据文件读取连接配置。

`p55_daily --asof` 是历史补跑专用，独立写入历史命名空间。盘后实际及时记录并且信号日期在所有模型冻结日期之后，才计入前瞻评价。同日冻结、历史补跑、零预测和无正式行情均保留在各自轨道。

`p55_mature` 在真实 T+6 开盘路径可见后尝试成熟原预测 ID。收益按 T+1 开盘至 T+6 开盘计算，遇到缺行情、未核实行为或证据不连续时保留缺失原因。Top-10／20 先依据原分数取名单，再检查标签覆盖。报告按 0–4、5–19、20+ 条训练序列分组，包含 Rank IC、成本后超额、覆盖和更替率。前瞻门槛为至少 20 个有成熟标签的交易日，且成熟预测的标签覆盖率不低于 80%。任一历史样本不得计入该门槛。

完整的本次实测、版本和目录见根目录 README 看板。生产行情库的查询保持只读，所有预测产物均为研究记录。
