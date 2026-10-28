# task1-7B：时间基、多尺度信号与相位候选共识

## 1. 执行结论

状态：**实现完成；Fit3D dev 阶段门通过，但 Fit3D regression holdout
为 `SIGNAL_STAGE_NO_GO`（2026-10-27）**。holdout 基线 `29/31`、
FPS 15 `28/31`、FPS 60 `29/31`、noise σ=2 `27/31`，并有 `7` 个变换
相对 T1-6G 发生结构变化。因此不进入 T1-7C，也不构成正式跨域 `GO`。

本轮已完成真实时间基、短/长缺口证据、多尺度相位候选、Top-K 假设、
左右侧与关节族一致性检查、事件级跨尺度共识、部分端点链闭合，以及
低采样半周期恢复、低幅度节点上下文桥接、局部周期缺口补边和端点
时序模型、噪声下的局部峰/谷稳定性桥接和长 partial 尾端剪枝。FPS 15
从 `24/32` 提升到 `32/32`，FPS 60 从 `29/32` 提升到 `30/32`，
基线保持 `32/32`；噪声 σ=2 从 `27/32` 提升到 `30/32`，全部 dev
阶段门通过。剩余噪声失败为两个 `s03` pushup 样本。

> P1 后续更新：采样感知运动门和近边界局部恢复已实现，dev/holdout
> 的最新数字与结论见本文第 10 节和
> [`task1-7b_p1_results.md`](task1-7b_p1_results.md)。

## 2. 实现范围

- `scripts/task1_7_signals.py`
  - 恒定 FPS、真实时间戳、可变 FPS 检测和 10–60 FPS 重采样；
  - 短缺口显式插值；孤立长缺口记录为 `degraded`，检测时显式插值，
    不静默当作完整观测；
  - `0.12 / 0.2 / 0.32 / 0.48s` 多尺度候选与周期、SNR、有效比例、
    缺口质量和尺度一致性评分；
  - 双侧均值优先、可配置的同族先验和 Top-K 相位假设；
  - 关节族周期冲突时返回 `phase_ambiguous`，零 set/rep 输出。
  - 多尺度 top/bottom 事件按周期容差聚类，输出 `scale_support`、
    尺度成员和锚点离散度；半周期内重复事件按尺度支持与平台稳定性
    择优保留，避免平滑窗口改变局部峰数量；
  - 保留选定尺度独有锚点作为闭合证据；低支持端点只有在更窄的
    `0.65–1.35` 周期窗内才可连接，不能独自成立链；
  - 相邻 top 间隔落在 `1.55–2.45` 周期时预测缺失半周期，并从各尺度
    的 singleton 候选中恢复 top；恢复事件必须同时满足相邻周期比、
    锚点容差和稳定性证据，写入 `half_cycle_recovered`、
    `recovery_scale_seconds`、`recovery_anchor_error_frames` 和
    `recovery_confidence`；
  - 相邻 top 之间没有 bottom 时，从各尺度 trough 候选中恢复一个
    bottom，使恢复后的 top 能构成完整 rep；恢复事件强制
    `multi_scale_consensus=false`，不会冒充多尺度共识；
  - 低采样（分析 FPS `<=20`）下，仅当链首尾都是多尺度共识、且较低
    稳定性端点 `<=0.6` 时，才进入端点剪枝候选；若端点时序模型判定该
    partial 端点间隔与全局/局部节拍一致，则保留，单尺度闭合端点不剪；
  - 端点时序模型使用组内 top-to-top 近邻间隔中位数、相对 MAD、端点
    全局/局部间隔比、端点 kind、稳定性和 bottom 支持度。partial 端点
    的全局间隔在 `0.9–1.1` 周期、局部间隔在 `0.72–1.08` 倍中位数且
    上下文稳定时标记为可保留；partial 端点局部间隔低于 `0.75` 倍且
    上下文稳定时标记为短周期重复端点并剪掉；局部间隔高于 `1.45` 倍
    时标记为长 partial 离群端点并剪掉。冻结阈值不读取数据集、相机、
    clip ID 或固定 rep 数；
  - 局部峰/谷稳定性桥接只处理未进入链的 `leading_partial` /
    `trailing_partial` 端点。端点须为多尺度共识且 top 稳定性过门；
    leading 端点的局部幅度按“该峰到下一个稳定谷”计算，而不是只复
    用峰前已分配的 bottom。局部幅度按组内 top-bottom 幅度中位数归一
    化，下一谷还须满足稳定性、多尺度支持度和 `0.75–1.35` 周期间隔
    门。leading/trailing 各最多恢复一个节点，并写入
    `local_extrema_stability_bridged`、局部相对幅度和 bottom 稳定性
    证据；
  - 低幅度节点上下文桥接只作用于分析 FPS `<=20`、多尺度一致但相对
    幅度低于节点门的 `interior` 节点。节点必须能扩展一个已经可成立的
    链；若加入节点后左右两侧都能独立成立，则视为可能跨 set 合并并
    拒绝。桥接节点保留 `low_amplitude_context_bridged`、
    `bridge_confidence` 和允许连接的上下文邻居，不提高原始节点幅度；
  - 局部周期缺口只补标准全局边窗拒绝的相邻 top 边。要求缺口左右各
    有连续可信边，使用左右邻域间隔的稳健中位数估计局部周期；局部
    周期离散度和缺口周期比仍须过门，边质量仍使用原边质量门。补边
    记录全局/局部周期、上下文间隔、离散度和质量，不放宽全局
    `gap_period_range`；
   - 单尺度链与共识链作为两个假设仲裁：共识需至少补齐 2 个 rep，
    或只增加低支持 leading partial，或只增加已选中的稳定局部峰，
    或净减少能由低采样端点清理/端点跨度解释；任意未解释的单 rep
    漂移仍被拒绝。未选中的其他 consensus state 不能冒充新增事件；
  - 同族 prior 命中当前最高分候选时，不再仅因跨关节族原始分数接近而
    返回 `phase_ambiguous`。
- `scripts/task1_6_chain_selection.py`
  - 新增可选 `phase_resolution` 接线；不传时保持 T1-6 原路径不变；
  - 新增可选 `phase_state_override` 和链策略接线；不传时保持 T1-6
    原路径不变；
  - 新增低幅度上下文桥接：先用现有边识别链分量，只允许低幅度节点
    扩展单侧可成立链；桥接后的边按允许上下文邻居裁剪，避免把两个
    已成立的 set 连成一个；
  - 新增局部周期缺口补边：只在相邻 top 的标准边缺失时，从两侧连续
    可信边估计局部周期并重建候选边；没有双侧上下文或上下文不稳定
    时拒绝；
  - 新增端点时序评估与剪枝：保留 `evaluate_endpoint_timing()` 的
    leading/trailing 两侧证据，短周期 partial 端点可在任意分析 FPS
    下剪除；长 partial 端点局部间隔超过 `1.45` 倍邻域中位数时剪除；
    低采样弱端点只有在时序证据不支持保留时才剪；
  - 新增局部峰/谷稳定性桥接：只启用多尺度一致、稳定性过门且具有
    局部稳定谷支持的 partial 端点；leading 使用峰到下一谷的局部幅度，
    避免把峰前归入其他周期的 bottom 当成峰谷幅度；
  - `ambiguous` 直接短路为 `phase_ambiguous`，不按最高分强行选相位；
  - 端点闭合边、共识事件数、端点闭合计数、低采样端点剪枝、端点时序
    决策/剪枝、局部峰谷稳定性桥接和双假设与局部周期缺口写入
    diagnostics；桥接、局部周期和剪枝策略未传入时 T1-6 行为不变；
    单尺度侧只接收端点时序子策略，不会因策略透传而改变原有端点闭合边。
- `configs/task1_7_signal.json`
  - `dataset_independent=true`；所有阈值为全局配置，不读取数据集、
    相机或 clip ID，也不使用固定 rep 数；
  - 新增 `event_consensus` 配置，冻结周期/锚点容差、重复峰间距、
   端点闭合窗、半周期恢复窗、低采样端点剪枝、端点时序保留/短周期
    剪枝阈值、长 partial 离群阈值、局部峰谷稳定性门、低幅度节点桥接、
    局部周期缺口和双假设仲裁容差。
- `tests/test_task1_7_signals.py`
  - 覆盖恒定/可变 FPS、逆序时间戳、短/长缺口、候选选择、髋膝冲突、
    先验同族选择、链选择接线、ambiguity 短路、周期共识、跨尺度事件
    聚类、重复峰压制、选定尺度锚点保留、端点闭合、纯端点拒绝和
    双假设仲裁；新增缺失 top/bottom 恢复、远锚点拒绝、正常间隔不
    改写、低采样端点剪枝、时序一致 partial 端点保留、低采样 partial
    离群端点剪枝、高采样短周期 partial 剪枝、长 partial 尾端剪枝、
    稳定 leading 局部峰桥接、伪谷拒绝、正常链不改写、单尺度端点保护、
    低幅度链尾桥接、孤点拒绝、跨 set 合并拒绝、足够采样关闭桥接、
    局部周期补边、双侧上下文要求、上下文离散度拒绝、局部峰仲裁和
    可解释端点对账测试。

## 3. Fit3D dev 结果

评测规模：`32` 个 clip，`8` 个 subject-action cluster。结构为
`(set 数 / rep 数 / 默认组 rep 数)`，质量排除数为 `0`。

| 变换 | 结构一致 | 平均 rep delta | phase ambiguous | 结果 |
|---|---:|---:|---:|---:|
| baseline | `32/32` | `0.000` | `0` | pass |
| amplitude 0.8 | `32/32` | `0.000` | `0` | pass |
| amplitude 1.2 | `31/32` | `+0.031` | `0` | 新增 1 个失败 |
| offset +15 | `32/32` | `0.000` | `0` | pass |
| flip LR | `32/32` | `0.000` | `0` | pass |
| FPS 15 | `32/32` | `0.000` | `0` | **pass（阶段门）** |
| FPS 60 | `30/32` | `-0.063` | `0` | **pass（阶段门）** |
| noise σ=2 | `30/32` | `+0.094` | `0` | **pass（阶段门）** |
| missing 5% | `30/32` | `+0.063` | `0` | 新增 2 个失败 |
| idle pad 2s | `28/32` | `+0.094` | `0` | 新增 4 个失败 |

阶段门：

| 门 | 观测 | 目标 | 结果 |
|---|---:|---:|---:|
| baseline 结构保持 | `32/32` | `32/32` | pass |
| FPS 15 结构一致 | `32/32` | `>=30/31` | pass |
| FPS 60 结构一致 | `30/32` | `>=30/31` | pass |
| noise σ=2 结构一致 | `30/32` | `>=30/31` | pass |
| 质量排除不新增 | `0` | baseline `0` | pass |

dev 没有 T1-6G 的 robustness summary，因此本轮不生成
`no_regression_vs_t1_6g` 门；该门留到 holdout。dev 阶段门已通过，
但正式跨域结论仍须用冻结配置运行 regression holdout，且不得用 holdout
调参或后续阶段补偿性放宽阈值。

## 4. Fit3D regression holdout

使用未修改的 `configs/task1_7_signal.json` 运行，配置 SHA-256 为
`e4fc5abb95f561765e4add7c0e05d72c03ad4f190d08fec13dd0a425ed19bc49`。
holdout 输入 `32` 个 clip，其中 `fit3d_s09_pushup_50591643` 被质量门
排除，eligible 为 `31`；质量排除数与 T1-6G 相同。

| 变换 | T1-7B | T1-6G | 平均 rep delta | 结果 |
|---|---:|---:|---:|---:|
| baseline | `29/31` | `31/31` | `+0.032` | fail |
| amplitude 0.8 | `29/31` | `30/31` | `+0.065` | regression |
| amplitude 1.2 | `29/31` | `31/31` | `+0.032` | regression |
| offset +15 | `29/31` | `31/31` | `+0.032` | regression |
| flip LR | `29/31` | `31/31` | `+0.032` | regression |
| FPS 15 | `28/31` | `26/31` | `-0.194` | 改善但仍 fail |
| FPS 60 | `29/31` | `28/31` | `-0.097` | 改善但仍 fail |
| noise σ=2 | `27/31` | `28/31` | `+0.032` | regression |
| missing 5% | `28/31` | `30/31` | `+0.065` | regression |
| idle pad 2s | `29/31` | `31/31` | `+0.032` | regression |

阶段门：

| 门 | 观测 | 目标 | 结果 |
|---|---:|---:|---:|
| baseline 结构保持 | `29/31` | `31/31` | fail |
| FPS 15 结构一致 | `28/31` | `>=30/31` | fail |
| FPS 60 结构一致 | `29/31` | `>=30/31` | fail |
| noise σ=2 结构一致 | `27/31` | `>=30/31` | fail |
| 相对 T1-6G 不回归 | `7` 个变换 | `0` | fail |
| 质量排除不新增 | `1` | T1-6G `1` | pass |

失败集中在两个共同结构差异：

- `fit3d_s10_pushup_58860488`：T1-6G `1/5/5` → T1-7B `1/4/4`，
  在 baseline、幅度、偏移、翻转、idle、missing 和 noise 下均出现；
- `fit3d_s11_pushup_58860488`：T1-6G `1/3/3` → T1-7B `1/5/5`，
  同样出现在多数变换；该变化增加了 rep，但按“相对冻结 T1-6G 结构”
  的定义仍计为 regression，而不是相对 GT 的回归。

变换特有失败：

- FPS 15：`fit3d_s08_pushup_50591643` `7→5`、
  `fit3d_s11_pushup_50591643` `5→4`，以及
  `fit3d_s11_pushup_58860488` 变为 `no_motion` 的 `3→0`；
- FPS 60：`fit3d_s08_pushup_50591643` `7→5`、
  `fit3d_s11_pushup_50591643` `5→4`；
- noise σ=2：`fit3d_s10_pushup_50591643` `7→6`、
  `fit3d_s11_pushup_50591643` `5→6`，并叠加两个共同差异；
- missing 5%：`fit3d_s08_pushup_65906101` `4→5`，并叠加两个共同差异。

局部峰谷稳定性桥接在 holdout 的 noise σ=2 上未触发，只在 missing 5%
中触发 `2` 次，说明 dev 上针对噪声端点的修复没有迁移到 holdout 的
主要失败型态。holdout 已被观测且当前没有独立的新 holdout，因此本轮
只把它作为 regression 结果，不据此调整阈值。

## 5. dev 失败归因

本轮的半周期恢复、端点时序模型和局部周期策略解决了 FPS 15/60 的
主要拓扑问题：

- `fit3d_s05_pushup_58860488` 从 `1/3/3` 恢复为 `1/6/6`：`0.48s`
  尺度找回约第 `94` 帧缺失 top，并由相邻 top 间 trough 补齐末端
  bottom；
- `fit3d_s03_pushup_58860488`、`fit3d_s07_pushup_50591643`、
  `fit3d_s07_pushup_58860488` 各多计 1 rep，低采样端点剪枝后分别恢复
  目标结构；
- `fit3d_s07_squat_65906101` 不再因膝/髋跨族分数接近而短路，同族
  prior 抑制了错误的 `phase_ambiguous`；
- `fit3d_s04_pushup_58860488` 的单尺度链少 1 rep，共识链通过低支持
  leading partial 闭合补齐到 `1/5/5`；
- `fit3d_s03_pushup_50591643` 的低幅度链尾节点满足多尺度支持、稳定性
  和单侧链扩展条件，桥接后从 `1/3/3` 恢复到目标 `2/6/3`；位于两个
  已成立链之间的低幅度节点会被拒绝，避免把两个 set 合并；
- `fit3d_s04_pushup_50591643` 的 `69→112` 缺口在全局周期 `24` 帧下为
  `1.79` 周期，但两侧连续可信边给出的局部周期为 `30.5` 帧，局部周期
  比 `1.41`、相对 MAD `0.131`；补边后该 clip 从 `1/3/3` 提升到
  `1/6/6`；端点时序模型进一步确认链尾 `216` 的 `24` 帧间隔等于全局
  周期、且局部间隔为中位数的 `0.787` 倍，因此拒绝低采样剪枝，恢复为
  `1/7/7`；
- `fit3d_s04_pushup_50591643` 的 FPS 60 链尾 `1074` 为 partial，但
  `82` 帧间隔仅为局部中位数的 `0.661` 倍，被端点时序模型判为短周期
  重复端点并剪掉，从 `1/8/8` 恢复为 `1/7/7`；
- `fit3d_s04_pushup_58860488` 在噪声 σ=2 下由局部峰 `186` 恢复：
  峰到下一稳定谷的归一化幅度为 `1.094`，谷稳定度 `0.763`、尺度支持
  `4`，结构恢复到 `1/5/5`；
- `fit3d_s05_pushup_58860488` 在噪声 σ=2 下由局部峰 `228` 恢复：
  局部幅度 `1.210`，谷稳定度 `0.901`、尺度支持 `4`，结构恢复到
  `1/6/6`；
- `fit3d_s07_pushup_50591643` 在噪声 σ=2 下由局部峰 `284` 补回
  稳定节点，同时链尾 `676` 的局部间隔为中位数的 `1.548` 倍，被长
  partial 规则剪除，结构恢复到 `1/5/5`；
- 局部峰桥接会拒绝幅度不足的伪峰：`s04_pushup_58860488` 的 `108`
  局部幅度约 `0.078`、`s05_pushup_58860488` 的 `147` 约 `0.465`、
  `s07_pushup_50591643` 的 `235` 约 `0.048`，均未进入链；
- 基线、幅度、偏移、翻转和缺失 5% 没有新增质量排除或结构回归。

剩余问题仍集中在 pushup 的链拓扑与噪声：

- FPS 15 已达到 `32/32`；端点时序模型保留 `s04` 链尾 `216` 后补齐
  最后 1 rep；
- FPS 60 仍有 2 个变化：`fit3d_s03_pushup_50591643` 仍是
  `2/6/3 → 1/3/3` 的 set 合并/漏检，`fit3d_s07_pushup_58860488`
  仍是 `1/4/4 → 1/5/5` 的尾端 1-rep 漂移；
- noise σ=2 下仍有 2 个变化：
  `fit3d_s03_pushup_50591643` 为 `2/6/3 → 1/8/8`，仍有跨 set 拓扑
  与多计 rep；`fit3d_s03_pushup_58860488` 为 `1/3/3 → 1/4/4`，仍有
  尾端多计 1 rep；
- 一个孤立长缺口案例（膝/髋角度各有约 `0.9–1.5s` 缺失）已改为
  显式 `degraded` 并保留缺口证据，避免了“全候选 insufficient”的
  基线回归，但仍不足以解决低采样下的状态拓扑变化。

当前恢复只使用“已有尺度中真实存在的 singleton 峰/谷”；低幅度桥接
和局部峰谷稳定性桥接也只重新启用已有多尺度一致、通过稳定性和局部
上下文复核的节点，不会凭空生成波形，也没有降低 candidate/edge 全局
门槛。因此它解决了可观测但未进入链的欠采样节点、可解释的局部长间隔
和噪声下仍有稳定谷支撑的 partial 峰；尚不能解决所有尺度都未形成稳定
候选的强噪声、幅度塌缩、跨 set 拓扑合并，以及没有可靠邻域上下文或
稳定谷支撑的长间隔场景。

## 6. 产物

```text
scripts/task1_7_signals.py
configs/task1_7_signal.json
tests/test_task1_7_signals.py
docs/task1-7b_signal_results.md

results/task1_7/signals/dev/
├── robustness.csv
├── signal_evidence.csv
├── summary.json
└── checksums.sha256

results/task1_7/signals/holdout/
├── robustness.csv
├── signal_evidence.csv
├── summary.json
└── checksums.sha256
```

复现：

```bash
cd /mnt/storage2/data_guest/VLS/V0
/mnt/storage2/data_guest/.venv/bin/python scripts/task1_7_signals.py \
  --evaluate --split dev \
  --output-dir results/task1_7/signals/dev
cd results/task1_7/signals/dev && sha256sum -c checksums.sha256

/mnt/storage2/data_guest/.venv/bin/python scripts/task1_7_signals.py \
  --evaluate --split holdout \
  --config configs/task1_7_signal.json \
  --output-dir results/task1_7/signals/holdout
cd results/task1_7/signals/holdout && sha256sum -c checksums.sha256
```

校验：T1-7B 定向测试 `71 passed, 4 subtests passed`；全量测试
`207 passed, 4 subtests passed`，忽略已知 TensorFlow/protobuf 环境
问题的 T1-4 clip/report 两项。

## 7. holdout 失败审计

详细审计见
[`task1-7b_holdout_audit.md`](task1-7b_holdout_audit.md)。审计未修改
算法、配置、阈值或冻结产物，结论仍为 `SIGNAL_STAGE_NO_GO`。

- holdout 共 `24` 条结构变化记录，集中在 `6` 个 clip；`31/31` 质量
  排除口径未变，仍然只排除 `fit3d_s09_pushup_50591643`。
- `structure_match` 是相对冻结 T1-6G 的结构比较，不是 GT 精度。
  `s11_pushup_50591643` noise 和 `s10_pushup_50591643` noise 在 GT
  对齐后并不比 T1-6G 差；`s08_pushup_65906101` 新增 leading `174`
  也与 GT `163` 匹配，不能直接算作桥接误报。
- 真实回归集中在：`s10_pushup_58860488` 的端点跨度清理丢掉 GT 可匹配
  leading 事件；`s11_pushup_50591643` FPS 15/60 选中了结构更差的右侧
  相位信号；`s08_pushup_50591643` FPS 15/60 的半周期恢复硬边界漏掉
  leading top。
- 通用层缺陷按优先级为：共识仲裁缺少事件保留不变量；端点跨度清理缺少
  显式删除证据；相位选择缺少事件级一致性；运动门的
  `state_amplitude` 对采样率过敏；半周期恢复硬边界。局部峰/谷桥接在
  `noise_sigma_2` 上触发 `0` 次，dev 的噪声修复没有迁移到 holdout。

下一轮修复先做重叠/保留、显式剪枝和事件一致性等不变量，再讨论
采样感知运动门和端点边界；当前 holdout 已被观测，只能作为 regression
集，不能用于调参。

## 8. 下一阶段建议

T1-7B holdout 未通过，当前停留在 T1-7B，不进入 T1-7C：

1. 保留本次冻结配置、CSV、summary 和 checksum，作为 regression 基线；
2. 先审计两个共同结构差异 `s10_pushup_58860488` 和
   `s11_pushup_58860488`，区分 T1-6G 参考结构、相机视角和当前链选择
   各自的责任，不直接改阈值；
3. 再处理 FPS 15/60 的 `s08/s11_pushup_50591643` 低采样漏检和
   `s11_pushup_58860488` 的 `no_motion` 拒答；
4. 修复只能在 dev/train 或新锁定的未见 holdout 上验证；当前 holdout
   已被观测，不能继续将其当作独立泛化证据。

## 9. P0 不变量修复回归

审计提出的三个 P0 不变量已实现并通过定向测试：共识净增事件必须保留
单尺度事件、端点删除必须有显式剪枝证据、单侧相位候选必须有同族双侧
事件支持。使用未修改的冻结配置重跑后，holdout baseline 从 `29/31`
恢复到 `31/31`，noise σ=2 从 `27/31` 恢复到 `28/31`，
`no_regression_vs_t1_6g` 从 `7` 个变换降到 `2` 个；FPS 15/60 和
noise 绝对门仍未通过，决策继续保持 `SIGNAL_STAGE_NO_GO`。

`fit3d_s10_pushup_58860488` 已恢复 `1/5/5`，
`fit3d_s11_pushup_58860488` baseline 不再接受错位共识，
`fit3d_s11_pushup_50591643` FPS 15/60 改为事件级相位歧义拒答。详细
结果见 [`task1-7b_invariant_results.md`](task1-7b_invariant_results.md)。

下一步进入 P1：采样感知运动门与半周期局部上下文；当前 holdout 只作
regression，不用于调参。

## 10. P1 采样感知运动门与半周期局部上下文

P1 已实现并完成回归（2026-10-27）。使用独立配置
`configs/task1_7_signal_p1.json`，P0 冻结配置与产物保持不变。

- 采样感知运动门只在 `state_amplitude` 是唯一失败原因、分析 FPS
  `<=20`，且 amplitude、dynamic range、SNR、periodicity 联合过门时
  生效；
- 近边界 top 恢复只接受多尺度 `no_stable_top_state` 拒峰，并要求
  局部 gap、锚点离散、峰值幅度和稳定性同时成立，不直接扩大候选门；
- dev 为 baseline `32/32`、FPS 15 `32/32`、FPS 60 `28/32`、
  noise `30/32`，失败集合与 P0 完全一致；P1 机制在 dev 未触发；
- holdout 为 baseline `31/31`、FPS 15 `29/31`、FPS 60 `30/31`、
  noise `28/31`。FPS 60 比 P0 提升 `1` 个案例，但阶段门仍未通过，
  决策保持 `SIGNAL_STAGE_NO_GO`；
- `s11_pushup_58860488` FPS 15 由运动门恢复为 `1/3/3`；
  `s08_pushup_50591643` FPS 60 由近边界恢复为 `1/7/7`。同一 clip
  FPS 15 为 `1/6/6`，leading top `21` 仍因 stability `0` 被端点
  剪枝，保持 fail-closed。
- 研究阶段仍为 `SIGNAL_STAGE_NO_GO`，但不阻塞产品真实视频试用：按
  `Plan/task1-7.md` 第 18 节，T1-7B 冻结为干净输入
  `LIMITED/EXPERIMENTAL` 基线，后续先做 C-lite/E-lite 和真实视频验收，
  不再继续追合成 holdout 门。

详细实现、触发证据和复现命令见
[`task1-7b_p1_results.md`](task1-7b_p1_results.md)。
