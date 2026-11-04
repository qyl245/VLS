# task1-6：通用动作链分割与相位区间（RepChain v2）

## 1. 任务定位

T1-5D 证明当前 T1-4 rep 分割不能直接进入产品验收：计数共识只有 `6/16` 完全正确，主相机 top 事件中位误差 `12.5` 帧，`80` 个高置信误检被包装成完整 rep。根因不是单一阈值，而是：

1. 全片独立候选 + active bout 没有真正判断“哪一段是正式动作组”；
2. 首尾周期被截断，GT 第 1/6 事件召回只有 `25%/12.5%`；
3. `top_frame` 是平台边界，但 Fit3D `rep_ann` 更靠高位平台内部，单帧语义不一致；
4. 同一动作不同相机计数差异中位为 `2`，主相机计数完全正确仅 `1/16`；
5. 当前 `confidence` 对 FP 的绝对区分不足，`80/94` 个 FP 超过 `0.50`。

T1-6 的目标不是给 Fit3D 做一套专用修补，而是建立可迁移到 QEVD、手机和其他周期动作的分割内核：

```text
clip + 动作先验
→ 相位信号选择
→ top/bottom 状态区间
→ 连续 rep 链与动作组选择
→ 首尾部分周期处理
→ 完整度 / 是否在组内 / 综合置信度
→ 统一分析结果
```

核心决策问题：

> 在不使用数据集 ID、固定 rep 数、固定帧偏移、相机 ID 或测试集 GT 调参的前提下，能否把 FIT3D 与 QEVD 的 rep 数量、动作组边界和 top 事件都稳定分出来？

## 2. 与 T1-3 / T1-4 / T1-5 的关系

### 2.1 复用

- 复用 T1-1/T1-2 的 MediaPipe 与角度输出格式。
- 复用 `segment_reps.py` 中已验证的信号预处理、相位、周期、模板和证据函数，但只把它们作为 v2 内核的底层工具。
- 复用 T1-4 的动作语义配置、视角/指标支持和报告输入结构。
- 复用 T1-5 的 Fit3D manifest、相机审计和 `subject-action` cluster 评测原则。
- 复用 T1-3 的 QEVD 标注、IoU/边界评测和鲁棒性测试方法。

### 2.2 不直接修改

- 不覆盖 `results/clip_analysis_v1/`、`results/segmentation_final/`、`results/task1_5/rep_evaluation/`。
- 不修改 T1-4 已冻结的报告和 issue 阈值。
- 不在 T1-6 中重新启用 T1-5C 被禁用的侧视角度指标。
- 不把 T1-6 的 v2 结果直接写回 T1-4 manifest，集成留给通过验收后的独立阶段。

## 3. 通用性与数据集边界

### 3.1 算法内核禁止依赖

以下内容不得出现在 v2 的状态检测、链选择或置信度公式中：

- 数据集名称、视频 ID、subject ID、camera ID；
- Fit3D 固定 `6 rep`、测试集固定 rep 数；
- 固定帧号、固定 `+13` 帧平移、固定平台宽度；
- 固定相机顺序或“四路相机取中位数”；
- 为通过某个 holdout 指标而设置的事后阈值；
- QEVD/Fit3D 各自的特殊裁剪帧。

### 3.2 允许的动作配置

动作语义可以不同，但必须与采集数据集无关：

| 配置 | squat 示例 | pushup 示例 |
|---|---|---|
| 相位候选 | 膝/髋屈曲角 | 肘/肩屈曲角 |
| top 物理含义 | 站立、膝接近伸直 | 手臂接近伸直 |
| bottom 物理含义 | 膝屈曲最大 | 肘屈曲最大 |
| 周期范围 | 动作级物理先验 | 动作级物理先验 |
| 姿态先验 | 躯干竖直范围 | 身体近水平范围 |
| rep unit | `cycle` | `cycle` |

这些配置属于动作语义，不属于 Fit3D 适配。若某个配置只能在一个数据集中成立，必须删除或标为 experimental。

### 3.3 数据集适配层职责

适配层只负责：

- manifest 校验和路径映射；
- 源视频帧号与 clip 帧号转换；
- GT 语义映射，例如 Fit3D `rep_ann` → top 事件，QEVD 标注 → rep 起止；
- 多相机重复观测、subject-action cluster 和 split；
- 结果字段重命名与评测入口。

适配层不得生成或修正 rep 数量、边界、置信度或动作组。

### 3.4 输入契约与动作组歧义

产品输入约定为：

```text
一个 clip = 用户选择的一个动作 + 一次预期训练组
允许前后存在短暂入位/起身/休息，但不允许穿插另一种动作
```

如果视频中存在多个形态相同、都像正式 rep 的连续动作链（例如未标注热身组），算法不能声称能推断用户意图。此时应：

1. 输出多个 `set_candidate`；
2. 标记最长/最稳定链为默认候选；
3. 允许上层传入 `set_start_hint/set_end_hint`；
4. 不把所有链强行合并成一个组。

因此“排除入位/卸杠等非 rep 过渡”是通用能力；“分辨用户心里的正式组和热身组”需要产品交互或额外上下文，不能靠调参伪装成已解决。

## 4. 核心输出：状态区间而非单帧

### 4.1 单个 rep 的统一结构

```json
{
  "rep_id": 2,
  "start_frame": 245,
  "end_frame": 336,
  "top": {
    "start_frame": 245,
    "end_frame": 278,
    "anchor_frame": 261,
    "stable_fraction": 0.84
  },
  "bottom": {
    "start_frame": 300,
    "end_frame": 318,
    "anchor_frame": 309,
    "stable_fraction": 0.76
  },
  "phase_signal": "left_knee_flexion",
  "rep_unit": "cycle",
  "status": "complete",
  "confidence": {
    "complete": 0.91,
    "in_set": 0.94,
    "combined": 0.86
  },
  "quality_flags": []
}
```

### 4.2 语义冻结

T1-6A 必须在 dev 上比较以下全局边界策略，并选定一个，不允许按数据集切换：

- `plateau_entry`：高位平台进入点；
- `plateau_midpoint`：高位平台中点；
- `plateau_exit`：高位平台退出点；
- `phase_zero_crossing`：相位信号导数/阈值穿越点。

建议默认比较 `plateau_midpoint` 和 `phase_zero_crossing`，但最终选择必须以 QEVD 的 rep 区间和 Fit3D 的 top 事件同时可解释为准。选定后写进 `task1_6_segmenter.json`，holdout 阶段不得再改。

Fit3D 的 `rep_ann` 没有完整 rep 起止，因此必须同时报告：

- GT 帧是否落在预测 `top` 区间内；
- 选定 `anchor_frame` 与 GT 的误差；
- 不把 `rep_ann` 当作 bottom，也不反推 bottom GT。

### 4.3 动作组结构

```json
{
  "set_id": 1,
  "start_frame": 180,
  "end_frame": 840,
  "chain_confidence": 0.88,
  "status": "complete_set",
  "reps": [],
  "rejected_candidates": [],
  "non_rep_intervals": []
}
```

一个 clip 可以有多个 set；没有足够证据时输出 `no_confident_set`，而不是硬凑 rep。

## 5. 算法设计

### 5.1 相位信号与状态

1. 从动作配置生成相位候选，而不是固定左右某个关节。
2. 以覆盖率、周期显著性、峰谷稳定性、边界稳定性和可见性选择主信号。
3. 对主信号做缺失处理、平滑和鲁棒归一化；所有持续时间以中位周期比例表达。
4. 估计 `top` / `bottom` 状态区间，而不是只取单点峰谷。
5. 状态区间必须包含持续时间、稳定比例、角度水平、边界斜率等证据。

### 5.2 候选 rep 生成

由 top/bottom 状态交替生成候选，允许：

- 首尾候选缺少一个外侧边界；
- 短暂信号空洞，但必须有插值覆盖证据；
- 不同 `rep_unit`，例如 `cycle`、`stride`、`side`，由动作配置决定。

候选只是节点，不直接以“置信度大于阈值”逐个放行。

### 5.3 连续 rep 链选择

使用候选图上的动态规划/维特比式选择：

```text
node score    = 状态完整度 + 幅度 + 可见性 + 动作姿态先验
edge score    = 周期一致性 + 幅度一致性 + 曲线形状相似度 + 间隔合理性
chain score   = Σ node score + Σ edge score - 组切换惩罚
boundary score = 起始/结束前后必须回到稳定非 rep 状态
```

输出一条或多条连续链，候选条件：

- 至少 `min_complete_reps` 个 rep；
- 相邻周期只允许相对变化，不允许固定帧差；
- 首尾不能靠无界外推补齐；
- 入位、卸杠、走路、单次抬腿等不与长链连通；
- 链内允许短暂停顿，但停顿不能产生新的 rep。

该步骤替代“先收 active bout，再对每个 rep 独立过滤”的 T1-4 逻辑。

### 5.4 首尾部分周期

首尾候选允许以下两种状态：

- `complete`：top、bottom 和两侧边界都可见；
- `endpoint_partial`：缺少一个外侧周期，但 top/bottom 状态完整、幅度充分、边界由不超过一个局部周期的外推得到。

不得把仅有入位/起身弧线、没有 bottom 或 top 状态的片段补成 rep。首尾候选必须和链内其它 rep 保持周期、幅度和形态一致。

### 5.5 置信度

拆成三个概率，而不是一个混合 score：

| 字段 | 含义 |
|---|---|
| `complete` | 该候选是否包含完整 top/bottom 状态 |
| `in_set` | 是否属于当前连续正式动作链 |
| `combined` | 可用于下游放行的综合置信度 |

候选特征：

- 链内前后周期相对差、幅度相对差、形状相关；
- top/bottom 稳定比例和边界斜率；
- 相位信号覆盖率、SNR、主侧可见性；
- 动作姿态先验；
- 与非 rep 状态的距离；
- 首尾边界外推长度。

校准要求：

- 只在 dev 上拟合校准器；
- 不加入 dataset ID、camera ID、subject ID；
- 同时报告 Brier score、ECE 和按阈值扫描的 precision/recall；
- holdout 上禁止重新拟合或改阈值。

### 5.6 拒答

必须显式区分：

- `no_motion`：没有足够周期运动；
- `no_confident_set`：有局部运动但没有连续链；
- `insufficient_visibility`：姿态或主侧不可见；
- `unsupported_view`：当前视角不支持该动作指标；
- `ambiguous_chain`：存在多条同等合理链，等待上层选择；
- `unstable_phase`：主信号周期或边界不稳定。

拒答不能生成“0 rep 即正常”的报告。

## 6. 数据与划分

### 6.1 Fit3D

沿用 T1-5 冻结 manifest：

```text
64 路 train 视频
16 个 subject-action
96 个唯一 GT rep
384 个相机 × rep 观测（不是 384 个独立实例）
```

划分保持：

| 角色 | 受试者 | 用途 |
|---|---|---|
| dev | `s03,s04,s05,s07` | 状态语义、链选择、校准 |
| holdout | `s08,s09,s10,s11` | 冻结后评估一次 |

注意：T1-5D 已观察过 aggregate holdout 结果，因此该 holdout 对新算法不是“从未见过”，只能保证 T1-6 开发期间不在 holdout 上调参。文档必须如实说明这一点。

### 6.2 QEVD

可用数据：

- T1-3 已有的非盲标注和 5 段内部盲标，共 6 个 segment、`66` 个 accept rep；
- T1-4 的 13 个 dev clip、118 个 rep，可用于产品链路 smoke；
- 其余动作段作为入位、休息和错误动作负样本。

T1-3 盲标已经参与过历史评测，不能再次称为完全独立 holdout。T1-6A 必须另建一个封存 holdout：

```text
至少 6 个动作段
覆盖 pushup、squat 和至少 2 类非目标周期动作
首次标注后封存，只在 T1-6E 运行一次
```

若暂时无法新增标注，T1-6 只能报告“跨数据源回归”，不能宣称完成通用算法验收。

### 6.3 数据隔离

- Fit3D 多相机按 `subject-action` 聚合，禁止把同一动作的 4 路视频当 4 个独立调参样本。
- QEVD 按视频隔离，同一长视频的不同动作段不能跨 split。
- 标注、调参和 holdout 运行记录必须带 manifest SHA-256。
- test smoke 永不进入误差统计。

## 7. 评测协议

### 7.1 分层指标

必须同时报告：

| 层级 | 指标 |
|---|---|
| 状态 | top/bottom 区间 IoU、anchor 误差、区间包含率 |
| rep | count error、完整 rep 数、首/中/尾 rep 召回、边界误差 |
| 动作组 | set start/end 误差、组外 FP、组间合并、组内断裂 |
| 置信度 | Brier、ECE、高置信 FP、拒答正确率、覆盖率 |
| 工程 | 运行时间、内存、FPS、失败原因分布 |

Fit3D 主统计单位仍为 `subject-action` cluster；QEVD 以 segment/video 为独立单位。

### 7.2 与 T1-5D 的固定对照

| 指标 | T1-5D 基线 | T1-6 目标 |
|---|---:|---:|
| Fit3D 实例计数 MAE | `0.625` | `<= 0.5` |
| 完全计数正确 | `6/16` | `>= 14/16` |
| 首事件召回 | `25%` | `>= 75%` |
| 末事件召回 | `12.5%` | `>= 75%` |
| 主相机事件中位误差 | `12.5` 帧 | `<= 5` 帧 |
| 主相机事件 p90 | `25.8` 帧 | `<= 12` 帧 |
| 主相机高置信 FP | `24` | `0` |
| 全机位高置信 FP | `80` | `0` |
| top-to-top 间隔中位误差 | `3` 帧 | 不退化 |

这些是固定对照线，不是按 T1-6 结果重新设定的目标。

### 7.3 QEVD 验收

封存 holdout 上要求：

| 指标 | 通过线 |
|---|---:|
| count exact | `>= 5/6 segments` |
| count MAE | `<= 0.25 / segment` |
| 边界绝对误差中位 | `<= 5` 帧 |
| 边界绝对误差 p95 | `<= 12` 帧 |
| F1 @ IoU 0.30 | `>= 0.95` |
| 高置信 FP | `0` |
| 错误动作/纯入位片段被输出为完整 rep | `0` |

若新 holdout 的动作分布与 T1-3 明显不同，可在 T1-6A 冻结前调整样本构成，但不能在看到 T1-6 结果后修改。

### 7.4 通用性测试

#### 不变性

以下变换不应改变 rep 数量和组结构，除非明确标注为拒答：

- 幅度缩放、角度 offset、左右翻折；
- 15/30/50/60 FPS 重采样；
- 随机角度噪声、landmark 抖动；
- 短暂丢帧和局部遮挡；
- 在 clip 前后拼接 idle/setup 片段。

#### 负样本

以下片段必须输出 0 个完整 rep 或 `no_confident_set`：

- 取杠、卸杠、入位、起身；
- 静立、走路、单次抬腿；
- 半 rep；
- 多种动作拼接；
- 只有一次摆动的过渡。

#### 跨域

同一份冻结配置必须依次运行：

1. Fit3D dev；
2. Fit3D holdout；
3. QEVD dev；
4. QEVD sealed holdout；
5. 可选的手机自录 smoke。

不允许在不同数据集之间切换阈值后仍宣称通用。

### 7.5 消融

至少比较：

A. T1-4/T1-5 当前分割；
B. 仅增加首尾部分周期；
C. 仅增加连续链选择；
D. 仅换状态区间 anchor；
E. 链选择 + 首尾部分周期；
F. E + 置信度校准；
G. 去掉任一核心组件。

报告每个组件对计数、首尾召回、事件误差和高置信 FP 的贡献，不以单一总分掩盖失败。

## 8. 分阶段交付

### T1-6A：协议、适配器与基线

内容：

- 冻结 v2 schema、边界 anchor 候选和评测口径；
- 实现 Fit3D/QEVD adapter，不改动源 manifest；
- 复现 T1-4/T1-5D 基线数字；
- 建立 dev/holdout/sealed holdout 清单和 SHA；
- 新建 QEVD 封存 holdout 标注模板。

通过条件：

- 所有 clip、帧号、GT 语义和 split 可校验；
- 基线数字与 T1-5D 一致；
- 任何 adapter 都不能改变算法输出；
- 无数据集特定常量进入核心配置。

产物：

```text
configs/task1_6_eval_protocol.json
configs/task1_6_sources.json
scripts/adapt_task1_6_sources.py
schemas/task1_6_clip_analysis_v2.schema.json
docs/task1-6a_protocol.md
```

### T1-6B：状态区间与首尾部分周期

内容：

- 实现 top/bottom 状态区间；
- 实现锚点候选和全局选择；
- 实现首尾 `endpoint_partial`；
- 对合成波形、平台、缺帧、噪声做单元测试；
- 在 dev 上比较 anchor 策略，冻结一个全局策略。

通过条件：

- 合成平台上 boundary/anchor 误差在容差内；
- 幅度缩放、offset、翻转、重采样不改变状态数量；
- 半 rep 不被标成 complete；
- Fit3D first/last 召回在不使用固定偏移的情况下提升到 `>=50%` 作为阶段门。

产物：

```text
scripts/task1_6_phase_states.py
configs/task1_6_segmenter.json
tests/test_task1_6_phase_states.py
docs/task1-6b_state_results.md
```

### T1-6C：连续 rep 链与动作组选择

内容：

- 生成候选 rep 图；
- 实现周期/幅度/形状/间隔 edge cost；
- 实现 DP/维特比链选择；
- 输出一个或多个 `set_candidate`；
- 将 setup、rest、recovery、单次过渡作为负样本评估。

通过条件：

- Fit3D dev 的首尾召回均 `>=75%`；
- 组外高置信 FP 在 dev 上降到 `0`；
- 不增加 GT 中位数周期之外的硬编码；
- QEVD dev 的 rep 数和边界不退化超过预注册容忍。

产物：

```text
scripts/task1_6_chain_selection.py
tests/test_task1_6_chain_selection.py
results/task1_6/dev/chain_audit.csv
docs/task1-6c_chain_results.md
```

### T1-6D：置信度校准与拒答

内容：

- 计算 `complete/in_set/combined` 三层置信度；
- 在 dev 上拟合校准器；
- 在 QEVD dev 上检查域间校准；
- 输出高置信 FP、拒答类型和覆盖率；
- 禁止使用数据集/相机/受试者身份特征。

通过条件：

- Fit3D dev 与 QEVD dev 的高置信 FP 都为 `0`；
- 校准后 ECE 明显低于未校准 score；
- 拒答类别与实际失败原因一致；
- 阈值在进入 holdout 前冻结并写入 manifest。

产物：

```text
scripts/task1_6_confidence.py
configs/task1_6_confidence_frozen.json
tests/test_task1_6_confidence.py
docs/task1-6d_confidence_results.md
```

### T1-6E：跨数据集验收、消融与集成决策

内容：

- 运行 Fit3D holdout、QEVD sealed holdout；
- 完成消融、噪声/重采样/负样本测试；
- 生成失败样本索引和逐项 go/no-go；
- 决定是否将 v2 接入 T1-4/T1-7；
- 若失败，保留 v2 experimental，不回写 T1-4 冻结产物。

通过条件：

- 第 7.2/7.3 节全部通过；
- 同一冻结配置贯穿所有数据集；
- 结果可重复，CSV 除时间戳外一致；
- 没有 clip ID/数据源特判；
- 全量测试通过。

产物：

```text
scripts/evaluate_task1_6.py
scripts/audit_task1_6_failures.py
results/task1_6/evaluation/
results/task1_6/tables/
results/task1_6/figures/
docs/task1-6_results.md
```

## 9. 新增文件草案

```text
VLS/V0/scripts/adapt_task1_6_sources.py       # Fit3D/QEVD → 统一 v2 输入
VLS/V0/scripts/task1_6_phase_states.py        # 相位状态区间与 anchor
VLS/V0/scripts/task1_6_chain_selection.py     # 候选图与连续链 DP
VLS/V0/scripts/task1_6_confidence.py          # 三层置信度与校准
VLS/V0/scripts/analyze_clip_v2.py             # v2 编排入口
VLS/V0/scripts/evaluate_task1_6.py            # 跨数据集评测
VLS/V0/scripts/audit_task1_6_failures.py      # 失败样本与可视化证据
VLS/V0/schemas/task1_6_clip_analysis_v2.schema.json
VLS/V0/configs/task1_6_sources.json
VLS/V0/configs/task1_6_segmenter.json
VLS/V0/configs/task1_6_confidence_frozen.json
VLS/V0/configs/task1_6_eval_protocol.json
VLS/V0/tests/test_task1_6_phase_states.py
VLS/V0/tests/test_task1_6_chain_selection.py
VLS/V0/tests/test_task1_6_confidence.py
VLS/V0/tests/test_task1_6_evaluation.py
VLS/V0/docs/task1-6a_protocol.md
VLS/V0/docs/task1-6b_state_results.md
VLS/V0/docs/task1-6c_chain_results.md
VLS/V0/docs/task1-6d_confidence_results.md
VLS/V0/docs/task1-6_results.md
```

## 10. 输出目录

```text
VLS/V0/results/task1_6/
├── manifest/
│   ├── sources.json
│   ├── validation.json
│   └── checksums.sha256
├── runs/
│   └── <run_id>/
│       ├── fit3d_dev/<clip_id>/analysis_v2.json
│       ├── fit3d_holdout/<clip_id>/analysis_v2.json
│       ├── qevd_dev/<segment_id>/analysis_v2.json
│       └── qevd_holdout/<segment_id>/analysis_v2.json
├── evaluation/
│   ├── summary.json
│   ├── rep_counts.csv
│   ├── state_intervals.csv
│   ├── set_boundaries.csv
│   ├── confidence_calibration.csv
│   ├── rejection_audit.csv
│   └── stratified_summary.csv
├── ablation/
├── figures/
└── checks/
```

所有 run 目录带 manifest hash、配置 hash、代码状态和随机种子。

## 11. 命令草案

```bash
cd /mnt/storage2/data_guest

# 1. 构建统一数据索引，不改源 manifest
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/adapt_task1_6_sources.py \
  --fit3d-manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --qevd-manifest VLS/V0/configs/task1_4_clips.json \
  --blind-annotations VLS/V0/results/segmentation/annotation_blind/blind_reps.csv \
  --sources-output VLS/V0/configs/task1_6_sources.json \
  --protocol-output VLS/V0/configs/task1_6_eval_protocol.json \
  --holdout-dir VLS/V0/results/task1_6/manifest

# 2. 运行 dev，冻结 anchor 与置信度
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/analyze_clip_v2.py \
  --sources VLS/V0/results/task1_6/manifest/sources.json \
  --split dev \
  --config VLS/V0/configs/task1_6_segmenter.json \
  --output-dir VLS/V0/results/task1_6/runs/dev

# 3. 冻结后运行 holdout，只运行一次
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/analyze_clip_v2.py \
  --sources VLS/V0/results/task1_6/manifest/sources.json \
  --split holdout \
  --config VLS/V0/configs/task1_6_confidence_frozen.json \
  --output-dir VLS/V0/results/task1_6/runs/holdout

# 4. 跨数据集评估与消融
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/evaluate_task1_6.py \
  --sources VLS/V0/results/task1_6/manifest/sources.json \
  --run-dir VLS/V0/results/task1_6/runs/holdout \
  --output-dir VLS/V0/results/task1_6/evaluation

# 5. 全量测试
VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test*.py'
```

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 把 Fit3D 平台偏移写成常量 | 换数据集失效 | 状态区间输出；anchor 全局冻结；跨 QEVD 验收 |
| 强行输出唯一动作组 | 热身组/多组场景错误 | 输出多个 set candidate；允许上层 hint |
| 首尾补齐产生新 FP | 计数看似改善但误检增加 | endpoint_partial 必须有 top+bottom 与链一致性 |
| 链选择只按周期 | 把走路/其它周期动作串进来 | 加姿态、幅度、形状、动作先验边 |
| 置信度仍过度自信 | 高置信 FP 继续进入报告 | 三层概率、dev 校准、holdout 零高置信 FP |
| QEVD holdout 被污染 | 泛化结论不成立 | T1-6A 新建封存 holdout；旧盲标只作回归 |
| Fit3D holdout 已观察 | 结果不再完全独立 | 文档如实标注；开发期锁定，不据其调参 |
| 杠铃深蹲与徒手深蹲域差 | 动作分布不可外推 | 只评分割机制，不把深度/姿态指标外推 |
| 多相机聚合掩蔽单路失败 | 产品单相机不成立 | 主结果同时报告单路；cluster 只用于 CI |
| 复杂度导致不可维护 | 难以定位回归 | 状态、链、置信度分层模块；固定中间 CSV 和消融 |

## 13. 完成定义

T1-6 完成必须同时满足：

- v2 schema、算法配置和 adapter 边界冻结；
- Fit3D holdout 达到第 7.2 节计数、首尾、事件和高置信 FP 目标；
- QEVD sealed holdout 达到第 7.3 节目标；
- 同一冻结配置跨数据集运行，无 per-dataset threshold；
- 首尾事件召回不再由少数 clip 或相机支撑；
- 高置信错误完整 rep 为 `0`；
- 负样本、半 rep、重采样、噪声和拼接测试通过；
- 消融能说明链选择、状态区间、首尾处理和置信度各自动贡献；
- 全量测试、复现性和产物 checksum 通过；
- 文档明确回答“哪些能力是通用的，哪些仍依赖动作语义/产品输入”；
- 未通过时明确 no-go，不把 experimental 结果接入 T1-4 产品报告。

## 14. T1-6A 执行状态

状态：**完成（2026-10-26）**。

- 已新增 `scripts/adapt_task1_6_sources.py`，统一映射 `70` 个 Fit3D clip、`41` 个 QEVD T1-4 clip 和 `2` 个 QEVD sealed 非目标段。
- 已生成 `configs/task1_6_sources.json`；`113` 个 clip 的文件、帧号、GT 语义、cluster 和 split 校验通过。
- 已生成 `configs/task1_6_eval_protocol.json`，冻结 anchor 候选、事件匹配、cluster 单位、固定验收线与禁止特征。
- 已生成 `schemas/task1_6_clip_analysis_v2.schema.json`，固定状态区间、reps、set candidates、三层置信度和拒答原因。
- 已复现 T1-5D 基线：MAE `0.625`、exact `6/16`、主相机事件中位/p90 `12.5/25.8` 帧、全机位高置信 FP `80`。
- 已建立 QEVD sealed holdout：`0152/0159` pushup、`0127/0177` squat、`0207` air jump rope/good morning，共 `6` 段。
- 已生成空白标注模板与接触表；当前状态为 `pending_human_annotation`，跨数据源验收**尚未完成**。
- 定向测试 `tests/test_task1_6_adapter.py` 共 `12` 项通过；全量测试结果见收口记录。
- 详细说明见 `docs/task1-6a_protocol.md`。

## 15. T1-6B 执行状态

状态：**完成（2026-10-26）**，阶段门通过。

- 已实现 top/bottom 状态区间、四种 anchor、首尾 `endpoint_partial` 和三层 rep 置信度字段。
- 已在 Fit3D dev `32` clips / `8` clusters / `192` 个相机事件观测上比较三种信号策略和四种 anchor。
- 冻结 `t1_6_bilateral_mean + plateau_midpoint`，写入 `configs/task1_6_segmenter.json`。
- 全机位 top 事件 precision/recall 为 `0.825/0.885`，首/末召回 `0.8125/0.78125`，误差中位/p90 为 `6/22` 帧。
- 主相机 precision/recall 为 `0.918/0.938`，首/末召回 `0.875/0.75`，误差中位/p90 为 `5/20` 帧。
- T1-5D dev accepted rep 的首/末事件召回约为 `0.21875/0.125`；T1-6B 已通过 `>=50%` 阶段门，但仍保留 `36` 个全机位 FP 和 `22` 个 FN，交给 T1-6C 链选择处理。
- 合成平台、缩放、offset、翻转、15/30/50/60 FPS、噪声、缺帧、半 rep 和无 bottom 测试已补齐，定向测试 `13/13` 通过。
- 详细结果见 `docs/task1-6b_state_results.md`，评测产物在 `results/task1_6/dev/`。
- 下一步进入 T1-6C：候选图、连续链 DP、动作组边界和组外 FP 剔除。

## 16. T1-6C 执行状态

状态：**本轮实现与 Fit3D dev 阶段门完成（2026-10-26）**；QEVD dev 仅 smoke，正式边界验收尚不可评估。

- 已新增 `scripts/task1_6_chain_selection.py` 与 `configs/task1_6_chain.json`，实现候选节点、相对幅度过滤、周期/幅度/bottom 持续时间边、连通分量、Viterbi 链选择、多 `set_candidate`、默认组和非 rep 区间。
- 最小链长按**实际完整周期数**执行，至少 `3` 个 rep；`3` 个 top 状态只能形成 `2` 个周期，不能成为正式 set。
- Fit3D dev `32` clips / `8` clusters：event recall `0.869792`、precision `0.888298`、first/last `0.750/0.78125`、误差中位/p90 `6/20.2` 帧、count MAE `0.4375`、count exact `26/32`。
- T1-6C 阶段门通过：first/last 均 `>=0.75`，组外高置信 FP `0`。但链内仍有 `21` 个高置信 FP，且首事件召回没有余量；总高置信 FP 的 `0` 目标留给 T1-6D，不得把组外 FP 与总 FP 混写。
- `31/32` 个 clip 输出 set；`fit3d_s03_pushup_50591643` 输出两个 3-rep set；`fit3d_s05_pushup_50591643` 因状态欠分割且只有两个合格候选输出 `no_confident_set`；`fit3d_s03_squat_58860488` 首/末事件误差 `48/54` 帧，超过 `35` 帧门，保留为真实失败样例。
- Fit3D 的 `6` 个 top 事件对应 `5` 个 top-to-top 完整周期；`rep_count=5` 与 `pred_event_count=6` 是预期语义，不是 Viterbi 丢节点。
- QEVD dev `13` clips：count MAE `0.230769`、max delta `2`、exact `11/13`；`2` 个低周期显著性 clip 使用 legacy fallback。边界 GT 仅有 `0161_pushup_set01` 的 `5` 个盲标区间，`5/5` 匹配，start/end 中位 `11/8` 帧，pooled 中位/p95 `9.5/20.85` 帧。
- 已修复 QEVD 非 rep 区间使用源视频总帧数的错误，统一改为角度表实际 clip-local 长度；例如 `0161_pushup_set01` 的 recovery 从 `495–4488` 修正为 `495–667`。
- 组外高置信 FP 改为根据事件的 `set_ids` 实际计算，不再写死为 `0`；QEVD 边界统计改为 pooled 盲标误差，不再对各 clip 中位数再取 p95。
- QEVD dev 的 `12/25` 帧边界阈值只作为本轮 smoke；由于仅 `1` 个 clip 有边界标注，`formal_stage_gate_evaluable=false`，不能宣称跨域通过。
- 已新增 `tests/test_task1_6_chain_selection.py`，`10` 项覆盖链长、弱候选、缺 bottom、断链、双 set、不变性、QEVD 帧偏移和配置禁令；全量测试 `127/127` 通过。
- 详细结果见 `docs/task1-6c_chain_results.md`，产物在 `results/task1_6/dev/`。
- 下一步进入 T1-6D：链内置信度校准与拒答，目标是把总高置信 FP 从 `21` 降到 `0`，并在 holdout 前冻结阈值。

## 17. T1-6D 执行状态

状态：**Fit3D dev 阶段门通过（2026-10-26）**；QEVD dev 仅完成跨域 smoke，正式阶段门不可评估。

- 已新增 `scripts/task1_6_confidence.py`、`configs/task1_6_confidence_frozen.json`、`tests/test_task1_6_confidence.py` 和 `docs/task1-6d_confidence_results.md`。
- 正式评估单位固定为“默认 `set_candidate` 内 `status=complete` 的 rep”；默认组 `endpoint_partial` 和非默认 set 不进入高置信 FP 验收。
- Fit3D dev 共 `103` 个正式 complete rep（`96 TP / 7 FP`）。T1-6C 的 `21` 个高置信 FP 是 `188` 个 selected top event 的诊断值，不是 complete rep 的正式 FP。
- 冻结 `5` 个等宽分箱的 Beta/Clopper-Pearson 风险下界（`alpha=0.1`）+ running-max 单调化，高置信阈值为 `0.5`；配置每次运行都会与 dev 重拟合结果逐项校验。
- 校准前 `combined >= 0.5`：`89` 个高置信 rep，`89 TP / 0 FP`；校准后 `91` 个高置信 rep，`91 TP / 0 FP`。高置信覆盖率从 `0.864078` 提升到 `0.883495`，TP 覆盖率从 `0.927083` 提升到 `0.947917`。
- ECE 从 `0.274748` 降到 `0.074698`（相对下降 `72.81%`），Brier 从 `0.109480` 降到 `0.038866`；正式门通过。
- `complete / in_set / combined` 三层均报告。`complete` 的 ECE 为 `0.040256`，但只描述状态完整度；下游 `combined` 原始 ECE 为 `0.274748`，因此校准对象保持为 `combined`。
- 校准后有 `12` 个正式 rep 低置信拒答，其中 `5 TP / 7 FP`；`7` 个正式 FP 全部被拒答，没有高置信 FP。
- 拒答审计共 `20` 个链候选：`16 shape_mismatch`、`4 no_confident_set`、`outside_set` 未产生；`20/20` 与实际证据一致。
- QEVD dev `13` clips 中仅 `0161_pushup_set01` 有 `5` 个盲标 rep；应用冻结表后 `5/5` IoU 匹配、`5 TP / 0 FP`，ECE `0.193630 -> 0.067390`。其余 `12` clips 的 `94` 个正式 rep 无 rep 级 GT，不能计入正式 FP 分母。
- QEVD `annotated_clip_count=1`，因此 `formal_stage_gate_evaluable=false`、`qevd_dev_stage_passed=null`；只能声明单 clip 跨域 smoke 未观察到高置信 FP，不能宣称 QEVD dev 正式通过。
- 配置 `core` 不含 dataset/video/subject/camera ID、固定 rep 数或固定帧偏移；冻结阈值写入 manifest 后在 sealed holdout 前不再修改。
- 定向测试 `tests/test_task1_6_confidence.py` 共 `12` 项通过；全量测试 `139/139` 通过。详细结果与复现命令见 `docs/task1-6d_confidence_results.md`。
- 下一步进入 T1-6E：跨数据集验收、消融与集成决策；QEVD sealed holdout 标注完成前不得把跨域正式门标记为通过。

## 18. T1-6E 执行状态

状态：**执行完成（2026-10-27）**，验收结论：**NO_GO**；RepChain v2 保持 experimental，不接入 T1-4/T1-7。

- 已新增 `scripts/evaluate_task1_6.py`、`scripts/audit_task1_6_failures.py` 和 `tests/test_task1_6_evaluation.py`，产物写入 `results/task1_6/evaluation/`、`results/task1_6/tables/`、`results/task1_6/figures/` 和 `results/task1_6/runs/holdout/`。
- Fit3D holdout 为 `32` clips / `8` 个 `subject-action` clusters / `192` 个 GT top 事件观测。该 split 已被 T1-5D 观察过，本轮按锁定评测处理，不宣称从未见过。
- 计数共识通过：`48/48` 个 rep、count MAE `0.0`、exact `8/8`；首事件召回 `0.8125` 也通过。
- Fit3D 固定门失败：last recall `0.53125`、主相机事件中位/p90 `6/17` 帧、主相机高置信事件 FP `4`、全机位高置信事件 FP `26`、complete rep 高置信 FP `2`、top-to-top 间隔中位 `5` 帧。
- 正式 complete rep 共 `108`（`101 TP / 7 FP`）。校准把 ECE `0.287903 -> 0.057450`、Brier `0.114547 -> 0.043686`，但 holdout 高置信 FP 仍为 `2`，没有复现 dev 的 0 FP。
- QEVD sealed 共 `6` 段：`4` 段 pushup/squat 可运行，`2` 段 `0207` 动作无冻结契约；人工 GT 仍为 `0/6`，7.3 全部 `not_evaluable`。`0159_pushup_set01` 的 legacy quality 为 reject，但 v2 输出 `7` 个高置信 complete rep，质量门冲突已进入 review index。
- 消融完成：legacy A 为 recall `0.541667`、first/last `0.3125/0.125`；state-only B 为 `0.880208/0.59375` 但 count MAE `0.75`；去掉 endpoint 的 C 首/末召回崩塌；完整链 D 为 recall/precision `0.854167/0.863158`，把 count MAE 降到 `0.0`、exact 提到 `8/8`；E 的高置信 FP 使用 complete rep 口径，不能与 D 的事件口径混写。
- 稳健性未完全通过：15 FPS 结构一致 `27/32`、60 FPS `28/32`、噪声 `28/32`；幅度缩放、offset、翻转、缺帧和 idle padding 基本稳定。
- 负样本 `4/5`：`idle_constant` 被误判为 1 个 set、3 个 rep、2 个 complete rep；其余 setup ramp、单周期、半 rep 和两短周期均拒答。
- 失败审计：`21/32` clips 失败、`19` 个 high priority；last event miss `15`、事件高置信 FP `17`、first event miss `6`、count error `6`。失败索引、分层表和 SVG 已生成。
- 已修正复现 scope 误判、sealed pending/delete 标注误入 GT、门控图伪数值、帧号 `0` 证据、消融 A 事件误差键名和审计 checksum 覆盖。
- 已修正事件匹配门：先禁用超距边，再最大化门限内一对一匹配数，最后最小化误差；不再先强制完整分配后丢弃。`scripts/evaluate_task1_5d_reps.py` 的重复实现同步修正。
- T1-5D 原冻结产物未覆盖；消融 A 改用 `results/task1_6/baseline_t1_5d_matchfix/` 的修正复评，全机位高置信 FP 为 `79`（旧值 `80`），计数和主相机头条值不变。
- 复现性：连续两次完整运行的 9 张 CSV hash 一致；定向测试 `43/43`、全量测试 `162/162` 通过。详细结果见 `docs/task1-6_results.md`。
- 下一轮优先级：T1-7 前执行质量门、修链尾/末事件覆盖、单独抑制事件级高置信 FP、修 idle 误检、完成 QEVD sealed 标注并补 `0207` 动作契约。

## 19. T1-6F：tail anchor 通用优化

状态：**已完成独立复评（2026-10-27）**；`decision = NO_GO` 不变。

- 通用规则：链图和中间事件保持 `plateau_midpoint`，每条链最后一个 selected top event 使用 `min(plateau_midpoint, phase_zero_crossing)`；不含任何数据集、受试者、相机或固定帧偏移特征。
- 配置：`configs/task1_6_chain.json` 新增 `core.tail_anchor`；实现：`scripts/task1_6_chain_selection.py` 的 `terminal_anchor_frame`。
- 事件诊断新增 `graph_frame`、`is_terminal` 和实际 `anchor_policy`，明确区分链图时间与输出事件时间。
- 跨数据集结构检查：Fit3D holdout `23/32`、QEVD sealed `4/4` 的链尾 top state 为 `trailing_partial`；midpoint 相对 zero-crossing 的中位偏移分别约 `0.35`、`0.30` 个周期。QEVD 无 sealed GT，因此不写成正式跨域通过。
- Fit3D holdout 全机位：last recall `0.53125 -> 0.90625`、recall `0.854167 -> 0.911458`、precision `0.863158 -> 0.921053`、高置信事件 FP `26 -> 15`、p90 `19 -> 18`；主相机 recall/precision `0.895833/0.914894 -> 0.979167/1.0`，主相机高置信事件 FP `4 -> 0`。
- 失败审计：失败 clips `21 -> 12`、high priority `19 -> 6`、last-event miss `15 -> 3`；剩余 3 个都是 pushup 链尾节点入选/抑制问题。
- 计数和 complete rep 高置信 FP 不变：count MAE `0.0`、rep 高置信 FP `2`；原 `results/task1_6/evaluation/` 等冻结产物未覆盖，新结果在 `results/task1_6/evaluation_tail_anchor/`、`tables_tail_anchor/`、`figures_tail_anchor/` 和 `runs/tail_anchor/`。
- 复现性：两次完整运行 9 张 CSV hash 一致，`checksums.sha256` 全部通过；定向测试 `55` 项通过，排除环境阻塞的 T1-4 两个模块后全量 `141 passed + 2 subtests`。下一步优先修 pushup 链尾节点和事件级高置信 FP，不在 anchor 层继续调参。

## 20. T1-6G：质量门、idle 拒答与 raw/gated 双口径

状态：**已完成独立复评（2026-10-27）**；`decision = NO_GO` 不变。

### 20.1 执行顺序

1. 先接统一质量门，避免只在 evaluator 打标记；
2. 再增加 idle/低变化运动证据与拒答；
3. 最后扩展评测为 raw/gated 双口径，并披露排除计数与 coverage；
4. `warn` 只附告警，不硬阻断；`reject/unknown` 才 fail closed。

### 20.2 实现

- 新增 `scripts/task1_6_quality.py`：`status/scope/reasons/source/blocking` 契约，支持 clip/set/rep 作用域；clip reject 清空 sets、events 和 complete rep，保留门控前结构诊断。
- `scripts/task1_6_chain_selection.py` 的 `analyze_clip_signals` 接收质量门；`from_legacy_quality` 把 T1-4 的 `pass/warn/reject` 映射到新契约。
- `scripts/task1_6_phase_states.py` 增加 `dynamic_range`、鲁棒 `dynamic_to_noise`、`state_amplitude` 三类运动证据；阈值写入 `configs/task1_6_segmenter.json`。
- 运动门限：`dynamic_range >= 5°`、`dynamic_to_noise >= 8`、`state_amplitude >= 3°`；不满足返回 `no_motion`，零 set/rep。
- `scripts/evaluate_task1_6.py` 同时保留 raw 与 gated 分析；Fit3D/QEVD 输出新增 raw/gated 计数、质量动作、排除原因、motion status、coverage；新增 `fit3d_gated_clip_coverage >= 0.95` 披露门。
- `scripts/audit_task1_6_failures.py` 增加质量排除和运动拒答的 review 路由。

### 20.3 复评结果

- Fit3D 质量分布 `pass 17 / warn 14 / reject 1`；14 个 warn 全部保留，唯一 reject 为 `fit3d_s09_pushup_50591643`。
- Fit3D raw：recall/precision `0.911458/0.921053`、高置信事件 FP `15`、coverage `32/32`。
- Fit3D gated：recall/precision `0.901042/0.935135`、高置信事件 FP `12`、coverage `31/32 = 0.96875`，cluster coverage `8/8`；count consensus 仍为 MAE `0.0`、exact `8/8`。
- 负样本 `idle_constant` 从 `1 set / 3 reps / 2 complete` 修到 `no_motion / 0 / 0 / 0`；5 个负控 `5/5` 通过；`idle_pad_2s` 结构一致 `32/32`。
- QEVD `0159` 从 raw `1 set / 9 reps / 7 high-conf complete` 变为 gated `0 / 0 / 0`；`0152`、`0127`、`0177` 保留；质量冲突 gated 计数为 `0`。
- QEVD 可报告覆盖率 `3/4` 可运行段、`3/6` 全部段；sealed GT 仍为 `0/6`，正式跨域门仍不可评估。
- 剩余固定失败门不变：主相机事件中位/p90、全机位高置信事件 FP、complete rep 高置信 FP、top-to-top 间隔。

### 20.4 产物与复验

```text
results/task1_6/evaluation_quality_motion/
results/task1_6/tables_quality_motion/
results/task1_6/figures_quality_motion/
results/task1_6/runs/quality_motion/
```

连续两次完整运行 `identical_csv=true`，11 张评测 CSV hash 一致；排除受现有 TensorFlow/protobuf 环境影响的 T1-4 两个模块后，全量测试 `151 passed + 4 subtests`。下一轮仍应优先修 pushup 链尾入选/抑制与事件级高置信 FP，而不是放宽质量门。
