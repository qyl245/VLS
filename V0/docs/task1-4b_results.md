# task1-4B：clip 级姿态、角度、质量门控与 rep/相位

## 1. 执行结论

状态：**完成（2026-10-26 更新）**

- 新增 `scripts/analyze_clip.py`，按 manifest 读取 clip，完成 `MediaPipe → 角度 → 姿态门控 → rep/相位 → 结构化输出`。
- 在 13 个 dev clip 上跑通：13 个 `ok`；共输出 118 个完整 rep，剔除 14 个不完整、错位或不符合目标动作姿态的 rep。
- 新增动作姿态门控：俯卧撑在 bottom 帧必须接近水平，深蹲在 bottom 帧必须保持直立；热身、起身和站立过渡 rep 写入 `rejected_reps.csv`，不参与点评。
- 新增相位信号候选分割：当整段角度评分被热身动作带偏时，用动作相位关节角（俯卧撑左肘、深蹲左膝）重新分割，经姿态门控后只保留有效 rep 更多的方案。`0004_pushup_set01` 因此从 0 个有效 rep 恢复为 5 个真实俯卧撑。
- 所有输出写入 `results/clip_analysis_v1/<clip_id>/`，未覆盖 `results/segmentation_final/` 或 task1-1/1-2 的冻结输出。
- 当前按“暂时无法补录侧视数据”执行，只完成纯正面垂直切片；深蹲深度、躯干前倾、俯卧撑身体直线等矢状面指标保持 disabled。
- `0161_pushup_set01` 有 5 个已确认盲标 rep；本版本输出同样为 5 个，IoU@0.30 全部匹配，平均 IoU `0.681`。

## 2. 输入范围与视角限制

| 动作 | dev clip | 完整 rep | 视角 |
|---|---:|---:|---|
| 深蹲 | 5 | 53 | 全部 `front` |
| 俯卧撑 | 8 | 65 | 全部 `front` |

T1-4A 已确认 QEVD 入选集没有可靠侧/斜侧样本。因此本轮可以验收姿态/角度适配、质量门控、正面指标输入和 rep/相位；不能验收深蹲深度、膝/髋矢状面 ROM、躯干前倾、俯卧撑下放深度、身体直线或塌腰/撅臀。

这些指标在 `quality.json.view_support.disabled_metrics` 中显式列出，T1-4C 不得为它们生成高/中严重度问题卡。相机运动当前为 `not_evaluated`。

## 3. 实现要点

### 3.1 帧号与时间基准

所有 rep、相位和证据时间都以原视频第 0 帧为时间零点：

```text
start_time_s  = start_frame / fps
bottom_time_s = bottom_frame / fps
end_time_s    = end_frame / fps
```

同时保留 clip 相对时间 `clip_start_time_s`、`clip_bottom_time_s`、`clip_end_time_s`。这样 T1-4C/D 的 `anchor_time_s` 可以直接用于原视频 seek。

例：`0020_squat_set01` 第一个完整 rep 从原视频帧 `1917` 开始，`start_time_s=63.9`；旧版本的 clip 相对时间 `0.833333` 保留为 `clip_start_time_s`。

### 3.2 相位定义

| 动作 | rep 边界 | 底部 |
|---|---|---|
| 深蹲 | `standing_knee_extended`，膝关节角峰值 | 膝关节角谷值 |
| 俯卧撑 | `arms_extended_top`，肘关节角峰值 | 肘关节角谷值 |

`task1_4_exercise_specs.json` 与 `phase_guidance.json` 均显式写入 `boundary_polarity=peak`、`bottom_polarity=trough`，避免把底部极性误用于 rep 边界。

### 3.3 rep 分割与完整性过滤

优先复用 task1-3 的 AIFit/active-bout。针对 clip 级使用再增加一个相位循环兜底：

1. 在相位关节角上检测显著谷值；
2. 以谷值间的高位相位穿越点作为相邻 rep 边界；
3. 仅当相位循环比 AIFit 产生更多完整 rep 时启用，`method` 记为 `phase_cycles`。

该兜底解决了两个已观察问题：`0034_squat_set01` 的 AIFit 边界整体落在底部；`0161_pushup_set01` 的 active-bout 把前 3 个 GT rep 误判为入场过渡。

每个候选 rep 输出 `quality_flags`、`width_ratio`、`start_phase_level`、`end_phase_level`。以下情况不进入完整 rep：

- 起止不在站立/伸臂相位；
- bottom 落在 rep 边界附近，缺少内部底点；
- 宽度明显异常且伴随低置信/低形状相似度；
- task1-3 已判定为 `low_local_amplitude`、`outside_active_bout` 等。

被剔除记录写入 `rejected_reps.csv`，不会静默消失。

### 3.4 动作姿态门控与相位信号候选

俯卧撑与深蹲共用同一套 rep 边界逻辑，但热身、起身和站立过渡也可能产生周期性肘/膝信号。为避免把这些片段当成训练 rep，T1-4B 在 bottom 帧增加动作姿态门控：

```text
torso_vertical_span_bottom_m = abs(肩中点 Y - 髋中点 Y)   # MediaPipe world 坐标

pushup: torso_vertical_span_bottom_m <= 0.30 m   # 接近水平
squat:  torso_vertical_span_bottom_m >= 0.30 m   # 保持直立
```

不满足的 rep 以 `exercise_posture_mismatch` 写入 `rejected_reps.csv`；通过的 rep 重新编号，`phases.csv` 同步重映射。`analysis.json` 的 `segmentation.posture_gate` 记录每个候选的 bottom 帧、测量值、rep 中位数和水平帧比例。

仅靠门控会消除误报，但可能把被热身带偏的真实 rep 一并清零。因此分割阶段同时生成两个候选：

1. 原有整段角度评分 + AIFit/active-bout；
2. 动作相位信号（俯卧撑 `left_elbow_flexion`、深蹲 `left_knee_flexion`）重新分割。

两个候选都先经过姿态门控，再按 `max_posture_gated_valid_reps` 选择有效 rep 更多的方案；平票保留原候选。选择证据写入 `segmentation.segmentation_candidates`。

### 3.5 质量门控

当前硬门控使用帧覆盖率、姿态检出率、关键关节 visibility 的 10% 分位、身体在画面内比例、视角是否已知，以及是否存在至少一个可用指标。

可见性同时输出三个诊断：

- `key_joint_visible_rate`：所有动作关节样本中 visibility ≥ 0.5 的比例；
- `all_exercise_joints_visible_rate`：同一帧全部动作关节均可见的严格比例；
- `all_required_visible_rate`：兼容旧字段的严格比例别名，不作为硬门控。

例如俯卧撑正面视角中手腕/肘部常在画面边缘，严格“全关节同帧可见率”会很低，但平均与 q10 visibility 仍可支持分割；因此不把严格比例误当作唯一质量门槛。

## 4. Dev 集结果

| clip | 动作 | quality | 完整 rep | 剔除 | 方法 | mean vis | q10 vis | warnings |
|---|---|---|---:|---:|---|---:|---:|---|
| `0001_pushup_set01` | pushup | pass | 11 | 0 | phase_cycles | 0.862 | 0.816 | - |
| `0004_pushup_set01` | pushup | warn | 5 | 10 | aifit | 0.925 | 0.816 | `exercise_posture_reps_rejected` |
| `0005_pushup_set01` | pushup | warn | 3 | 1 | aifit | 0.895 | 0.809 | `exercise_posture_reps_rejected` |
| `0007_pushup_set01` | pushup | pass | 8 | 2 | aifit | 0.852 | 0.815 | - |
| `0020_pushup_set01` | pushup | pass | 13 | 0 | phase_cycles | 0.841 | 0.814 | - |
| `0020_squat_set01` | squat | pass | 12 | 0 | aifit | 0.995 | 0.995 | - |
| `0021_pushup_set01` | pushup | pass | 11 | 0 | aifit | 0.848 | 0.812 | - |
| `0034_squat_set01` | squat | pass | 10 | 0 | phase_cycles | 0.991 | 0.989 | - |
| `0046_squat_set01` | squat | pass | 9 | 0 | aifit | 0.996 | 0.994 | - |
| `0104_squat_set01` | squat | pass | 9 | 0 | phase_cycles | 0.997 | 0.996 | - |
| `0156_pushup_set01` | pushup | warn | 9 | 1 | aifit | 0.845 | 0.793 | `exercise_posture_reps_rejected` |
| `0156_squat_set01` | squat | pass | 13 | 0 | aifit | 0.992 | 0.991 | - |
| `0161_pushup_set01` | pushup | pass | 5 | 0 | phase_cycles | 0.844 | 0.794 | - |

`0004_pushup_set01` 的原始整段分割只找到前段热身的 2 个候选；姿态门控将二者剔除后，相位信号候选恢复出后段 5 个真实俯卧撑。剔除的 10 个 rep 中，5 个为热身姿态不匹配，5 个为内部不完整候选。`0005_pushup_set01` 保留 3 个跪姿俯卧撑，剔除 1 个起身 rep；`0156_pushup_set01` 保留 9 个俯卧撑，剔除 1 个起身 rep。

## 5. 盲标对照：0161_pushup

输入 GT：`results/segmentation_final/data/segmentation/annotation_blind_eval/blind_reps.csv` 中 `segment_id=0161_pushup` 的 5 个 `accept` rep。冻结评估文件为 `results/clip_analysis_v1/evaluation_0161_vs_blind.json`。

| 指标 | 结果 |
|---|---:|
| prediction / GT | `5 / 5` |
| IoU@0.30 匹配 | `5 / 5` |
| mean matched IoU | `0.681` |
| mean abs start error | `13.4` 帧（0.45 s） |
| mean abs end error | `9.8` 帧（0.33 s） |

最大边界误差出现在第一个 rep 的起点（`+24` 帧）；首个 rep 前的入场段本身也处于伸臂高位，仅靠角度相位无法与动作开始时刻完全区分。该结果只作为 T1-4B 回放定位可用性的验收证据，不用于调 holdout 阈值。

## 6. 输出

每个 clip 目录：

```text
VLS/V0/results/clip_analysis_v1/<clip_id>/
├── analysis.json          # 完整机读结果、方法、质量、rep/相位证据
├── clip.json              # manifest 快照 + 实际处理 trim
├── quality.json           # 门控、可见性、视角支持
├── exercise.json          # 动作标签来源（当前为 manifest hint）
├── reps.csv               # 完整 rep、绝对/相对时间、相位与质量标记
├── rejected_reps.csv      # 被剔除 rep 及原因
├── phases.csv             # top_start/bottom/top_end
└── angles/                # 角度、世界坐标与 meta
```

聚合文件：

```text
VLS/V0/results/clip_analysis_v1/summary.json
VLS/V0/results/clip_analysis_v1/evaluation_0161_vs_blind.json
```

`summary.json` 现在从所有子目录的 `analysis.json` 重建，不再只保留最近一次命令的 clip。

## 7. 复现与测试

```bash
# 13 个 dev clip
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/analyze_clip.py \
  --clip-id 0001_pushup_set01 0004_pushup_set01 0005_pushup_set01 \
  0007_pushup_set01 0020_pushup_set01 0021_pushup_set01 \
  0156_pushup_set01 0161_pushup_set01 0020_squat_set01 \
  0034_squat_set01 0046_squat_set01 0104_squat_set01 \
  0156_squat_set01 --overwrite

# 0161 盲标对照
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/evaluate_task1_4b_blind.py

# 单元测试
VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test_task1_4_clip.py' -v
```

测试覆盖绝对时间、深蹲/俯卧撑边界极性、相位循环、尾部 rep 过滤、正面视角指标门控、姿态门控、相位信号候选择优和 summary 重建。全量测试 `54/54` 通过。

## 8. 已知限制

- 当前只跑 dev；holdout 尚未批量执行，也未用于规则调整。
- 相机运动未估计；手机手持场景仍需在 T1-4D 前加入 `camera_motion` 门控或警告。
- 多人场景未检测；当前 MediaPipe 固定 `num_poses=1`。
- 当前只输出 rep/相位/质量，不包含 `features.csv`、`issues.json`、`report.html` 和圈画帧；这些属于 T1-4C/D。
- 正面视角下不能把“深度不足”“身体直线破坏”等矢状面结论写入报告。
- 姿态门控阈值 `0.30 m` 是首版工程值，尚未用更多受试者和机位校准；如果起身动作在下蹲/撑地阶段刚好呈水平，仍可能被误保留。

## 9. 下一步

T1-4C 应直接消费本目录的 `reps.csv`、`phases.csv` 和 `angles/`：

1. 只对 `quality_status != reject` 且 `rep_count >= 1` 的 clip 计算指标；
2. 指标必须读取 `view_support.disabled_metrics`，disabled 项不得生成问题卡；
3. `anchor_frame/anchor_time_s` 直接使用 `reps.csv` 的原视频绝对时间和 bottom 帧；
4. 先完成正面可评估指标，侧视指标等待补录或 Fit3D，不伪造。
