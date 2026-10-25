# task1-4C：逐 rep 指标、问题规则与证据锚点

## 1. 执行结论

状态：**完成（2026-10-25）**

- 新增 `scripts/build_issue_cards.py`，直接消费 T1-4B 的 `reps.csv`、`phases.csv`、`angles/`，不重跑 MediaPipe。
- 13 个 dev clip 共 114 个完整 rep，输出 1031 行逐 rep 指标、43 张活动问题卡、35 条屏蔽/未评估记录。
- 活动问题卡中 `high=7`、`medium=10`、`low=26`；17 张 medium/high 卡全部有合法 `anchor_frame`、`anchor_time_s` 和 `evidence_interval`，anchor 均位于代表 rep 内且落在证据区间内。
- 所有正面视角不支持的矢状面指标均未生成 activity 卡，只输出 `suppressed=true` 的明确未评估记录。
- 当前没有教练参考 clip 或专家问题级标签，阈值全部是 `provisional_engineering`，不能据此报告 precision/recall 或医学结论。

## 2. 输入与输出

输入目录：

```text
VLS/V0/results/clip_analysis_v1/<clip_id>/
├── analysis.json
├── quality.json
├── reps.csv
├── phases.csv
└── angles/
    ├── angles.csv
    ├── angles_wide.csv
    └── landmarks_body_frame.json
```

输出目录：

```text
VLS/V0/results/feedback_v1/
├── <clip_id>/
│   ├── features.csv
│   └── issues.json
├── features_all.csv
├── issues_summary.json
└── thresholds_used.json
```

`features.csv` 使用长表，每个指标一行：

```text
clip_id,exercise,rep_id,metric,scope,raw_value,unit,confidence,source,
view_requirement,status,reference_value,deviation,severity_candidate,
anchor_frame,anchor_time_s,note
```

`status` 取值包括 `ok`、`low_confidence`、`disabled`、`insufficient_reps`、`unavailable` 和 `unusable_low_confidence`。没有教练参考时 `reference_value` 保持为空，严重度只由版本化 provisional 阈值给出。

## 3. 指标实现

### 3.1 正面垂直切片

| 动作 | 当前可评估指标 |
|---|---|
| 深蹲 | `rep_duration_s`、`rep_duration_cv`、`spine_stability_bottom`、`knee_valgus_bottom`、`knee_asymmetry_bottom`、`hip_asymmetry_bottom` |
| 俯卧撑 | `rep_duration_s`、`rep_duration_cv`、`spine_stability_bottom`、`elbow_asymmetry_bottom`、`shoulder_asymmetry_bottom` |

深蹲深度/ROM、躯干前倾、髋部下降比例，以及俯卧撑下放深度、身体直线和塌腰/撅臀仍因缺少侧/斜侧视角而 disabled。

### 3.2 膝内扣角修正

T1-4B 的 `left/right_knee_valgus_coronal` 是冠状面投影后的膝夹角；在深蹲底部会随屈膝从约 60° 变到 160°，不能直接解释为“内扣角度”。T1-4C 改为从身体坐标系中的髋-膝-踝几何重新计算：

```text
medial_offset = 膝到髋-踝冠状面直线的内侧偏移
knee_valgus_bottom = atan2(max(0, medial_offset), hip_ankle_coronal_length)
```

正值表示膝向身体中线偏移，取左右两侧较大值。输出 `source` 记作 `landmarks_body_frame.hip_knee_ankle_coronal`，保证该指标可追溯，而不是继续复用有歧义的膝夹角。

### 3.3 稳定性和左右差

- `spine_stability_bottom` 取 bottom 前后默认 15 帧窗口内的最大 `spine_stability`，单位修正为 `variance`。原 specs 中的 `deg` 是错误单位。
- `knee_asymmetry_bottom` / `hip_asymmetry_bottom` 读取 bottom 帧的预计算左右差。
- `elbow_asymmetry_bottom` / `shoulder_asymmetry_bottom` 分别计算 bottom 帧左右肘、左右肩 flexion 的绝对差。
- `rep_duration_cv` 使用完整 rep 时长的总体标准差除以均值；少于 3 个 rep 时为 `insufficient_reps`，不判断组级节奏。

## 4. 阈值与分布

阈值与中文文案位于 `configs/task1_4_feedback_templates.json`，运行时快照写入 `results/feedback_v1/thresholds_used.json`。

| 指标 | low | medium | high | 单位 |
|---|---:|---:|---:|---|
| `knee_valgus_bottom` | 12 | 20 | 30 | deg |
| `knee_asymmetry_bottom` | 5 | 8 | 15 | deg |
| `hip_asymmetry_bottom` | 5 | 8 | 15 | deg |
| `elbow_asymmetry_bottom` | 8 | 12 | 20 | deg |
| `shoulder_asymmetry_bottom` | 8 | 12 | 20 | deg |
| `rep_duration_cv` | 0.15 | 0.20 | 0.35 | ratio |
| `spine_stability_bottom` | 0.03 | 0.06 | 0.10 | variance |

Dev 集实际分布：

| 指标 | n | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| `knee_valgus_bottom` | 53 | 13.17 | 20.03 | 30.77 | 49.47 |
| `knee_asymmetry_bottom` | 53 | 0.55 | 1.71 | 4.93 | 9.50 |
| `hip_asymmetry_bottom` | 53 | 0.86 | 2.88 | 5.89 | 8.88 |
| `elbow_asymmetry_bottom` | 61 | 1.59 | 4.67 | 10.80 | 43.28 |
| `shoulder_asymmetry_bottom` | 61 | 0.68 | 3.97 | 11.24 | 25.94 |
| `rep_duration_cv` | 12 | 0.062 | 0.129 | 0.369 | 0.402 |
| `spine_stability_bottom` | 114 | 0.006 | 0.015 | 0.045 | 0.113 |

这些数值只用于初版工程分流和回放定位。尤其膝内扣、肩部不对称和躯干方差受 3D 姿态噪声影响明显，T1-4D 展示时必须保留低置信和相机运动警告。

## 5. 问题判定与证据选择

每个 issue 先按动作配置聚合指标，再按 rep 生成候选：

- 同一 `issue_code` 的连续 rep 合并为一张卡；不相邻 rep 分开，避免把中间正常 rep 误算进同一问题。
- `frequency` 使用“触发 rep 数 / 完整 rep 数”；组级节奏问题使用全部 rep，并以时长最异常的 rep 作为代表。
- 排序依次为严重度、频率、置信度；每段 clip 最多 3 张 `show_by_default=true`，完整问题列表仍保留。
- 置信度 `0.35–0.55` 最多降为 `low`；低于 `0.35` 不生成活动卡。当前 dev 中只有 2 行进入 `low_confidence`，没有低于最低置信度的活动问题。

Anchor 规则：

| 问题类型 | anchor |
|---|---|
| 膝内扣、左右差、稳定性 | 代表 rep 的 bottom 或 bottom 邻域极值帧 |
| 节奏 | 时长最异常 rep 的中点 |
| 已禁用矢状面指标 | 不生成 anchor，写 `suppressed=true` 与原因码 |

证据区间默认取 `anchor ± 8` 帧，并裁剪在代表 rep 内；节奏问题使用代表 rep 的完整区间。所有时间仍以原视频第 0 帧为零点，可直接用于 T1-4D 播放器 seek。

`issues.json` 同时保留：

- `evaluations`：每个 issue_code 的 `triggered / no_issue / suppressed` 状态、原因和指标状态；
- `issues`：含 suppressed 卡在内的完整卡片列表。

这样“没有发现问题”和“因为视角/置信度不评估”不会混淆。

## 6. Dev 集结果

| clip | 动作 | rep | 活动卡 | high | medium | low | 屏蔽记录 |
|---|---|---:|---:|---:|---:|---:|---:|
| `0001_pushup_set01` | pushup | 10 | 6 | 0 | 1 | 5 | 3 |
| `0004_pushup_set01` | pushup | 2 | 1 | 1 | 0 | 0 | 4 |
| `0005_pushup_set01` | pushup | 3 | 3 | 3 | 0 | 0 | 3 |
| `0007_pushup_set01` | pushup | 8 | 0 | 0 | 0 | 0 | 3 |
| `0020_pushup_set01` | pushup | 13 | 5 | 0 | 1 | 4 | 3 |
| `0020_squat_set01` | squat | 12 | 4 | 1 | 2 | 1 | 2 |
| `0021_pushup_set01` | pushup | 10 | 4 | 0 | 0 | 4 | 3 |
| `0034_squat_set01` | squat | 10 | 4 | 1 | 0 | 3 | 2 |
| `0046_squat_set01` | squat | 9 | 5 | 0 | 2 | 3 | 2 |
| `0104_squat_set01` | squat | 9 | 3 | 0 | 1 | 2 | 2 |
| `0156_pushup_set01` | pushup | 10 | 4 | 1 | 2 | 1 | 3 |
| `0156_squat_set01` | squat | 13 | 3 | 0 | 1 | 2 | 2 |
| `0161_pushup_set01` | pushup | 5 | 1 | 0 | 0 | 1 | 3 |
| **合计** | - | **114** | **43** | **7** | **10** | **26** | **35** |

问题代码计数：`pushup_asymmetry=15`、`pushup_stability_low=5`、`pushup_tempo_inconsistent=4`、`squat_asymmetry=9`、`squat_knee_valgus=6`、`squat_stability_low=4`。

35 条屏蔽记录由 24 条俯卧撑矢状面禁用、10 条深蹲矢状面禁用和 1 条 2-rep 节奏不足组成。`0004_pushup_set01` 不评估组级节奏，但仍允许输出单 rep 可见的左右差问题。

## 7. 复现与测试

```bash
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/build_issue_cards.py --split dev

VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test*.py' -v
```

新增 `tests/test_task1_4_issues.py`，覆盖：

- high_bad / low_bad 阈值边界；
- 低置信度降级和低于最低置信度拒答；
- 连续 rep 合并、非连续 rep 拆分；
- 视角禁用指标不进入活动问题；
- 少于 3 rep 不判断节奏；
- medium/high 问题必须有 anchor，且证据区间包含 anchor、不越出代表 rep。

全量测试 39/39 通过。

## 8. 已知限制

- 没有专家问题级 GT，本阶段只能验证 schema、规则分支、anchor 与拒答不变量，不能验证问题检测准确率。
- 所有 dev clip 为正面视角，矢状面问题仍未评估；不能用 suppressed 卡推断“动作正确”。
- 膝内扣由 MediaPipe 世界坐标推导，深蹲底部仍可能受深度估计误差和遮挡影响。
- 相机运动仍为 `not_evaluated`，稳定性阈值可能在手持视频中偏敏感。
- 问题卡当前只有文字、测量值和 overlay 指令，尚未生成 T1-4D 的实际圈画帧和 HTML 播放器。

## 9. 下一步

T1-4D 应直接消费 `results/feedback_v1/<clip_id>/issues.json`：

1. 用 `anchor_frame` 和 `anchor_time_s` seek 原视频并暂停；
2. 按 `visual_overlay.highlight_joints/draw` 画骨架、关节高亮和角度弧；
3. 用 `evidence_interval` 画时间线色带；
4. 展示 `evaluations` 中的 disabled/insufficient 原因，避免把未评估误解为正常；
5. 同步输出 `annotated_frames/<issue_id>_frame_<n>.png`，保证离线可检查。
