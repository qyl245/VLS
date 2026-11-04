# task1-4A：clip 候选、视角筛选与数据划分

## 1. 执行结论

状态：**pass**（2026-10-25）

- 从 QEVD-FIT-COACH 长视频中严格按 `action_display` 筛出 41 个纯深蹲/俯卧撑 clip：14 个 `squats`、27 个 `pushups`。
- 建立 26 个入选 clip：深蹲 `5 dev + 5 holdout`，俯卧撑 `8 dev + 8 holdout`；其余 15 个保留为 reserve。
- 所有划分按 `video_id` 隔离，`0020`、`0153`、`0156`、`0160` 同时含两个动作，仍整体进入同一 split。
- 41 个 clip 的视频均为 `640x360 / 30 FPS`；25 个需要 `90cw`，16 个原生方向正确。
- 人工复核发现，QEVD 这些纯深蹲/俯卧撑窗口几乎全部是正面视角：40 个 `front`、1 个 `oblique_left`、0 个可靠侧面。26 个入选 clip 全部为 `front`。

## 2. 关键限制

这是一处必须在 T1-4B 前处理的数据缺口，而不是筛选错误：

| 指标组 | 入选集可用性 | 原因 |
|---|---|---|
| 节奏、rep 时长、稳定性 | 可用 | 视角无关 |
| 深蹲膝外翻、膝/髋左右差 | 可用 | 正面/斜前视角支持 |
| 俯卧撑肘/肩左右差 | 可用 | 正面/斜前视角支持 |
| 深蹲深度、膝/髋屈曲 ROM、躯干前倾 | **全部 disabled** | 没有可靠侧/斜侧视角 |
| 俯卧撑下放深度、身体直线、塌腰/撅臀 | **全部 disabled** | 没有可靠侧/斜侧视角 |

因此当前 manifest 的 `screening_findings.sagittal_metric_status` 为
`disabled_for_all_selected_clips`。如果直接进入 T1-4B，只能先完成“正面指标 + rep/相位 + 回放报告”的垂直切片，不能诚实地验收下蹲深度、躯干前倾或俯卧撑身体直线。

建议优先级：

1. **推荐**：补录 4–6 条固定机位、全身可见的深蹲/俯卧撑侧或斜侧 clip，作为 2–3 条 dev、2–3 条 holdout。
2. 若暂时无法补录：T1-4B 先只做正面可评估指标，所有矢状面问题卡保持 `suppressed=true`，等数据补齐后再启用。
3. 等 FIT3D 申请通过后再补系统性的侧视/3D 评估，不把 QEVD 正面视角伪装成侧视结果。

## 3. 数据筛选

来源：`VLS/dataset/QEVD-FIT/QEVD-FIT-COACH/feedbacks_long_range.json`

| 项目 | 数量 |
|---|---:|
| 长视频 | 149 |
| 全部动作段 | 894 |
| 纯深蹲 `action_display == "squats"` | 14 |
| 纯俯卧撑 `action_display == "pushups"` | 27 |
| task1-4A 候选 clip | 41 |
| 入选 dev + holdout | 26 |
| 视频级 reserve 或 clip 级 reserve | 15 |

注意：现有 `segment_reps.py` 的 `ACTION_ALIASES` 使用子串匹配，会把 `squat jumps` 和 `squat kicks` 归成 `squat`。本轮没有使用该动作类别列，而是直接按原始 `action_display` 精确过滤，因此没有混入这两类动作。

## 4. 数据划分

### 深蹲

| role | video_id |
|---|---|
| dev | `0020`, `0034`, `0046`, `0104`, `0156` |
| holdout | `0075`, `0125`, `0153`, `0160`, `0184` |
| reserve | `0106`, `0127`, `0177`, `0211` |

### 俯卧撑

| role | video_id |
|---|---|
| dev | `0001`, `0004`, `0005`, `0007`, `0020`, `0021`, `0156`, `0161` |
| holdout | `0044`, `0065`, `0067`, `0090`, `0103`, `0137`, `0143`, `0160` |
| reserve | `0008`, `0047`, `0118`, `0122`, `0150`, `0151`, `0152`, `0153`, `0157`, `0159`, `0163` |

`0153` 的视频级 split 是 `holdout`（其深蹲入选），但 `0153_pushup_set01` 在 manifest 中为 `phase_role=reserve`，原因是前半段主要是跪姿入场，不能算作干净的 pushup 评估窗口。

主要 reserve 原因：

| clip | 原因 |
|---|---|
| `0106_squat_set01` | 中间有明显转身和动作不连续 |
| `0008_pushup_set01` | 大部分窗口为站立，pushup 只在尾部 |
| `0047_pushup_set01` | 大部分窗口为站立，仅中点附近有 pushup 姿态 |
| `0118_pushup_set01` | 从地面入场并站立恢复，动作窗口不纯 |
| `0153_pushup_set01` | 前半段跪姿入场 |
| `0150_pushup_set01` | 低机位广角，深度畸变明显 |
| `0157_pushup_set01` | 唯一可见斜角，但仍不足以做可靠侧视指标 |

## 5. 旋转与视角

### 旋转

- 自动候选：`none / 90cw / 180 / 90ccw`，每个 clip 在 8 个窗口分位取样。
- 深蹲优先“头 < 肩 < 髋 < 踝”的直立链；俯卧撑优先“头在手腕上方、手腕在肩下方、肩/髋线水平”。
- 全部 41 个 clip 的旋转 margin 均大于 `0.13`；26 个入选 clip 的最小 margin 为 `0.140`（`0007_pushup_set01`）。
- 25 个使用 `90cw`，16 个使用 `none`。

### 视角

- 自动肩宽/躯干长比在竖屏广角视频上会被透视和姿势压缩，首轮将 6 个正面深蹲误判为侧面。
- 最终以 `squat_selected_contact_sheet.jpg`、`pushup_selected_contact_sheet.jpg` 和 `candidate_contact_sheet.jpg` 逐条复核，人工覆盖写入 `task1_4_screening_overrides.json`。
- 最终分布：`front=40`、`oblique_left=1`（`0157_pushup_set01`）、side/rear 为 0。

## 6. 指标门控

### 正面深蹲

可用：

```text
rep_duration_s
rep_duration_cv
spine_stability_bottom
knee_valgus_bottom
knee_asymmetry_bottom
hip_asymmetry_bottom
```

禁用：

```text
knee_flexion_rom
bottom_knee_angle
hip_flexion_rom
torso_lean_bottom
hip_depth_ratio
```

### 正面俯卧撑

可用：

```text
rep_duration_s
rep_duration_cv
spine_stability_bottom
elbow_asymmetry_bottom
shoulder_asymmetry_bottom
```

禁用：

```text
elbow_flexion_rom
bottom_elbow_angle
body_line_angle
hip_sag_ratio
```

## 7. 校验

`validation.json` 与独立复核均通过：

- `clip_id` 唯一；
- `0 <= start_frame < end_frame < frame_count`；
- `action_display` 只含 `squats` / `pushups`；
- rotation、view 均在允许枚举中；
- 同一视频不跨 split；
- 每个 clip 的 usable/disabled 指标无交集；
- 入选数量严格为深蹲 `5+5`、俯卧撑 `8+8`；
- 26 个入选 clip 当前没有 `quality_flags`。

复现命令：

```bash
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/build_task1_4_manifest.py
```

MediaPipe 使用隔离环境 `VLS/V0/.venv-mediapipe`。根 `.venv` 同时安装了 TensorFlow，其 protobuf 版本与 MediaPipe 导入链冲突；不要为了跑本脚本修改根环境的 protobuf/TensorFlow。

## 8. 产物

```text
VLS/V0/configs/task1_4_clips.json
VLS/V0/configs/task1_4_exercise_specs.json
VLS/V0/configs/task1_4_screening_overrides.json
VLS/V0/scripts/build_task1_4_manifest.py
VLS/V0/outputs/task1_4/clip_screening/
├── validation.json
├── rotation_scores.json
├── view_screening.csv
├── manifest_preview.csv
├── split_summary.csv
├── candidate_contact_sheet.jpg
├── squat_selected_contact_sheet.jpg
└── pushup_selected_contact_sheet.jpg
```

## 9. 下一步

T1-4B 应只读取本 manifest，不覆盖 `segmentation_final`，也不把 task1-3 的 provisional 分割直接当作问题级 GT。若采用“先做正面垂直切片”的方案，建议先用 `0020_squat_set01` 和 `0020_pushup_set01` 验证 MediaPipe → 角度 → rep/相位 → 质量门控链路；holdout 保持冻结，不参与阈值调整。

如果目标是尽快验收“下蹲深度不足、躯干前倾、身体直线破坏”这类核心反馈，应先补录侧/斜侧 clip，否则这些能力在 T1-4B/C 只能显示为不可评估。
