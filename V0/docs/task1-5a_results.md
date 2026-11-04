# task1-5A：Fit3D 适配、受试者划分与视角审计

## 1. 执行结论

状态：**完成（2026-10-26）**

> **T1-5B 修订（2026-10-26）**：T1-5B 用逐 clip GT 相机空间深度复核发现，T1-5A 的 `camera_side` 符号整体写反，导致 `side_left/side_right` 与 `oblique_left/oblique_right` 的左右后缀互换。当前 manifest、overrides、view audit 已统一修正；64/64 与 GT 近侧深度一致。本文第 5 节的 pilot 数字保留为 T1-5A 历史快照，最终结果以 `docs/task1-5b_results.md` 为准。

- 完成 Fit3D → T1-4 manifest 适配，固定 `squat`/`pushup` 两个动作、8 个 train 受试者、4 路相机。
- 完成受试者级 dev/holdout 划分和 6 条 test smoke 记录；dev 与 holdout 无受试者重叠。
- 完成 64 路 train 视角的相机几何判定、contact sheet 和逐条人工复核；自动分类与人工分类 `64/64` 一致，修正 `0` 条。
- 完成 `s09` 的 2 动作 × 4 相机 pilot，共 8 条视频；MediaPipe、rep 分割、质量门控和输出结构全部实际跑通。
- 完成 8 条 pilot 的 3D GT 投影叠加和 2 张 rep 事件曲线。25 个 GT 关节在所有抽样帧均落入画面，外参约定与视频对齐。
- pilot 暴露出侧视下的真实问题：当前 T1-4 默认的左侧肘/肩与“整段角度选主信号”策略在多路侧视中不稳定。该问题进入 T1-5B 处理，不在 T1-5A 调阈值。
- 侧视指标继续保持 disabled。Fit3D 可以验证测量能力，但 T1-5A 不据此启用产品规则。

## 2. 数据与划分

| 项目 | 结果 |
|---|---:|
| train 动作实例 | `16`（8 人 × 2 动作） |
| train 相机视频 | `64`（16 × 4） |
| train 唯一 rep | `96`（每实例 6 个） |
| test smoke 视频 | `6`（3 人 × 2 动作，单视角） |
| test 3D / rep GT | `0`，只允许 smoke |
| 视频规格 | `900×900`、`50 FPS`、H.264 |

split 以受试者为单位：

| split | 受试者 | 用途 |
|---|---|---|
| dev | `s03,s04,s05,s07` | 调试与规则分析 |
| holdout | `s08,s09,s10,s11` | 留出验证，不参与调参 |
| smoke | `s02,s12,s13` | 只测读入/推理/输出结构，不计算 GT 误差 |

manifest 校验结果：

```text
passed=true
train_clips=64
smoke_clips=6
action_instances=16
unique_reps=96
```

冻结 manifest 为 `configs/task1_5_fit3d_clips.json`，校验记录为 `results/task1_5/manifest/validation.json`。

## 3. 视角审计

视角角定义为身体额状面法向与 `pelvis → camera` 射线的夹角；统计使用 rep 帧 ±3 帧邻域，避免整段动作的躯干前倾制造假 `unknown`。

| 动作 | 视角分布 | 角度范围 | 中位数 |
|---|---|---|---:|
| `squat` | `rear=15`、`front=12`、`oblique_left=2`、`oblique_right=3` | `17.8–34.9°` | `26.3°` |
| `pushup` | `side_right=15`、`side_left=16`、`oblique_right=1` | `59.3–89.5°` | `76.5°` |

关键结论：

- Fit3D 的 4 路相机不是固定前/后/左/右，不能按相机 ID 硬编码视角。
- `squat` 没有纯侧视，只有正面、背面和浅斜视；不能补齐 T1-4 缺失的“纯侧视徒手深蹲”视频验证。
- `pushup` 有 31 路纯侧视和 1 路斜侧视，可以进入 T1-5B 的侧视姿态验证。
- 64 路 contact sheet 均保留审计记录：16 张受试者-动作分组图，加 2 张动作总览图。

人工复核汇总：

```text
reviewed_clips=64
pending_clips=0
matches=64
corrections=0
```

上述 `corrections=0` 指 T1-5A 的 front/rear/side/oblique 类别复核没有改类；T1-5B 的深度审计随后修正了 32 条 side/oblique 的左右后缀，并同步更新了 override 版本。

## 4. 数据边界

- Fit3D `squat` 是**杠铃背蹲**，不是 T1-4 的徒手深蹲。它可以验证关节角定义、投影和视角门控，但不能直接提供徒手深蹲的指标分布或反馈基线。
- Fit3D `pushup` 是标准地面俯卧撑，可作为侧视验证的主要数据源。
- 384 个“视角 × rep”观测不是 384 个独立动作实例。统计主单位必须保持 `subject-action`，相机只作为同实例内的重复观测。
- test 没有 `joints3d_25` 和 `rep_ann`；manifest 中标记 `smoke_only=true`、`has_gt=false`、`view=unknown`，不得计算 MPJPE、角度误差或 rep IoU。

## 5. s09 pilot 结果

实际运行 8 条视频，未跳过失败样本：

| clip | view | status | quality | rep/GT | pose 检出 | q10 visibility |
|---|---|---|---|---:|---:|---:|
| `fit3d_s09_squat_50591643` | rear | ok | pass | `6/6` | `1.000` | `0.955` |
| `fit3d_s09_squat_58860488` | rear | ok | pass | `6/6` | `1.000` | `0.942` |
| `fit3d_s09_squat_60457274` | oblique_left | ok | warn | `5/6` | `1.000` | `0.908` |
| `fit3d_s09_squat_65906101` | front | ok | warn | `5/6` | `1.000` | `0.858` |
| `fit3d_s09_pushup_50591643` | side_right | rejected_quality | reject | `1/6` | `0.660` | `0.741` |
| `fit3d_s09_pushup_58860488` | side_left | partial | warn | `0/6` | `1.000` | `0.794` |
| `fit3d_s09_pushup_60457274` | side_left | ok | warn | `6/6` | `0.997` | `0.803` |
| `fit3d_s09_pushup_65906101` | side_right | ok | pass | `7/6` | `1.000` | `0.840` |

汇总：

| 指标 | 结果 |
|---|---:|
| status | `ok=6`、`partial=1`、`rejected_quality=1` |
| quality | `pass=3`、`warn=4`、`reject=1` |
| 完整 rep | `36 / 48` |
| rep count MAE | `1.75 / clip` |

pilot 不是验收通过，而是定位 T1-5B 的具体工作项：

1. `fit3d_s09_pushup_50591643` 的 pose/body-in-frame 只有 `66%`，先由质量门控正确拒绝，不能把 1 个 rep 包装成完整结果。
2. `fit3d_s09_pushup_58860488` 的左侧肘角几乎无变化，而右侧肘角范围为 `92.8–167.7°`；当前固定取左肘导致相位信号不可用、0 个完整 rep。
3. 部分 pushup clip 的“整段角度选主信号”落到 `torso_flexion` 或 `shoulder_flexion`，不是肘角。这说明侧视不能沿用正面默认的 `legacy` 选角逻辑。
4. squat 的两条 `5/6` 均由动作姿态门控剔除 1 个候选 rep；门控方向正确，但阈值需要在 T1-5B 用 GT 和近侧可见性复核。
5. `fit3d_s09_pushup_65906101` 的 `7/6` 是多余 rep，说明仅靠姿态门控不足以解决侧视边界误检。

因此 T1-5B 的第一优先级不是扫全量指标，而是：

```text
视角/相机侧 → 近侧关节可见性 → 选择相位信号与主角度 → 再做 rep 和角度误差评估
```

## 6. GT 投影与 rep 语义

已生成 `s09` 8 条 pilot 的 GT 骨架叠加和 2 张 rep 曲线：

```text
results/task1_5/manifest/gt_overlay/
├── overlays/<clip_id>.jpg
├── curves/s09_squat_gt_rep_curve.png
├── curves/s09_pushup_gt_rep_curve.png
└── summary.json
```

投影约定已经用画面叠加重合确认：

```text
camera_center = extrinsics.T
t = -R @ camera_center
K = intrinsics_wo_distortion
```

8 条 clip 的抽样帧均为 `25/25` 关节可见，GT 关节编号与画面左右侧一致。T1-5A 当时未单独核验 `camera_side` 的符号；该项由 T1-5B 的相机空间深度审计补上，并发现和修正了整体的左右后缀反转。

rep 语义有一个必须先冻结的限制：

- `rep_ann` 帧的膝/肘角接近伸展高位：s09 squat 标注中位 `164.5°`，曲线 p95 `168.9°`；s09 pushup 标注中位 `154.3°`，曲线 p95 `162.0°`。
- 相邻标注帧之间的谷值中位数分别约为 squat `64.2°`、pushup `70.2°`。
- 因此当前证据支持 `rep_ann` 是**接近伸展的 top/边界事件**，不是 bottom 帧。
- 在 T1-5D 明确语义前，`bottom_gt_status` 固定为 `not_confirmed_do_not_use_as_bottom`，不得直接把 `rep_ann` 当 bottom GT 计算误差。

## 7. 视角与指标策略

T1-5A 只冻结“视角支持”，不冻结指标可用性：

- `squat`：正面/背面/浅斜视可继续支持现有正面指标和 rep 指标；矢状面指标保持 disabled。
- `pushup`：纯侧视可作为 T1-5B 的主要验证数据；但在近侧关节选择完成前，`elbow_flexion_rom`、`body_line_angle`、`hip_sag_ratio` 等仍保持 disabled。
- manifest 中侧视指标只写入 `side_metric_candidates`，`side_metrics_enabled=false`，状态为 `pending_t1_5b_validation`。

## 8. 产物

| 路径 | 内容 |
|---|---|
| `scripts/build_task1_5_fit3d_manifest.py` | Fit3D → manifest、路径/N/rep/split 校验 |
| `scripts/audit_task1_5_views.py` | 相机几何视角、contact sheet、人工覆盖校验 |
| `scripts/render_task1_5_gt_overlay.py` | GT 骨架投影、rep 曲线、语义证据 |
| `configs/task1_5_fit3d_clips.json` | 冻结的 70 条 clip manifest |
| `configs/task1_5_view_overrides.json` | 64 条人工复核记录 |
| `results/task1_5/manifest/validation.json` | manifest 校验结果 |
| `results/task1_5/manifest/view_audit.csv` | 逐 clip 视角审计表 |
| `results/task1_5/manifest/view_audit_summary.json` | 视角分布与复核汇总 |
| `results/task1_5/manifest/contact_sheets/` | 16 张分组图 + 2 张总览图 |
| `results/task1_5/manifest/gt_overlay/` | pilot GT 叠加、rep 曲线和 JSON |
| `results/task1_5/pilot_analysis/` | 8 条 pilot 的完整 T1-4 输出与 `summary.json` |

## 9. 复现

```bash
cd /mnt/storage2/data_guest

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/build_task1_5_fit3d_manifest.py \
  --dataset-root VLS/dataset/Fit3D \
  --output VLS/V0/configs/task1_5_fit3d_clips.json \
  --view-overrides VLS/V0/configs/task1_5_view_overrides.json

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/audit_task1_5_views.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --output-dir VLS/V0/results/task1_5/manifest \
  --overrides VLS/V0/configs/task1_5_view_overrides.json

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/render_task1_5_gt_overlay.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --clip-id fit3d_s09_squat_50591643 fit3d_s09_squat_58860488 \
    fit3d_s09_squat_60457274 fit3d_s09_squat_65906101 \
    fit3d_s09_pushup_50591643 fit3d_s09_pushup_58860488 \
    fit3d_s09_pushup_60457274 fit3d_s09_pushup_65906101 \
  --output-dir VLS/V0/results/task1_5/manifest/gt_overlay

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/analyze_clip.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --clip-id fit3d_s09_squat_50591643 fit3d_s09_squat_58860488 \
    fit3d_s09_squat_60457274 fit3d_s09_squat_65906101 \
    fit3d_s09_pushup_50591643 fit3d_s09_pushup_58860488 \
    fit3d_s09_pushup_60457274 fit3d_s09_pushup_65906101 \
  --output-dir VLS/V0/results/task1_5/pilot_analysis --overwrite
```

## 10. 测试与下一步

全量单元测试 **71/71 通过**，覆盖 manifest 计数、受试者 split、路径与 rep 边界、视角分类、侧视指标禁启用策略、contact sheet 参考帧、GT 角度、rep 语义和 overlay 选帧。

T1-5B 应直接使用本 manifest，但先完成：

1. 用相机侧 + 近侧关节 visibility 选择相位信号，不再固定左侧肘/肩；
2. 先在 `s09` 的 8 条 pilot 上复跑，确认侧视 rep 数不再系统性缺失或多报；
3. 再跑 64 路 train，按 `exercise × view × joint` 评估 MediaPipe 与 GT 的角度误差；
4. 在 T1-5B 结论出来前，所有矢状面指标继续保持 disabled。
