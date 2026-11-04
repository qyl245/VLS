# task1-5B：MediaPipe / Fit3D 姿态与角度基线

## 1. 执行结论

状态：**完成（2026-10-26）**。

- 64 路 Fit3D train 全部重新分析：`ok=60`、`partial=3`、`rejected_quality=1`；质量状态为 `pass=37`、`warn=26`、`reject=1`。
- 用逐 clip 的 GT 相机空间深度复核左右侧，发现并修复 T1-5A 的整体符号错误：原 `camera_side` 与真实近侧深度 **64/64 相反**。修复后 `camera_side`、view 左右后缀、近侧选择均为 `64/64` 一致。
- 37 条 side/oblique clip 全部选择相机近侧信号，远侧替代 `0` 条，`selected_side_mismatch=0`。
- 俯卧撑纯侧视的 holdout 屈肘 MAE：`side_left=15.19°`、`side_right=12.50°`；前者超过 `15°` 禁用线，后者处于 `10–15°` 低置信区。因同一动作已有侧别无法通过验收，**俯卧撑侧视肘角指标继续 disabled**。
- Fit3D 深蹲只有 `17.8–34.9°` 的正面/背面/浅斜视，没有纯侧视；浅斜视 MAE 为 `10.15–12.03°`，只能作为低置信证据，不能据此启用徒手深蹲侧视指标。
- 高误差集中在相机 `50591643`/`58860488`：近侧肘 MAE `23.83°`/`21.48°`，而同一相机远侧肘 MAE 仅 `13.56°`/`10.86°`。近侧关节 visibility 仍有 `0.88/0.92`，说明问题是**近侧肢体自遮挡导致 MediaPipe 几何错误**，不是侧别选错；远侧只用于诊断，不能作为产品回退信号。

## 2. 左右侧与相机深度审计

Fit3D `joints3d_25` 的评估链映射已经用 GT 投影和正面/背面视频核对：

```text
left  = J1/J2/J3 (hip/knee/ankle), J11/J12/J13 (shoulder/elbow/wrist)
right = J4/J5/J6 (hip/knee/ankle), J14/J15/J16 (shoulder/elbow/wrist)
```

近侧判定使用相机空间深度，z 越小越靠近相机：

```text
camera_points = joints @ R.T + t
t = -R @ camera_center
near_side = argmin(median camera z of the exercise arm/leg chain)
```

审计结果：

| 项目 | 结果 |
|---|---:|
| train clip | `64` |
| `camera_side` 与 GT 近侧深度不一致 | `0` |
| view 左右后缀与 GT 近侧深度不一致 | `0` |
| side/oblique clip | `37` |
| 选中近侧信号 | `37/37` |
| 选中侧与 GT 近侧不一致 | `0` |
| 远侧替代次数 | `0` |

T1-5A 的 32 条 side/oblique override 左右后缀已统一交换并升至 `task1-5a-view-overrides-v2`。当前侧别定义为“相机位于受试者左侧/右侧”，不是画面左边/右边。

产物：

```text
results/task1_5/manifest/side_audit/side_audit.csv
results/task1_5/manifest/side_audit/summary.json
```

## 3. 全量姿态质量与 rep

64 路 train 的质量分布：

| 指标 | 结果 |
|---|---:|
| analysis status | `ok=60`、`partial=3`、`rejected_quality=1` |
| quality status | `pass=37`、`warn=26`、`reject=1` |
| 完整 rep | `325` |
| 被拒候选 rep | `305` |
| pushup 完整 rep | `153` |
| squat 完整 rep | `172` |

未通过 `ok` 的 4 条：

| clip | 原因 |
|---|---|
| `fit3d_s07_pushup_65906101` | 只有 1 个完整 rep |
| `fit3d_s09_pushup_50591643` | pose/body-in-frame 约 `66%`，质量门控拒绝 |
| `fit3d_s10_pushup_58860488` | 0 个完整 rep |
| `fit3d_s11_pushup_58860488` | 0 个完整 rep |

侧视姿态质量本身较高：pushup 侧视 31 条中 `30/31` 达到 pose 检出通过线，近侧最小 visibility 跨 clip 中位数为 `0.964`，最低 `0.698`。因此后续高角度误差不能简单归因于“没检测到人”。

## 4. s09 pilot 修复验证

修正近侧选择后，`s09` 8 路结果：

| clip | view | status | quality | reps | pose 检出 | q10 visibility | 选中侧 |
|---|---|---|---|---:|---:|---:|---|
| `fit3d_s09_squat_50591643` | rear | ok | pass | 6 | 1.000 | 0.955 | - |
| `fit3d_s09_squat_58860488` | rear | ok | pass | 6 | 1.000 | 0.942 | - |
| `fit3d_s09_squat_60457274` | oblique_right | ok | warn | 5 | 1.000 | 0.908 | right |
| `fit3d_s09_squat_65906101` | front | ok | warn | 5 | 1.000 | 0.858 | - |
| `fit3d_s09_pushup_50591643` | side_left | rejected_quality | reject | 1 | 0.660 | 0.741 | left |
| `fit3d_s09_pushup_58860488` | side_right | ok | pass | 5 | 1.000 | 0.794 | right |
| `fit3d_s09_pushup_60457274` | side_right | ok | pass | 5 | 0.997 | 0.803 | right |
| `fit3d_s09_pushup_65906101` | side_left | ok | pass | 5 | 1.000 | 0.840 | left |

对比 T1-5A：完整 rep 从 `36/48` 提升到 `38/48`；`fit3d_s09_pushup_58860488` 从 `0 reps / partial` 修复为 `5 reps / pass`。`fit3d_s09_pushup_50591643` 仍被质量门控拒绝，没有把 1 个 rep 包装成完整结果。

## 5. 近侧角度误差

统计单位为 `subject-action` cluster；下表为 holdout 的相机近侧屈曲角：

| exercise | view | angle | clips | clusters | cluster MAE | 95% CI | bias | mean p90 | gate |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| pushup | side_left | left elbow | 8 | 4 | `15.19°` | `12.65–17.72°` | `+8.52°` | `37.16°` | disabled |
| pushup | side_right | right elbow | 7 | 4 | `12.50°` | `11.30–13.69°` | `+6.59°` | `28.32°` | low_confidence |
| pushup | oblique_right | right elbow | 1 | 1 | `34.57°` | 单样本 | `+33.31°` | `94.38°` | disabled |
| squat | oblique_left | left knee | 2 | 2 | `10.15°` | `8.97–11.32°` | `+4.76°` | `21.48°` | low_confidence |
| squat | oblique_right | right knee | 2 | 2 | `12.03°` | `10.52–13.54°` | `+4.81°` | `26.43°` | low_confidence |

解释：

- 俯卧撑左右纯侧视没有同时达到可用线，不能只报较好的 `side_right` 后启用指标。
- 即使 MAE 接近 `10–15°`，p90 仍达到 `21–37°`，说明逐帧反馈会出现明显的尾部错误。
- 深蹲的 front/rear 结果不是侧视验证；浅斜视只有 6 条 clip，且动作是杠铃背蹲，不能补齐徒手深蹲的纯侧视缺口。

## 6. 高误差来源定位

按相机拆分 pushup 近侧屈肘：

| camera | view 分布 | clips | near MAE | far MAE（仅诊断） | near visibility |
|---|---|---:|---:|---:|---:|
| `50591643` | side_left | 8 | `23.83°` | `13.56°` | `0.879` |
| `58860488` | side_right | 8 | `21.48°` | `10.86°` | `0.917` |
| `60457274` | side_right | 8 | `10.77°` | `14.84°` | `0.981` |
| `65906101` | side_left | 8 | `9.34°` | `15.98°` | `0.990` |

结论：

1. 侧别选择已由 GT 深度确认正确，高误差不是 `left/right` 选反。
2. `50591643`/`58860488` 上，MediaPipe 的远侧肘角反而更接近 GT；结合高 visibility 和画面重叠，判断为近侧肢体自遮挡后模型把部分近侧点估到了错误位置。
3. `60457274`/`65906101` 上近侧明显优于远侧，说明不能全局改成远侧或左右平均。
4. 仅用 visibility 中位数不足以识别这种遮挡；T1-6 若继续单目侧视，需要增加 2D 肢体重合、左右肘/膝互换检查或多视角一致性门控。

逐 clip 的近/远侧诊断在 `results/task1_5/evaluation/near_far_side_diagnostic.csv`，其中 `diagnostic_only=true`，不进入产品规则。

## 7. 验收结论

| 项目 | 结论 |
|---|---|
| 左右侧选择 | 通过：64/64 深度一致，37/37 选中近侧 |
| pose 质量 | 基本通过：60/64 ok，侧视检出和 body-in-frame 高 |
| pushup 侧视肘角 | **不通过**：side_left MAE `15.19°`，且部分相机 `>21°` |
| squat 侧视膝角 | **不能验收**：无纯侧视，只有 6 条浅斜视 |
| 侧视产品指标 | 继续 disabled |
| 远侧信号替代 | 禁止；仅保留诊断用途 |

因此 T1-5C 不应启用 `bottom_elbow_angle`、`elbow_flexion_rom`、`body_line_angle`、`hip_sag_ratio` 或深蹲侧视指标。T1-5D 可以继续做 rep 事件评估，但不能把本轮的侧视角度当作可用产品测量。

## 8. 限制

- Fit3D `squat` 是杠铃背蹲，不是 T1-4 的徒手深蹲。
- Fit3D 深蹲没有纯侧视，不能用 3D GT 替代视频侧视验收。
- `rep_ann` 仍只支持“接近伸展的 top/边界事件”，不能作为 bottom GT。
- s09 在 T1-5A/B pilot 调试中被使用，holdout 不再视为完全未参与开发的独立测试集。
- Fit3D 是棚拍、固定相机、深色服装和有限受试者数据，不能直接外推到手机手持、遮挡和健身房场景。

## 9. 产物

| 路径 | 内容 |
|---|---|
| `scripts/audit_task1_5b_sides.py` | GT 相机深度、manifest 侧别、MediaPipe 选中侧审计 |
| `scripts/task1_5_metrics.py` | 近侧深度判定、近侧信号选择、角度误差与 cluster bootstrap |
| `scripts/evaluate_task1_5_fit3d.py` | 64 路角度、质量、逐视角/逐相机与近远侧诊断 |
| `results/task1_5/manifest/side_audit/` | 左右侧审计 CSV 和汇总 |
| `results/task1_5/clip_analysis/` | 64 路完整逐 clip 分析 |
| `results/task1_5/evaluation/summary.json` | 正式评测汇总 |
| `results/task1_5/evaluation/pose_angles.csv` | 逐 clip × angle 误差 |
| `results/task1_5/evaluation/pose_quality.csv` | 逐 clip 姿态质量与选中信号 |
| `results/task1_5/evaluation/selected_signal_summary.csv` | 按视角的近侧主信号汇总 |
| `results/task1_5/evaluation/camera_selected_summary.csv` | 按相机的误差诊断 |
| `results/task1_5/evaluation/near_far_side_diagnostic.csv` | 近/远侧误差对照，只用于诊断 |

## 10. 复现

```bash
cd /mnt/storage2/data_guest

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/audit_task1_5b_sides.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --analysis-dir VLS/V0/results/task1_5/clip_analysis \
  --output-dir VLS/V0/results/task1_5/manifest/side_audit

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/evaluate_task1_5_fit3d.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --analysis-dir VLS/V0/results/task1_5/clip_analysis \
  --output-dir VLS/V0/results/task1_5/evaluation

VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test*.py'
```

全量测试：`85/85` 通过。
