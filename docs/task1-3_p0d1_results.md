# task1-3 P0-D1：landmark 约束的整段相位选择

日期：2026-10-25

## 1. 结论

P0-D1 已实现并完成 QEVD-FIT 内部验证。D1 只在 Stage A 的起始相位选择上使用动作级 landmark 相位约定；Stage B 参数未改，`active bouts`、活动度阈值和 rep 数验收口径保持不变。

D1 结果：

- `0161/lunge_jumps`：绝对 phase shift `19 -> 1`，达到 `<=5` 目标。
- `0218/walking_lunges`：绝对 phase shift `8 -> 5`，达到 `<=5` 目标。
- `0173/boxing_squat_punches`：相位证据覆盖率只有 `0.727 < 0.75`，按设计回退；shift 保持 `26`，**未达到 `<=8` 目标**。
- `0161/pushup`：极值数不足，回退；start 抖动仍属于 P0-D2。
- `0218/air_jump_rope`：同相位支持率 `0.571 < 0.65`，回退；shift 保持 `0`。
- 22 段 AIFit rep 数全部不变，仍为 `366`；Peak 基线仍为 `341`。
- 5 段盲标 IoU=0.30 F1 保持 `0.977`；raw micro IoU `0.585 -> 0.629`，IoU=0.50 F1 `0.677 -> 0.722`。

本轮没有把逐段 GT 帧号写入配置，也没有在 boxing 上继续调相位偏移来强行满足验收值。

## 2. 实现

动作级配置：`VLS/V0/configs/phase_guidance.json`

配置只描述相位语义和信号来源，不包含 video/segment 帧号：

| action | phase landmark | 实际信号 | 极性 | D1 决策 |
|---|---|---|---|---|
| `pushup` | `arms_extended_top` | `left_elbow_flexion` | peak | fallback: `too_few_extrema` |
| `boxing_squat_punches` | `standing_knee_extended` | `left_knee_flexion` | peak | fallback: `insufficient_extrema_coverage` |
| `lunge_jumps` | `left_foot_contact` | `left_foot_index.y * -1` | trough | applied: shift `15` 帧 |
| `walking_lunges` | `right_leg_forward_lowest` | `right_ankle_dorsiflexion_sagittal` | peak | applied: shift `19` 帧 |
| `air_jump_rope` | `jump_apex_head_highest` | `nose.y * -1` | peak | fallback: `insufficient_phase_support` |

`left_foot_index.y * -1` 把图像/世界坐标中向下为正的 y 转成“足部抬高”，因此落地相位是 trough；这与盲标元数据一致。`nose.y * -1` 同理，跳跃最高点是 peak。

实现要点：

- 新增 `PhaseGuidance` / `PhaseSignalSpec`，支持复用 `angles_wide.csv` 角度或读取 `pose_landmarker_full/*_landmarks.json` 世界坐标。
- `stage_a()` 在原相关分数候选之外，计算相位信号同极性极值在周期内的分布；选择支持同一相位的起始偏移。
- 只有当极值数、覆盖率、支持率满足门槛，且不存在接近的第二相位簇时才吸附，否则保留原 Stage A。
- 若相位吸附会改变边界数量，回退原 Stage A，保证 D1 不改变 rep 数。
- Stage B 参数不变。额外加入 D1 安全门：phase-guided Stage B 若减少 rep 数则回退到 phase-guided Stage A；若末尾低幅值区间起点落在相位极值容差内，可保留为 phase-aligned edge rep。
- `0162/walking_lunges` 因相位候选会改变边界数量而回退，rep 数保持 `6`。

CLI 新增：

```bash
--phase-config VLS/V0/configs/phase_guidance.json
--landmarks-dir VLS/V0/outputs/pose_landmarker_full
--no-phase-guidance   # 复现 P0-C 基线
```

## 3. 盲标结果

IoU >= 0.30 的 AIFit 分段结果：

| segment | GT/Pred | phase shift | signed start/end | abs start/end | raw IoU | p95 |
|---|---:|---:|---:|---:|---:|---:|
| `0161/lunge_jumps` | 6/6 | **1** | -1.83 / -1.83 | 2.17 / 2.17 | **0.918** | 4.0 |
| `0161/pushup` | 5/5 | 4 | +24.00 / +3.00 | 28.40 / 9.00 | 0.578 | 55.4 |
| `0173/boxing_squat_punches` | 10/10 | -26 | -10.56 / -11.89 | 16.11 / 16.33 | 0.532 | 21.6 |
| `0218/air_jump_rope` | 39/40 | 0 | -2.33 / -2.44 | 4.90 / 4.79 | 0.571 | 9.3 |
| `0218/walking_lunges` | 6/6 | **5** | -3.83 / -0.33 | 4.17 / 6.00 | **0.904** | 10.5 |

总体指标：

| metric | P0-C | P0-D1 |
|---|---:|---:|
| IoU=0.30 Pred/TP/FP/FN | 67/65/2/1 | 67/65/2/1 |
| IoU=0.30 micro F1 | 0.977 | **0.977** |
| raw micro IoU | 0.585 | **0.629** |
| mean abs start error | 9.86 | **7.94** |
| mean abs end error | 8.20 | **6.58** |
| p95 boundary error | 20.10 | **19.65** |
| IoU=0.50 micro F1 | 0.677 | **0.722** |
| split / merge events | 4 / 4 | 4 / 4 |

## 4. 相位证据

### `0161/lunge_jumps`

- 5 个左足落地极值，覆盖率 `0.833`，支持率 `0.800`。
- 周期内目标相位 `11`，边界起点 `1868`，GT 首边界 `1869`。
- 末尾区间 `2129-2169` 低幅值，但起点与相位极值 `2131` 在容差内，保留为 edge rep，避免 rep 数从 6 降到 5。

### `0218/walking_lunges`

- 右踝背屈峰 8 个，覆盖率 `1.333`，支持率 `0.750`。
- 周期内目标相位 `52`，Stage A 起点 `2454`；Stage B 后首个 rep 起点 `2442`。
- 首边界误差从 P0-C 的约 `-10` 帧降到约 `-4` 帧，shift 达到 `5`。

### 回退项

- `pushup`：两个 bout 分别只有 1、2 个极值，低于最少 3 个的要求。
- `boxing_squat_punches`：8 个极值对应约 11 个基线周期，覆盖率 `0.727`；强行吸附会降低 raw IoU 并抬高 p95，因此回退。
- `air_jump_rope`：极值很多但相位分散，最高支持率仅 `0.571`；短周期下属于 `phase_ambiguity` 类似的低置信情形，回退。

## 5. 验证

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py
# 10 passed

.venv/bin/python VLS/V0/scripts/check_p0d1_regression.py \
  --use-existing \
  --segmentation-dir VLS/V0/results/segmentation_p0d1 \
  --report VLS/V0/results/segmentation_p0d1/p0d1_regression.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode guard \
  --segmentation-dir VLS/V0/results/segmentation_p0d1 \
  --use-existing \
  --report VLS/V0/results/segmentation_p0d1/p0c_guard.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode exact \
  --report VLS/V0/results/segmentation_p0d1/p0c_exact_after_d1.json
# pass: --no-phase-guidance reproduces frozen P0-C
```

synthetic 新增覆盖：

- 相位信号极值与主信号相关最优相位不同时，D1 选择 landmark 相位。
- 极值不足时回退原 Stage A。
- 相位候选会改变边界数量时回退，并保持边界数不变。

## 6. 产物

```text
VLS/V0/configs/phase_guidance.json
VLS/V0/configs/p0d1_golden.json
VLS/V0/scripts/segment_reps.py
VLS/V0/scripts/check_p0d1_regression.py
VLS/V0/tests/test_segment_reps.py
VLS/V0/results/segmentation_p0d1/
├── run.json
├── reps.csv
├── action_segments.csv
├── p0d1_regression.json
├── p0c_guard.json
├── p0c_exact_after_d1.json
├── annotation_blind_eval/
└── plots/
```

## 7. 剩余限制与 D2 输入

- `boxing_squat_punches` 的整段相位目标未达成。固定周期 Stage A 与当前膝伸展相位信号无法在不损失计数/IoU 的情况下把 shift 压到 8 帧以内；不继续用该段 GT 调偏移。
- `pushup` 的起点抖动和 boxing 的 bout 内边界抖动仍是 P0-D2 范围。
- `walking_lunges` 和 `lunge_jumps` 的改善依赖 phase-guided edge-rep 安全门，D2/P1 改动后必须重新跑 `check_p0d1_regression.py`。
- 仍是 QEVD-FIT 内部盲标，不是 Fit3D 官方指标；独立第二标注者仍为 `pending`。
- D1 的相位配置来自动作级语义和已有盲标元数据，参数只在 5 段样本上做过非回归检查，未做视频级留出。
