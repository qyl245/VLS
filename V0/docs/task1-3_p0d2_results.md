# task1-3 P0-D2：平台进入点约束的残差边界修正

日期：2026-10-25

## 1. 结论

P0-D2 已实现并通过冻结回归。D2 在 Stage B 之后，对 `pushup` 和 `boxing_squat_punches` 的 bout 内边界吸附到重复出现的相位高平台进入点；D1 的整段相位选择和 P0-C 的 active-bout 逻辑保持不变。

本轮验收为 **partial**：

- `0173/boxing_squat_punches` 达标：start/end error SD `12.73 / 12.18 -> 3.42 / 4.77`，均 `<8`；absolute phase shift `26 -> 3`。
- `0161/pushup` 的 start SD `27.22 -> 16.33`，明显改善但未达到 `<10`；两个 bout 的首边界都落在 fit-window 截断的高平台上，当前不可见真实进入点，D2 选择保守保留。
- 5 段盲标 rep 数不变；IoU=0.30 `TP/FP/FN 65/2/1 -> 66/1/0`，F1 `0.977 -> 0.992`。
- raw micro IoU `0.629 -> 0.685`；IoU=0.50 F1 `0.722 -> 0.827`；p95 boundary error `19.65 -> 12.00` 帧。
- `lunge_jumps`、`walking_lunges` 和 `air_jump_rope` 的 D1 结果未退化；air jump rope 不启用 D2。

## 2. 实现

### 2.1 配置

`VLS/V0/configs/phase_guidance.json` 新增动作级 `boundary_snap`，只对两个残差抖动动作启用：

```json
{
  "mode": "plateau_entry",
  "plateau_fraction": 0.8,
  "min_plateau_fraction": 0.08,
  "snap_radius_fraction": 0.85,
  "min_width_fraction": 0.25
}
```

配置不包含 video/segment 帧号。阈值由相位信号的 5/95 分位动态确定；吸附半径、最小平台长度和最小 rep 宽度均按局部周期比例计算。

### 2.2 边界吸附

新增 `snap_boundaries_to_phase()`：

- 对相位信号复用 D1 的插值、Savitzky-Golay 平滑、去趋势和标准化。
- 取同极性高平台，候选为持续至少 `min_plateau_fraction * period` 帧的连续区间起点。
- 按顺序一对一匹配候选与 Stage B 边界；仅接受半径内的候选。
- 严格检查边界递增和 `min_width_fraction * period`，失败则整组回退。
- 若末尾区间已经呈现 incomplete-tail 特征，保留最后两个 Stage B 边界，避免把尾部残段误扩成完整 rep。
- 吸附后重新构建 rep；若 rep 数与 Stage B 不同，整段回退并记录 `snap_changes_rep_count`。

每个 bout 在 `run.json` 中输出 `phase_guidance.boundary_snap`，包含：

- `status` / `reason`
- `low_phase_confidence`
- `leading_plateau_truncated`
- `candidates` / `plateau_runs`
- `boundaries_before` / `boundaries_after`
- 每个边界的 `original`、`candidate`、`shift`、`accepted`、`reason`
- `trailing_incomplete_protected` 和 `changed_boundaries`

### 2.3 CLI

```bash
--no-phase-snap   # 保持 D1，仅关闭 D2 局部吸附
```

`check_p0d1_regression.py` 的重跑命令已显式加入 `--no-phase-snap`；P0-C `exact` 模式同时关闭 D1/D2。

## 3. 盲标结果

### 3.1 分段指标

IoU >= 0.30 的 AIFit 结果：

| segment | GT/Pred | phase shift | signed start/end | abs start/end | raw IoU | p95 |
|---|---:|---:|---:|---:|---:|---:|
| `0161/lunge_jumps` | 6/6 | 1 | -1.83 / -1.83 | 2.17 / 2.17 | 0.918 | 4.00 |
| `0161/pushup` | 5/5 | 8 | +5.60 / -8.60 | 14.80 / 8.60 | **0.679** | 25.55 |
| `0173/boxing_squat_punches` | 10/10 | **3** | -3.90 / -1.20 | **3.90 / 3.80** | **0.862** | 13.05 |
| `0218/air_jump_rope` | 39/40 | 0 | -2.33 / -2.44 | 4.90 / 4.79 | 0.571 | 9.30 |
| `0218/walking_lunges` | 6/6 | 5 | -3.83 / -0.33 | 4.17 / 6.00 | 0.904 | 10.50 |

目标边界 SD：

| segment | D1 start/end SD | D2 start/end SD | 目标 | 状态 |
|---|---:|---:|---:|---|
| `0161/pushup` | 27.22 / 10.53 | 16.33 / 4.27 | start `<10` | **partial** |
| `0173/boxing_squat_punches` | 12.73 / 12.18 | 3.42 / 4.77 | start/end `<8` | **pass** |

### 3.2 总体指标

| metric | P0-C | P0-D1 | P0-D2 |
|---|---:|---:|---:|
| IoU=0.30 Pred/TP/FP/FN | 67/65/2/1 | 67/65/2/1 | **67/66/1/0** |
| IoU=0.30 micro F1 | 0.977 | 0.977 | **0.992** |
| raw micro IoU | 0.585 | 0.629 | **0.685** |
| mean abs start error | 9.86 | 7.94 | **5.18** |
| mean abs end error | 8.20 | 6.58 | **4.80** |
| p95 boundary error | 20.10 | 19.65 | **12.00** |
| IoU=0.50 micro F1 | 0.677 | 0.722 | **0.827** |
| IoU=0.50 raw IoU | 0.658 | 0.706 | **0.741** |
| split / merge events | 4 / 4 | 4 / 4 | **3 / 3** |

全量仍为 22 段、AIFit `366` rep、Peak `341` rep。`0218/toe_touchers` 继续保持 0 rep，未因 D2 改变。

## 4. pushup 未完全达标的原因

两个 active bout 的 fit window 都以高肘伸展平台开始：

- bout 1：fit `2465-2626`，平台从 frame `2465` 起已高于阈值，真实区间入口在窗口之前。
- bout 2：fit `2627-2810`，同样从窗口起点进入高平台。

D2 只能把首边界吸到当前可见的平台起点，不能凭空恢复窗口外的真实边界；若继续向前扩展 fit window，Stage A/B 会重新生成边界并引入恢复段 rep，违反“不得重新破坏 D1/C 相位选择”和 rep 数非退化约束。因此本轮选择保留 `from_fit_start` 边界，并记录为低置信限制，而不是用逐段帧号硬编码修正。

pushup 的其余四个边界已明显改善：end SD `10.53 -> 4.27`，raw IoU `0.578 -> 0.679`；剩余误差集中在两个首边界。

## 5. 验证

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py
# 14 passed

.venv/bin/python VLS/V0/scripts/check_p0d2_regression.py \
  --use-existing \
  --report VLS/V0/results/segmentation_p0d2/p0d2_regression.json
# status=pass, acceptance_status=partial, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0d1_regression.py \
  --report VLS/V0/results/segmentation_p0d2/p0d1_regression.json
# pass; --no-phase-snap exactly reproduces D1

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode guard \
  --segmentation-dir VLS/V0/results/segmentation_p0d2 \
  --use-existing \
  --report VLS/V0/results/segmentation_p0d2/p0c_guard.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode exact \
  --report VLS/V0/results/segmentation_p0d2/p0c_exact_after_d2.json
# pass; reproduces P0-C with D1/D2 disabled
```

synthetic 新增覆盖：

- 平台候选可将偏移边界吸附到周期平台进入点。
- 相位信号不可用时回退原边界。
- 吸附导致 rep 宽度非法时整组回退。
- 吸附前后 rep 数保持不变。

## 6. 产物与限制

```text
VLS/V0/configs/phase_guidance.json
VLS/V0/configs/p0d2_golden.json
VLS/V0/scripts/segment_reps.py
VLS/V0/scripts/check_p0d2_regression.py
VLS/V0/scripts/check_p0d1_regression.py
VLS/V0/scripts/check_p0c_regression.py
VLS/V0/tests/test_segment_reps.py
VLS/V0/results/segmentation_p0d2/
├── run.json
├── reps.csv
├── annotation_blind_eval/
├── p0d2_regression.json
├── p0d1_regression.json
├── p0c_guard.json
├── p0c_exact_after_d2.json
└── plots/
```

- 当前仍是 QEVD-FIT 内部盲标，不是 Fit3D 官方指标。
- 独立第二标注者仍为 `pending`；现有二读只是同一标注者复核，不能报告 inter-annotator agreement。
- 参数来自 5 段内部盲标的动作级非回归检查，未做视频级留出；后续不得继续用同一小样本反复调整 D2 阈值后宣称泛化。
- pushup 首边界若要达到 `<10`，需要单独设计 fit-window 外相位恢复，并重新证明不会生成恢复段 rep；该工作不在当前 D2 的安全改动范围内。
