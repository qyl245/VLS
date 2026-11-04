# task1-3 P1-B1：rep_unit 与边界相位元数据

日期：2026-10-25

## 1. 结论

P1-B1 已完成。22 个动作段都补齐了 `rep_unit`、`phase_landmark` 和 `phase_polarity`，这些字段为动作级/段级元数据，不包含 GT 帧号，也不参与分割决策。

- `rep_unit` 分布：`cycle` 15 段、`side` 5 段、`stride` 2 段。
- `0162/walking_lunges` 已补原始帧盲标：12 次交替迈步，按 `stride` 计为 **6 个完整 rep**。
- `0162` 的 AIFit 为 6 reps，与 GT **6/6** 匹配，IoU=0.30 F1 `1.0`；Peak 为 10 reps，6 TP + 4 FP，F1 `0.75`。因此“6 vs 10”中 **AIFit 的 stride 计数正确**，Peak 把单步事件混入了计数。
- `0218/walking_lunges` 仍是 6/6、F1 `1.0`。两段合并按 `stride` 计：AIFit `12/12`、F1 `1.0`；Peak `16` 预测、10 TP、F1 `0.714`。
- 没有证据表明单侧角度把半周期误计为完整 rep，也没有证据支持漏掉另一侧周期。**P1-B2 左右相位融合不启动。**

本轮没有改变任何 rep 边界、rep 数或既有盲标 GT。全量 AIFit/Peak 仍为 `366/341`，相对 P1-D 的边界差异为 `0`。

## 2. 0162 盲标依据

### 2.1 原始帧复核

只使用原始视频抽帧，不显示 AIFit/Peak 红线。证据包括：

- 10 FPS 总览：`VLS/V0/results/segmentation_p1b/annotation/frames_10fps/`。
- 30 FPS 边界 focus：`VLS/V0/results/segmentation_p1b/annotation/frames/focus/`。
- 左右髋/膝与踝部相位交叉核对，但不把姿态信号当作 GT。

动作从约 `f1624` 开始准备，第一组左腿前迈出现在 `f1650` 左右；之后左右交替，最后一组右腿前迈在 `f2230` 左右，`f2267` 后进入恢复。

### 2.2 GT 口径

| rep | GT 区间 | 包含步态 |
|---|---|---|
| 1 | `1675-1785` | 左步 + 右步 |
| 2 | `1786-1880` | 左步 + 右步 |
| 3 | `1881-1976` | 左步 + 右步 |
| 4 | `1977-2070` | 左步 + 右步 |
| 5 | `2071-2176` | 左步 + 右步 |
| 6 | `2177-2267` | 左步 + 右步 |

`0162` 的动作从**左腿**开始、以右腿结束，因此完整 stride 的边界落在 `left_leg_forward_lowest`。`0218` 从右腿开始、以左腿结束，边界落在 `right_leg_forward_lowest`。两段的 `rep_unit` 都是 `stride`，但起始脚不同；后续做相位对齐或左右融合时不能假设两段使用同一侧边界。

## 3. 评估结果

主阈值 `IoU=0.30`，按 `rep_unit=stride` 分组：

| segment | method | GT | Pred | TP | FP | FN | F1 | raw IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0162/walking_lunges | AIFit | 6 | 6 | 6 | 0 | 0 | 1.000 | 0.842 |
| 0162/walking_lunges | Peak | 6 | 10 | 6 | 4 | 0 | 0.750 | 0.720 |
| 0218/walking_lunges | AIFit | 6 | 6 | 6 | 0 | 0 | 1.000 | 0.904 |
| 0218/walking_lunges | Peak | 6 | 6 | 4 | 2 | 2 | 0.667 | 0.465 |

合并两段：

| method | GT | Pred | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| AIFit | 12 | 12 | 12 | 0 | 0 | 1.000 |
| Peak | 12 | 16 | 10 | 2 | 2 | 0.714 |

`0162` AIFit 的边界平均早于 GT：start `-9.50` 帧、end `-7.67` 帧，p95 边界误差 `15` 帧。这个偏移是稳定的相位差，不是计数错误；本轮不为了缩小该偏移改算法。`0218` 的 start/end 平均误差分别为 `-3.83/-0.33` 帧。

Peak 在 `0162` 的 10 个 rep 中有 4 个是典型的单步事件：它们与相邻完整 stride 有交集，但边界落在单个左/右步上，因此在 `stride` 口径下计为 duplicate/FP。

## 4. 22 段元数据

配置文件为 `VLS/V0/configs/rep_metadata.json`。`confirmed` 表示已有盲标 GT；`provisional` 表示按动作语义先定义、尚无盲标；`excluded` 表示段本身不进入定量评估。

| segment | rep_unit | phase_landmark | polarity | status | AIFit | Peak |
|---|---|---|---|---|---:|---:|
| 0153/jumping_jack | cycle | arms_down_lowest | trough | provisional | 30 | 27 |
| 0153/high_knees | side | alternating_knee_drive_top | peak | provisional | 32 | 27 |
| 0153/squat | cycle | standing_knee_extended | peak | provisional | 8 | 9 |
| 0153/pushup | cycle | arms_extended_top | peak | provisional | 3 | 3 |
| 0161/jumping_jack | cycle | arms_down_lowest | trough | provisional | 27 | 25 |
| 0161/butt_kickers | side | heel_to_glute_closest | trough | provisional | 17 | 17 |
| 0161/lunge_jumps | cycle | left_foot_contact | trough | confirmed | 6 | 6 |
| 0161/pushup | cycle | arms_extended_top | peak | confirmed | 5 | 5 |
| 0162/butt_kickers | side | heel_to_glute_closest | trough | provisional | 33 | 33 |
| 0162/jumping_jack | cycle | arms_down_lowest | trough | provisional | 27 | 26 |
| 0162/walking_lunges | stride | left_leg_forward_lowest | peak | confirmed | 6 | 10 |
| 0162/mountain_climbers | side | alternating_knee_drive | alternating | excluded | 12 | 2 |
| 0173/good_morning | cycle | standing_hip_extended | trough | provisional | 9 | 9 |
| 0173/air_jump_rope | cycle | jump_apex_head_highest | peak | provisional | 24 | 27 |
| 0173/standing_kicks | side | kick_extension_peak | peak | provisional | 13 | 13 |
| 0173/boxing_squat_punches | cycle | standing_knee_extended | peak | confirmed | 10 | 10 |
| 0218/good_morning | cycle | standing_hip_extended | trough | provisional | 16 | 16 |
| 0218/air_jump_rope | cycle | jump_apex_head_highest | peak | confirmed | 40 | 28 |
| 0218/boxing_squat_punches | cycle | standing_knee_extended | peak | provisional | 17 | 17 |
| 0218/walking_lunges | stride | right_leg_forward_lowest | peak | confirmed | 6 | 6 |
| 0218/toe_touchers | cycle | standing_torso_upright | trough | provisional | 0 | 0 |
| 0218/shoulder_gators | cycle | elbows_forward_peak | peak | provisional | 25 | 25 |

`side` 表示每个单侧动作计一个 rep（左右交替时每侧各算一次）；`stride` 表示左右各一步合成一个 rep；`cycle` 表示一次完整对称/全身动作周期。`phase_polarity` 描述边界在该段相位信号上的极值方向；`alternating` 只用于 `0162/mountain_climbers` 的暂定口径。

## 5. 输出接口

`segment_reps.py` 新增 `--rep-metadata` / `--no-rep-metadata`，默认读取 `configs/rep_metadata.json`。向后兼容扩展如下：

- `action_segments.csv`：追加 `rep_unit,phase_landmark,phase_polarity`。
- `summary.csv`：每个 method 行追加上述三列。
- `reps.csv` / `rejected_reps.csv`：每个 rep 行追加上述三列。
- `run.json`：顶层写入 `rep_metadata_config`，`parameters.rep_metadata=true`，每个 segment 写入完整 `rep_metadata`。

若所选动作段缺少元数据，脚本直接报错，避免生成“部分段有口径、部分段无口径”的评估文件。

## 6. P1-B2 决策

不启动 P1-B2。理由：

1. 两段 walking lunges 的 AIFit 在 `stride` 口径下都是 6/6，合并 12/12。
2. `0162` 的 GT 是 12 次交替单步，按 `stride` 应为 6 个完整周期；AIFit 没有把半周期算成一个 rep。
3. Peak 的 10 个事件才是单步级过分割，说明问题在 peak 口径，而不是 AIFit 漏掉了另一侧。
4. 本轮没有新增支持“单侧角度导致半周期错误”的盲标证据；按计划不引入左右膝/髋融合。

后续只有出现新的盲标段明确显示单侧信号漏计或半周期误计，才重新评估 P1-B2。

## 7. 验证

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py
# 19 passed

.venv/bin/python VLS/V0/scripts/evaluate_p1b1_rep_units.py
# 0162 AIFit 6/6 F1 1.0; Peak 6 TP + 4 FP F1 0.75

.venv/bin/python VLS/V0/scripts/check_p1b1_regression.py \
  --report VLS/V0/results/segmentation_p1b/p1b1_regression_rerun.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p1d_regression.py \
  --report VLS/V0/results/segmentation_p1b/p1d_regression.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode guard \
  --segmentation-dir VLS/V0/results/segmentation_p1b --use-existing \
  --skip-synthetic --report VLS/V0/results/segmentation_p1b/p0c_guard.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode exact \
  --skip-synthetic --report VLS/V0/results/segmentation_p1b/p0c_exact_after_p1b1.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0d1_regression.py --skip-synthetic \
  --report VLS/V0/results/segmentation_p1b/p0d1_regression.json
# pass, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0d2_regression.py --use-existing \
  --skip-synthetic --segmentation-dir VLS/V0/results/segmentation_p1b \
  --report VLS/V0/results/segmentation_p1b/p0d2_regression.json
# pass, difference_count=0
```

回归快照：

- 22 段均有元数据；无空字段。
- AIFit 366 reps、Peak 341 reps。
- 相对 P1-D 的 rep 边界差异 `0`。
- 5 段冻结盲标指标不变：IoU=0.30 `67 Pred / 66 TP / 1 FP / 0 FN`。
- `0162/walking_lunges` 已加入 P1-B1 独立盲标目录，没有修改 P0-A 的既有 GT 文件。

## 8. 限制

- `0162` 仍是同一标注者（`codex_blind_frame_review`）的原始帧复核；独立第二标注者仍为 `pending`，不能报告 inter-annotator agreement。
- 除已有盲标段外，其余动作的 `side/cycle` 和边界相位是按动作语义定义的 provisional 口径，尚无 GT 支持。它们可用于后续分组和标注准备，不能当作已验证指标。
- 本轮是 QEVD-FIT 内部 GT，不是 Fit3D 官方指标。
