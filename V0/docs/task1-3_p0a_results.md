# task1-3 P0-A：高风险段盲标结果

日期：2026-10-25

## 1. 结论

P0-A 已完成高风险动作段的原始抽帧盲标主体，并将第一轮 GT 与新盲标 GT 分开保存。新 GT 不读取 AIFit 红线、峰值蓝线或草稿 rep 边界。

- 盲标动作段：6 个。
- 入评估动作段：5 个，共 66 个 rep。
- 排除动作段：`0162/mountain_climbers`，8 个暂定 side rep 不进入定量评估。
- 原始抽帧：36 页 15 FPS 总览，加 23 页 ±12 帧边界精修。
- 结构校验：`blind_validation.csv` 无 issue。
- 独立第二标注者：**尚未完成**。15/66（22.7%）只做了同一标注者的原始帧重读，不能作为 inter-annotator agreement。

## 2. 盲标集

新 GT 不覆盖也不修改第一轮文件 `annotation/annotation_template.csv`。

| 文件 | 内容 |
|---|---|
| `annotation_blind/blind_reps.csv` | 74 行逐 rep GT；66 行入评估，8 行登山跑暂定并排除 |
| `annotation_blind/blind_segments.csv` | 动作窗口、active 区间、`rep_unit`、边界相位和极性 |
| `annotation_blind/blind_exclusions.csv` | 13 个 transition/rest/incomplete/unclear 区间 |
| `annotation_blind/second_pass_sample.csv` | 22.7% 重读样本及明确标注者身份 |
| `annotation_blind/frames/manifest.json` | 36 页总览和 23 页边界精修的完整帧清单 |

动作与计数：

| video/action | GT rep | `rep_unit` | 边界相位 | 状态 |
|---|---:|---|---|---|
| `0161/pushup` | 5 | `cycle` | `arms_extended_top` / `peak` | 入评估 |
| `0173/boxing_squat_punches` | 10 | `cycle` | `standing_knee_extended` / `peak` | 入评估 |
| `0161/lunge_jumps` | 6 | `cycle` | `left_foot_contact` / `trough` | 入评估 |
| `0218/walking_lunges` | 6 | `stride` | `right_leg_forward_lowest` / `peak` | 入评估 |
| `0218/air_jump_rope` | 39 | `cycle` | `jump_apex_head_highest` / `peak` | 入评估 |
| `0162/mountain_climbers` | 8 暂定 | `side` | `alternating_knee_drive` / `alternating` | 排除 |

## 3. 边界复核

### `0161/pushup`

边界为 `2440-2530-2600-2700-2740-2780`。姿态信号中只有 5 个肘屈谷：`2513`、`2568`、`2683`、`2717`、`2760`。`f2440` 是撑起位平台起点，不是站立准备位；`f2600` 是两次动作之间的撑起位停顿平台。AIFit 的 `2571-2611` 和 `2764-2809` 是相对新 GT 的两个 duplicate FP。

### `0173/boxing_squat_punches`

边界从 `2472` 的站立防守位到 `2988` 的下一次站立防守位，共 10 个完整 squat-punch 周期。`3018-3041` 只下降、未回到站立位，记为 `incomplete_rep`。

### `0161/lunge_jumps`

边界统一取每两次左脚踏地之间的完整左右周期，共 6 个 `cycle`。没有复用 AIFit 草稿边界。

### `0218/walking_lunges`

使用 `stride`：左右各一步合为一个 rep，共 6 个。边界取右腿前迈最低相位，避免把半周期误算成完整 rep。

### `0218/air_jump_rope`

保留 39 个完整跳跃周期，最后一个边界为 `f1662`。`f1663-1684` 已进入恢复，不构成第 40 个 rep。AIFit 的第 40 个 rep `1663-1679` 因此归类为 `transition`；峰值基线只得到 28 个，主要问题是漏检。

### `0162/mountain_climbers`

原始抽帧只能给出 8 个暂定左右交替相位，左右归属和姿态质量不稳定。该段整段标记 `exclude`，不以 8 作为可靠 GT，也不拿 AIFit 的 12 或峰值的 2 判断算法优劣。

## 4. 二标/重读

`second_pass_sample.csv` 抽取 15 个 rep，覆盖 66 个入评估 rep 的 22.7%。15 个样本均已完成同一标注者的原始帧重读，rep 数和边界差异均为 0。

这组结果只能说明同一口径下的重复性，**不是独立二标**：

- `second_pass_annotator = codex_same_annotator_recheck`
- `independent_annotation_claimed = false`
- `second_pass_status = partial`，表示只覆盖抽样，不表示样本未完成

正式验收仍建议增加另一名标注者，或至少隔天由未查看本轮结果的标注者重标同一批样本。

## 5. 评估结果

匹配阈值为 IoU ≥ 0.30；边界误差只在匹配 rep 上计算，并按匹配数加权。

| 方法 | GT | Pred | TP | FP | FN | Precision | Recall | F1 | raw IoU | phase IoU | 边界误差 s | p95 帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | 66 | 70 | 65 | 5 | 1 | 0.929 | 0.985 | 0.956 | 0.573 | 0.666 | 0.311 | 19 |
| Peak | 66 | 55 | 45 | 10 | 21 | 0.818 | 0.682 | 0.744 | 0.532 | 0.615 | 0.454 | 39 |

不同 IoU 阈值下的 summary：

| 方法 | IoU | TP | FP | FN | F1 | raw IoU | phase IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIFit | 0.10 | 65 | 5 | 1 | 0.956 | 0.573 | 0.688 |
| AIFit | 0.30 | 65 | 5 | 1 | 0.956 | 0.573 | 0.666 |
| AIFit | 0.50 | 42 | 28 | 24 | 0.618 | 0.653 | 0.794 |
| Peak | 0.10 | 49 | 6 | 17 | 0.810 | 0.507 | 0.587 |
| Peak | 0.30 | 45 | 10 | 21 | 0.744 | 0.532 | 0.615 |
| Peak | 0.50 | 28 | 27 | 38 | 0.463 | 0.603 | 0.650 |

严格阈值下的 mean IoU 只统计仍然匹配的 rep，不能把阈值 0.50 的较高 IoU 解释为边界整体更好；主结论以 0.30 的匹配集合为准。

分动作段主结果（IoU ≥ 0.30）：

| segment | 方法 | GT/Pred | TP/FP/FN | raw IoU | phase IoU | 边界误差帧 |
|---|---|---:|---:|---:|---:|---:|
| `0161/lunge_jumps` | AIFit | 6/6 | 6/0/0 | 0.509 | 0.925 | 16.92 |
| `0161/pushup` | AIFit | 5/7 | 5/2/0 | 0.422 | 0.699 | 22.40 |
| `0173/boxing_squat_punches` | AIFit | 10/11 | 9/2/1 | 0.532 | 0.703 | 16.22 |
| `0218/air_jump_rope` | AIFit | 39/40 | 39/1/0 | 0.571 | 0.571 | 4.85 |
| `0218/walking_lunges` | AIFit | 6/6 | 6/0/0 | 0.832 | 0.943 | 9.50 |
| `0161/lunge_jumps` | Peak | 6/6 | 4/2/2 | 0.544 | 0.627 | 14.75 |
| `0161/pushup` | Peak | 5/5 | 4/1/1 | 0.625 | 0.685 | 17.00 |
| `0173/boxing_squat_punches` | Peak | 10/10 | 8/2/2 | 0.526 | 0.710 | 17.13 |
| `0218/air_jump_rope` | Peak | 39/28 | 25/3/14 | 0.528 | 0.554 | 7.32 |
| `0218/walking_lunges` | Peak | 6/6 | 4/2/2 | 0.465 | 0.723 | 41.38 |

FP 类型（IoU ≥ 0.30）：

| 方法 | duplicate | transition | incomplete_rep | rest | unexplained |
|---|---:|---:|---:|---:|---:|
| AIFit | 3 | 1 | 1 | 0 | 0 |
| Peak | 9 | 1 | 0 | 0 | 0 |

主要失败模式：

- `0161/pushup`：AIFit 7 对 GT 5，多出 `2571-2611` 和 `2764-2809` 两个 duplicate。
- `0173/boxing_squat_punches`：AIFit 11 对 GT 10，末尾未完成深蹲被识别为 `incomplete_rep`。
- `0218/air_jump_rope`：AIFit 40 对 GT 39，末尾恢复被识别为 `transition`；峰值 28 明显漏检。

## 6. 与第一轮 GT 的差异

第一轮 GT 保留在 `annotation/annotation_template.csv`，新 GT 保存在 `annotation_blind/`。两者不是同一份真值。

| segment | 第一轮 GT | 盲标 GT | 平均起点差 | raw IoU |
|---|---:|---:|---:|---:|
| `0161/lunge_jumps` | 6 | 6 | +18.2 帧 | 0.509 |
| `0161/pushup` | 5 | 5 | -28.6 帧 | 0.445 |
| `0173/boxing_squat_punches` | 10 | 10 | -16.3 帧 | 0.498 |
| `0218/walking_lunges` | 6 | 6 | +9.0 帧 | 0.832 |

这说明第一轮的整段相位约定与盲标口径仍有明显差异，不能把第一轮 AIFit IoU 与本轮 AIFit IoU 直接混用。

## 7. 产物

```text
VLS/V0/results/segmentation/annotation_blind/
├── blind_reps.csv
├── blind_segments.csv
├── blind_exclusions.csv
├── focus.csv
├── second_pass_sample.csv
├── frames/manifest.json
├── blind_validation.csv
├── blind_eval.csv
├── blind_eval_summary.csv
├── blind_eval_by_rep_unit.csv
├── blind_eval.json
├── blind_false_positives.csv
├── blind_segmentation_errors.csv
├── blind_vs_first_round.csv
├── blind_second_pass.csv
└── blind_second_pass_summary.csv
```

复现命令：

```bash
.venv/bin/python VLS/V0/scripts/extract_blind_annotation_frames.py \
  --focus-file VLS/V0/results/segmentation/annotation_blind/focus.csv \
  --focus-radius 12 --focus-step 2 --focus-columns 7 \
  --focus-boundaries-per-page 2 --crop 180,0,520,360 --thumb-width 240

.venv/bin/python VLS/V0/scripts/evaluate_blind_annotations.py
```

## 8. 验收核对与剩余限制

- 高风险段不沿用草稿边界：满足；新 GT 与第一轮文件物理分离。
- 每个动作明确 `rep_unit` 和边界相位：满足。
- FP 按类型统计：满足。
- 20% 二标抽样：数量满足，但只有同一标注者重读，**独立性不满足**。
- 登山跑：已排除，不输出伪精确计数。
- 指标性质：QEVD-FIT 内部 GT，不是 Fit3D 官方 GT，不能写成 AIFit 论文指标复现。

P0-B 已直接在 `annotation_blind/` 上扩展过分割率、欠分割率、带符号边界误差和 macro/micro 对照，见 `task1-3_p0b_results.md`；第一轮标注未修改。
