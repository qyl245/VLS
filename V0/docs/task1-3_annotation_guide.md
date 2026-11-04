# task1-3 人工标注指南

## 1. 标注目标

人工标注不是从零画 rep，而是对 AIFit 的草稿边界做审核和修正，得到一份小型内部 GT，用于回答：

- AIFit 的 rep 数是否对。
- 起止帧是否落在正确动作周期。
- 峰值基线漏检或误检了多少。
- 哪种主角度、平滑方式和噪声水平更稳定。

这份 GT 不是 Fit3D 官方 GT，不能用于声称复现 AIFit 论文的 IoU。

## 2. 已准备的标注集

模板：`VLS/V0/results/segmentation/annotation/annotation_template.csv`

动作段清单：`VLS/V0/results/segmentation/annotation/annotation_segments.csv`

当前包含 7 个动作段、51 个 AIFit 草稿 rep：

| 视频 | 动作 | 草稿 rep 数 |
|---|---|---:|
| `0153` | squat | 9 |
| `0153` | pushup | 3 |
| `0161` | lunge jumps | 6 |
| `0161` | pushup | 7 |
| `0173` | good morning | 9 |
| `0173` | boxing squat punches | 11 |
| `0218` | walking lunges | 6 |

2026-10-25 已完成首轮抽帧审核：51 个草稿中保留 47 个 GT rep，删除 4 个非完整或停顿 rep。结果见 `task1-3_results.md` 第 9 节。

这组刻意避开了 `mountain climbers`、`air jump rope`、`toe touchers` 等已知低置信动作，先把标注口径跑通。

### 2.1 P0-A 盲标集（与第一轮 GT 分开）

2026-10-25 新增 6 个高风险动作段的原始抽帧盲标，保存在：

```text
VLS/V0/results/segmentation/annotation_blind/
```

盲标过程不显示 AIFit 红线、峰值蓝线或草稿 rep 边界，也不修改 `annotation/annotation_template.csv`。两份 GT 的用途不同：第一轮用于历史筛查，P0-A 盲标用于当前失败模式评估。

| video/action | rep 数 | `rep_unit` | 是否入评估 |
|---|---:|---|---|
| `0161/pushup` | 5 | `cycle` | 是 |
| `0173/boxing_squat_punches` | 10 | `cycle` | 是 |
| `0161/lunge_jumps` | 6 | `cycle` | 是 |
| `0218/walking_lunges` | 6 | `stride` | 是 |
| `0218/air_jump_rope` | 39 | `cycle` | 是 |
| `0162/mountain_climbers` | 8 暂定 | `side` | 否，整段排除 |

关键文件：

- `blind_reps.csv`：逐 rep 边界、`rep_unit`、相位、置信度和状态。
- `blind_segments.csv`：动作窗口、active 区间、边界极性和 `phase_landmark`。
- `blind_exclusions.csv`：rest、transition、incomplete_rep 和 unclear 区间。
- `second_pass_sample.csv`：抽样重读记录，必须明确标注者是否独立。
- `frames/manifest.json`：原始总览页和边界精修页的帧清单。

登山跑因左右相位和姿态质量不稳定整段排除，不继续猜 rep 数。`0161/pushup` 的 5 个边界与 5 个肘屈谷对应；`0218/air_jump_rope` 的 `f1663-1684` 是恢复段，不计第 40 个 rep。详细证据见 `task1-3_p0a_results.md`。

P0-A 评估命令：

```bash
.venv/bin/python VLS/V0/scripts/evaluate_blind_annotations.py
```

输出包括 `blind_eval*.csv/json`、`blind_eval_by_rep_unit.csv`、`blind_false_positives.csv`、`blind_segmentation_errors.csv`、`blind_vs_first_round.csv`、`blind_second_pass*.csv` 和 `blind_validation.csv`。主报告阈值为 IoU 0.30，同时保留 0.10/0.50。

### 2.2 P0-B 评估口径

P0-B 保留 raw IoU 和 phase-aligned IoU 两列，并同时输出：

- `micro_*`：汇总全部匹配 rep 后计算，长段和 rep 多的段权重更高。
- `macro_*`：先按动作段计算再等权平均，小段不会被淹没。
- `mean_signed_start/end_error_frames`：预测边界减 GT 边界，负值表示预测更早。
- `p95_abs_start/end_error_frames`：起点和终点各自的绝对误差 p95。
- `over_segmentation_rate` / `under_segmentation_rate`：majority-overlap split/merge 率。

split/merge 主口径：交集至少覆盖较短区间的 50%。一个 GT rep 与 ≥2 个预测 rep 达到该 overlap 记为 over-segmented；一个预测 rep 与 ≥2 个 GT rep 达到该 overlap 记为 under-segmented。逐 rep 证据在 `blind_segmentation_errors.csv`。不同 `rep_unit` 的结果必须看 `blind_eval_by_rep_unit.csv`，不能只读混合汇总。

完整定义和数字见 `task1-3_p0b_results.md`。P0-B 没有利用这批数据调参；后续调参必须按完整动作段或视频留出验收集。

### 2.3 P0-C 结果路径

P0-C 使用同一份盲标 GT，没有修改 `annotation_blind/blind_reps.csv` 或 `blind_segments.csv`。当前 active-bout 版本输出在 `results/segmentation_p0c/`，对应评估副本在 `results/segmentation_p0c/annotation_blind_eval/`。`results/segmentation/` 保留为 P0-B baseline。

P0-C 的 0.5 s 活动窗口和阈值 `0.10` 是在包含三段验收样本上确定的，尚未做视频级留出；因此当前 F1 只能作为 QEVD-FIT 内部诊断。后续若继续调参，应把完整动作段/视频分成调参集和验收集，并保留本版本作为 baseline。详细结果见 `task1-3_p0c_results.md`。

P0-C 收口已冻结 5 段盲标、三段验收、22 段 rep 数和 IoU=0.30/0.50 指标。后续算法改动先运行：

```bash
.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode guard
```

`exact` 模式用于复现当前 P0-C，`guard` 模式允许指标改善但不允许盲标计数变化或指标退化。收口记录见 `task1-3_p0c_closure.md`。

## 3. 什么算一个 rep

一个 rep 是**从某一相位到下一次同一相位的一个完整周期**，起止帧都使用解码帧索引，闭区间。

推荐统一用主角度曲线上清晰的转折点作为边界：

| 动作 | 建议边界相位 |
|---|---|
| squat | 站立伸展位；`*_knee_flexion` 接近最大，`*_hip_flexion_sagittal` 接近最小 |
| pushup | 撑起伸直位，肘角最大处 |
| good morning | 站立伸展位；`*_hip_flexion_sagittal` 为最小值（屈曲量最小） |
| lunge jumps | 双脚接触或最低点；左右腿必须统一 |
| walking lunges | 前脚落地或最低点；不要一轮左腿、一轮右腿各算一次 |
| boxing squat punches | 拳头回到防守位、肩肘角回到同一相位 |

边界规则：

- `start_frame` 和 `end_frame` 都是包含帧。
- 下一 rep 的 `start_frame` 应当等于上一 rep 的 `end_frame + 1`。
- 不要让边界落在 NaN、低可见度或动作过渡帧上。
- 如果曲线相位和 AIFit 草稿相差半周期，以曲线转折点为准；评估脚本会同时报告相位对齐后的 IoU。

具体边界相位请填在 `phase_landmark` 列，例如 `top`, `bottom`, `foot_contact`, `guard`。

### 示例：0153 / pushup 不能只数“谷—谷”

`0153/pushup` 的肘角谷底在约 `2181`、`2221`、`2272`。如果只取相邻谷底，会得到两段：

- 2181–2220
- 2221–2271

但视频抽帧显示动作段开始时人已经在撑起位：约 `2165` 撑起，`2175–2181` 下落到第一下底部，`2195–2206` 撑起；随后 `2221` 是第二下底部，`2272` 是第三下底部。

因此整段应计为 **3 个有效 pushup**，而不是 2 个。谷—谷只覆盖了后两个完整周期，漏掉了第一下。建议统一用撑起位/下降起点作为边界：

- Rep 1：约 `2166–2206`
- Rep 2：约 `2207–2255`
- Rep 3：约 `2256–2295`

这说明一个重要规则：**如果动作段从“谷”之外的其他相位开始，就不能只用谷—谷来定义整段 rep；要么统一用两端都存在的相位，要么把首尾的完整 rep 显式补上。**

## 4. 怎么操作 CSV

用 IDE 打开 `annotation_template.csv`。每一行是一个草稿 rep：

| 字段 | 用法 |
|---|---|
| `segment_id` | 不要改 |
| `video_id`, `action`, `rep_id` | 正常不要改 |
| `draft_start_frame`, `draft_end_frame` | AIFit 原边界，只读参考 |
| `gt_start_frame`, `gt_end_frame` | 人工修正后的边界 |
| `review_status` | `pending` / `accept` / `adjust` / `delete` / `add` |
| `phase_landmark` | 边界相位 |
| `boundary_confidence` | `high` / `medium` / `low` |
| `annotator` | 标注人 |
| `reviewed_at` | 日期，例如 `2026-10-25` |
| `notes` | 遮挡、过渡、动作不标准等 |

状态含义：

- `pending`：未审核，不进入评估。
- `accept`：AIFit 草稿边界正确，`gt_*` 可以保持原值。
- `adjust`：AIFit 边界存在，但已修改 `gt_*`。
- `delete`：AIFit 多检了一个 rep，该行不进入 GT。
- `add`：AIFit 漏检了一个 rep。新增一行，`draft_*` 留空，填 `gt_*`。

新增 rep 示例：

```csv
0153_squat,0153,squat,10,,,1780,1818,add,top,high,guests,2026-10-25,AIFit missed this rep
```

注意：新增行的 `segment_id`、`video_id`、`action` 必须与已有行一致；`rep_id` 建议接着现有最大编号往后编。

## 5. 标注步骤

1. 打开 `annotation_segments.csv`，找到当前段的 `plot_path`。
2. 打开对应 `plots/{video_id}_{action}_segmentation.png`。
3. 在图上查看 AIFit 红色边界和角度曲线，确认 rep 数和每个周期。
4. 运行抽帧脚本，检查每个 rep 的起止相位和动作完整性：

   ```bash
   .venv/bin/python VLS/V0/scripts/extract_annotation_frames.py
   ```

   默认输出到 `annotation/frames/`，每个 rep 抽样 `start-12`、`start`、`start+12`、`mid`、`end-12`、`end`、`end+12` 七帧。
5. 回到 CSV，把该段每行状态改为 `accept`、`adjust` 或 `delete`。
6. 漏掉的周期新增 `add` 行。
7. 每段至少检查第一个 rep、最后一个 rep、停顿前后的 rep 和置信度最低的 rep。
8. 全部标完后运行评估脚本。

建议不要一次性标完所有段。先标 `0153/squat` 和 `0153/pushup`，和另一人核对口径，再继续后面的段。

## 6. 质量控制

- 不要直接把全部 `pending` 改成 `accept`，否则评估只是在测 AIFit 自己。
- 每个段先确认 rep 数，再微调起止帧；不要一开始就纠结 1–2 帧。
- 对 `boundary_confidence=low` 的 rep，必须在 `notes` 里写原因。
- P0-A 至少抽取 20% 的 rep 做二标；优先让第二个人独立复核。只有同一标注者时，可做原始帧重读，但必须记录 `same_annotator`，不能把 100% 一致写成 inter-annotator agreement。
- 如果某段动作本身不标准、镜头切换、遮挡严重，整段标 `notes=exclude` 并在评估时删除对应 `segment_id` 行。

## 7. 运行评估

```bash
.venv/bin/python VLS/V0/scripts/evaluate_annotations.py
```

输出：

```text
VLS/V0/results/segmentation/annotation/
├── annotation_eval.csv
├── annotation_eval_summary.csv
├── annotation_eval.json
└── annotation_validation.csv
```

评估内容：

- `mean_iou`：人工 GT 与预测边界的原始 IoU。
- `phase_aligned_iou`：允许整段统一平移后取最优的 IoU，用于消除“起始相位约定不同”的影响。
- `precision` / `recall` / `f1`：按 IoU ≥ 0.10 视为匹配。
- `count_error`：预测 rep 数减人工 rep 数。
- `mean_boundary_error_s`：匹配 rep 的平均边界误差，单位秒。

如果只标了部分段，评估脚本只统计已标段；`pending` 不计入。

## 8. 怎么理解结果

内部 GT 的建议判断线：

- `phase_aligned_iou ≥ 0.70` 且 `|count_error| ≤ 1`：该动作段基本可用。
- `0.50–0.70`：边界有偏移或个别漏检，需要看失败案例。
- `< 0.50`：不建议直接进入下游训练，先查主角度、遮挡和动作段质量。

这些阈值只用于内部筛查，不等同于 Fit3D 官方验收。

## 9. 重新生成模板

如果之后想换标注集：

```bash
.venv/bin/python VLS/V0/scripts/prepare_annotation.py \
  --segments 0153:squat 0161:pushup 0173:good_morning \
  --overwrite
```

默认输出不会覆盖已有 `annotation_template.csv`；确认已备份后才加 `--overwrite`。
