# task1-3 P1-D：鲁棒模板与可解释置信度

日期：2026-10-25

## 1. 结论

P1-D 已实现并完成回归。本轮只做**证据拆分和人工复核排序**，没有启用自动删除，也没有改变任何 rep 边界或 rep 数。

- 默认形状模板改为 IRLS 鲁棒模板；`mean`、`medoid`、`trimmed` 可通过 `--confidence-template` 做消融。
- 每个 rep 新增 `visibility`、`shape_similarity`、`phase_consistency`、`motion_evidence` 四项证据，以及 `review_score`、`review_priority`、`reject_reason`。
- legacy `confidence` 保留，接口仍向后兼容。
- IoU=0.30 的唯一剩余 FP 是 `0218/air_jump_rope` 的 `1663-1679` transition（pred rep 40）。
- `motion_evidence` 能解释该 FP：它在该段 40 个 rep 中排倒数第 2，低于 TP 的 p05；legacy confidence 只排倒数第 8。
- 在 P0-C recall `0.98485` 下，单独使用 `motion_evidence` 可把 precision 从 `0.98507` 提到 `1.0`，但会从当前 D2/P1-D 的 66 个盲标 TP 中损失 1 个。首轮按计划不启用自动门控。
- 5 段盲标和全量 22 段的所有边界、rep 数、raw IoU、IoU=0.50 F1、boundary p95 与 P0-D2 完全一致。

因此本轮验收为 **pass_review_only / partial**：证据排序和可解释性目标达到，固定 recall 下出现了 precision 增益；但 LOSO 学习式校准不可识别，且自动删除按设计未启用。

## 2. 实现

### 2.1 鲁棒模板

`VLS/V0/scripts/segment_reps.py` 新增：

- `mean`：原均值模板，用作基线。
- `medoid`：选择与其余曲线平均相关性最高的真实 rep 曲线。
- `trimmed`：逐采样点裁掉上下 20% 后取均值（rep 数不足 5 时回退均值）。
- `irls`：以中位曲线初始化，按曲线到模板的相关距离做迭代重加权，默认用于 `shape_similarity`。

形状证据按整个动作段重新计算，而不是按 bout 各自拟合模板；bout 只影响分割，不改变证据口径。

### 2.2 四项证据

| 字段 | 定义 |
|---|---|
| `visibility` | rep 区间内主角度有效帧比例 |
| `shape_similarity` | rep 重采样曲线与鲁棒模板的相关系数，截断到 `[0, 1]` |
| `phase_consistency` | 边界到 landmark 同极性极值的偏移 + rep 宽度相位方差；不使用 phase-aligned IoU |
| `motion_evidence` | rep 内 activity median 相对同段 rep 的稳健对数偏离；过高和过低都会被降分 |

`phase_consistency` 沿用 P0-D 的相位证据，但仅在至少 3 个相位极值可用时给信息量分数；否则使用中性值 `0.5`，避免把不可用相位误判成错误。

`review_score` 是四项的等权几何均值，仅用于排序，不是删除概率。`review_priority` 和 `reject_reason` 只标明人工复核优先级与低分证据，低置信 rep 仍完整保留。

### 2.3 CLI 与输出

```bash
--confidence-template {mean,medoid,trimmed,irls}
```

`reps.csv` 和 `run.json` 在原字段后追加：

```text
visibility, shape_similarity, phase_consistency, motion_evidence,
review_score, review_priority, reject_reason
```

`run.json.parameters` 明确写入 `confidence_auto_delete=false`。

## 3. 剩余 FP 的证据

### 3.1 `0218/air_jump_rope` rep 40

| field | value |
|---|---:|
| frames | `1663-1679` |
| type | transition |
| legacy confidence | `0.35541`（该段倒数第 8） |
| visibility | `1.000` |
| shape_similarity | `0.320`（TP 百分位 `0.154`） |
| phase_consistency | `0.641`（TP 百分位 `0.615`） |
| motion_evidence | `0.00103`（该段倒数第 2，TP p05 `0.0743`） |
| review_score | `0.1205` |
| review_priority | `high` |
| reject_reason | `low_motion_evidence` |

这说明尾部恢复段不是“动作幅度不足”，而是恢复动作的活动度显著高于该段典型 rep；原有低活动度门控无法抓到，`motion_evidence` 可以。

### 3.2 IoU=0.30 分数消融

固定 recall 使用 P0-C 基线 `0.98485`（`65/66`）：

| score | ROC AUC | AP | fixed-recall precision | kept | TP retained |
|---|---:|---:|---:|---:|---:|
| legacy confidence | 0.8636 | 0.99785 | 0.98507 | 67 | 66 |
| visibility | 0.4848 | 0.98463 | 0.98507 | 67 | 66 |
| shape_similarity | 0.8485 | 0.99759 | 0.98507 | 67 | 66 |
| phase_consistency | 0.2955 | 0.98216 | 0.98507 | 67 | 66 |
| **motion_evidence** | **0.9848** | **0.99977** | **1.00000** | **65** | **65** |
| all-four geomean | 0.9242 | 0.99883 | 0.98507 | 67 | 66 |

相对 legacy confidence，`motion_evidence`：

- ROC AUC `+0.1212`；
- AP `+0.00192`；
- P0-C recall 下 precision `+0.01493`。

但它通过同时移除尾部 FP 和一个低活动 TP 达成，相对于当前 D2/P1-D 的 66 TP 仍损失 1 个 TP。因此该信号适合做 `review_priority=high`，不适合首轮自动删除。等权 all-four 分数没有固定 recall 增益，也说明不能把四项简单压成一个门控分数。

### 3.3 IoU=0.50 duplicate

IoU=0.50 下仍有 12 个 FP（11 duplicate + 1 transition）：

- 4 个 FP 的某一分项低于 TP p05：3 个由 `phase_consistency` 解释，1 个由 `motion_evidence` 解释。
- 其余 8 个没有单项证据达到该阈值，说明短周期 split/merge 仍需要边界级方法（P2）处理，不能靠首轮低置信删除解决。
- `shape_similarity` 在 IoU=0.50 的 AP 略高于 legacy（`+0.00449`），但 AUC 低 `0.00909`，不支持把它单独作为门控。

## 4. 模板消融

实现使用同一组预测边界，只替换形状模板；因此 raw IoU、IoU=0.50 F1 和 boundary p95 必须保持不变。

| template | IoU=0.30 AUC | IoU=0.30 AP | IoU=0.50 AUC | IoU=0.50 AP |
|---|---:|---:|---:|---:|
| mean | 0.9242 | 0.99880 | 0.5227 | 0.8391 |
| medoid | 0.9242 | 0.99880 | 0.5333 | 0.8423 |
| trimmed | 0.9242 | 0.99880 | 0.5242 | 0.8392 |
| **irls** | **0.9242** | **0.99880** | **0.5242** | **0.8394** |

模板替换在本轮只有很小的排序差异，没有观察到对边界指标的收益或退化。IRLS 作为默认值的理由是抗离群、无需标签且不牺牲现有边界；不能宣称它单独带来了 IoU 提升。

## 5. LOSO 与校准限制

评估脚本执行了 leave-one-segment-out 检查，但当前 5 段盲标中所有预测 FP 都集中在 `0218/air_jump_rope`：

- 留出该段时，训练集只有正样本；
- 留出其他段时，测试集没有负样本；
- 因此没有任何 fold 同时具备训练和测试两类的可评估条件。

本轮没有用同一批盲标拟合权重后再原样报告，而是使用预先定义的等权分数，并明确记录 LOSO 为 `not_estimable`。逐段 AP/AUC 和固定 recall 结果仍在 `p1d_score_by_segment.csv` 中，供后续数据扩充后复核。

补充更多有 FP/FN 的盲标动作段之前，不应启用学习式置信度门控。

## 6. 盲标指标

P1-D 与 P0-D2 完全一致：

| metric | P0-D2 | P1-D |
|---|---:|---:|
| IoU=0.30 Pred/TP/FP/FN | 67/66/1/0 | 67/66/1/0 |
| IoU=0.30 micro F1 | 0.99248 | 0.99248 |
| raw micro IoU | 0.68514 | 0.68514 |
| mean abs start/end | 5.18 / 4.80 | 5.18 / 4.80 |
| p95 boundary | 12.00 | 12.00 |
| IoU=0.50 F1 | 0.82707 | 0.82707 |
| IoU=0.50 raw IoU | 0.74142 | 0.74142 |

5 段盲标 rep 数保持 `6/5/10/40/6`（预测），全量保持 22 段、AIFit `366` rep、Peak `341` rep。

## 7. 验证

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py
# 17 passed

.venv/bin/python VLS/V0/scripts/check_p1d_regression.py \
  --use-existing \
  --report VLS/V0/results/segmentation_p1d/p1d_regression.json
# status=pass, acceptance_status=partial, difference_count=0

.venv/bin/python VLS/V0/scripts/check_p0d2_regression.py \
  --use-existing \
  --segmentation-dir VLS/V0/results/segmentation_p1d \
  --report VLS/V0/results/segmentation_p1d/p0d2_regression.json
# pass; D2 boundary behavior unchanged

.venv/bin/python VLS/V0/scripts/check_p0d1_regression.py \
  --report VLS/V0/results/segmentation_p1d/p0d1_regression.json
# pass

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode guard \
  --segmentation-dir VLS/V0/results/segmentation_p1d \
  --use-existing \
  --report VLS/V0/results/segmentation_p1d/p0c_guard.json
# pass

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode exact \
  --report VLS/V0/results/segmentation_p1d/p0c_exact_after_p1d.json
# pass
```

新增 synthetic 测试覆盖：

- IRLS 模板比均值模板更抗相位偏移离群曲线；
- 四项证据完整输出、低运动证据触发高复核优先级但不删除 rep；
- rep 重新编号保留全部置信度字段。

## 8. 产物

```text
VLS/V0/scripts/segment_reps.py
VLS/V0/scripts/evaluate_p1d_confidence.py
VLS/V0/scripts/check_p1d_regression.py
VLS/V0/tests/test_segment_reps.py
VLS/V0/results/segmentation_p1d/
├── reps.csv
├── run.json
├── annotation_blind_eval/
├── p1d_confidence_eval.json
├── p1d_score_ablation.csv
├── p1d_pr_curves.csv
├── p1d_rep_scores.csv
├── p1d_fp_evidence.csv
├── p1d_score_by_segment.csv
├── p1d_template_ablation.csv
├── p1d_loso_folds.csv
├── p1d_regression.json
├── p0d2_regression.json
├── p0d1_regression.json
├── p0c_guard.json
└── p0c_exact_after_p1d.json
```

## 9. 限制与下一步

- 当前仍是 QEVD-FIT 内部盲标，不是 Fit3D 官方指标；独立第二标注者仍为 `pending`。
- IoU=0.30 只有 1 个 FP，AUC/AP 对小样本非常敏感；本轮只把它作为错误排序证据，不宣称泛化性能。
- `phase_consistency` 在相位极值不足时使用中性值；pushup 两个 bout 的相位证据仍不充分。
- 剩余 8 个 IoU=0.50 duplicate 没有达到单项 TP p05 解释阈值，应该进入 P2 的短周期 split/merge 边界处理，而不是降低门控阈值。
- 自动删除继续保持关闭；若要启用，至少需要更多含 FP/FN 的盲标段、可识别的 LOSO 校准，以及“两项独立证据 + 不损失当前盲标 TP”的冻结回归。

