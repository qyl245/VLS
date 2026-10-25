# task1-3 P0-C 收口

日期：2026-10-25

## 1. 结论

P0-C 已完成收口，可以进入 P0-D1。收口后：

- synthetic 回归 `7/7` 通过。
- frozen golden regression 在 `exact` 模式下零差异通过。
- 三段指定验收计数保持不变：`0153/squat=8`、`0161/pushup=5`、`0173/boxing_squat_punches=10`。
- 5 段盲标指标保持不变：IoU=0.30 为 `67 Pred / 65 TP / 2 FP / 1 FN`、micro F1 `0.977`、raw IoU `0.585`；IoU=0.50 F1 `0.677`。
- AIFit 全量输出由 P0-B 的 `372` 个 rep 变为 `366` 个；其中 `0218/toe_touchers` 由 P0-B 的 2 个变为 0 个。

本轮没有把未盲标段加入调参真值，也没有修改 `annotation_blind/blind_reps.csv`、`blind_segments.csv` 或 `blind_exclusions.csv`。

## 2. 收口时修复的两个问题

### 2.1 越出 active core 的 rep

原 P0-C 允许 AIFit 在向低活动区间中点延伸的 fit window 内产生 rep，但没有检查 rep 是否真正碰到 active core。`0218/toe_touchers` 因此产生了 4 个完全位于 `transition 3139-3787`、且不与 active core `3130-3138` 相交的 rep。

修复：`segment_reps.py` 新增 `filter_reps_to_active_core()`。只有当 rep 与所属 active core 至少相交 1 帧时才保留；完全位于 fit extension 中的 rep 记为 `outside_active_bout` 并拒绝。该修复只删除越界 rep，不改变核心分段或阈值。

### 2.2 零 rep 段丢失证据

原流程在 AIFit 与 peak 都没有输出时跳过该段，导致被拒区间没有写入 `non_rep_intervals.csv` 和 `run.json`。

修复：零 rep 段仍写入 segment payload、active bout 和 non-rep 证据。`0218/toe_touchers` 现在可在输出中追溯 `outside_active_bout`、`low_local_amplitude` 和 `low_activity`。

## 3. 三段未盲标变化复核

抽帧产物：`VLS/V0/results/segmentation_p0c/closure_frames/`

| segment | P0-B | P0-C 收口后 | 复核结论 |
|---|---:|---:|---|
| `0153/jumping_jack` | 28 | **30** | 原始视频与 `left_shoulder_abduction_coronal` 信号支持 30 个周期；P0-B 漏掉首尾各一个边界 |
| `0161/butt_kickers` | 19 | **17** | 原始帧显示 `f1202-1219` 是进入动作前的过渡；稳定动作段约 `f1275-1594`，17 个周期与信号峰一致 |
| `0218/toe_touchers` | 2 | **0** | P0-B 的 `f3092-3103`、`f3728-3743` 是站姿到弯腰/弯腰到站立的过渡；原 P0-C 的 4 个 rep 位于连续弯腰保持区间，不与 active core 相交。该段仍无可靠 rep 真值，继续排除 |

`0153/jumping_jack` 和 `0161/butt_kickers` 的变化可作为当前版本的定性支持，但仍不是盲标 GT。`0218/toe_touchers` 不进入任何 F1 或调参结论。

## 4. 冻结内容

golden 配置：`VLS/V0/configs/p0c_golden.json`

冻结项：

- 4 份盲标 GT 文件的 SHA-256。
- P0-C 参数：0.5 s 活动窗口、阈值 `0.10`、0.2 s 缝隙合并、0.5 s 最短 non-rep、`incomplete_tail_ratio=0.75`。
- 22 段 AIFit/peak rep 数。
- 三段指定验收 rep 数。
- 5 段盲标在 IoU=0.30/0.50 的 Pred/TP/FP/FN、F1、raw IoU、边界误差、split/merge 和 FP 类型。

回归报告：`VLS/V0/results/segmentation_p0c/closure_regression.json`

## 5. 回归入口

合成测试覆盖：周期正弦、周期 + 休息平台、active-core 交集、越界 rep 过滤、变周期、低可见度缺帧和半周期结尾。

```bash
# 重现冻结 P0-C：synthetic + 全量重跑 + 盲标复算
.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode exact \
  --report VLS/V0/results/segmentation_p0c/closure_regression.json

# P0-D/P1 改动后的非退化检查：允许指标改善，不允许计数变化或指标退化
.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode guard
```

单测也可以直接运行：

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py
```

## 6. 剩余限制

- 仍是 QEVD-FIT 内部盲标 GT，不是 Fit3D 官方指标。
- 独立第二标注者仍为 `pending`；22.7% 只是同一标注者原始帧重读。
- P0-C 参数在包含验收段的样本上确定，未做视频级留出。
- `0218/toe_touchers` 的 rep 口径仍未解决，当前输出 0 个 rep 只表示“没有通过 active-core 证据的候选”，不表示已确认该动作是 0 个 rep。
- 边界相位问题仍未修；IoU=0.50 F1 仍只有 `0.677`，进入 P0-D1。
