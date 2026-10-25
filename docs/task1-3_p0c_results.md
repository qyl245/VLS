# task1-3 P0-C：active bout 拆分结果

日期：2026-10-25

## 1. 结论

P0-C 已完成规则版 V0：先用多角度活动度把动作窗口拆成 active bout，再在每个 bout 内运行 AIFit；bout 之间的低活动区间不生成 rep，并写入 `run.json`、`non_rep_intervals.csv` 和 `rejected_reps.csv`。

三段指定验收均通过：

| segment | 原 AIFit | P0-C AIFit | 处理结果 |
|---|---:|---:|---|
| `0153/squat` | 9 | **8** | `1773-1852` 记作 `transition`，末尾半跪不再产生 rep |
| `0161/pushup` | 7 | **5** | `2592-2661` 记作 `rest`；休息/恢复假 rep 被删除，5 个有效 rep 保留 |
| `0173/boxing_squat_punches` | 11 | **10** | `3002-3041` 记作 `incomplete_rep`，末尾未回到站立位的深蹲被删除 |

在 P0-A/P0-B 的 5 段盲标 GT 上，IoU=0.30 的主结果由：

- AIFit `70` 个预测、`65 TP / 5 FP / 1 FN`、micro F1 `0.956`

变为：

- AIFit `67` 个预测、`65 TP / 2 FP / 1 FN`、micro F1 `0.977`

剩余 AIFit FP 为 `0173/boxing_squat_punches` 的 1 个 duplicate 和 `0218/air_jump_rope` 的 1 个 transition。P0-C 没有继续追这两个非本阶段目标错误。

这些数字是 **QEVD-FIT 内部盲标 GT** 上的诊断结果，不是 Fit3D 官方指标，也不是 AIFit 论文复现指标。

**P0-C 收口补充（2026-10-25）**：进入 P0-D 前又补了 active-core 输出约束。完全落在 fit extension、但不与 active core 相交的 rep 记为 `outside_active_bout` 并拒绝；零 rep 段仍保留 non-rep 证据。该修复只影响 `0218/toe_touchers`，三段验收和 5 段盲标指标均不变。收口细节见 `task1-3_p0c_closure.md`。

## 2. 实现

代码：`VLS/V0/scripts/segment_reps.py`

核心流程：

1. 在原动作窗口内，对动作候选角度分别做插值、平滑、去趋势和幅度归一化。
2. 用 0.5 s 滚动窗口计算每个角度的 10–90 分位角度范围，按各自动态范围归一化后取跨角度中位数，作为多角度活动度。
3. 同时计算主角度滚动角速度、主角度有效率和低可见度比例，作为证据写入输出。
4. 低活动阈值固定为 `0.10`；先合并不超过 `0.2 s` 的活动缝隙，再保留长度至少为 `max(0.5 s, 0.5 × 全局周期)` 的低活动区间，避免把一个周期内部的短暂低活动误判为休息。
5. 低活动区间之间为 active bout。AIFit 的拟合窗口向相邻低活动区间中点延伸，以保留足够的起始相位上下文；实际 bout 边界仍按低活动边缘输出。
6. 每个 bout 独立运行原 AIFit 两阶段，外层仍受原方法的有效窗口 `source_start/source_end` 约束。
7. 局部幅度过滤 `local_range < 0.25 × global_range` 保留，仅作为 bout 内的辅助拒因，不重复叠加同类方差门控。
8. 对长周期动作，如果末尾 rep 临近 bout 结束且时长小于参考 rep 的 `0.75`，标记 `incomplete_rep`。
9. 没有被任何有效 bout 覆盖的区间写入 `non_rep_intervals`，并区分 `rest`、`transition`、`incomplete_rep`、`low_activity`。
10. AIFit 输出必须与所属 active core 至少相交 1 帧；完全落在低活动 fit extension 中的 rep 记为 `outside_active_bout`，不进入最终结果。

峰值基线仍在原有效窗口上独立运行，便于保留 AIFit、peak 两个方法的可比较性；active-bout 门控只应用到 AIFit。

## 3. 三段验收证据

### `0153/squat`

```text
active bout 1: 1349-1772, fit 1349-1786, 8 reps
active bout 2: 1801-1852, no reps
non-rep:       1327-1348 transition
non-rep:       1773-1852 transition
```

第二个 bout 无法稳定产生 rep，因此不强行输出。末尾半跪过渡归入 `transition`，AIFit 总数为 8。

### `0161/pushup`

```text
active bout 1: 2488-2591, fit 2465-2626, 2 reps
active bout 2: 2662-2810, fit 2627-2810, 3 reps
non-rep:       2280-2487 transition
non-rep:       2592-2661 rest
non-rep:       2811-2947 transition
```

`f2592-2661` 是撑起位休息平台。向低活动区间中点延伸拟合窗口，避免短 bout 重新选择相位；最终保留 5 个有效 rep。bout 内另有一个低局部幅度候选 `2777-2803` 被拒绝。

### `0173/boxing_squat_punches`

```text
active bout: 2458-3041, 10 reps
non-rep:     2368-2457 transition
non-rep:     3002-3041 incomplete_rep
```

末尾 AIFit 候选 `3002-3036` 只有 35 帧，参考宽度 51 帧，比例 `0.686`，且临近拟合窗口结束；因此删除并记录为 `incomplete_rep`。盲标排除说明中的未完成深蹲起点约为 `3018`，这里的输出从被拒绝的预测 rep 起点 `3002` 延伸到窗口末尾，覆盖范围更保守。

## 4. 盲标评估

IoU=0.30，5 段、66 个 GT rep：

| 方法/版本 | Pred | TP | FP | FN | micro P | micro R | micro F1 | macro F1 | raw micro IoU | phase micro IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0-B AIFit | 70 | 65 | 5 | 1 | 0.929 | 0.985 | 0.956 | 0.936 | 0.573 | 0.666 |
| P0-C AIFit | 67 | 65 | 2 | 1 | 0.970 | 0.985 | **0.977** | **0.977** | 0.585 | 0.660 |
| P0-C Peak | 55 | 45 | 10 | 21 | 0.818 | 0.682 | 0.744 | 0.736 | 0.532 | 0.615 |

边界与分割误差：

| 版本 | signed start/end | abs start/end | p95 start/end | mean boundary s | over/split | under/merge |
|---|---:|---:|---:|---:|---:|---:|
| P0-B AIFit | -4.95/-6.48 帧 | 9.63/9.00 帧 | 19/19 帧 | 0.311 | 7.6% / 5 | 5.7% / 4 |
| P0-C AIFit | -3.52/-4.75 帧 | 9.86/8.20 帧 | 20.6/19.0 帧 | 0.301 | 6.1% / 4 | 6.0% / 4 |

P0-C 减少了 3 个 FP，raw IoU 和平均边界误差小幅改善；start 的绝对误差和 p95 略有变差，说明本阶段主要修的是“是否生成 rep”，不是边界相位精度。边界相位问题仍留给 P0-D。

收口后的 active-core 过滤没有改变上述 5 段盲标指标。

分动作：

| segment | GT/Pred | TP/FP/FN | raw IoU | 结果 |
|---|---:|---:|---:|---|
| `0161/lunge_jumps` | 6/6 | 6/0/0 | 0.509 | 与 P0-B 相同 |
| `0161/pushup` | 5/5 | 5/0/0 | 0.578 | 2 个 duplicate FP 消失 |
| `0173/boxing_squat_punches` | 10/10 | 9/1/1 | 0.532 | incomplete FP 消失，仍有 1 个 duplicate |
| `0218/air_jump_rope` | 39/40 | 39/1/0 | 0.571 | transition FP 未修 |
| `0218/walking_lunges` | 6/6 | 6/0/0 | 0.832 | 与 P0-B 相同 |

## 5. 全量 rep 数变化

22 个动作段中，AIFit 总 rep 数由 P0-B 的 372 变为 P0-C 收口后的 366。除三段验收外，还有 3 个未在本轮盲标集内的段发生变化：

| segment | 原 rep | P0-C rep | 状态 |
|---|---:|---:|---|
| `0153/jumping_jack` | 28 | 30 | 抽帧与肩外展周期支持 30；P0-B 漏掉首尾边界，无盲标 GT |
| `0161/butt_kickers` | 19 | 17 | 抽帧确认 `f1202-1219` 是过渡；稳定段 17 个周期，无盲标 GT |
| `0218/toe_touchers` | 2 | 0 | P0-B 的 2 个是站起/弯腰过渡；原 P0-C 的 4 个越出 active core，收口后删除。该段口径仍未解决，继续排除 |

这些变化说明 active-bout 参数尚未在完整动作集上做留出验证，不能默认全是改善。它们没有进入 F1 结论。`0218/toe_touchers` 的 0 个 rep 表示没有通过 active-core 证据的候选，不表示已确认该动作是 0 个 rep。

峰值基线没有应用 active-bout 门控，因此 21 段、341 个 rep 与原基线一致。唯一失败仍是 `0218/toe_touchers` 峰值 0 个 rep。

## 6. 产物

```text
VLS/V0/results/segmentation_p0c/
├── summary.csv                    # 新增 active_bouts/non_rep/rejected 计数
├── reps.csv                       # 原接口字段保持兼容
├── action_segments.csv
├── non_rep_intervals.csv          # active bout 之外的区间及拒因/证据
├── rejected_reps.csv              # rep 级拒绝原因
├── run.json                       # active_bouts、non_rep_intervals、activity_evidence
├── plots/                         # 灰色 non-rep、绿色 bout、橙色拒绝 rep
├── annotation_blind_eval/         # 本版本的盲标评估结果与 GT 输入副本
├── closure_frames/                # 三段未盲标变化的原始抽帧与密集边界页
└── closure_regression.json        # synthetic + golden regression 报告
```

`reps.csv` 仍为 `{video_id, action, method, angle, rep_id, start_frame, end_frame, peak_frame, confidence}`；rep id 在每个方法内从 1 连续编号。

复现：

```bash
.venv/bin/python VLS/V0/scripts/segment_reps.py \
  --output-dir VLS/V0/results/segmentation_p0c

.venv/bin/python VLS/V0/scripts/evaluate_blind_annotations.py \
  --blind-dir VLS/V0/results/segmentation_p0c/annotation_blind_eval \
  --reps VLS/V0/results/segmentation_p0c/reps.csv \
  --segments VLS/V0/results/segmentation_p0c/action_segments.csv

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py \
  --mode exact \
  --use-existing \
  --report VLS/V0/results/segmentation_p0c/closure_regression.json
```

旧版 baseline 保留在 `VLS/V0/results/segmentation/`，未被覆盖。

## 7. 限制与后续

- 仍是 QEVD-FIT 内部 GT；Fit3D 尚未到位，不能把当前数字写成官方指标。
- 只有一名标注者；22.7% 是同一标注者重读，不是独立二标。
- 阈值 `0.10`、0.5 s 窗口和周期相关最短间隔是在包含验收段的样本上确定的，尚未做视频级留出；因此 P0-C 的 F1 提升是诊断结果，不是无偏泛化估计。
- active-bout 能删除明显的休息/恢复和末尾半周期，但不能自动把 mid-bout duplicate 全部归因；`boxing` 的 1 个 duplicate 仍存在。
- 低幅度真实周期可能被保守阈值漏掉；`air_jump_rope` 的尾部 transition FP 仍未解决。
- 边界相位和严格 IoU 仍不理想；下一步应做 P0-D，而不是继续提高活动度阈值。
- `0218/toe_touchers` 仍无可靠 rep 口径；收口只保证不输出没有 active-core 证据的候选，不能把它当作已确认的 0 rep GT。

完整收口检查、冻结内容和复现命令见 `task1-3_p0c_closure.md`。
