# task1-3 P0-B：评估口径扩展结果

日期：2026-10-25

## 1. 结论

P0-B 已在 P0-A 的盲标 GT 上完成，未修改 `blind_reps.csv`、`blind_segments.csv` 或第一轮 GT。

- AIFit 和峰值基线均在 IoU `0.10`、`0.30`、`0.50` 下评估。
- `0.30` 作为主匹配口径，`0.50` 作为严格边界口径；两者分别报告。
- 原始 IoU 和 phase-aligned IoU 分开报告，phase-aligned IoU 不替代原始边界精度。
- 新增 majority-overlap 的 split/merge 统计、signed start/end、start/end 各自 p95。
- 同时报告 micro（按匹配 rep 加权）和 macro（按动作段等权）指标。
- 新增 `rep_unit` 分组结果；当前 4 段为 `cycle`，1 段为 `stride`。
- 结构校验仍为 0 issue。

## 2. split/merge 定义

过分割和欠分割不只按 FP/FN 推断，而使用区间 majority overlap：

- 对每个 GT rep，找出与其交集至少占**较短区间 50%** 的预测 rep。若数量 ≥ 2，该 GT rep 记为 `over_segmented`，额外数量为 `split_events`。
- 对每个预测 rep，找出与其交集至少占**较短区间 50%** 的 GT rep。若数量 ≥ 2，该预测 rep 记为 `under_segmented`，额外数量为 `merge_events`。
- `over_segmentation_rate = over_segmented_gt_reps / gt_reps`。
- `under_segmentation_rate = under_segmented_pred_reps / pred_reps`。
- 详细到 rep ID 的证据保存在 `blind_segmentation_errors.csv`。

使用 50% majority overlap 是为了避免短周期动作的正常边界平移被误算成切分。首版按 GT 时长 20% 会把 44/66 个 GT rep 误标为 split，已废弃该口径。

split/merge 与 TP/FP/FN 是不同维度：一个短 rep 的边界平移可能同时产生一次 split 和一次 merge；FP 也可能只是过渡段假 rep，而不构成 split。

split/merge 只由 overlap share 和区间几何决定，不随 IoU 匹配阈值改变；因此同一方法在 0.10/0.30/0.50 三行中的 split/merge 数相同。

## 3. 主结果（IoU ≥ 0.30）

| 方法 | GT | Pred | TP | FP | FN | micro P | micro R | micro F1 | macro P | macro R | macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | 66 | 70 | 65 | 5 | 1 | 0.929 | 0.985 | **0.956** | 0.901 | 0.980 | 0.936 |
| Peak | 66 | 55 | 45 | 10 | 21 | 0.818 | 0.682 | 0.744 | 0.765 | 0.715 | 0.736 |

边界与分割误差：

| 方法 | signed start | signed end | abs start | abs end | p95 start | p95 end | mean boundary s | p95 boundary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | -4.95 | -6.48 | 9.63 | 9.00 | 19 | 19 | 0.311 | 19 |
| Peak | +5.64 | +10.56 | 12.71 | 14.51 | 39 | 39 | 0.454 | 39 |

单位均为帧，`mean boundary s` 除外。signed error 定义为预测帧减 GT 帧；负值表示预测边界更早。

| 方法 | raw micro IoU | raw macro IoU | phase micro IoU | phase macro IoU | over reps/rate | split events | under reps/rate | merge events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | 0.573 | 0.573 | 0.666 | 0.768 | 5/66, 7.6% | 5 | 4/70, 5.7% | 4 |
| Peak | 0.532 | 0.538 | 0.615 | 0.660 | 7/66, 10.6% | 7 | 14/55, 25.5% | 19 |

AIFit 的 micro/macro F1 差异较小；峰值基线的 merge 明显更多，尤其集中在短周期 `air_jump_rope`。

## 4. 阈值敏感性

| 方法 | IoU | TP | FP | FN | micro F1 | macro F1 | raw micro IoU | phase micro IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | 0.10 | 65 | 5 | 1 | 0.956 | 0.936 | 0.573 | 0.688 |
| AIFit | 0.30 | 65 | 5 | 1 | 0.956 | 0.936 | 0.573 | 0.666 |
| AIFit | 0.50 | 42 | 28 | 24 | 0.618 | 0.537 | 0.653 | 0.794 |
| Peak | 0.10 | 49 | 6 | 17 | 0.810 | 0.849 | 0.507 | 0.587 |
| Peak | 0.30 | 45 | 10 | 21 | 0.744 | 0.736 | 0.532 | 0.615 |
| Peak | 0.50 | 28 | 27 | 38 | 0.463 | 0.462 | 0.603 | 0.650 |

严格阈值下均值只覆盖通过 0.50 匹配的 rep，因此 raw IoU 上升不能解释为边界整体变好。主结论以 0.30 的匹配集合为准，0.50 反映严格一致性。

## 5. `rep_unit` 分组

IoU ≥ 0.30：

| 方法 | `rep_unit` | 段数 | GT/Pred | micro F1 | macro F1 | raw micro IoU | raw macro IoU | over rate | under rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIFit | `cycle` | 4 | 60/64 | 0.952 | 0.919 | 0.546 | 0.509 | 8.3% | 6.3% |
| AIFit | `stride` | 1 | 6/6 | 1.000 | 1.000 | 0.832 | 0.832 | 0.0% | 0.0% |
| Peak | `cycle` | 4 | 60/49 | 0.752 | 0.753 | 0.539 | 0.556 | 10.0% | 26.5% |
| Peak | `stride` | 1 | 6/6 | 0.667 | 0.667 | 0.465 | 0.465 | 16.7% | 16.7% |

不同 `rep_unit` 的 rep 数不可直接混算后解释。主 summary 仍提供 `rep_unit=all` 的汇总，分组表用于检查口径差异。

## 6. 分动作段结果（IoU ≥ 0.30）

AIFit：

| segment | `rep_unit` | GT/Pred | TP/FP/FN | raw/phase IoU | signed start/end | over rate | under rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `0161/lunge_jumps` | `cycle` | 6/6 | 6/0/0 | 0.509/0.925 | -18.17/-15.67 | 0.0% | 0.0% |
| `0161/pushup` | `cycle` | 5/7 | 5/2/0 | 0.422/0.699 | +5.40/-19.40 | 20.0% | 0.0% |
| `0173/boxing_squat_punches` | `cycle` | 10/11 | 9/2/1 | 0.532/0.703 | -10.56/-11.89 | 10.0% | 9.1% |
| `0218/air_jump_rope` | `cycle` | 39/40 | 39/1/0 | 0.571/0.571 | -2.33/-2.44 | 7.7% | 7.5% |
| `0218/walking_lunges` | `stride` | 6/6 | 6/0/0 | 0.832/0.943 | -9.00/-4.67 | 0.0% | 0.0% |

Peak：

| segment | GT/Pred | TP/FP/FN | raw/phase IoU | signed start/end | over rate | under rate |
|---|---:|---:|---:|---:|---:|---:|
| `0161/lunge_jumps` | 6/6 | 4/2/2 | 0.544/0.627 | +14.25/+13.25 | 16.7% | 16.7% |
| `0161/pushup` | 5/5 | 4/1/1 | 0.625/0.685 | +10.75/+10.25 | 20.0% | 20.0% |
| `0173/boxing_squat_punches` | 10/10 | 8/2/2 | 0.526/0.710 | +8.75/+11.25 | 10.0% | 10.0% |
| `0218/air_jump_rope` | 39/28 | 25/3/14 | 0.528/0.554 | -2.68/+4.44 | 7.7% | 35.7% |
| `0218/walking_lunges` | 6/6 | 4/2/2 | 0.465/0.723 | +37.75/+45.00 | 16.7% | 16.7% |

## 7. 失败模式

- AIFit 的主问题是 `pushup` 2 个 duplicate 和 `boxing_squat_punches` 末尾 incomplete rep。
- AIFit 在 `air_jump_rope` 只有 1 个 transition FP，但短周期边界平移会产生 3 次 split 和 3 次 merge；计数接近正确不代表边界完全一致。
- 峰值基线在 `air_jump_rope` 的 merge 事件为 15，核心原因是漏检后一个长区间覆盖多个 GT rep。
- `walking_lunges` 用 `stride` 后 AIFit 的 F1 和 IoU 都较好；峰值基线的 signed start error 仍约 38 帧，说明单侧相位偏移没有被 phase-aligned IoU 完全掩盖。

## 8. 参数选择纪律

P0-B 没有在本轮数据上搜索或调整分割参数。后续若做阈值、平滑或主角度选择：

- 以视频或动作段为分组单位划分调参集和验收集，不能在同一段上调参后原样报告。
- 至少留出一整个动作段，不按单个 rep 随机切分，避免同段周期泄漏。
- 同时报告调参集和留出集；最终结论只使用留出集。
- 同一份 P0-A/P0-B GT 可以用于诊断，但不能同时充当调参真值和最终测试真值。

## 9. 产物

```text
VLS/V0/results/segmentation/annotation_blind/
├── blind_eval.csv                  # 分动作段 × 方法 × 阈值
├── blind_eval_summary.csv          # rep_unit=all 的 micro/macro 汇总
├── blind_eval_by_rep_unit.csv      # cycle/stride 分组汇总
├── blind_eval.json                 # 全部 summary、参数和校验信息
├── blind_false_positives.csv       # FP 类型和去向
├── blind_segmentation_errors.csv   # split/merge 的逐 rep 证据
└── blind_validation.csv            # 结构校验
```

复现命令：

```bash
.venv/bin/python VLS/V0/scripts/evaluate_blind_annotations.py
```

## 10. 剩余限制

- 仍是 QEVD-FIT 内部 GT，不是 Fit3D 官方 GT。
- 当前只有一名标注者；22.7% 重读不是独立二标。
- split/merge 使用固定 50% majority-overlap 定义，适合当前报告；若改阈值，应作为独立敏感性分析，不能静默替换主口径。
- 0.50 严格 IoU 下的失败可以用于判断边界质量，但不应单独用来挑参数。
