# task1-6D：置信度校准与拒答

## 1. 执行结论

状态：**Fit3D dev 阶段门通过（2026-10-26）**；QEVD dev 仅完成跨域 smoke，正式阶段门不可评估。

本轮在 RepChain v2 的 `complete / in_set / combined` 三层原始置信度上增加：

- 只使用 Fit3D dev 的默认组完整 rep 拟合校准器；
- 5 个等宽分箱的 Beta/Clopper-Pearson 下界与单调化；
- `combined >= 0.5` 的高置信冻结阈值；
- 完整 rep、事件级诊断、拒答类型和覆盖率的分离统计；
- Fit3D 与 QEVD 的跨域置信度回归表。

Fit3D dev 正式结果：

| 指标 | 原始 combined | 校准后 | 正式门 |
|---|---:|---:|---:|
| 高置信 complete rep | `89` | `91` | - |
| 高置信 TP / FP | `89 / 0` | `91 / 0` | FP `= 0`，pass |
| 高置信覆盖率 | `0.864078` | `0.883495` | 诊断 |
| 高置信 TP 覆盖率 | `0.927083` | `0.947917` | 诊断 |
| precision | `1.000000` | `1.000000` | 诊断 |
| Brier | `0.109480` | `0.038866` | 诊断 |
| ECE（10 等宽箱） | `0.274748` | `0.074698` | 相对下降 `72.81%`，pass |

校准后仍有 `12` 个正式 complete rep 被拒答，其中 `5 TP / 7 FP`。也就是说，`7` 个正式 FP 全部保持低置信，没有被包装成高置信完整 rep。

## 2. 产物

```text
configs/task1_6_confidence_frozen.json
scripts/task1_6_confidence.py
tests/test_task1_6_confidence.py
docs/task1-6d_confidence_results.md

results/task1_6/dev/confidence/
├── confidence_calibration.csv
├── rep_confidence_audit.csv
├── rejection_audit.csv
├── qevd_confidence_regression.csv
├── confidence_eval_summary.json
└── checksums.sha256
```

复现：

```bash
cd /mnt/storage2/data_guest/VLS/V0
./.venv-mediapipe/bin/python scripts/task1_6_confidence.py
./.venv-mediapipe/bin/python -m unittest tests.test_task1_6_confidence -v
```

## 3. 正式评估单位

T1-6D 与 T1-6C 的统计单位不同，不能混用：

```text
正式单位 = 默认 set_candidate 中 status == complete 的 rep
诊断单位 = 所有 selected top event，包括 endpoint_partial 和链 anchor
```

默认组中另有 `50` 个 `endpoint_partial`。它们保留为候选和边界证据，但不进入“accepted complete rep”的高置信 FP 验收。非默认 set 的 `3` 个 rep 同样只做审计，不进入正式门。

T1-6C 的 `21` 个高置信 FP 是事件级诊断值：

- 来自 `188` 个 selected top event；
- 包含首尾 partial/anchor 语义；
- 不是 `103` 个正式 complete rep 的 FP 计数。

T1-6D 的正式计数是：

```text
accepted complete rep high-confidence FP
raw combined       = 0
calibrated combined = 0
```

不得把 T1-6C 的 `21` 直接写成 T1-6D 正式 rep FP，也不得用 `21` 否定本轮 complete rep 门。

## 4. 三层置信度

脚本直接消费 T1-6C 的 rep 输出，并审计三层原始分数：

| 层 | 含义 | Fit3D Brier | Fit3D ECE | 平均分 |
|---|---|---:|---:|---:|
| `complete` | 候选是否包含完整 top/bottom | `0.048423` | `0.040256` | `0.954083` |
| `in_set` | 是否属于连续正式链 | `0.096182` | `0.262772` | `0.680106` |
| `combined` | 下游放行综合分 | `0.109480` | `0.274748` | `0.658765` |

`complete` 的 ECE 较低只说明“状态完整度”本身较准，不代表候选一定属于正式动作组。`in_set` 和 `combined` 明显过度保守/失配，因此下游放行必须校准 `combined`，不能只看 `complete`。

## 5. 校准器

冻结配置：

```text
method = binned_beta_lower_bound
input_score = rep.confidence.combined
evaluation_unit = default_set_complete_rep
bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
lower_bound_alpha = 0.1
high_confidence_threshold = 0.5
```

每个分箱使用成功率的 Clopper-Pearson 单侧下界：

```text
TP / (TP + FP) 的 alpha=0.1 下界
再沿分数轴做 running max，保证校准分单调不降
```

Fit3D dev 拟合表：

| combined 区间 | n | TP | FP | 原始准确率 | Beta 下界 | 单调校准分 | 高置信 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0.0, 0.2)` | `4` | `1` | `3` | `0.250000` | `0.025996` | `0.025996` | no |
| `[0.2, 0.4)` | `8` | `4` | `4` | `0.500000` | `0.239662` | `0.239662` | no |
| `[0.4, 0.6)` | `10` | `10` | `0` | `1.000000` | `0.794328` | `0.794328` | yes |
| `[0.6, 0.8)` | `69` | `69` | `0` | `1.000000` | `0.967180` | `0.967180` | yes |
| `[0.8, 1.0]` | `12` | `12` | `0` | `1.000000` | `0.825404` | `0.967180` | yes |

最后一个箱的原始下界低于前一箱，说明高分区样本量较小；单调化后不允许校准分因箱样本数变化而下降。

校准器只用 Fit3D dev 的 `103` 个默认组 complete rep 拟合，其中 `96 TP / 7 FP`。脚本每次运行都会重新计算拟合表，并与冻结配置逐项比对；不一致则直接失败，避免 dev/holdout 之间静默改阈值。

## 6. Fit3D dev 结果

### 6.1 总体

| 项目 | 数值 |
|---|---:|
| clips | `32` |
| 有正式 complete rep 的 clips | `31` |
| clip 输出覆盖率 | `0.96875` |
| accepted reps（含 partial/非默认） | `156` |
| 正式 default complete reps | `103` |
| 默认组 endpoint partial | `50` |
| 非默认 set complete reps | `3` |

唯一没有正式 rep 的 clip 是 `fit3d_s05_pushup_50591643`，其原因为 `no_confident_set`。没有为了覆盖率硬补 rep。

### 6.2 高置信拒答

校准后阈值 `0.5` 下：

- `91` 个高置信 rep，全部为 TP；
- `12` 个低置信拒答，其中 `5 TP / 7 FP`；
- 所有 `7` 个正式 FP 都被拒答；
- 拒答没有造成任何高置信 FP。

这里的拒答是下游“不输出为高置信完整 rep”，不会生成“0 rep 即正常”的报告。

### 6.3 事件级诊断

如果回看 T1-6C 的所有 selected top event：

```text
event_count = 188
high_confidence_tp = 167
high_confidence_fp = 21
```

该统计保留在 `fit3d_dev.event_level_diagnostic` 中，用于说明锚点误检风险；正式 T1-6D 验收只使用 complete rep。两种口径必须同时报告、分开判定。

## 7. QEVD dev 跨域 smoke

QEVD dev 共 `13` 个 clips，但只有 `0161_pushup_set01` 有 `5` 个盲标 rep 区间：

| 指标 | 原始 combined | 校准后 |
|---|---:|---:|
| 可评估 rep | `5` | `5` |
| 匹配 GT 的 TP / FP | `5 / 0` | `5 / 0` |
| 高置信 accepted | `4` | `5` |
| 高置信 FP | `0` | `0` |
| Brier | `0.068236` | `0.009322` |
| ECE | `0.193630` | `0.067390` |
| rep 区间 IoU 匹配 | `5/5` | `5/5` |

应用冻结表后，原分数 `0.471599` 的第 3 个 rep 被提升到 `0.794328` 并达到高置信；五个 rep 均与盲标区间 IoU `>= 0.30`。

其余 `12` 个 QEVD dev clips 没有 rep 级 GT：

```text
unlabeled formal complete reps = 94
calibrated high-confidence = 90
```

这 `90` 个高置信输出的真假无法在 dev 上判定，不能把它们计作 FP，也不能计作已通过的 zero-FP 证据。

因此 QEVD 结论是：

```text
observed_high_confidence_fp = 0
annotated_clip_count = 1
formal_stage_gate_evaluable = false
qevd_dev_stage_passed = null
```

本轮只能声明跨域 smoke 未观察到高置信 FP，不能声明 QEVD dev 正式门通过。

## 8. 拒答审计

共审计 `20` 个链候选拒答：

| 类型 | 数量 | 证据规则 | 一致 |
|---|---:|---|---:|
| `shape_mismatch` | `16` | `eligible=false` | `16/16` |
| `no_confident_set` | `4` | `eligible=true`，未进入连续链 | `4/4` |
| `outside_set` | `0` | 当前实现未产生 | - |

`rejection_audit.csv` 逐条保留候选帧、拒答原因、相对幅度、是否在已接受 set 范围内和证据校验结果。`unsupported_count=0`。

`no_confident_set` 的语义是“候选幅度/形态合格，但没有进入任何连续链”，不能简单理解为“帧不在任何 set 区间内”；因此审计规则只看 `eligible=true`，不把 set 时间跨度作为必要条件。

## 9. 冻结与通用性

`configs/task1_6_confidence_frozen.json` 的 `core` 包含：

- 统一分箱、Beta 下界、单调化与 `0.5` 阈值；
- 正式评估单位和 endpoint partial 政策；
- ECE 分箱和最小相对改善要求；
- 无 dataset/video/subject/camera ID、固定 rep 数或固定帧偏移。

`provenance` 记录 sources、chain config、segmenter config 的 SHA-256、拟合样本数和拟合来源，但不进入算法特征。阈值在读取任何 sealed holdout GT 之前冻结。

## 10. 测试

`tests/test_task1_6_confidence.py` 共 `12` 项，覆盖：

- Beta 下界在零成功、全成功和小样本下的边界；
- 分箱校准的 running-max 单调化；
- 校准后 ECE 与 Brier 下降；
- `0.5` 阈值下的覆盖率和 FP 统计；
- 空输入安全；
- 只接受默认组 complete rep；
- `endpoint_partial` 和非默认 set 不进入正式门；
- rep anchor 与区间中点；
- `shape_mismatch` / `no_confident_set` 拒答证据；
- 冻结配置契约、禁用 token 和篡改检测；
- 校准分范围与输出字段。

全量测试结果：`139/139` 通过。

## 11. 下一步

进入 T1-6E 前仍需保留以下边界：

1. QEVD dev 只有 `1` 个盲标 clip，正式跨域门未评估；
2. T1-6C 事件级仍有 `21` 个高置信 FP，需在 T1-6E 的锚点/链集成决策中继续跟踪；
3. 当前冻结校准器只适用于 RepChain v2 的 `combined` 分数，不能用于 T1-4 旧分数；
4. sealed holdout 只能使用本配置，不得重新拟合或按结果改阈值。
