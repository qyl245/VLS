# task1-6E：跨数据集验收、消融与集成决策

## 1. 执行结论

状态：**T1-6E 已完成（2026-10-27）**，验收结论：**NO_GO**。

同一套冻结的 RepChain v2 配置已在 Fit3D holdout 和可运行的 QEVD sealed holdout 片段上执行。结论不是“指标略弱”，而是固定验收门存在多项明确失败：

- Fit3D holdout 的计数共识门通过，`8/8` 个 `subject-action` 完全计数正确；
- 首事件召回通过，但末事件召回只有 `0.53125`；
- 主相机事件误差中位/p90 为 `6/17` 帧，未达到 `5/12` 帧；
- 主相机/全机位事件级高置信 FP 为 `4/26`，complete rep 级高置信 FP 为 `2`；
- top-to-top 间隔误差中位 `5` 帧，相对 T1-5D 的 `3` 帧退化；
- QEVD sealed holdout 的 6 段 GT 仍全部未标注，7 个正式跨域门均不可评估，不能写成通过。

最终集成决策：

```text
decision = NO_GO
fit3d_holdout_stage_passed = false
qevd_sealed_formal_evaluable = false
qevd_sealed_stage_passed = null
integration = do_not_integrate_v2_into_T1-4_or_T1-7
```

RepChain v2 保持 experimental。不得回写 T1-4 冻结报告、`results/segmentation_final/`、T1-5 冻结评测或 T1-4 manifest。

## 2. 数据范围与隔离

### 2.1 Fit3D holdout

| 项目 | 数值 |
|---|---:|
| clips | `32` |
| `subject-action` clusters | `8` |
| GT top 事件观测 | `192` |
| 每个 cluster 的相机数 | `4` |

Fit3D 的 4 路相机只作为同一动作实例的重复观测。主计数指标使用每个 cluster 的 4 路计数中位数；单相机结果只作诊断。

Fit3D holdout 在 T1-5D 已被观察过聚合结果，因此本轮如实定义为**锁定评测**，不是“从未见过”的独立外部测试。T1-6 开发期没有根据 holdout 结果重新调参，但不能再宣称它是 pristine holdout。

### 2.2 QEVD sealed holdout

| 项目 | 数值 |
|---|---:|
| segments | `6` |
| pushup/squat 可运行片段 | `4` |
| 无冻结动作配置片段 | `2` |
| 已完成人工边界标注片段 | `0` |

`0152/0159` 为 pushup，`0127/0177` 为 squat，`0207` 的 `air_jump_rope` 和 `good_morning` 没有冻结动作契约，未运行。sealed 模板中的 `gt_start_frame/gt_end_frame` 仍为空，因此本轮只能检查预测和质量门冲突，不能计算 count/F1/boundary/high-confidence-FP 正式指标。

### 2.3 冻结配置

本轮所有数据集共用：

```text
configs/task1_6_segmenter.json
configs/task1_6_chain.json
configs/task1_6_confidence_frozen.json
```

核心算法不读取 dataset/video/subject/camera ID，不包含固定 rep 数、固定帧偏移或按数据源切换阈值。完整配置和输入 SHA-256 记录在 `results/task1_6/evaluation/summary.json`。

## 3. 产物与复现

```text
scripts/evaluate_task1_6.py
scripts/audit_task1_6_failures.py
tests/test_task1_6_evaluation.py

results/task1_6/evaluation/
├── summary.json
├── fit3d_holdout_clip_metrics.csv
├── fit3d_holdout_event_matches.csv
├── fit3d_holdout_rep_confidence.csv
├── ablation.csv
├── robustness.csv
├── robustness_summary.csv
├── negative_controls.csv
├── qevd_sealed_predictions.csv
├── qevd_review_index.csv
├── failure_index.csv
├── go_no_go.csv
├── audit_summary.json
└── checksums.sha256

results/task1_6/tables/
results/task1_6/figures/
results/task1_6/runs/holdout/
results/task1_6/baseline_t1_5d_matchfix/
```

复现：

```bash
cd /mnt/storage2/data_guest/VLS/V0
./.venv-mediapipe/bin/python scripts/evaluate_task1_6.py
./.venv-mediapipe/bin/python scripts/audit_task1_6_failures.py
./.venv-mediapipe/bin/python -m unittest tests.test_task1_6_evaluation -v
```

连续两次完整运行后，`summary.json` 中 9 张 CSV 的哈希一致：

```text
reproducibility.checked = true
reproducibility.identical_csv = true
reproducibility.scope = {skip_robustness: false, skip_sealed: false}
```

复现性记录现在包含命令范围。不同 `--skip-*` 组合会标记为 `scope_changed`，不会再被误判为算法输出不一致。

消融 A 使用 `results/task1_6/baseline_t1_5d_matchfix/`：它基于原 T1-5D 冻结预测，用修正后的门限内最大匹配重新评分。原 `results/task1_5/rep_evaluation/` 未被覆盖。修正后 T1-5D 全机位高置信 FP 为 `79`（旧值 `80`），计数和主相机头条指标不变。

## 4. Fit3D holdout 结果

以下为 tail anchor 优化前的 T1-6E 冻结结果；通用 tail anchor 的独立复评见第 13 节。

### 4.1 主结果

| 统计范围 | GT 事件 | 预测事件 | TP / FP / FN | recall | precision | first | last | 中位 / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全机位诊断 | `192` | `190` | `164 / 26 / 28` | `0.854167` | `0.863158` | `0.8125` | `0.53125` | `6 / 19` 帧 |
| 主相机 | `48` | `47` | `43 / 4 / 5` | `0.895833` | `0.914894` | `0.875` | `0.5` | `6 / 17` 帧 |

计数共识：

| 指标 | 结果 |
|---|---:|
| 预测 rep 总和 / GT rep 总和 | `48 / 48` |
| count MAE | `0.0` |
| 完全计数正确 | `8/8` |

计数改善是真实的：每个 `subject-action` cluster 的 4 路预测中位数都等于 GT。但事件定位、末事件覆盖和高置信拒答仍不达标，因此不能把计数全对解释为可交付。

### 4.2 固定门

`14/16` 的 exact-count 目标按预先比例折算到 8 个 holdout clusters，门槛为 `7/8`；不是看到结果后重设的阈值。

| 门 | 观测 | 目标 | 状态 |
|---|---:|---:|---:|
| cluster count MAE | `0.0` | `<= 0.5` | pass |
| exact count | `8/8` | `>= 7/8` | pass |
| first event recall | `0.8125` | `>= 0.75` | pass |
| last event recall | `0.53125` | `>= 0.75` | fail |
| primary event median | `6` 帧 | `<= 5` 帧 | fail |
| primary event p90 | `17` 帧 | `<= 12` 帧 | fail |
| primary high-confidence event FP | `4` | `= 0` | fail |
| all-camera high-confidence event FP | `26` | `= 0` | fail |
| complete rep high-confidence FP | `2` | `= 0` | fail |
| top-to-top interval median | `5` 帧 | `<= 3` 帧 | fail |

结果：Fit3D holdout 阶段门失败。

### 4.3 指标口径

三个高置信 FP 数字不能混写：

- `primary high-confidence event FP = 4`：8 路主相机 selected top event 诊断；
- `all-camera high-confidence event FP = 26`：32 个 clip 的事件级诊断；
- `complete rep high-confidence FP = 2`：默认 set 内 `status=complete` 的正式 rep 单位。

`outside_set_high_confidence_fp = 0` 只说明高置信误检事件在时间上落在某个 set 候选内，不代表它们是真 rep。

## 5. 置信度结果

Fit3D holdout 共有 `108` 个正式 complete rep，其中 `101 TP / 7 FP`。

| 指标 | 原始 combined | 校准后 combined |
|---|---:|---:|
| accepted complete reps | `93` | `98` |
| accepted TP / FP | `92 / 1` | `96 / 2` |
| high-confidence FP | `1` | `2` |
| precision | `0.989247` | `0.979592` |
| coverage | `0.861111` | `0.907407` |
| TP coverage | `0.910891` | `0.950495` |
| Brier | `0.114547` | `0.043686` |
| ECE | `0.287903` | `0.057450` |

T1-6D 的校准器显著降低了 ECE 和 Brier，但没有把 holdout 的高置信 FP 降到 0。开发集上“校准后 0 FP”没有跨 subject holdout 复现，这是本轮最重要的过拟合警告。

## 6. QEVD sealed holdout

4 个可运行片段的 v2 输出：

| segment | action | default complete rep | 高置信 complete rep | legacy quality | 备注 |
|---|---|---:|---:|---|---|
| `0152_pushup_set01` | pushup | `10` | `10` | pass | 待人工 GT |
| `0159_pushup_set01` | pushup | `7` | `7` | reject | **质量门冲突** |
| `0127_squat_set01` | squat | `8` | `8` | warn | 待人工 GT |
| `0177_squat_set01` | squat | `11` | `11` | warn | 待人工 GT |

`0159` 的 legacy 质量门为 reject，但 v2 仍输出了 7 个高置信 complete rep。该冲突已写入 `qevd_review_index.csv`，T1-7 输出前必须先执行质量门，不能让 reject 片段进入报告。

`0207` 两段不是 pushup/squat，且缺少冻结动作配置，不能临时借用其它动作配置求指标，应补正式契约或明确负样本预期。

sealed 正式门结果：

```text
annotation_ready_count = 0 / 6
runnable_segment_count = 4 / 6
formal_stage_gate_evaluable = false
stage_passed = null
```

因此 7.3 的 count exact、count MAE、边界中位/p95、F1@IoU0.30、高置信 FP 和非 rep 完整输出全部为 `not_evaluable`。

## 7. 消融

A–E 在同口径 Fit3D holdout 上比较；F 的 anchor 消融来自 T1-6B 的 **dev state-only**，不能当作 holdout 结果。

本节 D/E 仍是 tail anchor 优化前的数值；优化后的 D/E 见 13.2。

| variant | 范围 | recall | precision | first | last | 中位 / p90 | count MAE | exact | 高置信 FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A legacy T1-5D | holdout | `0.541667` | `0.658228` | `0.3125` | `0.125` | `15 / 27` | `0.5` | `4/8` | `45` event |
| B states only | holdout | `0.880208` | `0.793427` | `0.875` | `0.59375` | `6 / 20` | `0.75` | `3/8` | n/a |
| C chain no endpoint | holdout | `0.526042` | `0.935185` | `0.03125` | `0.03125` | `4 / 12` | `2.625` | `0/8` | n/a |
| D chain full raw | holdout | `0.854167` | `0.863158` | `0.8125` | `0.53125` | `6 / 19` | `0.0` | `8/8` | `26` event |
| E full calibrated | holdout | 同 D | 同 D | 同 D | 同 D | 同 D | `0.0` | `8/8` | `2` complete rep |
| F anchor entry | dev state-only | `0.588542` | `0.548544` | `0.1875` | `0.75` | `15 / 27.8` | - | - | n/a |
| F anchor midpoint | dev state-only | `0.885417` | `0.825243` | `0.8125` | `0.78125` | `6 / 22` | - | - | n/a |
| F anchor exit | dev state-only | `0.286458` | `0.26699` | `0.34375` | `0.0625` | `12 / 25` | - | - | n/a |
| F anchor zero-crossing | dev state-only | `0.822917` | `0.76699` | `0.875` | `0.9375` | `15 / 27.3` | - | - | n/a |

解释：

- B 的 state-only 事件 recall 较高，但 count MAE 和误检明显更差；
- C 去掉 endpoint partial 后首尾召回接近 0，证明首尾部分周期是覆盖关键，但它们不是正式 complete rep；
- D 把 count MAE 降到 0、exact 提到 8/8，同时保留较好的首事件覆盖，但末事件和高置信 FP 仍失败；
- E 的事件指标与 D 相同，高置信 FP 改为 complete rep 口径；不能把 E 的 `2` 与 D 的 `26` 当成同一单位的改善；
- F 的 midpoint 在 dev 上综合最好，但 p90 仍为 `22` 帧；zero-crossing 末召回更高，却把中位误差扩大到 `15` 帧。

## 8. 稳健性

9 种冻结变换，每种覆盖 32 个 holdout clips：

| 变换 | 结构完全一致 | 平均 rep 数变化 |
|---|---:|---:|
| amplitude 0.8 | `31/32 = 0.96875` | `-0.09375` |
| amplitude 1.2 | `31/32 = 0.96875` | `0.03125` |
| flip L/R | `32/32 = 1.0` | `0` |
| FPS 15 | `27/32 = 0.84375` | `0` |
| FPS 60 | `28/32 = 0.875` | `-0.28125` |
| idle pad 2s | `31/32 = 0.96875` | `0.03125` |
| missing 5% | `31/32 = 0.96875` | `-0.03125` |
| noise sigma 2 | `28/32 = 0.875` | `-0.0625` |
| offset +15 | `32/32 = 1.0` | `0` |

幅度、offset、左右翻转、缺帧和 idle padding 基本稳定；15/60 FPS 与角度噪声仍会改变 4–5 个 clip 的组结构，不满足严格的 rep 数量/组结构不变性。

## 9. 负样本

| 负样本 | 状态 | set / rep / complete | 结果 |
|---|---|---:|---:|
| `idle_constant` | no_motion（T1-6G） | `0 / 0 / 0` | pass |
| `setup_ramp` | no_confident_set | `0 / 0 / 0` | pass |
| `single_cycle` | no_confident_set | `0 / 0 / 0` | pass |
| `half_rep` | no_confident_set | `0 / 0 / 0` | pass |
| `two_short_cycles` | no_confident_set | `0 / 0 / 0` | pass |

合计 `5/5`。T1-6E 原结果中静立常量序列被误判为正式动作组并产生 2 个 complete rep；T1-6G 增加通用运动证据门后，该序列返回 `no_motion`，不再产生 set/rep，详见第 14 节。

## 10. 失败样本审计

Fit3D holdout：

```text
failed clips = 21 / 32
high-priority clips = 19
```

失败类别出现次数：

| 类别 | clip 数 |
|---|---:|
| last event missed | `15` |
| event high-confidence FP | `17` |
| first event missed | `6` |
| count error | `6` |
| p90 exceeds gate | `6` |
| median exceeds gate | `5` |
| complete-rep high-confidence FP | `2` |

分层：

- pushup：`9/16` clips 失败，事件高置信 FP `14`，rep 高置信 FP `2`；
- squat：`12/16` clips 失败，事件高置信 FP `12`，rep 高置信 FP `0`；
- 末事件缺失是最普遍的失败类别，占 15 个 clip；
- 最高优先级样本包括 `fit3d_s08_pushup_50591643`、`fit3d_s10_pushup_50591643` 和 `fit3d_s11_pushup_50591643`。

`failure_index.csv` 给出每个 clip 的严重度、主要失败类别和证据帧，可直接作为下一轮修复的回归集。

## 11. 集成决策

### 11.1 不接入

本轮不将 RepChain v2 接入 T1-4/T1-7：

- Fit3D holdout 有 7 个固定门失败；
- QEVD sealed 没有可评估 GT；
- 负样本静立误检已在 T1-6G 修复（`5/5` 通过）；
- 15/60 FPS 与噪声的结构不变性未通过；
- 校准器在 holdout 上仍产生 2 个高置信 complete rep FP。

### 11.2 下一轮优先级

1. **先修质量门和拒答**（T1-6G 已完成）：T1-7 输出前必须拒绝 legacy `reject` 片段；`0159` 已不再输出可报告 complete rep。
2. **修链尾与首尾 anchor 覆盖**：last recall 只有 `0.53125`，是 15 个 clip 的共同失败。
3. **单独处理事件级高置信 FP**：complete rep 校准不能代替 selected event 的误检抑制。
4. **增加静止/低变化负样本路径**（T1-6G 已完成）：`idle_constant` 已由 `1 set / 2 complete rep` 修到 `0 / 0`。
5. **完成 QEVD sealed 人工标注**：6 段全部完成后才能计算 7.3；同时为 `0207` 两动作补冻结契约或明确零 rep 预期。
6. **重做跨采样率与噪声鲁棒性**：优先定位 15/60 FPS 下结构改变的 4–5 个 clip。

## 12. 本轮实现修正

- 复现性比较加入运行 scope，避免 `--skip-sealed` 与完整运行被误判为算法不确定；
- QEVD sealed 标注只接受 `accept/adjust/add`，pending/delete 行不会进入 GT；
- 门控 SVG 使用 PASS/FAIL/NOT EVALUABLE 状态位置和文字，不再把所有柱画成 1.0；
- 失败证据允许合法帧号 `0`，不再把 `0` 当成空值；
- 修复消融 A 的事件误差键名，legacy baseline 现在正确输出中位/p90；
- 审计脚本校验整个 `results/task1_6/evaluation/` 目录，不再覆盖并遗漏评测主表。
- 修正 `match_events` 的匹配门实现：先按门限禁用超距边，再最大化门限内一对一匹配数，最后最小化误差；不再先做强制完整分配后丢弃。
- T1-5D 的重复匹配实现同步修正，并用独立目录重算消融基线；原 T1-5D 冻结目录保持不变。

定向测试 `43` 项通过；全量测试 `162/162` 通过。

## 13. 通用 tail anchor 优化（T1-6F follow-up）

状态：**已实现并独立复评（2026-10-27）**。本轮只改链路末事件的输出锚点，不改链选择、rep 边界、计数或置信度。

### 13.1 规则

- 链图和中间事件继续使用 `plateau_midpoint`；每条 accepted chain 的最后一个 selected top event 使用 `min(plateau_midpoint, phase_zero_crossing)`。
- 规则只依赖 state 的 anchor 和 terminal 位置，不使用 dataset/video/subject/camera ID、固定帧偏移或按动作切换阈值。
- 配置写入 `configs/task1_6_chain.json` 的 `core.tail_anchor`；实现位于 `scripts/task1_6_chain_selection.py`。
- `selected_top_events` 现在记录 `graph_frame`、`is_terminal` 和实际 `anchor_policy`，便于审计“链图帧”和“输出帧”的差异。

依据：末端 top plateau 常包含结束保持，midpoint 会偏晚；Fit3D 与 QEVD sealed 的链尾均以 `trailing_partial` 为主，midpoint 相对 zero-crossing 的中位偏移分别约为 `0.35` 和 `0.30` 个周期。QEVD sealed 尚未有 GT，因此这只是结构证据，不是正式跨域指标验证。

### 13.2 独立复评

原 `results/task1_6/evaluation/`、`tables/`、`figures/` 和 `runs/holdout/` 未覆盖。本轮产物在：

```text
results/task1_6/evaluation_tail_anchor/
results/task1_6/tables_tail_anchor/
results/task1_6/figures_tail_anchor/
results/task1_6/runs/tail_anchor/
```

Fit3D holdout 同口径对比：

| 指标 | 旧 T1-6E | tail anchor | 变化 |
|---|---:|---:|---:|
| 全机位 event recall | `0.854167` | `0.911458` | `+0.057291` |
| 全机位 event precision | `0.863158` | `0.921053` | `+0.057895` |
| last event recall | `0.53125` | `0.90625` | `+0.375` |
| 主相机 event recall | `0.895833` | `0.979167` | `+0.083334` |
| 主相机 event precision | `0.914894` | `1.0` | `+0.085106` |
| 主相机高置信事件 FP | `4` | `0` | `-4` |
| 全机位高置信事件 FP | `26` | `15` | `-11` |
| 事件误差中位 / p90 | `6 / 19` | `6 / 18` | p90 `-1` |
| 失败 clips | `21` | `12` | `-9` |
| high-priority clips | `19` | `6` | `-13` |
| last-event miss clips | `15` | `3` | `-12` |
| complete rep 高置信 FP | `2` | `2` | 不变 |
| count MAE | `0.0` | `0.0` | 不变 |

末事件仍失败的 3 个 clip 都是 pushup 链尾选择问题：`fit3d_s09_pushup_50591643`、`fit3d_s10_pushup_50591643`、`fit3d_s11_pushup_58860488`。它们需要修尾节点入选/抑制，不应继续调 anchor。

### 13.3 决策

`decision = NO_GO` 保持不变，仍失败的门是：主相机事件中位/p90、全机位高置信事件 FP、complete rep 高置信 FP、top-to-top 间隔。QEVD sealed 仍未完成 GT，正式跨域门仍为 `not_evaluable`。

复现性：连续两次完整运行后 9 张 CSV hash 一致；`summary.json` 记录 `identical_csv=true`，`checksums.sha256` 全部通过。相关测试 `55` 项通过；排除受现有 TensorFlow/protobuf 环境影响的 T1-4 两个模块后，全量 `141 passed + 2 subtests`。

## 14. 质量门、idle 拒答与 raw/gated 双口径（T1-6G）

状态：**已实现并独立复评（2026-10-27）**；`decision = NO_GO` 不变。本轮按“质量门接线 → idle 运动证据 → raw/gated 评测”的顺序落地，`warn` 不硬阻断。

### 14.1 统一质量门契约

新增 `scripts/task1_6_quality.py`，质量门统一为：

```text
status: pass | warn | reject | unknown
scope: clip | set | rep
reasons: [...]
source: pose | signal | action_contract
blocking: bool
```

- `pass`：正常输出；
- `warn`：保留 set/rep/event，并在输出中附质量问题；
- `reject`：clip 级硬短路，`sets=[]`、`selected_top_events=[]`、complete rep 为 `0`，只保留门控前结构与原因诊断；
- `unknown`：产品输出 fail closed，评测标记为不可评估；
- rep 级门只删除对应 rep，set 级门只删除对应 set。

链分析入口 `analyze_clip_signals` 已接收质量门，门控发生在链分析和置信度/报告之间；`from_legacy_quality` 负责把 T1-4 的 `pass/warn/reject` 映射到统一契约。

### 14.2 idle 运动证据与拒答

在 `scripts/task1_6_phase_states.py` 增加通用运动证据，不使用数据集或样本 ID：

```text
dynamic_range = p95(smooth) - p5(smooth)
noise_sigma = 1.4826 * MAD(diff(smooth)) / sqrt(2)
dynamic_to_noise = dynamic_range / noise_sigma
state_amplitude = median(|top.mean_angle - nearest_bottom.mean_angle|)
```

门限为 `dynamic_range >= 5°`、`dynamic_to_noise >= 8`、`state_amplitude >= 3°`，写入 `configs/task1_6_segmenter.json` 的 detector。任一失败返回 `status=no_motion`，零 set / 零 rep，并保留 `motion_evidence` 诊断；产品语义是“未检测到有效动作”，不是“画质不合格”。

负样本结果：

| 控制序列 | 状态 | set / rep / complete | 结果 |
|---|---|---:|---|
| `idle_constant` | no_motion | `0 / 0 / 0` | pass |
| `setup_ramp` | no_confident_set | `0 / 0 / 0` | pass |
| `single_cycle` | no_confident_set | `0 / 0 / 0` | pass |
| `half_rep` | no_confident_set | `0 / 0 / 0` | pass |
| `two_short_cycles` | no_confident_set | `0 / 0 / 0` | pass |

合计 `5/5`。

### 14.3 Gated 鲁棒性

鲁棒性统一在质量门后的 eligible 口径统计：输入 `32` 个 clip，其中 `fit3d_s09_pushup_50591643` 被质量门排除，eligible 为 `31` 个。下表不再与旧 raw 口径的 `32/32` 表混用。

| 变换 | 输入 / eligible | 结构一致 | 一致率 | 改变的 clip |
|---|---:|---:|---:|---:|
| amplitude ×0.8 | `32 / 31` | `30/31` | `0.967742` | `1` |
| amplitude ×1.2 | `32 / 31` | `31/31` | `1.0` | `0` |
| flip L/R | `32 / 31` | `31/31` | `1.0` | `0` |
| FPS 15 | `32 / 31` | `26/31` | `0.83871` | `5` |
| FPS 60 | `32 / 31` | `28/31` | `0.903226` | `3` |
| idle pad 2s | `32 / 31` | `31/31` | `1.0` | `0` |
| missing 5% | `32 / 31` | `30/31` | `0.967742` | `1` |
| noise σ=2 | `32 / 31` | `28/31` | `0.903226` | `3` |
| offset +15 | `32 / 31` | `31/31` | `1.0` | `0` |

`idle_pad_2s` 在 eligible 集上为 `31/31`，另有 `1` 个输入被质量门排除，说明真实动作前后补静立段不会被运动门误拒。当前主要鲁棒性缺口仍是 FPS 15/60、噪声和幅度缩小。

### 14.4 Fit3D raw / gated 双口径

质量分布：`pass 17 / warn 14 / reject 1`。唯一被排除的是 `fit3d_s09_pushup_50591643`；14 个 `warn` 全部保留输出，没有硬阻断。

| 指标 | raw | gated |
|---|---:|---:|
| reportable clip coverage | `32/32 = 1.0` | `31/32 = 0.96875` |
| cluster coverage | `8/8 = 1.0` | `8/8 = 1.0` |
| 全机位 event recall | `0.911458` | `0.901042` |
| 全机位 event precision | `0.921053` | `0.935135` |
| 全机位高置信事件 FP | `15` | `12` |
| last event recall | `0.90625` | `0.90625` |
| 主相机 recall / precision | `0.979167 / 1.0` | `0.979167 / 1.0` |
| count consensus MAE / exact | `0.0 / 8` | `0.0 / 8` |
| complete rep 高置信 FP | `2` | `2` |

被排除 clip 的 raw 结构为 `1 set / 4 reps / 3 default complete reps`，gated 全部为 `0`。Fit3D 的 warn 没有改变计数或高置信 rep FP，说明当前门控只修复 reject 泄漏，而不是靠剔除大量样本改善指标。

### 14.5 QEVD sealed

| clip | raw | gated | 质量动作 |
|---|---|---|---|
| `0152_pushup_set01` | `1 set / 12 reps / 10 high-conf complete` | 同 raw | pass，保留 |
| `0159_pushup_set01` | `1 set / 9 reps / 7 high-conf complete` | `0 / 0 / 0` | reject → blocked |
| `0127_squat_set01` | `1 / 10 / 8` | 同 raw | warn，保留 |
| `0177_squat_set01` | `1 / 13 / 11` | 同 raw | warn，保留 |

QEVD 可报告覆盖率：`3/4` 可运行段（`0.75`）、`3/6` 全部段（`0.5`）。`quality_gate_conflict_count` 由 raw 的 `1` 降为 gated 的 `0`。sealed GT 仍为 `0/6`，正式跨域门仍为 `not_evaluable`。

### 14.6 评测与产物

- `fit3d_holdout_clip_metrics.csv` 新增 raw/gated set、rep、complete rep、事件计数、质量动作、排除原因、motion status 和 coverage 相关字段；
- `qevd_sealed_predictions.csv` 新增 raw/gated 计数、`quality_action`、`excluded_by_quality`、`excluded_by_motion`、`raw_quality_gate_conflict`；
- `summary.json` 的 `fit3d_holdout.raw` 与 `.gated` 并列记录指标和 coverage，新增 `fit3d_gated_clip_coverage >= 0.95` 披露门；
- 独立产物未覆盖旧冻结目录：

```text
results/task1_6/evaluation_quality_motion/
results/task1_6/tables_quality_motion/
results/task1_6/figures_quality_motion/
results/task1_6/runs/quality_motion/
```

### 14.7 决策与复验

本轮修复了质量门泄漏、idle 误报和排除口径不透明，但剩余固定门仍失败：主相机事件中位/p90、全机位高置信事件 FP、complete rep 高置信 FP、top-to-top 间隔。因而继续保持 `NO_GO`，不接入 T1-4/T1-7。

复现性：连续两次完整运行 `identical_csv=true`，11 张评测 CSV hash 一致，`checksums.sha256` 全部通过。排除受现有 TensorFlow/protobuf 环境影响的 T1-4 两个模块后，全量测试为 `151 passed + 4 subtests`。
