# task1-3 P1-A / P2-A：主角度证据、尾部收口与鲁棒性

日期：2026-10-25

Canonical final snapshot: `VLS/V0/results/segmentation_final/`（主结果见 `VLS/V0/docs/task1-3_results.md`）。

## 1. 结论

本轮把 P1-A 和 P2-A 都实现为可复现实验，并保留独立结果目录。P2-A 已完成默认收口，新增统一 golden、平滑消融和噪声鲁棒性。

- **P1-A：证据完备，但未达到替换门槛**。新增动作相关周期范围、周期性、主峰显著度、周期内相位一致性、边界稳定性和周期簇融合；全量输出 136 条候选角度证据。当前数据上 19/22 段被判为 `angle_ambiguous` 并回退 legacy top-1，只有 3 段由簇选择，其中唯一实际换角度的是零 rep 的 `0218/toe_touchers`。5 段盲标和全量 rep 边界与 P1-D 完全一致，但没有计数或边界收益，因此默认仍使用 `legacy`。
- **P2-A：已收口并默认启用**。`0218/air_jump_rope` 的最后 rep `1663-1679` 在“尾部位置 + 相位 apex 缺失 + 低 motion evidence”三项证据同时成立时被删除，`0173/air_jump_rope` 保持 24 rep，其余 21 段逐边界不变。IoU=0.30 从 `67/66/1/0` 变为 `66/66/0/0`，micro F1 从 `0.99248` 升到 `1.0`。新基线为 `results/segmentation_p2a/`，AIFit `365`、Peak `341`，由 `configs/p2a_golden.json` 约束。
- **鲁棒性：默认平滑窗口不替换**。无平滑、`5/2` 和 `11/2` 都造成计数或盲标退化；`7/2` 保持 `365` 和 F1 `1.0`。`9/2` 在同一批 5 段盲标上略降 p95，但同时改变 33 个 segment-method 边界，属于事后候选，不作为默认。角度噪声在 `1–2°` 内可维持接近基线，`5°` 起明显退化，`10°` 接近失效。

默认值为 `--angle-scoring legacy --p2a-tail-filter guarded --smooth-window 7 --smooth-polyorder 2`。旧阶段回归脚本显式传 `--p2a-tail-filter off`，避免历史 golden 被静默改写；P2-A 新基线单独用 `check_p2a_regression.py` 验证。

## 2. P1-A 实现

`VLS/V0/scripts/segment_reps.py` 新增：

- `ACTION_PERIOD_RANGES_S`：按动作设置周期搜索范围，并与 CLI 的全局范围取交集。
- 硬门槛：`valid_fraction >= 0.50`、`dynamic_range >= 5.0`、必须有可估计周期。
- 四项评分：
  - `periodicity`：自相关主峰强度；
  - `peak_prominence`：归一化信号的重复峰显著度；
  - `phase_consistency`：最优相位下相邻周期的平均相关；
  - `boundary_stability`：Stage A 周期边界重采样后的形状一致性。
- 固定权重：周期性 `0.30`、峰显著度 `0.20`、相位一致性 `0.25`、边界稳定性 `0.25`。
- 晚期融合：先按周期相对差 `<= 15%` 聚类，再按簇内最佳候选分数、簇周期一致性和角度数选择；不平均原始角度信号。
- 歧义：候选或簇的分数差距小于 `0.03` 时标 `angle_ambiguous=true`，并回退 legacy top-1，不强行切换。
- 输出：`run.json.angle_selection` 和 `angle_evidence.csv`，包含候选的周期、周期性、簇周期一致性、相位一致性、动态范围、边界稳定性、分数、簇号和歧义边距。

CLI：

```bash
--angle-scoring {legacy,evidence}
```

## 3. P1-A 结果

全量结果目录：`VLS/V0/results/segmentation_p1a_p2a/full_evidence/`。

| metric | P1-D | P1-A evidence |
|---|---:|---:|
| AIFit reps | 366 | 366 |
| 5 段 GT/Pred/TP/FP/FN | 66/67/66/1/0 | 66/67/66/1/0 |
| IoU=0.30 micro F1 | 0.99248 | 0.99248 |
| raw IoU | 0.68514 | 0.68514 |
| mean abs start/end | 5.18 / 4.80 | 5.18 / 4.80 |
| p95 boundary | 12.00 | 12.00 |
| IoU=0.50 F1 | 0.82707 | 0.82707 |
| walking-lunge stride | 12/12 | 12/12 |

角度选择：

- 22 段共输出 136 条候选。
- 19 段歧义，均回退 legacy；3 段由周期簇选择。
- 仅 `0218/toe_touchers` 从 `right_hip_flexion_sagittal` 切到 `torso_flexion`，该段仍为 0 rep，不改变任何指标。
- `lunge_jumps=6`、`0218/air_jump_rope=40`、两段 walking lunges 各 6 均保持。

按验收口径，P1-A **未通过“同时改善计数和边界”的替换条件**，因此不启用。它现在提供的是可审计的候选证据，不是新的默认选择器。

## 4. P2-A 实现

新增 `reject_air_jump_rope_transition_tail()`。只有同时满足以下条件才删除最后一个 rep：

- 动作严格为 `air_jump_rope`；
- 最后一个 rep 距 fit end 不超过 `0.35 × period`；
- 至少两项独立证据成立：
  - rep 内没有相位同极性 apex；
  - `shape_similarity < max(0.25, 0.65 × 同段其余 rep 中位数)`；
  - `motion_evidence < min(0.05, 0.15 × 同段其余 rep 中位数)`。

删除后重新编号 rep、更新周期，并写入 `rejected_reps.csv` 和 `run.json` 的拒因证据。被删区间同时补为 `transition` non-rep interval。规则不使用视频 ID 之外的帧号特判。

CLI：

```bash
--p2a-tail-filter {off,guarded}
```

默认已为 `guarded`；`off` 仅用于复现历史冻结阶段。

## 5. P2-A 结果

全量结果目录：`VLS/V0/results/segmentation_p2a/`。统一 golden：`VLS/V0/configs/p2a_golden.json`。

`0218/air_jump_rope` rep 40 的证据：

| field | value |
|---|---:|
| frames | `1663-1679` |
| fit end / tail gap | `1684` / `5` frames |
| period | `17` frames |
| phase apex in rep | none |
| shape similarity / limit | `0.3200` / `0.4297` |
| motion evidence / limit | `0.00103` / `0.05` |
| support count | `3` |

关键非回归：

- `0173/air_jump_rope`：`24 -> 24`，未删除最后 rep。
- `0161/lunge_jumps`：`6 -> 6`。
- 两段 walking lunges：各 `6`，合并 stride `12/12`。
- 除 `0218/air_jump_rope` 删除参考集最后一项外，其余 21 段边界完全一致。
- AIFit 全量：`366 -> 365`；air jump rope `64 -> 63`。

| metric | P1-D | P2-A |
|---|---:|---:|
| IoU=0.30 Pred/TP/FP/FN | 67/66/1/0 | **66/66/0/0** |
| IoU=0.30 micro F1 | 0.99248 | **1.00000** |
| raw micro IoU | 0.68514 | 0.68514 |
| mean abs start/end | 5.18 / 4.80 | 5.18 / 4.80 |
| p95 boundary | 12.00 | 12.00 |
| IoU=0.50 Pred/FP/FN | 67/12/11 | 66/11/11 |
| IoU=0.50 micro F1 | 0.82707 | 0.83333 |

组合运行 `--angle-scoring evidence --p2a-tail-filter guarded` 与只开 P2-A 的所有 rep 边界一致，说明两个开关没有耦合副作用。

默认收口验证：

- `run.json.parameters` 为 `legacy`、`guarded`、`7/2`。
- AIFit `365`、Peak `341`；22 段 rep 数与 golden 一致。
- `0218/air_jump_rope=39` 且只删除参考集最后一项；`0173/air_jump_rope=24`。
- 5 段盲标 IoU=0.30 为 `66/66/0/0`，micro F1 `1.0`；walking-lunge stride `12/12`。

## 6. 平滑消融与噪声鲁棒性

运行入口：

```bash
.venv/bin/python VLS/V0/scripts/run_p2a_robustness.py
```

输出目录：`VLS/V0/results/segmentation_p2a_robustness/`。

- `smoothing_ablation.csv/json`：6 个 Savitzky-Golay 配置。
- `noise_robustness.csv/json`：角度噪声和姿态相位噪声。
- `robustness_summary.json`：按 sigma 聚合的 mean/std/min/max。
- 每个配置保留独立的 `reps.csv`、`run.json`、盲标评估和 walking-lunge 评估。

### 6.1 平滑消融

所有配置均使用 `guarded`、legacy 主角度和固定 seed，比较对象为 `segmentation_p2a`。

| 配置 | AIFit | Peak | 0218 | 0173 | P2-A | IoU=0.30 F1 | raw IoU | p95 边界 | IoU=0.50 F1 | 改变边界数 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 无平滑 | 363 | 326 | 39 | 24 | pass | 0.99237 | 0.68282 | 12.00 | 0.83969 | 38 |
| 5/2 | 362 | 340 | 39 | 24 | pass | 0.97674 | 0.68404 | 11.00 | 0.82171 | 33 |
| **7/2（默认）** | **365** | **341** | **39** | **24** | **pass** | **1.00000** | **0.68514** | **12.00** | **0.83333** | **0** |
| 9/2 | 365 | 324 | 39 | 24 | pass | 1.00000 | 0.68843 | 11.45 | 0.89394 | 33 |
| 11/2 | 342 | 317 | 12 | 25 | fail | 0.68571 | 0.72962 | 31.35 | 0.51429 | 38 |
| 7/3 | 365 | 341 | 39 | 24 | pass | 1.00000 | 0.68514 | 12.00 | 0.83333 | 2 |

结论：

- `7/2` 是唯一在不改变任何 segment-method 边界的前提下保持全部主指标的配置。
- `11/2` 把 `0218/air_jump_rope` 压到 12 rep，并显著恶化盲标，明确淘汰。
- `9/2` 在同批盲标上 p95 和 IoU=0.50 略好，但改变 33 个 segment-method 边界，且是在验收集上事后观察到的候选；没有独立留出收益，不替换默认。
- `7/3` 与 `7/2` 主指标相同，但多出 2 个边界变化，没有收益。

### 6.2 角度噪声

对 25 个 `deg` 角度列逐帧加入 i.i.d. Gaussian 噪声，保留 NaN 和时间列，裁剪到 `[0, 180]`；每个 sigma 跑 3 个固定 seed trial，sigma=0 只跑 1 次作为确定性基线。因为没有配对 3D 关节角 GT，这不是 MediaPipe 真实误差标定，而是敏感性范围；`5–10°` 可作为保守上界。

这里 `0218 计数正确率` 只要求 `air_jump_rope=39`；`P2-A 触发成功率` 还要求存在尾部拒绝记录，且拒绝区间与默认基线 `1663-1679` 的 IoU `>= 0.80`。两者分开报告，避免把“本来就没有第 40 个 rep”误判成规则成功。

| sigma | AIFit mean±sd | IoU=0.30 F1 | raw IoU | p95 边界 | IoU=0.50 F1 | 0218 计数正确率 | P2-A 触发成功率 | walking stride F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0° | 365.0 ± 0.0 | 1.00000 | 0.68514 | 12.00 | 0.83333 | 100% | 100% | 1.00000 |
| 1° | 366.7 ± 2.49 | 0.98735 | 0.66749 | 15.15 | 0.80480 | 100% | 66.7% | 1.00000 |
| 2° | 361.0 ± 1.63 | 0.97189 | 0.69001 | 11.38 | 0.80303 | 100% | 100% | 1.00000 |
| 5° | 361.3 ± 12.97 | 0.82748 | 0.66560 | 21.75 | 0.65800 | 0% | 0% | 1.00000 |
| 10° | 385.0 ± 17.80 | 0.55891 | 0.58412 | 28.60 | 0.32352 | 0% | 0% | 0.96970 |

解释：

- `1–2°` 下总 rep 数和主 F1 仍接近基线，边界 p95 的波动说明短周期尾部对噪声敏感。
- `1°` 的 P2-A 触发率是 `2/3`：有一个 trial 没有生成第 40 个尾部 rep，最终计数仍是 39；因此同时报告 `0218 计数正确率` 和 `P2-A 触发成功率`，不能把“规则没触发”误写成“计数错误”。
- `5°` 起 `0218` 不再是单尾部问题，计数和边界同时破坏；`10°` 已不能作为可用输入。失败来自候选信号和边界本身改变，不应通过放宽 P2-A 阈值掩盖。
- walking-lunge stride 在 `0–5°` 保持 `1.0`，`10°` 降到 `0.9697`，说明该动作对角度噪声更稳。

### 6.3 姿态相位噪声

对 `0173/0218` 的 nose world 坐标加入 Gaussian 噪声，测试 `air_jump_rope` 的 landmark 相位信号；其他 landmark 和全部角度输入保持不变。因此这一组只验证 P2-A 的相位输入鲁棒性，不等价于完整的姿态噪声实验。

| sigma (m) | 0218 计数正确率 | P2-A 触发成功率 | 盲标 F1 | 0173 变化 |
|---:|---:|---:|---:|---|
| 0.002 | 100% | 100% | 1.00000 | 无 |
| 0.005 | 100% | 100% | 1.00000 | 1/3 trial 变为 25 rep |
| 0.010 | 100% | 100% | 1.00000 | 无 |
| 0.020 | 100% | 100% | 1.00000 | 无 |

在本次 seed 和 nose-only 扰动范围内，P2-A 对相位噪声稳定；但 `0.005 m` 的单个 trial 已使 `0173` 增加 1 rep，说明非目标段仍可能被相位抖动改变。结果不是单调曲线，不能外推成一般姿态误差鲁棒性。

### 6.4 鲁棒性限制

- 噪声水平不是由成对 GT 标定得到；绝对值只能用于比较趋势，不能写成 MediaPipe 的实际角度误差。
- 姿态噪声只扰动 nose 相位信号，没有重新计算角度；完整 pose-to-angle 误差需要 Fit3D 或成对 3D GT。
- 平滑和噪声评估仍使用同一批 5 段 QEVD-FIT 内部盲标，`9/2` 的轻微收益不能作为独立验证。
- P2-A 仍只有一个确认尾部 transition 样本；鲁棒性实验不能替代更多动作/受试者样本。

## 7. 回归与产物

```bash
.venv/bin/python -m pytest -q VLS/V0/tests/test_segment_reps.py VLS/V0/tests/test_p2a_robustness.py
# 25 passed

.venv/bin/python VLS/V0/scripts/check_p1a_angle_evidence.py \
  --segmentation-dir VLS/V0/results/segmentation_p1a_p2a/full_evidence \
  --report VLS/V0/results/segmentation_p1a_p2a/full_evidence/p1a_regression.json
# pass; 22 segments, 136 candidates, 19 ambiguous

.venv/bin/python VLS/V0/scripts/check_p2a_regression.py \
  --report VLS/V0/results/segmentation_p2a/p2a_regression.json
# pass; 365 AIFit / 341 Peak; no differences

.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode guard
.venv/bin/python VLS/V0/scripts/check_p0c_regression.py --mode exact
.venv/bin/python VLS/V0/scripts/check_p0d1_regression.py
.venv/bin/python VLS/V0/scripts/check_p0d2_regression.py
.venv/bin/python VLS/V0/scripts/check_p1d_regression.py
.venv/bin/python VLS/V0/scripts/check_p1b1_regression.py
# all pass; old stages explicitly disable P2-A

.venv/bin/python VLS/V0/scripts/run_p2a_robustness.py
# 6 smoothing configs + 13 angle-noise runs + 12 landmark-noise runs
```

P1-A 还通过了 P1-D、P0-D2、P1-B1 和 P0-C guard 回归。P2-A 会按设计触发旧 exact 脚本的 `40 -> 39` 差异，因此使用专用 `check_p2a_regression.py` 验证默认参数、22 段计数、盲标指标和“只删目标尾部、其他边界不变”。P0-D1 回归也已显式关闭 P2-A，避免默认切换污染历史 golden。

新增/修改：

```text
VLS/V0/scripts/segment_reps.py
VLS/V0/scripts/check_p1a_angle_evidence.py
VLS/V0/scripts/check_p2a_regression.py
VLS/V0/scripts/check_p0c_regression.py
VLS/V0/scripts/check_p0d1_regression.py
VLS/V0/scripts/check_p0d2_regression.py
VLS/V0/scripts/check_p1d_regression.py
VLS/V0/scripts/check_p1b1_regression.py
VLS/V0/scripts/run_p2a_robustness.py
VLS/V0/tests/test_segment_reps.py
VLS/V0/tests/test_p2a_robustness.py
VLS/V0/configs/p2a_golden.json
VLS/V0/results/segmentation_p2a/
VLS/V0/results/segmentation_p2a_robustness/
VLS/V0/results/segmentation_p1a_p2a/full_evidence/
VLS/V0/results/segmentation_p1a_p2a/full_legacy_p2a/
VLS/V0/results/segmentation_p1a_p2a/full_evidence_p2a/
```

## 8. 限制与下一步

- 仍是 QEVD-FIT 内部盲标，不是 Fit3D 官方结果；独立第二标注者仍为 `pending`。
- P1-A 的动作周期范围是规则先验，不是从大数据拟合得到；当前 19 段歧义说明主角度选择信号高度接近，不能靠继续调权重制造“稳定性”。
- P2-A 只有一次已确认的尾部 transition 样本。规则已通过非目标段逐边界回归，但泛化性仍受样本量限制。
- 角度噪声 `>=5°` 时 P2-A 目标段和总量指标都会失效；当前规则不应对这种输入承诺鲁棒，应先提高姿态/角度质量或增加辅助信号。
- P1-A 继续保留 `evidence` 作为候选审计模式，等新增盲标动作出现计数或边界收益后再考虑替换 legacy。
- 下一步优先级：Fit3D 到位后复验 P2-A 与噪声曲线；补独立第二标注者；再决定是否需要把 `9/2` 放到留出集验证。
