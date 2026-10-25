# task1-3 V0 最终结果：QEVD-FIT 重复动作分割

更新日期：2026-10-25

权威产物：`VLS/V0/results/segmentation_final/`

## 1. 最终结论

已基于 QEVD-FIT-COACH 的 5 条长视频完成重复动作分割 V0，并把 P2-A 收口版固化为统一快照。最终默认配置为 `angle_scoring=legacy`、`p2a_tail_filter=guarded`、Savitzky-Golay 平滑 `7/2`。

- 动作段：22 个；有非零 rep 的定性图：21 张。
- AIFit：365 个 rep；峰值基线：341 个 rep。
- `0218/air_jump_rope`：`40 -> 39`，删除尾部 `1663-1679` transition，`0173/air_jump_rope` 保持 24。
- 盲标 IoU=0.30：Pred/TP/FP/FN `66/66/0/0`，micro F1 `1.0000`。
- raw micro IoU：`0.6851`；phase-aligned micro IoU：`0.7102`。
- 边界误差：mean |start|/|end| `5.18/4.80` 帧，平均边界误差 `0.1664 s`，p95 `12` 帧。
- 更严格的 IoU=0.50：matched `55/66`，FP/FN `11/11`，micro F1 `0.8333`。
- Walking lunge stride：GT/Pred `12/12`，F1 `1.0`。
- P2-A golden 回归：`pass`，AIFit 365、Peak 341，差异为空。

**口径边界**：这些数字来自 QEVD-FIT 内部盲标，不是 Fit3D 官方逐 rep 时间戳结果，也不是 AIFit 论文指标复现。当前不能据此声称达到论文中的 IoU 0.73。

## 2. 最终配置与产物

```text
angle_scoring=legacy
p2a_tail_filter=guarded
smooth_window=7
smooth_polyorder=2
active_bouts=true
phase_guidance=true
rep_metadata=true
confidence_template=irls
confidence_auto_delete=false
```

最终目录：

```text
VLS/V0/results/segmentation_final/
├── README.md
├── manifest.json
├── config/
├── data/
│   ├── segmentation/        # 冻结 P2-A reps、盲标评估和 run 配置
│   └── robustness/          # 消融摘要及逐 trial 审计数据
├── tables/                  # 主结果与消融表（CSV + Markdown）
├── figures/
│   ├── fig01_main_metrics.*
│   ├── fig02_smoothing_ablation.*
│   ├── fig03_angle_noise.*
│   ├── fig04_landmark_noise.*
│   ├── fig05_0218_tail_case.*
│   ├── fig06_qualitative_contact_sheet.*
│   └── qualitative/         # 21 张逐段分割图
└── checks/
```

关键文件：

| 文件 | 内容 |
|---|---|
| `tables/main_results.csv/.md` | P1-D 与 P2-A 最终主指标对比 |
| `tables/segment_counts.csv/.md` | 22 段逐段 rep 数 |
| `tables/ablation_all.csv` | 平滑、角度噪声和 landmark 相位噪声的统一机读表 |
| `data/segmentation/reps.csv` | 最终冻结的 rep 边界 |
| `data/segmentation/rejected_reps.csv` | P2-A 等规则的拒绝区间与证据 |
| `data/segmentation/annotation_blind_eval/` | 5 段盲标评估明细 |
| `figures/fig05_0218_tail_case.png` | P2-A 尾部案例标注图 |
| `checks/p2a_regression.json` | 最终快照 golden 校验结果 |
| `checks/file_checksums.sha256` | 产物 SHA-256 校验和 |

## 3. 动作段覆盖

| 视频 | 动作段 |
|---|---|
| `0153` | high knees、jumping jacks、pushups、squats |
| `0161` | butt kickers、jumping jacks、lunge jumps、pushups |
| `0162` | butt kickers、jumping jacks、walking lunges、mountain climbers |
| `0173` | good morning、air jump rope、standing kicks、boxing squat punches |
| `0218` | good morning、air jump rope、boxing squat punches、walking lunges、toe touchers、shoulder gators |

`0218/toe_touchers` 最终为 0 rep，继续作为排除段保留在 `action_segments.csv`，因此没有对应定性叠加图。`0162/mountain_climbers` 仍因动作窗口和左右相位质量不稳定而不进入定量评估。

## 4. 最终主结果

P1-D 与 P2-A 的对比：

| 指标 | P1-D | P2-A final | Delta |
|---|---:|---:|---:|
| AIFit reps | 366 | **365** | -1 |
| Peak reps | 341 | 341 | 0 |
| 盲标 Pred/TP/FP/FN @ IoU=0.30 | 67/66/1/0 | **66/66/0/0** | -1 FP |
| micro F1 @ IoU=0.30 | 0.9925 | **1.0000** | +0.0075 |
| raw micro IoU | 0.6851 | 0.6851 | 0.0000 |
| p95 boundary error | 12.00 帧 | 12.00 帧 | 0.00 |
| micro F1 @ IoU=0.50 | 0.8271 | **0.8333** | +0.0063 |
| `0218/air_jump_rope` | 40 | **39** | -1 |
| `0173/air_jump_rope` | 24 | 24 | 0 |

最终 AIFit 在 5 段盲标上的完整结果：

| IoU 阈值 | GT | Pred | TP | FP | FN | micro F1 | raw micro IoU | phase micro IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 66 | 66 | 66 | 0 | 0 | 1.0000 | 0.6851 | 0.7312 |
| 0.30 | 66 | 66 | 66 | 0 | 0 | 1.0000 | 0.6851 | 0.7102 |
| 0.50 | 66 | 66 | 55 | 11 | 11 | 0.8333 | 0.7414 | 0.8361 |

边界指标在 IoU=0.30 下为：signed start/end `-2.06/-2.47` 帧，mean |start|/|end| `5.18/4.80` 帧，mean boundary `0.1664 s`，p95 `12` 帧。IoU=0.50 的未匹配主要来自边界相位偏差，不是总 rep 数错误；两种阈值下 count MAE 都是 0。

## 5. 最终图

| 图 | 内容 |
|---|---|
| `figures/fig01_main_metrics.png` | 盲标 micro F1、TP/FP/FN、边界误差和 raw/phase IoU |
| `figures/fig02_smoothing_ablation.png` | 6 组平滑配置的 rep 数、0218 目标数和盲标指标 |
| `figures/fig03_angle_noise.png` | 角度噪声下的 F1、p95 和 P2-A 目标稳定性 |
| `figures/fig04_landmark_noise.png` | nose 相位噪声稳定性及 0173 非目标段计数 |
| `figures/fig05_0218_tail_case.png` | `0218/air_jump_rope` `1663-1679` 尾部拒绝证据 |
| `figures/fig06_qualitative_contact_sheet.png` | 21 张逐段定性图的联络表 |

每张图同时保存 PNG 和 PDF。逐段图索引在 `figures/qualitative_index.csv`，总图索引在 `figures/figure_index.csv`。

## 6. 消融与噪声鲁棒性

### 6.1 平滑消融

| 配置 | AIFit | Peak | 0218 | F1@0.30 | raw IoU | p95 | F1@0.50 | 改变边界数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 363 | 326 | 39 | 0.9924 | 0.6828 | 12.00 | 0.8397 | 38 |
| 5/2 | 362 | 340 | 39 | 0.9767 | 0.6840 | 11.00 | 0.8217 | 33 |
| **7/2** | **365** | **341** | **39** | **1.0000** | **0.6851** | **12.00** | **0.8333** | **0** |
| 9/2 | 365 | 324 | 39 | 1.0000 | 0.6884 | 11.45 | 0.8939 | 33 |
| 11/2 | 342 | 317 | 12 | 0.6857 | 0.7296 | 31.35 | 0.5143 | 38 |
| 7/3 | 365 | 341 | 39 | 1.0000 | 0.6851 | 12.00 | 0.8333 | 2 |

结论：`7/2` 是唯一在不改变任何 segment-method 边界时保持全部最终指标的配置。`9/2` 在同一批盲标上略降 p95，但同时改变 33 个边界，只能作为事后候选，需新的留出集验证。

### 6.2 角度噪声

| sigma | AIFit mean±sd | F1@0.30 | raw IoU | p95 | F1@0.50 | 0218 计数正确率 | P2-A 触发率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0° | 365.0±0.0 | 1.0000 | 0.6851 | 12.00 | 0.8333 | 100% | 100% |
| 1° | 366.7±2.49 | 0.9873 | 0.6675 | 15.15 | 0.8048 | 100% | 66.7% |
| 2° | 361.0±1.63 | 0.9719 | 0.6900 | 11.38 | 0.8030 | 100% | 100% |
| 5° | 361.3±12.97 | 0.8275 | 0.6656 | 21.75 | 0.6580 | 0% | 0% |
| 10° | 385.0±17.80 | 0.5589 | 0.5841 | 28.60 | 0.3235 | 0% | 0% |

`1–2°` 内接近基线，`5°` 起明显退化，`10°` 不可用。`1°` 时 P2-A 触发率为 `2/3`：有一个 trial 没有生成第 40 个尾部 rep，最终计数仍正确，因此必须把“计数正确率”和“规则触发率”分开报告。

### 6.3 Nose 相位噪声

| sigma (m) | AIFit mean±sd | 0218 计数正确率 | P2-A 触发率 | 0173 保持 24 rep | F1@0.30 |
|---:|---:|---:|---:|---:|---:|
| 0.002 | 365.0±0.0 | 100% | 100% | 3/3 | 1.0000 |
| 0.005 | 365.3±0.47 | 100% | 100% | 2/3 | 1.0000 |
| 0.010 | 365.0±0.0 | 100% | 100% | 3/3 | 1.0000 |
| 0.020 | 365.0±0.0 | 100% | 100% | 3/3 | 1.0000 |

该实验只扰动 nose world 坐标，用于测试 P2-A 的相位输入稳定性，不等价于完整的 MediaPipe 姿态误差标定。`0.005 m` 的单个 trial 已使 `0173` 增至 25 rep，说明非目标段仍可能受相位抖动影响；曲线不是单调的，不能外推为一般姿态鲁棒性。

## 7. 关键案例与解释

### 7.1 P2-A 尾部拒绝

`0218/air_jump_rope` 的最后一个候选 rep 为 `1663-1679`，位于 fit end `1684` 附近。删除时同时存在三项证据：

- rep 内缺少相位同极性 apex；
- shape similarity `0.3200 < 0.4297`；
- motion evidence `0.0010 < 0.0500`。

P1-D 到 P2-A 只删除这一个目标尾部，其余 21 段逐边界不变；`0173/air_jump_rope` 仍为 24 rep。实现不针对视频 ID 做帧号特判。

### 7.2 其他低置信项

- `0218/toe_touchers`：最终 0 rep，缺少可靠 active-core 证据；不把 0 当作已确认真值，整段排除。
- `0162/mountain_climbers`：自动主角度为 `right_hip_flexion`，AIFit 12、Peak 2，左右相位和姿态质量不稳定，整段不进入定量评估。
- `0218/walking_lunges` 与 `0162/walking_lunges`：以 stride 为 rep 单位；合并 AIFit 12/12，F1 `1.0`。
- P1-A 证据模式已实现并保留为审计入口，但 22 段中 19 段歧义并回退 legacy，未通过替换门槛。

## 8. 历史过程记录

下表仅用于追溯实现过程；最终指标以上文 P2-A 快照为准。

| 阶段 | 状态 | 对最终结果的作用 |
|---|---|---|
| P0-A 盲标 | 完成 | 5 段 66 rep 内部盲标；`independent_annotation_claimed=false` |
| P0-B 评估口径 | 完成 | 增加 IoU 0.10/0.30/0.50、macro/micro、split/merge 和边界误差 |
| P0-C active bouts | 完成 | 去掉休息/过渡假 rep，并输出拒因证据 |
| P0-D1/D2 相位约束 | 完成 | 改善 Phase A/B 边界；最终 mean boundary 降到 `0.1664 s` |
| P1-A 主角度证据 | 完成，未启用 | 提供候选证据；19/22 段歧义回退 legacy |
| P1-B1 `rep_unit` | 完成 | 明确 walking lunge 使用 stride；未触发双侧融合 |
| P1-D 鲁棒模板/置信度 | 完成 | 提供可解释置信度，不自动删除 rep |
| P2-A 尾部收口 | 完成并启用 | 删除 `0218` 尾部 transition，形成 365/341 新 golden |

各阶段细节分别见 `task1-3_p0a_results.md`、`task1-3_p0b_results.md`、`task1-3_p0c_results.md`、`task1-3_p0d1_results.md`、`task1-3_p0d2_results.md`、`task1-3_p1d_results.md`、`task1-3_p1b1_results.md` 和 `task1-3_p1a_p2a_results.md`。

## 9. 复现与校验

构建最终目录：

```bash
.venv/bin/python VLS/V0/scripts/build_final_outputs.py --overwrite
```

独立复核 golden：

```bash
.venv/bin/python VLS/V0/scripts/check_p2a_regression.py \
  --golden VLS/V0/configs/p2a_golden.json \
  --segmentation-dir VLS/V0/results/segmentation_final/data/segmentation \
  --reference-dir VLS/V0/results/segmentation_p1d \
  --report VLS/V0/results/segmentation_final/checks/p2a_regression.json
```

测试和 checksum：

```bash
.venv/bin/python -m pytest -q \
  VLS/V0/tests/test_segment_reps.py \
  VLS/V0/tests/test_p2a_robustness.py

cd VLS/V0/results/segmentation_final
sha256sum -c checks/file_checksums.sha256
```

本次收口结果：`25 passed`；P2-A 回归 `pass`；checksum 全部通过。

## 10. 数据与时间语义

- 输出边界使用解码帧索引，与 `angles_wide.csv` 的 `frame_id` 对齐。
- 真实时间必须使用 `long_range_videos/*_timestamps.npy` 映射，不能直接使用 `frame_id / 30`。
- `feedbacks_long_range.json` 的反馈时间为 UTC+8 秒，比较前需减 `28800`。
- `0153` 存在 23 个超过 100 ms 的帧间隔，时序分析时应单独检查或降权。

## 11. 限制

- QEVD-FIT 是内部盲标基准，不是 Fit3D 官方 GT；当前没有配对 3D GT，不能计算官方 AIFit 指标。
- 盲标仅覆盖 5 段 66 rep，一名标注者；22.7% 为同一标注者原始帧重读，独立第二标注者仍为 `pending`，因此不能报告 inter-annotator agreement。
- P2-A 只有 `0218/air_jump_rope` 一个已确认的尾部 transition 样本；泛化性受样本量限制。
- 角度噪声不是由 MediaPipe 真实误差标定得到；nose 噪声也没有重新计算角度，不能写成完整姿态误差鲁棒性。
- `9/2` 的轻微 p95 收益与 33 个边界变化同时出现，不能作为默认配置或独立验证收益。
- `0218/toe_touchers` 的零 rep 与 `0162/mountain_climbers` 的动作口径仍未解决，不能把排除当成负样本真值。

## 12. 下一步

1. Fit3D 到位后，将官方逐 rep 时间戳接到同一评估接口，重算 IoU、F1 和边界误差。
2. 补真正的独立第二标注者，至少覆盖当前 5 段盲标和新增尾部样本。
3. 新增 `air_jump_rope` 尾部 transition 或其他动作的同类样本，复验 P2-A 的泛化性。
4. 在独立留出集上验证 `9/2` 平滑是否仍有收益；在此之前保持 `7/2` 默认。
5. 补齐 `0218/toe_touchers` 和 `0162/mountain_climbers` 的动作窗口与 `rep_unit` 口径，再决定是否进入评估。
