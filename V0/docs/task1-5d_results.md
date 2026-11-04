# task1-5D：Fit3D rep 数量与 top 事件对标

## 1. 执行结论

状态：**完成（2026-10-26）**，验收：**未通过**。

T1-5D 在 64 路 Fit3D train 视频、16 个 `subject-action`、96 个唯一 GT rep 上完成评测。4 路相机只作为同一动作实例的重复观测，不按 4 个独立样本计入主统计。

核心结果：

- 动作实例级计数共识 MAE 为 `0.625`，高于 `0.5` 验收线。
- 完全计数正确仅 `6/16`，低于 `14/16` 验收线；错误全部为少计 1 个 rep。
- 最接近侧视的主相机事件绝对误差中位数 `12.5` 帧、p90 `25.8` 帧，均超过 `5/12` 帧验收线。
- 230 个成功匹配事件中 `210` 个预测早于 GT，事件 signed error 中位数为 `-13` 帧；主相机为 `-11` 帧。
- 325 个被接受完整 rep 中 `94` 个未匹配 GT，其中 `80` 个内部 `confidence >= 0.50`。当前置信度不能阻止错误 rep 被包装成完整结果。
- 305 个被拒候选中 `280` 个不在 GT 事件门限内、`25` 个靠近未被接受覆盖的真实事件；拒答本身保守，但不能抵消已接受的高置信误检。
- `bottom` 没有 GT，不能计算 bottom 误差；`rep_ann` 继续只按 top/边界事件使用。官方 rep 起止语义未确认，本轮不报告 IoU。

因此当前 T1-4 rep 分割不能作为 Fit3D 验收通过的反馈输入，也不应据此启用新的产品阈值。T1-5D 的结论是“失败可定位”，不是“通过但指标较弱”。

## 2. 评测协议

### 2.1 单位与规则

| 项目 | 规则 |
|---|---|
| 动作实例 | 以 `subject-action` 为 cluster；共 `16` 个 |
| 相机观测 | 每个实例 `4` 路，只用于重复观测和诊断 |
| 计数共识 | 对每个实例 4 路可用输出的完整 rep 数取中位数，`.5` 向上取整 |
| 主相机 | 每个实例选择“相对额状面法向角度最大”的一路；深蹲只能算最接近侧视，不是纯侧视 |
| 预测事件 | `segmentation.reps` 中 `status=complete` 的 `top_frame` |
| GT 事件 | `gt.rep_frames`，只解释为 top/伸展边界事件 |
| 事件匹配 | GT 与预测 top 事件做一对一 Hungarian 匹配 |
| 匹配门限 | `0.25 × GT top-to-top 中位周期`，四舍五入，最少 `1` 帧 |
| 高置信误检 | 未匹配 GT 且 `confidence >= 0.50` 的已接受 rep |
| 主验收范围 | 计数用实例共识；事件用主相机；全机位只作诊断 |

### 2.2 数据边界

- train clip `64`，dev/holdout 各 `32`。
- 唯一 GT rep `96`；按相机展开后的 GT 事件观测为 `384`，后者不构成独立样本。
- analysis status：`ok=60`、`partial=3`、`rejected_quality=1`。
- quality status：`pass=37`、`warn=26`、`reject=1`。
- segmentation method：`aifit=57`、`phase_cycles=7`。
- test smoke 无 GT，已排除。

## 3. rep 数量

### 3.1 主结果

| 统计范围 | MAE | 完全正确 | 说明 |
|---|---:|---:|---|
| 64 路单相机诊断 | `1.296875 / clip` | `11/64` | 会被 4 路同实例伪增样本 |
| 16 个动作实例共识 | `0.625 / instance` | `6/16` | **主结果**；cluster bootstrap 95% CI `0.375–0.875` |
| 16 个主相机 | `1.375 / instance` | `1/16` | 单路对角度的相机选择不稳定 |

共识预测 rep 总和为 `86/96`。16 个实例中 `10` 个少 1 个、`6` 个完全正确，没有实例在共识层多计。单相机层仍有明显多计，例如：

| clip | 预测 / GT | 高置信误检 |
|---|---:|---:|
| `fit3d_s10_pushup_50591643` | `9 / 6` | `3` |
| `fit3d_s08_squat_50591643` | `8 / 6` | `5` |
| `fit3d_s04_pushup_50591643` | `3 / 6` | `1` |

高置信误检不是孤立样本：`s08_squat_50591643` 和 `s10_pushup_50591643` 均为 `quality=warn`，仍分别产生 8 个和 9 个完整 rep。现有 `pass/warn/reject` 门控没有覆盖这类错误。

### 3.2 分层

| 分层 | 共识 MAE | 完全正确 | 事件中位 / p90 |
|---|---:|---:|---:|
| pushup | `0.75` | `2/8` | `9 / 17` 帧 |
| squat | `0.50` | `4/8` | `20 / 29` 帧 |
| dev | `0.75` | `2/8` | `11.5 / 26` 帧 |
| holdout | `0.50` | `4/8` | `15 / 26.8` 帧 |

holdout 的共识计数误差没有明显放大，但事件定位比 dev 更差。不能把计数近似正确解释为分割边界可用。

## 4. top 事件定位

### 4.1 主相机

| 指标 | 结果 | 验收线 |
|---|---:|---:|
| GT / 预测事件观测 | `96 / 78` | - |
| TP / FP / FN | `52 / 26 / 44` | - |
| precision / recall / F1 | `0.6667 / 0.5417 / 0.5977` | - |
| 绝对误差中位数 | `12.5` 帧 | `<= 5` |
| 绝对误差 p90 | `25.8` 帧 | `<= 12` |
| signed error 中位数 | `-11` 帧 | - |
| 高置信误检 rep | `24` | `0` |

### 4.2 全机位诊断

| 指标 | 结果 |
|---|---:|
| 预测事件观测 | `325` |
| TP / FP / FN | `231 / 94 / 153` |
| precision / recall / F1 | `0.7108 / 0.6016 / 0.6516` |
| 绝对误差中位数 / p90 | `14 / 26` 帧 |
| signed error 中位数 / 均值 | `-13 / -13.15` 帧 |
| 预测早于 GT / 晚于 GT | `210 / 20` |
| 高置信误检 rep | `80` |

配对事件误差存在强方向性：绝大多数预测 top 比 GT 早约 `0.22–0.26 s`（50 FPS）。这可能来自：

1. 预测 `top_frame` 是 top plateau 的进入边界，而 GT 标注点更接近伸展顶点；
2. 相位检测或 boundary snap 系统性提前；
3. Fit3D `rep_ann` 自身标注落在周期内的位置与 T1-4 定义不同。

现有证据无法区分三者。不能直接给所有预测统一加 `+13` 帧，因为仍有 `20/230` 个事件晚于 GT，且门限内误差分布较宽。T1-5E/T1-6 应先冻结“top 是 plateau 进入、中心还是退出”，再做事件对齐。

### 4.3 间隔与时长

只对 GT 与预测都连续、且两端都一对一匹配的相邻事件计算，属于派生辅助指标：

| 指标 | 样本数 | 中位绝对误差 | p90 |
|---|---:|---:|---:|
| top-to-top 间隔 | `168` | `3` 帧 | `10` 帧 |
| 预测 rep 时长 | `168` | `3` 帧 | `10` 帧 |

一旦事件匹配正确，周期长度本身较稳定；主要问题仍是事件整体相位偏早、漏检和多检，而不是单个 rep 周期长度完全失真。

## 5. 拒答与高置信误检

被拒候选分类：

| 分类 | 数量 | 含义 |
|---|---:|---|
| `safe_refusal_non_gt` | `280` | 不在任何 GT 事件门限内，拒绝正确 |
| `conservative_refusal_true_event` | `25` | 靠近未被接受 rep 覆盖的真实事件，保守拒答 |
| `safe_refusal_duplicate` | `0` | 没有发现靠近已匹配 GT 的重复候选 |

主要拒答原因：`low_local_amplitude=174`、`exercise_posture_mismatch=45`、`missing_interior_bottom=43`、`incomplete_short_rep=37`。这说明质量门控能清掉大量低幅或相位不完整候选，但“被拒候选”与“已接受候选”的置信度校准是两套问题。

最关键失败不是“拒答不够多”，而是 **94 个已接受 FP 中有 80 个 confidence >= 0.50**。因此不能用当前 `confidence` 判断一个 rep 是否可信；后续至少需要增加跨相机一致性、周期完整性、边界一致性和高幅活动段一致性检查。

## 6. 主相机选择审计

按视角几何选择主相机能避免用结果挑机位，但单路质量仍可能很差：

| action | 主相机视角 | 预测 / GT | 事件中位 |
|---|---|---:|---:|
| `s07_pushup` | `side_left 78.9°` | `1 / 6` | 无匹配 |
| `s09_squat` | `oblique_right 31.5°` | `5 / 6` | 无匹配 |
| `s11_squat` | `oblique_right 34.9°` | `3 / 6` | `29` 帧 |

所以“最接近侧视”只是合理的预注册选路规则，不等于该路 MediaPipe 质量最好。产品若只有单相机，不能依赖多机位共识兜底；仍需单路质量判定和失败拒答。

## 7. 未计算项

### 7.1 bottom 误差

`rep_ann` 在 T1-5A 已确认接近膝/肘伸展高位，是 top/边界事件，不是 bottom。Fit3D 当前没有独立 bottom 标注，因此：

```text
bottom_status = not_computable_no_bottom_gt
```

本轮不计算 bottom 帧误差，也不把 `rep_ann` 反推为 bottom。

### 7.2 IoU

官方 rep 起止语义尚未确认。若用相邻 top 中点构造区间，得到的是派生区间而非官方 rep 起止；用它计算 IoU 会把事件定义假设混入定位误差。因此：

```text
iou_status = not_reported_rep_interval_semantics_unconfirmed
```

本轮只报告 top 事件、数量、匹配后的相邻周期与时长辅助指标。

## 8. 验收结论

| 验收项 | 结果 | 结论 |
|---|---:|---|
| rep count MAE `<= 0.5 / instance` | `0.625` | 未通过 |
| 完全计数正确 `>= 14/16` | `6/16` | 未通过 |
| 主相机事件中位误差 `<= 5` 帧 | `12.5` | 未通过 |
| 主相机事件 p90 `<= 12` 帧 | `25.8` | 未通过 |
| 不包装高置信错误完整 rep | `80` 个高置信 FP | 未通过 |

总体：**未通过**。

失败优先级：

1. **事件语义与边界**：预测 top 系统性早约 `11–13` 帧，需冻结 top 定义并重做 boundary alignment。
2. **单路鲁棒性**：主相机计数仅 `1/16` 完全正确，说明不能把多视角共识当作单相机能力。
3. **置信度校准**：高置信 FP 占全部 FP 的 `85.1%`，当前 confidence 不能作为完整 rep 的放行依据。
4. **少数遮挡/相位失败**：0 rep 和 1 rep 的 clip 仍存在，但主要验收失败并非只由这些极端样本造成。

后续不应启用新的 rep 验收阈值，也不应把 T1-5D 的周期误差当作“分割已可用”。进入 T1-5E 时，应继续保留失败样本和拒绝策略，不以平滑或事后平移掩盖系统性事件偏移。

## 9. 产物

| 路径 | 内容 |
|---|---|
| `scripts/evaluate_task1_5d_reps.py` | 动作实例聚合、Hungarian 事件匹配、拒答分类与分层统计 |
| `tests/test_task1_5d_rep_eval.py` | 计数取整、主相机、唯一匹配、拒答分类和实例共识测试 |
| `results/task1_5/rep_evaluation/summary.json` | 正式汇总与验收布尔值 |
| `results/task1_5/rep_evaluation/rep_counts.csv` | 64 路逐 clip 数量、事件与质量指标 |
| `results/task1_5/rep_evaluation/rep_events.csv` | 被接受 rep 的 top、置信度与匹配状态 |
| `results/task1_5/rep_evaluation/rep_matches.csv` | 逐 GT 事件匹配、FN 与 signed error |
| `results/task1_5/rep_evaluation/rejected_candidates.csv` | 305 个拒答候选及安全分类 |
| `results/task1_5/rep_evaluation/rep_intervals.csv` | 匹配后的派生间隔和时长误差 |
| `results/task1_5/rep_evaluation/subject_action_summary.csv` | 16 个动作实例的共识计数与主相机事件结果 |
| `results/task1_5/rep_evaluation/stratified_summary.csv` | 动作、动作×视角、split、subject 分层 |

## 10. 复现

```bash
cd /mnt/storage2/data_guest

VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/evaluate_task1_5d_reps.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --analysis-dir VLS/V0/results/task1_5/clip_analysis \
  --output-dir VLS/V0/results/task1_5/rep_evaluation

VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test*.py'
```

## 11. 后续匹配门修正

2026-10-27 在 T1-6E 验收期间发现：旧 `match_events` 先对完整 cost matrix 做 Hungarian 分配，再按门限丢弃超距匹配。首尾存在额外事件时，这会为了降低总分配代价把整条事件序列错位，导致本可在门限内匹配的事件被判为 FN/FP。

实现已改为：

```text
1. 超出门限的边赋予大于任何合法总代价的惩罚；
2. Hungarian 优先最大化门限内一对一匹配数；
3. 在相同匹配数下最小化误差；
4. 最后仍按门限过滤未匹配边。
```

为保持已发布冻结产物不变，原 `results/task1_5/rep_evaluation/` 未覆盖。使用修正实现重新评分后的独立产物在：

```text
results/task1_6/baseline_t1_5d_matchfix/
```

修正后变化：

| 指标 | 原冻结值 | 修正复评 |
|---|---:|---:|
| 全机位 TP / FP / FN | `231 / 94 / 153` | `232 / 93 / 152` |
| 全机位 precision | `0.7108` | `0.713846` |
| 全机位 recall | `0.6016` | `0.604167` |
| 全机位 F1 | `0.6516` | `0.654443` |
| 全机位误差中位 / p90 | `14 / 26` 帧 | `14 / 26.9` 帧 |
| 全机位高置信 FP | `80` | `79` |

主相机 `TP/FP/FN=52/26/44`、recall `0.541667`、precision `0.666667`、中位/p90 `12.5/25.8` 帧、高置信 FP `24` 均不变；计数共识 MAE `0.625`、exact `6/16` 也不变。所有 T1-5D 验收结论继续保持未通过。
