# Multi-View push-up side-view expanded：RepChain 可评测性测试

日期：2026-11-02
数据：`VLS/dataset/Multi-View/`
范围：7 名受试者 × good/bad × side view，共 14 段 push-up

## 1. 结论

这轮验证了真实健身房手机视频上的完整离线链路：

```text
视频 → MediaPipe 姿态/角度 → RepChain v2 → T1-7 v3 质量门/拒答
```

14 段视频全部完成 pose/角度分析，质量门均为 `pass`。以逐段人工复核的 144 个完整 push-up rep 为 GT，RepChain v2 的 event-cycle 结果为：

| 指标 | 结果 |
| --- | ---: |
| GT rep | 144 |
| 预测 event cycle | 134 |
| TP / FP / FN (IoU30) | 128 / 6 / 16 |
| Precision | 0.955 |
| Recall | 0.889 |
| F1 | 0.921 |
| clip 计数 MAE | 1.43 |
| 完全计数正确 | 6 / 14 |
| 误差不超过 1 rep | 10 / 14 |
| 边界中位绝对误差 | 4 帧 |
| 边界 p95 绝对误差 | 13 帧 |

总体可用，但还不能直接视为产品验收通过。主要剩余风险是组内长停顿导致 set 被拆开、退出动作被误识别为 rep，以及低置信片段没有被 v3 全部拒答。

## 2. 数据切片

Multi-View 中共有 144 段 push-up、24 名受试者，front/side/diagonal 各 48 段。预筛结果中 side view 最稳定：

| 视角 | pass | warn | reject |
| --- | ---: | ---: | ---: |
| front | 22 | 16 | 10 |
| side | 44 | 3 | 1 |
| diagonal | 36 | 8 | 4 |

因此本轮只选 side view 做主测试，并用 good/bad 做来源质量分组。最终选取 `subject_003/004/006/009/012/023/024`，每人 good/bad 各一段，共 14 段。所有入选文件 SHA-256 唯一，近侧可见度充足，统一映射为 `side_left`。

分辨率分布：

| 受试者 | 分辨率 | 用途 |
| --- | --- | --- |
| s003 / s004 / s006 / s009 | 1440×810 | 主测试 |
| s012 | 960×540 | 中分辨率压力测试 |
| s023 / s024 | 636×360 | 低分辨率压力测试 |

配置：`configs/multiview_pushup_expanded.json`。

## 3. GT 与标注

数据集没有 rep/phase 标注。本轮先用模型候选定位，再逐段查看 every-frame crop 和每 3 帧 dense contact sheet 人工确认。模型候选没有直接作为 GT。

GT 采用 clip-local 闭区间：

```text
rep_i = [top_i, top_i+1 - 1]
```

冻结结果：

- 14 段、144 个完整 rep。
- 边界分辨率约 3 帧，置信度记为 `medium-high`。
- 排除了起身、站立、跪姿和支撑转换等非 rep 帧。
- `s012_bad` 补回了前段两次和中段一次被长周期合并的 top。
- `s024_good` 补回了中段一次被合并的 top，并排除了末尾站立帧。

产物：

- top 锚点：`configs/multiview_pushup_expanded_tops.json`
- 冻结 GT：`results/multiview_pushup_expanded/annotation/`
- GT manifest：`results/multiview_pushup_expanded/annotation/gt_manifest.json`

`gt_reps.csv` SHA-256：

```text
4715f20b43158d5318eef6617b83083d28bf9c21aecfc10d5d0fb636eeba9d3c
```

## 4. Pose 与质量门

14 段均完成 MediaPipe pose 检测和角度提取：

| 指标 | 范围 |
| --- | ---: |
| pose detection rate | 0.953–1.000 |
| q10 key visibility | 0.695–0.780 |
| body-in-frame rate | 0.952–1.000 |
| clip quality | 14 pass |

这说明本轮 side-view 样本没有重复先前 Multi-View squat side/diagonal 中“大面积丢 pose”的问题。低分辨率 s023/s024 仍能稳定检出姿态，但边界误差更大。

进入 T1-7 v3 后，14 段均挂有 set/rep 级 warn 证据，action 全部判为 `supported`，没有质量门直接 reject。最终 6 段因 `low_confidence` 拒答：good 2 段、bad 4 段。

## 5. good / bad 分组

| 来源标签 | clip | GT | 预测 | TP / FP / FN | Precision | Recall | F1 | 计数 MAE | v3 拒答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| good | 7 | 84 | 77 | 75 / 2 / 9 | 0.974 | 0.893 | 0.932 | 1.57 | 2 |
| bad | 7 | 60 | 57 | 53 / 4 / 7 | 0.930 | 0.883 | 0.906 | 1.29 | 4 |

good/bad 的 event F1 差距不大，说明核心分段能力没有明显依赖来源质量标签。bad 组 precision 更低，主要来自退出/跪姿转换和一次多余 cycle；good 组 recall 损失主要集中在 `s003_good` 的 set 拆分。

注意：good/bad 是视频级来源标签，不是 rep 级动作质量专家标注，不能用来推导 rep 数或边界。

## 6. 分受试者结果

| 受试者 | GT | 预测 | TP / FP / FN | P | R | F1 | 计数 MAE | p95 边界误差 | v3 拒答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s003 | 32 | 22 | 22 / 0 / 10 | 1.000 | 0.688 | 0.815 | 5.0 | 12.0 | 2 |
| s004 | 11 | 12 | 11 / 1 / 0 | 0.917 | 1.000 | 0.957 | 0.5 | 14.95 | 0 |
| s006 | 24 | 24 | 24 / 0 / 0 | 1.000 | 1.000 | 1.000 | 0.0 | 4.0 | 0 |
| s009 | 13 | 11 | 11 / 0 / 2 | 1.000 | 0.846 | 0.917 | 1.0 | 14.0 | 1 |
| s012 | 28 | 26 | 25 / 1 / 3 | 0.962 | 0.893 | 0.926 | 1.0 | 11.0 | 1 |
| s023 | 15 | 19 | 15 / 4 / 0 | 0.789 | 1.000 | 0.882 | 2.0 | 6.55 | 0 |
| s024 | 21 | 20 | 20 / 0 / 1 | 1.000 | 0.952 | 0.976 | 0.5 | 5.05 | 2 |

## 7. 分 clip 明细

| clip | 标签 | GT | 预测 | TP / FP / FN | P | R | F1 | 起/终误差中位 | v3 拒答 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s003_good` | good | 18 | 9 | 9 / 0 / 9 | 1.000 | 0.500 | 0.667 | 4 / 4 | low_confidence |
| `s003_bad` | bad | 14 | 13 | 13 / 0 / 1 | 1.000 | 0.929 | 0.963 | 7 / 7 | low_confidence |
| `s004_good` | good | 6 | 7 | 6 / 1 / 0 | 0.857 | 1.000 | 0.923 | 10 / 10 | — |
| `s004_bad` | bad | 5 | 5 | 5 / 0 / 0 | 1.000 | 1.000 | 1.000 | 13 / 11 | — |
| `s006_good` | good | 16 | 16 | 16 / 0 / 0 | 1.000 | 1.000 | 1.000 | 2 / 2 | — |
| `s006_bad` | bad | 8 | 8 | 8 / 0 / 0 | 1.000 | 1.000 | 1.000 | 3 / 3 | — |
| `s009_good` | good | 6 | 6 | 6 / 0 / 0 | 1.000 | 1.000 | 1.000 | 8.5 / 8.5 | — |
| `s009_bad` | bad | 7 | 5 | 5 / 0 / 2 | 1.000 | 0.714 | 0.833 | 13 / 8 | low_confidence |
| `s012_good` | good | 18 | 18 | 18 / 0 / 0 | 1.000 | 1.000 | 1.000 | 2 / 2 | — |
| `s012_bad` | bad | 10 | 8 | 7 / 1 / 3 | 0.875 | 0.700 | 0.778 | 4 / 6 | low_confidence |
| `s023_good` | good | 10 | 11 | 10 / 1 / 0 | 0.909 | 1.000 | 0.952 | 5 / 5 | — |
| `s023_bad` | bad | 5 | 8 | 5 / 3 / 0 | 0.625 | 1.000 | 0.769 | 4 / 4 | — |
| `s024_good` | good | 10 | 10 | 10 / 0 / 0 | 1.000 | 1.000 | 1.000 | 2.5 / 2.5 | low_confidence |
| `s024_bad` | bad | 11 | 10 | 10 / 0 / 1 | 1.000 | 0.909 | 0.952 | 2.5 / 2.5 | low_confidence |

## 8. 主要失败模式

1. **组内停顿导致 set 拆分。** `s003_good` 被分成两个 set，默认 set 只有 9 个 cycle，导致 9 个 FN；这是本轮最大的单点损失。该 clip 的后续 set 实际被正确识别，但当前 event-cycle 评测只统计默认 set。
2. **退出/跪姿转换被当成 rep。** `s023_bad` 把末尾支撑到跪姿的转换扩展成 3 个 FP，precision 降到 `0.625`。v3 仍给出 `supported` 且未拒答，是当前最需要优先处理的产品风险。
3. **慢动作和长顶撑造成边界漂移。** `s004_bad` 的计数完全正确，但起止边界的绝对值误差中位数为 `13/11` 帧；该段每个 rep 的顶撑停顿时长很长，单点 top 语义比连续动作更模糊。
4. **低幅/快速组仍有漏检。** `s012_bad` 和 `s009_bad` 分别漏 3 个和 2 个完整 rep，且二者都被 v3 以 `low_confidence` 拒答。拒答避免了高置信错误，但还不能把结果直接用于反馈。
5. **低分辨率不是当前主要瓶颈。** `s023/s024` 的 pose 检出率仍接近 1.0，`s024_good` 甚至达到 10/10；问题更多来自转换动作语义和边界，而不是完全丢姿态。

## 9. 对后续算法的含义

- 优先修 set 合并/跨停顿时长链闭合，而不是继续降低单 rep 阈值。`s003_good` 说明错误主要发生在 set 层。
- 为 push-up 增加“站立/跪姿退出”动作状态，不应让退出弧线进入 rep 候选链；`s023_bad` 是直接证据。
- v3 的 `low_confidence` 拒答覆盖了部分难样本，但没有覆盖 `s023_bad` 的 3 个 FP。需要把 set 内 cycle 数异常、退出转换证据和边界稳定性接入拒答，而不是只看动作置信度。
- 当前 good/bad 差异不显著，暂不应针对 bad 标签单独调参。真实产品更值得按“是否有退出转换、是否有长停顿、分辨率/帧率”分层验证。
- 本轮的 3 帧 GT 只能支撑约 3 帧分辨率的边界结论，不应把它当作逐帧专家标注。

## 10. 产物

| 产物 | 路径 |
| --- | --- |
| manifest | `configs/multiview_pushup_expanded.json` |
| top 锚点 | `configs/multiview_pushup_expanded_tops.json` |
| 接触表 | `results/multiview_pushup_expanded/annotation/frames/` |
| 候选 top 审计图 | `results/multiview_pushup_expanded/annotation/top_audit/` |
| 冻结 GT | `results/multiview_pushup_expanded/annotation/` |
| pose/角度 | `results/multiview_pushup_expanded/clip_analysis/` |
| RepChain v2 | `results/multiview_pushup_expanded/repchain_v2/` |
| T1-7 v3 | `results/multiview_pushup_expanded/lite_v3/` |
| 汇总 JSON | `results/multiview_pushup_expanded/evaluation/expanded_summary.json` |
| clip 指标 | `results/multiview_pushup_expanded/evaluation/clip_metrics_enriched.csv` |
| 受试者指标 | `results/multiview_pushup_expanded/evaluation/subject_metrics.csv` |

## 11. 复现

当前工作区没有 `.venv`，以下命令使用已有的 `.venv-mediapipe`：

```bash
cd /mnt/storage2/data_guest/VLS/V0

# 1. 预筛全部 push-up，并生成扩样 manifest
.venv-mediapipe/bin/python scripts/screen_multiview_exercise.py
.venv-mediapipe/bin/python scripts/prepare_multiview_pushup_expanded.py

# 2. 冻结人工复核后的 GT
.venv-mediapipe/bin/python scripts/build_multiview_pushup_expanded_gt.py

# 3. RepChain v2
.venv-mediapipe/bin/python scripts/run_collection_repchain.py \
  --manifest configs/multiview_pushup_expanded.json \
  --gt-reps results/multiview_pushup_expanded/annotation/gt_reps.csv \
  --gt-labels results/multiview_pushup_expanded/annotation/gt_clip_labels.csv \
  --analysis-dir results/multiview_pushup_expanded/clip_analysis \
  --output-dir results/multiview_pushup_expanded/repchain_v2

# 4. T1-7 v3
PYTHONPATH=/tmp/collection_pydeps .venv-mediapipe/bin/python \
  scripts/build_task1_7_lite.py \
  --analysis-dir results/multiview_pushup_expanded/repchain_v2/analysis \
  --clips configs/multiview_pushup_expanded.json \
  --output-dir results/multiview_pushup_expanded/lite_v3 \
  --gate-dir results/multiview_pushup_expanded/repchain_v2/gates

# 5. 汇总
.venv-mediapipe/bin/python scripts/summarize_multiview_pushup_expanded.py
```

## 12. 局限

- 只测 side view，不能代表 front/diagonal；本轮选择 side 是因为预筛质量最好。
- 仅 7 名受试者、每人 good/bad 各一段，不能形成受试者级泛化结论。
- GT 为 3 帧网格人工复核，边界精度应保留约 3 帧误差。
- 数据集视频存在明显运动模糊，顶部/底部平台本身可能不清晰。
- T1-7 v3 的 `warn` 和 `low_confidence` 阈值属于当前冻结配置，本轮没有为 push-up 调参。
