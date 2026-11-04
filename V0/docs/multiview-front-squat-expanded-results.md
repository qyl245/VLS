# Multi-View front-view squat 扩样复测：7 名受试者 / 195 rep

日期：2026-11-02  
数据：`VLS/dataset/Multi-View/`  
范围：7 名受试者的 good/bad front-view squat，共 14 段视频

## 1. 结论

本轮把先前的 `subject_001` pilot 扩到 7 名受试者、14 段视频、195 个完整 squat rep，用来判断 RepChain v2 + T1-7 v3 的结果是否只对单名受试者有效。

主要结果：

- 总体 event Precision / Recall / F1 为 `0.932 / 0.846 / 0.887`（IoU 30%）。
- good 组 F1 为 `0.931`，bad 组 F1 为 `0.833`；差距主要来自 bad 组召回下降到 `0.753`。
- 计数平均绝对误差为 `2.29` rep，完全计数一致 `5/14` 段，误差不超过 1 rep 的有 `7/14` 段。
- 全部匹配事件的起点/终点误差中位数为 `5 / 5` 帧，p95 为 `14` 帧。
- v3 正常输出 `13/14` 段；`mv_s024_squat_bad_front` 以 `action_ambiguous` 拒答。

这说明扩样后仍能保持可用的总体分段能力，但 bad/低幅动作是明确短板，不能只靠放宽质量门解决。

## 2. 样本筛选

先对 Multi-View 中全部 front-view squat 做 MediaPipe 预筛，共 `50` 段、`25` 名受试者，每段按 3 帧步长抽检：

| 筛选状态 | 数量 |
| --- | ---: |
| pass | 44 |
| warn | 5 |
| reject | 1 |

唯一 reject 是 `subject_005_bad`。最终只选择 good 和 bad 两段都通过预筛的受试者，得到 7 人 14 段：

| 受试者 | good GT rep | bad GT rep | 分辨率 | 用途 |
| --- | ---: | ---: | --- | --- |
| s001 | 17 | 15 | 810×1440 | 与既有 pilot 对齐 |
| s002 | 13 | 9 | 810×1440 | 扩样主体 |
| s003 | 15 | 18 | 810×1440 | 扩样主体 |
| s004 | 13 | 10 | 810×1440 | 扩样主体 |
| s006 | 19 | 18 | 810×1440 | 扩样主体 |
| s023 | 13 | 12 | 540×960 | 低分辨率压力测试 |
| s024 | 12 | 11 | 540×960 | 低分辨率压力测试 |

合计 `102` 个 good rep 和 `93` 个 bad rep。这里 good/bad 是数据集提供的动作质量分组，不是 rep 级标签，也没有用于反推 GT 边界或 rep 数。

筛选产物：

- `results/multiview_squat/front_screen/screen.csv`
- `results/multiview_squat/front_screen/summary.json`
- `scripts/screen_multiview_front_squat.py`

## 3. GT 协议

数据集没有 rep/phase 标注，本轮通过接触表人工复核生成 clip-local GT：

```text
rep_i = [top_i, top_i+1 - 1]
```

其中 `top_i` 是每次 squat 的站立端点。只标注相邻 top 之间的完整周期，片段首尾不完整的 rep 不计入。

标注过程：

- 每段生成一张每 15 帧的 coarse 接触表；
- 每段生成两张每 3 帧的 dense 接触表；
- 在 dense 网格上人工选择 top anchor，再按上述闭区间生成 rep；
- 冻结后不再用模型预测结果修改 GT。

最终得到 `14` 段、`195` 个完整 rep。GT 边界分辨率为 `3` 帧，置信度标为 `medium-high`，不能表述为逐帧精确标注。

冻结文件与 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `results/multiview_front_squat_expanded/annotation/gt_reps.csv` | `75e7e678cb81f1260008729e0c2a25ea917afa257b530830e8d21b6a77ba6d38` |
| `results/multiview_front_squat_expanded/annotation/gt_clip_labels.csv` | `8b49ab3dab334d1bfee831d245118da59c0313ce01af8a9db18ed53f066a9967` |

元数据见 `results/multiview_front_squat_expanded/annotation/gt_manifest.json`。

## 4. 总体指标

| 指标 | 结果 |
| --- | ---: |
| clips | 14 |
| GT rep | 195 |
| predicted event cycles | 177 |
| event TP / FP / FN @ IoU 30 | 165 / 12 / 30 |
| event Precision | 0.932 |
| event Recall | 0.846 |
| event F1 | 0.887 |
| count MAE | 2.29 rep |
| exact count clips | 5 / 14 |
| within-one count clips | 7 / 14 |
| boundary start median | 5 帧 |
| boundary end median | 5 帧 |
| boundary p95 | 14 帧 |
| v3 supported | 13 / 14 |
| v3 abstained | 1 / 14 |

## 5. good / bad 分组

| 组 | GT | Pred | TP / FP / FN | Precision | Recall | F1 | count MAE | v3 supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| good | 102 | 102 | 95 / 7 / 7 | 0.931 | 0.931 | 0.931 | 1.71 | 7 / 7 |
| bad | 93 | 75 | 70 / 5 / 23 | 0.933 | 0.753 | 0.833 | 2.86 | 6 / 7 |

bad 组的 Precision 与 good 接近，但 Recall 明显下降，说明当前主要错误不是“把非 rep 判成 rep”，而是欠深、节奏不标准或低幅阶段漏掉完整周期。

## 6. 分受试者结果

| 受试者 | GT | Pred | TP / FP / FN | Precision | Recall | F1 | count MAE | boundary median start / end | v3 abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s001 | 32 | 32 | 32 / 0 / 0 | 1.000 | 1.000 | 1.000 | 0.0 | 1 / 1 | 0 |
| s002 | 22 | 20 | 19 / 1 / 3 | 0.950 | 0.864 | 0.905 | 1.0 | 7 / 7 | 0 |
| s003 | 33 | 31 | 22 / 9 / 11 | 0.710 | 0.667 | 0.688 | 6.0 | 2.5 / 4 | 0 |
| s004 | 23 | 22 | 21 / 1 / 2 | 0.955 | 0.913 | 0.933 | 1.5 | 8 / 9 | 0 |
| s006 | 37 | 31 | 31 / 0 / 6 | 1.000 | 0.838 | 0.912 | 3.0 | 6 / 5 | 0 |
| s023 | 25 | 25 | 25 / 0 / 0 | 1.000 | 1.000 | 1.000 | 0.0 | 5 / 5 | 0 |
| s024 | 23 | 16 | 15 / 1 / 8 | 0.938 | 0.652 | 0.769 | 4.5 | 8 / 7 | 1 |

重点失败：

- `s003` 是最差受试者。good 段 `15 GT / 20 Pred`，多出 5 个 cycle；bad 段 `18 GT / 11 Pred`，只命中 7 个，漏 11 个，且该段终点误差中位数为 15 帧。
- `s024_bad` 只检出 `3/11` 个 rep，但检出的 3 个边界全部匹配。该段随后被 v3 以 `action_ambiguous` 拒答，属于“少报并拒答”，好于输出高置信错误结果。
- `s006` 的 Precision 为 1.000，但漏 6 个 rep；失败模式偏向保守漏检。
- `s004` 的 F1 为 0.933，但起点/终点误差中位数为 8/9 帧，说明匹配成立时边界仍明显偏移。
- `s002` 的 good 段 `13 GT / 11 Pred`，有 1 个 FP 和 3 个 FN；bad 段计数完全一致，但边界误差中位数为 11 帧。

## 7. 质量门与 v3

pose/角度阶段结果为 `10 ok / 4 partial`，clip 质量门为 `13 warn / 1 pass`，没有 reject。进入 v3 后，14 段均有 set/rep 级 warn 证据，但只有 `mv_s024_squat_bad_front` 被拒答：

| 项目 | 结果 |
| --- | ---: |
| v3 `ok` | 13 |
| v3 `no_confident_set` | 1 |
| action supported | 13 |
| action ambiguous | 1 |
| abstain reason `action_ambiguous` | 1 |

该拒答是合理行为：bad 组低幅动作只恢复了 3 个完整周期，动作契约无法稳定确认，继续输出反馈会把漏检包装成高置信结论。

前置姿态规则曾拒绝大量候选 rep，但 RepChain 使用冻结信号仍能恢复其中大部分，因此不能把 pose 阶段的 `rejected_rep_count` 直接等同于最终漏检数。

## 8. 产物

| 产物 | 路径 |
| --- | --- |
| manifest | `configs/multiview_front_squat_expanded.json` |
| top anchors | `configs/multiview_front_squat_expanded_tops.json` |
| 接触表 | `results/multiview_front_squat_expanded/annotation/frames/` |
| 冻结 GT | `results/multiview_front_squat_expanded/annotation/` |
| pose/角度 | `results/multiview_front_squat_expanded/clip_analysis/` |
| RepChain v2 | `results/multiview_front_squat_expanded/repchain_v2/` |
| T1-7 v3 | `results/multiview_front_squat_expanded/lite_v3/` |
| 汇总 JSON | `results/multiview_front_squat_expanded/evaluation/expanded_summary.json` |
| clip 指标 | `results/multiview_front_squat_expanded/evaluation/clip_metrics_enriched.csv` |
| 受试者指标 | `results/multiview_front_squat_expanded/evaluation/subject_metrics.csv` |

## 9. 复现

当前工作区没有 `.venv`，以下命令沿用已有的 `.venv-mediapipe`：

```bash
cd /mnt/storage2/data_guest/VLS/V0

# 1. 筛选全部 front-view squat
.venv-mediapipe/bin/python scripts/screen_multiview_front_squat.py

# 2. 生成扩样 manifest
.venv-mediapipe/bin/python scripts/prepare_multiview_front_expanded.py

# 3. 生成标注接触表
.venv-mediapipe/bin/python scripts/extract_multiview_annotation_frames.py

# 4. 生成并冻结 GT
.venv-mediapipe/bin/python scripts/build_multiview_front_expanded_gt.py

# 5. pose/角度
CLIPS=$(.venv-mediapipe/bin/python -c "import json; print(' '.join(c['clip_id'] for c in json.load(open('configs/multiview_front_squat_expanded.json'))['clips']))")
.venv-mediapipe/bin/python scripts/analyze_clip.py \
  --manifest configs/multiview_front_squat_expanded.json \
  --clip-id $CLIPS \
  --output-dir results/multiview_front_squat_expanded/clip_analysis \
  --overwrite

# 6. RepChain v2
.venv-mediapipe/bin/python scripts/run_collection_repchain.py \
  --manifest configs/multiview_front_squat_expanded.json \
  --gt-reps results/multiview_front_squat_expanded/annotation/gt_reps.csv \
  --gt-labels results/multiview_front_squat_expanded/annotation/gt_clip_labels.csv \
  --analysis-dir results/multiview_front_squat_expanded/clip_analysis \
  --output-dir results/multiview_front_squat_expanded/repchain_v2

# 7. T1-7 v3
PYTHONPATH=/tmp/collection_pydeps .venv-mediapipe/bin/python \
  scripts/build_task1_7_lite.py \
  --analysis-dir results/multiview_front_squat_expanded/repchain_v2/analysis \
  --clips configs/multiview_front_squat_expanded.json \
  --output-dir results/multiview_front_squat_expanded/lite_v3 \
  --gate-dir results/multiview_front_squat_expanded/repchain_v2/gates

# 8. 汇总
.venv-mediapipe/bin/python scripts/summarize_multiview_front_expanded.py
```

## 10. 局限与后续

- 只验证 front-view；side/diagonal 仍受原始视频模糊和 pose 覆盖率限制，不并入本轮精度结论。
- GT 为 3 帧网格人工标注，边界结论应保留约 3 帧分辨率误差；不能与逐帧 GT 直接宣称同等精度。
- good/bad 是视频来源标签，不等价于 rep 级动作质量，也不代表临床标准。
- `s023/s024` 为 540×960，样本量仍小，低分辨率结论只能作为压力测试，不足以形成分辨率曲线。
- 下一步应优先补真实手机固定机位视频，并重点覆盖 `s003_bad`、`s024_bad` 这类低幅/欠深动作，再决定是否调整 RepChain 通用层；不应针对单个 clip ID 或数据集特判。
