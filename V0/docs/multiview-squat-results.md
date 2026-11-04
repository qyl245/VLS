# Multi-View squat pilot：RepChain 可评测性测试

日期：2026-11-02  
数据：`VLS/dataset/Multi-View/`  
范围：`subject_001` 的 6 个 squat 视频（good/bad × front/side/diagonal）

## 1. 结论

这轮先验证的是 **数据能否进入现有 MediaPipe → 角度 → RepChain v2 → T1-7 v3 链路**，不是官方 benchmark。结果分成两类：

- **front 可用**：两段视频 pose 检出率均为 `100%`，RepChain event cycle 分别精确得到 `17/17` 和 `15/15`，没有 FP；事件边界中位误差为 `0` 帧和 `3` 帧。
- **side/diagonal 当前不可用**：pose 检出率只有 `63.7%–76.1%`，低于 `90%` 的 pass 线；四段全部被质量门拒答。抽帧确认人始终在画面内，主要问题是明显的运动模糊/失焦，而不是主体出画或近侧选错。

因此，Multi-View 适合先补充 **front-view squat** 的分段/计数压力测试；side/diagonal 在这些原始视频上不适合作为 phase boundary 的主评测集。

## 2. 数据切片

只选了 `subject_001`，先控制变量核对三种视角。六段均为 15 FPS、810×1440、约 31–33 秒。

| clip | 视角映射 | 标签 | 帧数 |
|---|---|---|---:|
| `mv_s001_squat_good_front` | `front` | good | 476 |
| `mv_s001_squat_bad_front` | `front` | bad | 468 |
| `mv_s001_squat_good_side` | `side_left` | good | 479 |
| `mv_s001_squat_bad_side` | `side_left` | bad | 485 |
| `mv_s001_squat_good_diagonal` | `oblique_left` | good | 480 |
| `mv_s001_squat_bad_diagonal` | `oblique_left` | bad | 494 |

数据集目录只写 `side` / `diagonal`；按左右关键点可见度复核后，近侧统一映射为左侧，避免把可用侧信号误判为不可用。

配置：`configs/multiview_squat_clips.json`。

## 3. GT 与可评测性

数据集没有 rep/phase 标注，GT 由本次抽帧复核生成，采用 clip-local 闭区间：

```text
rep_i = [top_i, top_i+1 - 1]
```

冻结结果共 6 段、101 个完整 squat rep：

- `results/multiview_squat/annotation/gt_reps.csv`
- `results/multiview_squat/annotation/gt_clip_labels.csv`
- `results/multiview_squat/annotation/gt_manifest.json`

`gt_reps.csv` SHA-256：

```text
5285fedb3f067a27ec518fa24c3d3e1572ad2e95c6a1a48fa331f8d3164de041
```

置信度分层：

| 数据 | GT 置信度 | 用途 |
|---|---|---|
| front good/bad | high | 可用于本轮计数、边界和拒答分析 |
| side good/bad | medium | 仅作覆盖率和拒答诊断 |
| diagonal good/bad | low | 视频模糊严重，不作边界精度结论 |

生成脚本：`scripts/build_multiview_squat_gt.py`。

## 4. Pose 与质量门

| clip | GT rep | pose 检出率 | q10 关键点可见度 | 质量门 | v3 输出 |
|---|---:|---:|---:|---|---|
| good front | 17 | 1.000 | 0.995 | pass | `ok`，不拒答 |
| bad front | 15 | 1.000 | 0.994 | warn | `ok`，不拒答 |
| good side | 18 | 0.637 | 0.766 | reject | `rejected_quality`，拒答 |
| bad side | 17 | 0.748 | 0.768 | reject | `rejected_quality`，拒答 |
| good diagonal | 18 | 0.702 | 0.765 | reject | `rejected_quality`，拒答 |
| bad diagonal | 16 | 0.761 | 0.567 | reject | `rejected_quality`，拒答 |

质量门要求 pose 检出率至少 `0.90` 才 pass，低于 `0.75` 直接 reject；body-in-frame 的 pass/reject 线为 `0.95/0.85`。side/diagonal 的失败集中在 pose 覆盖，不是 action 分类失败：RepChain 的 `motion_status` 仍为 `ok`，但 v3 正确以 `quality_reject` 拒答。

## 5. RepChain v2 结果

front 两段是当前唯一可做正式误差统计的样本：

| clip | GT | event cycles | TP / FP / FN | start 中位误差 | end 中位误差 | formal complete |
|---|---:|---:|---:|---:|---:|---:|
| good front | 17 | 17 | 17 / 0 / 0 | 0 帧 | 0 帧 | 16 |
| bad front | 15 | 15 | 15 / 0 / 0 | 3 帧 | 3 帧 | 13 |

front 合计：

- event cycles：`32 / 32`，Precision = Recall = F1 = `1.000`。
- clip 计数：两段均完全一致。
- 边界误差：good front p95 为 `1.8` 帧；bad front p95 为 `3.3` 帧。
- formal complete rep：`29 / 32`。少掉的 3 个都是每组首尾的 `endpoint_partial`，不是中间漏检。

side/diagonal 四段在质量门前被拒，RepChain 没有产生 set，因此不计入可用精度结论。评测表中的 69 个 pooled FN 主要是拒答，不应解释成“算法在这四段漏了 69 次”。

详细结果：

- `results/multiview_squat/repchain_v2/evaluation/clip_metrics.csv`
- `results/multiview_squat/repchain_v2/evaluation/rep_matches.csv`
- `results/multiview_squat/lite_v3/summary.json`

## 6. 对产品/后续实验的含义

1. **front-view 固定机位已经足够支撑第一版离线反馈。** 在清晰视频上，RepChain 的事件计数和边界稳定性都很好，尤其 good front 达到逐 rep 边界一致。
2. **不应为了 Multi-View 的 side/diagonal 放宽全局质量门。** 这些片段是视频退化，不是动作幅度小；降低门槛只会把低置信 pose 放进下游。
3. **先筛数据再扩样本。** 继续跑 Multi-View 时，建议先按 `pose_detection_rate >= 0.90` 和 `body_in_frame_rate >= 0.95` 预筛；side/diagonal 若要保留，应单独做去模糊/超分验证。
4. **good/bad 不是 rep 标签。** 本轮只把它当动作质量分组，没有用标签推导 rep 数或边界。
5. **真实手机拍摄应单独测。** 该数据集的 15 FPS 和滚动模糊比常见手机固定机位更差，不能据此直接判断产品场景不可用。

## 7. 产物

- manifest：`configs/multiview_squat_clips.json`
- 抽帧与角度图：`results/multiview_squat/annotation/frames/`、`results/multiview_squat/annotation/plots/`
- 冻结 GT：`results/multiview_squat/annotation/`
- pose/角度：`results/multiview_squat/clip_analysis/`
- RepChain v2：`results/multiview_squat/repchain_v2/`
- T1-7 v3 lite：`results/multiview_squat/lite_v3/`

复现 GT：

```bash
.venv-mediapipe/bin/python scripts/build_multiview_squat_gt.py
```

复现 RepChain：

```bash
.venv-mediapipe/bin/python scripts/run_collection_repchain.py \
  --manifest configs/multiview_squat_clips.json \
  --gt-reps results/multiview_squat/annotation/gt_reps.csv \
  --gt-labels results/multiview_squat/annotation/gt_clip_labels.csv \
  --analysis-dir results/multiview_squat/clip_analysis \
  --output-dir results/multiview_squat/repchain_v2
```

复现 v3：

```bash
PYTHONPATH=/tmp/collection_pydeps .venv-mediapipe/bin/python \
  scripts/build_task1_7_lite.py \
  --analysis-dir results/multiview_squat/repchain_v2/analysis \
  --clips configs/multiview_squat_clips.json \
  --output-dir results/multiview_squat/lite_v3 \
  --gate-dir results/multiview_squat/repchain_v2/gates
```
