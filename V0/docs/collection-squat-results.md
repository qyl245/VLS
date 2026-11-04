# 网上深蹲视频：逐帧 GT 与 RepChain 测试

日期：2026-11-02

## 1. 结论

5 个原始视频按动作边界拆成 8 个深蹲 clip，其中 2 个是下蹲后直接进入
burpee/push-up、没有回到站立位的负样本。逐帧复核后冻结 16 个完整深蹲
rep。

RepChain v2 在包含连续多 rep 的 3 个 clip 上事件计数全部正确：

| clip | GT rep | RepChain 事件周期 | 正式 complete rep |
|---|---:|---:|---:|
| `c02_5034577_squat_set01` | 5 | 5 | 3 |
| `c03_6454275_squat_set01` | 4 | 4 | 2 |
| `c04_6868320_squat_set01` | 4 | 4 | 2 |

其余 5 个 clip 的完整 rep 数为 `1 / 0 / 1 / 0 / 1`，RepChain 都返回
`no_confident_set`。其中两个 0-rep 负样本均正确未报 rep；三个 1-rep
clip 被冻结的最小链长门槛拦住，属于当前短片段限制，而不是误检。

## 2. 数据与切片

| 原始视频 | 分辨率 | FPS | 切片 |
|---|---:|---:|---|
| `4764175` | 2160x3840 | 29.970 | `c01` 1 个完整 rep |
| `5034577` | 3840x2160 | 29.970 | `c02` 5 个完整 rep |
| `6454275` | 2160x3840 | 23.976 | `c03` 4 个完整 rep |
| `6868320` | 1080x1920 | 25.000 | `c04` 4 个浅幅度 rep |
| `7901204` | 2160x4096 | 25.000 | `c05a` 0，`c05b` 1，`c05c` 0，`c05d` 1 |

`7901204` 混合了深蹲、俯卧撑和 burpee。共用的深蹲/过渡段被拆成四个
clip，避免俯卧撑污染深蹲计数。`c05a` 和 `c05c` 只有下蹲、没有回到
站立顶部的完整相位，因此明确标成 0-rep 负样本。

清单：`configs/collection_squat_clips.json`。

## 3. 逐帧 GT

GT 使用 clip-local decoded frame index，`start_frame/end_frame` 为闭区间。
边界统一取站立 top 相位，并用原始视频的一帧分辨率抽帧条带复核。

冻结文件：

- `results/collection_squat/annotation/gt_reps.csv`
- `results/collection_squat/annotation/gt_clip_labels.csv`
- `results/collection_squat/annotation/gt_manifest.json`
- `results/collection_squat/annotation/annotation_checksums.sha256`

`gt_reps.csv` SHA-256：

```text
53933f49d17dcbcf834716c8a4849dd4f66e524358e778b35a699cd2594d91f3
```

`gt_clip_labels.csv` SHA-256：

```text
710ebb49ee7654d1fc9ecebdfdda169cf7346819de8594937627f768115fd8a5
```

## 4. RepChain v2 评测

使用冻结的：

- `configs/task1_6_segmenter.json`
- `configs/task1_6_chain.json`

并以 `allow_legacy_fallback=False` 运行，未使用旧 AIFit 补洞。结果目录：
`results/collection_squat/repchain_v2/`。

两个统计口径必须分开看：

- **event cycles**：相邻 selected top anchor 形成的周期，包含首尾
  `endpoint_partial`，用于诊断事件级检测和边界。
- **formal complete rep**：只接受默认 set 中 `status=complete` 的内部
  rep，是 T1-7 正式门采用的严格口径。

### 4.1 Event cycles

| 指标 | 结果 |
|---|---:|
| GT 完整 rep | 16 |
| 检出周期 | 13 |
| TP / FP / FN (IoU>=0.30) | 13 / 0 / 3 |
| Precision / Recall / F1 | 1.000 / 0.8125 / 0.8966 |
| clip 计数完全一致 | 5 / 8 |
| clip 计数误差 <=1 | 8 / 8 |
| count MAE | 0.375 |
| start 边界误差中位 | 3 帧 |
| end 边界误差中位 | 6 帧 |
| 边界误差 p95 | 11.25 帧 |
| 0-rep 负样本正确拒答 | 2 / 2 |

3 个 FN 全是单 rep clip：`c01`、`c05b`、`c05d`。冻结链配置要求至少 3
个 rep，因此这些片段不会形成可报告 set。长组没有误检。

`c02` 最后一个周期的 end 误差为 51 帧，原因是 frozen tail-anchor 在
末端平台选择较早的 plateau-entry/zero-crossing。其余 12 个匹配周期的
边界误差为 0-12 帧。

### 4.2 Formal complete rep

| 指标 | 结果 |
|---|---:|
| 正式 complete rep | 7 |
| TP / FP / FN (IoU>=0.30) | 7 / 0 / 9 |
| Precision / Recall / F1 | 1.000 / 0.4375 / 0.6087 |
| start 边界误差中位 | 3 帧 |
| end 边界误差中位 | 2 帧 |
| 边界误差 p95 | 9 帧 |

严格口径把每组首尾 `endpoint_partial` 排除，因此 `c02/c03/c04` 分别只保留
`3/2/2` 个 complete rep。这个结果不应与 event-cycle 的 `5/4/4` 混写。

详细对齐表：

- `results/collection_squat/repchain_v2/evaluation/clip_metrics.csv`
- `results/collection_squat/repchain_v2/evaluation/rep_matches.csv`
- `results/collection_squat/repchain_v2/evaluation/summary.json`

## 5. T1-7 C-lite / E-lite 输出

结果目录：`results/collection_squat/lite_v3/`。

| 结果 | clip 数 |
|---|---:|
| `ok` | 1 |
| `no_confident_set` | 7 |
| `action_ambiguous` | 7 |
| 最终拒答 | 7 |

只有 `c02` 进入可报告状态。`c03/c04` 虽然检出 4 个 event cycle，但严格
complete rep 只有 2 个，低于 action verification 的最少 3-rep 门槛；
三个单 rep clip 同理被拒。两个 0-rep 负样本没有产生 rep，但仍以
`action_ambiguous` 拒答，而不是伪造结果。

这说明当前生产门偏保守：它不会在短片段上乱报，但会把真实单 rep 训练组
整体拒答。后续如果真实产品需要支持 `1-2 rep` 的短组，需要单独定义短组
策略，不能直接降低全局最小链长而不做回归。

## 6. 复现

pose/angles/quality：

```bash
.venv-mediapipe/bin/python scripts/analyze_clip.py \
  --manifest configs/collection_squat_clips.json \
  --output-dir results/collection_squat/clip_analysis \
  --clip-id c01_4764175_squat_set01 c02_5034577_squat_set01 \
    c03_6454275_squat_set01 c04_6868320_squat_set01 \
    c05a_7901204_squat_set01 c05b_7901204_squat_set02 \
    c05c_7901204_squat_set03 c05d_7901204_squat_set04 \
  --overwrite
```

RepChain v2 与 GT 对齐：

```bash
.venv-mediapipe/bin/python scripts/run_collection_repchain.py \
  --manifest configs/collection_squat_clips.json \
  --gt-reps results/collection_squat/annotation/gt_reps.csv \
  --gt-labels results/collection_squat/annotation/gt_clip_labels.csv \
  --analysis-dir results/collection_squat/clip_analysis \
  --output-dir results/collection_squat/repchain_v2
```

T1-7 v3：

```bash
PYTHONPATH=/tmp/collection_pydeps .venv-mediapipe/bin/python \
  scripts/build_task1_7_lite.py \
  --analysis-dir results/collection_squat/repchain_v2/analysis \
  --clips configs/collection_squat_clips.json \
  --output-dir results/collection_squat/lite_v3 \
  --gate-dir results/collection_squat/repchain_v2/gates
```

## 7. 环境与修复

- 工作区 `.venv` 不存在，且 `.venv-mediapipe` 是当前唯一可导入 MediaPipe
  的环境，因此本测试使用 `.venv-mediapipe`。
- `.venv-mediapipe` 缺少 `jsonschema` 且没有 pip；本次把依赖临时解包到
  `/tmp/collection_pydeps`，通过 `PYTHONPATH` 运行 v3 builder，没有写入
  项目环境。
- 修复了 `scripts/task1_7_analysis.py` 的真实 v3 契约 bug：v2 的
  `motion_status` 和 `event_anchor_policy` 被带入 v3，而冻结 schema 禁止
  额外字段。现在 v3 组装时移除这两个 v2-only 字段。
- `tests/test_task1_7_analysis.py` 的 fixture 已补上这两个字段防回归。
- 全量测试：`272/272` 通过。

## 8. 边界

这是 5 个网络视频、8 个深蹲 clip 的内部小样本测试，不是正式泛化验收。
视频视角、镜头透视和人物尺度差异较大；结果只能说明当前 RepChain 在
连续多 rep 片段上事件级计数和边界基本可用，不能代表 Fit3D/QEVD 或真实
产品采集分布上的最终指标。
