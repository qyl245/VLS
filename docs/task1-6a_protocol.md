# task1-6A：协议、适配器与基线

## 1. 执行结论

状态：**完成（2026-10-26）**。

T1-6A 已冻结 RepChain v2 的输出契约、边界语义候选、评测口径和数据隔离规则；Fit3D 与 QEVD 已通过独立 adapter 映射到统一源清单，且 adapter 不调用分割器、不修改任何算法输出。T1-5D 基线已从现有 `summary.json` 逐项复现，数值完全一致。

本轮**不能**宣称跨数据源验收完成：QEVD sealed holdout 只完成了选片、窗口清单、空白标注模板和抽帧接触表，尚未人工填写 GT。只有完成冻结标注并跑完 T1-6E 后，才能报告跨域通过或失败。

## 2. 产物

```text
configs/task1_6_eval_protocol.json
configs/task1_6_sources.json
scripts/adapt_task1_6_sources.py
schemas/task1_6_clip_analysis_v2.schema.json
tests/test_task1_6_adapter.py
docs/task1-6a_protocol.md

results/task1_6/manifest/
├── qevd_sealed_holdout_segments.csv
├── qevd_sealed_holdout_template.csv
├── qevd_sealed_holdout_contact_sheet.jpg
└── qevd_sealed_holdout_manifest.json
```

复现命令：

```bash
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/adapt_task1_6_sources.py
```

脚本默认检查视频文件是否存在、`frame_count/fps/trim` 是否一致。当前 113 个统一 clip 均通过文件级校验。

## 3. 统一源清单

`configs/task1_6_sources.json` 共 `113` 个 clip：

| 来源 | clip 数 | 说明 |
|---|---:|---|
| Fit3D train | `70` | `64` 个有 GT 的 train clip，`6` 个无 GT test smoke |
| QEVD T1-4 基础清单 | `41` | 原样映射，不修改 `task1_4_clips.json` |
| QEVD sealed 新增非目标段 | `2` | `0207` 的 air jump rope、good morning |
| 合计 | `113` | `70 + 41 + 2` |

Fit3D 校验结果：

- `64` 个 GT train clip，`16` 个 `subject-action` cluster，每 cluster `4` 路相机；
- `96` 个唯一 top 事件，同一 cluster 的 4 路相机事件序列完全一致；
- dev `32`、holdout `32`、smoke `6`；
- smoke clip 全部无 GT，永不进入误差统计；
- `rep_ann` 只映射为 `top_event`，不解释为 bottom、完整周期起点或区间真值。

QEVD 校验结果：

- T1-4 基础 `41` 个 clip：dev `13`、holdout `14`、reserve `14`；
- 从 reserve 中选 `4` 个目标段进入 sealed holdout 后，剩余 reserve `10`；
- 新增 `2` 个非目标周期段；
- sealed holdout 合计 `6` 段：pushup `2`、squat `2`、air jump rope `1`、good morning `1`；
- 按 `video_id` 隔离，sealed 视频与 T1-4 dev/holdout 无交集；
- 旧 T1-3 标注保留为独立 annotation set，语义为 `rep_interval`，不当作 top 事件或 bottom 状态标签。

## 4. Adapter 边界

`scripts/adapt_task1_6_sources.py` 只做以下事情：

- 读取并校验源 manifest；
- 映射视频路径、绝对帧号与 clip-local 帧号；
- 映射 GT 语义：Fit3D `rep_ann → top_event`，QEVD 标注 `→ rep_interval`；
- 映射 subject/video cluster 与 dev/holdout/reserve/sealed split；
- 输出校验结果、源文件 SHA-256 和基线复现结果。

它不读取 `results/clip_analysis_v1/`、`results/segmentation_final/` 或 T1-5D 分割输出，不调用 `segment_reps.py`，也不生成、修正或删除任何 rep、边界、置信度或 set 候选。测试使用 `deepcopy` 校验 adapter 前后源 manifest 完全不变。

## 5. 冻结的 v2 输出契约

`schemas/task1_6_clip_analysis_v2.schema.json` 固定以下结构：

- 单个 rep 必须有 `start_frame/end_frame/top/bottom/phase_signal/status/confidence`；
- `top/bottom` 是状态区间，包含 `start_frame/end_frame/anchor_frame/stable_fraction/boundary_slope`，不是单帧峰谷；
- rep 状态为 `complete / endpoint_partial / incomplete`，首尾半周期必须与完整 rep 区分；
- 置信度固定为三层：`complete / in_set / combined`，均为 `[0, 1]`；
- 一个 clip 可以输出多个 `set_candidate`，必须标记默认候选；
- 拒答原因固定在可见性、相位歧义、rep 不完整、组外、形状、时序和 `no_confident_set` 七类；
- setup/rest/recovery/transition 通过 `non_rep_intervals` 显式输出，不靠静默丢帧。

## 6. 边界与评测口径

边界 anchor 候选在 T1-6A 冻结为：

```text
plateau_entry
plateau_midpoint
plateau_exit
phase_zero_crossing
```

`plateau_midpoint` 只是 T1-6B 的初始默认候选，不是最终结论。T1-6B 必须用 dev 数据比较后选定一个**全局**策略，写入 `task1_6_segmenter.json`；进入 holdout 后不得按数据集切换。

事件匹配沿用 T1-5D：

- GT：Fit3D `gt.rep_frames`，仅 top/伸展边界事件；
- 预测：accepted complete rep 的 anchor frame；
- 匹配：一对一 Hungarian；
- 门限：`0.25 × GT top-to-top 中位周期`，四舍五入，最少 `1` 帧；
- 主统计单位：Fit3D `subject-action`，QEVD segment/video；
- Fit3D 4 路相机是重复观测，不是独立样本；
- 高置信误检阈值固定为 `0.50`。

算法核禁止使用数据集名、video/subject/camera ID、固定 rep 数、固定帧偏移、固定平台宽度、相机顺序/共识和 sealed GT 调参。动作语义配置仍可包含 squat/pushup 的物理先验，但不能绑定数据源。

## 7. T1-5D 基线复现

| 指标 | 冻结值 | 复现 |
|---|---:|---:|
| Fit3D train clip | `64` | `64` |
| `subject-action` | `16` | `16` |
| 唯一 GT rep | `96` | `96` |
| 共识计数 MAE | `0.625` | `0.625` |
| 完全计数正确 | `6/16` | `6/16` |
| 主相机事件误差中位 | `12.5` 帧 | `12.5` 帧 |
| 主相机事件误差 p90 | `25.8` 帧 | `25.8` 帧 |
| 主相机高置信 FP | `24` | `24` |
| 全机位高置信 FP | `80` | `80` |
| top-to-top 间隔中位误差 | `3` 帧 | `3` 帧 |

同时校验 T1-5D `summary.json` 记录的 Fit3D manifest SHA 与当前源 manifest SHA 一致，避免把不同清单上的数字误当作同一基线。

## 8. QEVD sealed holdout

| segment | action | 窗口帧 | 选择依据 |
|---|---|---:|---|
| `0152_pushup_set01` | pushup | `2313–2968` | T1-4 reserve，未进入 dev/holdout |
| `0159_pushup_set01` | pushup | `1710–2474` | T1-4 reserve，未进入 dev/holdout |
| `0127_squat_set01` | squat | `1840–2545` | T1-4 reserve，未进入 dev/holdout |
| `0177_squat_set01` | squat | `2412–3090` | T1-4 reserve，未进入 dev/holdout |
| `qeud_0207_air_jump_rope_sealed01` | air jump rope | `346–1053` | 未参与 T1-3/T1-4 调参的非目标周期动作 |
| `qeud_0207_good_morning_sealed01` | good morning | `1054–1739` | 同上，且与 air jump rope 同视频、同 split |

标注要求：

1. 打开 `qevd_sealed_holdout_contact_sheet.jpg` 和原始视频复核窗口；
2. 在 `qevd_sealed_holdout_template.csv` 中为每个完整周期复制一行并填写 `rep_id/gt_start_frame/gt_end_frame/phase_landmark`；
3. `frame_unit` 固定为 `source_video_absolute`；
4. 非目标动作也必须完整标注，用于验证“被误输出为完整 rep”的负样本指标；
5. 不得复用 T1-3 已有标注，也不得在看到 T1-6 结果后调整窗口；
6. 标注完成后冻结模板 SHA，再进入 T1-6E。

当前状态：**`pending_human_annotation`**。因此：

- 跨数据源验收尚未完成；
- 不能报告 QEVD sealed count/boundary/F1；
- 不能宣称 RepChain v2 已具备通用性。

## 9. 已知边界

- Fit3D holdout 在 T1-5 已被观察过，对 T1-6 不是完全盲测集；T1-6 仍应避免在 holdout 上调参，但不能把它当作从未见过的独立验证。
- Fit3D squat 是杠铃深蹲，且没有真正纯侧视；T1-6 不重新启用 T1-5C 禁用的侧视角度指标。
- QEVD 旧标注与新 sealed holdout 语义不同：前者是 rep 区间，后者也要求 rep 区间，但 Fit3D 只有 top 事件；两者不能混算同一套 IoU。
- 多个候选 set 的用户意图无法仅靠视觉可靠推断；算法只能输出多个候选并标记默认，产品层需要起始/结束提示。

## 10. 下一步

进入 T1-6B：

1. 实现 top/bottom 状态区间与 anchor frame；
2. 实现首尾 `endpoint_partial`；
3. 在 dev 上比较四个 anchor 候选并冻结一个全局策略；
4. 用合成平台、缺帧、噪声和幅度缩放做不变性测试；
5. 不读取 QEVD sealed GT，不开始跨域验收。
