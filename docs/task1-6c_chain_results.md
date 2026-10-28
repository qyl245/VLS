# task1-6C：连续 rep 链与动作组选择

## 1. 执行结论

状态：**本轮实现与 Fit3D dev 阶段门完成（2026-10-26）**。

RepChain v2 已在 T1-6B 的 top/bottom 状态区间上加入：

- 候选节点与相对幅度过滤；
- 周期、幅度和 bottom 持续时间一致的候选边；
- 连通分量与 Viterbi 连续链选择；
- 一个或多个 `set_candidate`、默认组和组外拒答；
- `setup / rest / recovery` 非 rep 区间；
- 低周期显著性片段的 legacy rep 回退。

Fit3D dev 结果：

| 指标 | 结果 | T1-6C 阶段门 |
|---|---:|---:|
| first event recall | `0.750000` | `>= 0.75`，pass |
| last event recall | `0.781250` | `>= 0.75`，pass |
| 组外高置信 FP | `0` | `= 0`，pass |
| event recall / precision | `0.869792 / 0.888298` | 诊断 |
| count MAE | `0.437500` | 诊断 |
| count exact | `26/32` | 诊断 |
| 有 set 的 clip | `31/32` | 诊断 |

本轮**没有**通过 T1-6D 的最终高置信 FP 门：链内仍有 `21` 个高置信未匹配 top 事件。它们全部位于某个 `set_candidate` 的时间范围内，因此“组外高置信 FP”为 `0`，但“总高置信 FP”仍为 `21`。这两个数不能混写；后者留给 T1-6D 的置信度校准与拒答处理。

QEVD dev 只完成跨域 smoke：

- `13` 个 clip 的 rep 数相对 T1-4 legacy 管线，MAE `0.230769`；
- 最大计数差 `2`，`11/13` 完全一致；
- 仅 `0161_pushup_set01` 有 5 个可用盲标 rep 区间，`5/5` 匹配；
- start/end 误差中位分别为 `11 / 8` 帧，合并边界误差中位/p95 为 `9.5 / 20.85` 帧。

由于边界 GT 只来自 `1` 个 clip，QEVD dev 的边界门**不可做正式统计验收**。文档中的 `12/25` 帧阈值只是本轮 smoke 阈值，不是 T1-6A 冻结的跨域验收线。

## 2. 产物

```text
configs/task1_6_chain.json
scripts/task1_6_chain_selection.py
tests/test_task1_6_chain_selection.py
docs/task1-6c_chain_results.md

results/task1_6/dev/
├── chain_eval_summary.json
├── chain_audit.csv
├── qevd_chain_regression.csv
├── clip_analysis_v2/                 # 32 个 Fit3D dev clip
├── qevd_clip_analysis_v2/            # 13 个 QEVD dev clip
└── checksums.sha256
```

复现：

```bash
cd /mnt/storage2/data_guest/VLS/V0
./.venv-mediapipe/bin/python scripts/task1_6_chain_selection.py
./.venv-mediapipe/bin/python -m unittest tests.test_task1_6_chain_selection -v
```

## 3. 算法

### 3.1 候选节点

每个 top 状态生成一个候选节点。节点证据包括：

- `plateau_midpoint` 事件帧；
- 与相邻 bottom 的幅度差；
- top/bottom 平台稳定比例；
- 相对同 clip 候选幅度的中位比例；
- top/bottom fallback 惩罚。

相对幅度低于 `0.5` 或缺少 bottom 的候选不进入链图，拒答为 `shape_mismatch`。该阈值是动作无关的全局配置，不按数据集或相机切换。

### 3.2 候选边

相邻候选的边同时检查：

- top-to-top 间隔必须在 `0.55–1.55` 个估计周期内；
- 周期一致性使用 log-gap Gaussian，`sigma=0.2`；
- 相邻节点幅度差；
- 相邻 bottom 持续时间占比差；
- 综合权重 `0.55 / 0.25 / 0.20`，边质量门限 `0.2`。

间隔越界直接不建边；质量不足的边不能进入 Viterbi 路径。

### 3.3 链选择与动作组

允许边形成连通分量，在每个分量内做 Viterbi 式最长得分路径：

```text
node score = fallback_penalty * (
    0.45 * top_stability +
    0.25 * bottom_stability +
    0.30 * relative_amplitude
)
chain score = sum(node score) + 0.30 * sum(edge quality)
```

最少要求 **3 个实际完整周期**，而不是 3 个 top 节点。一个包含 4 个 top 状态的路径可形成 3 个 top-to-top 周期；3 个 top 只形成 2 个周期，必须拒答。修复后当前 dev 结果未变化，但防止后续短链被误当成正式组。

如果同一 clip 存在多个不连通且各自满足最小周期的链，则输出多个 `set_candidate`。默认组按以下顺序选择：

1. rep 数最多；
2. 链置信度最高；
3. 起始帧最早。

算法不能区分“正式组”和形态相同的热身组；这时只能给出多个候选，由产品交互决定。

### 3.4 帧语义与非 rep 区间

Fit3D 和 QEVD v2 输出统一使用 **clip-local frame**。QEVD legacy rep 是源视频绝对帧，回退时减去 `trim.start_frame`；非 rep 区间使用角度表的实际长度，而不是源视频总帧数。

本轮修复了此前 recovery 区间延伸到源视频末尾的问题。例如 `0161_pushup_set01` 的 recovery 从错误的 `495–4488` 修正为 clip-local 的 `495–667`。

## 4. Fit3D dev 结果

### 4.1 总体

| 指标 | T1-6B 状态候选 | T1-6C 链选择 |
|---|---:|---:|
| GT event 观测 | `192` | `192` |
| 预测 event | `206` | `188` |
| TP / FP / FN | `170 / 36 / 22` | `167 / 21 / 25` |
| event precision | `0.825243` | `0.888298` |
| event recall | `0.885417` | `0.869792` |
| first event recall | `0.812500` | `0.750000` |
| last event recall | `0.781250` | `0.781250` |
| 绝对误差中位 / p90 | `6 / 22` 帧 | `6 / 20.2` 帧 |
| 高置信 FP | `36` | `21` |
| 组外高置信 FP | 未定义 | `0` |

链选择减少了 `15` 个 FP，并把误差 p90 从 `22` 帧降到 `20.2` 帧；代价是 `3` 个 FN，主要来自一个无 set 的 pushup clip。首事件召回仍为阶段线 `0.75`，没有余量。

### 4.2 分动作

| 动作 | clips | TP / FP / FN | first matched | last matched | 多 set / 无 set |
|---|---:|---:|---:|---:|---:|
| squat | `16` | `91 / 5 / 5` | `15/16` | `12/16` | `0 / 0` |
| pushup | `16` | `76 / 16 / 20` | `9/16` | `13/16` | `1 / 1` |

squat 的组外候选清理较稳定，但首尾事件仍有边界偏内问题；pushup 的主要误差在链内误检、漏检和组拆分。

### 4.3 6 个 top 事件为何对应 5 个 rep

Fit3D GT 的 `rep_ann` 在本协议中只映射为 top 事件。一个完整周期由相邻两个 top 之间定义。因此：

```text
6 个连续 top 状态 -> 5 个 top-to-top 完整周期
```

`squat` 的 `5` 个 rep 与 `6` 个 selected top events 同时存在是预期语义，不是 Viterbi 丢失节点。`rep_count` 统计完整周期，`pred_event_count` 统计 top anchor；两者不能直接相等比较。

### 4.4 失败样例

`fit3d_s05_pushup_50591643`：

- 状态检测只得到 `4` 个 top、`3` 个 bottom；
- 前两个候选相对幅度为 `0.372 / 0.203`，被 `shape_mismatch` 拒答；
- 后两个候选幅度合格，但只有 `2` 个节点，最多形成 `1` 个周期，低于最小 `3` rep；
- 最终输出 `no_confident_set`，没有为了计数硬补 rep。

该 clip 的主要问题在 T1-6B 状态检测和 bottom fallback，不在链选择阈值；当前保留为失败样例。

`fit3d_s03_pushup_50591643`：

- `8` 个 top 状态被分成两个各 `3` rep 的 set；
- GT 只有 `6` 个 top 事件，但预测 `8` 个；
- 默认组为第一个 3-rep 组，另一个组保留为非默认候选；
- 说明当前链可以在周期断裂处拒绝合并，但仍会把重复/过渡候选带入某一组，需 T1-6D 校准。

`fit3d_s03_squat_58860488`：

- 选中事件为 `[357, 535, 677, 820, 963, 1144]`；
- GT 为 `[405, 528, 663, 807, 952, 1090]`；
- 首事件误差 `48` 帧、末事件误差 `54` 帧，均超过 `35` 帧匹配门；
- 末尾另有相对幅度 `0.355` 的弱候选，被 `shape_mismatch` 正确拒绝。

这是真实的状态 anchor 偏差，不能通过放宽事件匹配门掩盖。

### 4.5 高置信 FP 的边界

`21` 个高置信 FP 均为已进入某个 set 的未匹配 top anchor。T1-6C 的预注册条件只要求组外高置信 FP 为 `0`；实现现在通过每个事件的 `set_ids` 实际计算该指标，不再写死常量。

T1-6D 仍要求最终总高置信 FP 为 `0`，当前明显未达到。不得把 T1-6C 的 `outside_set=0` 写成最终 FP 验收通过。

## 5. QEVD dev smoke

### 5.1 计数

| 指标 | 结果 |
|---|---:|
| clips | `13` |
| legacy fallback | `2` |
| count MAE | `0.230769` |
| max count delta | `2` |
| 完全一致 | `11/13` |

两个 fallback clip：

- `0005_pushup_set01`：periodicity `0.135582`，v2/legacy 均为 `3`；
- `0161_pushup_set01`：periodicity `0.105365`，v2 无回退为 `3`，回退后为 `5`，与 legacy 一致。

其余 `11` 个 clip 使用 v2 链输出。计数差不为零的只有：

- `0007_pushup_set01`：legacy `8`，v2 `10`，差 `2`；
- `0034_squat_set01`：legacy `10`，v2 `9`，差 `1`。

### 5.2 边界

只有 `0161_pushup_set01` 有 5 个可用盲标区间，且 `5/5` 匹配：

| 指标 | 结果 |
|---|---:|
| start 绝对误差中位 | `11` 帧 |
| end 绝对误差中位 | `8` 帧 |
| pooled boundary 中位 / p95 | `9.5 / 20.85` 帧 |

这只能证明该 clip 的帧偏移换算和边界量级可用，不能代表 QEVD dev 整体分布。正式跨域边界验收仍需更多盲标区间或 T1-6E sealed holdout。

## 6. 配置与通用性边界

`configs/task1_6_chain.json` 的 `core` 只包含动作无关的链参数；测试检查其中不含 dataset/video/subject/camera 标识、固定 rep 数、固定帧偏移或固定平台宽度。

仍依赖产品/动作语义的部分：

- 相位信号来自动作级关节候选与左右均值规则；
- `rep_unit` 由动作配置决定；
- 多个形态相同动作组的真实意图无法只靠视觉判断；
- 低周期显著性时回退到 legacy 管线，回退质量受上游限制。

因此当前可声明的能力是：**通用链选择协议已在 Fit3D dev 通过阶段门，并在 QEVD dev 完成有限 smoke；尚未完成跨数据源正式验收。**

## 7. 测试

`tests/test_task1_6_chain_selection.py` 共 `10` 项，覆盖：

- 6 个 top 状态形成 1 个 5 周期 set；
- 首尾弱候选被 `shape_mismatch` 剔除；
- 2/3 个 top 状态因实际周期不足而拒答；
- 缺 bottom 不生成 complete rep；
- 周期突变断开链且不合并；
- 幅度/bottom 形态异常时边被拒绝；
- 幅度缩放、offset、15/30/50/60 FPS、噪声和缺帧不改变完整链结构；
- 两个独立组输出两个 set，最长链为默认；
- QEVD legacy 绝对帧到 clip-local 帧的换算；
- `core` 无数据源特定 token，输出满足 v2 必需字段和取值范围。

本轮全量测试 `127/127` 通过。

## 8. 下一步

进入 T1-6D：

1. 对链内高置信 FP 做置信度校准和拒答；
2. 区分 `no_confident_set`、`shape_mismatch` 与实际边界偏差；
3. 在 dev 上重新检查组拆分与重复候选；
4. 最终总高置信 FP 必须降到 `0`，不能只看组外 FP；
5. 在 holdout 前冻结阈值，不读取 sealed GT 调参。
