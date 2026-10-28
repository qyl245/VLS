# task1-6B：状态区间、anchor 与首尾部分周期

## 1. 执行结论

状态：**完成（2026-10-26）**，T1-6B 阶段门通过。

本轮在 Fit3D **dev** 的 `32` 个相机 clip、`8` 个 `subject-action` cluster 上比较了：

- 三种全局相位信号策略：T1-5 已选信号、T1-6 自动候选、动作语义双侧均值；
- 四种全局 anchor：`plateau_entry / plateau_midpoint / plateau_exit / phase_zero_crossing`。

最终冻结：

```text
phase signal = 双侧屈曲均值（knee / elbow）
anchor       = plateau_midpoint
```

全机位 top 事件召回 `88.54%`、首事件召回 `81.25%`、末事件召回 `78.13%`，事件绝对误差中位 `6` 帧、p90 `22` 帧。按 T1-6B 预注册的阶段门（首/末均 `>=50%`）判定为 **pass**。

本轮验证的是 top/bottom 状态区间和事件 anchor，不是最终 rep 计数。T1-6C 仍需用连续链选择去掉组外候选并形成正式 `set_candidate`。

## 2. 产物

```text
scripts/task1_6_phase_states.py
configs/task1_6_segmenter.json
tests/test_task1_6_phase_states.py
docs/task1-6b_state_results.md

results/task1_6/dev/
├── state_eval_summary.json
├── state_intervals.csv
├── anchor_comparison.csv
├── clip_anchor_metrics_<signal_mode>_<anchor>.csv
├── signal_selection_audit.csv
└── checksums.sha256
```

复现命令：

```bash
cd /mnt/storage2/data_guest
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/task1_6_phase_states.py
VLS/V0/.venv-mediapipe/bin/python -m unittest VLS/V0/tests/test_task1_6_phase_states.py
```

## 3. 评测口径

- 数据：Fit3D dev `32` clips、`8` clusters、`192` 个相机事件观测；
- GT：只使用 Fit3D `rep_ann` 映射的 top event，不把其当作 bottom 或完整 rep 区间；
- 预测：每个 top 状态区间的 `anchor_frame`；
- 匹配：一对一 Hungarian，门限为 `0.25 x GT top-to-top 中位间隔`，最少 `1` 帧；
- 首/末召回：每个相机 clip 的 GT 首/末事件是否被匹配；
- 主相机：每个 `subject-action` 选视角最接近侧视的相机，仅用于诊断，不参与算法；
- 相机是同一动作的重复观测，不把 `192` 个观测当作 `192` 个独立动作实例。

## 4. 最终策略

### 4.1 全机位

| 指标 | 结果 |
|---|---:|
| GT event 观测 | `192` |
| 预测 event | `206` |
| TP / FP / FN | `170 / 36 / 22` |
| precision | `0.825243` |
| recall | `0.885417` |
| F1 | `0.854272` |
| first event recall | `0.812500` |
| last event recall | `0.781250` |
| 绝对误差中位 | `6` 帧 |
| 绝对误差 p90 | `22` 帧 |
| 绝对误差最大 | `35` 帧 |
| 有符号误差中位 | `+3` 帧 |
| GT 落在任一 top 区间 | `0.901042` |
| 已匹配 GT 落在对应 top 区间 | `0.970588` |

### 4.2 主相机诊断

| 指标 | 结果 |
|---|---:|
| clips / clusters | `8 / 8` |
| GT event 观测 | `48` |
| TP / FP / FN | `45 / 4 / 3` |
| precision / recall | `0.918367 / 0.937500` |
| first / last event recall | `0.875000 / 0.750000` |
| 绝对误差中位 / p90 | `5 / 20` 帧 |
| GT 落在任一 top 区间 | `0.979167` |

### 4.3 分动作

| 动作 | precision | recall | first | last |
|---|---:|---:|---:|---:|
| squat | `0.938144` | `0.947917` | `0.937500` | `0.750000` |
| pushup | `0.724771` | `0.822917` | `0.687500` | `0.812500` |

pushup 的误检和首事件问题更集中，是 T1-6C 链选择与组边界的主要压力来源。

## 5. Anchor 对比

选择双侧均值信号后，四种 anchor 的全机位结果：

| anchor | precision | recall | first | last | 中位误差 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| `plateau_entry` | `0.549` | `0.589` | `0.188` | `0.750` | `15` | `27.8` |
| `plateau_midpoint` | **`0.825`** | **`0.885`** | `0.812` | `0.781` | **`6`** | **`22`** |
| `plateau_exit` | `0.267` | `0.286` | `0.344` | `0.062` | `12` | `25` |
| `phase_zero_crossing` | `0.767` | `0.823` | **`0.875`** | **`0.938`** | `15` | `27.3` |

`phase_zero_crossing` 在首/末召回上更高，但整体 recall、precision 和事件误差都明显差于 `plateau_midpoint`。因此不能只按首尾召回选 anchor；本轮排序规则预先固定为：

1. 首、末召回都必须 `>=0.50`；
2. 最大化全机位事件 recall；
3. 最小化误差中位、p90；
4. 再比较首/末召回、precision 和 top 区间覆盖率。

## 6. 相位信号对比

固定 `plateau_midpoint` 时的全机位结果：

| 信号策略 | precision | recall | first | last | 中位误差 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| T1-5 已选相位信号 | `0.786` | `0.859` | `0.844` | `0.719` | `6` | **`19`** |
| T1-6 自动候选评分 | `0.768` | `0.792` | `0.750` | `0.656` | `6` | **`19`** |
| T1-6 双侧均值 | **`0.825`** | **`0.885`** | `0.812` | **`0.781`** | `6` | `22` |

自动候选评分偏向单侧大振幅信号；双侧均值降低左右角度提取差异，对全部相机更稳。最终配置只依赖动作语义中的左右关节对，不依赖数据集、视频、受试者或相机 ID。

## 7. 状态区间与首尾部分周期

选中策略在 dev 上产生 `206` 个 top 状态、`177` 个 bottom 状态：

- top 区间长度占周期比例中位 `0.561`，p10/p90 为 `0.238/0.800`；
- bottom 区间长度占周期比例中位 `0.349`，p10/p90 为 `0.171/0.478`；
- fallback top 状态 `1` 个，fallback bottom 状态 `12` 个，均保留显式状态标记。

首尾规则：

- 首/末 top 状态分别来自 `leading_partial / trailing_partial` 窗口；
- 只有同时具备 top、bottom 和合理周期/幅度证据才生成 `endpoint_partial`；
- 中间周期的 rep 才是 `complete`；
- 没有 top+bottom 或只有单侧半周期时，不补成 `complete rep`。

合成 5 周期平台波的测试结果为 `endpoint_partial, complete, complete, complete, endpoint_partial`。

## 8. 冻结配置

`configs/task1_6_segmenter.json` 的 `core` 固定：

```text
phase_signal.policy = t1_6_bilateral_mean
anchor.policy       = plateau_midpoint
```

配置包含动作级候选、检测器参数、状态阈值比例、首尾部分周期规则和 fallback。`core` 中不含数据集名、视频/受试者/相机 ID、固定 rep 数、固定帧偏移或固定平台宽度；这些约束由测试检查。数据来源、dev 样本量和 SHA-256 只记录在 `provenance`，不参与运行时决策。

## 9. 测试

`tests/test_task1_6_phase_states.py` 覆盖：

- anchor 顺序与状态区间包含；
- 合成平台中点误差；
- 首尾 `endpoint_partial`；
- 幅度缩放、offset、极性翻转；
- `15/30/50/60 FPS` 重采样；
- 噪声、landmark 抖动、缺帧；
- 短片段、半 rep、无 bottom 不得产生 complete rep；
- 双侧均值和单侧缺失 fallback；
- 一对一 Hungarian 匹配与门限；
- 冻结配置 `core` 不含数据源特定 token。

本轮定向测试 `13/13` 通过。

## 10. 与 T1-5D 的对照和边界

T1-5D dev 的 accepted complete-rep top 事件首/末召回约为 `21.88% / 12.50%`。T1-6B 的 top 状态 anchor 已达到 `81.25% / 78.13%`，说明状态区间和首尾窗口解决了主要的事件漏检问题。

该比较不是最终 rep 计数对照：T1-6B 输出的是 top 状态候选，尚未经过 T1-6C 的连续链、组外剔除和 set 选择。当前仍有 `36` 个全机位 FP 和 `22` 个 FN，尤其集中在 pushup；这些必须交给 T1-6C 处理，不能通过改动 anchor 或增加固定偏移掩盖。

本轮只在 Fit3D dev 上选择并冻结策略，未使用 Fit3D holdout 调参，也未使用 QEVD sealed holdout GT。跨数据源结论仍为 **not yet validated**。
