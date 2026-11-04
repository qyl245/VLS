# T1-7B P0 不变量修复记录

状态：**P0 不变量实现与回归完成（2026-10-27）**。冻结配置未修改，
T1-7B 仍为 `SIGNAL_STAGE_NO_GO`，不进入 T1-7C。

## 1. 修复范围

本轮只处理 holdout 审计确认的三个通用层 P0 缺陷，不调整阈值，不读取
数据集、相机或 clip ID 作为策略条件。

1. **共识事件保留不变量**
   - `consensus_reps >= single_reps + 2` 不再直接选择共识；
   - 共识必须保留全部单尺度事件，或至少与一个单尺度事件重叠且所有
     单尺度事件仍落在共识事件跨度内；
   - 整体相位偏移、与原链几乎不重叠的共识仍回退到单尺度结果。
2. **端点删除证据化**
   - `endpoint_span_cleanup` 删除的每个事件都必须出现在
     `low_sampling_endpoint_pruned_events` 或
     `endpoint_timing_pruned_events`；
   - 仅因事件落在共识跨度外而删除、但没有显式剪枝证据时，仲裁拒绝
     共识结果。
3. **相位候选事件级一致性**
   - 双侧候选作为聚合信号使用；
   - 单侧候选必须由同族双侧候选在其超过半数的 interior 事件锚点上
     支持；
   - 事件级冲突返回 `phase_signal_event_disagreement`，直接短路为
     `phase_ambiguous` / `0 set, 0 rep`，不再仅凭标量分数选中单侧信号。

实现位置：

- `scripts/task1_7_signals.py`：`select_analysis_hypothesis()`、
  `_event_consensus_evidence()`、`_explicit_endpoint_pruned_frames()`
  及事件匹配辅助函数；
- `tests/test_task1_7_signals.py`：覆盖错位共识拒绝、合法跨度增益、
  无证据端点删除拒绝、显式剪枝接受和事件级相位冲突。

## 2. 审计案例回归

使用冻结配置
`e4fc5abb95f561765e4add7c0e05d72c03ad4f190d08fec13dd0a425ed19bc49`
在 `/tmp` 独立目录重跑，未覆盖 `results/task1_7/signals/` 下的冻结
产物。

- `fit3d_s10_pushup_58860488` baseline 保持 `1/5/5`，不再因无证据的
  端点跨度清理删除事件 `237`；
- `fit3d_s11_pushup_58860488` baseline 从错位共识的 `1/5/5` 回退为
  单尺度 `1/3/3`，不再把整体相位偏移当作净增 rep；
- `fit3d_s11_pushup_50591643` FPS 15/60 因单侧右肘候选与同族双侧事件
  冲突，输出 `phase_ambiguous`，不再接受结构更差的单侧链；
- `fit3d_s08_pushup_50591643` 的 FPS 15/60 半周期漏检仍为 P1；
- `fit3d_s11_pushup_58860488` 的 FPS 15 `no_motion` 仍为 P1 运动门问题。

## 3. 回归结果

Fit3D dev：

| 变换 | 不变量修复后 | 修复前 | 说明 |
|---|---:|---:|---|
| baseline | `32/32` | `32/32` | 保持 |
| FPS 15 | `32/32` | `32/32` | 保持 |
| FPS 60 | `28/32` | `30/32` | 两个共识结果因缺少保留/剪枝证据回退为单尺度 |
| noise σ=2 | `30/32` | `30/32` | 保持 |
| quality exclusion | `0` | `0` | 保持 |

FPS 60 新增的差异为 `fit3d_s03_pushup_58860488` 和
`fit3d_s07_squat_60457274`。这是 fail-closed 结果：当前证据不支持共识
替换，不应通过放宽仲裁门恢复旧计数。

Fit3D regression holdout（已观测，只作回归检查）：

| 门 | 不变量修复后 | 修复前 |
|---|---:|---:|
| baseline structure match | `31/31` | `29/31` |
| FPS 15 structure match | `28/31` | `28/31` |
| FPS 60 structure match | `29/31` | `29/31` |
| noise σ=2 structure match | `28/31` | `27/31` |
| no regression vs T1-6G | 2 个变换 | 7 个变换 |
| quality exclusion | `1` | `1` |

剩余 no-regression 差异为 `amplitude_scale_1.2` 和 `idle_pad_2s`；FPS
15/60 与 noise 的绝对结构门仍未达到 `>=30/31`，因此阶段决策保持
`NO_GO`。

## 4. 校验

- `pytest tests/test_task1_7_signals.py -q`：`51 passed`；
- 全量测试排除已知 TensorFlow/protobuf 环境问题的 T1-4
  clip/report 两个模块：`212 passed, 4 subtests passed`；
- `configs/task1_7_signal.json` SHA-256 保持不变；
- 未修改冻结配置、阈值、holdout 产物或 `Plan/方案.md`。

## 5. 下一步

P0 不变量已收口，后续按审计顺序处理 P1：

1. 采样感知运动门，解决低帧率 `no_motion`；
2. 半周期恢复的局部上下文，替代当前接近硬边界的判定；
3. 仅在出现跨 set 误连接证据后，再评估局部桥接的 set-boundary 约束。

当前 holdout 不用于调参；修复应在 dev/train 或新锁定的未见 holdout
上验证。
