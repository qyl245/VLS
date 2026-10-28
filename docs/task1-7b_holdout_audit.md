# T1-7B Fit3D regression holdout 失败审计

状态：**审计完成（2026-10-27）**。本轮只审计冻结结果，不修改算法、
配置、阈值或 holdout 产物；T1-7B 继续保持 `SIGNAL_STAGE_NO_GO`。

## 1. 审计结论

本轮 holdout 失败不是单一阈值问题，至少包含以下相互独立的通用层缺陷：

1. 多尺度共识链在“净增 2 个 rep”时直接胜出，没有要求与单尺度链保留
   足够的事件重叠，导致共识链可以整体相位偏移后替换原链；
2. 端点跨度清理把“落在共识链跨度之外的未匹配事件”当作可删事件，没有
   要求显式的剪枝或端点时序证据，实际删掉了 GT 可匹配事件；
3. 相位信号选择偏重周期性与 SNR，缺少事件级左右侧/关节族一致性，
   在低帧率 pushup 上选中了结构更差的单侧信号；
4. 运动门的 `state_amplitude` 在降采样后过于脆弱，动态范围和
   dynamic-to-noise 均正常时仍可能返回 `no_motion`；
5. 半周期恢复使用硬边界，两个 near-boundary 案例分别落在上限外
   `0.05` 和 `0.013` 个周期，直接丢失 leading top。

同时，结构差异不能全部解释为精度回归。T1-6G 参考本身并非 GT：
部分 clip 的 T1-6G rep 数与 GT 明显不一致，部分 T1-7B 变化在 GT 对齐后
反而更好或至少不差。后续修复应先补不变量，再讨论阈值。

当前 holdout 已被观测，只能作为 regression set，不能再作为独立泛化证据。

## 2. 冻结执行与门禁

执行配置：

- `configs/task1_7_signal.json`
- SHA-256：`e4fc5abb95f561765e4add7c0e05d72c03ad4f190d08fec13dd0a425ed19bc49`
- 结果目录：`results/task1_7/signals/holdout/`
- `checksums.sha256`：`robustness.csv`、`signal_evidence.csv`、
  `summary.json` 全部校验通过。

holdout 输入 `32` 个 clip，质量门排除 `1` 个，进入结构评测 `31` 个。
排除项仍为 `fit3d_s09_pushup_50591643`：`pose_detection_rate=0.660432`、
`body_in_frame_rate=0.660432`，与 T1-6G 的排除项一致，不是 T1-7B 新引入
的覆盖率损失。

| 门 | 观测 | 目标 | 结果 |
|---|---:|---:|---|
| baseline structure match | `29/31` | `31/31` | fail |
| FPS 15 structure match | `28/31` | `>=30/31` | fail |
| FPS 60 structure match | `29/31` | `>=30/31` | fail |
| noise `sigma=2` structure match | `27/31` | `>=30/31` | fail |
| no regression vs T1-6G | `7` transforms regressed | `0` | fail |
| quality exclusion count | `1` | `1` | pass |

`SIGNAL_STAGE_NO_GO` 成立。FPS 15/60 的绝对门未通过；baseline、noise 和
no-regression 门也未通过，不能以 FPS 变换相对 T1-6G 有改善为由进入
T1-7C。

## 3. 失败清单

`robustness.csv` 中共有 `24` 条 `structure_match=false` 记录，集中在
`6` 个 clip。下表同时给出结构变化和 GT 对齐后的审计判断。

| clip | 受影响变换 | 结构变化 | GT 对齐审计 | 主要归因 |
|---|---|---:|---|---|
| `fit3d_s10_pushup_58860488` | baseline、amplitude 0.8/1.2、offset、flip、noise、missing 5%、idle | `1/5/5 -> 1/4/4` | **真实回归**：T1-6G 事件 `6 TP/0 FP/0 FN`，当前 consensus `5 TP/0 FP/1 FN` | 端点跨度清理 |
| `fit3d_s11_pushup_58860488` | baseline、amplitude 1.2、offset、flip、noise、missing 5%、idle | `1/3/3 -> 1/5/5` | **不能视为精度提升**：T1-6G `0 TP/4 FP/6 FN`，当前 `1 TP/5 FP/5 FN`；二者都严重偏离 GT | 共识仲裁 |
| `fit3d_s11_pushup_58860488` | amplitude 0.8 | `1/3/3 -> 1/6/6` | 同上；净增 3 rep 来自相位偏移后的共识段 | 共识仲裁 |
| `fit3d_s11_pushup_58860488` | FPS 15 | `1/3/3 -> 0/0/0` | **真实拒答**：运动证据正常，状态幅度边界失败 | 运动门采样敏感性 |
| `fit3d_s11_pushup_50591643` | FPS 15、FPS 60 | `1/5/5 -> 1/4/4` | **真实回归**：当前事件分别 `1 TP/4 FP/5 FN` 和 `0 TP/5 FP/6 FN` | 相位信号选择 |
| `fit3d_s11_pushup_50591643` | noise `sigma=2` | `1/5/5 -> 1/6/6` | **不是精度回归**：T1-6G `4 TP/2 FP/2 FN`，当前 `5 TP/2 FP/1 FN` | 参考基线差异 |
| `fit3d_s08_pushup_50591643` | FPS 15、FPS 60 | `1/7/7 -> 1/5/5` | **真实回归**：当前事件分别 `1 TP/5 FP/5 FN`、`2 TP/4 FP/4 FN` | 半周期硬边界 |
| `fit3d_s08_pushup_65906101` | missing 5% | `1/4/4 -> 1/5/5` | **不是桥接误报**：T1-6G `5 TP/0 FP/1 FN`，当前 `5 TP/1 FP/1 FN`；新增 leading `174` 与 GT `163` 相差 `11` 帧，事件门为 `19`；但尾端 `574` 丢失了参考中与 GT 可匹配的 `569` | 参考尾端差异，桥接本身基本正确 |
| `fit3d_s10_pushup_50591643` | noise `sigma=2` | `1/7/7 -> 1/6/6` | **不是精度回归**：T1-6G `4 TP/4 FP/2 FN`，当前 `4 TP/3 FP/2 FN` | 参考基线差异 |

表中的 GT 对齐使用与 T1-6 相同的一对一 Hungarian 事件匹配和
`0.25 x 中位 GT 间隔` 事件门；FPS 变换的 GT 帧按重采样后的帧数缩放。

## 4. 先区分“相对 T1-6G”与“相对 GT”

T1-7B 的 `structure_match` 只比较当前输出的 set/rep/default-rep 计数与
冻结 T1-6G 参考，不直接比较 GT。因此以下判断必须拆开：

- `structure_match=false`：当前结构与 T1-6G 不同；
- `regression vs GT`：当前事件、计数或边界比 T1-6G 更接近 GT；
- `reference artifact`：T1-6G 自身并不符合 GT，不能把差异自动算成退步。

直接证据：

- `fit3d_s11_pushup_58860488` 的 GT 为 `[283, 390, 489, 594, 692, 802]`；
  T1-6G 参考只有 `3` 个 rep，`4` 个 top event 为
  `[161, 254, 363, 462]`，全部落在 GT 起始段之前或边缘。当前 consensus
  变为 `1/5/5`，但 event 为 `[424, 530, 645, 737, 817, 905]`，仍只有
  `1` 个 GT 匹配。这里既不能把“+2 rep”视为真实改善，也不能把所有
  结构差异都归咎于 T1-7B。
- `fit3d_s11_pushup_50591643` 的 noise 变换从 `1/5/5` 变为 `1/6/6`，
  但 GT 匹配从 `4 TP/2 FP/2 FN` 改善为 `5 TP/2 FP/1 FN`。该结构变化
  不应出现在“真实精度回归”清单中。
- `fit3d_s10_pushup_50591643` 的 noise 变换从 `1/7/7` 变为 `1/6/6`，
  GT 匹配由 `4 TP/4 FP/2 FN` 变为 `4 TP/3 FP/2 FN`，FP 反而减少。

## 5. 根因

### 5.1 共识仲裁缺少事件重叠/保留不变量

代码位置：`scripts/task1_7_signals.py:1646`。

```python
if consensus_reps >= single_reps + 2:
    return consensus
```

该捷径只检查净 rep 数，不检查共识链是否保留单尺度链的事件，也不要求
新增事件来自同一时间邻域。实际后果是共识链可以整体移位并丢弃原链：

`fit3d_s11_pushup_58860488` baseline：

- 单尺度：`1/3/3`，events `[161, 254, 363, 462]`；
- 共识：`1/5/5`，events `[424, 530, 645, 737, 817, 905]`；
- 仲裁直接选择共识，尽管两组事件几乎不重叠。

这不是“多尺度补了两个 rep”，而是“换成了另一个后段链”。仅在
`+1 rep` 分支存在 `events_preserved` 检查，`+2 rep` 分支反而绕过了
不变量。

修复方向：所有净增分支都必须要求单尺度核心事件被保留，或对每个丢失
事件给出显式、可审计的替换/剪枝证据；不能只按 rep 数阈值胜负。

### 5.2 端点跨度清理接受了无证据的 leading 删除

代码位置：`scripts/task1_7_signals.py:1842`。

`endpoint_span_cleanup` 只要求未匹配事件位于共识事件跨度之外，没有要求
该事件出现在 `low_sampling_endpoint_pruned_events`、端点时序模型或其他
可信剪枝记录中。

`fit3d_s10_pushup_58860488` baseline：

- GT：`[250, 345, 444, 537, 644, 750]`；
- T1-6G 单尺度：`[237, 343, 446, 543, 648, 770]`，GT 匹配
  `6 TP/0 FP/0 FN`；
- T1-7B 单尺度：与 T1-6G 相同；
- T1-7B 共识：`[343, 446, 543, 648, 770]`，匹配
  `5 TP/0 FP/1 FN`；
- 仲裁选择共识，导致真实漏掉 `237`。

`237` 与 GT 首个事件 `250` 只差 `13` 帧，事件门为 `25` 帧。它不是可在
无证据情况下被删除的伪峰。

修复方向：端点跨度位置只能作为“候选解释”，不能单独成为删除理由；
必须要求显式剪枝/时序证据，并记录被删事件、原因和保留事件。

### 5.3 相位信号选择缺少事件级一致性

代码位置：`scripts/task1_7_signals.py:1416`。

候选选择是 `bilateral preferred pool`，但只有状态为 `ok/degraded` 的
双侧候选才进入 preferred pool。`fit3d_s11_pushup_50591643` 的双侧均值
候选并非被候选池代码漏掉，而是因轻微低于周期门被过滤：

| 变换 | 双侧 mean periodicity | 状态 | 左肘 periodicity | 右肘 periodicity | 选中 |
|---|---:|---|---:|---:|---|
| FPS 15 | `0.147406` | `insufficient_signal` | `0.351848` | `0.472798` | right |
| FPS 60 | `0.146357` | `insufficient_signal` | `0.308305` | `0.421072` | right |

最小值配置为 `0.15`。因而这里不是“bilateral pool 实现 bug”，而是：

1. bilateral mean 只差约 `0.003` 就过门；
2. 双侧候选失效后，排序主要由周期性和 SNR 驱动；
3. 右侧信号分数更高，但其 top 事件与 GT 明显不符。

GT 缩放后的直接检查：

- FPS 15，GT `[85, 117, 147, 178, 208, 241]`，事件门 `8`；
  选中右侧的最终 top 为 `[97, 125, 158, 187, 222]`，
  `1 TP/4 FP/5 FN`；
- FPS 60，GT `[340, 468, 587, 713, 831, 963]`，事件门 `32`；
  选中右侧的最终 top 为 `[384, 501, 635, 750, 892]`，
  `0 TP/5 FP/6 FN`；
- 同一候选的双侧 mean consensus top 在 FPS 15 包含
  `[87, 119, 149, 193, 223, ...]`，在 FPS 60 包含
  `[346, 478, 598, 772, 937, ...]`；与 GT 的匹配明显优于右侧。
  左肘候选的 raw top 在 FPS 15 还包含全部 6 个 GT 缩放事件，但直接
  只输入左肘时链未能成立，因此不能把“改选左侧”当作已完成的修复。

修复方向：对单侧候选增加事件级一致性证据，而不是只看波形周期性；
双侧候选接近门限时应同时保留其事件结构，不能用单一标量分数直接淘汰。
但不得针对该 holdout 样本降低全局 periodicity 门。

### 5.4 运动门对降采样后的状态幅度过敏

`fit3d_s11_pushup_58860488` FPS 15：

```text
status = no_motion
reasons = [state_amplitude_below_minimum]
state_amplitude = 2.417776  < 3.0
dynamic_range = 16.61054    >= 5.0
dynamic_to_noise = 14.152055 >= 8.0
periodicity = 0.479276
```

信号明显存在动态和周期性，拒答来自 `state_amplitude` 单指标。
在降采样和短平滑窗口组合下，平台幅度估计会比高帧率更脆弱。直接下调
全局 `3.0` 会同时放宽真正 idle 场景，不能作为修复。

修复方向：让运动判定感知分析 FPS/平滑尺度，或要求
`state_amplitude` 与 dynamic-range、periodicity、跨尺度一致性联合
决策；在 dev/train 或新 holdout 上验证。

### 5.5 半周期恢复硬边界

`fit3d_s08_pushup_50591643`：

- FPS 15：leading top `21` 到下一个 top `76`，gap `55`，period `22`，
  ratio `2.50`；配置上限 `2.45`；
- FPS 60：leading top `84` 到 `302`，gap `218`，period `88.5`，
  ratio `2.46`；配置上限 `2.45`。

两个案例都只差极小量，却因硬窗被排除。FPS 15 的 `21` 稳定性为 `0`，
不应无条件恢复；FPS 60 的 `84` 有尺度支持 `4`、稳定性 `1`，更接近可
恢复节点。说明边界本身不足以区分真假 leading partial。

修复方向：使用局部上下文、邻域稳定性和端点时序决定是否恢复；不要
只把 `2.45` 改成更大常数。当前 GT 事件匹配显示该 clip 的 FPS 15/60
确有真实事件损失，因此该问题应保留在修复队列。

### 5.6 局部峰/谷桥接：本次不是主要精度缺陷

`fit3d_s08_pushup_65906101` missing 5%：

- 参考结构 `1/4/4`，当前 consensus `1/5/5`；
- 新增 leading `174` 的局部相对幅度 `0.955448`、top stability `1.0`、
  bottom stability `1.0`、bottom scale support `4`、gap ratio `1.12`；
- GT 首事件为 `163`，事件门 `19`，`174` 是 GT 可匹配事件。

因此该 leading 新增本身不是误报。变化还包含尾端 `574` 取代参考
`569`，后者在 GT 对齐中少一个匹配。该案例更适合归入“尾端参考差异”，
而不是“桥接过度触发”。

不过，桥接目前没有显式 set-boundary 冲突门；如果后续在其他数据上
出现跨 set 误连接，应增加“加入后左右两侧是否同时独立成立”的证据检查。

### 5.7 dev 噪声修复没有迁移到 holdout

- `noise_sigma_2`：`local_extrema_stability_bridge` 触发 `0` 次；
- `missing_5pct`：触发 `2` 次：
  `fit3d_s08_pushup_65906101` 和 `fit3d_s10_pushup_60457274`。

dev 上由局部峰桥接修复的噪声失败型态，没有在 holdout 的主要噪声失败中
重现。这是明确的负向泛化结果，不能把 dev 的噪声改善当作机制已经泛化。

## 6. 修复优先级

先做不变量，后做阈值；所有修复都不得使用本次 holdout 调参。

1. **P0：共识事件保留不变量**
   - 所有净增分支都要求单尺度核心事件被保留，或逐个事件提供显式替换
     证据；禁止仅凭 `+2 rep` 直接选择共识。
2. **P0：端点清理证据化**
   - `endpoint_span_cleanup` 必须依赖显式剪枝/端点时序记录；删除事件
     必须写入 diagnostics，且不能在无证据时删掉 GT 可匹配事件。
3. **P0：相位候选事件级一致性**
   - 对左右侧和关节族增加事件匹配/拓扑一致性；bilateral 候选接近门限
     时不能只靠标量分数淘汰。修复在 dev/train 上验证。
4. **P1：采样感知运动门**
   - 用 FPS/尺度条件归一化 `state_amplitude`，或采用多证据联合判定；
     不直接下调全局阈值。
5. **P1：半周期局部上下文**
   - 用邻域周期、稳定性和端点时序处理 near-boundary，而不是扩大硬窗。
6. **P2：局部桥接 set-boundary 门**
   - 当前 holdout 没有证明 leading 桥接是主要精度缺陷；先保留机制，
     后续只在出现跨 set 误连接证据时增加冲突门。
7. **验收**
   - 只在 dev/train，或修复后新锁定的未见 holdout 上验证；当前 holdout
     已污染，不能再作为独立泛化结论。

## 7. 产物与复现

本次审计读取：

```text
configs/task1_7_signal.json
results/task1_7/signals/holdout/summary.json
results/task1_7/signals/holdout/robustness.csv
results/task1_7/signals/holdout/signal_evidence.csv
results/task1_6/evaluation_quality_motion/summary.json
results/task1_6/runs/quality_motion/fit3d_holdout/<clip_id>/analysis_v2.json
configs/task1_5_fit3d_clips.json
```

未修改任何代码、配置、冻结结果或 checksum。审计结论可复现依据：

- holdout checksum 全部通过；
- `scripts/task1_7_signals.py` 当前分支与冻结配置 hash 对应；
- 事件匹配使用 T1-6 的 Hungarian 匹配逻辑与 `0.25 x median interval`
  事件门；
- 结构计数使用 `set_count / rep_count / default_rep_count` 三元组。
