# task1-7B P1：采样感知运动门与半周期局部上下文

## 1. 执行结论

P1 已实现并完成 dev/holdout 回归，时间 2026-10-27。冻结 P0 配置和
冻结产物未修改；P1 使用独立配置
`configs/task1_7_signal_p1.json`（SHA-256
`177797161f15b7b326c7491d672fd613d95dda90e16c8d287d0fe5a89bc0ad27`）。

T1-7B 阶段决策仍为 **`SIGNAL_STAGE_NO_GO`**。P1 解决了
`fit3d_s11_pushup_58860488` 的 FPS 15 运动门误拒答，并把
`fit3d_s08_pushup_50591643` FPS 60 从 `1/5/5` 恢复到与参考结构一致的
`1/7/7`；但 holdout FPS 15 仍为 `29/31`，noise σ=2 仍为 `28/31`，
且相对 T1-6G 仍有 `2` 个变换结构变化。研究阶段因此不进入完整的
T1-7C；产品第一版按 `Plan/task1-7.md` 第 18 节冻结为干净输入的
`LIMITED/EXPERIMENTAL` 基线，先做真实视频端到端测试。

当前 holdout 在 T1-7B 审计中已经观测，本轮的 holdout 结果只作为
regression 证据，不再作为独立泛化证据。P1 策略不包含 clip ID、相机、
数据集或固定 rep 数条件；后续仍应使用新的未见 holdout 或 train/dev
验证泛化。

## 2. 实现内容

### 2.1 采样感知运动门

`scripts/task1_6_phase_states.py` 和 `scripts/task1_7_signals.py` 增加
联合运动门。只有当运动拒答的**唯一**原因是
`state_amplitude_below_minimum` 时才评估 override，其他 dynamic-range、
SNR 或有效比例问题不会被覆盖。P1 配置要求：

- 分析 FPS `<= 20`；
- `state_amplitude` 至少为全局门限的 `0.75` 倍；
- dynamic range 至少为全局门限的 `2.0` 倍；
- dynamic-to-noise 至少为全局门限的 `1.5` 倍；
- periodicity 至少为 `0.35`。

override 证据写入 `motion_evidence.joint_state_amplitude`，包含每个检查
项的布尔值、分析 FPS、periodicity 和 `override_applied`。默认 detector
仍为 `enabled=false`，只有 P1 配置显式启用。

### 2.2 近边界拒峰的多尺度局部恢复

`detect_phase_states()` 保留因 `no_stable_top_state` 未进入候选的
interior top，并记录峰值相对幅度。`build_multiscale_state_consensus()`
把这些状态带入 `rejected_top_states`。P1 只在以下局部证据同时成立时恢复
一个 top：

- 相邻 top 的 gap ratio 在 `2.45–2.65` 周期；
- 同一锚点附近至少有 `3` 个平滑尺度支持；
- 拒峰 relative peak fraction `>= 0.4`；
- 拒峰 stability `>= 0.5`；
- 跨尺度锚点离散不超过 `0.2` 周期；
- 左右相邻 gap ratio 仍在 `0.5–1.5` 周期；
- 恢复置信度达到既有 half-cycle recovery 门限。

恢复事件写入 `near_boundary_rejected_top`、`near_boundary_gap_ratio`、
左右 gap ratio、锚点离散和相对峰值证据。`select_analysis_hypothesis()`
在 `+1 rep` 仲裁中显式接受该恢复来源，但仍要求单尺度事件保留和置信度
不退化。

### 2.3 证据列

`robustness.csv` 新增：

- `near_boundary_recovered_top_count`
- `motion_joint_override_count`

## 3. 回归结果

使用的命令：

```bash
/mnt/storage2/data_guest/.venv/bin/python scripts/task1_7_signals.py \
  --evaluate --split dev \
  --config configs/task1_7_signal_p1.json \
  --output-dir results/task1_7/signals_p1/dev

/mnt/storage2/data_guest/.venv/bin/python scripts/task1_7_signals.py \
  --evaluate --split holdout \
  --config configs/task1_7_signal_p1.json \
  --output-dir results/task1_7/signals_p1/holdout
```

| split | baseline | FPS 15 | FPS 60 | noise σ=2 | quality exclusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | `32/32` | `32/32` | `28/32` | `30/32` | `0` |
| holdout | `31/31` | `29/31` | `30/31` | `28/31` | `1`（未变） |

dev 的失败集合与 P0 完全一致，P1 两条机制在 dev 上均未触发；holdout
相对 P0 的 FPS 60 从 `29/31` 提升到 `30/31`，`stage_passed` 仍为
`false`。holdout 的 `2` 个 regression transform 仍是
`amplitude_scale_1.2` 和 `idle_pad_2s`。

## 4. 触发案例

- `fit3d_s11_pushup_58860488` FPS 15：运动门在 `3` 个尺度上应用
  override，从 P0 的 `no_motion` 恢复为 `1/3/3`，与参考结构一致。
- `fit3d_s08_pushup_50591643` FPS 60：近边界恢复补回拒峰，输出
  `1/7/7`，与参考结构一致。
- `fit3d_s08_pushup_50591643` FPS 15：近边界恢复补回一个中间 top，
  输出从 `1/5/5` 提升到 `1/6/6`，但仍比参考少 `1` rep。leading top
  `21` 的 top stability 为 `0`，虽然局部上下文稳定、相对幅度充足，
  仍被既有低采样端点剪枝规则移除。该案例保留 fail-closed，不通过
  放宽稳定性门或写死 clip 条件强行恢复。

其余 holdout 失败仍来自 P0 已记录的相位歧义、噪声链拓扑和缺失缺口，
没有证据表明 P1 引入了新的广泛回归。

## 5. 校验

- 定向测试：`tests/test_task1_7_signals.py`，`56 passed`。
- 全量测试（排除已知 T1-4 TensorFlow/protobuf 环境问题）：
  `217 passed, 4 subtests passed`。
- `git diff --check`：通过。
- 冻结 P0 配置 SHA-256：
  `e4fc5abb95f561765e4add7c0e05d72c03ad4f190d08fec13dd0a425ed19bc49`，
  未变化。
- `results/task1_7/signals/dev`、`results/task1_7/signals/holdout` 和
  `results/task1_7/manifest` 的 checksum 全部通过；P1 产物写入独立目录
  `results/task1_7/signals_p1/{dev,holdout}`，不覆盖冻结结果。

## 6. 结论与下一步

P1 的两条通用机制已落地，并验证了低采样运动门和小幅近边界拒峰的收益；
但 T1-7B 的绝对结构门仍未通过，研究阶段保持 `NO_GO`。产品路线不再继续
追该合成 holdout：T1-7B 只作为干净输入 `LIMITED/EXPERIMENTAL` 基线，
先做 C-lite/E-lite 和真实视频验收。剩余 `s08_pushup_50591643` FPS 15
的 leading top 暂不继续修；只有真实视频暴露同类端点问题，再按新的未见
数据验证“低采样端点上下文保留”，不扩大 near-boundary 硬窗。
