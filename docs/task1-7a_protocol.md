# T1-7A 协议、基线与 RepChain v3 Schema

状态：**已完成（2026-10-27）**；T1-7A 只冻结协议、基线、split 和 schema，不改变 RepChain 算法输出。

## 1. 本阶段目标

T1-7A 为后续 T1-7B～G 建立不可漂移的比较基线和工作契约：

- 固定 T1-6G 的 raw/gated 指标、质量分布、鲁棒性和 sealed 状态；
- 锁定 Fit3D dev / regression holdout / smoke 与 QEVD sealed 清单；
- 冻结 RepChain v3 analysis schema；
- 明确时间基、质量证据、动作验证、置信度和 abstention 契约；
- 建立可由 checksum 验证的 manifest；
- 不在本阶段修改 T1-6 冻结产物或启动算法重跑。

## 2. 冻结基线

基线来源：

```text
results/task1_6/evaluation_quality_motion/summary.json
```

基线决策仍为 `NO_GO`，锁定值包括：

| 指标 | T1-6G gated 基线 |
|---|---:|
| Fit3D clip / cluster | `32 / 8` |
| reportable clip coverage | `31/32 = 0.96875` |
| 全机位 event recall / precision | `0.901042 / 0.935135` |
| 全机位高置信事件 FP | `12` |
| 主相机事件中位 / p90 | `6.0 / 16.8` 帧 |
| 主相机高置信事件 FP | `0` |
| count consensus MAE / exact | `0.0 / 8` |
| complete rep 高置信 FP | `2` |
| top-to-top interval median | `5.0` 帧 |
| 质量分布 | `pass 17 / warn 14 / reject 1` |
| idle 负控 | `5/5` |
| QEVD reportable segment | `3/6` |
| QEVD sealed annotation ready | `0/6` |

若后续产生新的基线，不得覆盖本阶段快照，应新建独立 baseline id。

## 3. Split Lock

Fit3D：

| split | subject | clip | cluster | 用途 |
|---|---|---:|---:|---|
| dev | `s03/s04/s05/s07` | `32` | `8` | 调参与冻结 |
| regression holdout | `s08/s09/s10/s11` | `32` | `8` | 仅回归，不再视为独立 holdout |
| smoke | `s02/s12/s13` | `6` | - | 不进入正式错误统计 |

QEVD sealed：

- `6` 个 segment；
- `0` 个 segment 已完成人工 GT；
- 正式 count/boundary/F1/high-confidence-FP 门继续为 `not_evaluable`；
- 在 GT 完成前，任何结果只能标记为跨源诊断，不能宣称跨域验收通过。

## 4. RepChain v3 Schema

新增：

```text
schemas/task1_7_clip_analysis_v3.schema.json
```

顶层必需字段：

```text
schema_version
clip_id
algorithm
status
timebase
signal_evidence
phase_hypotheses
action_verification
quality_evidence
confidence
sets
rejected_candidates
non_rep_intervals
```

关键契约：

- `timebase` 记录源/有效 FPS、时间戳、可变采样、最大缺口和插值比例；
- `signal_evidence` 保存 Top-K 候选、共识分数和 `ambiguous`；
- `phase_hypotheses` 允许多个相位假设，不要求强制选单一信号；
- `action_verification` 只允许 `supported/mismatch/unsupported/ambiguous`；
- `quality_evidence` 支持 `clip/set/rep/frame` 作用域及时间区间；
- `confidence` 必须包含组件分数、校准后的三层置信和是否 abstain；
- rep/set 保留帧号，并允许记录秒时间、端点证据和质量证据；
- `diagnostics` 保持开放，便于失败审计，但不能替代正式字段。

## 5. 通用层约束

协议明确禁止核心使用：

- `dataset_name/source_id/video_id/subject_id/camera_id/clip_id`；
- 固定 rep 数、固定帧偏移、固定平台宽度；
- 相机顺序或相机中位数作为核心逻辑；
- per-dataset threshold；
- sealed holdout GT 调参；
- 通过增加排除率改善 headline 指标。

多视角融合只能在核心外进行。单相机路径必须独立可运行。

## 6. 产物与 Checksum

```text
configs/task1_7_protocol.json
schemas/task1_7_clip_analysis_v3.schema.json
results/task1_7/manifest/baseline_t1_6g.json
results/task1_7/manifest/sources.json
results/task1_7/manifest/split_lock.json
results/task1_7/manifest/validation.json
results/task1_7/manifest/checksums.sha256
```

当前冻结 hash：

```text
protocol  505a64bba0df2b7c80215b7def0bf98095e16f429fe81fbcbee633e07877f39f
schema    03cb6466a1584c8a5075f064dd5979b4a0591e22baef2a7fd7c59515d969e281
baseline  1ae60554136627d4e6414fc5ed01a40b1fd44cb63088792a7e259ec094e37d59
split      b85caa8f279b58b2c72c688802aadfb72466989b82f45dfce77fb1b01643f579
```

`checksums.sha256` 覆盖 manifest 下 4 个 JSON 文件；协议和 schema 的 hash 同时写入 `validation.json` 和协议本体。

## 7. 复现命令

```bash
cd /mnt/storage2/data_guest/VLS/V0

/mnt/storage2/data_guest/.venv/bin/python scripts/analyze_clip_v3.py \
  --build-protocol \
  --baseline-dir results/task1_6/evaluation_quality_motion \
  --output-dir results/task1_7/manifest \
  --config-output configs/task1_7_protocol.json

cd results/task1_7/manifest
sha256sum -c checksums.sha256
```

如需固定时间戳做确定性比较，可传入 `--generated-at`。时间戳不参与指标或 split 判断。

## 8. 验证结果

- 基线 25 项关键值全部匹配；
- 3 个源 manifest hash 全部匹配；
- Fit3D dev `32`、regression holdout `32`、smoke `6`，subjects 互斥；
- QEVD sealed `6` 段，annotation ready `0`；
- v3 schema 通过 Draft 2020-12 元 schema 校验；
- manifest 4 个 JSON 的 checksum 全部通过；
- T1-7A 定向测试 `6 passed`；排除既有 TensorFlow/protobuf 环境阻塞的 T1-4 两个模块后，全量测试 `157 passed + 4 subtests`。

## 9. 下一阶段

进入 T1-7B 前必须保持：

- `configs/task1_7_protocol.json`、baseline、split lock 和 schema 不再无记录地修改；
- 所有算法改动写入新的 T1-7 run 目录；
- Fit3D regression holdout 不参与新阈值选择；
- QEVD sealed GT 在最终运行前保持不可见。

T1-7B 的范围是真实时间基、多尺度信号、短缺口处理和相位候选共识，不在本阶段实现。
