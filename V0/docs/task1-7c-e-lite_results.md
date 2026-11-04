# T1-7C-lite + T1-7E-lite：质量拒答与动作契约

状态：**已实现并完成 dev 回归（2026-10-27）**。

## 1. 执行结论

本轮按 `Plan/task1-7.md` 第 18 节冻结 T1-7B：研究阶段结论继续保留
`SIGNAL_STAGE_NO_GO`，干净输入基线标记为 `LIMITED/EXPERIMENTAL`，
不再针对已观测的合成 holdout 调参。产品第一版只接入最小可用的
C-lite 与 E-lite：

- C-lite 把既有质量门转成 clip/set/rep 证据，`warn` 真实参与置信度降权，
  `reject/unknown` 继续 fail-closed；
- E-lite 只接受 `squat` 和 `pushup`，检查相位信号族、完整 rep 数和周期，
  不匹配或未支持动作直接拒答；
- 低置信、相位歧义、动作歧义时输出 `abstained`，不静默生成高置信结果。

这不是完整的 T1-7C/E，也不是正式跨域 `GO`。本轮没有做风险-覆盖率研究、
完整 ROM/姿态检查、多视角融合或真实视频验收。

## 2. 实现内容

### 2.1 C-lite 连续质量证据

`scripts/task1_7_quality.py` 复用 T1-6 的 blocking 语义，输出：

- clip 级质量门证据；
- set 链状态证据；
- rep 状态与 `quality_flags` 证据；
- `c_lite_quality` 诊断，明确 `new_blocking_rules=false`。

`pass` 不改结构；`warn` 保留输出但降低置信度；`reject/unknown` 继续清除
set、rep 和事件结构，分别映射为 `rejected_quality` 和
`not_evaluable_quality`。本轮没有增加新的阻断阈值。

`scripts/task1_7_confidence.py` 继承 T1-6 冻结校准器，不重新拟合。降权规则：

- 顶层输出置信度使用 clip/set/rep 中最差的 `warn`，偏保守；
- rep 的局部降权只使用其所属 clip/set 与自身证据，单个 rep 警告不会直接
  改写其他 rep；
- set 级 `warn` 同时降低该 set 的 `chain_confidence` 和组内 rep；
- `quality=reject/unknown`、`phase_ambiguous`、`action=mismatch/unsupported`
  时明确 abstain。

批量构建器 `scripts/build_task1_7_lite.py` 可通过 `--gate-dir` 接入逐 clip
质量门，通过 `--resolution-dir` 接入相位候选结果。当前 dev v2 产物没有
独立 `quality_gate` 字段，因此本轮批量回归使用默认 `pass`，只验证 C-lite
证据与降权接线；真实视频链路必须显式传入姿态质量门，不能把默认 `pass`
当作质量验收。

### 2.2 E-lite 动作契约

`scripts/task1_7_actions.py` 使用数据集无关的动作注册表：

| 动作 | 允许相位族 | 最少完整 rep | 周期范围 |
| --- | --- | ---: | --- |
| squat | knee/hip flexion | 3 | 0.7–5.0 s |
| pushup | elbow flexion | 3 | 0.7–5.0 s |

验证结果只允许 `supported/mismatch/unsupported/ambiguous`。非 supported 时：

- `mismatch -> action_mismatch`；
- `unsupported -> unsupported_action`；
- `ambiguous -> no_confident_set`。

已有的质量拒绝和相位歧义优先，不被动作验证覆盖。E-lite 不按 clip ID、
受试者、相机或数据集做特判。

当前 E-lite 不能单独完成“热身动作识别”。如果热身动作恰好使用同一动作
且 rep 数足够，仍可能被当作正式组；第一版只能依靠 rep 数不足、周期异常、
相位不匹配或低置信 abstain 拦截。真正的热身/正式组区分需要后续动作阶段
或真实视频数据支持。

## 3. 冻结产物

| 产物 | 路径 | SHA-256 |
| --- | --- | --- |
| C-lite 配置 | `configs/task1_7_quality.json` | `7892bd8f15e0658bc69723523e0fc52f20fb49d132c8f7bea0aa7875d4a7900f` |
| C-lite 置信配置 | `configs/task1_7_confidence_frozen.json` | `4b55b9ca81453b49c6aa314990a2547269b48af4e6a8310ed1bfa32fa2ae42e5` |
| E-lite 配置 | `configs/task1_7_actions.json` | `f9bbcb294a0a8aaf15e01d28266cbe49c6b933468550a41ecf8f5415027f892c` |
| T1-6 校准器 | `configs/task1_6_confidence_frozen.json` | `795e24491dfec5e2f988f7a69abea626cbe7256295b226e17787ab04468bb612` |
| v3 schema | `schemas/task1_7_clip_analysis_v3.schema.json` | `03cb6466a1584c8a5075f064dd5979b4a0591e22baef2a7fd7c59515d969e281` |

T1-7B 冻结配置未改动：P0 为
`e4fc5abb95f561765e4add7c0e05d72c03ad4f190d08fec13dd0a425ed19bc49`，
P1 为 `177797161f15b7b326c7491d672fd613d95dda90e16c8d287d0fe5a89bc0ad27`。

## 4. dev 回归

批量输出目录：

```text
results/task1_7/lite/dev/fit3d/
results/task1_7/lite/dev/qevd/
```

每个目录包含逐 clip v3 JSON、`summary.json` 和 `checksums.sha256`。

| 数据 | clip | `ok` | `no_confident_set` | action supported | action ambiguous | abstained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit3D dev | 32 | 31 | 1 | 31 | 1 | 2 |
| QEVD dev | 13 | 11 | 2 | 11 | 2 | 2 |
| 合计 | 45 | 42 | 3 | 42 | 3 | 4 |

Fit3D：

- 质量状态为 `pass 4 / warn 28`，无 `reject/unknown`；
- 共 `50` 条 rep `warn`，全部来自 `endpoint_partial` 端点证据；
- `fit3d_s05_pushup_50591643` 没有完整 rep，动作验证为 `ambiguous`，按
  `no_confident_set` 拒答；
- `fit3d_s04_pushup_50591643` 动作匹配，但 rep 中位置信度约 `0.215`，
  叠加 set 级/端点 `warn` 后校准置信度低于报告门限，按 `low_confidence`
  拒答。

QEVD：

- 质量状态为 `pass 2 / warn 11`，无 `reject/unknown`；
- 共 `20` 条 rep `warn`，全部来自 `endpoint_partial`；
- `0005_pushup_set01` 和 `0161_pushup_set01` 的 T1-6 回退 rep 使用
  `legacy_t1_4`，缺少可验证相位族，E-lite 按 `action_ambiguous` 拒答。

`warn` 不删除 set/rep，只降低输出置信度；上述拒答均由无完整 rep、
低置信或动作/相位不可判定触发，不是新增的质量阻断阈值。

## 5. 验证

- C/E-lite 定向测试：`32 passed`；
- 全量测试（排除已知 T1-4 TensorFlow/protobuf 环境阻塞）：
  `249 passed, 4 subtests passed`；
- 两个 dev 输出目录的 `sha256sum -c checksums.sha256` 均通过；
- v3 批量输出全部通过 `task1_7_clip_analysis_v3.schema.json`；
- `git diff --check` 通过。

复现命令：

```bash
cd /mnt/storage2/data_guest/VLS/V0

/mnt/storage2/data_guest/.venv/bin/python scripts/build_task1_7_lite.py \
  --analysis-dir results/task1_6/dev/clip_analysis_v2 \
  --clips configs/task1_5_fit3d_clips.json \
  --output-dir results/task1_7/lite/dev/fit3d

/mnt/storage2/data_guest/.venv/bin/python scripts/build_task1_7_lite.py \
  --analysis-dir results/task1_6/dev/qevd_clip_analysis_v2 \
  --clips configs/task1_4_clips.json \
  --output-dir results/task1_7/lite/dev/qevd
```

## 6. 后续边界

- 下一步是真实手机视频的 G-lite 验收，不继续追合成 holdout 阈值；
- 真实视频出现首尾 rep 偏移、set 串联或 top-to-top 不稳定时，再触发 D-lite；
- 只有真实数据证明需要时，才补完整 ROM/姿态、热身检测和风险-覆盖率研究；
- 当前结果只能表述为 C/E-lite `LIMITED/EXPERIMENTAL` 可用，不能表述为
  正式跨域 `GO`。
