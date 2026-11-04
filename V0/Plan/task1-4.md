# task1-4：离线训练片段评估与可视化反馈（深蹲 / 俯卧撑）

## 1. 任务定位

本任务面向真实使用方式：用户完成一组动作后，摄像头或手机录下一段包含若干个 rep 的 clip；算法在 clip 结束后离线分析，生成可回放的训练报告。初版不要求实时反馈，也不要求训练过程中触发语音。

核心产物不是单纯 rep 计数，而是：

```text
clip → rep/相位 → 结构化指标 → 问题卡 → 回放定位到证据帧的图文报告
```

本任务完成后，应能对一个输入 clip 输出：

- 动作类型和动作窗口；
- rep 起止、top/bottom、向心/离心相位；
- 每个 rep 的 ROM、对称性、躯干/身体直线、节奏和稳定性指标；
- 按严重度排序的问题卡；
- 每张问题卡对应的 `anchor_frame`、证据区间、圈画目标和解释文字；
- 可直接在浏览器中回放并跳转到问题帧的 HTML 报告。

## 2. 范围与非目标

### 2.1 V0.5 范围

- 动作：`squat`（纯深蹲）和 `pushup`（标准俯卧撑）。
- 单个 clip 默认只包含一个动作、一组训练；允许组内有短暂休息。
- 初版允许用户在输入时指定 `exercise` 和 `view`，自动动作识别作为并行能力接入。
- 允许单人、固定摄像头、全身可见的室内场景。
- 数据格式优先复用 task1-1/task1-2/task1-3 的 MediaPipe、角度和 rep 输出。
- 离线处理，不限制分析延迟；一组 30–60 秒视频应在分钟级完成。

### 2.2 非目标

- 不做实时逐帧反馈、TTS 播报和训练中打断。
- 不覆盖深蹲跳、深蹲踢、负重深蹲、器械动作或多人场景。
- 不在本任务中训练完整的 148 类错误分类器。
- 不做医学诊断或康复处方；反馈只描述可观测的动作偏差。
- 不在低置信或视角不支持的指标上强行给出结论。

## 3. 数据准备与划分

### 3.1 可用数据

当前 QEVD-FIT-COACH 长视频中，按动作文本可筛出：

| 动作 | 纯动作段 | 视频数 | 备注 |
|---|---:|---:|---|
| `squat` | 14 | 14 | 仅使用 `action_display == "squats"` |
| `pushup` | 27 | 27 | 直接使用 |
| 其他动作段 | 853 | 149 个视频 | 作为动作识别负样本/其他类 |

注意：当前 `segment_reps.py` 的子串别名会把 `squat jumps` 和 `squat kicks` 也归为 `squat`。task1-4 必须按原始 `action_display` 过滤，或显式增加 `squat_jump`、`squat_kick` 类别，不能直接使用现有的全部 `action=squat` 段。

### 3.2 建议划分

按视频划分，禁止同一视频跨训练/验证/测试：

| 用途 | 深蹲 | 俯卧撑 | 说明 |
|---|---:|---:|---|
| 开发与阈值标定 | 5 段 | 8 段 | 只用于规则和模板调整 |
| 留出测试 | 5 段 | 8 段 | 冻结规则后只评一次 |
| 其他/动作识别负样本 | 其余动作段 | 其余动作段 | 按视频隔离 |

初版不要求所有 QEVD 视频都重新提姿态；先选 10 段深蹲 + 16 段俯卧撑完成端到端和质量验收。若需要动作识别，再从剩余数据补负样本。

### 3.3 新标注

每段留出 clip 至少标注：

- rep 起止帧；
- 每个 rep 的 `bottom_frame`；
- 1–3 个问题类型及严重度；
- 每个问题的代表证据帧或证据区间；
- 标注是否受视角、遮挡或置信度限制。

问题级标注优先覆盖深蹲和俯卧撑各 5 段，先建立小规模高一致性的评测集，不追求一次标满 41 段。

## 4. 输入与输出接口

### 4.1 输入 clip manifest

```json
{
  "schema_version": "task1-4-v1",
  "clip_id": "0153_squat_set01",
  "video_path": "VLS/dataset/QEVD-FIT/QEVD-FIT-COACH/long_range_videos/0153.mp4",
  "trim": {"start_frame": 1327, "end_frame": 1852},
  "exercise_hint": "squat",
  "view": "oblique_left",
  "fps": 30,
  "subject_id": "optional",
  "notes": "one set, no camera motion"
}
```

`exercise_hint` 和 `view` 在初版由用户或上层业务提供。`exercise_hint` 缺失时才调用动作识别；`view` 缺失时，视角依赖指标标为不可评估。

### 4.2 单 clip 输出目录

```text
VLS/V0/results/clip_analysis_v1/<clip_id>/
├── analysis.json              # 完整机读结果
├── clip.json                  # 输入快照和参数
├── quality.json               # 质量门控与拒答原因
├── exercise.json              # 动作标签、置信度和来源
├── reps.csv                   # rep、相位和置信度
├── phases.csv                 # 逐 rep 相位边界
├── features.csv               # 逐 rep 结构化指标
├── issues.json                # 问题卡及证据定位
├── report.html                # 可回放报告
├── annotated_frames/          # 代表帧圈画 PNG
└── checks/
    ├── schema_validation.json
    └── regression.json
```

## 5. 实现步骤

### 步骤一：clip 预处理与姿态/角度适配

新增 `scripts/analyze_clip.py` 作为编排入口。复用已有模块，不复制核心算法：

1. 读取 clip manifest，校验视频、trim、fps、动作和视角字段。
2. 抽取 clip 区间，或把 trim 信息传给下游模块。
3. 调用 task1-1 的 MediaPipe Pose Landmarker 输出逐帧 33 点世界坐标和 visibility。
4. 调用 task1-2 的角度提取逻辑，输出 `angles_wide.csv`、`angles.csv` 和 meta。
5. 保留 `source="vision"`，为未来 IMU/EMG 追加留出同构字段。

实现约束：

- 不覆盖 `outputs/pose_landmarker_full/`、`outputs/angles/` 和 `results/segmentation_final/`。
- 新结果统一写入 `results/clip_analysis_v1/<clip_id>/`。
- clip 内帧号必须映射回原视频帧号；报告和证据定位以原视频帧号为准。

### 步骤二：质量门控

在分割和评估前输出 `quality.json`。以下任一条件失败时，只生成拒答报告，不生成问题卡：

- 视频解码失败或有效帧比例过低；
- 检测到多人或主体频繁离开画面；
- 全身关键点缺失比例超过阈值；
- 有效角度覆盖率低于阈值；
- clip 中少于 3 个完整 rep；
- 视角不支持当前指标的判断。

建议初始阈值：

| 检查 | 初始阈值 | 失败处理 |
|---|---:|---|
| 姿态检出率 | `< 85%` | `quality_failed` |
| 关键关节平均 visibility | `< 0.50` | 降低 confidence 或拒答 |
| 有效主角度覆盖率 | `< 60%` | 不分割 |
| 完整 rep 数 | `< 3` | 不评估，建议重录 |
| 视频分辨率短边 | `< 480 px` | 警告，视角指标降级 |
| 人物框边缘距离 | 过近边缘 | 警告，相关关节不评 |

质量门控必须区分“动作错误”和“看不见/拍不清”。后者不能进入问题卡。

### 步骤三：动作窗口与动作识别

由于输入 clip 默认只包含一个动作，V0.5 不做实时动作开始检测，只做 clip 级动作归一化：

1. 若 `exercise_hint` 存在，标签来源记为 `user`，但仍运行分类器做一致性检查。
2. 若缺失，则在 `squat / pushup / other` 三分类中预测。
3. 若置信度低于阈值，报告显示“无法确认动作”，停止动作特定评估。
4. 检测组内低活动段，复用当前 active-bout 逻辑去除休息和过渡。

建议基线：

- 规则特征：主角度、周期、左右相位、躯干角度、上下肢参与度；
- 轻量分类器：对 clip 级角度序列提取统计特征，使用逻辑回归/随机森林；
- 预留 TCN/Transformer 接口，但不在第一轮直接依赖深度模型。

验收不把“用户已指定动作”算作自动识别成功；动作识别指标只在缺少 `exercise_hint` 的留出集上报告。

### 步骤四：rep 分割与相位细化

复用 task1-3 的 AIFit 两阶段和 active-bout 逻辑，但增加 clip 适配器：

```text
clip angles → active bouts → AIFit rep boundaries → phase landmarks → rep records
```

要求：

- `rep_id` 在 clip 内连续编号；
- 输出 `start_frame/end_frame`、`start_time_s/end_time_s`；
- 输出 `top_frame`、`bottom_frame`、`concentric_start/end`、`eccentric_start/end`；
- 输出 `rep_unit`、`phase_landmark`、`phase_polarity`、`confidence`、`quality_flags`；
- 少于 3 个完整 rep 时不做组级节奏和一致性问题判断。

相位定义：

| 动作 | top | bottom |
|---|---|---|
| 深蹲 | 站立、膝接近伸直、rep 边界 | 膝屈曲最大处，即膝关节角最小处 |
| 俯卧撑 | 手臂伸直、rep 边界 | 肘屈曲最大处，即肘关节角最小处 |

注意角度命名：`*_knee_flexion` 和 `*_elbow_flexion` 存的是关节夹角，伸直接近 180°，所以“屈曲最大”对应角度值最小。派生指标统一显式使用 `flexion_amount = 180 - joint_angle`，避免符号混淆。

### 步骤五：结构化指标

逐 rep 计算以下指标，写入 `features.csv`。每个指标必须带 `unit/confidence/source/view_requirement`。

#### 深蹲

| 指标 | 计算 | 用途 |
|---|---|---|
| `knee_flexion_rom` | 每侧 `max(180-膝角)-min(180-膝角)` | 下蹲深度/ROM |
| `bottom_knee_angle` | `bottom_frame` 的左右膝角 | 可读的底部角度 |
| `hip_flexion_rom` | 每侧矢状髋屈曲 ROM | 髋参与度 |
| `torso_lean_bottom` | `bottom_frame` 的 `torso_flexion` | 躯干前倾 |
| `knee_valgus_bottom` | 冠状面膝外翻角 | 膝盖内扣，需正面/斜前视角 |
| `knee_asymmetry_bottom` | 左右膝角差 | 左右不对称 |
| `hip_asymmetry_bottom` | 左右矢状髋角差 | 髋部代偿 |
| `rep_duration_s` | `(end_frame-start_frame)/fps` | 节奏 |
| `rep_duration_cv` | 组内 rep 时长变异系数 | 节奏一致性 |
| `spine_stability_bottom` | 底部窗口的躯干方差 | 稳定性 |
| `hip_depth_ratio` | 站立到最低点髋高下降/站立髋高 | 辅助深度指标，低置信 |

#### 俯卧撑

| 指标 | 计算 | 用途 |
|---|---|---|
| `elbow_flexion_rom` | 每侧 `max(180-肘角)-min(180-肘角)` | 下放深度/ROM |
| `bottom_elbow_angle` | `bottom_frame` 的左右肘角 | 可读的底部角度 |
| `body_line_angle` | 肩-髋-踝在矢状面的夹角，伸直接近 180° | 身体直线 |
| `hip_sag_ratio` | 髋偏离肩-踝连线距离/躯干长度 | 塌腰/撅臀 |
| `elbow_asymmetry_bottom` | 左右肘角差 | 左右不对称 |
| `shoulder_asymmetry_bottom` | 左右肩位置/角度差 | 代偿，低置信 |
| `rep_duration_s` | 每个 rep 时长 | 节奏 |
| `rep_duration_cv` | 组内 rep 时长变异系数 | 节奏一致性 |
| `spine_stability_bottom` | 底部窗口的躯干方差 | 稳定性 |

所有指标同时写入 `raw_value`、`reference_value`、`deviation`、`severity_candidate`。没有参考模板时，`reference_value` 为空，改用配置中的 provisional 阈值。

### 步骤六：问题判定

第一阶段使用可解释规则和教练参考 clip，不训练错误多分类器。

#### 6.1 参考模式

规则优先级：

1. 有同动作、同视角的教练参考 clip：使用相位对齐后的参考值，偏差阈值由配置给出；
2. 无参考 clip：使用 provisional 绝对阈值；
3. 视角或置信度不支持：不生成高/中严重度问题。

初版每个动作至少准备 1 条正面/斜前参考和 1 条侧面参考。参考值只用于动作形态，不作为医学标准。

#### 6.2 问题类型

| 动作 | issue_code | 主要证据 | 视角限制 |
|---|---|---|---|
| 深蹲 | `squat_depth_insufficient` | `knee_flexion_rom`、`hip_depth_ratio` | 侧/斜侧 |
| 深蹲 | `squat_torso_lean_excess` | `torso_lean_bottom` | 侧/斜侧 |
| 深蹲 | `squat_knee_valgus` | `knee_valgus_bottom` | 正面/斜前，否则禁用 |
| 深蹲 | `squat_asymmetry` | 膝/髋左右差 | 正面/斜前 |
| 深蹲 | `squat_tempo_inconsistent` | `rep_duration_cv`、时长序列 | 任意 |
| 深蹲 | `squat_stability_low` | `spine_stability_bottom` | 任意，需足够 visibility |
| 俯卧撑 | `pushup_depth_insufficient` | `elbow_flexion_rom` | 侧/斜侧 |
| 俯卧撑 | `pushup_body_line_break` | `body_line_angle` | 侧/斜侧 |
| 俯卧撑 | `pushup_hip_sag` | `hip_sag_ratio` | 侧/斜侧 |
| 俯卧撑 | `pushup_asymmetry` | 肘/肩左右差 | 正面/斜前/后视 |
| 俯卧撑 | `pushup_tempo_inconsistent` | 时长和 CV | 任意 |
| 俯卧撑 | `pushup_stability_low` | 躯干/髋部抖动 | 任意 |

严重度按“偏差量相对阈值的倍数”分 `low/medium/high`，并乘置信度降级规则。低置信结果最多报告为 `low`；低于最低置信度则不生成问题。

### 步骤七：证据帧与问题卡

每个问题必须能回答“第几个 rep、哪一帧、哪个关节、偏了多少”。

#### 7.1 anchor frame 选择

| 问题类型 | anchor frame |
|---|---|
| 深度、ROM、躯干前倾、身体直线、膝外翻 | 代表 rep 的 `bottom_frame` |
| 左右不对称 | 左右差值最大的帧，限制在 `bottom` 附近 |
| 节奏问题 | 最慢/最快 rep 的中点，或报告时间线的对应区间 |
| 稳定性问题 | 方差最大的滑窗中心帧 |

代表 rep 优先选择偏差最大、置信度最高的 rep；同一问题跨多个 rep 出现时，先报告频率，再挑 1–2 个代表 rep。

#### 7.2 问题卡 schema

```json
{
  "issue_id": "squat_depth_insufficient_01",
  "issue_code": "squat_depth_insufficient",
  "exercise": "squat",
  "rep_ids": [3, 4, 5],
  "representative_rep_id": 4,
  "frequency": "3/8",
  "severity": "medium",
  "confidence": 0.84,
  "anchor_frame": 1634,
  "anchor_time_s": 54.47,
  "evidence_interval": [1608, 1660],
  "measurements": {
    "knee_flexion_rom_deg": 72.4,
    "bottom_knee_angle_deg": 107.6,
    "reference_rom_deg": 92.0
  },
  "visual_overlay": {
    "view": "oblique_left",
    "highlight_joints": ["left_knee", "right_knee"],
    "draw": ["skeleton", "angle_arc", "reference_arc", "deviation_label"],
    "camera_zoom": 1.2
  },
  "text": {
    "title": "下蹲深度不足",
    "detail": "第 3–5 个 rep 膝关节屈曲约 72°，低于参考 92°。",
    "cue": "尝试再向下坐一些，同时保持膝盖朝向脚尖。"
  },
  "suppressed": false
}
```

输出规则：

- 每个 `medium/high` 问题必须有 `anchor_frame` 和 `evidence_interval`；
- 同一 `issue_code + 主要关节 + 相邻 rep` 合并，不重复生成卡片；
- 按 `severity`、`frequency`、`confidence` 排序，默认最多显示 3 张；
- 质量门控失败、动作不确定或视角不支持的指标使用 `suppressed=true` 和原因码。

### 步骤八：圈画与回放报告

新增 `scripts/render_clip_report.py`，生成静态 HTML 报告。V0.5 不要求重新编码整段视频，只要求：

1. HTML 内嵌或引用原视频；
2. 点击问题卡，播放器 seek 到 `anchor_time_s` 并暂停；
3. 在代表帧上叠加骨架、关节高亮、角度弧、参考角和偏差文字；
4. 同时生成 `annotated_frames/<issue_id>_frame_<n>.png`，保证没有前端也能查看；
5. 时间线标出 rep 区间、bottom 事件和问题证据区间。

视觉语法固定为：

| 元素 | 含义 |
|---|---|
| 绿色骨架 | 当前姿态 |
| 黄色关节 | 需要关注但未超过阈值 |
| 红色关节/角度弧 | 超过阈值的偏差 |
| 虚线弧 | 参考角度 |
| 箭头 | 偏差方向或身体直线偏移 |
| 半透明色带 | 证据区间/问题 rep 区间 |
| 灰色遮罩 | 低置信、不可评估关节 |

报告结构：

1. 组级摘要：动作、rep 数、总时长、平均节奏、质量问题数；
2. 质量提示：视角、遮挡、低置信和拒答项；
3. 问题卡列表：按严重度排序，每张卡片可跳转到证据帧；
4. rep 时间线：每个 rep 的时长、底部角度、主要偏差；
5. 原始数据下载：`analysis.json`、`features.csv`、`issues.json`。

## 6. 评估方案

### 6.1 分割与动作识别

| 指标 | 说明 |
|---|---|
| rep count MAE | 每个 clip 的 GT 与预测 rep 数绝对差 |
| IoU@0.30 / 0.50 | 沿用 task1-3 的 rep 匹配协议 |
| boundary error | start/end 平均绝对误差和 p95 |
| bottom frame error | 预测 bottom 与人工标注 bottom 的帧误差 |
| action macro F1 | `squat / pushup / other`，按视频留出 |
| action abstention rate | 低置信时拒答比例 |

### 6.2 问题定位

| 指标 | 定义 |
|---|---|
| issue precision/recall/F1 | 与专家问题级标注匹配 |
| severity agreement | 严重度完全一致或相差一级的比例 |
| anchor frame error | 专家证据帧与本系统 `anchor_frame` 的时间差 |
| evidence interval IoU | 本系统证据区间与专家区间的 IoU |
| unsupported claim rate | 在低置信/错误视角下仍给出高/中严重度结论的比例 |
| evidence coverage | 每个高/中严重度问题带可回放证据的比例 |

### 6.3 产品可用性

- 专家对“圈画位置是否正确”“文字是否可操作”“是否会造成误解”分别打 1–5 分；
- 统计用户从打开报告到定位第一个问题的耗时；
- 与纯文字报告对比，检查回放定位是否降低理解时间；
- 所有主观评分报告样本量和评分者一致性，不用单次演示替代验证。

### 6.4 建议验收线（provisional）

以下门槛用于 V0.5 工程验收，不是论文指标：

- 分割：留出 clip 的 rep count MAE `<= 0.5`，不得有超过 1 个 rep 的严重计数错误；
- bottom 定位：中位误差 `<= 5` 帧，p90 `<= 12` 帧；
- 证据覆盖：`100%` 的 medium/high 问题有 anchor frame 和 evidence interval；
- 问题检测：在 5+5 段专家标注留出集上，medium/high 问题 precision `>= 0.80`；
- 安全：质量门控失败时，错误生成动作问题卡的比例为 `0`；
- 报告：点击问题卡能在 1 秒内跳到对应证据帧；
- 离线性能：1 分钟 30 FPS clip 在单机 CPU 上 2 分钟内完成分析（不含首次模型加载）。

## 7. 工程结构与测试

### 7.1 新增文件

```text
VLS/V0/scripts/analyze_clip.py                 # 端到端编排入口
VLS/V0/scripts/build_issue_cards.py            # 指标→问题卡
VLS/V0/scripts/render_clip_report.py           # HTML 与代表帧圈画
VLS/V0/scripts/serve_reports.py                # 支持 Range 的报告预览服务
VLS/V0/scripts/evaluate_task1_4.py             # 分割/问题定位评估
VLS/V0/configs/task1_4_exercise_specs.json     # 动作、相位、指标和视角规则
VLS/V0/configs/task1_4_feedback_templates.json # 问题卡文案模板
VLS/V0/tests/test_task1_4_rules.py             # 指标和问题规则
VLS/V0/tests/test_task1_4_report.py            # schema、anchor 和报告生成
VLS/V0/tests/test_task1_4_regression.py        # 现有 5 段/新增留出集回归
```

### 7.2 回归要求

- 现有 `segmentation_final` 的 22 段计数和 P2-A golden 不得改变；
- 新编排器只能读取旧输出或调用旧函数，不得覆盖旧目录；
- 对同一 clip 重复分析，`analysis.json` 中除时间戳外的内容应稳定；
- 低置信、视角不支持、少于 3 rep 三类拒答必须有测试；
- 问题卡 schema、anchor frame 合法范围、evidence interval 包住 anchor 必须有测试。

### 7.3 命令草案

```bash
# 分析一个 clip
.venv/bin/python VLS/V0/scripts/analyze_clip.py \
  --manifest VLS/V0/configs/task1_4_clips.json \
  --clip-id 0153_squat_set01 \
  --output-dir VLS/V0/results/clip_analysis_v1

# 重新生成报告
.venv/bin/python VLS/V0/scripts/render_clip_report.py \
  --analysis VLS/V0/results/clip_analysis_v1/0153_squat_set01/analysis.json \
  --output VLS/V0/results/clip_analysis_v1/0153_squat_set01/report.html

# 评估留出集
.venv/bin/python VLS/V0/scripts/evaluate_task1_4.py \
  --predictions VLS/V0/results/clip_analysis_v1 \
  --annotations VLS/V0/annotations/task1_4_holdout \
  --output VLS/V0/results/clip_analysis_v1/evaluation.json
```

## 8. 分阶段交付

| 子阶段 | 内容 | 交付 |
|---|---|---|
| T1-4A | clip manifest、数据筛选、视角和动作规范 | 10+16 段候选 clip 清单 |
| T1-4B | MediaPipe/角度适配、质量门控、rep/相位输出 | 单 clip `analysis.json`，深蹲可用 |
| T1-4C | 指标、问题规则、anchor/evidence 选择 | `features.csv`、`issues.json` |
| T1-4D | 代表帧圈画、HTML 回放报告 | `report.html` + `annotated_frames/` |
| T1-4E | 加俯卧撑、动作识别和留出评估 | 两动作评估报告与冻结快照 |

建议先完成 T1-4B/C/D 的深蹲 vertical slice，确认“问题能定位、报告能回放、专家能判断”后，再把俯卧撑接入。不要并行扩展动作和 GUI。

## 9. 风险与决策点

| 风险 | 应对 |
|---|---|
| 只有 1–2 个 rep，自相关周期不稳定 | 要求至少 3 个完整 rep；短 clip 单独做模板/峰值 fallback |
| 视角不支持膝外翻或左右差 | 根据 `view` 禁用指标，报告“不评估”而不是猜 |
| MediaPipe 角度噪声导致错误圈画 | confidence 低于阈值不生成问题；证据帧必须显示原始画面和骨架 |
| 问题规则阈值不适配不同人群 | 先使用教练参考和 provisional 阈值；所有阈值版本化并记录来源 |
| 同一组内问题重复过多 | 按 issue_code 合并，报告频率，只展示 1–2 个代表 rep |
| QEVD 只有视频级反馈，没有证据帧 GT | 新增小规模专家标注；视频级教练文本只作弱参考 |
| 动作识别和评估错误互相放大 | 初版允许人工指定动作；自动识别低置信时直接停止动作特定评估 |

关键决策点：

1. 深蹲和俯卧撑是否都要求侧/斜侧视角？如果 QEVD 中正面视频不足，膝外翻只作为低优先级可选指标。
2. 问题卡默认最多显示 3 张还是按严重度全部展示？建议默认 3 张，完整列表可展开。
3. 代表帧是使用单帧还是 1–2 秒短回放？建议报告层先做单帧圈画 + 播放器区间，不重新编码视频。
4. 动作识别是否进入 V0.5 硬门槛？建议不作为首个 vertical slice 的门槛，但留出集必须报告。

## 10. 完成定义

满足以下条件视为 task1-4 V0.5 完成：

- 输入一个含 3 个以上 rep 的深蹲或俯卧撑 clip，一条命令生成完整报告；
- 报告包含 rep 时间线、bottom 相位、逐 rep 指标和可回放问题卡；
- 每张 medium/high 问题卡都定位到原视频帧，并在该帧圈画相关关节和角度；
- 动作、rep 分割、问题检测和证据定位在视频级留出集上有可复现指标；
- 质量门控能拒绝遮挡、少 rep、低置信和错误视角，不产生高置信错误结论；
- 现有 task1-3 冻结结果和 golden 回归保持不变。

## 11. T1-4A 执行状态

**完成（2026-10-25）**：已生成 41 个候选 clip，入选深蹲 `5 dev + 5 holdout`、俯卧撑 `8 dev + 8 holdout`，按视频隔离并完成旋转/视角人工复核。核心发现是 QEVD 纯深蹲/俯卧撑窗口几乎全部为正面视角，26 个入选 clip 中没有可靠侧/斜侧样本，因此当前入选集的矢状面指标全部标为 disabled；T1-4B 前应补录少量侧视 clip，或先完成正面指标垂直切片。详见 `VLS/V0/docs/task1-4a_results.md`。

## 12. T1-4B 执行状态

**完成（2026-10-26 更新）**：按“暂时无法补录侧视数据”执行纯正面垂直切片。新增 `scripts/analyze_clip.py`，完成 MediaPipe 姿态提取、角度复用、质量门控、rep/相位输出和原视频绝对时间映射；新增相位循环兜底和 `rejected_reps.csv`，避免不完整 rep 进入后续指标。针对热身/起身被误判为训练 rep 的问题，新增动作姿态门控（俯卧撑 bottom 躯干接近水平、深蹲保持直立）和相位信号候选分割，经门控后按有效 rep 数择优。13 个 dev clip 全部为 `ok`，共 118 个完整 rep、14 个剔除 rep；`0004_pushup_set01` 从 0 个有效 rep 恢复为 5 个真实俯卧撑，`0005_pushup_set01` 保留 3 个跪姿俯卧撑并剔除起身 rep，`0156_pushup_set01` 保留 9 个俯卧撑并剔除起身 rep。`0161_pushup_set01` 与 5 个盲标 rep 的 IoU@0.30 仍为 `5/5`。所有入选 clip 仍为正面视角，矢状面指标继续 disabled，未伪造侧视结论。详见 `VLS/V0/docs/task1-4b_results.md`。

## 13. T1-4C 执行状态

**完成（2026-10-26 更新）**：新增 `scripts/build_issue_cards.py` 与 `configs/task1_4_feedback_templates.json`，直接消费 T1-4B 输出，生成逐 rep `features.csv`、`issues.json`、顶层 `features_all.csv` 和 `issues_summary.json`。13 个 dev clip 共 118 个完整 rep，输出 1063 行指标、47 张活动问题卡和 34 条屏蔽/未评估记录；14 张 medium/high 问题卡全部通过 anchor、代表 rep 边界和 evidence interval 校验。热身/起身 rep 已在 T1-4B 剔除，不进入指标聚合；`0004_pushup_set01` 不再点评热身，`0005_pushup_set01` 的两个活动问题均落在真实跪姿俯卧撑帧。正面视角不支持的矢状面指标只输出 `suppressed=true`，少于 3 rep 不判断组级节奏。由于尚无教练参考或专家问题级 GT，阈值统一标记为 provisional，不报告准确率。期间修正了 `spine_stability_bottom` 的错误单位，并将膝内扣改为由髋-膝-踝冠状面对齐推导，避免把屈膝投影夹角误当内扣角。详见 `VLS/V0/docs/task1-4c_results.md`。

## 14. T1-4D 执行状态

**完成（2026-10-26 更新）**：新增 `scripts/render_clip_report.py`，生成 13 个 dev clip 的静态 `report.html`、13 个裁剪 VP8/WebM 回放代理、47 张 `annotated_frames/<issue_id>_frame_<n>.png` 和顶层 `index.html`。问题卡可 seek 到 clip 相对 `anchor_time_s` 并暂停，也可播放 `evidence_interval`；证据播放会等待 `seeked` 完成后再 `play()`，避免异步 seek 被 `timeupdate` 提前终止；视频 metadata 未就绪时 seek 会排队到 `loadedmetadata` 后执行。报告包含组级摘要、质量/未评估项、默认最多 3 张问题卡、rep/bottom/证据时间线、逐 rep 明细和原始数据下载。标注使用绿色骨架、黄色普通关节、红色问题关节/角度弧、虚线参考线、偏移箭头和灰色低可见关节。47 张标注帧全部完成姿态叠加；其中 4 张因单帧检测缺失或关键关节可见性/几何不合理，通过 VIDEO 跟踪回退恢复，最终 `pose_detection_failures=0`。质量警告改为中文展示，姿态门控剔除热身/起身时报告会明确说明；被 suppression 或重新编号淘汰的旧标注帧自动清理。源视频为浏览器不支持的 MPEG-4 Visual，已改为裁剪 VP8/WebM 代理播放；新增支持 HTTP Range 的 `scripts/serve_reports.py`，并提示普通 `http.server` 不支持 Range 会导致 seek 失效。13 个报告共 138 个本地资产引用全部可解析。全量测试 `56/56` 通过。当前环境无浏览器，未做真实浏览器截图回归。详见 `VLS/V0/docs/task1-4d_results.md`。
