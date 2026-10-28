# task1-5：Fit3D 3D GT 对齐与可用视角审计

## 1. 任务定位

task1-4 已完成 QEVD-FIT 正面视角下的深蹲/俯卧撑垂直切片：

```text
clip → MediaPipe → rep/相位 → 指标 → 问题卡 → 可回放报告
```

当前明确遗留的是一个数据缺口：QEVD 入选 clip 没有可靠侧/斜侧视角，因此深蹲深度、躯干前倾、俯卧撑下放深度、身体直线、塌腰/撅臀等矢状面指标全部处于 `disabled`。Fit3D 下载完成后，可以用它的多视角视频、逐帧 3D 关节和 rep 时间戳补齐三类验证：

1. MediaPipe 在侧视下的逐帧姿态和关节角是否足够可靠；
2. T1-4 定义的矢状面指标能否从视频中恢复 Fit3D GT 的数值；
3. T1-4 的 rep 分割能否与官方 rep 时间戳对齐。

初步几何审计还发现一个必须先接受的限制：Fit3D 的 4 路相机不是固定的“前/后/左/右”，而是随受试者动作朝向形成不同视角。按 8 人 rep 帧的 3D 骨架与外参计算：

| 动作 | 相机相对身体额状面法向的中位夹角 | 实际覆盖 | 对本任务的影响 |
|---|---:|---|---|
| `squat` | 约 18–35° | 正面、背面、浅斜前/斜后，没有纯侧视 | 可验证 3D 角度定义和正面/浅斜视角；不能补齐“纯侧视深蹲”视频验证 |
| `pushup` | 约 60–90° | 侧视与斜侧视 | 可用于验证俯卧撑肘角、身体直线和髋塌陷的侧视测量 |

此外，Fit3D 的 `squat` 实际是杠铃背蹲，不是 T1-4 的徒手深蹲。两者可用于比较关节角测量，但不能把 Fit3D 的指标分布直接当成徒手深蹲的基线。

本任务的主线不是“再做一个 demo”，而是回答一个决策问题：

> 现有 MediaPipe + 规则指标能否支撑侧视反馈；如果不行，差距是视角门控、指标实现，还是姿态模型本身，需要升级到什么程度。

## 2. 范围与非目标

### 2.1 本任务范围

- 动作仍聚焦 `squat` 和 `pushup`，与 T1-4 保持同一垂直切片。
- 使用 Fit3D train 的 8 个受试者、4 路相机、3D 标注和 rep 标注做外部验证，但不假设四路相机的视角组成相同。
- 建立 Fit3D → T1-4 manifest 的适配层，不复制一套新的分析管线。
- 实现并校验 T1-4 已定义但尚未启用的矢状面指标。
- 增加基于相机外参和 3D 骨架的视角判定，替代仅靠 2D 画面比例的人工/启发式判断。
- 在 Fit3D 上评估 rep 数量、事件定位和逐 rep 指标误差。
- 输出逐受试者、逐动作、逐相机的结果，以及明确的 go/no-go 结论。

### 2.2 非目标

- 不把动作范围扩到 Fit3D 的全部 47 类动作。
- 不在本任务中训练新的 2D/3D 姿态网络，也不微调 MubyNet/RTMPose；先测出现有基线，再决定 T1-6 是否需要模型升级。
- 不用 Fit3D test 计算 MPJPE、rep IoU 或角度准确率；test 没有 3D/rep GT。
- 不把 Fit3D 的动作质量分布当作“正确/错误动作”的专家 GT。Fit3D 能验证测量值，不能验证反馈文案是否专业。
- 不因为 Fit3D 是棚拍数据，就直接宣称模型能泛化到手机、手持、健身房和遮挡场景。
- 不把 Fit3D 杠铃深蹲当作徒手深蹲的外观、负荷或反馈基线。
- 不把“Fit3D 有 4 路相机”误写成“深蹲已有 4 个侧视验证视角”。
- 不覆盖 `results/segmentation_final/`、`results/clip_analysis_v1/` 或 T1-4 已冻结产物。

## 3. 审计结论与数据边界

依据 `VLS/dataset/Fit3D/QC/REPORT.md`，本任务只依赖以下已核实事实：

| 项目 | train | test |
|---|---:|---:|
| 受试者 | 8 | 3 |
| 动作类别 | 47 | 47 |
| 深蹲/俯卧撑动作实例 | 8 人 × 2 动作 = 16 | 3 人 × 2 动作 = 6 |
| 深蹲/俯卧撑视频 | 16 × 4 相机 = 64 | 6 单视角视频 |
| 3D 标注 | 16/16 动作实例有 `joints3d_25` | 无 |
| rep 标注 | 16/16 动作实例各 6 个 = 96 个唯一次数 | 无 |
| 视频规格 | 900×900、50 FPS、H.264 | 900×900、50 FPS、H.264 |

额外数据属性：

- Fit3D `squat` 为杠铃背蹲，T1-4 目标为徒手深蹲；
- Fit3D `pushup` 为地面俯卧撑，可作为侧视测量的主要数据源；
- 8 人 × 2 动作 × 4 相机的 18–90° 视角跨度，可用于测试“指标在不同视角下的误差如何变化”，而不是简单宣称每类指标都有侧视 GT 验证。
- 审计可验证的 rep 标签共 1,822 个；项目方案中提到的 2,964 个并未全部出现在当前 train/test 包内。本任务只使用深蹲/俯卧撑的 96 个，不把缺失的 1,142 个当作已获得数据。

因此本任务对侧视指标的核心验证样本是：

```text
16 个受试者-动作实例
64 路视频视角
96 个唯一 rep
384 个“视角 × rep”观测（不是 384 个独立动作实例）
```

统计和置信区间必须以受试者-动作实例为主要独立单位，不能把同一动作的 4 个相机当成 4 个独立样本。

侧视指标的可用样本必须再按实际夹角筛选；`64 路视频` 是总视角数，不是 `64 路侧视`。在最终报告中必须同时给出“可用侧视/斜侧视频数”和“被视角门控排除的视频数”。

### 3.1 数据使用边界

- train 的逐帧 3D 和 rep 标注可用于评价与误差分析，但 dev/holdout 必须按受试者隔离。
- test 只能做推理 smoke test，检查视频可读、姿态可检出、报告可生成；不能报告准确率或用于阈值调参。
- `fit3d_template.json` 是静态模板，不是 GT，禁止用于评价。
- 每条视频都以对应相机、对应动作的 JSON 标定为准；不套用受试者级固定外参。
- 视频头帧数可能比标注多 0–10 帧。所有训练/评估统一截取前 `N = len(joints3d_25)` 帧，禁止用“视频最后一帧”代替标注终点。
- `rep_ann.json` 的时间值当前是帧索引，落在 `[0, N)` 且严格递增。其语义必须先做核对，尤其不能把“事件帧”直接当作 rep 起止边界。

### 3.2 dev / holdout 划分

按受试者隔离，沿用 T1-4 的“先 dev 调规则、holdout 只评一次”原则：

| 角色 | 受试者 | 说明 |
|---|---|---|
| dev | `s03`, `s04`, `s05`, `s07` | 可用来看误差、调视角门控和指标实现 |
| holdout | `s08`, `s09`, `s10`, `s11` | 规则与门控冻结后只做最终评估 |
| smoke | `s02`, `s12`, `s13` | test，仅检查管线可运行，不算指标 |

每个受试者固定包含 `squat` 和 `pushup` 两个动作，每个动作 4 路相机。不得按相机或帧随机切分，避免同一动作同时进入 dev 和 holdout。

## 4. 关键决策关口

### Gate A：视角判定是否可靠

在 64 路 train 视频上生成视角审计表。若自动视角判定与人工复核的一致率不足 90%，先修视角算法；不得直接进入侧视指标评估。

### Gate B：侧视姿态是否可用

在 holdout 侧视视频上检查姿态检出率、关键关节可见性和角度误差。若膝/肘角 MAE 超过 15°，禁止在 QEVD 上启用对应问题卡，并将“姿态模型升级”列为 T1-6 候选。

### Gate C：矢状面指标是否可启用

按下表分流，不把“模型不擅长”包装成“动作正常”：

| 侧视指标误差 | 处理 |
|---|---|
| MAE ≤ 10°（角度）或 ≤ 0.05（比例） | 可启用，正常生成评估 |
| 10° < MAE ≤ 15°，或 0.05 < MAE ≤ 0.08 | 只允许 low confidence，并保留“测量不确定”提示 |
| MAE > 15°，或 > 0.08 | 继续 disabled，不生成矢状面问题卡 |

比例指标的最终门槛需要在 dev 上确认测量误差与 T1-4 阈值是否同量级；不能用误差大于阈值的指标做正式报警。

Gate C 分动作执行：

- `pushup`：在 60–90° 侧/斜侧视频上验证；通过后可启用侧视指标。
- `squat`：在 18–35° 浅斜视角上只能验证“斜前/斜后是否够用”。若纯侧视缺失且斜视误差超线，`squat_side` 继续 disabled；3D GT 仍可用于验证指标公式和关节映射，但不能替代视频侧视验收。

### Gate D：是否在 T1-6 升级姿态模型

只有满足以下任一条件才进入模型升级：

- 侧视 MediaPipe 的膝/肘角 MAE 持续高于 15°；
- 多视角一致性显示 MediaPipe 世界坐标存在明显系统性偏差；
- 即使角度误差可接受，`hip_sag_ratio`、`hip_depth_ratio` 等比例指标仍无稳定相关性。

如果现有基线通过 Gate C，T1-6 应优先扩动作和反馈，而不是为了“更高级模型”而训练。

## 5. 输入、输出与接口

### 5.1 Fit3D clip manifest

新增 `scripts/build_task1_5_fit3d_manifest.py`，把 Fit3D 映射成 T1-4 可读取的 manifest，并额外保留 GT 索引。

单条记录示例：

```json
{
  "clip_id": "fit3d_s09_squat_50591643",
  "video_path": "VLS/dataset/Fit3D/train/s09/videos/50591643/squat.mp4",
  "trim": {"start_frame": 0, "end_frame": 1212},
  "exercise_hint": "squat",
  "rotation": "none",
  "fps": 50,
  "subject_id": "s09",
  "camera_id": "50591643",
  "split_role": "holdout",
  "view": "unknown",
  "view_angle_deg": 26.9,
  "view_label_source": "camera_geometry_pending_manual_audit",
  "annotation_frames": 1213,
  "video_frames": 1214,
  "gt": {
    "joints3d_25": "VLS/dataset/Fit3D/train/s09/joints3d_25/squat.json",
    "rep_frames": [370, 484, 609, 724, 832, 931],
    "camera_parameters": "VLS/dataset/Fit3D/train/s09/camera_parameters/50591643/squat.json"
  },
  "usable_metrics": [],
  "disabled_metrics": []
}
```

最终 manifest 必须通过：

- 16/16 动作实例、64/64 视频、96/96 rep 均存在；
- 每个视频的 `trim.end_frame = N - 1`，`N` 与 `joints3d_25` 长度一致；
- 视频实际帧数 ≥ `N`；
- 相机、3D、rep 路径均可解析；
- `clip_id` 唯一；
- 4 路相机的 `exercise/subject/action` 指向同一动作实例；
- dev/holdout 的受试者集合不重叠。

### 5.2 输出目录

```text
VLS/V0/results/task1_5/
├── manifest/
│   ├── fit3d_clips.json
│   ├── validation.json
│   └── view_audit.csv
├── gt/
│   └── <clip_id>_gt.json
├── clip_analysis/
│   └── <clip_id>/                 # 复用 T1-4 逐 clip 输出结构
├── evaluation/
│   ├── summary.json
│   ├── pose_angles.csv
│   ├── sagittal_metrics.csv
│   ├── rep_events.csv
│   ├── view_confusion.csv
│   └── runtime.csv
└── report/
    ├── index.html
    └── task1-5_results.md
```

`clip_analysis/` 使用独立目录，不能写回 `results/clip_analysis_v1/`。T1-4 的 QEVD 结果继续作为冻结回归集。

## 6. GT 与指标定义

### 6.1 关节映射与统一公式

Fit3D 使用 25 关节，MediaPipe 使用 33 关节点。第一阶段只映射评估必需关节：

```text
shoulder / elbow / wrist / hip / knee / ankle
```

Fit3D 的左右编号必须在投影图上逐类目视核对，不能仅按索引猜测。建议新增一个可视化脚本，把 GT 关节按对应相机参数投影到视频帧，确认左右侧、关节点和视频方向一致后再冻结映射。

GT 指标和视觉指标必须共用同一套几何函数，避免“GT 用 SMPL 定义、视觉用另一个公式”。优先抽取 `scripts/task1_5_metrics.py`：

- `three_point_angle()`
- `body_frame_projection()`
- `knee_flexion_deg()`
- `elbow_flexion_deg()`
- `hip_flexion_deg()`
- `torso_flexion_deg()`
- `body_line_angle_deg()`
- `hip_sag_ratio()`
- `hip_depth_ratio()`

T1-4 现有 `extract_angles.py` 的公式作为基线，不得在 T1-5 里悄悄改定义。若必须改，先在 T1-4 dev 上做等值回归并记录版本差异。

### 6.2 矢状面指标

| 指标 | 定义 | 数据源 |
|---|---|---|
| `knee_flexion_rom` | rep 内膝角最大值 − 最小值 | 侧视 MediaPipe / Fit3D GT |
| `bottom_knee_angle` | bottom 帧膝角 | 同上 |
| `hip_flexion_rom` | rep 内髋角最大值 − 最小值 | 同上 |
| `torso_lean_bottom` | bottom 帧躯干前倾角 | 同上 |
| `hip_depth_ratio` | (站立髋高 − bottom 髋高) / 站立髋高 | 同上 |
| `elbow_flexion_rom` | rep 内肘角最大值 − 最小值 | 同上 |
| `bottom_elbow_angle` | bottom 帧肘角 | 同上 |
| `body_line_angle` | 肩-髋-踝在矢状面的夹角，伸直接近 180° | 同上 |
| `hip_sag_ratio` | 髋偏离肩-踝连线距离 / 躯干长度 | 同上 |

侧视下只使用靠近相机的一侧肢体计算膝/肘指标。选择规则为：

1. 依据相机几何判断哪一侧朝向镜头；
2. 用该侧关键关节 visibility 做二次校验；
3. 两侧都低可见时不生成数值，标记 `insufficient_side_visibility`；
4. 不允许把左右两侧角度取平均后冒充“靠近镜头侧”的结果。

### 6.3 rep 标注语义

`rep_ann.json` 的 6 个帧号先做三种假设的核查：

- 假设 A：每次 rep 的起点/边界；
- 假设 B：每次 rep 的 bottom 事件；
- 假设 C：每次 rep 的某一固定相位点。

核对方法：在 4 个受试者 × 2 动作上，把标注帧与 3D 膝/肘角曲线叠画，检查标注相对峰值/谷值的偏移。只有语义确认后才计算 IoU；在此之前只报告 rep 数量、事件误差和相邻标注间隔。

若语义为“边界”：

- 相邻边界构成完整 rep 区间；
- 首尾未被两个边界完整包住的区间不参与 IoU；
- 以边界 F1、起点/终点误差和 rep 数补充报告。

若语义为“事件帧”：

- 主指标为事件匹配误差和 rep 数；
- IoU 只作为辅助指标，用中点法构造临时区间并明确标注这是派生区间，不是官方区间。

## 7. 实现步骤

### 步骤一：Fit3D 适配与审计

新增 `scripts/build_task1_5_fit3d_manifest.py`：

1. 枚举 train 8 人的 `squat`、`pushup`；
2. 校验每个动作的 4 个相机视频、相机参数、`joints3d_25` 和 `rep_ann`；
3. 用 `N = len(joints3d_25)` 生成 trim；
4. 读取每条视频的 FPS、帧数、方向；
5. 将 `subject_id`、`camera_id`、GT 路径和 split role 写入 manifest；
6. 生成 `validation.json` 和受试者级 split 统计。

先做 pilot：`s09` 的 squat/pushup × 4 相机，共 8 条视频。pilot 必须完成视频读帧、MediaPipe 姿态、3D 投影叠加和 rep 曲线可视化，再跑全量。

### 步骤二：相机几何视角判定

新增视角计算函数，输入为 3D 关节和相机标定，输出 `view`、`view_angle_deg` 和置信度：

1. 先用 3D 关节投影叠加确认 Fit3D 外参约定；不能直接假设 `C = -RᵀT`，本轮 pilot 已确认按投影结果校正后 `T` 才给出合理的相机位置；
2. 从肩/髋构造身体侧向轴和躯干轴；
3. 计算身体额状面法向；
4. 计算 pelvis 到相机中心的方向；
5. 用法向与相机射线的夹角定义 front/rear/oblique/side；
6. 对动作内稳定帧取中位数，并记录角度 IQR；
7. 若动作内视角变化过大，标为 `unknown` 并禁用视角依赖指标。

自动结果必须和人工 contact sheet 对照。64 条视频全部留审计记录，不能只抽查 4 条后直接相信分类器。

拟合结果只用于“该视角支持哪些指标”，不直接改变 T1-4 的 provisional 问题阈值。

### 步骤三：姿态与角度基线评估

用现有 MediaPipe 管线处理 64 条视频，分别评估：

- 姿态检出率、关键关节 q10 visibility、全身入镜率；
- 膝/肘/髋/躯干角与 GT 的 MAE、bias、p90；
- 左右侧选择是否正确；
- 正面视角的现有指标是否在 Fit3D 上同样成立；
- 侧视指标是否只在侧/斜侧视角启用。

对 `squat` 必须单列“18–35° 浅斜视”结果，不得把它混入 `side` 上报；如果侧视指标在该角度误差超线，结论应写“Fit3D 不能补齐深蹲侧视验证”，而不是用 3D GT 替代视频结果。

主报告按 `exercise × view × joint` 分层，并给出逐受试者结果。所有置信区间按 `subject-action` cluster，不能按帧或按相机独立计算。

### 步骤四：矢状面指标实现与校准

在 `build_issue_cards.py` 或新的指标模块中实现 T1-4 已定义但当前未计算的矢状面指标：

- 深蹲：`knee_flexion_rom`、`bottom_knee_angle`、`hip_flexion_rom`、`torso_lean_bottom`、`hip_depth_ratio`；
- 俯卧撑：`elbow_flexion_rom`、`bottom_elbow_angle`、`body_line_angle`、`hip_sag_ratio`。

每个指标输出：

```json
{
  "metric": "bottom_knee_angle",
  "value": 104.3,
  "gt_value": 101.8,
  "abs_error": 2.5,
  "confidence": 0.84,
  "camera_side": "left",
  "view": "oblique_left",
  "status": "ok"
}
```

本轮不直接修改 T1-4 的 issue 阈值。先用 Fit3D 确认指标测得准，再决定是否在 T1-6 重新标定严重度阈值。

### 步骤五：rep / bottom 评估

把 T1-4 的 rep、bottom 和相位输出映射到 Fit3D 原视频帧，评估：

- rep 数量误差和完全匹配率；
- 相邻 rep 间隔/时长误差；
- bottom 或标注事件帧误差；
- 若标注语义允许，再计算 IoU@0.30 和 IoU@0.50；
- 低置信、遮挡、视角不支持时的拒答是否正确。

需同时报告“所有视角”和“侧视主相机”两种结果，避免 4 路相机等权平均掩盖单视角失败。

### 步骤六：QEVD 回归与报告

Fit3D 验证通过后，把新的视角判定和矢状面指标接到 T1-4 QEVD manifest：

1. 先只在 dev clip 上验证不改变既有正面问题卡；
2. 运行全量回归，确认 T1-4 的 13 个 dev clip、118 个 rep、47 个问题卡不被无意改变；
3. 对真实侧视或斜侧 QEVD clip 启用对应指标；
4. 输出 `task1-5_results.md` 和可回放的失败样本索引。

若 QEVD 仍没有可靠侧视样本，Fit3D 的结论只用于“测量能力验证”和“视角门控”，不能声称 QEVD 产品指标已被端到端验证。

## 8. 评估与验收

### 8.1 数据与工程验收

| 项目 | 通过线 |
|---|---|
| 适配完整率 | 16/16 动作实例、64/64 视频、96/96 rep |
| GT 映射 | 投影叠加人工复核通过，无左右侧错位 |
| 视角判定 | 与 64 条人工审计一致率 ≥ 90%，且无 front/rear 被静默当作 side |
| 测试可复现 | 固定 manifest 和 seed 后，指标表除时间戳外一致 |
| 回归 | T1-4 既有测试全部通过，QEVD 冻结计数不变 |

### 8.2 rep 验收（provisional）

| 指标 | 通过线 |
|---|---|
| rep count MAE | ≤ 0.5 / 动作实例 |
| 完全计数正确 | ≥ 14/16 动作实例 |
| 事件/bottom 中位误差 | ≤ 5 帧（50 FPS 下 0.1 s） |
| 事件/bottom p90 误差 | ≤ 12 帧 |
| 低置信拒答 | 不得把错误分割包装成高置信完整 rep |

### 8.3 侧视指标验收（provisional）

| 指标组 | 可启用 | 低置信 | 继续禁用 |
|---|---:|---:|---:|
| 膝/肘 flexion 及 ROM | MAE ≤ 10° | 10–15° | > 15° |
| 躯干前倾 / 身体直线 | MAE ≤ 10° | 10–15° | > 15° |
| 髋深 / 髋塌陷比例 | MAE ≤ 0.05 | 0.05–0.08 | > 0.08 |

同时报告 Pearson/Spearman 相关和按受试者聚类的 bootstrap 95% CI；相关只能辅助解释，不能替代绝对误差门槛。

### 8.4 结论模板

最终报告必须明确给出：

- 哪些侧视指标可以进入产品规则；
- 哪些只能低置信展示；
- 哪些必须继续 disabled；
- Fit3D 实际提供了多少条可用于侧视视频验证的角度，而不是只报“64 路”；
- 杠铃深蹲与徒手深蹲的域差限制了哪些结论；
- 误差主要来自 MediaPipe、相机视角、左右侧选择、bottom 定位还是指标定义；
- 是否需要 T1-6 做姿态模型升级；
- Fit3D 结论到 QEVD/手机场景不能外推的部分。

## 9. 分阶段交付

| 子阶段 | 内容 | 产物 |
|---|---|---|
| T1-5A | Fit3D 适配、split、视角审计 | manifest、validation、view audit |
| T1-5B | MediaPipe/GT 姿态与角度基线 | pose_angles、逐视角误差表 |
| T1-5C | 矢状面指标实现与启用决策 | sagittal_metrics、go/no-go 阈值表 |
| T1-5D | rep/bottom 对标 | rep_events、IoU/事件误差 |
| T1-5E | QEVD 回归、失败样本、最终报告 | `task1-5_results.md`、回归快照 |

建议顺序：A → B → C → D → E。只有 A 的视角判定和 B 的侧视姿态都通过，才进入 C 的指标启用。

## 10. 新增文件草案

```text
VLS/V0/scripts/build_task1_5_fit3d_manifest.py   # Fit3D → T1-4 manifest
VLS/V0/scripts/task1_5_metrics.py                 # 共用 3D/投影指标公式
VLS/V0/scripts/evaluate_task1_5_fit3d.py          # 姿态、指标和 rep 评估
VLS/V0/scripts/audit_task1_5_views.py             # 相机几何视角 + contact sheet
VLS/V0/scripts/audit_task1_5b_sides.py            # GT 相机深度 + 近侧选择审计
VLS/V0/configs/task1_5_fit3d_clips.json           # 冻结 manifest
VLS/V0/configs/task1_5_view_overrides.json        # 人工复核覆盖，可选
VLS/V0/tests/test_task1_5_adapter.py              # 路径、N、split、rep 完整性
VLS/V0/tests/test_task1_5_metrics.py              # 向量、角度、比例和退化输入
VLS/V0/tests/test_task1_5_evaluation.py           # 配对、聚类和错误分流
VLS/V0/docs/task1-5_results.md                    # 最终结果与决策
VLS/V0/docs/task1-5b_results.md                   # 姿态/角度基线与 go/no-go 证据
```

## 11. 命令草案

```bash
cd /mnt/storage2/data_guest

# 1. 构建并校验 Fit3D manifest
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/build_task1_5_fit3d_manifest.py \
  --dataset-root VLS/dataset/Fit3D \
  --output VLS/V0/configs/task1_5_fit3d_clips.json

# 2. 生成四路相机 contact sheet，人工复核视角
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/audit_task1_5_views.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --output-dir VLS/V0/results/task1_5/manifest \
  --overrides VLS/V0/configs/task1_5_view_overrides.json

# 3. pilot：s09 的 squat/pushup × 4 相机
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/analyze_clip.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --clip-id fit3d_s09_squat_50591643 fit3d_s09_pushup_50591643 \
  --output-dir VLS/V0/results/task1_5/clip_analysis \
  --overwrite

# 4. 全量 train 评估
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/evaluate_task1_5_fit3d.py \
  --manifest VLS/V0/configs/task1_5_fit3d_clips.json \
  --analysis-dir VLS/V0/results/task1_5/clip_analysis \
  --output-dir VLS/V0/results/task1_5/evaluation

# 5. 全量回归
VLS/V0/.venv-mediapipe/bin/python -m unittest discover \
  -s VLS/V0/tests -p 'test*.py'
```

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 多相机同动作被当作独立样本 | 指标置信区间被高估 | 以 subject-action 为 cluster；相机结果单独分层 |
| rep 时间戳语义不明 | IoU 定义错误 | 先做峰值/谷值叠加核查；不确定时只报事件误差和数量 |
| 侧视左右侧选错 | 膝/肘角系统性偏差 | 相机几何选侧 + visibility 校验 + 投影叠加复核 |
| 深色服装/遮挡导致 MediaPipe 丢失 | 侧视指标误报 | 质量门控、低可见拒答、记录失败样本 |
| Fit3D 棚拍与 QEVD 手机场景存在域差 | 不能直接外推 | 报告分层说明；QEVD 只做回归和真实侧视 smoke |
| 指标定义在 GT 与视觉侧不一致 | 误差不可解释 | 单一公式模块；对 T1-4 做等值回归 |
| test 被误用为独立评测集 | 结论不成立 | manifest 标记 `smoke_only`；评估脚本拒绝读取 test GT |
| 视角判定依赖人工覆盖 | 不可复现 | 覆盖文件版本化，人工标签逐条留痕 |

## 13. 完成定义

满足以下条件视为 task1-5 完成：

- 64 路 Fit3D train 视频和 96 个 rep 全部被稳定索引；
- 4 路相机视角有可复现判定和人工审计记录；
- 明确报告深蹲无纯侧视、俯卧撑有侧/斜侧的数据覆盖差异；
- 侧视 MediaPipe 与 3D GT 的角度误差按受试者/动作/视角报告；
- 深蹲和俯卧撑的矢状面指标均给出“启用 / 低置信 / 禁用”结论；
- 若深蹲因缺少纯侧视无法启用，明确保留 disabled，不以杠铃深蹲或 3D GT 绕过视频验收；
- rep 数量、事件/底部定位和（语义确认后的）IoU 有明确结果；
- T1-4 dev/holdout 回归和全量测试不退化；
- 最终文档明确回答“MediaPipe 是否足够支撑侧视反馈，若不够该升级什么”。

## 14. T1-5A 执行状态

状态：**完成（2026-10-26）**。详见 `docs/task1-5a_results.md`。

- manifest 校验 `passed=true`：`70` 条 clip（64 train + 6 test smoke）、`16` 个 subject-action、`96` 个唯一 rep。
- 受试者级 split：dev `s03,s04,s05,s07`；holdout `s08,s09,s10,s11`；test smoke `s02,s12,s13`。
- 视角审计 `64/64` 已人工复核，自动分类一致 `64/64`，修正 `0` 条。`squat` 为 `17.8–34.9°` 正面/背面/浅斜视，无纯侧视；`pushup` 为 `59.3–89.5°` 侧/斜侧视。
- `s09` pilot 8 路全部执行并保留失败结果：`ok=6`、`partial=1`、`rejected_quality=1`，完整 rep `36/48`。侧视下固定左侧肘/肩和整段角度选主信号不稳定，列入 T1-5B。
- GT 投影叠加 8 条 clip，抽样帧 `25/25` 关节入画；rep 曲线 2 张。`rep_ann` 证据支持其为接近伸展的 top/边界事件，不是 bottom GT；T1-5D 前不得按 bottom 使用。
- 侧视指标继续 disabled：`side_metrics_enabled=false`，`side_metrics_status=pending_t1_5b_validation`。
- 全量单元测试 `71/71` 通过。

产物：`configs/task1_5_fit3d_clips.json`、`configs/task1_5_view_overrides.json`、`results/task1_5/manifest/`、`results/task1_5/pilot_analysis/`。

## 15. T1-5B 执行状态

状态：**完成（2026-10-26）**。详见 `docs/task1-5b_results.md`。

- 发现并修正 T1-5A 的 `camera_side` 整体符号错误：修正前 64/64 与 GT 相机空间深度相反；修正后 `camera_side`、view 左右后缀、近侧选择均为 `64/64` 一致。32 条 side/oblique override 的后缀同步更新。
- 新增 `scripts/audit_task1_5b_sides.py`，逐 clip 输出 GT 左/右链深度、实际近侧、manifest 侧别和 MediaPipe 选中侧。37 条 side/oblique clip 全部选中近侧，远侧替代 `0` 条。
- 修改 `scripts/task1_5_metrics.py` 与 `scripts/analyze_clip.py`：侧视只使用相机近侧做相位/主角度，近侧 visibility、覆盖率或动态范围不足时拒答，不再切换到远侧。
- 64 路 train 全量重跑完成：`ok=60`、`partial=3`、`rejected_quality=1`；质量 `pass=37`、`warn=26`、`reject=1`；完整 rep `325`。`s09` pilot 从 `36/48` 修复到 `38/48`。
- 正式评测输出 `results/task1_5/evaluation/`。holdout 俯卧撑纯侧视屈肘 MAE：`side_left=15.19°`、`side_right=12.50°`；深蹲浅斜视膝角 MAE `10.15–12.03°`，但无纯侧视。
- 高误差集中在相机 `50591643`/`58860488` 的近侧肘（MAE `23.83°`/`21.48°`），而近侧 visibility 仍为 `0.88/0.92`；证据指向近侧肢体自遮挡，不是左右选错。远侧仅保留诊断，不进入产品规则。
- 结论：侧视角度基线未通过可用线，T1-5C 不启用任何侧视产品指标；T1-5D 可继续 rep 事件评估，但不能把侧视角度当作已验证测量。全量测试 `85/85` 通过。

## 16. T1-5D 执行状态

状态：**完成（2026-10-26）**，rep 验收：**未通过**。详见 `docs/task1-5d_results.md`。

- 在 `64` 路 train、`16` 个 `subject-action`、`96` 个唯一 GT rep 上完成评测。4 路相机只作为同实例重复观测，主统计单位为 `subject-action`。
- 计数采用动作实例内 4 路完整 rep 数中位数（`.5` 向上取整）：MAE `0.625/instance`，完全正确 `6/16`，未达到 `<=0.5` 和 `>=14/16` 验收线。
- 主相机按每实例相对额状面法向角度最大预注册选择；事件使用 Hungarian 一对一匹配，门限为 `0.25 × GT top-to-top 中位周期`。主相机事件绝对误差中位 `12.5` 帧、p90 `25.8` 帧，未达到 `5/12` 帧验收线。
- 230 个全机位匹配事件中 `210` 个预测早于 GT，signed error 中位 `-13` 帧；主相机中位 `-11` 帧。事件存在系统性相位偏早，且不能用一个全局平移安全修复。
- `325` 个已接受完整 rep 中 `94` 个为 FP，其中 `80` 个 `confidence>=0.50`。当前置信度和 `pass/warn` 质量门控不能阻止错误 rep 被包装成完整结果。
- 拒答候选 `305` 个：`280` 个不在 GT 门限内、`25` 个保守拒绝真实事件、`0` 个重复候选。拒答本身安全，但不能抵消已接受的高置信误检。
- `rep_ann` 继续只作为 top/边界事件，Fit3D 无独立 bottom GT，因此 bottom 误差不可计算；官方 rep 起止语义未确认，本轮不报告 IoU。匹配后的派生 top-to-top 间隔/时长中位误差均为 `3` 帧、p90 均为 `10` 帧，只作辅助诊断。
- 结论：当前 T1-4 rep 分割不能作为 Fit3D 验收通过的反馈输入，不应据此启用新产品阈值。后续优先冻结 top 事件定义与边界对齐，再解决单路鲁棒性和置信度校准。
- 产物：`scripts/evaluate_task1_5d_reps.py`、`tests/test_task1_5d_rep_eval.py`、`results/task1_5/rep_evaluation/`。
