# task1-2：QEVD-FIT 姿态角度特征提取与可视化

## 1. 结论

已基于 task1-1 的 MediaPipe 世界坐标完成 5 条 QEVD-FIT-COACH 视频的角度特征提取、平滑、异常处理、身体坐标系转换和曲线检查。

- 处理视频：5 条
- 处理帧数：21,703
- 角度数量：27
- 长表行数：585,981（`21,703 × 27`）
- 宽表列数：29（`frame_id`、`timestamp_ms` 和 27 个角度）
- 有效角度单元：531,661
- NaN 角度单元：54,320（9.27%）
- 跳变异常点：1,526（占全部角度单元 0.26%）
- 曲线图：9 张，覆盖 6 类动作
- 批量角度计算耗时：28.4 s（每条视频 4.58–6.17 s，不含 task1-1 推理）

独立审计脚本完成约 1,015 万项检查，`outputs/angles/audit.json` 状态为 `pass`。所有长表、宽表、meta 统计、NaN/outlier 标记和时间戳一致；身体坐标系为右手正交系，坐标轴正交误差最大 `5.6e-16`，由原始关键点重建身体坐标的舍入误差为 `0`。

## 2. 实现方法

### 2.1 关键点映射

| 关节 | MediaPipe 左/右索引 |
|---|---|
| 肩 | 11 / 12 |
| 肘 | 13 / 14 |
| 腕 | 15 / 16 |
| 髋 | 23 / 24 |
| 膝 | 25 / 26 |
| 踝 | 27 / 28 |
| 足跟 / 足尖 | 29, 31 / 30, 32 |
| 脊柱参考 | 肩中点、髋中点 |

### 2.2 身体坐标系

令肩中点为 `s`、髋中点为 `h`，本实现使用：

1. `y = normalize(h - s)`，即肩指向髋；直立时与重力同向。
2. `x_raw = right_shoulder - left_shoulder`，对 `y` 做 Gram-Schmidt 正交化并归一化。
3. `z = normalize(x × y)`。
4. 以髋中点为原点，将 33 个世界坐标点转换到 `[x, y, z]` 坐标系。

身体坐标写入 `landmarks_body_frame.json`，每帧包含 `frame_id`、`timestamp_ms`、33×3 坐标、`source` 和 `detected`。未检出姿态时坐标为 `null`。

### 2.3 27 个角度

角度命名统一为 `joint + feature`，详细定义见 `outputs/angles/angle_vocabulary.csv`。`deg` 角度均为无符号夹角，范围为 `[0, 180]`。名称中的 `flexion` 表示解剖学关节夹角：伸直接近 180°，屈曲越大，角度越小。

| 类别 | 数量 | 角度 |
|---|---:|---|
| 3D 关节屈曲 | 10 | 左右肩、肘、腕、髋、膝，均由相邻三点构造向量夹角 |
| 躯干特征 | 2 | `torso_flexion`、`spine_stability` |
| 平面投影 | 12 | 左右膝外翻、左右踝背屈、左右肩外展、左右肩水平外展、左右肩伸展、左右髋矢状屈曲 |
| 对称性 | 3 | 左右膝角差、左右腕高差、左右髋角差 |

主要定义：

- `torso_flexion`：躯干向量（肩中点到髋中点）与重力参考（髋中点到踝中点）的夹角；直立时接近 0°。
- `spine_stability`：15 帧中心窗口内单位躯干向量三个分量的方差之和，单位记为 `variance`。
- 膝外翻：髋-膝与踝-膝向量在身体冠状面的投影夹角，再转换为偏离伸直位的角度。
- 踝背屈：小腿向量在矢状面的投影与身体竖直方向的夹角。
- 肩外展：上臂向量在冠状面内相对身体竖直方向的夹角。
- 肩水平外展：上臂在横断面内相对身体侧向轴的夹角，按无符号线角归一到 `[0, 90]`。
- 肩伸展：上臂在矢状面内相对身体竖直方向的夹角。
- 髋矢状屈曲：躯干与大腿向量在矢状面内的投影夹角，再转换为相对伸直位的屈曲量。
- `wrist_height_difference`：左右腕在身体坐标 `y` 轴上的高度差绝对值，单位为米。

### 2.4 置信度与缺失处理

- `confidence`：参与该角度计算的全部关键点 `visibility` 的最小值。
- `visibility < 0.5`：角度置为 `NaN`，记录 `invalid_reason=low_visibility`，不做插值。
- 未检出姿态：该帧全部角度置为 `NaN`，记录 `no_pose`。
- 对称性角度在上游角度缺失时记录 `upstream_nan`。
- 原始列 `angle_raw` 保留平滑前结果；`angle` 为平滑、裁剪和异常处理后的最终值。

### 2.5 平滑与异常处理

- Savitzky–Golay 滤波：窗口 7 帧、2 阶，约 0.23 s（30 FPS）。
- 按连续有效段分别滤波，不跨越缺失段填充。
- 平滑后按角度单位裁剪：`deg` 到 `[0, 180]`，距离和方差非负。
- 仅比较相邻连续有效帧；单帧变化超过 30° 时将该帧置 `NaN`，记录 `angle_jump_gt_30deg`。
- 异常点只标记跳变后的当前帧，不与缺失段两侧跨段比较。

## 3. 质量统计

| video_id | 帧数 | NaN 率 | 异常点 | 低可见度行数 | NaN 最高的角度 |
|---|---:|---:|---:|---:|---|
| `0153` | 3,556 | 13.03% | 513 | 9,138 | `knee_flexion_asymmetry` (29.22%) |
| `0161` | 4,489 | 9.56% | 215 | 6,577 | `knee_flexion_asymmetry` (16.62%) |
| `0162` | 4,444 | 13.88% | 147 | 14,821 | `knee_flexion_asymmetry` (33.42%) |
| `0173` | 4,505 | 3.35% | 400 | 2,942 | `right_wrist_flexion` (16.18%) |
| `0218` | 4,709 | 7.46% | 251 | 4,392 | `left_wrist_flexion` (24.51%) |
| **合计** | **21,703** | **9.27%** | **1,526** | **37,870** | - |

NaN 原因合计与全部 NaN 单元严格一致：

| 原因 | 行数 | 占比 |
|---|---:|---:|
| `low_visibility` | 37,870 | 6.46% |
| `no_pose` | 11,070 | 1.89% |
| `upstream_nan` | 3,853 | 0.66% |
| `angle_jump_gt_30deg` | 1,526 | 0.26% |
| `insufficient_window` | 1 | <0.01% |

其中 `no_pose` 为 `410 帧 × 27 角度`，与 task1-1 的未检出帧数完全对应。

按 5 条视频平均的 NaN 率，缺失最高的角度依次为：

| 角度 | NaN 率 |
|---|---:|
| `knee_flexion_asymmetry` | 17.72% |
| `torso_flexion` | 17.30% |
| `left_wrist_flexion` | 15.44% |
| `left_knee_valgus_coronal` | 15.42% |
| `left_ankle_dorsiflexion_sagittal` | 15.28% |
| `left_knee_flexion` | 15.10% |

左侧膝踝链、腕部和躯干角度缺失相对较多，主要来自侧面或局部遮挡时的低可见度。`spine_stability` 平均 NaN 率仅 1.93%，但它在有效窗口内依赖躯干向量，不能把低方差直接解释为动作质量好。

## 4. 可视化检查

共生成 9 张图，索引见 `outputs/angles/plots/index.json`，背景阴影标出反馈对应的帧区间。

| 视频 | 动作 | 曲线 |
|---|---|---|
| `0153` | squat | 左右膝屈曲、躯干屈曲 |
| `0153` | jumping jack | 左右肩外展、左膝屈曲 |
| `0161` | pushup | 左右肘屈曲、躯干屈曲 |
| `0161` | plank | 躯干屈曲、脊柱方差 |
| `0162` | jumping jack | 左右肩外展、左膝屈曲 |
| `0162` | lunge | 左右膝屈曲、躯干屈曲 |
| `0173` | squat | 左右膝屈曲、躯干屈曲 |
| `0173` | armcrosschest | 左右肩水平外展、躯干屈曲 |
| `0218` | lunge | 左右膝屈曲、躯干屈曲 |

肉眼检查结论：

- 深蹲：`0153` 的左右膝角呈现清晰屈伸周期，底部为约 85–100° 的谷值，站立约 160–165°；`0173` 的同一屈伸形态清晰。左右曲线基本同步，未见过度毛刺。
- 开合跳：`0153`、`0162` 的左右肩外展同步升降，肩角从接近体侧升至约 130–145°，相位差较小，可作为左右同步性的 sanity check。
- 俯卧撑：`0161` 的左右肘角同步屈伸，约在 90° 与 170° 之间变化，符合俯卧撑最低点与撑起位置的交替。
- 弓步：`0162`、`0218` 能看到左右膝角交替变化和一次清晰屈曲过程；片段内周期数有限，不能据此声称周期性已经得到充分验证。
- 手臂交叉：`0173` 的左右肩水平外展曲线平滑，动作区间内出现同步变化。
- 平板：`0161` 的片段包含 moving plank/过渡动作，躯干曲线并不平稳；该图只能验证信号连续性和方差计算，不能当作标准静态平板的质量证据。

## 5. 局限与后续验证

- 当前 5 条视频没有真实侧平举样本；开合跳只能验证左右肩外展的同步性，不能验证侧平举幅度。
- 膝屈曲、肘屈曲等均为无符号夹角，不能区分屈曲方向；方向信息应由后续模型结合时序和身体坐标使用。
- MediaPipe 世界坐标在遮挡、侧面视角和快速换向时仍有噪声；本任务只做平滑和跳变剔除，没有伪造成对缺失帧的插值。
- 所有结果是可视化和结构 sanity check，没有逐帧人体测量 ground truth；准确的定量误差、分割 IoU 和动作评估指标留待 task 2.1。
- `meta.json` 的 `view_label` 当前统一为 `unknown`，后续需要人工补充视角标签。
- `angles_wide.csv` 只保存角度值；confidence、source、valid、outlier 和 NaN 原因保存在长表 `angles.csv`，训练建模时应按需从长表关联。

## 6. 产物

| 路径 | 内容 |
|---|---|
| `scripts/extract_angles.py` | 批量角度提取、平滑、异常处理、身体坐标转换 |
| `scripts/plot_angles.py` | 基于反馈区间的关键角度曲线图 |
| `scripts/audit_angles.py` | 长宽表、meta、视角坐标与曲线产物审计 |
| `outputs/angles/angle_vocabulary.json` | 27 个角度的机器可读词表 |
| `outputs/angles/angle_vocabulary.csv` | 角度词表表格版 |
| `outputs/angles/quality_report.csv` | 每条视频、每个角度的有效数、NaN、异常点和低可见度统计 |
| `outputs/angles/summary.json` | 批量运行汇总与平滑参数 |
| `outputs/angles/audit.json` | 独立审计结果与逐视频几何误差 |
| `outputs/angles/plots/*.png` | 9 张角度曲线图 |
| `outputs/angles/{video_id}/angles.csv` | 27 角度长表，含 confidence/source/NaN 原因 |
| `outputs/angles/{video_id}/angles_wide.csv` | 帧 × 27 角度宽表 |
| `outputs/angles/{video_id}/landmarks_body_frame.json` | 身体坐标系下的逐帧 33×3 关键点 |
| `outputs/angles/{video_id}/meta.json` | fps、平滑参数、逐角度质量统计 |

`angles.csv` 字段为：`frame_id, timestamp_ms, joint, feature, angle_name, plane, unit, angle, angle_raw, confidence, source, valid, outlier, invalid_reason`。`source` 预留为 `vision`，未来 IMU 结果可按相同 schema 追加。

## 7. 复现

在仓库根目录 `/mnt/storage2/data_guest` 执行：

```bash
.venv/bin/python VLS/V0/scripts/extract_angles.py --overwrite
.venv/bin/python VLS/V0/scripts/plot_angles.py
.venv/bin/python VLS/V0/scripts/audit_angles.py
```

task1-2 直接依赖见 `requirements-task1-2.txt`；输入来自 task1-1 的 `outputs/pose_landmarker_full/{video_id}_landmarks.json` 与 `{video_id}_meta.json`。若从空的 Python 环境重建整条链路，还需按 `requirements-task1-1.txt` 安装 MediaPipe 并先运行 task1-1。
