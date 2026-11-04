# task1-1：QEVD-FIT MediaPipe Pose Landmarker 长视频姿态提取

## 1. 结论

已在服务器上跑通 MediaPipe Pose Landmarker 的 `VIDEO` 模式，并完成 5 条 QEVD-FIT-COACH 长视频的逐帧姿态提取。每条视频均输出 33 个世界坐标关键点及其 visibility，VIDEO 时间戳严格单调递增，所有文件通过 schema 与数值校验。

- 处理视频：5 条
- 处理帧数：21,703
- 成功检出：21,293 帧
- 未检出：410 帧
- 总失败率：1.89%
- 端到端速度：34.46 FPS
- 单帧平均推理耗时：27.85 ms
- 总处理耗时：629.8 s（约 10.5 min）

## 2. 环境与模型

| 项目 | 值 |
|---|---|
| Python | 3.10.12 |
| MediaPipe | 0.10.21 |
| OpenCV | 4.11.0 |
| NumPy | 1.26.4 |
| 推理后端 | MediaPipe Tasks，VIDEO 模式，XNNPACK CPU |
| 模型 | `models/pose_landmarker_full.task` |
| 模型大小 | 9,398,247 B |
| 模型 SHA-256 | `eb8e404e1aa4e7fa4c017ce911f10d93199fad28d21eedc3aab0657e77cb6182` |

共享根 `.venv` 的 `protobuf 6.33.6` 与 MediaPipe 0.10.21 要求的 `protobuf<5` 冲突。本次运行使用隔离依赖目录，未修改根 `.venv`。复现时建议使用独立的 `VLS/V0/.venv-mediapipe`。

## 3. 视频选择

从 149 条长视频中选择 5 条未旋转视频，覆盖深蹲、开合跳、俯卧撑和弓步。选择依据为 `feedbacks_long_range.json` 中的动作关键词，不代表人工逐帧复核。

| video_id | 覆盖动作 | 帧数 |
|---|---|---:|
| `0153` | squat, jumping jack, push-up | 3,556 |
| `0161` | jumping jack, push-up, lunge | 4,489 |
| `0162` | jumping jack, lunge | 4,444 |
| `0173` | squat, push-up | 4,505 |
| `0218` | squat, lunge | 4,709 |

选择配置：`configs/task1_1_selection.json`。

## 4. 速度与稳定性

| video_id | 帧数 | 检出 | 失败 | 失败率 | 端到端 FPS | 平均推理 ms | P95 推理 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0153` | 3,556 | 3,487 | 69 | 1.94% | 34.17 | 28.08 | 30.85 |
| `0161` | 4,489 | 4,342 | 147 | 3.27% | 33.86 | 28.37 | 31.13 |
| `0162` | 4,444 | 4,441 | 3 | 0.07% | 34.52 | 27.79 | 30.68 |
| `0173` | 4,505 | 4,478 | 27 | 0.60% | 34.45 | 27.87 | 30.75 |
| `0218` | 4,709 | 4,545 | 164 | 3.48% | 35.23 | 27.22 | 30.46 |
| **合计** | **21,703** | **21,293** | **410** | **1.89%** | **34.46** | **27.85** | - |

所有视频的 `decode_failures` 均为 0，说明检测失败来自姿态模型未返回结果，而不是视频解码中断。

## 5. 失败区间与初步判断

逐区间记录见 `outputs/pose_landmarker_full/failure_ranges.csv`。主要区间如下：

| video_id | 帧区间 | 时长 | 上下文 | 初步判断 |
|---|---|---:|---|---|
| `0153` | 2801–2863 | 2.1 s | quad stretch left | 人物靠近镜头，上半身/支撑腿被裁切 |
| `0161` | 1215–1273 | 1.9 s | "Don't leave!" | 人物离开有效画面区域 |
| `0161` | 4391–4475 | 2.8 s | 片尾 | 人物已离开或只剩局部肢体 |
| `0162` | 2843–2845 | 0.1 s | 无 | 3 帧瞬态失败，下一帧恢复 |
| `0173` | 0–10.3 s | 10.3 s 内 27 帧 | 片头 / good morning beginner | 片头构图不稳，并有少量单帧漏检 |
| `0218` | 0–13 | 0.43 s | 片头 | 人物尚未进入稳定检测区域 |
| `0218` | 4559–4708 | 5.0 s | session end | 训练结束后人物离开画面 |

未发现“快速动作导致连续大段跟踪失败”的证据。大部分失败是人物离开画面、身体局部被裁切或片头片尾空镜；`0162` 属于短暂漏检。

## 6. 输出格式

每帧记录包含：

```json
{
  "frame_id": 0,
  "timestamp_ms": 0,
  "landmarks_world": [[x, y, z]],
  "visibility": [0.99],
  "source": "vision",
  "detected": true
}
```

字段说明：

- `landmarks_world`：33 个世界坐标点，单位约为米，原点为髋部中心。
- `visibility`：33 个关键点的可见度，范围为 `[0, 1]`。
- `timestamp_ms`：按容器帧率计算的相对时间，`round(frame_id * 1000 / fps)`。
- `detected=false` 时，`landmarks_world` 和 `visibility` 均为 `null`。
- 本次未使用 `_timestamps.npy` 或反馈时间戳生成时间轴。

校验结果：5 条视频全部通过 `validation.json`，包括帧数、检出数、失败索引、时间戳单调性、33×3 坐标、数值有限性、visibility 范围和髋部原点检查。

## 7. 产物

| 路径 | 内容 |
|---|---|
| `scripts/extract_pose_landmarker.py` | 主提取脚本，支持 VIDEO 模式、批量视频、stride、固定或自动方向 |
| `scripts/audit_extractions.py` | 输出 schema 校验与连续失败区间汇总 |
| `configs/task1_1_selection.json` | 5 条视频及动作覆盖配置 |
| `models/pose_landmarker_full.task` | 官方 full 模型 |
| `outputs/pose_landmarker_full/*_landmarks.json` | 逐帧姿态结果 |
| `outputs/pose_landmarker_full/*_meta.json` | 单视频元数据、失败帧和耗时 |
| `outputs/pose_landmarker_full/summary.json` | 批量汇总 |
| `outputs/pose_landmarker_full/speed_stats.csv` | 速度统计表 |
| `outputs/pose_landmarker_full/failure_ranges.csv` | 连续失败区间与反馈上下文 |
| `outputs/pose_landmarker_full/validation.json` | 自动校验结果 |

## 8. 复现

```bash
python3 -m venv VLS/V0/.venv-mediapipe
VLS/V0/.venv-mediapipe/bin/pip install -r VLS/V0/requirements-task1-1.txt
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/extract_pose_landmarker.py --overwrite
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/audit_extractions.py
```

QEVD-FIT 中大量长视频以 90° 旋转存放。本次 5 条样本均为正向；处理其他视频时使用 `--rotation auto`，或逐视频明确指定 `90cw`、`180`、`90ccw`。自动方向模式已在 `0000.mp4` 上验证，可选择 `90cw`。
