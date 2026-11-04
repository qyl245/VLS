**在服务器上跑通 MediaPipe Pose Landmarker 的 VIDEO 模式，对 QevD-Fit-COACH 的长视频做逐帧 3D 姿态提取，并测量处理速度**。不追求精度调优，只验证“能稳定输出 33 个关键点 + 世界坐标 + 置信度”。

### 前置准备

确认 `mediapipe` 已安装且版本 ≥ 0.10，模型文件 `pose_landmarker_full.task` 已放在服务器可访问路径。QevD-Fit-COACH 解压后，长视频在 `long_range_videos/` 下，反馈标注在 `feedbacks_long_range.json` 中（时间戳为 UNIX 格式，本任务先不用）。

### 步骤一：跑通单视频 VIDEO 模式

MediaPipe 的 VIDEO 模式要求逐帧传入 `mp.Image` 并附带**单调递增的时间戳（毫秒）**，不能用 IMAGE 模式逐帧独立推理。关键配置：

```python
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)
```

对每个视频帧调用 `landmarker.detect_for_video(mp_image, timestamp_ms)`，返回的 `PoseLandmarkerResult` 包含两组关键数据：

- `pose_landmarks`：归一化图像坐标（x, y 在 [0,1]，z 为相对深度）
- `pose_world_landmarks`：**世界坐标（米级，以髋部中心为原点）**——这是你后续计算角度特征要用的，比归一化坐标更适合跨帧、跨人的角度对比。

**本步骤验收**：从 `long_range_videos/` 选 1 个视频，成功跑完，输出每帧 33 个关键点的 `pose_world_landmarks` 和 visibility 分数到 JSON/CSV。

### 步骤二：批量处理 3–5 个长视频

从 QevD-Fit-COACH 的 149 个训练视频中，按动作类型抽 3–5 个（覆盖深蹲、开合跳、俯卧撑、弓步等），批量跑姿态提取。每个视频输出两个文件：

- `{video_id}_landmarks.json`：逐帧 `pose_world_landmarks`（33×3） + `visibility`（33）
- `{video_id}_meta.json`：视频帧率、总帧数、处理耗时、检测失败的帧索引

**注意**：QevD 长视频约 3.5 分钟，30fps 约 6300 帧。如果处理速度太慢，可以先降采样（每 2 帧取 1 帧），但**记录降采样策略**，后续角度计算时按原始帧率还原时间轴。

### 步骤三：速度与稳定性统计

在服务器上记录：

- **单帧平均处理耗时**（ms）
- **端到端 FPS**（总帧数 / 总耗时）
- **检测失败率**（`pose_landmarks` 为空的帧数 / 总帧数）

参考基准：MediaPipe Pose Landmarker 在移动端 GPU 上约 12–18ms/帧，服务器 CPU 上会慢一些，但你的目标是先跑通，速度记录用于后续决定是否需要降采样或换 lite 模型。

### 步骤四：输出格式定版

按周计划里约定的 schema 输出，至少包含：

```json
{
  "frame_id": 0,
  "timestamp_ms": 0,
  "landmarks_world": [[x, y, z], ...],  // 33个
  "visibility": [v0, v1, ...],           // 33个
  "source": "vision"
}
```

后续 IMU 融合时，`source` 字段区分来源，不改结构。

### 交付物

| 交付物                         | 内容                                                  |
| ------------------------------ | ----------------------------------------------------- |
| 姿态提取脚本                   | 输入视频目录，输出逐帧 JSON                           |
| 3–5 个 QevD 长视频的姿态 JSON | 含 world landmarks + visibility                       |
| 速度统计表                     | 单帧耗时、FPS、失败率                                 |
| 问题记录                       | 哪些帧/动作检测失败，原因初步判断（遮挡？快速运动？） |
