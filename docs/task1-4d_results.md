# task1-4D：代表帧圈画与 HTML 回放报告

## 1. 执行结论

状态：**完成（2026-10-26 更新）**

- 新增 `scripts/render_clip_report.py`，消费 T1-4C 的 `issues.json` 与 T1-4B 的 `analysis.json`，生成静态 HTML 报告和代表帧标注图。
- 13 个 dev clip 全部生成报告，共 47 张活动问题标注帧；47 张图均成功恢复姿态，无 text-only 回退。其中 4 张因单帧 `IMAGE` 检测缺失或关键关节不可信，自动改用 `VIDEO` 跟踪。
- 每个报告生成一个裁剪到 clip 区间的 VP8/WebM 代理视频，解决源 MP4 的 `MPEG-4 Visual / FMP4` 编码无法被主流浏览器播放的问题。
- 每张活动问题卡都可点击定位到对应的 clip 相对 `anchor_time_s` 并暂停；证据区间会等待 `seeked` 完成后再播放，避免异步 seek 被提前终止。
- 报告包含组级摘要、质量/未评估项、最多 3 张默认问题卡、完整问题列表、rep/bottom/证据时间线、逐 rep 明细和原始数据下载。
- 原始长视频保留“仅下载”入口；报告播放使用裁剪代理，不复制完整长视频。
- 质量警告改为中文展示；当姿态门控剔除热身/起身 rep 时，报告明确说明这些动作未参与点评。被 suppression 或重新编号淘汰的旧标注帧会自动清理。

## 2. 输出结构

```text
VLS/V0/results/reports_v1/
├── <clip_id>/
│   ├── report.html
│   ├── media/
│   │   └── <clip_id>_clip.webm
│   └── annotated_frames/
│       └── <issue_id>_frame_<anchor_frame>.png
├── index.html
└── summary.json
```

`summary.json` 记录 13 个 clip、13 个 Web 代理视频、47 张标注帧、47 张活动问题卡、姿态回退次数和检测失败数。`index.html` 提供全部 clip 的报告索引。

## 3. 标注帧实现

### 3.1 定点姿态恢复与合理性门控

T1-4B 的 `landmarks.json` 只保存世界坐标和 visibility，没有图像归一化坐标。T1-4D 对每个活动问题的 anchor 帧重新运行 MediaPipe：

1. 默认使用 `RunningMode.IMAGE` 单帧检测；
2. 单帧结果缺失，或关键关节 visibility `< 0.5`、俯卧撑肩-腕/深蹲髋-膝-踝几何塌缩到同一点时，判定该姿态不可信；
3. 不可信时从 anchor 前 30 帧开始用 `RunningMode.VIDEO` 跟踪到目标帧，并再次执行同一套合理性校验；
4. 跟踪结果仍不可信时保留原画面和文字说明，不绘制虚假骨架。

本次 47 张图中有 4 张由 VIDEO 跟踪回退恢复：

- `0001_pushup_set01` / `pushup_stability_low_01` / frame `2895`
- `0001_pushup_set01` / `pushup_stability_low_02` / frame `3090`
- `0004_pushup_set01` / `pushup_asymmetry_01` / frame `2957`
- `0156_pushup_set01` / `pushup_asymmetry_02` / frame `2623`

最终 `pose_detection_failures=0`。

其中 `0004_pushup_set01` frame `2957` 的单帧结果把手肘/手腕压到头部附近，且 visibility 仅约 `0.09`；该结果此前被直接绘制，现在会触发跟踪回退并使用正确的全身姿态。

### 3.2 视觉语法

| 元素 | 实现 |
|---|---|
| 绿色骨架 | MediaPipe 33 点标准连接 |
| 黄色关节 | 当前可见但未作为问题焦点的关节 |
| 红色关节点/角度弧 | 当前问题高亮关节和超阈值几何 |
| 虚线参考线 | 膝内扣的髋-踝对齐线、阈值辅助弧 |
| 红色箭头 | 膝相对髋-踝参考线的内侧偏移方向 |
| 灰色关节点 | visibility `< 0.5` 的不可信关节 |
| 半透明阴影 | 稳定性问题的躯干重点区域 |
| 文字面板 | 标题、rep/frame/time/confidence、测量值/阈值、可操作 cue |

问题类型对应的几何：

- 膝内扣：髋-膝-踝骨架、髋-踝虚线、内侧偏移箭头、膝部角度弧；
- 左右不对称：按 primary metric 选择膝/髋/肘/肩，左右两侧角度弧和肘/肩数值；
- 稳定性：肩中点到髋中点的躯干线 + 半透明椭圆重点区域；
- 节奏：骨架 + 报告时间线中的证据色带，不伪造关节角度。

## 4. 报告交互

- 点击问题卡的覆盖按钮：`video.currentTime = clip 相对 anchor_time_s`，暂停播放并滚动到播放器；若视频 metadata 尚未加载，seek 会排队到 `loadedmetadata` 后执行。
- 点击 `rep` 色带：跳到该 rep 起点。
- 点击 bottom 标记：跳到该 rep 的 bottom 帧。
- 点击问题证据色带：跳到问题 anchor，并记录证据起止时间。
- `播放当前证据区间`：先 seek 到 evidence start，收到 `seeked` 后调用 `play()`；到 evidence end 自动暂停，并允许再次播放。
- 视频 metadata 加载后默认定位到 clip 代理的 `0` 秒。
- 源视频为 MPEG-4 Visual，浏览器无法直接解码；报告播放 `media/<clip_id>_clip.webm`，原始 MP4 仅保留下载链接。
- 必须使用 `scripts/serve_reports.py` 等支持 HTTP Range 的本地服务。普通 `python -m http.server` 对 WebM 请求返回 `200 OK` 而非 `206 Partial Content`，视频未完整缓冲时会导致问题卡、时间线和 rep 明细的 seek 失效。

播放器、时间线、rep 表格和标注图统一使用 clip 相对时间；`frame` 编号仍保留 T1-4B 的原始视频帧号，便于追溯。

## 5. Dev 集输出

| clip | 活动问题 | 标注帧 | high | medium | low | 回退 |
|---|---:|---:|---:|---:|---:|---:|
| `0001_pushup_set01` | 8 | 8 | 0 | 1 | 7 | 2 |
| `0004_pushup_set01` | 3 | 3 | 0 | 0 | 3 | 1 |
| `0005_pushup_set01` | 2 | 2 | 0 | 1 | 1 | 0 |
| `0007_pushup_set01` | 0 | 0 | 0 | 0 | 0 | 0 |
| `0020_pushup_set01` | 5 | 5 | 0 | 1 | 4 | 0 |
| `0020_squat_set01` | 4 | 4 | 1 | 2 | 1 | 0 |
| `0021_pushup_set01` | 5 | 5 | 0 | 0 | 5 | 0 |
| `0034_squat_set01` | 4 | 4 | 1 | 0 | 3 | 0 |
| `0046_squat_set01` | 5 | 5 | 0 | 2 | 3 | 0 |
| `0104_squat_set01` | 3 | 3 | 0 | 1 | 2 | 0 |
| `0156_pushup_set01` | 4 | 4 | 1 | 2 | 1 | 1 |
| `0156_squat_set01` | 3 | 3 | 0 | 1 | 2 | 0 |
| `0161_pushup_set01` | 1 | 1 | 0 | 0 | 1 | 0 |
| **合计** | **47** | **47** | **3** | **11** | **33** | **4** |

`0007_pushup_set01` 没有活动问题，报告仍包含质量、时间线、逐 rep 明细和未评估项，并明确提示“未检测到活动问题不代表动作正常”。

`0004_pushup_set01` 报告现在展示后段 5 个真实俯卧撑及其 3 个低严重度问题，不再点评前段热身；`0005_pushup_set01` 的 2 个问题分别落在真实跪姿俯卧撑帧 `1999` 和 `2150`，原起身帧不再生成卡片。`0007_pushup_set01`、`0004_pushup_set01` 和 `0005_pushup_set01` 重生成时分别清理了 0、1、3 张陈旧标注帧。

## 6. 复现

```bash
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/render_clip_report.py --split dev

# 仅重绘 HTML，复用已有标注帧
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/render_clip_report.py --split dev --skip-frames

# 仅重绘 HTML，复用已有标注帧和 Web 代理视频
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/render_clip_report.py --split dev --skip-frames --skip-video

# 启动支持 Range 的报告服务
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/serve_reports.py --port 8000

# 单个 clip
VLS/V0/.venv-mediapipe/bin/python VLS/V0/scripts/render_clip_report.py \
  --clip-id 0020_squat_set01 --split dev
```

## 7. 校验与测试

- 47/47 活动问题有 `annotated_frames/<issue_id>_frame_<anchor>.png`。
- 13/13 报告和 13/13 VP8/WebM 代理存在；所有代理均为 30 FPS、帧数等于对应 clip 的 `trim` 帧数，并可由 OpenCV/GStreamer 解码和 seek。
- 13/13 报告存在，138 个视频/图片/下载资产引用全部可解析，无断链。
- `serve_reports.py` 的 Range 请求实测返回 `206 Partial Content`、正确的 `Content-Range` 和 `Content-Length`。
- HTML 中每个活动 issue 都包含对应的 `selectIssue(...)` seek 参数。
- 新增 `tests/test_task1_4_report.py`，覆盖时间线位置、rep/bottom/证据色带、骨架+文字叠加、姿态合理性门控、无姿态 fallback、中文标点换行、VP8 代理裁剪与编码、HTML 转义、seek 参数、metadata 未就绪时的延迟 seek、Range 服务提示、suppressed 展示、质量警告中文映射、无 rep 说明、陈旧标注帧清理和索引链接。
- 全量测试 `56/56` 通过。

当前环境没有 Chromium/Firefox，因此未做真实浏览器截图回归；已完成 HTML 结构解析、资产路径检查和 JS/交互代码审阅。这不替代 T1-4D 的人工浏览器验收。

## 8. 已知限制

- 阈值仍是 provisional，问题卡内容不能作为准确率证据。
- 代表帧是单帧静态圈画；证据区间通过 clip 代理视频回看。
- Web 代理仅覆盖 clip 区间，不是完整长视频；播放器时间因此为 clip 相对时间。
- VP8/WebM 在现代 Chrome、Edge 和 Firefox 中可用；旧版 Safari 需要另做 H.264 代理。
- 当前只在 front-view dev 集验证，矢状面问题的圈画逻辑虽已预留，但没有真实侧视样本。
- 4 张标注图依赖 VIDEO 跟踪回退，说明单帧姿态检测在俯卧撑底部仍可能丢失或产生低可见度误检；报告必须在图上保留该帧和文字，不能用邻近帧冒充。
- 相机运动仍为 `not_evaluated`；手持视频的稳定性问题需要后续加入门控或警告。
- 姿态门控阈值仍是 provisional；跪姿俯卧撑与标准俯卧撑目前共用同一套指标和文案。

## 9. 下一步

T1-4E 可在现有报告上继续做：

1. 动作识别与 `exercise_hint` 缺失时的拒答；
2. holdout 集问题定位与 anchor/evidence 误差评估；
3. 小规模专家可用性评分，验证“圈画位置、文字可操作性和回放理解速度”；
4. 在真实浏览器中完成播放、seek、证据区间自动暂停和再次播放验收，再冻结 T1-4D 报告快照与校验清单。
