# VLS V0：离线健身动作反馈系统

V0 是面向“训练结束后回放分析”的第一版纯视觉原型，覆盖：

```text
视频 clip → 姿态/角度 → 动作分段与 rep → 结构化问题卡 → 可视化回放报告
```

当前重点验证深蹲和俯卧撑的 rep 分割、问题定位、代表帧圈画与 HTML 报告。

## 目录

```text
scripts/    分析与报告生成脚本
configs/    动作规则、问题模板、clip 清单和回归 golden
docs/       各阶段结果、标注说明与校验记录
Plan/       任务设计与迭代计划
tests/      单元测试和回归测试
models/     MediaPipe Pose Landmarker 模型
outputs/    生成性姿态/角度数据（不纳入 Git）
results/    分段、问题卡与可视化报告产物（不纳入 Git）
```

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-task1-1.txt \
  -r requirements-task1-2.txt \
  -r requirements-task1-3.txt
```

## 测试

```bash
python -m unittest discover -s tests -p 'test*.py'
```

## 报告复现

T1-4 默认读取 `results/clip_analysis_v1/` 和 `results/feedback_v1/`，并将产物写入 `results/reports_v1/`：

```bash
python scripts/render_clip_report.py --split dev
```

仅重绘 HTML、复用已有标注帧和 Web 视频：

```bash
python scripts/render_clip_report.py --split dev --skip-frames --skip-video
```

启动支持 HTTP Range 的本地预览服务：

```bash
python scripts/serve_reports.py --port 8000
```

浏览器打开 `http://localhost:8000/results/reports_v1/index.html`。报告播放使用裁剪后的 VP8/WebM 代理视频，源 MP4 仅保留下载链接。

## 数据与产物说明

- QEVD-FIT 原始数据和由它派生的 `outputs/`、`results/` 体积较大，未提交到 Git。
- clip 清单中的视频路径可能保留生成时的工作站绝对路径；迁移环境后需要重新生成本地 manifest。
- 当前阈值仍是 provisional，报告用于工程验证与回放定位，不作为医学或动作标准结论。

