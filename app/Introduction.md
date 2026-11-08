
这是一个面向**健身动作复盘的产品原型**，核心不是生成式 AI，也不是完整健身教练系统，而是：

> 用 MediaPipe 提取人体姿态 → 将关节角度转换成周期信号 → RepChain 自动切分动作次数 → 量化动作指标 → 通过规则和置信度门控生成问题卡 → 在视频时间轴上回放证据。

它目前主要支持**俯卧撑和深蹲**，适合单人、单机位、3–120 秒的视频。项目说明见 [README.md](README.md)。

## 整体处理流程

```text
上传视频
  ↓
FastAPI 创建任务，SQLite 排队
  ↓
探测视频参数、生成任务 manifest
  ↓
MediaPipe Pose Landmarker 提取人体关键点
  ↓
计算关节角度、身体姿态和动作质量
  ↓
RepChain v3 识别动作周期，切分每次 rep
  ↓
阶段化测量：底部姿态、动作幅度、左右差异、节奏等
  ↓
质量门 + 置信度 + 问题规则
  ↓
问题卡、证据帧、证据时间区间
  ↓
HTML 报告 + Vue 交互式回放页面
```

后端阶段定义在 [stages.py](web/backend/app/services/pipeline/stages.py#L18)，依次为：

`validate_upload → probe_video → build_job_manifest → analysis_v1 → repchain → issues → feedback_v2 → build_report → validate_artifacts → publish_complete`

## 核心算法：RepChain

RepChain 是该项目最有技术含量的部分，位于 [repchain](web/backend/app/repchain)。

它不是直接看视频猜 “这是一次深蹲”，而是：

1. 使用预训练的 MediaPipe Pose Landmarker 逐帧识别一个人的人体关键点。
2. 计算肘、膝、髋、肩、躯干等关节角度。
3. 自动选择周期性和动态范围较好的角度信号。例如俯卧撑优先看左右肘部弯曲，深蹲优先看左右膝或髋部弯曲。
4. 检测动作顶部、底部等相位状态。
5. 构建候选动作节点和连接边。
6. 使用 **Viterbi 路径选择**挑出最连续、最合理的一串 rep，相关代码在 [repchain_chain_selection.py](web/backend/app/repchain/repchain_chain_selection.py#L726)。
7. 处理低幅度动作、动作间断、首尾不完整、局部漏检等情况。
8. 给每个 rep 和整条 rep 链计算置信度。
9. 最后适配成旧版分析契约，供测量、评估和报告模块继续使用。

[repchain_pipeline.py](web/backend/app/repchain/repchain_pipeline.py) 是新版 RepChain 与旧版分析格式之间的适配层。最终周期通过 `vls-rep-selection-v1` 契约显式记录选择模式和边界来源，详见 [REP_CONTRACT.md](REP_CONTRACT.md)。如果 RepChain 执行失败，整个任务不会立刻失败，而是按契约降级到旧版 `analysis_v1` 的切分结果。

## 后端结构

后端是 FastAPI：

- [main.py](web/backend/app/main.py)：应用入口、API 路由、前端静态文件托管。
- `api/v1/jobs.py`：上传、任务状态、任务结果、取消、视频和证据帧接口。
- `api/v1/samples.py`：冻结样本库、样本详情、报告和产物浏览。
- `worker/manager.py`：单机任务队列，目前只有一个 worker，因此任务串行执行。
- `storage/db.py`：SQLite 保存任务状态，`status.json` 只是便于人工查看的镜像。
- `services/registry.py`：把分析、RepChain、问题卡和反馈 JSON 组装成前端需要的统一 DTO。
- `configs/`：动作规则、阈值、质量门、问题分类和反馈模板。
- `schemas/`：分析结果和反馈结果 JSON Schema。

这种设计比较强调**工程可追溯性**：配置文件有版本和部分 SHA256 依赖，运行产物经过完整性检查，并生成 `complete.json` 清单和 SHA256。

## 前端能力

前端是 Vue 3 + Vite，入口在 [App.vue](web/frontend/src/App.vue)，主要页面包括：

- **样本库**：浏览已冻结的分析样本。
- **我的分析**：查看上传任务和重分析任务，轮询进度。
- **分析详情**：视频播放、rep 时间轴、证据区间播放、问题卡、置信度、质量门和动作契约。
- **报告页**：浏览已经生成的 HTML 报告。

分析详情页会把视频时间和问题证据联动起来，并支持 URL 携带时间点和 rep，方便分享具体问题位置。核心页面是 [AnalysisView.vue](web/frontend/src/views/AnalysisView.vue)。

## 当前能做什么

当前主要指标包括：

- 动作周期时长、节奏一致性。
- 膝、肘、髋的活动幅度。
- 底部膝角、肘角。
- 躯干前倾、身体直线、髋部下沉。
- 左右膝、髋、肘、肩的对称性。
- 底部阶段稳定性。

系统会根据用户提供的机位选择可用指标：

- 侧视或斜侧视主要启用矢状面指标。
- 正视或背视主要启用左右对称和膝内扣等额状面指标。
- 视角选择不正确时，相关指标会被禁用。

## 关键限制

这是整个项目最重要的部分：

- **目前是研究 / Shadow 状态**。`assessment_rules.json` 明确写着 `expert_calibrated: false`、`shadow_only`、`product_actions_enabled: false`。
- 反馈会明确标注“研究候选反馈”，不能替代教练或医学判断。
- 大多数动作质量指标仍被标记为 `diagnostic_only` 或 `experimental`。
- 当前真正允许形成较完整问题决策的，主要是节奏一致性，以及正视/背视下的部分对称性指标；膝角、肘角、身体直线等更偏诊断信息。
- 只有两个侧视俯卧撑演示样本，没有深蹲演示样本。
- 只支持一个人，使用 `num_poses=1`。
- 用户必须自己选择动作类型和拍摄视角，系统没有可靠的全自动视角识别。
- 当前是本地单机原型：单 worker、SQLite、没有用户认证、权限和分布式任务调度。
- 计算使用 MediaPipe CPU 推理，没有 GPU 也可以运行。
- 当前 `results/` 和 `webdata/` 中包含运行时数据。README 认为这些不应作为源码发布，部署时应通过 `VLS_WEBDATA_ROOT` 放到源码目录之外。
- 如果要在 Windows 上用 `D:\Anaconda` 运行，Makefile 中的 `.venv/bin/python` 和 shell 写法不能直接照搬，应直接使用 Anaconda 的解释器运行 Uvicorn 和 npm 命令。

一句话评价：**项目的数据链路、动作切分、证据回放和工程契约做得比较完整，已经是一个可演示、可复核的动作分析原型；但动作问题判断仍处于研究候选阶段，尚不能作为成熟自动教练或医学产品使用。**
