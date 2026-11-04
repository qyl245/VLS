# VLS App：离线健身动作反馈产品区

`app/` 只接收已经验证过的产品补丁，不是日常开发目录。产品闭环为：

```text
上传视频 -> 姿态/角度 -> RepChain 动作分段 -> 问题卡 -> 反馈 v2 -> 回放报告
```

## 目录边界

- `V0/`：算法与研究开发区，保留完整开发历史和实验资产。
- `V0-web/`：web 联调开发区，后续功能和修复仍在这里开发和跑全量测试。
- `app/`：产品发布区，只保留 web 后端、前端和 RepChain 运行闭环。

`app/` 不保存历史任务计划、研究文档、测试集、历史结果或实验脚本。需要回溯时去
`V0/`、`V0-web/`；需要修改产品时，先在开发区验证，再向 `app/` 打最小补丁。

## 产品目录

```text
web/backend/                 FastAPI 服务、存储和上传流水线
web/backend/app/repchain/    RepChain 与报告运行模块
web/frontend/                Vue 前端源码
configs/                     运行时规则与模板
schemas/                     运行时输出契约
models/                      MediaPipe Pose Landmarker 模型
```

`results/`、`outputs/`、`webdata/`、`dist/`、`node_modules/` 均不是源码。上传任务和
报告默认写入 `VLS_WEBDATA_ROOT`，部署时应放在源码目录之外。

## 运行时配置

`configs/` 中的文件都参与上传分析闭环，不是历史任务配置：

| 配置 | 作用 |
| --- | --- |
| `exercise_specs.json` | 动作定义、指标和视角契约 |
| `phase_guidance.json` | 传统分段器的动作相位先验 |
| `repchain_segmenter.json` | RepChain 相位信号、峰值和端点策略 |
| `repchain_chain.json` | rep 链、缺口桥接和端点仲裁策略 |
| `repchain_confidence.json` | RepChain 置信度校准器 |
| `confidence_calibration.json` | 运行时校准器引用、质量/动作置信度倍率 |
| `quality_gates.json` | 质量门规则 |
| `action_verification.json` | 动作核验规则 |
| `assessment_rules.json` | 评估阈值与决策策略 |
| `issue_taxonomy.json` | 问题分类、严重度和动作建议 |
| `metric_contracts.json` | 指标可用性、方向和报告契约 |
| `feedback_templates_v1.json` | 问题卡与旧版报告模板 |
| `feedback_templates_v2.json` | 反馈 v2 文案与优先级模板 |
| `sample_clips.json` | 样本重分析清单；空清单也用于契约校验 |

`repchain_segmenter.json`、`repchain_chain.json` 和 `repchain_confidence.json` 的哈希被
校准链引用。修改这些文件后，必须按依赖顺序重算 `confidence_calibration.json`，再运行
`make check`。

输出契约统一使用 `vls-*` 版本的 `schema_version`，不携带开发阶段编号。`task`/`stage`
这类过程元数据统一改为 `component`；流水线数据库中的 `stage` 仅表示执行阶段。

## 本地示例样本

当前内置两组来自 `V0-web/webdata` 的 push-up 侧视对照样本：

| 样本 | 来源 | rep |
| --- | --- | --- |
| `demo_pushup_good_side` | `subject_003_push_up_good_side.mp4` | 18 |
| `demo_pushup_bad_side` | `subject_003_push_up_bad_side.mp4` | 14 |

原视频保存在 `webdata/jobs/<job_id>/input/source.mp4`，派生分析、回放和反馈分别位于
`results/clip_analysis_v1`、`results/reports_v1` 和 `results/feedback_v2`。`webdata/`、
`results/` 和视频文件默认不进入 Git；只提交代码时，克隆后需要重新导入这些示例数据。

## 开发到发布

1. 在 `V0/` 或 `V0-web/` 完成修改并通过对应回归测试。
2. 只提取产品运行所需差异，生成相对 `app/` 的 patch。
3. 在 `app/` 应用 patch，不在 `app/` 补历史测试或实验结果。
4. 运行 `make check`、`make frontend-build`，再上传一段短视频做链路 smoke。

补丁路径映射：

```text
V0-web/web/**                 -> app/web/**
V0-web/scripts/<runtime>.py   -> app/web/backend/app/repchain/<runtime>.py
V0-web/configs/<live>.json    -> app/configs/<live>.json
V0-web/schemas/<live>.json    -> app/schemas/<live>.json
V0-web/models/**              -> app/models/**
```

`<runtime>` 只包括后端的五个入口及其导入闭包，不要同步审计、标注、评测和历史实验脚本。

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make check
make frontend-build
make backend
```

打开 `http://127.0.0.1:8011/`。前端联调可另开终端运行 `make frontend-dev`。

`make check` 只验证产品运行模块可导入、CLI 入口和前端构建；完整算法回归仍应在
`V0-web/` 执行。当前阈值用于工程验证与回放定位，不作为医学或动作标准结论。
