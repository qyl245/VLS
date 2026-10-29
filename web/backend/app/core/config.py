"""运行配置：目录白名单、解释器、并发与限额。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# web/backend/app/core/config.py -> 仓库根
ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = ROOT / "results"
CLIP_ANALYSIS_DIR = RESULTS_DIR / "clip_analysis_v1"
FEEDBACK_DIR = RESULTS_DIR / "feedback_v1"
REPORTS_DIR = RESULTS_DIR / "reports_v1"

WEBDATA_ROOT = Path(os.environ.get("VLS_WEBDATA_ROOT", ROOT / "webdata")).resolve()
UPLOADS_DIR = WEBDATA_ROOT / "uploads"
JOBS_DIR = WEBDATA_ROOT / "jobs"
DB_PATH = WEBDATA_ROOT / "app.db"

# 只读根：任何解析到此范围之外的路径一律拒绝
READONLY_ROOTS = (RESULTS_DIR.resolve(), ROOT.resolve() / "outputs")

# 分析阶段使用的解释器，必须显式指定，不依赖当前 PATH
PYTHON_EXE = os.environ.get("VLS_PYTHON_EXE") or sys.executable

MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
CLIPS_MANIFEST = ROOT / "configs" / "task1_4_clips.json"
EXERCISE_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
FEEDBACK_TEMPLATES = ROOT / "configs" / "task1_4_feedback_templates.json"

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_UPLOAD_DURATION_S = 120.0
MIN_UPLOAD_DURATION_S = 3.0

JOB_TIMEOUT_S = int(os.environ.get("VLS_JOB_TIMEOUT_S", "1800"))
WORKER_CONCURRENCY = 1

FRONTEND_DIST = ROOT / "web" / "frontend" / "dist"


def ensure_webdata() -> None:
    for path in (WEBDATA_ROOT, UPLOADS_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------- 源视频定位（与 scripts/paths.py 的候选顺序保持一致） ----------

DATASET_NAME = "QEVD-FIT-COACH"


def data_root() -> Path:
    candidates = []
    env_value = os.environ.get("VLS_DATA_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            ROOT / "rawdata" / DATASET_NAME,
            ROOT / "dataset" / "QEVD-FIT" / DATASET_NAME,
            ROOT.parent / "dataset" / "QEVD-FIT" / DATASET_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return (ROOT / "rawdata" / DATASET_NAME).resolve()


def resolve_source_path(value: str) -> Path:
    """把历史 manifest 里的路径（可能是 Linux 绝对路径）重映射到本地数据根。"""
    path = Path(value)
    if path.is_file():
        return path
    root = data_root()
    parts = path.parts
    for index, part in enumerate(parts):
        if part.upper() == DATASET_NAME:
            candidate = root.joinpath(*parts[index + 1 :])
            if candidate.is_file():
                return candidate
    candidate = root / path.name
    return candidate if candidate.is_file() else path

