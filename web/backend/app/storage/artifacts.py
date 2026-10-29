"""ArtifactStore：把 id 映射到白名单内的真实文件，阻断任意路径访问。"""

from __future__ import annotations

import re
from pathlib import Path

from ..core import config
from ..core.errors import not_found

# 仅允许字母数字下划线，杜绝 / \ .. 等路径成分
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_FRAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}\.png$")

REQUIRED_SAMPLE_FILES = ("analysis.json", "clip.json", "exercise.json", "quality.json", "reps.csv", "phases.csv")


def check_id(value: str, kind: str = "artifact") -> str:
    if not _ID_PATTERN.fullmatch(value or ""):
        raise not_found("artifact_not_found", f"{kind} not found: {value}")
    return value


def check_frame_name(value: str) -> str:
    if not _FRAME_PATTERN.fullmatch(value or ""):
        raise not_found("artifact_not_found", f"frame not found: {value}")
    return value


def resolve_within(root: Path, *parts: str) -> Path:
    """解析路径并要求结果落在 root 内，否则按不存在处理。"""
    candidate = (root.joinpath(*parts)).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise not_found("artifact_not_found", "path outside allowed root") from None
    return candidate


# ---------- 冻结样本 ----------

def sample_analysis_dir(sample_id: str) -> Path:
    path = resolve_within(config.CLIP_ANALYSIS_DIR, check_id(sample_id, "sample"))
    if not path.is_dir():
        raise not_found("sample_not_found", f"sample not found: {sample_id}")
    return path


def sample_feedback_dir(sample_id: str) -> Path:
    path = resolve_within(config.FEEDBACK_DIR, check_id(sample_id, "sample"))
    if not path.is_dir():
        raise not_found("sample_not_found", f"sample not found: {sample_id}")
    return path


def sample_report_dir(sample_id: str) -> Path | None:
    path = resolve_within(config.REPORTS_DIR, check_id(sample_id, "sample"))
    return path if path.is_dir() else None


def sample_media_path(sample_id: str) -> Path:
    report_dir = sample_report_dir(sample_id)
    if report_dir is None:
        raise not_found("artifact_not_found", f"media not found for {sample_id}")
    path = resolve_within(report_dir, "media", f"{sample_id}_clip.webm")
    if not path.is_file():
        raise not_found("artifact_not_found", f"media not found for {sample_id}")
    return path


def sample_frame_path(sample_id: str, frame_name: str) -> Path:
    report_dir = sample_report_dir(sample_id)
    if report_dir is None:
        raise not_found("artifact_not_found", f"frames not found for {sample_id}")
    path = resolve_within(report_dir, "annotated_frames", check_frame_name(frame_name))
    if not path.is_file():
        raise not_found("artifact_not_found", f"frame not found: {frame_name}")
    return path


def sample_ready(sample_id: str) -> bool:
    analysis_dir = config.CLIP_ANALYSIS_DIR / sample_id
    if not analysis_dir.is_dir():
        return False
    if not all((analysis_dir / name).is_file() for name in REQUIRED_SAMPLE_FILES):
        return False
    report_dir = config.REPORTS_DIR / sample_id
    return (report_dir / "media" / f"{sample_id}_clip.webm").is_file()


# ---------- 任务 ----------

def job_dir(job_id: str) -> Path:
    path = resolve_within(config.JOBS_DIR, check_id(job_id, "job"))
    if not path.is_dir():
        raise not_found("job_not_found", f"job not found: {job_id}")
    return path


def job_artifacts_dir(job_id: str) -> Path:
    """任务产物目录，与 reports_v1/<clip_id>/ 同构。"""
    path = resolve_within(job_dir(job_id), "artifacts", check_id(job_id, "job"))
    if not path.is_dir():
        raise not_found("artifact_not_found", f"artifacts not ready for job {job_id}")
    return path


def job_media_path(job_id: str) -> Path:
    path = resolve_within(job_artifacts_dir(job_id), "media", f"{job_id}_clip.webm")
    if not path.is_file():
        raise not_found("artifact_not_found", f"media not ready for job {job_id}")
    return path


def job_frame_dir(job_id: str) -> Path:
    return resolve_within(job_artifacts_dir(job_id), "annotated_frames")


def job_frame_path(job_id: str, frame_name: str) -> Path:
    path = resolve_within(job_frame_dir(job_id), check_frame_name(frame_name))
    if not path.is_file():
        raise not_found("artifact_not_found", f"frame not found: {frame_name}")
    return path


def job_artifact_path(job_id: str, relative: str) -> Path:
    """读取任务产物目录内的单个文件（如 analysis.v1.json）。"""
    if not _ID_PATTERN.fullmatch(relative.replace(".", "_").replace("-", "_")):
        raise not_found("artifact_not_found", f"artifact not found: {relative}")
    path = resolve_within(job_dir(job_id), "artifacts", relative)
    if not path.is_file():
        raise not_found("artifact_not_found", f"artifact not ready: {relative}")
    return path
