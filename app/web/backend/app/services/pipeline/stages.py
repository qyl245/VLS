"""流水线阶段定义：上传路径显式分阶段，每阶段独立记录状态与日志。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...core import config
from ...core.errors import AppError

STAGE_NAMES = (
    "validate_upload",
    "probe_video",
    "build_job_manifest",
    "analysis_v1",
    "repchain",
    "issues",
    "feedback_v2",
    "build_report",
    "validate_artifacts",
    "publish_complete",
)

SIDE_VIEWS = {"side_left", "side_right", "oblique_left", "oblique_right"}
FRONTAL_VIEWS = {"front", "rear", "oblique_left", "oblique_right"}
BASE_METRICS = ("rep_duration_s", "rep_duration_cv", "spine_stability_bottom")
SQUAT_SIDE_METRICS = ("knee_flexion_rom", "bottom_knee_angle", "hip_flexion_rom", "torso_lean_bottom", "hip_depth_ratio")
SQUAT_FRONTAL_METRICS = ("knee_valgus_bottom", "knee_asymmetry_bottom", "hip_asymmetry_bottom")
PUSHUP_SIDE_METRICS = ("elbow_flexion_rom", "bottom_elbow_angle", "body_line_angle", "hip_sag_ratio")
PUSHUP_FRONTAL_METRICS = ("elbow_asymmetry_bottom", "shoulder_asymmetry_bottom")
VALID_ROTATIONS = {"none", "90cw", "180", "90ccw"}
VALID_VIEWS = {"front", "rear", "oblique_left", "oblique_right", "side_left", "side_right", "unknown"}


@dataclass
class StageResult:
    stage: str
    detail: dict[str, Any]


def metrics_for_view(exercise: str, view: str) -> tuple[list[str], list[str]]:
    if exercise == "squat":
        side, frontal = SQUAT_SIDE_METRICS, SQUAT_FRONTAL_METRICS
    else:
        side, frontal = PUSHUP_SIDE_METRICS, PUSHUP_FRONTAL_METRICS
    all_metrics = tuple(dict.fromkeys((*BASE_METRICS, *side, *frontal)))
    usable = set(BASE_METRICS)
    if view in SIDE_VIEWS:
        usable.update(side)
    if view in FRONTAL_VIEWS:
        usable.update(frontal)
    ordered = [metric for metric in all_metrics if metric in usable]
    return ordered, [metric for metric in all_metrics if metric not in usable]


def _run_script(job_dir: Path, args: list[str], log_name: str, timeout_s: int) -> None:
    """子进程调用现有脚本：参数数组、固定 cwd、日志落盘。"""
    log_path = job_dir / "logs" / f"{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [config.PYTHON_EXE, *args]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(config.ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise AppError("job_timeout", f"{log_name} 超时（>{timeout_s}s）", status_code=500)
    if proc.returncode != 0:
        raise AppError("stage_failed", f"{log_name} 失败，退出码 {proc.returncode}，详见 logs/{log_name}.log", status_code=500)


def _rewrite_analysis_outputs(clip_dir: Path) -> None:
    """Point copied fallback analysis at its published working directory."""
    path = clip_dir / "analysis.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outputs"] = {
        name: str((clip_dir / relative).resolve())
        for name, relative in {
            "clip": "clip.json",
            "quality": "quality.json",
            "exercise": "exercise.json",
            "reps": "reps.csv",
            "rejected_reps": "rejected_reps.csv",
            "phases": "phases.csv",
            "angles": "angles/angles.csv",
            "angles_wide": "angles/angles_wide.csv",
            "landmarks": "angles/landmarks.json",
        }.items()
    }
    angle_extraction = payload.setdefault("angle_extraction", {})
    angle_extraction["outputs"] = {
        name: str((clip_dir / relative).resolve())
        for name, relative in {
            "landmarks": "angles/landmarks.json",
            "landmarks_body_frame": "angles/landmarks_body_frame.json",
            "angles": "angles/angles.csv",
            "angles_wide": "angles/angles_wide.csv",
        }.items()
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def probe_video(source: Path, enforce_limits: bool = True) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise AppError("invalid_video", f"无法解码视频：{source.name}", status_code=400)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise AppError("invalid_video", "视频缺少有效的 fps/帧数信息", status_code=400)
    duration_s = frame_count / fps
    if enforce_limits and duration_s < config.MIN_UPLOAD_DURATION_S:
        raise AppError("video_too_short", f"视频过短（{duration_s:.1f}s）", status_code=400)
    if enforce_limits and duration_s > config.MAX_UPLOAD_DURATION_S:
        raise AppError("video_too_long", f"视频过长（{duration_s:.1f}s，上限 {config.MAX_UPLOAD_DURATION_S:.0f}s）", status_code=400)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_s, 3),
    }


def build_manifest(
    job_id: str,
    exercise: str,
    view: str,
    rotation: str,
    probe: dict[str, Any],
    source: Path,
    trim: dict[str, int],
    clip_entry: dict[str, Any] | None = None,
) -> Path:
    if clip_entry:
        # 冻结样本重分析：以原 manifest 条目为基底，保证与既有产物同构可复现
        entry = json.loads(json.dumps(clip_entry))
        entry.update(
            {
                "clip_id": job_id,
                "video_id": job_id,
                "video_path": str(source.resolve()),
                "trim": {"start_frame": trim["start_frame"], "end_frame": trim["end_frame"]},
                "processed_trim": {"start_frame": trim["start_frame"], "end_frame": trim["end_frame"]},
            }
        )
    else:
        usable, disabled = metrics_for_view(exercise, view)
        entry = {
            "clip_id": job_id,
            "video_id": job_id,
            "video_path": str(source.resolve()),
            "exercise": exercise,
            "action_display": "pushups" if exercise == "pushup" else "squats",
            "trim": {"start_frame": trim["start_frame"], "end_frame": trim["end_frame"]},
            "fps": probe["fps"],
            "frame_count": probe["frame_count"],
            "source_size": {"width": probe["width"], "height": probe["height"]},
            "rotation": rotation,
            "view": view,
            "auto_view": view,
            "view_source": "user_selected",
            "split": "upload",
            "phase_role": "upload",
            "selected_for_phase": view in SIDE_VIEWS,
            "usable_metrics": usable,
            "disabled_metrics": disabled,
            "quality": {
                "rotation_confidence": 0.0,
                "rotation_margin": 0.0,
                "pose_detection_rate": 0.0,
                "mean_visible_fraction": 0.0,
                "view_confidence": 0.0,
                "width_ratio": 0.0,
                "shoulder_tilt": 0.0,
                "nose_offset": 0.0,
                "quality_flags": ["upload_rotation_not_evaluated"],
            },
            "notes": "web upload",
            "processed_trim": {"start_frame": trim["start_frame"], "end_frame": trim["end_frame"]},
        }
    path = config.JOBS_DIR / job_id / "work" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "vls-sample-clips-v1", "clips": [entry]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_stages(job_id: str, request: dict[str, Any], on_stage: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """顺序执行全部阶段；任何阶段抛错即中断。"""
    job_dir = config.JOBS_DIR / job_id
    if request.get("source_path"):
        source = config.resolve_source_path(str(request["source_path"]))
    else:
        suffix = str(request.get("suffix") or ".mp4")
        source = job_dir / "input" / f"source{suffix}"
    if not source.is_file():
        raise AppError("missing_input", "源视频不可用", status_code=500)
    trim = request.get("trim") or {}
    trim = {
        "start_frame": int(trim.get("start_frame") or 0),
        "end_frame": int(trim.get("end_frame") or 0),
    }

    on_stage("validate_upload", {"source_bytes": source.stat().st_size})

    probe = probe_video(source, enforce_limits=not request.get("source_path"))
    exercise = str(request.get("exercise") or "pushup")
    view = str(request.get("view") or "unknown")
    rotation = str(request.get("rotation") or "none")
    if exercise not in {"pushup", "squat"}:
        raise AppError("invalid_exercise", f"不支持的动作类型：{exercise}", status_code=400)
    if view not in VALID_VIEWS:
        raise AppError("invalid_view", f"不支持的视角：{view}", status_code=400)
    if rotation not in VALID_ROTATIONS:
        raise AppError("invalid_rotation", f"不支持的旋转：{rotation}", status_code=400)
    on_stage("probe_video", probe)

    if trim["end_frame"] <= 0 or trim["end_frame"] > probe["frame_count"] - 1:
        trim["end_frame"] = probe["frame_count"] - 1
    if trim["start_frame"] >= trim["end_frame"]:
        raise AppError("invalid_trim", "裁剪区间无效", status_code=400)
    manifest = build_manifest(job_id, exercise, view, rotation, probe, source, trim, request.get("clip_entry"))
    on_stage("build_job_manifest", {"manifest": str(manifest.name)})

    legacy_dir = job_dir / "work" / "analysis_v1"
    _run_script(
        job_dir,
        [
            str(config.REPCHAIN_DIR / "analysis_v1.py"),
            "--manifest", str(manifest),
            "--clip-id", job_id,
            "--output-dir", str(legacy_dir),
            "--overwrite",
        ],
        "analysis_v1",
        config.JOB_TIMEOUT_S,
    )
    on_stage("analysis_v1", {"output_dir": "work/analysis_v1"})

    work_dir = job_dir / "work" / "analysis"
    repchain_dir = job_dir / "work" / "repchain"
    repchain_status = "ok"
    repchain_summary: dict[str, Any] = {}
    try:
        _run_script(
            job_dir,
            [
                str(config.REPCHAIN_DIR / "repchain_pipeline.py"),
                "--manifest", str(manifest),
                "--clip-id", job_id,
                "--analysis-dir", str(legacy_dir),
                "--output-dir", str(work_dir),
                "--repchain-dir", str(repchain_dir),
            ],
            "repchain",
            config.JOB_TIMEOUT_S,
        )
        summary_path = repchain_dir / "summary.json"
        if summary_path.is_file():
            repchain_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except AppError as exc:
        repchain_status = "fallback_legacy"
        repchain_summary = {
            "status": repchain_status,
            "error_code": exc.code,
            "error_message": exc.message,
        }
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        shutil.copytree(legacy_dir, work_dir)
        _rewrite_analysis_outputs(work_dir / job_id)
        repchain_dir.mkdir(parents=True, exist_ok=True)
        (repchain_dir / "fallback.json").write_text(
            json.dumps(
                {
                    "status": repchain_status,
                    "error_code": exc.code,
                    "error_message": exc.message,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    on_stage(
        "repchain",
        {
            "status": repchain_status,
            "output_dir": "work/analysis",
            "raw_dir": "work/repchain",
        },
    )

    feedback_dir = job_dir / "work" / "feedback"
    _run_script(
        job_dir,
        [
            str(config.REPCHAIN_DIR / "build_issue_cards.py"),
            "--analysis-dir", str(work_dir),
            "--output-dir", str(feedback_dir),
            "--clip-id", job_id,
            "--split", "all",
        ],
        "issues",
        config.JOB_TIMEOUT_S,
    )
    on_stage("issues", {"output_dir": "work/feedback"})

    feedback_v2_dir = job_dir / "work" / "feedback_v2"
    _run_script(
        job_dir,
        [
            str(config.REPCHAIN_DIR / "feedback_builder.py"),
            "--analysis-dir", str(work_dir),
            "--output-dir", str(feedback_v2_dir),
            "--clip-id", job_id,
            "--split", "all",
            "--overwrite",
            "--render-frames",
            "--media-url-template", f"/api/v1/jobs/{job_id}/media",
            "--frame-url-template", f"/api/v1/jobs/{job_id}/frames/{{frame_name}}",
        ],
        "feedback_v2",
        config.JOB_TIMEOUT_S,
    )
    on_stage("feedback_v2", {"output_dir": "work/feedback_v2"})

    artifacts_root = job_dir / "artifacts"
    _run_script(
        job_dir,
        [
            str(config.REPCHAIN_DIR / "render_clip_report.py"),
            "--analysis-dir", str(work_dir),
            "--feedback-dir", str(feedback_dir),
            "--output-dir", str(artifacts_root),
            "--clip-id", job_id,
            "--split", "all",
        ],
        "build_report",
        config.JOB_TIMEOUT_S,
    )
    on_stage("build_report", {"output_dir": "artifacts"})

    # 与 reports_v1 同构：artifacts/<job_id>/ 下汇总分析、反馈与报告资源
    artifacts_dir = artifacts_root / job_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("analysis.json", "quality.json", "clip.json", "exercise.json", "reps.csv", "phases.csv", "rejected_reps.csv"):
        src = work_dir / job_id / name
        if src.is_file():
            shutil.copy2(src, artifacts_dir / name)
    issues_src = feedback_dir / job_id / "issues.json"
    if issues_src.is_file():
        shutil.copy2(issues_src, artifacts_dir / "issues.json")
    feedback_v2_clip = feedback_v2_dir / job_id
    for name in ("feedback_v2.json", "assessment_v2.json", "evidence.json", "report_v2.html"):
        source = feedback_v2_clip / name
        if source.is_file():
            shutil.copy2(source, artifacts_dir / name)
    frames_src = feedback_v2_clip / "frames"
    if frames_src.is_dir():
        shutil.copytree(frames_src, artifacts_dir / "annotated_frames", dirs_exist_ok=True)
    if repchain_dir.is_dir():
        shutil.copytree(repchain_dir, artifacts_dir / "repchain", dirs_exist_ok=True)

    missing = [
        name
        for name in (
            "analysis.json",
            "quality.json",
            "clip.json",
            "exercise.json",
            "reps.csv",
            "phases.csv",
            "issues.json",
            "feedback_v2.json",
            "assessment_v2.json",
            "evidence.json",
            "report_v2.html",
        )
        if not (artifacts_dir / name).is_file()
    ]
    if not (artifacts_dir / "media" / f"{job_id}_clip.webm").is_file():
        missing.append(f"media/{job_id}_clip.webm")
    if missing:
        raise AppError("artifact_incomplete", "产物缺失：" + ", ".join(missing), status_code=500)
    on_stage("validate_artifacts", {"checked": True})

    manifest_payload = {
        "job_id": job_id,
        "pipeline_version": "v0-repchain-v3",
        "artifacts": [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(artifacts_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifacts_dir.rglob("*"))
            if path.is_file() and path.name not in {"complete.json"}
        ],
    }
    complete_path = job_dir / "complete.json"
    complete_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    on_stage("publish_complete", {"complete": str(complete_path.name)})

    analysis = json.loads((artifacts_dir / "analysis.json").read_text(encoding="utf-8"))
    return {
        "exercise": analysis.get("exercise") or exercise,
        "view": analysis.get("view") or view,
        "rep_count": int(analysis.get("rep_count") or 0),
        "rejected_rep_count": int(analysis.get("rejected_rep_count") or 0),
        "quality_status": str(analysis.get("quality_status") or "unknown"),
        "summary": {
            "probe": probe,
            "manifest": manifest_payload,
            "repchain": repchain_summary,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
