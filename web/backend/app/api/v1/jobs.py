"""任务 API：上传/样本分析、状态查询、结果与媒体。"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from ...core import config
from ...core.errors import AppError, bad_request, not_found
from ...schemas.jobs import JobCreateRequest, JobRecord
from ...schemas.samples import SampleDetail
from ...services import registry
from ...storage import artifacts, db
from ...workers.manager import manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _manifest_clip(sample_id: str) -> dict:
    payload = json.loads(config.CLIPS_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task1-4-v1":
        raise AppError("manifest_invalid", "clip manifest schema 不正确", status_code=500)
    for clip in payload.get("clips", []):
        if clip.get("clip_id") == sample_id:
            return clip
    raise bad_request("unknown_sample", f"manifest 中没有该样本：{sample_id}")


def _new_record(job_id: str, source_kind: str, sample_id: str | None) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        source_kind=source_kind,  # type: ignore[arg-type]
        sample_id=sample_id,
        status="queued",
        stage="queued",
        progress=0.0,
        created_at=db.utcnow(),
    )


def _persist_request(job_id: str, request: dict) -> None:
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")


@router.post("", status_code=202, response_model=JobRecord)
def create_job(payload: JobCreateRequest) -> JobRecord:
    if payload.source_kind != "sample" or not payload.sample_id:
        raise bad_request("invalid_source", "JSON 建任务仅支持 source_kind=sample；上传请用 /jobs/upload")
    clip = _manifest_clip(payload.sample_id)
    existing = db.find_active_for_sample(payload.sample_id)
    if existing is not None:
        return existing
    source = config.resolve_source_path(str(clip.get("video_path") or ""))
    if not source.is_file():
        raise bad_request("source_unavailable", f"源视频不可用：{payload.sample_id}")
    request = {
        "source_kind": "sample",
        "sample_id": payload.sample_id,
        "source_path": str(source),
        "suffix": source.suffix.lower(),
        "trim": clip.get("trim") or {},
        "exercise": payload.exercise or clip.get("exercise"),
        "view": payload.view or clip.get("view"),
        "rotation": payload.rotation or clip.get("rotation"),
        "clip_entry": clip,
    }
    job_id = uuid.uuid4().hex[:12]
    record = _new_record(job_id, "sample", payload.sample_id)
    _persist_request(job_id, request)
    db.insert_job(record, request)
    manager.enqueue(job_id)
    return record


@router.post("/upload", status_code=202, response_model=JobRecord)
async def create_upload_job(
    file: UploadFile = File(...),
    exercise: str = Form(...),
    view: str = Form("unknown"),
    rotation: str = Form("none"),
) -> JobRecord:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_VIDEO_SUFFIXES:
        raise bad_request("unsupported_file_type", f"不支持的视频格式：{suffix or '(无扩展名)'}")
    if exercise not in {"pushup", "squat"}:
        raise bad_request("invalid_exercise", f"不支持的动作类型：{exercise}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = config.JOBS_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    staging = config.UPLOADS_DIR / f"{job_id}{suffix}"
    total = 0
    try:
        with staging.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.MAX_UPLOAD_BYTES:
                    raise AppError(
                        "file_too_large",
                        f"文件超过上限（{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB）",
                        status_code=413,
                    )
                out.write(chunk)
        target = input_dir / f"source{suffix}"
        shutil.move(str(staging), str(target))
    finally:
        if staging.exists():
            staging.unlink()

    request = {
        "source_kind": "upload",
        "sample_id": None,
        "suffix": suffix,
        "exercise": exercise,
        "view": view,
        "rotation": rotation,
        "filename": file.filename,
        "bytes": total,
    }
    record = _new_record(job_id, "upload", None)
    _persist_request(job_id, request)
    db.insert_job(record, request)
    manager.enqueue(job_id)
    return record


@router.get("", response_model=list[JobRecord])
def list_jobs(limit: int = 50) -> list[JobRecord]:
    return db.list_jobs(limit=limit)


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    record = db.get_job(job_id)
    if record is None:
        raise not_found("job_not_found", f"job not found: {job_id}")
    return record


@router.post("/{job_id}/cancel", response_model=JobRecord)
def cancel_job(job_id: str) -> JobRecord:
    record = db.get_job(job_id)
    if record is None:
        raise not_found("job_not_found", f"job not found: {job_id}")
    if not manager.cancel(job_id):
        raise bad_request("job_cancel_rejected", f"当前状态无法取消：{record.status}")
    updated = db.get_job(job_id)
    return updated if updated else record


@router.get("/{job_id}/result", response_model=SampleDetail)
def get_result(job_id: str) -> SampleDetail:
    record = db.get_job(job_id)
    if record is None:
        raise not_found("job_not_found", f"job not found: {job_id}")
    if record.status != "succeeded":
        raise AppError(
            "job_not_ready",
            f"任务尚未完成：{record.status}",
            status_code=409,
            job_status=record.status,
            stage=record.stage,
        )
    return registry.detail_of(registry.job_source(job_id))


@router.get("/{job_id}/media")
def get_job_media(job_id: str) -> FileResponse:
    path = artifacts.job_media_path(job_id)
    return FileResponse(path, media_type="video/webm", filename=path.name)


@router.get("/{job_id}/frames/{frame_name}")
def get_job_frame(job_id: str, frame_name: str) -> FileResponse:
    path = artifacts.job_frame_path(job_id, frame_name)
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.get("/{job_id}/report")
def get_job_report(job_id: str) -> FileResponse:
    path = artifacts.job_artifacts_dir(job_id) / "report.html"
    if not path.is_file():
        raise not_found("report_not_found", f"report not ready for job {job_id}")
    return FileResponse(path, media_type="text/html", filename=path.name)
