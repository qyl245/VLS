"""任务状态 DTO。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatusValue = Literal["queued", "validating", "running", "succeeded", "failed", "cancelled"]
SourceKind = Literal["sample", "upload"]


class JobStage(BaseModel):
    name: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    error: str | None = None


class JobRecord(BaseModel):
    job_id: str
    source_kind: SourceKind
    sample_id: str | None = None
    status: JobStatusValue
    stage: str
    progress: float = 0.0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    pipeline_version: str = "v0"
    stages: list[JobStage] = Field(default_factory=list)
    artifact_manifest_url: str | None = None


class JobCreateRequest(BaseModel):
    source_kind: SourceKind
    sample_id: str | None = None
    exercise: str | None = None
    view: str | None = None
    rotation: str | None = None


class JobResult(BaseModel):
    job_id: str
    sample_id: str
    status: JobStatusValue
    exercise: str
    view: str
    rep_count: int
    rejected_rep_count: int
    quality_status: str
    active_issue_count: int
    summary: dict[str, Any] = Field(default_factory=dict)


class ArtifactEntry(BaseModel):
    name: str
    relative_path: str
    size_bytes: int
    sha256: str


class ArtifactManifest(BaseModel):
    job_id: str
    pipeline_version: str
    created_at: str
    artifacts: list[ArtifactEntry] = Field(default_factory=list)
