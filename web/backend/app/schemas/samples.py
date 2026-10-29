"""样本库与单样本分析结果的对外 DTO（不含任何服务器路径）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Timebase(BaseModel):
    fps: float
    duration_seconds: float
    clip_local_origin_s: float = 0.0
    source_trim_start_frame: int = 0
    video_time_origin: Literal["clip_start"] = "clip_start"
    variable_fps: bool = False


class VideoRef(BaseModel):
    url: str
    timebase: Timebase


class SampleSummary(BaseModel):
    sample_id: str
    exercise: str
    action_display: str | None = None
    view: str
    split: str
    status: str
    quality_status: str
    rep_count: int
    rejected_rep_count: int
    active_issue_count: int
    suppressed_issue_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    has_media: bool
    has_report: bool
    duration_seconds: float


class RepItem(BaseModel):
    rep_id: int
    start_s: float
    end_s: float
    bottom_s: float
    duration_s: float
    confidence: float
    review_priority: str
    status: str
    quality_flags: list[str] = Field(default_factory=list)
    reject_reasons: list[str] = Field(default_factory=list)


class PhaseItem(BaseModel):
    rep_id: int
    phase: str
    time_s: float


class EvidenceInterval(BaseModel):
    issue_id: str
    severity: str
    start_s: float
    end_s: float
    anchor_s: float


class Timeline(BaseModel):
    reps: list[RepItem]
    phases: list[PhaseItem]
    evidence_intervals: list[EvidenceInterval]


class IssueItem(BaseModel):
    issue_id: str
    issue_code: str
    title: str
    detail: str
    cue: str
    severity: str
    confidence: float | None = None
    rep_ids: list[int] = Field(default_factory=list)
    representative_rep_id: int | None = None
    frequency: str
    frequency_ratio: float
    anchor_s: float | None = None
    evidence_interval_s: list[float] | None = None
    frame_url: str | None = None
    primary_metric: str | None = None
    primary_value: float | None = None
    unit: str | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    highlight_joints: list[str] = Field(default_factory=list)
    suppressed: bool
    show_by_default: bool


class QualityGate(BaseModel):
    status: str
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    view_support: dict[str, Any] = Field(default_factory=dict)


class Contract(BaseModel):
    exercise: str
    view: str
    view_definition: str | None = None
    action_display: str | None = None
    rep_unit: str | None = None
    min_complete_reps: int | None = None
    usable_metrics: list[str] = Field(default_factory=list)
    disabled_metrics: list[str] = Field(default_factory=list)
    metric_groups: dict[str, list[str]] = Field(default_factory=dict)
    phase: dict[str, Any] = Field(default_factory=dict)


class ConfidenceRep(BaseModel):
    rep_id: int
    confidence: float
    review_priority: str


class ConfidencePanel(BaseModel):
    mean_confidence: float
    min_confidence: float
    low_confidence_rep_ids: list[int] = Field(default_factory=list)
    reps: list[ConfidenceRep] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SampleDetail(BaseModel):
    sample: SampleSummary
    video: VideoRef
    timeline: Timeline
    issues: list[IssueItem]
    quality: QualityGate
    contract: Contract
    confidence: ConfidencePanel
    report_url: str | None = None
