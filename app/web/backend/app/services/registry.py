"""产物注册表：把任意一组同构产物目录组装成对外 DTO。

时间基准约定：所有下发时间均为 ``video_time_s``（相对于预裁剪 clip.webm 起点）。
源视频时间仅在 timebase 元数据中体现。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core import config
from ..core.errors import not_found
from ..schemas.samples import (
    AssessmentSummary,
    ConfidencePanel,
    ConfidenceRep,
    Contract,
    EvidenceInterval,
    FeedbackSummary,
    IssueItem,
    PhaseItem,
    QualityGate,
    RepItem,
    RepProvenance,
    RepSelection,
    SampleDetail,
    SampleSummary,
    Timebase,
    Timeline,
    VideoRef,
)
from ..storage import artifacts

LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class Source:
    """一组同构产物目录 + 对外 URL 前缀。"""

    sample_id: str
    analysis_dir: Path
    issues_dir: Path | None
    report_dir: Path | None
    media_url: str
    frame_url_base: str | None
    feedback_v2_dir: Path | None = None
    report_url: str | None = None
    feedback_report_url: str | None = None

    @property
    def media_path(self) -> Path | None:
        if self.report_dir is None:
            return None
        path = self.report_dir / "media" / f"{self.sample_id}_clip.webm"
        return path if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise not_found("artifact_not_found", f"missing artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise not_found("artifact_not_found", f"missing artifact: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_to_clip(source_time_s: Any, start_frame: int, fps: float) -> float | None:
    if source_time_s is None:
        return None
    return round(float(source_time_s) - start_frame / fps, 6)


def _load_feedback_v2(source: Source) -> dict[str, Any] | None:
    if source.feedback_v2_dir is None:
        return None
    path = source.feedback_v2_dir / "feedback_v2.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _feedback_frame_available(source: Source, name: str | None) -> bool:
    if not name:
        return False
    candidates = []
    if source.feedback_v2_dir is not None:
        candidates.append(source.feedback_v2_dir / "frames" / name)
    if source.report_dir is not None:
        candidates.append(source.report_dir / "annotated_frames" / name)
    return any(path.is_file() for path in candidates)


@lru_cache(maxsize=1)
def _exercise_specs() -> dict[str, Any]:
    return json.loads(config.EXERCISE_SPECS.read_text(encoding="utf-8"))


# ---------- 样本源 ----------

def list_sample_ids() -> list[str]:
    root = config.CLIP_ANALYSIS_DIR
    if not root.is_dir():
        return []
    return [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / "analysis.json").is_file()
    ]


def sample_source(sample_id: str) -> Source:
    analysis_dir = artifacts.sample_analysis_dir(sample_id)
    report_dir = artifacts.sample_report_dir(sample_id)
    feedback_dir = config.FEEDBACK_DIR / sample_id
    feedback_v2_dir = config.FEEDBACK_V2_DIR / sample_id
    return Source(
        sample_id=sample_id,
        analysis_dir=analysis_dir,
        issues_dir=feedback_dir if (feedback_dir / "issues.json").is_file() else None,
        report_dir=report_dir,
        media_url=f"/api/v1/samples/{sample_id}/media",
        frame_url_base=f"/api/v1/samples/{sample_id}/frames",
        feedback_v2_dir=(
            feedback_v2_dir if (feedback_v2_dir / "feedback_v2.json").is_file() else None
        ),
        report_url=(
            f"/reports/{sample_id}/report.html"
            if report_dir and (report_dir / "report.html").is_file()
            else None
        ),
        feedback_report_url=(
            f"/api/v1/samples/{sample_id}/feedback-report"
            if (feedback_v2_dir / "report_v2.html").is_file()
            else None
        ),
    )


def job_source(job_id: str) -> Source:
    analysis_dir = artifacts.job_artifacts_dir(job_id)
    return Source(
        sample_id=job_id,
        analysis_dir=analysis_dir,
        issues_dir=analysis_dir,
        report_dir=analysis_dir,
        media_url=f"/api/v1/jobs/{job_id}/media",
        frame_url_base=f"/api/v1/jobs/{job_id}/frames",
        feedback_v2_dir=(
            analysis_dir if (analysis_dir / "feedback_v2.json").is_file() else None
        ),
        report_url=(
            f"/api/v1/jobs/{job_id}/report"
            if (analysis_dir / "report.html").is_file()
            else None
        ),
        feedback_report_url=(
            f"/api/v1/jobs/{job_id}/feedback-report"
            if (analysis_dir / "report_v2.html").is_file()
            else None
        ),
    )


# ---------- 组装 ----------

def _timebase(analysis: dict[str, Any]) -> Timebase:
    fps = _float(analysis.get("fps"), 30.0) or 30.0
    trim = analysis.get("trim") or {}
    return Timebase(
        fps=fps,
        duration_seconds=round(_int(analysis.get("frame_count")) / fps, 3),
        clip_local_origin_s=0.0,
        source_trim_start_frame=_int(trim.get("start_frame")),
        video_time_origin="clip_start",
        variable_fps=False,
    )


def summary_of(source: Source) -> SampleSummary:
    analysis = _read_json(source.analysis_dir / "analysis.json")
    exercise_meta = _read_json(source.analysis_dir / "exercise.json")
    feedback_v2 = _load_feedback_v2(source)
    if feedback_v2 is not None:
        cards = feedback_v2.get("cards") or []
        issue_summary = {
            "active_issue_count": len(cards),
            "suppressed_issue_count": _int(
                (feedback_v2.get("assessment_summary") or {}).get("suppressed_issue_count")
            ),
            "severity_counts": dict(
                sorted(
                    {
                        severity: sum(card.get("severity") == severity for card in cards)
                        for severity in ("high", "medium", "low")
                    }.items()
                )
            ),
        }
        quality_status = str(
            (feedback_v2.get("assessment_summary") or {}).get("quality_status")
            or analysis.get("quality_status")
            or "unknown"
        )
    else:
        issues_payload: dict[str, Any] = {}
        if source.issues_dir is not None:
            issues_payload = json.loads((source.issues_dir / "issues.json").read_text(encoding="utf-8"))
        issue_summary = issues_payload.get("summary") or {}
        quality_status = str(analysis.get("quality_status") or "unknown")
    fps = _float(analysis.get("fps"), 30.0) or 30.0
    return SampleSummary(
        sample_id=source.sample_id,
        exercise=str(analysis.get("exercise") or exercise_meta.get("exercise_hint") or "unknown"),
        action_display=exercise_meta.get("action_display"),
        view=str(analysis.get("view") or "unknown"),
        split=str(analysis.get("split") or "unknown"),
        status=str(analysis.get("status") or "unknown"),
        quality_status=quality_status,
        rep_count=_int(analysis.get("rep_count")),
        rejected_rep_count=_int(analysis.get("rejected_rep_count")),
        active_issue_count=_int(issue_summary.get("active_issue_count")),
        suppressed_issue_count=_int(issue_summary.get("suppressed_issue_count")),
        severity_counts={str(k): _int(v) for k, v in (issue_summary.get("severity_counts") or {}).items()},
        has_media=source.media_path is not None,
        has_report=bool(source.report_dir and (source.report_dir / "report.html").is_file()),
        duration_seconds=round(_int(analysis.get("frame_count")) / fps, 3),
    )


def _build_reps(source: Source) -> list[RepItem]:
    rows = _read_csv(source.analysis_dir / "reps.csv")
    reps = []
    for row in rows:
        provenance = None
        if row.get("provenance.selection"):
            provenance = RepProvenance(
                selection=row["provenance.selection"],
                top_boundary_source=row.get("provenance.top_boundary_source") or "unknown",
                bottom_boundary_source=row.get("provenance.bottom_boundary_source") or "unknown",
            )
        reps.append(
            RepItem(
                rep_id=_int(row.get("rep_id")),
                start_s=_float(row.get("clip_start_time_s")),
                end_s=_float(row.get("clip_end_time_s")),
                bottom_s=_float(row.get("clip_bottom_time_s")),
                duration_s=_float(row.get("duration_s")),
                confidence=_float(row.get("confidence")),
                review_priority=row.get("review_priority") or "unknown",
                status=row.get("status") or "unknown",
                quality_flags=[x for x in (row.get("quality_flags") or "").split(";") if x],
                reject_reasons=[x for x in (row.get("reject_reasons") or "").split(";") if x],
                provenance=provenance,
            )
        )
    return reps


def _build_rep_selection(analysis: dict[str, Any]) -> RepSelection | None:
    payload = analysis.get("rep_selection")
    if not payload:
        return None
    return RepSelection.model_validate(payload)


def _build_phases(source: Source) -> list[PhaseItem]:
    rows = _read_csv(source.analysis_dir / "phases.csv")
    return [
        PhaseItem(
            rep_id=_int(row.get("rep_id")),
            phase=row.get("phase") or "unknown",
            time_s=_float(row.get("clip_time_s")),
        )
        for row in rows
    ]


def _build_issues(source: Source, analysis: dict[str, Any]) -> list[IssueItem]:
    feedback_v2 = _load_feedback_v2(source)
    if feedback_v2 is not None:
        items: list[IssueItem] = []
        state = str(feedback_v2.get("state") or "no_reliable_issue")
        for card in feedback_v2.get("cards") or []:
            frame_name = card.get("frame_filename")
            frame_url = None
            if source.frame_url_base and _feedback_frame_available(source, frame_name):
                frame_url = f"{source.frame_url_base}/{frame_name}"
            interval = card.get("evidence_interval_s") or []
            items.append(
                IssueItem(
                    issue_id=str(card.get("issue_id") or "unknown"),
                    issue_code=str(card.get("issue_code") or "unknown"),
                    title=str(card.get("title") or ""),
                    detail=str(card.get("detail") or ""),
                    cue=str(card.get("cue") or ""),
                    severity=str(card.get("severity") or "none"),
                    confidence=_optional_float(card.get("confidence")),
                    rep_ids=[_int(item) for item in card.get("support_rep_ids") or []],
                    representative_rep_id=card.get("representative_rep_id"),
                    frequency=str(card.get("frequency") or ""),
                    frequency_ratio=_float(card.get("frequency_ratio")),
                    anchor_s=_optional_float(card.get("anchor_time_s")),
                    evidence_interval_s=(
                        [_float(interval[0]), _float(interval[1])]
                        if len(interval) >= 2
                        else None
                    ),
                    frame_url=frame_url,
                    primary_metric=card.get("primary_metric"),
                    primary_value=_optional_float(card.get("primary_value")),
                    unit=card.get("unit"),
                    thresholds=card.get("thresholds") or {},
                    highlight_joints=list(card.get("highlight_joints") or []),
                    suppressed=False,
                    show_by_default=True,
                    decision=card.get("decision"),
                    family=card.get("issue_family"),
                    support_rep_ids=[_int(item) for item in card.get("support_rep_ids") or []],
                    anchor_fallback=bool(card.get("anchor_fallback")),
                    feedback_state=state,
                    shadow_only=bool(card.get("shadow_only", True)),
                )
            )
        return items
    if source.issues_dir is None:
        return []
    payload = json.loads((source.issues_dir / "issues.json").read_text(encoding="utf-8"))
    fps = _float(analysis.get("fps"), 30.0) or 30.0
    trim = analysis.get("trim") or {}
    start_frame = _int(trim.get("start_frame"))

    items: list[IssueItem] = []
    for raw in payload.get("issues", []):
        anchor_frame = raw.get("anchor_frame")
        anchor_s = _source_to_clip(raw.get("anchor_time_s"), start_frame, fps)
        interval = raw.get("evidence_interval")
        interval_s = None
        if interval:
            interval_s = [
                round((int(interval[0]) - start_frame) / fps, 6),
                round((int(interval[1]) - start_frame) / fps, 6),
            ]
        frame_url = None
        if (
            not raw.get("suppressed")
            and anchor_frame is not None
            and source.report_dir is not None
            and source.frame_url_base
        ):
            name = f"{raw['issue_id']}_frame_{int(anchor_frame)}.png"
            if (source.report_dir / "annotated_frames" / name).is_file():
                frame_url = f"{source.frame_url_base}/{name}"
        measurements = raw.get("measurements") or {}
        overlay = raw.get("visual_overlay") or {}
        text = raw.get("text") or {}
        items.append(
            IssueItem(
                issue_id=raw.get("issue_id") or "unknown",
                issue_code=raw.get("issue_code") or "unknown",
                title=text.get("title") or "",
                detail=text.get("detail") or "",
                cue=text.get("cue") or "",
                severity=raw.get("severity") or "none",
                confidence=raw.get("confidence"),
                rep_ids=[_int(x) for x in (raw.get("rep_ids") or [])],
                representative_rep_id=raw.get("representative_rep_id"),
                frequency=raw.get("frequency") or "",
                frequency_ratio=_float(raw.get("frequency_ratio")),
                anchor_s=anchor_s,
                evidence_interval_s=interval_s,
                frame_url=frame_url,
                primary_metric=measurements.get("primary_metric"),
                primary_value=measurements.get("primary_value"),
                unit=measurements.get("unit"),
                thresholds=measurements.get("thresholds") or {},
                highlight_joints=list(overlay.get("highlight_joints") or []),
                suppressed=bool(raw.get("suppressed")),
                show_by_default=bool(raw.get("show_by_default")),
            )
        )
    return items


def _build_evidence(issues: list[IssueItem]) -> list[EvidenceInterval]:
    out = []
    for issue in issues:
        if issue.suppressed or issue.anchor_s is None:
            continue
        start, end = (issue.evidence_interval_s or [issue.anchor_s, issue.anchor_s])[:2]
        out.append(
            EvidenceInterval(
                issue_id=issue.issue_id,
                severity=issue.severity,
                start_s=start,
                end_s=end,
                anchor_s=issue.anchor_s,
            )
        )
    return out


def _build_quality(source: Source, analysis: dict[str, Any]) -> QualityGate:
    path = source.analysis_dir / "quality.json"
    merged = _read_json(path) if path.is_file() else (analysis.get("quality") or {})
    return QualityGate(
        status=str(merged.get("status") or "unknown"),
        failures=list(merged.get("failures") or []),
        warnings=list(merged.get("warnings") or []),
        checks=merged.get("checks") or {},
        thresholds=merged.get("thresholds") or {},
        view_support=merged.get("view_support") or {},
    )


def _build_contract(source: Source, analysis: dict[str, Any]) -> Contract:
    exercise_meta = _read_json(source.analysis_dir / "exercise.json")
    specs = _exercise_specs()
    exercise = str(analysis.get("exercise") or exercise_meta.get("exercise_hint") or "unknown")
    view = str(analysis.get("view") or "unknown")
    exercise_spec = (specs.get("exercises") or {}).get(exercise) or {}
    view_support = analysis.get("quality", {}).get("view_support") or {}
    return Contract(
        exercise=exercise,
        view=view,
        view_definition=(specs.get("view_definitions") or {}).get(view),
        action_display=exercise_meta.get("action_display") or exercise_spec.get("action_display"),
        rep_unit=exercise_spec.get("rep_unit"),
        min_complete_reps=exercise_spec.get("min_complete_reps"),
        usable_metrics=list(view_support.get("usable_metrics") or []),
        disabled_metrics=list(view_support.get("disabled_metrics") or []),
        metric_groups=specs.get("metric_groups") or {},
        phase=exercise_spec.get("phase") or {},
    )


def _build_confidence(reps: list[RepItem], quality: dict[str, Any]) -> ConfidencePanel:
    checks = quality.get("checks") or {}
    notes = list((quality.get("visibility_notes") or {}).values())
    if not reps:
        return ConfidencePanel(mean_confidence=0.0, min_confidence=0.0, checks=checks, notes=notes)
    values = [rep.confidence for rep in reps]
    return ConfidencePanel(
        mean_confidence=round(sum(values) / len(values), 6),
        min_confidence=round(min(values), 6),
        low_confidence_rep_ids=[rep.rep_id for rep in reps if rep.confidence < LOW_CONFIDENCE_THRESHOLD],
        reps=[
            ConfidenceRep(rep_id=rep.rep_id, confidence=rep.confidence, review_priority=rep.review_priority)
            for rep in reps
        ],
        checks=checks,
        notes=notes,
    )


def _build_assessment_summary(feedback_v2: dict[str, Any] | None) -> AssessmentSummary | None:
    if feedback_v2 is None:
        return None
    summary = feedback_v2.get("assessment_summary") or {}
    return AssessmentSummary(
        conclusion=str(summary.get("conclusion") or "unknown"),
        quality_status=str(summary.get("quality_status") or "unknown"),
        rep_count=_int(summary.get("rep_count")),
        evaluable_metric_count=_int(summary.get("evaluable_metric_count")),
        candidate_issue_count=_int(summary.get("candidate_issue_count")),
        reportable_issue_count=_int(summary.get("reportable_issue_count")),
        diagnostic_issue_count=_int(summary.get("diagnostic_issue_count")),
        suppressed_issue_count=_int(summary.get("suppressed_issue_count")),
        abstained=bool(summary.get("abstained")),
        abstain_reason=summary.get("abstain_reason"),
        shadow_only=bool(summary.get("shadow_only", True)),
        product_actions_enabled=bool(summary.get("product_actions_enabled", False)),
    )


def _build_feedback_summary(feedback_v2: dict[str, Any] | None) -> FeedbackSummary | None:
    if feedback_v2 is None:
        return None
    cards = feedback_v2.get("cards") or []
    summary = feedback_v2.get("assessment_summary") or {}
    policy = feedback_v2.get("policy") or {}
    return FeedbackSummary(
        schema_version=str(feedback_v2.get("schema_version") or "vls-feedback-v2"),
        state=feedback_v2.get("state") or "no_reliable_issue",
        message=str(feedback_v2.get("state_message") or ""),
        card_count=len(cards),
        candidate_card_count=sum(card.get("decision") == "candidate" for card in cards),
        reportable_card_count=sum(card.get("decision") == "reportable" for card in cards),
        diagnostic_issue_count=_int(summary.get("diagnostic_issue_count")),
        suppressed_issue_count=_int(summary.get("suppressed_issue_count")),
        shadow_only=bool(policy.get("shadow_only", True)),
        product_actions_enabled=bool(policy.get("product_actions_enabled", False)),
        expert_calibrated=bool(policy.get("expert_calibrated", False)),
    )


def _fallback_state(
    issues: list[IssueItem], reps: list[RepItem], quality: dict[str, Any]
) -> str:
    if str(quality.get("status") or "unknown") == "reject":
        return "quality_reject"
    if not reps:
        return "no_reps"
    if any(not issue.suppressed for issue in issues):
        return "active_issues"
    if reps and all(rep.confidence < LOW_CONFIDENCE_THRESHOLD for rep in reps):
        return "low_confidence"
    return "no_reliable_issue"


def detail_of(source: Source) -> SampleDetail:
    analysis = _read_json(source.analysis_dir / "analysis.json")
    quality_json: dict[str, Any] = {}
    quality_path = source.analysis_dir / "quality.json"
    if quality_path.is_file():
        quality_json = json.loads(quality_path.read_text(encoding="utf-8"))
    reps = _build_reps(source)
    issues = _build_issues(source, analysis)
    feedback_v2 = _load_feedback_v2(source)
    quality = _build_quality(source, analysis)
    feedback_summary = _build_feedback_summary(feedback_v2)
    state = (
        feedback_summary.state
        if feedback_summary is not None
        else _fallback_state(issues, reps, quality_json)
    )
    return SampleDetail(
        sample=summary_of(source),
        video=VideoRef(url=source.media_url, timebase=_timebase(analysis)),
        timeline=Timeline(reps=reps, phases=_build_phases(source), evidence_intervals=_build_evidence(issues)),
        issues=issues,
        quality=quality,
        contract=_build_contract(source, analysis),
        confidence=_build_confidence(reps, quality_json),
        rep_selection=_build_rep_selection(analysis),
        assessment=_build_assessment_summary(feedback_v2),
        feedback=feedback_summary,
        state=state,
        report_url=source.report_url,
        feedback_report_url=source.feedback_report_url,
    )


def detail(sample_id: str) -> SampleDetail:
    return detail_of(sample_source(sample_id))


def list_summaries() -> list[SampleSummary]:
    return [summary_of(sample_source(sample_id)) for sample_id in list_sample_ids()]
