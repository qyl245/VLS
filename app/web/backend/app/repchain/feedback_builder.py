#!/usr/bin/env python3
"""Build evidence-backed feedback from assessment results.

The script deliberately keeps assessment, feedback ordering and rendering as
separate contracts. It can consume an existing assessment directory, or adapt
an arbitrary clip-analysis/RepChain directory at runtime without claiming a
benchmark split.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

import assessment_measurement as measurement_stage
import assessment_decision as decision_stage


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "clip_analysis_v1"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "feedback_v2"
DEFAULT_TEMPLATES = ROOT / "configs" / "feedback_templates_v2.json"
DEFAULT_RULES = ROOT / "configs" / "assessment_rules.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "issue_taxonomy.json"
DEFAULT_CONTRACTS = ROOT / "configs" / "metric_contracts.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "feedback_v2.schema.json"

ACTIVE_DECISIONS = {"candidate", "reportable"}
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SEVERITY_LABEL = {"low": "低", "medium": "中", "high": "高"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--assessment-dir",
        type=Path,
        default=None,
        help="Optional existing assessment root; rebuild when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument(
        "--split",
        default="all",
        help="Filter analysis split; use 'all' for the Web runtime adapter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--render-frames",
        action="store_true",
        help="Render annotated evidence frames into <output>/<clip>/frames/.",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_full.task")
    parser.add_argument(
        "--media-url-template",
        default="/api/v1/samples/{clip_id}/media",
        help="Standalone report video URL; supports {clip_id}.",
    )
    parser.add_argument(
        "--frame-url-template",
        default="/api/v1/samples/{clip_id}/frames/{frame_name}",
        help="Standalone report evidence-frame URL; supports {clip_id}, {frame_name}.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def severity_rank(value: str | None) -> int:
    return SEVERITY_RANK.get(str(value or "none"), 0)


def select_clip_ids(args: argparse.Namespace, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(args.clip_id or [])
    selected = []
    for analysis in analyses:
        clip_id = str(analysis.get("clip_id") or "")
        if not clip_id or (wanted and clip_id not in wanted):
            continue
        split = str(analysis.get("split") or "unknown")
        if args.split != "all" and split != args.split:
            continue
        selected.append(analysis)
    selected.sort(key=lambda item: str(item["clip_id"]))
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    if wanted:
        missing = wanted - {str(item["clip_id"]) for item in selected}
        if missing:
            raise FileNotFoundError("analysis missing for: " + ", ".join(sorted(missing)))
    return selected


def discover_analyses(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.glob("*/analysis.json")):
        payload = load_json(path)
        payload["_analysis_dir"] = str(path.parent)
        output.append(payload)
    return output


def subject_id_for(clip_id: str, clip_meta: dict[str, Any], analysis: dict[str, Any]) -> str:
    explicit = clip_meta.get("subject_id") or analysis.get("subject_id")
    if explicit:
        return str(explicit)
    match = re.search(r"subject[_-](\d+)", clip_id, flags=re.IGNORECASE)
    if match:
        return f"subject_{match.group(1)}"
    return f"runtime_{clip_id}"


def runtime_clip(
    analysis: dict[str, Any], clip_meta: dict[str, Any]
) -> dict[str, Any]:
    clip_id = str(analysis["clip_id"])
    source_path = clip_meta.get("video_path") or clip_meta.get("source_path")
    source = Path(str(source_path)).expanduser() if source_path else None
    return {
        "clip_id": clip_id,
        "subject_id": subject_id_for(clip_id, clip_meta, analysis),
        "exercise": str(analysis["exercise"]),
        "view": str(analysis.get("view") or "unknown"),
        "split": str(analysis.get("split") or "dev"),
        "fps": as_float(analysis.get("fps")) or as_float(clip_meta.get("fps")) or 30.0,
        "source_path": str(source) if source else None,
        "sha256": sha256_file(source) if source else None,
        "runtime_adapter": True,
    }


def write_runtime_benchmark(
    output_dir: Path, clips: list[dict[str, Any]], generated_at: str
) -> Path:
    path = output_dir / "_runtime_benchmark.json"
    write_json(
        path,
        {
            "schema_version": "vls-runtime-benchmark-v1",
            "component": "runtime_benchmark_adapter",
            "generated_at": generated_at,
            "status": "runtime_adapter_not_a_benchmark_split",
            "clips": clips,
        },
    )
    return path


def build_runtime_assessment(
    clip: dict[str, Any],
    analysis_dir: Path,
    contracts: dict[str, Any],
    rules: dict[str, Any],
    taxonomy: dict[str, Any],
    benchmark_path: Path,
    contracts_path: Path,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = measurement_stage.load_clip_data(clip, analysis_dir)
    measurement = measurement_stage.build_assessment(
        clip,
        data,
        contracts,
        benchmark_path,
        contracts_path,
        generated_at,
    )
    assessment = decision_stage.build_assessment(
        measurement,
        clip,
        rules,
        taxonomy,
        generated_at,
    )
    return measurement, assessment


def load_existing_assessment(assessment_dir: Path, clip_id: str) -> dict[str, Any] | None:
    path = assessment_dir / clip_id / "assessment.json"
    return load_json(path) if path.is_file() else None


def support_evaluations(issue: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in issue.get("support_metric_evaluations") or []
        if item.get("decision") in ACTIVE_DECISIONS
    ]


def choose_primary_evaluation(issue: dict[str, Any], templates: dict[str, Any]) -> dict[str, Any] | None:
    evaluations = support_evaluations(issue)
    if not evaluations:
        return None
    representative = issue.get("representative_measurement_id")
    if templates.get("evidence", {}).get("prefer_representative_measurement", True):
        for item in evaluations:
            if item.get("measurement_id") == representative:
                return item
    return max(
        evaluations,
        key=lambda item: (
            severity_rank(item.get("magnitude_severity")),
            as_float(item.get("deviation")) or 0.0,
            as_float(item.get("adjusted_confidence")) or 0.0,
        ),
    )


def choose_rep(issue: dict[str, Any], reps: list[dict[str, Any]], anchor: int | None) -> dict[str, Any] | None:
    by_id = {as_int(rep.get("rep_id")): rep for rep in reps if as_int(rep.get("rep_id")) is not None}
    if anchor is not None:
        for rep in reps:
            if int(rep["start_frame"]) <= anchor <= int(rep["end_frame"]):
                return rep
    representative = as_int(issue.get("representative_rep_id") or issue.get("rep_id"))
    if representative in by_id:
        return by_id[representative]
    for rep_id in issue.get("support_rep_ids") or []:
        rep = by_id.get(as_int(rep_id))
        if rep is not None:
            return rep
    if anchor is not None and reps:
        return min(
            reps,
            key=lambda rep: min(
                abs(int(rep["start_frame"]) - anchor),
                abs(int(rep["end_frame"]) - anchor),
            ),
        )
    return reps[0] if reps else None


def select_evidence(
    issue: dict[str, Any],
    reps: list[dict[str, Any]],
    fps: float,
    frame_count: int,
    trim_start: int,
    templates: dict[str, Any],
) -> dict[str, Any]:
    radius = int(templates.get("evidence", {}).get("radius_frames", 8))
    max_distance = int(templates.get("evidence", {}).get("max_fallback_distance_frames", 45))
    anchor = as_int(issue.get("anchor_frame"))
    rep = choose_rep(issue, reps, anchor)
    fallback = False
    lower = 0
    upper = max(0, frame_count - 1)
    preferred_anchor: int | None = None
    if rep is not None:
        lower = int(rep.get("start_frame") or 0)
        upper = int(rep.get("end_frame") or upper)
        preferred_anchor = as_int(rep.get("bottom_frame")) or lower
    if anchor is None or not (lower <= anchor <= upper):
        if preferred_anchor is not None and abs((anchor or 0) - preferred_anchor) > max_distance:
            fallback = True
        anchor = preferred_anchor if preferred_anchor is not None else max(lower, min(upper, anchor or lower))
        fallback = True
    anchor = max(lower, min(upper, int(anchor)))

    interval = issue.get("evidence_interval") or []
    start = as_int(interval[0]) if len(interval) >= 2 else None
    end = as_int(interval[1]) if len(interval) >= 2 else None
    if start is None or end is None or end < start:
        start, end = anchor - radius, anchor + radius
        fallback = True
    start = max(lower, min(upper, int(start)))
    end = max(start, min(upper, int(end)))
    if not (start <= anchor <= end):
        start = max(lower, anchor - radius)
        end = min(upper, anchor + radius)
        fallback = fallback or not (start <= anchor <= end)
        if not (start <= anchor <= end):
            anchor = min(max(anchor, start), end)

    source_anchor = issue.get("anchor_source_frame")
    if not fallback and source_anchor is not None:
        anchor_source = int(source_anchor)
    else:
        anchor_source = trim_start + anchor
    return {
        "anchor_frame": int(anchor),
        "anchor_source_frame": int(anchor_source),
        "anchor_time_s": round(anchor / fps, 6) if fps > 0 else 0.0,
        "evidence_interval_frames": [int(start), int(end)],
        "evidence_interval_s": [
            round(start / fps, 6) if fps > 0 else 0.0,
            round(end / fps, 6) if fps > 0 else 0.0,
        ],
        "anchor_fallback": bool(fallback),
        "representative_rep_id": as_int(rep.get("rep_id")) if rep else None,
    }


def format_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "-"
    if unit == "deg":
        return f"{value:.1f}°"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "variance":
        return f"{value:.3f}"
    if unit == "s":
        return f"{value:.2f}s"
    return f"{value:.2f}{unit or ''}"


def rep_phrase(issue: dict[str, Any], representative: int | None) -> str:
    rep_ids = [as_int(item) for item in issue.get("support_rep_ids") or [] if as_int(item) is not None]
    if len(rep_ids) == 1:
        return f"rep {rep_ids[0]} "
    if len(rep_ids) > 1:
        return f"rep {rep_ids[0]}–{rep_ids[-1]} "
    if representative is not None:
        return f"rep {representative} "
    return "本组 "


def card_frequency(issue: dict[str, Any], rep_count: int, templates: dict[str, Any]) -> tuple[str, float]:
    priority_cfg = templates.get("priority", {})
    if issue.get("scope") == "clip":
        ratio = float(priority_cfg.get("clip_scope_frequency_ratio", 1.0))
        return "整组", max(0.0, min(1.0, ratio))
    support_count = int(issue.get("support_rep_count") or len(issue.get("support_rep_ids") or []))
    if rep_count <= 0:
        return f"{support_count} rep", 0.0
    ratio = max(0.0, min(1.0, support_count / rep_count))
    ratio = max(float(priority_cfg.get("min_frequency_ratio", 0.0)), ratio)
    return f"{support_count}/{rep_count} reps", ratio


def build_card(
    issue: dict[str, Any],
    assessment: dict[str, Any],
    templates: dict[str, Any],
    trim_start: int,
) -> dict[str, Any]:
    primary = choose_primary_evaluation(issue, templates)
    if primary is None:
        raise ValueError(f"active issue has no product-eligible support: {issue.get('issue_id')}")
    evidence = select_evidence(
        issue,
        assessment.get("reps") or [],
        float(assessment.get("source_measurement", {}).get("input_contract", {}).get("fps") or 30.0),
        int(assessment.get("source_measurement", {}).get("input_contract", {}).get("frame_count") or 0),
        trim_start,
        templates,
    )
    template = (templates.get("issues") or {}).get(issue["issue_code"])
    if not template:
        raise ValueError(f"missing feedback template: {issue['issue_code']}")
    severity = str(issue["severity"])
    confidence = max(0.0, min(1.0, float(issue.get("confidence") or 0.0)))
    frequency_text, frequency_ratio = card_frequency(issue, int(assessment["summary"]["rep_count"]), templates)
    priority_cfg = templates.get("priority", {})
    priority_components = {
        "severity": float(priority_cfg.get("severity_weights", {}).get(severity, 1.0)),
        "frequency": round(frequency_ratio, 6),
        "confidence": round(confidence, 6),
        "actionability": float(
            priority_cfg.get("decision_weights", {}).get(issue.get("decision"), 0.0)
        ),
    }
    priority = (
        priority_components["severity"]
        * priority_components["frequency"]
        * priority_components["confidence"]
        * priority_components["actionability"]
    )
    metric = str(primary["metric"])
    unit = primary.get("unit")
    value = as_float(primary.get("raw_value"))
    detail = template["detail_template"].format(
        rep_phrase=rep_phrase(issue, evidence["representative_rep_id"]),
        metric_label=(templates.get("metric_labels") or {}).get(metric, metric),
        value_text=format_value(value, unit),
        severity_label=SEVERITY_LABEL.get(severity, severity),
    )
    threshold = primary.get("threshold") or {}
    thresholds = {
        key: threshold.get(key)
        for key in ("low", "medium", "high", "direction")
        if threshold.get(key) is not None
    }
    return {
        "issue_id": str(issue["issue_id"]),
        "issue_code": str(issue["issue_code"]),
        "issue_family": str(issue["issue_family"]),
        "decision": str(issue["decision"]),
        "severity": severity,
        "confidence": round(confidence, 6),
        "priority": round(priority, 6),
        "priority_components": priority_components,
        "title": template["title"],
        "detail": detail,
        "cue": template["cue"],
        "limitation": template["limitation"],
        "support_rep_ids": [int(item) for item in issue.get("support_rep_ids") or []],
        "representative_rep_id": evidence["representative_rep_id"],
        "support_measurement_ids": list(issue.get("support_measurement_ids") or []),
        "frequency": frequency_text,
        "frequency_ratio": round(frequency_ratio, 6),
        "primary_metric": metric,
        "primary_value": rounded(value),
        "unit": unit,
        "thresholds": thresholds,
        "highlight_joints": list(template.get("highlight_joints") or []),
        "anchor_frame": evidence["anchor_frame"],
        "anchor_source_frame": evidence["anchor_source_frame"],
        "anchor_time_s": evidence["anchor_time_s"],
        "evidence_interval_frames": evidence["evidence_interval_frames"],
        "evidence_interval_s": evidence["evidence_interval_s"],
        "anchor_fallback": evidence["anchor_fallback"],
        "frame_filename": None,
        "frame_available": False,
        "shadow_only": True,
        "product_actions_enabled": False,
    }


def prioritize_issues(
    assessment: dict[str, Any], templates: dict[str, Any], trim_start: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active = [
        issue
        for issue in assessment.get("issues") or []
        if issue.get("decision") in ACTIVE_DECISIONS
    ]
    cards = [build_card(issue, assessment, templates, trim_start) for issue in active]
    cards.sort(
        key=lambda card: (
            -float(card["priority"]),
            -severity_rank(card["severity"]),
            card["issue_id"],
        )
    )
    selected = []
    omitted = []
    families = set()
    max_one = bool(templates.get("priority", {}).get("max_one_per_family", True))
    top_k = int(templates.get("priority", {}).get("top_k", 3))
    for card in cards:
        if len(selected) >= top_k:
            omitted.append({**card, "omission_reason": "top_k"})
            continue
        if max_one and card["issue_family"] in families:
            omitted.append({**card, "omission_reason": "family_duplicate"})
            continue
        selected.append(card)
        families.add(card["issue_family"])
    return selected, omitted


def feedback_state(assessment: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    summary = assessment.get("summary") or {}
    quality = assessment.get("quality") or {}
    if quality.get("status") == "reject" or summary.get("abstain_reason") == "quality_reject":
        return "quality_reject"
    if int(summary.get("rep_count") or 0) == 0:
        return "no_reps"
    if cards:
        return "active_issues"
    if int(summary.get("evaluable_metric_count") or 0) == 0:
        return "low_confidence"
    if int(summary.get("diagnostic_issue_count") or 0) > 0:
        return "low_confidence"
    if summary.get("abstained"):
        return "low_confidence"
    return "no_reliable_issue"


def build_limitations(
    assessment: dict[str, Any], omitted: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    summary = assessment.get("summary") or {}
    diagnostic = int(summary.get("diagnostic_issue_count") or 0)
    if diagnostic:
        output.append(
            {
                "code": "diagnostic_only",
                "message": f"{diagnostic} 个诊断项因视角、指标可用性或置信度不足未进入主卡。",
                "count": diagnostic,
            }
        )
    suppressed = int(summary.get("suppressed_issue_count") or 0)
    if suppressed:
        output.append(
            {
                "code": "suppressed",
                "message": f"{suppressed} 个问题组被质量、置信度或物理范围门抑制。",
                "count": suppressed,
            }
        )
    omitted_by_reason = Counter(item["omission_reason"] for item in omitted)
    if omitted_by_reason.get("family_duplicate"):
        count = omitted_by_reason["family_duplicate"]
        output.append(
            {
                "code": "family_deduplicated",
                "message": f"{count} 个同族候选已合并，仅保留排序最高的一张卡。",
                "count": count,
            }
        )
    if omitted_by_reason.get("top_k"):
        count = omitted_by_reason["top_k"]
        output.append(
            {
                "code": "top_k",
                "message": f"{count} 个低优先级候选未进入默认 top-k 反馈。",
                "count": count,
            }
        )
    if (assessment.get("quality") or {}).get("status") == "warn":
        output.append(
            {
                "code": "quality_warn",
                "message": "输入质量存在警告，问题严重度已按规则降级或封顶。",
                "count": 1,
            }
        )
    return output


def build_feedback(
    assessment: dict[str, Any],
    templates: dict[str, Any],
    *,
    runtime_adapter: bool,
    benchmark_path: str | None = None,
    benchmark_sha256: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    input_contract = assessment.get("source_measurement", {}).get("input_contract") or {}
    trim_start = int((input_contract.get("trim") or {}).get("start_frame") or 0)
    cards, omitted = prioritize_issues(assessment, templates, trim_start)
    state = feedback_state(assessment, cards)
    if state in {"quality_reject", "no_reps"}:
        cards = []
    summary = assessment.get("summary") or {}
    return {
        "schema_version": "vls-feedback-v2",
        "component": "feedback_builder",
        "generated_at": generated_at or utc_now(),
        "clip_id": str(assessment["clip_id"]),
        "exercise": str(assessment["exercise"]),
        "view": str(assessment["view"]),
        "state": state,
        "state_message": str(templates["states"][state]),
        "runtime_contract": {
            "runtime_adapter": bool(runtime_adapter),
            "source_assessment_schema": str(assessment.get("schema_version")),
            "source_split": str(assessment.get("split")),
            "benchmark_path": benchmark_path,
            "benchmark_sha256": benchmark_sha256,
        },
        "assessment_summary": {
            "conclusion": str(summary.get("conclusion")),
            "quality_status": str((assessment.get("quality") or {}).get("status") or "unknown"),
            "rep_count": int(summary.get("rep_count") or 0),
            "evaluable_metric_count": int(summary.get("evaluable_metric_count") or 0),
            "candidate_issue_count": int(summary.get("candidate_issue_count") or 0),
            "reportable_issue_count": int(summary.get("reportable_issue_count") or 0),
            "diagnostic_issue_count": int(summary.get("diagnostic_issue_count") or 0),
            "suppressed_issue_count": int(summary.get("suppressed_issue_count") or 0),
            "abstained": bool(summary.get("abstained")),
            "abstain_reason": summary.get("abstain_reason"),
            "shadow_only": True,
            "product_actions_enabled": False,
        },
        "cards": cards,
        "limitations": build_limitations(assessment, omitted),
        "policy": {
            "top_k": int(templates.get("priority", {}).get("top_k", 3)),
            "max_one_per_family": bool(
                templates.get("priority", {}).get("max_one_per_family", True)
            ),
            "shadow_only": True,
            "product_actions_enabled": False,
            "expert_calibrated": False,
        },
    }


def validate_schema(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    ]


def invariant_checks(feedback: dict[str, Any], assessment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cards = feedback.get("cards") or []
    if feedback["state"] == "quality_reject" and cards:
        errors.append("quality reject must not emit cards")
    if feedback["state"] == "active_issues" and not cards:
        errors.append("active state requires at least one card")
    if len(cards) > int(feedback["policy"]["top_k"]):
        errors.append("top-k exceeded")
    families = Counter(card["issue_family"] for card in cards)
    if feedback["policy"]["max_one_per_family"] and any(count > 1 for count in families.values()):
        errors.append("multiple cards in one family")
    reps = {
        as_int(rep.get("rep_id")): rep
        for rep in assessment.get("reps") or []
        if as_int(rep.get("rep_id")) is not None
    }
    product_measurements = {
        item.get("measurement_id")
        for item in assessment.get("metric_evaluations") or []
        if item.get("eligibility") == "provisional" and item.get("report_view_allowed")
    }
    for card in cards:
        if card.get("decision") not in ACTIVE_DECISIONS:
            errors.append(f"non-active decision in cards: {card['issue_id']}")
        start, end = card.get("evidence_interval_frames") or [None, None]
        anchor = card.get("anchor_frame")
        if start is None or end is None or anchor is None or not (int(start) <= int(anchor) <= int(end)):
            errors.append(f"anchor outside evidence: {card['issue_id']}")
        if severity_rank(card.get("severity")) >= severity_rank("medium") and start is None:
            errors.append(f"medium/high lacks evidence: {card['issue_id']}")
        rep_id = card.get("representative_rep_id")
        rep = reps.get(rep_id)
        if rep is not None and start is not None and end is not None:
            if int(start) < int(rep["start_frame"]) or int(end) > int(rep["end_frame"]):
                errors.append(f"evidence crosses representative rep: {card['issue_id']}")
        if not set(card.get("support_measurement_ids") or []).intersection(product_measurements):
            errors.append(f"card lacks product-eligible support: {card['issue_id']}")
        if not (0.0 <= float(card.get("confidence") or 0.0) <= 1.0):
            errors.append(f"invalid card confidence: {card['issue_id']}")
    if feedback["policy"]["product_actions_enabled"]:
        errors.append("product actions must remain disabled")
    return errors


def frame_url(template: str, clip_id: str, frame_name: str | None) -> str | None:
    if not frame_name:
        return None
    return template.format(clip_id=clip_id, frame_name=frame_name)


def render_card_frames(
    feedback: dict[str, Any],
    clip_meta: dict[str, Any],
    analysis: dict[str, Any],
    frames_dir: Path,
    model_path: Path,
) -> tuple[int, list[str]]:
    """Render optional evidence frames; failures degrade to timestamp-only evidence."""
    cards = feedback.get("cards") or []
    if not cards:
        return 0, []
    source = Path(str(clip_meta.get("video_path") or clip_meta.get("source_path") or ""))
    if not source.is_file() or not model_path.is_file():
        return 0, ["evidence_frame_source_or_model_unavailable"]

    import cv2

    from render_clip_report import (
        VideoFrameReader,
        create_landmarker,
        detect_normalized_landmarks,
        render_issue_frame,
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    warnings: list[str] = []
    landmarker = None
    try:
        landmarker = create_landmarker(model_path)
        with VideoFrameReader(source, str(analysis.get("rotation") or "none")) as reader:
            for card in cards:
                filename = f"{card['issue_id']}_frame_{int(card['anchor_frame'])}.png"
                target = frames_dir / filename
                card["frame_filename"] = filename
                if target.is_file():
                    card["frame_available"] = True
                    generated += 1
                    continue
                try:
                    frame = reader.read(int(card["anchor_source_frame"]))
                    landmarks = detect_normalized_landmarks(landmarker, frame)
                    fake_issue = {
                        "issue_id": card["issue_id"],
                        "issue_code": card["issue_code"],
                        "severity": card["severity"],
                        "confidence": card["confidence"],
                        "anchor_frame": card["anchor_source_frame"],
                        "anchor_time_s": card["anchor_source_frame"] / float(analysis["fps"]),
                        "representative_rep_id": card["representative_rep_id"],
                        "text": {
                            "title": card["title"],
                            "detail": card["detail"],
                            "cue": card["cue"],
                        },
                        "measurements": {
                            "primary_metric": card["primary_metric"],
                            "primary_value": card["primary_value"],
                            "unit": card["unit"],
                            "thresholds": card["thresholds"],
                        },
                        "visual_overlay": {
                            "highlight_joints": card["highlight_joints"],
                            "camera_zoom": 1.2,
                        },
                    }
                    annotated, _ = render_issue_frame(
                        frame,
                        landmarks,
                        fake_issue,
                        {},
                        float(analysis["trim"]["start_frame"]) / float(analysis["fps"]),
                    )
                    if not cv2.imwrite(str(target), annotated):
                        raise RuntimeError("cv2.imwrite returned false")
                    card["frame_available"] = True
                    generated += 1
                except Exception as exc:  # noqa: BLE001 - evidence fallback must not abort feedback
                    warnings.append(f"{card['issue_id']}:frame_render_failed:{type(exc).__name__}")
                    card["frame_available"] = False
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"frame_render_unavailable:{type(exc).__name__}")
    finally:
        if landmarker is not None:
            landmarker.close()
    return generated, warnings


def build_report_html(
    feedback: dict[str, Any],
    *,
    media_url: str,
    frame_url_template: str,
) -> str:
    clip_id = feedback["clip_id"]
    cards_html = []
    for card in feedback.get("cards") or []:
        image = None
        if card.get("frame_available") and card.get("frame_filename"):
            image = frame_url(frame_url_template, clip_id, card["frame_filename"])
        image_html = f'<img src="{html.escape(image)}" alt="证据帧">' if image else ""
        start, end = card["evidence_interval_s"]
        cards_html.append(
            f"""
            <article class="card {html.escape(card['severity'])}" data-anchor="{card['anchor_time_s']}"
              data-start="{start}" data-end="{end}">
              <div class="card-head"><span class="sev">{html.escape(card['severity'])}</span>
                <strong>{html.escape(card['title'])}</strong>
                <span class="tag">{'研究候选' if card['decision'] == 'candidate' else 'Shadow 可报告'}</span></div>
              <p>{html.escape(card['detail'])}</p>
              <p class="cue">建议：{html.escape(card['cue'])}</p>
              <p class="limit">限制：{html.escape(card['limitation'])}</p>
              <div class="meta">rep {html.escape(str(card['representative_rep_id'] or '-'))} ·
                {html.escape(card['frequency'])} · {start:.2f}–{end:.2f}s ·
                置信度 {card['confidence']:.2f}{' · 锚点回退' if card['anchor_fallback'] else ''}</div>
              {image_html}
              <button type="button" class="play">播放证据区间</button>
            </article>
            """
        )
    limitations = "".join(
        f"<li>{html.escape(item['message'])}</li>" for item in feedback.get("limitations") or []
    ) or "<li>无额外限制项。</li>"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(clip_id)} · 动作反馈</title>
<style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:24px}}
.banner,.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;margin:12px 0}}
.banner{{border-color:#9e6a03;color:#f0c56b}}.card{{cursor:pointer}}.card.high{{border-left:5px solid #f85149}}
.card.medium{{border-left:5px solid #d29922}}.card.low{{border-left:5px solid #58a6ff}}
.card-head{{display:flex;gap:8px;align-items:center}}.sev,.tag{{font-size:12px;border:1px solid currentColor;border-radius:999px;padding:1px 7px}}
.meta,.limit,.cue{{font-size:13px;color:#9da7b3}}.cue{{color:#b9cbcf}}img{{max-width:100%;border-radius:8px;margin-top:10px}}
video{{width:100%;max-height:60vh;background:#000;border-radius:10px}}button{{padding:5px 10px;border-radius:7px;border:1px solid #30363d;background:#21262d;color:#e6edf3;cursor:pointer}}
</style></head><body><main>
<h1>{html.escape(clip_id)} · 反馈 v2</h1>
<div class="banner">{html.escape(feedback['state_message'])}<br><small>Shadow-only；product_actions_enabled=false。</small></div>
<video id="player" controls preload="metadata" src="{html.escape(media_url)}"></video>
<h2>问题卡</h2>{''.join(cards_html) or '<p>无主问题卡。</p>'}
<h2>限制与未展示项</h2><ul>{limitations}</ul>
<script>
const player=document.getElementById('player');
for(const card of document.querySelectorAll('.card')){{
  const seek=()=>{{player.currentTime=Number(card.dataset.start||card.dataset.anchor);player.play();}};
  card.addEventListener('click',seek);
}}
</script></main></body></html>
"""


def clip_meta_for(clip_id: str, analysis_dir: Path) -> dict[str, Any]:
    path = analysis_dir / clip_id / "clip.json"
    return load_json(path) if path.is_file() else {}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates = load_json(args.templates)
    rules = load_json(args.rules)
    taxonomy = load_json(args.taxonomy)
    contracts = load_json(args.contracts)
    schema = load_json(args.schema)
    generated_at = args.generated_at or utc_now()
    analyses = select_clip_ids(args, discover_analyses(args.analysis_dir))
    if not analyses:
        raise ValueError("no clip matched")

    clips = [
        runtime_clip(analysis, clip_meta_for(str(analysis["clip_id"]), args.analysis_dir))
        for analysis in analyses
    ]
    runtime_benchmark = write_runtime_benchmark(args.output_dir, clips, generated_at)
    summaries = []
    for analysis, clip in zip(analyses, clips):
        clip_id = str(clip["clip_id"])
        output = args.output_dir / clip_id
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {output}")
        if args.assessment_dir is not None:
            assessment = load_existing_assessment(args.assessment_dir, clip_id)
            if assessment is None:
                raise FileNotFoundError(f"assessment missing: {args.assessment_dir / clip_id}")
            runtime_adapter = False
        else:
            _, assessment = build_runtime_assessment(
                clip,
                args.analysis_dir,
                contracts,
                rules,
                taxonomy,
                runtime_benchmark,
                args.contracts,
                generated_at,
            )
            runtime_adapter = True
        benchmark_path = (assessment.get("source_measurement") or {}).get("input_contract", {}).get(
            "benchmark_path"
        )
        benchmark_sha = (assessment.get("source_measurement") or {}).get("input_contract", {}).get(
            "benchmark_sha256"
        )
        feedback = build_feedback(
            assessment,
            templates,
            runtime_adapter=runtime_adapter,
            benchmark_path=benchmark_path,
            benchmark_sha256=benchmark_sha,
            generated_at=generated_at,
        )
        clip_meta = clip_meta_for(clip_id, args.analysis_dir)
        frame_warnings: list[str] = []
        if args.render_frames:
            _, frame_warnings = render_card_frames(
                feedback,
                clip_meta,
                analysis,
                output / "frames",
                args.model,
            )
        feedback["limitations"].extend(
            {
                "code": "evidence_frame_fallback",
                "message": warning,
                "count": 1,
            }
            for warning in frame_warnings
        )
        feedback["cards"] = [
            card for card in feedback["cards"] if card.get("decision") in ACTIVE_DECISIONS
        ]
        feedback_errors = validate_schema(feedback, schema)
        invariant_errors = invariant_checks(feedback, assessment)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "assessment_v2.json", assessment)
        write_json(output / "feedback_v2.json", feedback)
        write_json(
            output / "evidence.json",
            {
                "schema_version": "vls-feedback-evidence-v1",
                "clip_id": clip_id,
                "policy": templates.get("evidence", {}),
                "cards": [
                    {
                        key: card[key]
                        for key in (
                            "issue_id",
                            "anchor_frame",
                            "anchor_source_frame",
                            "anchor_time_s",
                            "evidence_interval_frames",
                            "evidence_interval_s",
                            "anchor_fallback",
                            "support_measurement_ids",
                            "frame_filename",
                            "frame_available",
                        )
                    }
                    for card in feedback["cards"]
                ],
            },
        )
        media_url = args.media_url_template.format(clip_id=clip_id)
        (output / "report_v2.html").write_text(
            build_report_html(
                feedback,
                media_url=media_url,
                frame_url_template=args.frame_url_template,
            ),
            encoding="utf-8",
        )
        write_json(
            output / "checks" / "schema_validation.json",
            {"ok": not feedback_errors, "errors": feedback_errors},
        )
        write_json(
            output / "checks" / "invariants.json",
            {"ok": not invariant_errors, "errors": invariant_errors},
        )
        if feedback_errors or invariant_errors:
            raise ValueError(
                f"feedback validation failed for {clip_id}: "
                + "; ".join(feedback_errors + invariant_errors)
            )
        summaries.append(
            {
                "clip_id": clip_id,
                "exercise": feedback["exercise"],
                "view": feedback["view"],
                "state": feedback["state"],
                "card_count": len(feedback["cards"]),
                "candidate_cards": sum(card["decision"] == "candidate" for card in feedback["cards"]),
                "reportable_cards": sum(card["decision"] == "reportable" for card in feedback["cards"]),
                "evidence_frames": sum(card["frame_available"] for card in feedback["cards"]),
                "runtime_adapter": runtime_adapter,
            }
        )

    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": "vls-feedback-summary-v2",
            "component": "feedback_builder",
            "generated_at": generated_at,
            "clip_count": len(summaries),
            "state_counts": dict(sorted(Counter(item["state"] for item in summaries).items())),
            "card_count": sum(item["card_count"] for item in summaries),
            "results": summaries,
            "policy": {
                "shadow_only": True,
                "product_actions_enabled": False,
                "expert_calibrated": False,
            },
        },
    )
    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"}
    )
    checksums = [f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}" for path in artifact_paths]
    (args.output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": "vls-feedback-manifest-v2",
            "component": "feedback_builder",
            "generated_at": generated_at,
            "clip_count": len(summaries),
            "artifacts": [
                {
                    "relative_path": path.relative_to(args.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
            "policy": {
                "shadow_only": True,
                "product_actions_enabled": False,
                "expert_calibrated": False,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
