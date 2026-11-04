#!/usr/bin/env python3
"""Build per-rep features and evidence-backed issue cards from task1-4B outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "clip_analysis_v1"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_TEMPLATES = ROOT / "configs" / "task1_4_feedback_templates.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "feedback_v1"

ISSUE_SCHEMA_VERSION = "task1-4-issues-v1"
SUMMARY_SCHEMA_VERSION = "task1-4-feedback-summary-v1"
FEATURE_FIELDS = [
    "clip_id",
    "exercise",
    "rep_id",
    "metric",
    "scope",
    "raw_value",
    "unit",
    "confidence",
    "source",
    "view_requirement",
    "status",
    "reference_value",
    "deviation",
    "severity_candidate",
    "anchor_frame",
    "anchor_time_s",
    "note",
]
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
ACTIVE_FEATURE_STATUSES = {"ok", "low_confidence"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument(
        "--split",
        default="dev",
        help="Only process this analysis split; use 'all' to disable filtering.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def source_time_s(frame: int | None, fps: float) -> float | None:
    return round(frame / fps, 6) if frame is not None and fps > 0 else None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def load_clip_data(clip_dir: Path) -> dict[str, Any]:
    analysis = read_json(clip_dir / "analysis.json")
    trim_start = int(analysis["trim"]["start_frame"])
    fps = float(analysis["fps"])
    reps = []
    with (clip_dir / "reps.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rep = dict(row)
            for name in (
                "rep_id",
                "start_frame",
                "end_frame",
                "top_frame",
                "bottom_frame",
            ):
                rep[name] = to_int(row.get(name))
            for name in ("start_time_s", "end_time_s", "bottom_time_s", "duration_s", "confidence"):
                rep[name] = to_float(row.get(name))
            rep["start_local"] = rep["start_frame"] - trim_start
            rep["end_local"] = rep["end_frame"] - trim_start
            rep["bottom_local"] = rep["bottom_frame"] - trim_start
            reps.append(rep)

    wide = {}
    with (clip_dir / "angles" / "angles_wide.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = to_int(row.get("frame_id"))
            if frame is not None:
                wide[frame] = row

    angle_rows = {}
    with (clip_dir / "angles" / "angles.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = to_int(row.get("frame_id"))
            name = row.get("angle_name")
            if frame is not None and name:
                angle_rows[(frame, name)] = row

    body_frames = {}
    body_path = clip_dir / "angles" / "landmarks_body_frame.json"
    for record in read_json(body_path):
        frame = to_int(record.get("frame_id"))
        if frame is not None:
            body_frames[frame] = record.get("landmarks_body_frame")

    return {
        "clip_dir": clip_dir,
        "analysis": analysis,
        "trim_start": trim_start,
        "fps": fps,
        "reps": sorted(reps, key=lambda rep: rep["rep_id"]),
        "wide": wide,
        "angle_rows": angle_rows,
        "body_frames": body_frames,
    }


def wide_float(wide: dict[int, dict[str, str]], frame: int, name: str) -> float | None:
    row = wide.get(frame)
    return to_float(row.get(name)) if row else None


def angle_confidence(
    angle_rows: dict[tuple[int, str], dict[str, str]],
    frame: int,
    name: str,
) -> float | None:
    row = angle_rows.get((frame, name))
    if not row or row.get("valid") != "1":
        return None
    return to_float(row.get("confidence"))


def min_confidence(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return min(value for value in values if value is not None)


def view_allows(views: str, view: str) -> bool:
    if views == "any":
        return True
    if views == "frontal_or_oblique":
        return view in {"front", "rear", "oblique_left", "oblique_right"}
    if views == "side_or_oblique":
        return view in {"side_left", "side_right", "oblique_left", "oblique_right"}
    return False


def metric_enabled(metric: str, metric_cfg: dict[str, Any], quality: dict[str, Any]) -> bool:
    view_support = quality.get("view_support") or {}
    usable = view_support.get("usable_metrics")
    if usable is not None:
        return metric in usable
    return view_allows(str(metric_cfg.get("views", "any")), str(quality.get("view", "unknown")))


def inward_knee_valgus_deg(points: list[list[float]], side: str) -> float | None:
    hip_index, knee_index, ankle_index = (23, 25, 27) if side == "left" else (24, 26, 28)
    hip = points[hip_index]
    knee = points[knee_index]
    ankle = points[ankle_index]
    dx = ankle[0] - hip[0]
    dy = ankle[1] - hip[1]
    leg_length = math.hypot(dx, dy)
    if leg_length < 1e-8:
        return None
    lateral = (dx * (knee[1] - hip[1]) - dy * (knee[0] - hip[0])) / leg_length
    medial = lateral if side == "left" else -lateral
    return math.degrees(math.atan2(max(0.0, medial), leg_length))


def severity_from_value(
    value: float | None,
    confidence: float | None,
    threshold_cfg: dict[str, Any] | None,
    confidence_cfg: dict[str, Any],
) -> tuple[str, str, float | None]:
    if value is None:
        return "none", "unavailable", None
    if not threshold_cfg:
        if confidence is not None and confidence < float(confidence_cfg["low_below"]):
            return "none", "low_confidence", None
        return "none", "ok", None
    if confidence is None:
        return "none", "unavailable", None

    low = float(threshold_cfg["low"])
    medium = float(threshold_cfg["medium"])
    high = float(threshold_cfg["high"])
    direction = threshold_cfg.get("direction", "high_bad")
    if direction == "low_bad":
        if value <= high:
            raw_severity, chosen_threshold = "high", high
        elif value <= medium:
            raw_severity, chosen_threshold = "medium", medium
        elif value <= low:
            raw_severity, chosen_threshold = "low", low
        else:
            raw_severity, chosen_threshold = "none", None
    else:
        if value >= high:
            raw_severity, chosen_threshold = "high", high
        elif value >= medium:
            raw_severity, chosen_threshold = "medium", medium
        elif value >= low:
            raw_severity, chosen_threshold = "low", low
        else:
            raw_severity, chosen_threshold = "none", None

    if confidence < float(confidence_cfg["suppress_below"]):
        return "none", "unusable_low_confidence", chosen_threshold
    if confidence < float(confidence_cfg["low_below"]) and raw_severity in {"medium", "high"}:
        return "low", "low_confidence", chosen_threshold
    status = "low_confidence" if confidence < float(confidence_cfg["low_below"]) else "ok"
    return raw_severity, status, chosen_threshold


def deviation_ratio(value: float | None, threshold: float | None, direction: str) -> float:
    if value is None or threshold is None or threshold <= 0:
        return 0.0
    if direction == "low_bad":
        return threshold / max(abs(value), 1e-8)
    return abs(value) / threshold


def feature_payload(
    data: dict[str, Any],
    rep_id: int | None,
    metric: str,
    metric_cfg: dict[str, Any],
    scope: str,
    value: float | None,
    confidence: float | None,
    source: str,
    anchor_local: int | None,
    note: str,
    severity: str = "none",
    status: str = "ok",
) -> dict[str, Any]:
    analysis = data["analysis"]
    anchor_frame = data["trim_start"] + anchor_local if anchor_local is not None else None
    return {
        "clip_id": analysis["clip_id"],
        "exercise": analysis["exercise"],
        "rep_id": rep_id,
        "metric": metric,
        "scope": scope,
        "raw_value": round_or_none(value),
        "unit": metric_cfg.get("unit", ""),
        "confidence": round_or_none(confidence),
        "source": source,
        "view_requirement": metric_cfg.get("views", "any"),
        "status": status,
        "reference_value": None,
        "deviation": None,
        "severity_candidate": severity,
        "anchor_frame": anchor_frame,
        "anchor_time_s": source_time_s(anchor_frame, data["fps"]),
        "note": note,
    }


def metric_value_and_anchor(
    data: dict[str, Any],
    rep: dict[str, Any],
    metric: str,
    exercise: str,
    templates: dict[str, Any],
) -> tuple[float | None, float | None, int | None, str, str]:
    bottom = rep["bottom_local"]
    start = rep["start_local"]
    end = rep["end_local"]
    wide = data["wide"]
    angle_rows = data["angle_rows"]

    if metric == "rep_duration_s":
        anchor = (start + end) // 2
        return rep.get("duration_s"), rep.get("confidence"), anchor, "reps.duration_s", "rep duration"

    if metric == "spine_stability_bottom":
        radius = int(templates["evidence"]["stability_window_frames"])
        candidates = []
        for frame in range(max(start, bottom - radius), min(end, bottom + radius) + 1):
            value = wide_float(wide, frame, "spine_stability")
            if value is not None:
                candidates.append((value, frame))
        if not candidates:
            return None, None, None, "angles_wide.spine_stability", "no finite stability value"
        value, anchor = max(candidates)
        confidence = angle_confidence(angle_rows, anchor, "spine_stability")
        return (
            value,
            confidence,
            anchor,
            "angles_wide.spine_stability",
            f"maximum variance within +/-{radius} frames of bottom",
        )

    if metric == "knee_valgus_bottom":
        points = data["body_frames"].get(bottom)
        if not points:
            return None, None, None, "landmarks_body_frame", "body-frame landmarks unavailable"
        side_values = {
            side: inward_knee_valgus_deg(points, side) for side in ("left", "right")
        }
        finite_sides = [side for side, value in side_values.items() if value is not None]
        if not finite_sides:
            return None, None, None, "landmarks_body_frame", "degenerate knee alignment"
        side = max(finite_sides, key=lambda name: side_values[name] or 0.0)
        confidence = angle_confidence(angle_rows, bottom, f"{side}_knee_valgus_coronal")
        return (
            side_values[side],
            confidence,
            bottom,
            "landmarks_body_frame.hip_knee_ankle_coronal",
            f"larger inward deviation side={side}",
        )

    if metric == "knee_asymmetry_bottom":
        value = wide_float(wide, bottom, "knee_flexion_asymmetry")
        confidence = angle_confidence(angle_rows, bottom, "knee_flexion_asymmetry")
        return value, confidence, bottom, "angles_wide.knee_flexion_asymmetry", "at bottom"

    if metric == "hip_asymmetry_bottom":
        value = wide_float(wide, bottom, "hip_flexion_asymmetry")
        confidence = angle_confidence(angle_rows, bottom, "hip_flexion_asymmetry")
        return value, confidence, bottom, "angles_wide.hip_flexion_asymmetry", "at bottom"

    if metric == "elbow_asymmetry_bottom":
        left = wide_float(wide, bottom, "left_elbow_flexion")
        right = wide_float(wide, bottom, "right_elbow_flexion")
        value = abs(left - right) if left is not None and right is not None else None
        confidence = min_confidence(
            angle_confidence(angle_rows, bottom, "left_elbow_flexion"),
            angle_confidence(angle_rows, bottom, "right_elbow_flexion"),
        )
        return value, confidence, bottom, "angles_wide.left_right_elbow_flexion", "at bottom"

    if metric == "shoulder_asymmetry_bottom":
        left = wide_float(wide, bottom, "left_shoulder_flexion")
        right = wide_float(wide, bottom, "right_shoulder_flexion")
        value = abs(left - right) if left is not None and right is not None else None
        confidence = min_confidence(
            angle_confidence(angle_rows, bottom, "left_shoulder_flexion"),
            angle_confidence(angle_rows, bottom, "right_shoulder_flexion"),
        )
        return value, confidence, bottom, "angles_wide.left_right_shoulder_flexion", "at bottom"

    return None, None, None, "unavailable", f"unsupported metric for {exercise}"


def compute_features(
    data: dict[str, Any],
    specs: dict[str, Any],
    templates: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis = data["analysis"]
    exercise = analysis["exercise"]
    exercise_spec = specs["exercises"][exercise]
    metrics_cfg = exercise_spec["metrics"]
    quality = analysis.get("quality", {})
    thresholds = templates["thresholds"]
    confidence_cfg = templates["confidence"]
    features = []

    for rep in data["reps"]:
        for metric, metric_cfg in metrics_cfg.items():
            if metric == "rep_duration_cv":
                continue
            if not metric_enabled(metric, metric_cfg, quality):
                features.append(
                    feature_payload(
                        data,
                        rep["rep_id"],
                        metric,
                        metric_cfg,
                        "rep",
                        None,
                        None,
                        "not_evaluated",
                        None,
                        "disabled by view support",
                        status="disabled",
                    )
                )
                continue
            value, confidence, anchor, source, note = metric_value_and_anchor(
                data, rep, metric, exercise, templates
            )
            severity, status, _ = severity_from_value(
                value, confidence, thresholds.get(metric), confidence_cfg
            )
            features.append(
                feature_payload(
                    data,
                    rep["rep_id"],
                    metric,
                    metric_cfg,
                    "rep",
                    value,
                    confidence,
                    source,
                    anchor,
                    note,
                    severity,
                    status,
                )
            )

    if "rep_duration_cv" in metrics_cfg:
        metric_cfg = metrics_cfg["rep_duration_cv"]
        if not metric_enabled("rep_duration_cv", metric_cfg, quality):
            features.append(
                feature_payload(
                    data,
                    None,
                    "rep_duration_cv",
                    metric_cfg,
                    "clip",
                    None,
                    None,
                    "not_evaluated",
                    None,
                    "disabled by view support",
                    status="disabled",
                )
            )
        elif len(data["reps"]) < int(metric_cfg.get("min_reps", 3)):
            features.append(
                feature_payload(
                    data,
                    None,
                    "rep_duration_cv",
                    metric_cfg,
                    "clip",
                    None,
                    None,
                    "reps.duration_s",
                    None,
                    f"requires at least {metric_cfg.get('min_reps', 3)} complete reps",
                    status="insufficient_reps",
                )
            )
        else:
            durations = [rep["duration_s"] for rep in data["reps"] if rep.get("duration_s") is not None]
            mean_duration = statistics.mean(durations)
            value = statistics.pstdev(durations) / mean_duration if mean_duration > 0 else None
            confidences = [rep.get("confidence") for rep in data["reps"]]
            confidence = min_confidence(*confidences)
            severity, status, _ = severity_from_value(
                value,
                confidence,
                thresholds.get("rep_duration_cv"),
                confidence_cfg,
            )
            features.append(
                feature_payload(
                    data,
                    None,
                    "rep_duration_cv",
                    metric_cfg,
                    "clip",
                    value,
                    confidence,
                    "reps.duration_s",
                    None,
                    "population CV over complete reps",
                    severity,
                    status,
                )
            )
    return features


def metric_statuses(
    metric_names: Iterable[str],
    rep_features: dict[tuple[int, str], dict[str, Any]],
    group_features: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    result = {}
    for metric in metric_names:
        group = group_features.get(metric)
        if group is not None:
            result[metric] = [group["status"]]
            continue
        statuses = sorted(
            {
                feature["status"]
                for (rep_id, name), feature in rep_features.items()
                if name == metric and rep_id is not None
            }
        )
        result[metric] = statuses or ["unavailable"]
    return result


def flattened_statuses(statuses: dict[str, list[str]]) -> set[str]:
    return {status for values in statuses.values() for status in values}


def view_requirement_text(metric_names: Iterable[str], exercise_spec: dict[str, Any], templates: dict[str, Any]) -> str:
    labels = templates["view_requirement_labels"]
    views = {
        exercise_spec["metrics"][metric].get("views", "any")
        for metric in metric_names
        if metric in exercise_spec["metrics"]
    }
    return "、".join(labels.get(view, view) for view in sorted(views)) or labels["any"]


def text_context(
    data: dict[str, Any],
    exercise_spec: dict[str, Any],
    templates: dict[str, Any],
    primary_metric: str,
    value: float | None,
    threshold: float | None,
    representative_rep_id: int | None,
    group_rep_ids: list[int] | None = None,
) -> dict[str, str]:
    rep_ids = group_rep_ids or ([representative_rep_id] if representative_rep_id is not None else [])
    if len(rep_ids) == 1:
        rep_phrase = f"第 {rep_ids[0]} 个 rep"
    elif rep_ids:
        rep_phrase = f"第 {min(rep_ids)}–{max(rep_ids)} 个 rep"
        if representative_rep_id is not None:
            rep_phrase += f"（代表 rep {representative_rep_id}）"
    else:
        rep_phrase = "本组"
    return {
        "rep_phrase": rep_phrase,
        "view": str(data["analysis"].get("view", "unknown")),
        "view_requirement_text": view_requirement_text([primary_metric], exercise_spec, templates),
        "metric_label": templates["metric_labels"].get(primary_metric, primary_metric),
        "value_text": format_measurement(value, templates["thresholds"].get(primary_metric, {}).get("unit", "")),
        "threshold_text": format_measurement(
            threshold, templates["thresholds"].get(primary_metric, {}).get("unit", "")
        ),
        "representative_rep_id": str(representative_rep_id or ""),
    }


def format_measurement(value: float | None, unit: str) -> str:
    if value is None:
        return "不可用"
    if unit == "deg":
        return f"{value:.1f}°"
    if unit == "s":
        return f"{value:.2f} s"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "variance":
        return f"{value:.3f}"
    return f"{value:.2f} {unit}".strip()


def safe_format(template: str, context: dict[str, str]) -> str:
    try:
        return template.format(**context)
    except KeyError:
        return template


def visual_overlay(data: dict[str, Any], issue_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "view": data["analysis"].get("view", "unknown"),
        "highlight_joints": issue_cfg.get("highlight_joints", []),
        "draw": issue_cfg.get("draw", []),
        "camera_zoom": issue_cfg.get("camera_zoom", 1.0),
    }


def suppressed_issue(
    data: dict[str, Any],
    issue_code: str,
    reason: str,
    issue_cfg: dict[str, Any],
    exercise_spec: dict[str, Any],
    templates: dict[str, Any],
    metric_names: list[str],
    statuses: dict[str, list[str]],
) -> dict[str, Any]:
    context = text_context(
        data,
        exercise_spec,
        templates,
        metric_names[0] if metric_names else "unknown",
        None,
        None,
        None,
    )
    context["view_requirement_text"] = view_requirement_text(metric_names, exercise_spec, templates)
    detail_template = issue_cfg.get("disabled_detail", "本项未评估。")
    return {
        "issue_id": f"{issue_code}_suppressed",
        "issue_code": issue_code,
        "exercise": data["analysis"]["exercise"],
        "scope": "not_evaluated",
        "rep_ids": [],
        "representative_rep_id": None,
        "frequency": f"0/{len(data['reps'])}",
        "frequency_ratio": 0.0,
        "severity": "none",
        "confidence": None,
        "anchor_frame": None,
        "anchor_time_s": None,
        "evidence_interval": None,
        "measurements": {
            "metric_statuses": statuses,
            "reason": reason,
        },
        "visual_overlay": visual_overlay(data, issue_cfg),
        "text": {
            "title": f"{issue_cfg.get('title', issue_code)}（未评估）",
            "detail": safe_format(detail_template, context),
            "cue": templates.get("suppressed_cue", ""),
        },
        "suppressed": True,
        "suppression_reason": reason,
        "show_by_default": False,
    }


def group_consecutive(rep_ids: Iterable[int]) -> list[list[int]]:
    groups = []
    for rep_id in sorted(set(rep_ids)):
        if not groups or rep_id != groups[-1][-1] + 1:
            groups.append([rep_id])
        else:
            groups[-1].append(rep_id)
    return groups


def feature_candidate(
    feature: dict[str, Any],
    threshold_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    severity = feature["severity_candidate"]
    if severity not in SEVERITY_RANK or SEVERITY_RANK[severity] == 0:
        return None
    direction = threshold_cfg.get("direction", "high_bad")
    threshold = float(threshold_cfg[severity])
    return {
        "feature": feature,
        "severity": severity,
        "severity_rank": SEVERITY_RANK[severity],
        "threshold": threshold,
        "deviation_ratio": deviation_ratio(feature.get("raw_value"), threshold, direction),
        "confidence": feature.get("confidence") or 0.0,
    }


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            item["severity_rank"],
            item["deviation_ratio"],
            item["confidence"],
        ),
    )


def build_rep_issue(
    data: dict[str, Any],
    issue_code: str,
    rep_ids: list[int],
    candidate: dict[str, Any],
    issue_cfg: dict[str, Any],
    exercise_spec: dict[str, Any],
    templates: dict[str, Any],
    threshold_cfg: dict[str, Any],
) -> dict[str, Any]:
    feature = candidate["feature"]
    rep = next(rep for rep in data["reps"] if rep["rep_id"] == feature["rep_id"])
    anchor_frame = feature["anchor_frame"]
    radius = int(templates["evidence"]["radius_frames"])
    evidence_start = max(rep["start_frame"], anchor_frame - radius)
    evidence_end = min(rep["end_frame"], anchor_frame + radius)
    context = text_context(
        data,
        exercise_spec,
        templates,
        feature["metric"],
        feature["raw_value"],
        candidate["threshold"],
        feature["rep_id"],
        rep_ids,
    )
    return {
        "issue_id": "",
        "issue_code": issue_code,
        "exercise": data["analysis"]["exercise"],
        "scope": "rep",
        "rep_ids": rep_ids,
        "representative_rep_id": feature["rep_id"],
        "frequency": f"{len(rep_ids)}/{len(data['reps'])}",
        "frequency_ratio": round(len(rep_ids) / len(data["reps"]), 6) if data["reps"] else 0.0,
        "severity": candidate["severity"],
        "confidence": feature.get("confidence"),
        "anchor_frame": anchor_frame,
        "anchor_time_s": feature.get("anchor_time_s"),
        "evidence_interval": [evidence_start, evidence_end],
        "measurements": {
            "primary_metric": feature["metric"],
            "primary_value": feature.get("raw_value"),
            "unit": feature.get("unit"),
            "thresholds": threshold_cfg,
            "representative_source": feature.get("source"),
            "deviation_ratio": round(candidate["deviation_ratio"], 6),
        },
        "visual_overlay": visual_overlay(data, issue_cfg),
        "text": {
            "title": issue_cfg.get("title", issue_code),
            "detail": safe_format(issue_cfg.get("detail", ""), context),
            "cue": issue_cfg.get("cue", ""),
        },
        "suppressed": False,
        "suppression_reason": None,
        "show_by_default": False,
    }


def build_tempo_issue(
    data: dict[str, Any],
    issue_code: str,
    feature: dict[str, Any],
    issue_cfg: dict[str, Any],
    exercise_spec: dict[str, Any],
    templates: dict[str, Any],
) -> dict[str, Any]:
    durations = [rep["duration_s"] for rep in data["reps"] if rep.get("duration_s") is not None]
    mean_duration = statistics.mean(durations)
    representative = max(
        data["reps"],
        key=lambda rep: abs((rep.get("duration_s") or 0.0) - mean_duration),
    )
    rep_ids = [rep["rep_id"] for rep in data["reps"]]
    anchor_frame = (representative["start_frame"] + representative["end_frame"]) // 2
    threshold_cfg = templates["thresholds"]["rep_duration_cv"]
    threshold = float(threshold_cfg[feature["severity_candidate"]])
    context = text_context(
        data,
        exercise_spec,
        templates,
        "rep_duration_cv",
        feature["raw_value"],
        threshold,
        representative["rep_id"],
        rep_ids,
    )
    return {
        "issue_id": "",
        "issue_code": issue_code,
        "exercise": data["analysis"]["exercise"],
        "scope": "clip",
        "rep_ids": rep_ids,
        "representative_rep_id": representative["rep_id"],
        "frequency": f"{len(rep_ids)}/{len(rep_ids)}",
        "frequency_ratio": 1.0,
        "severity": feature["severity_candidate"],
        "confidence": feature.get("confidence"),
        "anchor_frame": anchor_frame,
        "anchor_time_s": source_time_s(anchor_frame, data["fps"]),
        "evidence_interval": [representative["start_frame"], representative["end_frame"]],
        "measurements": {
            "primary_metric": "rep_duration_cv",
            "primary_value": feature.get("raw_value"),
            "unit": "ratio",
            "thresholds": threshold_cfg,
            "rep_durations_s": durations,
            "representative_duration_s": representative.get("duration_s"),
        },
        "visual_overlay": visual_overlay(data, issue_cfg),
        "text": {
            "title": issue_cfg.get("title", "tempo"),
            "detail": safe_format(issue_cfg.get("detail", ""), context),
            "cue": issue_cfg.get("cue", ""),
        },
        "suppressed": False,
        "suppression_reason": None,
        "show_by_default": False,
    }


def build_issue_output(
    data: dict[str, Any],
    features: list[dict[str, Any]],
    specs: dict[str, Any],
    templates: dict[str, Any],
) -> dict[str, Any]:
    analysis = data["analysis"]
    exercise = analysis["exercise"]
    exercise_spec = specs["exercises"][exercise]
    issue_cfgs = exercise_spec["issues"]
    thresholds = templates["thresholds"]
    rep_features = {
        (feature["rep_id"], feature["metric"]): feature
        for feature in features
        if feature["scope"] == "rep"
    }
    group_features = {
        feature["metric"]: feature for feature in features if feature["scope"] == "clip"
    }
    issues = []
    evaluations = []
    quality_failed = analysis.get("quality_status") == "reject"

    for issue_code, metric_names in issue_cfgs.items():
        issue_cfg = templates["issues"][issue_code]
        statuses = metric_statuses(metric_names, rep_features, group_features)
        unique_statuses = flattened_statuses(statuses)
        if quality_failed:
            status = "suppressed"
            reason = "quality_gate_failed"
            evaluations.append(
                {
                    "issue_code": issue_code,
                    "status": status,
                    "reason": reason,
                    "metric_statuses": statuses,
                }
            )
            issues.append(
                suppressed_issue(
                    data,
                    issue_code,
                    reason,
                    issue_cfg,
                    exercise_spec,
                    templates,
                    metric_names,
                    statuses,
                )
            )
            continue

        if metric_names == ["rep_duration_cv"]:
            feature = group_features.get("rep_duration_cv")
            if feature is None:
                unique_statuses = {"unavailable"}
            if unique_statuses == {"disabled"}:
                reason = "view_unsupported"
            elif "insufficient_reps" in unique_statuses:
                reason = "insufficient_reps"
            elif unique_statuses <= {"unavailable", "unusable_low_confidence"}:
                reason = "low_confidence" if "unusable_low_confidence" in unique_statuses else "metric_unavailable"
            else:
                reason = None
            if reason is not None:
                evaluations.append(
                    {
                        "issue_code": issue_code,
                        "status": "suppressed",
                        "reason": reason,
                        "metric_statuses": statuses,
                    }
                )
                if reason != "low_confidence":
                    issues.append(
                        suppressed_issue(
                            data,
                            issue_code,
                            reason,
                            issue_cfg,
                            exercise_spec,
                            templates,
                            metric_names,
                            statuses,
                        )
                    )
                continue
            candidate = feature_candidate(feature, thresholds["rep_duration_cv"])
            if candidate is None:
                evaluations.append(
                    {
                        "issue_code": issue_code,
                        "status": "no_issue",
                        "reason": "below_threshold",
                        "metric_statuses": statuses,
                    }
                )
                continue
            issue = build_tempo_issue(
                data, issue_code, feature, issue_cfg, exercise_spec, templates
            )
            issues.append(issue)
            evaluations.append(
                {
                    "issue_code": issue_code,
                    "status": "triggered",
                    "reason": "threshold_exceeded",
                    "metric_statuses": statuses,
                    "triggered_rep_ids": issue["rep_ids"],
                }
            )
            continue

        active_metrics = [
            metric
            for metric in metric_names
            if any(
                rep_features.get((rep["rep_id"], metric), {}).get("status") in ACTIVE_FEATURE_STATUSES
                for rep in data["reps"]
            )
        ]
        if not active_metrics:
            if unique_statuses == {"disabled"}:
                reason = "view_unsupported"
            elif "unusable_low_confidence" in unique_statuses:
                reason = "low_confidence"
            else:
                reason = "metric_unavailable"
            evaluations.append(
                {
                    "issue_code": issue_code,
                    "status": "suppressed",
                    "reason": reason,
                    "metric_statuses": statuses,
                }
            )
            if reason != "low_confidence":
                issues.append(
                    suppressed_issue(
                        data,
                        issue_code,
                        reason,
                        issue_cfg,
                        exercise_spec,
                        templates,
                        metric_names,
                        statuses,
                    )
                )
            continue

        best_by_rep = {}
        for rep in data["reps"]:
            candidates = []
            for metric in active_metrics:
                feature = rep_features.get((rep["rep_id"], metric))
                threshold_cfg = thresholds.get(metric)
                if feature is None or not threshold_cfg:
                    continue
                candidate = feature_candidate(feature, threshold_cfg)
                if candidate is not None:
                    candidates.append(candidate)
            if candidates:
                best_by_rep[rep["rep_id"]] = choose_candidate(candidates)

        if not best_by_rep:
            evaluations.append(
                {
                    "issue_code": issue_code,
                    "status": "no_issue",
                    "reason": "below_threshold",
                    "metric_statuses": statuses,
                }
            )
            continue

        groups = group_consecutive(best_by_rep)
        for rep_ids in groups:
            candidate = choose_candidate([best_by_rep[rep_id] for rep_id in rep_ids])
            metric = candidate["feature"]["metric"]
            issue = build_rep_issue(
                data,
                issue_code,
                rep_ids,
                candidate,
                issue_cfg,
                exercise_spec,
                templates,
                thresholds[metric],
            )
            issues.append(issue)
        evaluations.append(
            {
                "issue_code": issue_code,
                "status": "triggered",
                "reason": "threshold_exceeded",
                "metric_statuses": statuses,
                "triggered_rep_ids": sorted(best_by_rep),
            }
        )

    active_issues = sorted(
        [issue for issue in issues if not issue["suppressed"]],
        key=lambda issue: (
            -SEVERITY_RANK[issue["severity"]],
            -issue["frequency_ratio"],
            -(issue["confidence"] or 0.0),
            issue["issue_code"],
        ),
    )
    for issue in active_issues[:3]:
        issue["show_by_default"] = True
    suppressed = [issue for issue in issues if issue["suppressed"]]
    ordered = active_issues + sorted(suppressed, key=lambda issue: issue["issue_code"])
    counters = defaultdict(int)
    for issue in ordered:
        counters[issue["issue_code"]] += 1
        suffix = "suppressed" if issue["suppressed"] else f"{counters[issue['issue_code']]:02d}"
        issue["issue_id"] = f"{issue['issue_code']}_{suffix}"

    severity_counts = Counter(
        issue["severity"] for issue in ordered if not issue["suppressed"]
    )
    return {
        "schema_version": ISSUE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "task": "task1-4c",
        "clip_id": analysis["clip_id"],
        "exercise": exercise,
        "view": analysis.get("view"),
        "quality_status": analysis.get("quality_status"),
        "rep_count": len(data["reps"]),
        "threshold_policy": templates.get("threshold_policy"),
        "evaluations": evaluations,
        "issues": ordered,
        "summary": {
            "active_issue_count": len(active_issues),
            "suppressed_issue_count": len(suppressed),
            "default_issue_count": min(3, len(active_issues)),
            "severity_counts": {
                severity: severity_counts.get(severity, 0)
                for severity in ("high", "medium", "low")
            },
        },
    }


def write_features_csv(path: Path, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for feature in features:
            writer.writerow({field: csv_value(feature.get(field)) for field in FEATURE_FIELDS})


def select_clip_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = []
    requested = set(args.clip_id or [])
    for clip_dir in sorted(args.analysis_dir.iterdir()):
        analysis_path = clip_dir / "analysis.json"
        if not clip_dir.is_dir() or not analysis_path.exists():
            continue
        analysis = read_json(analysis_path)
        clip_id = analysis.get("clip_id", clip_dir.name)
        if requested and clip_id not in requested:
            continue
        if args.split != "all" and analysis.get("split") != args.split:
            continue
        dirs.append(clip_dir)
    if requested:
        found = {read_json(path / "analysis.json").get("clip_id", path.name) for path in dirs}
        missing = sorted(requested - found)
        if missing:
            raise ValueError("unknown or filtered clip_id: " + ", ".join(missing))
    return dirs


def main() -> None:
    args = parse_args()
    specs = read_json(args.specs)
    templates = read_json(args.templates)
    clip_dirs = select_clip_dirs(args)
    if not clip_dirs:
        raise ValueError("no clip analysis directories matched")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_features = []
    results = []
    for clip_dir in clip_dirs:
        data = load_clip_data(clip_dir)
        features = compute_features(data, specs, templates)
        issue_output = build_issue_output(data, features, specs, templates)
        clip_output_dir = args.output_dir / data["analysis"]["clip_id"]
        write_features_csv(clip_output_dir / "features.csv", features)
        write_json(clip_output_dir / "issues.json", issue_output)
        all_features.extend(features)
        results.append(
            {
                "clip_id": data["analysis"]["clip_id"],
                "exercise": data["analysis"]["exercise"],
                "view": data["analysis"].get("view"),
                "split": data["analysis"].get("split"),
                "quality_status": data["analysis"].get("quality_status"),
                "rep_count": len(data["reps"]),
                "feature_count": len(features),
                "active_issue_count": issue_output["summary"]["active_issue_count"],
                "suppressed_issue_count": issue_output["summary"]["suppressed_issue_count"],
                "severity_counts": issue_output["summary"]["severity_counts"],
                "outputs": {
                    "features": str((clip_output_dir / "features.csv").resolve()),
                    "issues": str((clip_output_dir / "issues.json").resolve()),
                },
            }
        )
        print(
            f"{data['analysis']['clip_id']}: {len(features)} features, "
            f"{issue_output['summary']['active_issue_count']} active issues",
            flush=True,
        )

    write_features_csv(args.output_dir / "features_all.csv", all_features)
    all_issues = []
    for result in results:
        payload = read_json(Path(result["outputs"]["issues"]))
        all_issues.extend(
            {"clip_id": result["clip_id"], **issue}
            for issue in payload["issues"]
            if not issue["suppressed"]
        )
    severity_counts = Counter(issue["severity"] for issue in all_issues)
    issue_code_counts = Counter(issue["issue_code"] for issue in all_issues)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "task": "task1-4c",
        "threshold_policy": templates.get("threshold_policy"),
        "provisional_thresholds": True,
        "analysis_dir": str(args.analysis_dir.resolve()),
        "clip_count": len(results),
        "total_rep_count": sum(result["rep_count"] for result in results),
        "active_issue_count": len(all_issues),
        "suppressed_issue_count": sum(result["suppressed_issue_count"] for result in results),
        "severity_counts": {
            severity: severity_counts.get(severity, 0)
            for severity in ("high", "medium", "low")
        },
        "issue_code_counts": dict(sorted(issue_code_counts.items())),
        "results": results,
        "outputs": {
            "features_all": str((args.output_dir / "features_all.csv").resolve()),
            "thresholds_used": str((args.output_dir / "thresholds_used.json").resolve()),
        },
    }
    write_json(args.output_dir / "issues_summary.json", summary)
    write_json(
        args.output_dir / "thresholds_used.json",
        {
            "schema_version": templates["schema_version"],
            "source": str(args.templates.resolve()),
            "threshold_policy": templates.get("threshold_policy"),
            "disclaimer": templates.get("disclaimer"),
            "confidence": templates["confidence"],
            "evidence": templates["evidence"],
            "thresholds": templates["thresholds"],
        },
    )
    print(
        f"wrote {len(results)} clips, {len(all_features)} feature rows, "
        f"{len(all_issues)} active issues to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
