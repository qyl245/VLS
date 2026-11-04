#!/usr/bin/env python3
"""Build phase-aware measurements without making issue decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TAXONOMY = ROOT / "configs" / "issue_taxonomy.json"
DEFAULT_CONTRACTS = ROOT / "configs" / "metric_contracts.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "assessment" / "input_analysis"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "assessment" / "measurement"

FRAME_BASIS = "clip_local"
SOURCE_FRAME_BASIS = "source_video_absolute"
ACTIVE_STATUSES = {"ok", "low_confidence"}
IMPLEMENTED_REP_METRICS = {
    "rep_duration_s",
    "spine_stability_bottom",
    "knee_flexion_rom",
    "bottom_knee_angle",
    "hip_flexion_rom",
    "torso_lean_bottom",
    "hip_depth_ratio",
    "knee_valgus_bottom",
    "knee_asymmetry_bottom",
    "hip_asymmetry_bottom",
    "elbow_flexion_rom",
    "bottom_elbow_angle",
    "body_line_angle",
    "hip_sag_ratio",
    "elbow_asymmetry_bottom",
    "shoulder_asymmetry_bottom",
}
IMPLEMENTED_CLIP_METRICS = {"rep_duration_cv"}
FEATURE_FIELDS = [
    "clip_id",
    "subject_id",
    "exercise",
    "view",
    "split",
    "measurement_id",
    "scope",
    "rep_id",
    "metric",
    "phase",
    "raw_value",
    "normalized_value",
    "unit",
    "view_requirement",
    "source",
    "status",
    "confidence",
    "signal_confidence",
    "rep_boundary_confidence",
    "view_confidence",
    "quality_confidence",
    "metric_scale",
    "anchor_frame",
    "anchor_source_frame",
    "anchor_time_s",
    "window_start_frame",
    "window_end_frame",
    "window_center_frame",
    "aggregation",
    "quality_flags",
    "reason",
    "source_signals",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-id", nargs="+", default=None)
    parser.add_argument(
        "--split",
        nargs="+",
        default=["dev", "calibration"],
        choices=["dev", "calibration", "holdout"],
        help="Benchmark splits to measure; holdout is excluded by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def parse_flags(value: str | None) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
            return [str(item) for item in payload] if isinstance(payload, list) else [text]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.replace("|", ";").split(";") if item.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_contracts(
    contracts: dict[str, Any], taxonomy: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    metric_names = set(contracts.get("metrics") or {})
    for metrics in (contracts.get("exercise_metrics") or {}).values():
        metric_names.update(metrics)
    families = contracts.get("view_families") or {}
    for action, issue_map in (contracts.get("issue_metrics") or {}).items():
        for issue, metrics in issue_map.items():
            if issue not in (taxonomy.get("issues") or {}):
                errors.append(f"unknown taxonomy issue in contract: {issue}")
            for metric in metrics:
                if metric not in metric_names:
                    errors.append(f"{action}.{issue} references unknown metric {metric}")
    for issue, issue_cfg in (taxonomy.get("issues") or {}).items():
        action = issue_cfg["action"]
        contract_metrics = (contracts.get("issue_metrics") or {}).get(action, {})
        issue_metrics = contract_metrics.get(issue) or []
        if issue not in contract_metrics:
            errors.append(f"taxonomy issue not expressible in contract: {issue}")
        for metric in issue_cfg.get("metrics") or []:
            if metric not in metric_names:
                errors.append(f"{issue} taxonomy metric not measured: {metric}")
            if metric not in issue_metrics:
                errors.append(f"{issue} taxonomy metric missing from issue contract: {metric}")
        views = issue_cfg.get("views") or []
        if views == ["any"]:
            if not issue_metrics:
                errors.append(f"{issue} has no metric for any-view taxonomy entry")
        else:
            for view in views:
                if not any(
                    metric_supported(all_metrics(contracts)[metric], view, contracts)
                    for metric in issue_metrics
                ):
                    errors.append(
                        f"{issue} has no metric available from taxonomy view {view}"
                    )
    for metric, metric_cfg in all_metrics(contracts).items():
        family = metric_cfg.get("view_family")
        if family not in families:
            errors.append(f"metric {metric} has unknown view family: {family}")
        if metric_cfg.get("scope") not in {"rep", "clip"}:
            errors.append(f"metric {metric} has invalid scope")
        if not metric_cfg.get("source_signals"):
            errors.append(f"metric {metric} has no source signals")
        implemented = (
            IMPLEMENTED_REP_METRICS
            if metric_cfg.get("scope") == "rep"
            else IMPLEMENTED_CLIP_METRICS
        )
        if metric not in implemented:
            errors.append(f"metric {metric} has no measurement implementation")
    if not bool((contracts.get("clip_policy") or {}).get("quality_reject_abstains")):
        errors.append("quality reject policy must abstain")
    if len((contracts.get("issue_metrics") or {}).get("squat", {})) != 6:
        errors.append("squat issue contract must contain 6 issues")
    if len((contracts.get("issue_metrics") or {}).get("pushup", {})) != 6:
        errors.append("pushup issue contract must contain 6 issues")
    return errors


def all_metrics(contracts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {name: dict(cfg) for name, cfg in (contracts.get("metrics") or {}).items()}
    for metrics in (contracts.get("exercise_metrics") or {}).values():
        output.update({name: dict(cfg) for name, cfg in metrics.items()})
    return output


def exercise_metrics(contracts: dict[str, Any], exercise: str) -> dict[str, dict[str, Any]]:
    output = {name: dict(cfg) for name, cfg in (contracts.get("metrics") or {}).items()}
    output.update(
        {
            name: dict(cfg)
            for name, cfg in (contracts.get("exercise_metrics") or {}).get(exercise, {}).items()
        }
    )
    return output


def near_side(contracts: dict[str, Any], view: str) -> str | None:
    return (contracts.get("near_side_by_view") or {}).get(view)


def resolve_signals(
    metric_cfg: dict[str, Any], side: str | None
) -> list[str]:
    return [
        str(signal).replace("{side}", side or "unknown")
        for signal in metric_cfg.get("source_signals") or []
    ]


def metric_supported(
    metric_cfg: dict[str, Any], view: str, contracts: dict[str, Any]
) -> bool:
    family = str(metric_cfg.get("view_family"))
    return view in ((contracts.get("view_families") or {}).get(family) or [])


def capability_for(
    contracts: dict[str, Any], exercise: str, view: str
) -> dict[str, Any]:
    metrics = exercise_metrics(contracts, exercise)
    usable = sorted(
        metric for metric, cfg in metrics.items() if metric_supported(cfg, view, contracts)
    )
    disabled = sorted(
        metric for metric, cfg in metrics.items() if not metric_supported(cfg, view, contracts)
    )
    reasons = {
        metric: f"view {view!r} is outside view family {metrics[metric]['view_family']!r}"
        for metric in disabled
    }
    return {
        "view": view,
        "usable_metrics": usable,
        "disabled_metrics": disabled,
        "reasons": reasons,
    }


def quality_factor(contracts: dict[str, Any], quality_status: str) -> float:
    factors = (contracts.get("confidence") or {}).get("quality_factors") or {}
    return float(factors.get(quality_status, factors.get("unknown", 0.8)))


def view_factor(contracts: dict[str, Any], view: str) -> float:
    return float((contracts.get("view_confidence_factors") or {}).get(view, 0.8))


def combine_confidence(
    signal_confidence: float | None,
    rep_confidence: float | None,
    metric_view_factor: float,
    metric_quality_factor: float,
    metric_scale: float,
) -> float | None:
    if signal_confidence is None or rep_confidence is None:
        return None
    value = (
        max(0.0, min(1.0, signal_confidence))
        * max(0.0, min(1.0, rep_confidence))
        * metric_view_factor
        * metric_quality_factor
        * metric_scale
    )
    return rounded(max(0.0, min(1.0, value)))


def load_clip_data(
    clip: dict[str, Any], analysis_dir: Path
) -> dict[str, Any]:
    clip_dir = analysis_dir / clip["clip_id"]
    analysis_path = clip_dir / "analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"missing analysis output: {clip_dir}")
    analysis = load_json(analysis_path)
    if analysis.get("clip_id") != clip["clip_id"]:
        raise ValueError(f"clip id mismatch in {analysis_path}")
    if analysis.get("exercise") != clip["exercise"] or analysis.get("view") != clip["view"]:
        raise ValueError(f"exercise/view mismatch in {analysis_path}")

    trim_start = int((analysis.get("trim") or {}).get("start_frame", 0))
    fps = float(analysis.get("fps") or clip.get("fps") or 0.0)
    reps = []
    for row in read_csv(clip_dir / "reps.csv"):
        start_source = to_int(row.get("start_frame"))
        end_source = to_int(row.get("end_frame"))
        bottom_source = to_int(row.get("bottom_frame"))
        top_source = to_int(row.get("top_frame"))
        if start_source is None or end_source is None or bottom_source is None:
            continue
        reps.append(
            {
                "rep_id": int(row["rep_id"]),
                "start_frame": max(0, start_source - trim_start),
                "end_frame": max(0, end_source - trim_start),
                "bottom_frame": max(0, bottom_source - trim_start),
                "top_frame": max(0, (top_source if top_source is not None else start_source) - trim_start),
                "start_source_frame": start_source,
                "end_source_frame": end_source,
                "bottom_source_frame": bottom_source,
                "duration_s": to_float(row.get("duration_s")),
                "boundary_confidence": to_float(row.get("confidence")),
                "quality_flags": parse_flags(row.get("quality_flags")),
                "status": row.get("status") or "complete",
            }
        )
    reps.sort(key=lambda item: item["rep_id"])

    wide: dict[int, dict[str, str]] = {}
    for row in read_csv(clip_dir / "angles" / "angles_wide.csv"):
        frame = to_int(row.get("frame_id"))
        if frame is not None:
            wide[frame] = row

    angle_confidence: dict[tuple[int, str], float] = {}
    for row in read_csv(clip_dir / "angles" / "angles.csv"):
        frame = to_int(row.get("frame_id"))
        name = row.get("angle_name")
        confidence = to_float(row.get("confidence"))
        if frame is not None and name and row.get("valid") == "1" and confidence is not None:
            angle_confidence[(frame, name)] = confidence

    world_frames: dict[int, dict[str, Any]] = {}
    for record in load_json(clip_dir / "angles" / "landmarks.json"):
        frame = to_int(record.get("frame_id"))
        if frame is not None:
            world_frames[frame] = record
    body_frames: dict[int, list[list[float]]] = {}
    for record in load_json(clip_dir / "angles" / "landmarks_body_frame.json"):
        frame = to_int(record.get("frame_id"))
        points = record.get("landmarks_body_frame")
        if frame is not None and points:
            body_frames[frame] = points

    return {
        "clip_dir": clip_dir,
        "analysis": analysis,
        "trim_start": trim_start,
        "fps": fps,
        "frame_count": len(wide),
        "reps": reps,
        "wide": wide,
        "angle_confidence": angle_confidence,
        "world_frames": world_frames,
        "body_frames": body_frames,
    }


def signal_value(data: dict[str, Any], frame: int, name: str) -> float | None:
    row = data["wide"].get(frame)
    return to_float(row.get(name)) if row else None


def signal_confidence(data: dict[str, Any], frame: int, name: str) -> float | None:
    return data["angle_confidence"].get((frame, name))


def min_signal_confidence(
    data: dict[str, Any], frame: int, names: list[str]
) -> float | None:
    values = [signal_confidence(data, frame, name) for name in names]
    if any(value is None for value in values):
        return None
    return min(float(value) for value in values if value is not None)


def visibility_confidence(
    data: dict[str, Any], frame: int, indices: list[int]
) -> float | None:
    record = data["world_frames"].get(frame)
    visibility = (record or {}).get("visibility") or []
    if not visibility or len(visibility) <= max(indices):
        return None
    values = [to_float(visibility[index]) for index in indices]
    if any(value is None for value in values):
        return None
    return min(float(value) for value in values if value is not None)


def bounded_window(
    data: dict[str, Any], center: int, radius: int
) -> tuple[int, int]:
    start = max(0, center - radius)
    end = min(max(0, data["frame_count"] - 1), center + radius)
    return start, end


def series(
    data: dict[str, Any],
    start: int,
    end: int,
    names: list[str],
    transform: Callable[[list[float]], float | None] | None = None,
) -> list[tuple[int, float, float | None]]:
    output = []
    for frame in range(start, end + 1):
        values = [signal_value(data, frame, name) for name in names]
        if any(value is None for value in values):
            continue
        numeric = [float(value) for value in values if value is not None]
        value = transform(numeric) if transform else numeric[0]
        if value is None or not math.isfinite(value):
            continue
        output.append((frame, float(value), min_signal_confidence(data, frame, names)))
    return output


def aggregate_series(
    values: list[tuple[int, float, float | None]],
    aggregation: str,
    anchor_rule: str,
) -> tuple[float | None, int | None, float | None]:
    if not values:
        return None, None, None
    frames = [item[0] for item in values]
    numeric = np.asarray([item[1] for item in values], dtype=np.float64)
    confidences = finite(item[2] for item in values)
    signal_conf = float(np.median(confidences)) if confidences else None
    if aggregation == "median":
        value = float(np.median(numeric))
    elif aggregation == "p90":
        value = float(np.percentile(numeric, 90))
    elif aggregation == "p10":
        value = float(np.percentile(numeric, 10))
    elif aggregation == "max":
        value = float(np.max(numeric))
    elif aggregation == "min":
        value = float(np.min(numeric))
    elif aggregation == "p95_minus_p5":
        value = float(np.percentile(numeric, 95) - np.percentile(numeric, 5))
    else:
        raise ValueError(f"unsupported aggregation: {aggregation}")

    if anchor_rule in {"max", "max_flexion"}:
        anchor = frames[int(np.argmax(numeric))]
    elif anchor_rule == "min":
        anchor = frames[int(np.argmin(numeric))]
    elif anchor_rule == "nearest_median":
        anchor = frames[int(np.argmin(np.abs(numeric - np.median(numeric))))]
    else:
        anchor = frames[len(frames) // 2]
    return value, int(anchor), signal_conf


def inward_knee_valgus_deg(points: list[list[float]], side: str) -> float | None:
    hip_index, knee_index, ankle_index = (23, 25, 27) if side == "left" else (24, 26, 28)
    hip = np.asarray(points[hip_index], dtype=np.float64)
    knee = np.asarray(points[knee_index], dtype=np.float64)
    ankle = np.asarray(points[ankle_index], dtype=np.float64)
    dx = ankle[0] - hip[0]
    dy = ankle[1] - hip[1]
    leg_length = math.hypot(dx, dy)
    if leg_length < 1e-8:
        return None
    lateral = (dx * (knee[1] - hip[1]) - dy * (knee[0] - hip[0])) / leg_length
    medial = lateral if side == "left" else -lateral
    return math.degrees(math.atan2(max(0.0, medial), leg_length))


def world_midpoint(points: list[list[float]], first: int, second: int) -> np.ndarray:
    return (np.asarray(points[first], dtype=np.float64) + np.asarray(points[second], dtype=np.float64)) / 2.0


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-10:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def body_line_geometry(points: list[list[float]]) -> tuple[float | None, float | None]:
    shoulder = world_midpoint(points, 11, 12)
    hip = world_midpoint(points, 23, 24)
    ankle = world_midpoint(points, 27, 28)
    angle = vector_angle_deg(shoulder - hip, ankle - hip)
    torso_length = float(np.linalg.norm(shoulder - hip))
    line = ankle - shoulder
    line_length = float(np.linalg.norm(line))
    if torso_length < 1e-8 or line_length < 1e-8:
        return angle, None
    distance = float(np.linalg.norm(np.cross(hip - shoulder, line)) / line_length)
    return angle, distance / torso_length


def observation_payload(
    data: dict[str, Any],
    *,
    metric: str,
    metric_cfg: dict[str, Any],
    rep: dict[str, Any] | None,
    raw_value: float | None,
    anchor_frame: int | None,
    window_start: int | None,
    window_end: int | None,
    signal_conf: float | None,
    status: str,
    reason: str | None,
    source_file: str,
    signals: list[str],
    view_factor_value: float,
    quality_factor_value: float,
    metric_scale: float,
    quality_flags: list[str],
    rep_boundary_override: float | None = None,
) -> dict[str, Any]:
    rep_id = int(rep["rep_id"]) if rep is not None else None
    boundary_conf = (
        rep.get("boundary_confidence")
        if rep is not None
        else rep_boundary_override
    )
    confidence = combine_confidence(
        signal_conf,
        boundary_conf,
        view_factor_value,
        quality_factor_value,
        metric_scale,
    )
    if confidence is not None:
        low_below = float(((data.get("_contracts") or {}).get("confidence") or {}).get("low_below", 0.55))
        if confidence < low_below and status == "ok":
            status = "low_confidence"
    anchor_source = (
        data["trim_start"] + anchor_frame if anchor_frame is not None else None
    )
    anchor_time = (
        round(anchor_frame / data["fps"], 6)
        if anchor_frame is not None and data["fps"] > 0
        else None
    )
    scope = str(metric_cfg.get("scope"))
    return {
        "measurement_id": (
            f"rep:{rep_id}:{metric}" if scope == "rep" else f"clip:{metric}"
        ),
        "metric": metric,
        "scope": scope,
        "rep_id": rep_id,
        "phase": str(metric_cfg.get("phase")),
        "raw_value": rounded(raw_value),
        "normalized_value": rounded(raw_value),
        "unit": str(metric_cfg.get("unit", "")),
        "view_requirement": str(metric_cfg.get("view_family")),
        "source": str(metric_cfg.get("source")),
        "status": status,
        "confidence": confidence,
        "confidence_components": {
            "signal": rounded(signal_conf),
            "rep_boundary": rounded(boundary_conf),
            "view": rounded(view_factor_value),
            "quality": rounded(quality_factor_value),
            "metric_scale": rounded(metric_scale),
        },
        "quality_flags": sorted(set(quality_flags)),
        "reason": reason,
        "anchor_frame": anchor_frame,
        "anchor_source_frame": anchor_source,
        "anchor_time_s": anchor_time,
        "window": {
            "start_frame": window_start,
            "end_frame": window_end,
            "center_frame": anchor_frame,
            "aggregation": str(metric_cfg.get("aggregation")),
        },
        "trace": {
            "source_signals": signals,
            "frame_basis": FRAME_BASIS,
            "anchor_rule": str(metric_cfg.get("anchor_rule")),
            "aggregation": str(metric_cfg.get("aggregation")),
            "source_file": source_file,
        },
    }


def disabled_observation(
    data: dict[str, Any],
    rep: dict[str, Any] | None,
    metric: str,
    metric_cfg: dict[str, Any],
    signals: list[str],
    reason: str,
) -> dict[str, Any]:
    return observation_payload(
        data,
        metric=metric,
        metric_cfg=metric_cfg,
        rep=rep,
        raw_value=None,
        anchor_frame=None,
        window_start=None,
        window_end=None,
        signal_conf=None,
        status="disabled",
        reason=reason,
        source_file="not_evaluated",
        signals=signals,
        view_factor_value=0.0,
        quality_factor_value=quality_factor(
            data["_contracts"], data["analysis"].get("quality_status", "unknown")
        ),
        metric_scale=float(metric_cfg.get("confidence_scale", 1.0)),
        quality_flags=data["_quality_flags"],
    )


def unavailable_observation(
    data: dict[str, Any],
    rep: dict[str, Any] | None,
    metric: str,
    metric_cfg: dict[str, Any],
    signals: list[str],
    reason: str,
    status: str = "unavailable",
    window: tuple[int, int] | None = None,
    source_file: str = "unavailable",
) -> dict[str, Any]:
    return observation_payload(
        data,
        metric=metric,
        metric_cfg=metric_cfg,
        rep=rep,
        raw_value=None,
        anchor_frame=None,
        window_start=window[0] if window else None,
        window_end=window[1] if window else None,
        signal_conf=None,
        status=status,
        reason=reason,
        source_file=source_file,
        signals=signals,
        view_factor_value=view_factor(data["_contracts"], data["analysis"]["view"]),
        quality_factor_value=quality_factor(
            data["_contracts"], data["analysis"].get("quality_status", "unknown")
        ),
        metric_scale=float(metric_cfg.get("confidence_scale", 1.0)),
        quality_flags=data["_quality_flags"] + (rep.get("quality_flags") if rep else []),
    )


def compute_rep_observation(
    data: dict[str, Any],
    rep: dict[str, Any],
    metric: str,
    metric_cfg: dict[str, Any],
    contracts: dict[str, Any],
    side: str | None,
) -> dict[str, Any]:
    signals = resolve_signals(metric_cfg, side)
    view_factor_value = view_factor(contracts, data["analysis"]["view"])
    quality_factor_value = quality_factor(
        contracts, data["analysis"].get("quality_status", "unknown")
    )
    metric_scale = float(metric_cfg.get("confidence_scale", 1.0))
    flags = data["_quality_flags"] + rep.get("quality_flags", [])
    start = int(rep["start_frame"])
    end = int(rep["end_frame"])
    bottom = int(rep["bottom_frame"])
    top = int(rep.get("top_frame", start))
    radii = contracts.get("windows") or {}
    bottom_radius = int(radii.get("bottom_radius_frames", 8))
    stability_radius = int(radii.get("stability_radius_frames", 15))
    top_radius = int(radii.get("top_radius_frames", 3))

    def payload(
        value: float | None,
        anchor: int | None,
        window: tuple[int, int] | None,
        signal_conf: float | None,
        status: str = "ok",
        reason: str | None = None,
        source_file: str = "angles/angles_wide.csv",
    ) -> dict[str, Any]:
        return observation_payload(
            data,
            metric=metric,
            metric_cfg=metric_cfg,
            rep=rep,
            raw_value=value,
            anchor_frame=anchor,
            window_start=window[0] if window else None,
            window_end=window[1] if window else None,
            signal_conf=signal_conf,
            status=status,
            reason=reason,
            source_file=source_file,
            signals=signals,
            view_factor_value=view_factor_value,
            quality_factor_value=quality_factor_value,
            metric_scale=metric_scale,
            quality_flags=flags,
        )

    if metric == "rep_duration_s":
        anchor = (start + end) // 2
        return payload(
            rep.get("duration_s"),
            anchor,
            (start, end),
            rep.get("boundary_confidence"),
            source_file="reps.csv",
        )

    if metric == "spine_stability_bottom":
        window = bounded_window(data, bottom, stability_radius)
        values = series(data, *window, ["spine_stability"])
        value, anchor, signal_conf = aggregate_series(values, "p90", "max")
        if value is None:
            return payload(None, None, window, None, "unavailable", "no finite stability samples")
        return payload(value, anchor, window, signal_conf)

    if metric == "knee_flexion_rom":
        if side is None:
            return payload(None, None, (start, end), None, "unavailable", "near side unresolved")
        values = series(
            data,
            start,
            end,
            [f"{side}_knee_flexion"],
            transform=lambda values: 180.0 - values[0],
        )
        value, anchor, signal_conf = aggregate_series(values, "p95_minus_p5", "max_flexion")
        if value is None:
            return payload(None, None, (start, end), None, "unavailable", "no finite knee angle samples")
        return payload(value, anchor, (start, end), signal_conf)

    if metric == "bottom_knee_angle":
        if side is None:
            return payload(None, None, None, None, "unavailable", "near side unresolved")
        window = bounded_window(data, bottom, bottom_radius)
        values = series(data, *window, [f"{side}_knee_flexion"])
        value, anchor, signal_conf = aggregate_series(values, "median", "nearest_median")
        if value is None:
            return payload(None, None, window, None, "unavailable", "no finite knee angle samples")
        return payload(value, anchor, window, signal_conf)

    if metric == "hip_flexion_rom":
        if side is None:
            return payload(None, None, (start, end), None, "unavailable", "near side unresolved")
        values = series(data, start, end, [f"{side}_hip_flexion_sagittal"])
        value, anchor, signal_conf = aggregate_series(values, "p95_minus_p5", "max_flexion")
        if value is None:
            return payload(None, None, (start, end), None, "unavailable", "no finite hip angle samples")
        return payload(value, anchor, (start, end), signal_conf)

    if metric == "elbow_flexion_rom":
        if side is None:
            return payload(None, None, (start, end), None, "unavailable", "near side unresolved")
        values = series(
            data,
            start,
            end,
            [f"{side}_elbow_flexion"],
            transform=lambda values: 180.0 - values[0],
        )
        value, anchor, signal_conf = aggregate_series(values, "p95_minus_p5", "max_flexion")
        if value is None:
            return payload(None, None, (start, end), None, "unavailable", "no finite elbow angle samples")
        return payload(value, anchor, (start, end), signal_conf)

    if metric == "bottom_elbow_angle":
        if side is None:
            return payload(None, None, None, None, "unavailable", "near side unresolved")
        window = bounded_window(data, bottom, bottom_radius)
        values = series(data, *window, [f"{side}_elbow_flexion"])
        value, anchor, signal_conf = aggregate_series(values, "median", "nearest_median")
        if value is None:
            return payload(None, None, window, None, "unavailable", "no finite elbow angle samples")
        return payload(value, anchor, window, signal_conf)

    if metric == "torso_lean_bottom":
        window = bounded_window(data, bottom, bottom_radius)
        values = series(data, *window, ["torso_flexion"])
        value, anchor, signal_conf = aggregate_series(values, "median", "max")
        if value is None:
            return payload(None, None, window, None, "unavailable", "no finite torso angle samples")
        return payload(value, anchor, window, signal_conf)

    if metric == "hip_depth_ratio":
        top_window = bounded_window(data, top, top_radius)
        bottom_window_bounds = bounded_window(data, bottom, bottom_radius)
        heights = []
        confidences = []
        for frame in range(min(top_window[0], bottom_window_bounds[0]), max(top_window[1], bottom_window_bounds[1]) + 1):
            record = data["world_frames"].get(frame)
            points = (record or {}).get("landmarks_world")
            if not points:
                continue
            hip_y = (float(points[23][1]) + float(points[24][1])) / 2.0
            ankle_y = (float(points[27][1]) + float(points[28][1])) / 2.0
            confidence = visibility_confidence(data, frame, [23, 24, 27, 28])
            heights.append((frame, ankle_y - hip_y, confidence))
        top_values = [item for item in heights if top_window[0] <= item[0] <= top_window[1]]
        bottom_values = [item for item in heights if bottom_window_bounds[0] <= item[0] <= bottom_window_bounds[1]]
        if not top_values or not bottom_values:
            return payload(None, None, (top_window[0], bottom_window_bounds[1]), None, "unavailable", "hip/ankle world coordinates unavailable", "angles/landmarks.json")
        top_height = float(np.median([item[1] for item in top_values]))
        bottom_height = float(np.median([item[1] for item in bottom_values]))
        if top_height <= 0.05:
            return payload(None, None, (top_window[0], bottom_window_bounds[1]), None, "unavailable", "invalid top hip height", "angles/landmarks.json")
        value = (top_height - bottom_height) / top_height
        signal_conf = min(
            [item[2] for item in top_values + bottom_values if item[2] is not None]
            or [0.0]
        )
        return payload(value, bottom, (top_window[0], bottom_window_bounds[1]), signal_conf, source_file="angles/landmarks.json")

    if metric == "knee_valgus_bottom":
        window = bounded_window(data, bottom, bottom_radius)
        values = []
        for frame in range(window[0], window[1] + 1):
            points = data["body_frames"].get(frame)
            if not points:
                continue
            candidates = []
            for current_side in ("left", "right"):
                value = inward_knee_valgus_deg(points, current_side)
                confidence = signal_confidence(data, frame, f"{current_side}_knee_valgus_coronal")
                if value is not None:
                    candidates.append((value, confidence))
            if candidates:
                value, confidence = max(candidates, key=lambda item: item[0])
                values.append((frame, value, confidence))
        result, anchor, signal_conf = aggregate_series(values, "p90", "max")
        if result is None:
            return payload(None, None, window, None, "unavailable", "body-frame knee geometry unavailable", "angles/landmarks_body_frame.json")
        return payload(result, anchor, window, signal_conf, source_file="angles/landmarks_body_frame.json")

    if metric in {"knee_asymmetry_bottom", "hip_asymmetry_bottom"}:
        signal_name = (
            "knee_flexion_asymmetry" if metric.startswith("knee_") else "hip_flexion_asymmetry"
        )
        window = bounded_window(data, bottom, bottom_radius)
        values = series(data, *window, [signal_name])
        value, anchor, signal_conf = aggregate_series(values, "p90", "max")
        if value is None:
            return payload(None, None, window, None, "unavailable", f"no finite {signal_name} samples")
        return payload(value, anchor, window, signal_conf)

    if metric in {"elbow_asymmetry_bottom", "shoulder_asymmetry_bottom"}:
        prefix = "elbow" if metric.startswith("elbow_") else "shoulder"
        window = bounded_window(data, bottom, bottom_radius)
        values = series(
            data,
            *window,
            [f"left_{prefix}_flexion", f"right_{prefix}_flexion"],
            transform=lambda values: abs(values[0] - values[1]),
        )
        value, anchor, signal_conf = aggregate_series(values, "p90", "max")
        if value is None:
            return payload(None, None, window, None, "unavailable", f"no finite bilateral {prefix} samples")
        return payload(value, anchor, window, signal_conf)

    if metric in {"body_line_angle", "hip_sag_ratio"}:
        values = []
        for frame in range(start, end + 1):
            record = data["world_frames"].get(frame)
            points = (record or {}).get("landmarks_world")
            if not points:
                continue
            angle, ratio = body_line_geometry(points)
            raw_value = angle if metric == "body_line_angle" else ratio
            if raw_value is None:
                continue
            confidence = visibility_confidence(data, frame, [11, 12, 23, 24, 27, 28])
            values.append((frame, float(raw_value), confidence))
        if metric == "body_line_angle":
            value, anchor, signal_conf = aggregate_series(values, "p10", "min")
        else:
            window_values = [item for item in values if bottom - bottom_radius <= item[0] <= bottom + bottom_radius]
            value, anchor, signal_conf = aggregate_series(window_values, "p90", "max")
            window = (max(start, bottom - bottom_radius), min(end, bottom + bottom_radius))
            if value is not None:
                return payload(value, anchor, window, signal_conf, source_file="angles/landmarks.json")
            return payload(None, None, window, None, "unavailable", "no finite body-line geometry", "angles/landmarks.json")
        if value is None:
            return payload(None, None, (start, end), None, "unavailable", "no finite body-line geometry", "angles/landmarks.json")
        return payload(value, anchor, (start, end), signal_conf, source_file="angles/landmarks.json")

    return payload(None, None, None, None, "unavailable", f"unsupported metric implementation: {metric}")


def compute_clip_observation(
    data: dict[str, Any],
    metric: str,
    metric_cfg: dict[str, Any],
    contracts: dict[str, Any],
) -> dict[str, Any]:
    signals = resolve_signals(metric_cfg, None)
    view_factor_value = view_factor(contracts, data["analysis"]["view"])
    quality_factor_value = quality_factor(
        contracts, data["analysis"].get("quality_status", "unknown")
    )
    metric_scale = float(metric_cfg.get("confidence_scale", 1.0))
    min_reps = int(metric_cfg.get("min_reps", 3))
    if len(data["reps"]) < min_reps:
        return unavailable_observation(
            data,
            None,
            metric,
            metric_cfg,
            signals,
            f"requires at least {min_reps} complete reps",
            status="insufficient_reps",
            source_file="reps.csv",
        )
    durations = finite(rep.get("duration_s") for rep in data["reps"])
    if not durations or float(np.mean(durations)) <= 0:
        return unavailable_observation(
            data,
            None,
            metric,
            metric_cfg,
            signals,
            "rep duration unavailable",
            source_file="reps.csv",
        )
    value = float(np.std(durations, ddof=0) / np.mean(durations))
    mean_duration = float(np.mean(durations))
    anchor_rep = max(
        data["reps"],
        key=lambda rep: abs(float(rep.get("duration_s") or 0.0) - mean_duration),
    )
    anchor = (int(anchor_rep["start_frame"]) + int(anchor_rep["end_frame"])) // 2
    signal_conf_values = finite(rep.get("boundary_confidence") for rep in data["reps"])
    signal_conf = min(signal_conf_values) if signal_conf_values else None
    clip_boundary_component = float(
        (contracts.get("confidence") or {}).get(
            "clip_rep_boundary_component", 1.0
        )
    )
    return observation_payload(
        data,
        metric=metric,
        metric_cfg=metric_cfg,
        rep=None,
        raw_value=value,
        anchor_frame=anchor,
        window_start=min(int(rep["start_frame"]) for rep in data["reps"]),
        window_end=max(int(rep["end_frame"]) for rep in data["reps"]),
        signal_conf=signal_conf,
        status="ok",
        reason=None,
        source_file="reps.csv",
        signals=signals,
        view_factor_value=view_factor_value,
        quality_factor_value=quality_factor_value,
        metric_scale=metric_scale,
        quality_flags=data["_quality_flags"],
        rep_boundary_override=clip_boundary_component,
    )


def apply_quality_policy(
    observations: list[dict[str, Any]], quality_status: str
) -> None:
    if quality_status != "reject":
        return
    for observation in observations:
        if observation["status"] not in ACTIVE_STATUSES:
            continue
        observation["status"] = "unavailable"
        observation["raw_value"] = None
        observation["normalized_value"] = None
        observation["confidence"] = None
        observation["reason"] = "quality_reject"


def input_contract(
    data: dict[str, Any],
    clip: dict[str, Any],
    benchmark_path: Path,
    contracts_path: Path,
) -> dict[str, Any]:
    files = {}
    for name in (
        "analysis.json",
        "reps.csv",
        "angles/angles.csv",
        "angles/angles_wide.csv",
        "angles/landmarks.json",
        "angles/landmarks_body_frame.json",
    ):
        path = data["clip_dir"] / name
        files[name] = {"path": relative_path(path), "sha256": sha256_file(path)}
    analysis_schema = data["analysis"].get("schema_version")
    if analysis_schema == "vls-clip-analysis-v3":
        upstream_kind = "repchain-v3"
        segmentation_source = "repchain-v3-sets"
        is_repchain_v3 = True
    elif analysis_schema == "vls-clip-analysis-v1":
        upstream_kind = "clip-analysis-v1"
        segmentation_source = "legacy-reps-csv-frozen-fallback"
        is_repchain_v3 = False
    else:
        upstream_kind = "unknown"
        segmentation_source = "unknown"
        is_repchain_v3 = False
    return {
        "analysis_schema_version": analysis_schema,
        "analysis_status": data["analysis"].get("status"),
        "upstream_contract": {
            "kind": upstream_kind,
            "segmentation_source": segmentation_source,
            "is_repchain_v3": is_repchain_v3,
        },
        "frame_basis": FRAME_BASIS,
        "source_frame_basis": SOURCE_FRAME_BASIS,
        "trim": data["analysis"].get("trim"),
        "fps": data["fps"],
        "frame_count": data["frame_count"],
        "benchmark_path": relative_path(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "metric_contracts_path": relative_path(contracts_path),
        "metric_contracts_sha256": sha256_file(contracts_path),
        "analysis_files": files,
        "source_video": clip.get("source_path"),
        "source_video_sha256": clip.get("sha256"),
    }


def build_assessment(
    clip: dict[str, Any],
    data: dict[str, Any],
    contracts: dict[str, Any],
    benchmark_path: Path,
    contracts_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    view = str(clip["view"])
    exercise = str(clip["exercise"])
    capability = capability_for(contracts, exercise, view)
    side = near_side(contracts, view)
    metrics = exercise_metrics(contracts, exercise)
    quality = data["analysis"].get("quality") or {}
    quality_status = str(quality.get("status") or data["analysis"].get("quality_status") or "unknown")
    data["_contracts"] = contracts
    data["_quality_flags"] = [
        f"input_quality:{quality_status}",
        *[f"quality_warning:{item}" for item in quality.get("warnings") or []],
        *[f"quality_failure:{item}" for item in quality.get("failures") or []],
    ]

    rep_measurements = []
    clip_measurements = []
    for rep in data["reps"]:
        for metric, metric_cfg in metrics.items():
            if metric_cfg.get("scope") != "rep":
                continue
            signals = resolve_signals(metric_cfg, side)
            if not metric_supported(metric_cfg, view, contracts):
                rep_measurements.append(
                    disabled_observation(data, rep, metric, metric_cfg, signals, capability["reasons"][metric])
                )
                continue
            rep_measurements.append(
                compute_rep_observation(data, rep, metric, metric_cfg, contracts, side)
            )

    for metric, metric_cfg in metrics.items():
        if metric_cfg.get("scope") != "clip":
            continue
        signals = resolve_signals(metric_cfg, side)
        if not metric_supported(metric_cfg, view, contracts):
            clip_measurements.append(
                disabled_observation(data, None, metric, metric_cfg, signals, capability["reasons"][metric])
            )
            continue
        clip_measurements.append(
            compute_clip_observation(data, metric, metric_cfg, contracts)
        )

    observations = rep_measurements + clip_measurements
    apply_quality_policy(observations, quality_status)
    status_counts = Counter(item["status"] for item in observations)
    evaluable_statuses = ACTIVE_STATUSES
    evaluable_metrics = {
        item["metric"] for item in observations if item["status"] in evaluable_statuses
    }
    applicable = [
        item for item in observations if item["status"] not in {"disabled", "insufficient_reps"}
    ]
    evaluable_observations = [
        item for item in applicable if item["status"] in evaluable_statuses
    ]
    rep_count = len(data["reps"])
    abstained = quality_status == "reject" or rep_count == 0
    abstain_reason = None
    if quality_status == "reject":
        abstain_reason = "quality_reject"
    elif rep_count == 0:
        abstain_reason = "no_complete_reps"
    confidences = finite(item.get("confidence") for item in observations)

    assessment = {
        "schema_version": "vls-assessment-v1",
        "component": "assessment_measurement",
        "generated_at": generated_at,
        "clip_id": clip["clip_id"],
        "subject_id": clip["subject_id"],
        "exercise": exercise,
        "view": view,
        "split": clip["split"],
        "workflow_stage": "measurement_only",
        "input_contract": input_contract(data, clip, benchmark_path, contracts_path),
        "quality": {
            "status": quality_status,
            "decision": "abstain" if quality_status == "reject" else "continue",
            "warnings": quality.get("warnings") or [],
            "failures": quality.get("failures") or [],
            "checks": quality.get("checks") or {},
        },
        "view_capability": capability,
        "reps": [
            {
                key: rep[key]
                for key in (
                    "rep_id",
                    "start_frame",
                    "end_frame",
                    "bottom_frame",
                    "start_source_frame",
                    "end_source_frame",
                    "bottom_source_frame",
                    "duration_s",
                    "boundary_confidence",
                    "quality_flags",
                    "status",
                )
            }
            for rep in data["reps"]
        ],
        "rep_measurements": rep_measurements,
        "clip_measurements": clip_measurements,
        "issues": [],
        "summary": {
            "rep_count": rep_count,
            "observation_count": len(observations),
            "rep_observation_count": len(rep_measurements),
            "clip_observation_count": len(clip_measurements),
            "status_counts": dict(sorted(status_counts.items())),
            "evaluable_metric_count": len(evaluable_metrics),
            "evaluable_observation_ratio": (
                round(len(evaluable_observations) / len(applicable), 6)
                if applicable
                else 0.0
            ),
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "active_issue_count": 0,
        },
        "confidence": {
            "components": {
                "view": view_factor(contracts, view),
                "quality": quality_factor(contracts, quality_status),
                "mean_observation_confidence": (
                    round(float(np.mean(confidences)), 6) if confidences else None
                ),
            },
            "calibrated": None,
            "coverage": (
                round(len(evaluable_observations) / len(applicable), 6)
                if applicable
                else 0.0
            ),
        },
    }
    return assessment


def invariant_checks(
    assessment: dict[str, Any], contracts: dict[str, Any]
) -> list[dict[str, Any]]:
    errors = []
    all_metric_names = set(all_metrics(contracts))
    disabled = set(assessment["view_capability"]["disabled_metrics"])
    for observation in assessment["rep_measurements"] + assessment["clip_measurements"]:
        metric = observation["metric"]
        if metric not in all_metric_names:
            errors.append({"metric": metric, "error": "metric missing from contract"})
        if metric in disabled and observation["status"] != "disabled":
            errors.append({"metric": metric, "error": "disabled view emitted a usable status"})
        if metric in disabled and observation["raw_value"] is not None:
            errors.append({"metric": metric, "error": "disabled view emitted a raw value"})
        if observation["status"] in ACTIVE_STATUSES and observation["raw_value"] is None:
            errors.append({"metric": metric, "error": "active status has no raw value"})
        if observation["status"] == "disabled" and observation["raw_value"] is not None:
            errors.append({"metric": metric, "error": "disabled status has a raw value"})
        if observation["status"] in ACTIVE_STATUSES:
            if observation["confidence"] is None:
                errors.append({"metric": metric, "error": "active status has no confidence"})
            components = observation.get("confidence_components") or {}
            for component in ("signal", "rep_boundary", "view", "quality", "metric_scale"):
                if components.get(component) is None:
                    errors.append(
                        {"metric": metric, "error": f"active status lacks {component} confidence"}
                    )
            if not (observation.get("trace") or {}).get("source_signals"):
                errors.append({"metric": metric, "error": "active status has no source signals"})
        anchor = observation.get("anchor_frame")
        window = observation.get("window") or {}
        if observation["status"] in ACTIVE_STATUSES:
            if anchor is None or window.get("start_frame") is None:
                errors.append({"metric": metric, "error": "active status lacks an evidence window"})
        if anchor is not None and window.get("start_frame") is not None:
            if not (window["start_frame"] <= anchor <= window["end_frame"]):
                errors.append({"metric": metric, "error": "anchor outside measurement window"})
            if window.get("center_frame") != anchor:
                errors.append({"metric": metric, "error": "window center differs from anchor"})
            if observation.get("anchor_source_frame") != (
                assessment["input_contract"].get("trim", {}).get("start_frame", 0)
                + anchor
            ):
                errors.append({"metric": metric, "error": "source anchor mapping mismatch"})
    if assessment["quality"]["status"] == "reject":
        if not assessment["summary"]["abstained"]:
            errors.append({"metric": None, "error": "quality reject did not abstain"})
        if any(
            observation["status"] in ACTIVE_STATUSES
            for observation in assessment["rep_measurements"] + assessment["clip_measurements"]
        ):
            errors.append({"metric": None, "error": "quality reject left active measurements"})
    if assessment["issues"]:
        errors.append({"metric": None, "error": "measurement stage must not emit issues"})
    return errors


def features_rows(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for observation in assessment["rep_measurements"] + assessment["clip_measurements"]:
        components = observation["confidence_components"]
        window = observation["window"]
        trace = observation["trace"]
        rows.append(
            {
                "clip_id": assessment["clip_id"],
                "subject_id": assessment["subject_id"],
                "exercise": assessment["exercise"],
                "view": assessment["view"],
                "split": assessment["split"],
                "measurement_id": observation["measurement_id"],
                "scope": observation["scope"],
                "rep_id": observation["rep_id"],
                "metric": observation["metric"],
                "phase": observation["phase"],
                "raw_value": observation["raw_value"],
                "normalized_value": observation["normalized_value"],
                "unit": observation["unit"],
                "view_requirement": observation["view_requirement"],
                "source": observation["source"],
                "status": observation["status"],
                "confidence": observation["confidence"],
                "signal_confidence": components["signal"],
                "rep_boundary_confidence": components["rep_boundary"],
                "view_confidence": components["view"],
                "quality_confidence": components["quality"],
                "metric_scale": components["metric_scale"],
                "anchor_frame": observation["anchor_frame"],
                "anchor_source_frame": observation["anchor_source_frame"],
                "anchor_time_s": observation["anchor_time_s"],
                "window_start_frame": window["start_frame"],
                "window_end_frame": window["end_frame"],
                "window_center_frame": window["center_frame"],
                "aggregation": window["aggregation"],
                "quality_flags": ";".join(observation["quality_flags"]),
                "reason": observation["reason"] or "",
                "source_signals": ";".join(trace["source_signals"]),
                "source_file": trace["source_file"],
            }
        )
    return rows


def write_features(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in FEATURE_FIELDS})


def validate_schema(
    assessment: dict[str, Any], schema: dict[str, Any]
) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(assessment), key=lambda item: list(item.path))
    ]


def select_clips(
    benchmark: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    wanted = set(args.clip_id or [])
    selected = []
    for clip in benchmark.get("clips") or []:
        if clip["split"] not in args.split:
            continue
        if wanted and clip["clip_id"] not in wanted:
            continue
        selected.append(clip)
    missing = wanted - {clip["clip_id"] for clip in selected}
    if missing:
        raise ValueError("requested clips not selected by split: " + ", ".join(sorted(missing)))
    return selected


def main() -> int:
    args = parse_args()
    generated_at = args.generated_at or utc_now()
    benchmark = load_json(args.benchmark)
    taxonomy = load_json(args.taxonomy)
    contracts = load_json(args.contracts)
    schema = load_json(args.schema)
    contract_errors = validate_contracts(contracts, taxonomy)
    if contract_errors:
        raise ValueError("invalid metric contract:\n" + "\n".join(contract_errors))

    clips = select_clips(benchmark, args)
    if not clips:
        raise ValueError("no clips selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for clip in clips:
        clip_output = args.output_dir / clip["clip_id"]
        if clip_output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {clip_output}")
        clip_output.mkdir(parents=True, exist_ok=True)
        data = load_clip_data(clip, args.analysis_dir)
        assessment = build_assessment(
            clip,
            data,
            contracts,
            args.benchmark,
            args.contracts,
            generated_at,
        )
        schema_errors = validate_schema(assessment, schema)
        invariant_errors = invariant_checks(assessment, contracts)
        write_json(clip_output / "assessment.json", assessment)
        write_json(clip_output / "quality.json", assessment["quality"])
        write_features(clip_output / "features.csv", features_rows(assessment))
        write_json(
            clip_output / "checks" / "schema_validation.json",
            {
                "schema_version": "vls-schema-validation-v1",
                "valid": not schema_errors,
                "errors": schema_errors,
            },
        )
        write_json(
            clip_output / "checks" / "invariants.json",
            {
                "schema_version": "vls-assessment-invariants-v1",
                "valid": not invariant_errors,
                "errors": invariant_errors,
            },
        )
        if schema_errors or invariant_errors:
            raise ValueError(
                f"measurement validation failed for {clip['clip_id']}: "
                f"schema={len(schema_errors)} invariants={len(invariant_errors)}"
            )
        all_rows.extend(features_rows(assessment))
        summaries.append(
            {
                "clip_id": clip["clip_id"],
                "split": clip["split"],
                "exercise": clip["exercise"],
                "view": clip["view"],
                "quality_status": assessment["quality"]["status"],
                "rep_count": assessment["summary"]["rep_count"],
                "observation_count": assessment["summary"]["observation_count"],
                "status_counts": assessment["summary"]["status_counts"],
                "evaluable_metric_count": assessment["summary"]["evaluable_metric_count"],
                "evaluable_observation_ratio": assessment["summary"]["evaluable_observation_ratio"],
                "abstained": assessment["summary"]["abstained"],
                "abstain_reason": assessment["summary"]["abstain_reason"],
                "upstream_contract": assessment["input_contract"]["upstream_contract"],
            }
        )

    write_features(args.output_dir / "features_all.csv", all_rows)
    status_counts = Counter(row["status"] for row in all_rows)
    quality_status_counts = Counter(
        summary["quality_status"] for summary in summaries
    )
    abstain_reason_counts = Counter(
        summary["abstain_reason"]
        for summary in summaries
        if summary["abstain_reason"] is not None
    )
    upstream_kind_counts = Counter(
        summary["upstream_contract"]["kind"] for summary in summaries
    )
    status_counts_by_view: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for summary in summaries:
        status_counts_by_view[summary["view"]].update(summary["status_counts"])
        status_counts_by_action[summary["exercise"]].update(summary["status_counts"])
    manifest = {
        "schema_version": "vls-measurement-manifest-v1",
        "component": "assessment_measurement",
        "generated_at": generated_at,
        "workflow_stage": "measurement_only",
        "benchmark_path": relative_path(args.benchmark),
        "benchmark_sha256": sha256_file(args.benchmark),
        "metric_contracts_path": relative_path(args.contracts),
        "metric_contracts_sha256": sha256_file(args.contracts),
        "taxonomy_path": relative_path(args.taxonomy),
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "schema_path": relative_path(args.schema),
        "schema_sha256": sha256_file(args.schema),
        "analysis_dir": relative_path(args.analysis_dir),
        "selected_splits": args.split,
        "clip_count": len(clips),
        "feature_rows": len(all_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "status_counts_by_view": {
            view: dict(sorted(counts.items()))
            for view, counts in sorted(status_counts_by_view.items())
        },
        "status_counts_by_action": {
            action: dict(sorted(counts.items()))
            for action, counts in sorted(status_counts_by_action.items())
        },
        "quality_status_counts": dict(sorted(quality_status_counts.items())),
        "abstained_clip_count": sum(
            bool(summary["abstained"]) for summary in summaries
        ),
        "abstain_reason_counts": dict(sorted(abstain_reason_counts.items())),
        "upstream_contract_counts": dict(sorted(upstream_kind_counts.items())),
        "issue_count": 0,
        "clips": summaries,
        "command": (
            ".venv/bin/python web/backend/app/repchain/assessment_measurement.py "
            f"--split {' '.join(args.split)}"
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"measured {len(clips)} clips, wrote {len(all_rows)} feature rows to "
        f"{relative_path(args.output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
