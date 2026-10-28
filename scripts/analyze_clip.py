#!/usr/bin/env python3
"""Analyze one task1-4 clip and emit pose, angle, quality, and rep outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import extract_angles
import extract_pose_landmarker
import segment_reps
import task1_5_metrics
from paths import resolve_data_path


DEFAULT_MANIFEST = ROOT / "configs" / "task1_4_clips.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT = ROOT / "results" / "clip_analysis_v1"

SCHEMA_VERSION = "task1-4-clip-analysis-v1"
QUALITY_SCHEMA_VERSION = "task1-4-quality-v1"
POSTURE_GATE_SCHEMA_VERSION = "task1-4-posture-gate-v1"
SEGMENTATION_CANDIDATES_SCHEMA_VERSION = "task1-4-segmentation-candidates-v1"

MIN_PERIOD_S = 0.40
MAX_PERIOD_S = 5.00
PROMINENCE_FRAC = 0.20
ACTIVITY_WINDOW_S = segment_reps.ACTIVITY_WINDOW_S
ACTIVITY_THRESHOLD = segment_reps.ACTIVITY_THRESHOLD
MIN_NONREP_S = segment_reps.MIN_NONREP_S
ACTIVITY_CLOSE_GAP_S = segment_reps.ACTIVITY_CLOSE_GAP_S
SHAPE_TEMPLATE = segment_reps.DEFAULT_SHAPE_TEMPLATE

POSE_DETECTION_PASS = 0.90
POSE_DETECTION_REJECT = 0.75
VISIBILITY_Q10_PASS = 0.50
VISIBILITY_Q10_REJECT = 0.35
BODY_IN_FRAME_PASS = 0.95
BODY_IN_FRAME_REJECT = 0.85
MIN_COMPLETE_REPS = 3
PHASE_BOUNDARY_LEVEL_MIN = 0.25
MIN_REP_WIDTH_RATIO = 0.65
MAX_REP_WIDTH_RATIO = 1.50
REP_LOW_CONFIDENCE = 0.35
REP_LOW_SHAPE_SIMILARITY = 0.50

EXERCISE_JOINTS = {
    "squat": (11, 12, 23, 24, 25, 26, 27, 28),
    "pushup": (11, 12, 13, 14, 15, 16, 23, 24, 27, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--clip-id", nargs="+", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 10:
        parser.error("--max-frames must be >= 10")
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "task1-4-v1":
        raise ValueError(f"unexpected manifest schema: {payload.get('schema_version')}")
    return payload


def select_clips(manifest: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    by_id = {clip["clip_id"]: clip for clip in manifest["clips"]}
    missing = [clip_id for clip_id in requested if clip_id not in by_id]
    if missing:
        raise ValueError("unknown clip_id: " + ", ".join(missing))
    return [by_id[clip_id] for clip_id in requested]


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    return extract_pose_landmarker.rotate_frame(frame, rotation)


def make_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def landmark_quality(result: Any, exercise: str) -> dict[str, Any]:
    if not result.pose_landmarks:
        return {
            "detected": False,
            "mean_visibility": 0.0,
            "min_visibility": 0.0,
            "visible_joint_fraction": 0.0,
            "all_joints_visible": False,
            "in_frame": False,
        }
    landmarks = result.pose_landmarks[0]
    indices = EXERCISE_JOINTS[exercise]
    visibility = np.asarray([landmarks[index].visibility for index in indices], dtype=np.float64)
    xy = np.asarray(
        [[landmarks[index].x, landmarks[index].y] for index in indices],
        dtype=np.float64,
    )
    in_frame = bool(np.all((xy >= -0.02) & (xy <= 1.02)))
    return {
        "detected": True,
        "mean_visibility": float(visibility.mean()),
        "min_visibility": float(visibility.min()),
        "visible_joint_fraction": float(np.mean(visibility >= 0.5)),
        "all_joints_visible": bool(np.all(visibility >= 0.5)),
        "in_frame": in_frame,
    }


def source_time_s(source_frame: int, fps: float) -> float | None:
    """Map an original-video frame index to seconds from the start of that video."""
    return round(source_frame / fps, 6) if fps > 0 else None


def clip_time_s(source_frame: int, clip_start_frame: int, fps: float) -> float | None:
    """Return seconds relative to the first frame of this clip."""
    return round((source_frame - clip_start_frame) / fps, 6) if fps > 0 else None


def extract_clip_pose(
    clip: dict[str, Any],
    args: argparse.Namespace,
    pose_source_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    clip_id = clip["clip_id"]
    video_path = resolve_data_path(clip["video_path"])
    start_frame = int(clip["trim"]["start_frame"])
    end_frame = int(clip["trim"]["end_frame"])
    expected_frames = end_frame - start_frame + 1
    if args.max_frames is not None:
        end_frame = min(end_frame, start_frame + args.max_frames - 1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pose_source_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = pose_source_dir / f"{clip_id}_landmarks.json"
    meta_path = pose_source_dir / f"{clip_id}_meta.json"
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_detection_confidence,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    frame_quality = []
    decoded = 0
    detected = 0
    started = time.perf_counter()
    try:
        with landmarks_path.open("w") as handle:
            handle.write("[\n")
            first = True
            for local_index, source_frame in enumerate(range(start_frame, end_frame + 1)):
                ok, frame = cap.read()
                if not ok:
                    break
                rotated = rotate_frame(frame, clip["rotation"])
                timestamp_ms = int(round(local_index * 1000.0 / fps)) if fps > 0 else local_index
                result = landmarker.detect_for_video(make_image(rotated), timestamp_ms)
                payload = extract_pose_landmarker.serialize_result(result)
                quality = landmark_quality(result, clip["exercise"])
                quality.update({"source_frame": source_frame, "clip_frame_index": local_index})
                frame_quality.append(quality)
                record = {
                    "frame_id": local_index,
                    "source_frame": source_frame,
                    "timestamp_ms": timestamp_ms,
                    "landmarks_world": payload["landmarks_world"],
                    "visibility": payload["visibility"],
                    "source": "vision",
                    "detected": payload["detected"],
                }
                if not first:
                    handle.write(",\n")
                json.dump(record, handle, separators=(",", ":"), allow_nan=False)
                first = False
                decoded += 1
                detected += int(payload["detected"])
                if args.log_every and decoded % args.log_every == 0:
                    print(f"{clip_id}: {decoded} frames decoded", flush=True)
            handle.write("\n]\n")
    finally:
        cap.release()
        landmarker.close()

    elapsed = time.perf_counter() - started
    meta = {
        "fps": fps,
        "stride": 1,
        "source_video": str(video_path.resolve()),
        "source_size": {"width": width, "height": height},
        "rotation": clip["rotation"],
        "trim": {"start_frame": start_frame, "end_frame": end_frame},
        "expected_frames": expected_frames,
        "processed_frames": decoded,
        "detected_frames": detected,
        "processing_seconds": elapsed,
        "processing_fps": decoded / elapsed if elapsed > 0 else None,
    }
    write_json(meta_path, meta)

    detections = [item for item in frame_quality if item["detected"]]
    mean_visibilities = [item["mean_visibility"] for item in detections]
    summary = {
        "expected_frames": expected_frames,
        "processed_frames": decoded,
        "frame_coverage": decoded / expected_frames if expected_frames else 0.0,
        "detected_frames": detected,
        "pose_detection_rate": detected / decoded if decoded else 0.0,
        "mean_key_visibility": float(np.mean(mean_visibilities)) if mean_visibilities else 0.0,
        "q10_key_visibility": (
            float(np.percentile(mean_visibilities, 10)) if mean_visibilities else 0.0
        ),
        "key_joint_visible_rate": (
            float(np.mean([item["visible_joint_fraction"] for item in frame_quality]))
            if frame_quality
            else 0.0
        ),
        "all_exercise_joints_visible_rate": (
            sum(item["all_joints_visible"] for item in frame_quality) / decoded
            if decoded
            else 0.0
        ),
        # Kept as a compatibility alias for the strict per-frame all-joint diagnostic.
        "all_required_visible_rate": (
            sum(item["all_joints_visible"] for item in frame_quality) / decoded
            if decoded
            else 0.0
        ),
        "body_in_frame_rate": (
            sum(item["in_frame"] for item in detections) / decoded if decoded else 0.0
        ),
        "processing_fps": meta["processing_fps"],
    }
    return landmarks_path, meta_path, summary


def run_angle_extraction(
    clip_id: str,
    pose_source_dir: Path,
    angle_staging_parent: Path,
    final_angles_dir: Path,
    view: str,
    clip: dict[str, Any],
) -> dict[str, Any]:
    angle_args = SimpleNamespace(
        source_dir=pose_source_dir,
        output_dir=angle_staging_parent,
        overwrite=True,
        visibility_threshold=extract_angles.VISIBILITY_THRESHOLD,
        smooth_window=extract_angles.SMOOTH_WINDOW,
        smooth_polyorder=extract_angles.SMOOTH_POLYORDER,
        outlier_threshold_deg=extract_angles.OUTLIER_THRESHOLD_DEG,
        spine_window=extract_angles.SPINE_WINDOW,
    )
    meta = extract_angles.process_video(clip_id, angle_args)
    staged_dir = angle_staging_parent / clip_id
    if final_angles_dir.exists():
        shutil.rmtree(final_angles_dir)
    shutil.move(str(staged_dir), str(final_angles_dir))
    final_files = {
        "landmarks": final_angles_dir / "landmarks.json",
        "landmarks_body_frame": final_angles_dir / "landmarks_body_frame.json",
        "angles": final_angles_dir / "angles.csv",
        "angles_wide": final_angles_dir / "angles_wide.csv",
    }
    meta["view_label"] = view
    meta["clip_id"] = clip_id
    meta["clip_trim"] = clip["trim"]
    meta["source_task1_1_landmarks"] = str(final_files["landmarks"].resolve())
    meta["outputs"] = {name: str(path.resolve()) for name, path in final_files.items()}
    write_json(final_angles_dir / "meta.json", meta)
    return meta


def phase_guidance_for(exercise: str, specs: dict[str, Any]) -> segment_reps.PhaseGuidance:
    guidance_by_action = segment_reps.load_phase_config(segment_reps.DEFAULT_PHASE_CONFIG)
    if exercise in guidance_by_action:
        return guidance_by_action[exercise]
    phase = specs["exercises"][exercise]["phase"]
    polarity = str(phase.get("boundary_polarity", "peak"))
    return segment_reps.PhaseGuidance(
        phase_landmark=str(phase.get("phase_landmark", phase["top"])),
        annotation_polarity=polarity,
        signal=segment_reps.PhaseSignalSpec(source="angle", name=str(phase["signal"])),
        polarity=polarity,
    )


def side_signal_visibility(exercise: str, landmarks_path: Path) -> dict[str, float]:
    specs = task1_5_metrics.SIDE_SIGNAL_SPECS.get(exercise, {})
    if not specs or not landmarks_path.is_file():
        return {}
    records = json.loads(landmarks_path.read_text())
    values: dict[str, list[float]] = {side: [] for side in specs}
    for record in records:
        if not record.get("detected"):
            continue
        visibility = np.asarray(record.get("visibility") or [], dtype=np.float64)
        for side, spec in specs.items():
            indices = spec["mediapipe_joints"]
            if len(visibility) <= max(indices):
                continue
            values[side].append(float(np.min(visibility[list(indices)])))
    return {
        side: float(np.median(samples)) if samples else 0.0
        for side, samples in values.items()
    }


def select_view_phase_signal(
    clip: dict[str, Any], angles_dir: Path
) -> dict[str, Any] | None:
    view = str(clip.get("view", "unknown"))
    if view not in task1_5_metrics.SIDE_VIEWS:
        return None
    wide_path = angles_dir / "angles_wide.csv"
    if not wide_path.is_file():
        return None
    _, wide = segment_reps.load_wide(wide_path)
    visibility = side_signal_visibility(clip["exercise"], angles_dir / "landmarks.json")
    geometry = clip.get("view_geometry") or {}
    return task1_5_metrics.choose_side_phase_signal(
        clip["exercise"],
        view,
        geometry.get("camera_side"),
        wide,
        visibility,
    )


def phase_level(value: float | None, p10: float, p90: float) -> float | None:
    """Normalize a joint angle into its clip-local low/high phase range."""
    if value is None or p90 - p10 < 1e-8:
        return None
    return float(np.clip((value - p10) / (p90 - p10), 0.0, 1.0))


def find_phase_cycle_boundaries(
    phase_values: np.ndarray,
    fps: float,
    min_period_s: float,
    max_period_s: float,
) -> tuple[list[int], dict[str, Any]]:
    """Find top-to-top boundaries from pronounced phase troughs and high-level crossings."""
    finite = phase_values[np.isfinite(phase_values)]
    if len(finite) < 5 or fps <= 0:
        return [], {"status": "fallback", "reason": "insufficient_phase_signal"}
    p10, p90 = np.percentile(finite, [10, 90])
    dynamic_range = float(p90 - p10)
    if dynamic_range < 1e-8:
        return [], {"status": "fallback", "reason": "flat_phase_signal"}
    min_gap = max(4, int(round(0.35 * min_period_s * fps)))
    troughs, properties = find_peaks(
        -phase_values,
        distance=min_gap,
        prominence=0.55 * dynamic_range,
    )
    troughs = np.asarray(troughs, dtype=np.int64)
    if len(troughs) == 0:
        return [], {"status": "fallback", "reason": "no_prominent_troughs"}
    min_gap_frames = 0.55 * min_period_s * fps
    max_gap_frames = 1.25 * max_period_s * fps
    plausible = []
    for index, trough in enumerate(troughs):
        gaps = []
        if index > 0:
            gaps.append(int(trough - troughs[index - 1]))
        if index + 1 < len(troughs):
            gaps.append(int(troughs[index + 1] - trough))
        if not gaps or all(min_gap_frames <= gap <= max_gap_frames for gap in gaps):
            plausible.append(int(trough))
    troughs = np.asarray(plausible, dtype=np.int64)
    if len(troughs) == 0:
        return [], {"status": "fallback", "reason": "implausible_rep_intervals"}

    gaps = np.diff(troughs)
    median_gap = float(np.median(gaps)) if len(gaps) else min_period_s * fps
    first_boundary = max(0, int(troughs[0] - round(median_gap)))
    high_threshold = float(p10 + 0.65 * dynamic_range)
    boundaries = [first_boundary]
    for index, trough in enumerate(troughs):
        search_end = int(troughs[index + 1]) if index + 1 < len(troughs) else len(phase_values)
        candidates = np.flatnonzero(phase_values[trough:search_end] >= high_threshold)
        if len(candidates) == 0:
            break
        boundaries.append(int(trough + candidates[0]))
    if len(boundaries) - 1 < 1:
        return [], {"status": "fallback", "reason": "no_complete_phase_cycles"}
    evidence = {
        "status": "applied",
        "method": "phase_cycles",
        "trough_frames": [int(value) for value in troughs[: len(boundaries) - 1]],
        "boundary_frames": boundaries,
        "phase_p10": float(p10),
        "phase_p90": float(p90),
        "high_threshold": high_threshold,
        "trough_prominences": [
            float(value) for value in properties.get("prominences", [])[: len(boundaries) - 1]
        ],
    }
    return boundaries, evidence


def build_raw_rep_records(
    reps: list[segment_reps.Rep],
    phase_values: np.ndarray,
    frame_offset: int,
    fps: float,
) -> list[dict[str, Any]]:
    records = []
    frame_count = len(phase_values)
    for rep in reps:
        start = max(0, min(frame_count - 1, rep.start_frame))
        end = max(start, min(frame_count - 1, rep.end_frame))
        values = phase_values[start : end + 1]
        bottom_local = (
            start + int(np.nanargmin(values))
            if values.size and np.isfinite(values).any()
            else start
        )
        start_value = float(phase_values[start]) if np.isfinite(phase_values[start]) else None
        end_value = float(phase_values[end]) if np.isfinite(phase_values[end]) else None
        bottom_value = (
            float(phase_values[bottom_local]) if np.isfinite(phase_values[bottom_local]) else None
        )
        source_start = frame_offset + start
        source_end = frame_offset + end
        source_bottom = frame_offset + bottom_local
        records.append(
            {
                "rep_id": rep.rep_id,
                "start_frame": source_start,
                "end_frame": source_end,
                "top_frame": source_start,
                "bottom_frame": source_bottom,
                "start_time_s": source_time_s(source_start, fps),
                "end_time_s": source_time_s(source_end, fps),
                "bottom_time_s": source_time_s(source_bottom, fps),
                "clip_start_time_s": clip_time_s(source_start, frame_offset, fps),
                "clip_end_time_s": clip_time_s(source_end, frame_offset, fps),
                "clip_bottom_time_s": clip_time_s(source_bottom, frame_offset, fps),
                "duration_s": round((end - start + 1) / fps, 6) if fps > 0 else 0.0,
                "concentric_start_frame": source_bottom,
                "concentric_end_frame": source_end,
                "eccentric_start_frame": source_start,
                "eccentric_end_frame": source_bottom,
                "phase_angle": "",
                "top_angle_deg": None if start_value is None else round(start_value, 6),
                "start_angle_deg": None if start_value is None else round(start_value, 6),
                "end_angle_deg": None if end_value is None else round(end_value, 6),
                "bottom_angle_deg": None if bottom_value is None else round(bottom_value, 6),
                "confidence": round(float(rep.confidence), 6),
                "visibility": round(float(rep.visibility), 6),
                "shape_similarity": round(float(rep.shape_similarity), 6),
                "phase_consistency": round(float(rep.phase_consistency), 6),
                "motion_evidence": round(float(rep.motion_evidence), 6),
                "review_score": round(float(rep.review_score), 6),
                "review_priority": rep.review_priority,
                "reject_reason": rep.reject_reason,
            }
        )
    return records


def assess_rep_records(
    reps: list[dict[str, Any]], phase_values: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flag and remove rep-shaped segments that are not complete phase cycles."""
    if not reps:
        return [], []
    finite_phase = phase_values[np.isfinite(phase_values)]
    if len(finite_phase):
        phase_p10, phase_p90 = np.percentile(finite_phase, [10, 90])
    else:
        phase_p10 = phase_p90 = 0.0
    widths = np.asarray(
        [rep["end_frame"] - rep["start_frame"] + 1 for rep in reps],
        dtype=np.float64,
    )
    reference_width = float(np.median(widths)) if len(widths) else 0.0
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rep in reps:
        width = rep["end_frame"] - rep["start_frame"] + 1
        width_ratio = width / reference_width if reference_width > 0 else None
        start_level = phase_level(rep["start_angle_deg"], float(phase_p10), float(phase_p90))
        end_level = phase_level(rep["end_angle_deg"], float(phase_p10), float(phase_p90))
        flags = []
        fatal_flags = []
        if (
            rep["bottom_frame"] <= rep["start_frame"] + 1
            or rep["bottom_frame"] >= rep["end_frame"] - 1
        ):
            flags.append("bottom_near_boundary")
            fatal_flags.append("missing_interior_bottom")
        if start_level is not None and start_level < PHASE_BOUNDARY_LEVEL_MIN:
            flags.append("start_not_boundary_phase")
            fatal_flags.append("start_not_boundary_phase")
        if end_level is not None and end_level < PHASE_BOUNDARY_LEVEL_MIN:
            flags.append("end_not_boundary_phase")
            fatal_flags.append("end_not_boundary_phase")
        if width_ratio is not None and width_ratio < MIN_REP_WIDTH_RATIO:
            flags.append("short_rep")
        elif width_ratio is not None and width_ratio > MAX_REP_WIDTH_RATIO:
            flags.append("long_rep")
        low_evidence = (
            rep["confidence"] < REP_LOW_CONFIDENCE
            and rep["shape_similarity"] < REP_LOW_SHAPE_SIMILARITY
        )
        if low_evidence:
            flags.append("low_rep_evidence")
        if "short_rep" in flags and (low_evidence or fatal_flags):
            fatal_flags.append("incomplete_short_rep")
        if "long_rep" in flags and (low_evidence or fatal_flags):
            fatal_flags.append("misaligned_long_rep")
        record = {
            **rep,
            "width_ratio": None if width_ratio is None else round(float(width_ratio), 6),
            "start_phase_level": None if start_level is None else round(start_level, 6),
            "end_phase_level": None if end_level is None else round(end_level, 6),
            "quality_flags": ";".join(flags),
        }
        if fatal_flags:
            record["status"] = "rejected"
            record["reject_reasons"] = ";".join(dict.fromkeys(fatal_flags))
            rejected.append(record)
        else:
            record["status"] = "complete"
            record["reject_reasons"] = ""
            kept.append(record)
    for rep_id, rep in enumerate(kept, start=1):
        rep["rep_id"] = rep_id
    return kept, rejected


def phase_cycle_method(
    phase_values: np.ndarray,
    prepared: segment_reps.PreparedSignal,
    angle: str,
    guidance: segment_reps.PhaseGuidance,
    fps: float,
    min_period_s: float,
    max_period_s: float,
) -> tuple[segment_reps.MethodResult | None, dict[str, Any]]:
    boundaries, evidence = find_phase_cycle_boundaries(
        phase_values, fps, min_period_s, max_period_s
    )
    if len(boundaries) < 3:
        return None, evidence
    local_boundaries = (
        np.asarray(boundaries, dtype=np.int64) - int(prepared.source_start)
    )
    if local_boundaries[0] < 0 or local_boundaries[-1] >= len(prepared.normalized):
        return None, {**evidence, "status": "fallback", "reason": "boundaries_outside_signal"}
    reps, similarity, rejected = segment_reps.reps_from_boundaries(
        local_boundaries,
        prepared.valid,
        prepared.normalized,
        prepared.source_start,
    )
    if len(reps) < 2:
        return None, {**evidence, "status": "fallback", "reason": "too_few_cycle_reps"}
    period_frames = float(np.median(np.diff(local_boundaries)))
    phase_evidence = {
        **evidence,
        "phase_landmark": guidance.phase_landmark,
        "annotation_polarity": guidance.annotation_polarity,
        "signal": {
            "source": guidance.signal.source,
            "name": guidance.signal.name,
            "axis": guidance.signal.axis,
            "sign": guidance.signal.sign,
        },
        "polarity": guidance.polarity,
        "tolerance_frames": max(1, int(round(guidance.tolerance_fraction * period_frames))),
        "extrema_frames": [
            int(value - prepared.source_start) for value in boundaries[1:-1]
        ],
        "period_frames": int(round(period_frames)),
        "start_frame": int(boundaries[0]),
    }
    reps = segment_reps.enrich_rep_confidences(
        reps,
        signal=prepared.normalized,
        valid=prepared.valid,
        offset=prepared.source_start,
        template_method=SHAPE_TEMPLATE,
        phase_evidence=phase_evidence,
        period_frames=period_frames,
        phase_signal_available=True,
    )
    result = segment_reps.MethodResult(
        method="phase_cycles",
        angle=angle,
        reps=reps,
        period_frames=period_frames,
        similarity=float(np.mean([rep.shape_similarity for rep in reps])) if reps else similarity,
        valid_fraction=float(prepared.valid.mean()),
        objective=0.0,
        rejected_reps=rejected,
        phase_guidance=phase_evidence,
    )
    return result, evidence


def angle_selection_for(
    wide: dict[str, np.ndarray],
    segment: segment_reps.ActionSegment,
    fps: float,
    angle: str,
) -> segment_reps.AngleSelection | None:
    """Build an angle selection without the legacy whole-clip angle scorer."""
    if angle not in wide:
        return None
    values = wide[angle][segment.start_frame : segment.end_frame + 1]
    prepared = segment_reps.prepare_signal(values)
    if prepared is None:
        return None
    period_result = segment_reps.estimate_period(
        prepared.normalized, fps, MIN_PERIOD_S, MAX_PERIOD_S
    )
    return segment_reps.AngleSelection(
        angle=angle,
        prepared=prepared,
        periodicity=float(period_result[1]) if period_result is not None else 0.0,
        evidence={"method": "phase_signal_override"},
    )


def segment_clip(
    clip: dict[str, Any],
    specs: dict[str, Any],
    fps: float,
    angle_override: str | None = None,
    phase_signal_override: str | None = None,
) -> dict[str, Any]:
    angles_dir = Path(clip["_angles_dir"])
    timestamps_ms, wide = segment_reps.load_wide(angles_dir / "angles_wide.csv")
    frame_count = len(timestamps_ms)
    segment = segment_reps.ActionSegment(
        video_id=clip["clip_id"],
        action=clip["exercise"],
        action_display=clip["action_display"],
        start_frame=0,
        end_frame=frame_count - 1,
        start_time_s=0.0,
        end_time_s=(frame_count - 1) / fps if fps > 0 else 0.0,
    )
    guidance = phase_guidance_for(clip["exercise"], specs)
    if phase_signal_override is not None:
        guidance = replace(
            guidance,
            signal=segment_reps.PhaseSignalSpec(
                source="angle", name=str(phase_signal_override)
            ),
        )
    phase_signal = wide.get(guidance.signal.name)
    if angle_override is not None:
        selection = angle_selection_for(wide, segment, fps, angle_override)
    else:
        selection = segment_reps.choose_auto_angle(
            wide,
            segment,
            fps,
            MIN_PERIOD_S,
            MAX_PERIOD_S,
            scoring="legacy",
        )
    if selection is None:
        return {
            "status": "no_usable_angle",
            "angle_selection": None,
            "reps": [],
            "phases": [],
            "non_rep_intervals": [],
            "active_bouts": [],
            "method": None,
        }

    angle = selection.angle
    prepared = selection.prepared
    effective_offset = prepared.source_start
    period_result = segment_reps.estimate_period(
        prepared.normalized, fps, MIN_PERIOD_S, MAX_PERIOD_S
    )
    global_period_frames = float(period_result[0]) if period_result is not None else 0.0
    segment_values = wide[angle][segment.start_frame : segment.end_frame + 1]
    trace = segment_reps.build_activity_trace(
        wide,
        segment,
        angle,
        fps,
        ACTIVITY_WINDOW_S,
        ACTIVITY_THRESHOLD,
        ACTIVITY_CLOSE_GAP_S,
    )
    active_bouts = []
    non_rep_intervals = []
    activity_evidence = None
    if trace is None:
        phase_values = (
            phase_signal[prepared.source_start : prepared.source_end + 1]
            if phase_signal is not None
            else None
        )
        aifit = segment_reps.run_aifit(
            prepared.normalized,
            prepared.valid,
            angle,
            effective_offset,
            fps,
            MIN_PERIOD_S,
            MAX_PERIOD_S,
            True,
            phase_signal=phase_values,
            phase_guidance=guidance if phase_values is not None else None,
            use_phase_snap=True,
            shape_template_method=SHAPE_TEMPLATE,
        )
        activity_evidence = {"status": "unavailable", "fallback": "global AIFit"}
    else:
        low_intervals = segment_reps.significant_low_intervals(
            trace, global_period_frames, fps, MIN_NONREP_S
        )
        aifit, active_bouts, non_rep_intervals = segment_reps.run_aifit_with_active_bouts(
            segment_values,
            prepared,
            trace,
            low_intervals,
            segment,
            angle,
            fps,
            MIN_PERIOD_S,
            MAX_PERIOD_S,
            True,
            phase_values=phase_signal,
            phase_guidance=guidance,
            use_phase_snap=True,
            shape_template_method=SHAPE_TEMPLATE,
        )
        activity_evidence = dict(trace.evidence)
        activity_evidence.update(
            {
                "global_period_frames": global_period_frames,
                "significant_low_intervals": [
                    {"start_frame": start, "end_frame": end, "frames": end - start + 1}
                    for start, end in low_intervals
                ],
            }
        )

    if aifit is not None:
        aifit.reps = segment_reps.enrich_rep_confidences(
            aifit.reps,
            signal=prepared.normalized,
            valid=prepared.valid,
            offset=effective_offset,
            template_method=SHAPE_TEMPLATE,
            trace=trace,
            segment_start_frame=segment.start_frame,
        )
    peak = segment_reps.run_peak_baseline(
        prepared.detrended,
        prepared.valid,
        angle,
        effective_offset,
        fps,
        MIN_PERIOD_S,
        MAX_PERIOD_S,
        PROMINENCE_FRAC,
    )
    if peak is not None:
        peak.reps = segment_reps.enrich_rep_confidences(
            peak.reps,
            signal=prepared.detrended,
            valid=prepared.valid,
            offset=effective_offset,
            template_method=SHAPE_TEMPLATE,
            trace=trace,
            segment_start_frame=segment.start_frame,
        )
    chosen = aifit if aifit is not None and aifit.reps else peak
    chosen_method = chosen.method if chosen is not None else None
    if chosen is None:
        return {
            "status": "no_reps",
            "angle_selection": selection.evidence,
            "angle": angle,
            "reps": [],
            "phases": [],
            "non_rep_intervals": [item.__dict__ for item in non_rep_intervals],
            "active_bouts": [item.__dict__ for item in active_bouts],
            "activity_evidence": activity_evidence,
            "method": chosen_method,
            "aifit_peak_iou": None,
        }

    frame_offset = int(clip["trim"]["start_frame"])
    phase_angle_name = guidance.signal.name if guidance.signal.name in wide else angle
    phase_values = wide[phase_angle_name]
    raw_reps = build_raw_rep_records(chosen.reps, phase_values, frame_offset, fps)
    reps, rejected_reps = assess_rep_records(raw_reps, phase_values)
    aifit_complete_rep_count = len(reps)
    phase_result, phase_cycle_evidence = phase_cycle_method(
        phase_values,
        prepared,
        angle,
        guidance,
        fps,
        MIN_PERIOD_S,
        MAX_PERIOD_S,
    )
    phase_complete_rep_count = 0
    if phase_result is not None:
        phase_raw_reps = build_raw_rep_records(
            phase_result.reps, phase_values, frame_offset, fps
        )
        phase_reps, phase_rejected_reps = assess_rep_records(phase_raw_reps, phase_values)
        phase_complete_rep_count = len(phase_reps)
        if phase_complete_rep_count > len(reps):
            chosen = phase_result
            chosen_method = chosen.method
            reps = phase_reps
            rejected_reps = phase_rejected_reps
    for rep in reps:
        rep["phase_angle"] = phase_angle_name
    phases = []
    for rep in reps:
        start = rep["start_frame"] - frame_offset
        end = rep["end_frame"] - frame_offset
        bottom_local = rep["bottom_frame"] - frame_offset
        top_local = rep["top_frame"] - frame_offset
        top_value = rep["start_angle_deg"]
        bottom_value = rep["bottom_angle_deg"]
        end_value = rep["end_angle_deg"]
        for phase, local_frame, value, confidence in (
            ("top_start", top_local, top_value, rep["confidence"]),
            ("bottom", bottom_local, bottom_value, rep["confidence"]),
            ("top_end", end, end_value, rep["confidence"]),
        ):
            source_frame = frame_offset + local_frame
            phases.append(
                {
                    "rep_id": rep["rep_id"],
                    "phase": phase,
                    "frame_index": source_frame,
                    "clip_frame_index": local_frame,
                    "time_s": source_time_s(source_frame, fps),
                    "clip_time_s": clip_time_s(source_frame, frame_offset, fps),
                    "angle_name": phase_angle_name,
                    "angle_deg": None if value is None else round(float(value), 6),
                    "confidence": round(float(confidence), 6),
                }
            )

    internal_rejected = []
    for rep_id, rejected in enumerate(chosen.rejected_reps, start=1):
        start_frame = frame_offset + int(rejected.start_frame)
        end_frame = frame_offset + int(rejected.end_frame)
        internal_rejected.append(
            {
                "candidate_rep_id": rep_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": source_time_s(start_frame, fps),
                "end_time_s": source_time_s(end_frame, fps),
                "quality_flags": rejected.reason,
                "reject_reasons": rejected.reason,
                "evidence": rejected.evidence,
            }
        )

    aifit_peak_iou = (
        segment_reps.agreement(aifit.reps, peak.reps)
        if aifit is not None and peak is not None
        else None
    )
    phase_evidence = chosen.phase_guidance or {}
    phase_bouts = phase_evidence.get("bouts") or [phase_evidence]
    boundary_snap = next(
        (
            bout.get("boundary_snap")
            for bout in phase_bouts
            if isinstance(bout, dict) and bout.get("boundary_snap")
        ),
        None,
    )
    return {
        "status": "ok",
        "angle_selection": selection.evidence,
        "angle": angle,
        "phase_angle": phase_angle_name,
        "method": chosen_method,
        "rep_unit": specs["exercises"][clip["exercise"]].get("rep_unit", "cycle"),
        "phase_guidance": {
            "phase_landmark": guidance.phase_landmark,
            "boundary_polarity": guidance.polarity,
            "bottom_polarity": specs["exercises"][clip["exercise"]]["phase"].get(
                "bottom_polarity", "trough"
            ),
            "signal": guidance.signal.name,
            "phase_status": phase_evidence.get("status"),
            "boundary_snap": boundary_snap,
        },
        "candidate_sources": {
            "aifit_complete_rep_count": aifit_complete_rep_count,
            "phase_cycle_complete_rep_count": phase_complete_rep_count,
            "selected": chosen_method,
        },
        "phase_cycle_detection": phase_cycle_evidence,
        "reps": reps,
        "rejected_reps": internal_rejected + rejected_reps,
        "phases": phases,
        "non_rep_intervals": [item.__dict__ for item in non_rep_intervals],
        "active_bouts": [item.__dict__ for item in active_bouts],
        "activity_evidence": activity_evidence,
        "aifit_peak_iou": None if aifit_peak_iou is None else round(float(aifit_peak_iou), 6),
    }


def torso_vertical_span_by_frame(
    landmarks_path: Path,
    trim_start: int,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in json.loads(landmarks_path.read_text()):
        landmarks = record.get("landmarks_world")
        if not isinstance(landmarks, list) or len(landmarks) < 25:
            continue
        try:
            shoulder_y = (float(landmarks[11][1]) + float(landmarks[12][1])) / 2.0
            hip_y = (float(landmarks[23][1]) + float(landmarks[24][1])) / 2.0
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(shoulder_y) and np.isfinite(hip_y)):
            continue
        source_frame = record.get("source_frame")
        if source_frame is None:
            source_frame = trim_start + int(record.get("frame_id", 0))
        values[int(source_frame)] = abs(shoulder_y - hip_y)
    return values


def apply_exercise_posture_gate(
    segmentation: dict[str, Any],
    clip: dict[str, Any],
    angles_dir: Path,
    specs: dict[str, Any],
) -> dict[str, Any]:
    gate = specs["exercises"][clip["exercise"]].get("posture_gate")
    if not gate:
        return segmentation

    trim_start = int(clip["trim"]["start_frame"])
    values = torso_vertical_span_by_frame(angles_dir / "landmarks.json", trim_start)
    rule = gate["rule"]
    threshold = float(gate["threshold"])
    kept = []
    rejected = []
    rep_evidence = []
    old_to_new = {}

    for rep in segmentation.get("reps", []):
        old_rep_id = int(rep["rep_id"])
        bottom_frame = int(rep["bottom_frame"])
        rep_values = [
            values[frame]
            for frame in range(int(rep["start_frame"]), int(rep["end_frame"]) + 1)
            if frame in values
        ]
        bottom_value = values.get(bottom_frame)
        evidence = {
            "metric": gate["metric"],
            "rule": rule,
            "threshold": threshold,
            "bottom_frame": bottom_frame,
            "bottom_value": None if bottom_value is None else round(float(bottom_value), 6),
            "rep_median": None
            if not rep_values
            else round(float(np.median(rep_values)), 6),
            "horizontal_fraction": None
            if not rep_values
            else round(float(np.mean(np.asarray(rep_values) <= threshold)), 6),
            "valid_frames": len(rep_values),
        }
        if bottom_value is None:
            accepted = True
            reason = "posture_unavailable"
        elif rule == "max":
            accepted = float(bottom_value) <= threshold
            reason = None if accepted else "exercise_posture_mismatch"
        elif rule == "min":
            accepted = float(bottom_value) >= threshold
            reason = None if accepted else "exercise_posture_mismatch"
        else:
            raise ValueError(f"unsupported posture rule: {rule}")

        rep_evidence.append(
            {
                "candidate_rep_id": old_rep_id,
                "accepted": accepted,
                "reason": reason,
                **evidence,
            }
        )
        if accepted:
            new_id = len(kept) + 1
            old_to_new[old_rep_id] = new_id
            accepted_rep = dict(rep)
            accepted_rep["rep_id"] = new_id
            kept.append(accepted_rep)
            continue

        rejected.append(
            {
                "candidate_rep_id": old_rep_id,
                "start_frame": int(rep["start_frame"]),
                "end_frame": int(rep["end_frame"]),
                "start_time_s": float(rep["start_time_s"]),
                "end_time_s": float(rep["end_time_s"]),
                "quality_flags": reason,
                "reject_reasons": reason,
                "evidence": evidence,
            }
        )

    segmentation["reps"] = kept
    segmentation["phases"] = [
        {**phase, "rep_id": old_to_new[int(phase["rep_id"])]}
        for phase in segmentation.get("phases", [])
        if int(phase["rep_id"]) in old_to_new
    ]
    segmentation["rejected_reps"] = list(segmentation.get("rejected_reps", [])) + rejected
    if not kept:
        segmentation["status"] = "no_reps"
    segmentation["candidate_sources"] = {
        **segmentation.get("candidate_sources", {}),
        "posture_rejected_rep_count": len(rejected),
    }
    segmentation["posture_gate"] = {
        "schema_version": POSTURE_GATE_SCHEMA_VERSION,
        "metric": gate["metric"],
        "rule": rule,
        "threshold": threshold,
        "candidate_rep_count": len(rep_evidence),
        "accepted_rep_count": len(kept),
        "rejected_rep_count": len(rejected),
        "reps": rep_evidence,
    }
    return segmentation


def segment_with_posture_gate(
    clip: dict[str, Any],
    specs: dict[str, Any],
    fps: float,
    angles_dir: Path,
) -> dict[str, Any]:
    """Choose the posture-gated segmentation with the most valid reps.

    The legacy scorer can lock onto a warm-up or floor-transition movement and
    miss the actual exercise later in the clip.  Re-segmenting on the exercise
    phase signal is a cheap second candidate; the posture gate decides whether
    either candidate contains usable reps.
    """
    signal_selection = select_view_phase_signal(clip, angles_dir)
    if signal_selection is not None:
        signal_angle = signal_selection["selected_angle"]
        if signal_angle is None:
            return {
                "status": "no_usable_side_signal",
                "angle_selection": None,
                "reps": [],
                "rejected_reps": [],
                "phases": [],
                "non_rep_intervals": [],
                "active_bouts": [],
                "method": None,
                "signal_selection": signal_selection,
                "segmentation_candidates": {
                    "schema_version": SEGMENTATION_CANDIDATES_SCHEMA_VERSION,
                    "selection_rule": "view_aware_phase_signal",
                    "view_aware": {
                        "angle": None,
                        "selected_side": None,
                        "selection_reason": signal_selection["selection_reason"],
                    },
                },
            }
        selected = apply_exercise_posture_gate(
            segment_clip(
                clip,
                specs,
                fps,
                angle_override=signal_angle,
                phase_signal_override=signal_angle,
            ),
            clip,
            angles_dir,
            specs,
        )
        selected["signal_selection"] = signal_selection
        selected["segmentation_candidates"] = {
            "schema_version": SEGMENTATION_CANDIDATES_SCHEMA_VERSION,
            "selection_rule": "view_aware_phase_signal",
            "view_aware": {
                "angle": signal_angle,
                "selected_side": signal_selection["selected_side"],
                "selection_reason": signal_selection["selection_reason"],
            },
        }
        return selected

    primary = apply_exercise_posture_gate(
        segment_clip(clip, specs, fps), clip, angles_dir, specs
    )
    phase_angle = specs["exercises"][clip["exercise"]].get("phase", {}).get("signal")
    if not phase_angle or phase_angle == primary.get("angle"):
        return primary

    alternative = apply_exercise_posture_gate(
        segment_clip(clip, specs, fps, angle_override=str(phase_angle)),
        clip,
        angles_dir,
        specs,
    )
    primary_count = len(primary.get("reps", []))
    alternative_count = len(alternative.get("reps", []))
    selected = alternative if alternative_count > primary_count else primary
    selected["segmentation_candidates"] = {
        "schema_version": SEGMENTATION_CANDIDATES_SCHEMA_VERSION,
        "selection_rule": "max_posture_gated_valid_reps",
        "primary": {
            "angle": primary.get("angle"),
            "valid_rep_count": primary_count,
            "rejected_rep_count": len(primary.get("rejected_reps", [])),
        },
        "phase_signal": {
            "angle": alternative.get("angle"),
            "valid_rep_count": alternative_count,
            "rejected_rep_count": len(alternative.get("rejected_reps", [])),
        },
        "selected": "phase_signal" if selected is alternative else "primary",
    }
    return selected


def build_quality(
    clip: dict[str, Any],
    pose_quality: dict[str, Any],
    segmentation: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    warnings = []
    if pose_quality["frame_coverage"] < 1.0:
        failures.append("frame_coverage_incomplete")
    if pose_quality["pose_detection_rate"] < POSE_DETECTION_REJECT:
        failures.append("pose_detection_rate_below_reject_threshold")
    elif pose_quality["pose_detection_rate"] < POSE_DETECTION_PASS:
        warnings.append("pose_detection_rate_below_pass_threshold")
    if pose_quality["q10_key_visibility"] < VISIBILITY_Q10_REJECT:
        failures.append("key_visibility_q10_below_reject_threshold")
    elif pose_quality["q10_key_visibility"] < VISIBILITY_Q10_PASS:
        warnings.append("key_visibility_q10_below_pass_threshold")
    if pose_quality["body_in_frame_rate"] < BODY_IN_FRAME_REJECT:
        failures.append("body_in_frame_rate_below_reject_threshold")
    elif pose_quality["body_in_frame_rate"] < BODY_IN_FRAME_PASS:
        warnings.append("body_in_frame_rate_below_pass_threshold")
    if clip["view"] == "unknown":
        failures.append("view_unknown")
    if not clip["usable_metrics"]:
        failures.append("no_usable_metrics")
    rep_count = len(segmentation.get("reps", []))
    posture_rejected = int(
        segmentation.get("posture_gate", {}).get("rejected_rep_count", 0)
    )
    if posture_rejected:
        warnings.append("exercise_posture_reps_rejected")
    if rep_count == 0:
        warnings.append("no_complete_reps")
    elif rep_count < MIN_COMPLETE_REPS:
        warnings.append("fewer_than_three_complete_reps")
    status = "reject" if failures else ("warn" if warnings else "pass")
    signal_selection = segmentation.get("signal_selection") or {}
    side_signal_rejected = bool(
        signal_selection and signal_selection.get("selected_angle") is None
    )
    if side_signal_rejected:
        warnings.append("side_phase_signal_rejected")
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "side_signal_rejected": side_signal_rejected,
        "checks": {
            "frame_coverage": round(float(pose_quality["frame_coverage"]), 6),
            "pose_detection_rate": round(float(pose_quality["pose_detection_rate"]), 6),
            "mean_key_visibility": round(float(pose_quality["mean_key_visibility"]), 6),
            "q10_key_visibility": round(float(pose_quality["q10_key_visibility"]), 6),
            "key_joint_visible_rate": round(float(pose_quality["key_joint_visible_rate"]), 6),
            "all_exercise_joints_visible_rate": round(
                float(pose_quality["all_exercise_joints_visible_rate"]), 6
            ),
            "all_required_visible_rate": round(float(pose_quality["all_required_visible_rate"]), 6),
            "body_in_frame_rate": round(float(pose_quality["body_in_frame_rate"]), 6),
            "rep_count": rep_count,
            "candidate_rep_count": len(segmentation.get("reps", []))
            + len(segmentation.get("rejected_reps", [])),
            "rejected_rep_count": len(segmentation.get("rejected_reps", [])),
            "camera_motion": {"status": "not_evaluated"},
        },
        "thresholds": {
            "pose_detection_pass": POSE_DETECTION_PASS,
            "pose_detection_reject": POSE_DETECTION_REJECT,
            "visibility_q10_pass": VISIBILITY_Q10_PASS,
            "visibility_q10_reject": VISIBILITY_Q10_REJECT,
            "body_in_frame_pass": BODY_IN_FRAME_PASS,
            "body_in_frame_reject": BODY_IN_FRAME_REJECT,
            "min_complete_reps": MIN_COMPLETE_REPS,
            "phase_boundary_level_min": PHASE_BOUNDARY_LEVEL_MIN,
            "min_rep_width_ratio": MIN_REP_WIDTH_RATIO,
            "max_rep_width_ratio": MAX_REP_WIDTH_RATIO,
        },
        "visibility_notes": {
            "key_joint_visible_rate": (
                "fraction of all exercise-joint samples with visibility >= 0.5"
            ),
            "all_exercise_joints_visible_rate": (
                "strict diagnostic: all exercise joints visible in the same frame; "
                "not a hard gate"
            ),
            "all_required_visible_rate": "compatibility alias of all_exercise_joints_visible_rate",
        },
        "view_support": {
            "view": clip["view"],
            "usable_metrics": clip["usable_metrics"],
            "disabled_metrics": clip["disabled_metrics"],
            "sagittal_metrics_disabled": any(
                metric
                in {
                    "knee_flexion_rom",
                    "bottom_knee_angle",
                    "hip_flexion_rom",
                    "torso_lean_bottom",
                    "hip_depth_ratio",
                    "elbow_flexion_rom",
                    "bottom_elbow_angle",
                    "body_line_angle",
                    "hip_sag_ratio",
                }
                for metric in clip["disabled_metrics"]
            ),
        },
    }


def write_reps_csv(path: Path, reps: list[dict[str, Any]]) -> None:
    fields = [
        "rep_id",
        "start_frame",
        "end_frame",
        "top_frame",
        "bottom_frame",
        "start_time_s",
        "end_time_s",
        "bottom_time_s",
        "clip_start_time_s",
        "clip_end_time_s",
        "clip_bottom_time_s",
        "duration_s",
        "eccentric_start_frame",
        "eccentric_end_frame",
        "concentric_start_frame",
        "concentric_end_frame",
        "phase_angle",
        "top_angle_deg",
        "start_angle_deg",
        "end_angle_deg",
        "bottom_angle_deg",
        "confidence",
        "visibility",
        "shape_similarity",
        "phase_consistency",
        "motion_evidence",
        "review_score",
        "review_priority",
        "reject_reason",
        "width_ratio",
        "start_phase_level",
        "end_phase_level",
        "quality_flags",
        "status",
        "reject_reasons",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reps)


def write_phases_csv(path: Path, phases: list[dict[str, Any]]) -> None:
    fields = [
        "rep_id",
        "phase",
        "frame_index",
        "clip_frame_index",
        "time_s",
        "clip_time_s",
        "angle_name",
        "angle_deg",
        "confidence",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(phases)


def write_rejected_reps_csv(path: Path, rejected_reps: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_rep_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "quality_flags",
        "reject_reasons",
        "evidence",
    ]
    rows = []
    for index, item in enumerate(rejected_reps, start=1):
        row = {key: item.get(key) for key in fields}
        if row["candidate_rep_id"] is None:
            row["candidate_rep_id"] = item.get("rep_id", index)
        row["evidence"] = json.dumps(item.get("evidence", {}), ensure_ascii=False, sort_keys=True)
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_one(
    clip: dict[str, Any],
    specs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = args.output_dir / clip["clip_id"]
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work_dir = output_dir / ".work" / f"{clip['clip_id']}_{os.getpid()}"
    pose_source_dir = work_dir / "pose"
    angle_staging_parent = work_dir / "angles"
    angles_dir = output_dir / "angles"
    started = time.perf_counter()

    landmarks_path, pose_meta_path, pose_quality = extract_clip_pose(
        clip, args, pose_source_dir
    )
    pose_meta = json.loads(pose_meta_path.read_text())
    fps = float(pose_meta["fps"])
    clip_snapshot = {
        **clip,
        "fps": fps,
        "processed_trim": pose_meta["trim"],
    }
    write_json(output_dir / "clip.json", clip_snapshot)
    exercise_payload = {
        "exercise_hint": clip["exercise"],
        "action_display": clip["action_display"],
        "source": "manifest",
        "classifier_run": False,
        "consistency_check": "deferred_to_task1_4e",
        "confidence": 1.0,
    }
    write_json(output_dir / "exercise.json", exercise_payload)

    angle_meta = run_angle_extraction(
        clip["clip_id"],
        pose_source_dir,
        angle_staging_parent,
        angles_dir,
        clip["view"],
        clip,
    )
    clip["_angles_dir"] = str(angles_dir)
    segmentation = segment_with_posture_gate(clip, specs, fps, angles_dir)
    quality = build_quality(clip, pose_quality, segmentation)
    rep_count = len(segmentation.get("reps", []))
    rejected_rep_count = len(segmentation.get("rejected_reps", []))
    if quality["status"] == "reject":
        analysis_status = "rejected_quality"
    elif segmentation["status"] != "ok" or rep_count < MIN_COMPLETE_REPS:
        analysis_status = "partial"
    else:
        analysis_status = "ok"

    elapsed = time.perf_counter() - started
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "task1-4b",
        "clip_id": clip["clip_id"],
        "status": analysis_status,
        "exercise": clip["exercise"],
        "view": clip["view"],
        "split": clip.get("split"),
        "rotation": clip["rotation"],
        "trim": clip["trim"],
        "time_base": "source_video_zero",
        "fps": fps,
        "frame_count": pose_meta["processed_frames"],
        "rep_count": rep_count,
        "rejected_rep_count": rejected_rep_count,
        "quality_status": quality["status"],
        "quality": quality,
        "segmentation": segmentation,
        "pose": pose_quality,
        "angle_extraction": {
            "angle_count": angle_meta["angle_count"],
            "processing_seconds": angle_meta["processing_seconds"],
            "outputs": angle_meta["outputs"],
        },
        "outputs": {
            "clip": str((output_dir / "clip.json").resolve()),
            "quality": str((output_dir / "quality.json").resolve()),
            "exercise": str((output_dir / "exercise.json").resolve()),
            "reps": str((output_dir / "reps.csv").resolve()),
            "rejected_reps": str((output_dir / "rejected_reps.csv").resolve()),
            "phases": str((output_dir / "phases.csv").resolve()),
            "angles": str((angles_dir / "angles.csv").resolve()),
            "angles_wide": str((angles_dir / "angles_wide.csv").resolve()),
            "landmarks": str((angles_dir / "landmarks.json").resolve()),
        },
        "processing_seconds": elapsed,
    }
    write_json(output_dir / "quality.json", quality)
    write_json(output_dir / "analysis.json", analysis)
    write_reps_csv(output_dir / "reps.csv", segmentation.get("reps", []))
    write_rejected_reps_csv(
        output_dir / "rejected_reps.csv", segmentation.get("rejected_reps", [])
    )
    write_phases_csv(output_dir / "phases.csv", segmentation.get("phases", []))
    shutil.rmtree(work_dir, ignore_errors=True)
    return analysis


def summary_entry(analysis: dict[str, Any]) -> dict[str, Any]:
    quality = analysis.get("quality", {})
    segmentation = analysis.get("segmentation", {})
    return {
        "clip_id": analysis["clip_id"],
        "exercise": analysis.get("exercise"),
        "view": analysis.get("view"),
        "split": analysis.get("split"),
        "status": analysis["status"],
        "quality_status": analysis["quality_status"],
        "rep_count": analysis["rep_count"],
        "rejected_rep_count": analysis.get("rejected_rep_count", 0),
        "angle": segmentation.get("angle"),
        "phase_angle": segmentation.get("phase_angle"),
        "method": segmentation.get("method"),
        "quality_warnings": quality.get("warnings", []),
        "quality_failures": quality.get("failures", []),
    }


def rebuild_summary(output_dir: Path) -> dict[str, Any]:
    """Rebuild the aggregate summary from all completed clip directories."""
    results = []
    for analysis_path in sorted(output_dir.glob("*/analysis.json")):
        try:
            analysis = json.loads(analysis_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if analysis.get("schema_version") != SCHEMA_VERSION:
            continue
        results.append(summary_entry(analysis))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "quality_status_counts": dict(Counter(item["quality_status"] for item in results)),
        "complete_rep_count": sum(item["rep_count"] for item in results),
        "rejected_rep_count": sum(item["rejected_rep_count"] for item in results),
        "results": results,
    }


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.manifest = args.manifest.resolve()
    args.specs = args.specs.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    manifest = load_manifest(args.manifest)
    specs = json.loads(args.specs.read_text())
    clips = select_clips(manifest, args.clip_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in clips:
        print(f"analyzing {clip['clip_id']}", flush=True)
        analysis = analyze_one(clip, specs, args)
        results.append(analysis)
        print(
            f"finished {clip['clip_id']}: status={analysis['status']} "
            f"quality={analysis['quality_status']} reps={analysis['rep_count']}",
            flush=True,
        )
    summary = rebuild_summary(args.output_dir)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
