#!/usr/bin/env python3
"""Screen QEVD clips and build the task1-4 clip manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from paths import DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = DATA_ROOT / "feedbacks_long_range.json"
DEFAULT_VIDEO_DIR = DATA_ROOT / "long_range_videos"
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "task1_4" / "clip_screening"
DEFAULT_MANIFEST = ROOT / "configs" / "task1_4_clips.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_OVERRIDES = ROOT / "configs" / "task1_4_screening_overrides.json"

ROTATION_CODES = {
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
ROTATION_ORDER = ["none", "90cw", "180", "90ccw"]
VIEWS = {
    "front",
    "rear",
    "oblique_left",
    "oblique_right",
    "side_left",
    "side_right",
    "unknown",
}
FRONTAL_VIEWS = {"front", "rear", "oblique_left", "oblique_right"}
SIDE_VIEWS = {"side_left", "side_right", "oblique_left", "oblique_right"}

ROTATION_SAMPLE_FRACTIONS = (0.08, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90)
VIEW_SAMPLE_FRACTIONS = (0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.90)
CONTACT_FRACTIONS = (0.25, 0.50, 0.75)

KEY_LANDMARKS = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32)

DEV_VIDEOS = {"0020", "0034", "0046", "0104", "0156", "0001", "0004", "0005", "0007", "0021", "0161"}
HOLDOUT_VIDEOS = {
    "0153",
    "0075",
    "0125",
    "0184",
    "0160",
    "0044",
    "0065",
    "0067",
    "0090",
    "0103",
    "0137",
    "0143",
}
RESERVE_CLIPS = {"0153_pushup_set01"}

SQUAT_BASE_METRICS = ("rep_duration_s", "rep_duration_cv", "spine_stability_bottom")
SQUAT_SIDE_METRICS = (
    "knee_flexion_rom",
    "bottom_knee_angle",
    "hip_flexion_rom",
    "torso_lean_bottom",
    "hip_depth_ratio",
)
SQUAT_FRONTAL_METRICS = (
    "knee_valgus_bottom",
    "knee_asymmetry_bottom",
    "hip_asymmetry_bottom",
)
PUSHUP_BASE_METRICS = ("rep_duration_s", "rep_duration_cv", "spine_stability_bottom")
PUSHUP_SIDE_METRICS = (
    "elbow_flexion_rom",
    "bottom_elbow_angle",
    "body_line_angle",
    "hip_sag_ratio",
)
PUSHUP_FRONTAL_METRICS = ("elbow_asymmetry_bottom", "shoulder_asymmetry_bottom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def clip_id(video_id: str, exercise: str) -> str:
    return f"{video_id}_{exercise}_set01"


def load_candidates(feedback_path: Path, video_dir: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import segment_reps

    payload = json.loads(feedback_path.read_text())
    video_ids = [Path(item["long_range_video_file"]).stem for item in payload]
    segments = segment_reps.load_feedback_segments(feedback_path, video_ids)
    candidates = []
    for segment in segments:
        if segment.action_display == "squats":
            exercise = "squat"
        elif segment.action_display == "pushups":
            exercise = "pushup"
        else:
            continue
        video_path = video_dir / f"{segment.video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        candidates.append(
            {
                "clip_id": clip_id(segment.video_id, exercise),
                "video_id": segment.video_id,
                "video_path": str(video_path),
                "exercise": exercise,
                "action_display": segment.action_display,
                "trim": {
                    "start_frame": int(segment.start_frame),
                    "end_frame": int(segment.end_frame),
                },
                "fps": fps,
                "frame_count": frame_count,
                "source_size": {"width": width, "height": height},
            }
        )
    candidates.sort(key=lambda item: (item["exercise"], item["video_id"]))
    return candidates


def sample_indexes(start: int, end: int, fractions: tuple[float, ...]) -> list[int]:
    indexes = sorted({int(round(start + (end - start) * fraction)) for fraction in fractions})
    return [int(np.clip(index, start, end)) for index in indexes]


def read_frames(video_path: Path, indexes: list[int]) -> tuple[list[int], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    kept_indexes: list[int] = []
    frames: list[np.ndarray] = []
    for index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            kept_indexes.append(index)
            frames.append(frame)
    cap.release()
    return kept_indexes, frames


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    code = ROTATION_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def make_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def make_landmarker(args: argparse.Namespace) -> Any:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_detection_confidence,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def clipped_ratio(numerator: float, denominator: float, scale: float = 1.0) -> float:
    return float(np.clip(abs(numerator) / max(abs(denominator), 1e-6) / scale, 0.0, 1.0))


def positive_ratio(value: float, scale: float = 1.0) -> float:
    return float(np.clip(value / max(scale, 1e-6), 0.0, 1.0))


def horizontal_alignment(a: Any, b: Any) -> float:
    return 1.0 - clipped_ratio(a.y - b.y, a.x - b.x, 1.0)


def pose_view_features(result: Any, width: int, height: int) -> dict[str, float] | None:
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    visibility = float(np.mean([landmarks[index].visibility for index in KEY_LANDMARKS]))
    aspect = width / max(height, 1)

    def point(index: int) -> tuple[float, float]:
        landmark = landmarks[index]
        return float(landmark.x) * aspect, float(landmark.y)

    def midpoint(first: int, second: int) -> tuple[float, float]:
        a = point(first)
        b = point(second)
        return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0

    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    sh_l = point(11)
    sh_r = point(12)
    hip_l = point(23)
    hip_r = point(24)
    knee_l = point(25)
    knee_r = point(26)
    ankle_l = point(27)
    ankle_r = point(28)
    wrist_l = point(15)
    wrist_r = point(16)
    head = point(0)
    sh_center = midpoint(11, 12)
    hip_center = midpoint(23, 24)
    ankle_center = midpoint(27, 28)
    wrist_center = midpoint(15, 16)
    torso_length = max(distance(sh_center, hip_center), 1e-6)
    shoulder_width = distance(sh_l, sh_r)
    hip_width = distance(hip_l, hip_r)
    return {
        "visibility": visibility,
        "shoulder_width_ratio": shoulder_width / torso_length,
        "hip_width_ratio": hip_width / torso_length,
        "width_ratio": ((shoulder_width + hip_width) / 2.0) / torso_length,
        "shoulder_tilt": abs(sh_l[1] - sh_r[1]) / max(abs(sh_l[0] - sh_r[0]), 1e-6),
        "hip_tilt": abs(hip_l[1] - hip_r[1]) / max(abs(hip_l[0] - hip_r[0]), 1e-6),
        "nose_offset": (head[0] - sh_center[0]) / torso_length,
        "head_above_shoulder": (sh_center[1] - head[1]) / torso_length,
        "shoulder_above_hip": (hip_center[1] - sh_center[1]) / torso_length,
        "hip_above_ankle": (ankle_center[1] - hip_center[1]) / torso_length,
        "wrist_below_shoulder": (wrist_center[1] - sh_center[1]) / torso_length,
        "head_above_wrists": (wrist_center[1] - head[1]) / torso_length,
        "ankle_x_span_ratio": abs(ankle_l[0] - ankle_r[0]) / torso_length,
        "knee_x_span_ratio": abs(knee_l[0] - knee_r[0]) / torso_length,
        "body_x_span_ratio": (max(sh_l[0], sh_r[0], hip_l[0], hip_r[0], ankle_l[0], ankle_r[0]) - min(sh_l[0], sh_r[0], hip_l[0], hip_r[0], ankle_l[0], ankle_r[0])) / torso_length,
    }


def rotation_shape_score(exercise: str, features: dict[str, float]) -> dict[str, float]:
    shoulder_alignment = 1.0 - float(np.clip(features["shoulder_tilt"], 0.0, 1.0))
    hip_alignment = 1.0 - float(np.clip(features["hip_tilt"], 0.0, 1.0))
    if exercise == "squat":
        upright = float(
            features["head_above_shoulder"] > 0.05
            and features["shoulder_above_hip"] > 0.05
            and features["hip_above_ankle"] > 0.05
        )
        ankles_below = positive_ratio(features["hip_above_ankle"], 0.5)
        score = 0.35 * upright + 0.20 * shoulder_alignment + 0.15 * hip_alignment + 0.15 * ankles_below + 0.15 * features["visibility"]
        return {
            "upright": upright,
            "shoulder_alignment": shoulder_alignment,
            "hip_alignment": hip_alignment,
            "ankles_below": ankles_below,
            "score": score,
        }
    wrists_below = positive_ratio(features["wrist_below_shoulder"], 0.5)
    head_above_wrists = positive_ratio(features["head_above_wrists"], 0.5)
    score = (
        0.25 * shoulder_alignment
        + 0.15 * hip_alignment
        + 0.15 * wrists_below
        + 0.30 * head_above_wrists
        + 0.15 * features["visibility"]
    )
    return {
        "shoulder_alignment": shoulder_alignment,
        "hip_alignment": hip_alignment,
        "wrists_below": wrists_below,
        "head_above_wrists": head_above_wrists,
        "score": score,
    }


def evaluate_rotation(
    landmarker: Any,
    frames: list[np.ndarray],
    exercise: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for rotation in ROTATION_ORDER:
        detections = 0
        visibilities: list[float] = []
        shape_parts: defaultdict[str, list[float]] = defaultdict(list)
        for frame in frames:
            rotated = rotate_frame(frame, rotation)
            result = landmarker.detect(make_image(rotated))
            features = pose_view_features(result, width, height)
            if features is None:
                continue
            detections += 1
            visibilities.append(features["visibility"])
            for key, value in rotation_shape_score(exercise, features).items():
                shape_parts[key].append(float(value))
        detection_rate = detections / max(len(frames), 1)
        shape_mean = {key: float(np.mean(values)) for key, values in shape_parts.items()}
        score = 0.45 * detection_rate + 0.55 * shape_mean.get("score", 0.0)
        scores[rotation] = {
            "score": round(score, 6),
            "detection_rate": round(detection_rate, 6),
            "mean_visibility": round(float(np.mean(visibilities)), 6) if visibilities else 0.0,
            **{key: round(value, 6) for key, value in shape_mean.items() if key != "score"},
            "shape_score": round(shape_mean.get("score", 0.0), 6),
        }
    ranked = sorted(ROTATION_ORDER, key=lambda name: scores[name]["score"], reverse=True)
    selected = ranked[0]
    margin = scores[selected]["score"] - scores[ranked[1]]["score"]
    return {
        "selected": selected,
        "margin": round(float(margin), 6),
        "scores": scores,
        "confidence": round(float(np.clip(margin / 0.10, 0.0, 1.0)), 6),
    }


def median_feature(features: list[dict[str, float]], key: str) -> float:
    return float(np.median([item[key] for item in features]))


def classify_view(features: list[dict[str, float]]) -> dict[str, Any]:
    if not features:
        return {
            "view": "unknown",
            "reason": "no_pose",
            "confidence": 0.0,
            "width_ratio": None,
            "shoulder_tilt": None,
            "nose_offset": None,
        }
    width_ratio = median_feature(features, "width_ratio")
    shoulder_tilt = median_feature(features, "shoulder_tilt")
    nose_offset = median_feature(features, "nose_offset")
    if width_ratio <= 0.62:
        family = "side"
        distance_to_boundary = 0.62 - width_ratio
    elif width_ratio >= 1.05 and shoulder_tilt <= 0.45:
        family = "front"
        distance_to_boundary = min(width_ratio - 1.05, 0.45 - shoulder_tilt)
    elif width_ratio <= 0.92:
        family = "oblique"
        distance_to_boundary = min(width_ratio - 0.62, 0.92 - width_ratio)
    else:
        family = "front"
        distance_to_boundary = width_ratio - 0.92
    if family == "front":
        view = "front"
    else:
        direction = "left" if nose_offset < 0 else "right"
        view = f"{family}_{direction}"
    confidence = float(np.clip(distance_to_boundary / 0.20, 0.0, 1.0))
    return {
        "view": view,
        "reason": "auto",
        "confidence": round(confidence, 6),
        "width_ratio": round(width_ratio, 6),
        "shoulder_tilt": round(shoulder_tilt, 6),
        "nose_offset": round(nose_offset, 6),
    }


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return dict(payload.get("clips", {}))


def metrics_for_view(exercise: str, view: str) -> tuple[list[str], list[str]]:
    if exercise == "squat":
        base, side, frontal = SQUAT_BASE_METRICS, SQUAT_SIDE_METRICS, SQUAT_FRONTAL_METRICS
    else:
        base, side, frontal = PUSHUP_BASE_METRICS, PUSHUP_SIDE_METRICS, PUSHUP_FRONTAL_METRICS
    all_metrics = tuple(dict.fromkeys((*base, *side, *frontal)))
    usable = set(base)
    if view in SIDE_VIEWS:
        usable.update(side)
    if view in FRONTAL_VIEWS:
        usable.update(frontal)
    disabled = [metric for metric in all_metrics if metric not in usable]
    return [metric for metric in all_metrics if metric in usable], disabled


def split_for_video(video_id: str) -> tuple[str, bool]:
    if video_id in DEV_VIDEOS:
        return "dev", True
    if video_id in HOLDOUT_VIDEOS:
        return "holdout", True
    return "reserve", False


def annotate_tile(frame: np.ndarray, lines: list[str], tile_width: int = 320) -> np.ndarray:
    tile_height = int(tile_width * frame.shape[0] / frame.shape[1])
    tile = cv2.resize(frame, (tile_width, tile_height))
    footer_height = 18 * len(lines) + 8
    canvas = np.full((tile_height + footer_height, tile_width, 3), 20, dtype=np.uint8)
    canvas[:tile_height] = tile
    for row, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (6, tile_height + 16 + row * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_contact_sheet(
    rows: list[list[np.ndarray]],
    labels: list[list[str]],
    path: Path,
    tile_width: int = 320,
    columns: int = 1,
) -> None:
    if not rows:
        return
    item_rows = []
    for row_frames, row_labels in zip(rows, labels):
        row_tiles = [annotate_tile(frame, label, tile_width) for frame, label in zip(row_frames, row_labels)]
        height = max(tile.shape[0] for tile in row_tiles)
        row = np.full((height, tile_width * len(row_tiles), 3), 20, dtype=np.uint8)
        for column, tile in enumerate(row_tiles):
            row[: tile.shape[0], column * tile_width : (column + 1) * tile_width] = tile
        item_rows.append(row)
    tiles = []
    for start in range(0, len(item_rows), columns):
        group = item_rows[start : start + columns]
        width = max(row.shape[1] for row in group)
        height = max(row.shape[0] for row in group)
        row = np.full((height, width * columns, 3), 20, dtype=np.uint8)
        for column, tile in enumerate(group):
            row[: tile.shape[0], column * width : column * width + tile.shape[1]] = tile
        tiles.append(row)
    width = max(row.shape[1] for row in tiles)
    sheet = np.full((sum(row.shape[0] for row in tiles), width, 3), 20, dtype=np.uint8)
    offset = 0
    for row in tiles:
        sheet[offset : offset + row.shape[0], : row.shape[1]] = row
        offset += row.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_specs() -> dict[str, Any]:
    return {
        "schema_version": "task1-4-exercise-specs-v1",
        "view_definitions": {
            "front": "frontal camera; left/right joints visible for asymmetry checks",
            "rear": "rear camera; asymmetry checks only",
            "oblique_left": "camera obliquely ahead-left; supports side and selected frontal checks",
            "oblique_right": "camera obliquely ahead-right; supports side and selected frontal checks",
            "side_left": "camera on the subject's left or subject facing image-left; sagittal checks",
            "side_right": "camera on the subject's right or subject facing image-right; sagittal checks",
            "unknown": "unresolved view; view-dependent metrics must be disabled",
        },
        "metric_groups": {
            "base": ["rep_duration_s", "rep_duration_cv", "spine_stability_bottom"],
            "squat_side": list(SQUAT_SIDE_METRICS),
            "squat_frontal": list(SQUAT_FRONTAL_METRICS),
            "pushup_side": list(PUSHUP_SIDE_METRICS),
            "pushup_frontal": list(PUSHUP_FRONTAL_METRICS),
        },
        "exercises": {
            "squat": {
                "action_display": "squats",
                "rep_unit": "cycle",
                "min_complete_reps": 3,
                "phase": {
                    "top": "standing, knee extended, rep boundary",
                    "bottom": "minimum knee angle",
                    "signal": "left_knee_flexion",
                    "bottom_polarity": "trough",
                },
                "metrics": {
                    "rep_duration_s": {"unit": "s", "views": "any"},
                    "rep_duration_cv": {"unit": "ratio", "views": "any", "min_reps": 3},
                    "spine_stability_bottom": {"unit": "variance", "views": "any"},
                    "knee_flexion_rom": {"unit": "deg", "views": "side_or_oblique"},
                    "bottom_knee_angle": {"unit": "deg", "views": "side_or_oblique"},
                    "hip_flexion_rom": {"unit": "deg", "views": "side_or_oblique"},
                    "torso_lean_bottom": {"unit": "deg", "views": "side_or_oblique"},
                    "hip_depth_ratio": {"unit": "ratio", "views": "side_or_oblique", "confidence": "low"},
                    "knee_valgus_bottom": {"unit": "deg", "views": "frontal_or_oblique"},
                    "knee_asymmetry_bottom": {"unit": "deg", "views": "frontal_or_oblique"},
                    "hip_asymmetry_bottom": {"unit": "deg", "views": "frontal_or_oblique"},
                },
                "issues": {
                    "squat_depth_insufficient": ["knee_flexion_rom", "hip_depth_ratio"],
                    "squat_torso_lean_excess": ["torso_lean_bottom"],
                    "squat_knee_valgus": ["knee_valgus_bottom"],
                    "squat_asymmetry": ["knee_asymmetry_bottom", "hip_asymmetry_bottom"],
                    "squat_tempo_inconsistent": ["rep_duration_cv"],
                    "squat_stability_low": ["spine_stability_bottom"],
                },
            },
            "pushup": {
                "action_display": "pushups",
                "rep_unit": "cycle",
                "min_complete_reps": 3,
                "phase": {
                    "top": "arms extended, rep boundary",
                    "bottom": "minimum elbow angle",
                    "signal": "left_elbow_flexion",
                    "bottom_polarity": "trough",
                },
                "metrics": {
                    "rep_duration_s": {"unit": "s", "views": "any"},
                    "rep_duration_cv": {"unit": "ratio", "views": "any", "min_reps": 3},
                    "spine_stability_bottom": {"unit": "variance", "views": "any"},
                    "elbow_flexion_rom": {"unit": "deg", "views": "side_or_oblique"},
                    "bottom_elbow_angle": {"unit": "deg", "views": "side_or_oblique"},
                    "body_line_angle": {"unit": "deg", "views": "side_or_oblique"},
                    "hip_sag_ratio": {"unit": "ratio", "views": "side_or_oblique", "confidence": "low"},
                    "elbow_asymmetry_bottom": {"unit": "deg", "views": "frontal_or_oblique"},
                    "shoulder_asymmetry_bottom": {"unit": "deg", "views": "frontal_or_oblique", "confidence": "low"},
                },
                "issues": {
                    "pushup_depth_insufficient": ["elbow_flexion_rom", "bottom_elbow_angle"],
                    "pushup_body_line_break": ["body_line_angle"],
                    "pushup_hip_sag": ["hip_sag_ratio"],
                    "pushup_asymmetry": ["elbow_asymmetry_bottom", "shoulder_asymmetry_bottom"],
                    "pushup_tempo_inconsistent": ["rep_duration_cv"],
                    "pushup_stability_low": ["spine_stability_bottom"],
                },
            },
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    clips = manifest["clips"]
    ids = [item["clip_id"] for item in clips]
    if len(ids) != len(set(ids)):
        errors.append("clip_id is not unique")
    video_splits: defaultdict[str, set[str]] = defaultdict(set)
    for item in clips:
        start = item["trim"]["start_frame"]
        end = item["trim"]["end_frame"]
        if not 0 <= start < end < item["frame_count"]:
            errors.append(f"{item['clip_id']}: invalid trim")
        if item["action_display"] not in {"squats", "pushups"}:
            errors.append(f"{item['clip_id']}: unexpected action_display")
        if item["rotation"] not in ROTATION_ORDER:
            errors.append(f"{item['clip_id']}: invalid rotation")
        if item["view"] not in VIEWS:
            errors.append(f"{item['clip_id']}: invalid view")
        overlap = set(item["usable_metrics"]) & set(item["disabled_metrics"])
        if overlap:
            errors.append(f"{item['clip_id']}: metrics both usable and disabled: {sorted(overlap)}")
        video_splits[item["video_id"]].add(item["split"])
    for video_id, splits in video_splits.items():
        if len(splits) != 1:
            errors.append(f"video {video_id} crosses splits: {sorted(splits)}")
    selected = [item for item in clips if item["selected_for_phase"]]
    counts = Counter((item["exercise"], item["phase_role"]) for item in selected)
    expected = {
        ("squat", "dev"): 5,
        ("squat", "holdout"): 5,
        ("pushup", "dev"): 8,
        ("pushup", "holdout"): 8,
    }
    for key, value in expected.items():
        if counts[key] != value:
            errors.append(f"selected count {key}: expected {value}, got {counts[key]}")
    return errors


def main() -> None:
    args = parse_args()
    overrides = load_overrides(args.overrides)
    candidates = load_candidates(args.feedback, args.video_dir)
    landmarker = make_landmarker(args)
    rotation_records: dict[str, Any] = {}
    view_rows: list[dict[str, Any]] = []
    manifest_clips: list[dict[str, Any]] = []
    contact_rows: list[list[np.ndarray]] = []
    contact_labels: list[list[str]] = []

    try:
        for index, candidate in enumerate(candidates, start=1):
            cid = candidate["clip_id"]
            start = candidate["trim"]["start_frame"]
            end = candidate["trim"]["end_frame"]
            width = candidate["source_size"]["width"]
            height = candidate["source_size"]["height"]
            rotation_indexes = sample_indexes(start, end, ROTATION_SAMPLE_FRACTIONS)
            _, rotation_frames = read_frames(Path(candidate["video_path"]), rotation_indexes)
            rotation_result = evaluate_rotation(
                landmarker, rotation_frames, candidate["exercise"], width, height
            )
            view_indexes = sample_indexes(start, end, VIEW_SAMPLE_FRACTIONS)
            _, view_frames = read_frames(Path(candidate["video_path"]), view_indexes)
            override = overrides.get(cid, {})
            rotation = override.get("rotation", rotation_result["selected"])
            features = []
            for frame in view_frames:
                result = landmarker.detect(make_image(rotate_frame(frame, rotation)))
                pose_features = pose_view_features(result, width, height)
                if pose_features is not None:
                    features.append(pose_features)
            view_result = classify_view(features)
            auto_view = view_result["view"]
            view = override.get("view", auto_view)
            view_source = "manual_override" if "view" in override else "auto"
            usable_metrics, disabled_metrics = metrics_for_view(candidate["exercise"], view)
            split, selected_for_phase = split_for_video(candidate["video_id"])
            selected_for_phase = selected_for_phase and cid not in RESERVE_CLIPS
            phase_role = split if selected_for_phase else "reserve"
            detection_rate = float(np.mean([item["visibility"] for item in features])) if features else 0.0
            quality_flags = []
            if rotation_result["margin"] < 0.03:
                quality_flags.append("rotation_margin_low")
            if len(features) / max(len(view_frames), 1) < 0.75:
                quality_flags.append("pose_detection_low")
            if view_source == "auto" and view_result["confidence"] < 0.25:
                quality_flags.append("view_confidence_low")
            if view == "unknown":
                quality_flags.append("view_unknown")
            notes = override.get("notes", "")
            manifest_clips.append(
                {
                    **candidate,
                    "rotation": rotation,
                    "view": view,
                    "auto_view": auto_view,
                    "view_source": view_source,
                    "split": split,
                    "phase_role": phase_role,
                    "selected_for_phase": selected_for_phase,
                    "usable_metrics": usable_metrics,
                    "disabled_metrics": disabled_metrics,
                    "quality": {
                        "rotation_confidence": rotation_result["confidence"],
                        "rotation_margin": rotation_result["margin"],
                        "pose_detection_rate": round(len(features) / max(len(view_frames), 1), 6),
                        "mean_visible_fraction": round(detection_rate, 6),
                        "view_confidence": view_result["confidence"],
                        "width_ratio": view_result["width_ratio"],
                        "shoulder_tilt": view_result["shoulder_tilt"],
                        "nose_offset": view_result["nose_offset"],
                        "quality_flags": quality_flags,
                    },
                    "notes": notes,
                }
            )
            rotation_records[cid] = {
                "exercise": candidate["exercise"],
                "video_id": candidate["video_id"],
                "trim": candidate["trim"],
                "sample_frames": rotation_indexes,
                **rotation_result,
                "override": override.get("rotation"),
            }
            view_rows.append(
                {
                    "clip_id": cid,
                    "exercise": candidate["exercise"],
                    "view": view,
                    "auto_view": auto_view,
                    "view_source": view_source,
                    "split": split,
                    "phase_role": phase_role,
                    "selected_for_phase": selected_for_phase,
                    "rotation": rotation,
                    "rotation_margin": rotation_result["margin"],
                    "view_confidence": view_result["confidence"],
                    "width_ratio": view_result["width_ratio"],
                    "shoulder_tilt": view_result["shoulder_tilt"],
                    "nose_offset": view_result["nose_offset"],
                    "pose_detection_rate": round(len(features) / max(len(view_frames), 1), 6),
                    "usable_metrics": ";".join(usable_metrics),
                    "disabled_metrics": ";".join(disabled_metrics),
                    "quality_flags": ";".join(quality_flags),
                    "notes": notes,
                }
            )

            contact_index = sample_indexes(start, end, (0.50,))[0]
            _, contact_frames = read_frames(Path(candidate["video_path"]), [contact_index])
            if contact_frames:
                contact_rows.append([rotate_frame(contact_frames[0], rotation)])
                contact_labels.append(
                    [
                        [
                            f"{cid} | {rotation} | {view}",
                            f"{split} | selected={selected_for_phase}",
                        ]
                    ]
                )
            print(
                f"[{index:02d}/{len(candidates)}] {cid}: rotation={rotation} "
                f"margin={rotation_result['margin']:.3f} view={view} "
                f"conf={view_result['confidence']:.2f} split={split}",
                flush=True,
            )
    finally:
        landmarker.close()

    selected_side_clips = [
        item["clip_id"]
        for item in manifest_clips
        if item["selected_for_phase"] and item["view"] in SIDE_VIEWS
    ]
    manifest = {
        "schema_version": "task1-4-v1",
        "task": "task1-4a",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "QEVD-FIT-COACH",
        "source_feedback": str(args.feedback),
        "selection_policy": {
            "action_filter": "action_display in {squats, pushups}",
            "excluded_actions": ["squat jumps", "squat kicks"],
            "split_unit": "video_id",
            "selected_for_phase": {
                "squat": {"dev": 5, "holdout": 5},
                "pushup": {"dev": 8, "holdout": 8},
            },
            "reserve_policy": "remaining clips retained in manifest but excluded from dev and holdout",
            "reserve_clips": sorted(RESERVE_CLIPS),
        },
        "video_splits": {
            video_id: split_for_video(video_id)[0]
            for video_id in sorted({item["video_id"] for item in manifest_clips})
        },
        "screening_findings": {
            "candidate_view_counts": dict(sorted(Counter(item["view"] for item in manifest_clips).items())),
            "selected_view_counts": dict(
                sorted(Counter(item["view"] for item in manifest_clips if item["selected_for_phase"]).items())
            ),
            "selected_side_or_oblique_clips": selected_side_clips,
            "sagittal_metric_status": "available" if selected_side_clips else "disabled_for_all_selected_clips",
            "note": "QEVD pure squat/pushup windows are predominantly frontal; side-view depth, torso-lean, and body-line metrics require supplemental clips.",
        },
        "clips": manifest_clips,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError("manifest validation failed:\n" + "\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.manifest, manifest)
    write_json(args.specs, build_specs())
    write_json(args.output_dir / "rotation_scores.json", {"clips": rotation_records})
    write_csv(
        args.output_dir / "view_screening.csv",
        view_rows,
        [
            "clip_id",
            "exercise",
            "view",
            "auto_view",
            "view_source",
            "split",
            "phase_role",
            "selected_for_phase",
            "rotation",
            "rotation_margin",
            "view_confidence",
            "width_ratio",
            "shoulder_tilt",
            "nose_offset",
            "pose_detection_rate",
            "usable_metrics",
            "disabled_metrics",
            "quality_flags",
            "notes",
        ],
    )
    preview_rows = [
        {
            **item,
            "start_frame": item["trim"]["start_frame"],
            "end_frame": item["trim"]["end_frame"],
            "usable_metrics": ";".join(item["usable_metrics"]),
            "disabled_metrics": ";".join(item["disabled_metrics"]),
        }
        for item in manifest_clips
    ]
    write_csv(
        args.output_dir / "manifest_preview.csv",
        preview_rows,
        [
            "clip_id",
            "video_id",
            "exercise",
            "action_display",
            "start_frame",
            "end_frame",
            "split",
            "phase_role",
            "selected_for_phase",
            "rotation",
            "view",
            "auto_view",
            "usable_metrics",
            "disabled_metrics",
        ],
    )
    summary_rows = []
    for exercise in ("squat", "pushup"):
        for phase_role in ("dev", "holdout", "reserve"):
            group = [
                item
                for item in manifest_clips
                if item["exercise"] == exercise and item["phase_role"] == phase_role
            ]
            summary_rows.append(
                {
                    "exercise": exercise,
                    "phase_role": phase_role,
                    "total_clips": len(group),
                    "selected_clips": sum(item["selected_for_phase"] for item in group),
                    "video_splits": ";".join(
                        f"{video_id}:{split}"
                        for video_id, split in sorted(
                            {item["video_id"]: item["split"] for item in group}.items()
                        )
                    ),
                    "views": ";".join(f"{key}:{value}" for key, value in sorted(Counter(item["view"] for item in group).items())),
                }
            )
    write_csv(
        args.output_dir / "split_summary.csv",
        summary_rows,
        ["exercise", "phase_role", "total_clips", "selected_clips", "video_splits", "views"],
    )
    build_contact_sheet(
        contact_rows,
        contact_labels,
        args.output_dir / "candidate_contact_sheet.jpg",
        columns=4,
    )

    for exercise in ("squat", "pushup"):
        rows = []
        labels = []
        for item in manifest_clips:
            if item["exercise"] != exercise or not item["selected_for_phase"]:
                continue
            indexes = sample_indexes(
                item["trim"]["start_frame"], item["trim"]["end_frame"], CONTACT_FRACTIONS
            )
            _, frames = read_frames(Path(item["video_path"]), indexes)
            if not frames:
                continue
            rows.append([rotate_frame(frame, item["rotation"]) for frame in frames])
            labels.append(
                [
                    [
                        f"{item['clip_id']} | {item['view']}",
                        f"{item['split']} | {item['rotation']}",
                    ]
                    for _ in frames
                ]
            )
        build_contact_sheet(rows, labels, args.output_dir / f"{exercise}_selected_contact_sheet.jpg")

    validation = {
        "status": "pass",
        "clip_count": len(manifest_clips),
        "selected_count": sum(item["selected_for_phase"] for item in manifest_clips),
        "errors": errors,
    }
    write_json(args.output_dir / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
