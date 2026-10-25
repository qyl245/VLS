#!/usr/bin/env python3
"""Extract biomechanical angle features from task1-1 MediaPipe landmarks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "outputs" / "pose_landmarker_full"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "angles"
DEFAULT_SELECTION = ROOT / "configs" / "task1_1_selection.json"

VISIBILITY_THRESHOLD = 0.5
OUTLIER_THRESHOLD_DEG = 30.0
SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2
SPINE_WINDOW = 15


@dataclass(frozen=True)
class AngleDefinition:
    name: str
    joint: str
    feature: str
    plane: str
    unit: str
    landmarks: tuple[int, ...]
    description: str


ANGLE_DEFINITIONS = (
    AngleDefinition("left_shoulder_flexion", "left_shoulder", "flexion", "3d", "deg", (11, 13, 23), "Angle at the left shoulder between elbow and hip."),
    AngleDefinition("right_shoulder_flexion", "right_shoulder", "flexion", "3d", "deg", (12, 14, 24), "Angle at the right shoulder between elbow and hip."),
    AngleDefinition("left_elbow_flexion", "left_elbow", "flexion", "3d", "deg", (13, 11, 15), "Angle at the left elbow between shoulder and wrist."),
    AngleDefinition("right_elbow_flexion", "right_elbow", "flexion", "3d", "deg", (14, 12, 16), "Angle at the right elbow between shoulder and wrist."),
    AngleDefinition("left_wrist_flexion", "left_wrist", "flexion", "3d", "deg", (15, 13, 19), "Angle at the left wrist between elbow and index finger."),
    AngleDefinition("right_wrist_flexion", "right_wrist", "flexion", "3d", "deg", (16, 14, 20), "Angle at the right wrist between elbow and index finger."),
    AngleDefinition("left_hip_flexion", "left_hip", "flexion", "3d", "deg", (23, 11, 25), "Angle at the left hip between shoulder and knee."),
    AngleDefinition("right_hip_flexion", "right_hip", "flexion", "3d", "deg", (24, 12, 26), "Angle at the right hip between shoulder and knee."),
    AngleDefinition("left_knee_flexion", "left_knee", "flexion", "3d", "deg", (25, 23, 27), "Angle at the left knee between hip and ankle."),
    AngleDefinition("right_knee_flexion", "right_knee", "flexion", "3d", "deg", (26, 24, 28), "Angle at the right knee between hip and ankle."),
    AngleDefinition("torso_flexion", "torso", "flexion", "3d", "deg", (11, 12, 23, 24, 27, 28), "Angle between the shoulder-to-hip torso vector and hip-to-ankle gravity reference."),
    AngleDefinition("spine_stability", "spine", "stability", "3d", "variance", (11, 12, 23, 24), "Rolling variance of the unit torso vector over a centered window."),
    AngleDefinition("left_knee_valgus_coronal", "left_knee", "valgus_deviation", "coronal", "deg", (23, 25, 27), "Deviation from a straight knee in the body-frame coronal plane."),
    AngleDefinition("right_knee_valgus_coronal", "right_knee", "valgus_deviation", "coronal", "deg", (24, 26, 28), "Deviation from a straight knee in the body-frame coronal plane."),
    AngleDefinition("left_ankle_dorsiflexion_sagittal", "left_ankle", "dorsiflexion", "sagittal", "deg", (25, 27), "Angle of the left shank projection relative to the body vertical in the sagittal plane."),
    AngleDefinition("right_ankle_dorsiflexion_sagittal", "right_ankle", "dorsiflexion", "sagittal", "deg", (26, 28), "Angle of the right shank projection relative to the body vertical in the sagittal plane."),
    AngleDefinition("left_shoulder_abduction_coronal", "left_shoulder", "abduction", "coronal", "deg", (11, 13), "Left upper-arm angle relative to body vertical in the coronal plane."),
    AngleDefinition("right_shoulder_abduction_coronal", "right_shoulder", "abduction", "coronal", "deg", (12, 14), "Right upper-arm angle relative to body vertical in the coronal plane."),
    AngleDefinition("left_shoulder_horizontal_abduction_transverse", "left_shoulder", "horizontal_abduction", "transverse", "deg", (11, 13), "Left upper-arm angle relative to the lateral axis in the transverse plane."),
    AngleDefinition("right_shoulder_horizontal_abduction_transverse", "right_shoulder", "horizontal_abduction", "transverse", "deg", (12, 14), "Right upper-arm angle relative to the lateral axis in the transverse plane."),
    AngleDefinition("left_shoulder_extension_sagittal", "left_shoulder", "extension", "sagittal", "deg", (11, 13), "Left upper-arm angle relative to body vertical in the sagittal plane."),
    AngleDefinition("right_shoulder_extension_sagittal", "right_shoulder", "extension", "sagittal", "deg", (12, 14), "Right upper-arm angle relative to body vertical in the sagittal plane."),
    AngleDefinition("left_hip_flexion_sagittal", "left_hip", "flexion", "sagittal", "deg", (11, 23, 25), "Left hip flexion from the torso and thigh sagittal projections."),
    AngleDefinition("right_hip_flexion_sagittal", "right_hip", "flexion", "sagittal", "deg", (12, 24, 26), "Right hip flexion from the torso and thigh sagittal projections."),
    AngleDefinition("knee_flexion_asymmetry", "bilateral_knees", "flexion_asymmetry", "3d", "deg", (25, 26), "Absolute difference between left and right knee flexion."),
    AngleDefinition("wrist_height_difference", "bilateral_wrists", "height_difference", "body_y", "m", (15, 16), "Absolute left-right wrist height difference in the body frame."),
    AngleDefinition("hip_flexion_asymmetry", "bilateral_hips", "flexion_asymmetry", "3d", "deg", (23, 24), "Absolute difference between left and right hip flexion."),
)
ANGLE_NAMES = tuple(item.name for item in ANGLE_DEFINITIONS)
ANGLE_BY_NAME = {item.name: item for item in ANGLE_DEFINITIONS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--visibility-threshold", type=float, default=VISIBILITY_THRESHOLD)
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--smooth-polyorder", type=int, default=SMOOTH_POLYORDER)
    parser.add_argument("--outlier-threshold-deg", type=float, default=OUTLIER_THRESHOLD_DEG)
    parser.add_argument("--spine-window", type=int, default=SPINE_WINDOW)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.smooth_window < 3 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window must be an odd integer >= 3")
    if args.smooth_polyorder >= args.smooth_window:
        parser.error("--smooth-polyorder must be smaller than --smooth-window")
    if args.spine_window < 3:
        parser.error("--spine-window must be >= 3")
    return args


def load_video_ids(selection_path: Path, videos: list[str] | None) -> list[str]:
    if videos:
        return [Path(name).stem for name in videos]
    payload = json.loads(selection_path.read_text())
    return [entry["video_id"] for entry in payload["videos"]]


def unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def angle_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-10:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def make_body_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    left_shoulder = points[11]
    right_shoulder = points[12]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (points[23] + points[24]) / 2.0
    y_axis = unit(hip_mid - shoulder_mid)
    if y_axis is None:
        return None
    x_raw = right_shoulder - left_shoulder
    x_axis = unit(x_raw - np.dot(x_raw, y_axis) * y_axis)
    if x_axis is None:
        return None
    z_axis = unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    axes = np.stack([x_axis, y_axis, z_axis], axis=0)
    body_points = (points - hip_mid) @ axes.T
    return body_points, axes


def project(vector: np.ndarray, plane: str) -> np.ndarray:
    if plane == "coronal":
        return np.array([vector[0], vector[1], 0.0])
    if plane == "sagittal":
        return np.array([0.0, vector[1], vector[2]])
    if plane == "transverse":
        return np.array([vector[0], 0.0, vector[2]])
    raise ValueError(plane)


def init_frame_arrays(count: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    raw = {name: np.full(count, np.nan, dtype=np.float64) for name in ANGLE_NAMES}
    confidence = {name: np.full(count, np.nan, dtype=np.float64) for name in ANGLE_NAMES}
    reasons = {name: ["" for _ in range(count)] for name in ANGLE_NAMES}
    return raw, confidence, reasons


def set_three_point_angle(
    raw: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    reasons: dict[str, list[str]],
    frame_index: int,
    name: str,
    points: np.ndarray,
    visibility: np.ndarray,
    center: int,
    first: int,
    second: int,
    threshold: float,
) -> None:
    conf = float(np.min(visibility[[center, first, second]]))
    confidence[name][frame_index] = conf
    if conf < threshold:
        reasons[name][frame_index] = "low_visibility"
        return
    value = angle_deg(points[first] - points[center], points[second] - points[center])
    if value is None:
        reasons[name][frame_index] = "degenerate_vector"
        return
    raw[name][frame_index] = value


def set_projected_angle(
    raw: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    reasons: dict[str, list[str]],
    frame_index: int,
    name: str,
    body_points: np.ndarray | None,
    visibility: np.ndarray,
    landmarks: tuple[int, ...],
    vector_builder: Any,
    reference: np.ndarray,
    plane: str,
    threshold: float,
    complement: bool = False,
    line_angle: bool = False,
) -> None:
    conf = float(np.min(visibility[list(landmarks)]))
    confidence[name][frame_index] = conf
    if conf < threshold:
        reasons[name][frame_index] = "low_visibility"
        return
    if body_points is None:
        reasons[name][frame_index] = "body_frame_unavailable"
        return
    vector = vector_builder(body_points)
    projected = project(vector, plane)
    if float(np.linalg.norm(projected)) < 1e-8:
        reasons[name][frame_index] = "degenerate_projection"
        return
    value = angle_deg(projected, reference)
    if value is None:
        reasons[name][frame_index] = "degenerate_vector"
        return
    if complement:
        value = 180.0 - value
    if line_angle:
        value = min(value, 180.0 - value)
    raw[name][frame_index] = value


def set_two_vector_projected_angle(
    raw: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    reasons: dict[str, list[str]],
    frame_index: int,
    name: str,
    body_points: np.ndarray | None,
    visibility: np.ndarray,
    landmarks: tuple[int, ...],
    first_builder: Any,
    second_builder: Any,
    plane: str,
    threshold: float,
    complement: bool = False,
) -> None:
    conf = float(np.min(visibility[list(landmarks)]))
    confidence[name][frame_index] = conf
    if conf < threshold:
        reasons[name][frame_index] = "low_visibility"
        return
    if body_points is None:
        reasons[name][frame_index] = "body_frame_unavailable"
        return
    first = project(first_builder(body_points), plane)
    second = project(second_builder(body_points), plane)
    if float(np.linalg.norm(first)) < 1e-8 or float(np.linalg.norm(second)) < 1e-8:
        reasons[name][frame_index] = "degenerate_projection"
        return
    value = angle_deg(first, second)
    if value is None:
        reasons[name][frame_index] = "degenerate_vector"
        return
    raw[name][frame_index] = 180.0 - value if complement else value


def compute_record(
    record: dict[str, Any],
    frame_index: int,
    raw: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    reasons: dict[str, list[str]],
    threshold: float,
    spine_vectors: np.ndarray,
    spine_confidence: np.ndarray,
) -> np.ndarray | None:
    if not record.get("detected"):
        for name in ANGLE_NAMES:
            reasons[name][frame_index] = "no_pose"
        return None

    points = np.asarray(record["landmarks_world"], dtype=np.float64)
    visibility = np.asarray(record["visibility"], dtype=np.float64)
    body = make_body_frame(points)
    body_points = body[0] if body is not None else None

    joint_map = {
        "left_shoulder_flexion": (11, 13, 23),
        "right_shoulder_flexion": (12, 14, 24),
        "left_elbow_flexion": (13, 11, 15),
        "right_elbow_flexion": (14, 12, 16),
        "left_wrist_flexion": (15, 13, 19),
        "right_wrist_flexion": (16, 14, 20),
        "left_hip_flexion": (23, 11, 25),
        "right_hip_flexion": (24, 12, 26),
        "left_knee_flexion": (25, 23, 27),
        "right_knee_flexion": (26, 24, 28),
    }
    for name, indices in joint_map.items():
        set_three_point_angle(
            raw, confidence, reasons, frame_index, name, points, visibility, *indices, threshold
        )

    torso_conf = float(np.min(visibility[[11, 12, 23, 24, 27, 28]]))
    confidence["torso_flexion"][frame_index] = torso_conf
    shoulder_mid = (points[11] + points[12]) / 2.0
    hip_mid = (points[23] + points[24]) / 2.0
    ankle_mid = (points[27] + points[28]) / 2.0
    torso_vector = hip_mid - shoulder_mid
    if torso_conf < threshold:
        reasons["torso_flexion"][frame_index] = "low_visibility"
    else:
        value = angle_deg(torso_vector, ankle_mid - hip_mid)
        if value is None:
            reasons["torso_flexion"][frame_index] = "degenerate_vector"
        else:
            raw["torso_flexion"][frame_index] = value

    torso_spine_conf = float(np.min(visibility[[11, 12, 23, 24]]))
    spine_confidence[frame_index] = torso_spine_conf
    if torso_spine_conf >= threshold:
        normalized_torso = unit(torso_vector)
        if normalized_torso is not None:
            spine_vectors[frame_index] = normalized_torso

    if body_points is None:
        for name in (
            "left_knee_valgus_coronal",
            "right_knee_valgus_coronal",
            "left_ankle_dorsiflexion_sagittal",
            "right_ankle_dorsiflexion_sagittal",
            "left_shoulder_abduction_coronal",
            "right_shoulder_abduction_coronal",
            "left_shoulder_horizontal_abduction_transverse",
            "right_shoulder_horizontal_abduction_transverse",
            "left_shoulder_extension_sagittal",
            "right_shoulder_extension_sagittal",
            "left_hip_flexion_sagittal",
            "right_hip_flexion_sagittal",
            "wrist_height_difference",
        ):
            if confidence[name][frame_index] != confidence[name][frame_index]:
                confidence[name][frame_index] = float(
                    np.min(visibility[list(ANGLE_BY_NAME[name].landmarks)])
                )
            reasons[name][frame_index] = "body_frame_unavailable"
    else:
        set_two_vector_projected_angle(
            raw, confidence, reasons, frame_index, "left_knee_valgus_coronal", body_points,
            visibility, (23, 25, 27), lambda p: p[23] - p[25], lambda p: p[27] - p[25],
            "coronal", threshold, complement=True,
        )
        set_two_vector_projected_angle(
            raw, confidence, reasons, frame_index, "right_knee_valgus_coronal", body_points,
            visibility, (24, 26, 28), lambda p: p[24] - p[26], lambda p: p[28] - p[26],
            "coronal", threshold, complement=True,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "left_ankle_dorsiflexion_sagittal", body_points,
            visibility, (25, 27), lambda p: p[25] - p[27], np.array([0.0, -1.0, 0.0]),
            "sagittal", threshold,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "right_ankle_dorsiflexion_sagittal", body_points,
            visibility, (26, 28), lambda p: p[26] - p[28], np.array([0.0, -1.0, 0.0]),
            "sagittal", threshold,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "left_shoulder_abduction_coronal", body_points,
            visibility, (11, 13), lambda p: p[13] - p[11], np.array([0.0, 1.0, 0.0]),
            "coronal", threshold,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "right_shoulder_abduction_coronal", body_points,
            visibility, (12, 14), lambda p: p[14] - p[12], np.array([0.0, 1.0, 0.0]),
            "coronal", threshold,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "left_shoulder_horizontal_abduction_transverse",
            body_points, visibility, (11, 13), lambda p: p[13] - p[11], np.array([1.0, 0.0, 0.0]),
            "transverse", threshold, line_angle=True,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "right_shoulder_horizontal_abduction_transverse",
            body_points, visibility, (12, 14), lambda p: p[14] - p[12], np.array([1.0, 0.0, 0.0]),
            "transverse", threshold, line_angle=True,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "left_shoulder_extension_sagittal", body_points,
            visibility, (11, 13), lambda p: p[13] - p[11], np.array([0.0, 1.0, 0.0]),
            "sagittal", threshold,
        )
        set_projected_angle(
            raw, confidence, reasons, frame_index, "right_shoulder_extension_sagittal", body_points,
            visibility, (12, 14), lambda p: p[14] - p[12], np.array([0.0, 1.0, 0.0]),
            "sagittal", threshold,
        )
        set_two_vector_projected_angle(
            raw, confidence, reasons, frame_index, "left_hip_flexion_sagittal", body_points,
            visibility, (11, 23, 25), lambda p: p[11] - p[23], lambda p: p[25] - p[23],
            "sagittal", threshold, complement=True,
        )
        set_two_vector_projected_angle(
            raw, confidence, reasons, frame_index, "right_hip_flexion_sagittal", body_points,
            visibility, (12, 24, 26), lambda p: p[12] - p[24], lambda p: p[26] - p[24],
            "sagittal", threshold, complement=True,
        )
        confidence["wrist_height_difference"][frame_index] = float(np.min(visibility[[15, 16]]))
        if confidence["wrist_height_difference"][frame_index] < threshold:
            reasons["wrist_height_difference"][frame_index] = "low_visibility"
        else:
            raw["wrist_height_difference"][frame_index] = abs(body_points[15, 1] - body_points[16, 1])

    for name, left_name, right_name in (
        ("knee_flexion_asymmetry", "left_knee_flexion", "right_knee_flexion"),
        ("hip_flexion_asymmetry", "left_hip_flexion", "right_hip_flexion"),
    ):
        confidence[name][frame_index] = float(
            np.nanmin([confidence[left_name][frame_index], confidence[right_name][frame_index]])
        )
        left_value = raw[left_name][frame_index]
        right_value = raw[right_name][frame_index]
        if np.isfinite(left_value) and np.isfinite(right_value):
            raw[name][frame_index] = abs(float(left_value - right_value))
        else:
            reasons[name][frame_index] = "upstream_nan"

    return body_points


def smooth_series(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    result = values.copy()
    valid = np.isfinite(values)
    start = None
    for index in range(len(values) + 1):
        is_valid = index < len(values) and valid[index]
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            end = index
            if end - start >= window:
                result[start:end] = savgol_filter(
                    values[start:end], window_length=window, polyorder=polyorder, mode="interp"
                )
            start = None
    return result


def remove_angle_jumps(
    values: np.ndarray,
    reasons: list[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    outliers = np.zeros(len(values), dtype=bool)
    previous_valid = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            previous_valid = None
            continue
        if (
            previous_valid is not None
            and index == previous_valid + 1
            and abs(value - values[previous_valid]) > threshold
        ):
            result[index] = np.nan
            outliers[index] = True
            reasons[index] = f"angle_jump_gt_{threshold:g}deg"
            previous_valid = None
            continue
        previous_valid = index
    return result, outliers


def clip_physical_range(values: np.ndarray, unit: str) -> np.ndarray:
    result = values.copy()
    if unit == "deg":
        result = np.clip(result, 0.0, 180.0)
    elif unit in {"m", "variance"}:
        result = np.clip(result, 0.0, None)
    return result


def compute_spine_stability(
    spine_vectors: np.ndarray,
    spine_confidence: np.ndarray,
    threshold: float,
    window: int,
    reasons: list[str],
) -> np.ndarray:
    count = len(spine_vectors)
    result = np.full(count, np.nan, dtype=np.float64)
    half = window // 2
    for index in range(count):
        if not np.isfinite(spine_confidence[index]) or spine_confidence[index] < threshold:
            if not reasons[index]:
                reasons[index] = "low_visibility"
            continue
        start = max(0, index - half)
        end = min(count, index + half + 1)
        vectors = spine_vectors[start:end]
        vectors = vectors[np.isfinite(vectors).all(axis=1)]
        if len(vectors) < 3:
            reasons[index] = "insufficient_window"
            continue
        result[index] = float(np.var(vectors, axis=0).sum())
    return result


def format_number(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def write_vocabulary(output_dir: Path) -> None:
    payload = {
        "task": "task1-2",
        "version": "1.0",
        "angles": [asdict(item) for item in ANGLE_DEFINITIONS],
    }
    (output_dir / "angle_vocabulary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "angle_vocabulary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "joint", "feature", "plane", "unit", "landmarks", "description"])
        for item in ANGLE_DEFINITIONS:
            writer.writerow(
                [
                    item.name,
                    item.joint,
                    item.feature,
                    item.plane,
                    item.unit,
                    " ".join(str(index) for index in item.landmarks),
                    item.description,
                ]
            )


def process_video(video_id: str, args: argparse.Namespace) -> dict[str, Any]:
    source_landmarks = args.source_dir / f"{video_id}_landmarks.json"
    source_meta_path = args.source_dir / f"{video_id}_meta.json"
    if not source_landmarks.exists() or not source_meta_path.exists():
        raise FileNotFoundError(f"missing task1-1 output for {video_id}")

    output_dir = args.output_dir / video_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    source_meta = json.loads(source_meta_path.read_text())
    records = json.loads(source_landmarks.read_text())
    count = len(records)
    raw, confidence, reasons = init_frame_arrays(count)
    spine_vectors = np.full((count, 3), np.nan, dtype=np.float64)
    spine_confidence = np.full(count, np.nan, dtype=np.float64)

    body_path = output_dir / "landmarks_body_frame.json"
    with body_path.open("w") as handle:
        handle.write("[\n")
        for index, record in enumerate(records):
            body_points = compute_record(
                record,
                index,
                raw,
                confidence,
                reasons,
                args.visibility_threshold,
                spine_vectors,
                spine_confidence,
            )
            payload = {
                "frame_id": int(record["frame_id"]),
                "timestamp_ms": int(record["timestamp_ms"]),
                "landmarks_body_frame": None
                if body_points is None
                else [[round(float(value), 6) for value in point] for point in body_points],
                "source": "vision",
                "detected": body_points is not None,
            }
            if index:
                handle.write(",\n")
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n]\n")

    raw["spine_stability"] = compute_spine_stability(
        spine_vectors,
        spine_confidence,
        args.visibility_threshold,
        args.spine_window,
        reasons["spine_stability"],
    )
    confidence["spine_stability"] = spine_confidence

    final = {}
    outliers = {}
    for definition in ANGLE_DEFINITIONS:
        smoothed = smooth_series(raw[definition.name], args.smooth_window, args.smooth_polyorder)
        smoothed = clip_physical_range(smoothed, definition.unit)
        if definition.unit == "deg":
            smoothed, outlier_mask = remove_angle_jumps(
                smoothed, reasons[definition.name], args.outlier_threshold_deg
            )
        else:
            outlier_mask = np.zeros(count, dtype=bool)
        final[definition.name] = smoothed
        outliers[definition.name] = outlier_mask

    angles_long_path = output_dir / "angles.csv"
    with angles_long_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "timestamp_ms",
                "joint",
                "feature",
                "angle_name",
                "plane",
                "unit",
                "angle",
                "angle_raw",
                "confidence",
                "source",
                "valid",
                "outlier",
                "invalid_reason",
            ]
        )
        for index, record in enumerate(records):
            for definition in ANGLE_DEFINITIONS:
                value = final[definition.name][index]
                writer.writerow(
                    [
                        record["frame_id"],
                        record["timestamp_ms"],
                        definition.joint,
                        definition.feature,
                        definition.name,
                        definition.plane,
                        definition.unit,
                        format_number(value),
                        format_number(raw[definition.name][index]),
                        format_number(confidence[definition.name][index]),
                        "vision",
                        int(np.isfinite(value)),
                        int(outliers[definition.name][index]),
                        reasons[definition.name][index],
                    ]
                )

    wide_path = output_dir / "angles_wide.csv"
    with wide_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "timestamp_ms", *ANGLE_NAMES])
        for index, record in enumerate(records):
            writer.writerow(
                [
                    record["frame_id"],
                    record["timestamp_ms"],
                    *[format_number(final[name][index]) for name in ANGLE_NAMES],
                ]
            )

    quality = []
    for definition in ANGLE_DEFINITIONS:
        valid = np.isfinite(final[definition.name])
        reason_counts = Counter(
            reason for reason in reasons[definition.name] if reason
        )
        quality.append(
            {
                "name": definition.name,
                "unit": definition.unit,
                "valid_frames": int(valid.sum()),
                "nan_frames": int((~valid).sum()),
                "nan_ratio": float((~valid).mean()),
                "outlier_count": int(outliers[definition.name].sum()),
                "low_visibility_count": int(reason_counts.get("low_visibility", 0)),
                "reason_counts": dict(reason_counts),
            }
        )

    shutil.copyfile(source_landmarks, output_dir / "landmarks.json")
    elapsed = time.perf_counter() - started
    meta = {
        "task": "task1-2",
        "schema_version": "task1-2.v1",
        "video_id": video_id,
        "source_task1_1_landmarks": str(source_landmarks.resolve()),
        "source_task1_1_meta": str(source_meta_path.resolve()),
        "source": "vision",
        "fps": source_meta["fps"],
        "stride": source_meta["stride"],
        "downsample": {"stride": source_meta["stride"], "effective_fps": source_meta["fps"] / source_meta["stride"]},
        "frame_count": count,
        "view_label": "unknown",
        "visibility_threshold": args.visibility_threshold,
        "smoothing": {
            "method": "savitzky_golay",
            "window": args.smooth_window,
            "polyorder": args.smooth_polyorder,
            "mode": "interp",
            "missing_policy": "segment-wise smoothing; no interpolation into missing frames",
        },
        "outlier_rule": {
            "unit": "deg",
            "threshold_per_frame": args.outlier_threshold_deg,
            "applied_after_smoothing": True,
        },
        "spine_stability": {
            "window_frames": args.spine_window,
            "definition": "sum of component variances of the unit torso vector",
        },
        "body_frame": {
            "origin": "hip midpoint",
            "x_axis": "right shoulder - left shoulder, orthogonalized against y",
            "y_axis": "hip midpoint - shoulder midpoint",
            "z_axis": "cross(x_axis, y_axis)",
            "units": "m",
            "gram_schmidt": True,
        },
        "angle_count": len(ANGLE_DEFINITIONS),
        "quality": quality,
        "processing_seconds": elapsed,
        "outputs": {
            "landmarks": str((output_dir / "landmarks.json").resolve()),
            "landmarks_body_frame": str(body_path.resolve()),
            "angles": str(angles_long_path.resolve()),
            "angles_wide": str(wide_path.resolve()),
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def main() -> int:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    video_ids = load_video_ids(args.selection.resolve(), args.videos)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vocabulary(args.output_dir)

    metas = []
    for video_id in video_ids:
        print(f"extracting angles for {video_id}", flush=True)
        meta = process_video(video_id, args)
        metas.append(meta)
        print(
            f"finished {video_id}: {meta['frame_count']} frames, {meta['angle_count']} angles, "
            f"{meta['processing_seconds']:.2f}s",
            flush=True,
        )

    with (args.output_dir / "quality_report.csv").open("w", newline="") as handle:
        fields = [
            "video_id",
            "angle_name",
            "unit",
            "valid_frames",
            "nan_frames",
            "nan_ratio",
            "outlier_count",
            "low_visibility_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for meta in metas:
            for row in meta["quality"]:
                writer.writerow({"video_id": meta["video_id"], "angle_name": row["name"], **{key: row[key] for key in fields[2:]}})

    summary = {
        "task": "task1-2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "angle_count": len(ANGLE_DEFINITIONS),
        "videos": metas,
        "totals": {
            "videos": len(metas),
            "frames": sum(meta["frame_count"] for meta in metas),
            "angle_rows": sum(meta["frame_count"] for meta in metas) * len(ANGLE_DEFINITIONS),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
