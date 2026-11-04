#!/usr/bin/env python3
"""Shared Fit3D GT and MediaPipe angle utilities for task1-5."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SIDE_VIEWS = {"side_left", "side_right", "oblique_left", "oblique_right"}
MIN_SIGNAL_RANGE_DEG = 8.0
MIN_SIGNAL_COVERAGE = 0.50
MIN_SIGNAL_VISIBILITY = 0.50

FIT3D_JOINTS = {
    "left_hip": 1,
    "left_knee": 2,
    "left_ankle": 3,
    "right_hip": 4,
    "right_knee": 5,
    "right_ankle": 6,
    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}

SIDE_SIGNAL_SPECS = {
    "squat": {
        "left": {
            "angle": "left_knee_flexion",
            "fit3d_joints": (1, 2, 3),
            "mediapipe_joints": (23, 25, 27),
        },
        "right": {
            "angle": "right_knee_flexion",
            "fit3d_joints": (4, 5, 6),
            "mediapipe_joints": (24, 26, 28),
        },
    },
    "pushup": {
        "left": {
            "angle": "left_elbow_flexion",
            "fit3d_joints": (11, 12, 13),
            "mediapipe_joints": (11, 13, 15),
        },
        "right": {
            "angle": "right_elbow_flexion",
            "fit3d_joints": (14, 15, 16),
            "mediapipe_joints": (12, 14, 16),
        },
    },
}

SIDE_DEPTH_JOINTS = {
    "squat": {"left": (1, 2, 3), "right": (4, 5, 6)},
    "pushup": {"left": (11, 12, 13), "right": (14, 15, 16)},
}

GT_ANGLE_SPECS = {
    "squat": {
        "left_knee_flexion": (1, 2, 3),
        "right_knee_flexion": (4, 5, 6),
        "left_hip_flexion": (11, 1, 2),
        "right_hip_flexion": (14, 4, 5),
        "torso_flexion": None,
    },
    "pushup": {
        "left_elbow_flexion": (11, 12, 13),
        "right_elbow_flexion": (14, 15, 16),
        "left_hip_flexion": (11, 1, 2),
        "right_hip_flexion": (14, 4, 5),
        "torso_flexion": None,
    },
}


def side_from_view(view: str, camera_side: str | None = None) -> str | None:
    if camera_side in {"left", "right"} and view in SIDE_VIEWS:
        return camera_side
    if view in {"side_left", "oblique_left"}:
        return "left"
    if view in {"side_right", "oblique_right"}:
        return "right"
    if view in {"front", "rear"}:
        return None
    return None


def near_side_from_camera_points(
    camera_points: np.ndarray,
    exercise: str,
    frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Infer the near side from GT joints already transformed to camera space."""
    if exercise not in SIDE_DEPTH_JOINTS:
        raise ValueError(f"unsupported exercise: {exercise}")
    points = np.asarray(camera_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] < 25 or points.shape[2] != 3:
        raise ValueError(f"unexpected camera-space joints shape: {points.shape}")
    if frame_indices is not None:
        points = points[np.asarray(frame_indices, dtype=np.int64)]
    if points.shape[0] == 0:
        raise ValueError("at least one frame is required for depth side audit")

    depths: dict[str, float] = {}
    for side, indices in SIDE_DEPTH_JOINTS[exercise].items():
        values = points[:, list(indices), 2]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"non-finite {side} camera depth")
        depths[side] = float(np.median(finite))
    near_side = "left" if depths["left"] < depths["right"] else "right"
    return {
        "near_side": near_side,
        "left_depth": round(depths["left"], 6),
        "right_depth": round(depths["right"], 6),
        "signed_depth_margin": round(depths["right"] - depths["left"], 6),
        "absolute_depth_margin": round(abs(depths["right"] - depths["left"]), 6),
    }


def angle_between_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cosine = np.sum(first * second, axis=-1) / np.maximum(denominator, 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def three_point_angle(points: np.ndarray, indices: tuple[int, int, int]) -> np.ndarray:
    center_point = points[..., indices[1], :]
    return angle_between_vectors(
        points[..., indices[0], :] - center_point,
        points[..., indices[2], :] - center_point,
    )


def fit3d_angle_series(
    joints: np.ndarray, exercise: str, angle_name: str
) -> np.ndarray:
    if exercise not in GT_ANGLE_SPECS:
        raise ValueError(f"unsupported exercise: {exercise}")
    if angle_name not in GT_ANGLE_SPECS[exercise]:
        raise ValueError(f"unsupported angle for {exercise}: {angle_name}")
    if joints.ndim != 3 or joints.shape[1] < 25 or joints.shape[2] != 3:
        raise ValueError(f"unexpected Fit3D joints shape: {joints.shape}")
    indices = GT_ANGLE_SPECS[exercise][angle_name]
    if indices is not None:
        return three_point_angle(joints, indices)

    shoulder_mid = (joints[:, 11] + joints[:, 14]) * 0.5
    hip_mid = (joints[:, 1] + joints[:, 4]) * 0.5
    ankle_mid = (joints[:, 3] + joints[:, 6]) * 0.5
    return angle_between_vectors(hip_mid - shoulder_mid, ankle_mid - hip_mid)


def finite_signal_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"coverage": 0.0, "range_deg": 0.0, "p10_deg": 0.0, "p90_deg": 0.0}
    p10, p90 = np.percentile(finite, [10.0, 90.0])
    return {
        "coverage": float(finite.size / max(1, len(values))),
        "range_deg": float(p90 - p10),
        "p10_deg": float(p10),
        "p90_deg": float(p90),
    }


def choose_side_phase_signal(
    exercise: str,
    view: str,
    camera_side: str | None,
    signals: dict[str, np.ndarray],
    visibility_scores: dict[str, float],
) -> dict[str, Any] | None:
    """Choose the camera-near phase signal for a side/oblique clip.

    A far-side signal is never substituted for the camera-near side. If the
    near side fails visibility, coverage, or dynamic-range checks, the clip is
    rejected for side measurement instead of producing a plausible-looking but
    semantically wrong angle.
    """
    if exercise not in SIDE_SIGNAL_SPECS or view not in SIDE_VIEWS:
        return None
    near_side = side_from_view(view, camera_side)
    candidates: dict[str, dict[str, Any]] = {}
    for side, spec in SIDE_SIGNAL_SPECS[exercise].items():
        signal = signals.get(spec["angle"])
        if signal is None:
            continue
        stats = finite_signal_stats(signal)
        visibility = float(visibility_scores.get(side, 0.0))
        eligible = (
            stats["coverage"] >= MIN_SIGNAL_COVERAGE
            and stats["range_deg"] >= MIN_SIGNAL_RANGE_DEG
            and visibility >= MIN_SIGNAL_VISIBILITY
        )
        candidates[side] = {
            "side": side,
            "angle": spec["angle"],
            "visibility_median": round(visibility, 6),
            "signal_coverage": round(stats["coverage"], 6),
            "signal_range_deg": round(stats["range_deg"], 3),
            "signal_p10_deg": round(stats["p10_deg"], 3),
            "signal_p90_deg": round(stats["p90_deg"], 3),
            "eligible": eligible,
            "score": round(stats["coverage"] * stats["range_deg"] * max(visibility, 0.0), 4),
        }
    if not candidates:
        return None

    if near_side in candidates:
        near = candidates[near_side]
    else:
        near = None

    selected: dict[str, Any] | None = None
    reason: str
    if near is None:
        reason = "camera_side_unknown"
    elif near["eligible"]:
        selected = near
        reason = "near_side_preferred"
    elif near["visibility_median"] < MIN_SIGNAL_VISIBILITY:
        reason = "insufficient_near_side_visibility"
    elif near["signal_coverage"] < MIN_SIGNAL_COVERAGE:
        reason = "insufficient_near_side_signal_coverage"
    else:
        reason = "insufficient_near_side_signal_range"

    low_confidence = selected is None
    return {
        "schema_version": "task1-5b-side-signal-v2",
        "exercise": exercise,
        "view": view,
        "camera_side": camera_side,
        "near_side": near_side,
        "selected_side": None if selected is None else selected["side"],
        "selected_angle": None if selected is None else selected["angle"],
        "selection_reason": reason,
        "low_confidence": low_confidence,
        "candidates": candidates,
        "far_side_substitution": False,
    }


def angle_error_stats(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    frame_count = min(len(pred), len(ref))
    pred = pred[:frame_count]
    ref = ref[:frame_count]
    valid = np.isfinite(pred) & np.isfinite(ref)
    count = int(valid.sum())
    if count == 0:
        return {
            "frame_count": frame_count,
            "valid_frames": 0,
            "valid_fraction": 0.0,
            "mae_deg": None,
            "bias_deg": None,
            "rmse_deg": None,
            "p90_abs_error_deg": None,
            "pearson_r": None,
        }
    error = pred[valid] - ref[valid]
    abs_error = np.abs(error)
    pearson_r = None
    if count >= 3 and np.std(pred[valid]) > 1e-9 and np.std(ref[valid]) > 1e-9:
        pearson_r = float(np.corrcoef(pred[valid], ref[valid])[0, 1])
    return {
        "frame_count": frame_count,
        "valid_frames": count,
        "valid_fraction": round(float(count / frame_count), 6) if frame_count else 0.0,
        "mae_deg": round(float(np.mean(abs_error)), 6),
        "bias_deg": round(float(np.mean(error)), 6),
        "rmse_deg": round(float(np.sqrt(np.mean(error**2))), 6),
        "p90_abs_error_deg": round(float(np.percentile(abs_error, 90.0)), 6),
        "pearson_r": None if pearson_r is None else round(pearson_r, 6),
    }


def cluster_bootstrap_ci(
    cluster_values: dict[str, float],
    seed: int = 20261026,
    samples: int = 4000,
) -> dict[str, Any]:
    values = np.asarray(
        [value for value in cluster_values.values() if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"clusters": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if values.size == 1:
        value = float(values[0])
        return {
            "clusters": 1,
            "mean": round(value, 6),
            "ci95_low": round(value, 6),
            "ci95_high": round(value, 6),
        }
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indexes].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "clusters": int(values.size),
        "mean": round(float(values.mean()), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
    }
