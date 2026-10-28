#!/usr/bin/env python3
"""Detect top/bottom phase states and freeze the global T1-6 anchor policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps


DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_FIT3D_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_CONFIG_OUTPUT = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_RESULTS_DIR = ROOT / "results" / "task1_6" / "dev"

CONFIG_SCHEMA_VERSION = "task1-6-segmenter-config-v1"
SUMMARY_SCHEMA_VERSION = "task1-6b-state-eval-v1"

ANCHOR_POLICIES = (
    "plateau_entry",
    "plateau_midpoint",
    "plateau_exit",
    "phase_zero_crossing",
)

ACTION_PHASE_CANDIDATES = {
    "squat": (
        "left_knee_flexion",
        "right_knee_flexion",
        "left_hip_flexion_sagittal",
        "right_hip_flexion_sagittal",
    ),
    "pushup": (
        "left_elbow_flexion",
        "right_elbow_flexion",
    ),
}

ACTION_BILATERAL_SIGNAL_PAIRS = {
    "squat": (
        (
            "knee_flexion_bilateral_mean",
            "left_knee_flexion",
            "right_knee_flexion",
        ),
        (
            "hip_flexion_bilateral_mean",
            "left_hip_flexion_sagittal",
            "right_hip_flexion_sagittal",
        ),
    ),
    "pushup": (
        (
            "elbow_flexion_bilateral_mean",
            "left_elbow_flexion",
            "right_elbow_flexion",
        ),
    ),
}

ACTION_PERIOD_RANGE_S = {
    "squat": (0.70, 5.00),
    "pushup": (0.70, 5.00),
}

PHASE_SIGNAL_MODES = (
    "t1_5_phase_angle",
    "t1_6_auto_candidate",
    "t1_6_bilateral_mean",
)

DEFAULT_SIGNAL_MODE = "t1_6_bilateral_mean"
DEFAULT_ANCHOR_POLICY = "plateau_midpoint"
DEV_DATASET = "Fit3D"
EXPECTED_DEV_CLIPS = 32
EXPECTED_DEV_CLUSTERS = 8
EVENT_GATE_PERIOD_FRACTION = 0.25

DEFAULT_DETECTOR = {
    "smoothing_seconds": 0.12,
    "minimum_valid_fraction": 0.30,
    "period_percentiles": [5.0, 95.0],
    "top_threshold_fraction": 0.55,
    "bottom_threshold_fraction": 0.35,
    "extrema_prominence_fraction": 0.35,
    "extrema_min_distance_period_fraction": 0.55,
    "cycle_chain_min_period_fraction": 0.60,
    "cycle_chain_max_period_fraction": 1.50,
    "minimum_state_period_fraction": 0.05,
    "fallback_state_radius_period_fraction": 0.12,
    "stable_derivative_fraction": 0.08,
    "motion_min_dynamic_range_deg": 5.0,
    "motion_min_dynamic_to_noise": 8.0,
    "motion_min_state_amplitude_deg": 3.0,
    "motion_joint_state_amplitude": {
        "enabled": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument(
        "--fit3d-manifest", type=Path, default=DEFAULT_FIT3D_MANIFEST
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def action_config(action: str, specs: dict[str, Any]) -> dict[str, Any]:
    if action not in specs.get("exercises", {}):
        raise ValueError(f"no exercise spec for action: {action}")
    spec = specs["exercises"][action]
    phase = spec.get("phase") or {}
    min_period_s, max_period_s = ACTION_PERIOD_RANGE_S.get(action, (0.5, 6.0))
    return {
        "action": action,
        "rep_unit": spec.get("rep_unit", "cycle"),
        "phase_signal_candidates": list(ACTION_PHASE_CANDIDATES.get(action, ())),
        "phase_signal_pairs": [
            {
                "name": name,
                "left": left,
                "right": right,
            }
            for name, left, right in ACTION_BILATERAL_SIGNAL_PAIRS.get(action, ())
        ],
        "top_polarity": phase.get("boundary_polarity", "peak"),
        "bottom_polarity": phase.get("bottom_polarity", "trough"),
        "phase_landmark": phase.get("phase_landmark", "top"),
        "min_period_s": min_period_s,
        "max_period_s": max_period_s,
    }


def _prepare_motion_series(
    values: Iterable[float],
    fps: float,
    detector: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = {**DEFAULT_DETECTOR, **(detector or {})}
    values = np.asarray(list(values), dtype=np.float64)
    count = len(values)
    if count < 20 or fps <= 0:
        return {}, settings
    valid = np.isfinite(values)
    valid_fraction = float(valid.mean())
    if valid_fraction < float(settings["minimum_valid_fraction"]):
        return {}, settings
    indexes = np.arange(count, dtype=np.float64)
    filled = np.interp(indexes, indexes[valid], values[valid])
    missing_indexes = np.flatnonzero(~valid)
    max_gap = 0
    if missing_indexes.size:
        runs = np.split(
            missing_indexes,
            np.where(np.diff(missing_indexes) > 1)[0] + 1,
        )
        max_gap = max(len(run) for run in runs)
    window = max(5, int(round(float(settings["smoothing_seconds"]) * fps)))
    max_window = count if count % 2 == 1 else count - 1
    window = min(window if window % 2 == 1 else window + 1, max_window)
    smooth = (
        savgol_filter(filled, window, 2, mode="interp")
        if window >= 5
        else filled
    )
    p_low, p_high = np.percentile(
        smooth, [float(value) for value in settings["period_percentiles"]]
    )
    dynamic_range = float(p_high - p_low)
    differences = np.diff(smooth)
    difference_median = (
        float(np.median(differences)) if differences.size else 0.0
    )
    mad = (
        float(np.median(np.abs(differences - difference_median)))
        if differences.size
        else 0.0
    )
    noise_sigma = 1.4826 * mad / math.sqrt(2.0)
    dynamic_to_noise = (
        dynamic_range / noise_sigma if noise_sigma > 1e-12 else None
    )
    return {
        "values": values,
        "fps": float(fps),
        "smooth": smooth,
        "valid": valid,
        "valid_fraction": valid_fraction,
        "max_gap_frames": int(max_gap),
        "dynamic_range": dynamic_range,
        "noise_sigma": noise_sigma,
        "dynamic_to_noise": dynamic_to_noise,
        "p_low": float(p_low),
        "p_high": float(p_high),
        "settings": settings,
        "window_frames": int(window),
    }, settings


def build_motion_evidence(
    prepared: dict[str, Any],
    *,
    state_amplitude: float | None = None,
) -> dict[str, Any]:
    if not prepared:
        return {
            "status": "insufficient_signal",
            "reasons": ["insufficient_signal"],
            "dynamic_range": None,
            "noise_sigma": None,
            "dynamic_to_noise": None,
            "state_amplitude": None,
            "thresholds": {
                "minimum_dynamic_range_deg": DEFAULT_DETECTOR[
                    "motion_min_dynamic_range_deg"
                ],
                "minimum_dynamic_to_noise": DEFAULT_DETECTOR[
                    "motion_min_dynamic_to_noise"
                ],
                "minimum_state_amplitude_deg": DEFAULT_DETECTOR[
                    "motion_min_state_amplitude_deg"
                ],
            },
        }
    settings = prepared["settings"]
    reasons = []
    if float(prepared["dynamic_range"]) < float(
        settings["motion_min_dynamic_range_deg"]
    ):
        reasons.append("dynamic_range_below_minimum")
    dynamic_to_noise = prepared.get("dynamic_to_noise")
    if (
        dynamic_to_noise is not None
        and float(dynamic_to_noise)
        < float(settings["motion_min_dynamic_to_noise"])
    ):
        reasons.append("dynamic_to_noise_below_minimum")
    state_amplitude_minimum = float(
        settings["motion_min_state_amplitude_deg"]
    )
    if state_amplitude is not None and float(state_amplitude) < state_amplitude_minimum:
        reasons.append("state_amplitude_below_minimum")
    joint = settings.get("motion_joint_state_amplitude") or {}
    joint_evidence = None
    if (
        reasons == ["state_amplitude_below_minimum"]
        and bool(joint.get("enabled"))
        and state_amplitude is not None
    ):
        minimum_fps = joint.get("minimum_analysis_fps")
        maximum_fps = joint.get("maximum_analysis_fps")
        analysis_fps = prepared.get("fps")
        periodicity = prepared.get("periodicity")
        checks = {
            "fps_in_range": bool(
                analysis_fps is not None
                and (minimum_fps is None or float(analysis_fps) >= float(minimum_fps))
                and (maximum_fps is None or float(analysis_fps) <= float(maximum_fps))
            ),
            "state_amplitude_fraction": bool(
                float(state_amplitude)
                >= float(joint.get("minimum_amplitude_fraction", 0.75))
                * state_amplitude_minimum
            ),
            "dynamic_range_pass": bool(
                float(prepared["dynamic_range"])
                >= float(joint.get("minimum_dynamic_range_multiple", 2.0))
                * float(settings["motion_min_dynamic_range_deg"])
            ),
            "dynamic_to_noise_pass": bool(
                dynamic_to_noise is not None
                and float(dynamic_to_noise)
                >= float(
                    joint.get("minimum_dynamic_to_noise_multiple", 1.5)
                )
                * float(settings["motion_min_dynamic_to_noise"])
            ),
            "periodicity_pass": bool(
                periodicity is not None
                and float(periodicity)
                >= float(joint.get("minimum_periodicity", 0.35))
            ),
        }
        joint_evidence = {
            "enabled": True,
            "checks": checks,
            "analysis_fps": (
                round(float(analysis_fps), 6)
                if analysis_fps is not None
                else None
            ),
            "periodicity": (
                round(float(periodicity), 6)
                if periodicity is not None
                else None
            ),
            "minimum_amplitude_fraction": float(
                joint.get("minimum_amplitude_fraction", 0.75)
            ),
        }
        if all(checks.values()):
            reasons = []
            joint_evidence["override_applied"] = True
        else:
            joint_evidence["override_applied"] = False
    return {
        "status": "no_motion" if reasons else "ok",
        "reasons": reasons,
        "dynamic_range": round(float(prepared["dynamic_range"]), 6),
        "noise_sigma": round(float(prepared["noise_sigma"]), 6),
        "dynamic_to_noise": (
            round(float(dynamic_to_noise), 6)
            if dynamic_to_noise is not None
            else None
        ),
        "state_amplitude": (
            round(float(state_amplitude), 6)
            if state_amplitude is not None
            else None
        ),
        "joint_state_amplitude": joint_evidence,
        "thresholds": {
            "minimum_dynamic_range_deg": float(
                settings["motion_min_dynamic_range_deg"]
            ),
            "minimum_dynamic_to_noise": float(
                settings["motion_min_dynamic_to_noise"]
            ),
            "minimum_state_amplitude_deg": float(
                settings["motion_min_state_amplitude_deg"]
            ),
        },
    }


def motion_evidence(
    values: Iterable[float],
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del action
    prepared, _ = _prepare_motion_series(values, fps, detector)
    return build_motion_evidence(prepared)


def assess_motion_signals(
    signals: dict[str, np.ndarray],
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expanded = dict(signals)
    bilateral_names = add_bilateral_signal_candidates(expanded, action)
    candidate_names = [
        *bilateral_names,
        *[
            name
            for name in action.get("phase_signal_candidates") or []
            if name not in bilateral_names
        ],
    ]
    candidates = []
    for name in candidate_names:
        if name not in expanded:
            continue
        evidence = motion_evidence(expanded[name], fps, action, detector)
        candidates.append({"phase_signal": name, **evidence})
    if not candidates:
        return {
            **motion_evidence([], fps, action, detector),
            "phase_signal": None,
        }
    candidates.sort(
        key=lambda item: (
            item["status"] == "ok",
            float(item.get("dynamic_to_noise") or 0.0),
            float(item.get("dynamic_range") or 0.0),
            str(item.get("phase_signal") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def prepare_analysis_signal(
    values: Iterable[float],
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prepared, settings = _prepare_motion_series(values, fps, detector)
    if not prepared:
        return None
    smooth = prepared["smooth"]
    indexes = np.arange(len(smooth), dtype=np.float64)
    coefficients = np.polyfit(indexes, smooth, 1)
    detrended = smooth - np.polyval(coefficients, indexes)
    period_result = segment_reps.estimate_period(
        detrended,
        fps,
        float(action["min_period_s"]),
        float(action["max_period_s"]),
    )
    if period_result is None:
        return None
    period_frames, periodicity = period_result
    dynamic_range = float(prepared["dynamic_range"])
    if dynamic_range < 1e-6:
        return None
    return {
        "fps": float(fps),
        "smooth": smooth,
        "valid": prepared["valid"],
        "valid_fraction": prepared["valid_fraction"],
        "max_gap_frames": prepared["max_gap_frames"],
        "dynamic_range": dynamic_range,
        "noise_sigma": prepared["noise_sigma"],
        "dynamic_to_noise": prepared["dynamic_to_noise"],
        "p_low": prepared["p_low"],
        "p_high": prepared["p_high"],
        "period_frames": float(period_frames),
        "periodicity": float(periodicity),
        "settings": settings,
        "motion_evidence": build_motion_evidence(prepared),
    }


def candidate_signal_score(prepared: dict[str, Any]) -> float:
    return (
        max(0.0, float(prepared["periodicity"]))
        * float(prepared["dynamic_range"])
        * float(prepared["valid_fraction"])
    )


def add_bilateral_signal_candidates(
    signals: dict[str, np.ndarray],
    action: dict[str, Any],
) -> list[str]:
    names = []
    for pair in action.get("phase_signal_pairs") or []:
        left = signals.get(pair["left"])
        right = signals.get(pair["right"])
        if left is None or right is None:
            continue
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.shape != right.shape or left.ndim != 1:
            continue
        stack = np.vstack([left, right])
        valid_count = np.sum(np.isfinite(stack), axis=0)
        total = np.zeros(left.shape, dtype=np.float64)
        np.nansum(stack, axis=0, out=total)
        mean = np.full(left.shape, np.nan, dtype=np.float64)
        supported = valid_count > 0
        mean[supported] = total[supported] / valid_count[supported]
        signals[pair["name"]] = mean
        names.append(pair["name"])
    return names


def select_phase_signal(
    signals: dict[str, Iterable[float]],
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    expanded_signals = {
        name: np.asarray(list(values), dtype=np.float64)
        for name, values in signals.items()
    }
    bilateral_names = add_bilateral_signal_candidates(expanded_signals, action)
    candidate_names = [
        *bilateral_names,
        *[
            name
            for name in action["phase_signal_candidates"]
            if name not in bilateral_names
        ],
    ]
    candidates = []
    for name in candidate_names:
        if name not in expanded_signals:
            continue
        prepared = prepare_analysis_signal(
            expanded_signals[name], fps, action, detector
        )
        if prepared is None:
            continue
        candidates.append(
            {
                "name": name,
                "prepared": prepared,
                "score": candidate_signal_score(prepared),
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["prepared"]["periodicity"],
            -item["prepared"]["max_gap_frames"],
            item["name"],
        ),
        reverse=True,
    )
    best = candidates[0]
    return {
        "name": best["name"],
        "prepared": best["prepared"],
        "score": best["score"],
        "candidates": [
            {
                "name": item["name"],
                "score": item["score"],
                "period_frames": item["prepared"]["period_frames"],
                "periodicity": item["prepared"]["periodicity"],
                "dynamic_range": item["prepared"]["dynamic_range"],
                "valid_fraction": item["prepared"]["valid_fraction"],
                "max_gap_frames": item["prepared"]["max_gap_frames"],
            }
            for item in candidates
        ],
    }


def select_dominant_extrema_chain(
    extrema: Iterable[int],
    period_frames: float,
    min_fraction: float,
    max_fraction: float,
) -> list[int]:
    values = sorted({int(value) for value in extrema})
    count = len(values)
    if count == 0:
        return []
    lengths = [1] * count
    costs = [0.0] * count
    previous = [-1] * count
    minimum_gap = min_fraction * period_frames
    maximum_gap = max_fraction * period_frames
    for index in range(count):
        for prior in range(index):
            gap = values[index] - values[prior]
            if not minimum_gap <= gap <= maximum_gap:
                continue
            edge_cost = (math.log(gap / period_frames) / 0.25) ** 2
            candidate_length = lengths[prior] + 1
            candidate_cost = costs[prior] + edge_cost
            if candidate_length > lengths[index] or (
                candidate_length == lengths[index]
                and candidate_cost < costs[index]
            ):
                lengths[index] = candidate_length
                costs[index] = candidate_cost
                previous[index] = prior
    cursor = max(
        range(count),
        key=lambda index: (lengths[index], -costs[index], -index),
    )
    chain = []
    while cursor >= 0:
        chain.append(values[cursor])
        cursor = previous[cursor]
    return list(reversed(chain))


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def strongest_run(
    mask: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    minimum_width: int,
) -> tuple[int, int] | None:
    start = max(0, int(start))
    end = min(len(mask) - 1, int(end))
    if end < start:
        return None
    runs = [
        (run_start, run_end)
        for run_start, run_end in contiguous_runs(mask[start : end + 1])
        if run_end - run_start + 1 >= minimum_width
    ]
    if not runs:
        return None
    shifted = [(run_start + start, run_end + start) for run_start, run_end in runs]
    return max(
        shifted,
        key=lambda run: (
            float(np.mean(values[run[0] : run[1] + 1])),
            run[1] - run[0] + 1,
        ),
    )


def run_containing_or_nearest(
    mask: np.ndarray,
    start: int,
    end: int,
    center: int,
    minimum_width: int,
) -> tuple[int, int] | None:
    start = max(0, int(start))
    end = min(len(mask) - 1, int(end))
    runs = [
        (run_start + start, run_end + start)
        for run_start, run_end in contiguous_runs(mask[start : end + 1])
        if run_end - run_start + 1 >= minimum_width
    ]
    if not runs:
        return None
    containing = [run for run in runs if run[0] <= center <= run[1]]
    if containing:
        return max(containing, key=lambda run: run[1] - run[0])
    return min(
        runs,
        key=lambda run: min(abs(center - run[0]), abs(center - run[1])),
    )


def local_slope(values: np.ndarray, start: int, end: int, edge: str) -> float:
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end <= start:
        return 0.0
    radius = min(3, max(1, (end - start) // 4))
    if edge == "start":
        indexes = np.arange(start, min(end + 1, start + radius + 1), dtype=float)
    else:
        indexes = np.arange(max(start, end - radius), end + 1, dtype=float)
    if len(indexes) < 2:
        return 0.0
    return float(np.polyfit(indexes, values[indexes.astype(int)], 1)[0])


def stable_fraction(
    values: np.ndarray,
    start: int,
    end: int,
    dynamic_range: float,
    derivative_fraction: float,
) -> float:
    segment = values[max(0, start) : min(len(values), end + 1)]
    if len(segment) < 2:
        return 0.0
    tolerance = max(1e-8, derivative_fraction * dynamic_range)
    return float(np.mean(np.abs(np.diff(segment)) <= tolerance))


def crossing_anchor(
    oriented: np.ndarray,
    state_mean: float,
    threshold: float,
    start: int,
    end: int,
    period_frames: float,
    high_state: bool,
) -> int:
    target = 0.5 * (state_mean + threshold)
    lower = max(1, start - int(round(0.25 * period_frames)))
    upper = min(len(oriented) - 1, end + int(round(0.10 * period_frames)))
    if upper <= lower:
        return int((start + end) // 2)
    before = oriented[lower - 1 : upper]
    after = oriented[lower : upper + 1]
    if high_state:
        crossings = np.flatnonzero((before < target) & (after >= target)) + lower
    else:
        crossings = np.flatnonzero((before > target) & (after <= target)) + lower
    if len(crossings) == 0:
        return int((start + end) // 2)
    midpoint = 0.5 * (start + end)
    candidate = int(crossings[np.argmin(np.abs(crossings - midpoint))])
    return int(np.clip(candidate, start, end))


def build_phase_state(
    values: np.ndarray,
    oriented: np.ndarray,
    start: int,
    end: int,
    period_frames: float,
    dynamic_range: float,
    threshold: float,
    high_state: bool,
    kind: str,
    fallback: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    state_mean = float(np.mean(oriented[start : end + 1]))
    anchors = {
        "plateau_entry": int(start),
        "plateau_midpoint": int((start + end) // 2),
        "plateau_exit": int(end),
        "phase_zero_crossing": crossing_anchor(
            oriented,
            state_mean,
            threshold,
            start,
            end,
            period_frames,
            high_state,
        ),
    }
    slope_start = local_slope(values, start, end, "start")
    slope_end = local_slope(values, start, end, "exit")
    evidence = {
        "amplitude_level": state_mean,
        "threshold": float(threshold),
        "duration_fraction": float((end - start + 1) / period_frames),
        "slope_start": slope_start,
        "slope_end": slope_end,
        "fallback": 1.0 if fallback else 0.0,
    }
    return {
        "start_frame": int(start),
        "end_frame": int(end),
        "anchor_frame": anchors["plateau_midpoint"],
        "stable_fraction": stable_fraction(
            values,
            start,
            end,
            dynamic_range,
            float(settings["stable_derivative_fraction"]),
        ),
        "boundary_slope": float(0.5 * (slope_start + slope_end)),
        "mean_angle": float(np.mean(values[start : end + 1])),
        "evidence": evidence,
        "kind": kind,
        "status": "fallback" if fallback else "valid",
        "anchors": anchors,
        "quality_flags": ["fallback_peak_state"] if fallback else [],
    }


def observed_state_amplitude(
    top_states: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
) -> float | None:
    if not top_states or not bottom_states:
        return None
    amplitudes = []
    for top in top_states:
        top_midpoint = 0.5 * (top["start_frame"] + top["end_frame"])
        bottom = min(
            bottom_states,
            key=lambda state: abs(
                0.5 * (state["start_frame"] + state["end_frame"])
                - top_midpoint
            ),
        )
        amplitudes.append(abs(float(top["mean_angle"] - bottom["mean_angle"])))
    return float(np.median(amplitudes)) if amplitudes else None


def detect_phase_states(
    values: Iterable[float],
    fps: float,
    action: dict[str, Any],
    phase_signal: str,
    anchor_policy: str = "plateau_midpoint",
    detector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if anchor_policy not in ANCHOR_POLICIES:
        raise ValueError(f"unknown anchor policy: {anchor_policy}")
    values = np.asarray(list(values), dtype=np.float64)
    prepared = prepare_analysis_signal(values, fps, action, detector)
    if prepared is None:
        return {
            "status": "no_motion",
            "motion_status": "no_motion",
            "phase_signal": phase_signal,
            "anchor_policy": anchor_policy,
            "period_frames": None,
            "top_states": [],
            "bottom_states": [],
            "reps": [],
            "rejected_states": [],
            "motion_evidence": motion_evidence(values, fps, action, detector),
            "diagnostics": {"reason": "insufficient_periodic_signal"},
        }
    evidence = prepared["motion_evidence"]
    if evidence["status"] == "no_motion":
        return {
            "status": "no_motion",
            "motion_status": "no_motion",
            "phase_signal": phase_signal,
            "anchor_policy": anchor_policy,
            "period_frames": prepared["period_frames"],
            "periodicity": prepared["periodicity"],
            "top_states": [],
            "bottom_states": [],
            "reps": [],
            "rejected_states": [],
            "motion_evidence": evidence,
            "diagnostics": {
                "reason": "motion_threshold_not_met",
                "motion_reasons": evidence["reasons"],
            },
        }
    smooth = prepared["smooth"]
    valid = prepared["valid"]
    settings = prepared["settings"]
    period_frames = float(prepared["period_frames"])
    dynamic_range = float(prepared["dynamic_range"])
    top_polarity = action.get("top_polarity", "peak")
    oriented = smooth if top_polarity == "peak" else -smooth
    bottom_series = -oriented
    oriented_low, _ = np.percentile(oriented, [5.0, 95.0])
    top_threshold = float(oriented_low) + float(
        settings["top_threshold_fraction"]
    ) * dynamic_range
    bottom_threshold = float(oriented_low) + float(
        settings["bottom_threshold_fraction"]
    ) * dynamic_range
    minimum_width = max(3, int(round(float(settings["minimum_state_period_fraction"]) * period_frames)))
    fallback_radius = max(2, int(round(float(settings["fallback_state_radius_period_fraction"]) * period_frames)))
    extrema, _ = find_peaks(
        bottom_series,
        distance=max(
            4,
            int(
                round(
                    float(settings["extrema_min_distance_period_fraction"])
                    * period_frames
                )
            ),
        ),
        prominence=max(
            1e-8,
            float(settings["extrema_prominence_fraction"]) * dynamic_range,
        ),
    )
    chain = select_dominant_extrema_chain(
        extrema.tolist(),
        period_frames,
        float(settings["cycle_chain_min_period_fraction"]),
        float(settings["cycle_chain_max_period_fraction"]),
    )
    if len(chain) < 2:
        return {
            "status": "unstable_phase",
            "motion_status": evidence["status"],
            "phase_signal": phase_signal,
            "anchor_policy": anchor_policy,
            "period_frames": period_frames,
            "periodicity": prepared["periodicity"],
            "top_states": [],
            "bottom_states": [],
            "reps": [],
            "rejected_states": [],
            "motion_evidence": evidence,
            "diagnostics": {
                "reason": "insufficient_cycle_chain",
                "extrema_frames": [int(value) for value in extrema],
            },
        }

    high_mask = oriented >= top_threshold
    low_mask = oriented <= bottom_threshold
    bottom_states = []
    rejected_states = []
    for index, frame in enumerate(chain, start=1):
        window = int(round(0.35 * period_frames))
        run = run_containing_or_nearest(
            low_mask,
            frame - window,
            frame + window,
            frame,
            minimum_width,
        )
        fallback = run is None
        if run is None:
            run = (
                max(0, frame - fallback_radius),
                min(len(smooth) - 1, frame + fallback_radius),
            )
        state = build_phase_state(
            smooth,
            oriented,
            run[0],
            run[1],
            period_frames,
            dynamic_range,
            bottom_threshold,
            False,
            "interior",
            fallback,
            settings,
        )
        state["state_index"] = index
        state["extrema_frame"] = int(frame)
        state["anchor_frame"] = state["anchors"][anchor_policy]
        bottom_states.append(state)

    windows = [
        ("leading_partial", chain[0] - period_frames, chain[0]),
        *[
            ("interior", left, right)
            for left, right in zip(chain[:-1], chain[1:])
        ],
        ("trailing_partial", chain[-1], chain[-1] + period_frames),
    ]
    top_states = []
    for index, (kind, window_start, window_end) in enumerate(windows, start=1):
        run = strongest_run(
            high_mask,
            smooth,
            window_start,
            window_end,
            minimum_width,
        )
        fallback = run is None
        if run is None:
            start = max(0, int(window_start))
            end = min(len(smooth) - 1, int(window_end))
            local_peak = start + int(np.argmax(oriented[start : end + 1]))
            run = (
                max(start, local_peak - fallback_radius),
                min(end, local_peak + fallback_radius),
            )
            local_peak_value = float(
                np.max(oriented[run[0] : run[1] + 1])
            )
            if local_peak_value < top_threshold:
                rejected = build_phase_state(
                    smooth,
                    oriented,
                    run[0],
                    run[1],
                    period_frames,
                    dynamic_range,
                    top_threshold,
                    True,
                    kind,
                    True,
                    settings,
                )
                rejected.update(
                    {
                        "reason": "no_stable_top_state",
                        "relative_peak_fraction": round(
                            float(
                                (local_peak_value - oriented_low)
                                / max(dynamic_range, 1e-8)
                            ),
                            6,
                        ),
                    }
                )
                rejected_states.append(rejected)
                continue
        state = build_phase_state(
            smooth,
            oriented,
            run[0],
            run[1],
            period_frames,
            dynamic_range,
            top_threshold,
            True,
            kind,
            fallback,
            settings,
        )
        state["state_index"] = index
        state["anchor_frame"] = state["anchors"][anchor_policy]
        top_states.append(state)

    reps = build_rep_candidates(
        top_states,
        bottom_states,
        phase_signal=phase_signal,
        rep_unit=action.get("rep_unit", "cycle"),
        period_frames=period_frames,
        periodicity=prepared["periodicity"],
    )
    state_amplitude = observed_state_amplitude(top_states, bottom_states)
    evidence = build_motion_evidence(
        prepared,
        state_amplitude=state_amplitude,
    )
    if evidence["status"] == "no_motion":
        return {
            "status": "no_motion",
            "motion_status": "no_motion",
            "phase_signal": phase_signal,
            "top_polarity": top_polarity,
            "anchor_policy": anchor_policy,
            "period_frames": period_frames,
            "periodicity": prepared["periodicity"],
            "valid_fraction": prepared["valid_fraction"],
            "max_gap_frames": prepared["max_gap_frames"],
            "dynamic_range": dynamic_range,
            "top_states": top_states,
            "bottom_states": bottom_states,
            "reps": [],
            "rejected_states": rejected_states,
            "motion_evidence": evidence,
            "diagnostics": {
                "reason": "motion_threshold_not_met",
                "motion_reasons": evidence["reasons"],
            },
        }
    status = "ok" if reps else "no_confident_set"
    return {
        "status": status,
        "motion_status": evidence["status"],
        "phase_signal": phase_signal,
        "top_polarity": top_polarity,
        "anchor_policy": anchor_policy,
        "period_frames": period_frames,
        "periodicity": prepared["periodicity"],
        "valid_fraction": prepared["valid_fraction"],
        "max_gap_frames": prepared["max_gap_frames"],
        "dynamic_range": dynamic_range,
        "top_states": top_states,
        "bottom_states": bottom_states,
        "reps": reps,
        "rejected_states": rejected_states,
        "motion_evidence": evidence,
        "diagnostics": {
            "top_threshold": top_threshold,
            "bottom_threshold": bottom_threshold,
            "extrema_frames": [int(value) for value in extrema],
            "selected_chain_frames": [int(value) for value in chain],
            "window_count": len(windows),
        },
    }


def public_phase_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_frame": int(state["start_frame"]),
        "end_frame": int(state["end_frame"]),
        "anchor_frame": int(state["anchor_frame"]),
        "stable_fraction": float(state["stable_fraction"]),
        "boundary_slope": float(state["boundary_slope"]),
        "mean_angle": float(state["mean_angle"]),
        "evidence": {
            key: float(value)
            for key, value in state.get("evidence", {}).items()
            if isinstance(value, (int, float))
        },
    }


def build_rep_candidates(
    top_states: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    *,
    phase_signal: str,
    rep_unit: str,
    period_frames: float,
    periodicity: float,
) -> list[dict[str, Any]]:
    if len(top_states) < 2 or not bottom_states:
        return []
    tops = sorted(top_states, key=lambda state: state["anchor_frame"])
    bottoms = sorted(bottom_states, key=lambda state: state["anchor_frame"])
    reps = []
    for index, (left_top, right_top) in enumerate(zip(tops[:-1], tops[1:]), start=1):
        left_mid = 0.5 * (left_top["start_frame"] + left_top["end_frame"])
        right_mid = 0.5 * (right_top["start_frame"] + right_top["end_frame"])
        candidates = [
            state
            for state in bottoms
            if left_mid < 0.5 * (state["start_frame"] + state["end_frame"]) < right_mid
        ]
        if not candidates:
            continue
        target = 0.5 * (left_mid + right_mid)
        bottom = min(
            candidates,
            key=lambda state: abs(
                0.5 * (state["start_frame"] + state["end_frame"]) - target
            ),
        )
        endpoint = (
            left_top["kind"] != "interior" or right_top["kind"] != "interior"
        )
        status = "endpoint_partial" if endpoint else "complete"
        partial_side = None
        if endpoint:
            partial_side = "start" if left_top["kind"] == "leading_partial" else "end"
        cycle_frames = max(
            1.0,
            (right_top["start_frame"] + right_top["end_frame"])
            - (left_top["start_frame"] + left_top["end_frame"]),
        ) * 0.5
        period_ratio = cycle_frames / max(period_frames, 1e-6)
        timing_score = math.exp(-0.5 * ((math.log(max(period_ratio, 1e-6))) / 0.25) ** 2)
        state_stability = np.mean(
            [
                left_top["stable_fraction"],
                right_top["stable_fraction"],
                bottom["stable_fraction"],
            ]
        )
        amplitude = max(
            0.0,
            float(left_top["mean_angle"] - bottom["mean_angle"]),
        )
        amplitude_score = min(1.0, amplitude / 15.0)
        fallback_penalty = 0.85 if any(
            state["status"] == "fallback"
            for state in (left_top, right_top, bottom)
        ) else 1.0
        complete_confidence = float(
            np.clip(
                fallback_penalty
                * (0.55 * state_stability + 0.30 * amplitude_score + 0.15 * timing_score),
                0.0,
                1.0,
            )
        )
        in_set_confidence = float(
            np.clip(
                (0.55 * max(0.0, periodicity) + 0.45 * timing_score) * fallback_penalty,
                0.0,
                1.0,
            )
        )
        combined = float(complete_confidence * in_set_confidence)
        flags = []
        for state in (left_top, right_top, bottom):
            flags.extend(state.get("quality_flags") or [])
        if endpoint:
            flags.append("endpoint_partial")
        reps.append(
            {
                "rep_id": index,
                "start_frame": int(left_top["start_frame"]),
                "end_frame": int(right_top["end_frame"]),
                "top": public_phase_state(left_top),
                "bottom": public_phase_state(bottom),
                "phase_signal": phase_signal,
                "rep_unit": rep_unit,
                "status": status,
                "partial_side": partial_side,
                "confidence": {
                    "complete": round(complete_confidence, 6),
                    "in_set": round(in_set_confidence, 6),
                    "combined": round(combined, 6),
                },
                "quality_flags": sorted(set(flags)),
            }
        )
    return reps


def load_angle_signals(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty angle table: {path}")
    signals = {}
    for name in rows[0]:
        if name in {"frame_id", "timestamp_ms"}:
            continue
        values = []
        for row in rows:
            text = (row.get(name) or "").strip()
            try:
                values.append(float(text))
            except ValueError:
                values.append(np.nan)
        signals[name] = np.asarray(values, dtype=np.float64)
    return signals


def select_bilateral_signal(
    signals: dict[str, np.ndarray],
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None = None,
) -> str | None:
    names = add_bilateral_signal_candidates(signals, action)
    for name in names:
        if prepare_analysis_signal(
            signals[name], fps, action, detector
        ) is not None:
            return name
    return None


def resolve_phase_signal(
    signals: dict[str, np.ndarray],
    fps: float,
    action: dict[str, Any],
    policy: str,
    *,
    prior_signal: str | None = None,
    detector: dict[str, Any] | None = None,
) -> str | None:
    if policy not in PHASE_SIGNAL_MODES:
        raise ValueError(f"unknown phase signal policy: {policy}")
    if policy == "t1_5_phase_angle":
        if (
            prior_signal in signals
            and prepare_analysis_signal(
                signals[prior_signal], fps, action, detector
            )
            is not None
        ):
            return prior_signal
    elif policy == "t1_6_bilateral_mean":
        selected = select_bilateral_signal(signals, fps, action, detector)
        if selected is not None:
            return selected

    auto = select_phase_signal(signals, fps, action, detector)
    if auto is not None:
        return str(auto["name"])
    for name in action.get("phase_signal_candidates") or []:
        if name in signals and prepare_analysis_signal(
            signals[name], fps, action, detector
        ) is not None:
            return str(name)
    return None


def event_gate_frames(gt_frames: list[int]) -> int:
    if len(gt_frames) < 2:
        return 1
    median_interval = float(np.median(np.diff(sorted(gt_frames))))
    return max(
        1,
        int(math.floor(EVENT_GATE_PERIOD_FRACTION * median_interval + 0.5)),
    )


def match_events(
    gt_frames: list[int],
    pred_frames: list[int],
    gate_frames: int,
) -> list[tuple[int, int, int]]:
    if not gt_frames or not pred_frames:
        return []
    cost = np.abs(
        np.asarray(gt_frames, dtype=np.float64)[:, None]
        - np.asarray(pred_frames, dtype=np.float64)[None, :]
    )
    assignment_limit = min(cost.shape)
    invalid_penalty = (max(int(gate_frames), 0) + 1) * (assignment_limit + 1)
    gated_cost = np.where(cost <= gate_frames, cost, float(invalid_penalty))
    rows, columns = linear_sum_assignment(gated_cost)
    matches = [
        (int(row), int(column), int(round(cost[row, column])))
        for row, column in zip(rows, columns)
        if cost[row, column] <= gate_frames
    ]
    return sorted(matches)


def percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, quantile)), 6)


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return round(2.0 * precision * recall / (precision + recall), 6)


def evaluate_predictions(
    gt_frames: list[int],
    top_states: list[dict[str, Any]],
    anchor_policy: str,
) -> dict[str, Any]:
    predictions = sorted(
        [
            {
                "anchor_frame": int(state["anchors"][anchor_policy]),
                "start_frame": int(state["start_frame"]),
                "end_frame": int(state["end_frame"]),
            }
            for state in top_states
        ],
        key=lambda item: item["anchor_frame"],
    )
    pred_frames = [item["anchor_frame"] for item in predictions]
    gate = event_gate_frames(gt_frames)
    matches = match_events(gt_frames, pred_frames, gate)
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_pred = {pred_index for _, pred_index, _ in matches}
    abs_errors = [error for _, _, error in matches]
    signed_errors = [
        int(pred_frames[pred_index] - gt_frames[gt_index])
        for gt_index, pred_index, _ in matches
    ]
    covered_gt = [
        index
        for index, frame in enumerate(gt_frames)
        if any(
            prediction["start_frame"] <= frame <= prediction["end_frame"]
            for prediction in predictions
        )
    ]
    matched_contained_gt = [
        gt_index
        for gt_index, pred_index, _ in matches
        if predictions[pred_index]["start_frame"]
        <= gt_frames[gt_index]
        <= predictions[pred_index]["end_frame"]
    ]
    tp = len(matches)
    return {
        "event_gate_frames": gate,
        "gt_event_count": len(gt_frames),
        "pred_event_count": len(predictions),
        "tp": tp,
        "fp": len(predictions) - tp,
        "fn": len(gt_frames) - tp,
        "event_recall": ratio(tp, len(gt_frames)),
        "event_precision": ratio(tp, len(predictions)),
        "event_f1": f1_score(
            ratio(tp, len(predictions)),
            ratio(tp, len(gt_frames)),
        ),
        "first_gt_matched": bool(gt_frames) and 0 in matched_gt,
        "last_gt_matched": bool(gt_frames) and len(gt_frames) - 1 in matched_gt,
        "matched_pred_event_count": len(matched_pred),
        "gt_in_any_top_interval_count": len(covered_gt),
        "gt_in_any_top_interval_fraction": ratio(len(covered_gt), len(gt_frames)),
        "matched_gt_in_top_interval_count": len(matched_contained_gt),
        "matched_gt_in_top_interval_fraction": ratio(
            len(matched_contained_gt), tp
        ),
        "event_abs_error_median_frames": percentile(abs_errors, 50),
        "event_abs_error_p90_frames": percentile(abs_errors, 90),
        "event_signed_error_median_frames": percentile(signed_errors, 50),
        "abs_errors": abs_errors,
        "signed_errors": signed_errors,
    }


def aggregate_clip_metrics(
    rows: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    gt_count = sum(int(row["gt_event_count"]) for row in rows)
    pred_count = sum(int(row["pred_event_count"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    abs_errors = [
        error for row in rows for error in row.get("_abs_errors", [])
    ]
    signed_errors = [
        error for row in rows for error in row.get("_signed_errors", [])
    ]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["event_recall"] is not None:
            clusters[row["cluster_id"]].append(float(row["event_recall"]))
    cluster_recalls = [
        float(np.mean(values)) for values in clusters.values() if values
    ]
    first_recall = ratio(
        sum(bool(row["first_gt_matched"]) for row in rows), len(rows)
    )
    last_recall = ratio(
        sum(bool(row["last_gt_matched"]) for row in rows), len(rows)
    )
    precision = ratio(tp, pred_count)
    recall = ratio(tp, gt_count)
    return {
        "scope": scope,
        "clip_count": len(rows),
        "cluster_count": len(clusters),
        "gt_event_count": gt_count,
        "pred_event_count": pred_count,
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1_score(precision, recall),
        "first_event_recall": first_recall,
        "last_event_recall": last_recall,
        "first_last_min_recall": (
            round(min(first_recall, last_recall), 6)
            if first_recall is not None and last_recall is not None
            else None
        ),
        "clip_event_recall_mean": round(
            float(np.mean([row["event_recall"] for row in rows])), 6
        )
        if rows
        else None,
        "cluster_event_recall_mean": round(
            float(np.mean(cluster_recalls)), 6
        )
        if cluster_recalls
        else None,
        "event_abs_error_median_frames": percentile(abs_errors, 50),
        "event_abs_error_p90_frames": percentile(abs_errors, 90),
        "event_abs_error_max_frames": max(abs_errors) if abs_errors else None,
        "event_signed_error_median_frames": percentile(signed_errors, 50),
        "gt_in_any_top_interval_fraction": ratio(
            sum(int(row["gt_in_any_top_interval_count"]) for row in rows),
            gt_count,
        ),
        "matched_gt_in_top_interval_fraction": ratio(
            sum(int(row["matched_gt_in_top_interval_count"]) for row in rows),
            tp,
        ),
        "status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in rows)
        ),
    }


def choose_primary_clip_ids(
    clips: list[dict[str, Any]],
    fit3d_metadata: dict[str, dict[str, Any]],
) -> dict[str, str]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_cluster[clip["cluster_id"]].append(clip)

    def priority(clip: dict[str, Any]) -> tuple[float, int, str]:
        metadata = fit3d_metadata.get(clip["clip_id"]) or {}
        geometry = metadata.get("view_geometry") or {}
        angle = float(geometry.get("angle_from_frontal_normal_deg", -1.0))
        is_side = int(
            metadata.get("view")
            in {"side_left", "side_right", "oblique_left", "oblique_right"}
        )
        return angle, is_side, str(clip.get("camera_id") or "")

    return {
        cluster_id: str(max(cluster_clips, key=priority)["clip_id"])
        for cluster_id, cluster_clips in by_cluster.items()
    }


def combination_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    preferred_signals = (
        "t1_6_bilateral_mean",
        "t1_6_auto_candidate",
        "t1_5_phase_angle",
    )
    preferred_anchors = (
        "plateau_midpoint",
        "phase_zero_crossing",
        "plateau_entry",
        "plateau_exit",
    )
    all_cameras = row["all_cameras"]
    first_last_min = float(all_cameras["first_last_min_recall"] or 0.0)
    passed_stage_gate = first_last_min >= 0.50
    return (
        not passed_stage_gate,
        -float(all_cameras["event_recall"] or 0.0),
        float(all_cameras["event_abs_error_median_frames"] or math.inf),
        float(all_cameras["event_abs_error_p90_frames"] or math.inf),
        -first_last_min,
        -float(all_cameras["event_precision"] or 0.0),
        -float(all_cameras["gt_in_any_top_interval_fraction"] or 0.0),
        preferred_signals.index(row["signal_mode"]),
        preferred_anchors.index(row["anchor_policy"]),
    )


def build_frozen_config(
    *,
    selected: dict[str, Any],
    actions: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "task": "T1-6B",
        "frozen": True,
        "core": {
            "dataset_independent": True,
            "phase_signal": {
                "policy": selected["signal_mode"],
                "auto_score": "periodicity * dynamic_range * valid_fraction",
                "bilateral_mean": "nanmean(left, right) when either side is valid",
                "fallback": (
                    "t1_6_auto_candidate_then_first_available_action_candidate"
                ),
            },
            "actions": actions,
            "detector": DEFAULT_DETECTOR,
            "anchor": {
                "policy": selected["anchor_policy"],
                "candidates": list(ANCHOR_POLICIES),
                "scope": "single_global_policy",
            },
            "endpoint_partial": {
                "rule": "leading_or_trailing_top_plus_interior_bottom",
                "requires_top_and_bottom": True,
                "requires_period_and_amplitude_consistency": True,
            },
        },
        "provenance": provenance,
    }


def write_checksums(directory: Path, names: list[str]) -> None:
    lines = [
        f"{sha256_path(directory / name)}  {name}"
        for name in sorted(names)
        if (directory / name).is_file()
    ]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n")


def run_dev_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    sources = load_json(args.sources)
    specs = load_json(args.specs)
    fit3d_manifest = load_json(args.fit3d_manifest)
    fit3d_metadata = {
        clip["clip_id"]: clip for clip in fit3d_manifest.get("clips", [])
    }
    clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == DEV_DATASET
        and clip.get("split_role") == "dev"
        and clip.get("has_gt")
        and not clip.get("smoke_only")
    ]
    clusters = {clip["cluster_id"] for clip in clips}
    if len(clips) != EXPECTED_DEV_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_DEV_CLIPS} Fit3D dev clips, found {len(clips)}"
        )
    if len(clusters) != EXPECTED_DEV_CLUSTERS:
        raise ValueError(
            f"expected {EXPECTED_DEV_CLUSTERS} dev clusters, found {len(clusters)}"
        )
    primary_clip_ids = choose_primary_clip_ids(clips, fit3d_metadata)
    clip_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    state_rows: list[dict[str, Any]] = []
    signal_audit_rows: list[dict[str, Any]] = []
    for clip in clips:
        analysis_dir = (
            args.analysis_dir / clip["clip_id"] / "angles" / "angles_wide.csv"
        )
        analysis = load_json(
            args.analysis_dir / clip["clip_id"] / "analysis.json"
        )
        signals = load_angle_signals(analysis_dir)
        action = action_config(clip["action"], specs)
        auto = select_phase_signal(signals, clip["fps"], action)
        prior_signal = (analysis.get("segmentation") or {}).get("phase_angle")
        bilateral_signal = select_bilateral_signal(
            signals, clip["fps"], action
        )
        signal_audit_rows.append(
            {
                "clip_id": clip["clip_id"],
                "cluster_id": clip["cluster_id"],
                "action": clip["action"],
                "prior_phase_angle": prior_signal,
                "auto_phase_angle": auto["name"] if auto else None,
                "auto_score": round(float(auto["score"]), 6) if auto else None,
                "bilateral_phase_signal": bilateral_signal,
                "auto_candidates_json": json.dumps(
                    auto["candidates"] if auto else [], ensure_ascii=False
                ),
            }
        )
        signal_by_mode = {
            "t1_5_phase_angle": prior_signal,
            "t1_6_auto_candidate": auto["name"] if auto else None,
            "t1_6_bilateral_mean": bilateral_signal,
        }
        gt_frames = sorted(
            int(event["clip_frame"])
            for event in (clip.get("gt") or {}).get("events", [])
            if event.get("event") == "top"
        )
        metadata = fit3d_metadata.get(clip["clip_id"]) or {}
        base = {
            "dataset": clip.get("dataset"),
            "clip_id": clip["clip_id"],
            "video_id": clip.get("video_id"),
            "cluster_id": clip["cluster_id"],
            "subject_id": clip.get("subject_id"),
            "camera_id": clip.get("camera_id"),
            "action": clip.get("action"),
            "view": clip.get("view"),
            "view_angle_deg": (metadata.get("view_geometry") or {}).get(
                "angle_from_frontal_normal_deg"
            ),
            "split_role": clip.get("split_role"),
            "is_primary_camera": clip["clip_id"]
            == primary_clip_ids.get(clip["cluster_id"]),
        }
        for signal_mode in PHASE_SIGNAL_MODES:
            phase_signal = signal_by_mode[signal_mode]
            if phase_signal is None or phase_signal not in signals:
                states = {
                    "status": "phase_signal_unavailable",
                    "period_frames": None,
                    "top_states": [],
                    "bottom_states": [],
                    "reps": [],
                }
            else:
                states = detect_phase_states(
                    signals[phase_signal],
                    clip["fps"],
                    action,
                    phase_signal,
                    DEFAULT_ANCHOR_POLICY,
                )
            for state_type, state_key in (
                ("top", "top_states"),
                ("bottom", "bottom_states"),
            ):
                for state_index, state in enumerate(
                    states.get(state_key) or [], start=1
                ):
                    state_rows.append(
                        {
                            **base,
                            "signal_mode": signal_mode,
                            "phase_signal": phase_signal,
                            "state_type": state_type,
                            "state_index": state_index,
                            "kind": state.get("kind"),
                            "status": state.get("status"),
                            "start_frame": state["start_frame"],
                            "end_frame": state["end_frame"],
                            "duration_frames": state["end_frame"]
                            - state["start_frame"]
                            + 1,
                            "period_frames": states.get("period_frames"),
                            "duration_period_fraction": state.get(
                                "evidence", {}
                            ).get("duration_fraction"),
                            "stable_fraction": state.get("stable_fraction"),
                            "boundary_slope": state.get("boundary_slope"),
                            "mean_angle": state.get("mean_angle"),
                            "anchor_plateau_entry": state.get("anchors", {}).get(
                                "plateau_entry"
                            ),
                            "anchor_plateau_midpoint": state.get(
                                "anchors", {}
                            ).get("plateau_midpoint"),
                            "anchor_plateau_exit": state.get("anchors", {}).get(
                                "plateau_exit"
                            ),
                            "anchor_phase_zero_crossing": state.get(
                                "anchors", {}
                            ).get("phase_zero_crossing"),
                        }
                    )
            for anchor_policy in ANCHOR_POLICIES:
                evaluation = evaluate_predictions(
                    gt_frames,
                    states.get("top_states") or [],
                    anchor_policy,
                )
                row = {
                    **base,
                    "signal_mode": signal_mode,
                    "anchor_policy": anchor_policy,
                    "phase_signal": phase_signal,
                    "status": states.get("status"),
                    "period_frames": states.get("period_frames"),
                    "top_state_count": len(states.get("top_states") or []),
                    "bottom_state_count": len(
                        states.get("bottom_states") or []
                    ),
                    "rep_candidate_count": len(states.get("reps") or []),
                    "endpoint_partial_count": sum(
                        rep.get("status") == "endpoint_partial"
                        for rep in states.get("reps") or []
                    ),
                    **{
                        key: value
                        for key, value in evaluation.items()
                        if key not in {"abs_errors", "signed_errors"}
                    },
                    "_abs_errors": evaluation["abs_errors"],
                    "_signed_errors": evaluation["signed_errors"],
                }
                clip_rows[(signal_mode, anchor_policy)].append(row)

    primary_ids = set(primary_clip_ids.values())
    comparison_rows = []
    for signal_mode in PHASE_SIGNAL_MODES:
        for anchor_policy in ANCHOR_POLICIES:
            rows = clip_rows[(signal_mode, anchor_policy)]
            primary_rows = [row for row in rows if row["clip_id"] in primary_ids]
            comparison_rows.append(
                {
                    "signal_mode": signal_mode,
                    "anchor_policy": anchor_policy,
                    "all_cameras": aggregate_clip_metrics(rows, "all_cameras"),
                    "primary_camera": aggregate_clip_metrics(
                        primary_rows, "primary_camera"
                    ),
                }
            )
    selected = min(comparison_rows, key=combination_rank)
    selected_signal_mode = selected["signal_mode"]
    selected_anchor = selected["anchor_policy"]
    for row in state_rows:
        row["selected_signal_mode"] = row["signal_mode"] == selected_signal_mode
        row["selected_anchor_policy"] = selected_anchor
        row["selected_anchor_frame"] = row.get(
            f"anchor_{selected_anchor}"
        )

    actions = {
        action: action_config(action, specs)
        for action in sorted({clip["action"] for clip in clips})
    }
    provenance = {
        "selected_on": "Fit3D dev",
        "selection_scope": "global",
        "selection_rule": (
            "require first and last recall >= 0.50, then maximize all-camera "
            "recall, minimize median/p90 error, then first/last recall, "
            "precision and top-interval containment"
        ),
        "dev_clip_count": len(clips),
        "dev_cluster_count": len(clusters),
        "sources_sha256": sha256_path(args.sources),
        "specs_sha256": sha256_path(args.specs),
        "fit3d_manifest_sha256": sha256_path(args.fit3d_manifest),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "generated_at": utc_now(),
    }
    config = build_frozen_config(
        selected=selected,
        actions=actions,
        provenance=provenance,
    )
    write_json(args.config_output, config)
    config_sha = sha256_path(args.config_output)
    all_cameras = selected["all_cameras"]
    primary_camera = selected["primary_camera"]
    stage_gate = (
        all_cameras["first_event_recall"] is not None
        and all_cameras["last_event_recall"] is not None
        and all_cameras["first_event_recall"] >= 0.50
        and all_cameras["last_event_recall"] >= 0.50
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task": "T1-6B",
        "generated_at": utc_now(),
        "source": {
            "sources": str(args.sources.resolve()),
            "sources_sha256": provenance["sources_sha256"],
            "specs": str(args.specs.resolve()),
            "specs_sha256": provenance["specs_sha256"],
            "fit3d_manifest": str(args.fit3d_manifest.resolve()),
            "fit3d_manifest_sha256": provenance["fit3d_manifest_sha256"],
            "analysis_dir": str(args.analysis_dir.resolve()),
        },
        "protocol": {
            "split": "Fit3D dev",
            "gt_semantics": "top_event_only",
            "prediction": "top_state_anchor_frame",
            "event_match": "one_to_one_hungarian",
            "event_gate": "0.25 x median GT top-to-top interval, min 1 frame",
            "signal_modes": list(PHASE_SIGNAL_MODES),
            "anchor_candidates": list(ANCHOR_POLICIES),
            "selection_scope": "single_global_signal_and_anchor_policy",
        },
        "selected": {
            "signal_mode": selected_signal_mode,
            "anchor_policy": selected_anchor,
            "config_path": str(args.config_output.resolve()),
            "config_sha256": config_sha,
            "selection_reason": provenance["selection_rule"],
            "all_cameras": all_cameras,
            "primary_camera": primary_camera,
            "t1_6b_stage_gate_passed": stage_gate,
        },
        "anchor_comparison": comparison_rows,
        "artifacts": {
            "state_intervals": "state_intervals.csv",
            "anchor_comparison": "anchor_comparison.csv",
            "clip_anchor_metrics": "clip_anchor_metrics.csv",
            "signal_selection_audit": "signal_selection_audit.csv",
            "checksums": "checksums.sha256",
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    state_fields = [
        "dataset",
        "clip_id",
        "video_id",
        "cluster_id",
        "subject_id",
        "camera_id",
        "action",
        "view",
        "view_angle_deg",
        "split_role",
        "is_primary_camera",
        "signal_mode",
        "phase_signal",
        "state_type",
        "state_index",
        "kind",
        "status",
        "start_frame",
        "end_frame",
        "duration_frames",
        "period_frames",
        "duration_period_fraction",
        "stable_fraction",
        "boundary_slope",
        "mean_angle",
        "anchor_plateau_entry",
        "anchor_plateau_midpoint",
        "anchor_plateau_exit",
        "anchor_phase_zero_crossing",
        "selected_signal_mode",
        "selected_anchor_policy",
        "selected_anchor_frame",
    ]
    write_csv(args.results_dir / "state_intervals.csv", state_rows, state_fields)
    comparison_fields = [
        "signal_mode",
        "anchor_policy",
        "scope",
        "clip_count",
        "cluster_count",
        "gt_event_count",
        "pred_event_count",
        "tp",
        "fp",
        "fn",
        "event_recall",
        "event_precision",
        "event_f1",
        "first_event_recall",
        "last_event_recall",
        "first_last_min_recall",
        "clip_event_recall_mean",
        "cluster_event_recall_mean",
        "event_abs_error_median_frames",
        "event_abs_error_p90_frames",
        "event_abs_error_max_frames",
        "event_signed_error_median_frames",
        "gt_in_any_top_interval_fraction",
        "matched_gt_in_top_interval_fraction",
        "status_counts",
    ]
    comparison_csv_rows = []
    for row in comparison_rows:
        for scope in ("all_cameras", "primary_camera"):
            comparison_csv_rows.append(
                {
                    "signal_mode": row["signal_mode"],
                    "anchor_policy": row["anchor_policy"],
                    **{
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in row[scope].items()
                    },
                }
            )
    write_csv(
        args.results_dir / "anchor_comparison.csv",
        comparison_csv_rows,
        comparison_fields,
    )
    clip_fields = [
        "dataset",
        "clip_id",
        "cluster_id",
        "subject_id",
        "camera_id",
        "action",
        "view",
        "view_angle_deg",
        "signal_mode",
        "anchor_policy",
        "phase_signal",
        "status",
        "period_frames",
        "top_state_count",
        "bottom_state_count",
        "rep_candidate_count",
        "endpoint_partial_count",
        "gt_event_count",
        "pred_event_count",
        "tp",
        "fp",
        "fn",
        "event_recall",
        "event_precision",
        "event_f1",
        "first_gt_matched",
        "last_gt_matched",
        "gt_in_any_top_interval_fraction",
        "matched_gt_in_top_interval_fraction",
        "event_abs_error_median_frames",
        "event_abs_error_p90_frames",
        "event_signed_error_median_frames",
        "event_gate_frames",
    ]
    for signal_mode in PHASE_SIGNAL_MODES:
        for anchor_policy in ANCHOR_POLICIES:
            rows = clip_rows[(signal_mode, anchor_policy)]
            public_rows = [
                {key: row.get(key) for key in clip_fields}
                for row in rows
            ]
            write_csv(
                args.results_dir
                / (
                    f"clip_anchor_metrics_{signal_mode}_{anchor_policy}.csv"
                ),
                public_rows,
                clip_fields,
            )
    write_csv(
        args.results_dir / "signal_selection_audit.csv",
        signal_audit_rows,
        [
            "clip_id",
            "cluster_id",
            "action",
            "prior_phase_angle",
            "auto_phase_angle",
            "auto_score",
            "bilateral_phase_signal",
            "auto_candidates_json",
        ],
    )
    write_json(args.results_dir / "state_eval_summary.json", summary)
    write_checksums(
        args.results_dir,
        [
            "state_intervals.csv",
            "anchor_comparison.csv",
            "signal_selection_audit.csv",
            "state_eval_summary.json",
            *[
                f"clip_anchor_metrics_{mode}_{anchor}.csv"
                for mode in PHASE_SIGNAL_MODES
                for anchor in ANCHOR_POLICIES
            ],
        ],
    )
    return summary


def main() -> int:
    summary = run_dev_evaluation(parse_args())
    selected = summary["selected"]
    all_cameras = selected["all_cameras"]
    print(
        "selected "
        f"{selected['signal_mode']} + {selected['anchor_policy']}: "
        f"recall={all_cameras['event_recall']} "
        f"first={all_cameras['first_event_recall']} "
        f"last={all_cameras['last_event_recall']} "
        f"median={all_cameras['event_abs_error_median_frames']} "
        f"p90={all_cameras['event_abs_error_p90_frames']}"
    )
    print(f"stage_gate={'pass' if selected['t1_6b_stage_gate_passed'] else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
