#!/usr/bin/env python3
"""Timebase-aware multi-scale phase signal analysis for RepChain v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_task1_6 as baseline_evaluation
import task1_6_chain_selection as chain_selection
import task1_6_phase_states as phase_states
import task1_6_quality as quality_gates


DEFAULT_CONFIG = ROOT / "configs" / "task1_7_signal_p1.json"
DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_SEGMENTER_CONFIG = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_CHAIN_CONFIG = ROOT / "configs" / "task1_6_chain.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_FROZEN_BASELINE_DIR = (
    ROOT / "results" / "task1_6" / "runs" / "quality_motion" / "fit3d_holdout"
)
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "results" / "task1_6" / "evaluation_quality_motion" / "summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_7" / "signals" / "dev"

CONFIG_SCHEMA_VERSION = "task1-7-signal-config-v1"
RESOLUTION_SCHEMA_VERSION = "task1-7-phase-resolution-v1"
EVALUATION_SCHEMA_VERSION = "task1-7b-signal-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument(
        "--segmenter-config", type=Path, default=DEFAULT_SEGMENTER_CONFIG
    )
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN_CONFIG)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--frozen-baseline-dir", type=Path, default=DEFAULT_FROZEN_BASELINE_DIR
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path, names: Iterable[str]) -> None:
    lines = [
        f"{sha256_path(directory / name)}  {name}"
        for name in sorted(names)
        if (directory / name).is_file()
    ]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected signal config: {config.get('schema_version')}")
    core = config.get("core") or {}
    if not core.get("dataset_independent"):
        raise ValueError("signal config must be dataset independent")
    scales = [float(value) for value in core.get("smoothing_scales_seconds") or []]
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("smoothing scales must be positive")
    return config


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_timebase(
    frame_count: int,
    fps: float | None,
    config: dict[str, Any],
    *,
    timestamps: Iterable[float] | None = None,
) -> dict[str, Any]:
    core = config["core"]
    settings = core["timebase"]
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    timestamps_present = timestamps is not None
    source_times = None
    effective_fps = None
    frame_dt_cv = None
    max_source_gap_seconds = 0.0
    if timestamps_present:
        source_times = np.asarray(list(timestamps), dtype=np.float64)
        if source_times.shape != (frame_count,):
            raise ValueError("timestamps length must match frame_count")
        if not np.isfinite(source_times).all():
            raise ValueError("timestamps must be finite")
        source_times = source_times - float(source_times[0])
        deltas = np.diff(source_times)
        if deltas.size and (deltas <= 0).any():
            raise ValueError("timestamps must be strictly increasing")
        if deltas.size:
            median_dt = float(np.median(deltas))
            effective_fps = 1.0 / median_dt if median_dt > 0 else None
            mean_dt = float(np.mean(deltas))
            frame_dt_cv = (
                float(np.std(deltas) / mean_dt) if mean_dt > 0 else None
            )
            max_source_gap_seconds = float(np.max(deltas))
        duration_seconds = float(source_times[-1]) if frame_count > 1 else 0.0
    else:
        if fps is None or float(fps) <= 0:
            raise ValueError("fps must be positive when timestamps are absent")
        effective_fps = float(fps)
        source_times = np.arange(frame_count, dtype=np.float64) / effective_fps
        duration_seconds = (
            float(source_times[-1]) if frame_count > 1 else 0.0
        )
        frame_dt_cv = 0.0
        max_source_gap_seconds = 1.0 / effective_fps

    minimum_fps = float(settings["minimum_analysis_fps"])
    maximum_fps = float(settings["maximum_analysis_fps"])
    analysis_fps = _clamp(float(effective_fps), minimum_fps, maximum_fps)
    tolerance = float(settings["variable_fps_cv_threshold"])
    variable_fps = bool(frame_dt_cv is not None and frame_dt_cv > tolerance)
    if variable_fps:
        output_count = max(2, int(round(duration_seconds * analysis_fps)) + 1)
        target_times = np.arange(output_count, dtype=np.float64) / analysis_fps
    elif abs(analysis_fps - float(fps or effective_fps)) <= 1e-9:
        output_count = frame_count
        target_times = source_times.copy()
    else:
        output_count = max(2, int(round(duration_seconds * analysis_fps)) + 1)
        target_times = np.arange(output_count, dtype=np.float64) / analysis_fps

    return {
        "frame_count": frame_count,
        "output_frame_count": output_count,
        "source_fps": float(fps) if fps is not None else None,
        "effective_fps": effective_fps,
        "analysis_fps": analysis_fps,
        "duration_seconds": duration_seconds,
        "timestamps_present": timestamps_present,
        "variable_fps": variable_fps,
        "frame_dt_cv": frame_dt_cv,
        "max_gap_seconds": max_source_gap_seconds,
        "resampled": output_count != frame_count,
        "source_times": source_times,
        "target_times": target_times,
    }


def _resample_series(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    max_unresolved_gap_seconds: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return np.full(target_times.shape, np.nan, dtype=np.float64)
    output = np.interp(
        target_times,
        source_times[valid],
        values[valid],
    )
    valid_indexes = np.flatnonzero(valid)
    for left_index, right_index in zip(valid_indexes[:-1], valid_indexes[1:]):
        if right_index - left_index <= 1:
            continue
        gap_seconds = float(
            source_times[right_index] - source_times[left_index]
        )
        if gap_seconds <= max_unresolved_gap_seconds:
            continue
        inside = (target_times > source_times[left_index]) & (
            target_times < source_times[right_index]
        )
        output[inside] = np.nan
    return output


def resample_signals(
    signals: dict[str, np.ndarray],
    timebase: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not timebase["resampled"]:
        return {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in signals.items()
        }
    maximum_gap = float(
        config["core"]["gaps"]["maximum_unresolved_gap_seconds"]
    )
    return {
        name: _resample_series(
            np.asarray(values, dtype=np.float64),
            timebase["source_times"],
            timebase["target_times"],
            maximum_gap,
        )
        for name, values in signals.items()
    }


def _missing_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(~valid)
    if indexes.size == 0:
        return []
    groups = np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
    return [(int(group[0]), int(group[-1])) for group in groups]


def interpolate_signal_gaps(
    values: Iterable[float],
    times: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(list(values), dtype=np.float64)
    valid = np.isfinite(values)
    original_valid_fraction = float(valid.mean()) if values.size else 0.0
    settings = config["core"]["gaps"]
    maximum_short_gap = float(settings["maximum_short_gap_seconds"])
    maximum_unresolved_gap = float(
        settings["maximum_unresolved_gap_seconds"]
    )
    maximum_missing_fraction = float(settings["maximum_missing_fraction"])
    filled = values.copy()
    short_gap_count = 0
    long_gap_count = 0
    unresolved_gap_count = 0
    maximum_gap_seconds = 0.0
    for start, end in _missing_runs(valid):
        left = start - 1
        right = end + 1
        if left < 0 or right >= values.size:
            unresolved_gap_count += 1
            continue
        if not np.isfinite(values[left]) or not np.isfinite(values[right]):
            unresolved_gap_count += 1
            continue
        gap_seconds = float(times[right] - times[left])
        maximum_gap_seconds = max(maximum_gap_seconds, gap_seconds)
        if gap_seconds > maximum_unresolved_gap:
            unresolved_gap_count += 1
            continue
        filled[start : end + 1] = np.interp(
            times[start : end + 1],
            [times[left], times[right]],
            [values[left], values[right]],
        )
        if gap_seconds <= maximum_short_gap:
            short_gap_count += 1
        else:
            long_gap_count += 1
    missing_fraction = (
        float(np.mean(~np.isfinite(values))) if values.size else 1.0
    )
    interpolated_fraction = (
        float(np.sum(~np.isfinite(values) & np.isfinite(filled)) / values.size)
        if values.size
        else 0.0
    )
    evidence = {
        "original_valid_fraction": original_valid_fraction,
        "missing_fraction": missing_fraction,
        "interpolated_fraction": interpolated_fraction,
        "short_gap_count": short_gap_count,
        "long_gap_count": long_gap_count,
        "unresolved_gap_count": unresolved_gap_count,
        "max_gap_seconds": maximum_gap_seconds,
        "status": "ok",
        "reasons": [],
    }
    if unresolved_gap_count:
        evidence["status"] = "insufficient_signal"
        evidence["reasons"].append("unresolved_gap_too_long")
    if missing_fraction > maximum_missing_fraction:
        evidence["status"] = "insufficient_signal"
        evidence["reasons"].append("missing_fraction_above_maximum")
    if original_valid_fraction < float(settings["minimum_original_valid_fraction"]):
        evidence["status"] = "insufficient_signal"
        evidence["reasons"].append("valid_fraction_below_minimum")
    return filled, evidence


def signal_family(name: str) -> str:
    family = name
    if family.endswith("_bilateral_mean"):
        family = family[: -len("_bilateral_mean")]
    for prefix in ("left_", "right_"):
        if family.startswith(prefix):
            family = family[len(prefix) :]
            break
    return family


def _prior_signal_factor(
    candidate: dict[str, Any],
    prior_signal: str | None,
    settings: dict[str, Any],
) -> float:
    if prior_signal and candidate["family"] == signal_family(prior_signal):
        return 1.0 + float(settings["prior_signal_bonus"])
    return 1.0


def _scale_detector(
    base_detector: dict[str, Any] | None,
    scale_seconds: float,
) -> dict[str, Any]:
    detector = dict(base_detector or {})
    detector["smoothing_seconds"] = float(scale_seconds)
    return detector


def _motion_gate_detector(
    base_detector: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    detector = dict(base_detector or {})
    policy = (
        (config.get("core") or {}).get("motion_gate") or {}
    ).get("sampling_aware_state_amplitude") or {}
    if not bool(policy.get("enabled")):
        return detector
    detector["motion_joint_state_amplitude"] = {
        "enabled": True,
        "minimum_analysis_fps": policy.get("minimum_analysis_fps"),
        "maximum_analysis_fps": policy.get("maximum_analysis_fps"),
        "minimum_amplitude_fraction": float(
            policy.get("minimum_amplitude_fraction", 0.75)
        ),
        "minimum_periodicity": float(
            policy.get("minimum_periodicity", 0.35)
        ),
        "minimum_dynamic_range_multiple": float(
            policy.get("minimum_dynamic_range_multiple", 2.0)
        ),
        "minimum_dynamic_to_noise_multiple": float(
            policy.get("minimum_dynamic_to_noise_multiple", 1.5)
        ),
    }
    return detector


def _scale_prior(
    scale_seconds: float,
    base_detector: dict[str, Any] | None,
    settings: dict[str, Any],
) -> float:
    prior = float((base_detector or {}).get("smoothing_seconds", 0.12))
    sigma = float(settings["scale_prior_log_sigma"])
    distance = math.log(max(scale_seconds, 1e-6) / max(prior, 1e-6))
    return math.exp(-0.5 * (distance / max(sigma, 1e-6)) ** 2)


def _select_period_consensus(
    scale_results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    tolerance = float(settings["period_cluster_tolerance_fraction"])
    groups = []
    for seed in scale_results:
        period = float(seed["period_frames"])
        members = [
            item
            for item in scale_results
            if abs(float(item["period_frames"]) - period)
            / max(period, 1e-6)
            <= tolerance
        ]
        groups.append(members)
    if not groups:
        raise ValueError("cannot cluster an empty period set")
    best = max(
        groups,
        key=lambda members: (
            len(members),
            -float(np.median([item["smoothing_seconds"] for item in members])),
        ),
    )
    periods = [float(item["period_frames"]) for item in best]
    return float(np.median(periods)), sorted(
        best, key=lambda item: item["smoothing_seconds"]
    )


def _cluster_events_across_scales(
    scale_results: list[dict[str, Any]],
    period_frames: float,
    settings: dict[str, Any],
    *,
    state_key: str,
    selected_scale_seconds: float | None = None,
) -> list[dict[str, Any]]:
    tolerance = (
        float(settings["anchor_tolerance_period_fraction"])
        * max(period_frames, 1.0)
    )
    records = sorted(
        [
            {
                "scale": float(result["smoothing_seconds"]),
                "state": state,
                "anchor": float(state["anchor_frame"]),
                "selected_scale": (
                    selected_scale_seconds is not None
                    and math.isclose(
                        float(result["smoothing_seconds"]),
                        float(selected_scale_seconds),
                    )
                ),
            }
            for result in scale_results
            for state in result.get(state_key) or []
        ],
        key=lambda item: (item["anchor"], item["scale"]),
    )
    clusters: list[dict[str, Any]] = []
    for record in records:
        candidates = []
        for cluster in clusters:
            anchors = [item["anchor"] for item in cluster["members"]]
            median_anchor = float(np.median(anchors + [record["anchor"]]))
            spread = max(anchors + [record["anchor"]]) - min(
                anchors + [record["anchor"]]
            )
            distance = abs(record["anchor"] - median_anchor)
            if distance <= tolerance and spread <= 2.0 * tolerance:
                candidates.append((distance, cluster))
        if not candidates:
            clusters.append({"members": [record]})
            continue
        candidates.sort(key=lambda item: item[0])
        candidates[0][1]["members"].append(record)

    active_scale_count = max(
        1, len({float(result["smoothing_seconds"]) for result in scale_results})
    )
    minimum_support = int(settings["minimum_interior_scale_support"])
    minimum_edge_support = int(settings["minimum_edge_scale_support"])
    include_edge_singletons = bool(settings["include_edge_singletons"])
    states = []
    for cluster in clusters:
        members = cluster["members"]
        support_scales = sorted({item["scale"] for item in members})
        support_fraction = len(support_scales) / active_scale_count
        selected_scale_anchor = any(
            item["selected_scale"] for item in members
        )
        member_states = [item["state"] for item in members]
        best = max(
            member_states,
            key=lambda state: (
                float(state.get("stable_fraction", 0.0)),
                -float(state.get("boundary_slope", 0.0)),
            ),
        )
        kind_counts = defaultdict(int)
        for state in member_states:
            kind_counts[str(state.get("kind") or "interior")] += 1
        kind = max(kind_counts, key=lambda value: (kind_counts[value], value))
        if (
            kind == "interior"
            and active_scale_count > 1
            and len(support_scales) < min(minimum_support, active_scale_count)
            and not (
                selected_scale_anchor
                and bool(settings["preserve_selected_scale_anchors"])
            )
        ):
            continue
        if kind != "interior":
            if not include_edge_singletons:
                continue
            if len(support_scales) < min(
                minimum_edge_support, active_scale_count
            ):
                continue
        anchors = sorted(int(state["anchor_frame"]) for state in member_states)
        anchor_frame = int(round(float(np.median(anchors))))
        start_frame = int(
            round(float(np.median([state["start_frame"] for state in member_states])))
        )
        end_frame = int(
            round(float(np.median([state["end_frame"] for state in member_states])))
        )
        start_frame = min(start_frame, anchor_frame)
        end_frame = max(end_frame, anchor_frame)
        anchor_names = set().union(
            *[
                set((state.get("anchors") or {}).keys())
                for state in member_states
            ]
        )
        anchors_consensus = {
            name: int(
                round(
                    float(
                        np.median(
                            [
                                int((state.get("anchors") or {})[name])
                                for state in member_states
                                if name in (state.get("anchors") or {})
                            ]
                        )
                    )
                )
            )
            for name in sorted(anchor_names)
        }
        evidence = {
            **(best.get("evidence") or {}),
            "scale_support": len(support_scales),
            "scale_support_fraction": round(float(support_fraction), 6),
            "scale_members": support_scales,
            "member_count": len(members),
            "anchor_spread_frames": max(anchors) - min(anchors),
            "multi_scale_consensus": len(support_scales) >= 2,
            "selected_scale_anchor": selected_scale_anchor,
        }
        flags = sorted(
            set().union(*[set(state.get("quality_flags") or []) for state in member_states])
            | ({"multi_scale_consensus"} if len(support_scales) >= 2 else set())
        )
        states.append(
            {
                **best,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "anchor_frame": anchor_frame,
                "anchors": anchors_consensus,
                "mean_angle": round(
                    float(np.median([state["mean_angle"] for state in member_states])),
                    6,
                ),
                "stable_fraction": round(
                    float(
                        np.median(
                            [state.get("stable_fraction", 0.0) for state in member_states]
                        )
                    ),
                    6,
                ),
                "boundary_slope": round(
                    float(
                        np.median(
                            [state.get("boundary_slope", 0.0) for state in member_states]
                        )
                    ),
                    6,
                ),
                "kind": kind,
                "evidence": evidence,
                "quality_flags": flags,
            }
        )
    states.sort(key=lambda state: state["anchor_frame"])
    for index, state in enumerate(states, start=1):
        state["state_index"] = index
    return states


def _suppress_near_duplicate_events(
    states: list[dict[str, Any]],
    period_frames: float,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep one event when two consensus clusters describe the same peak."""

    if len(states) < 2 or period_frames <= 0:
        return states
    minimum_gap = (
        float(settings["duplicate_spacing_period_fraction"]) * period_frames
    )
    groups: list[list[dict[str, Any]]] = []
    for state in sorted(states, key=lambda item: item["anchor_frame"]):
        if groups:
            previous_anchor = float(groups[-1][-1]["anchor_frame"])
            if float(state["anchor_frame"]) - previous_anchor < minimum_gap:
                groups[-1].append(state)
                continue
        groups.append([state])

    def strength(state: dict[str, Any]) -> tuple[float, ...]:
        evidence = state.get("evidence") or {}
        return (
            float(bool(evidence.get("multi_scale_consensus"))),
            float(evidence.get("scale_support_fraction") or 0.0),
            float(state.get("kind") == "interior"),
            float(evidence.get("member_count") or 0),
            float(state.get("stable_fraction") or 0.0),
            -float(state["anchor_frame"]),
        )

    kept = []
    for group in groups:
        winner = max(group, key=strength)
        if len(group) > 1:
            suppressed = [int(item["anchor_frame"]) for item in group if item is not winner]
            evidence = dict(winner.get("evidence") or {})
            evidence["near_duplicate_suppressed_frames"] = sorted(suppressed)
            evidence["near_duplicate_suppressed_count"] = len(suppressed)
            winner = {
                **winner,
                "evidence": evidence,
                "quality_flags": sorted(
                    set(winner.get("quality_flags") or [])
                    | {"near_duplicate_suppressed"}
                ),
            }
        kept.append(winner)
    for index, state in enumerate(kept, start=1):
        state["state_index"] = index
    return kept


def _recovered_state_evidence(
    state: dict[str, Any],
    *,
    role: str,
    predicted_anchor: float,
    period_frames: float,
    scale_seconds: list[float],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    evidence = dict(state.get("evidence") or {})
    anchor = float(state["anchor_frame"])
    error = anchor - float(predicted_anchor)
    recovery = settings["half_cycle_recovery"]
    anchor_sigma = float(
        recovery["predicted_anchor_tolerance_period_fraction"]
    ) * max(float(period_frames), 1.0)
    closeness = math.exp(-0.5 * (error / max(anchor_sigma, 1e-6)) ** 2)
    stability = max(0.0, min(1.0, float(state.get("stable_fraction") or 0.0)))
    if role == "bottom":
        stability = max(stability, 0.2)
    support_factor = (
        1.0
        if role == "bottom"
        else min(1.0, 0.75 + 0.1 * (len(scale_seconds) - 1))
    )
    confidence = float(np.clip(closeness * stability * support_factor, 0.0, 1.0))
    evidence.update(
        {
            "half_cycle_recovered": True,
            "recovery_role": role,
            "recovery_scale_seconds": scale_seconds[0],
            "recovery_scale_seconds_all": scale_seconds,
            "recovery_anchor_error_frames": round(float(error), 6),
            "predicted_anchor_frame": round(float(predicted_anchor), 6),
            "recovery_confidence": round(confidence, 6),
            "scale_support": len(scale_seconds),
            "multi_scale_consensus": False,
            "scale_members": scale_seconds,
        }
    )
    state = {
        **state,
        "kind": "interior",
        "evidence": evidence,
        "quality_flags": sorted(
            set(state.get("quality_flags") or [])
            | {f"recovered_{role}_half_cycle"}
        ),
    }
    return state, confidence


def _recover_missing_tops(
    top_states: list[dict[str, Any]],
    scale_results: list[dict[str, Any]],
    period_frames: float,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovery = settings["half_cycle_recovery"]
    minimum_gap, maximum_gap = (
        float(value) for value in recovery["gap_period_range"]
    )
    minimum_neighbor_gap, maximum_neighbor_gap = (
        float(value) for value in recovery["neighbor_gap_period_range"]
    )
    tolerance = (
        float(recovery["predicted_anchor_tolerance_period_fraction"])
        * max(float(period_frames), 1.0)
    )
    tops = sorted(top_states, key=lambda state: state["anchor_frame"])
    recovered = []
    records = []
    for left, right in zip(tops[:-1], tops[1:]):
        gap = float(right["anchor_frame"] - left["anchor_frame"])
        gap_ratio = gap / max(float(period_frames), 1e-6)
        if not minimum_gap <= gap_ratio <= maximum_gap:
            near_boundary = _near_boundary_rejected_top(
                left,
                right,
                scale_results,
                period_frames,
                gap_ratio,
                settings,
            )
            if near_boundary is not None:
                state, record = near_boundary
                recovered.append(state)
                records.append(record)
            continue
        predicted = 0.5 * (
            float(left["anchor_frame"]) + float(right["anchor_frame"])
        )
        candidates = []
        for result in scale_results:
            for state in result.get("top_states") or []:
                if state.get("kind") != "interior":
                    continue
                error = float(state["anchor_frame"]) - predicted
                if abs(error) > tolerance:
                    continue
                left_ratio = (
                    float(state["anchor_frame"]) - float(left["anchor_frame"])
                ) / max(float(period_frames), 1e-6)
                right_ratio = (
                    float(right["anchor_frame"]) - float(state["anchor_frame"])
                ) / max(float(period_frames), 1e-6)
                if not (
                    minimum_neighbor_gap
                    <= left_ratio
                    <= maximum_neighbor_gap
                    and minimum_neighbor_gap
                    <= right_ratio
                    <= maximum_neighbor_gap
                ):
                    continue
                candidates.append((result["smoothing_seconds"], state, error))
        if not candidates:
            continue
        scale_seconds = sorted({float(item[0]) for item in candidates})
        _, best_state, _ = max(
            candidates,
            key=lambda item: (
                float(item[1].get("stable_fraction") or 0.0),
                -abs(float(item[2])),
                float(item[0]),
            ),
        )
        state, confidence = _recovered_state_evidence(
            best_state,
            role="top",
            predicted_anchor=predicted,
            period_frames=period_frames,
            scale_seconds=scale_seconds,
            settings=settings,
        )
        if confidence < float(recovery["minimum_recovery_confidence"]):
            continue
        recovered.append(state)
        records.append(
            {
                "role": "top",
                "gap_frames": int(round(gap)),
                "gap_ratio": round(float(gap_ratio), 6),
                "predicted_anchor_frame": round(float(predicted), 6),
                "anchor_frame": int(state["anchor_frame"]),
                "anchor_error_frames": round(
                    float(state["anchor_frame"]) - predicted, 6
                ),
                "scale_seconds": scale_seconds,
                "confidence": round(float(confidence), 6),
            }
        )
    if not recovered:
        return tops, []
    merged = sorted([*tops, *recovered], key=lambda state: state["anchor_frame"])
    for index, state in enumerate(merged, start=1):
        state["state_index"] = index
    return merged, records


def _near_boundary_rejected_top(
    left: dict[str, Any],
    right: dict[str, Any],
    scale_results: list[dict[str, Any]],
    period_frames: float,
    gap_ratio: float,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    recovery = settings["half_cycle_recovery"]
    policy = recovery.get("near_boundary_rejected_top") or {}
    if not bool(policy.get("enabled")):
        return None
    minimum_gap, maximum_gap = (
        float(value)
        for value in policy.get("gap_period_range", [2.45, 2.65])
    )
    if not minimum_gap <= gap_ratio <= maximum_gap:
        return None
    minimum_relative_peak = float(
        policy.get("minimum_relative_peak_fraction", 0.4)
    )
    minimum_stability = float(policy.get("minimum_stability", 0.5))
    minimum_support = int(policy.get("minimum_scale_support", 3))
    maximum_anchor_spread = float(
        policy.get("maximum_anchor_spread_period_fraction", 0.2)
    )
    minimum_neighbor_gap, maximum_neighbor_gap = (
        float(value) for value in recovery["neighbor_gap_period_range"]
    )
    anchor_tolerance = (
        float(recovery["predicted_anchor_tolerance_period_fraction"])
        * max(float(period_frames), 1.0)
    )
    left_anchor = float(left["anchor_frame"])
    right_anchor = float(right["anchor_frame"])
    candidates = []
    for result in scale_results:
        scale_seconds = float(result["smoothing_seconds"])
        for state in result.get("rejected_top_states") or []:
            if state.get("reason") != "no_stable_top_state":
                continue
            if float(state.get("stable_fraction") or 0.0) < minimum_stability:
                continue
            relative_peak = float(state.get("relative_peak_fraction") or 0.0)
            if relative_peak < minimum_relative_peak:
                continue
            anchor = float(state["anchor_frame"])
            if not left_anchor < anchor < right_anchor:
                continue
            left_ratio = (
                anchor - left_anchor
            ) / max(float(period_frames), 1e-6)
            right_ratio = (
                right_anchor - anchor
            ) / max(float(period_frames), 1e-6)
            if not (
                minimum_neighbor_gap
                <= left_ratio
                <= maximum_neighbor_gap
                and minimum_neighbor_gap
                <= right_ratio
                <= maximum_neighbor_gap
            ):
                continue
            candidates.append(
                {
                    "state": state,
                    "scale_seconds": scale_seconds,
                    "anchor": anchor,
                    "relative_peak_fraction": relative_peak,
                    "left_ratio": left_ratio,
                    "right_ratio": right_ratio,
                }
            )
    if not candidates:
        return None

    groups: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: item["anchor"]):
        if groups:
            group_anchor = float(
                np.median([item["anchor"] for item in groups[-1]])
            )
            if abs(candidate["anchor"] - group_anchor) <= anchor_tolerance:
                groups[-1].append(candidate)
                continue
        groups.append([candidate])
    supported = []
    for group in groups:
        scale_seconds = sorted({item["scale_seconds"] for item in group})
        if len(scale_seconds) < minimum_support:
            continue
        anchors = [item["anchor"] for item in group]
        if max(anchors) - min(anchors) > maximum_anchor_spread * period_frames:
            continue
        supported.append((group, scale_seconds))
    if not supported:
        return None
    group, scale_seconds = max(
        supported,
        key=lambda item: (
            len(item[1]),
            float(
                np.median(
                    [
                        candidate["relative_peak_fraction"]
                        for candidate in item[0]
                    ]
                )
            ),
            float(
                np.median(
                    [candidate["state"]["stable_fraction"] for candidate in item[0]]
                )
            ),
        ),
    )
    best = max(
        group,
        key=lambda item: (
            float(item["state"].get("stable_fraction") or 0.0),
            item["relative_peak_fraction"],
        ),
    )
    predicted_anchor = 0.5 * (left_anchor + right_anchor)
    state_for_recovery = {
        **best["state"],
        "evidence": {
            **(best["state"].get("evidence") or {}),
            "relative_peak_fraction": round(
                float(best["relative_peak_fraction"]), 6
            ),
        },
    }
    recovered_state, confidence = _recovered_state_evidence(
        state_for_recovery,
        role="top",
        predicted_anchor=predicted_anchor,
        period_frames=period_frames,
        scale_seconds=scale_seconds,
        settings=settings,
    )
    if confidence < float(recovery["minimum_recovery_confidence"]):
        return None
    anchors = [candidate["anchor"] for candidate in group]
    evidence = recovered_state["evidence"]
    evidence.update(
        {
            "near_boundary_rejected_top": True,
            "near_boundary_gap_ratio": round(float(gap_ratio), 6),
            "near_boundary_left_ratio": round(
                float(best["left_ratio"]), 6
            ),
            "near_boundary_right_ratio": round(
                float(best["right_ratio"]), 6
            ),
            "near_boundary_anchor_spread_frames": round(
                float(max(anchors) - min(anchors)), 6
            ),
            "near_boundary_relative_peak_fraction": round(
                float(best["relative_peak_fraction"]), 6
            ),
        }
    )
    recovered_state["quality_flags"] = sorted(
        set(recovered_state.get("quality_flags") or [])
        | {"recovered_top_near_boundary"}
    )
    record = {
        "role": "top",
        "recovery_source": "near_boundary_rejected_top",
        "gap_frames": int(round(right_anchor - left_anchor)),
        "gap_ratio": round(float(gap_ratio), 6),
        "predicted_anchor_frame": round(float(predicted_anchor), 6),
        "anchor_frame": int(recovered_state["anchor_frame"]),
        "anchor_error_frames": round(
            float(recovered_state["anchor_frame"]) - predicted_anchor, 6
        ),
        "scale_seconds": scale_seconds,
        "relative_peak_fraction": round(
            float(best["relative_peak_fraction"]), 6
        ),
        "left_ratio": round(float(best["left_ratio"]), 6),
        "right_ratio": round(float(best["right_ratio"]), 6),
        "confidence": round(float(confidence), 6),
    }
    return recovered_state, record


def _recover_missing_bottoms(
    top_states: list[dict[str, Any]],
    bottom_states: list[dict[str, Any]],
    scale_results: list[dict[str, Any]],
    period_frames: float,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovery = settings["half_cycle_recovery"]
    minimum_gap, maximum_gap = (
        float(value) for value in recovery["neighbor_gap_period_range"]
    )
    bottoms = sorted(bottom_states, key=lambda state: state["anchor_frame"])
    tops = sorted(top_states, key=lambda state: state["anchor_frame"])
    recovered = []
    records = []
    for left, right in zip(tops[:-1], tops[1:]):
        left_anchor = float(left["anchor_frame"])
        right_anchor = float(right["anchor_frame"])
        gap_ratio = (right_anchor - left_anchor) / max(
            float(period_frames), 1e-6
        )
        if not minimum_gap <= gap_ratio <= maximum_gap:
            continue
        if any(
            left_anchor < float(state["anchor_frame"]) < right_anchor
            for state in bottoms
        ):
            continue
        predicted = 0.5 * (left_anchor + right_anchor)
        candidates = []
        for result in scale_results:
            for state in result.get("bottom_states") or []:
                anchor = float(state["anchor_frame"])
                if not left_anchor < anchor < right_anchor:
                    continue
                candidates.append((result["smoothing_seconds"], state))
        if not candidates:
            continue
        scale_seconds = sorted({float(item[0]) for item in candidates})
        _, best_state = max(
            candidates,
            key=lambda item: (
                float(item[1].get("stable_fraction") or 0.0),
                -abs(float(item[1]["anchor_frame"]) - predicted),
                float(item[0]),
            ),
        )
        state, confidence = _recovered_state_evidence(
            best_state,
            role="bottom",
            predicted_anchor=predicted,
            period_frames=period_frames,
            scale_seconds=scale_seconds,
            settings=settings,
        )
        if confidence < float(recovery["minimum_bottom_recovery_confidence"]):
            continue
        recovered.append(state)
        records.append(
            {
                "role": "bottom",
                "predicted_anchor_frame": round(float(predicted), 6),
                "anchor_frame": int(state["anchor_frame"]),
                "anchor_error_frames": round(
                    float(state["anchor_frame"]) - predicted, 6
                ),
                "scale_seconds": scale_seconds,
                "confidence": round(float(confidence), 6),
            }
        )
    if not recovered:
        return bottoms, []
    merged = sorted(
        [*bottoms, *recovered], key=lambda state: state["anchor_frame"]
    )
    for index, state in enumerate(merged, start=1):
        state["state_index"] = index
    return merged, records


def build_multiscale_state_consensus(
    values: Iterable[float],
    fps: float,
    action: dict[str, Any],
    phase_signal: str,
    config: dict[str, Any],
    *,
    detector: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = config["core"]["event_consensus"]
    values = np.asarray(list(values), dtype=np.float64)
    chain_policy = {
        "minimum_chain_consensus_events": int(
            settings["minimum_chain_consensus_events"]
        ),
        "endpoint_closure_gap_period_range": [
            float(value)
            for value in settings["edge_closure_gap_period_range"]
        ],
        "low_sampling_endpoint_pruning": {
            "enabled": bool(
                float(fps)
                <= float(
                    settings["low_sampling_endpoint_pruning"][
                        "maximum_analysis_fps"
                    ]
                )
            ),
            "minimum_retained_events": int(
                settings["low_sampling_endpoint_pruning"][
                    "minimum_retained_events"
                ]
            ),
            "maximum_pruned_endpoint_stability": float(
                settings["low_sampling_endpoint_pruning"][
                    "maximum_pruned_endpoint_stability"
                ]
            ),
        },
        "endpoint_timing_model": {
            "enabled": bool(settings["endpoint_timing_model"]["enabled"]),
            "context_intervals": int(
                settings["endpoint_timing_model"]["context_intervals"]
            ),
            "minimum_context_intervals": int(
                settings["endpoint_timing_model"][
                    "minimum_context_intervals"
                ]
            ),
            "maximum_context_relative_mad": float(
                settings["endpoint_timing_model"][
                    "maximum_context_relative_mad"
                ]
            ),
            "minimum_retained_events": int(
                settings["endpoint_timing_model"]["minimum_retained_events"]
            ),
            "retain_partial_gap_period_range": [
                float(value)
                for value in settings["endpoint_timing_model"][
                    "retain_partial_gap_period_range"
                ]
            ],
            "retain_partial_local_gap_ratio_range": [
                float(value)
                for value in settings["endpoint_timing_model"][
                    "retain_partial_local_gap_ratio_range"
                ]
            ],
            "short_partial_local_gap_ratio": float(
                settings["endpoint_timing_model"][
                    "short_partial_local_gap_ratio"
                ]
            ),
            "long_partial_local_gap_ratio": float(
                settings["endpoint_timing_model"][
                    "long_partial_local_gap_ratio"
                ]
            ),
        },
        "local_extrema_stability": {
            "enabled": bool(
                settings["local_extrema_stability"]["enabled"]
            ),
            "gap_period_range": [
                float(value)
                for value in settings["local_extrema_stability"][
                    "gap_period_range"
                ]
            ],
            "minimum_relative_amplitude": float(
                settings["local_extrema_stability"][
                    "minimum_relative_amplitude"
                ]
            ),
            "minimum_top_stability": float(
                settings["local_extrema_stability"][
                    "minimum_top_stability"
                ]
            ),
            "minimum_bottom_stability": float(
                settings["local_extrema_stability"][
                    "minimum_bottom_stability"
                ]
            ),
            "minimum_bottom_scale_support": int(
                settings["local_extrema_stability"][
                    "minimum_bottom_scale_support"
                ]
            ),
        },
        "low_amplitude_context_bridge": {
            "enabled": bool(
                float(fps)
                <= float(
                    settings["low_amplitude_context_bridge"][
                        "maximum_analysis_fps"
                    ]
                )
            ),
            "minimum_relative_amplitude": float(
                settings["low_amplitude_context_bridge"][
                    "minimum_relative_amplitude"
                ]
            ),
            "minimum_top_stability": float(
                settings["low_amplitude_context_bridge"][
                    "minimum_top_stability"
                ]
            ),
            "minimum_bottom_stability": float(
                settings["low_amplitude_context_bridge"][
                    "minimum_bottom_stability"
                ]
            ),
            "minimum_scale_support_fraction": float(
                settings["low_amplitude_context_bridge"][
                    "minimum_scale_support_fraction"
                ]
            ),
            "minimum_bridge_confidence": float(
                settings["low_amplitude_context_bridge"][
                    "minimum_bridge_confidence"
                ]
            ),
        },
        "local_period_gap": {
            "enabled": bool(settings["local_period_gap"]["enabled"]),
            "context_intervals_per_side": int(
                settings["local_period_gap"]["context_intervals_per_side"]
            ),
            "minimum_context_intervals": int(
                settings["local_period_gap"]["minimum_context_intervals"]
            ),
            "maximum_relative_mad": float(
                settings["local_period_gap"]["maximum_relative_mad"]
            ),
        },
    }
    scale_results = []
    selected_scale_seconds = float(
        (detector or {}).get(
            "smoothing_seconds",
            config["core"]["smoothing_scales_seconds"][0],
        )
    )
    for scale_seconds in config["core"]["smoothing_scales_seconds"]:
        scale_detector = _scale_detector(detector, float(scale_seconds))
        states = phase_states.detect_phase_states(
            values,
            fps,
            action,
            phase_signal,
            "plateau_midpoint",
            scale_detector,
        )
        if states.get("period_frames") is None:
            continue
        scale_results.append(
            {
                "smoothing_seconds": float(scale_seconds),
                "period_frames": float(states["period_frames"]),
                "periodicity": float(states.get("periodicity") or 0.0),
                "states": states,
                "top_states": states.get("top_states") or [],
                "bottom_states": states.get("bottom_states") or [],
                "rejected_top_states": [
                    state
                    for state in states.get("rejected_states") or []
                    if state.get("kind") == "interior"
                    and state.get("reason") == "no_stable_top_state"
                ],
            }
        )
    if not scale_results:
        return {
            "status": "no_motion",
            "motion_status": "no_motion",
            "phase_signal": phase_signal,
            "period_frames": None,
            "periodicity": 0.0,
            "top_states": [],
            "bottom_states": [],
            "motion_evidence": {},
            "evidence": {
                "status": "no_motion",
                "scale_results": [],
                "reasons": ["no_periodic_scale"],
            },
            "chain_policy": chain_policy,
        }
    period_frames, period_members = _select_period_consensus(
        scale_results, settings
    )
    top_states = _cluster_events_across_scales(
        period_members,
        period_frames,
        settings,
        state_key="top_states",
        selected_scale_seconds=selected_scale_seconds,
    )
    raw_top_count = len(top_states)
    top_states = _suppress_near_duplicate_events(
        top_states, period_frames, settings
    )
    deduplicated_top_count = len(top_states)
    bottom_states = _cluster_events_across_scales(
        period_members,
        period_frames,
        settings,
        state_key="bottom_states",
        selected_scale_seconds=selected_scale_seconds,
    )
    raw_bottom_count = len(bottom_states)
    bottom_states = _suppress_near_duplicate_events(
        bottom_states, period_frames, settings
    )
    deduplicated_bottom_count = len(bottom_states)
    top_states, recovered_tops = _recover_missing_tops(
        top_states, period_members, period_frames, settings
    )
    bottom_states, recovered_bottoms = _recover_missing_bottoms(
        top_states,
        bottom_states,
        period_members,
        period_frames,
        settings,
    )
    periodicity = float(
        np.median([item["periodicity"] for item in period_members])
    )
    best_motion = max(
        period_members,
        key=lambda item: (
            (item["states"].get("motion_evidence") or {}).get("status") == "ok",
            item["periodicity"],
        ),
    )
    evidence = {
        "status": "ok" if top_states else "insufficient_consensus",
        "period_frames": round(float(period_frames), 6),
        "periodicity": round(periodicity, 6),
        "scale_members": [
            item["smoothing_seconds"] for item in period_members
        ],
        "scale_periods": {
            f"{item['smoothing_seconds']:g}": round(
                float(item["period_frames"]), 6
            )
            for item in period_members
        },
        "top_cluster_count": len(top_states),
        "bottom_cluster_count": len(bottom_states),
        "raw_top_cluster_count": raw_top_count,
        "raw_bottom_cluster_count": raw_bottom_count,
        "near_duplicate_top_count": raw_top_count - deduplicated_top_count,
        "near_duplicate_bottom_count": (
            raw_bottom_count - deduplicated_bottom_count
        ),
        "consensus_top_count": sum(
            bool((state.get("evidence") or {}).get("multi_scale_consensus"))
            for state in top_states
        ),
        "recovered_top_count": len(recovered_tops),
        "recovered_bottom_count": len(recovered_bottoms),
        "near_boundary_recovered_top_count": sum(
            record.get("recovery_source") == "near_boundary_rejected_top"
            for record in recovered_tops
        ),
        "motion_joint_override_count": sum(
            bool(
                (
                    (item["states"].get("motion_evidence") or {}).get(
                        "joint_state_amplitude"
                    )
                    or {}
                ).get("override_applied")
            )
            for item in period_members
        ),
        "half_cycle_recovery": [*recovered_tops, *recovered_bottoms],
    }
    return {
        "status": evidence["status"],
        "motion_status": best_motion["states"].get("motion_status"),
        "phase_signal": phase_signal,
        "period_frames": period_frames,
        "periodicity": periodicity,
        "top_states": top_states,
        "bottom_states": bottom_states,
        "motion_evidence": best_motion["states"].get("motion_evidence"),
        "evidence": evidence,
        "chain_policy": chain_policy,
    }


def _candidate_score(
    prepared: dict[str, Any],
    gap_evidence: dict[str, Any],
    settings: dict[str, Any],
) -> float:
    dynamic_to_noise = max(float(prepared.get("dynamic_to_noise") or 0.0), 0.0)
    snr_score = min(
        1.0,
        math.log1p(dynamic_to_noise)
        / math.log1p(float(settings["target_dynamic_to_noise"])),
    )
    periodicity = max(0.0, min(1.0, float(prepared["periodicity"])))
    valid_fraction = max(
        0.0,
        min(1.0, float(gap_evidence["original_valid_fraction"])),
    )
    interpolated_fraction = max(
        0.0, min(1.0, float(gap_evidence["interpolated_fraction"]))
    )
    gap_score = max(
        0.0,
        1.0
        - interpolated_fraction
        / max(
            float(settings["gaps"]["maximum_missing_fraction"]),
            1e-6,
        ),
    )
    return float(
        periodicity ** 0.60
        * snr_score ** 0.25
        * valid_fraction ** 0.10
        * max(gap_score, 0.05) ** 0.05
    )


def _candidate_event_evidence(
    values: Iterable[float],
    *,
    name: str,
    fps: float,
    action: dict[str, Any],
    detector: dict[str, Any] | None,
) -> dict[str, Any]:
    states = phase_states.detect_phase_states(
        values,
        fps,
        action,
        name,
        "plateau_midpoint",
        detector,
    )
    top_states = states.get("top_states") or []
    bottom_states = states.get("bottom_states") or []
    return {
        "status": states.get("status"),
        "period_frames": (
            round(float(states["period_frames"]), 6)
            if states.get("period_frames") is not None
            else None
        ),
        "periodicity": (
            round(float(states.get("periodicity") or 0.0), 6)
            if states.get("periodicity") is not None
            else None
        ),
        "top_anchors": [
            int(state["anchor_frame"]) for state in top_states
        ],
        "top_kinds": [
            str(state.get("kind") or "interior") for state in top_states
        ],
        "bottom_anchors": [
            int(state["anchor_frame"]) for state in bottom_states
        ],
    }


def evaluate_signal_candidate(
    values: Iterable[float],
    *,
    name: str,
    fps: float,
    action: dict[str, Any],
    config: dict[str, Any],
    detector: dict[str, Any] | None,
    bilateral: bool,
) -> dict[str, Any]:
    settings = config["core"]
    values = np.asarray(list(values), dtype=np.float64)
    times = np.arange(len(values), dtype=np.float64) / max(fps, 1e-6)
    filled, gap_evidence = interpolate_signal_gaps(values, times, config)
    unresolved_gap_only = bool(
        gap_evidence["reasons"]
        and all(
            reason == "unresolved_gap_too_long"
            for reason in gap_evidence["reasons"]
        )
    )
    if unresolved_gap_only:
        gap_evidence = {
            **gap_evidence,
            "status": "degraded",
            "reasons": [
                *gap_evidence["reasons"],
                "unresolved_gap_interpolated_for_detection",
            ],
        }
    scales = []
    if gap_evidence["status"] in {"ok", "degraded"}:
        for scale_seconds in settings["smoothing_scales_seconds"]:
            scale_seconds = float(scale_seconds)
            scale_detector = _scale_detector(detector, scale_seconds)
            prepared = phase_states.prepare_analysis_signal(
                filled if gap_evidence["status"] == "ok" else values,
                fps,
                action,
                scale_detector,
            )
            if prepared is None:
                continue
            base_score = _candidate_score(prepared, gap_evidence, settings)
            prior = _scale_prior(scale_seconds, detector, settings)
            scales.append(
                {
                    "smoothing_seconds": scale_seconds,
                    "detector": scale_detector,
                    "prepared": prepared,
                    "base_score": base_score,
                    "adjusted_score": base_score
                    * (1.0 + float(settings["scale_prior_bonus"]) * prior),
                }
            )
    if not scales:
        return {
            "phase_signal": name,
            "score": 0.0,
            "status": "insufficient_signal",
            "periodicity": None,
            "period_seconds": None,
            "dynamic_range": None,
            "noise_sigma": None,
            "dynamic_to_noise": None,
            "valid_fraction": gap_evidence["original_valid_fraction"],
            "max_gap_seconds": gap_evidence["max_gap_seconds"],
            "multiscale_agreement": 0.0,
            "selected_scale_seconds": None,
            "scale_count": 0,
            "bilateral": bilateral,
            "family": signal_family(name),
            "gap_evidence": gap_evidence,
            "detector": None,
            "event_evidence": None,
            "reasons": list(gap_evidence["reasons"])
            or ["no_valid_periodic_scale"],
        }
    minimum_periodicity = float(settings["minimum_periodicity"])
    periodicity_eligible = [
        item
        for item in scales
        if float(item["prepared"]["periodicity"]) >= minimum_periodicity
    ]
    best = max(
        periodicity_eligible or scales,
        key=lambda item: item["adjusted_score"],
    )
    selection_tolerance = float(settings["scale_selection_tolerance"])
    for candidate in sorted(scales, key=lambda item: item["smoothing_seconds"]):
        period = float(candidate["prepared"]["period_frames"])
        best_period = float(best["prepared"]["period_frames"])
        period_difference = abs(period - best_period) / max(best_period, 1e-6)
        if (
            period_difference
            <= float(settings["period_agreement_fraction"])
            and candidate["adjusted_score"]
            >= best["adjusted_score"] * (1.0 - selection_tolerance)
        ):
            best = candidate
            break
    periods = [
        float(item["prepared"]["period_frames"]) / max(fps, 1e-6)
        for item in scales
    ]
    median_period = float(np.median(periods))
    tolerance = float(settings["period_agreement_fraction"])
    agreement = float(
        np.mean(
            [
                abs(period - median_period) / max(median_period, 1e-6)
                <= tolerance
                for period in periods
            ]
        )
    )
    score = float(
        best["base_score"] * (0.80 + 0.20 * agreement)
    )
    prepared = best["prepared"]
    reasons = []
    if float(prepared["periodicity"]) < float(settings["minimum_periodicity"]):
        reasons.append("periodicity_below_minimum")
    if score < float(settings["minimum_candidate_score"]):
        reasons.append("candidate_score_below_minimum")
    if agreement < float(settings["minimum_multiscale_agreement"]):
        reasons.append("multiscale_agreement_below_minimum")
    status = "ok" if not reasons else "insufficient_signal"
    if status == "ok" and gap_evidence["status"] == "degraded":
        status = "degraded"
    detection_values = filled if gap_evidence["status"] == "ok" else values
    event_evidence = _candidate_event_evidence(
        detection_values,
        name=name,
        fps=fps,
        action=action,
        detector=best["detector"],
    )
    return {
        "phase_signal": name,
        "score": round(score, 6),
        "status": status,
        "periodicity": round(float(prepared["periodicity"]), 6),
        "period_seconds": round(
            float(prepared["period_frames"]) / max(fps, 1e-6),
            6,
        ),
        "dynamic_range": round(float(prepared["dynamic_range"]), 6),
        "noise_sigma": round(float(prepared["noise_sigma"]), 6),
        "dynamic_to_noise": (
            round(float(prepared["dynamic_to_noise"]), 6)
            if prepared.get("dynamic_to_noise") is not None
            else None
        ),
        "valid_fraction": round(
            float(gap_evidence["original_valid_fraction"]), 6
        ),
        "max_gap_seconds": round(float(gap_evidence["max_gap_seconds"]), 6),
        "multiscale_agreement": round(agreement, 6),
        "selected_scale_seconds": round(
            float(best["smoothing_seconds"]), 6
        ),
        "scale_count": len(scales),
        "bilateral": bilateral,
        "family": signal_family(name),
        "gap_evidence": gap_evidence,
        "detector": best["detector"],
        "event_evidence": event_evidence,
        "reasons": reasons,
    }


def _event_consensus_evidence(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    selected_evidence = selected.get("event_evidence") or {}
    anchors = [
        int(frame) for frame in selected_evidence.get("top_anchors") or []
    ]
    kinds = list(selected_evidence.get("top_kinds") or [])
    required_indices = [
        index
        for index, kind in enumerate(kinds)
        if kind == "interior" and index < len(anchors)
    ]
    required_frames = [anchors[index] for index in required_indices]
    supporting = {frame: [] for frame in required_frames}
    selected_is_bilateral = bool(selected.get("bilateral"))
    peer_evidences = [
        candidate
        for candidate in candidates
        if candidate is not selected
        and bool(candidate.get("bilateral"))
        and candidate.get("family") == selected.get("family")
        and (candidate.get("event_evidence") or {}).get("top_anchors")
    ]
    if selected_is_bilateral:
        return {
            "status": "aggregate_signal",
            "selected_phase_signal": selected["phase_signal"],
            "required_event_frames": required_frames,
            "corroborated_event_frames": required_frames,
            "uncorroborated_event_frames": [],
            "supporting_phase_signals": {},
            "tolerance_frames": None,
        }
    if not peer_evidences:
        return {
            "status": "not_applicable" if required_frames else "missing",
            "selected_phase_signal": selected["phase_signal"],
            "required_event_frames": required_frames,
            "corroborated_event_frames": required_frames,
            "uncorroborated_event_frames": [],
            "supporting_phase_signals": {},
            "tolerance_frames": None,
        }
    if not required_frames:
        return {
            "status": "missing",
            "selected_phase_signal": selected["phase_signal"],
            "required_event_frames": [],
            "corroborated_event_frames": [],
            "uncorroborated_event_frames": [],
            "supporting_phase_signals": {},
            "tolerance_frames": None,
        }
    selected_period = float(selected_evidence.get("period_frames") or 0.0)
    peer_periods = [
        float(
            (candidate.get("event_evidence") or {}).get("period_frames")
            or 0.0
        )
        for candidate in peer_evidences
    ]
    periods = [period for period in [selected_period, *peer_periods] if period > 0]
    tolerance = (
        float(settings["period_agreement_fraction"])
        * min(periods)
        if periods
        else 0.0
    )
    for candidate in peer_evidences:
        peer_frames = [
            int(frame)
            for frame in (
                (candidate.get("event_evidence") or {}).get("top_anchors")
                or []
            )
        ]
        for frame in required_frames:
            if any(
                abs(frame - peer_frame) <= tolerance
                for peer_frame in peer_frames
            ):
                supporting[frame].append(candidate["phase_signal"])
    corroborated = [
        frame for frame in required_frames if supporting[frame]
    ]
    uncorroborated = [
        frame for frame in required_frames if not supporting[frame]
    ]
    event_consensus_ok = len(corroborated) * 2 > len(required_frames)
    return {
        "status": "ok" if event_consensus_ok else "disagreement",
        "selected_phase_signal": selected["phase_signal"],
        "required_event_frames": required_frames,
        "corroborated_event_frames": corroborated,
        "uncorroborated_event_frames": uncorroborated,
        "supporting_phase_signals": {
            str(frame): names for frame, names in supporting.items()
        },
        "tolerance_frames": round(float(tolerance), 6),
    }


def _serializable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key != "detector"
    }


def analyze_phase_signals(
    signals: dict[str, np.ndarray],
    fps: float,
    action: dict[str, Any],
    config: dict[str, Any],
    *,
    timestamps: Iterable[float] | None = None,
    prior_signal: str | None = None,
    detector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_count = (
        int(next(iter(signals.values())).shape[0]) if signals else 0
    )
    timebase = build_timebase(
        frame_count,
        fps,
        config,
        timestamps=timestamps,
    )
    resampled = resample_signals(signals, timebase, config)
    effective_detector = _motion_gate_detector(detector, config)
    expanded = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in resampled.items()
    }
    bilateral_names = phase_states.add_bilateral_signal_candidates(
        expanded, action
    )
    candidate_names = list(dict.fromkeys([
        *bilateral_names,
        *[
            name
            for name in action.get("phase_signal_candidates") or []
            if name in expanded
        ],
    ]))
    candidates = []
    for name in candidate_names:
        candidates.append(
            evaluate_signal_candidate(
                expanded[name],
                name=name,
                fps=float(timebase["analysis_fps"]),
                action=action,
                config=config,
                detector=effective_detector,
                bilateral=name in bilateral_names,
            )
        )
    settings = config["core"]
    eligible = [
        candidate
        for candidate in candidates
        if candidate["status"] in {"ok", "degraded"}
    ]
    preferred = [candidate for candidate in eligible if candidate["bilateral"]]
    pool = preferred if preferred else eligible
    pool.sort(
        key=lambda item: (
            item["score"]
            * _prior_signal_factor(item, prior_signal, settings),
            item["score"],
            item["phase_signal"],
        ),
        reverse=True,
    )
    top_k = int(settings["top_k_hypotheses"])
    hypotheses = pool[:top_k]
    selected = hypotheses[0] if hypotheses else None
    status = "insufficient_signal"
    consensus_score = 0.0
    ambiguity_reason = None
    if selected is not None:
        status = "ok"
        consensus_score = selected["score"] * 0.85
        if len(hypotheses) > 1:
            second = hypotheses[1]
            period_denominator = max(
                float(selected["period_seconds"] or 0.0),
                1e-6,
            )
            period_difference = abs(
                float(selected["period_seconds"] or 0.0)
                - float(second["period_seconds"] or 0.0)
            ) / period_denominator
            score_ratio = min(
                1.0,
                float(second["score"])
                / max(float(selected["score"]), 1e-6),
            )
            period_agreement = max(
                0.0,
                1.0
                - period_difference
                / max(float(settings["period_agreement_fraction"]), 1e-6),
            )
            consensus_score = selected["score"] * (
                0.75 + 0.15 * period_agreement + 0.10 * (1.0 - score_ratio)
            )
            if (
                selected["family"] != second["family"]
                and period_difference
                > float(settings["period_agreement_fraction"])
                and score_ratio
                >= float(settings["ambiguity_score_ratio"])
                and not (
                    prior_signal
                    and selected["family"] == signal_family(prior_signal)
                )
            ):
                status = "ambiguous"
                ambiguity_reason = "phase_signal_family_disagreement"
        if consensus_score < float(settings["minimum_consensus_score"]):
            status = "ambiguous"
            ambiguity_reason = "consensus_score_below_minimum"
    else:
        consensus_score = 0.0
    event_consensus = (
        _event_consensus_evidence(selected, candidates, settings)
        if selected is not None
        else {
            "status": "not_applicable",
            "selected_phase_signal": None,
            "required_event_frames": [],
            "corroborated_event_frames": [],
            "uncorroborated_event_frames": [],
            "supporting_phase_signals": {},
            "tolerance_frames": None,
        }
    )
    if status == "ok" and event_consensus["status"] == "disagreement":
        status = "ambiguous"
        ambiguity_reason = "phase_signal_event_disagreement"
    elif status == "ok" and event_consensus["status"] == "missing":
        status = "ambiguous"
        ambiguity_reason = "phase_signal_event_evidence_missing"
    evidence = {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "status": status,
        "selected_phase_signal": (
            selected["phase_signal"] if selected is not None else None
        ),
        "consensus_score": round(float(consensus_score), 6),
        "ambiguous": status == "ambiguous",
        "ambiguity_reason": ambiguity_reason,
        "event_consensus": event_consensus,
        "candidates": [
            _serializable_candidate(candidate) for candidate in candidates
        ],
        "phase_hypotheses": [
            {
                "hypothesis_id": f"phase_{index}",
                "phase_signal": candidate["phase_signal"],
                "score": candidate["score"],
                "period_seconds": candidate["period_seconds"],
                "status": (
                    "ambiguous" if status == "ambiguous" else "ok"
                ),
                "evidence": {
                    "family": candidate["family"],
                    "multiscale_agreement": candidate[
                        "multiscale_agreement"
                    ],
                    "selected_scale_seconds": candidate[
                        "selected_scale_seconds"
                    ],
                },
            }
            for index, candidate in enumerate(hypotheses, start=1)
        ],
        "prior_signal": prior_signal,
        "timebase": {
            key: value
            for key, value in timebase.items()
            if key not in {"source_times", "target_times"}
        },
    }
    return {
        **evidence,
        "detector": selected["detector"] if selected is not None else detector,
        "analysis_fps": float(timebase["analysis_fps"]),
        "signals": expanded,
        "evidence": evidence,
    }


def load_fit3d_clips(
    sources: dict[str, Any],
    split_role: str,
) -> list[dict[str, Any]]:
    clips = [
        clip
        for clip in sources.get("clips") or []
        if clip.get("dataset") == "Fit3D"
        and clip.get("split_role") == split_role
        and clip.get("has_gt")
        and not clip.get("smoke_only")
    ]
    if len(clips) != 32:
        raise ValueError(f"expected 32 Fit3D {split_role} clips, found {len(clips)}")
    return clips


def phase_ambiguous_analysis(
    clip_id: str,
    resolution: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    analysis = {
        "schema_version": chain_selection.ANALYSIS_SCHEMA_VERSION,
        "clip_id": clip_id,
        "algorithm": {"name": "repchain", "version": "v2"},
        "anchor_policy": "plateau_midpoint",
        "status": "phase_ambiguous",
        "motion_status": "unknown",
        "sets": [],
        "rejected_candidates": [],
        "non_rep_intervals": [],
        "diagnostics": {
            "chain_source": "phase_resolution_ambiguous",
            "selected_phase_signal": resolution.get("selected_phase_signal"),
            "phase_signal_evidence": resolution.get("evidence") or resolution,
        },
    }
    return quality_gates.apply_quality_gates(analysis, [gate])


def reference_analyses(
    clips: list[dict[str, Any]],
    *,
    analysis_dir: Path,
    split: str,
    specs: dict[str, Any],
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
    frozen_baseline_dir: Path,
) -> dict[str, dict[str, Any]]:
    analyses = {}
    for clip in clips:
        if split == "holdout":
            frozen_path = (
                frozen_baseline_dir / clip["clip_id"] / "analysis_v2.json"
            )
            if frozen_path.is_file():
                analyses[clip["clip_id"]] = load_json(frozen_path)
                continue
        clip_dir = analysis_dir / clip["clip_id"]
        legacy = load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        action = phase_states.action_config(clip["action"], specs)
        gate = quality_gates.from_legacy_quality(legacy)
        analysis, _ = chain_selection.analyze_clip_signals(
            clip,
            signals,
            action,
            prior_signal=(legacy.get("segmentation") or {}).get("phase_angle"),
            phase_policy=segmenter_config["core"]["phase_signal"]["policy"],
            chain_config=chain_config,
            quality_gate=gate,
            detector=segmenter_config["core"].get("detector"),
        )
        analyses[clip["clip_id"]] = analysis
    return analyses


def _analysis_structure(analysis: dict[str, Any]) -> tuple[int, int, int]:
    sets = analysis.get("sets") or []
    return (
        len(sets),
        sum(len(item.get("reps") or []) for item in sets),
        sum(
            len(item.get("reps") or [])
            for item in sets
            if item.get("is_default")
        ),
    )


def _selected_top_event_frames(analysis: dict[str, Any]) -> list[int]:
    return [
        int(event.get("graph_frame", event["frame"]))
        for event in (
            (analysis.get("diagnostics") or {}).get("selected_top_events") or []
        )
    ]


def _event_frames_match(
    frames: Iterable[int],
    references: Iterable[int],
    tolerance: float,
) -> list[int]:
    reference_frames = [int(frame) for frame in references]
    return [
        int(frame)
        for frame in frames
        if any(abs(int(frame) - other) <= tolerance for other in reference_frames)
    ]


def _event_frames_preserved(
    source_frames: Iterable[int],
    target_frames: Iterable[int],
    tolerance: float,
) -> bool:
    source = [int(frame) for frame in source_frames]
    target = [int(frame) for frame in target_frames]
    return bool(source) and len(_event_frames_match(source, target, tolerance)) == len(
        source
    )


def _event_span_preserved(
    source_frames: Iterable[int],
    target_frames: Iterable[int],
    tolerance: float,
) -> bool:
    source = [int(frame) for frame in source_frames]
    target = [int(frame) for frame in target_frames]
    if not source or not target:
        return False
    target_start = min(target) - tolerance
    target_end = max(target) + tolerance
    return bool(_event_frames_match(source, target, tolerance)) and all(
        target_start <= frame <= target_end for frame in source
    )


def _explicit_endpoint_pruned_frames(
    analysis: dict[str, Any],
) -> list[int]:
    diagnostics = analysis.get("diagnostics") or {}
    records = [
        *(
            diagnostics.get("low_sampling_endpoint_pruned_events")
            or []
        ),
        *(diagnostics.get("endpoint_timing_pruned_events") or []),
    ]
    return sorted(
        {
            int(record["frame"])
            for record in records
            if record.get("frame") is not None
        }
    )


def select_analysis_hypothesis(
    single_scale: dict[str, Any],
    consensus: dict[str, Any] | None,
    consensus_state: dict[str, Any] | None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select consensus only when it materially improves a usable chain."""

    if consensus is None or consensus_state is None:
        return single_scale
    if consensus_state.get("status") != "ok":
        return single_scale
    single_sets, single_reps, _ = _analysis_structure(single_scale)
    consensus_sets, consensus_reps, _ = _analysis_structure(consensus)
    if consensus_sets == 0:
        return single_scale
    if single_sets == 0:
        return consensus
    if consensus_sets < single_sets:
        return single_scale
    period_frames = float(consensus_state.get("period_frames") or 0.0)
    tolerance = (
        float(
            (settings or {}).get(
                "arbitration_match_tolerance_period_fraction", 0.45
            )
        )
        * period_frames
    )
    single_frames = _selected_top_event_frames(single_scale)
    consensus_frames = _selected_top_event_frames(consensus)
    events_preserved = _event_frames_preserved(
        single_frames, consensus_frames, tolerance
    )
    event_span_preserved = _event_span_preserved(
        single_frames, consensus_frames, tolerance
    )
    if consensus_reps >= single_reps + 2:
        if events_preserved or event_span_preserved:
            return consensus
        return single_scale
    if consensus_reps == single_reps + 1:
        consensus_events = consensus_state.get("top_states") or []
        extra_states = [
            state
            for state in consensus_events
            if not any(
                abs(int(state["anchor_frame"]) - frame) <= tolerance
                for frame in single_frames
            )
        ]
        regular_endpoint_addition = bool(extra_states) and all(
            (
                state.get("kind") == "leading_partial"
                or bool((state.get("evidence") or {}).get("half_cycle_recovered"))
            )
            and int((state.get("evidence") or {}).get("scale_support") or 0)
            < 2
            and not bool(
                (state.get("evidence") or {}).get("multi_scale_consensus")
            )
            for state in extra_states
        )
        consensus_selected_frames = [
            int(event.get("graph_frame", event["frame"]))
            for event in (
                (consensus.get("diagnostics") or {}).get(
                    "selected_top_events"
                )
                or []
            )
        ]
        selected_extra_states = [
            state
            for state in extra_states
            if any(
                abs(int(state["anchor_frame"]) - frame) <= tolerance
                for frame in consensus_selected_frames
            )
        ]
        local_extrema_addition = bool(selected_extra_states) and all(
            bool(
                (state.get("evidence") or {}).get(
                    "local_extrema_stable"
                )
            )
            for state in selected_extra_states
        )
        near_boundary_addition = bool(selected_extra_states) and all(
            bool(
                (state.get("evidence") or {}).get(
                    "near_boundary_rejected_top"
                )
            )
            and int((state.get("evidence") or {}).get("scale_support") or 0)
            >= 3
            for state in selected_extra_states
        )
        endpoint_addition = (
            regular_endpoint_addition or local_extrema_addition
        )
        single_confidence = max(
            float(item.get("chain_confidence") or 0.0)
            for item in single_scale.get("sets") or []
        )
        consensus_confidence = max(
            float(item.get("chain_confidence") or 0.0)
            for item in consensus.get("sets") or []
        )
        maximum_confidence_drop = float(
            (settings or {}).get(
                "arbitration_single_rep_max_confidence_drop", 0.10
            )
        )
        closure_evidence = bool(
            (consensus.get("diagnostics") or {}).get(
                "endpoint_closure_edges"
            )
        )
        if (
            events_preserved
            and endpoint_addition
            and closure_evidence
            and consensus_confidence
            >= single_confidence - maximum_confidence_drop
        ):
            return consensus
        if (
            events_preserved
            and near_boundary_addition
            and consensus_confidence
            >= single_confidence - maximum_confidence_drop
        ):
            return consensus
    if consensus_reps < single_reps:
        single_confidence = max(
            float(item.get("chain_confidence") or 0.0)
            for item in single_scale.get("sets") or []
        )
        consensus_confidence = max(
            float(item.get("chain_confidence") or 0.0)
            for item in consensus.get("sets") or []
        )
        maximum_confidence_drop = float(
            (settings or {}).get(
                "arbitration_single_rep_max_confidence_drop", 0.10
            )
        )
        evidence = consensus_state.get("evidence") or {}
        cleanup_evidence = bool(
            int(evidence.get("near_duplicate_top_count") or 0) > 0
            or (
                (consensus.get("diagnostics") or {}).get(
                    "endpoint_closure_edges"
                )
                or []
            )
        )
        pruned_events = (
            (consensus.get("diagnostics") or {}).get(
                "low_sampling_endpoint_pruned_events"
            )
            or []
        )
        pruned_frames = [int(item["frame"]) for item in pruned_events]
        consensus_selected_frames = [
            int(event.get("graph_frame", event["frame"]))
            for event in (
                (consensus.get("diagnostics") or {}).get(
                    "selected_top_events"
                )
                or []
            )
        ]
        retained_single_frames = [
            frame
            for frame in single_frames
            if not any(abs(frame - pruned) <= tolerance for pruned in pruned_frames)
        ]
        endpoint_only_cleanup = bool(pruned_events) and (
            len(retained_single_frames) == len(single_frames) - len(pruned_events)
        ) and all(
            any(abs(frame - other) <= tolerance for other in consensus_selected_frames)
            for frame in retained_single_frames
        ) and all(
            any(abs(frame - other) <= tolerance for other in single_frames)
            for frame in consensus_selected_frames
        )
        consensus_span = (
            (min(consensus_selected_frames), max(consensus_selected_frames))
            if consensus_selected_frames
            else None
        )
        unmatched_single_frames = [
            frame
            for frame in single_frames
            if not any(
                abs(frame - other) <= tolerance
                for other in consensus_selected_frames
            )
        ]
        explicit_pruned_frames = _explicit_endpoint_pruned_frames(consensus)
        unexplained_single_frames = [
            frame
            for frame in unmatched_single_frames
            if not any(
                abs(frame - pruned) <= tolerance
                for pruned in explicit_pruned_frames
            )
        ]
        endpoint_span_cleanup = bool(
            unmatched_single_frames
            and not unexplained_single_frames
            and consensus_span is not None
            and all(
                frame < consensus_span[0] - tolerance
                or frame > consensus_span[1] + tolerance
                for frame in unmatched_single_frames
            )
            and all(
                any(abs(frame - other) <= tolerance for other in single_frames)
                for frame in consensus_selected_frames
            )
        )
        if (
            (
                cleanup_evidence
                and events_preserved
                and consensus_confidence >= single_confidence + 0.05
            )
            or (
                endpoint_only_cleanup
                and consensus_confidence
                >= single_confidence - maximum_confidence_drop
            )
            or (
                endpoint_span_cleanup
                and consensus_confidence
                >= single_confidence - maximum_confidence_drop
            )
        ):
            return consensus
    return single_scale


def evaluate_signals(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    sources = load_json(args.sources)
    specs = load_json(args.specs)
    segmenter_config = load_json(args.segmenter_config)
    chain_config = chain_selection.load_chain_config(args.chain_config)
    clips = load_fit3d_clips(
        sources,
        "holdout" if args.split == "holdout" else "dev",
    )
    reference = reference_analyses(
        clips,
        analysis_dir=args.analysis_dir,
        split=args.split,
        specs=specs,
        segmenter_config=segmenter_config,
        chain_config=chain_config,
        frozen_baseline_dir=args.frozen_baseline_dir,
    )
    baseline_summary = (
        load_json(args.baseline_summary)
        if args.split == "holdout" and args.baseline_summary.is_file()
        else {}
    )
    baseline_robustness = {
        row["transform_id"]: row
        for row in baseline_summary.get("robustness") or []
    }
    rows = []
    signal_rows = []
    transforms = ["baseline", *baseline_evaluation.robustness_variants()]
    for clip in clips:
        clip_dir = args.analysis_dir / clip["clip_id"]
        legacy = load_json(clip_dir / "analysis.json")
        gate = quality_gates.from_legacy_quality(legacy)
        action = phase_states.action_config(clip["action"], specs)
        original_signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        base_sets, base_reps, base_default_reps = baseline_evaluation.structure_counts(
            reference[clip["clip_id"]]
        )
        for transform_id in transforms:
            if transform_id == "baseline":
                transformed = original_signals
                transformed_fps = float(clip["fps"])
            else:
                transformed, transformed_fps = baseline_evaluation.transform_signals(
                    original_signals,
                    transform_id,
                    fps=float(clip["fps"]),
                    seed=zlib.crc32(
                        f"{clip['clip_id']}:{transform_id}".encode("utf-8")
                    ),
                )
            resolution = analyze_phase_signals(
                transformed,
                transformed_fps,
                action,
                config,
                prior_signal=(legacy.get("segmentation") or {}).get(
                    "phase_angle"
                ),
                detector=segmenter_config["core"].get("detector"),
            )
            consensus_state = None
            if (
                resolution["status"] != "ambiguous"
                and resolution.get("selected_phase_signal")
            ):
                selected_signal = resolution["selected_phase_signal"]
                consensus_state = build_multiscale_state_consensus(
                    resolution["signals"][selected_signal],
                    float(resolution["analysis_fps"]),
                    action,
                    selected_signal,
                    config,
                    detector=resolution.get("detector"),
                )
            transformed_clip = {
                **clip,
                "fps": float(resolution["analysis_fps"]),
            }
            if resolution["status"] == "ambiguous":
                analysis = phase_ambiguous_analysis(
                    clip["clip_id"], resolution, gate
                )
                meta = {"selected_top_events": []}
                analysis_hypothesis = "phase_ambiguous"
            else:
                single_analysis, single_meta = (
                    chain_selection.analyze_clip_signals(
                        transformed_clip,
                        resolution["signals"],
                        action,
                        prior_signal=(legacy.get("segmentation") or {}).get(
                            "phase_angle"
                        ),
                        phase_policy=segmenter_config["core"]["phase_signal"][
                            "policy"
                        ],
                        chain_config=chain_config,
                        quality_gate=gate,
                        detector=segmenter_config["core"].get("detector"),
                        phase_resolution=resolution,
                        chain_policy=(
                            {
                                "endpoint_timing_model": consensus_state[
                                    "chain_policy"
                                ]["endpoint_timing_model"]
                            }
                            if consensus_state is not None
                            else None
                        ),
                    )
                )
                consensus_analysis = None
                if consensus_state is not None:
                    consensus_analysis, _ = (
                        chain_selection.analyze_clip_signals(
                            transformed_clip,
                            resolution["signals"],
                            action,
                            prior_signal=(
                                legacy.get("segmentation") or {}
                            ).get("phase_angle"),
                            phase_policy=segmenter_config["core"][
                                "phase_signal"
                            ]["policy"],
                            chain_config=chain_config,
                            quality_gate=gate,
                            detector=segmenter_config["core"].get(
                                "detector"
                            ),
                            phase_resolution=resolution,
                            phase_state_override=consensus_state,
                        )
                    )
                analysis = select_analysis_hypothesis(
                    single_analysis,
                    consensus_analysis,
                    consensus_state,
                    config["core"]["event_consensus"],
                )
                analysis_hypothesis = (
                    "multiscale_consensus"
                    if consensus_analysis is not None
                    and analysis is consensus_analysis
                    else "single_scale"
                )
                meta = (
                    single_meta
                    if analysis is single_analysis
                    else {
                        "selected_top_events": (
                            analysis.get("diagnostics") or {}
                        ).get("selected_top_events", []),
                        "phase_state_source": "multiscale_consensus",
                    }
                )
                analysis["diagnostics"]["analysis_hypothesis"] = (
                    analysis_hypothesis
                )
                analysis["diagnostics"][
                    "single_scale_structure"
                ] = _analysis_structure(single_analysis)
                if consensus_analysis is not None:
                    analysis["diagnostics"][
                        "consensus_structure"
                    ] = _analysis_structure(consensus_analysis)
            sets, reps, default_reps = baseline_evaluation.structure_counts(
                analysis
            )
            rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "cluster_id": clip["cluster_id"],
                    "action": clip["action"],
                    "transform_id": transform_id,
                    "status": analysis.get("status"),
                    "quality_excluded": bool(gate["blocking"]),
                    "baseline_set_count": base_sets,
                    "transformed_set_count": sets,
                    "baseline_rep_count": base_reps,
                    "transformed_rep_count": reps,
                    "baseline_default_rep_count": base_default_reps,
                    "transformed_default_rep_count": default_reps,
                    "rep_count_delta": reps - base_reps,
                    "set_count_delta": sets - base_sets,
                    "structure_match": (
                        sets == base_sets
                        and reps == base_reps
                        and default_reps == base_default_reps
                    ),
                    "phase_resolution_status": resolution["status"],
                    "selected_phase_signal": resolution.get(
                        "selected_phase_signal"
                    ),
                    "consensus_score": resolution.get("consensus_score"),
                    "phase_state_source": analysis_hypothesis,
                    "event_consensus_status": (
                        consensus_state.get("status")
                        if consensus_state is not None
                        else None
                    ),
                    "consensus_top_count": (
                        len(consensus_state.get("top_states") or [])
                        if consensus_state is not None
                        else None
                    ),
                    "half_cycle_recovered_top_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "recovered_top_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "half_cycle_recovered_bottom_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "recovered_bottom_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "near_boundary_recovered_top_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "near_boundary_recovered_top_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "motion_joint_override_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "motion_joint_override_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "low_sampling_endpoint_pruned_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "low_sampling_endpoint_pruned_events"
                        )
                        or []
                    ),
                    "endpoint_timing_pruned_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "endpoint_timing_pruned_events"
                        )
                        or []
                    ),
                    "endpoint_timing_decision_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "endpoint_timing_decisions"
                        )
                        or []
                    ),
                    "low_amplitude_context_bridged_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "low_amplitude_context_bridged_events"
                        )
                        or []
                    ),
                    "local_extrema_stability_bridged_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "local_extrema_stability_bridged_events"
                        )
                        or []
                    ),
                    "local_period_gap_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "local_period_gap_edges"
                        )
                        or []
                    ),
                    "endpoint_closure_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "endpoint_closure_edges"
                        )
                        or []
                    ),
                }
            )
            signal_rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "action": clip["action"],
                    "transform_id": transform_id,
                    "analysis_fps": resolution["analysis_fps"],
                    "variable_fps": resolution["timebase"]["variable_fps"],
                    "resampled": resolution["timebase"]["resampled"],
                    "selected_phase_signal": resolution.get(
                        "selected_phase_signal"
                    ),
                    "phase_resolution_status": resolution["status"],
                    "consensus_score": resolution.get("consensus_score"),
                    "ambiguity_reason": resolution.get("ambiguity_reason"),
                    "candidate_count": len(resolution.get("candidates") or []),
                    "selected_scale_seconds": (
                        (resolution.get("detector") or {}).get(
                            "smoothing_seconds"
                        )
                    ),
                    "phase_state_source": analysis_hypothesis,
                    "event_consensus_status": (
                        consensus_state.get("status")
                        if consensus_state is not None
                        else None
                    ),
                    "event_period_frames": (
                        consensus_state.get("period_frames")
                        if consensus_state is not None
                        else None
                    ),
                    "event_scale_support": (
                        len(
                            (consensus_state.get("evidence") or {}).get(
                                "scale_members"
                            )
                            or []
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "consensus_top_count": (
                        len(consensus_state.get("top_states") or [])
                        if consensus_state is not None
                        else None
                    ),
                    "half_cycle_recovered_top_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "recovered_top_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "half_cycle_recovered_bottom_count": (
                        int(
                            (consensus_state.get("evidence") or {}).get(
                                "recovered_bottom_count"
                            )
                            or 0
                        )
                        if consensus_state is not None
                        else None
                    ),
                    "low_sampling_endpoint_pruned_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "low_sampling_endpoint_pruned_events"
                        )
                        or []
                    ),
                    "endpoint_timing_pruned_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "endpoint_timing_pruned_events"
                        )
                        or []
                    ),
                    "endpoint_timing_decision_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "endpoint_timing_decisions"
                        )
                        or []
                    ),
                    "low_amplitude_context_bridged_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "low_amplitude_context_bridged_events"
                        )
                        or []
                    ),
                    "local_extrema_stability_bridged_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "local_extrema_stability_bridged_events"
                        )
                        or []
                    ),
                    "local_period_gap_count": len(
                        (analysis.get("diagnostics") or {}).get(
                            "local_period_gap_edges"
                        )
                        or []
                    ),
                }
            )
    summary = summarize_signal_evaluation(
        rows,
        baseline_robustness,
        split=args.split,
    )
    summary.update(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "task": "T1-7B",
            "split": args.split,
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_path(args.config),
            "clip_count": len(clips),
        }
    )
    write_csv(args.output_dir / "robustness.csv", rows)
    write_csv(args.output_dir / "signal_evidence.csv", signal_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_checksums(
        args.output_dir,
        ["robustness.csv", "signal_evidence.csv", "summary.json"],
    )
    return summary


def summarize_signal_evaluation(
    rows: list[dict[str, Any]],
    baseline_robustness: dict[str, dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["transform_id"]].append(row)
    summary_rows = []
    for transform_id, transform_rows in sorted(grouped.items()):
        eligible = [
            row for row in transform_rows if not row.get("quality_excluded")
        ]
        match_count = sum(bool(row["structure_match"]) for row in eligible)
        baseline_match_count = baseline_robustness.get(transform_id, {}).get(
            "structure_match_count",
            31 if transform_id == "baseline" else None,
        )
        summary_rows.append(
            {
                "transform_id": transform_id,
                "input_clip_count": len(transform_rows),
                "clip_count": len(eligible),
                "excluded_by_quality_clip_count": len(transform_rows)
                - len(eligible),
                "structure_match_count": match_count,
                "structure_match_rate": round(
                    match_count / len(eligible), 6
                )
                if eligible
                else None,
                "changed_structure_count": len(eligible) - match_count,
                "mean_rep_count_delta": round(
                    float(
                        np.mean(
                            [int(row["rep_count_delta"]) for row in eligible]
                        )
                    ),
                    6,
                )
                if eligible
                else None,
                "phase_ambiguous_count": sum(
                    row["phase_resolution_status"] == "ambiguous"
                    for row in eligible
                ),
                "t1_6g_structure_match_count": baseline_match_count,
                "no_regression_vs_t1_6g": (
                    baseline_match_count is None
                    or match_count >= int(baseline_match_count)
                ),
            }
        )
    by_transform = {
        row["transform_id"]: row for row in summary_rows
    }
    expected_quality_excluded = sum(
        bool(row.get("quality_excluded"))
        for row in grouped.get("baseline", [])
    )

    def match_count(transform_id: str) -> int:
        return int(by_transform[transform_id]["structure_match_count"])

    gates = []
    if "baseline" in by_transform:
        baseline_eligible = int(by_transform["baseline"]["clip_count"])
        gates.append(
            {
                "gate_id": "baseline_structure_match",
                "observed": (
                    f"{match_count('baseline')}/{baseline_eligible}"
                ),
                "target": (
                    f"{baseline_eligible}/{baseline_eligible}"
                ),
                "passed": (
                    match_count("baseline") == baseline_eligible
                ),
            }
        )
    for transform_id in ("fps_15", "fps_60", "noise_sigma_2"):
        if transform_id in by_transform:
            transform_eligible = int(by_transform[transform_id]["clip_count"])
            gates.append(
                {
                    "gate_id": f"{transform_id}_structure_match_at_least_30",
                    "observed": (
                        f"{match_count(transform_id)}/{transform_eligible}"
                    ),
                    "target": ">=30/31",
                    "passed": match_count(transform_id) >= 30,
                }
            )
    regression_rows = [
        row
        for row in summary_rows
        if row["transform_id"] != "baseline"
        and not row["no_regression_vs_t1_6g"]
    ]
    if baseline_robustness:
        gates.append(
            {
                "gate_id": "no_regression_vs_t1_6g",
                "observed": len(regression_rows),
                "target": "0 regressed transforms",
                "passed": not regression_rows,
            }
        )
    quality_counts = {
        row["transform_id"]: row["excluded_by_quality_clip_count"]
        for row in summary_rows
    }
    gates.append(
        {
            "gate_id": "quality_exclusion_count_unchanged",
            "observed": quality_counts,
            "target": (
                f"same {expected_quality_excluded} quality-excluded clips"
            ),
            "passed": all(
                value == expected_quality_excluded
                for value in quality_counts.values()
            ),
        }
    )
    return {
        "stage": "T1-7B",
        "stage_passed": all(gate["passed"] for gate in gates),
        "decision": (
            "SIGNAL_STAGE_PASS"
            if all(gate["passed"] for gate in gates)
            else "SIGNAL_STAGE_NO_GO"
        ),
        "split_role": split,
        "robustness": summary_rows,
        "gates": gates,
        "regressed_transforms": [
            row["transform_id"] for row in regression_rows
        ],
    }


def main() -> int:
    args = parse_args()
    if not args.evaluate:
        raise SystemExit("T1-7B requires --evaluate")
    summary = evaluate_signals(args)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "split": summary["split"],
                "decision": summary["decision"],
                "stage_passed": summary["stage_passed"],
                "gates": summary["gates"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
