#!/usr/bin/env python3
"""Calibrate T1-6 rep confidence and audit explicit rejection reasons."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_chain_selection as chain_selection
import task1_6_phase_states as phase_states


DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_CHAIN_CONFIG = ROOT / "configs" / "task1_6_chain.json"
DEFAULT_SEGMENTER_CONFIG = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_FIT3D_ANALYSIS_DIR = ROOT / "results" / "task1_6" / "dev" / "clip_analysis_v2"
DEFAULT_QEVD_ANALYSIS_DIR = (
    ROOT / "results" / "task1_6" / "dev" / "qevd_clip_analysis_v2"
)
DEFAULT_RESULTS_DIR = ROOT / "results" / "task1_6" / "dev" / "confidence"
DEFAULT_FROZEN_CONFIG = ROOT / "configs" / "task1_6_confidence_frozen.json"

CONFIG_SCHEMA_VERSION = "task1-6-confidence-config-v1"
SUMMARY_SCHEMA_VERSION = "task1-6d-confidence-eval-v1"
ANALYSIS_SCHEMA_VERSION = "task1-6-clip-analysis-v2"
FIT3D_DATASET = "Fit3D"
QEVD_DATASET = "QEVD-FIT-COACH"
EXPECTED_FIT3D_DEV_CLIPS = 32
EXPECTED_QEVD_DEV_CLIPS = 13
HIGH_CONFIDENCE_THRESHOLD = 0.5
CALIBRATION_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
LOWER_BOUND_ALPHA = 0.1
ECE_BIN_COUNT = 10
ECE_RELATIVE_IMPROVEMENT_MIN = 0.5
QEVD_MINIMUM_IOU = 0.30
CONFIDENCE_LAYERS = ("complete", "in_set", "combined")
FORBIDDEN_CORE_TOKENS = (
    "fit3d",
    "qevd",
    "subject_id",
    "camera_id",
    "video_id",
    "clip_id",
    "s03",
    "s04",
    "s05",
    "s07",
    "50591643",
    "58860488",
    "60457274",
    "65906101",
    "fixed_rep_count",
    "fixed_frame_offset",
    "camera_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN_CONFIG)
    parser.add_argument(
        "--segmenter-config",
        type=Path,
        default=DEFAULT_SEGMENTER_CONFIG,
    )
    parser.add_argument(
        "--fit3d-analysis-dir",
        type=Path,
        default=DEFAULT_FIT3D_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--qevd-analysis-dir",
        type=Path,
        default=DEFAULT_QEVD_ANALYSIS_DIR,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=DEFAULT_FROZEN_CONFIG,
    )
    return parser.parse_args()


def finite_score(value: Any, *, label: str = "score") -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label} must be finite in [0, 1], got {value!r}")
    return score


def binary_target(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target must be 0 or 1, got {value!r}") from exc
    if numeric not in (0.0, 1.0):
        raise ValueError(f"target must be 0 or 1, got {value!r}")
    return int(numeric)


def validate_bin_edges(bin_edges: Iterable[float]) -> list[float]:
    edges = [finite_score(value, label="bin edge") for value in bin_edges]
    if len(edges) < 2:
        raise ValueError("at least two bin edges are required")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("bin edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("bin edges must be strictly increasing")
    return edges


def calibration_bin_index(score: float, bin_edges: Iterable[float]) -> int:
    edges = validate_bin_edges(bin_edges)
    value = finite_score(score)
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def beta_lower_bound(successes: int, total: int, alpha: float) -> float:
    """Return a one-sided Clopper-Pearson lower bound for a binomial rate."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must satisfy 0 <= successes <= total")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if total == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def fit_binned_calibrator(
    samples: Iterable[dict[str, Any]],
    *,
    bin_edges: Iterable[float] = CALIBRATION_BIN_EDGES,
    alpha: float = LOWER_BOUND_ALPHA,
) -> dict[str, Any]:
    edges = validate_bin_edges(bin_edges)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    positives = np.zeros(len(edges) - 1, dtype=np.int64)
    for sample in samples:
        score = finite_score(sample.get("score"))
        target = binary_target(sample.get("target"))
        bin_index = calibration_bin_index(score, edges)
        counts[bin_index] += 1
        positives[bin_index] += target

    raw_lower_bounds = []
    bins = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        total = int(counts[index])
        positive = int(positives[index])
        negative = total - positive
        lower_bound = beta_lower_bound(positive, total, alpha)
        raw_lower_bounds.append(lower_bound)
        bins.append(
            {
                "bin_index": index,
                "score_min": left,
                "score_max": right,
                "n": total,
                "tp": positive,
                "fp": negative,
                "raw_accuracy": (
                    round(positive / total, 6) if total else None
                ),
                "beta_lower_bound": round(lower_bound, 12),
            }
        )

    monotonic_lower_bounds = list(
        np.maximum.accumulate(np.asarray(raw_lower_bounds, dtype=np.float64))
    )
    for row, calibrated in zip(bins, monotonic_lower_bounds):
        row["monotonic_lower_bound"] = round(float(calibrated), 12)
    return {
        "method": "binned_beta_lower_bound",
        "bin_edges": edges,
        "lower_bound_alpha": alpha,
        "sample_count": int(counts.sum()),
        "positive_count": int(positives.sum()),
        "negative_count": int(counts.sum() - positives.sum()),
        "lower_bounds": [round(value, 12) for value in raw_lower_bounds],
        "monotonic_lower_bounds": [
            round(float(value), 12) for value in monotonic_lower_bounds
        ],
        "bins": bins,
    }


def calibrate_score(score: float, core: dict[str, Any]) -> float:
    bin_index = calibration_bin_index(score, core["bin_edges"])
    bounds = core["lower_bounds"]
    if len(bounds) != len(core["bin_edges"]) - 1:
        raise ValueError("lower_bounds length must equal bin count")
    return finite_score(bounds[bin_index], label="calibrated score")


def rep_unit_eligible(is_default: bool, rep_status: str) -> bool:
    return bool(is_default) and rep_status == "complete"


def rep_anchor_frame(rep: dict[str, Any]) -> int:
    if rep.get("top"):
        return int(rep["top"]["anchor_frame"])
    return int(round(0.5 * (rep["start_frame"] + rep["end_frame"])))


def empty_metric() -> dict[str, Any]:
    return {
        "n": 0,
        "positive_count": 0,
        "negative_count": 0,
        "brier": None,
        "ece": None,
        "mean_score": None,
        "accuracy": None,
    }


def binary_calibration_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    score_key: str,
    ece_bins: int = ECE_BIN_COUNT,
) -> dict[str, Any]:
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")
    values = [
        (
            finite_score(row.get(score_key), label=score_key),
            binary_target(row.get("target")),
        )
        for row in rows
    ]
    if not values:
        return empty_metric()
    scores = np.asarray([value[0] for value in values], dtype=np.float64)
    targets = np.asarray([value[1] for value in values], dtype=np.float64)
    assignments = np.minimum(
        (scores * ece_bins).astype(np.int64), ece_bins - 1
    )
    ece = 0.0
    for index in range(ece_bins):
        mask = assignments == index
        count = int(mask.sum())
        if not count:
            continue
        ece += (
            count
            / len(values)
            * abs(float(scores[mask].mean() - targets[mask].mean()))
        )
    return {
        "n": len(values),
        "positive_count": int(targets.sum()),
        "negative_count": int(len(values) - targets.sum()),
        "brier": round(float(np.mean((scores - targets) ** 2)), 6),
        "ece": round(float(ece), 6),
        "mean_score": round(float(scores.mean()), 6),
        "accuracy": round(float(targets.mean()), 6),
    }


def threshold_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    score_key: str,
    threshold: float,
) -> dict[str, Any]:
    values = [
        (
            finite_score(row.get(score_key), label=score_key),
            binary_target(row.get("target")),
        )
        for row in rows
    ]
    if not values:
        return {
            "accepted_count": 0,
            "accepted_tp": 0,
            "accepted_fp": 0,
            "rejected_count": 0,
            "rejected_tp": 0,
            "rejected_fp": 0,
            "coverage": None,
            "precision": None,
            "true_positive_coverage": None,
            "high_confidence_fp": 0,
        }
    accepted = [(score, target) for score, target in values if score >= threshold]
    rejected = [(score, target) for score, target in values if score < threshold]
    accepted_tp = sum(target for _, target in accepted)
    accepted_fp = len(accepted) - accepted_tp
    positive_count = sum(target for _, target in values)
    return {
        "accepted_count": len(accepted),
        "accepted_tp": accepted_tp,
        "accepted_fp": accepted_fp,
        "rejected_count": len(rejected),
        "rejected_tp": sum(target for _, target in rejected),
        "rejected_fp": len(rejected) - sum(target for _, target in rejected),
        "coverage": phase_states.ratio(len(accepted), len(values)),
        "precision": phase_states.ratio(accepted_tp, len(accepted)),
        "true_positive_coverage": phase_states.ratio(
            accepted_tp, positive_count
        ),
        "high_confidence_fp": accepted_fp,
    }


def load_analysis(path: Path) -> dict[str, Any]:
    analysis = phase_states.load_json(path)
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected analysis schema in {path}: "
            f"{analysis.get('schema_version')}"
        )
    return analysis


def default_set(analysis: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (set_candidate for set_candidate in analysis["sets"] if set_candidate["is_default"]),
        None,
    )


def base_rep_row(
    *,
    dataset: str,
    clip: dict[str, Any],
    analysis: dict[str, Any],
    set_candidate: dict[str, Any],
    rep: dict[str, Any],
    unit_eligible: bool,
    gt_available: bool,
) -> dict[str, Any]:
    confidence = rep["confidence"]
    exclusion_reason = None
    if not set_candidate["is_default"]:
        exclusion_reason = "non_default_set"
    elif rep["status"] != "complete":
        exclusion_reason = "endpoint_partial_not_formal"
    elif not gt_available:
        exclusion_reason = "gt_unavailable"
    return {
        "dataset": dataset,
        "clip_id": clip["clip_id"],
        "cluster_id": clip.get("cluster_id"),
        "action": clip.get("action"),
        "clip_status": analysis["status"],
        "set_id": int(set_candidate["set_id"]),
        "is_default_set": bool(set_candidate["is_default"]),
        "set_chain_confidence": float(set_candidate["chain_confidence"]),
        "rep_id": int(rep["rep_id"]),
        "rep_status": rep["status"],
        "partial_side": rep.get("partial_side"),
        "anchor_frame": rep_anchor_frame(rep),
        "start_frame": int(rep["start_frame"]),
        "end_frame": int(rep["end_frame"]),
        "raw_complete": finite_score(confidence["complete"]),
        "raw_in_set": finite_score(confidence["in_set"]),
        "raw_combined": finite_score(confidence["combined"]),
        "unit_eligible": bool(unit_eligible),
        "gt_evaluable": bool(unit_eligible and gt_available),
        "exclusion_reason": exclusion_reason,
        "is_tp": None,
        "gt_match_iou": None,
        "gt_match_error_frames": None,
        "outcome": "excluded",
    }


def build_fit3d_rep_rows(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for clip in sorted(clips, key=lambda item: item["clip_id"]):
        analysis = analyses[clip["clip_id"]]
        gt_frames = sorted(
            int(event["clip_frame"])
            for event in (clip.get("gt") or {}).get("events", [])
            if event.get("event") == "top"
        )
        selected_default = default_set(analysis)
        eligible_keys = []
        eligible_anchors = []
        if selected_default:
            for rep in selected_default["reps"]:
                if rep_unit_eligible(True, rep["status"]):
                    eligible_keys.append((selected_default["set_id"], rep["rep_id"]))
                    eligible_anchors.append(rep_anchor_frame(rep))
        gate = chain_selection.event_gate_frames(gt_frames)
        matches = phase_states.match_events(gt_frames, eligible_anchors, gate)
        match_by_key = {
            eligible_keys[pred_index]: (error, gt_index)
            for gt_index, pred_index, error in matches
        }
        for set_candidate in analysis["sets"]:
            for rep in set_candidate["reps"]:
                unit_eligible = rep_unit_eligible(
                    bool(set_candidate["is_default"]), rep["status"]
                )
                row = base_rep_row(
                    dataset=FIT3D_DATASET,
                    clip=clip,
                    analysis=analysis,
                    set_candidate=set_candidate,
                    rep=rep,
                    unit_eligible=unit_eligible,
                    gt_available=bool(gt_frames),
                )
                key = (set_candidate["set_id"], rep["rep_id"])
                if row["gt_evaluable"]:
                    match = match_by_key.get(key)
                    row["is_tp"] = match is not None
                    row["outcome"] = "tp" if match else "fp"
                    if match:
                        row["gt_match_error_frames"] = int(match[0])
                rows.append(row)
    return rows


def build_qevd_rep_rows(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for clip in sorted(clips, key=lambda item: item["clip_id"]):
        analysis = analyses[clip["clip_id"]]
        trim_start = int((clip.get("trim") or {}).get("start_frame", 0))
        intervals = [
            {
                "start_frame": int(interval["start_frame"]) - trim_start,
                "end_frame": int(interval["end_frame"]) - trim_start,
            }
            for interval in chain_selection.annotation_intervals(clip)
        ]
        selected_default = default_set(analysis)
        eligible_keys = []
        eligible_intervals = []
        if selected_default:
            for rep in selected_default["reps"]:
                if rep_unit_eligible(True, rep["status"]):
                    eligible_keys.append((selected_default["set_id"], rep["rep_id"]))
                    eligible_intervals.append(
                        {
                            "start_frame": int(rep["start_frame"]),
                            "end_frame": int(rep["end_frame"]),
                        }
                    )
        matches, _, _ = chain_selection.match_intervals_by_iou(
            intervals, eligible_intervals, minimum_iou=QEVD_MINIMUM_IOU
        )
        match_by_key = {
            eligible_keys[pred_index]: iou for _, pred_index, iou in matches
        }
        for set_candidate in analysis["sets"]:
            for rep in set_candidate["reps"]:
                unit_eligible = rep_unit_eligible(
                    bool(set_candidate["is_default"]), rep["status"]
                )
                row = base_rep_row(
                    dataset=QEVD_DATASET,
                    clip=clip,
                    analysis=analysis,
                    set_candidate=set_candidate,
                    rep=rep,
                    unit_eligible=unit_eligible,
                    gt_available=bool(intervals),
                )
                key = (set_candidate["set_id"], rep["rep_id"])
                if row["gt_evaluable"]:
                    iou = match_by_key.get(key)
                    row["is_tp"] = iou is not None
                    row["outcome"] = "tp" if iou is not None else "fp"
                    row["gt_match_iou"] = round(float(iou), 6) if iou else None
                rows.append(row)
    return rows


def apply_calibration(rows: list[dict[str, Any]], core: dict[str, Any]) -> None:
    for row in rows:
        raw_score = finite_score(row["raw_combined"], label="raw_combined")
        calibrated = calibrate_score(raw_score, core)
        row["calibration_bin"] = calibration_bin_index(
            raw_score, core["bin_edges"]
        )
        row["calibrated_combined"] = round(calibrated, 6)
        row["high_confidence"] = bool(
            calibrated >= float(core["high_confidence_threshold"])
        )


def target_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "target": int(bool(row["is_tp"])),
        }
        for row in rows
        if row["gt_evaluable"] and row["is_tp"] is not None
    ]


def reason_supported(
    reason: str,
    *,
    eligible: bool,
    event_in_accepted_set: bool,
) -> bool:
    if reason == "shape_mismatch":
        return not eligible
    if reason == "no_confident_set":
        return eligible
    if reason == "outside_set":
        return eligible and event_in_accepted_set
    return False


def build_rejection_rows(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for clip in sorted(clips, key=lambda item: item["clip_id"]):
        analysis = analyses[clip["clip_id"]]
        accepted_ranges = [
            (int(set_candidate["start_frame"]), int(set_candidate["end_frame"]))
            for set_candidate in analysis["sets"]
        ]
        for candidate in analysis.get("rejected_candidates", []):
            evidence = candidate.get("evidence") or {}
            event_frame = int(
                evidence.get(
                    "event_frame",
                    round(
                        0.5
                        * (
                            candidate["start_frame"]
                            + candidate["end_frame"]
                        )
                    ),
                )
            )
            in_set = any(start <= event_frame <= end for start, end in accepted_ranges)
            eligible = bool(evidence.get("eligible"))
            reasons = list(candidate.get("reasons") or [])
            supports = [
                reason_supported(
                    reason,
                    eligible=eligible,
                    event_in_accepted_set=in_set,
                )
                for reason in reasons
            ]
            rows.append(
                {
                    "dataset": clip["dataset"],
                    "clip_id": clip["clip_id"],
                    "clip_status": analysis["status"],
                    "candidate_id": candidate["candidate_id"],
                    "event_frame": event_frame,
                    "reasons": "|".join(reasons),
                    "eligible": eligible,
                    "event_in_accepted_set": in_set,
                    "relative_amplitude": evidence.get("relative_amplitude"),
                    "candidate_confidence_combined": finite_score(
                        (candidate.get("confidence") or {}).get("combined", 0.0)
                    ),
                    "reason_supported": bool(supports) and all(supports),
                }
            )
    return rows


def selected_event_diagnostics(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    high_confidence_fp = 0
    high_confidence_tp = 0
    event_count = 0
    for clip in clips:
        analysis = analyses[clip["clip_id"]]
        events = (analysis.get("diagnostics") or {}).get("selected_top_events") or []
        gt_frames = sorted(
            int(event["clip_frame"])
            for event in (clip.get("gt") or {}).get("events", [])
            if event.get("event") == "top"
        )
        pred_frames = [int(event["frame"]) for event in events]
        matches = phase_states.match_events(
            gt_frames,
            pred_frames,
            chain_selection.event_gate_frames(gt_frames),
        )
        matched = {pred_index for _, pred_index, _ in matches}
        for pred_index, event in enumerate(events):
            event_count += 1
            score = float(event["confidence"]["combined"])
            if score < threshold:
                continue
            if pred_index in matched:
                high_confidence_tp += 1
            else:
                high_confidence_fp += 1
    return {
        "scope": (
            "all_selected_top_events_including_endpoint_partial_and_chain_anchors"
        ),
        "event_count": event_count,
        "high_confidence_tp": high_confidence_tp,
        "high_confidence_fp": high_confidence_fp,
    }


def aggregate_hashes(paths: Iterable[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_frozen_config(
    fit: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "task": "T1-6D",
        "frozen": True,
        "core": {
            "dataset_independent": True,
            "method": "binned_beta_lower_bound",
            "input_score": "rep.confidence.combined",
            "reported_layers": list(CONFIDENCE_LAYERS),
            "evaluation_unit": "default_set_complete_rep",
            "accepted_rep_status": "complete",
            "endpoint_partial_policy": (
                "candidate_only_not_accepted_complete_rep"
            ),
            "bin_edges": list(fit["bin_edges"]),
            "lower_bound_alpha": float(fit["lower_bound_alpha"]),
            "raw_beta_lower_bounds": list(fit["lower_bounds"]),
            "lower_bounds": list(fit["monotonic_lower_bounds"]),
            "monotonic_enforced": True,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "ece_bins": ECE_BIN_COUNT,
            "ece_relative_improvement_min": ECE_RELATIVE_IMPROVEMENT_MIN,
        },
        "provenance": provenance,
    }


def validate_frozen_config(
    payload: dict[str, Any],
    *,
    fitted: dict[str, Any] | None = None,
) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unexpected confidence config schema")
    if payload.get("task") != "T1-6D" or payload.get("frozen") is not True:
        raise ValueError("confidence config must be frozen for T1-6D")
    core = payload.get("core") or {}
    required = {
        "dataset_independent": True,
        "method": "binned_beta_lower_bound",
        "input_score": "rep.confidence.combined",
        "evaluation_unit": "default_set_complete_rep",
        "accepted_rep_status": "complete",
        "endpoint_partial_policy": (
            "candidate_only_not_accepted_complete_rep"
        ),
        "monotonic_enforced": True,
    }
    for key, expected in required.items():
        if core.get(key) != expected:
            raise ValueError(f"invalid confidence core field {key}: {core.get(key)!r}")
    edges = validate_bin_edges(core.get("bin_edges") or [])
    bounds = [finite_score(value, label="lower bound") for value in core["lower_bounds"]]
    raw_bounds = [
        finite_score(value, label="raw beta lower bound")
        for value in core["raw_beta_lower_bounds"]
    ]
    if len(bounds) != len(edges) - 1 or len(raw_bounds) != len(edges) - 1:
        raise ValueError("confidence lower bounds do not match bin count")
    if any(right < left for left, right in zip(bounds[:-1], bounds[1:])):
        raise ValueError("confidence lower bounds must be monotonic")
    finite_score(core.get("lower_bound_alpha"), label="lower_bound_alpha")
    finite_score(
        core.get("high_confidence_threshold"),
        label="high_confidence_threshold",
    )
    if float(core["lower_bound_alpha"]) != LOWER_BOUND_ALPHA:
        raise ValueError("frozen lower_bound_alpha changed after registration")
    if float(core["high_confidence_threshold"]) != HIGH_CONFIDENCE_THRESHOLD:
        raise ValueError("frozen high-confidence threshold changed")
    if int(core.get("ece_bins", 0)) != ECE_BIN_COUNT:
        raise ValueError("frozen ECE bin count changed")
    core_text = json.dumps(core, ensure_ascii=False, sort_keys=True).lower()
    forbidden = [token for token in FORBIDDEN_CORE_TOKENS if token in core_text]
    if forbidden:
        raise ValueError(f"forbidden confidence core tokens: {forbidden}")
    if fitted is not None:
        if core["raw_beta_lower_bounds"] != fitted["lower_bounds"]:
            raise ValueError("frozen raw beta lower bounds do not match dev fit")
        if core["lower_bounds"] != fitted["monotonic_lower_bounds"]:
            raise ValueError("frozen monotonic lower bounds do not match dev fit")
        if core["bin_edges"] != fitted["bin_edges"]:
            raise ValueError("frozen bin edges do not match dev fit")
        if float(core["lower_bound_alpha"]) != float(
            fitted["lower_bound_alpha"]
        ):
            raise ValueError("frozen alpha does not match dev fit")


def load_or_create_frozen_config(
    path: Path,
    fit: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        payload = phase_states.load_json(path)
        validate_frozen_config(payload, fitted=fit)
        return payload
    payload = build_frozen_config(fit, provenance)
    validate_frozen_config(payload, fitted=fit)
    path.parent.mkdir(parents=True, exist_ok=True)
    phase_states.write_json(path, payload)
    return payload


def formal_population(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    values = list(rows)
    unit_rows = [row for row in values if row["unit_eligible"]]
    return {
        "accepted_rep_count": len(values),
        "unit_eligible_count": len(unit_rows),
        "gt_evaluable_count": sum(row["gt_evaluable"] for row in unit_rows),
        "endpoint_partial_default_count": sum(
            row["is_default_set"] and row["rep_status"] != "complete"
            for row in values
        ),
        "non_default_set_rep_count": sum(not row["is_default_set"] for row in values),
        "clips_with_formal_rep": len(
            {row["clip_id"] for row in unit_rows}
        ),
    }


def build_summary(
    *,
    args: argparse.Namespace,
    fit3d_clips: list[dict[str, Any]],
    qevd_clips: list[dict[str, Any]],
    fit3d_analyses: dict[str, dict[str, Any]],
    qevd_analyses: dict[str, dict[str, Any]],
    fit3d_rows: list[dict[str, Any]],
    qevd_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    fit: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    core = config["core"]
    threshold = float(core["high_confidence_threshold"])
    fit3d_targets = target_rows(fit3d_rows)
    qevd_targets = target_rows(qevd_rows)
    raw_layer_metrics = {
        layer: binary_calibration_metrics(
            fit3d_targets, score_key=f"raw_{layer}"
        )
        for layer in CONFIDENCE_LAYERS
    }
    raw_combined = raw_layer_metrics["combined"]
    calibrated = binary_calibration_metrics(
        fit3d_targets, score_key="calibrated_combined"
    )
    raw_threshold = threshold_metrics(
        fit3d_targets, score_key="raw_combined", threshold=threshold
    )
    calibrated_threshold = threshold_metrics(
        fit3d_targets, score_key="calibrated_combined", threshold=threshold
    )
    ece_relative_improvement = (
        round(
            (raw_combined["ece"] - calibrated["ece"]) / raw_combined["ece"],
            6,
        )
        if raw_combined["ece"]
        else None
    )
    event_diagnostics = selected_event_diagnostics(
        fit3d_clips, fit3d_analyses, threshold
    )
    reason_counts = Counter()
    for row in rejection_rows:
        for reason in str(row["reasons"]).split("|"):
            if reason:
                reason_counts[reason] += 1
    all_rejections_supported = all(
        bool(row["reason_supported"]) for row in rejection_rows
    )
    qevd_raw = binary_calibration_metrics(
        qevd_targets, score_key="raw_combined"
    )
    qevd_calibrated = binary_calibration_metrics(
        qevd_targets, score_key="calibrated_combined"
    )
    qevd_raw_threshold = threshold_metrics(
        qevd_targets, score_key="raw_combined", threshold=threshold
    )
    qevd_calibrated_threshold = threshold_metrics(
        qevd_targets, score_key="calibrated_combined", threshold=threshold
    )
    annotated_qevd_clips = {
        clip["clip_id"]
        for clip in qevd_clips
        if chain_selection.annotation_intervals(clip)
    }
    qevd_unlabeled_unit_rows = [
        row
        for row in qevd_rows
        if row["unit_eligible"] and not row["gt_evaluable"]
    ]
    fit3d_gates = {
        "calibrated_complete_rep_high_confidence_fp_zero": (
            calibrated_threshold["high_confidence_fp"] == 0
        ),
        "raw_complete_rep_high_confidence_fp_zero": (
            raw_threshold["high_confidence_fp"] == 0
        ),
        "calibrated_ece_lt_raw_ece": bool(
            calibrated["ece"] is not None
            and raw_combined["ece"] is not None
            and calibrated["ece"] < raw_combined["ece"]
        ),
        "calibrated_ece_relative_improvement_gte_min": bool(
            ece_relative_improvement is not None
            and ece_relative_improvement
            >= float(core["ece_relative_improvement_min"])
        ),
        "all_rejection_reasons_supported": all_rejections_supported,
        "frozen_before_holdout": bool(config["frozen"]),
    }
    qevd_formal_evaluable = len(annotated_qevd_clips) > 1
    qevd_smoke_gates = {
        "observed_high_confidence_fp_zero": (
            qevd_calibrated_threshold["high_confidence_fp"] == 0
        ),
        "calibrated_ece_lt_raw_ece": bool(
            qevd_calibrated["ece"] is not None
            and qevd_raw["ece"] is not None
            and qevd_calibrated["ece"] < qevd_raw["ece"]
        ),
    }
    qevd_summary = {
        "clip_count": len(qevd_clips),
        "annotated_clip_count": len(annotated_qevd_clips),
        "annotated_rep_count": len(qevd_targets),
        "unlabeled_formal_rep_count": len(qevd_unlabeled_unit_rows),
        "unlabeled_high_confidence_count": sum(
            row["high_confidence"] for row in qevd_unlabeled_unit_rows
        ),
        "raw_combined": qevd_raw,
        "calibrated_combined": qevd_calibrated,
        "raw_threshold": qevd_raw_threshold,
        "calibrated_threshold": qevd_calibrated_threshold,
        "smoke_gates": qevd_smoke_gates,
        "smoke_passed": all(qevd_smoke_gates.values()),
        "formal_stage_gate_evaluable": qevd_formal_evaluable,
        "stage_passed": None,
        "limitation": (
            "Only one QEVD dev clip has blind rep-interval GT; zero FP is "
            "observed on five matched reps, not a formal cross-domain gate."
        ),
    }
    fit3d_summary = {
        "clip_count": len(fit3d_clips),
        "population": formal_population(fit3d_rows),
        "raw_layers": raw_layer_metrics,
        "calibrated_combined": calibrated,
        "raw_threshold": raw_threshold,
        "calibrated_threshold": calibrated_threshold,
        "ece_relative_improvement": ece_relative_improvement,
        "event_level_diagnostic": event_diagnostics,
        "stage_gates": fit3d_gates,
        "stage_passed": all(fit3d_gates.values()),
    }
    low_confidence_rows = [
        row
        for row in fit3d_targets
        if not row["high_confidence"]
    ]
    rejection_summary = {
        "candidate_rejection_count": len(rejection_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "all_reasons_supported": all_rejections_supported,
        "unsupported_count": sum(
            not row["reason_supported"] for row in rejection_rows
        ),
        "low_confidence_formal_rep_count": len(low_confidence_rows),
        "low_confidence_formal_fp_count": sum(
            not row["is_tp"] for row in low_confidence_rows
        ),
        "low_confidence_formal_tp_count": sum(
            bool(row["is_tp"]) for row in low_confidence_rows
        ),
    }
    clip_output_coverage = phase_states.ratio(
        len({row["clip_id"] for row in fit3d_rows if row["is_default_set"]}),
        len(fit3d_clips),
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "task": "T1-6D",
        "generated_at": phase_states.utc_now(),
        "source": {
            "sources": str(args.sources.resolve()),
            "sources_sha256": phase_states.sha256_path(args.sources),
            "chain_config": str(args.chain_config.resolve()),
            "chain_config_sha256": phase_states.sha256_path(args.chain_config),
            "segmenter_config": str(args.segmenter_config.resolve()),
            "segmenter_config_sha256": phase_states.sha256_path(
                args.segmenter_config
            ),
            "frozen_config": str(config_path.resolve()),
            "frozen_config_sha256": phase_states.sha256_path(config_path),
            "fit3d_analysis_sha256": aggregate_hashes(
                args.fit3d_analysis_dir.glob("*.json")
            ),
            "qevd_analysis_sha256": aggregate_hashes(
                args.qevd_analysis_dir.glob("*.json")
            ),
        },
        "calibration": {
            "fit_scope": "fit3d_dev_default_complete_rep_only",
            "sample_count": fit["sample_count"],
            "positive_count": fit["positive_count"],
            "negative_count": fit["negative_count"],
            "bin_edges": fit["bin_edges"],
            "raw_beta_lower_bounds": fit["lower_bounds"],
            "monotonic_lower_bounds": fit["monotonic_lower_bounds"],
            "lower_bound_alpha": fit["lower_bound_alpha"],
            "high_confidence_threshold": threshold,
        },
        "fit3d_dev": {
            **fit3d_summary,
            "clip_output_coverage": clip_output_coverage,
        },
        "qevd_dev": qevd_summary,
        "rejection_audit": rejection_summary,
        "verdict": {
            "fit3d_dev_stage_passed": all(fit3d_gates.values()),
            "qevd_dev_formal_gate_evaluable": qevd_formal_evaluable,
            "qevd_dev_stage_passed": None,
            "overall_stage_passed": None if not qevd_formal_evaluable else all(
                [all(fit3d_gates.values()), qevd_summary["stage_passed"]]
            ),
            "note": (
                "Fit3D dev calibration gate is complete. QEVD dev has only "
                "one annotated clip, so its zero-FP observation remains a "
                "cross-domain smoke result and cannot be promoted to a formal "
                "stage pass."
            ),
        },
        "artifacts": {
            "confidence_calibration": "confidence_calibration.csv",
            "rep_confidence_audit": "rep_confidence_audit.csv",
            "rejection_audit": "rejection_audit.csv",
            "qevd_confidence_regression": "qevd_confidence_regression.csv",
            "summary": "confidence_eval_summary.json",
        },
    }


def write_outputs(
    results_dir: Path,
    *,
    fit: dict[str, Any],
    fit3d_rows: list[dict[str, Any]],
    qevd_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows = []
    for row in fit["bins"]:
        calibrated = float(row["monotonic_lower_bound"])
        calibration_rows.append(
            {
                **row,
                "calibrated_confidence": round(calibrated, 6),
                "high_confidence": calibrated >= HIGH_CONFIDENCE_THRESHOLD,
            }
        )
    calibration_fields = [
        "bin_index",
        "score_min",
        "score_max",
        "n",
        "tp",
        "fp",
        "raw_accuracy",
        "beta_lower_bound",
        "monotonic_lower_bound",
        "calibrated_confidence",
        "high_confidence",
    ]
    phase_states.write_csv(
        results_dir / "confidence_calibration.csv",
        calibration_rows,
        calibration_fields,
    )
    rep_fields = [
        "dataset",
        "clip_id",
        "cluster_id",
        "action",
        "clip_status",
        "set_id",
        "is_default_set",
        "set_chain_confidence",
        "rep_id",
        "rep_status",
        "partial_side",
        "anchor_frame",
        "start_frame",
        "end_frame",
        "raw_complete",
        "raw_in_set",
        "raw_combined",
        "calibration_bin",
        "calibrated_combined",
        "high_confidence",
        "unit_eligible",
        "gt_evaluable",
        "exclusion_reason",
        "is_tp",
        "outcome",
        "gt_match_iou",
        "gt_match_error_frames",
    ]
    all_rep_rows = sorted(
        fit3d_rows + qevd_rows,
        key=lambda row: (row["dataset"], row["clip_id"], row["set_id"], row["rep_id"]),
    )
    phase_states.write_csv(
        results_dir / "rep_confidence_audit.csv",
        all_rep_rows,
        rep_fields,
    )
    phase_states.write_csv(
        results_dir / "qevd_confidence_regression.csv",
        qevd_rows,
        rep_fields,
    )
    rejection_fields = [
        "dataset",
        "clip_id",
        "clip_status",
        "candidate_id",
        "event_frame",
        "reasons",
        "eligible",
        "event_in_accepted_set",
        "relative_amplitude",
        "candidate_confidence_combined",
        "reason_supported",
    ]
    phase_states.write_csv(
        results_dir / "rejection_audit.csv",
        rejection_rows,
        rejection_fields,
    )
    phase_states.write_json(
        results_dir / "confidence_eval_summary.json", summary
    )
    phase_states.write_checksums(
        results_dir,
        [
            "confidence_calibration.csv",
            "rep_confidence_audit.csv",
            "rejection_audit.csv",
            "qevd_confidence_regression.csv",
            "confidence_eval_summary.json",
        ],
    )


def main() -> int:
    args = parse_args()
    sources = phase_states.load_json(args.sources)
    fit3d_clips = chain_selection.load_fit3d_clips(sources)
    qevd_clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == QEVD_DATASET
        and clip.get("split_role") == "dev"
    ]
    if len(qevd_clips) != EXPECTED_QEVD_DEV_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_QEVD_DEV_CLIPS} QEVD dev clips, "
            f"found {len(qevd_clips)}"
        )
    fit3d_analyses = {
        clip["clip_id"]: load_analysis(
            args.fit3d_analysis_dir / f"{clip['clip_id']}.json"
        )
        for clip in fit3d_clips
    }
    qevd_analyses = {
        clip["clip_id"]: load_analysis(
            args.qevd_analysis_dir / f"{clip['clip_id']}.json"
        )
        for clip in qevd_clips
    }
    fit3d_rows = build_fit3d_rep_rows(fit3d_clips, fit3d_analyses)
    qevd_rows = build_qevd_rep_rows(qevd_clips, qevd_analyses)
    calibration_samples = [
        {"score": row["raw_combined"], "target": int(bool(row["is_tp"]))}
        for row in fit3d_rows
        if row["gt_evaluable"]
    ]
    fit = fit_binned_calibrator(
        calibration_samples,
        bin_edges=CALIBRATION_BIN_EDGES,
        alpha=LOWER_BOUND_ALPHA,
    )
    provenance = {
        "generated_at": phase_states.utc_now(),
        "fit_scope": "fit3d_dev_default_complete_rep_only",
        "sample_count": fit["sample_count"],
        "positive_count": fit["positive_count"],
        "negative_count": fit["negative_count"],
        "cluster_count": len({clip["cluster_id"] for clip in fit3d_clips}),
        "sources_sha256": phase_states.sha256_path(args.sources),
        "chain_config_sha256": phase_states.sha256_path(args.chain_config),
        "segmenter_config_sha256": phase_states.sha256_path(
            args.segmenter_config
        ),
    }
    config = load_or_create_frozen_config(
        args.frozen_config,
        fit,
        provenance,
    )
    apply_calibration(fit3d_rows, config["core"])
    apply_calibration(qevd_rows, config["core"])
    rejection_rows = build_rejection_rows(
        fit3d_clips + qevd_clips,
        {**fit3d_analyses, **qevd_analyses},
    )
    summary = build_summary(
        args=args,
        fit3d_clips=fit3d_clips,
        qevd_clips=qevd_clips,
        fit3d_analyses=fit3d_analyses,
        qevd_analyses=qevd_analyses,
        fit3d_rows=fit3d_rows,
        qevd_rows=qevd_rows,
        rejection_rows=rejection_rows,
        fit=fit,
        config=config,
        config_path=args.frozen_config,
    )
    write_outputs(
        args.results_dir,
        fit=fit,
        fit3d_rows=fit3d_rows,
        qevd_rows=qevd_rows,
        rejection_rows=rejection_rows,
        summary=summary,
    )
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
