#!/usr/bin/env python3
"""Evaluate frozen RepChain v2 on Fit3D holdout and available QEVD holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_chain_selection as chain_selection
import task1_6_confidence as confidence
import task1_6_phase_states as phase_states
import task1_6_quality as quality_gates


DEFAULT_SOURCES = ROOT / "configs" / "task1_6_sources.json"
DEFAULT_SPECS = ROOT / "configs" / "task1_4_exercise_specs.json"
DEFAULT_SEGMENTER_CONFIG = ROOT / "configs" / "task1_6_segmenter.json"
DEFAULT_CHAIN_CONFIG = ROOT / "configs" / "task1_6_chain.json"
DEFAULT_CONFIDENCE_CONFIG = ROOT / "configs" / "task1_6_confidence_frozen.json"
DEFAULT_FIT3D_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_FIT3D_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_T1_5_EVALUATION_DIR = (
    ROOT / "results" / "task1_6" / "baseline_t1_5d_matchfix"
)
DEFAULT_SEALED_MANIFEST = (
    ROOT / "results" / "task1_6" / "manifest" / "qevd_sealed_holdout_manifest.json"
)
DEFAULT_SEALED_TEMPLATE = (
    ROOT / "results" / "task1_6" / "manifest" / "qevd_sealed_holdout_template.csv"
)
DEFAULT_RUNS_DIR = ROOT / "results" / "task1_6" / "runs" / "holdout"
DEFAULT_SEALED_INPUT_DIR = DEFAULT_RUNS_DIR / "qevd_sealed_holdout_input"
DEFAULT_EVALUATION_DIR = ROOT / "results" / "task1_6" / "evaluation"
DEFAULT_TABLES_DIR = ROOT / "results" / "task1_6" / "tables"
DEFAULT_FIGURES_DIR = ROOT / "results" / "task1_6" / "figures"

SCHEMA_VERSION = "task1-6e-evaluation-v1"
FIT3D_DATASET = "Fit3D"
QEVD_DATASET = "QEVD-FIT-COACH"
EXPECTED_FIT3D_HOLDOUT_CLIPS = 32
EXPECTED_FIT3D_HOLDOUT_CLUSTERS = 8
EXPECTED_QEVD_SEALED_SEGMENTS = 6
HIGH_CONFIDENCE_THRESHOLD = 0.5
COUNT_MAE_MAX = 0.5
EXACT_COUNT_NUMERATOR = 14
EXACT_COUNT_DENOMINATOR = 16
FIRST_LAST_RECALL_MIN = 0.75
PRIMARY_EVENT_MEDIAN_MAX = 5.0
PRIMARY_EVENT_P90_MAX = 12.0
TOP_INTERVAL_MEDIAN_BASELINE = 3.0
ROBUSTNESS_FPS_TARGETS = (15.0, 60.0)
ROBUSTNESS_AMPLITUDES = (0.8, 1.2)
ROBUSTNESS_OFFSET = 15.0
ROBUSTNESS_NOISE_SIGMA = 2.0
ROBUSTNESS_MISSING_FRACTION = 0.05
ROBUSTNESS_IDLE_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument(
        "--segmenter-config", type=Path, default=DEFAULT_SEGMENTER_CONFIG
    )
    parser.add_argument("--chain-config", type=Path, default=DEFAULT_CHAIN_CONFIG)
    parser.add_argument(
        "--confidence-config",
        type=Path,
        default=DEFAULT_CONFIDENCE_CONFIG,
    )
    parser.add_argument(
        "--fit3d-manifest", type=Path, default=DEFAULT_FIT3D_MANIFEST
    )
    parser.add_argument(
        "--fit3d-analysis-dir",
        type=Path,
        default=DEFAULT_FIT3D_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--t1-5-evaluation-dir",
        type=Path,
        default=DEFAULT_T1_5_EVALUATION_DIR,
    )
    parser.add_argument(
        "--sealed-manifest", type=Path, default=DEFAULT_SEALED_MANIFEST
    )
    parser.add_argument(
        "--sealed-template", type=Path, default=DEFAULT_SEALED_TEMPLATE
    )
    parser.add_argument(
        "--sealed-input-dir", type=Path, default=DEFAULT_SEALED_INPUT_DIR
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR
    )
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-sealed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return phase_states.load_json(path)


def write_json(path: Path, payload: Any) -> None:
    phase_states.write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    phase_states.write_csv(path, rows, fields)


def sha256_path(path: Path) -> str:
    return phase_states.sha256_path(path)


def sha256_tree(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def load_fit3d_holdout_clips(sources: dict[str, Any]) -> list[dict[str, Any]]:
    clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == FIT3D_DATASET
        and clip.get("split_role") == "holdout"
        and clip.get("has_gt")
        and not clip.get("smoke_only")
    ]
    if len(clips) != EXPECTED_FIT3D_HOLDOUT_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_FIT3D_HOLDOUT_CLIPS} Fit3D holdout clips, "
            f"found {len(clips)}"
        )
    clusters = {clip["cluster_id"] for clip in clips}
    if len(clusters) != EXPECTED_FIT3D_HOLDOUT_CLUSTERS:
        raise ValueError(
            f"expected {EXPECTED_FIT3D_HOLDOUT_CLUSTERS} holdout clusters, "
            f"found {len(clusters)}"
        )
    return clips


def load_qevd_sealed_clips(sources: dict[str, Any]) -> list[dict[str, Any]]:
    clips = [
        clip
        for clip in sources.get("clips", [])
        if clip.get("dataset") == QEVD_DATASET
        and clip.get("split_role") == "sealed_holdout"
    ]
    if len(clips) != EXPECTED_QEVD_SEALED_SEGMENTS:
        raise ValueError(
            f"expected {EXPECTED_QEVD_SEALED_SEGMENTS} sealed clips, "
            f"found {len(clips)}"
        )
    return clips


def gt_top_frames(clip: dict[str, Any]) -> list[int]:
    return sorted(
        int(event["clip_frame"])
        for event in (clip.get("gt") or {}).get("events", [])
        if event.get("event") == "top"
    )


def event_metrics(
    gt_frames: list[int],
    events: list[dict[str, Any]],
    *,
    high_confidence_threshold: float,
    require_set: bool = True,
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: int(event["frame"]))
    pred_frames = [int(event["frame"]) for event in ordered]
    gate = phase_states.event_gate_frames(gt_frames)
    matches = phase_states.match_events(gt_frames, pred_frames, gate)
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_pred = {pred_index for _, pred_index, _ in matches}
    errors = [error for _, _, error in matches]
    signed_errors = [
        pred_frames[pred_index] - gt_frames[gt_index]
        for gt_index, pred_index, _ in matches
    ]
    high_confidence_fp = 0
    outside_set_high_confidence_fp = 0
    for pred_index, event in enumerate(ordered):
        score = as_float((event.get("confidence") or {}).get("combined"))
        if pred_index in matched_pred or score is None:
            continue
        if score >= high_confidence_threshold:
            high_confidence_fp += 1
            if require_set and not event.get("set_ids"):
                outside_set_high_confidence_fp += 1
    tp = len(matches)
    return {
        "event_gate_frames": gate,
        "gt_event_count": len(gt_frames),
        "pred_event_count": len(ordered),
        "tp": tp,
        "fp": len(ordered) - tp,
        "fn": len(gt_frames) - tp,
        "event_recall": phase_states.ratio(tp, len(gt_frames)),
        "event_precision": phase_states.ratio(tp, len(ordered)),
        "first_gt_matched": bool(gt_frames) and 0 in matched_gt,
        "last_gt_matched": bool(gt_frames) and len(gt_frames) - 1 in matched_gt,
        "event_abs_error_median_frames": phase_states.percentile(errors, 50),
        "event_abs_error_p90_frames": phase_states.percentile(errors, 90),
        "high_confidence_event_fp": high_confidence_fp,
        "outside_set_high_confidence_fp": outside_set_high_confidence_fp,
        "matches": matches,
        "abs_errors": errors,
        "signed_errors": signed_errors,
    }


def aggregate_event_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_count = sum(int(row["gt_event_count"]) for row in rows)
    pred_count = sum(int(row["pred_event_count"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    errors = [value for row in rows for value in row.get("_abs_errors", [])]
    count_errors = [
        int(row["pred_event_count"]) - int(row["gt_event_count"])
        for row in rows
    ]
    precision = phase_states.ratio(tp, pred_count)
    recall = phase_states.ratio(tp, gt_count)
    return {
        "clip_count": len(rows),
        "cluster_count": len({row["cluster_id"] for row in rows}),
        "gt_event_count": gt_count,
        "pred_event_count": pred_count,
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": phase_states.f1_score(precision, recall),
        "first_event_recall": phase_states.ratio(
            sum(bool(row["first_gt_matched"]) for row in rows), len(rows)
        ),
        "last_event_recall": phase_states.ratio(
            sum(bool(row["last_gt_matched"]) for row in rows), len(rows)
        ),
        "event_abs_error_median_frames": phase_states.percentile(errors, 50),
        "event_abs_error_p90_frames": phase_states.percentile(errors, 90),
        "count_mae": round(float(np.mean(np.abs(count_errors))), 6)
        if count_errors
        else None,
        "count_exact_correct": sum(error == 0 for error in count_errors),
        "high_confidence_event_fp": sum(
            int(row.get("high_confidence_event_fp", row.get("high_confidence_fp", 0)))
            for row in rows
        ),
        "outside_set_high_confidence_fp": sum(
            int(row.get("outside_set_high_confidence_fp", 0))
            for row in rows
        ),
    }


def count_consensus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    cluster_rows = []
    for cluster_id, cluster_rows_raw in sorted(grouped.items()):
        gt_counts = {int(row["gt_event_count"]) for row in cluster_rows_raw}
        if len(gt_counts) != 1:
            raise ValueError(f"inconsistent GT count for {cluster_id}: {gt_counts}")
        gt_count = next(iter(gt_counts))
        predictions = [int(row["pred_event_count"]) for row in cluster_rows_raw]
        consensus = int(math.floor(float(np.median(predictions)) + 0.5))
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "camera_count": len(predictions),
                "camera_pred_counts": "|".join(str(value) for value in predictions),
                "gt_count": gt_count,
                "consensus_count": consensus,
                "count_error": consensus - gt_count,
                "abs_count_error": abs(consensus - gt_count),
                "exact": consensus == gt_count,
            }
        )
    return {
        "cluster_count": len(cluster_rows),
        "gt_rep_sum": sum(row["gt_count"] for row in cluster_rows),
        "pred_rep_sum": sum(row["consensus_count"] for row in cluster_rows),
        "count_mae": round(
            float(np.mean([row["abs_count_error"] for row in cluster_rows])), 6
        )
        if cluster_rows
        else None,
        "exact_count_correct": sum(row["exact"] for row in cluster_rows),
        "rows": cluster_rows,
    }


def top_to_top_interval_errors(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    *,
    selected_clip_ids: set[str] | None = None,
) -> list[int]:
    errors = []
    for clip in clips:
        if selected_clip_ids is not None and clip["clip_id"] not in selected_clip_ids:
            continue
        analysis = analyses[clip["clip_id"]]
        events = (analysis.get("diagnostics") or {}).get("selected_top_events") or []
        pred_frames = [int(event["frame"]) for event in events]
        gt_frames = gt_top_frames(clip)
        matches = phase_states.match_events(
            gt_frames,
            pred_frames,
            phase_states.event_gate_frames(gt_frames),
        )
        matched = {
            gt_index: pred_index for gt_index, pred_index, _ in matches
        }
        for gt_left in range(len(gt_frames) - 1):
            gt_right = gt_left + 1
            if gt_left not in matched or gt_right not in matched:
                continue
            pred_left = matched[gt_left]
            pred_right = matched[gt_right]
            if pred_right <= pred_left:
                continue
            errors.append(
                abs(
                    (pred_frames[pred_right] - pred_frames[pred_left])
                    - (gt_frames[gt_right] - gt_frames[gt_left])
                )
            )
    return errors


def default_set(analysis: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            set_candidate
            for set_candidate in analysis.get("sets", [])
            if set_candidate.get("is_default")
        ),
        None,
    )


def structure_counts(analysis: dict[str, Any]) -> tuple[int, int, int]:
    sets = analysis.get("sets", [])
    reps = sum(len(set_candidate.get("reps", [])) for set_candidate in sets)
    selected = default_set(analysis)
    default_reps = len(selected.get("reps", [])) if selected else 0
    return len(sets), reps, default_reps


def coverage_summary(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    reportable = [
        row
        for row in rows
        if int(row.get("set_count") or 0) > 0
        and row.get("status")
        not in {
            "rejected_quality",
            "not_evaluable_quality",
            "no_motion",
        }
    ]
    reportable_ids = {row["clip_id"] for row in reportable}
    cluster_ids = {row["cluster_id"] for row in rows}
    covered_clusters = {
        row["cluster_id"] for row in reportable if row["clip_id"] in reportable_ids
    }
    total = len(rows)
    return {
        "label": label,
        "clip_count": total,
        "cluster_count": len(cluster_ids),
        "reportable_clip_count": len(reportable),
        "reportable_cluster_count": len(covered_clusters),
        "clip_coverage": phase_states.ratio(len(reportable), total),
        "cluster_coverage": phase_states.ratio(
            len(covered_clusters), len(cluster_ids)
        ),
        "excluded_by_quality_clip_count": sum(
            bool(row.get("excluded_by_quality")) for row in rows
        ),
        "excluded_by_motion_clip_count": sum(
            bool(row.get("excluded_by_motion")) for row in rows
        ),
        "no_motion_clip_count": sum(
            row.get("status") == "no_motion" for row in rows
        ),
        "no_confident_set_clip_count": sum(
            row.get("status") == "no_confident_set" for row in rows
        ),
    }


def chain_events(
    analysis: dict[str, Any], mode: str
) -> list[dict[str, Any]]:
    if mode == "full":
        return list(
            (analysis.get("diagnostics") or {}).get("selected_top_events") or []
        )
    if mode == "complete_only":
        selected = default_set(analysis)
        if not selected:
            return []
        return [
            {
                "frame": (
                    int(rep["top"]["anchor_frame"])
                    if rep.get("top")
                    else int(round(0.5 * (rep["start_frame"] + rep["end_frame"])))
                ),
                "confidence": rep["confidence"],
                "set_ids": [selected["set_id"]],
            }
            for rep in selected.get("reps", [])
            if rep.get("status") == "complete"
        ]
    raise ValueError(f"unknown chain event mode: {mode}")


def build_chain_event_rows(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for clip in clips:
        analysis = analyses[clip["clip_id"]]
        evaluation = event_metrics(
            gt_top_frames(clip),
            chain_events(analysis, mode),
            high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        )
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "cluster_id": clip["cluster_id"],
                **evaluation,
                "_abs_errors": evaluation.pop("abs_errors"),
            }
        )
    return rows


def analyze_state_only(
    clips: list[dict[str, Any]],
    *,
    analysis_dir: Path,
    segmenter_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    clip_rows = []
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    anchor_policy = segmenter_config["core"]["anchor"]["policy"]
    detector = segmenter_config["core"].get("detector")
    actions = segmenter_config["core"]["actions"]
    for clip in clips:
        clip_dir = analysis_dir / clip["clip_id"]
        legacy = load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        action = actions.get(clip["action"])
        if action is None:
            raise ValueError(f"missing action config for {clip['action']}")
        prior_signal = (legacy.get("segmentation") or {}).get("phase_angle")
        phase_signal = phase_states.resolve_phase_signal(
            signals,
            float(clip["fps"]),
            action,
            phase_policy,
            prior_signal=prior_signal,
            detector=detector,
        )
        if phase_signal is None:
            events = []
        else:
            states = phase_states.detect_phase_states(
                signals[phase_signal],
                float(clip["fps"]),
                action,
                phase_signal,
                anchor_policy,
                detector,
            )
            events = [
                {
                    "frame": int(state["anchors"][anchor_policy]),
                    "confidence": {},
                }
                for state in states.get("top_states") or []
            ]
        evaluation = event_metrics(
            gt_top_frames(clip),
            events,
            high_confidence_threshold=2.0,
            require_set=False,
        )
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "cluster_id": clip["cluster_id"],
                **evaluation,
                "_abs_errors": evaluation.pop("abs_errors"),
            }
        )
        clip_rows.append(
            {
                "clip_id": clip["clip_id"],
                "cluster_id": clip["cluster_id"],
                "gt_event_count": evaluation["gt_event_count"],
                "pred_event_count": evaluation["pred_event_count"],
            }
        )
    return rows, clip_rows


def load_t1_5_baseline(evaluation_dir: Path) -> dict[str, Any]:
    matches = [
        row
        for row in load_csv(evaluation_dir / "rep_matches.csv")
        if row.get("split") == "holdout"
    ]
    events = [
        row
        for row in load_csv(evaluation_dir / "rep_events.csv")
        if row.get("split") == "holdout"
    ]
    subject_actions = [
        row
        for row in load_csv(evaluation_dir / "subject_action_summary.csv")
        if row.get("split") == "holdout"
    ]
    intervals = [
        row
        for row in load_csv(evaluation_dir / "rep_intervals.csv")
        if row.get("split") == "holdout"
    ]
    by_clip_gt: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_clip_pred: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matches:
        by_clip_gt[row["clip_id"]].append(row)
    for row in events:
        by_clip_pred[row["clip_id"]].append(row)
    clip_rows = []
    for clip_id in sorted(by_clip_gt):
        gt_rows = by_clip_gt[clip_id]
        pred_rows = by_clip_pred.get(clip_id, [])
        matched_gt = {
            int(row["gt_event_index"])
            for row in gt_rows
            if row["match_status"] == "matched"
        }
        matched_pred = {
            int(row["pred_event_index"])
            for row in gt_rows
            if row["match_status"] == "matched"
        }
        tp = len(matched_gt)
        clip_rows.append(
            {
                "clip_id": clip_id,
                "cluster_id": f"subject_action:{gt_rows[0]['subject_id']}:{gt_rows[0]['exercise']}",
                "gt_event_count": len(gt_rows),
                "pred_event_count": len(pred_rows),
                "tp": tp,
                "fp": len(pred_rows) - tp,
                "fn": len(gt_rows) - tp,
                "first_gt_matched": 1 in matched_gt,
                "last_gt_matched": len(gt_rows) in matched_gt,
                "high_confidence_event_fp": sum(
                    row.get("high_confidence_false_positive") == "True"
                    for row in pred_rows
                ),
                "_abs_errors": [
                    int(row["abs_error_frames"])
                    for row in gt_rows
                    if row["match_status"] == "matched"
                    and row.get("abs_error_frames")
                ],
            }
        )
    aggregate = aggregate_event_rows(clip_rows)
    consensus_errors = [
        int(row["consensus_abs_count_error"]) for row in subject_actions
    ]
    exact = sum(row["consensus_count_exact"] == "True" for row in subject_actions)
    interval_errors = [
        abs(int(row["interval_error_frames"]))
        for row in intervals
        if row.get("interval_error_frames")
    ]
    primary_matches = [
        row for row in matches if row.get("camera_role") == "primary_closest_to_side"
    ]
    primary_errors = [
        int(row["abs_error_frames"])
        for row in primary_matches
        if row["match_status"] == "matched" and row.get("abs_error_frames")
    ]
    primary_events = [
        row for row in events if row.get("camera_role") == "primary_closest_to_side"
    ]
    return {
        "all_cameras": aggregate,
        "primary_camera": {
            "clip_count": len(primary_matches),
            "gt_event_count": len(primary_matches),
            "pred_event_count": len(primary_events),
            "tp": sum(
                row["match_status"] == "matched" for row in primary_matches
            ),
            "fp": sum(
                row["match_status"] == "false_positive" for row in primary_events
            ),
            "event_abs_error_median_frames": phase_states.percentile(
                primary_errors, 50
            ),
            "event_abs_error_p90_frames": phase_states.percentile(
                primary_errors, 90
            ),
            "high_confidence_event_fp": sum(
                row.get("high_confidence_false_positive") == "True"
                for row in primary_events
            ),
        },
        "count_consensus": {
            "cluster_count": len(subject_actions),
            "count_mae": round(float(np.mean(consensus_errors)), 6)
            if consensus_errors
            else None,
            "exact_count_correct": exact,
        },
        "top_to_top_interval_median_frames": phase_states.percentile(
            interval_errors, 50
        ),
    }


def build_confidence_audit(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    core: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = confidence.build_fit3d_rep_rows(clips, analyses)
    confidence.apply_calibration(rows, core)
    targets = confidence.target_rows(rows)
    raw_metrics = confidence.binary_calibration_metrics(
        targets, score_key="raw_combined"
    )
    calibrated_metrics = confidence.binary_calibration_metrics(
        targets, score_key="calibrated_combined"
    )
    raw_threshold = confidence.threshold_metrics(
        targets, score_key="raw_combined", threshold=HIGH_CONFIDENCE_THRESHOLD
    )
    calibrated_threshold = confidence.threshold_metrics(
        targets,
        score_key="calibrated_combined",
        threshold=HIGH_CONFIDENCE_THRESHOLD,
    )
    by_clip: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "default_complete_rep_count": 0,
            "high_confidence_rep_count": 0,
            "high_confidence_rep_tp": 0,
            "high_confidence_rep_fp": 0,
        }
    )
    for row in targets:
        bucket = by_clip[row["clip_id"]]
        bucket["default_complete_rep_count"] += 1
        if row["high_confidence"]:
            bucket["high_confidence_rep_count"] += 1
            if row["is_tp"]:
                bucket["high_confidence_rep_tp"] += 1
            else:
                bucket["high_confidence_rep_fp"] += 1
    return rows, {
        "raw_combined": raw_metrics,
        "calibrated_combined": calibrated_metrics,
        "raw_threshold": raw_threshold,
        "calibrated_threshold": calibrated_threshold,
        "evaluation_unit_count": len(targets),
        "by_clip": dict(by_clip),
    }


def primary_clip_ids(
    clips: list[dict[str, Any]], fit3d_manifest: dict[str, Any]
) -> dict[str, str]:
    metadata = {
        clip["clip_id"]: clip for clip in fit3d_manifest.get("clips", [])
    }
    return phase_states.choose_primary_clip_ids(clips, metadata)


def build_clip_metrics(
    clips: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    raw_analyses: dict[str, dict[str, Any]],
    *,
    primary_by_cluster: dict[str, str],
    confidence_summary: dict[str, Any],
    raw_confidence_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    row_by_clip = {row["clip_id"]: row for row in rows}
    raw_row_by_clip = {row["clip_id"]: row for row in raw_rows}
    output = []
    for clip in clips:
        row = row_by_clip[clip["clip_id"]]
        confidence_row = confidence_summary["by_clip"].get(
            clip["clip_id"],
            {
                "default_complete_rep_count": 0,
                "high_confidence_rep_count": 0,
                "high_confidence_rep_tp": 0,
                "high_confidence_rep_fp": 0,
            },
        )
        raw_confidence_row = raw_confidence_summary["by_clip"].get(
            clip["clip_id"],
            {
                "default_complete_rep_count": 0,
                "high_confidence_rep_count": 0,
                "high_confidence_rep_tp": 0,
                "high_confidence_rep_fp": 0,
            },
        )
        raw_row = raw_row_by_clip[clip["clip_id"]]
        output.append(
            {
                "dataset": FIT3D_DATASET,
                "clip_id": clip["clip_id"],
                "cluster_id": clip["cluster_id"],
                "subject_id": clip.get("subject_id"),
                "camera_id": clip.get("camera_id"),
                "action": clip.get("action"),
                "view": clip.get("view"),
                "quality_status": row.get("quality_status"),
                "quality_source": row.get("quality_source"),
                "quality_action": row.get("quality_action"),
                "quality_reasons": row.get("quality_reasons"),
                "excluded_by_quality": row.get("excluded_by_quality"),
                "excluded_by_motion": row.get("excluded_by_motion"),
                "motion_status": row.get("motion_status"),
                "is_primary_camera": (
                    primary_by_cluster.get(clip["cluster_id"]) == clip["clip_id"]
                ),
                "analysis_status": row["status"],
                "raw_analysis_status": raw_row["status"],
                "raw_motion_status": raw_row["motion_status"],
                "set_count": row["set_count"],
                "raw_set_count": raw_row["set_count"],
                "rep_count": row["rep_count"],
                "raw_rep_count": raw_row["rep_count"],
                "default_set_rep_count": row["default_set_rep_count"],
                "raw_default_set_rep_count": raw_row["default_set_rep_count"],
                "complete_rep_count": row["complete_rep_count"],
                "raw_complete_rep_count": raw_row["complete_rep_count"],
                "gt_event_count": row["gt_event_count"],
                "pred_event_count": row["pred_event_count"],
                "count_error": row["count_error"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "event_recall": row["event_recall"],
                "event_precision": row["event_precision"],
                "first_gt_matched": row["first_gt_matched"],
                "last_gt_matched": row["last_gt_matched"],
                "event_abs_error_median_frames": row[
                    "event_abs_error_median_frames"
                ],
                "event_abs_error_p90_frames": row[
                    "event_abs_error_p90_frames"
                ],
                "high_confidence_event_fp": row["high_confidence_fp"],
                "raw_pred_event_count": raw_row["pred_event_count"],
                "raw_tp": raw_row["tp"],
                "raw_fp": raw_row["fp"],
                "raw_fn": raw_row["fn"],
                "outside_set_high_confidence_fp": row[
                    "outside_set_high_confidence_fp"
                ],
                "raw_default_complete_rep_count": raw_confidence_row[
                    "default_complete_rep_count"
                ],
                "raw_high_confidence_rep_tp": raw_confidence_row[
                    "high_confidence_rep_tp"
                ],
                "raw_high_confidence_rep_fp": raw_confidence_row[
                    "high_confidence_rep_fp"
                ],
                **confidence_row,
            }
        )
    return output


def build_event_match_rows(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for clip in clips:
        analysis = analyses[clip["clip_id"]]
        events = (analysis.get("diagnostics") or {}).get("selected_top_events") or []
        events = sorted(events, key=lambda item: int(item["frame"]))
        gt_frames = gt_top_frames(clip)
        pred_frames = [int(event["frame"]) for event in events]
        matches = phase_states.match_events(
            gt_frames,
            pred_frames,
            phase_states.event_gate_frames(gt_frames),
        )
        matched_gt = {gt_index for gt_index, _, _ in matches}
        matched_pred = {pred_index for _, pred_index, _ in matches}
        match_by_gt = {
            gt_index: (pred_index, error)
            for gt_index, pred_index, error in matches
        }
        for gt_index, gt_frame in enumerate(gt_frames):
            match = match_by_gt.get(gt_index)
            if match:
                pred_index, error = match
                rows.append(
                    {
                        "clip_id": clip["clip_id"],
                        "cluster_id": clip["cluster_id"],
                        "action": clip.get("action"),
                        "view": clip.get("view"),
                        "camera_id": clip.get("camera_id"),
                        "is_primary_camera": False,
                        "gt_event_index": gt_index + 1,
                        "gt_frame": gt_frame,
                        "pred_event_index": pred_index + 1,
                        "pred_frame": pred_frames[pred_index],
                        "match_status": "matched",
                        "signed_error_frames": pred_frames[pred_index] - gt_frame,
                        "abs_error_frames": error,
                        "high_confidence_false_positive": False,
                    }
                )
            else:
                rows.append(
                    {
                        "clip_id": clip["clip_id"],
                        "cluster_id": clip["cluster_id"],
                        "action": clip.get("action"),
                        "view": clip.get("view"),
                        "camera_id": clip.get("camera_id"),
                        "is_primary_camera": False,
                        "gt_event_index": gt_index + 1,
                        "gt_frame": gt_frame,
                        "pred_event_index": None,
                        "pred_frame": None,
                        "match_status": "false_negative",
                        "signed_error_frames": None,
                        "abs_error_frames": None,
                        "high_confidence_false_positive": False,
                    }
                )
        for pred_index, event in enumerate(events):
            if pred_index in matched_pred:
                continue
            score = float((event.get("confidence") or {}).get("combined", 0.0))
            rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "cluster_id": clip["cluster_id"],
                    "action": clip.get("action"),
                    "view": clip.get("view"),
                    "camera_id": clip.get("camera_id"),
                    "is_primary_camera": False,
                    "gt_event_index": None,
                    "gt_frame": None,
                    "pred_event_index": pred_index + 1,
                    "pred_frame": pred_frames[pred_index],
                    "match_status": "false_positive",
                    "signed_error_frames": None,
                    "abs_error_frames": None,
                    "high_confidence_false_positive": (
                        score >= HIGH_CONFIDENCE_THRESHOLD
                    ),
                }
            )
    return rows


def replace_or_resample(values: np.ndarray, target_count: int, fps: float, target_fps: float) -> np.ndarray:
    source_count = len(values)
    if source_count == 0 or target_count <= 0:
        return np.full(target_count, np.nan, dtype=np.float64)
    x_old = np.arange(source_count, dtype=np.float64) / max(fps, 1e-6)
    x_new = np.arange(target_count, dtype=np.float64) / max(target_fps, 1e-6)
    valid = np.isfinite(values)
    if not valid.any():
        return np.full(target_count, np.nan, dtype=np.float64)
    return np.interp(x_new, x_old[valid], values[valid])


def flip_signal_names(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    flipped = {}
    for name, values in signals.items():
        if "left_" in name:
            target = name.replace("left_", "right_", 1)
        elif "right_" in name:
            target = name.replace("right_", "left_", 1)
        else:
            target = name
        flipped[target] = values
    return flipped


def transform_signals(
    signals: dict[str, np.ndarray],
    transform_id: str,
    *,
    fps: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    copied = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in signals.items()
    }
    if transform_id.startswith("amplitude_scale_"):
        scale = float(transform_id.rsplit("_", 1)[-1])
        output = {}
        for name, values in copied.items():
            center = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            output[name] = center + scale * (values - center)
        return output, fps
    if transform_id == "offset_plus_15":
        return {name: values + ROBUSTNESS_OFFSET for name, values in copied.items()}, fps
    if transform_id == "flip_lr":
        return flip_signal_names(copied), fps
    if transform_id.startswith("fps_"):
        target_fps = float(transform_id.split("_", 1)[1])
        target_count = max(2, int(round(len(next(iter(copied.values()))) * target_fps / fps)))
        return {
            name: replace_or_resample(values, target_count, fps, target_fps)
            for name, values in copied.items()
        }, target_fps
    if transform_id == "noise_sigma_2":
        rng = np.random.default_rng(seed)
        return {
            name: values + rng.normal(0.0, ROBUSTNESS_NOISE_SIGMA, len(values))
            for name, values in copied.items()
        }, fps
    if transform_id == "missing_5pct":
        rng = np.random.default_rng(seed)
        output = {}
        for name, values in copied.items():
            values = values.copy()
            count = len(values)
            missing = rng.choice(
                count,
                size=max(1, int(round(count * ROBUSTNESS_MISSING_FRACTION))),
                replace=False,
            )
            values[missing] = np.nan
            gap = max(1, int(round(0.20 * fps)))
            for _ in range(2):
                start = int(rng.integers(0, max(1, count - gap)))
                values[start : start + gap] = np.nan
            output[name] = values
        return output, fps
    if transform_id == "idle_pad_2s":
        pad = int(round(ROBUSTNESS_IDLE_SECONDS * fps))
        output = {}
        for name, values in copied.items():
            finite = values[np.isfinite(values)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            output[name] = np.concatenate(
                [np.full(pad, fill), values, np.full(pad, fill)]
            )
        return output, fps
    raise ValueError(f"unknown transformation: {transform_id}")


def robustness_variants() -> list[str]:
    variants = []
    for scale in ROBUSTNESS_AMPLITUDES:
        variants.append(f"amplitude_scale_{scale:g}")
    variants.extend(["offset_plus_15", "flip_lr"])
    for fps in ROBUSTNESS_FPS_TARGETS:
        variants.append(f"fps_{fps:g}")
    variants.extend(["noise_sigma_2", "missing_5pct", "idle_pad_2s"])
    return variants


def evaluate_robustness(
    clips: list[dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
    *,
    analysis_dir: Path,
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    actions = segmenter_config["core"]["actions"]
    for clip in clips:
        clip_dir = analysis_dir / clip["clip_id"]
        legacy = load_json(clip_dir / "analysis.json")
        signals = phase_states.load_angle_signals(
            clip_dir / "angles" / "angles_wide.csv"
        )
        action = actions.get(clip["action"])
        if action is None:
            continue
        prior_signal = (legacy.get("segmentation") or {}).get("phase_angle")
        gate = quality_gates.from_legacy_quality(legacy)
        base_sets, base_reps, base_default_reps = structure_counts(
            analyses[clip["clip_id"]]
        )
        for transform_id in robustness_variants():
            transformed, transformed_fps = transform_signals(
                signals,
                transform_id,
                fps=float(clip["fps"]),
                seed=zlib.crc32(f"{clip['clip_id']}:{transform_id}".encode("utf-8")),
            )
            transformed_clip = {**clip, "fps": transformed_fps}
            transformed_analysis, _ = chain_selection.analyze_clip_signals(
                transformed_clip,
                transformed,
                action,
                prior_signal=prior_signal,
                phase_policy=phase_policy,
                chain_config=chain_config,
                quality_gate=gate,
                detector=segmenter_config["core"].get("detector"),
            )
            sets, reps, default_reps = structure_counts(transformed_analysis)
            rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "cluster_id": clip["cluster_id"],
                    "action": clip.get("action"),
                    "transform_id": transform_id,
                    "status": transformed_analysis["status"],
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
                }
            )
    return rows


def summarize_robustness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["transform_id"]].append(row)
    output = []
    for transform_id, transform_rows in sorted(grouped.items()):
        eligible = [
            row for row in transform_rows if not row.get("quality_excluded")
        ]
        output.append(
            {
                "transform_id": transform_id,
                "input_clip_count": len(transform_rows),
                "clip_count": len(eligible),
                "excluded_by_quality_clip_count": (
                    len(transform_rows) - len(eligible)
                ),
                "structure_match_count": sum(
                    bool(row["structure_match"]) for row in eligible
                ),
                "structure_match_rate": phase_states.ratio(
                    sum(bool(row["structure_match"]) for row in eligible),
                    len(eligible),
                ),
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
                "changed_structure_count": sum(
                    not bool(row["structure_match"]) for row in eligible
                ),
            }
        )
    return output


def negative_control_series(control_id: str, *, fps: float, seconds: float) -> np.ndarray:
    count = int(round(fps * seconds))
    times = np.arange(count, dtype=np.float64) / fps
    baseline = 120.0
    rng = np.random.default_rng(zlib.crc32(control_id.encode("utf-8")))
    if control_id == "idle_constant":
        return baseline + rng.normal(0.0, 0.3, count)
    if control_id == "setup_ramp":
        ramp = np.clip(times / max(seconds, 1e-6), 0.0, 1.0) * 20.0
        return baseline + ramp
    if control_id == "single_cycle":
        active = (times >= 2.0) & (times <= 5.0)
        values = np.full(count, baseline)
        local = times[active] - 2.0
        values[active] = baseline + 35.0 * (0.5 - 0.5 * np.cos(2.0 * np.pi * local / 3.0))
        return values
    if control_id == "half_rep":
        active = (times >= 2.0) & (times <= 3.5)
        values = np.full(count, baseline)
        local = times[active] - 2.0
        values[active] = baseline + 35.0 * (0.5 - 0.5 * np.cos(np.pi * local / 1.5))
        return values
    if control_id == "two_short_cycles":
        active = (times >= 2.0) & (times <= 6.0)
        values = np.full(count, baseline)
        local = times[active] - 2.0
        values[active] = baseline + 30.0 * (0.5 - 0.5 * np.cos(2.0 * np.pi * local / 2.0))
        return values
    raise ValueError(f"unknown negative control: {control_id}")


def evaluate_negative_controls(
    *,
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
) -> list[dict[str, Any]]:
    action = segmenter_config["core"]["actions"]["pushup"]
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    rows = []
    for control_id in (
        "idle_constant",
        "setup_ramp",
        "single_cycle",
        "half_rep",
        "two_short_cycles",
    ):
        values = negative_control_series(control_id, fps=30.0, seconds=10.0)
        clip = {
            "clip_id": f"synthetic_{control_id}",
            "fps": 30.0,
        }
        analysis, _ = chain_selection.analyze_clip_signals(
            clip,
            {
                "left_elbow_flexion": values,
                "right_elbow_flexion": values,
            },
            action,
            prior_signal=None,
            phase_policy=phase_policy,
            chain_config=chain_config,
            detector=segmenter_config["core"].get("detector"),
        )
        sets, reps, default_reps = structure_counts(analysis)
        complete_reps = sum(
            rep.get("status") == "complete"
            for set_candidate in analysis.get("sets", [])
            for rep in set_candidate.get("reps", [])
        )
        rows.append(
            {
                "control_id": control_id,
                "status": analysis["status"],
                "set_count": sets,
                "rep_count": reps,
                "default_rep_count": default_reps,
                "complete_rep_count": complete_reps,
                "expected_zero_complete_rep": True,
                "passed": complete_reps == 0 and sets == 0,
            }
        )
    return rows


def load_sealed_annotations(template_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not template_path.is_file():
        return {}
    for row in load_csv(template_path):
        review_status = str(row.get("review_status", "")).strip().lower()
        if review_status not in {"accept", "adjust", "add"}:
            continue
        start = as_int(row.get("gt_start_frame"))
        end = as_int(row.get("gt_end_frame"))
        if start is None or end is None or end < start:
            continue
        grouped[row["segment_id"]].append(
            {
                "start_frame": start,
                "end_frame": end,
                "review_status": review_status,
                "boundary_confidence": row.get("boundary_confidence"),
            }
        )
    return dict(grouped)


def evaluate_sealed_holdout(
    clips: list[dict[str, Any]],
    *,
    segmenter_config: dict[str, Any],
    chain_config: dict[str, Any],
    confidence_core: dict[str, Any],
    input_dir: Path,
    runs_dir: Path,
    template_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = load_sealed_annotations(template_path)
    actions = segmenter_config["core"]["actions"]
    phase_policy = segmenter_config["core"]["phase_signal"]["policy"]
    output_dir = runs_dir / "qevd_sealed_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = runs_dir / "qevd_sealed_holdout_raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    selection_roles = {
        row.get("segment_id"): row.get("selection_role")
        for row in load_csv(
            ROOT
            / "results"
            / "task1_6"
            / "manifest"
            / "qevd_sealed_holdout_segments.csv"
        )
    }
    detector = segmenter_config["core"].get("detector")
    rows = []
    raw_analyses = {}
    analyses = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        action = actions.get(clip.get("action"))
        input_analysis = input_dir / clip_id / "analysis.json"
        legacy = None
        if input_analysis.is_file():
            legacy = load_json(input_analysis)
        gate = quality_gates.from_legacy_quality(legacy)
        status = "ok"
        if action is None:
            status = "unsupported_action_config"
        elif not input_analysis.is_file():
            status = "preprocessed_input_missing"
        processing_status = status
        if status == "ok":
            signals = phase_states.load_angle_signals(
                input_dir / clip_id / "angles" / "angles_wide.csv"
            )
            analysis, _ = chain_selection.analyze_clip_signals(
                clip,
                signals,
                action,
                prior_signal=(legacy.get("segmentation") or {}).get("phase_angle"),
                phase_policy=phase_policy,
                chain_config=chain_config,
                legacy_analysis=legacy,
                allow_legacy_fallback=True,
                detector=detector,
            )
            gated_analysis = quality_gates.apply_quality_gates(
                analysis, [gate]
            )
            raw_analyses[clip_id] = analysis
            analyses[clip_id] = gated_analysis
            write_json(
                raw_output_dir / clip_id / "analysis_v2.json", analysis
            )
            write_json(
                output_dir / clip_id / "analysis_v2.json", gated_analysis
            )
            status = gated_analysis["status"]
        rows.append(
            {
                "dataset": QEVD_DATASET,
                "clip_id": clip_id,
                "video_id": clip.get("video_id"),
                "action": clip.get("action"),
                "selection_role": selection_roles.get(clip_id),
                "processing_status": processing_status,
                "status": status,
                "quality_status": (
                    gate["status"] if processing_status == "ok" else "not_applicable"
                ),
                "quality_source": gate["source"],
                "quality_action": (
                    "blocked"
                    if processing_status == "ok" and gate["blocking"]
                    else "reported"
                    if processing_status == "ok"
                    else "not_applicable"
                ),
                "quality_reasons": "|".join(gate["reasons"]),
                "excluded_by_quality": bool(
                    processing_status == "ok" and gate["blocking"]
                ),
                "excluded_by_motion": bool(
                    processing_status == "ok"
                    and not gate["blocking"]
                    and analyses[clip_id].get("status") == "no_motion"
                ),
                "raw_analysis_status": (
                    raw_analyses[clip_id]["status"]
                    if clip_id in raw_analyses
                    else None
                ),
                "motion_status": (
                    analyses[clip_id].get("motion_status")
                    if clip_id in analyses
                    else None
                ),
                "raw_motion_status": (
                    raw_analyses[clip_id].get("motion_status")
                    if clip_id in raw_analyses
                    else None
                ),
                "input_analysis_status": (
                    legacy.get("status") if legacy is not None else None
                ),
                "input_quality_status": (
                    legacy.get("quality_status") if legacy is not None else None
                ),
                "input_quality_failures": (
                    "|".join(
                        (legacy.get("quality") or {}).get("failures") or []
                    )
                    if legacy is not None
                    else None
                ),
                "annotation_rep_count": len(annotations.get(clip_id, [])),
                "gt_evaluable": bool(annotations.get(clip_id)),
            }
        )
    def build_confidence_rows(
        analysis_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not analysis_map:
            return []
        result = confidence.build_qevd_rep_rows(
            [clip for clip in clips if clip["clip_id"] in analysis_map],
            analysis_map,
        )
        confidence.apply_calibration(result, confidence_core)
        return result

    confidence_rows = build_confidence_rows(analyses)
    raw_confidence_rows = build_confidence_rows(raw_analyses)

    def structure_bucket(
        analysis_map: dict[str, dict[str, Any]],
        confidence_rows_for_map: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        by_clip: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "set_count": 0,
                "rep_count": 0,
                "complete_rep_count": 0,
                "endpoint_partial_count": 0,
                "high_confidence_complete_rep_count": 0,
            }
        )
        for confidence_row in confidence_rows_for_map:
            bucket = by_clip[confidence_row["clip_id"]]
            if confidence_row["is_default_set"]:
                if confidence_row["unit_eligible"]:
                    bucket["complete_rep_count"] += 1
                    if confidence_row["high_confidence"]:
                        bucket["high_confidence_complete_rep_count"] += 1
                elif confidence_row["rep_status"] == "endpoint_partial":
                    bucket["endpoint_partial_count"] += 1
        for clip_id, analysis in analysis_map.items():
            sets, reps, _ = structure_counts(analysis)
            by_clip[clip_id]["set_count"] = sets
            by_clip[clip_id]["rep_count"] = reps
        return dict(by_clip)

    by_clip = structure_bucket(analyses, confidence_rows)
    raw_by_clip = structure_bucket(raw_analyses, raw_confidence_rows)
    for row in rows:
        row.update(by_clip.get(row["clip_id"], {}))
        raw_bucket = raw_by_clip.get(row["clip_id"], {})
        row.update(
            {
                f"raw_{key}": value
                for key, value in raw_bucket.items()
            }
        )
        row["quality_gate_conflict"] = bool(
            row.get("input_quality_status") == "reject"
            and int(row.get("high_confidence_complete_rep_count") or 0) > 0
        )
        row["raw_quality_gate_conflict"] = bool(
            row.get("input_quality_status") == "reject"
            and int(row.get("raw_high_confidence_complete_rep_count") or 0) > 0
        )
    annotation_ready_count = sum(bool(annotations.get(clip["clip_id"])) for clip in clips)
    runnable_count = sum(row["processing_status"] == "ok" for row in rows)
    raw_reportable_count = sum(int(row.get("raw_set_count") or 0) > 0 for row in rows)
    reportable_count = sum(
        int(row.get("set_count") or 0) > 0
        and not row.get("excluded_by_quality")
        and not row.get("excluded_by_motion")
        for row in rows
    )
    formal_evaluable = annotation_ready_count == len(clips) and runnable_count == len(clips)
    summary = {
        "segment_count": len(clips),
        "annotation_ready_count": annotation_ready_count,
        "runnable_segment_count": runnable_count,
        "raw_reportable_segment_count": raw_reportable_count,
        "reportable_segment_count": reportable_count,
        "reportable_coverage_of_runnable": phase_states.ratio(
            reportable_count, runnable_count
        ),
        "reportable_coverage_of_all_segments": phase_states.ratio(
            reportable_count, len(clips)
        ),
        "excluded_by_quality_count": sum(
            bool(row.get("excluded_by_quality")) for row in rows
        ),
        "excluded_by_motion_count": sum(
            bool(row.get("excluded_by_motion")) for row in rows
        ),
        "quality_status_counts": dict(
            Counter(str(row.get("quality_status")) for row in rows)
        ),
        "raw_gated": {
            "raw": {
                "reportable_segment_count": raw_reportable_count,
                "high_confidence_complete_rep_count": sum(
                    int(row.get("raw_high_confidence_complete_rep_count") or 0)
                    for row in rows
                ),
            },
            "gated": {
                "reportable_segment_count": reportable_count,
                "high_confidence_complete_rep_count": sum(
                    int(row.get("high_confidence_complete_rep_count") or 0)
                    for row in rows
                ),
            },
        },
        "unsupported_action_count": sum(
            row["processing_status"] == "unsupported_action_config" for row in rows
        ),
        "missing_input_count": sum(
            row["processing_status"] == "preprocessed_input_missing" for row in rows
        ),
        "quality_gate_conflict_count": sum(
            bool(row.get("quality_gate_conflict")) for row in rows
        ),
        "raw_quality_gate_conflict_count": sum(
            bool(row.get("raw_quality_gate_conflict")) for row in rows
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path)
        if manifest_path.is_file()
        else None,
        "template": str(template_path),
        "template_sha256": sha256_path(template_path)
        if template_path.is_file()
        else None,
        "formal_stage_gate_evaluable": formal_evaluable,
        "stage_passed": None if not formal_evaluable else False,
        "metrics": None,
        "limitation": (
            "Sealed GT is incomplete; pushup/squat predictions can be inspected "
            "but no count/boundary/F1/high-confidence-FP gate is computed. "
            "air_jump_rope and good_morning also lack frozen action configs."
        ),
    }
    return rows, summary


def gate_row(
    gate_id: str,
    observed: Any,
    target: str,
    passed: bool | None,
    evidence: str,
    dataset: str = FIT3D_DATASET,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "gate_id": gate_id,
        "observed": observed,
        "target": target,
        "status": "pass"
        if passed is True
        else "fail"
        if passed is False
        else "not_evaluable",
        "passed": passed,
        "evidence": evidence,
    }


def build_go_no_go(
    *,
    all_cameras: dict[str, Any],
    primary: dict[str, Any],
    consensus: dict[str, Any],
    confidence_summary: dict[str, Any],
    top_interval_median: float | None,
    sealed_summary: dict[str, Any],
    coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    required_exact = int(
        math.ceil(
            EXACT_COUNT_NUMERATOR
            / EXACT_COUNT_DENOMINATOR
            * consensus["cluster_count"]
        )
    )
    calibrated_fp = confidence_summary["calibrated_threshold"][
        "high_confidence_fp"
    ]
    gates = [
        gate_row(
            "fit3d_count_mae_lte_0_5",
            consensus["count_mae"],
            "<= 0.5",
            bool(
                consensus["count_mae"] is not None
                and consensus["count_mae"] <= COUNT_MAE_MAX
            ),
            "cluster-level median event count across four cameras",
        ),
        gate_row(
            "fit3d_exact_count",
            f"{consensus['exact_count_correct']}/{consensus['cluster_count']}",
            f">= {required_exact}/{consensus['cluster_count']}",
            consensus["exact_count_correct"] >= required_exact,
            "14/16 proportional holdout target",
        ),
        gate_row(
            "fit3d_first_event_recall",
            all_cameras["first_event_recall"],
            ">= 0.75",
            bool(
                all_cameras["first_event_recall"] is not None
                and all_cameras["first_event_recall"] >= FIRST_LAST_RECALL_MIN
            ),
            "all-camera clip-level first GT top matched",
        ),
        gate_row(
            "fit3d_last_event_recall",
            all_cameras["last_event_recall"],
            ">= 0.75",
            bool(
                all_cameras["last_event_recall"] is not None
                and all_cameras["last_event_recall"] >= FIRST_LAST_RECALL_MIN
            ),
            "all-camera clip-level last GT top matched",
        ),
        gate_row(
            "fit3d_primary_event_median",
            primary["event_abs_error_median_frames"],
            "<= 5 frames",
            bool(
                primary["event_abs_error_median_frames"] is not None
                and primary["event_abs_error_median_frames"]
                <= PRIMARY_EVENT_MEDIAN_MAX
            ),
            "primary camera per subject-action",
        ),
        gate_row(
            "fit3d_primary_event_p90",
            primary["event_abs_error_p90_frames"],
            "<= 12 frames",
            bool(
                primary["event_abs_error_p90_frames"] is not None
                and primary["event_abs_error_p90_frames"]
                <= PRIMARY_EVENT_P90_MAX
            ),
            "primary camera per subject-action",
        ),
        gate_row(
            "fit3d_primary_high_confidence_fp",
            primary["high_confidence_event_fp"],
            "= 0",
            primary["high_confidence_event_fp"] == 0,
            "event-level diagnostic, raw combined >= 0.5",
        ),
        gate_row(
            "fit3d_all_camera_high_confidence_fp",
            all_cameras["high_confidence_event_fp"],
            "= 0",
            all_cameras["high_confidence_event_fp"] == 0,
            "event-level diagnostic, raw combined >= 0.5",
        ),
        gate_row(
            "fit3d_complete_rep_high_confidence_fp",
            calibrated_fp,
            "= 0",
            calibrated_fp == 0,
            "T1-6D calibrated default complete rep",
        ),
        gate_row(
            "fit3d_top_to_top_interval_not_degraded",
            top_interval_median,
            f"<= {TOP_INTERVAL_MEDIAN_BASELINE} frames",
            bool(
                top_interval_median is not None
                and top_interval_median <= TOP_INTERVAL_MEDIAN_BASELINE
            ),
            "matched consecutive top-to-top interval error",
        ),
    ]
    if coverage is not None:
        gates.append(
            gate_row(
                "fit3d_gated_clip_coverage",
                coverage.get("clip_coverage"),
                ">= 0.95",
                bool(
                    coverage.get("clip_coverage") is not None
                    and coverage["clip_coverage"] >= 0.95
                ),
                "reportable clip coverage after quality and motion gates",
            )
        )
    qevd_targets = {
        "qevd_count_exact": ">= 5/6 segments",
        "qevd_count_mae": "<= 0.25",
        "qevd_boundary_median": "<= 5 frames",
        "qevd_boundary_p95": "<= 12 frames",
        "qevd_f1_iou_0_30": ">= 0.95",
        "qevd_high_confidence_fp": "= 0",
        "qevd_non_rep_as_complete_rep": "= 0",
    }
    for gate_id, target in qevd_targets.items():
        gates.append(
            gate_row(
                gate_id,
                None,
                target,
                None,
                (
                    "sealed GT pending; formal cross-domain metric unavailable"
                    if sealed_summary["annotation_ready_count"] == 0
                    else "sealed evaluation incomplete"
                ),
                dataset=QEVD_DATASET,
            )
        )
    return gates


def build_ablation_rows(
    *,
    baseline: dict[str, Any],
    state_only_rows: list[dict[str, Any]],
    state_only_clip_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    complete_only_rows: list[dict[str, Any]],
    confidence_summary: dict[str, Any],
    anchor_dev_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_only = aggregate_event_rows(state_only_rows)
    state_consensus = count_consensus(state_only_clip_rows)
    full = aggregate_event_rows(full_rows)
    full_consensus = count_consensus(full_rows)
    complete_only = aggregate_event_rows(complete_only_rows)
    complete_consensus = count_consensus(complete_only_rows)
    rows = [
        {
            "variant_id": "A_legacy_t1_5d",
            "scope": "fit3d_holdout",
            "state_source": "t1_4_legacy",
            "chain_selection": False,
            "endpoint_partial": False,
            "anchor_policy": "legacy",
            "confidence_calibration": False,
            "event_recall": baseline["all_cameras"]["event_recall"],
            "event_precision": baseline["all_cameras"]["event_precision"],
            "first_event_recall": baseline["all_cameras"]["first_event_recall"],
            "last_event_recall": baseline["all_cameras"]["last_event_recall"],
            "event_median_frames": baseline["all_cameras"][
                "event_abs_error_median_frames"
            ],
            "event_p90_frames": baseline["all_cameras"][
                "event_abs_error_p90_frames"
            ],
            "count_mae": baseline["count_consensus"]["count_mae"],
            "count_exact_correct": baseline["count_consensus"][
                "exact_count_correct"
            ],
            "count_cluster_count": baseline["count_consensus"]["cluster_count"],
            "high_confidence_fp": baseline["all_cameras"][
                "high_confidence_event_fp"
            ],
            "high_confidence_fp_unit": "event",
            "notes": (
                "T1-5D frozen predictions re-evaluated with corrected "
                "in-gate maximum matching"
            ),
        },
        {
            "variant_id": "B_states_only",
            "scope": "fit3d_holdout",
            "state_source": "t1_6_phase_states",
            "chain_selection": False,
            "endpoint_partial": False,
            "anchor_policy": "plateau_midpoint",
            "confidence_calibration": False,
            "event_recall": state_only["event_recall"],
            "event_precision": state_only["event_precision"],
            "first_event_recall": state_only["first_event_recall"],
            "last_event_recall": state_only["last_event_recall"],
            "event_median_frames": state_only["event_abs_error_median_frames"],
            "event_p90_frames": state_only["event_abs_error_p90_frames"],
            "count_mae": state_consensus["count_mae"],
            "count_exact_correct": state_consensus["exact_count_correct"],
            "count_cluster_count": state_consensus["cluster_count"],
            "high_confidence_fp": None,
            "high_confidence_fp_unit": "not_defined",
            "notes": "all detected top states, no chain or endpoint policy",
        },
        {
            "variant_id": "C_chain_without_endpoint",
            "scope": "fit3d_holdout",
            "state_source": "t1_6_phase_states",
            "chain_selection": True,
            "endpoint_partial": False,
            "anchor_policy": "plateau_midpoint",
            "confidence_calibration": False,
            "event_recall": complete_only["event_recall"],
            "event_precision": complete_only["event_precision"],
            "first_event_recall": complete_only["first_event_recall"],
            "last_event_recall": complete_only["last_event_recall"],
            "event_median_frames": complete_only[
                "event_abs_error_median_frames"
            ],
            "event_p90_frames": complete_only["event_abs_error_p90_frames"],
            "count_mae": complete_consensus["count_mae"],
            "count_exact_correct": complete_consensus["exact_count_correct"],
            "count_cluster_count": complete_consensus["cluster_count"],
            "high_confidence_fp": None,
            "high_confidence_fp_unit": "not_defined",
            "notes": "Viterbi chain events from status=complete reps only",
        },
        {
            "variant_id": "D_chain_full_raw",
            "scope": "fit3d_holdout",
            "state_source": "t1_6_phase_states",
            "chain_selection": True,
            "endpoint_partial": True,
            "anchor_policy": "plateau_midpoint_with_terminal_early",
            "confidence_calibration": False,
            "event_recall": full["event_recall"],
            "event_precision": full["event_precision"],
            "first_event_recall": full["first_event_recall"],
            "last_event_recall": full["last_event_recall"],
            "event_median_frames": full["event_abs_error_median_frames"],
            "event_p90_frames": full["event_abs_error_p90_frames"],
            "count_mae": full_consensus["count_mae"],
            "count_exact_correct": full_consensus["exact_count_correct"],
            "count_cluster_count": full_consensus["cluster_count"],
            "high_confidence_fp": full["high_confidence_event_fp"],
            "high_confidence_fp_unit": "event",
            "notes": (
                "current chain diagnostics with raw event confidence; "
                "terminal event uses earlier of midpoint and zero crossing"
            ),
        },
        {
            "variant_id": "E_chain_full_calibrated",
            "scope": "fit3d_holdout",
            "state_source": "t1_6_phase_states",
            "chain_selection": True,
            "endpoint_partial": True,
            "anchor_policy": "plateau_midpoint_with_terminal_early",
            "confidence_calibration": True,
            "event_recall": full["event_recall"],
            "event_precision": full["event_precision"],
            "first_event_recall": full["first_event_recall"],
            "last_event_recall": full["last_event_recall"],
            "event_median_frames": full["event_abs_error_median_frames"],
            "event_p90_frames": full["event_abs_error_p90_frames"],
            "count_mae": full_consensus["count_mae"],
            "count_exact_correct": full_consensus["exact_count_correct"],
            "count_cluster_count": full_consensus["cluster_count"],
            "high_confidence_fp": confidence_summary["calibrated_threshold"][
                "high_confidence_fp"
            ],
            "high_confidence_fp_unit": "complete_rep",
            "notes": (
                "T1-6D calibrated complete-rep acceptance with terminal "
                "event anchor enabled"
            ),
        },
    ]
    for anchor_row in anchor_dev_rows:
        all_cameras = anchor_row["all_cameras"]
        rows.append(
            {
                "variant_id": f"F_anchor_{anchor_row['anchor_policy']}_dev",
                "scope": "fit3d_dev_state_only",
                "state_source": "t1_6_phase_states",
                "chain_selection": False,
                "endpoint_partial": False,
                "anchor_policy": anchor_row["anchor_policy"],
                "confidence_calibration": False,
                "event_recall": all_cameras["event_recall"],
                "event_precision": all_cameras["event_precision"],
                "first_event_recall": all_cameras["first_event_recall"],
                "last_event_recall": all_cameras["last_event_recall"],
                "event_median_frames": all_cameras[
                    "event_abs_error_median_frames"
                ],
                "event_p90_frames": all_cameras["event_abs_error_p90_frames"],
                "count_mae": None,
                "count_exact_correct": None,
                "count_cluster_count": all_cameras["cluster_count"],
                "high_confidence_fp": None,
                "high_confidence_fp_unit": "not_defined",
                "notes": "T1-6B dev anchor ablation, selected bilateral signal",
            }
        )
    return rows


def reproducibility_check(
    output_dir: Path,
    csv_names: list[str],
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_path = output_dir / "artifact_hashes.json"
    current = {
        name: sha256_path(output_dir / name)
        for name in csv_names
        if (output_dir / name).is_file()
    }
    current_record = {
        "schema_version": "task1-6e-artifact-hashes-v1",
        "scope": scope or {},
        "artifacts": current,
    }
    if not previous_path.is_file():
        result = {
            "checked": False,
            "identical_csv": None,
            "previous_artifacts": None,
            "current_artifacts": current,
            "reason": "no_previous_run",
            "scope": scope or {},
        }
    else:
        previous_payload = load_json(previous_path)
        if (
            isinstance(previous_payload, dict)
            and "artifacts" in previous_payload
            and isinstance(previous_payload.get("artifacts"), dict)
        ):
            previous = previous_payload["artifacts"]
            previous_scope = previous_payload.get("scope")
        else:
            previous = previous_payload if isinstance(previous_payload, dict) else {}
            previous_scope = None
        if scope is not None and previous_scope is None:
            result = {
                "checked": False,
                "identical_csv": None,
                "previous_artifacts": previous,
                "current_artifacts": current,
                "reason": "previous_scope_missing",
                "scope": scope,
            }
        elif scope is not None and previous_scope != scope:
            result = {
                "checked": False,
                "identical_csv": None,
                "previous_artifacts": previous,
                "current_artifacts": current,
                "reason": "scope_changed",
                "previous_scope": previous_scope,
                "scope": scope,
            }
        else:
            identical = set(previous) == set(current) and all(
                previous[name] == current[name] for name in current
            )
            result = {
                "checked": True,
                "identical_csv": identical,
                "previous_artifacts": previous,
                "current_artifacts": current,
                "reason": None if identical else "csv_hash_mismatch",
                "scope": scope or {},
                "previous_scope": previous_scope,
            }
    write_json(previous_path, current_record)
    return result


def write_outputs(
    *,
    output_dir: Path,
    tables_dir: Path,
    clip_metrics: list[dict[str, Any]],
    event_matches: list[dict[str, Any]],
    raw_event_matches: list[dict[str, Any]],
    rep_audit: list[dict[str, Any]],
    raw_rep_audit: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    robustness_summary: list[dict[str, Any]],
    negative_controls: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    go_no_go: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    clip_fields = list(clip_metrics[0]) if clip_metrics else []
    event_fields = [
        "clip_id",
        "cluster_id",
        "action",
        "view",
        "camera_id",
        "is_primary_camera",
        "gt_event_index",
        "gt_frame",
        "pred_event_index",
        "pred_frame",
        "match_status",
        "signed_error_frames",
        "abs_error_frames",
        "high_confidence_false_positive",
    ]
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
    ablation_fields = list(ablation_rows[0]) if ablation_rows else []
    robustness_fields = list(robustness_rows[0]) if robustness_rows else []
    robustness_summary_fields = (
        list(robustness_summary[0]) if robustness_summary else []
    )
    negative_fields = list(negative_controls[0]) if negative_controls else []
    sealed_fields = list(sealed_rows[0]) if sealed_rows else []
    gate_fields = [
        "dataset",
        "gate_id",
        "observed",
        "target",
        "status",
        "passed",
        "evidence",
    ]
    write_csv(output_dir / "fit3d_holdout_clip_metrics.csv", clip_metrics, clip_fields)
    write_csv(output_dir / "fit3d_holdout_event_matches.csv", event_matches, event_fields)
    write_csv(
        output_dir / "fit3d_holdout_raw_event_matches.csv",
        raw_event_matches,
        event_fields,
    )
    write_csv(output_dir / "fit3d_holdout_rep_confidence.csv", rep_audit, rep_fields)
    write_csv(
        output_dir / "fit3d_holdout_raw_rep_confidence.csv",
        raw_rep_audit,
        rep_fields,
    )
    write_csv(output_dir / "ablation.csv", ablation_rows, ablation_fields)
    write_csv(output_dir / "robustness.csv", robustness_rows, robustness_fields)
    write_csv(
        output_dir / "robustness_summary.csv",
        robustness_summary,
        robustness_summary_fields,
    )
    write_csv(
        output_dir / "negative_controls.csv",
        negative_controls,
        negative_fields,
    )
    write_csv(output_dir / "qevd_sealed_predictions.csv", sealed_rows, sealed_fields)
    write_csv(output_dir / "go_no_go.csv", go_no_go, gate_fields)
    write_csv(tables_dir / "ablation_summary.csv", ablation_rows, ablation_fields)
    write_csv(
        tables_dir / "robustness_summary.csv",
        robustness_summary,
        robustness_summary_fields,
    )
    write_csv(tables_dir / "go_no_go.csv", go_no_go, gate_fields)
    write_json(output_dir / "summary.json", summary)


def main() -> int:
    args = parse_args()
    sources = load_json(args.sources)
    fit3d_manifest = load_json(args.fit3d_manifest)
    segmenter_config = load_json(args.segmenter_config)
    chain_config = chain_selection.load_chain_config(args.chain_config)
    confidence_config = load_json(args.confidence_config)
    confidence.validate_frozen_config(confidence_config)
    fit3d_clips = load_fit3d_holdout_clips(sources)
    sealed_clips = load_qevd_sealed_clips(sources)
    primary_by_cluster = primary_clip_ids(fit3d_clips, fit3d_manifest)

    raw_rows, raw_analyses, rows, analyses = (
        chain_selection.run_fit3d_dual_evaluation(
        fit3d_clips,
        specs=load_json(args.specs),
        segmenter_config=segmenter_config,
        chain_config=chain_config,
        analysis_dir=args.fit3d_analysis_dir,
        chain_config_hash=sha256_path(args.chain_config),
        )
    )
    analyses_by_id = {analysis["clip_id"]: analysis for analysis in analyses}
    raw_analyses_by_id = {
        analysis["clip_id"]: analysis for analysis in raw_analyses
    }
    fit3d_run_dir = args.runs_dir / "fit3d_holdout"
    for analysis in analyses:
        write_json(
            fit3d_run_dir / analysis["clip_id"] / "analysis_v2.json", analysis
        )
    fit3d_raw_run_dir = args.runs_dir / "fit3d_holdout_raw"
    for analysis in raw_analyses:
        write_json(
            fit3d_raw_run_dir / analysis["clip_id"] / "analysis_v2.json",
            analysis,
        )

    raw_all_cameras = aggregate_event_rows(raw_rows)
    all_cameras = aggregate_event_rows(rows)
    primary_rows = [
        row
        for row in rows
        if primary_by_cluster.get(row["cluster_id"]) == row["clip_id"]
    ]
    raw_primary_rows = [
        row
        for row in raw_rows
        if primary_by_cluster.get(row["cluster_id"]) == row["clip_id"]
    ]
    raw_primary = aggregate_event_rows(raw_primary_rows)
    primary = aggregate_event_rows(primary_rows)
    raw_consensus = count_consensus(raw_rows)
    consensus = count_consensus(rows)
    raw_interval_errors = top_to_top_interval_errors(
        fit3d_clips, raw_analyses_by_id
    )
    interval_errors = top_to_top_interval_errors(fit3d_clips, analyses_by_id)
    primary_interval_errors = top_to_top_interval_errors(
        fit3d_clips,
        analyses_by_id,
        selected_clip_ids=set(primary_by_cluster.values()),
    )
    rep_audit, confidence_summary = build_confidence_audit(
        fit3d_clips,
        analyses_by_id,
        confidence_config["core"],
    )
    raw_rep_audit, raw_confidence_summary = build_confidence_audit(
        fit3d_clips,
        raw_analyses_by_id,
        confidence_config["core"],
    )
    gated_coverage = coverage_summary(rows, label="gated")
    raw_coverage = coverage_summary(raw_rows, label="raw")
    clip_metrics = build_clip_metrics(
        fit3d_clips,
        rows,
        analyses_by_id,
        raw_rows,
        raw_analyses_by_id,
        primary_by_cluster=primary_by_cluster,
        confidence_summary=confidence_summary,
        raw_confidence_summary=raw_confidence_summary,
    )
    for row in clip_metrics:
        clip = next(clip for clip in fit3d_clips if clip["clip_id"] == row["clip_id"])
        row["top_to_top_interval_median_frames"] = phase_states.percentile(
            top_to_top_interval_errors(
                [clip], analyses_by_id
            ),
            50,
        )
        row["raw_top_to_top_interval_median_frames"] = phase_states.percentile(
            top_to_top_interval_errors([clip], raw_analyses_by_id),
            50,
        )
    event_matches = build_event_match_rows(fit3d_clips, analyses_by_id)
    raw_event_matches = build_event_match_rows(
        fit3d_clips, raw_analyses_by_id
    )
    primary_ids = set(primary_by_cluster.values())
    for row in event_matches:
        row["is_primary_camera"] = row["clip_id"] in primary_ids
    for row in raw_event_matches:
        row["is_primary_camera"] = row["clip_id"] in primary_ids

    baseline = load_t1_5_baseline(args.t1_5_evaluation_dir)
    state_only_rows, state_only_clip_rows = analyze_state_only(
        fit3d_clips,
        analysis_dir=args.fit3d_analysis_dir,
        segmenter_config=segmenter_config,
    )
    complete_only_rows = build_chain_event_rows(
        fit3d_clips, analyses_by_id, mode="complete_only"
    )
    state_eval_summary = load_json(
        ROOT / "results" / "task1_6" / "dev" / "state_eval_summary.json"
    )
    anchor_dev_rows = [
        row
        for row in state_eval_summary.get("anchor_comparison", [])
        if row.get("signal_mode") == "t1_6_bilateral_mean"
    ]
    ablation_rows = build_ablation_rows(
        baseline=baseline,
        state_only_rows=state_only_rows,
        state_only_clip_rows=state_only_clip_rows,
        full_rows=rows,
        complete_only_rows=complete_only_rows,
        confidence_summary=confidence_summary,
        anchor_dev_rows=anchor_dev_rows,
    )

    if args.skip_robustness:
        robustness_rows = []
        robustness_summary = []
    else:
        robustness_rows = evaluate_robustness(
            fit3d_clips,
            analyses_by_id,
            analysis_dir=args.fit3d_analysis_dir,
            segmenter_config=segmenter_config,
            chain_config=chain_config,
        )
        robustness_summary = summarize_robustness(robustness_rows)
    negative_controls = evaluate_negative_controls(
        segmenter_config=segmenter_config,
        chain_config=chain_config,
    )

    if args.skip_sealed:
        sealed_rows = []
        sealed_summary = {
            "segment_count": len(sealed_clips),
            "annotation_ready_count": 0,
            "runnable_segment_count": 0,
            "formal_stage_gate_evaluable": False,
            "stage_passed": None,
            "metrics": None,
            "limitation": "sealed evaluation skipped by command line",
        }
    else:
        sealed_rows, sealed_summary = evaluate_sealed_holdout(
            sealed_clips,
            segmenter_config=segmenter_config,
            chain_config=chain_config,
            confidence_core=confidence_config["core"],
            input_dir=args.sealed_input_dir,
            runs_dir=args.runs_dir,
            template_path=args.sealed_template,
            manifest_path=args.sealed_manifest,
        )
    go_no_go = build_go_no_go(
        all_cameras=all_cameras,
        primary=primary,
        consensus=consensus,
        confidence_summary=confidence_summary,
        top_interval_median=phase_states.percentile(interval_errors, 50),
        sealed_summary=sealed_summary,
        coverage=gated_coverage,
    )
    failed_gates = [row["gate_id"] for row in go_no_go if row["status"] == "fail"]
    not_evaluable_gates = [
        row["gate_id"] for row in go_no_go if row["status"] == "not_evaluable"
    ]
    fit3d_gates = [row for row in go_no_go if row["dataset"] == FIT3D_DATASET]
    fit3d_stage_passed = all(row["passed"] is True for row in fit3d_gates)
    overall_stage_passed = bool(
        fit3d_stage_passed and sealed_summary["formal_stage_gate_evaluable"]
        and sealed_summary["stage_passed"] is True
    )
    decision = "GO" if overall_stage_passed else "NO_GO"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": "T1-6E",
        "generated_at": phase_states.utc_now(),
        "source": {
            "sources": str(args.sources.resolve()),
            "sources_sha256": sha256_path(args.sources),
            "segmenter_config": str(args.segmenter_config.resolve()),
            "segmenter_config_sha256": sha256_path(args.segmenter_config),
            "chain_config": str(args.chain_config.resolve()),
            "chain_config_sha256": sha256_path(args.chain_config),
            "confidence_config": str(args.confidence_config.resolve()),
            "confidence_config_sha256": sha256_path(args.confidence_config),
            "fit3d_manifest": str(args.fit3d_manifest.resolve()),
            "fit3d_manifest_sha256": sha256_path(args.fit3d_manifest),
            "t1_5_evaluation_dir": str(args.t1_5_evaluation_dir.resolve()),
            "t1_5_evaluation_summary_sha256": sha256_path(
                args.t1_5_evaluation_dir / "summary.json"
            ),
            "fit3d_analysis_sha256": sha256_tree(
                args.fit3d_analysis_dir.glob("*/analysis.json")
            ),
            "run_analysis_sha256": sha256_tree(
                fit3d_run_dir.glob("*/analysis_v2.json")
            ),
            "raw_run_analysis_sha256": sha256_tree(
                fit3d_raw_run_dir.glob("*/analysis_v2.json")
            ),
        },
        "split": {
            "fit3d": "holdout",
            "fit3d_note": (
                "T1-5D had observed aggregate holdout before T1-6; this is a "
                "locked evaluation, not a never-seen external test."
            ),
            "qevd": "sealed_holdout",
        },
        "fit3d_holdout": {
            "clip_count": len(fit3d_clips),
            "cluster_count": consensus["cluster_count"],
            "all_cameras": all_cameras,
            "primary_camera": primary,
            "count_consensus": consensus,
            "top_to_top_interval_median_frames": phase_states.percentile(
                interval_errors, 50
            ),
            "top_to_top_interval_p90_frames": phase_states.percentile(
                interval_errors, 90
            ),
            "primary_top_to_top_interval_median_frames": phase_states.percentile(
                primary_interval_errors, 50
            ),
            "confidence": confidence_summary,
            "raw": {
                "clip_count": len(fit3d_clips),
                "cluster_count": raw_consensus["cluster_count"],
                "all_cameras": raw_all_cameras,
                "primary_camera": raw_primary,
                "count_consensus": raw_consensus,
                "top_to_top_interval_median_frames": phase_states.percentile(
                    raw_interval_errors, 50
                ),
                "top_to_top_interval_p90_frames": phase_states.percentile(
                    raw_interval_errors, 90
                ),
                "confidence": raw_confidence_summary,
                "coverage": raw_coverage,
            },
            "gated": {
                "clip_count": len(fit3d_clips),
                "cluster_count": consensus["cluster_count"],
                "all_cameras": all_cameras,
                "primary_camera": primary,
                "count_consensus": consensus,
                "top_to_top_interval_median_frames": phase_states.percentile(
                    interval_errors, 50
                ),
                "top_to_top_interval_p90_frames": phase_states.percentile(
                    interval_errors, 90
                ),
                "confidence": confidence_summary,
                "coverage": gated_coverage,
            },
            "quality_status_counts": dict(
                Counter(str(row.get("quality_status")) for row in rows)
            ),
            "quality_excluded_clip_ids": [
                row["clip_id"] for row in rows if row.get("excluded_by_quality")
            ],
            "motion_excluded_clip_ids": [
                row["clip_id"] for row in rows if row.get("excluded_by_motion")
            ],
            "coverage": {
                "raw": raw_coverage,
                "gated": gated_coverage,
            },
            "stage_passed": fit3d_stage_passed,
        },
        "qevd_sealed_holdout": sealed_summary,
        "ablation": ablation_rows,
        "robustness": robustness_summary,
        "negative_controls": {
            "count": len(negative_controls),
            "passed": sum(bool(row["passed"]) for row in negative_controls),
            "all_passed": all(bool(row["passed"]) for row in negative_controls),
        },
        "go_no_go": go_no_go,
        "verdict": {
            "decision": decision,
            "fit3d_holdout_stage_passed": fit3d_stage_passed,
            "qevd_sealed_formal_evaluable": sealed_summary[
                "formal_stage_gate_evaluable"
            ],
            "qevd_sealed_stage_passed": sealed_summary["stage_passed"],
            "overall_stage_passed": overall_stage_passed,
            "failed_required_gates": failed_gates,
            "not_evaluable_gates": not_evaluable_gates,
            "integration": (
                "do_not_integrate_v2_into_T1-4_or_T1-7"
                if decision == "NO_GO"
                else "integrate_v2_after_reproducibility_review"
            ),
            "note": (
                "Fit3D holdout fails multiple fixed gates and QEVD sealed GT "
                "is incomplete. Keep RepChain v2 experimental and do not write "
                "back T1-4 frozen artifacts."
            ),
        },
        "artifacts": {
            "summary": "summary.json",
            "clip_metrics": "fit3d_holdout_clip_metrics.csv",
            "event_matches": "fit3d_holdout_event_matches.csv",
            "raw_event_matches": "fit3d_holdout_raw_event_matches.csv",
            "rep_confidence": "fit3d_holdout_rep_confidence.csv",
            "raw_rep_confidence": "fit3d_holdout_raw_rep_confidence.csv",
            "ablation": "ablation.csv",
            "robustness": "robustness.csv",
            "negative_controls": "negative_controls.csv",
            "sealed_predictions": "qevd_sealed_predictions.csv",
            "go_no_go": "go_no_go.csv",
        },
    }
    write_outputs(
        output_dir=args.output_dir,
        tables_dir=args.tables_dir,
        clip_metrics=clip_metrics,
        event_matches=event_matches,
        raw_event_matches=raw_event_matches,
        rep_audit=rep_audit,
        raw_rep_audit=raw_rep_audit,
        ablation_rows=ablation_rows,
        robustness_rows=robustness_rows,
        robustness_summary=robustness_summary,
        negative_controls=negative_controls,
        sealed_rows=sealed_rows,
        go_no_go=go_no_go,
        summary=summary,
    )
    csv_names = [
        "fit3d_holdout_clip_metrics.csv",
        "fit3d_holdout_event_matches.csv",
        "fit3d_holdout_raw_event_matches.csv",
        "fit3d_holdout_rep_confidence.csv",
        "fit3d_holdout_raw_rep_confidence.csv",
        "ablation.csv",
        "robustness.csv",
        "robustness_summary.csv",
        "negative_controls.csv",
        "qevd_sealed_predictions.csv",
        "go_no_go.csv",
    ]
    summary["reproducibility"] = reproducibility_check(
        args.output_dir,
        csv_names,
        scope={
            "skip_robustness": args.skip_robustness,
            "skip_sealed": args.skip_sealed,
        },
    )
    write_json(args.output_dir / "summary.json", summary)
    phase_states.write_checksums(
        args.output_dir,
        csv_names + ["summary.json", "artifact_hashes.json"],
    )
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
