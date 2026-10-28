#!/usr/bin/env python3
"""Evaluate T1-4 rep counts and top events against Fit3D annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_5_metrics as metrics


DEFAULT_MANIFEST = ROOT / "configs" / "task1_5_fit3d_clips.json"
DEFAULT_ANALYSIS_DIR = ROOT / "results" / "task1_5" / "clip_analysis"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task1_5" / "rep_evaluation"

SCHEMA_VERSION = "task1-5d-rep-eval-v1"
EVENT_GATE_PERIOD_FRACTION = 0.25
HIGH_CONFIDENCE_THRESHOLD = 0.50
EXPECTED_TRAIN_CLIPS = 64
EXPECTED_ACTION_INSTANCES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "task1-4-v1":
        raise ValueError(f"unexpected manifest schema: {payload.get('schema_version')}")
    return payload


def select_train_clips(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    clips = [
        clip
        for clip in manifest["clips"]
        if clip.get("has_gt") and not clip.get("smoke_only")
    ]
    action_keys = subject_action_keys(clips)
    if len(clips) != EXPECTED_TRAIN_CLIPS:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_CLIPS} train clips, found {len(clips)}"
        )
    if len(action_keys) != EXPECTED_ACTION_INSTANCES:
        raise ValueError(
            "expected "
            f"{EXPECTED_ACTION_INSTANCES} subject-action instances, "
            f"found {len(action_keys)}"
        )
    if any(clip["split"] not in {"dev", "holdout"} for clip in clips):
        raise ValueError("train evaluation contains a non-dev/holdout clip")
    return clips


def subject_action_key(clip: dict[str, Any]) -> str:
    return f"{clip['subject_id']}_{clip['exercise']}"


def subject_action_keys(clips: list[dict[str, Any]]) -> set[str]:
    return {subject_action_key(clip) for clip in clips}


def median_round_half_up(values: list[int | float]) -> int:
    if not values:
        raise ValueError("at least one value is required")
    return int(math.floor(float(np.median(values)) + 0.5))


def choose_primary_camera(clips: list[dict[str, Any]]) -> dict[str, Any]:
    if not clips:
        raise ValueError("at least one clip is required")

    def priority(clip: dict[str, Any]) -> tuple[float, int, str]:
        angle = float(
            (clip.get("view_geometry") or {}).get(
                "angle_from_frontal_normal_deg", -1.0
            )
        )
        is_side = int(clip.get("view") in metrics.SIDE_VIEWS)
        return angle, is_side, clip["camera_id"]

    return max(clips, key=priority)


def load_analysis(analysis_dir: Path, clip_id: str) -> dict[str, Any] | None:
    path = analysis_dir / clip_id / "analysis.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def accepted_events(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    segmentation = analysis.get("segmentation") or {}
    events = []
    for index, rep in enumerate(segmentation.get("reps") or [], start=1):
        if rep.get("status", "complete") != "complete":
            continue
        top_frame = rep.get("top_frame", rep.get("start_frame"))
        if top_frame is None:
            continue
        events.append(
            {
                "rep_id": rep.get("rep_id", index),
                "top_frame": int(top_frame),
                "start_frame": int(rep.get("start_frame", top_frame)),
                "end_frame": int(rep.get("end_frame", top_frame)),
                "confidence": finite_float(rep.get("confidence")),
            }
        )
    return sorted(events, key=lambda item: (item["top_frame"], item["rep_id"]))


def rejected_events(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    segmentation = analysis.get("segmentation") or {}
    events = []
    for index, rep in enumerate(segmentation.get("rejected_reps") or [], start=1):
        top_frame = rep.get("top_frame", rep.get("start_frame"))
        if top_frame is None:
            continue
        events.append(
            {
                "candidate_rep_id": rep.get(
                    "candidate_rep_id", rep.get("rep_id", index)
                ),
                "top_frame": int(top_frame),
                "start_frame": int(rep.get("start_frame", top_frame)),
                "end_frame": int(rep.get("end_frame", top_frame)),
                "confidence": finite_float(rep.get("confidence")),
                "reject_reasons": rep.get("reject_reasons")
                or rep.get("reject_reason")
                or "",
            }
        )
    return sorted(
        events, key=lambda item: (item["top_frame"], str(item["candidate_rep_id"]))
    )


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def event_gate_frames(gt_frames: list[int]) -> int:
    if len(gt_frames) < 2:
        return 1
    median_interval = float(np.median(np.diff(sorted(gt_frames))))
    return max(1, int(math.floor(EVENT_GATE_PERIOD_FRACTION * median_interval + 0.5)))


def match_events(
    gt_frames: list[int], pred_frames: list[int], gate_frames: int
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


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return round(2.0 * precision * recall / (precision + recall), 6)


def interval_rows(
    gt_frames: list[int],
    events: list[dict[str, Any]],
    matches: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    gt_to_pred = {gt_index: pred_index for gt_index, pred_index, _ in matches}
    rows = []
    for gt_index, pred_index, _ in matches:
        next_gt_index = gt_index + 1
        next_pred_index = pred_index + 1
        if gt_to_pred.get(next_gt_index) != next_pred_index:
            continue
        if next_gt_index >= len(gt_frames) or next_pred_index >= len(events):
            continue
        gt_interval = gt_frames[next_gt_index] - gt_frames[gt_index]
        pred_interval = (
            events[next_pred_index]["top_frame"] - events[pred_index]["top_frame"]
        )
        pred_duration = (
            events[pred_index]["end_frame"]
            - events[pred_index]["start_frame"]
            + 1
        )
        rows.append(
            {
                "gt_event_index": gt_index + 1,
                "pred_event_index": pred_index + 1,
                "gt_top_frame": gt_frames[gt_index],
                "pred_top_frame": events[pred_index]["top_frame"],
                "gt_interval_frames": gt_interval,
                "pred_interval_frames": pred_interval,
                "interval_error_frames": pred_interval - gt_interval,
                "pred_duration_frames": pred_duration,
                "duration_error_frames": pred_duration - gt_interval,
            }
        )
    return rows


def classify_rejections(
    rejected: list[dict[str, Any]],
    gt_frames: list[int],
    matches: list[tuple[int, int, int]],
    gate_frames: int,
) -> list[dict[str, Any]]:
    matched_gt_indexes = {gt_index for gt_index, _, _ in matches}
    rows = []
    for candidate in rejected:
        if not gt_frames:
            nearest_index = None
            nearest_distance = None
            within_gate = False
            classification = "unscoreable_no_gt"
        else:
            distances = [abs(candidate["top_frame"] - frame) for frame in gt_frames]
            nearest_index = int(np.argmin(distances))
            nearest_distance = int(distances[nearest_index])
            within_gate = nearest_distance <= gate_frames
            if not within_gate:
                classification = "safe_refusal_non_gt"
            elif nearest_index in matched_gt_indexes:
                classification = "safe_refusal_duplicate"
            else:
                classification = "conservative_refusal_true_event"
        rows.append(
            {
                **candidate,
                "nearest_gt_event_index": None
                if nearest_index is None
                else nearest_index + 1,
                "nearest_gt_frame": None
                if nearest_index is None
                else gt_frames[nearest_index],
                "nearest_gt_distance_frames": nearest_distance,
                "within_event_gate": within_gate,
                "classification": classification,
            }
        )
    return rows


def evaluate_clip(
    clip: dict[str, Any], analysis_dir: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    analysis = load_analysis(analysis_dir, clip["clip_id"])
    base = {
        "clip_id": clip["clip_id"],
        "subject_id": clip["subject_id"],
        "camera_id": clip["camera_id"],
        "exercise": clip["exercise"],
        "split": clip["split"],
        "view": clip["view"],
        "view_angle_deg": (clip.get("view_geometry") or {}).get(
            "angle_from_frontal_normal_deg"
        ),
        "camera_side": (clip.get("view_geometry") or {}).get("camera_side"),
    }
    gt_frames = sorted(
        int(frame) for frame in (clip.get("gt") or {}).get("rep_frames", [])
    )
    if analysis is None:
        summary = {
            **base,
            "analysis_status": "missing",
            "quality_status": "missing",
            "segmentation_method": None,
            "phase_angle": None,
            "gt_rep_count": len(gt_frames),
            "declared_rep_count": None,
            "pred_rep_count": 0,
            "rep_count_error": None,
            "abs_rep_count_error": None,
            "event_gate_frames": event_gate_frames(gt_frames),
            "event_tp": 0,
            "event_fp": 0,
            "event_fn": len(gt_frames),
            "event_precision": None,
            "event_recall": 0.0 if gt_frames else None,
            "event_f1": None,
            "event_abs_error_median_frames": None,
            "event_abs_error_p90_frames": None,
            "event_abs_error_max_frames": None,
            "interval_pair_count": 0,
            "interval_abs_error_median_frames": None,
            "interval_abs_error_p90_frames": None,
            "duration_abs_error_median_frames": None,
            "duration_abs_error_p90_frames": None,
            "rejected_candidate_count": 0,
            "rejected_true_event_count": 0,
            "rejected_duplicate_count": 0,
            "rejected_non_gt_count": 0,
            "high_confidence_fp_count": 0,
            "status_ok": False,
            "quality_ok": False,
            "count_usable": False,
        }
        return summary, [], [], [], []

    events = accepted_events(analysis)
    rejected = rejected_events(analysis)
    gate = event_gate_frames(gt_frames)
    matches = match_events(gt_frames, [item["top_frame"] for item in events], gate)
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_pred = {pred_index for _, pred_index, _ in matches}
    errors = [error for _, _, error in matches]
    intervals = interval_rows(gt_frames, events, matches)
    rejects = classify_rejections(rejected, gt_frames, matches, gate)

    event_rows = []
    for pred_index, event in enumerate(events):
        match = next(
            (item for item in matches if item[1] == pred_index), None
        )
        gt_index = None if match is None else match[0]
        event_rows.append(
            {
                **base,
                "event_index": pred_index + 1,
                "rep_id": event["rep_id"],
                "top_frame": event["top_frame"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "confidence": event["confidence"],
                "match_status": "matched" if match else "false_positive",
                "gt_event_index": None if gt_index is None else gt_index + 1,
                "gt_frame": None if gt_index is None else gt_frames[gt_index],
                "signed_error_frames": None
                if gt_index is None
                else event["top_frame"] - gt_frames[gt_index],
                "abs_error_frames": None if match is None else match[2],
                "high_confidence_false_positive": bool(
                    match is None
                    and event["confidence"] is not None
                    and event["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
                ),
            }
        )

    match_rows = []
    for gt_index, gt_frame in enumerate(gt_frames):
        match = next((item for item in matches if item[0] == gt_index), None)
        pred_index = None if match is None else match[1]
        match_rows.append(
            {
                **base,
                "gt_event_index": gt_index + 1,
                "gt_frame": gt_frame,
                "match_status": "matched" if match else "false_negative",
                "pred_event_index": None if pred_index is None else pred_index + 1,
                "pred_top_frame": None
                if pred_index is None
                else events[pred_index]["top_frame"],
                "pred_confidence": None
                if pred_index is None
                else events[pred_index]["confidence"],
                "signed_error_frames": None
                if pred_index is None
                else events[pred_index]["top_frame"] - gt_frame,
                "abs_error_frames": None if match is None else match[2],
                "event_gate_frames": gate,
            }
        )

    interval_error_values = [
        abs(int(row["interval_error_frames"])) for row in intervals
    ]
    duration_error_values = [
        abs(int(row["duration_error_frames"])) for row in intervals
    ]
    tp = len(matches)
    fp = len(events) - tp
    fn = len(gt_frames) - tp
    precision = ratio(tp, len(events))
    recall = ratio(tp, len(gt_frames))
    summary = {
        **base,
        "analysis_status": analysis.get("status"),
        "quality_status": analysis.get("quality_status"),
        "segmentation_method": (analysis.get("segmentation") or {}).get("method"),
        "phase_angle": (analysis.get("segmentation") or {}).get("phase_angle"),
        "gt_rep_count": len(gt_frames),
        "declared_rep_count": analysis.get("rep_count"),
        "pred_rep_count": len(events),
        "rep_count_error": len(events) - len(gt_frames),
        "abs_rep_count_error": abs(len(events) - len(gt_frames)),
        "event_gate_frames": gate,
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1_score(precision, recall),
        "event_abs_error_median_frames": percentile(errors, 50),
        "event_abs_error_p90_frames": percentile(errors, 90),
        "event_abs_error_max_frames": max(errors) if errors else None,
        "interval_pair_count": len(intervals),
        "interval_abs_error_median_frames": percentile(interval_error_values, 50),
        "interval_abs_error_p90_frames": percentile(interval_error_values, 90),
        "duration_abs_error_median_frames": percentile(duration_error_values, 50),
        "duration_abs_error_p90_frames": percentile(duration_error_values, 90),
        "rejected_candidate_count": len(rejects),
        "rejected_true_event_count": sum(
            row["classification"] == "conservative_refusal_true_event"
            for row in rejects
        ),
        "rejected_duplicate_count": sum(
            row["classification"] == "safe_refusal_duplicate" for row in rejects
        ),
        "rejected_non_gt_count": sum(
            row["classification"] == "safe_refusal_non_gt" for row in rejects
        ),
        "high_confidence_fp_count": sum(
            row["high_confidence_false_positive"] for row in event_rows
        ),
        "status_ok": analysis.get("status") == "ok",
        "quality_ok": analysis.get("quality_status") in {"pass", "warn"},
        "count_usable": analysis.get("status") in {"ok", "partial", "rejected_quality"},
    }
    return summary, event_rows, match_rows, rejects, intervals


def percentile(values: list[int | float], value: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), value)), 6)


def build_subject_action_rows(
    clips: list[dict[str, Any]], clip_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_clip = {row["clip_id"]: row for row in clip_rows}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[subject_action_key(clip)].append(clip)

    action_rows = []
    for action_key, action_clips in sorted(grouped.items()):
        usable = [
            clip
            for clip in action_clips
            if by_clip[clip["clip_id"]]["count_usable"]
        ]
        if not usable:
            continue
        rows = [by_clip[clip["clip_id"]] for clip in usable]
        gt_counts = {row["gt_rep_count"] for row in rows}
        if len(gt_counts) != 1:
            raise ValueError(f"inconsistent GT count for {action_key}: {gt_counts}")
        gt_count = int(next(iter(gt_counts)))
        predicted_counts = [int(row["pred_rep_count"]) for row in rows]
        consensus = median_round_half_up(predicted_counts)
        primary_clip = choose_primary_camera(usable)
        primary = by_clip[primary_clip["clip_id"]]
        action_rows.append(
            {
                "action_instance": action_key,
                "subject_id": primary["subject_id"],
                "exercise": primary["exercise"],
                "split": primary["split"],
                "gt_rep_count": gt_count,
                "camera_count": len(rows),
                "camera_ids": "|".join(clip["camera_id"] for clip in usable),
                "camera_pred_counts": "|".join(str(value) for value in predicted_counts),
                "consensus_pred_count": consensus,
                "consensus_count_error": consensus - gt_count,
                "consensus_abs_count_error": abs(consensus - gt_count),
                "consensus_count_exact": consensus == gt_count,
                "primary_clip_id": primary["clip_id"],
                "primary_camera_id": primary["camera_id"],
                "primary_view": primary["view"],
                "primary_view_angle_deg": primary["view_angle_deg"],
                "primary_pred_count": primary["pred_rep_count"],
                "primary_count_error": primary["pred_rep_count"] - gt_count,
                "primary_abs_count_error": abs(primary["pred_rep_count"] - gt_count),
                "primary_count_exact": primary["pred_rep_count"] == gt_count,
                "primary_event_tp": primary["event_tp"],
                "primary_event_fp": primary["event_fp"],
                "primary_event_fn": primary["event_fn"],
                "primary_event_abs_error_median_frames": primary[
                    "event_abs_error_median_frames"
                ],
                "primary_event_abs_error_p90_frames": primary[
                    "event_abs_error_p90_frames"
                ],
                "all_camera_event_tp": sum(row["event_tp"] for row in rows),
                "all_camera_event_fp": sum(row["event_fp"] for row in rows),
                "all_camera_event_fn": sum(row["event_fn"] for row in rows),
                "all_camera_rejected_candidate_count": sum(
                    row["rejected_candidate_count"] for row in rows
                ),
                "all_camera_high_confidence_fp_count": sum(
                    row["high_confidence_fp_count"] for row in rows
                ),
            }
        )
    return action_rows


def event_scope_summary(
    clip_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    clip_ids: set[str],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    clip_subset = [row for row in clip_rows if row["clip_id"] in clip_ids]
    matches = [row for row in match_rows if row["clip_id"] in clip_ids]
    matched = [row for row in matches if row["match_status"] == "matched"]
    errors = [int(row["abs_error_frames"]) for row in matched]
    signed_errors = [int(row["signed_error_frames"]) for row in matched]
    tp = len(matched)
    pred_count = sum(row["pred_rep_count"] for row in clip_subset)
    gt_count = sum(row["gt_rep_count"] for row in clip_subset)
    precision = ratio(tp, pred_count)
    recall = ratio(tp, gt_count)
    cluster_values: defaultdict[str, list[float]] = defaultdict(list)
    for row in matched:
        cluster_values[subject_action_key(row)].append(float(row["abs_error_frames"]))
    cluster_means = {
        cluster: float(np.mean(values)) for cluster, values in cluster_values.items()
    }
    bootstrap = metrics.cluster_bootstrap_ci(
        cluster_means, seed=seed, samples=samples
    )
    return {
        "clip_observations": len(clip_subset),
        "gt_event_observations": gt_count,
        "pred_event_observations": pred_count,
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "event_abs_error_median_frames": percentile(errors, 50),
        "event_abs_error_p90_frames": percentile(errors, 90),
        "event_abs_error_max_frames": max(errors) if errors else None,
        "event_signed_error_median_frames": percentile(signed_errors, 50),
        "event_signed_error_mean_frames": round(
            float(np.mean(signed_errors)), 6
        )
        if signed_errors
        else None,
        "event_pred_earlier_than_gt_count": sum(error < 0 for error in signed_errors),
        "event_pred_later_than_gt_count": sum(error > 0 for error in signed_errors),
        "event_cluster_count": bootstrap["clusters"],
        "event_cluster_mean_abs_error_frames": bootstrap["mean"],
        "event_cluster_bootstrap_ci95_frames": [
            bootstrap["ci95_low"],
            bootstrap["ci95_high"],
        ],
        "high_confidence_fp_count": sum(
            row["high_confidence_fp_count"] for row in clip_subset
        ),
    }


def count_cluster_summary(
    action_rows: list[dict[str, Any]],
    prediction_field: str,
    error_field: str,
    exact_field: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values = [int(row[error_field]) for row in action_rows]
    cluster_values = {
        row["action_instance"]: float(row[error_field]) for row in action_rows
    }
    bootstrap = metrics.cluster_bootstrap_ci(
        cluster_values, seed=seed, samples=samples
    )
    return {
        "action_instances": len(action_rows),
        "gt_unique_reps": sum(int(row["gt_rep_count"]) for row in action_rows),
        "predicted_rep_sum": sum(int(row[prediction_field]) for row in action_rows),
        "count_mae": round(float(np.mean(values)), 6) if values else None,
        "count_median_abs_error": round(float(np.median(values)), 6)
        if values
        else None,
        "count_error_p90": round(float(np.percentile(values, 90.0)), 6)
        if values
        else None,
        "exact_count_correct": sum(bool(row[exact_field]) for row in action_rows),
        "exact_count_accuracy": ratio(
            sum(bool(row[exact_field]) for row in action_rows), len(action_rows)
        ),
        "cluster_bootstrap_ci95_mae": [
            bootstrap["ci95_low"],
            bootstrap["ci95_high"],
        ],
    }


def build_stratified_rows(
    clips: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    seed: int,
    samples: int,
) -> list[dict[str, Any]]:
    clip_by_id = {clip["clip_id"]: clip for clip in clips}
    action_by_key = {row["action_instance"]: row for row in action_rows}
    strata: list[tuple[str, str, list[dict[str, Any]]]] = []
    for exercise in sorted({clip["exercise"] for clip in clips}):
        strata.append(
            (
                "exercise",
                exercise,
                [clip for clip in clips if clip["exercise"] == exercise],
            )
        )
    for exercise_view in sorted(
        {(clip["exercise"], clip["view"]) for clip in clips}
    ):
        strata.append(
            (
                "exercise_view",
                f"{exercise_view[0]}/{exercise_view[1]}",
                [
                    clip
                    for clip in clips
                    if (clip["exercise"], clip["view"]) == exercise_view
                ],
            )
        )
    for split in ("dev", "holdout"):
        strata.append(
            ("split", split, [clip for clip in clips if clip["split"] == split])
        )
    for subject in sorted({clip["subject_id"] for clip in clips}):
        strata.append(
            (
                "subject",
                subject,
                [clip for clip in clips if clip["subject_id"] == subject],
            )
        )

    rows = []
    for stratum_type, stratum, stratum_clips in strata:
        clip_ids = {clip["clip_id"] for clip in stratum_clips}
        subset_clip_rows = [row for row in clip_rows if row["clip_id"] in clip_ids]
        action_keys = subject_action_keys(stratum_clips)
        subset_actions = [action_by_key[key] for key in action_keys if key in action_by_key]
        event_summary = event_scope_summary(
            clip_rows, match_rows, clip_ids, seed=seed, samples=samples
        )
        count_errors = [int(row["abs_rep_count_error"]) for row in subset_clip_rows]
        rows.append(
            {
                "stratum_type": stratum_type,
                "stratum": stratum,
                "clip_observations": len(subset_clip_rows),
                "action_instances": len(subset_actions),
                "camera_count_mae": round(float(np.mean(count_errors)), 6)
                if count_errors
                else None,
                "consensus_count_mae": round(
                    float(
                        np.mean(
                            [int(row["consensus_abs_count_error"]) for row in subset_actions]
                        )
                    ),
                    6,
                )
                if subset_actions
                else None,
                "consensus_exact_count_correct": sum(
                    bool(row["consensus_count_exact"]) for row in subset_actions
                ),
                "event_tp": event_summary["tp"],
                "event_fp": event_summary["fp"],
                "event_fn": event_summary["fn"],
                "event_precision": event_summary["precision"],
                "event_recall": event_summary["recall"],
                "event_f1": event_summary["f1"],
                "event_abs_error_median_frames": event_summary[
                    "event_abs_error_median_frames"
                ],
                "event_abs_error_p90_frames": event_summary[
                    "event_abs_error_p90_frames"
                ],
                "high_confidence_fp_count": event_summary[
                    "high_confidence_fp_count"
                ],
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    clips = select_train_clips(manifest)
    primary_ids = {
        choose_primary_camera(group)["clip_id"]
        for group in group_clips_by_action(clips).values()
    }

    clip_rows = []
    event_rows = []
    match_rows = []
    rejected_rows = []
    interval_rows_all = []
    for clip in clips:
        summary, events, matches, rejects, intervals = evaluate_clip(
            clip, args.analysis_dir.resolve()
        )
        summary["camera_role"] = (
            "primary_closest_to_side"
            if clip["clip_id"] in primary_ids
            else "secondary"
        )
        clip_rows.append(summary)
        event_rows.extend(events)
        match_rows.extend(matches)
        rejected_rows.extend(rejects)
        interval_rows_all.extend(
            {
                **row,
                "clip_id": clip["clip_id"],
                "subject_id": clip["subject_id"],
                "camera_id": clip["camera_id"],
                "exercise": clip["exercise"],
                "split": clip["split"],
                "view": clip["view"],
            }
            for row in intervals
        )
        for item in (*events, *matches, *rejects, *intervals):
            item["camera_role"] = summary["camera_role"]

    action_rows = build_subject_action_rows(clips, clip_rows)
    all_ids = {clip["clip_id"] for clip in clips}
    all_events = event_scope_summary(
        clip_rows,
        match_rows,
        all_ids,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    primary_events = event_scope_summary(
        clip_rows,
        match_rows,
        primary_ids,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    consensus_counts = count_cluster_summary(
        action_rows,
        "consensus_pred_count",
        "consensus_abs_count_error",
        "consensus_count_exact",
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    primary_counts = count_cluster_summary(
        action_rows,
        "primary_pred_count",
        "primary_abs_count_error",
        "primary_count_exact",
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    camera_errors = [int(row["abs_rep_count_error"]) for row in clip_rows]
    camera_counts = {
        "clip_observations": len(clip_rows),
        "count_mae": round(float(np.mean(camera_errors)), 6),
        "count_median_abs_error": round(float(np.median(camera_errors)), 6),
        "exact_count_correct": sum(error == 0 for error in camera_errors),
        "exact_count_accuracy": ratio(
            sum(error == 0 for error in camera_errors), len(camera_errors)
        ),
    }
    rejection_reason_counts = Counter(
        reason
        for row in rejected_rows
        for reason in str(row["reject_reasons"]).split(";")
        if reason
    )
    rejection_classification_counts = Counter(
        row["classification"] for row in rejected_rows
    )
    interval_errors = [
        abs(int(row["interval_error_frames"])) for row in interval_rows_all
    ]
    duration_errors = [
        abs(int(row["duration_error_frames"])) for row in interval_rows_all
    ]
    acceptance = {
        "rep_count_mae_lte_0_5": consensus_counts["count_mae"] is not None
        and consensus_counts["count_mae"] <= 0.5,
        "exact_count_at_least_14_of_16": consensus_counts[
            "exact_count_correct"
        ]
        >= 14,
        "primary_event_median_lte_5_frames": primary_events[
            "event_abs_error_median_frames"
        ]
        is not None
        and primary_events["event_abs_error_median_frames"] <= 5.0,
        "primary_event_p90_lte_12_frames": primary_events[
            "event_abs_error_p90_frames"
        ]
        is not None
        and primary_events["event_abs_error_p90_frames"] <= 12.0,
        "no_high_confidence_false_positive_rep": all_events[
            "high_confidence_fp_count"
        ]
        == 0,
    }
    acceptance["passed"] = all(acceptance.values())

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "T1-5D",
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "analysis_dir": str(args.analysis_dir.resolve()),
        },
        "dataset": {
            "train_clips": len(clips),
            "action_instances": len(action_rows),
            "unique_gt_reps": sum(row["gt_rep_count"] for row in action_rows),
            "camera_gt_event_observations": sum(
                row["gt_rep_count"] for row in clip_rows
            ),
            "split_counts": dict(Counter(row["split"] for row in clip_rows)),
            "exercise_counts": dict(
                Counter(row["exercise"] for row in clip_rows)
            ),
            "view_counts": dict(Counter(row["view"] for row in clip_rows)),
            "analysis_status_counts": dict(
                Counter(row["analysis_status"] for row in clip_rows)
            ),
            "quality_status_counts": dict(
                Counter(row["quality_status"] for row in clip_rows)
            ),
            "segmentation_method_counts": dict(
                Counter(row["segmentation_method"] for row in clip_rows)
            ),
        },
        "protocol": {
            "gt_semantics": "rep_ann_is_a_top_or_extension_boundary_event",
            "bottom_status": "not_computable_no_bottom_gt",
            "iou_status": "not_reported_rep_interval_semantics_unconfirmed",
            "count_consensus": (
                "median complete-rep count across usable cameras for each "
                "subject-action, rounded half up"
            ),
            "primary_camera": (
                "largest absolute angle from the frontal normal per "
                "subject-action; for squat this is closest-to-side only, not a "
                "validated pure side view"
            ),
            "event_match": (
                "one-to-one Hungarian matching between GT top events and "
                "accepted complete-rep top_frame values"
            ),
            "event_gate": (
                "0.25 x median GT top-to-top interval, rounded half up, "
                "minimum 1 frame"
            ),
            "accepted_event": "segmentation.reps entry with status complete",
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "interval_metrics": (
                "auxiliary, only consecutive GT and predicted events that are "
                "both one-to-one matched; GT interval is top-to-top"
            ),
            "cluster_unit": "subject-action",
            "camera_warning": (
                "four camera clips are repeated observations of the same "
                "subject-action and are never counted as four action instances"
            ),
        },
        "acceptance_thresholds": {
            "rep_count_mae": 0.5,
            "exact_count_correct": ">=14/16",
            "primary_event_median_abs_error_frames": 5,
            "primary_event_p90_abs_error_frames": 12,
            "high_confidence_false_positive_reps_allowed": 0,
        },
        "results": {
            "camera_count_diagnostic": camera_counts,
            "consensus_count_primary": consensus_counts,
            "primary_camera_count_diagnostic": primary_counts,
            "events_all_cameras_diagnostic": all_events,
            "events_primary_camera": primary_events,
            "interval_auxiliary": {
                "pair_count": len(interval_rows_all),
                "interval_abs_error_median_frames": percentile(interval_errors, 50),
                "interval_abs_error_p90_frames": percentile(interval_errors, 90),
                "duration_abs_error_median_frames": percentile(duration_errors, 50),
                "duration_abs_error_p90_frames": percentile(duration_errors, 90),
            },
            "rejections": {
                "candidate_count": len(rejected_rows),
                "classification_counts": dict(rejection_classification_counts),
                "reason_counts": dict(rejection_reason_counts),
            },
            "acceptance": acceptance,
        },
        "subject_actions": action_rows,
    }

    output_dir = args.output_dir.resolve()
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "rep_counts.csv", clip_rows)
    write_csv(output_dir / "rep_events.csv", event_rows)
    write_csv(output_dir / "rep_matches.csv", match_rows)
    write_csv(output_dir / "rejected_candidates.csv", rejected_rows)
    write_csv(output_dir / "rep_intervals.csv", interval_rows_all)
    write_csv(output_dir / "subject_action_summary.csv", action_rows)
    write_csv(
        output_dir / "stratified_summary.csv",
        build_stratified_rows(
            clips,
            clip_rows,
            match_rows,
            action_rows,
            seed=args.seed,
            samples=args.bootstrap_samples,
        ),
    )
    print(
        "T1-5D "
        f"consensus_count_mae={consensus_counts['count_mae']} "
        f"exact={consensus_counts['exact_count_correct']}/{len(action_rows)} "
        f"primary_event_median={primary_events['event_abs_error_median_frames']} "
        f"primary_event_p90={primary_events['event_abs_error_p90_frames']} "
        f"high_conf_fp={all_events['high_confidence_fp_count']} "
        f"passed={acceptance['passed']}"
    )
    print(f"wrote {output_dir}")
    return 0


def group_clips_by_action(
    clips: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[subject_action_key(clip)].append(clip)
    return dict(grouped)


if __name__ == "__main__":
    raise SystemExit(main())
