#!/usr/bin/env python3
"""Validate and evaluate the P0-A blind QEVD-FIT annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_REPS = ROOT / "results" / "segmentation" / "reps.csv"
DEFAULT_SEGMENTS = ROOT / "results" / "segmentation" / "action_segments.csv"
DEFAULT_FIRST_ROUND = ROOT / "results" / "segmentation" / "annotation" / "annotation_template.csv"

VALID_REP_UNITS = {"side", "stride", "cycle"}
VALID_REP_STATUSES = {"accept", "adjust", "add", "exclude", "pending"}
VALID_EXCLUSION_REASONS = {"rest", "transition", "incomplete_rep", "unclear"}
VALID_BOUNDARY_CONFIDENCES = {"high", "medium", "low"}
VALID_PHASE_POLARITIES = {"peak", "trough", "alternating"}
VALID_SECOND_PASS_STATUSES = {"complete", "pending"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--reps", type=Path, default=DEFAULT_REPS)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--first-round", type=Path, default=DEFAULT_FIRST_ROUND)
    parser.add_argument("--methods", nargs="*", default=("aifit", "peak"))
    parser.add_argument("--iou-thresholds", nargs="*", type=float, default=(0.10, 0.30, 0.50))
    parser.add_argument("--primary-iou", type=float, default=0.30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--second-pass-min-ratio", type=float, default=0.20)
    parser.add_argument(
        "--split-overlap-share",
        type=float,
        default=0.50,
        help="Majority-overlap share of the shorter interval used to count split/merge errors.",
    )
    args = parser.parse_args()
    if any(not 0.0 < value <= 1.0 for value in args.iou_thresholds):
        parser.error("--iou-thresholds values must be in (0, 1]")
    if args.primary_iou not in args.iou_thresholds:
        parser.error("--primary-iou must be one of --iou-thresholds")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0.0 <= args.second_pass_min_ratio <= 1.0:
        parser.error("--second-pass-min-ratio must be in [0, 1]")
    if not 0.0 < args.split_overlap_share <= 1.0:
        parser.error("--split-overlap-share must be in (0, 1]")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = min(first[1], second[1]) - max(first[0], second[0]) + 1
    if intersection <= 0:
        return 0.0
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union)


def interval_overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)


def match_with_pairs(
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    threshold: float,
) -> dict[str, Any]:
    if not gt or not pred:
        return {
            "pairs": [],
            "unmatched_gt": list(range(len(gt))),
            "unmatched_pred": list(range(len(pred))),
            "mean_iou": 0.0,
        }
    matrix = np.asarray(
        [[interval_iou((left[0], left[1]), (right[0], right[1])) for right in pred] for left in gt],
        dtype=np.float64,
    )
    row_indices, column_indices = linear_sum_assignment(-matrix)
    pairs = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for row, column in zip(row_indices, column_indices):
        value = float(matrix[row, column])
        if value < threshold:
            continue
        pairs.append((row, column, value))
        matched_gt.add(int(row))
        matched_pred.add(int(column))
    return {
        "pairs": pairs,
        "unmatched_gt": [index for index in range(len(gt)) if index not in matched_gt],
        "unmatched_pred": [index for index in range(len(pred)) if index not in matched_pred],
        "mean_iou": float(np.mean([pair[2] for pair in pairs])) if pairs else 0.0,
    }


def phase_aligned_match(
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    threshold: float,
) -> tuple[dict[str, Any], int]:
    if not gt or not pred:
        return match_with_pairs(gt, pred, threshold), 0
    durations = [end - start + 1 for start, end, _ in gt]
    max_shift = max(3, min(90, int(round(0.5 * float(np.median(durations))))))
    best_result: dict[str, Any] | None = None
    best_shift = 0
    best_key: tuple[int, float, int] | None = None
    for shift in range(-max_shift, max_shift + 1):
        shifted = [(start + shift, end + shift, rep_id) for start, end, rep_id in pred]
        result = match_with_pairs(gt, shifted, threshold)
        iou_sum = float(sum(pair[2] for pair in result["pairs"]))
        key = (len(result["pairs"]), iou_sum, -abs(shift))
        if best_key is None or key > best_key:
            best_key = key
            best_result = result
            best_shift = shift
    assert best_result is not None
    return best_result, best_shift


def load_windows(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        (row["video_id"].zfill(4), row["action"]): (
            int(row["start_frame"]),
            int(row["end_frame"]),
        )
        for row in read_csv(path)
    }


def parse_int(value: str, field: str, segment_id: str, issue_rows: list[dict[str, Any]]) -> int | None:
    try:
        return int(float((value or "").strip()))
    except ValueError:
        issue_rows.append(
            {
                "segment_id": segment_id,
                "rep_id": "",
                "issue": "invalid_integer",
                "detail": f"{field}={value}",
            }
        )
        return None


def load_blind_segments(
    path: Path,
    windows: dict[tuple[str, str], tuple[int, int]],
    issue_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    segments: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        segment_id = row["segment_id"]
        if segment_id in segments:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "duplicate_segment", "detail": ""}
            )
            continue
        key = (row["video_id"].zfill(4), row["action"])
        if key not in windows:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "missing_action_window",
                    "detail": f"{key[0]}/{key[1]}",
                }
            )
        values: dict[str, Any] = dict(row)
        for field in ("window_start_frame", "window_end_frame", "active_start_frame", "active_end_frame", "complete_reps"):
            parsed = parse_int(row.get(field, ""), field, segment_id, issue_rows)
            values[field] = parsed
        values["included_in_eval"] = row.get("included_in_eval", "").strip().lower() == "true"
        if values["rep_unit"] not in VALID_REP_UNITS:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "invalid_rep_unit",
                    "detail": values["rep_unit"],
                }
            )
        if values.get("phase_polarity", "").strip().lower() not in VALID_PHASE_POLARITIES:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "invalid_phase_polarity",
                    "detail": values.get("phase_polarity", ""),
                }
            )
        if not row.get("phase_landmark", "").strip():
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "missing_phase", "detail": ""}
            )
        if not row.get("evidence", "").strip():
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "missing_evidence", "detail": ""}
            )
        if key in windows and values["window_start_frame"] is not None:
            if (values["window_start_frame"], values["window_end_frame"]) != windows[key]:
                issue_rows.append(
                    {
                        "segment_id": segment_id,
                        "rep_id": "",
                        "issue": "window_mismatch",
                        "detail": f"{values['window_start_frame']}-{values['window_end_frame']} vs {windows[key]}",
                    }
                )
            if values["active_start_frame"] is not None and not (
                values["window_start_frame"] <= values["active_start_frame"] <= values["active_end_frame"] <= values["window_end_frame"]
            ):
                issue_rows.append(
                    {
                        "segment_id": segment_id,
                        "rep_id": "",
                        "issue": "active_window_outside_action_window",
                        "detail": f"{values['active_start_frame']}-{values['active_end_frame']}",
                    }
                )
        segments[segment_id] = values
    return segments


def load_blind_reps(
    path: Path,
    segments: dict[str, dict[str, Any]],
    issue_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, list[dict[str, Any]]], Counter[str]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    full_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in read_csv(path):
        segment_id = row["segment_id"]
        status = row.get("review_status", "").strip().lower()
        counts[f"status_{status or 'missing'}"] += 1
        if status not in VALID_REP_STATUSES:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row["rep_id"], "issue": "invalid_status", "detail": status}
            )
            continue
        if segment_id not in segments:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row["rep_id"], "issue": "unknown_segment", "detail": ""}
            )
            continue
        segment = segments[segment_id]
        if row["video_id"].zfill(4) != segment["video_id"] or row["action"] != segment["action"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": row["rep_id"],
                    "issue": "rep_segment_metadata_mismatch",
                    "detail": f"{row['video_id']}/{row['action']}",
                }
            )
        if row.get("rep_unit", "") != segment["rep_unit"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": row["rep_id"],
                    "issue": "rep_unit_mismatch",
                    "detail": f"row={row.get('rep_unit', '')} segment={segment['rep_unit']}",
                }
            )
        if row.get("phase_landmark", "") != segment["phase_landmark"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": row["rep_id"],
                    "issue": "phase_mismatch",
                    "detail": f"row={row.get('phase_landmark', '')} segment={segment['phase_landmark']}",
                }
            )
        confidence = row.get("boundary_confidence", "").strip().lower()
        if confidence not in VALID_BOUNDARY_CONFIDENCES:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": row["rep_id"],
                    "issue": "invalid_boundary_confidence",
                    "detail": confidence,
                }
            )
        if not row.get("annotator", "").strip():
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row["rep_id"], "issue": "missing_annotator", "detail": ""}
            )
        if not row.get("reviewed_at", "").strip():
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row["rep_id"], "issue": "missing_reviewed_at", "detail": ""}
            )
        rep_id = parse_int(row.get("rep_id", ""), "rep_id", segment_id, issue_rows)
        start = parse_int(row.get("gt_start_frame", ""), "gt_start_frame", segment_id, issue_rows)
        end = parse_int(row.get("gt_end_frame", ""), "gt_end_frame", segment_id, issue_rows)
        if rep_id is None or start is None or end is None:
            continue
        if start > end:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "start_after_end", "detail": f"{start}>{end}"}
            )
            continue
        if not segment["active_start_frame"] <= start <= end <= segment["active_end_frame"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "rep_outside_active_window",
                    "detail": f"{start}-{end} not in {segment['active_start_frame']}-{segment['active_end_frame']}",
                }
            )
        entry = {
            "segment_id": segment_id,
            "video_id": row["video_id"].zfill(4),
            "action": row["action"],
            "rep_id": rep_id,
            "start": start,
            "end": end,
            "rep_unit": row["rep_unit"],
            "phase_landmark": row["phase_landmark"],
            "review_status": status,
        }
        full_rows[segment_id].append(entry)
        if segment["included_in_eval"] and status in {"accept", "adjust", "add"}:
            grouped[segment_id].append((start, end, rep_id))
    for segment_id, values in grouped.items():
        values.sort(key=lambda item: item[0])
        rep_ids = [item[2] for item in values]
        if rep_ids != list(range(1, len(values) + 1)):
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "non_sequential_rep_ids",
                    "detail": str(rep_ids),
                }
            )
        for previous, current in zip(values, values[1:]):
            if current[0] != previous[1] + 1:
                issue_rows.append(
                    {
                        "segment_id": segment_id,
                        "rep_id": f"{previous[2]},{current[2]}",
                        "issue": "non_contiguous_reps",
                        "detail": f"{previous[0]}-{previous[1]} then {current[0]}-{current[1]}",
                    }
                )
        expected = segments[segment_id]["complete_reps"]
        if expected is not None and expected != len(values):
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "complete_reps_mismatch",
                    "detail": f"metadata={expected} rows={len(values)}",
                }
            )
    return grouped, full_rows, counts


def load_exclusions(
    path: Path,
    segments: dict[str, dict[str, Any]],
    reps: dict[str, list[tuple[int, int, int]]],
    issue_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        segment_id = row["segment_id"]
        start = parse_int(row.get("start_frame", ""), "start_frame", segment_id, issue_rows)
        end = parse_int(row.get("end_frame", ""), "end_frame", segment_id, issue_rows)
        if start is None or end is None:
            continue
        reason = row.get("exclude_reason", "").strip().lower()
        if reason not in VALID_EXCLUSION_REASONS:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "invalid_exclusion_reason", "detail": reason}
            )
        if segment_id not in segments:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "exclusion_unknown_segment", "detail": ""}
            )
            continue
        if start > end:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": "", "issue": "exclusion_start_after_end", "detail": f"{start}>{end}"}
            )
            continue
        segment = segments[segment_id]
        if not segment["window_start_frame"] <= start <= end <= segment["window_end_frame"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": "",
                    "issue": "exclusion_outside_window",
                    "detail": f"{start}-{end}",
                }
            )
        entry = {"start": start, "end": end, "exclude_reason": reason, "notes": row.get("notes", "")}
        grouped[segment_id].append(entry)
        for rep_start, rep_end, rep_id in reps.get(segment_id, []):
            if interval_overlap((start, end), (rep_start, rep_end)) > 0:
                issue_rows.append(
                    {
                        "segment_id": segment_id,
                        "rep_id": str(rep_id),
                        "issue": "exclusion_overlaps_rep",
                        "detail": f"exclusion {start}-{end} overlaps rep {rep_start}-{rep_end}",
                    }
                )
    for segment_id, values in grouped.items():
        values.sort(key=lambda item: item["start"])
        for previous, current in zip(values, values[1:]):
            if interval_overlap((previous["start"], previous["end"]), (current["start"], current["end"])) > 0:
                issue_rows.append(
                    {
                        "segment_id": segment_id,
                        "rep_id": "",
                        "issue": "overlapping_exclusions",
                        "detail": f"{previous['start']}-{previous['end']} and {current['start']}-{current['end']}",
                    }
                )
    return grouped


def load_second_pass_sample(
    path: Path,
    segments: dict[str, dict[str, Any]],
    all_rep_rows: dict[str, list[dict[str, Any]]],
    included_segments: set[str],
    issue_rows: list[dict[str, Any]],
    min_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = {
        (segment_id, row["rep_id"]): row
        for segment_id, rows in all_rep_rows.items()
        for row in rows
        if segment_id in included_segments
        and row["review_status"] in {"accept", "adjust", "add"}
    }
    rows = read_csv(path)
    sample_keys: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    completed_keys: set[tuple[str, int]] = set()
    count_deltas_by_segment: dict[str, int] = {}
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        segment_id = row.get("segment_id", "").strip()
        if not sample_id:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row.get("rep_id", ""), "issue": "missing_sample_id", "detail": ""}
            )
        elif sample_id in sample_ids:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row.get("rep_id", ""), "issue": "duplicate_sample_id", "detail": sample_id}
            )
        sample_ids.add(sample_id)
        if segment_id not in segments or segment_id not in included_segments:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": row.get("rep_id", ""), "issue": "second_pass_unknown_segment", "detail": ""}
            )
            continue
        segment = segments[segment_id]
        if row.get("video_id", "").zfill(4) != segment["video_id"] or row.get("action", "") != segment["action"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": row.get("rep_id", ""),
                    "issue": "second_pass_metadata_mismatch",
                    "detail": f"{row.get('video_id', '')}/{row.get('action', '')}",
                }
            )
        rep_id = parse_int(row.get("rep_id", ""), "rep_id", segment_id, issue_rows)
        blind_start = parse_int(row.get("blind_start_frame", ""), "blind_start_frame", segment_id, issue_rows)
        blind_end = parse_int(row.get("blind_end_frame", ""), "blind_end_frame", segment_id, issue_rows)
        if rep_id is None or blind_start is None or blind_end is None:
            continue
        key = (segment_id, rep_id)
        if key in sample_keys:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "duplicate_second_pass_rep", "detail": ""}
            )
        sample_keys.add(key)
        entry = entries.get(key)
        if entry is None:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "second_pass_rep_not_in_gt", "detail": ""}
            )
            continue
        if (blind_start, blind_end) != (entry["start"], entry["end"]):
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "second_pass_blind_boundary_mismatch",
                    "detail": f"sample={blind_start}-{blind_end} blind={entry['start']}-{entry['end']}",
                }
            )
        if row.get("rep_unit", "") != entry["rep_unit"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "second_pass_rep_unit_mismatch",
                    "detail": f"sample={row.get('rep_unit', '')} blind={entry['rep_unit']}",
                }
            )
        status = row.get("second_pass_status", "").strip().lower()
        if status not in VALID_SECOND_PASS_STATUSES:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "invalid_second_pass_status", "detail": status}
            )
            continue
        enriched: dict[str, Any] = {
            "sample_id": sample_id,
            "segment_id": segment_id,
            "video_id": entry["video_id"],
            "action": entry["action"],
            "rep_id": rep_id,
            "rep_unit": entry["rep_unit"],
            "blind_start_frame": entry["start"],
            "blind_end_frame": entry["end"],
            "second_pass_start_frame": "",
            "second_pass_end_frame": "",
            "start_delta_frames": "",
            "end_delta_frames": "",
            "abs_start_delta_frames": "",
            "abs_end_delta_frames": "",
            "second_pass_rep_count_delta": "",
            "boundary_confidence_second": "",
            "second_pass_status": status,
            "second_pass_annotator": row.get("second_pass_annotator", "").strip(),
            "notes": row.get("notes", "").strip(),
        }
        if status == "pending":
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "second_pass_pending", "detail": sample_id}
            )
            enriched_rows.append(enriched)
            continue
        second_start = parse_int(row.get("second_pass_start_frame", ""), "second_pass_start_frame", segment_id, issue_rows)
        second_end = parse_int(row.get("second_pass_end_frame", ""), "second_pass_end_frame", segment_id, issue_rows)
        count_delta = parse_int(
            row.get("second_pass_rep_count_delta", ""), "second_pass_rep_count_delta", segment_id, issue_rows
        )
        confidence = row.get("boundary_confidence_second", "").strip().lower()
        if second_start is None or second_end is None or count_delta is None:
            enriched_rows.append(enriched)
            continue
        if second_start > second_end:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "second_pass_start_after_end",
                    "detail": f"{second_start}>{second_end}",
                }
            )
        if not segment["active_start_frame"] <= second_start <= second_end <= segment["active_end_frame"]:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "second_pass_outside_active_window",
                    "detail": f"{second_start}-{second_end}",
                }
            )
        if confidence not in VALID_BOUNDARY_CONFIDENCES:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "invalid_second_pass_confidence",
                    "detail": confidence,
                }
            )
        if not enriched["second_pass_annotator"]:
            issue_rows.append(
                {"segment_id": segment_id, "rep_id": str(rep_id), "issue": "missing_second_pass_annotator", "detail": ""}
            )
        previous_delta = count_deltas_by_segment.get(segment_id)
        if previous_delta is not None and previous_delta != count_delta:
            issue_rows.append(
                {
                    "segment_id": segment_id,
                    "rep_id": str(rep_id),
                    "issue": "inconsistent_second_pass_count_delta",
                    "detail": f"{previous_delta} vs {count_delta}",
                }
            )
        count_deltas_by_segment[segment_id] = count_delta
        start_delta = second_start - entry["start"]
        end_delta = second_end - entry["end"]
        enriched.update(
            {
                "second_pass_start_frame": second_start,
                "second_pass_end_frame": second_end,
                "start_delta_frames": start_delta,
                "end_delta_frames": end_delta,
                "abs_start_delta_frames": abs(start_delta),
                "abs_end_delta_frames": abs(end_delta),
                "second_pass_rep_count_delta": count_delta,
                "boundary_confidence_second": confidence,
            }
        )
        completed_keys.add(key)
        enriched_rows.append(enriched)

    eligible_count = len(entries)
    sample_ratio = len(sample_keys & set(entries)) / eligible_count if eligible_count else 0.0
    completed_ratio = len(completed_keys) / eligible_count if eligible_count else 0.0
    if sample_ratio < min_ratio:
        issue_rows.append(
            {
                "segment_id": "",
                "rep_id": "",
                "issue": "second_pass_sample_below_minimum",
                "detail": f"{sample_ratio:.3f} < {min_ratio:.3f}",
            }
        )
    if completed_ratio < min_ratio:
        issue_rows.append(
            {
                "segment_id": "",
                "rep_id": "",
                "issue": "second_pass_completed_below_minimum",
                "detail": f"{completed_ratio:.3f} < {min_ratio:.3f}",
            }
        )
    completed_rows = [
        row for row in enriched_rows
        if row["second_pass_status"] == "complete" and row["start_delta_frames"] != ""
    ]
    boundary_abs_deltas = [
        value
        for row in completed_rows
        for value in (row["abs_start_delta_frames"], row["abs_end_delta_frames"])
    ]
    annotated_by = sorted({row["second_pass_annotator"] for row in completed_rows if row["second_pass_annotator"]})
    independent = bool(annotated_by) and not any(
        marker in annotator.lower()
        for annotator in annotated_by
        for marker in ("same_annotator", "same-annotator", "self_recheck")
    )
    summary = {
        "sample_path": str(path),
        "samples": len(enriched_rows),
        "eligible_reps": eligible_count,
        "sample_ratio": sample_ratio,
        "completed": len(completed_rows),
        "completed_ratio": completed_ratio,
        "pending": len(enriched_rows) - len(completed_rows),
        "second_pass_status": "complete" if completed_ratio == 1.0 else "partial",
        "second_pass_annotators": ";".join(annotated_by),
        "independent_annotation_claimed": independent,
        "mean_abs_start_delta_frames": float(np.mean([row["abs_start_delta_frames"] for row in completed_rows])) if completed_rows else 0.0,
        "mean_abs_end_delta_frames": float(np.mean([row["abs_end_delta_frames"] for row in completed_rows])) if completed_rows else 0.0,
        "max_abs_start_delta_frames": max((row["abs_start_delta_frames"] for row in completed_rows), default=0),
        "max_abs_end_delta_frames": max((row["abs_end_delta_frames"] for row in completed_rows), default=0),
        "boundary_exact_agreement": float(np.mean([value == 0 for value in boundary_abs_deltas])) if boundary_abs_deltas else 0.0,
        "boundary_within_3f_agreement": float(np.mean([value <= 3 for value in boundary_abs_deltas])) if boundary_abs_deltas else 0.0,
        "segments_with_count_delta": sum(1 for value in count_deltas_by_segment.values() if value != 0),
        "total_abs_count_delta": sum(abs(value) for value in count_deltas_by_segment.values()),
    }
    return enriched_rows, summary


def load_predictions(
    path: Path,
    method: str,
    included_segments: set[str],
) -> dict[str, list[tuple[int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in read_csv(path):
        if row["method"] != method:
            continue
        segment_id = f"{row['video_id'].zfill(4)}_{row['action']}"
        if segment_id not in included_segments:
            continue
        grouped[segment_id].append((int(row["start_frame"]), int(row["end_frame"]), int(float(row["rep_id"]))))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def classify_false_positive(
    pred: tuple[int, int, int],
    gt: list[tuple[int, int, int]],
    matched_gt: set[int],
    exclusions: list[dict[str, Any]],
) -> tuple[str, str, str]:
    pred_interval = (pred[0], pred[1])
    best_gt_index: int | None = None
    best_gt_overlap = 0
    best_gt_iou = 0.0
    for index, item in enumerate(gt):
        overlap = interval_overlap(pred_interval, (item[0], item[1]))
        if overlap > best_gt_overlap:
            best_gt_index = index
            best_gt_overlap = overlap
        best_gt_iou = max(best_gt_iou, interval_iou(pred_interval, (item[0], item[1])))
    if best_gt_index is not None and best_gt_index in matched_gt:
        return "duplicate", str(gt[best_gt_index][2]), f"overlaps matched GT rep by {best_gt_overlap} frames"
    best_reason = "unexplained"
    best_overlap = 0
    best_note = ""
    for exclusion in exclusions:
        overlap = interval_overlap(pred_interval, (exclusion["start"], exclusion["end"]))
        if overlap > best_overlap:
            best_reason = exclusion["exclude_reason"]
            best_overlap = overlap
            best_note = exclusion.get("notes", "")
    if best_overlap > 0:
        return best_reason, "", f"overlaps exclusion by {best_overlap} frames; {best_note}".strip()
    if best_gt_index is not None and best_gt_overlap > 0:
        return "duplicate", str(gt[best_gt_index][2]), f"partial overlap with GT rep ({best_gt_iou:.3f} IoU)"
    return "unexplained", "", f"best GT IoU={best_gt_iou:.3f}"


def segmentation_error_analysis(
    segment_id: str,
    video_id: str,
    action: str,
    rep_unit: str,
    method: str,
    threshold: float,
    gt_values: list[tuple[int, int, int]],
    pred_values: list[tuple[int, int, int]],
    overlap_share: float,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    split_gt_reps = 0
    split_events = 0
    merge_pred_reps = 0
    merge_events = 0
    event_rows: list[dict[str, Any]] = []
    for gt_rep in gt_values:
        candidates = [
            pred_rep
            for pred_rep in pred_values
            if interval_overlap((gt_rep[0], gt_rep[1]), (pred_rep[0], pred_rep[1]))
            / min(gt_rep[1] - gt_rep[0] + 1, pred_rep[1] - pred_rep[0] + 1)
            >= overlap_share
        ]
        if len(candidates) >= 2:
            split_gt_reps += 1
            split_events += len(candidates) - 1
            event_rows.append(
                {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "action": action,
                    "method": method,
                    "iou_threshold": threshold,
                    "rep_unit": rep_unit,
                    "error_type": "over_segmentation",
                    "source_rep_id": gt_rep[2],
                    "source_start_frame": gt_rep[0],
                    "source_end_frame": gt_rep[1],
                    "overlapping_rep_ids": ";".join(str(item[2]) for item in candidates),
                    "overlapping_count": len(candidates),
                    "event_count": len(candidates) - 1,
                    "overlap_share": overlap_share,
                }
            )
    for pred_rep in pred_values:
        candidates = [
            gt_rep
            for gt_rep in gt_values
            if interval_overlap((pred_rep[0], pred_rep[1]), (gt_rep[0], gt_rep[1]))
            / min(gt_rep[1] - gt_rep[0] + 1, pred_rep[1] - pred_rep[0] + 1)
            >= overlap_share
        ]
        if len(candidates) >= 2:
            merge_pred_reps += 1
            merge_events += len(candidates) - 1
            event_rows.append(
                {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "action": action,
                    "method": method,
                    "iou_threshold": threshold,
                    "rep_unit": rep_unit,
                    "error_type": "under_segmentation",
                    "source_rep_id": pred_rep[2],
                    "source_start_frame": pred_rep[0],
                    "source_end_frame": pred_rep[1],
                    "overlapping_rep_ids": ";".join(str(item[2]) for item in candidates),
                    "overlapping_count": len(candidates),
                    "event_count": len(candidates) - 1,
                    "overlap_share": overlap_share,
                }
            )
    return {
        "over_segmented_gt_reps": split_gt_reps,
        "split_events": split_events,
        "under_segmented_pred_reps": merge_pred_reps,
        "merge_events": merge_events,
    }, event_rows


def evaluate_threshold(
    gt: dict[str, list[tuple[int, int, int]]],
    predictions: dict[str, list[tuple[int, int, int]]],
    rep_units: dict[str, str],
    method: str,
    threshold: float,
    exclusions: dict[str, list[dict[str, Any]]],
    overlap_share: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    segmentation_error_rows: list[dict[str, Any]] = []
    for segment_id in sorted(gt):
        video_id, action = segment_id.split("_", 1)
        rep_unit = rep_units[segment_id]
        gt_values = gt[segment_id]
        pred_values = predictions.get(segment_id, [])
        raw = match_with_pairs(gt_values, pred_values, threshold)
        aligned, shift = phase_aligned_match(gt_values, pred_values, threshold)
        matched = len(raw["pairs"])
        false_positives = len(pred_values) - matched
        false_negatives = len(gt_values) - matched
        precision = matched / len(pred_values) if pred_values else 0.0
        recall = matched / len(gt_values) if gt_values else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        signed_start_errors = []
        signed_end_errors = []
        boundary_errors = []
        for gt_index, pred_index, _ in raw["pairs"]:
            start_error = pred_values[pred_index][0] - gt_values[gt_index][0]
            end_error = pred_values[pred_index][1] - gt_values[gt_index][1]
            signed_start_errors.append(start_error)
            signed_end_errors.append(end_error)
            boundary_errors.extend([abs(start_error), abs(end_error)])
        segmentation_counts, segment_event_rows = segmentation_error_analysis(
            segment_id,
            video_id,
            action,
            rep_unit,
            method,
            threshold,
            gt_values,
            pred_values,
            overlap_share,
        )
        segmentation_error_rows.extend(segment_event_rows)
        gt_count = len(gt_values)
        pred_count = len(pred_values)
        metric_rows.append(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "action": action,
                "rep_unit": rep_unit,
                "method": method,
                "iou_threshold": threshold,
                "gt_reps": gt_count,
                "pred_reps": pred_count,
                "count_error": pred_count - gt_count,
                "matched": matched,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou": raw["mean_iou"],
                "phase_aligned_iou": aligned["mean_iou"],
                "phase_shift_frames": shift,
                "mean_signed_start_error_frames": float(np.mean(signed_start_errors)) if signed_start_errors else 0.0,
                "mean_signed_end_error_frames": float(np.mean(signed_end_errors)) if signed_end_errors else 0.0,
                "mean_abs_start_error_frames": float(np.mean(np.abs(signed_start_errors))) if signed_start_errors else 0.0,
                "mean_abs_end_error_frames": float(np.mean(np.abs(signed_end_errors))) if signed_end_errors else 0.0,
                "mean_boundary_error_frames": float(np.mean(boundary_errors)) if boundary_errors else 0.0,
                "p95_boundary_error_frames": float(np.percentile(boundary_errors, 95)) if boundary_errors else 0.0,
                "boundary_error_count": len(boundary_errors),
                "boundary_error_sum_frames": float(sum(boundary_errors)),
                "p95_abs_start_error_frames": float(np.percentile(np.abs(signed_start_errors), 95)) if signed_start_errors else 0.0,
                "p95_abs_end_error_frames": float(np.percentile(np.abs(signed_end_errors), 95)) if signed_end_errors else 0.0,
                "boundary_errors": boundary_errors,
                "signed_start_errors": signed_start_errors,
                "signed_end_errors": signed_end_errors,
                "over_segmented_gt_reps": segmentation_counts["over_segmented_gt_reps"],
                "over_segmentation_rate": segmentation_counts["over_segmented_gt_reps"] / gt_count if gt_count else 0.0,
                "split_events": segmentation_counts["split_events"],
                "under_segmented_pred_reps": segmentation_counts["under_segmented_pred_reps"],
                "under_segmentation_rate": segmentation_counts["under_segmented_pred_reps"] / pred_count if pred_count else 0.0,
                "merge_events": segmentation_counts["merge_events"],
            }
        )
        matched_gt = {pair[0] for pair in raw["pairs"]}
        for pred_index in raw["unmatched_pred"]:
            pred = pred_values[pred_index]
            fp_type, gt_rep_id, note = classify_false_positive(
                pred,
                gt_values,
                matched_gt,
                exclusions.get(segment_id, []),
            )
            false_positive_rows.append(
                {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "action": action,
                    "rep_unit": rep_unit,
                    "method": method,
                    "iou_threshold": threshold,
                    "pred_rep_id": pred[2],
                    "pred_start_frame": pred[0],
                    "pred_end_frame": pred[1],
                    "overlaps_gt_rep_id": gt_rep_id,
                    "false_positive_type": fp_type,
                    "notes": note,
                }
            )
    return metric_rows, false_positive_rows, segmentation_error_rows


def summarize(
    rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    segmentation_error_rows: list[dict[str, Any]],
    method: str,
    threshold: float,
    fps: float,
    rep_unit: str | None = None,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["method"] == method
        and row["iou_threshold"] == threshold
        and (rep_unit is None or row["rep_unit"] == rep_unit)
    ]
    selected_fp = [
        row for row in false_positive_rows
        if row["method"] == method and row["iou_threshold"] == threshold
        and (rep_unit is None or row["rep_unit"] == rep_unit)
    ]
    selected_segmentation_errors = [
        row for row in segmentation_error_rows
        if row["method"] == method and row["iou_threshold"] == threshold
        and (rep_unit is None or row["rep_unit"] == rep_unit)
    ]
    gt_total = sum(row["gt_reps"] for row in selected)
    pred_total = sum(row["pred_reps"] for row in selected)
    tp = sum(row["matched"] for row in selected)
    fp = sum(row["false_positives"] for row in selected)
    fn = sum(row["false_negatives"] for row in selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fp_counts = Counter(row["false_positive_type"] for row in selected_fp)
    weighted_iou = sum(row["mean_iou"] * row["matched"] for row in selected) / tp if tp else 0.0
    weighted_aligned = sum(row["phase_aligned_iou"] * row["matched"] for row in selected) / tp if tp else 0.0
    boundary_errors = [error for row in selected for error in row["boundary_errors"]]
    signed_start_errors = [error for row in selected for error in row["signed_start_errors"]]
    signed_end_errors = [error for row in selected for error in row["signed_end_errors"]]
    abs_start_errors = [abs(error) for error in signed_start_errors]
    abs_end_errors = [abs(error) for error in signed_end_errors]
    weighted_boundary_error = float(np.mean(boundary_errors)) if boundary_errors else 0.0
    over_segmented_gt_reps = sum(row["over_segmented_gt_reps"] for row in selected)
    split_events = sum(row["split_events"] for row in selected)
    under_segmented_pred_reps = sum(row["under_segmented_pred_reps"] for row in selected)
    merge_events = sum(row["merge_events"] for row in selected)
    return {
        "method": method,
        "iou_threshold": threshold,
        "rep_unit": rep_unit or "all",
        "segments": len(selected),
        "gt_reps": gt_total,
        "pred_reps": pred_total,
        "count_mae": float(np.mean([abs(row["count_error"]) for row in selected])) if selected else 0.0,
        "matched": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_precision": float(np.mean([row["precision"] for row in selected])) if selected else 0.0,
        "macro_recall": float(np.mean([row["recall"] for row in selected])) if selected else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in selected])) if selected else 0.0,
        "mean_iou": weighted_iou,
        "micro_iou": weighted_iou,
        "macro_iou": float(np.mean([row["mean_iou"] for row in selected])) if selected else 0.0,
        "phase_aligned_iou": weighted_aligned,
        "micro_phase_aligned_iou": weighted_aligned,
        "macro_phase_aligned_iou": float(np.mean([row["phase_aligned_iou"] for row in selected])) if selected else 0.0,
        "mean_signed_start_error_frames": float(np.mean(signed_start_errors)) if signed_start_errors else 0.0,
        "mean_signed_end_error_frames": float(np.mean(signed_end_errors)) if signed_end_errors else 0.0,
        "mean_abs_start_error_frames": float(np.mean(abs_start_errors)) if abs_start_errors else 0.0,
        "mean_abs_end_error_frames": float(np.mean(abs_end_errors)) if abs_end_errors else 0.0,
        "p95_abs_start_error_frames": float(np.percentile(abs_start_errors, 95)) if abs_start_errors else 0.0,
        "p95_abs_end_error_frames": float(np.percentile(abs_end_errors, 95)) if abs_end_errors else 0.0,
        "mean_boundary_error_frames": weighted_boundary_error,
        "macro_boundary_error_frames": float(np.mean([row["mean_boundary_error_frames"] for row in selected])) if selected else 0.0,
        "mean_boundary_error_s": weighted_boundary_error / fps,
        "p95_boundary_error_frames": float(np.percentile(boundary_errors, 95)) if boundary_errors else 0.0,
        "boundary_error_count": len(boundary_errors),
        "over_segmented_gt_reps": over_segmented_gt_reps,
        "over_segmentation_rate": over_segmented_gt_reps / gt_total if gt_total else 0.0,
        "split_events": split_events,
        "under_segmented_pred_reps": under_segmented_pred_reps,
        "under_segmentation_rate": under_segmented_pred_reps / pred_total if pred_total else 0.0,
        "merge_events": merge_events,
        "segmentation_error_rows": len(selected_segmentation_errors),
        "fp_rest": fp_counts.get("rest", 0),
        "fp_transition": fp_counts.get("transition", 0),
        "fp_incomplete_rep": fp_counts.get("incomplete_rep", 0),
        "fp_duplicate": fp_counts.get("duplicate", 0),
        "fp_unexplained": fp_counts.get("unexplained", 0),
    }


def load_first_round(
    path: Path,
    included_segments: set[str],
) -> dict[str, list[tuple[int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in read_csv(path):
        segment_id = row["segment_id"]
        if segment_id not in included_segments:
            continue
        status = row.get("review_status", "").strip().lower()
        if status not in {"accept", "adjust", "add"}:
            continue
        grouped[segment_id].append(
            (int(row["gt_start_frame"]), int(row["gt_end_frame"]), int(float(row["rep_id"])))
        )
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def compare_first_round(
    blind: dict[str, list[tuple[int, int, int]]],
    first_round: dict[str, list[tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    rows = []
    for segment_id in sorted(set(blind) & set(first_round)):
        reference = first_round[segment_id]
        candidate = blind[segment_id]
        result = match_with_pairs(reference, candidate, 0.10)
        start_errors = [candidate[pred][0] - reference[gt][0] for gt, pred, _ in result["pairs"]]
        end_errors = [candidate[pred][1] - reference[gt][1] for gt, pred, _ in result["pairs"]]
        strict_matches = sum(1 for _, _, iou in result["pairs"] if iou >= 0.50)
        rows.append(
            {
                "segment_id": segment_id,
                "first_round_reps": len(reference),
                "blind_reps": len(candidate),
                "count_delta": len(candidate) - len(reference),
                "matched_iou_010": len(result["pairs"]),
                "matched_iou_050": strict_matches,
                "mean_iou": result["mean_iou"],
                "mean_signed_start_error_frames": float(np.mean(start_errors)) if start_errors else 0.0,
                "mean_signed_end_error_frames": float(np.mean(end_errors)) if end_errors else 0.0,
                "mean_abs_start_error_frames": float(np.mean(np.abs(start_errors))) if start_errors else 0.0,
                "mean_abs_end_error_frames": float(np.mean(np.abs(end_errors))) if end_errors else 0.0,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    windows = load_windows(args.segments)
    issue_rows: list[dict[str, Any]] = []
    segments = load_blind_segments(args.blind_dir / "blind_segments.csv", windows, issue_rows)
    gt, all_rep_rows, status_counts = load_blind_reps(
        args.blind_dir / "blind_reps.csv", segments, issue_rows
    )
    exclusions = load_exclusions(
        args.blind_dir / "blind_exclusions.csv", segments, gt, issue_rows
    )
    included_segments = {
        segment_id for segment_id, segment in segments.items() if segment["included_in_eval"]
    }
    rep_units = {
        segment_id: segment["rep_unit"]
        for segment_id, segment in segments.items()
        if segment_id in included_segments
    }
    second_pass_rows, second_pass_summary = load_second_pass_sample(
        args.blind_dir / "second_pass_sample.csv",
        segments,
        all_rep_rows,
        included_segments,
        issue_rows,
        args.second_pass_min_ratio,
    )
    missing_gt_segments = included_segments - set(gt)
    for segment_id in sorted(missing_gt_segments):
        issue_rows.append(
            {"segment_id": segment_id, "rep_id": "", "issue": "included_segment_without_gt", "detail": ""}
        )

    metric_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    segmentation_error_rows: list[dict[str, Any]] = []
    for method in args.methods:
        predictions = load_predictions(args.reps, method, included_segments)
        for threshold in args.iou_thresholds:
            threshold_rows, threshold_fps, threshold_segmentation_errors = evaluate_threshold(
                gt,
                predictions,
                rep_units,
                method,
                threshold,
                exclusions,
                args.split_overlap_share,
            )
            metric_rows.extend(threshold_rows)
            false_positive_rows.extend(threshold_fps)
            segmentation_error_rows.extend(threshold_segmentation_errors)

    summary_rows = [
        summarize(
            metric_rows,
            false_positive_rows,
            segmentation_error_rows,
            method,
            threshold,
            args.fps,
        )
        for method in args.methods
        for threshold in args.iou_thresholds
    ]
    rep_unit_values = sorted(set(rep_units.values()))
    summary_by_rep_unit_rows = [
        summarize(
            metric_rows,
            false_positive_rows,
            segmentation_error_rows,
            method,
            threshold,
            args.fps,
            rep_unit=unit,
        )
        for method in args.methods
        for threshold in args.iou_thresholds
        for unit in rep_unit_values
    ]
    first_round = load_first_round(args.first_round, included_segments)
    first_round_comparison = compare_first_round(gt, first_round)

    metric_fields = [
        "segment_id", "video_id", "action", "rep_unit", "method", "iou_threshold", "gt_reps", "pred_reps",
        "count_error", "matched", "false_positives", "false_negatives", "precision", "recall", "f1",
        "mean_iou", "phase_aligned_iou", "phase_shift_frames", "mean_signed_start_error_frames",
        "mean_signed_end_error_frames", "mean_abs_start_error_frames", "mean_abs_end_error_frames",
        "mean_boundary_error_frames", "p95_boundary_error_frames", "p95_abs_start_error_frames",
        "p95_abs_end_error_frames", "boundary_error_count", "boundary_error_sum_frames",
        "over_segmented_gt_reps", "over_segmentation_rate", "split_events",
        "under_segmented_pred_reps", "under_segmentation_rate", "merge_events",
    ]
    summary_fields = [
        "method", "iou_threshold", "rep_unit", "segments", "gt_reps", "pred_reps", "count_mae",
        "matched", "false_positives", "false_negatives", "precision", "recall", "f1",
        "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1",
        "mean_iou", "micro_iou", "macro_iou", "phase_aligned_iou", "micro_phase_aligned_iou",
        "macro_phase_aligned_iou", "mean_signed_start_error_frames", "mean_signed_end_error_frames",
        "mean_abs_start_error_frames", "mean_abs_end_error_frames", "p95_abs_start_error_frames",
        "p95_abs_end_error_frames", "mean_boundary_error_s", "mean_boundary_error_frames",
        "macro_boundary_error_frames", "p95_boundary_error_frames", "boundary_error_count",
        "over_segmented_gt_reps", "over_segmentation_rate", "split_events", "under_segmented_pred_reps",
        "under_segmentation_rate", "merge_events", "segmentation_error_rows",
        "fp_rest", "fp_transition", "fp_incomplete_rep", "fp_duplicate", "fp_unexplained",
    ]
    fp_fields = [
        "segment_id", "video_id", "action", "rep_unit", "method", "iou_threshold", "pred_rep_id",
        "pred_start_frame", "pred_end_frame", "overlaps_gt_rep_id", "false_positive_type", "notes",
    ]
    segmentation_error_fields = [
        "segment_id", "video_id", "action", "method", "iou_threshold", "rep_unit", "error_type",
        "source_rep_id", "source_start_frame", "source_end_frame", "overlapping_rep_ids",
        "overlapping_count", "event_count", "overlap_share",
    ]
    comparison_fields = [
        "segment_id", "first_round_reps", "blind_reps", "count_delta", "matched_iou_010",
        "matched_iou_050", "mean_iou", "mean_signed_start_error_frames", "mean_signed_end_error_frames",
        "mean_abs_start_error_frames", "mean_abs_end_error_frames",
    ]
    second_pass_fields = [
        "sample_id", "segment_id", "video_id", "action", "rep_id", "rep_unit",
        "blind_start_frame", "blind_end_frame", "second_pass_start_frame", "second_pass_end_frame",
        "start_delta_frames", "end_delta_frames", "abs_start_delta_frames", "abs_end_delta_frames",
        "second_pass_rep_count_delta", "boundary_confidence_second", "second_pass_status",
        "second_pass_annotator", "notes",
    ]
    second_pass_summary_fields = [
        "sample_path", "samples", "eligible_reps", "sample_ratio", "completed", "completed_ratio",
        "pending", "second_pass_status", "second_pass_annotators", "independent_annotation_claimed",
        "mean_abs_start_delta_frames", "mean_abs_end_delta_frames", "max_abs_start_delta_frames",
        "max_abs_end_delta_frames", "boundary_exact_agreement", "boundary_within_3f_agreement",
        "segments_with_count_delta", "total_abs_count_delta",
    ]
    write_csv(args.blind_dir / "blind_validation.csv", issue_rows, ["segment_id", "rep_id", "issue", "detail"])
    write_csv(args.blind_dir / "blind_eval.csv", metric_rows, metric_fields)
    write_csv(args.blind_dir / "blind_eval_summary.csv", summary_rows, summary_fields)
    write_csv(args.blind_dir / "blind_eval_by_rep_unit.csv", summary_by_rep_unit_rows, summary_fields)
    write_csv(args.blind_dir / "blind_false_positives.csv", false_positive_rows, fp_fields)
    write_csv(
        args.blind_dir / "blind_segmentation_errors.csv",
        segmentation_error_rows,
        segmentation_error_fields,
    )
    write_csv(args.blind_dir / "blind_vs_first_round.csv", first_round_comparison, comparison_fields)
    write_csv(args.blind_dir / "blind_second_pass.csv", second_pass_rows, second_pass_fields)
    write_csv(
        args.blind_dir / "blind_second_pass_summary.csv",
        [second_pass_summary],
        second_pass_summary_fields,
    )

    payload = {
        "blind_reps": str(args.blind_dir / "blind_reps.csv"),
        "blind_segments": str(args.blind_dir / "blind_segments.csv"),
        "blind_exclusions": str(args.blind_dir / "blind_exclusions.csv"),
        "predictions": str(args.reps),
        "first_round": str(args.first_round),
        "included_segments": sorted(included_segments),
        "excluded_segments": sorted(set(segments) - included_segments),
        "included_rep_count": sum(len(values) for values in gt.values()),
        "rep_status_counts": dict(status_counts),
        "iou_thresholds": args.iou_thresholds,
        "primary_iou": args.primary_iou,
        "fps": args.fps,
        "split_overlap_share": args.split_overlap_share,
        "second_pass_min_ratio": args.second_pass_min_ratio,
        "second_pass": second_pass_summary,
        "summary": summary_rows,
        "summary_by_rep_unit": summary_by_rep_unit_rows,
        "first_round_comparison": first_round_comparison,
        "issues": issue_rows,
    }
    (args.blind_dir / "blind_eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(
        {
            "included_rep_count": payload["included_rep_count"],
            "included_segments": payload["included_segments"],
            "excluded_segments": payload["excluded_segments"],
            "second_pass": second_pass_summary,
            "validation_issues": len(issue_rows),
            "summary": summary_rows,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if not issue_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
