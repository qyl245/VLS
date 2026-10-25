#!/usr/bin/env python3
"""Evaluate task1-3 rep predictions against a manually reviewed annotation table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPS = ROOT / "results" / "segmentation" / "reps.csv"
DEFAULT_SEGMENTS = ROOT / "results" / "segmentation" / "action_segments.csv"
DEFAULT_ANNOTATIONS = ROOT / "results" / "segmentation" / "annotation" / "annotation_template.csv"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation" / "annotation"

VALID_STATUSES = {"accept", "adjust", "delete", "add", "pending"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--reps", type=Path, default=DEFAULT_REPS)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="*", default=("aifit", "peak"))
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--no-phase-align", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in [0, 1]")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = min(first[1], second[1]) - max(first[0], second[0]) + 1
    if intersection <= 0:
        return 0.0
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union)


def match_intervals(
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    threshold: float,
) -> dict[str, Any]:
    if not gt or not pred:
        return {
            "matched": 0,
            "iou_sum": 0.0,
            "mean_iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positives": len(pred),
            "false_negatives": len(gt),
            "boundary_errors": [],
        }
    matrix = np.asarray(
        [[interval_iou((left[0], left[1]), (right[0], right[1])) for right in pred] for left in gt],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(-matrix)
    pairs = []
    boundary_errors = []
    for row, column in zip(rows, columns):
        value = float(matrix[row, column])
        if value < threshold:
            continue
        pairs.append((row, column, value))
        boundary_errors.extend(
            [
                abs(gt[row][0] - pred[column][0]),
                abs(gt[row][1] - pred[column][1]),
            ]
        )
    matched = len(pairs)
    iou_sum = float(sum(item[2] for item in pairs))
    precision = float(matched / len(pred)) if pred else 0.0
    recall = float(matched / len(gt)) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "matched": matched,
        "iou_sum": iou_sum,
        "mean_iou": float(iou_sum / matched) if matched else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": len(pred) - matched,
        "false_negatives": len(gt) - matched,
        "boundary_errors": boundary_errors,
    }


def phase_aligned_match(
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    threshold: float,
) -> tuple[dict[str, Any], int, int]:
    if not gt or not pred:
        return match_intervals(gt, pred, threshold), 0, 0
    durations = [end - start + 1 for start, end, _ in gt]
    max_shift = max(3, min(90, int(round(0.5 * float(np.median(durations))))))
    best = None
    best_shift = 0
    best_abs_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        shifted = [(start + shift, end + shift, rep_id) for start, end, rep_id in pred]
        result = match_intervals(gt, shifted, threshold)
        key = (result["matched"], result["iou_sum"], -abs(shift))
        if best is None or key > best[0]:
            best = (key, result, shift)
    assert best is not None
    return best[1], int(best[2]), max_shift


def parse_int(value: str, field: str, segment_id: str, rep_id: str, issues: list[dict[str, Any]]) -> int | None:
    text = (value or "").strip()
    if not text:
        issues.append(
            {
                "segment_id": segment_id,
                "rep_id": rep_id,
                "issue": "missing_gt_frame",
                "detail": field,
            }
        )
        return None
    try:
        return int(float(text))
    except ValueError:
        issues.append(
            {
                "segment_id": segment_id,
                "rep_id": rep_id,
                "issue": "invalid_gt_frame",
                "detail": f"{field}={text}",
            }
        )
        return None


def load_gt(
    path: Path, windows: dict[tuple[str, str], tuple[int, int]]
) -> tuple[dict[tuple[str, str], list[tuple[int, int, int]]], list[dict[str, Any]], dict[str, int]]:
    rows = read_csv(path)
    issues: list[dict[str, Any]] = []
    counts = {"rows": len(rows), "pending": 0, "accepted": 0, "deleted": 0, "added": 0}
    grouped: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        segment_id = row.get("segment_id", "")
        video_id = row.get("video_id", "").zfill(4)
        action = row.get("action", "")
        rep_id_text = row.get("rep_id", "")
        status = row.get("review_status", "").strip().lower()
        if status not in VALID_STATUSES:
            issues.append(
                {
                    "segment_id": segment_id,
                    "rep_id": rep_id_text,
                    "issue": "invalid_status",
                    "detail": status,
                }
            )
            continue
        if status == "pending":
            counts["pending"] += 1
            continue
        if status == "delete":
            counts["deleted"] += 1
            continue
        start_value = row.get("gt_start_frame", "")
        end_value = row.get("gt_end_frame", "")
        if status == "accept" and (not start_value.strip() or not end_value.strip()):
            start_value = row.get("draft_start_frame", "")
            end_value = row.get("draft_end_frame", "")
        start = parse_int(start_value, "gt_start_frame", segment_id, rep_id_text, issues)
        end = parse_int(end_value, "gt_end_frame", segment_id, rep_id_text, issues)
        if start is None or end is None:
            continue
        if start > end:
            issues.append(
                {
                    "segment_id": segment_id,
                    "rep_id": rep_id_text,
                    "issue": "start_after_end",
                    "detail": f"{start}>{end}",
                }
            )
            continue
        window = windows.get((video_id, action))
        if window is None:
            issues.append(
                {
                    "segment_id": segment_id,
                    "rep_id": rep_id_text,
                    "issue": "missing_action_window",
                    "detail": f"{video_id}/{action}",
                }
            )
        elif not (window[0] <= start <= end <= window[1]):
            issues.append(
                {
                    "segment_id": segment_id,
                    "rep_id": rep_id_text,
                    "issue": "gt_outside_window",
                    "detail": f"{start}-{end} not in {window[0]}-{window[1]}",
                }
            )
        try:
            rep_id = int(float(rep_id_text))
        except ValueError:
            rep_id = len(grouped[(video_id, action)]) + 1
        grouped[(video_id, action)].append((start, end, rep_id))
        counts["accepted"] += 1
        if status == "add":
            counts["added"] += 1
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        for previous, current in zip(values, values[1:]):
            if current[0] <= previous[1]:
                issues.append(
                    {
                        "segment_id": f"{key[0]}_{key[1]}",
                        "rep_id": f"{previous[2]},{current[2]}",
                        "issue": "overlapping_gt",
                        "detail": f"{previous[0]}-{previous[1]} and {current[0]}-{current[1]}",
                    }
                )
    return grouped, issues, counts


def load_windows(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    rows = read_csv(path)
    return {
        (row["video_id"].zfill(4), row["action"]): (int(row["start_frame"]), int(row["end_frame"]))
        for row in rows
    }


def load_predictions(path: Path, method: str) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for row in read_csv(path):
        if row["method"] != method:
            continue
        key = (row["video_id"].zfill(4), row["action"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"]), int(float(row["rep_id"]))))
    for values in grouped.values():
        values.sort(key=lambda item: item[2])
    return grouped


def metric_row(
    video_id: str,
    action: str,
    method: str,
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    raw: dict[str, Any],
    aligned: dict[str, Any],
    shift: int,
    max_shift: int,
) -> dict[str, Any]:
    boundary_errors = raw["boundary_errors"]
    return {
        "video_id": video_id,
        "action": action,
        "segment_id": f"{video_id}_{action}",
        "method": method,
        "gt_reps": len(gt),
        "pred_reps": len(pred),
        "count_error": len(pred) - len(gt),
        "matched": raw["matched"],
        "false_positives": raw["false_positives"],
        "false_negatives": raw["false_negatives"],
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        "mean_iou": raw["mean_iou"],
        "phase_aligned_iou": aligned["mean_iou"],
        "phase_shift_frames": shift,
        "max_phase_shift_frames": max_shift,
        "mean_boundary_error_frames": float(np.mean(boundary_errors)) if boundary_errors else 0.0,
        "mean_boundary_error_s": float(np.mean(boundary_errors) / 30.0) if boundary_errors else 0.0,
    }


def summarize(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    gt_total = sum(int(row["gt_reps"]) for row in method_rows)
    pred_total = sum(int(row["pred_reps"]) for row in method_rows)
    tp = sum(int(row["matched"]) for row in method_rows)
    fp = sum(int(row["false_positives"]) for row in method_rows)
    fn = sum(int(row["false_negatives"]) for row in method_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou_sum = sum(float(row["mean_iou"]) * int(row["matched"]) for row in method_rows)
    aligned_sum = sum(float(row["phase_aligned_iou"]) * int(row["matched"]) for row in method_rows)
    return {
        "method": method,
        "segments": len(method_rows),
        "gt_reps": gt_total,
        "pred_reps": pred_total,
        "count_mae": float(np.mean([abs(int(row["count_error"])) for row in method_rows])) if method_rows else 0.0,
        "matched": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": iou_sum / tp if tp else 0.0,
        "phase_aligned_iou": aligned_sum / tp if tp else 0.0,
        "mean_boundary_error_s": float(np.mean([float(row["mean_boundary_error_s"]) for row in method_rows])) if method_rows else 0.0,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows(args.segments)
    gt, issues, counts = load_gt(args.annotations, windows)
    labeled_keys = set(gt)
    predictions = {method: load_predictions(args.reps, method) for method in args.methods}
    rows: list[dict[str, Any]] = []
    for key in sorted(labeled_keys):
        video_id, action = key
        gt_values = gt[key]
        for method in args.methods:
            pred_values = predictions[method].get(key, [])
            raw = match_intervals(gt_values, pred_values, args.iou_threshold)
            if args.no_phase_align:
                aligned = raw
                shift = 0
                max_shift = 0
            else:
                aligned, shift, max_shift = phase_aligned_match(gt_values, pred_values, args.iou_threshold)
            rows.append(metric_row(video_id, action, method, gt_values, pred_values, raw, aligned, shift, max_shift))
    rows.sort(key=lambda row: (row["video_id"], row["action"], row["method"]))
    summary_rows = [summarize(rows, method) for method in args.methods]
    write_csv(
        args.output_dir / "annotation_eval.csv",
        rows,
        [
            "video_id",
            "action",
            "segment_id",
            "method",
            "gt_reps",
            "pred_reps",
            "count_error",
            "matched",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "mean_iou",
            "phase_aligned_iou",
            "phase_shift_frames",
            "max_phase_shift_frames",
            "mean_boundary_error_frames",
            "mean_boundary_error_s",
        ],
    )
    write_csv(
        args.output_dir / "annotation_eval_summary.csv",
        summary_rows,
        [
            "method",
            "segments",
            "gt_reps",
            "pred_reps",
            "count_mae",
            "matched",
            "precision",
            "recall",
            "f1",
            "mean_iou",
            "phase_aligned_iou",
            "mean_boundary_error_s",
        ],
    )
    write_csv(
        args.output_dir / "annotation_validation.csv",
        issues,
        ["segment_id", "rep_id", "issue", "detail"],
    )
    payload = {
        "annotations": str(args.annotations),
        "reps": str(args.reps),
        "iou_threshold": args.iou_threshold,
        "phase_aligned": not args.no_phase_align,
        "counts": counts,
        "labeled_segments": len(labeled_keys),
        "summary": summary_rows,
        "segments": rows,
        "issues": issues,
    }
    (args.output_dir / "annotation_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({"labeled_segments": len(labeled_keys), "counts": counts, "summary": summary_rows}, ensure_ascii=False, indent=2))
    if issues:
        print(f"validation_issues={len(issues)} -> {args.output_dir / 'annotation_validation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
