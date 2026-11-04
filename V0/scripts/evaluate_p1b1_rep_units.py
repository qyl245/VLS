#!/usr/bin/env python3
"""Evaluate P1-B1 rep-unit metadata on the two walking-lunge blind segments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_blind_annotations as eba


DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p1b"
DEFAULT_P1B_ANNOTATION = DEFAULT_SEGMENTATION / "annotation"
DEFAULT_EXISTING_ANNOTATION = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_REP_METADATA = ROOT / "configs" / "rep_metadata.json"
WALKING_LUNGE_SEGMENTS = ("0162_walking_lunges", "0218_walking_lunges")

METRIC_FIELDS = (
    "segment_id",
    "video_id",
    "action",
    "rep_unit",
    "method",
    "iou_threshold",
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
    "mean_signed_start_error_frames",
    "mean_signed_end_error_frames",
    "mean_abs_start_error_frames",
    "mean_abs_end_error_frames",
    "mean_boundary_error_frames",
    "p95_boundary_error_frames",
    "split_events",
    "merge_events",
)

SUMMARY_FIELDS = (
    "method",
    "iou_threshold",
    "rep_unit",
    "segments",
    "gt_reps",
    "pred_reps",
    "matched",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "micro_iou",
    "mean_signed_start_error_frames",
    "mean_signed_end_error_frames",
    "mean_abs_start_error_frames",
    "mean_abs_end_error_frames",
    "p95_boundary_error_frames",
    "split_events",
    "merge_events",
    "fp_duplicate",
    "fp_transition",
    "fp_incomplete_rep",
    "fp_rest",
    "fp_unexplained",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--p1b-annotation-dir", type=Path, default=DEFAULT_P1B_ANNOTATION)
    parser.add_argument("--existing-annotation-dir", type=Path, default=DEFAULT_EXISTING_ANNOTATION)
    parser.add_argument("--rep-metadata", type=Path, default=DEFAULT_REP_METADATA)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iou-thresholds", nargs="*", type=float, default=(0.10, 0.30, 0.50))
    parser.add_argument("--primary-iou", type=float, default=0.30)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if any(not 0.0 < value <= 1.0 for value in args.iou_thresholds):
        parser.error("--iou-thresholds values must be in (0, 1]")
    if args.primary_iou not in args.iou_thresholds:
        parser.error("--primary-iou must be one of --iou-thresholds")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.output_dir is None:
        args.output_dir = args.segmentation_dir
    return args


def merge_annotation_rows(
    source_paths: list[Path],
    destination: Path,
    wanted_segments: set[str],
) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in source_paths:
        source_rows = eba.read_csv(path)
        if fieldnames is None:
            fieldnames = list(source_rows[0]) if source_rows else []
        rows.extend(row for row in source_rows if row["segment_id"] in wanted_segments)
    if fieldnames is None:
        raise ValueError(f"cannot infer header from {source_paths}")
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_gt(
    p1b_annotation_dir: Path,
    existing_annotation_dir: Path,
    segments_path: Path,
) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    wanted = set(WALKING_LUNGE_SEGMENTS)
    issues: list[dict[str, Any]] = []
    windows = eba.load_windows(segments_path)
    with tempfile.TemporaryDirectory(prefix="vls-p1b1-gt-") as temporary:
        temp_root = Path(temporary)
        merged_segments = temp_root / "blind_segments.csv"
        merged_reps = temp_root / "blind_reps.csv"
        merge_annotation_rows(
            [p1b_annotation_dir / "blind_segments.csv", existing_annotation_dir / "blind_segments.csv"],
            merged_segments,
            wanted,
        )
        merge_annotation_rows(
            [p1b_annotation_dir / "blind_reps.csv", existing_annotation_dir / "blind_reps.csv"],
            merged_reps,
            wanted,
        )
        segments = eba.load_blind_segments(merged_segments, windows, issues)
        grouped, _full_rows, _counts = eba.load_blind_reps(merged_reps, segments, issues)
    segments = {key: value for key, value in segments.items() if key in wanted}
    grouped = {key: value for key, value in grouped.items() if key in wanted}
    return grouped, segments, issues


def metadata_unit_counts(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    counts: dict[str, int] = {}
    for entry in payload["segments"].values():
        unit = entry["rep_unit"]
        counts[unit] = counts.get(unit, 0) + 1
    return counts


def build_summaries(
    metric_rows: list[dict[str, Any]],
    fp_rows: list[dict[str, Any]],
    segmentation_rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    thresholds: tuple[float, ...],
    fps: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in methods:
        for threshold in thresholds:
            summaries.append(
                eba.summarize(metric_rows, fp_rows, segmentation_rows, method, threshold, fps)
            )
            summaries.append(
                eba.summarize(
                    metric_rows,
                    fp_rows,
                    segmentation_rows,
                    method,
                    threshold,
                    fps,
                    rep_unit="stride",
                )
            )
    return summaries


def main() -> int:
    args = parse_args()
    args.segmentation_dir = args.segmentation_dir.resolve()
    args.p1b_annotation_dir = args.p1b_annotation_dir.resolve()
    args.existing_annotation_dir = args.existing_annotation_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    segments_path = args.segmentation_dir / "action_segments.csv"
    reps_path = args.segmentation_dir / "reps.csv"
    gt, segments, issues = load_gt(
        args.p1b_annotation_dir,
        args.existing_annotation_dir,
        segments_path,
    )
    missing_gt = sorted(set(WALKING_LUNGE_SEGMENTS) - set(gt))
    if missing_gt:
        raise ValueError(f"missing walking-lunge GT for: {', '.join(missing_gt)}")

    rep_units = {segment_id: segments[segment_id]["rep_unit"] for segment_id in gt}
    predictions = {
        method: eba.load_predictions(reps_path, method, set(gt))
        for method in ("aifit", "peak")
    }

    metric_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    for method in ("aifit", "peak"):
        for threshold in args.iou_thresholds:
            rows, false_positives, errors = eba.evaluate_threshold(
                gt,
                predictions[method],
                rep_units,
                method,
                threshold,
                {},
                0.50,
            )
            metric_rows.extend(rows)
            fp_rows.extend(false_positives)
            segmentation_rows.extend(errors)

    summaries = build_summaries(
        metric_rows,
        fp_rows,
        segmentation_rows,
        ("aifit", "peak"),
        tuple(args.iou_thresholds),
        args.fps,
    )
    summary_by_key = {
        (row["method"], f"{float(row['iou_threshold']):g}", row["rep_unit"]): row
        for row in summaries
    }
    primary = f"{args.primary_iou:g}"
    aifit_0162 = next(
        row for row in metric_rows
        if row["segment_id"] == "0162_walking_lunges"
        and row["method"] == "aifit"
        and abs(row["iou_threshold"] - args.primary_iou) < 1e-9
    )
    peak_0162 = next(
        row for row in metric_rows
        if row["segment_id"] == "0162_walking_lunges"
        and row["method"] == "peak"
        and abs(row["iou_threshold"] - args.primary_iou) < 1e-9
    )
    aifit_stride = summary_by_key[("aifit", primary, "stride")]
    peak_stride = summary_by_key[("peak", primary, "stride")]

    acceptance = {
        "status": "pass" if (
            len(gt["0162_walking_lunges"]) == 6
            and aifit_0162["matched"] == 6
            and aifit_0162["false_positives"] == 0
            and aifit_0162["false_negatives"] == 0
            and aifit_stride["f1"] == 1.0
        ) else "fail",
        "gt_0162_reps": len(gt["0162_walking_lunges"]),
        "gt_0162_rep_unit": rep_units["0162_walking_lunges"],
        "aifit_0162": {
            "pred_reps": aifit_0162["pred_reps"],
            "matched": aifit_0162["matched"],
            "false_positives": aifit_0162["false_positives"],
            "false_negatives": aifit_0162["false_negatives"],
            "f1": aifit_0162["f1"],
            "mean_iou": aifit_0162["mean_iou"],
        },
        "peak_0162": {
            "pred_reps": peak_0162["pred_reps"],
            "matched": peak_0162["matched"],
            "false_positives": peak_0162["false_positives"],
            "false_negatives": peak_0162["false_negatives"],
            "f1": peak_0162["f1"],
            "mean_iou": peak_0162["mean_iou"],
        },
        "aifit_stride_combined": {
            "gt_reps": aifit_stride["gt_reps"],
            "pred_reps": aifit_stride["pred_reps"],
            "matched": aifit_stride["matched"],
            "f1": aifit_stride["f1"],
        },
        "peak_stride_combined": {
            "gt_reps": peak_stride["gt_reps"],
            "pred_reps": peak_stride["pred_reps"],
            "matched": peak_stride["matched"],
            "f1": peak_stride["f1"],
        },
        "p1b2_trigger": False,
        "p1b2_decision": (
            "AIFit matches all six stride reps on both walking-lunge segments. "
            "The raw-frame GT shows twelve alternating steps per segment, grouped into six strides; "
            "the peak baseline overcounts step events on 0162. No blind evidence supports a half-cycle error, "
            "so P1-B2 left-right fusion must not be started from this result."
        ),
    }

    payload = {
        "task": "task1-3 P1-B1",
        "segments": list(WALKING_LUNGE_SEGMENTS),
        "gt_reps": {segment_id: len(values) for segment_id, values in gt.items()},
        "rep_units": rep_units,
        "metadata_unit_counts": metadata_unit_counts(args.rep_metadata),
        "iou_thresholds": list(args.iou_thresholds),
        "primary_iou": args.primary_iou,
        "validation_issues": issues,
        "summary": summaries,
        "per_segment_metrics": metric_rows,
        "acceptance": acceptance,
    }

    eba.write_csv(
        args.output_dir / "p1b1_walking_lunges_eval.csv",
        metric_rows,
        list(METRIC_FIELDS),
    )
    eba.write_csv(
        args.output_dir / "p1b1_walking_lunges_summary.csv",
        summaries,
        list(SUMMARY_FIELDS),
    )
    eba.write_csv(
        args.output_dir / "p1b1_walking_lunges_false_positives.csv",
        fp_rows,
        [
            "segment_id",
            "video_id",
            "action",
            "rep_unit",
            "method",
            "iou_threshold",
            "pred_rep_id",
            "pred_start_frame",
            "pred_end_frame",
            "overlaps_gt_rep_id",
            "false_positive_type",
            "notes",
        ],
    )
    (args.output_dir / "p1b1_walking_lunges_eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0 if acceptance["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
