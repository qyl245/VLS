#!/usr/bin/env python3
"""Compare one task1-4 clip's rep boundaries with the frozen blind annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results" / "clip_analysis_v1"
DEFAULT_BLIND_REPS = (
    ROOT
    / "results"
    / "segmentation_final"
    / "data"
    / "segmentation"
    / "annotation_blind_eval"
    / "blind_reps.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "clip_analysis_v1" / "evaluation_0161_vs_blind.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--clip-id", default="0161_pushup_set01")
    parser.add_argument("--segment-id", default="0161_pushup")
    parser.add_argument("--blind-reps", type=Path, default=DEFAULT_BLIND_REPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match-iou", type=float, default=0.30)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return intersection / union if union > 0 else 0.0


def greedy_match(
    predictions: list[tuple[int, int]],
    ground_truth: list[tuple[int, int]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(ground_truth):
            score = iou(pred, gt)
            if score >= threshold:
                candidates.append((score, pred_index, gt_index))
    matched_predictions = set()
    matched_ground_truth = set()
    matches = []
    for score, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index, score))
    return sorted(matches)


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    analysis = json.loads((analysis_dir / args.clip_id / "analysis.json").read_text())
    predictions = [
        (int(rep["start_frame"]), int(rep["end_frame"]))
        for rep in analysis["segmentation"]["reps"]
    ]
    ground_truth = [
        (int(row["gt_start_frame"]), int(row["gt_end_frame"]))
        for row in read_csv(args.blind_reps)
        if row["segment_id"] == args.segment_id and row.get("review_status") == "accept"
    ]
    matches = greedy_match(predictions, ground_truth, args.match_iou)
    paired_errors = []
    if len(predictions) == len(ground_truth):
        for pred, gt in zip(predictions, ground_truth):
            paired_errors.append(
                {
                    "prediction": {"start_frame": pred[0], "end_frame": pred[1]},
                    "ground_truth": {"start_frame": gt[0], "end_frame": gt[1]},
                    "iou": round(iou(pred, gt), 6),
                    "start_error_frames": pred[0] - gt[0],
                    "end_error_frames": pred[1] - gt[1],
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "task1-4b-blind-eval-v1",
        "clip_id": args.clip_id,
        "segment_id": args.segment_id,
        "prediction_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "match_iou_threshold": args.match_iou,
        "matched_count": len(matches),
        "matched_iou_mean": round(float(sum(item[2] for item in matches) / len(matches)), 6)
        if matches
        else None,
        "count_matched_pairing": paired_errors,
        "mean_abs_start_error_frames": (
            round(
                float(
                    sum(abs(item["start_error_frames"]) for item in paired_errors)
                    / len(paired_errors)
                ),
                6,
            )
            if paired_errors
            else None
        ),
        "mean_abs_end_error_frames": (
            round(
                float(
                    sum(abs(item["end_error_frames"]) for item in paired_errors)
                    / len(paired_errors)
                ),
                6,
            )
            if paired_errors
            else None
        ),
        "analysis_method": analysis["segmentation"].get("method"),
        "quality_status": analysis["quality_status"],
        "notes": "Index pairing is used only when prediction and GT counts match.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
