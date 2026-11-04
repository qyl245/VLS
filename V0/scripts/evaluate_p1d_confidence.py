#!/usr/bin/env python3
"""Evaluate P1-D evidence components with recall-preserving PR ablations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps as sr


DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p1d"
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_ANGLES_DIR = ROOT / "outputs" / "angles"
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "results" / "segmentation_p0c" / "annotation_blind_eval" / "blind_eval_summary.csv"
)
DEFAULT_BASELINE_EVAL = (
    ROOT / "results" / "segmentation_p0c" / "annotation_blind_eval" / "blind_eval.csv"
)
COMPONENTS = (
    "visibility",
    "shape_similarity",
    "phase_consistency",
    "motion_evidence",
)
SCORE_SPECS = (
    ("legacy_confidence", ("legacy_confidence",)),
    ("visibility", ("visibility",)),
    ("shape_similarity", ("shape_similarity",)),
    ("phase_consistency", ("phase_consistency",)),
    ("motion_evidence", ("motion_evidence",)),
    ("visibility_shape", ("visibility", "shape_similarity")),
    ("shape_phase", ("shape_similarity", "phase_consistency")),
    ("shape_motion", ("shape_similarity", "motion_evidence")),
    ("legacy_plus_motion", ("legacy_confidence", "motion_evidence")),
    ("all_four_geomean", COMPONENTS),
)
TEMPLATE_METHODS = ("mean", "medoid", "trimmed", "irls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--angles-dir", type=Path, default=DEFAULT_ANGLES_DIR)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iou-thresholds", nargs="*", type=float, default=(0.30, 0.50))
    parser.add_argument("--method", default="aifit")
    parser.add_argument("--explain-percentile", type=float, default=0.05)
    parser.add_argument("--skip-template-ablation", action="store_true")
    return parser.parse_args()


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


def load_blind_data(
    blind_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, int]]],
    dict[str, list[dict[str, Any]]],
]:
    included: dict[str, dict[str, Any]] = {}
    for row in read_csv(blind_dir / "blind_segments.csv"):
        if row.get("included_in_eval", "").strip().lower() != "true":
            continue
        segment_id = row["segment_id"]
        included[segment_id] = {
            **row,
            "window_start_frame": int(row["window_start_frame"]),
            "window_end_frame": int(row["window_end_frame"]),
            "active_start_frame": int(row["active_start_frame"]),
            "active_end_frame": int(row["active_end_frame"]),
        }
    ground_truth: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in read_csv(blind_dir / "blind_reps.csv"):
        segment_id = row["segment_id"]
        if segment_id not in included:
            continue
        if row.get("review_status", "").strip().lower() not in {"accept", "adjust", "add"}:
            continue
        ground_truth[segment_id].append(
            {
                "rep_id": int(row["rep_id"]),
                "start_frame": int(row["gt_start_frame"]),
                "end_frame": int(row["gt_end_frame"]),
            }
        )
    for values in ground_truth.values():
        values.sort(key=lambda item: item["start_frame"])
    exclusions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(blind_dir / "blind_exclusions.csv"):
        segment_id = row["segment_id"]
        if segment_id not in included:
            continue
        exclusions[segment_id].append(
            {
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "reason": row["exclude_reason"].strip().lower(),
            }
        )
    return included, ground_truth, exclusions


def load_predictions(path: Path, method: str, included: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        if row["method"] != method:
            continue
        segment_id = f"{row['video_id'].zfill(4)}_{row['action']}"
        if segment_id not in included:
            continue
        values: dict[str, Any] = {
            "segment_id": segment_id,
            "video_id": row["video_id"].zfill(4),
            "action": row["action"],
            "method": method,
            "rep_id": int(float(row["rep_id"])),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "angle": row.get("angle", ""),
            "legacy_confidence": float(row.get("confidence") or 0.0),
        }
        for component in COMPONENTS:
            values[component] = float(row.get(component) or 0.5)
        values["review_score"] = float(
            row.get("review_score")
            or score_from_components(values, COMPONENTS)
        )
        values["review_priority"] = row.get("review_priority", "")
        values["reject_reason"] = row.get("reject_reason", "")
        grouped[segment_id].append(values)
    for values in grouped.values():
        values.sort(key=lambda item: item["start_frame"])
    return grouped


def score_from_components(values: dict[str, float], names: tuple[str, ...]) -> float:
    clipped = np.clip(
        np.asarray([values[name] for name in names], dtype=np.float64),
        1e-6,
        1.0,
    )
    return float(np.exp(np.mean(np.log(clipped))))


def add_score_columns(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for score_name, names in SCORE_SPECS:
            if score_name == "legacy_confidence":
                row[score_name] = row["legacy_confidence"]
            else:
                row[score_name] = score_from_components(row, names)
        row["all_four_geomean"] = row["review_score"]


def classify_false_positive(
    prediction: dict[str, Any],
    ground_truth: list[dict[str, int]],
    matched_gt: set[int],
    exclusions: list[dict[str, Any]],
) -> tuple[str, int | None, str]:
    pred_interval = (prediction["start_frame"], prediction["end_frame"])
    overlapping_exclusions = [
        item for item in exclusions
        if interval_overlap(pred_interval, (item["start_frame"], item["end_frame"])) > 0
    ]
    if overlapping_exclusions:
        best = max(
            overlapping_exclusions,
            key=lambda item: interval_overlap(
                pred_interval, (item["start_frame"], item["end_frame"])
            ),
        )
        overlap = interval_overlap(pred_interval, (best["start_frame"], best["end_frame"]))
        return best["reason"], None, f"overlaps {best['reason']} exclusion by {overlap} frames"

    overlaps = [
        (index, interval_overlap(pred_interval, (item["start_frame"], item["end_frame"])))
        for index, item in enumerate(ground_truth)
    ]
    overlaps = [(index, overlap) for index, overlap in overlaps if overlap > 0]
    if overlaps:
        index, overlap = max(overlaps, key=lambda item: item[1])
        rep_id = ground_truth[index]["rep_id"]
        return "duplicate", rep_id, f"overlaps GT rep {rep_id} by {overlap} frames"
    return "unexplained", None, "no GT or exclusion overlap"


def label_predictions(
    predictions: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, list[dict[str, int]]],
    exclusions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> list[dict[str, Any]]:
    labeled = []
    for segment_id, rows in predictions.items():
        gt_rows = ground_truth.get(segment_id, [])
        matrix = np.asarray(
            [
                [
                    interval_iou(
                        (gt["start_frame"], gt["end_frame"]),
                        (row["start_frame"], row["end_frame"]),
                    )
                    for row in rows
                ]
                for gt in gt_rows
            ],
            dtype=np.float64,
        )
        pairs = []
        if matrix.size:
            row_indices, column_indices = linear_sum_assignment(-matrix)
            pairs = [
                (int(gt_index), int(pred_index), float(matrix[gt_index, pred_index]))
                for gt_index, pred_index in zip(row_indices, column_indices)
                if matrix[gt_index, pred_index] >= threshold
            ]
        matched_by_pred = {pred_index: (gt_index, iou) for gt_index, pred_index, iou in pairs}
        matched_gt = {gt_index for gt_index, _, _ in pairs}
        for pred_index, row in enumerate(rows):
            enriched = dict(row)
            enriched["iou_threshold"] = threshold
            match = matched_by_pred.get(pred_index)
            if match is not None:
                gt_index, iou = match
                enriched.update(
                    {
                        "label": "tp",
                        "matched_gt_rep_id": gt_rows[gt_index]["rep_id"],
                        "matched_iou": iou,
                        "fp_type": "",
                        "fp_note": "",
                    }
                )
            else:
                fp_type, gt_rep_id, note = classify_false_positive(
                    row,
                    gt_rows,
                    matched_gt,
                    exclusions.get(segment_id, []),
                )
                enriched.update(
                    {
                        "label": "fp",
                        "matched_gt_rep_id": gt_rep_id,
                        "matched_iou": 0.0,
                        "fp_type": fp_type,
                        "fp_note": note,
                    }
                )
            labeled.append(enriched)
    return labeled


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    differences = positives[:, None] - negatives[None, :]
    return float((np.sum(differences > 0) + 0.5 * np.sum(differences == 0)) / differences.size)


def score_metrics(
    rows: list[dict[str, Any]],
    score_name: str,
    target_recall: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([1 if row["label"] == "tp" else 0 for row in rows], dtype=np.int64)
    scores = np.asarray([float(row[score_name]) for row in rows], dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    points = []
    for threshold in sorted(np.unique(scores), reverse=True):
        selected = scores >= threshold
        true_positive = int(np.sum(labels[selected] == 1))
        false_positive = int(np.sum(labels[selected] == 0))
        precision = true_positive / (true_positive + false_positive) if selected.any() else 0.0
        recall = true_positive / positives if positives else 0.0
        points.append(
            {
                "score": score_name,
                "threshold": float(threshold),
                "tp": true_positive,
                "fp": false_positive,
                "precision": precision,
                "recall": recall,
                "kept": true_positive + false_positive,
                "flagged": len(rows) - true_positive - false_positive,
            }
        )
    previous_recall = 0.0
    average_precision = 0.0
    for point in points:
        if point["recall"] > previous_recall:
            average_precision += (point["recall"] - previous_recall) * point["precision"]
            previous_recall = point["recall"]
    eligible = [point for point in points if point["recall"] + 1e-12 >= target_recall]
    fixed = max(
        eligible,
        key=lambda point: (point["precision"], point["threshold"]),
    ) if eligible else None
    summary = {
        "score": score_name,
        "positives": positives,
        "negatives": negatives,
        "auc_roc": roc_auc(labels, scores),
        "average_precision": float(average_precision),
        "target_recall": target_recall,
        "fixed_recall_precision": fixed["precision"] if fixed else None,
        "fixed_recall_threshold": fixed["threshold"] if fixed else None,
        "fixed_recall_kept": fixed["kept"] if fixed else None,
        "fixed_recall_flagged": fixed["flagged"] if fixed else None,
        "fixed_recall_fp_retained": fixed["fp"] if fixed else None,
        "fixed_recall_tp_retained": fixed["tp"] if fixed else None,
    }
    return summary, points


def load_baseline_recalls(
    summary_path: Path,
    eval_path: Path,
    thresholds: list[float],
) -> tuple[dict[float, float], dict[tuple[float, str], float]]:
    overall: dict[float, float] = {}
    if summary_path.exists():
        for row in read_csv(summary_path):
            if row["method"] == "aifit" and row["rep_unit"] == "all":
                overall[round(float(row["iou_threshold"]), 6)] = float(row["recall"])
    by_segment: dict[tuple[float, str], float] = {}
    if eval_path.exists():
        for row in read_csv(eval_path):
            if row["method"] == "aifit":
                by_segment[
                    (round(float(row["iou_threshold"]), 6), row["segment_id"])
                ] = float(row["recall"])
    for threshold in thresholds:
        overall.setdefault(round(threshold, 6), 1.0)
    return overall, by_segment


def compute_template_shapes(
    predictions: dict[str, list[dict[str, Any]]],
    segments: dict[str, dict[str, Any]],
    angles_dir: Path,
) -> dict[str, dict[str, list[float]]]:
    shapes: dict[str, dict[str, list[float]]] = {}
    for segment_id, rows in predictions.items():
        if not rows:
            continue
        segment = segments[segment_id]
        angle = rows[0]["angle"]
        path = angles_dir / segment["video_id"] / "angles_wide.csv"
        if not path.exists() or not angle:
            continue
        _, wide = sr.load_wide(path)
        values = wide[angle][
            segment["window_start_frame"] : segment["window_end_frame"] + 1
        ]
        prepared = sr.prepare_signal(values)
        if prepared is None:
            continue
        offset = segment["window_start_frame"] + prepared.source_start
        rep_objects = [
            sr.Rep(
                rep_id=row["rep_id"],
                start_frame=row["start_frame"],
                end_frame=row["end_frame"],
                peak_frame=None,
                confidence=row["legacy_confidence"],
            )
            for row in rows
        ]
        segment_shapes = {}
        for method in TEMPLATE_METHODS:
            scores = sr.rep_shape_similarities(
                rep_objects,
                prepared.normalized,
                offset,
                method,
            )
            if scores is not None:
                segment_shapes[method] = scores
        shapes[segment_id] = segment_shapes
    return shapes


def add_template_score(
    rows: list[dict[str, Any]],
    template_shapes: dict[str, dict[str, list[float]]],
) -> None:
    row_index: dict[str, int] = defaultdict(int)
    for row in rows:
        segment_shapes = template_shapes.get(row["segment_id"], {})
        index = row_index[row["segment_id"]]
        row_index[row["segment_id"]] += 1
        for method, values in segment_shapes.items():
            if index >= len(values):
                continue
            replacement = {**row, "shape_similarity": values[index]}
            row[f"review_score_{method}_template"] = score_from_components(
                replacement,
                COMPONENTS,
            )


def percentile_below(values: np.ndarray, value: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values < value))


def build_fp_evidence(
    labeled: list[dict[str, Any]],
    threshold: float,
    explain_percentile: float,
) -> list[dict[str, Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_segment[row["segment_id"]].append(row)
    evidence = []
    for segment_id, rows in sorted(by_segment.items()):
        positives = [row for row in rows if row["label"] == "tp"]
        for row in rows:
            if row["label"] != "fp":
                continue
            tp_values = {
                component: np.asarray(
                    [positive[component] for positive in positives],
                    dtype=np.float64,
                )
                for component in COMPONENTS
            }
            explained = []
            item: dict[str, Any] = {
                "segment_id": segment_id,
                "iou_threshold": threshold,
                "pred_rep_id": row["rep_id"],
                "pred_start_frame": row["start_frame"],
                "pred_end_frame": row["end_frame"],
                "false_positive_type": row["fp_type"],
                "overlaps_gt_rep_id": row["matched_gt_rep_id"] or "",
                "legacy_confidence": row["legacy_confidence"],
                "review_score": row["review_score"],
                "review_priority": row["review_priority"],
                "reject_reason": row["reject_reason"],
            }
            for component in COMPONENTS:
                segment_values = np.asarray(
                    [candidate[component] for candidate in rows],
                    dtype=np.float64,
                )
                tp_percentile = (
                    percentile_below(tp_values[component], row[component])
                    if len(tp_values[component])
                    else 0.0
                )
                item[f"{component}_tp_percentile"] = tp_percentile
                item[f"{component}_tp_p05"] = (
                    float(np.percentile(tp_values[component], 5))
                    if len(tp_values[component])
                    else None
                )
                if (
                    tp_percentile <= explain_percentile
                    and item[f"{component}_tp_p05"] is not None
                    and row[component] < item[f"{component}_tp_p05"]
                ):
                    explained.append(component)
            for score_name, _ in SCORE_SPECS:
                values = np.asarray([candidate[score_name] for candidate in rows], dtype=np.float64)
                item[f"{score_name}_percentile"] = percentile_below(values, row[score_name])
                item[f"{score_name}_rank"] = int(np.sum(values < row[score_name]) + 1)
            item["explained_by"] = ";".join(explained)
            item["explained"] = bool(explained)
            evidence.append(item)
    return evidence


def load_baseline_segment_recalls(path: Path) -> dict[tuple[float, str], float]:
    values: dict[tuple[float, str], float] = {}
    if not path.exists():
        return values
    for row in read_csv(path):
        if row["method"] != "aifit":
            continue
        values[(round(float(row["iou_threshold"]), 6), row["segment_id"])] = float(row["recall"])
    return values


def build_segment_scores(
    all_labeled: dict[float, list[dict[str, Any]]],
    segment_recalls: dict[tuple[float, str], float],
) -> list[dict[str, Any]]:
    rows = []
    for threshold, labeled in all_labeled.items():
        by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in labeled:
            by_segment[row["segment_id"]].append(row)
        for segment_id, values in sorted(by_segment.items()):
            target = segment_recalls.get((round(threshold, 6), segment_id), 1.0)
            for score_name, _ in SCORE_SPECS:
                summary, _ = score_metrics(values, score_name, target)
                rows.append(
                    {
                        "segment_id": segment_id,
                        "iou_threshold": threshold,
                        **summary,
                    }
                )
    return rows


def fit_logistic(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    labels = np.asarray([1 if row["label"] == "tp" else 0 for row in rows], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    features = np.asarray([[row[name] for name in COMPONENTS] for row in rows], dtype=np.float64)
    transformed = np.log(np.clip(features, 1e-6, 1.0))
    mean = transformed.mean(axis=0)
    scale = transformed.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (transformed - mean) / scale
    design = np.column_stack([np.ones(len(rows)), normalized])

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        value = float(np.mean(np.logaddexp(0.0, logits) - labels * logits) + 0.05 * np.sum(weights[1:] ** 2))
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) / len(rows)
        gradient[1:] += 0.10 * weights[1:]
        return value, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        return None
    return result.x, mean, scale


def apply_logistic(
    rows: list[dict[str, Any]],
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[float]:
    weights, mean, scale = model
    features = np.asarray([[row[name] for name in COMPONENTS] for row in rows], dtype=np.float64)
    transformed = (np.log(np.clip(features, 1e-6, 1.0)) - mean) / scale
    design = np.column_stack([np.ones(len(rows)), transformed])
    logits = design @ weights
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def build_loso(
    labeled: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_segment[row["segment_id"]].append(row)
    fold_rows = []
    held_out_scores = []
    for segment_id in sorted(by_segment):
        train = [row for other, rows in by_segment.items() if other != segment_id for row in rows]
        test = by_segment[segment_id]
        train_positive = sum(row["label"] == "tp" for row in train)
        train_negative = sum(row["label"] == "fp" for row in train)
        test_positive = sum(row["label"] == "tp" for row in test)
        test_negative = sum(row["label"] == "fp" for row in test)
        model = fit_logistic(train)
        if model is None:
            status = "not_estimable_single_class_training"
        elif test_negative == 0:
            status = "fit_ok_but_no_test_negatives"
        elif test_positive == 0:
            status = "fit_ok_but_no_test_positives"
        else:
            status = "evaluated"
            for row, score in zip(test, apply_logistic(test, model)):
                held_out_scores.append({**row, "loso_score": score})
        fold_rows.append(
            {
                "iou_threshold": threshold,
                "held_out_segment": segment_id,
                "train_positives": train_positive,
                "train_negatives": train_negative,
                "test_positives": test_positive,
                "test_negatives": test_negative,
                "status": status,
            }
        )
    aggregate: dict[str, Any] = {
        "evaluable_folds": sum(row["status"] == "evaluated" for row in fold_rows),
        "status": "evaluated" if held_out_scores else "not_estimable",
    }
    if held_out_scores:
        target = 1.0
        summary, _ = score_metrics(held_out_scores, "loso_score", target)
        aggregate.update(summary)
    else:
        aggregate["reason"] = (
            "All prediction errors are concentrated in one segment; every LOSO fold "
            "has either a single-class training set or a single-class test set."
        )
    return fold_rows, aggregate


def main() -> int:
    args = parse_args()
    segmentation_dir = args.segmentation_dir.resolve()
    output_dir = (args.output_dir or segmentation_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({round(value, 6) for value in args.iou_thresholds})
    included, ground_truth, exclusions = load_blind_data(args.blind_dir.resolve())
    predictions = load_predictions(
        segmentation_dir / "reps.csv",
        args.method,
        set(included),
    )
    for rows in predictions.values():
        add_score_columns(rows)
    baseline_recalls, baseline_segment_recalls = load_baseline_recalls(
        args.baseline_summary.resolve(),
        args.baseline_eval.resolve(),
        thresholds,
    )

    labeled_by_threshold: dict[float, list[dict[str, Any]]] = {}
    pr_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    loso_by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        labeled = label_predictions(
            predictions,
            ground_truth,
            exclusions,
            threshold,
        )
        labeled_by_threshold[threshold] = labeled
        target_recall = baseline_recalls.get(threshold, 1.0)
        summaries = []
        for score_name, _ in SCORE_SPECS:
            summary, curve = score_metrics(labeled, score_name, target_recall)
            summaries.append(summary)
            for point in curve:
                pr_rows.append({"iou_threshold": threshold, **point})
        legacy = next(row for row in summaries if row["score"] == "legacy_confidence")
        for summary in summaries:
            summary.update(
                {
                    "iou_threshold": threshold,
                    "precision_gain_vs_legacy": (
                        summary["fixed_recall_precision"] - legacy["fixed_recall_precision"]
                        if summary["fixed_recall_precision"] is not None
                        and legacy["fixed_recall_precision"] is not None
                        else None
                    ),
                    "ap_gain_vs_legacy": summary["average_precision"] - legacy["average_precision"],
                    "auc_gain_vs_legacy": (
                        summary["auc_roc"] - legacy["auc_roc"]
                        if summary["auc_roc"] is not None and legacy["auc_roc"] is not None
                        else None
                    ),
                }
            )
            score_rows.append(summary)
        fp_rows.extend(build_fp_evidence(labeled, threshold, args.explain_percentile))
        folds, aggregate = build_loso(labeled, threshold)
        loso_rows.extend(folds)
        loso_by_threshold[str(threshold)] = aggregate

    segment_rows = build_segment_scores(
        labeled_by_threshold,
        load_baseline_segment_recalls(args.baseline_eval.resolve()),
    )

    template_rows: list[dict[str, Any]] = []
    template_validation: dict[str, Any] = {"status": "skipped"}
    if not args.skip_template_ablation:
        shapes = compute_template_shapes(predictions, included, args.angles_dir.resolve())
        template_validation = {"status": "pass", "segments": len(shapes), "max_abs_diff": {}}
        for threshold, labeled in labeled_by_threshold.items():
            add_template_score(labeled, shapes)
            target_recall = baseline_recalls.get(threshold, 1.0)
            legacy = next(
                row for row in score_rows
                if row["iou_threshold"] == threshold and row["score"] == "legacy_confidence"
            )
            for method in TEMPLATE_METHODS:
                score_name = f"review_score_{method}_template"
                if not all(score_name in row for row in labeled):
                    continue
                summary, _ = score_metrics(labeled, score_name, target_recall)
                template_rows.append(
                    {
                        "iou_threshold": threshold,
                        "template_method": method,
                        **summary,
                        "ap_gain_vs_legacy": summary["average_precision"] - legacy["average_precision"],
                    }
                )
        if any("irls" in segment_shapes for segment_shapes in shapes.values()):
            differences = []
            row_index: dict[str, int] = defaultdict(int)
            for row in labeled_by_threshold[thresholds[0]]:
                values = shapes.get(row["segment_id"], {}).get("irls")
                index = row_index[row["segment_id"]]
                row_index[row["segment_id"]] += 1
                if values is not None and index < len(values):
                    differences.append(abs(values[index] - row["shape_similarity"]))
            template_validation["max_abs_diff"]["irls"] = max(differences, default=0.0)

    acceptance: dict[str, Any] = {
        "auto_delete_enabled": False,
        "auto_delete_reason": "P1-D first pass ranks reps for review only; no rep was removed.",
    }
    for threshold in thresholds:
        threshold_scores = [row for row in score_rows if row["iou_threshold"] == threshold]
        threshold_fps = [row for row in fp_rows if row["iou_threshold"] == threshold]
        legacy = next(row for row in threshold_scores if row["score"] == "legacy_confidence")
        component_rows = [
            row for row in threshold_scores
            if row["score"] in COMPONENTS
        ]
        best_auc = max(
            component_rows,
            key=lambda row: row["auc_roc"] if row["auc_roc"] is not None else -np.inf,
        )
        best_ap = max(component_rows, key=lambda row: row["average_precision"])
        best_fixed = max(
            component_rows,
            key=lambda row: (
                row["fixed_recall_precision"]
                if row["fixed_recall_precision"] is not None
                else -np.inf
            ),
        )
        all_four = next(row for row in threshold_scores if row["score"] == "all_four_geomean")
        acceptance[str(threshold)] = {
            "remaining_fp": len(threshold_fps),
            "explained_fp": sum(bool(row["explained"]) for row in threshold_fps),
            "best_auc_component": best_auc["score"],
            "best_auc_gain_vs_legacy": best_auc["auc_gain_vs_legacy"],
            "best_ap_component": best_ap["score"],
            "best_ap_gain_vs_legacy": best_ap["ap_gain_vs_legacy"],
            "best_fixed_recall_component": best_fixed["score"],
            "best_fixed_recall_precision": best_fixed["fixed_recall_precision"],
            "best_fixed_recall_precision_gain_vs_legacy": best_fixed["precision_gain_vs_legacy"],
            "best_fixed_recall_tp_retained": best_fixed["fixed_recall_tp_retained"],
            "positives": best_fixed["positives"],
            "all_four_precision_gain_vs_legacy": all_four["precision_gain_vs_legacy"],
            "recall_at_fixed_threshold": legacy["target_recall"],
            "auto_gate_recommended": False,
        }

    summary = {
        "task": "P1-D",
        "status": "complete",
        "segmentation_dir": str(segmentation_dir),
        "blind_dir": str(args.blind_dir.resolve()),
        "method": args.method,
        "iou_thresholds": thresholds,
        "segments": len(included),
        "gt_reps": sum(len(values) for values in ground_truth.values()),
        "pred_reps": sum(len(values) for values in predictions.values()),
        "score_specs": {name: list(names) for name, names in SCORE_SPECS},
        "acceptance": acceptance,
        "loso_by_threshold": loso_by_threshold,
        "template_ablation_validation": template_validation,
        "artifacts": {
            "score_ablation": "p1d_score_ablation.csv",
            "pr_curves": "p1d_pr_curves.csv",
            "rep_scores": "p1d_rep_scores.csv",
            "fp_evidence": "p1d_fp_evidence.csv",
            "score_by_segment": "p1d_score_by_segment.csv",
            "template_ablation": "p1d_template_ablation.csv",
            "loso_folds": "p1d_loso_folds.csv",
        },
    }

    write_csv(
        output_dir / "p1d_score_ablation.csv",
        score_rows,
        list(score_rows[0]),
    )
    write_csv(output_dir / "p1d_pr_curves.csv", pr_rows, list(pr_rows[0]))
    rep_fields = [
        "iou_threshold", "segment_id", "video_id", "action", "method", "rep_id",
        "start_frame", "end_frame", "angle", "label", "matched_gt_rep_id", "matched_iou",
        "fp_type", "fp_note", "legacy_confidence", *COMPONENTS, "review_score",
        "review_priority", "reject_reason",
    ]
    for score_name, _ in SCORE_SPECS:
        if score_name not in rep_fields:
            rep_fields.append(score_name)
    write_csv(
        output_dir / "p1d_rep_scores.csv",
        [row for rows in labeled_by_threshold.values() for row in rows],
        rep_fields,
    )
    write_csv(output_dir / "p1d_fp_evidence.csv", fp_rows, list(fp_rows[0]) if fp_rows else [])
    write_csv(output_dir / "p1d_score_by_segment.csv", segment_rows, list(segment_rows[0]))
    if template_rows:
        write_csv(output_dir / "p1d_template_ablation.csv", template_rows, list(template_rows[0]))
    write_csv(output_dir / "p1d_loso_folds.csv", loso_rows, list(loso_rows[0]))
    (output_dir / "p1d_confidence_eval.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary["acceptance"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
