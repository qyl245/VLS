#!/usr/bin/env python3
"""Run synthetic tests and the frozen P0-D2 plateau-snap regression."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "configs" / "p0d2_golden.json"
DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p0d2"
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
EVALUATION_SCRIPT = ROOT / "scripts" / "evaluate_blind_annotations.py"
ANNOTATION_FILES = (
    "blind_reps.csv",
    "blind_segments.csv",
    "blind_exclusions.csv",
    "second_pass_sample.csv",
)
FLOAT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"{label} failed with exit code {result.returncode}\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result


def add_difference(
    differences: list[dict[str, Any]],
    scope: str,
    key: str,
    expected: Any,
    actual: Any,
) -> None:
    differences.append(
        {
            "scope": scope,
            "key": key,
            "expected": expected,
            "actual": actual,
        }
    )


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= FLOAT_TOLERANCE
    return expected == actual


def read_eval_summary(path: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    integer_fields = {
        "gt_reps",
        "pred_reps",
        "matched",
        "false_positives",
        "false_negatives",
        "split_events",
        "merge_events",
        "fp_transition",
        "fp_duplicate",
    }
    for row in read_csv(path):
        if row["method"] != "aifit" or row["rep_unit"] != "all":
            continue
        threshold = f"{float(row['iou_threshold']):g}"
        metrics: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"method", "iou_threshold", "rep_unit"}:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            metrics[key] = int(number) if key in integer_fields else number
        summary[threshold] = metrics
    return summary


def snap_statuses(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(run_path.read_text())
    statuses: dict[str, list[dict[str, Any]]] = {}
    for segment in payload.get("segments", []):
        key = f"{segment['video_id']}/{segment['action']}"
        values = []
        for bout in segment.get("active_bouts", []):
            evidence = bout.get("evidence", {}).get("phase_guidance") or {}
            snap = evidence.get("boundary_snap")
            if snap:
                values.append(
                    {
                        "status": snap.get("status"),
                        "reason": snap.get("reason"),
                        "changed_boundaries": snap.get("changed_boundaries", 0),
                        "trailing_incomplete_protected": snap.get(
                            "trailing_incomplete_protected", False
                        ),
                    }
                )
        if values:
            statuses[key] = values
    return statuses


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = min(first[1], second[1]) - max(first[0], second[0]) + 1
    if intersection <= 0:
        return 0.0
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union)


def boundary_error_stats(
    reps_path: Path,
    blind_reps_path: Path,
    threshold: float = 0.30,
) -> dict[str, dict[str, float]]:
    ground_truth: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(blind_reps_path):
        ground_truth[(row["video_id"], row["action"])].append(
            (int(row["gt_start_frame"]), int(row["gt_end_frame"]))
        )
    predictions: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(reps_path):
        if row["method"] == "aifit":
            predictions[(row["video_id"], row["action"])].append(
                (int(row["start_frame"]), int(row["end_frame"]))
            )

    stats = {}
    for key, gt in ground_truth.items():
        pred = predictions.get(key, [])
        if not pred:
            continue
        matrix = np.asarray(
            [[-interval_iou(left, right) for right in pred] for left in gt],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(matrix)
        starts = []
        ends = []
        for row, column in zip(rows, columns):
            if interval_iou(gt[row], pred[column]) < threshold:
                continue
            starts.append(pred[column][0] - gt[row][0])
            ends.append(pred[column][1] - gt[row][1])
        if not starts:
            continue
        stats[f"{key[0]}/{key[1]}"] = {
            "matched": len(starts),
            "start_error_mean_frames": float(np.mean(starts)),
            "start_error_std_frames": float(np.std(starts)),
            "end_error_mean_frames": float(np.mean(ends)),
            "end_error_std_frames": float(np.std(ends)),
        }
    return stats


def main() -> int:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    differences: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "pass",
        "acceptance_status": "pass",
        "golden": str(args.golden.resolve()),
        "synthetic_tests": "skipped",
    }

    if not args.skip_synthetic:
        result = run_command(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_segment_reps.py")],
            "synthetic tests",
        )
        report["synthetic_tests"] = "pass" if result.returncode == 0 else "fail"
        if result.returncode != 0:
            add_difference(differences, "synthetic_tests", "pytest", "pass", "fail")

    with tempfile.TemporaryDirectory(prefix="vls-p0d2-regression-") as temporary:
        temp_root = Path(temporary)
        segmentation_dir = args.segmentation_dir.resolve()
        if not args.use_existing:
            segmentation_dir = temp_root / "segmentation"
            result = run_command(
                [
                    sys.executable,
                    str(SEGMENT_SCRIPT),
                    "--output-dir",
                    str(segmentation_dir),
                    "--no-plots",
                    "--p2a-tail-filter",
                    "off",
                ],
                "P0-D2 segmentation",
            )
            if result.returncode != 0:
                report["status"] = "fail"
                if args.report:
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                return 1

        reps_path = segmentation_dir / "reps.csv"
        run_path = segmentation_dir / "run.json"
        aifit_count = sum(1 for row in read_csv(reps_path) if row["method"] == "aifit")
        report["aifit_reps"] = aifit_count
        if aifit_count != golden["expected_aifit_reps"]:
            add_difference(
                differences,
                "segment_counts",
                "aifit",
                golden["expected_aifit_reps"],
                aifit_count,
            )

        statuses = snap_statuses(run_path)
        report["boundary_snap"] = statuses
        for key, expected in golden["boundary_snap"].items():
            actual = statuses.get(key)
            if actual is None:
                add_difference(differences, "boundary_snap", key, expected, None)
                continue
            for index, expected_bout in enumerate(expected):
                actual_bout = actual[index] if index < len(actual) else None
                for field, expected_value in expected_bout.items():
                    actual_value = actual_bout.get(field) if actual_bout else None
                    if not values_equal(expected_value, actual_value):
                        add_difference(
                            differences,
                            "boundary_snap",
                            f"{key}/bout{index + 1}/{field}",
                            expected_value,
                            actual_value,
                        )

        eval_dir = temp_root / "blind_eval"
        eval_dir.mkdir()
        for name in ANNOTATION_FILES:
            shutil.copy2(args.blind_dir / name, eval_dir / name)
        result = run_command(
            [
                sys.executable,
                str(EVALUATION_SCRIPT),
                "--blind-dir",
                str(eval_dir),
                "--reps",
                str(reps_path),
                "--segments",
                str(segmentation_dir / "action_segments.csv"),
            ],
            "blind evaluation",
        )
        if result.returncode not in (0, 1):
            report["status"] = "fail"
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            return 1

        evaluation = read_eval_summary(eval_dir / "blind_eval_summary.csv")
        report["blind_eval"] = evaluation
        for threshold, expected_metrics in golden["blind_eval"].items():
            actual_metrics = evaluation.get(threshold)
            if actual_metrics is None:
                add_difference(differences, "blind_eval", threshold, expected_metrics, None)
                continue
            for field, expected in expected_metrics.items():
                if not values_equal(expected, actual_metrics.get(field)):
                    add_difference(
                        differences,
                        "blind_eval",
                        f"IoU={threshold}/{field}",
                        expected,
                        actual_metrics.get(field),
                    )

        error_stats = boundary_error_stats(reps_path, args.blind_dir / "blind_reps.csv")
        report["boundary_error_stats"] = error_stats
        for key, expected in golden["boundary_error_stats"].items():
            actual = error_stats.get(key)
            if actual is None:
                add_difference(differences, "boundary_error_stats", key, expected, None)
                continue
            for field, expected_value in expected.items():
                if not values_equal(expected_value, actual.get(field)):
                    add_difference(
                        differences,
                        "boundary_error_stats",
                        f"{key}/{field}",
                        expected_value,
                        actual.get(field),
                    )

        targets = {}
        for key, fields in golden.get("targets", {}).items():
            actual = error_stats.get(key, {})
            targets[key] = {}
            for field, target in fields.items():
                actual_value = actual.get(field)
                passed = actual_value is not None and float(actual_value) <= float(target) + FLOAT_TOLERANCE
                targets[key][field] = {
                    "actual": actual_value,
                    "required_max": target,
                    "pass": passed,
                }
        report["targets"] = targets
        if any(not values["pass"] for fields in targets.values() for values in fields.values()):
            report["acceptance_status"] = "partial"

    report["status"] = "pass" if not differences else "fail"
    report["difference_count"] = len(differences)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceptance_status": report["acceptance_status"],
                "synthetic_tests": report["synthetic_tests"],
                "aifit_reps": report.get("aifit_reps"),
                "difference_count": len(differences),
                "targets": report.get("targets"),
            },
            indent=2,
        )
    )
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
