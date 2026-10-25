#!/usr/bin/env python3
"""Run synthetic tests and the frozen P0-D1 phase-guidance regression."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "configs" / "p0d1_golden.json"
DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p0d1"
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


def read_eval_summary(path: Path) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    integer_fields = {
        "gt_reps",
        "pred_reps",
        "matched",
        "false_positives",
        "false_negatives",
        "split_events",
        "merge_events",
    }
    for row in read_csv(path):
        if row["method"] != "aifit" or row["rep_unit"] != "all":
            continue
        threshold = f"{float(row['iou_threshold']):g}"
        values: dict[str, float] = {}
        for key, value in row.items():
            if key in {"method", "iou_threshold", "rep_unit"}:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            values[key] = int(number) if key in integer_fields else number
        summary[threshold] = values
    return summary


def phase_statuses(run_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(run_path.read_text())
    statuses: dict[str, dict[str, Any]] = {}
    for segment in payload.get("segments", []):
        aifit = segment.get("angles", {}).get("aifit")
        if not aifit:
            continue
        bouts = aifit.get("phase_guidance", {}).get("bouts", [])
        key = f"{segment['video_id']}/{segment['action']}"
        statuses[key] = {
            "bouts": bouts,
            "status": (
                "applied"
                if bouts and all(bout.get("status") == "applied" for bout in bouts)
                else "fallback"
                if bouts
                else "disabled"
            ),
        }
    return statuses


def main() -> int:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    differences: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "pass",
        "golden": str(args.golden.resolve()),
        "synthetic_tests": "skipped",
        "differences": differences,
    }

    if not args.skip_synthetic:
        result = run_command(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_segment_reps.py")],
            "synthetic tests",
        )
        report["synthetic_tests"] = "pass" if result.returncode == 0 else "fail"
        if result.returncode != 0:
            add_difference(differences, "synthetic_tests", "pytest", "pass", "fail")

    with tempfile.TemporaryDirectory(prefix="vls-p0d1-regression-") as temporary:
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
                    "--no-phase-snap",
                    "--p2a-tail-filter",
                    "off",
                ],
                "P0-D1 segmentation",
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

        statuses = phase_statuses(run_path)
        report["phase_guidance"] = statuses
        for key, expected in golden["phase_guidance"].items():
            actual = statuses.get(key)
            if actual is None:
                add_difference(differences, "phase_guidance", key, expected, None)
                continue
            bouts = actual["bouts"]
            first = bouts[0] if bouts else {}
            for field in ("status", "reason", "final_low_amplitude_accepted"):
                if field in expected and not values_equal(expected[field], first.get(field)):
                    add_difference(
                        differences,
                        "phase_guidance",
                        f"{key}/{field}",
                        expected[field],
                        first.get(field),
                    )
            if "start_frame" in expected and not values_equal(
                expected["start_frame"], first.get("start_frame")
            ):
                add_difference(
                    differences,
                    "phase_guidance",
                    f"{key}/start_frame",
                    expected["start_frame"],
                    first.get("start_frame"),
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

    report["status"] = "pass" if not differences else "fail"
    report["difference_count"] = len(differences)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "synthetic_tests": report["synthetic_tests"],
                "aifit_reps": report.get("aifit_reps"),
                "difference_count": len(differences),
            },
            indent=2,
        )
    )
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
