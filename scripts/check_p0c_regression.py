#!/usr/bin/env python3
"""Run P0-C synthetic tests and the frozen QEVD-FIT golden regression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "configs" / "p0c_golden.json"
DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p0c"
DEFAULT_BLIND_DIR = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_FIRST_ROUND = ROOT / "results" / "segmentation" / "annotation" / "annotation_template.csv"
SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
EVALUATION_SCRIPT = ROOT / "scripts" / "evaluate_blind_annotations.py"
TESTS_DIR = ROOT / "tests"
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
    parser.add_argument("--first-round", type=Path, default=DEFAULT_FIRST_ROUND)
    parser.add_argument(
        "--mode",
        choices=("exact", "guard"),
        default="exact",
        help="exact reproduces frozen P0-C; guard allows metric improvements for later changes.",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Evaluate the existing --segmentation-dir instead of rerunning segmentation.",
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command: list[str], cwd: Path, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"{label} failed with exit code {result.returncode}\n")
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result


def segment_counts(segments_path: Path, reps_path: Path) -> dict[str, dict[str, int]]:
    counts = {
        f"{row['video_id']}/{row['action']}": {"aifit": 0, "peak": 0}
        for row in read_csv(segments_path)
    }
    for row in read_csv(reps_path):
        key = f"{row['video_id']}/{row['action']}"
        method = row["method"]
        if key not in counts or method not in counts[key]:
            continue
        counts[key][method] += 1
    return dict(sorted(counts.items()))


def read_eval_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    summary: dict[str, dict[str, Any]] = {}
    int_fields = {
        "segments",
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
    for row in rows:
        if row["method"] != "aifit" or row["rep_unit"] != "all":
            continue
        threshold = f"{float(row['iou_threshold']):g}"
        metrics: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"method", "iou_threshold", "rep_unit"}:
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                continue
            metrics[key] = int(numeric_value) if key in int_fields else numeric_value
        summary[threshold] = metrics
    return summary


def read_blind_segment_counts(path: Path) -> dict[str, dict[str, int]]:
    return {
        row["segment_id"]: {
            "gt_reps": int(float(row["gt_reps"])),
            "pred_reps": int(float(row["pred_reps"])),
        }
        for row in read_csv(path)
        if row["method"] == "aifit" and abs(float(row["iou_threshold"]) - 0.30) < 1e-9
    }


def append_difference(
    differences: list[dict[str, Any]],
    scope: str,
    key: str,
    expected: Any,
    actual: Any,
    reason: str,
) -> None:
    differences.append(
        {
            "scope": scope,
            "key": key,
            "expected": expected,
            "actual": actual,
            "reason": reason,
        }
    )


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= FLOAT_TOLERANCE
    return expected == actual


def compare_counts(
    golden: dict[str, Any],
    actual: dict[str, dict[str, int]],
    differences: list[dict[str, Any]],
) -> None:
    expected_counts = golden["segment_counts"]
    for key, expected in expected_counts.items():
        if key not in actual:
            append_difference(differences, "segment_counts", key, expected, None, "missing segment")
            continue
        for method, count in expected.items():
            if actual[key].get(method) != count:
                append_difference(
                    differences,
                    "segment_counts",
                    f"{key}/{method}",
                    count,
                    actual[key].get(method),
                    "rep count changed",
                )
    for key in sorted(set(actual) - set(expected_counts)):
        append_difference(
            differences,
            "segment_counts",
            key,
            None,
            actual[key],
            "unexpected segment",
        )

    for key, expected in golden["acceptance_segments"].items():
        actual_count = actual.get(key, {}).get("aifit")
        if actual_count != expected:
            append_difference(
                differences,
                "acceptance_segments",
                key,
                expected,
                actual_count,
                "specified acceptance count changed",
            )


def compare_blind_segments(
    golden: dict[str, Any],
    actual: dict[str, dict[str, int]],
    differences: list[dict[str, Any]],
) -> None:
    expected_segments = golden["blind_segment_counts"]
    for key, expected in expected_segments.items():
        if key not in actual:
            append_difference(differences, "blind_segments", key, expected, None, "missing segment")
            continue
        for field, value in expected.items():
            if actual[key].get(field) != value:
                append_difference(
                    differences,
                    "blind_segments",
                    f"{key}/{field}",
                    value,
                    actual[key].get(field),
                    "blind GT or prediction count changed",
                )


def compare_blind_eval_exact(
    golden: dict[str, Any],
    actual: dict[str, dict[str, Any]],
    differences: list[dict[str, Any]],
) -> None:
    for threshold, expected_metrics in golden["blind_eval"].items():
        if threshold not in actual:
            append_difference(differences, "blind_eval", threshold, expected_metrics, None, "missing threshold")
            continue
        for field, expected in expected_metrics.items():
            actual_value = actual[threshold].get(field)
            if not values_equal(expected, actual_value):
                append_difference(
                    differences,
                    "blind_eval",
                    f"IoU={threshold}/{field}",
                    expected,
                    actual_value,
                    "frozen P0-C metric changed",
                )


def compare_blind_eval_guard(
    golden: dict[str, Any],
    actual: dict[str, dict[str, Any]],
    differences: list[dict[str, Any]],
) -> None:
    lower_is_better = {
        "false_positives",
        "false_negatives",
        "mean_abs_start_error_frames",
        "mean_abs_end_error_frames",
        "p95_boundary_error_frames",
        "split_events",
        "merge_events",
        "fp_transition",
        "fp_duplicate",
    }
    higher_is_better = {"matched", "micro_f1", "micro_iou"}
    for threshold, expected_metrics in golden["blind_eval"].items():
        if threshold not in actual:
            append_difference(differences, "blind_eval", threshold, expected_metrics, None, "missing threshold")
            continue
        for field, expected in expected_metrics.items():
            actual_value = actual[threshold].get(field)
            if actual_value is None:
                append_difference(differences, "blind_eval", f"IoU={threshold}/{field}", expected, None, "missing metric")
                continue
            if field in {"gt_reps", "pred_reps"}:
                passed = values_equal(expected, actual_value)
                reason = "blind eval count changed"
            elif field in lower_is_better:
                passed = float(actual_value) <= float(expected) + FLOAT_TOLERANCE
                reason = "metric regressed above frozen P0-C"
            elif field in higher_is_better:
                passed = float(actual_value) >= float(expected) - FLOAT_TOLERANCE
                reason = "metric regressed below frozen P0-C"
            else:
                passed = values_equal(expected, actual_value)
                reason = "undefined guard direction"
            if not passed:
                append_difference(
                    differences,
                    "blind_eval",
                    f"IoU={threshold}/{field}",
                    expected,
                    actual_value,
                    reason,
                )


def check_annotation_hashes(
    golden: dict[str, Any],
    blind_dir: Path,
    differences: list[dict[str, Any]],
) -> dict[str, str]:
    actual = {}
    for name, expected in golden["annotation_sha256"].items():
        path = blind_dir / name
        if not path.exists():
            append_difference(differences, "annotation_sha256", name, expected, None, "missing GT file")
            continue
        digest = sha256(path)
        actual[name] = digest
        if digest != expected:
            append_difference(
                differences,
                "annotation_sha256",
                name,
                expected,
                digest,
                "frozen blind GT changed",
            )
    return actual


def check_active_core_invariant(
    run_path: Path,
    reps_path: Path,
    differences: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(run_path.read_text())
    active_bouts: dict[tuple[str, str], list[tuple[int, int]]] = {}
    zero_rep_segments: list[str] = []
    for segment in payload.get("segments", []):
        key = (segment["video_id"], segment["action"])
        active_bouts[key] = [
            (int(bout["start_frame"]), int(bout["end_frame"]))
            for bout in segment.get("active_bouts", [])
            if bout.get("status") == "active"
        ]
        aifit = segment.get("angles", {}).get("aifit", {})
        if int(aifit.get("num_reps", 0)) == 0:
            key_text = f"{key[0]}/{key[1]}"
            zero_rep_segments.append(key_text)
            if active_bouts[key] and not segment.get("non_rep_intervals"):
                append_difference(
                    differences,
                    "no_rep_evidence",
                    key_text,
                    "non_rep_intervals recorded",
                    [],
                    "zero-rep segment has active bouts but no rejection evidence",
                )

    checked = 0
    for row in read_csv(reps_path):
        if row["method"] != "aifit":
            continue
        key = (row["video_id"], row["action"])
        if key not in active_bouts:
            continue
        checked += 1
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        max_overlap = max(
            (
                max(0, min(end_frame, bout_end) - max(start_frame, bout_start) + 1)
                for bout_start, bout_end in active_bouts[key]
            ),
            default=0,
        )
        if max_overlap == 0:
            append_difference(
                differences,
                "active_core",
                f"{key[0]}/{key[1]}/rep{row['rep_id']}",
                ">0 overlap frames",
                0,
                "rep lies entirely outside active bout cores",
            )
    return {
        "checked_aifit_reps": checked,
        "segments_with_active_bouts": len(active_bouts),
        "zero_rep_segments": sorted(zero_rep_segments),
    }


def main() -> int:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    differences: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "mode": args.mode,
        "golden": str(args.golden.resolve()),
        "synthetic_tests": "skipped",
        "differences": differences,
    }

    if not args.skip_synthetic:
        test_result = run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(TESTS_DIR),
                "-p",
                "test_*.py",
            ],
            cwd=ROOT,
            label="synthetic tests",
        )
        report["synthetic_tests"] = "pass" if test_result.returncode == 0 else "fail"
        if test_result.returncode != 0:
            append_difference(
                differences,
                "synthetic_tests",
                "unittest",
                "pass",
                "fail",
                "synthetic regression failed",
            )

    with tempfile.TemporaryDirectory(prefix="vls-p0c-regression-") as temporary:
        temp_root = Path(temporary)
        segmentation_dir = args.segmentation_dir.resolve()
        if not args.use_existing:
            segmentation_dir = temp_root / "segmentation"
            segmentation_command = [
                sys.executable,
                str(SEGMENT_SCRIPT),
                "--output-dir",
                str(segmentation_dir),
                "--no-plots",
                "--p2a-tail-filter",
                "off",
            ]
            if args.mode == "exact":
                segmentation_command.append("--no-phase-guidance")
                segmentation_command.append("--no-phase-snap")
            segmentation_result = run_command(
                segmentation_command,
                cwd=ROOT,
                label="P0-C segmentation",
            )
            if segmentation_result.returncode != 0:
                report["status"] = "fail"
                if args.report:
                    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                return 1

        segments_path = segmentation_dir / "action_segments.csv"
        reps_path = segmentation_dir / "reps.csv"
        run_path = segmentation_dir / "run.json"
        counts = segment_counts(segments_path, reps_path)
        report["segment_counts"] = counts
        compare_counts(golden, counts, differences)

        run_payload = json.loads(run_path.read_text())
        actual_parameters = run_payload.get("parameters", {})
        for key, expected in golden["parameters"].items():
            if not values_equal(expected, actual_parameters.get(key)):
                append_difference(
                    differences,
                    "parameters",
                    key,
                    expected,
                    actual_parameters.get(key),
                    "frozen P0-C parameter changed",
                )
        report["invariants"] = {
            "active_core": check_active_core_invariant(run_path, reps_path, differences),
            "annotation_sha256": check_annotation_hashes(golden, args.blind_dir, differences),
        }

        eval_dir = temp_root / "blind_eval"
        eval_dir.mkdir()
        for name in ANNOTATION_FILES:
            shutil.copy2(args.blind_dir / name, eval_dir / name)
        evaluation_result = run_command(
            [
                sys.executable,
                str(EVALUATION_SCRIPT),
                "--blind-dir",
                str(eval_dir),
                "--reps",
                str(reps_path),
                "--segments",
                str(segments_path),
                "--first-round",
                str(args.first_round),
            ],
            cwd=ROOT,
            label="blind evaluation",
        )
        if evaluation_result.returncode not in (0, 1):
            report["status"] = "fail"
            if args.report:
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            return 1

        blind_eval = read_eval_summary(eval_dir / "blind_eval_summary.csv")
        blind_counts = read_blind_segment_counts(eval_dir / "blind_eval.csv")
        report["blind_eval"] = blind_eval
        report["blind_segment_counts"] = blind_counts
        compare_blind_segments(golden, blind_counts, differences)
        if args.mode == "exact":
            compare_blind_eval_exact(golden, blind_eval, differences)
        else:
            compare_blind_eval_guard(golden, blind_eval, differences)

    report["status"] = "pass" if not differences else "fail"
    report["difference_count"] = len(differences)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": args.mode,
                "synthetic_tests": report["synthetic_tests"],
                "difference_count": len(differences),
                "segments": len(report.get("segment_counts", {})),
                "blind_eval": report.get("blind_eval"),
                "differences": differences[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
