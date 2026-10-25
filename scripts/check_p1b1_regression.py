#!/usr/bin/env python3
"""Validate P1-B1 metadata coverage and unchanged P1-D boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps as sr


DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p1b"
DEFAULT_REFERENCE = ROOT / "results" / "segmentation_p1d"
DEFAULT_P1B_ANNOTATION = DEFAULT_SEGMENTATION / "annotation"
DEFAULT_EXISTING_ANNOTATION = ROOT / "results" / "segmentation" / "annotation_blind"
DEFAULT_REP_METADATA = ROOT / "configs" / "rep_metadata.json"
SEGMENT_SCRIPT = ROOT / "scripts" / "segment_reps.py"
EVALUATION_SCRIPT = ROOT / "scripts" / "evaluate_p1b1_rep_units.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--p1b-annotation-dir", type=Path, default=DEFAULT_P1B_ANNOTATION)
    parser.add_argument("--existing-annotation-dir", type=Path, default=DEFAULT_EXISTING_ANNOTATION)
    parser.add_argument("--rep-metadata", type=Path, default=DEFAULT_REP_METADATA)
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


def rep_boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["video_id"], row["action"], row["method"])].append(
            (int(row["start_frame"]), int(row["end_frame"]))
        )
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def metadata_row_matches(row: dict[str, str], metadata: dict[str, Any]) -> bool:
    return all(
        row.get(field) == str(metadata.get(field))
        for field in ("rep_unit", "phase_landmark", "phase_polarity")
    )


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "pass",
        "segmentation_dir": str(args.segmentation_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "differences": [],
        "synthetic_tests": "skipped",
    }
    differences: list[dict[str, Any]] = report["differences"]

    if not args.skip_synthetic:
        result = run_command(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_segment_reps.py")],
            "synthetic tests",
        )
        report["synthetic_tests"] = "pass" if result.returncode == 0 else "fail"
        if result.returncode != 0:
            add_difference(differences, "synthetic_tests", "pytest", "pass", "fail")

    with tempfile.TemporaryDirectory(prefix="vls-p1b1-regression-") as temporary:
        temp_root = Path(temporary)
        if args.use_existing:
            segmentation_dir = args.segmentation_dir.resolve()
        else:
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
                "P1-B1 segmentation",
            )
            if result.returncode != 0:
                report["status"] = "fail"
                if args.report:
                    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                return 1

        evaluation_dir = temp_root / "evaluation"
        result = run_command(
            [
                sys.executable,
                str(EVALUATION_SCRIPT),
                "--segmentation-dir",
                str(segmentation_dir),
                "--p1b-annotation-dir",
                str(args.p1b_annotation_dir.resolve()),
                "--existing-annotation-dir",
                str(args.existing_annotation_dir.resolve()),
                "--rep-metadata",
                str(args.rep_metadata.resolve()),
                "--output-dir",
                str(evaluation_dir),
            ],
            "P1-B1 walking-lunge evaluation",
        )
        if result.returncode != 0:
            add_difference(differences, "evaluation", "command", 0, result.returncode)
            evaluation: dict[str, Any] = {}
        else:
            evaluation = json.loads(
                (evaluation_dir / "p1b1_walking_lunges_eval.json").read_text()
            )

        metadata = sr.load_rep_metadata(args.rep_metadata.resolve())
        segments = read_csv(segmentation_dir / "action_segments.csv")
        summary_rows = read_csv(segmentation_dir / "summary.csv")
        rep_rows = read_csv(segmentation_dir / "reps.csv")
        expected_keys = {
            (row["video_id"].zfill(4), row["action"])
            for row in read_csv(ROOT / "results" / "segmentation" / "action_segments.csv")
        }
        report["metadata"] = {
            "configured_segments": len(metadata),
            "expected_segments": len(expected_keys),
            "rep_unit_counts": dict(Counter(entry["rep_unit"] for entry in metadata.values())),
            "phase_polarity_counts": dict(
                Counter(entry["phase_polarity"] for entry in metadata.values())
            ),
        }
        if set(metadata) != expected_keys:
            add_difference(
                differences,
                "metadata_coverage",
                "segment_keys",
                sorted("/".join(key) for key in expected_keys),
                sorted("/".join(key) for key in metadata),
            )
        if len(segments) != 22:
            add_difference(differences, "metadata_coverage", "action_segments", 22, len(segments))
        for row in segments:
            key = (row["video_id"].zfill(4), row["action"])
            entry = metadata.get(key)
            if entry is None or not metadata_row_matches(row, entry):
                add_difference(
                    differences,
                    "action_segments_metadata",
                    "/".join(key),
                    entry,
                    {
                        field: row.get(field)
                        for field in ("rep_unit", "phase_landmark", "phase_polarity")
                    },
                )

        for row in summary_rows:
            key = (row["video_id"].zfill(4), row["action"])
            entry = metadata.get(key)
            if entry is None or not metadata_row_matches(row, entry):
                add_difference(differences, "summary_metadata", "/".join(key), entry, row)
        for row in rep_rows:
            key = (row["video_id"].zfill(4), row["action"])
            entry = metadata.get(key)
            if entry is None or not metadata_row_matches(row, entry):
                add_difference(differences, "reps_metadata", "/".join(key), entry, row)

        aifit_count = sum(1 for row in rep_rows if row["method"] == "aifit")
        peak_count = sum(1 for row in rep_rows if row["method"] == "peak")
        report["rep_counts"] = {"aifit": aifit_count, "peak": peak_count}
        if aifit_count != 366:
            add_difference(differences, "rep_counts", "aifit", 366, aifit_count)
        if peak_count != 341:
            add_difference(differences, "rep_counts", "peak", 341, peak_count)

        reference = rep_boundaries(args.reference_dir.resolve() / "reps.csv")
        actual = rep_boundaries(segmentation_dir / "reps.csv")
        boundary_differences = 0
        for key in sorted(set(reference) | set(actual)):
            if reference.get(key) != actual.get(key):
                boundary_differences += 1
                add_difference(
                    differences,
                    "rep_boundaries",
                    "/".join(key),
                    reference.get(key),
                    actual.get(key),
                )
        report["boundary_difference_count"] = boundary_differences

        acceptance = evaluation.get("acceptance", {})
        report["walking_lunge_acceptance"] = acceptance
        if acceptance.get("status") != "pass":
            add_difference(
                differences,
                "walking_lunge_acceptance",
                "status",
                "pass",
                acceptance.get("status"),
            )
        if acceptance.get("gt_0162_reps") != 6:
            add_difference(
                differences,
                "walking_lunge_acceptance",
                "gt_0162_reps",
                6,
                acceptance.get("gt_0162_reps"),
            )
        if acceptance.get("p1b2_trigger") is not False:
            add_difference(
                differences,
                "walking_lunge_acceptance",
                "p1b2_trigger",
                False,
                acceptance.get("p1b2_trigger"),
            )
        if evaluation.get("rep_units") != {
            "0162_walking_lunges": "stride",
            "0218_walking_lunges": "stride",
        }:
            add_difference(
                differences,
                "walking_lunge_acceptance",
                "rep_units",
                {"0162_walking_lunges": "stride", "0218_walking_lunges": "stride"},
                evaluation.get("rep_units"),
            )

    report["status"] = "pass" if not differences else "fail"
    report["difference_count"] = len(differences)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "synthetic_tests": report["synthetic_tests"],
                "difference_count": report["difference_count"],
                "metadata": report.get("metadata"),
                "rep_counts": report.get("rep_counts"),
                "boundary_difference_count": report.get("boundary_difference_count"),
                "walking_lunge_acceptance": report.get("walking_lunge_acceptance"),
                "differences": differences[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
