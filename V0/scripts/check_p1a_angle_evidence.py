#!/usr/bin/env python3
"""Validate P1-A angle evidence, ambiguity flags, and non-regression."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import segment_reps as sr


DEFAULT_SEGMENTATION = ROOT / "results" / "segmentation_p1a_p2a" / "full_evidence"
DEFAULT_REFERENCE = ROOT / "results" / "segmentation_p1d"
REQUIRED_CANDIDATE_FIELDS = {
    "angle",
    "period_frames",
    "periodicity",
    "peak_prominence",
    "phase_consistency",
    "boundary_stability",
    "dynamic_range",
    "valid_fraction",
    "score",
    "cluster_id",
    "selected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def boundaries(path: Path) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        key = (row["video_id"], row["action"], row["method"])
        grouped[key].append((int(row["start_frame"]), int(row["end_frame"])))
    for values in grouped.values():
        values.sort()
    return dict(grouped)


def main() -> int:
    args = parse_args()
    segmentation_dir = args.segmentation_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    payload = json.loads((segmentation_dir / "run.json").read_text())
    evidence_rows = read_csv(segmentation_dir / "angle_evidence.csv")
    differences: list[dict[str, Any]] = []
    candidate_count = 0
    ambiguous_count = 0

    for segment in payload["segments"]:
        selection = segment.get("angle_selection", {})
        label = f"{segment['video_id']}/{segment['action']}"
        if selection.get("method") != "evidence":
            differences.append({"scope": "selection", "key": label, "reason": "not evidence mode"})
            continue
        candidates = selection.get("candidates", [])
        candidate_count += len(candidates)
        if not candidates:
            differences.append({"scope": "selection", "key": label, "reason": "no candidates"})
            continue
        for candidate in candidates:
            missing = REQUIRED_CANDIDATE_FIELDS - set(candidate)
            if missing:
                differences.append(
                    {
                        "scope": "candidate_fields",
                        "key": f"{label}/{candidate.get('angle')}",
                        "missing": sorted(missing),
                    }
                )
        selected = [item for item in candidates if item.get("selected")]
        if len(selected) != 1 or selected[0]["angle"] != selection.get("selected_angle"):
            differences.append(
                {"scope": "selected_candidate", "key": label, "reason": "inconsistent selection"}
            )
        if selection.get("angle_ambiguous") and selection.get("selected_by") != "legacy_tie_break":
            differences.append(
                {"scope": "ambiguity", "key": label, "reason": "ambiguous without legacy tie-break"}
            )
        if selection.get("angle_ambiguous"):
            ambiguous_count += 1
        minimum_s, maximum_s = sr.action_period_range_s(segment["action"], 0.4, 5.0)
        for candidate in candidates:
            period_s = float(candidate["period_frames"]) / 30.0
            if not minimum_s - 1e-9 <= period_s <= maximum_s + 1e-9:
                differences.append(
                    {
                        "scope": "period_range",
                        "key": f"{label}/{candidate['angle']}",
                        "period_s": period_s,
                        "range_s": [minimum_s, maximum_s],
                    }
                )

    if candidate_count != len(evidence_rows):
        differences.append(
            {
                "scope": "angle_evidence_csv",
                "run_candidates": candidate_count,
                "csv_rows": len(evidence_rows),
            }
        )

    actual = boundaries(segmentation_dir / "reps.csv")
    expected = boundaries(reference_dir / "reps.csv")
    if actual != expected:
        for key in sorted(set(actual) | set(expected)):
            if actual.get(key) != expected.get(key):
                differences.append(
                    {
                        "scope": "unchanged_boundaries",
                        "key": "/".join(key),
                        "expected_reps": len(expected.get(key, [])),
                        "actual_reps": len(actual.get(key, [])),
                    }
                )

    report = {
        "status": "pass" if not differences else "fail",
        "segmentation_dir": str(segmentation_dir),
        "reference_dir": str(reference_dir),
        "segments": len(payload["segments"]),
        "candidate_rows": candidate_count,
        "ambiguous_segments": ambiguous_count,
        "aifit_reps": sum(
            len(value) for key, value in actual.items() if key[2] == "aifit"
        ),
        "differences": differences,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
