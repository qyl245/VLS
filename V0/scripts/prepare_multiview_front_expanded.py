#!/usr/bin/env python3
"""Build the expanded front-view Multi-View squat manifest from screening CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN = ROOT / "results" / "multiview_squat" / "front_screen" / "screen.csv"
DEFAULT_OUTPUT = ROOT / "configs" / "multiview_front_squat_expanded.json"
DEFAULT_SUBJECTS = (
    "subject_001",
    "subject_002",
    "subject_003",
    "subject_004",
    "subject_006",
    "subject_023",
    "subject_024",
)

FRONT_USABLE_METRICS = [
    "rep_duration_s",
    "rep_duration_cv",
    "spine_stability_bottom",
    "knee_valgus_bottom",
    "knee_asymmetry_bottom",
    "hip_asymmetry_bottom",
]
FRONT_DISABLED_METRICS = [
    "knee_flexion_rom",
    "bottom_knee_angle",
    "hip_flexion_rom",
    "torso_lean_bottom",
    "hip_depth_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-csv", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    return parser.parse_args()


def read_screen(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["subject_id"], row["source_quality_label"]): row
        for row in rows
        if row["screen_status"] == "pass"
    }


def short_subject_id(subject_id: str) -> str:
    return "s" + subject_id.removeprefix("subject_")


def build_clip(row: dict[str, str]) -> dict[str, Any]:
    subject = row["subject_id"]
    quality = row["source_quality_label"]
    clip_id = f"mv_{short_subject_id(subject)}_squat_{quality}_front"
    frame_count = int(row["frame_count"])
    return {
        "clip_id": clip_id,
        "video_id": Path(row["video_path"]).stem,
        "video_path": row["video_path"],
        "exercise": "squat",
        "action_display": "squats",
        "trim": {"start_frame": 0, "end_frame": frame_count - 1},
        "fps": float(row["fps"]),
        "frame_count": frame_count,
        "source_size": {"width": int(row["width"]), "height": int(row["height"])},
        "rotation": "none",
        "view": "front",
        "view_geometry": {"dataset_label": "front"},
        "source_quality_label": quality,
        "split": "pilot_front_expanded",
        "phase_role": "pilot",
        "selected_for_phase": True,
        "usable_metrics": FRONT_USABLE_METRICS,
        "disabled_metrics": FRONT_DISABLED_METRICS,
    }


def main() -> int:
    args = parse_args()
    available = read_screen(args.screen_csv)
    clips = []
    for subject in args.subjects:
        for quality in ("good", "bad"):
            key = (subject, quality)
            if key not in available:
                raise ValueError(f"{subject} {quality} is missing from passing screen rows")
            clips.append(build_clip(available[key]))

    payload = {
        "schema_version": "task1-4-v1",
        "manifest_profile": "multiview-front-squat-expanded-v1",
        "task": "Expanded Multi-View front-view squat test",
        "dataset": "Multi-View Fitness Exercise Video Dataset",
        "subject_count": len(args.subjects),
        "screen_csv": str(args.screen_csv.resolve()),
        "selection_rule": "good and bad front clips must both pass MediaPipe screening",
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}: {len(clips)} clips / {len(args.subjects)} subjects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
