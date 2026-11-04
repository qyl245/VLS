#!/usr/bin/env python3
"""Build an expanded Multi-View side-view push-up manifest from screening CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN = ROOT / "results" / "multiview_pushup" / "screen" / "screen.csv"
DEFAULT_OUTPUT = ROOT / "configs" / "multiview_pushup_expanded.json"
DEFAULT_SUBJECTS = (
    "subject_003",
    "subject_004",
    "subject_006",
    "subject_009",
    "subject_012",
    "subject_023",
    "subject_024",
)

PUSHUP_SIDE_METRICS = [
    "rep_duration_s",
    "rep_duration_cv",
    "spine_stability_bottom",
    "elbow_flexion_rom",
    "bottom_elbow_angle",
    "body_line_angle",
    "hip_sag_ratio",
]
PUSHUP_FRONTAL_METRICS = [
    "elbow_asymmetry_bottom",
    "shoulder_asymmetry_bottom",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-csv", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--view", default="side")
    parser.add_argument("--allow-warn", action="store_true")
    return parser.parse_args()


def read_screen(
    path: Path, view: str, allow_warn: bool
) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        eligible_status = {"pass", "warn"} if allow_warn else {"pass"}
        if row["view"] != view or row["screen_status"] not in eligible_status:
            continue
        key = (row["subject_id"], row["source_quality_label"])
        if key in output:
            raise ValueError(f"duplicate screening row for {key}")
        output[key] = row
    return output


def short_subject_id(subject_id: str) -> str:
    return "s" + subject_id.removeprefix("subject_")


def camera_side(row: dict[str, str]) -> str:
    left = float(row["left_side_visibility_median"])
    right = float(row["right_side_visibility_median"])
    return "left" if left >= right else "right"


def near_side_visibility(row: dict[str, str]) -> float:
    return max(
        float(row["left_side_visibility_median"]),
        float(row["right_side_visibility_median"]),
    )


def build_clip(row: dict[str, str]) -> dict[str, Any]:
    subject = row["subject_id"]
    quality = row["source_quality_label"]
    side = camera_side(row)
    frame_count = int(row["frame_count"])
    return {
        "clip_id": f"mv_{short_subject_id(subject)}_pushup_{quality}_side",
        "video_id": Path(row["video_path"]).stem,
        "video_path": row["video_path"],
        "exercise": "pushup",
        "action_display": "pushups",
        "trim": {"start_frame": 0, "end_frame": frame_count - 1},
        "fps": float(row["fps"]),
        "frame_count": frame_count,
        "source_size": {"width": int(row["width"]), "height": int(row["height"])},
        "rotation": "none",
        "view": f"side_{side}",
        "view_geometry": {"dataset_label": "side", "camera_side": side},
        "source_quality_label": quality,
        "source_sha256": row["sha256"],
        "split": "pilot_pushup_side_expanded",
        "phase_role": "pilot",
        "selected_for_phase": True,
        "usable_metrics": PUSHUP_SIDE_METRICS,
        "disabled_metrics": PUSHUP_FRONTAL_METRICS,
    }


def main() -> int:
    args = parse_args()
    available = read_screen(args.screen_csv, args.view, args.allow_warn)
    clips = []
    for subject in args.subjects:
        for quality in ("good", "bad"):
            key = (subject, quality)
            if key not in available:
                raise ValueError(f"{subject} {quality} {args.view} is not a passing screen row")
            if near_side_visibility(available[key]) < 0.5:
                raise ValueError(
                    f"{subject} {quality} {args.view} has insufficient near-side visibility"
                )
            clips.append(build_clip(available[key]))

    hashes = [clip["source_sha256"] for clip in clips]
    if len(hashes) != len(set(hashes)):
        raise ValueError("selected push-up files contain duplicate content")

    payload = {
        "schema_version": "task1-4-v1",
        "manifest_profile": "multiview-pushup-side-expanded-v1",
        "task": "Expanded Multi-View side-view push-up test",
        "dataset": "Multi-View Fitness Exercise Video Dataset",
        "subject_count": len(args.subjects),
        "screen_csv": str(args.screen_csv.resolve()),
        "selection_rule": (
            "good and bad side clips must both pass MediaPipe screening "
            "(warn allowed when explicitly requested), near-side visibility must be >= 0.5, "
            "and content hashes must be unique"
        ),
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}: {len(clips)} clips / {len(args.subjects)} subjects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
