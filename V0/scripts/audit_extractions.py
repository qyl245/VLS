#!/usr/bin/env python3
"""Validate task1-1 pose outputs and summarize contiguous failure ranges."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "dataset" / "QEVD-FIT" / "QEVD-FIT-COACH"
DEFAULT_OUTPUT = ROOT / "outputs" / "pose_landmarker_full"
DEFAULT_FEEDBACK = DATA_ROOT / "feedbacks_long_range.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feedback-json", type=Path, default=DEFAULT_FEEDBACK)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.feedback_json = args.feedback_json.resolve()
    return args


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


def load_feedback(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        Path(item["long_range_video_file"]).name: item.get("feedbacks", [])
        for item in payload
    }


def validate_video(video_id: str, output_dir: Path) -> tuple[dict, list[int]]:
    landmarks_path = output_dir / f"{video_id}_landmarks.json"
    meta = json.loads((output_dir / f"{video_id}_meta.json").read_text())
    records = json.loads(landmarks_path.read_text())
    failures = []
    shape_errors = 0
    nonfinite_errors = 0
    visibility_errors = 0
    detected = 0
    hip_offsets = []

    for record in records:
        if record.get("source") != "vision":
            shape_errors += 1
        if record.get("detected"):
            detected += 1
            world = record.get("landmarks_world")
            visibility = record.get("visibility")
            if not isinstance(world, list) or len(world) != 33 or not all(
                isinstance(point, list) and len(point) == 3 for point in world
            ):
                shape_errors += 1
                continue
            if not isinstance(visibility, list) or len(visibility) != 33:
                shape_errors += 1
                continue
            values = np.asarray(world, dtype=np.float64)
            if not np.isfinite(values).all() or not all(
                math.isfinite(float(value)) for value in visibility
            ):
                nonfinite_errors += 1
            if any(float(value) < 0.0 or float(value) > 1.0 for value in visibility):
                visibility_errors += 1
            hip_offsets.append(float(np.linalg.norm((values[23] + values[24]) / 2.0)))
        else:
            failures.append(int(record["frame_id"]))
            if record.get("landmarks_world") is not None or record.get("visibility") is not None:
                shape_errors += 1

    frame_ids = [int(record["frame_id"]) for record in records]
    timestamps = [int(record["timestamp_ms"]) for record in records]
    stride = int(meta["stride"])
    checks = {
        "entries_match_meta": len(records) == int(meta["processed_frames"]),
        "detected_match_meta": detected == int(meta["detected_frames"]),
        "failures_match_meta": failures == list(meta["failed_frame_indices"]),
        "monotonic_frame_ids": all(
            later - earlier == stride for earlier, later in zip(frame_ids, frame_ids[1:])
        ),
        "monotonic_timestamps": all(
            later > earlier for earlier, later in zip(timestamps, timestamps[1:])
        ),
        "landmark_shapes": shape_errors == 0,
        "finite_values": nonfinite_errors == 0,
        "visibility_range": visibility_errors == 0,
        "hips_near_origin": bool(hip_offsets) and float(np.mean(hip_offsets)) < 0.05,
    }
    result = {
        "video_id": video_id,
        "entries": len(records),
        "detected_frames": detected,
        "failed_frames": len(failures),
        "mean_hip_origin_offset_m": float(np.mean(hip_offsets)) if hip_offsets else None,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return result, failures


def main() -> int:
    args = parse_args()
    feedback = load_feedback(args.feedback_json)
    video_ids = sorted(
        path.name.removesuffix("_meta.json") for path in args.output_dir.glob("*_meta.json")
    )
    if not video_ids:
        raise FileNotFoundError(f"no metadata files in {args.output_dir}")

    results = []
    failure_rows = []
    for video_id in video_ids:
        result, failures = validate_video(video_id, args.output_dir)
        results.append(result)
        texts = feedback.get(f"{video_id}.mp4", [])
        fps = json.loads((args.output_dir / f"{video_id}_meta.json").read_text())["fps"]
        for start, end in contiguous_ranges(failures):
            context = Counter(
                text.strip()
                for text in texts[max(0, start - 15) : min(len(texts), end + 16)]
                if text.strip()
            )
            failure_rows.append(
                {
                    "video_id": video_id,
                    "start_frame": start,
                    "end_frame": end,
                    "frames": end - start + 1,
                    "start_seconds": round(start / fps, 3),
                    "end_seconds": round(end / fps, 3),
                    "context": " | ".join(text for text, _ in context.most_common(2)),
                }
            )

    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    total_processed = sum(item["entries"] for item in results)
    total_detected = sum(item["detected_frames"] for item in results)
    total_failed = sum(item["failed_frames"] for item in results)
    checks = {
        "all_videos_passed": all(item["passed"] for item in results),
        "summary_totals_match": summary is None
        or (
            summary["totals"]["processed_frames"] == total_processed
            and summary["totals"]["detected_frames"] == total_detected
            and summary["totals"]["failed_frames"] == total_failed
        ),
    }
    report = {
        "task": "task1-1",
        "output_dir": str(args.output_dir),
        "totals": {
            "videos": len(results),
            "processed_frames": total_processed,
            "detected_frames": total_detected,
            "failed_frames": total_failed,
            "detection_failure_rate": total_failed / total_processed,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "videos": results,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "failure_ranges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0].keys()))
        writer.writeheader()
        writer.writerows(failure_rows)
    print(json.dumps({"passed": report["passed"], **report["totals"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
