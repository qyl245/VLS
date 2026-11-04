#!/usr/bin/env python3
"""Screen front-view Multi-View squat clips with sampled MediaPipe pose checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_clip


DEFAULT_ROOT = Path("/mnt/storage2/data_guest/VLS/dataset/Multi-View/Dataset Subject-wise")
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT = ROOT / "results" / "multiview_squat" / "front_screen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser.parse_args()


def make_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def classify_screen(row: dict[str, Any]) -> str:
    if (
        row["pose_detection_rate"] >= analyze_clip.POSE_DETECTION_PASS
        and row["body_in_frame_rate"] >= analyze_clip.BODY_IN_FRAME_PASS
        and row["q10_key_visibility"] >= analyze_clip.VISIBILITY_Q10_PASS
    ):
        return "pass"
    if (
        row["pose_detection_rate"] < analyze_clip.POSE_DETECTION_REJECT
        or row["body_in_frame_rate"] < analyze_clip.BODY_IN_FRAME_REJECT
        or row["q10_key_visibility"] < analyze_clip.VISIBILITY_Q10_REJECT
    ):
        return "reject"
    return "warn"


def screen_clip(
    path: Path,
    landmarker: mp.tasks.vision.PoseLandmarker,
    stride: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    checks: list[dict[str, Any]] = []
    sharpness: list[float] = []
    decoded = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded % stride == 0:
            result = landmarker.detect_for_video(
                make_image(frame), int(round(decoded * 1000.0 / fps))
            )
            checks.append(analyze_clip.landmark_quality(result, "squat"))
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (180, 320),
                interpolation=cv2.INTER_AREA,
            )
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        decoded += 1
    cap.release()

    detections = [item for item in checks if item["detected"]]
    mean_visibilities = [float(item["mean_visibility"]) for item in detections]
    sample_count = len(checks)
    row = {
        "clip_id": f"mv_{path.parents[3].name}_squat_{path.parents[1].name}_front",
        "subject_id": path.parents[3].name,
        "source_quality_label": path.parents[1].name,
        "video_path": str(path.resolve()),
        "frame_count": frame_count,
        "fps": round(fps, 6),
        "duration_s": round(frame_count / fps, 6) if fps > 0 else None,
        "width": width,
        "height": height,
        "sample_stride": stride,
        "sample_count": sample_count,
        "pose_detection_rate": round(len(detections) / sample_count, 6) if sample_count else 0.0,
        "mean_key_visibility": round(float(np.mean(mean_visibilities)), 6) if mean_visibilities else 0.0,
        "q10_key_visibility": round(float(np.percentile(mean_visibilities, 10)), 6) if mean_visibilities else 0.0,
        "key_joint_visible_rate": round(
            float(np.mean([item["visible_joint_fraction"] for item in checks])), 6
        ) if checks else 0.0,
        "all_exercise_joints_visible_rate": round(
            float(np.mean([item["all_joints_visible"] for item in checks])), 6
        ) if checks else 0.0,
        "body_in_frame_rate": round(
            sum(bool(item["in_frame"]) for item in checks) / sample_count, 6
        ) if sample_count else 0.0,
        "median_sharpness": round(float(np.median(sharpness)), 6) if sharpness else 0.0,
        "processing_seconds": round(time.perf_counter() - started, 6),
    }
    row["screen_status"] = classify_screen(row)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    files = sorted(args.root.glob("subject_*/squat/*/front/*.mp4"))
    if not files:
        raise FileNotFoundError(f"no front squat videos under {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_detection_confidence,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        try:
            row = screen_clip(path, landmarker, args.stride)
        finally:
            landmarker.close()
        rows.append(row)
        print(
            f"  {row['screen_status']}: detect={row['pose_detection_rate']:.3f} "
            f"body={row['body_in_frame_rate']:.3f} q10={row['q10_key_visibility']:.3f}",
            flush=True,
        )

    write_csv(args.output_dir / "screen.csv", rows)
    summary = {
        "schema_version": "multiview-front-squat-screen-v1",
        "clip_count": len(rows),
        "sample_stride": args.stride,
        "status_counts": dict(Counter(row["screen_status"] for row in rows)),
        "subject_count": len({row["subject_id"] for row in rows}),
        "quality_label_counts": dict(Counter(row["source_quality_label"] for row in rows)),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: summary[k] for k in ("clip_count", "status_counts", "subject_count")}, indent=2))


if __name__ == "__main__":
    main()
