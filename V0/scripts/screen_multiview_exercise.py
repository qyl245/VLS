#!/usr/bin/env python3
"""Screen Multi-View exercise clips with sampled MediaPipe pose checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
EXERCISE_ALIASES = {"push_up": "pushup", "pushup": "pushup", "squat": "squat"}
SIDE_JOINTS = {
    "pushup": {"left": (11, 13, 15), "right": (12, 14, 16)},
    "squat": {"left": (23, 25, 27), "right": (24, 26, 28)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--exercise", default="push_up")
    parser.add_argument("--views", nargs="+", default=["front", "side", "diagonal"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser.parse_args()


def make_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def side_visibility(landmarks: Any, exercise: str) -> dict[str, float]:
    output = {}
    for side, indices in SIDE_JOINTS[exercise].items():
        output[side] = float(min(landmarks[index].visibility for index in indices))
    return output


def screen_clip(
    path: Path,
    exercise_folder: str,
    exercise: str,
    view: str,
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
    left_visibility: list[float] = []
    right_visibility: list[float] = []
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
            checks.append(analyze_clip.landmark_quality(result, exercise))
            if result.pose_landmarks:
                sides = side_visibility(result.pose_landmarks[0], exercise)
                left_visibility.append(sides["left"])
                right_visibility.append(sides["right"])
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
    subject = path.parents[3].name
    quality = path.parents[1].name
    clip_id = f"mv_s{subject.removeprefix('subject_')}_{exercise}_{quality}_{view}"
    row = {
        "clip_id": clip_id,
        "subject_id": subject,
        "exercise_folder": exercise_folder,
        "exercise": exercise,
        "source_quality_label": quality,
        "view": view,
        "video_path": str(path.resolve()),
        "sha256": sha256_path(path),
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
        "left_side_visibility_median": round(float(np.median(left_visibility)), 6) if left_visibility else 0.0,
        "right_side_visibility_median": round(float(np.median(right_visibility)), 6) if right_visibility else 0.0,
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


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.exercise not in EXERCISE_ALIASES:
        raise ValueError(f"unsupported exercise: {args.exercise}")
    model_exercise = EXERCISE_ALIASES[args.exercise]
    files = []
    for view in args.views:
        files.extend(sorted(args.root.glob(f"subject_*/{args.exercise}/*/{view}/*.mp4")))
    if not files:
        raise FileNotFoundError(f"no {args.exercise} videos under {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        view = path.parent.name
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
            row = screen_clip(
                path, args.exercise, model_exercise, view, landmarker, args.stride
            )
        finally:
            landmarker.close()
        rows.append(row)
        print(
            f"  {row['screen_status']} view={view} detect={row['pose_detection_rate']:.3f} "
            f"body={row['body_in_frame_rate']:.3f} q10={row['q10_key_visibility']:.3f} "
            f"L/R={row['left_side_visibility_median']:.3f}/{row['right_side_visibility_median']:.3f}",
            flush=True,
        )

    hash_counts = Counter(row["sha256"] for row in rows)
    for row in rows:
        row["sha256_occurrences"] = hash_counts[row["sha256"]]
    with (args.output_dir / "screen.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "multiview-exercise-screen-v1",
        "exercise_folder": args.exercise,
        "exercise": model_exercise,
        "views": args.views,
        "clip_count": len(rows),
        "sample_stride": args.stride,
        "status_counts": dict(Counter(row["screen_status"] for row in rows)),
        "view_status_counts": {
            view: dict(Counter(row["screen_status"] for row in rows if row["view"] == view))
            for view in args.views
        },
        "subject_count": len({row["subject_id"] for row in rows}),
        "quality_label_counts": dict(Counter(row["source_quality_label"] for row in rows)),
        "duplicate_hash_count": sum(count > 1 for count in hash_counts.values()),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "clip_count": summary["clip_count"],
                "status_counts": summary["status_counts"],
                "view_status_counts": summary["view_status_counts"],
                "subject_count": summary["subject_count"],
                "duplicate_hash_count": summary["duplicate_hash_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
