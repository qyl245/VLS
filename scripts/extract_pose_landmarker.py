#!/usr/bin/env python3
"""Extract per-frame MediaPipe Pose Landmarker results from QEVD-FIT videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from paths import DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_VIDEO_DIR = DATA_ROOT / "long_range_videos"
DEFAULT_SELECTION = ROOT / "configs" / "task1_1_selection.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "pose_landmarker_full"
DEFAULT_FEEDBACK = DATA_ROOT / "feedbacks_long_range.json"

ROTATION_CODES = {
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
ROTATION_DEGREES = {"none": 0, "90cw": 90, "180": 180, "90ccw": -90}
ROTATION_ORDER = ["none", "90cw", "180", "90ccw"]
KEY_LANDMARKS = [0, 7, 8, 11, 12, 23, 24, 27, 28, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--videos", nargs="*", default=None, help="Override selected video filenames.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feedback-json", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument(
        "--rotation",
        choices=["auto", *ROTATION_CODES],
        default=None,
        help="Override rotation for every video. Defaults to the per-video selection value.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth decoded frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test frame limit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path, videos: list[str] | None, rotation_override: str | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text())
        entries = list(payload.get("videos", []))
    if videos:
        by_file = {entry["file"]: entry for entry in entries}
        entries = []
        for name in videos:
            entry = dict(by_file.get(name, {}))
            entry["file"] = name
            entry.setdefault("video_id", Path(name).stem)
            entry.setdefault("actions", [])
            entry.setdefault("rotation", "none")
            entries.append(entry)
    for entry in entries:
        entry["rotation"] = rotation_override or entry.get("rotation", "none")
        entry.setdefault("video_id", Path(entry["file"]).stem)
        entry.setdefault("actions", [])
    return entries


def load_feedback_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        Path(item["long_range_video_file"]).name: item.get("feedbacks", [])
        for item in payload
    }


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    code = ROTATION_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def read_frame_near(cap: cv2.VideoCapture, target: int, back: int = 40) -> np.ndarray | None:
    start = max(0, target - back)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame = None
    for _ in range(target - start + 1):
        ok, candidate = cap.read()
        if not ok:
            break
        frame = candidate
    return frame


def make_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def landmark_score(result: Any) -> tuple[int, int, float, float]:
    if not result.pose_landmarks:
        return 0, 0, 0.0, 0.0
    landmarks = result.pose_landmarks[0]
    visibility = float(np.mean([landmarks[index].visibility for index in KEY_LANDMARKS]))
    points = np.array([[landmarks[index].x, landmarks[index].y] for index in KEY_LANDMARKS])
    width = float(np.ptp(points[:, 0])) + 1e-6
    height = float(np.ptp(points[:, 1]))
    head = landmarks[0]
    shoulders = (landmarks[11].y + landmarks[12].y) / 2
    hips = (landmarks[23].y + landmarks[24].y) / 2
    ankles = (landmarks[27].y + landmarks[28].y + landmarks[31].y + landmarks[32].y) / 4
    upright = int(head.y < shoulders < hips < ankles)
    return 1, upright, visibility, height / width


def resolve_rotation(
    video_path: Path,
    requested: str,
    image_landmarker: Any,
) -> tuple[str, dict[str, Any]]:
    if requested != "auto":
        return requested, {"method": "fixed", "requested": requested}

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]:
        frame = read_frame_near(cap, int(frame_count * fraction))
        if frame is not None:
            frames.append(frame)
    cap.release()

    scores: dict[str, Any] = {}
    for rotation in ROTATION_ORDER:
        detections = upright = 0
        confidences: list[float] = []
        aspects: list[float] = []
        for frame in frames:
            result = image_landmarker.detect(make_image(rotate_frame(frame, rotation)))
            detected, is_upright, confidence, aspect = landmark_score(result)
            detections += detected
            upright += is_upright
            if detected:
                confidences.append(confidence)
                aspects.append(aspect)
        scores[rotation] = {
            "detections": detections,
            "upright": upright,
            "mean_key_visibility": round(float(np.mean(confidences)), 6) if confidences else 0.0,
            "mean_height_width_ratio": round(float(np.mean(aspects)), 6) if aspects else 0.0,
        }

    ranked = sorted(
        ROTATION_ORDER,
        key=lambda name: (
            scores[name]["upright"],
            scores[name]["detections"],
            scores[name]["mean_key_visibility"],
            scores[name]["mean_height_width_ratio"],
        ),
        reverse=True,
    )
    best = ranked[0]
    return best, {
        "method": "auto",
        "sampled_frames": len(frames),
        "scores": scores,
        "selected": best,
    }


def make_options(args: argparse.Namespace, mode: Any) -> Any:
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )


def serialize_result(result: Any) -> dict[str, Any]:
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return {"detected": False, "landmarks_world": None, "visibility": None}
    normalized = result.pose_landmarks[0]
    world = result.pose_world_landmarks[0]
    return {
        "detected": True,
        "landmarks_world": [
            [round(float(point.x), 6), round(float(point.y), 6), round(float(point.z), 6)]
            for point in world
        ],
        "visibility": [round(float(point.visibility), 6) for point in normalized],
    }


def failure_context(
    failures: list[int],
    feedback: list[str],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = []
    counts: Counter[str] = Counter()
    for frame_id in failures:
        text = feedback[frame_id].strip() if frame_id < len(feedback) else ""
        if text:
            counts[text] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": int(round(frame_id * 1000.0 / fps)) if fps > 0 else None,
                    "feedback": text,
                }
            )
    top = [{"feedback": text, "failed_frames": count} for text, count in counts.most_common(20)]
    return examples, top


def process_video(
    entry: dict[str, Any],
    args: argparse.Namespace,
    feedback_map: dict[str, list[str]],
    image_landmarker: Any,
) -> dict[str, Any]:
    video_path = args.video_dir / entry["file"]
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    video_id = entry["video_id"]
    landmarks_path = args.output_dir / f"{video_id}_landmarks.json"
    meta_path = args.output_dir / f"{video_id}_meta.json"
    if not args.overwrite and (landmarks_path.exists() or meta_path.exists()):
        raise FileExistsError(f"output exists for {video_id}; pass --overwrite")

    started = time.perf_counter()
    rotation, rotation_meta = resolve_rotation(video_path, entry["rotation"], image_landmarker)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_declared = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_width, output_height = (height, width) if rotation in {"90cw", "90ccw"} else (width, height)

    inference_ms: list[float] = []
    failures: list[int] = []
    processed = 0
    frame_id = -1
    video_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        make_options(args, mp.tasks.vision.RunningMode.VIDEO)
    )
    detected = 0
    try:
        with landmarks_path.open("w") as handle:
            handle.write("[\n")
            first = True
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                if frame_id % args.stride:
                    continue
                if args.max_frames is not None and processed >= args.max_frames:
                    break

                rotated = rotate_frame(frame, rotation)
                timestamp_ms = int(round(frame_id * 1000.0 / fps)) if fps > 0 else frame_id
                inference_started = time.perf_counter()
                result = video_landmarker.detect_for_video(make_image(rotated), timestamp_ms)
                inference_ms.append((time.perf_counter() - inference_started) * 1000.0)
                payload = serialize_result(result)
                if payload["detected"]:
                    detected += 1
                else:
                    failures.append(frame_id)

                record = {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "landmarks_world": payload["landmarks_world"],
                    "visibility": payload["visibility"],
                    "source": "vision",
                    "detected": payload["detected"],
                }
                if not first:
                    handle.write(",\n")
                json.dump(record, handle, separators=(",", ":"), allow_nan=False)
                first = False
                processed += 1
                if args.log_every and processed % args.log_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"{video_id}: {processed} frames, {detected / processed:.2%} detected, "
                        f"{processed / elapsed:.1f} FPS",
                        flush=True,
                    )
            handle.write("\n]\n")
    finally:
        cap.release()
        video_landmarker.close()

    wall_seconds = time.perf_counter() - started
    inference_array = np.asarray(inference_ms, dtype=np.float64)
    failures_examples, failures_top = failure_context(
        failures, feedback_map.get(entry["file"], []), fps
    )
    meta = {
        "task": "task1-1",
        "schema_version": "task1-1.v1",
        "video_id": video_id,
        "video_file": str(video_path.resolve()),
        "output_file": str(landmarks_path.resolve()),
        "source": "vision",
        "actions_expected": entry.get("actions", []),
        "selection_notes": entry.get("notes", ""),
        "model": str(args.model.resolve()),
        "running_mode": "VIDEO",
        "num_poses": 1,
        "rotation": rotation,
        "rotation_degrees": ROTATION_DEGREES[rotation],
        "rotation_selection": rotation_meta,
        "fps": fps,
        "input_width": width,
        "input_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "frames_declared": frames_declared,
        "frames_decoded": frame_id + 1,
        "decode_failures": max(0, frames_declared - (frame_id + 1))
        if args.max_frames is None
        else None,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "processed_frames": processed,
        "detected_frames": detected,
        "failed_frames": len(failures),
        "detection_failure_rate": len(failures) / processed if processed else None,
        "failed_frame_indices": failures,
        "failed_frame_examples": failures_examples,
        "failed_frame_feedback_top": failures_top,
        "processing": {
            "wall_seconds": wall_seconds,
            "inference_seconds": float(inference_array.sum() / 1000.0) if inference_array.size else 0.0,
            "mean_inference_ms": float(inference_array.mean()) if inference_array.size else None,
            "p50_inference_ms": float(np.percentile(inference_array, 50)) if inference_array.size else None,
            "p95_inference_ms": float(np.percentile(inference_array, 95)) if inference_array.size else None,
            "end_to_end_fps": processed / wall_seconds if wall_seconds > 0 else None,
            "inference_fps": processed / (inference_array.sum() / 1000.0)
            if inference_array.size and inference_array.sum() > 0
            else None,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return meta


def write_speed_csv(path: Path, metas: list[dict[str, Any]]) -> None:
    fields = [
        "video_id",
        "processed_frames",
        "detected_frames",
        "failed_frames",
        "detection_failure_rate",
        "wall_seconds",
        "end_to_end_fps",
        "mean_inference_ms",
        "p50_inference_ms",
        "p95_inference_ms",
        "inference_fps",
        "rotation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for meta in metas:
            processing = meta["processing"]
            writer.writerow(
                {
                    "video_id": meta["video_id"],
                    "processed_frames": meta["processed_frames"],
                    "detected_frames": meta["detected_frames"],
                    "failed_frames": meta["failed_frames"],
                    "detection_failure_rate": meta["detection_failure_rate"],
                    "wall_seconds": processing["wall_seconds"],
                    "end_to_end_fps": processing["end_to_end_fps"],
                    "mean_inference_ms": processing["mean_inference_ms"],
                    "p50_inference_ms": processing["p50_inference_ms"],
                    "p95_inference_ms": processing["p95_inference_ms"],
                    "inference_fps": processing["inference_fps"],
                    "rotation": meta["rotation"],
                }
            )


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.video_dir = args.video_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    entries = load_selection(args.selection, args.videos, args.rotation)
    if not entries:
        raise ValueError("no videos selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feedback_map = load_feedback_map(args.feedback_json)

    image_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        make_options(args, mp.tasks.vision.RunningMode.IMAGE)
    )
    metas = []
    try:
        for entry in entries:
            print(f"processing {entry['file']} ({entry['rotation']})", flush=True)
            meta = process_video(entry, args, feedback_map, image_landmarker)
            metas.append(meta)
            print(
                f"finished {entry['video_id']}: {meta['processing']['end_to_end_fps']:.2f} FPS, "
                f"failure={meta['detection_failure_rate']:.4%}",
                flush=True,
            )
    finally:
        image_landmarker.close()

    total_processed = sum(meta["processed_frames"] for meta in metas)
    total_detected = sum(meta["detected_frames"] for meta in metas)
    total_failed = sum(meta["failed_frames"] for meta in metas)
    total_wall = sum(meta["processing"]["wall_seconds"] for meta in metas)
    summary = {
        "task": "task1-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "QEVD-FIT-COACH",
        "environment": {
            "python": platform.python_version(),
            "mediapipe": mp.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python_executable": sys.executable,
        },
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "sha256": sha256_file(args.model),
        },
        "settings": {
            "running_mode": "VIDEO",
            "stride": args.stride,
            "max_frames": args.max_frames,
            "min_pose_detection_confidence": args.min_pose_detection_confidence,
            "min_pose_presence_confidence": args.min_pose_presence_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "timestamp_definition": "round(frame_id * 1000 / container_fps)",
        },
        "videos": metas,
        "totals": {
            "videos": len(metas),
            "processed_frames": total_processed,
            "detected_frames": total_detected,
            "failed_frames": total_failed,
            "detection_failure_rate": total_failed / total_processed if total_processed else None,
            "wall_seconds": total_wall,
            "end_to_end_fps": total_processed / total_wall if total_wall else None,
            "mean_end_to_end_fps": float(np.mean([m["processing"]["end_to_end_fps"] for m in metas])),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_speed_csv(args.output_dir / "speed_stats.csv", metas)
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
