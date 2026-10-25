#!/usr/bin/env python3
"""Extract raw video frame sheets for blind rep annotation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_SEGMENTS = ROOT / "results" / "segmentation" / "action_segments.csv"
DEFAULT_VIDEOS = ROOT.parent / "dataset" / "QEVD-FIT" / "QEVD-FIT-COACH" / "long_range_videos"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation" / "annotation_blind" / "frames"

DEFAULT_SEGMENTS = (
    ("0161", "pushup"),
    ("0173", "boxing_squat_punches"),
    ("0161", "lunge_jumps"),
    ("0218", "walking_lunges"),
    ("0218", "air_jump_rope"),
    ("0162", "mountain_climbers"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-segments", type=Path, default=DEFAULT_ACTION_SEGMENTS)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Segments as video_id:action. Defaults to the six P0-A segments.",
    )
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--frames-per-page", type=int, default=64)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=300)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional fixed crop as x1,y1,x2,y2 before tiling.",
    )
    parser.add_argument(
        "--focus-file",
        type=Path,
        default=None,
        help="Optional CSV with segment_id,center_frame,label for dense boundary review.",
    )
    parser.add_argument("--focus-radius", type=int, default=12)
    parser.add_argument("--focus-step", type=int, default=2)
    parser.add_argument("--focus-columns", type=int, default=7)
    parser.add_argument("--focus-boundaries-per-page", type=int, default=2)
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.frames_per_page <= 0 or args.columns <= 0:
        parser.error("--frames-per-page and --columns must be positive")
    if args.focus_radius < 0 or args.focus_step <= 0:
        parser.error("--focus-radius must be non-negative and --focus-step positive")
    if args.focus_columns <= 0 or args.focus_boundaries_per_page <= 0:
        parser.error("--focus-columns and --focus-boundaries-per-page must be positive")
    if args.crop is not None:
        try:
            values = tuple(int(value) for value in args.crop.split(","))
        except ValueError:
            parser.error("--crop must use x1,y1,x2,y2")
        if len(values) != 4:
            parser.error("--crop must use x1,y1,x2,y2")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            parser.error("--crop requires 0 <= x1 < x2 and 0 <= y1 < y2")
        args.crop_box = values
    else:
        args.crop_box = None
    if args.segments:
        selected = []
        for item in args.segments:
            if ":" not in item:
                parser.error("--segments entries must use video_id:action")
            video_id, action = item.split(":", 1)
            selected.append((video_id.zfill(4), action))
        args.selected_segments = selected
    else:
        args.selected_segments = list(DEFAULT_SEGMENTS)
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.48,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def load_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode frame {index}")
    return frame


def thumb_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.crop_box is None:
        aspect = 9 / 16
    else:
        x1, y1, x2, y2 = args.crop_box
        aspect = (y2 - y1) / (x2 - x1)
    return args.thumb_width, round(args.thumb_width * aspect)


def add_tile(
    canvas: np.ndarray,
    frame: np.ndarray,
    x: int,
    y: int,
    frame_index: int,
    fps: float,
    args: argparse.Namespace,
) -> None:
    width, height = thumb_size(args)
    if args.crop_box is not None:
        x1, y1, x2, y2 = args.crop_box
        frame = frame[y1:y2, x1:x2]
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (70, 70, 70), 1)
    put_label(canvas, f"f{frame_index}  {frame_index / fps:.2f}s", (x + 5, y + height - 8), 0.45)


def overview_frames(start: int, end: int, fps: float, sample_fps: float) -> list[int]:
    step = max(1, int(round(fps / sample_fps)))
    values = list(range(start, end + 1, step))
    if not values or values[-1] != end:
        values.append(end)
    return values


def make_overview_pages(
    video_id: str,
    action: str,
    start: int,
    end: int,
    frames: list[int],
    capture: cv2.VideoCapture,
    fps: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    width, height = thumb_size(args)
    cell_height = height + 28
    header_height = 54
    label_width = 0
    pages = []
    for page_index, offset in enumerate(range(0, len(frames), args.frames_per_page), start=1):
        page_frames = frames[offset : offset + args.frames_per_page]
        rows = math.ceil(len(page_frames) / args.columns)
        canvas = np.full(
            (header_height + rows * cell_height, label_width + args.columns * width, 3),
            26,
            dtype=np.uint8,
        )
        title = (
            f"BLIND RAW | {video_id}/{action} | f{start}-{end} | "
            f"page {page_index:02d} | {args.sample_fps:g} FPS | no segmentation overlay"
        )
        put_label(canvas, title, (14, 35), 0.72)
        for local_index, frame_index in enumerate(page_frames):
            row = local_index // args.columns
            column = local_index % args.columns
            x = label_width + column * width
            y = header_height + row * cell_height
            add_tile(canvas, load_frame(capture, frame_index), x, y, frame_index, fps, args)
        output_path = args.output_dir / "overview" / (
            f"{video_id}_{action}_page{page_index:02d}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        pages.append(
            {
                "segment_id": f"{video_id}_{action}",
                "mode": "overview",
                "page": page_index,
                "path": str(output_path),
                "frames": page_frames,
            }
        )
        print(f"wrote {output_path}")
    return pages


def focus_samples(center: int, start: int, end: int, args: argparse.Namespace) -> list[int]:
    lower = max(start, center - args.focus_radius)
    upper = min(end, center + args.focus_radius)
    values = list(range(lower, upper + 1, args.focus_step))
    if center not in values:
        values.append(center)
    return sorted(set(values))


def make_focus_pages(
    rows: list[dict[str, str]],
    segments: dict[tuple[str, str], tuple[int, int]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["segment_id"], []).append(row)
    width, height = thumb_size(args)
    cell_height = height + 28
    header_height = 54
    label_width = 250
    pages = []
    for segment_id, segment_rows in grouped.items():
        video_id, action = segment_id.split("_", 1)
        window = segments.get((video_id, action))
        if window is None:
            raise ValueError(f"segment not found in action_segments.csv: {segment_id}")
        video_path = args.videos_dir / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_rows.sort(key=lambda row: int(row["center_frame"]))
        for page_index, offset in enumerate(
            range(0, len(segment_rows), args.focus_boundaries_per_page), start=1
        ):
            page_rows = segment_rows[offset : offset + args.focus_boundaries_per_page]
            samples = [
                focus_samples(int(row["center_frame"]), window[0], window[1], args)
                for row in page_rows
            ]
            rows_per_boundary = max(
                math.ceil(len(values) / args.focus_columns) for values in samples
            )
            block_height = rows_per_boundary * cell_height
            canvas = np.full(
                (header_height + len(page_rows) * block_height, label_width + args.focus_columns * width, 3),
                26,
                dtype=np.uint8,
            )
            title = (
                f"BLIND BOUNDARY FOCUS | {video_id}/{action} | page {page_index:02d} | "
                f"step {args.focus_step} frames | raw only"
            )
            put_label(canvas, title, (14, 35), 0.72)
            manifest_rows = []
            for row_index, (row, frame_values) in enumerate(zip(page_rows, samples)):
                block_y = header_height + row_index * block_height
                put_label(
                    canvas,
                    f"{row.get('label', 'boundary')} f{row['center_frame']}",
                    (10, block_y + 92),
                    0.56,
                )
                for index, frame_index in enumerate(frame_values):
                    if not 0 <= frame_index < frame_count:
                        raise ValueError(f"frame {frame_index} outside video {video_id}")
                    row_offset, column = divmod(index, args.focus_columns)
                    x = label_width + column * width
                    y = block_y + row_offset * cell_height
                    add_tile(canvas, load_frame(capture, frame_index), x, y, frame_index, fps, args)
                manifest_rows.append(
                    {
                        "label": row.get("label", ""),
                        "center_frame": int(row["center_frame"]),
                        "frames": frame_values,
                    }
                )
            output_path = args.output_dir / "focus" / (
                f"{segment_id}_focus_page{page_index:02d}.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            pages.append(
                {
                    "segment_id": segment_id,
                    "mode": "focus",
                    "page": page_index,
                    "path": str(output_path),
                    "rows": manifest_rows,
                }
            )
            print(f"wrote {output_path}")
        capture.release()
    return pages


def main() -> int:
    args = parse_args()
    rows = read_csv(args.action_segments)
    windows = {
        (row["video_id"].zfill(4), row["action"]): (
            int(row["start_frame"]),
            int(row["end_frame"]),
        )
        for row in rows
    }
    manifest: dict[str, Any] = {
        "action_segments": str(args.action_segments.resolve()),
        "videos_dir": str(args.videos_dir.resolve()),
        "sample_fps": args.sample_fps,
        "overview": [],
        "focus": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "action_segments": existing.get("action_segments", manifest["action_segments"]),
                "videos_dir": existing.get("videos_dir", manifest["videos_dir"]),
                "sample_fps": existing.get("sample_fps", manifest["sample_fps"]),
                "overview": existing.get("overview", []),
                "focus": existing.get("focus", []),
            }
        )
    if args.focus_file is None:
        manifest["overview"] = []
        manifest["sample_fps"] = args.sample_fps
        for video_id, action in args.selected_segments:
            key = (video_id, action)
            if key not in windows:
                raise ValueError(f"segment not found in action_segments.csv: {video_id}/{action}")
            start, end = windows[key]
            video_path = args.videos_dir / f"{video_id}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 <= start <= end < frame_count):
                raise ValueError(f"invalid window for {video_id}/{action}: {start}-{end}")
            frames = overview_frames(start, end, fps, args.sample_fps)
            manifest["overview"].extend(
                make_overview_pages(
                    video_id,
                    action,
                    start,
                    end,
                    frames,
                    capture,
                    fps,
                    args,
                )
            )
            capture.release()
    else:
        focus_rows = read_csv(args.focus_file)
        manifest["focus"] = make_focus_pages(focus_rows, windows, args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"overview_pages={len(manifest['overview'])} focus_pages={len(manifest['focus'])}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
