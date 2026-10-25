#!/usr/bin/env python3
"""Extract draft-boundary frame strips for manual rep annotation review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "results" / "segmentation" / "annotation" / "annotation_template.csv"
DEFAULT_VIDEOS = ROOT.parent / "dataset" / "QEVD-FIT" / "QEVD-FIT-COACH" / "long_range_videos"
DEFAULT_OUTPUT = ROOT / "results" / "segmentation" / "annotation" / "frames"
COLORS = {
    "start": (60, 80, 230),
    "end": (220, 170, 40),
    "mid": (60, 190, 220),
    "context": (170, 170, 170),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=12)
    parser.add_argument("--reps-per-page", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frame_samples(row: dict[str, str], offset: int) -> list[tuple[str, int, str]]:
    start = int(row["draft_start_frame"])
    end = int(row["draft_end_frame"])
    midpoint = (start + end) // 2
    samples = [
        ("S-off", start - offset, "context"),
        ("START", start, "start"),
        ("S+off", start + offset, "context"),
        ("MID", midpoint, "mid"),
        ("E-off", end - offset, "context"),
        ("END", end, "end"),
        ("E+off", end + offset, "context"),
    ]
    return [(tag, max(0, index), color) for tag, index, color in samples]


def put_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.48) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def load_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode frame {index}")
    return frame


def make_page(
    segment_id: str,
    page_index: int,
    rows: list[dict[str, str]],
    capture: cv2.VideoCapture,
    fps: float,
    args: argparse.Namespace,
    output_path: Path,
) -> dict:
    thumb_width = args.thumb_width
    thumb_height = round(thumb_width * 9 / 16)
    label_width = 150
    header_height = 50
    cell_height = thumb_height + 30
    canvas = np.full(
        (header_height + len(rows) * cell_height, label_width + 7 * thumb_width, 3),
        28,
        dtype=np.uint8,
    )
    title = f"{segment_id} | draft boundary review | page {page_index:02d} | offset +/-{args.offset} frames"
    put_label(canvas, title, (16, 33), scale=0.72)
    manifest_rows = []
    for row_index, row in enumerate(rows):
        y = header_height + row_index * cell_height
        row_text = f"rep {row['rep_id']}  {row['draft_start_frame']}-{row['draft_end_frame']}"
        put_label(canvas, row_text, (10, y + 76), scale=0.52)
        duration = int(row["draft_end_frame"]) - int(row["draft_start_frame"]) + 1
        put_label(canvas, f"{duration} frames", (10, y + 104), scale=0.48)
        sample_rows = []
        for column, (tag, frame_index, role) in enumerate(frame_samples(row, args.offset)):
            frame = load_frame(capture, frame_index)
            frame = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            x = label_width + column * thumb_width
            canvas[y : y + thumb_height, x : x + thumb_width] = frame
            cv2.rectangle(
                canvas,
                (x, y),
                (x + thumb_width - 1, y + thumb_height - 1),
                COLORS[role],
                3 if role in {"start", "end", "mid"} else 1,
            )
            label = f"{tag} f{frame_index} {frame_index / fps:.2f}s"
            put_label(canvas, label, (x + 5, y + thumb_height - 8), scale=0.45)
            sample_rows.append(
                {
                    "tag": tag,
                    "frame_id": frame_index,
                    "timestamp_s": round(frame_index / fps, 6),
                    "role": role,
                }
            )
        manifest_rows.append(
            {
                "rep_id": int(row["rep_id"]),
                "draft_start_frame": int(row["draft_start_frame"]),
                "draft_end_frame": int(row["draft_end_frame"]),
                "samples": sample_rows,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    return {"segment_id": segment_id, "page": page_index, "path": str(output_path), "rows": manifest_rows}


def main() -> int:
    args = parse_args()
    rows = read_rows(args.annotations)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["segment_id"], []).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for segment_id, segment_rows in grouped.items():
        segment_rows.sort(key=lambda row: int(row["rep_id"]))
        video_id = segment_rows[0]["video_id"]
        video_path = args.videos_dir / f"{video_id}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index, row in enumerate(segment_rows):
            for field in ("draft_start_frame", "draft_end_frame"):
                frame_id = int(row[field])
                if not 0 <= frame_id < frame_count:
                    raise ValueError(f"{segment_id} rep {row['rep_id']} {field}={frame_id} outside video")
        for page_index, start in enumerate(range(0, len(segment_rows), args.reps_per_page), start=1):
            page_rows = segment_rows[start : start + args.reps_per_page]
            output_path = args.output_dir / f"{segment_id}_page{page_index:02d}.jpg"
            pages.append(make_page(segment_id, page_index, page_rows, capture, fps, args, output_path))
            print(f"wrote {output_path}")
        capture.release()
    manifest = {
        "annotations": str(args.annotations.resolve()),
        "videos_dir": str(args.videos_dir.resolve()),
        "offset_frames": args.offset,
        "pages": pages,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"segments={len(grouped)} draft_reps={len(rows)} pages={len(pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
