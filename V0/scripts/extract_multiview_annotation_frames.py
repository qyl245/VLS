#!/usr/bin/env python3
"""Extract coarse and dense contact sheets for Multi-View squat annotation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "multiview_front_squat_expanded.json"
DEFAULT_OUTPUT = ROOT / "results" / "multiview_front_squat_expanded" / "annotation" / "frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dense-stride", type=int, default=3)
    parser.add_argument("--coarse-stride", type=int, default=15)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    return parser.parse_args()


def put_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def load_frames(path: Path, indices: list[int]) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot decode {path} frame {frame_index}")
        frames.append(frame)
    cap.release()
    return frames, fps


def make_pages(
    clip_id: str,
    path: Path,
    frame_count: int,
    stride: int,
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    indices = list(range(0, frame_count, stride))
    frames, fps = load_frames(path, indices)
    per_page = args.columns * args.rows
    pages = []
    for page_number, start in enumerate(range(0, len(indices), per_page), start=1):
        page_indices = indices[start : start + per_page]
        page_frames = frames[start : start + per_page]
        thumb_height = round(args.thumb_width * page_frames[0].shape[0] / page_frames[0].shape[1])
        cell_height = thumb_height + 26
        header_height = 48
        canvas = np.full(
            (header_height + args.rows * cell_height, args.columns * args.thumb_width, 3),
            28,
            dtype=np.uint8,
        )
        title = (
            f"{clip_id} | {mode} every {stride} frames | page {page_number:02d} | "
            f"frames {page_indices[0]}-{page_indices[-1]}"
        )
        put_label(canvas, title, (16, 32), 0.68)
        for cell, (frame_index, frame) in enumerate(zip(page_indices, page_frames)):
            row, column = divmod(cell, args.columns)
            x = column * args.thumb_width
            y = header_height + row * cell_height
            thumb = cv2.resize(
                frame,
                (args.thumb_width, thumb_height),
                interpolation=cv2.INTER_AREA,
            )
            canvas[y : y + thumb_height, x : x + args.thumb_width] = thumb
            cv2.rectangle(
                canvas,
                (x, y),
                (x + args.thumb_width - 1, y + thumb_height - 1),
                (80, 80, 80),
                1,
            )
            put_label(
                canvas,
                f"f{frame_index} {frame_index / fps:.2f}s",
                (x + 4, y + thumb_height - 7),
                0.43,
            )
        output_path = output_dir / f"{path.stem}_{mode}_p{page_number:02d}.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        pages.append(
            {
                "mode": mode,
                "page": page_number,
                "path": str(output_path),
                "frame_start": page_indices[0],
                "frame_end": page_indices[-1],
                "sample_indices": page_indices,
            }
        )
        print(f"wrote {output_path}", flush=True)
    return pages


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for clip in manifest["clips"]:
        path = Path(clip["video_path"])
        frame_count = int(clip["frame_count"])
        pages.extend(
            make_pages(
                clip["clip_id"],
                path,
                frame_count,
                args.coarse_stride,
                "coarse",
                args.output_dir,
                args,
            )
        )
        pages.extend(
            make_pages(
                clip["clip_id"],
                path,
                frame_count,
                args.dense_stride,
                "dense",
                args.output_dir,
                args,
            )
        )
    payload = {
        "schema_version": "multiview-front-annotation-frames-v1",
        "manifest": str(args.manifest.resolve()),
        "dense_stride": args.dense_stride,
        "coarse_stride": args.coarse_stride,
        "page_count": len(pages),
        "pages": pages,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"clips={len(manifest['clips'])} pages={len(pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
