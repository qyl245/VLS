#!/usr/bin/env python3
"""Freeze expanded Multi-View front-squat GT from reviewed top anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "multiview_front_squat_expanded.json"
DEFAULT_TOPS = ROOT / "configs" / "multiview_front_squat_expanded_tops.json"
DEFAULT_OUTPUT = ROOT / "results" / "multiview_front_squat_expanded" / "annotation"
ANNOTATOR = "codex-visual+dense-contact-sheets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tops", type=Path, default=DEFAULT_TOPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    top_payload = json.loads(args.tops.read_text())
    tops_by_clip = top_payload["top_boundaries"]
    clips = {clip["clip_id"]: clip for clip in manifest["clips"]}
    missing = sorted(set(clips) - set(tops_by_clip))
    extra = sorted(set(tops_by_clip) - set(clips))
    if missing or extra:
        raise ValueError(f"GT/manifest clip mismatch: missing={missing}, extra={extra}")

    rep_rows = []
    label_rows = []
    for clip_id, clip in clips.items():
        tops = [int(value) for value in tops_by_clip[clip_id]]
        frame_count = int(clip["frame_count"])
        if len(tops) < 2 or tops != sorted(set(tops)):
            raise ValueError(f"invalid top anchors for {clip_id}: {tops}")
        if tops[0] < 0 or tops[-1] >= frame_count:
            raise ValueError(f"top anchor outside {clip_id}: {tops[0]}..{tops[-1]}")
        notes = "Front-view dense contact-sheet review; top anchors sampled every 3 frames."
        for rep_id, (start, next_top) in enumerate(zip(tops[:-1], tops[1:]), start=1):
            rep_rows.append(
                {
                    "clip_id": clip_id,
                    "rep_id": rep_id,
                    "start_frame": start,
                    "end_frame": next_top - 1,
                    "bottom_frame": (start + next_top - 1) // 2,
                    "phase_landmark": "top",
                    "boundary_confidence": "medium-high",
                    "annotator": ANNOTATOR,
                    "reviewed_at": top_payload["reviewed_at"],
                    "notes": notes,
                }
            )
        label_rows.append(
            {
                "clip_id": clip_id,
                "expected_complete_reps": len(tops) - 1,
                "label": "complete_squat",
                "review_status": "accept",
                "notes": notes,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reps_path = args.output_dir / "gt_reps.csv"
    labels_path = args.output_dir / "gt_clip_labels.csv"
    rep_fields = list(rep_rows[0])
    with reps_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rep_fields)
        writer.writeheader()
        writer.writerows(rep_rows)
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)

    payload = {
        "schema_version": "multiview-front-squat-gt-v1",
        "frozen": True,
        "reviewed_at": top_payload["reviewed_at"],
        "annotator": ANNOTATOR,
        "frame_domain": "clip-local decoded frame index, inclusive",
        "boundary_protocol": "top_i through top_i+1 - 1",
        "boundary_resolution_frames": 3,
        "manifest": str(args.manifest.resolve()),
        "top_anchors": str(args.tops.resolve()),
        "clip_count": len(label_rows),
        "complete_rep_count": len(rep_rows),
        "files": {
            "gt_reps.csv": sha256_path(reps_path),
            "gt_clip_labels.csv": sha256_path(labels_path),
        },
    }
    write_json(args.output_dir / "gt_manifest.json", payload)
    checksum_lines = [
        f"{payload['files']['gt_reps.csv']}  gt_reps.csv",
        f"{payload['files']['gt_clip_labels.csv']}  gt_clip_labels.csv",
    ]
    (args.output_dir / "annotation_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
