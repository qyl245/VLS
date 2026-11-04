#!/usr/bin/env python3
"""Build the reviewed Multi-View subject_001 squat GT tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "multiview_squat_clips.json"
OUTPUT_DIR = ROOT / "results" / "multiview_squat" / "annotation"
REVIEWED_AT = "2026-11-02"
ANNOTATOR = "codex-visual+phase-review"

# Each list contains clip-local top-phase anchors. Rep n is [top_n, top_n+1 - 1].
# Front clips are frame-by-frame reviewed. Side/diagonal anchors are provisional
# because those recordings contain severe rolling/defocus blur; their GT must not
# be treated as a high-confidence boundary benchmark.
TOP_BOUNDARIES = {
    "mv_s001_squat_good_front": [0, 30, 54, 78, 101, 126, 152, 178, 204, 231, 259, 288, 316, 343, 370, 399, 427, 455],
    "mv_s001_squat_bad_front": [0, 28, 58, 83, 119, 145, 176, 207, 238, 275, 302, 333, 366, 397, 428, 461],
    "mv_s001_squat_good_side": [2, 28, 54, 80, 106, 132, 158, 184, 210, 236, 262, 288, 314, 340, 366, 392, 418, 444, 470],
    "mv_s001_squat_bad_side": [18, 45, 72, 99, 126, 153, 180, 207, 234, 261, 288, 315, 342, 369, 396, 423, 450, 477],
    "mv_s001_squat_good_diagonal": [0, 21, 43, 70, 95, 117, 141, 167, 195, 222, 249, 279, 309, 335, 362, 391, 420, 448, 475],
    "mv_s001_squat_bad_diagonal": [0, 29, 60, 87, 117, 148, 180, 212, 244, 275, 306, 338, 369, 400, 432, 464, 491],
}

CLIP_REVIEW = {
    "mv_s001_squat_good_front": ("high", "accept", "Clear frontal clip; top-phase anchors verified in one-frame strips."),
    "mv_s001_squat_bad_front": ("high", "accept", "Clear frontal clip; top-phase anchors verified in one-frame strips."),
    "mv_s001_squat_good_side": ("medium", "provisional", "Side view has motion blur; rep count and top anchors reviewed against dense raw-frame strips."),
    "mv_s001_squat_bad_side": ("medium", "provisional", "Side view has motion blur; rep count and top anchors reviewed against dense raw-frame strips."),
    "mv_s001_squat_good_diagonal": ("low", "provisional", "Diagonal view has severe rolling/defocus blur; boundaries are phase-reviewed but not frame-perfect."),
    "mv_s001_squat_bad_diagonal": ("low", "provisional", "Diagonal view has severe rolling/defocus blur; boundaries are phase-reviewed but not frame-perfect."),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    clips = {clip["clip_id"]: clip for clip in manifest["clips"]}
    missing = sorted(set(clips) - set(TOP_BOUNDARIES))
    extra = sorted(set(TOP_BOUNDARIES) - set(clips))
    if missing or extra:
        raise ValueError(f"GT/manifest clip mismatch: missing={missing}, extra={extra}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rep_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for clip_id, tops in TOP_BOUNDARIES.items():
        clip = clips[clip_id]
        frame_count = int(clip["frame_count"])
        if tops != sorted(tops) or len(tops) < 2:
            raise ValueError(f"invalid top anchors for {clip_id}: {tops}")
        if tops[0] < 0 or tops[-1] >= frame_count:
            raise ValueError(f"top anchor outside clip for {clip_id}: {tops[0]}..{tops[-1]}")
        confidence, review_status, notes = CLIP_REVIEW[clip_id]
        for rep_index, (start, next_top) in enumerate(zip(tops[:-1], tops[1:]), start=1):
            if next_top <= start:
                raise ValueError(f"non-positive rep interval for {clip_id}: {start}..{next_top}")
            rep_rows.append(
                {
                    "clip_id": clip_id,
                    "rep_id": rep_index,
                    "start_frame": start,
                    "end_frame": next_top - 1,
                    "bottom_frame": (start + next_top - 1) // 2,
                    "phase_landmark": "top",
                    "boundary_confidence": confidence,
                    "annotator": ANNOTATOR,
                    "reviewed_at": REVIEWED_AT,
                    "notes": notes,
                }
            )
        label_rows.append(
            {
                "clip_id": clip_id,
                "expected_complete_reps": len(tops) - 1,
                "label": "complete_squat",
                "review_status": review_status,
                "notes": notes,
            }
        )

    reps_path = OUTPUT_DIR / "gt_reps.csv"
    labels_path = OUTPUT_DIR / "gt_clip_labels.csv"
    fields = [
        "clip_id",
        "rep_id",
        "start_frame",
        "end_frame",
        "bottom_frame",
        "phase_landmark",
        "boundary_confidence",
        "annotator",
        "reviewed_at",
        "notes",
    ]
    with reps_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rep_rows)
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip_id", "expected_complete_reps", "label", "review_status", "notes"],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    manifest_payload = {
        "schema_version": "multiview-squat-gt-v1",
        "frozen": True,
        "reviewed_at": REVIEWED_AT,
        "annotator": ANNOTATOR,
        "frame_domain": "clip-local decoded frame index, inclusive",
        "boundary_protocol": "top_i through top_i+1 - 1",
        "clip_count": len(label_rows),
        "complete_rep_count": len(rep_rows),
        "high_confidence_clips": ["mv_s001_squat_good_front", "mv_s001_squat_bad_front"],
        "provisional_clips": [
            "mv_s001_squat_good_side",
            "mv_s001_squat_bad_side",
            "mv_s001_squat_good_diagonal",
            "mv_s001_squat_bad_diagonal",
        ],
        "files": {
            "gt_reps.csv": sha256_path(reps_path),
            "gt_clip_labels.csv": sha256_path(labels_path),
        },
        "environment": "VLS/V0/.venv-mediapipe (workspace .venv unavailable)",
    }
    write_json(OUTPUT_DIR / "gt_manifest.json", manifest_payload)
    checksum_lines = [
        f"{manifest_payload['files']['gt_reps.csv']}  gt_reps.csv",
        f"{manifest_payload['files']['gt_clip_labels.csv']}  gt_clip_labels.csv",
    ]
    (OUTPUT_DIR / "annotation_checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    (OUTPUT_DIR / "README.md").write_text(
        "# Multi-View subject_001 squat annotation\n\n"
        "Reviewed on 2026-11-02 using dense raw-frame strips and phase plots. "
        "Frames are clip-local decoded indices; each rep is the closed interval "
        "`[top_i, top_i+1 - 1]`.\n\n"
        "- Front clips: high-confidence frame-by-frame boundary review.\n"
        "- Side/diagonal clips: provisional GT because the recordings contain "
        "severe rolling/defocus blur. They are suitable for coverage and "
        "abstention diagnostics, not a high-confidence boundary benchmark.\n"
        "- `bottom_frame` is the interval midpoint for bookkeeping; RepChain "
        "evaluation uses only the start/end boundaries.\n"
    )
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
