#!/usr/bin/env python3
"""Build C-lite/E-lite v3 analyses from an existing T1-6 analysis directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_7_actions as action_lite
import task1_7_analysis as analysis_v3
import task1_7_confidence as confidence_lite
import task1_7_quality as quality_lite


BATCH_SCHEMA_VERSION = "task1-7-lite-build-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = analysis_v3.load_json(path)
    clips = payload.get("clips") if isinstance(payload, dict) else payload
    if not isinstance(clips, list):
        raise ValueError(f"clip manifest has no clip list: {path}")
    output = {}
    for clip in clips:
        clip_id = str(clip.get("clip_id") or "")
        if not clip_id:
            raise ValueError(f"clip manifest contains a clip without clip_id: {path}")
        if clip_id in output:
            raise ValueError(f"duplicate clip_id in manifest: {clip_id}")
        output[clip_id] = clip
    return output


def _normalized_clip(clip: dict[str, Any]) -> dict[str, Any]:
    action = str(clip.get("action") or clip.get("exercise") or "").strip().lower()
    if not action:
        raise ValueError(f"clip {clip.get('clip_id')} has no action/exercise")
    return {**clip, "action": action}


def _optional_payload(directory: Path | None, clip_id: str) -> dict[str, Any] | None:
    if directory is None:
        return None
    path = directory / f"{clip_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing optional input for {clip_id}: {path}")
    return analysis_v3.load_json(path)


def _complete_rep_count(output: dict[str, Any]) -> int:
    sets = output.get("sets") or []
    default = next(
        (item for item in sets if item.get("is_default")),
        sets[0] if sets else None,
    )
    if not default:
        return 0
    return sum(rep.get("status") == "complete" for rep in default.get("reps") or [])


def _write_checksums(output_dir: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths):
        lines.append(f"{sha256_path(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


def build_lite(
    analysis_dir: Path,
    clip_manifest: Path,
    output_dir: Path,
    *,
    gate_dir: Path | None = None,
    resolution_dir: Path | None = None,
    quality_config: dict[str, Any] | None = None,
    action_config: dict[str, Any] | None = None,
    confidence_config: dict[str, Any] | None = None,
    schema_path: Path = analysis_v3.DEFAULT_SCHEMA,
) -> dict[str, Any]:
    quality_config = quality_config or quality_lite.load_config()
    action_config = action_config or action_lite.load_config()
    confidence_config = confidence_config or confidence_lite.load_config()
    clips = _clip_index(clip_manifest)
    analysis_paths = sorted(analysis_dir.glob("*.json"))
    if not analysis_paths:
        raise ValueError(f"no analysis JSON files found in {analysis_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    written_paths = []
    for analysis_path in analysis_paths:
        analysis = analysis_v3.load_json(analysis_path)
        clip_id = str(analysis.get("clip_id") or analysis_path.stem)
        if clip_id not in clips:
            raise KeyError(f"clip_id {clip_id} is absent from {clip_manifest}")
        output = analysis_v3.build_v3_analysis(
            analysis,
            _normalized_clip(clips[clip_id]),
            gate=_optional_payload(gate_dir, clip_id),
            resolution=_optional_payload(resolution_dir, clip_id),
            quality_config=quality_config,
            action_config=action_config,
            confidence_config=confidence_config,
        )
        analysis_v3.validate_v3_analysis(output, schema_path)
        output_path = output_dir / f"{clip_id}.json"
        analysis_v3.write_json(output_path, output)
        written_paths.append(output_path)
        quality_status = quality_lite.worst_status(
            output.get("quality_evidence") or []
        )
        warn_count = sum(
            item.get("status") == "warn"
            for item in output.get("quality_evidence") or []
        )
        rows.append(
            {
                "clip_id": clip_id,
                "status": output.get("status"),
                "action_status": (output.get("action_verification") or {}).get(
                    "status"
                ),
                "abstained": bool((output.get("confidence") or {}).get("abstained")),
                "abstain_reason": (output.get("confidence") or {}).get(
                    "abstain_reason"
                ),
                "quality_status": quality_status,
                "warn_evidence_count": warn_count,
                "complete_rep_count": _complete_rep_count(output),
                "output": output_path.name,
            }
        )

    summary = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "stage": "T1-7C-lite+E-lite",
        "clip_count": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "action_status_counts": dict(
            sorted(Counter(row["action_status"] for row in rows).items())
        ),
        "abstained_count": sum(row["abstained"] for row in rows),
        "abstain_reason_counts": dict(
            sorted(
                Counter(
                    row["abstain_reason"] for row in rows if row["abstain_reason"]
                ).items()
            )
        ),
        "quality_status_counts": dict(
            sorted(Counter(row["quality_status"] for row in rows).items())
        ),
        "clips": rows,
        "inputs": {
            "analysis_dir": str(analysis_dir.resolve()),
            "clip_manifest": str(clip_manifest.resolve()),
            "clip_manifest_sha256": sha256_path(clip_manifest),
            "gate_dir": str(gate_dir.resolve()) if gate_dir else None,
            "resolution_dir": (
                str(resolution_dir.resolve()) if resolution_dir else None
            ),
        },
        "configs": {
            "quality_schema": quality_config.get("schema_version"),
            "quality_revision": quality_config.get("revision"),
            "action_schema": action_config.get("schema_version"),
            "action_revision": action_config.get("revision"),
            "confidence_schema": confidence_config.get("schema_version"),
            "confidence_revision": confidence_config.get("revision"),
            "calibrator_sha256": confidence_config["core"]["calibrator_sha256"],
        },
    }
    summary_path = output_dir / "summary.json"
    analysis_v3.write_json(summary_path, summary)
    _write_checksums(output_dir, [*written_paths, summary_path])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, default=None)
    parser.add_argument("--resolution-dir", type=Path, default=None)
    parser.add_argument("--quality-config", type=Path, default=quality_lite.DEFAULT_CONFIG)
    parser.add_argument("--action-config", type=Path, default=action_lite.DEFAULT_CONFIG)
    parser.add_argument(
        "--confidence-config",
        type=Path,
        default=confidence_lite.DEFAULT_CONFIG,
    )
    parser.add_argument("--schema", type=Path, default=analysis_v3.DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_lite(
        args.analysis_dir,
        args.clips,
        args.output_dir,
        gate_dir=args.gate_dir,
        resolution_dir=args.resolution_dir,
        quality_config=quality_lite.load_config(args.quality_config),
        action_config=action_lite.load_config(args.action_config),
        confidence_config=confidence_lite.load_config(args.confidence_config),
        schema_path=args.schema,
    )
    print(
        json.dumps(
            {
                "clip_count": summary["clip_count"],
                "status_counts": summary["status_counts"],
                "action_status_counts": summary["action_status_counts"],
                "abstained_count": summary["abstained_count"],
                "abstain_reason_counts": summary["abstain_reason_counts"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
