#!/usr/bin/env python3
"""Assemble a RepChain v3 analysis with C-lite and E-lite safeguards."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_7_actions as action_lite
import task1_7_confidence as confidence_lite
import task1_7_quality as quality_lite


DEFAULT_SCHEMA = ROOT / "schemas" / "task1_7_clip_analysis_v3.schema.json"
ANALYSIS_SCHEMA_VERSION = "task1-7-clip-analysis-v3"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _analysis_frame_count(analysis: dict[str, Any], clip: dict[str, Any]) -> int:
    return int(
        clip.get("frame_count")
        or (clip.get("trim") or {}).get("end_frame", 0) + 1
        or 0
    )


def _timebase(
    analysis: dict[str, Any],
    clip: dict[str, Any],
    resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    frame_count = _analysis_frame_count(analysis, clip)
    if resolution and isinstance(resolution.get("timebase"), dict):
        base = resolution["timebase"]
        return {
            **base,
            "frame_count": int(base.get("frame_count") or frame_count),
            "timestamps_present": bool(base.get("timestamps_present")),
            "variable_fps": bool(base.get("variable_fps")),
            "max_gap_seconds": float(base.get("max_gap_seconds") or 0.0),
            "interpolated_fraction": float(base.get("interpolated_fraction") or 0.0),
        }
    fps = float(clip.get("fps") or 0.0) or None
    return {
        "frame_count": frame_count,
        "source_fps": fps,
        "effective_fps": fps,
        "duration_seconds": (
            round(frame_count / fps, 6) if fps and frame_count else None
        ),
        "timestamps_present": False,
        "variable_fps": False,
        "max_gap_seconds": round(1.0 / fps, 6) if fps else 0.0,
        "interpolated_fraction": 0.0,
    }


def _signal_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    status = str(candidate.get("status") or "insufficient_signal")
    status = {
        "degraded": "ok",
        "ambiguous": "ambiguous",
        "ok": "ok",
        "no_motion": "no_motion",
        "insufficient_signal": "insufficient_signal",
    }.get(status, "insufficient_signal")
    return {
        **copy.deepcopy(candidate),
        "phase_signal": str(candidate.get("phase_signal") or "unknown"),
        "score": max(0.0, float(candidate.get("score") or 0.0)),
        "status": status,
        "periodicity": candidate.get("periodicity"),
        "dynamic_range": candidate.get("dynamic_range"),
        "noise_sigma": candidate.get("noise_sigma"),
        "dynamic_to_noise": candidate.get("dynamic_to_noise"),
        "valid_fraction": max(
            0.0, min(1.0, float(candidate.get("valid_fraction") or 0.0))
        ),
        "max_gap_seconds": max(0.0, float(candidate.get("max_gap_seconds") or 0.0)),
        "multiscale_agreement": max(
            0.0, min(1.0, float(candidate.get("multiscale_agreement") or 0.0))
        ),
    }


def _signal_evidence(
    analysis: dict[str, Any],
    resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    if resolution:
        return {
            "selected_phase_signal": resolution.get("selected_phase_signal"),
            "consensus_score": max(
                0.0, min(1.0, float(resolution.get("consensus_score") or 0.0))
            ),
            "ambiguous": resolution.get("status") == "ambiguous",
            "candidates": [
                _signal_candidate(candidate)
                for candidate in resolution.get("candidates") or []
            ],
            "ambiguity_reason": resolution.get("ambiguity_reason"),
        }
    diagnostics = analysis.get("diagnostics") or {}
    return {
        "selected_phase_signal": diagnostics.get("selected_phase_signal"),
        "consensus_score": max(
            0.0, min(1.0, float(diagnostics.get("periodicity") or 0.0))
        ),
        "ambiguous": analysis.get("status") == "phase_ambiguous",
        "candidates": [],
    }


def _phase_hypotheses(
    resolution: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not resolution:
        return []
    output = []
    for hypothesis in resolution.get("phase_hypotheses") or []:
        if not hypothesis.get("phase_signal"):
            continue
        status = str(hypothesis.get("status") or "insufficient_signal")
        if status not in {"ok", "ambiguous", "rejected", "insufficient_signal"}:
            status = "insufficient_signal"
        output.append(
            {
                **copy.deepcopy(hypothesis),
                "hypothesis_id": str(
                    hypothesis.get("hypothesis_id") or f"phase_{len(output) + 1}"
                ),
                "phase_signal": str(hypothesis["phase_signal"]),
                "score": max(0.0, float(hypothesis.get("score") or 0.0)),
                "period_seconds": hypothesis.get("period_seconds"),
                "status": status,
            }
        )
    return output


def _apply_phase_ambiguity(analysis: dict[str, Any]) -> None:
    diagnostics = analysis.setdefault("diagnostics", {})
    diagnostics["pre_phase_ambiguity_structure"] = {
        "status": analysis.get("status"),
        "set_count": len(analysis.get("sets") or []),
        "rep_count": sum(
            len(item.get("reps") or []) for item in analysis.get("sets") or []
        ),
    }
    analysis["status"] = "phase_ambiguous"
    analysis["sets"] = []
    analysis["rejected_candidates"] = []
    analysis["non_rep_intervals"] = []
    diagnostics["selected_top_events"] = []


def build_v3_analysis(
    analysis: dict[str, Any],
    clip: dict[str, Any],
    *,
    gate: dict[str, Any] | None,
    resolution: dict[str, Any] | None = None,
    quality_config: dict[str, Any] | None = None,
    action_config: dict[str, Any] | None = None,
    confidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality_config = quality_config or quality_lite.load_config()
    action_config = action_config or action_lite.load_config()
    confidence_config = confidence_config or confidence_lite.load_config()
    calibrator = confidence_lite.load_calibrator(confidence_config)
    frame_count = _analysis_frame_count(analysis, clip)

    signal_evidence = _signal_evidence(analysis, resolution)
    quality_output = quality_lite.apply_quality_lite(
        analysis,
        gate,
        quality_config,
        frame_count=frame_count,
    )
    if resolution is not None and resolution.get("status") == "ambiguous":
        if quality_output.get("status") not in {
            "rejected_quality",
            "not_evaluable_quality",
        }:
            _apply_phase_ambiguity(quality_output)
    verification = action_lite.verify_action(
        analysis,
        str(clip.get("action") or clip.get("exercise") or "unknown"),
        fps=float(clip.get("fps") or 0.0) or None,
        config=action_config,
        signal_evidence=signal_evidence,
    )
    action_output = action_lite.apply_action_verification(
        quality_output,
        verification,
        action_config,
    )
    output = confidence_lite.apply_confidence_lite(
        action_output,
        quality_evidence=quality_output["quality_evidence"],
        action_verification=verification,
        signal_evidence=signal_evidence,
        config=confidence_config,
        calibrator=calibrator,
    )
    output["schema_version"] = ANALYSIS_SCHEMA_VERSION
    output["algorithm"] = {"name": "repchain", "version": "v3"}
    output["timebase"] = _timebase(analysis, clip, resolution)
    output["signal_evidence"] = signal_evidence
    output["phase_hypotheses"] = _phase_hypotheses(resolution)
    output["action_verification"] = verification
    output.setdefault("diagnostics", {})["v3_assembly"] = {
        "stage": "T1-7C-lite+E-lite",
        "quality_config_schema": quality_config.get("schema_version"),
        "action_config_schema": action_config.get("schema_version"),
        "confidence_config_schema": confidence_config.get("schema_version"),
        "resolution_supplied": resolution is not None,
    }
    return output


def validate_v3_analysis(
    analysis: dict[str, Any],
    schema_path: Path = DEFAULT_SCHEMA,
) -> None:
    from jsonschema import Draft202012Validator

    schema = load_json(schema_path)
    Draft202012Validator(schema).validate(analysis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, default=None)
    parser.add_argument("--gate", type=Path, default=None)
    parser.add_argument("--quality-config", type=Path, default=quality_lite.DEFAULT_CONFIG)
    parser.add_argument("--action-config", type=Path, default=action_lite.DEFAULT_CONFIG)
    parser.add_argument(
        "--confidence-config",
        type=Path,
        default=confidence_lite.DEFAULT_CONFIG,
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_v3_analysis(
        load_json(args.analysis),
        load_json(args.clip),
        gate=load_json(args.gate) if args.gate else None,
        resolution=load_json(args.resolution) if args.resolution else None,
        quality_config=quality_lite.load_config(args.quality_config),
        action_config=action_lite.load_config(args.action_config),
        confidence_config=confidence_lite.load_config(args.confidence_config),
    )
    validate_v3_analysis(output, args.schema)
    write_json(args.output, output)
    print(json.dumps({
        "status": output.get("status"),
        "action_status": (output.get("action_verification") or {}).get("status"),
        "abstained": (output.get("confidence") or {}).get("abstained"),
        "abstain_reason": (output.get("confidence") or {}).get("abstain_reason"),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
