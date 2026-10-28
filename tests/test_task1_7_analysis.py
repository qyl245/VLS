from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_quality as base_quality
import task1_7_analysis as analysis_v3
import build_task1_7_lite as lite_build


def phase_state(frame: int, angle: float, *, stable: float = 0.9) -> dict:
    return {
        "start_frame": max(0, frame - 4),
        "end_frame": frame + 4,
        "anchor_frame": frame,
        "stable_fraction": stable,
        "boundary_slope": 0.0,
        "mean_angle": angle,
    }


def make_rep(rep_id: int, start: int, phase_signal: str) -> dict:
    return {
        "rep_id": rep_id,
        "start_frame": start,
        "end_frame": start + 39,
        "start_seconds": round(start / 30.0, 6),
        "end_seconds": round((start + 39) / 30.0, 6),
        "top": phase_state(start, 160.0),
        "bottom": phase_state(start + 20, 90.0),
        "phase_signal": phase_signal,
        "rep_unit": "cycle",
        "status": "complete",
        "partial_side": None,
        "confidence": {"complete": 0.8, "in_set": 0.8, "combined": 0.8},
        "endpoint_confidence": 0.8,
        "quality_flags": [],
    }


def make_analysis(
    *,
    phase_signal: str = "knee_flexion_bilateral_mean",
    status: str = "ok",
) -> dict:
    reps = [make_rep(index + 1, 10 + index * 40, phase_signal) for index in range(3)]
    return {
        "schema_version": "task1-6-clip-analysis-v2",
        "clip_id": "synthetic_clip",
        "algorithm": {"name": "repchain", "version": "v2"},
        "status": status,
        "anchor_policy": "plateau_midpoint",
        "sets": [
            {
                "set_id": 1,
                "start_frame": 10,
                "end_frame": 129,
                "start_seconds": round(10 / 30.0, 6),
                "end_seconds": round(129 / 30.0, 6),
                "status": "complete_set",
                "chain_confidence": 0.9,
                "is_default": True,
                "reps": reps,
                "rejected_candidates": [],
                "non_rep_intervals": [],
            }
        ],
        "rejected_candidates": [],
        "non_rep_intervals": [],
        "diagnostics": {
            "selected_phase_signal": phase_signal,
            "periodicity": 0.8,
            "selected_top_events": [
                {"frame": rep["start_frame"], "confidence": {"combined": 0.8}}
                for rep in reps
            ],
        },
    }


def make_clip(action: str = "squat") -> dict:
    return {
        "clip_id": "synthetic_clip",
        "exercise": action,
        "action": action,
        "fps": 30.0,
        "frame_count": 200,
    }


def resolution(
    *,
    status: str = "ok",
    selected: str | None = "knee_flexion_bilateral_mean",
    ambiguous: bool = False,
) -> dict:
    candidates = []
    if selected:
        candidates.append(
            {
                "phase_signal": selected,
                "score": 0.8,
                "status": "ok",
                "periodicity": 0.8,
                "dynamic_range": 60.0,
                "noise_sigma": 1.0,
                "dynamic_to_noise": 60.0,
                "valid_fraction": 1.0,
                "max_gap_seconds": 0.0,
                "multiscale_agreement": 1.0,
            }
        )
    return {
        "status": status,
        "selected_phase_signal": selected,
        "consensus_score": 0.8 if not ambiguous else 0.2,
        "ambiguity_reason": "phase_signal_family_disagreement" if ambiguous else None,
        "candidates": candidates,
        "phase_hypotheses": [
            {
                "hypothesis_id": "phase_1",
                "phase_signal": selected or "unknown",
                "score": 0.8,
                "period_seconds": 1.33,
                "status": "ok",
            }
        ]
        if selected
        else [],
        "timebase": {
            "frame_count": 200,
            "source_fps": 30.0,
            "effective_fps": 30.0,
            "timestamps_present": False,
            "variable_fps": False,
            "max_gap_seconds": 0.033333,
            "interpolated_fraction": 0.0,
        },
    }


class AnalysisV3Tests(unittest.TestCase):
    def test_supported_output_matches_v3_schema(self) -> None:
        output = analysis_v3.build_v3_analysis(
            make_analysis(),
            make_clip(),
            gate=None,
            resolution=resolution(),
        )
        analysis_v3.validate_v3_analysis(output)
        self.assertEqual(output["schema_version"], "task1-7-clip-analysis-v3")
        self.assertEqual(output["algorithm"], {"name": "repchain", "version": "v3"})
        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["action_verification"]["status"], "supported")
        self.assertFalse(output["confidence"]["abstained"])
        self.assertEqual(len(output["quality_evidence"]), 5)
        self.assertEqual(output["phase_hypotheses"][0]["hypothesis_id"], "phase_1")
        self.assertEqual(output["timebase"]["frame_count"], 200)

    def test_manifest_exercise_field_is_accepted_as_action(self) -> None:
        clip = make_clip()
        clip.pop("action")
        output = analysis_v3.build_v3_analysis(
            make_analysis(),
            clip,
            gate=None,
            resolution=resolution(),
        )
        analysis_v3.validate_v3_analysis(output)
        self.assertEqual(output["action_verification"]["status"], "supported")

    def test_phase_ambiguity_fails_closed_before_action(self) -> None:
        output = analysis_v3.build_v3_analysis(
            make_analysis(),
            make_clip(),
            gate=None,
            resolution=resolution(status="ambiguous", ambiguous=True),
        )
        analysis_v3.validate_v3_analysis(output)
        self.assertEqual(output["status"], "phase_ambiguous")
        self.assertTrue(output["signal_evidence"]["ambiguous"])
        self.assertEqual(output["sets"], [])
        self.assertTrue(output["confidence"]["abstained"])
        self.assertEqual(output["confidence"]["abstain_reason"], "phase_ambiguous")
        self.assertEqual(
            output["diagnostics"]["action_verification_disposition"],
            "status_precedence",
        )

    def test_action_mismatch_and_unsupported_clear_structure(self) -> None:
        mismatch = analysis_v3.build_v3_analysis(
            make_analysis(phase_signal="elbow_flexion_bilateral_mean"),
            make_clip("squat"),
            gate=None,
            resolution=resolution(selected="elbow_flexion_bilateral_mean"),
        )
        analysis_v3.validate_v3_analysis(mismatch)
        self.assertEqual(mismatch["status"], "action_mismatch")
        self.assertEqual(mismatch["sets"], [])
        self.assertEqual(
            mismatch["confidence"]["abstain_reason"], "action_mismatch"
        )

        unsupported = analysis_v3.build_v3_analysis(
            make_analysis(),
            make_clip("lunge"),
            gate=None,
            resolution=resolution(),
        )
        analysis_v3.validate_v3_analysis(unsupported)
        self.assertEqual(unsupported["status"], "unsupported_action")
        self.assertEqual(unsupported["sets"], [])
        self.assertEqual(
            unsupported["confidence"]["abstain_reason"], "action_unsupported"
        )

    def test_quality_reject_has_precedence_over_action(self) -> None:
        gate = base_quality.make_quality_gate(
            "reject",
            reasons=["pose_missing"],
            source="pose",
            scope="clip",
        )
        output = analysis_v3.build_v3_analysis(
            make_analysis(),
            make_clip("lunge"),
            gate=gate,
            resolution=resolution(),
        )
        analysis_v3.validate_v3_analysis(output)
        self.assertEqual(output["status"], "rejected_quality")
        self.assertEqual(output["sets"], [])
        self.assertEqual(output["quality_evidence"][0]["status"], "reject")
        self.assertEqual(output["confidence"]["abstain_reason"], "quality_reject")

    def test_invalid_v3_output_is_rejected(self) -> None:
        output = analysis_v3.build_v3_analysis(
            make_analysis(),
            make_clip(),
            gate=None,
            resolution=resolution(),
        )
        invalid = copy.deepcopy(output)
        del invalid["quality_evidence"]
        with self.assertRaises(ValidationError):
            analysis_v3.validate_v3_analysis(invalid)

    def test_build_does_not_mutate_inputs(self) -> None:
        analysis = make_analysis()
        clip = make_clip()
        resolved = resolution()
        before = copy.deepcopy((analysis, clip, resolved))
        analysis_v3.build_v3_analysis(
            analysis,
            clip,
            gate=None,
            resolution=resolved,
        )
        self.assertEqual((analysis, clip, resolved), before)

    def test_batch_builder_writes_valid_outputs_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_dir = root / "analysis"
            output_dir = root / "output"
            analysis_dir.mkdir()
            (analysis_dir / "synthetic_clip.json").write_text(
                json.dumps(make_analysis())
            )
            manifest = root / "clips.json"
            manifest.write_text(json.dumps({"clips": [make_clip()]}))

            summary = lite_build.build_lite(analysis_dir, manifest, output_dir)

            self.assertEqual(summary["clip_count"], 1)
            self.assertEqual(summary["status_counts"], {"ok": 1})
            self.assertEqual(summary["action_status_counts"], {"supported": 1})
            self.assertTrue((output_dir / "synthetic_clip.json").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "checksums.sha256").is_file())
            output = analysis_v3.load_json(output_dir / "synthetic_clip.json")
            analysis_v3.validate_v3_analysis(output)


if __name__ == "__main__":
    unittest.main()
