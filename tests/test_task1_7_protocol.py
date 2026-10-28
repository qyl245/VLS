from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_clip_v3 as protocol


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def minimal_analysis() -> dict:
    return {
        "schema_version": "task1-7-clip-analysis-v3",
        "clip_id": "clip_a",
        "algorithm": {"name": "repchain", "version": "v3"},
        "status": "ok",
        "timebase": {
            "frame_count": 10,
            "source_fps": 30.0,
            "effective_fps": 30.0,
            "duration_seconds": 0.3,
            "timestamps_present": False,
            "variable_fps": False,
            "max_gap_seconds": 0.0,
            "interpolated_fraction": 0.0,
        },
        "signal_evidence": {
            "selected_phase_signal": "left_elbow_flexion",
            "consensus_score": 0.9,
            "ambiguous": False,
            "candidates": [
                {
                    "phase_signal": "left_elbow_flexion",
                    "score": 0.9,
                    "status": "ok",
                    "periodicity": 0.8,
                    "period_seconds": 1.2,
                    "dynamic_range": 60.0,
                    "noise_sigma": 1.0,
                    "dynamic_to_noise": 60.0,
                    "valid_fraction": 1.0,
                    "max_gap_seconds": 0.0,
                    "multiscale_agreement": 1.0,
                }
            ],
        },
        "phase_hypotheses": [
            {
                "hypothesis_id": "phase_1",
                "phase_signal": "left_elbow_flexion",
                "score": 0.9,
                "period_seconds": 1.2,
                "status": "ok",
                "evidence": {},
            }
        ],
        "action_verification": {
            "action": "pushup",
            "status": "supported",
            "supported": True,
            "confidence": 0.9,
            "reasons": [],
            "evidence": {},
        },
        "quality_evidence": [],
        "confidence": {
            "components": {"signal": 0.9, "chain": 0.9},
            "calibrated": {
                "complete": 0.9,
                "in_set": 0.9,
                "combined": 0.81,
            },
            "abstained": False,
            "abstain_reason": None,
        },
        "sets": [],
        "rejected_candidates": [],
        "non_rep_intervals": [],
        "diagnostics": {},
    }


class AnalysisSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_json(
            ROOT / "schemas" / "task1_7_clip_analysis_v3.schema.json"
        )
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_minimal_v3_analysis_is_valid(self) -> None:
        self.validator.validate(minimal_analysis())

    def test_missing_quality_evidence_is_invalid(self) -> None:
        analysis = minimal_analysis()
        del analysis["quality_evidence"]
        with self.assertRaises(ValidationError):
            self.validator.validate(analysis)

    def test_invalid_action_verification_status_is_rejected(self) -> None:
        analysis = minimal_analysis()
        analysis["action_verification"]["status"] = "maybe"
        with self.assertRaises(ValidationError):
            self.validator.validate(analysis)


class ProtocolTests(unittest.TestCase):
    def make_args(self, output_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            build_protocol=True,
            baseline_dir=ROOT / "results" / "task1_6" / "evaluation_quality_motion",
            sources_config=ROOT / "configs" / "task1_6_sources.json",
            t1_6_protocol=ROOT / "configs" / "task1_6_eval_protocol.json",
            schema=ROOT / "schemas" / "task1_7_clip_analysis_v3.schema.json",
            output_dir=output_dir,
            config_output=output_dir / "task1_7_protocol.json",
            generated_at="2026-10-27T00:00:00+00:00",
        )

    def test_build_protocol_lock_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            validation = protocol.build_protocol(self.make_args(output_dir))
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["baseline_decision"], "NO_GO")
            self.assertEqual(validation["counts"]["fit3d_dev_clips"], 32)
            self.assertEqual(
                validation["counts"]["fit3d_regression_holdout_clips"], 32
            )
            self.assertEqual(validation["counts"]["fit3d_smoke_clips"], 6)
            self.assertEqual(validation["counts"]["qevd_sealed_segments"], 6)
            self.assertEqual(validation["counts"]["qevd_sealed_annotation_ready"], 0)

            config = load_json(output_dir / "task1_7_protocol.json")
            self.assertEqual(config["stage"], "T1-7A")
            self.assertTrue(config["algorithm_core"]["dataset_independent"])
            self.assertFalse(
                config["generalization"]["allow_cross_domain_claim"]
            )
            self.assertEqual(config["phase_status"]["T1-7B"], "not_started")

            split_lock = load_json(output_dir / "split_lock.json")
            self.assertTrue(split_lock["locked"])
            self.assertTrue(split_lock["fit3d"]["holdout_is_observed"])
            self.assertFalse(
                split_lock["qevd_sealed"]["formal_metric_evaluable"]
            )

    def test_checksums_match_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            protocol.build_protocol(self.make_args(output_dir))
            checksum_lines = (
                output_dir / "checksums.sha256"
            ).read_text().splitlines()
            self.assertEqual(len(checksum_lines), 4)
            for line in checksum_lines:
                expected, name = line.split("  ", 1)
                observed = hashlib.sha256(
                    (output_dir / name).read_bytes()
                ).hexdigest()
                self.assertEqual(observed, expected)

    def test_baseline_mismatch_fails_closed(self) -> None:
        summary = load_json(
            ROOT
            / "results"
            / "task1_6"
            / "evaluation_quality_motion"
            / "summary.json"
        )
        summary["fit3d_holdout"]["primary_camera"][
            "event_abs_error_median_frames"
        ] = 5.0
        with self.assertRaises(ValueError):
            protocol.validate_baseline(summary)


if __name__ == "__main__":
    unittest.main()
