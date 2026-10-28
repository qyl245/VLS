from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_quality as base_quality
import task1_7_quality as quality


def make_analysis() -> dict:
    return {
        "clip_id": "synthetic_clip",
        "status": "ok",
        "sets": [
            {
                "set_id": 1,
                "start_frame": 10,
                "end_frame": 120,
                "status": "complete_set",
                "chain_confidence": 0.9,
                "is_default": True,
                "reps": [
                    {
                        "rep_id": 1,
                        "start_frame": 10,
                        "end_frame": 45,
                        "status": "complete",
                        "confidence": {"complete": 0.8, "in_set": 0.8, "combined": 0.8},
                        "quality_flags": [],
                    },
                    {
                        "rep_id": 2,
                        "start_frame": 45,
                        "end_frame": 80,
                        "status": "complete",
                        "confidence": {"complete": 0.8, "in_set": 0.8, "combined": 0.8},
                        "quality_flags": [],
                    },
                ],
            }
        ],
        "rejected_candidates": [{"candidate_id": "top_1"}],
        "non_rep_intervals": [{"start_frame": 0, "end_frame": 9}],
        "diagnostics": {"selected_top_events": [{"frame": 10}, {"frame": 45}]},
    }


class QualityLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = quality.load_config()

    def test_pass_preserves_structure_and_emits_evidence(self) -> None:
        analysis = make_analysis()
        gate = base_quality.make_quality_gate("pass", source="pose")
        output = quality.apply_quality_lite(
            analysis,
            gate,
            self.config,
            frame_count=121,
        )
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["sets"]), 1)
        self.assertEqual(len(output["sets"][0]["reps"]), 2)
        self.assertEqual(
            [item["evidence_id"] for item in output["quality_evidence"]],
            ["clip:quality_gate", "set:1", "rep:1:1", "rep:1:2"],
        )
        self.assertTrue(
            all(item["status"] == "pass" for item in output["quality_evidence"])
        )
        self.assertFalse(output["diagnostics"]["c_lite_quality"]["new_blocking_rules"])

    def test_warn_gate_keeps_output_and_adds_clip_evidence(self) -> None:
        gate = base_quality.make_quality_gate(
            "warn",
            reasons=["pose_jitter"],
            source="pose",
        )
        output = quality.apply_quality_lite(make_analysis(), gate, self.config)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["sets"][0]["reps"]), 2)
        self.assertEqual(output["quality_evidence"][0]["status"], "warn")
        self.assertEqual(output["quality_evidence"][0]["reasons"], ["pose_jitter"])

    def test_reject_and_unknown_reuse_fail_closed_gate(self) -> None:
        for status, expected in (
            ("reject", "rejected_quality"),
            ("unknown", "not_evaluable_quality"),
        ):
            gate = base_quality.make_quality_gate(
                status,
                reasons=["bad_pose"],
                source="pose",
                scope="clip",
            )
            output = quality.apply_quality_lite(make_analysis(), gate, self.config)
            self.assertEqual(output["status"], expected)
            self.assertEqual(output["sets"], [])
            self.assertEqual(output["rejected_candidates"], [])
            self.assertEqual(output["quality_evidence"][0]["status"], status)
            self.assertEqual(
                output["diagnostics"]["pre_gate_structure"]["rep_count"], 2
            )

    def test_set_and_rep_warnings_are_reporting_only(self) -> None:
        analysis = make_analysis()
        analysis["sets"][0]["status"] = "partial_set"
        analysis["sets"][0]["reps"][1]["status"] = "endpoint_partial"
        analysis["sets"][0]["reps"][1]["quality_flags"] = ["low_evidence"]
        gate = base_quality.make_quality_gate("pass", source="pose")
        output = quality.apply_quality_lite(analysis, gate, self.config)
        evidence = {item["evidence_id"]: item for item in output["quality_evidence"]}
        self.assertEqual(evidence["set:1"]["status"], "warn")
        self.assertEqual(evidence["rep:1:1"]["status"], "pass")
        self.assertEqual(evidence["rep:1:2"]["status"], "warn")
        self.assertEqual(
            evidence["rep:1:2"]["reasons"],
            ["rep_status:endpoint_partial", "low_evidence"],
        )
        self.assertEqual(output["status"], "ok")
        self.assertEqual(len(output["sets"][0]["reps"]), 2)

    def test_worst_status_uses_explicit_scope(self) -> None:
        evidence = copy.deepcopy(
            quality.build_quality_evidence(
                make_analysis(),
                base_quality.make_quality_gate("pass", source="pose"),
                self.config,
            )
        )
        evidence[2]["status"] = "warn"
        self.assertEqual(quality.worst_status(evidence), "warn")
        self.assertEqual(quality.worst_status(evidence, scopes=("clip", "set")), "pass")
        self.assertEqual(quality.worst_status([], scopes=("clip",)), "unknown")

    def test_config_keeps_existing_gate_boundary(self) -> None:
        self.assertTrue(self.config["core"]["dataset_independent"])
        self.assertTrue(self.config["core"]["inherit_base_quality_gate"])
        self.assertFalse(self.config["core"]["add_new_blocking_rules"])
        with self.assertRaises(ValueError):
            quality.load_config(ROOT / "configs" / "task1_7_actions.json")


if __name__ == "__main__":
    unittest.main()
