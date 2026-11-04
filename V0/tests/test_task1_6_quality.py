from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_quality as quality


def make_analysis() -> dict:
    return {
        "clip_id": "clip_a",
        "status": "ok",
        "sets": [
            {
                "set_id": 1,
                "start_frame": 10,
                "end_frame": 80,
                "chain_confidence": 0.9,
                "is_default": True,
                "reps": [
                    {"rep_id": 1, "status": "complete"},
                    {"rep_id": 2, "status": "complete"},
                ],
            }
        ],
        "rejected_candidates": [{"candidate_id": "top_1"}],
        "non_rep_intervals": [{"start_frame": 0, "end_frame": 9}],
        "diagnostics": {
            "selected_top_events": [{"frame": 10}, {"frame": 40}],
        },
    }


class QualityGateTests(unittest.TestCase):
    def test_legacy_status_mapping_and_reasons(self) -> None:
        reject = quality.from_legacy_quality(
            {
                "quality_status": "reject",
                "quality": {"failures": ["pose_low"], "warnings": ["ignored"]},
            }
        )
        self.assertEqual(reject["status"], "reject")
        self.assertTrue(reject["blocking"])
        self.assertEqual(reject["reasons"], ["pose_low"])

        warn = quality.from_legacy_quality(
            {
                "quality_status": "warn",
                "quality": {"warnings": ["posture"]},
            }
        )
        self.assertEqual(warn["status"], "warn")
        self.assertFalse(warn["blocking"])
        self.assertEqual(warn["reasons"], ["posture"])

    def test_pass_and_warn_preserve_structure(self) -> None:
        for status in ("pass", "warn"):
            gate = quality.make_quality_gate(
                status, reasons=["diagnostic"], source="pose"
            )
            gated = quality.apply_quality_gates(make_analysis(), [gate])
            self.assertEqual(gated["status"], "ok")
            self.assertEqual(len(gated["sets"]), 1)
            self.assertEqual(len(gated["diagnostics"]["selected_top_events"]), 2)
            self.assertEqual(gated["quality_gate"]["status"], status)

    def test_reject_and_unknown_clear_clip_outputs(self) -> None:
        for status, expected in (
            ("reject", "rejected_quality"),
            ("unknown", "not_evaluable_quality"),
        ):
            gate = quality.make_quality_gate(
                status, reasons=["bad"], source="pose", scope="clip"
            )
            gated = quality.apply_quality_gates(make_analysis(), [gate])
            self.assertEqual(gated["status"], expected)
            self.assertEqual(gated["sets"], [])
            self.assertEqual(gated["rejected_candidates"], [])
            self.assertEqual(gated["diagnostics"]["selected_top_events"], [])
            self.assertEqual(
                gated["diagnostics"]["pre_gate_structure"]["rep_count"], 2
            )

    def test_rep_scope_removes_only_target_rep(self) -> None:
        gate = quality.make_quality_gate(
            "reject",
            reasons=["rep_artifact"],
            source="signal",
            scope="rep",
            target_id=1,
        )
        gated = quality.apply_quality_gates(make_analysis(), [gate])
        self.assertEqual(gated["status"], "ok")
        self.assertEqual(
            [rep["rep_id"] for rep in gated["sets"][0]["reps"]], [2]
        )
        self.assertEqual(
            gated["diagnostics"]["quality_gate_removed"]["rep_ids"], [1]
        )

    def test_missing_legacy_quality_fails_closed(self) -> None:
        gate = quality.from_legacy_quality(None)
        self.assertEqual(gate["status"], "unknown")
        self.assertTrue(gate["blocking"])
        gated = quality.apply_quality_gates(make_analysis(), [gate])
        self.assertEqual(gated["status"], "not_evaluable_quality")
        self.assertEqual(gated["sets"], [])


if __name__ == "__main__":
    unittest.main()
