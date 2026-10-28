from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_7_actions as actions


def make_analysis(
    *,
    phase_signal: str | None = "elbow_flexion_bilateral_mean",
    rep_count: int = 3,
    start_stride: int = 60,
) -> dict:
    reps = []
    for index in range(rep_count):
        start = 10 + index * start_stride
        reps.append(
            {
                "rep_id": index + 1,
                "start_frame": start,
                "end_frame": start + start_stride - 1,
                "status": "complete",
                "phase_signal": phase_signal or "legacy_t1_4",
                "confidence": {
                    "complete": 0.8,
                    "in_set": 0.8,
                    "combined": 0.8,
                },
            }
        )
    return {
        "clip_id": "synthetic_clip",
        "status": "ok",
        "sets": [
            {
                "set_id": 1,
                "start_frame": reps[0]["start_frame"] if reps else 0,
                "end_frame": reps[-1]["end_frame"] if reps else 0,
                "status": "complete_set",
                "chain_confidence": 0.9,
                "is_default": True,
                "reps": reps,
            }
        ],
        "rejected_candidates": [{"candidate_id": "top_0"}],
        "non_rep_intervals": [{"start_frame": 0, "end_frame": 9}],
        "diagnostics": {
            "selected_phase_signal": phase_signal,
            "periodicity": 0.8,
            "selected_top_events": [{"frame": rep["start_frame"]} for rep in reps],
        },
    }


class ActionLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = actions.load_config()

    def verify(
        self,
        analysis: dict,
        action: str,
        *,
        phase_signal: str | None = None,
    ) -> dict:
        evidence = None
        if phase_signal is not None:
            evidence = {"selected_phase_signal": phase_signal}
        return actions.verify_action(
            analysis,
            action,
            fps=30.0,
            config=self.config,
            signal_evidence=evidence,
        )

    def test_squat_and_pushup_contracts_are_supported(self) -> None:
        squat = self.verify(make_analysis(), "squat", phase_signal="knee_flexion")
        self.assertEqual(squat["status"], "supported")
        self.assertTrue(squat["supported"])
        self.assertEqual(squat["evidence"]["phase_family"], "knee_flexion")
        self.assertEqual(squat["evidence"]["complete_rep_count"], 3)

        pushup = self.verify(make_analysis(), "pushup")
        self.assertEqual(pushup["status"], "supported")
        self.assertEqual(pushup["evidence"]["phase_family"], "elbow_flexion")

    def test_wrong_phase_family_is_mismatch(self) -> None:
        verification = self.verify(
            make_analysis(),
            "pushup",
            phase_signal="knee_flexion_bilateral_mean",
        )
        self.assertEqual(verification["status"], "mismatch")
        self.assertIn("phase_signal_family_mismatch", verification["reasons"])

    def test_unsupported_action_fails_closed(self) -> None:
        output = actions.apply_action_verification(
            make_analysis(),
            self.verify(make_analysis(), "lunge"),
            self.config,
        )
        self.assertEqual(output["status"], "unsupported_action")
        self.assertEqual(output["sets"], [])
        self.assertEqual(output["diagnostics"]["selected_top_events"], [])
        self.assertEqual(
            output["diagnostics"]["action_verification_disposition"], "fail_closed"
        )

    def test_insufficient_complete_reps_is_ambiguous(self) -> None:
        analysis = make_analysis(rep_count=2)
        verification = self.verify(analysis, "pushup")
        self.assertEqual(verification["status"], "ambiguous")
        self.assertIn("insufficient_complete_reps", verification["reasons"])
        output = actions.apply_action_verification(analysis, verification, self.config)
        self.assertEqual(output["status"], "no_confident_set")
        self.assertEqual(output["sets"], [])
        self.assertEqual(output["diagnostics"]["pre_action_structure"]["rep_count"], 2)

    def test_period_outside_contract_is_mismatch(self) -> None:
        analysis = make_analysis(start_stride=10)
        verification = self.verify(analysis, "pushup")
        self.assertEqual(verification["status"], "mismatch")
        self.assertIn("period_out_of_contract", verification["reasons"])

    def test_quality_and_phase_statuses_take_precedence(self) -> None:
        for status in (
            "rejected_quality",
            "not_evaluable_quality",
            "phase_ambiguous",
        ):
            analysis = make_analysis()
            analysis["status"] = status
            output = actions.apply_action_verification(
                analysis,
                self.verify(make_analysis(), "lunge"),
                self.config,
            )
            self.assertEqual(output["status"], status)
            self.assertEqual(
                output["diagnostics"]["action_verification_disposition"],
                "status_precedence",
            )

    def test_phase_signal_family_normalization(self) -> None:
        self.assertEqual(actions.signal_family("left_knee_flexion"), "knee_flexion")
        self.assertEqual(
            actions.signal_family("right_elbow_flexion_bilateral_mean"),
            "elbow_flexion",
        )

    def test_config_is_dataset_independent_and_minimal(self) -> None:
        core = self.config["core"]
        self.assertTrue(core["dataset_independent"])
        self.assertEqual(set(core["supported_actions"]), {"squat", "pushup"})
        self.assertFalse(core["full_rom_and_posture_checks"])
        self.assertTrue(core["fail_closed_on_mismatch"])

    def test_verification_does_not_mutate_input(self) -> None:
        analysis = make_analysis()
        before = copy.deepcopy(analysis)
        self.verify(analysis, "unsupported")
        self.assertEqual(analysis, before)


if __name__ == "__main__":
    unittest.main()
