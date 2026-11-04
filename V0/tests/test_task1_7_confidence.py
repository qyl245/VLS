from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_7_confidence as confidence


def make_analysis(*, rep_count: int = 3, rep_status: str = "complete") -> dict:
    reps = []
    for index in range(rep_count):
        start = 10 + index * 40
        reps.append(
            {
                "rep_id": index + 1,
                "start_frame": start,
                "end_frame": start + 39,
                "status": rep_status,
                "confidence": {
                    "complete": 0.8,
                    "in_set": 0.8,
                    "combined": 0.8,
                },
                "quality_flags": [],
            }
        )
    return {
        "clip_id": "synthetic_clip",
        "status": "ok",
        "sets": [
            {
                "set_id": 1,
                "start_frame": 10,
                "end_frame": 129,
                "status": "complete_set",
                "chain_confidence": 0.9,
                "is_default": True,
                "reps": reps,
            }
        ],
        "rejected_candidates": [],
        "non_rep_intervals": [],
        "diagnostics": {
            "selected_phase_signal": "elbow_flexion_bilateral_mean",
            "periodicity": 0.8,
        },
    }


def evidence(
    *,
    clip_status: str = "pass",
    set_status: str = "pass",
    rep_statuses: dict[int, str] | None = None,
    rep_count: int = 3,
) -> list[dict]:
    rep_statuses = rep_statuses or {}
    output = [
        {
            "evidence_id": "clip:quality_gate",
            "status": clip_status,
            "scope": "clip",
            "source": "pose",
            "reasons": [],
            "target_id": None,
            "start_frame": 0,
            "end_frame": 129,
        },
        {
            "evidence_id": "set:1",
            "status": set_status,
            "scope": "set",
            "source": "chain",
            "reasons": [],
            "target_id": 1,
            "start_frame": 10,
            "end_frame": 129,
        },
    ]
    for rep_id in range(1, rep_count + 1):
        output.append(
            {
                "evidence_id": f"rep:1:{rep_id}",
                "status": rep_statuses.get(rep_id, "pass"),
                "scope": "rep",
                "source": "chain",
                "reasons": [],
                "target_id": rep_id,
                "start_frame": 10 + (rep_id - 1) * 40,
                "end_frame": 49 + (rep_id - 1) * 40,
            }
        )
    return output


def supported_action() -> dict:
    return {"status": "supported", "confidence": 0.9}


def signal_evidence(*, ambiguous: bool = False, score: float = 0.8) -> dict:
    return {"consensus_score": score, "ambiguous": ambiguous}


class ConfidenceLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = confidence.load_config()
        cls.calibrator = confidence.load_calibrator(cls.config)

    def apply(
        self,
        analysis: dict,
        quality_evidence: list[dict],
        *,
        action: dict | None = None,
        signal: dict | None = None,
    ) -> dict:
        return confidence.apply_confidence_lite(
            analysis,
            quality_evidence=quality_evidence,
            action_verification=action or supported_action(),
            signal_evidence=signal or signal_evidence(),
            config=self.config,
            calibrator=self.calibrator,
        )

    def test_clean_input_is_reportable_and_does_not_mutate(self) -> None:
        analysis = make_analysis()
        before = copy.deepcopy(analysis)
        output = self.apply(analysis, evidence())
        self.assertFalse(output["confidence"]["abstained"])
        self.assertIsNone(output["confidence"]["abstain_reason"])
        self.assertEqual(analysis, before)
        self.assertEqual(len(output["sets"][0]["reps"]), 3)
        self.assertEqual(output["sets"][0]["reps"][0]["confidence"]["combined"], 0.8)

    def test_rep_warn_lowers_output_and_only_affected_rep(self) -> None:
        clean = self.apply(make_analysis(), evidence())
        warned = self.apply(
            make_analysis(),
            evidence(rep_statuses={2: "warn"}),
        )
        self.assertLess(
            warned["confidence"]["calibrated"]["combined"],
            clean["confidence"]["calibrated"]["combined"],
        )
        self.assertEqual(warned["confidence"]["components"]["quality"], 0.75)
        reps = warned["sets"][0]["reps"]
        self.assertEqual(reps[0]["confidence"]["combined"], 0.8)
        self.assertEqual(reps[1]["confidence"]["combined"], 0.6)
        self.assertEqual(reps[2]["confidence"]["combined"], 0.8)
        self.assertIn(
            "c_lite_confidence_downweighted", reps[1]["quality_flags"]
        )
        self.assertNotIn(
            "c_lite_confidence_downweighted", reps[0]["quality_flags"]
        )

    def test_set_warn_lowers_set_chain_and_all_reps(self) -> None:
        output = self.apply(make_analysis(), evidence(set_status="warn"))
        set_candidate = output["sets"][0]
        self.assertEqual(set_candidate["chain_confidence"], 0.675)
        self.assertEqual(
            [rep["confidence"]["combined"] for rep in set_candidate["reps"]],
            [0.6, 0.6, 0.6],
        )
        self.assertIn(
            "c_lite_set_confidence_downweighted",
            set_candidate["quality_flags"],
        )

    def test_clip_warn_propagates_to_set_and_reps(self) -> None:
        output = self.apply(make_analysis(), evidence(clip_status="warn"))
        self.assertEqual(output["sets"][0]["chain_confidence"], 0.675)
        self.assertTrue(
            all(
                rep["confidence"]["combined"] == 0.6
                for rep in output["sets"][0]["reps"]
            )
        )

    def test_reject_abstains_and_zeroes_confidence(self) -> None:
        output = self.apply(make_analysis(), evidence(clip_status="reject"))
        self.assertTrue(output["confidence"]["abstained"])
        self.assertEqual(output["confidence"]["abstain_reason"], "quality_reject")
        self.assertLess(
            output["confidence"]["calibrated"]["combined"],
            self.config["core"]["minimum_reportable_confidence"],
        )
        self.assertEqual(output["sets"][0]["chain_confidence"], 0.0)

    def test_action_and_phase_gates_abstain(self) -> None:
        action = self.apply(
            make_analysis(),
            evidence(),
            action={"status": "mismatch", "confidence": 0.1},
        )
        self.assertTrue(action["confidence"]["abstained"])
        self.assertEqual(action["confidence"]["abstain_reason"], "action_mismatch")
        self.assertLess(
            action["confidence"]["calibrated"]["combined"],
            self.config["core"]["minimum_reportable_confidence"],
        )

        phase = self.apply(
            make_analysis(),
            evidence(),
            signal=signal_evidence(ambiguous=True),
        )
        self.assertTrue(phase["confidence"]["abstained"])
        self.assertEqual(phase["confidence"]["abstain_reason"], "phase_ambiguous")

    def test_no_complete_rep_abstains_but_preserves_partial_structure(self) -> None:
        analysis = make_analysis(rep_status="endpoint_partial")
        output = self.apply(analysis, evidence())
        self.assertTrue(output["confidence"]["abstained"])
        self.assertEqual(
            output["confidence"]["abstain_reason"], "no_confident_set"
        )
        self.assertEqual(len(output["sets"][0]["reps"]), 3)

    def test_frozen_calibrator_hash_and_provenance(self) -> None:
        self.assertEqual(
            self.config["core"]["calibrator_sha256"],
            confidence.sha256_path(ROOT / self.config["core"]["calibrator_config"]),
        )
        self.assertFalse(self.config["provenance"]["refit_on_real_video"])
        self.assertFalse(self.config["provenance"]["refit_on_qevd_sealed"])

    def test_diagnostics_report_conservative_quality_scope(self) -> None:
        output = self.apply(
            make_analysis(),
            evidence(set_status="warn", rep_statuses={3: "warn"}),
        )
        diagnostics = output["diagnostics"]["c_lite_confidence"]
        self.assertEqual(diagnostics["quality_status"], "warn")
        self.assertEqual(diagnostics["rep_context_quality_status"], "warn")
        self.assertFalse(diagnostics["abstained"])


if __name__ == "__main__":
    unittest.main()
