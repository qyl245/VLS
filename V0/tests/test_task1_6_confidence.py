from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_confidence as confidence


def make_rep(
    *,
    status: str = "complete",
    rep_id: int = 1,
    start_frame: int = 10,
    end_frame: int = 50,
    anchor_frame: int = 20,
    combined: float = 0.8,
) -> dict:
    return {
        "rep_id": rep_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "top": {
            "anchor_frame": anchor_frame,
        },
        "status": status,
        "partial_side": "start" if status == "endpoint_partial" else None,
        "confidence": {
            "complete": combined,
            "in_set": combined,
            "combined": combined,
        },
    }


def make_analysis() -> dict:
    return {
        "status": "ok",
        "sets": [],
        "rejected_candidates": [],
    }


def make_clip() -> dict:
    return {
        "clip_id": "synthetic_clip",
        "cluster_id": "synthetic_cluster",
        "action": "pushup",
    }


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_beta_lower_bound_edge_cases(self) -> None:
        self.assertEqual(confidence.beta_lower_bound(0, 0, 0.1), 0.0)
        self.assertEqual(confidence.beta_lower_bound(0, 10, 0.1), 0.0)
        all_success = confidence.beta_lower_bound(10, 10, 0.1)
        self.assertGreater(all_success, 0.0)
        self.assertLess(all_success, 1.0)
        self.assertLess(
            confidence.beta_lower_bound(8, 10, 0.1),
            confidence.beta_lower_bound(10, 10, 0.1),
        )

    def test_fit_enforces_monotonic_lower_bounds(self) -> None:
        samples = [
            *[{"score": 0.5, "target": 1} for _ in range(10)],
            {"score": 0.7, "target": 1},
        ]
        fitted = confidence.fit_binned_calibrator(samples)
        self.assertEqual(
            fitted["monotonic_lower_bounds"],
            sorted(fitted["monotonic_lower_bounds"]),
        )
        self.assertGreater(
            fitted["monotonic_lower_bounds"][3],
            fitted["lower_bounds"][3],
        )
        self.assertEqual(
            fitted["monotonic_lower_bounds"][4],
            fitted["monotonic_lower_bounds"][3],
        )

    def test_calibration_reduces_ece_and_brier(self) -> None:
        samples = [
            *[{"score": 0.3, "target": 0} for _ in range(50)],
            *[{"score": 0.7, "target": 1} for _ in range(50)],
        ]
        fitted = confidence.fit_binned_calibrator(samples)
        core = confidence.build_frozen_config(
            fitted,
            {"fit_scope": "synthetic"},
        )["core"]
        raw_rows = [
            {
                "score": sample["score"],
                "target": sample["target"],
            }
            for sample in samples
        ]
        calibrated_rows = [
            {
                "score": confidence.calibrate_score(sample["score"], core),
                "target": sample["target"],
            }
            for sample in samples
        ]
        raw = confidence.binary_calibration_metrics(
            raw_rows, score_key="score"
        )
        calibrated = confidence.binary_calibration_metrics(
            calibrated_rows, score_key="score"
        )
        self.assertLess(calibrated["ece"], raw["ece"])
        self.assertLess(calibrated["brier"], raw["brier"])

    def test_threshold_metrics(self) -> None:
        rows = [
            {"score": 0.1, "target": 0},
            {"score": 0.5, "target": 1},
            {"score": 0.9, "target": 1},
        ]
        metrics = confidence.threshold_metrics(
            rows, score_key="score", threshold=0.5
        )
        self.assertEqual(metrics["accepted_count"], 2)
        self.assertEqual(metrics["accepted_tp"], 2)
        self.assertEqual(metrics["accepted_fp"], 0)
        self.assertEqual(metrics["coverage"], 0.666667)
        self.assertEqual(metrics["true_positive_coverage"], 1.0)

    def test_empty_inputs_are_safe(self) -> None:
        fitted = confidence.fit_binned_calibrator([])
        self.assertEqual(fitted["sample_count"], 0)
        self.assertTrue(all(value == 0.0 for value in fitted["lower_bounds"]))
        self.assertEqual(
            confidence.binary_calibration_metrics([], score_key="score")["n"],
            0,
        )
        self.assertEqual(
            confidence.threshold_metrics(
                [], score_key="score", threshold=0.5
            )["accepted_count"],
            0,
        )

    def test_formal_unit_is_default_set_complete_rep_only(self) -> None:
        self.assertTrue(
            confidence.rep_unit_eligible(True, "complete")
        )
        self.assertFalse(
            confidence.rep_unit_eligible(True, "endpoint_partial")
        )
        self.assertFalse(
            confidence.rep_unit_eligible(False, "complete")
        )

    def test_base_row_excludes_partial_and_non_default_reps(self) -> None:
        clip = make_clip()
        analysis = make_analysis()
        default_set = {
            "set_id": 1,
            "is_default": True,
            "chain_confidence": 0.9,
        }
        partial_row = confidence.base_rep_row(
            dataset="synthetic",
            clip=clip,
            analysis=analysis,
            set_candidate=default_set,
            rep=make_rep(status="endpoint_partial"),
            unit_eligible=False,
            gt_available=True,
        )
        self.assertFalse(partial_row["unit_eligible"])
        self.assertFalse(partial_row["gt_evaluable"])
        self.assertEqual(
            partial_row["exclusion_reason"],
            "endpoint_partial_not_formal",
        )
        non_default_row = confidence.base_rep_row(
            dataset="synthetic",
            clip=clip,
            analysis=analysis,
            set_candidate={**default_set, "is_default": False},
            rep=make_rep(),
            unit_eligible=False,
            gt_available=True,
        )
        self.assertEqual(
            non_default_row["exclusion_reason"], "non_default_set"
        )
        self.assertEqual(
            {key: partial_row[key] for key in (
                "raw_complete",
                "raw_in_set",
                "raw_combined",
            )},
            {"raw_complete": 0.8, "raw_in_set": 0.8, "raw_combined": 0.8},
        )

    def test_rep_anchor_uses_top_anchor_or_interval_midpoint(self) -> None:
        self.assertEqual(confidence.rep_anchor_frame(make_rep(anchor_frame=37)), 37)
        no_top = make_rep(start_frame=10, end_frame=30)
        no_top["top"] = None
        self.assertEqual(confidence.rep_anchor_frame(no_top), 20)

    def test_rejection_reason_support(self) -> None:
        self.assertTrue(
            confidence.reason_supported(
                "shape_mismatch",
                eligible=False,
                event_in_accepted_set=False,
            )
        )
        self.assertFalse(
            confidence.reason_supported(
                "shape_mismatch",
                eligible=True,
                event_in_accepted_set=False,
            )
        )
        self.assertTrue(
            confidence.reason_supported(
                "no_confident_set",
                eligible=True,
                event_in_accepted_set=True,
            )
        )
        self.assertFalse(
            confidence.reason_supported(
                "unknown_reason",
                eligible=True,
                event_in_accepted_set=True,
            )
        )

    def test_frozen_config_contract_and_forbidden_tokens(self) -> None:
        fitted = confidence.fit_binned_calibrator(
            [
                *[{"score": 0.3, "target": 0} for _ in range(4)],
                *[{"score": 0.7, "target": 1} for _ in range(4)],
            ]
        )
        payload = confidence.build_frozen_config(
            fitted,
            {"fit_scope": "synthetic_cluster"},
        )
        confidence.validate_frozen_config(payload, fitted=fitted)
        core_text = str(payload["core"]).lower()
        for token in confidence.FORBIDDEN_CORE_TOKENS:
            self.assertNotIn(token, core_text)
        self.assertTrue(payload["frozen"])
        self.assertEqual(
            payload["core"]["high_confidence_threshold"], 0.5
        )

    def test_changed_frozen_lower_bound_is_rejected(self) -> None:
        fitted = confidence.fit_binned_calibrator(
            [{"score": 0.7, "target": 1} for _ in range(10)]
        )
        payload = confidence.build_frozen_config(
            fitted,
            {"fit_scope": "synthetic_cluster"},
        )
        changed = copy.deepcopy(payload)
        changed["core"]["lower_bounds"][2] = 0.2
        with self.assertRaises(ValueError):
            confidence.validate_frozen_config(changed, fitted=fitted)

    def test_apply_calibration_outputs_bounded_scores(self) -> None:
        fitted = confidence.fit_binned_calibrator(
            [
                *[{"score": 0.1, "target": 0} for _ in range(5)],
                *[{"score": 0.9, "target": 1} for _ in range(5)],
            ]
        )
        payload = confidence.build_frozen_config(
            fitted,
            {"fit_scope": "synthetic_cluster"},
        )
        rows = [
            {"raw_combined": 0.05},
            {"raw_combined": 0.55},
            {"raw_combined": 0.95},
        ]
        confidence.apply_calibration(rows, payload["core"])
        for row in rows:
            self.assertGreaterEqual(row["calibrated_combined"], 0.0)
            self.assertLessEqual(row["calibrated_combined"], 1.0)
            self.assertIn(row["calibration_bin"], range(5))
            self.assertIsInstance(row["high_confidence"], bool)


if __name__ == "__main__":
    unittest.main()
