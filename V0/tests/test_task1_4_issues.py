from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_issue_cards


class RuleAndConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = json.loads(
            (ROOT / "configs" / "task1_4_feedback_templates.json").read_text()
        )
        self.confidence = self.templates["confidence"]
        self.thresholds = self.templates["thresholds"]

    def test_high_bad_threshold_boundaries(self) -> None:
        cfg = self.thresholds["knee_asymmetry_bottom"]
        cases = (
            (4.99, "ok", "none"),
            (5.0, "ok", "low"),
            (8.0, "ok", "medium"),
            (15.0, "ok", "high"),
        )
        for value, expected_status, expected_severity in cases:
            severity, status, _ = build_issue_cards.severity_from_value(
                value, 0.90, cfg, self.confidence
            )
            self.assertEqual(status, expected_status)
            self.assertEqual(severity, expected_severity)

    def test_low_bad_threshold_and_confidence_suppression(self) -> None:
        cfg = self.thresholds["knee_flexion_rom"]
        severity, status, _ = build_issue_cards.severity_from_value(
            35.0, 0.90, cfg, self.confidence
        )
        self.assertEqual((severity, status), ("high", "ok"))
        severity, status, _ = build_issue_cards.severity_from_value(
            35.0, 0.30, cfg, self.confidence
        )
        self.assertEqual((severity, status), ("none", "unusable_low_confidence"))
        severity, status, _ = build_issue_cards.severity_from_value(
            35.0, 0.50, cfg, self.confidence
        )
        self.assertEqual((severity, status), ("low", "low_confidence"))

    def test_adjacent_rep_grouping(self) -> None:
        self.assertEqual(
            build_issue_cards.group_consecutive([1, 2, 3, 5, 7, 8]),
            [[1, 2, 3], [5], [7, 8]],
        )

    def test_stability_unit_is_variance(self) -> None:
        specs = json.loads((ROOT / "configs" / "task1_4_exercise_specs.json").read_text())
        for exercise in ("squat", "pushup"):
            unit = specs["exercises"][exercise]["metrics"]["spine_stability_bottom"]["unit"]
            self.assertEqual(unit, "variance")


class IssueOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = json.loads((ROOT / "configs" / "task1_4_exercise_specs.json").read_text())
        cls.templates = json.loads(
            (ROOT / "configs" / "task1_4_feedback_templates.json").read_text()
        )

    def make_data(self, rep_count: int, exercise: str = "squat") -> dict:
        reps = []
        for rep_id in range(1, rep_count + 1):
            start = (rep_id - 1) * 60
            reps.append(
                {
                    "rep_id": rep_id,
                    "start_frame": start,
                    "end_frame": start + 59,
                    "bottom_frame": start + 30,
                    "duration_s": 2.0,
                    "confidence": 0.95,
                    "start_local": start,
                    "end_local": start + 59,
                    "bottom_local": start + 30,
                }
            )
        return {
            "analysis": {
                "clip_id": f"synthetic_{exercise}",
                "exercise": exercise,
                "view": "front",
                "quality_status": "pass",
                "split": "dev",
                "trim": {"start_frame": 0, "end_frame": rep_count * 60 - 1},
                "fps": 30.0,
            },
            "trim_start": 0,
            "fps": 30.0,
            "reps": reps,
            "wide": {},
            "angle_rows": {},
            "body_frames": {},
        }

    def make_features(
        self,
        data: dict,
        overrides: dict[tuple[int, str], tuple[float, str, float]] | None = None,
    ) -> list[dict]:
        exercise = data["analysis"]["exercise"]
        metrics_cfg = self.specs["exercises"][exercise]["metrics"]
        overrides = overrides or {}
        features = []
        for rep in data["reps"]:
            for metric, cfg in metrics_cfg.items():
                if metric == "rep_duration_cv":
                    continue
                disabled = metric in {
                    "knee_flexion_rom",
                    "bottom_knee_angle",
                    "hip_flexion_rom",
                    "torso_lean_bottom",
                    "hip_depth_ratio",
                    "elbow_flexion_rom",
                    "bottom_elbow_angle",
                    "body_line_angle",
                    "hip_sag_ratio",
                }
                value, severity, confidence = overrides.get(
                    (rep["rep_id"], metric), (1.0, "none", 0.95)
                )
                features.append(
                    {
                        "clip_id": data["analysis"]["clip_id"],
                        "exercise": exercise,
                        "rep_id": rep["rep_id"],
                        "metric": metric,
                        "scope": "rep",
                        "raw_value": None if disabled else value,
                        "unit": cfg["unit"],
                        "confidence": None if disabled else confidence,
                        "source": "synthetic",
                        "view_requirement": cfg["views"],
                        "status": "disabled" if disabled else "ok",
                        "reference_value": None,
                        "deviation": None,
                        "severity_candidate": "none" if disabled else severity,
                        "anchor_frame": None if disabled else rep["bottom_frame"],
                        "anchor_time_s": None if disabled else rep["bottom_frame"] / 30.0,
                        "note": "synthetic",
                    }
                )
        features.append(
            {
                "clip_id": data["analysis"]["clip_id"],
                "exercise": exercise,
                "rep_id": None,
                "metric": "rep_duration_cv",
                "scope": "clip",
                "raw_value": 0.05,
                "unit": "ratio",
                "confidence": 0.95,
                "source": "synthetic",
                "view_requirement": "any",
                "status": "ok",
                "reference_value": None,
                "deviation": None,
                "severity_candidate": "none",
                "anchor_frame": None,
                "anchor_time_s": None,
                "note": "synthetic",
            }
        )
        return features

    def test_disabled_metrics_do_not_become_active_issues(self) -> None:
        data = self.make_data(4)
        features = self.make_features(data)
        output = build_issue_cards.build_issue_output(
            data, features, self.specs, self.templates
        )
        self.assertEqual(output["summary"]["active_issue_count"], 0)
        suppressed = {
            issue["issue_code"]: issue
            for issue in output["issues"]
            if issue["suppressed"]
        }
        self.assertEqual(suppressed["squat_depth_insufficient"]["suppression_reason"], "view_unsupported")
        self.assertEqual(suppressed["squat_torso_lean_excess"]["suppression_reason"], "view_unsupported")
        self.assertFalse(any(issue["show_by_default"] for issue in output["issues"]))

    def test_adjacent_reps_merge_and_evidence_contains_anchor(self) -> None:
        data = self.make_data(4)
        features = self.make_features(
            data,
            {
                (1, "knee_valgus_bottom"): (35.0, "high", 0.90),
                (2, "knee_valgus_bottom"): (36.0, "high", 0.92),
            },
        )
        output = build_issue_cards.build_issue_output(
            data, features, self.specs, self.templates
        )
        valgus = [issue for issue in output["issues"] if issue["issue_code"] == "squat_knee_valgus"]
        self.assertEqual(len(valgus), 1)
        issue = valgus[0]
        self.assertEqual(issue["rep_ids"], [1, 2])
        self.assertEqual(issue["representative_rep_id"], 2)
        self.assertEqual(issue["frequency"], "2/4")
        self.assertIsNotNone(issue["anchor_frame"])
        self.assertLessEqual(issue["evidence_interval"][0], issue["anchor_frame"])
        self.assertGreaterEqual(issue["evidence_interval"][1], issue["anchor_frame"])
        self.assertTrue(issue["show_by_default"])

    def test_fewer_than_three_reps_suppresses_tempo(self) -> None:
        data = self.make_data(2)
        features = self.make_features(data)
        features[-1]["status"] = "insufficient_reps"
        features[-1]["raw_value"] = None
        output = build_issue_cards.build_issue_output(
            data, features, self.specs, self.templates
        )
        tempo_eval = next(
            item
            for item in output["evaluations"]
            if item["issue_code"] == "squat_tempo_inconsistent"
        )
        self.assertEqual(tempo_eval["status"], "suppressed")
        self.assertEqual(tempo_eval["reason"], "insufficient_reps")
        self.assertFalse(
            any(
                issue["issue_code"] == "squat_tempo_inconsistent" and not issue["suppressed"]
                for issue in output["issues"]
            )
        )

    def test_all_active_medium_high_issues_have_valid_evidence(self) -> None:
        data = self.make_data(5)
        overrides = {
            (1, "knee_valgus_bottom"): (35.0, "high", 0.90),
            (2, "knee_asymmetry_bottom"): (10.0, "medium", 0.80),
            (3, "spine_stability_bottom"): (0.08, "medium", 0.70),
        }
        features = self.make_features(data, overrides)
        output = build_issue_cards.build_issue_output(
            data, features, self.specs, self.templates
        )
        active = [issue for issue in output["issues"] if not issue["suppressed"]]
        self.assertGreaterEqual(len(active), 3)
        rep_by_id = {rep["rep_id"]: rep for rep in data["reps"]}
        for issue in active:
            if issue["severity"] not in {"medium", "high"}:
                continue
            self.assertIsNotNone(issue["anchor_frame"])
            self.assertIsNotNone(issue["evidence_interval"])
            start, end = issue["evidence_interval"]
            rep = rep_by_id[issue["representative_rep_id"]]
            self.assertLessEqual(start, issue["anchor_frame"])
            self.assertGreaterEqual(end, issue["anchor_frame"])
            self.assertGreaterEqual(start, rep["start_frame"])
            self.assertLessEqual(end, rep["end_frame"])


if __name__ == "__main__":
    unittest.main()
