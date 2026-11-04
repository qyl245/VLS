from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adapt_task1_6_sources as adapter


class SourceAdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources, cls.protocol = adapter.build_sources(
            fit3d_manifest_path=adapter.DEFAULT_FIT3D_MANIFEST,
            qevd_manifest_path=adapter.DEFAULT_QEVD_MANIFEST,
            first_round_annotations_path=adapter.DEFAULT_FIRST_ROUND_ANNOTATIONS,
            blind_annotations_path=adapter.DEFAULT_BLIND_ANNOTATIONS,
            feedback_path=adapter.DEFAULT_FEEDBACK,
            baseline_summary_path=adapter.DEFAULT_BASELINE_SUMMARY,
            check_files=False,
        )

    def test_validation_counts_and_splits(self) -> None:
        validation = self.sources["validation"]
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertEqual(validation["counts"]["total_clips"], 113)
        self.assertEqual(validation["counts"]["fit3d"], adapter.FIT3D_EXPECTED)
        self.assertEqual(validation["counts"]["qeud"]["dev"], 13)
        self.assertEqual(validation["counts"]["qeud"]["holdout"], 14)
        self.assertEqual(validation["counts"]["qeud"]["reserve"], 10)
        self.assertEqual(validation["counts"]["qeud"]["sealed_holdout"], 6)

    def test_fit3d_gt_is_top_event_only(self) -> None:
        fit3d = [
            clip
            for clip in self.sources["clips"]
            if clip["source_id"] == "fit3d_train"
        ]
        train = [
            clip for clip in fit3d if clip["has_gt"] and not clip["smoke_only"]
        ]
        self.assertEqual(len(train), 64)
        self.assertTrue(all(clip["gt"]["semantics"] == "top_event" for clip in train))
        self.assertTrue(
            all(clip["gt"]["bottom_status"] == "not_computable_no_bottom_gt" for clip in train)
        )
        self.assertTrue(
            all(not clip["has_gt"] for clip in fit3d if clip["smoke_only"])
        )
        self.assertEqual(
            sum(len(clip["gt"]["events"]) for clip in train) // 4,
            96,
        )

    def test_qevd_legacy_annotations_are_rep_intervals(self) -> None:
        sets = {
            item["annotation_set_id"]: item
            for item in self.sources["legacy_annotation_sets"]
        }
        self.assertEqual(sets["t1_3_first_round"]["accepted_count"], 47)
        self.assertEqual(len(sets["t1_3_first_round"]["segments"]), 7)
        self.assertEqual(sets["t1_3_blind"]["accepted_count"], 66)
        self.assertEqual(len(sets["t1_3_blind"]["segments"]), 5)
        for annotation_set in sets.values():
            self.assertEqual(annotation_set["gt_semantics"], "rep_interval")
            for interval in annotation_set["intervals"]:
                self.assertLessEqual(interval["start_frame"], interval["end_frame"])
                self.assertEqual(interval["frame_unit"], "source_video_absolute")

    def test_sealed_holdout_composition(self) -> None:
        sealed = [
            clip
            for clip in self.sources["clips"]
            if clip["split_role"] == "sealed_holdout"
        ]
        actions = [clip["action"] for clip in sealed]
        self.assertEqual(len(sealed), 6)
        self.assertEqual(actions.count("pushup"), 2)
        self.assertEqual(actions.count("squat"), 2)
        self.assertGreaterEqual(
            len({action for action in actions if action not in {"pushup", "squat"}}),
            2,
        )
        self.assertTrue(
            all(clip["gt"]["status"] == "sealed_unannotated" for clip in sealed)
        )
        sealed_videos = {clip["video_id"] for clip in sealed}
        observed_videos = {
            clip["video_id"]
            for clip in self.sources["clips"]
            if clip["source_id"] == "qeud_fit_coach"
            and clip["split_role"] in {"dev", "holdout"}
        }
        self.assertFalse(sealed_videos & observed_videos)

    def test_baseline_reproduction_matches_frozen_t1_5d(self) -> None:
        baseline = self.sources["baseline_reproduction"]
        self.assertTrue(baseline["passed"], baseline["errors"])
        self.assertEqual(baseline["observed"], adapter.T1_5D_BASELINE)

    def test_adapter_does_not_mutate_source_manifests(self) -> None:
        fit3d_before = json.loads(adapter.DEFAULT_FIT3D_MANIFEST.read_text())
        qevd_before = json.loads(adapter.DEFAULT_QEVD_MANIFEST.read_text())
        fit3d_copy = copy.deepcopy(fit3d_before)
        qevd_copy = copy.deepcopy(qevd_before)
        for clip in fit3d_copy["clips"]:
            adapter.fit3d_clip_record(clip, "synthetic-sha")
        for clip in qevd_copy["clips"]:
            adapter.qevd_clip_record(clip, "synthetic-sha", [])
        self.assertEqual(fit3d_before, fit3d_copy)
        self.assertEqual(qevd_before, qevd_copy)
        self.assertFalse(self.sources["adapter_contract"]["changes_algorithm_output"])
        self.assertFalse(self.sources["adapter_contract"]["invokes_segmenter"])

    def test_protocol_freezes_cross_domain_gate(self) -> None:
        protocol = self.protocol
        self.assertFalse(protocol["sealed_holdout"]["cross_domain_acceptance_complete"])
        self.assertEqual(
            protocol["sealed_holdout"]["annotation_status"], "pending_human_labels"
        )
        self.assertFalse(
            protocol["generalization_claims"]["cross_domain_acceptance_complete"]
        )
        core_values = {
            key: value
            for key, value in protocol["algorithm_core"].items()
            if key != "forbidden_core_constants"
        }
        serialized = json.dumps(core_values, ensure_ascii=False)
        for forbidden in ("Fit3D", "QEVD", "50591643", "camera_median", "6_reps"):
            self.assertNotIn(forbidden, serialized)


class BaselineValidationTests(unittest.TestCase):
    def _summary(self, manifest_sha: str, count_mae: float = 0.625) -> dict:
        return {
            "schema_version": "task1-5d-rep-eval-v1",
            "source": {"manifest_sha256": manifest_sha},
            "dataset": {
                "train_clips": 64,
                "action_instances": 16,
                "unique_gt_reps": 96,
            },
            "results": {
                "consensus_count_primary": {
                    "count_mae": count_mae,
                    "exact_count_correct": 6,
                },
                "events_primary_camera": {
                    "event_abs_error_median_frames": 12.5,
                    "event_abs_error_p90_frames": 25.8,
                    "high_confidence_fp_count": 24,
                },
                "events_all_cameras_diagnostic": {
                    "high_confidence_fp_count": 80
                },
                "interval_auxiliary": {
                    "interval_abs_error_median_frames": 3.0
                },
            },
        }

    def test_exact_baseline_values_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            path.write_text(json.dumps(self._summary("manifest-sha")))
            result = adapter.validate_t1_5d_baseline(path, "manifest-sha")
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["observed"], adapter.T1_5D_BASELINE)

    def test_baseline_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            path.write_text(json.dumps(self._summary("manifest-sha", count_mae=0.75)))
            result = adapter.validate_t1_5d_baseline(path, "manifest-sha")
        self.assertFalse(result["passed"])
        self.assertTrue(any("consensus_count_mae" in error for error in result["errors"]))


class HoldoutArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources, _ = adapter.build_sources(
            fit3d_manifest_path=adapter.DEFAULT_FIT3D_MANIFEST,
            qevd_manifest_path=adapter.DEFAULT_QEVD_MANIFEST,
            first_round_annotations_path=adapter.DEFAULT_FIRST_ROUND_ANNOTATIONS,
            blind_annotations_path=adapter.DEFAULT_BLIND_ANNOTATIONS,
            feedback_path=adapter.DEFAULT_FEEDBACK,
            baseline_summary_path=adapter.DEFAULT_BASELINE_SUMMARY,
            check_files=False,
        )

    def test_holdout_artifacts_are_pending_not_fake_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = adapter.write_holdout_artifacts(self.sources, output_dir)
            self.assertEqual(result["status"], "pending_human_annotation")
            self.assertEqual(result["segments"], 6)
            self.assertTrue(Path(result["contact_sheet_path"]).is_file())
            with Path(result["segments_path"]).open(newline="") as handle:
                segments = list(csv.DictReader(handle))
            with Path(result["template_path"]).open(newline="") as handle:
                template = list(csv.DictReader(handle))
            self.assertEqual(len(segments), 6)
            self.assertEqual(len(template), 6)
            self.assertTrue(all(row["review_status"] == "pending" for row in template))
            self.assertTrue(all(row["gt_start_frame"] == "" for row in template))
            self.assertTrue(all(row["gt_end_frame"] == "" for row in template))


class SchemaContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (ROOT / "schemas" / "task1_6_clip_analysis_v2.schema.json").read_text()
        )

    def test_schema_freezes_state_and_confidence_contract(self) -> None:
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        definitions = self.schema["$defs"]
        self.assertEqual(
            set(definitions["phase_state"]["required"]),
            {
                "start_frame",
                "end_frame",
                "anchor_frame",
                "stable_fraction",
                "boundary_slope",
            },
        )
        self.assertEqual(
            set(definitions["confidence_layers"]["required"]),
            {"complete", "in_set", "combined"},
        )
        self.assertEqual(
            definitions["rep_candidate"]["properties"]["status"]["enum"],
            ["complete", "endpoint_partial", "incomplete"],
        )

    def test_root_requires_sets_and_rejection_fields(self) -> None:
        required = set(self.schema["required"])
        self.assertIn("sets", required)
        self.assertIn("rejected_candidates", required)
        self.assertIn("non_rep_intervals", required)
        self.assertEqual(
            self.schema["properties"]["schema_version"]["const"],
            "task1-6-clip-analysis-v2",
        )


if __name__ == "__main__":
    unittest.main()
