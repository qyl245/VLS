from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_chain_selection as chain_selection
import task1_6_phase_states as phase_states


def make_state(
    state_type: str,
    midpoint: int,
    *,
    mean_angle: float,
    width: int = 30,
    kind: str = "interior",
    status: str = "valid",
    stable_fraction: float = 1.0,
    bottom_duration_fraction: float = 0.3,
) -> dict:
    start = max(0, midpoint - width // 2)
    end = midpoint + width // 2
    return {
        "state_type": state_type,
        "start_frame": start,
        "end_frame": end,
        "kind": kind,
        "status": status,
        "stable_fraction": stable_fraction,
        "mean_angle": mean_angle,
        "boundary_slope": 0.0,
        "anchors": {
            "plateau_entry": start,
            "plateau_midpoint": midpoint,
            "plateau_exit": end,
            "phase_zero_crossing": midpoint,
        },
        "anchor_frame": midpoint,
        "evidence": {"duration_fraction": bottom_duration_fraction},
        "quality_flags": [],
    }


def make_cycle_states(
    top_midpoints: list[int],
    *,
    period_frames: float = 100.0,
    top_angles: list[float] | None = None,
    bottom_angles: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    top_angles = top_angles or [160.0] * len(top_midpoints)
    bottom_midpoints = [
        int(round((left + right) / 2.0))
        for left, right in zip(top_midpoints[:-1], top_midpoints[1:])
    ]
    bottom_angles = bottom_angles or [100.0] * len(bottom_midpoints)
    tops = [
        make_state(
            "top",
            midpoint,
            mean_angle=angle,
            kind=(
                "leading_partial"
                if index == 0
                else "trailing_partial"
                if index == len(top_midpoints) - 1
                else "interior"
            ),
        )
        for index, (midpoint, angle) in enumerate(
            zip(top_midpoints, top_angles)
        )
    ]
    bottoms = [
        make_state("bottom", midpoint, mean_angle=angle)
        for midpoint, angle in zip(bottom_midpoints, bottom_angles)
    ]
    return tops, bottoms


def synthetic_wave(
    fps: float,
    duration_s: float,
    period_s: float,
    *,
    amplitude: float = 40.0,
    offset: float = 120.0,
    noise: float = 0.0,
    missing_fraction: float = 0.0,
    seed: int = 20261026,
) -> np.ndarray:
    count = int(round(fps * duration_s))
    times = np.arange(count, dtype=np.float64) / fps
    wave = np.cos(2.0 * np.pi * times / period_s)
    values = np.clip(wave, -0.55, 0.60) * amplitude + offset
    rng = np.random.default_rng(seed)
    if noise:
        values = values + rng.normal(0.0, noise, count)
    if missing_fraction:
        missing = rng.choice(
            count,
            size=int(round(count * missing_fraction)),
            replace=False,
        )
        values[missing] = np.nan
    return values


class ChainSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = chain_selection.load_chain_config(
            ROOT / "configs" / "task1_6_chain.json"
        )
        self.action = {
            "rep_unit": "cycle",
            "top_polarity": "peak",
            "bottom_polarity": "trough",
            "min_period_s": 0.70,
            "max_period_s": 5.00,
            "phase_signal_candidates": [],
            "phase_signal_pairs": [],
        }

    def analyze(
        self,
        top_midpoints: list[int],
        *,
        top_angles: list[float] | None = None,
        bottom_angles: list[float] | None = None,
        frame_count: int = 700,
        period_frames: float = 100.0,
    ) -> dict:
        tops, bottoms = make_cycle_states(
            top_midpoints,
            period_frames=period_frames,
            top_angles=top_angles,
            bottom_angles=bottom_angles,
        )
        analysis, _ = chain_selection.analyze_phase_states(
            "synthetic_clip",
            tops,
            bottoms,
            period_frames=period_frames,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=frame_count,
            config=self.config,
        )
        return analysis

    def analyze_detected(self, values: np.ndarray, fps: float) -> dict:
        states = phase_states.detect_phase_states(
            values,
            fps,
            self.action,
            "synthetic_flexion",
            "plateau_midpoint",
        )
        analysis, _ = chain_selection.analyze_phase_states(
            "synthetic_clip",
            states.get("top_states") or [],
            states.get("bottom_states") or [],
            period_frames=float(states.get("period_frames") or 0.0),
            periodicity=float(states.get("periodicity") or 0.0),
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=len(values),
            config=self.config,
        )
        return analysis

    def assert_contract_fields(self, analysis: dict) -> None:
        for field in (
            "schema_version",
            "clip_id",
            "algorithm",
            "anchor_policy",
            "status",
            "sets",
            "rejected_candidates",
            "non_rep_intervals",
        ):
            self.assertIn(field, analysis)
        self.assertIn(
            analysis["status"],
            {"ok", "partial", "no_confident_set", "rejected_quality"},
        )
        for set_candidate in analysis["sets"]:
            for field in (
                "set_id",
                "start_frame",
                "end_frame",
                "status",
                "chain_confidence",
                "is_default",
                "reps",
                "rejected_candidates",
                "non_rep_intervals",
            ):
                self.assertIn(field, set_candidate)
            self.assertIn(
                set_candidate["status"], {"complete_set", "partial_set"}
            )
            self.assertGreaterEqual(set_candidate["chain_confidence"], 0.0)
            self.assertLessEqual(set_candidate["chain_confidence"], 1.0)
            for rep in set_candidate["reps"]:
                for field in (
                    "rep_id",
                    "start_frame",
                    "end_frame",
                    "top",
                    "bottom",
                    "phase_signal",
                    "rep_unit",
                    "status",
                    "confidence",
                    "quality_flags",
                ):
                    self.assertIn(field, rep)
                self.assertEqual(rep["rep_unit"], "cycle")
                self.assertIn(
                    rep["status"],
                    {"complete", "endpoint_partial", "incomplete"},
                )
                for layer in ("complete", "in_set", "combined"):
                    self.assertGreaterEqual(rep["confidence"][layer], 0.0)
                    self.assertLessEqual(rep["confidence"][layer], 1.0)

    def test_six_top_states_form_one_five_cycle_set(self) -> None:
        analysis = self.analyze([100, 200, 300, 400, 500, 600])
        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 5)
        self.assertTrue(analysis["sets"][0]["is_default"])
        self.assertEqual(len(analysis["diagnostics"]["selected_top_events"]), 6)
        self.assert_contract_fields(analysis)

    def test_terminal_event_anchor_is_applied_per_chain(self) -> None:
        tops, bottoms = make_cycle_states(
            [100, 200, 300, 400, 500, 600, 1000, 1100, 1200, 1300]
        )
        tops[5]["anchors"]["phase_zero_crossing"] = 587
        tops[9]["anchors"]["phase_zero_crossing"] = 1287
        analysis, _ = chain_selection.analyze_phase_states(
            "synthetic_clip",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=1400,
            config=self.config,
        )
        self.assertEqual(len(analysis["sets"]), 2)
        events = analysis["diagnostics"]["selected_top_events"]
        self.assertEqual(
            [event["frame"] for event in events],
            [100, 200, 300, 400, 500, 587, 1000, 1100, 1200, 1287],
        )
        self.assertEqual(
            [event["graph_frame"] for event in events],
            [100, 200, 300, 400, 500, 600, 1000, 1100, 1200, 1300],
        )
        self.assertEqual(
            [event["is_terminal"] for event in events],
            [False, False, False, False, False, True, False, False, False, True],
        )
        self.assertEqual(
            [len(item["reps"]) for item in analysis["sets"]], [5, 3]
        )

    def test_terminal_anchor_never_moves_later_than_midpoint(self) -> None:
        tops, bottoms = make_cycle_states([100, 200, 300, 400])
        tops[-1]["anchors"]["phase_zero_crossing"] = 410
        analysis, _ = chain_selection.analyze_phase_states(
            "synthetic_clip",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=500,
            config=self.config,
        )
        event = analysis["diagnostics"]["selected_top_events"][-1]
        self.assertEqual(event["frame"], 400)
        self.assertEqual(event["graph_frame"], 400)

    def test_weak_end_candidates_are_rejected_as_shape_mismatch(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500, 600],
            top_angles=[105.0, 160.0, 160.0, 160.0, 160.0, 105.0],
            bottom_angles=[100.0, 100.0, 100.0, 100.0, 155.0],
        )
        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        rejected = {
            item["candidate_id"]: item["reasons"]
            for item in analysis["rejected_candidates"]
        }
        self.assertEqual(rejected["top_state_1"], ["shape_mismatch"])
        self.assertEqual(rejected["top_state_6"], ["shape_mismatch"])

    def test_two_or_three_top_states_do_not_meet_minimum_rep_chain(self) -> None:
        for tops in ([100, 200], [100, 200, 300]):
            with self.subTest(tops=tops):
                analysis = self.analyze(tops)
                self.assertEqual(analysis["status"], "no_confident_set")
                self.assertEqual(analysis["sets"], [])
                self.assertTrue(analysis["rejected_candidates"])
                self.assertTrue(
                    all(
                        item["reasons"] == ["no_confident_set"]
                        for item in analysis["rejected_candidates"]
                    )
                )

    def test_missing_bottom_never_becomes_complete_rep(self) -> None:
        tops, _ = make_cycle_states([100, 200, 300, 400, 500, 600])
        analysis, _ = chain_selection.analyze_phase_states(
            "missing_bottom",
            tops,
            [],
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=700,
            config=self.config,
        )
        self.assertEqual(analysis["status"], "no_confident_set")
        self.assertFalse(
            any(
                rep["status"] == "complete"
                for set_candidate in analysis["sets"]
                for rep in set_candidate["reps"]
            )
        )

    def test_period_break_separates_chains_without_merging(self) -> None:
        analysis = self.analyze([100, 200, 300, 400, 800, 900])
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        rejected = {
            item["candidate_id"]: item["reasons"]
            for item in analysis["rejected_candidates"]
        }
        self.assertEqual(rejected["top_state_5"], ["no_confident_set"])
        self.assertEqual(rejected["top_state_6"], ["no_confident_set"])

    def test_amplitude_and_bottom_mismatch_can_reject_edge(self) -> None:
        left = {
            "index": 0,
            "event_frame": 100,
            "amplitude": 60.0,
            "bottom_duration_fraction": 0.3,
        }
        right = {
            "index": 1,
            "event_frame": 245,
            "amplitude": 5.0,
            "bottom_duration_fraction": 0.02,
        }
        edge = chain_selection.build_edge(
            left, right, 100.0, self.config
        )
        self.assertIsNotNone(edge)
        self.assertLess(edge["amplitude_score"], 0.01)
        self.assertFalse(edge["allowed"])

    def test_partial_endpoint_closes_consensus_backbone_chain(self) -> None:
        tops, bottoms = make_cycle_states([100, 200, 300, 400])
        for top in tops[1:]:
            top["evidence"].update(
                scale_support=4,
                scale_support_fraction=1.0,
                member_count=4,
                multi_scale_consensus=True,
            )
        tops[0]["evidence"].update(
            scale_support=1,
            scale_support_fraction=0.25,
            member_count=1,
            multi_scale_consensus=False,
        )
        analysis, _ = chain_selection.analyze_phase_states(
            "endpoint_closure",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=500,
            config=self.config,
            chain_policy={
                "minimum_chain_consensus_events": 2,
                "endpoint_closure_gap_period_range": [0.65, 1.35],
            },
        )
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        self.assertEqual(analysis["sets"][0]["status"], "partial_set")
        self.assertGreaterEqual(
            analysis["sets"][0]["endpoint_closure_count"], 1
        )
        self.assertTrue(analysis["diagnostics"]["endpoint_closure_edges"])

    def test_no_chain_policy_preserves_t1_6_endpoint_behavior(self) -> None:
        tops, bottoms = make_cycle_states([100, 200, 300, 400])
        analysis, _ = chain_selection.analyze_phase_states(
            "t1_6_behavior",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=500,
            config=self.config,
        )
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(analysis["sets"][0]["status"], "complete_set")
        self.assertEqual(
            analysis["diagnostics"]["endpoint_closure_edges"], []
        )

    def test_edge_only_chain_cannot_form_a_set(self) -> None:
        tops, bottoms = make_cycle_states([100, 200, 300, 400])
        for top in tops:
            top["evidence"].update(
                scale_support=1,
                scale_support_fraction=0.25,
                member_count=1,
                multi_scale_consensus=False,
            )
        analysis, _ = chain_selection.analyze_phase_states(
            "edge_only",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=500,
            config=self.config,
            chain_policy={
                "minimum_chain_consensus_events": 2,
                "endpoint_closure_gap_period_range": [0.65, 1.35],
            },
        )
        self.assertEqual(analysis["status"], "no_confident_set")
        self.assertEqual(analysis["sets"], [])

    def test_edge_singleton_closure_uses_tighter_gap_range(self) -> None:
        def node(index: int, frame: int, *, partial: bool) -> dict:
            return {
                "index": index,
                "event_frame": frame,
                "amplitude": 50.0,
                "bottom_duration_fraction": 0.3,
                "partial_endpoint": partial,
                "multi_scale_consensus": not partial,
                "eligible": True,
            }

        policy = {
            "minimum_chain_consensus_events": 2,
            "endpoint_closure_gap_period_range": [0.65, 1.35],
        }
        allowed = chain_selection.build_edges(
            [node(0, 100, partial=True), node(1, 220, partial=False)],
            100.0,
            self.config,
            chain_policy=policy,
        )[(0, 1)]
        self.assertTrue(allowed["endpoint_closure"])
        self.assertTrue(allowed["allowed"])
        rejected = chain_selection.build_edges(
            [node(0, 100, partial=True), node(1, 250, partial=False)],
            100.0,
            self.config,
            chain_policy=policy,
        )[(0, 1)]
        self.assertTrue(rejected["endpoint_closure_gap_rejected"])
        self.assertFalse(rejected["allowed"])

    def test_endpoint_arbitration_restores_low_rom_leading_cycle(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500],
            top_angles=[120.0, 160.0, 160.0, 160.0, 160.0],
        )

        reps = analysis["sets"][0]["reps"]
        self.assertEqual(len(reps), 4)
        self.assertEqual(reps[0]["status"], "endpoint_partial")
        self.assertIn("low_rom", reps[0]["quality_flags"])
        first_event = analysis["diagnostics"]["selected_top_events"][0]
        self.assertTrue(first_event["endpoint_arbitrated"])
        self.assertTrue(first_event["low_rom"])
        decision = analysis["diagnostics"]["endpoint_arbitration_decisions"][0]
        self.assertEqual(decision["decision"], "add_rep")
        self.assertEqual(decision["role"], "leading")
        self.assertGreater(
            decision["hypotheses"]["rep"],
            decision["hypotheses"]["context"],
        )

    def test_endpoint_arbitration_restores_low_rom_trailing_cycle(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0],
        )

        reps = analysis["sets"][0]["reps"]
        self.assertEqual(len(reps), 4)
        self.assertEqual(reps[-1]["status"], "endpoint_partial")
        decision = analysis["diagnostics"]["endpoint_arbitration_decisions"][0]
        self.assertEqual(decision["decision"], "add_rep")
        self.assertEqual(decision["role"], "trailing")

    def test_endpoint_arbitration_keeps_context_below_amplitude_floor(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500],
            top_angles=[101.0, 160.0, 160.0, 160.0, 160.0],
        )

        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        decision = analysis["diagnostics"]["endpoint_arbitration_decisions"][0]
        self.assertEqual(decision["decision"], "keep_context")
        self.assertEqual(decision["reason"], "below_endpoint_amplitude")

    def test_endpoint_arbitration_rejects_outlier_gap(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 600],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0],
        )

        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        decision = analysis["diagnostics"]["endpoint_arbitration_decisions"][0]
        self.assertEqual(decision["decision"], "keep_context")
        self.assertEqual(decision["reason"], "endpoint_gap_or_edge_rejected")

    def test_endpoint_arbitration_cannot_chain_beyond_restored_node(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500, 600],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0, 160.0],
        )

        self.assertEqual(len(analysis["sets"][0]["reps"]), 4)
        frames = [
            event["frame"]
            for event in analysis["diagnostics"]["selected_top_events"]
        ]
        self.assertIn(500, frames)
        self.assertNotIn(600, frames)

    def test_endpoint_arbitration_can_be_disabled(self) -> None:
        config = copy.deepcopy(self.config)
        config["core"]["endpoint_arbitration"]["enabled"] = False
        tops, bottoms = make_cycle_states(
            [100, 200, 300, 400, 500],
            top_angles=[120.0, 160.0, 160.0, 160.0, 160.0],
        )
        analysis, _ = chain_selection.analyze_phase_states(
            "endpoint_arbitration_disabled",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=700,
            config=config,
        )

        self.assertEqual(len(analysis["sets"][0]["reps"]), 3)
        self.assertEqual(
            analysis["diagnostics"]["endpoint_arbitration_decisions"], []
        )

    def test_low_rom_interior_bridge_merges_two_core_chains(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500, 600, 700, 800],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0, 160.0, 160.0, 160.0],
            frame_count=1000,
        )

        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 7)
        bridged = [
            event
            for event in analysis["diagnostics"]["selected_top_events"]
            if event["low_rom_interior_bridged"]
        ]
        self.assertEqual([event["frame"] for event in bridged], [500])
        decision = analysis["diagnostics"][
            "low_rom_interior_bridge_decisions"
        ][0]
        self.assertEqual(decision["decision"], "add_rep")
        self.assertEqual(decision["frame"], 500)
        self.assertEqual(decision["combined_gap_ratio"], 2.0)
        self.assertIn("low_rom", analysis["sets"][0]["reps"][4]["quality_flags"])

    def test_low_rom_interior_bridge_rejects_outlier_combined_gap(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 520, 650, 750, 850],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0, 160.0, 160.0, 160.0],
            frame_count=1200,
        )

        decision = analysis["diagnostics"][
            "low_rom_interior_bridge_decisions"
        ][0]
        self.assertEqual(decision["decision"], "keep_context")
        self.assertEqual(
            decision["reason"], "combined_gap_outside_missing_event_range"
        )

    def test_low_rom_interior_bridge_does_not_insert_into_direct_edge(self) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500, 600, 700, 800],
            top_angles=[160.0, 160.0, 160.0, 160.0, 120.0, 160.0, 160.0, 160.0],
            frame_count=1000,
            period_frames=200.0,
        )

        frames = [
            event["frame"]
            for event in analysis["diagnostics"]["selected_top_events"]
        ]
        self.assertNotIn(500, frames)
        self.assertFalse(
            any(
                decision["decision"] == "add_rep"
                for decision in analysis["diagnostics"][
                    "low_rom_interior_bridge_decisions"
                ]
            )
        )

    def test_low_rom_interior_bridge_can_be_disabled(self) -> None:
        config = copy.deepcopy(self.config)
        config["core"]["low_rom_interior_bridge"]["enabled"] = False
        tops, bottoms = make_cycle_states(
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            top_angles=[
                160.0,
                160.0,
                160.0,
                160.0,
                120.0,
                160.0,
                160.0,
                160.0,
                160.0,
            ],
        )
        analysis, _ = chain_selection.analyze_phase_states(
            "low_rom_interior_bridge_disabled",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=1100,
            config=config,
        )

        self.assertEqual(len(analysis["sets"]), 2)
        self.assertEqual(
            analysis["diagnostics"]["low_rom_interior_bridge_decisions"], []
        )

    def test_endpoint_context_pruning_removes_setup_like_leading_cycle(self) -> None:
        tops, bottoms = make_cycle_states(
            [100, 165, 265, 365, 465, 565],
            top_angles=[130.0, 160.0, 160.0, 160.0, 160.0, 160.0],
            bottom_angles=[95.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )
        phase_values = np.full(700, 120.0)
        for index, (left, right) in enumerate(
            zip([100, 165, 265, 365, 465], [165, 265, 365, 465, 565])
        ):
            amplitude = 20.0 if index == 0 else 40.0
            times = np.linspace(0.0, np.pi, right - left)
            phase_values[left:right] -= amplitude * np.sin(times) ** 2
        analysis, _ = chain_selection.analyze_phase_states(
            "endpoint_context_setup",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=700,
            config=self.config,
            phase_signal_values=phase_values,
        )

        frames = [
            event["frame"]
            for event in analysis["diagnostics"]["selected_top_events"]
        ]
        self.assertNotIn(100, frames)
        record = analysis["diagnostics"]["endpoint_context_pruned_events"][0]
        self.assertEqual(record["role"], "leading")
        self.assertEqual(record["reason"], "setup_recovery_context_supported")
        self.assertGreaterEqual(record["evidence_count"], 3)
        self.assertIn("attenuated_amplitude", record["evidence"])
        self.assertIn("endpoint_gap_outlier", record["evidence"])
        self.assertIn("level_shift", record["evidence"])

    def test_endpoint_context_pruning_protects_arbitrated_low_rom_rep(self) -> None:
        tops, bottoms = make_cycle_states(
            [100, 165, 265, 365, 465, 565],
            top_angles=[120.0, 160.0, 160.0, 160.0, 160.0, 160.0],
            bottom_angles=[100.0] * 6,
        )
        phase_values = np.full(700, 120.0)
        for index, (left, right) in enumerate(
            zip([100, 165, 265, 365, 465], [165, 265, 365, 465, 565])
        ):
            amplitude = 20.0 if index == 0 else 40.0
            times = np.linspace(0.0, np.pi, right - left)
            phase_values[left:right] -= amplitude * np.sin(times) ** 2
        analysis, _ = chain_selection.analyze_phase_states(
            "endpoint_context_low_rom",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action,
            phase_signal="synthetic_flexion",
            frame_count=700,
            config=self.config,
            phase_signal_values=phase_values,
        )

        frames = [
            event["frame"]
            for event in analysis["diagnostics"]["selected_top_events"]
        ]
        self.assertIn(100, frames)
        self.assertTrue(frames)
        self.assertEqual(
            analysis["diagnostics"]["endpoint_context_pruned_events"], []
        )

    def test_scale_offset_frame_rate_noise_and_missing_frames_are_invariant(
        self,
    ) -> None:
        variants = [
            (50.0, synthetic_wave(50.0, 8.0, 1.6)),
            (15.0, synthetic_wave(15.0, 8.0, 1.6)),
            (30.0, synthetic_wave(30.0, 8.0, 1.6)),
            (60.0, synthetic_wave(60.0, 8.0, 1.6)),
            (
                50.0,
                synthetic_wave(
                    50.0, 8.0, 1.6, amplitude=11.0, offset=-35.0
                ),
            ),
            (
                50.0,
                synthetic_wave(
                    50.0,
                    8.0,
                    1.6,
                    noise=1.5,
                    missing_fraction=0.12,
                    seed=7,
                ),
            ),
        ]
        results = []
        for fps, values in variants:
            analysis = self.analyze_detected(values, fps)
            results.append(
                (
                    len(analysis["sets"]),
                    sum(len(item["reps"]) for item in analysis["sets"]),
                )
            )
        self.assertEqual(results, [(1, 5)] * len(variants))

    def test_two_independent_groups_produce_two_sets_and_longest_default(
        self,
    ) -> None:
        analysis = self.analyze(
            [100, 200, 300, 400, 500, 600, 1000, 1100, 1200, 1300]
        )
        self.assertEqual(len(analysis["sets"]), 2)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 5)
        self.assertEqual(len(analysis["sets"][1]["reps"]), 3)
        self.assertTrue(analysis["sets"][0]["is_default"])
        self.assertFalse(analysis["sets"][1]["is_default"])

    def test_legacy_fallback_converts_absolute_frames_to_clip_local(self) -> None:
        legacy = {
            "segmentation": {
                "reps": [
                    {
                        "start_frame": 1100,
                        "end_frame": 1140,
                        "status": "complete",
                        "confidence": 0.8,
                    },
                    {
                        "start_frame": 1141,
                        "end_frame": 1180,
                        "status": "complete",
                        "confidence": 0.7,
                    },
                ]
            }
        }
        analysis = chain_selection.legacy_sets_from_analysis(
            "qeud_clip",
            legacy,
            frame_count=200,
            reason="unit_test",
            frame_offset=1000,
        )
        reps = analysis["sets"][0]["reps"]
        self.assertEqual(
            [(rep["start_frame"], rep["end_frame"]) for rep in reps],
            [(100, 140), (141, 180)],
        )
        self.assertEqual(
            analysis["non_rep_intervals"][-1],
            {
                "start_frame": 181,
                "end_frame": 199,
                "kind": "recovery",
                "confidence": 0.8,
            },
        )
        self.assert_contract_fields(analysis)

    def test_quality_reject_and_unknown_short_circuit_chain_outputs(self) -> None:
        values = synthetic_wave(50.0, 10.0, 1.6)
        for status, expected in (
            ("reject", "rejected_quality"),
            ("unknown", "not_evaluable_quality"),
        ):
            with self.subTest(status=status):
                analysis, meta = chain_selection.analyze_clip_signals(
                    {"clip_id": "synthetic_clip", "fps": 50.0},
                    {
                        "left_elbow_flexion": values,
                        "right_elbow_flexion": values,
                    },
                    self.action,
                    prior_signal=None,
                    phase_policy="t1_6_bilateral_mean",
                    chain_config=self.config,
                    quality_gate={
                        "status": status,
                        "scope": "clip",
                        "reasons": ["test"],
                        "source": "pose",
                    },
                )
                self.assertEqual(analysis["status"], expected)
                self.assertEqual(analysis["sets"], [])
                self.assertEqual(
                    analysis["diagnostics"]["selected_top_events"], []
                )
                self.assertEqual(meta["selected_top_events"], [])

    def test_quality_warn_preserves_chain_outputs(self) -> None:
        values = synthetic_wave(50.0, 10.0, 1.6)
        action = {
            **self.action,
            "phase_signal_candidates": [
                "left_elbow_flexion",
                "right_elbow_flexion",
            ],
            "phase_signal_pairs": [
                {
                    "name": "elbow_flexion_bilateral_mean",
                    "left": "left_elbow_flexion",
                    "right": "right_elbow_flexion",
                }
            ],
        }
        analysis, _ = chain_selection.analyze_clip_signals(
            {"clip_id": "synthetic_clip", "fps": 50.0},
            {
                "left_elbow_flexion": values,
                "right_elbow_flexion": values,
            },
            action,
            prior_signal=None,
            phase_policy="t1_6_bilateral_mean",
            chain_config=self.config,
            quality_gate={
                "status": "warn",
                "scope": "clip",
                "reasons": ["posture"],
                "source": "pose",
            },
        )
        self.assertTrue(analysis["sets"])
        self.assertEqual(analysis["quality_gate"]["status"], "warn")

    def test_idle_padding_does_not_change_rep_structure(self) -> None:
        action = {
            **self.action,
            "phase_signal_candidates": [
                "left_elbow_flexion",
                "right_elbow_flexion",
            ],
            "phase_signal_pairs": [
                {
                    "name": "elbow_flexion_bilateral_mean",
                    "left": "left_elbow_flexion",
                    "right": "right_elbow_flexion",
                }
            ],
        }
        values = synthetic_wave(50.0, 8.0, 1.6)
        padding = np.full(100, float(np.median(values)))
        counts = []
        for signal in (values, np.concatenate([padding, values, padding])):
            analysis, _ = chain_selection.analyze_clip_signals(
                {"clip_id": "synthetic_clip", "fps": 50.0},
                {
                    "left_elbow_flexion": signal,
                    "right_elbow_flexion": signal,
                },
                action,
                prior_signal=None,
                phase_policy="t1_6_bilateral_mean",
                chain_config=self.config,
            )
            counts.append(
                (
                    len(analysis["sets"]),
                    sum(len(item["reps"]) for item in analysis["sets"]),
                )
            )
        self.assertEqual(counts[0], counts[1])
        self.assertGreater(counts[1][1], 0)

    def test_chain_config_core_has_no_source_specific_tokens(self) -> None:
        core_text = json.dumps(self.config["core"], ensure_ascii=False)
        for token in (
            "Fit3D",
            "QEVD",
            "s03",
            "s11",
            "50591643",
            "58860488",
            "60457274",
            "65906101",
            "fixed_rep_count",
            "fixed_frame_offset",
            "camera_id",
            "subject_id",
            "video_id",
        ):
            self.assertNotIn(token, core_text)
        self.assertTrue(self.config["core"]["dataset_independent"])
        self.assertEqual(
            self.config["core"]["tail_anchor"]["policy"],
            chain_selection.TAIL_ANCHOR_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
