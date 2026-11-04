from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import task1_6_chain_selection as chain_selection
import task1_6_phase_states as phase_states
import task1_7_signals as signals_v3


def sine_signal(period_seconds: float, *, fps: float, seconds: float) -> np.ndarray:
    times = np.arange(int(round(fps * seconds)), dtype=np.float64) / fps
    return 100.0 + 40.0 * (
        0.5 - 0.5 * np.cos(2.0 * np.pi * times / period_seconds)
    )


def consensus_event(
    anchor: int,
    *,
    kind: str = "interior",
    stable_fraction: float = 1.0,
    scale_support: int = 1,
) -> dict:
    return {
        "start_frame": anchor - 5,
        "end_frame": anchor + 5,
        "anchor_frame": anchor,
        "anchors": {
            "plateau_entry": anchor - 5,
            "plateau_midpoint": anchor,
            "plateau_exit": anchor + 5,
        },
        "status": "valid",
        "kind": kind,
        "stable_fraction": stable_fraction,
        "boundary_slope": 0.0,
        "mean_angle": 150.0,
        "evidence": {
            "duration_fraction": 0.3,
            "scale_support": scale_support,
            "scale_support_fraction": scale_support / 4.0,
            "member_count": scale_support,
            "multi_scale_consensus": scale_support >= 2,
        },
        "quality_flags": [],
    }


def bridge_node(
    index: int,
    anchor: int,
    *,
    eligible: bool,
    relative_amplitude: float,
    top_stability: float = 0.8,
    bottom_stability: float = 0.5,
    support_fraction: float = 1.0,
) -> dict:
    top = consensus_event(
        anchor,
        stable_fraction=top_stability,
        scale_support=4,
    )
    bottom = consensus_event(anchor + 50)
    bottom["mean_angle"] = 100.0
    return {
        "index": index,
        "top": top,
        "bottom": bottom,
        "event_frame": anchor,
        "output_frame": anchor,
        "amplitude": 20.0 if relative_amplitude < 0.5 else 60.0,
        "bottom_duration_fraction": 0.3,
        "top_stability": top_stability,
        "bottom_stability": bottom_stability,
        "fallback_penalty": 1.0,
        "scale_support": 4,
        "multi_scale_consensus": True,
        "partial_endpoint": False,
        "relative_amplitude": relative_amplitude,
        "eligible": eligible,
        "quality": 0.8,
        "initial_confidence": {
            "complete": 0.8,
            "in_set": 0.8,
            "combined": 0.8,
        },
    }


def timing_node(
    index: int,
    frame: int,
    *,
    kind: str = "interior",
    stable_fraction: float = 0.9,
) -> dict:
    top = consensus_event(
        frame,
        kind=kind,
        stable_fraction=stable_fraction,
        scale_support=3,
    )
    return {
        "index": index,
        "top": top,
        "bottom": consensus_event(frame + 10),
        "event_frame": frame,
        "output_frame": frame,
        "amplitude": 60.0,
        "bottom_duration_fraction": 0.3,
        "top_stability": stable_fraction,
        "bottom_stability": 0.5,
        "fallback_penalty": 1.0,
        "scale_support": 3,
        "multi_scale_consensus": True,
        "partial_endpoint": kind != "interior",
        "relative_amplitude": 1.0,
        "eligible": True,
        "quality": 0.8,
        "initial_confidence": {
            "complete": 0.8,
            "in_set": 0.8,
            "combined": 0.8,
        },
    }


class SignalLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = signals_v3.load_config(
            ROOT / "configs" / "task1_7_signal_p1.json"
        )
        cls.specs = json.loads(
            (ROOT / "configs" / "task1_4_exercise_specs.json").read_text()
        )
        cls.chain_config = chain_selection.load_chain_config(
            ROOT / "configs" / "task1_6_chain.json"
        )

    def action(self, name: str) -> dict:
        return phase_states.action_config(name, self.specs)

    def test_constant_fps_timebase(self) -> None:
        timebase = signals_v3.build_timebase(
            300, 30.0, self.config
        )
        self.assertFalse(timebase["timestamps_present"])
        self.assertFalse(timebase["variable_fps"])
        self.assertFalse(timebase["resampled"])
        self.assertEqual(timebase["frame_count"], 300)
        self.assertEqual(timebase["output_frame_count"], 300)
        self.assertAlmostEqual(timebase["duration_seconds"], 299.0 / 30.0)

    def test_variable_timestamps_are_detected_and_resampled(self) -> None:
        rng = np.random.default_rng(7)
        count = 150
        deltas = 1.0 / 30.0 + rng.normal(0.0, 0.004, count - 1)
        deltas = np.clip(deltas, 0.02, 0.06)
        timestamps = np.concatenate([[0.0], np.cumsum(deltas)])
        timebase = signals_v3.build_timebase(
            count,
            30.0,
            self.config,
            timestamps=timestamps,
        )
        self.assertTrue(timebase["timestamps_present"])
        self.assertTrue(timebase["variable_fps"])
        self.assertGreaterEqual(timebase["analysis_fps"], 10.0)
        self.assertLessEqual(timebase["analysis_fps"], 60.0)

    def test_decreasing_timestamps_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            signals_v3.build_timebase(
                4,
                30.0,
                self.config,
                timestamps=[0.0, 0.1, 0.05, 0.2],
            )

    def test_short_gap_is_interpolated(self) -> None:
        values = np.full(120, 100.0)
        values[40:44] = np.nan
        times = np.arange(120, dtype=np.float64) / 30.0
        filled, evidence = signals_v3.interpolate_signal_gaps(
            values, times, self.config
        )
        self.assertTrue(np.isfinite(filled).all())
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["short_gap_count"], 1)
        self.assertEqual(evidence["unresolved_gap_count"], 0)

    def test_long_gap_fails_closed(self) -> None:
        values = np.full(120, 100.0)
        values[40:75] = np.nan
        times = np.arange(120, dtype=np.float64) / 30.0
        _, evidence = signals_v3.interpolate_signal_gaps(
            values, times, self.config
        )
        self.assertEqual(evidence["status"], "insufficient_signal")
        self.assertGreaterEqual(evidence["unresolved_gap_count"], 1)

    def test_isolated_long_gap_degrades_candidate_with_evidence(self) -> None:
        values = sine_signal(1.2, fps=50.0, seconds=12.0)
        values[240:300] = np.nan
        candidate = signals_v3.evaluate_signal_candidate(
            values,
            name="elbow_flexion_bilateral_mean",
            fps=50.0,
            action=self.action("pushup"),
            config=self.config,
            detector={"smoothing_seconds": 0.12},
            bilateral=True,
        )
        self.assertEqual(candidate["status"], "degraded")
        self.assertEqual(candidate["gap_evidence"]["status"], "degraded")
        self.assertIn(
            "unresolved_gap_interpolated_for_detection",
            candidate["gap_evidence"]["reasons"],
        )

    def test_pushup_bilateral_signal_is_selected(self) -> None:
        values = sine_signal(1.2, fps=30.0, seconds=10.0)
        resolution = signals_v3.analyze_phase_signals(
            {
                "left_elbow_flexion": values.copy(),
                "right_elbow_flexion": values.copy(),
            },
            30.0,
            self.action("pushup"),
            self.config,
            detector={"smoothing_seconds": 0.12},
        )
        self.assertEqual(resolution["status"], "ok")
        self.assertEqual(
            resolution["selected_phase_signal"],
            "elbow_flexion_bilateral_mean",
        )
        selected = next(
            candidate
            for candidate in resolution["candidates"]
            if candidate["phase_signal"]
            == resolution["selected_phase_signal"]
        )
        self.assertGreater(selected["consensus_score"] if "consensus_score" in selected else resolution["consensus_score"], 0.0)
        self.assertGreater(selected["multiscale_agreement"], 0.0)
        self.assertIn(
            selected["selected_scale_seconds"],
            self.config["core"]["smoothing_scales_seconds"],
        )
        self.assertEqual(selected["selected_scale_seconds"], 0.12)

    def test_conflicting_squat_families_are_ambiguous(self) -> None:
        fps = 30.0
        knee = sine_signal(1.0, fps=fps, seconds=10.0)
        hip = sine_signal(1.7, fps=fps, seconds=10.0)
        resolution = signals_v3.analyze_phase_signals(
            {
                "left_knee_flexion": knee,
                "right_knee_flexion": knee.copy(),
                "left_hip_flexion_sagittal": hip,
                "right_hip_flexion_sagittal": hip.copy(),
            },
            fps,
            self.action("squat"),
            self.config,
            detector={"smoothing_seconds": 0.12},
        )
        self.assertEqual(resolution["status"], "ambiguous")
        self.assertEqual(
            resolution["ambiguity_reason"],
            "phase_signal_family_disagreement",
        )

    def test_phase_signal_event_consensus_rejects_single_side_conflict(
        self,
    ) -> None:
        selected = {
            "phase_signal": "right_elbow_flexion",
            "family": "elbow_flexion",
            "bilateral": False,
            "event_evidence": {
                "period_frames": 30.0,
                "top_anchors": [130, 230, 330, 430],
                "top_kinds": ["interior"] * 4,
            },
        }
        bilateral = {
            "phase_signal": "elbow_flexion_bilateral_mean",
            "family": "elbow_flexion",
            "bilateral": True,
            "event_evidence": {
                "period_frames": 30.0,
                "top_anchors": [100, 200, 300, 400],
                "top_kinds": ["interior"] * 4,
            },
        }
        evidence = signals_v3._event_consensus_evidence(
            selected,
            [selected, bilateral],
            self.config["core"],
        )
        self.assertEqual(evidence["status"], "disagreement")
        self.assertEqual(
            evidence["uncorroborated_event_frames"],
            [130, 230, 330, 430],
        )

    def test_phase_signal_event_consensus_accepts_bilateral_support(
        self,
    ) -> None:
        selected = {
            "phase_signal": "right_elbow_flexion",
            "family": "elbow_flexion",
            "bilateral": False,
            "event_evidence": {
                "period_frames": 30.0,
                "top_anchors": [130, 230, 330, 430],
                "top_kinds": ["interior"] * 4,
            },
        }
        bilateral = {
            "phase_signal": "elbow_flexion_bilateral_mean",
            "family": "elbow_flexion",
            "bilateral": True,
            "event_evidence": {
                "period_frames": 30.0,
                "top_anchors": [132, 228, 333, 428],
                "top_kinds": ["interior"] * 4,
            },
        }
        evidence = signals_v3._event_consensus_evidence(
            selected,
            [selected, bilateral],
            self.config["core"],
        )
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["uncorroborated_event_frames"], [])

    def test_prior_family_breaks_near_tie(self) -> None:
        fps = 30.0
        knee = sine_signal(1.2, fps=fps, seconds=10.0)
        hip = sine_signal(1.2, fps=fps, seconds=10.0)
        resolution = signals_v3.analyze_phase_signals(
            {
                "left_knee_flexion": knee,
                "right_knee_flexion": knee.copy(),
                "left_hip_flexion_sagittal": hip,
                "right_hip_flexion_sagittal": hip.copy(),
            },
            fps,
            self.action("squat"),
            self.config,
            prior_signal="left_knee_flexion",
            detector={"smoothing_seconds": 0.12},
        )
        self.assertEqual(resolution["status"], "ok")
        self.assertEqual(
            resolution["selected_phase_signal"],
            "knee_flexion_bilateral_mean",
        )

    def test_prior_family_suppresses_cross_family_period_ambiguity(self) -> None:
        fps = 30.0
        knee = sine_signal(1.0, fps=fps, seconds=10.0)
        hip = sine_signal(1.7, fps=fps, seconds=10.0)
        resolution = signals_v3.analyze_phase_signals(
            {
                "left_knee_flexion": knee,
                "right_knee_flexion": knee.copy(),
                "left_hip_flexion_sagittal": hip,
                "right_hip_flexion_sagittal": hip.copy(),
            },
            fps,
            self.action("squat"),
            self.config,
            prior_signal="left_knee_flexion",
            detector={"smoothing_seconds": 0.12},
        )
        self.assertEqual(resolution["status"], "ok")
        self.assertIsNone(resolution["ambiguity_reason"])
        self.assertEqual(
            resolution["selected_phase_signal"],
            "knee_flexion_bilateral_mean",
        )

    def test_phase_resolution_integrates_with_chain_selection(self) -> None:
        fps = 30.0
        values = sine_signal(1.0, fps=fps, seconds=8.0)
        resolution = signals_v3.analyze_phase_signals(
            {
                "left_elbow_flexion": values.copy(),
                "right_elbow_flexion": values.copy(),
            },
            fps,
            self.action("pushup"),
            self.config,
            detector={"smoothing_seconds": 0.12},
        )
        analysis, meta = chain_selection.analyze_clip_signals(
            {"clip_id": "clip_a", "fps": resolution["analysis_fps"]},
            resolution["signals"],
            self.action("pushup"),
            prior_signal=None,
            phase_policy="t1_6_bilateral_mean",
            chain_config=self.chain_config,
            detector={"smoothing_seconds": 0.12},
            phase_resolution=resolution,
        )
        self.assertEqual(analysis["status"], "ok")
        self.assertGreaterEqual(len(analysis["sets"]), 1)
        self.assertGreater(meta["periodicity"], 0.0)
        self.assertIn("phase_signal_evidence", analysis["diagnostics"])

    def test_period_consensus_uses_the_most_supported_cluster(self) -> None:
        settings = self.config["core"]["event_consensus"]
        results = [
            {"period_frames": 100.0, "smoothing_seconds": 0.12},
            {"period_frames": 101.0, "smoothing_seconds": 0.2},
            {"period_frames": 103.0, "smoothing_seconds": 0.32},
            {"period_frames": 150.0, "smoothing_seconds": 0.48},
        ]
        period, members = signals_v3._select_period_consensus(results, settings)
        self.assertAlmostEqual(period, 101.0)
        self.assertEqual(
            [item["smoothing_seconds"] for item in members],
            [0.12, 0.2, 0.32],
        )

    def test_half_cycle_recovery_restores_missing_singleton_top(self) -> None:
        settings = self.config["core"]["event_consensus"]
        scale_results = [
            {
                "smoothing_seconds": 0.48,
                "top_states": [consensus_event(203)],
            }
        ]
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(300)],
            scale_results,
            100.0,
            settings,
        )
        self.assertEqual([state["anchor_frame"] for state in recovered], [100, 203, 300])
        self.assertEqual(len(records), 1)
        evidence = recovered[1]["evidence"]
        self.assertTrue(evidence["half_cycle_recovered"])
        self.assertEqual(evidence["recovery_role"], "top")
        self.assertEqual(evidence["recovery_scale_seconds"], 0.48)
        self.assertFalse(evidence["multi_scale_consensus"])
        self.assertEqual(recovered[1]["kind"], "interior")

    def test_half_cycle_recovery_rejects_far_anchor(self) -> None:
        settings = self.config["core"]["event_consensus"]
        scale_results = [
            {
                "smoothing_seconds": 0.48,
                "top_states": [consensus_event(250)],
            }
        ]
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(300)],
            scale_results,
            100.0,
            settings,
        )
        self.assertEqual([state["anchor_frame"] for state in recovered], [100, 300])
        self.assertEqual(records, [])

    def test_half_cycle_recovery_fills_bottom_for_open_rep_interval(self) -> None:
        settings = self.config["core"]["event_consensus"]
        bottom = consensus_event(150)
        bottom["stable_fraction"] = 0.0
        bottom["mean_angle"] = 120.0
        scale_results = [
            {
                "smoothing_seconds": 0.48,
                "bottom_states": [bottom],
            }
        ]
        recovered, records = signals_v3._recover_missing_bottoms(
            [consensus_event(100), consensus_event(200)],
            [],
            scale_results,
            100.0,
            settings,
        )
        self.assertEqual([state["anchor_frame"] for state in recovered], [150])
        self.assertEqual(len(records), 1)
        self.assertEqual(
            recovered[0]["evidence"]["recovery_role"], "bottom"
        )
        self.assertTrue(recovered[0]["evidence"]["half_cycle_recovered"])

    def test_half_cycle_recovery_does_not_change_normal_gap(self) -> None:
        settings = self.config["core"]["event_consensus"]
        scale_results = [
            {
                "smoothing_seconds": 0.48,
                "top_states": [consensus_event(150)],
            }
        ]
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(200)],
            scale_results,
            100.0,
            settings,
        )
        self.assertEqual([state["anchor_frame"] for state in recovered], [100, 200])
        self.assertEqual(records, [])

    def test_near_boundary_recovery_uses_multiscale_rejected_top(self) -> None:
        settings = self.config["core"]["event_consensus"]
        rejected = []
        for scale in (0.12, 0.2, 0.32, 0.48):
            state = consensus_event(250)
            state.update(
                {
                    "reason": "no_stable_top_state",
                    "relative_peak_fraction": 0.5,
                }
            )
            rejected.append(
                {
                    "smoothing_seconds": scale,
                    "rejected_top_states": [state],
                }
            )
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(350)],
            rejected,
            100.0,
            settings,
        )
        self.assertEqual(
            [state["anchor_frame"] for state in recovered], [100, 250, 350]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["recovery_source"], "near_boundary_rejected_top"
        )
        evidence = recovered[1]["evidence"]
        self.assertTrue(evidence["half_cycle_recovered"])
        self.assertTrue(evidence["near_boundary_rejected_top"])
        self.assertEqual(evidence["scale_support"], 4)

    def test_near_boundary_recovery_rejects_single_scale_support(self) -> None:
        settings = self.config["core"]["event_consensus"]
        state = consensus_event(250)
        state.update(
            {
                "reason": "no_stable_top_state",
                "relative_peak_fraction": 0.5,
            }
        )
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(350)],
            [{"smoothing_seconds": 0.12, "rejected_top_states": [state]}],
            100.0,
            settings,
        )
        self.assertEqual(
            [state["anchor_frame"] for state in recovered], [100, 350]
        )
        self.assertEqual(records, [])

    def test_near_boundary_recovery_rejects_weak_peak(self) -> None:
        settings = self.config["core"]["event_consensus"]
        scale_results = []
        for scale in (0.12, 0.2, 0.32, 0.48):
            state = consensus_event(250)
            state.update(
                {
                    "reason": "no_stable_top_state",
                    "relative_peak_fraction": 0.3,
                }
            )
            scale_results.append(
                {
                    "smoothing_seconds": scale,
                    "rejected_top_states": [state],
                }
            )
        recovered, records = signals_v3._recover_missing_tops(
            [consensus_event(100), consensus_event(350)],
            scale_results,
            100.0,
            settings,
        )
        self.assertEqual(
            [state["anchor_frame"] for state in recovered], [100, 350]
        )
        self.assertEqual(records, [])

    def test_sampling_aware_motion_gate_requires_joint_evidence(self) -> None:
        policy = self.config["core"]["motion_gate"][
            "sampling_aware_state_amplitude"
        ]
        settings = {
            **phase_states.DEFAULT_DETECTOR,
            "motion_joint_state_amplitude": policy,
        }
        prepared = {
            "fps": 15.0,
            "periodicity": 0.479276,
            "dynamic_range": 16.61054,
            "noise_sigma": 1.17372,
            "dynamic_to_noise": 14.152055,
            "settings": settings,
        }
        evidence = phase_states.build_motion_evidence(
            prepared,
            state_amplitude=2.417776,
        )
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["reasons"], [])
        self.assertTrue(
            evidence["joint_state_amplitude"]["override_applied"]
        )

        weak_periodicity = {
            **prepared,
            "periodicity": 0.2,
        }
        rejected = phase_states.build_motion_evidence(
            weak_periodicity,
            state_amplitude=2.417776,
        )
        self.assertEqual(rejected["status"], "no_motion")
        self.assertEqual(
            rejected["reasons"], ["state_amplitude_below_minimum"]
        )

    def test_low_sampling_pruning_removes_only_unstable_multi_endpoint(self) -> None:
        nodes = {
            index: {
                "event_frame": 100 + index * 20,
                "quality": 0.8,
                "multi_scale_consensus": True,
                "top": {
                    "kind": (
                        "leading_partial"
                        if index == 0
                        else "trailing_partial"
                        if index == 5
                        else "interior"
                    ),
                    "stable_fraction": 0.4 if index == 5 else 0.9,
                    "evidence": {},
                },
            }
            for index in range(6)
        }
        policy = {
            "low_sampling_endpoint_pruning": {
                "enabled": True,
                "minimum_retained_events": 4,
                "maximum_pruned_endpoint_stability": 0.6,
            }
        }
        pruned = chain_selection.low_sampling_endpoint_pruning(
            list(range(6)), nodes, policy
        )
        self.assertIsNotNone(pruned)
        self.assertEqual(pruned[0], [0, 1, 2, 3, 4])
        self.assertEqual(pruned[1]["frame"], 200)
        self.assertEqual(pruned[1]["role"], "trailing")

        nodes[5]["top"]["stable_fraction"] = 0.8
        self.assertIsNone(
            chain_selection.low_sampling_endpoint_pruning(
                list(range(6)), nodes, policy
            )
        )

    def test_low_sampling_pruning_keeps_singleton_endpoint(self) -> None:
        nodes = {
            index: {
                "event_frame": 100 + index * 20,
                "quality": 0.8,
                "multi_scale_consensus": index > 0,
                "top": {
                    "kind": "leading_partial" if index == 0 else "interior",
                    "stable_fraction": 0.4,
                    "evidence": {},
                },
            }
            for index in range(6)
        }
        nodes[5]["top"]["kind"] = "trailing_partial"
        policy = {
            "low_sampling_endpoint_pruning": {
                "enabled": True,
                "minimum_retained_events": 4,
                "maximum_pruned_endpoint_stability": 0.6,
            }
        }
        self.assertIsNone(
            chain_selection.low_sampling_endpoint_pruning(
                list(range(6)), nodes, policy
            )
        )

    def test_endpoint_timing_retains_weak_partial_at_expected_period(self) -> None:
        frames = [12, 47, 69, 112, 146, 173, 192, 216]
        kinds = ["leading_partial", *(["interior"] * 6), "trailing_partial"]
        stability = [0.95, 1.0, 0.333, 0.5, 1.0, 0.375, 0.909, 0.4375]
        nodes = {
            index: timing_node(
                index,
                frame,
                kind=kinds[index],
                stable_fraction=stability[index],
            )
            for index, frame in enumerate(frames)
        }
        policy = {
            "low_sampling_endpoint_pruning": {
                "enabled": True,
                "minimum_retained_events": 4,
                "maximum_pruned_endpoint_stability": 0.6,
            },
            "endpoint_timing_model": {
                **self.config["core"]["event_consensus"][
                    "endpoint_timing_model"
                ]
            },
        }
        timing = chain_selection.evaluate_endpoint_timing(
            list(range(8)), nodes, 24.0, policy
        )
        self.assertTrue(timing["endpoints"]["trailing"]["timing_plausible"])
        self.assertIsNone(
            chain_selection.low_sampling_endpoint_pruning(
                list(range(8)), nodes, policy, 24.0
            )
        )

    def test_endpoint_timing_prunes_low_sampling_partial_outlier(self) -> None:
        frames = [89, 138, 169, 202, 240]
        kinds = ["leading_partial", *(["interior"] * 3), "trailing_partial"]
        stability = [0.889, 0.778, 0.5, 0.889, 0.414]
        nodes = {
            index: timing_node(
                index,
                frame,
                kind=kinds[index],
                stable_fraction=stability[index],
            )
            for index, frame in enumerate(frames)
        }
        policy = {
            "low_sampling_endpoint_pruning": {
                "enabled": True,
                "minimum_retained_events": 4,
                "maximum_pruned_endpoint_stability": 0.6,
            },
            "endpoint_timing_model": {
                **self.config["core"]["event_consensus"][
                    "endpoint_timing_model"
                ]
            },
        }
        pruned = chain_selection.low_sampling_endpoint_pruning(
            list(range(5)), nodes, policy, 32.0
        )
        self.assertIsNotNone(pruned)
        self.assertEqual(pruned[0], [0, 1, 2, 3])
        self.assertEqual(pruned[1]["frame"], 240)

    def test_endpoint_timing_prunes_short_partial_tail(self) -> None:
        frames = [61, 182, 320, 447, 585, 694, 878, 992, 1074]
        nodes = {
            index: timing_node(
                index,
                frame,
                kind=(
                    "leading_partial"
                    if index == 0
                    else "trailing_partial"
                    if index == len(frames) - 1
                    else "interior"
                ),
            )
            for index, frame in enumerate(frames)
        }
        policy = {
            "endpoint_timing_model": {
                **self.config["core"]["event_consensus"][
                    "endpoint_timing_model"
                ]
            }
        }
        timing = chain_selection.evaluate_endpoint_timing(
            list(range(len(frames))), nodes, 126.0, policy
        )
        self.assertEqual(timing["decision"], "prune")
        self.assertEqual(timing["prune_role"], "trailing")
        pruned = chain_selection.endpoint_timing_pruning(
            list(range(len(frames))), nodes, 126.0, policy
        )
        self.assertIsNotNone(pruned)
        self.assertEqual(pruned[0], list(range(len(frames) - 1)))
        self.assertEqual(pruned[1]["reason"], "short_partial_endpoint")

    def test_endpoint_timing_keeps_regular_chain(self) -> None:
        nodes = {
            index: timing_node(index, 100 + index * 50)
            for index in range(6)
        }
        policy = {
            "endpoint_timing_model": {
                **self.config["core"]["event_consensus"][
                    "endpoint_timing_model"
                ]
            }
        }
        self.assertIsNone(
            chain_selection.endpoint_timing_pruning(
                list(range(6)), nodes, 50.0, policy
            )
        )
        self.assertIsNone(
            chain_selection.evaluate_endpoint_timing(
                list(range(6)), nodes, 50.0, None
            )
        )

    def test_endpoint_timing_prunes_long_partial_tail(self) -> None:
        frames = [100, 200, 300, 400, 500, 850]
        nodes = {
            index: timing_node(
                index,
                frame,
                kind=(
                    "trailing_partial"
                    if index == len(frames) - 1
                    else "interior"
                ),
            )
            for index, frame in enumerate(frames)
        }
        policy = {
            "endpoint_timing_model": {
                **self.config["core"]["event_consensus"][
                    "endpoint_timing_model"
                ]
            }
        }
        timing = chain_selection.evaluate_endpoint_timing(
            list(range(len(frames))), nodes, 100.0, policy
        )
        self.assertEqual(timing["decision"], "prune")
        self.assertEqual(timing["reason"], "long_partial_endpoint")
        self.assertEqual(timing["prune_role"], "trailing")

    def test_local_extrema_stability_bridges_stable_leading_peak(self) -> None:
        def build_nodes() -> tuple[list[dict], list[dict]]:
            nodes = [
                bridge_node(0, 100, eligible=False, relative_amplitude=0.2),
                bridge_node(1, 200, eligible=False, relative_amplitude=0.2),
                bridge_node(2, 300, eligible=True, relative_amplitude=1.0),
            ]
            nodes[0]["top"]["kind"] = "leading_partial"
            nodes[0]["top"]["mean_angle"] = 150.0
            nodes[1]["top"]["kind"] = "leading_partial"
            nodes[1]["top"]["mean_angle"] = 160.0
            nodes[2]["top"]["mean_angle"] = 160.0
            nodes[0]["amplitude"] = 10.0
            nodes[1]["amplitude"] = 10.0
            nodes[2]["amplitude"] = 60.0
            spurious_bottom = consensus_event(150)
            spurious_bottom["mean_angle"] = 155.0
            valid_bottom = consensus_event(250, scale_support=4)
            valid_bottom["mean_angle"] = 100.0
            return nodes, [spurious_bottom, valid_bottom]

        policy = {
            "local_extrema_stability": {
                **self.config["core"]["event_consensus"][
                    "local_extrema_stability"
                ]
            }
        }
        nodes, bottoms = build_nodes()
        records = chain_selection.local_extrema_stability_bridge(
            nodes, bottoms, 100.0, self.chain_config, policy
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame"], 200)
        self.assertEqual(records[0]["role"], "leading")
        self.assertEqual(records[0]["kind"], "leading_partial")
        self.assertFalse(nodes[0]["eligible"])
        self.assertTrue(nodes[1]["eligible"])
        self.assertTrue(
            nodes[1]["top"]["evidence"]["local_extrema_stable"]
        )

        nodes, bottoms = build_nodes()
        bottoms[1]["mean_angle"] = 155.0
        self.assertEqual(
            chain_selection.local_extrema_stability_bridge(
                nodes, bottoms, 100.0, self.chain_config, policy
            ),
            [],
        )
        self.assertFalse(nodes[1]["eligible"])

    def test_low_amplitude_bridge_extends_one_viable_chain(self) -> None:
        nodes = [
            bridge_node(0, 100, eligible=True, relative_amplitude=1.0),
            bridge_node(1, 200, eligible=True, relative_amplitude=1.0),
            bridge_node(2, 300, eligible=True, relative_amplitude=1.0),
            bridge_node(3, 400, eligible=False, relative_amplitude=0.3),
        ]
        policy = {
            "low_amplitude_context_bridge": {
                "enabled": True,
                "minimum_relative_amplitude": 0.2,
                "minimum_top_stability": 0.4,
                "minimum_bottom_stability": 0.2,
                "minimum_scale_support_fraction": 0.5,
                "minimum_bridge_confidence": 0.45,
            }
        }
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        records = chain_selection.low_amplitude_context_bridge(
            nodes,
            edges,
            100.0,
            self.chain_config,
            policy,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(nodes[3]["eligible"])
        self.assertTrue(nodes[3]["low_amplitude_context_bridged"])
        self.assertEqual(nodes[3]["context_bridge_allowed_neighbors"], [0, 1, 2])
        self.assertEqual(records[0]["primary_side"], "left")
        rebuilt = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        self.assertTrue(rebuilt[(2, 3)]["allowed"])
        self.assertEqual(
            [path for path in chain_selection.select_chains(
                nodes, rebuilt, self.chain_config
            )],
            [[0, 1, 2, 3]],
        )

    def test_low_amplitude_bridge_rejects_independent_chain_merge(self) -> None:
        nodes = [
            bridge_node(0, 100, eligible=True, relative_amplitude=1.0),
            bridge_node(1, 200, eligible=True, relative_amplitude=1.0),
            bridge_node(2, 300, eligible=True, relative_amplitude=1.0),
            bridge_node(3, 400, eligible=False, relative_amplitude=0.3),
            bridge_node(4, 500, eligible=True, relative_amplitude=1.0),
            bridge_node(5, 600, eligible=True, relative_amplitude=1.0),
            bridge_node(6, 700, eligible=True, relative_amplitude=1.0),
        ]
        policy = {
            "low_amplitude_context_bridge": {
                "enabled": True,
                "minimum_relative_amplitude": 0.2,
                "minimum_top_stability": 0.4,
                "minimum_bottom_stability": 0.2,
                "minimum_scale_support_fraction": 0.5,
                "minimum_bridge_confidence": 0.45,
            }
        }
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        records = chain_selection.low_amplitude_context_bridge(
            nodes,
            edges,
            100.0,
            self.chain_config,
            policy,
        )
        self.assertEqual(records, [])
        self.assertFalse(nodes[3]["eligible"])
        self.assertNotIn("context_bridge_allowed_neighbors", nodes[3])

    def test_low_amplitude_bridge_rejects_missing_context(self) -> None:
        nodes = [
            bridge_node(0, 100, eligible=False, relative_amplitude=0.3),
        ]
        policy = {
            "low_amplitude_context_bridge": {
                "enabled": True,
                "minimum_relative_amplitude": 0.2,
                "minimum_top_stability": 0.4,
                "minimum_bottom_stability": 0.2,
                "minimum_scale_support_fraction": 0.5,
                "minimum_bridge_confidence": 0.45,
            }
        }
        records = chain_selection.low_amplitude_context_bridge(
            nodes,
            {},
            100.0,
            self.chain_config,
            policy,
        )
        self.assertEqual(records, [])
        self.assertFalse(nodes[0]["eligible"])

    def test_low_amplitude_bridge_is_disabled_when_sampling_is_sufficient(
        self,
    ) -> None:
        nodes = [
            bridge_node(0, 100, eligible=True, relative_amplitude=1.0),
            bridge_node(1, 200, eligible=True, relative_amplitude=1.0),
            bridge_node(2, 300, eligible=True, relative_amplitude=1.0),
            bridge_node(3, 400, eligible=False, relative_amplitude=0.3),
        ]
        policy = {
            "low_amplitude_context_bridge": {
                "enabled": False,
                "minimum_relative_amplitude": 0.2,
                "minimum_top_stability": 0.4,
                "minimum_bottom_stability": 0.2,
                "minimum_scale_support_fraction": 0.5,
                "minimum_bridge_confidence": 0.45,
            }
        }
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        self.assertEqual(
            chain_selection.low_amplitude_context_bridge(
                nodes,
                edges,
                100.0,
                self.chain_config,
                policy,
            ),
            [],
        )
        self.assertFalse(nodes[3]["eligible"])
        self.assertEqual(
            chain_selection.low_amplitude_context_bridge(
                nodes,
                edges,
                100.0,
                self.chain_config,
                None,
            ),
            [],
        )

    def test_low_amplitude_bridge_policy_uses_analysis_fps(self) -> None:
        for fps, expected in ((15.0, True), (60.0, False)):
            consensus = signals_v3.build_multiscale_state_consensus(
                sine_signal(1.2, fps=fps, seconds=8.0),
                fps,
                self.action("pushup"),
                "elbow_flexion_bilateral_mean",
                self.config,
                detector={"smoothing_seconds": 0.12},
            )
            self.assertEqual(
                consensus["chain_policy"]["low_amplitude_context_bridge"][
                    "enabled"
                ],
                expected,
            )

    def test_low_amplitude_bridge_extends_recovered_chain_structure(self) -> None:
        tops = [
            consensus_event(100, scale_support=4),
            consensus_event(200, scale_support=4),
            consensus_event(300, scale_support=4),
            consensus_event(400, scale_support=4),
            consensus_event(500, scale_support=4),
        ]
        tops[0]["kind"] = "leading_partial"
        tops[0]["mean_angle"] = 160.0
        for top in tops[:4]:
            top["mean_angle"] = 160.0
        tops[4]["mean_angle"] = 120.0
        bottoms = []
        for anchor in (150, 250, 350, 450):
            bottom = consensus_event(anchor)
            bottom["mean_angle"] = 100.0
            bottoms.append(bottom)
        policy = {
            "minimum_chain_consensus_events": 2,
            "endpoint_closure_gap_period_range": [0.65, 1.35],
            "low_amplitude_context_bridge": {
                "enabled": True,
                "minimum_relative_amplitude": 0.2,
                "minimum_top_stability": 0.4,
                "minimum_bottom_stability": 0.2,
                "minimum_scale_support_fraction": 0.5,
                "minimum_bridge_confidence": 0.45,
            },
        }
        analysis, _ = chain_selection.analyze_phase_states(
            "bridge_extension",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action("pushup"),
            phase_signal="elbow_flexion_bilateral_mean",
            frame_count=600,
            config=self.chain_config,
            chain_policy=policy,
        )
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 4)
        bridge_records = analysis["diagnostics"][
            "low_amplitude_context_bridged_events"
        ]
        self.assertEqual(len(bridge_records), 1)
        self.assertEqual(bridge_records[0]["frame"], 500)
        self.assertTrue(
            analysis["diagnostics"]["selected_top_events"][-1][
                "low_amplitude_context_bridged"
            ]
        )

    def test_local_period_gap_rescues_adjacent_edge(self) -> None:
        nodes = [
            bridge_node(index, frame, eligible=True, relative_amplitude=1.0)
            for index, frame in enumerate((100, 210, 320, 480, 590, 700))
        ]
        policy = {
            "local_period_gap": {
                "enabled": True,
                "context_intervals_per_side": 2,
                "minimum_context_intervals": 3,
                "maximum_relative_mad": 0.25,
            }
        }
        self.assertIsNone(
            chain_selection.build_edge(
                nodes[2], nodes[3], 100.0, self.chain_config
            )
        )
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        edge = edges[(2, 3)]
        self.assertTrue(edge["allowed"])
        self.assertTrue(edge["local_period_gap"])
        self.assertAlmostEqual(edge["global_gap_ratio"], 1.6)
        self.assertAlmostEqual(edge["local_period_frames"], 110.0)
        self.assertAlmostEqual(
            edge["local_gap_ratio"], 160.0 / 110.0, places=5
        )
        self.assertEqual(edge["local_period_context_intervals"], [110] * 4)
        self.assertEqual(
            [path for path in chain_selection.select_chains(
                nodes, edges, self.chain_config
            )],
            [[0, 1, 2, 3, 4, 5]],
        )

    def test_local_period_gap_rejects_unstable_context(self) -> None:
        frames = (100, 160, 300, 460, 520, 660)
        nodes = [
            bridge_node(index, frame, eligible=True, relative_amplitude=1.0)
            for index, frame in enumerate(frames)
        ]
        policy = {
            "local_period_gap": {
                "enabled": True,
                "context_intervals_per_side": 2,
                "minimum_context_intervals": 3,
                "maximum_relative_mad": 0.25,
            }
        }
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        self.assertNotIn((2, 3), edges)

    def test_local_period_gap_requires_bilateral_context(self) -> None:
        nodes = [
            bridge_node(index, frame, eligible=True, relative_amplitude=1.0)
            for index, frame in enumerate((100, 210, 320, 480))
        ]
        policy = {
            "local_period_gap": {
                "enabled": True,
                "context_intervals_per_side": 2,
                "minimum_context_intervals": 3,
                "maximum_relative_mad": 0.25,
            }
        }
        edges = chain_selection.build_edges(
            nodes, 100.0, self.chain_config, chain_policy=policy
        )
        self.assertNotIn((2, 3), edges)

    def test_local_period_gap_merges_chain_and_writes_diagnostics(self) -> None:
        frames = (100, 210, 320, 480, 590, 700)
        tops = [consensus_event(frame, scale_support=4) for frame in frames]
        for top in tops:
            top["mean_angle"] = 160.0
        bottoms = []
        for left, right in zip(frames[:-1], frames[1:]):
            bottom = consensus_event(int(round((left + right) / 2.0)))
            bottom["mean_angle"] = 100.0
            bottoms.append(bottom)
        policy = {
            "minimum_chain_consensus_events": 2,
            "endpoint_closure_gap_period_range": [0.65, 1.35],
            "local_period_gap": {
                "enabled": True,
                "context_intervals_per_side": 2,
                "minimum_context_intervals": 3,
                "maximum_relative_mad": 0.25,
            },
        }
        without_bridge, _ = chain_selection.analyze_phase_states(
            "local_gap_off",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action("pushup"),
            phase_signal="elbow_flexion_bilateral_mean",
            frame_count=800,
            config=self.chain_config,
            chain_policy={
                "minimum_chain_consensus_events": 2,
                "endpoint_closure_gap_period_range": [0.65, 1.35],
            },
        )
        self.assertEqual(without_bridge["status"], "no_confident_set")
        analysis, _ = chain_selection.analyze_phase_states(
            "local_gap_on",
            tops,
            bottoms,
            period_frames=100.0,
            periodicity=0.95,
            action=self.action("pushup"),
            phase_signal="elbow_flexion_bilateral_mean",
            frame_count=800,
            config=self.chain_config,
            chain_policy=policy,
        )
        self.assertEqual(len(analysis["sets"]), 1)
        self.assertEqual(len(analysis["sets"][0]["reps"]), 5)
        local_edges = analysis["diagnostics"]["local_period_gap_edges"]
        self.assertEqual(len(local_edges), 1)
        self.assertEqual(
            (local_edges[0]["left_frame"], local_edges[0]["right_frame"]),
            (320, 480),
        )
        self.assertTrue(
            any(
                event["local_period_gap"]
                for event in analysis["diagnostics"]["selected_top_events"]
            )
        )

    def test_event_clustering_merges_scale_drift_and_filters_singletons(
        self,
    ) -> None:
        settings = self.config["core"]["event_consensus"]
        results = [
            {
                "smoothing_seconds": 0.12,
                "top_states": [
                    consensus_event(99),
                    consensus_event(250),
                ],
            },
            {
                "smoothing_seconds": 0.2,
                "top_states": [consensus_event(101)],
            },
            {
                "smoothing_seconds": 0.32,
                "top_states": [consensus_event(100)],
            },
        ]
        states = signals_v3._cluster_events_across_scales(
            results, 100.0, settings, state_key="top_states"
        )
        self.assertEqual(len(states), 1)
        self.assertAlmostEqual(states[0]["anchor_frame"], 100.0, delta=1.0)
        self.assertEqual(states[0]["evidence"]["scale_support"], 3)
        self.assertTrue(states[0]["evidence"]["multi_scale_consensus"])

    def test_near_duplicate_prefers_interior_event(self) -> None:
        settings = self.config["core"]["event_consensus"]
        interior = consensus_event(100, scale_support=2)
        endpoint = consensus_event(
            110,
            kind="trailing_partial",
            stable_fraction=1.0,
            scale_support=2,
        )
        states = signals_v3._suppress_near_duplicate_events(
            [endpoint, interior], 100.0, settings
        )
        self.assertEqual([state["anchor_frame"] for state in states], [100])
        self.assertEqual(
            states[0]["evidence"]["near_duplicate_suppressed_frames"], [110]
        )

    def test_selected_scale_anchor_is_preserved_for_closure(self) -> None:
        settings = self.config["core"]["event_consensus"]
        results = [
            {
                "smoothing_seconds": 0.12,
                "top_states": [consensus_event(99), consensus_event(250)],
            },
            {
                "smoothing_seconds": 0.2,
                "top_states": [consensus_event(101)],
            },
            {
                "smoothing_seconds": 0.32,
                "top_states": [consensus_event(100)],
            },
        ]
        states = signals_v3._cluster_events_across_scales(
            results,
            100.0,
            settings,
            state_key="top_states",
            selected_scale_seconds=0.12,
        )
        preserved = next(
            state for state in states if state["anchor_frame"] == 250
        )
        self.assertEqual(preserved["evidence"]["scale_support"], 1)
        self.assertTrue(preserved["evidence"]["selected_scale_anchor"])

    def test_multiscale_consensus_and_chain_override_are_wired(self) -> None:
        fps = 30.0
        values = sine_signal(1.0, fps=fps, seconds=8.0)
        signals = {
            "left_elbow_flexion": values.copy(),
            "right_elbow_flexion": values.copy(),
        }
        resolution = signals_v3.analyze_phase_signals(
            signals,
            fps,
            self.action("pushup"),
            self.config,
            detector={"smoothing_seconds": 0.12},
        )
        selected = resolution["selected_phase_signal"]
        consensus = signals_v3.build_multiscale_state_consensus(
            resolution["signals"][selected],
            resolution["analysis_fps"],
            self.action("pushup"),
            selected,
            self.config,
            detector=resolution["detector"],
        )
        self.assertEqual(consensus["status"], "ok")
        self.assertGreaterEqual(len(consensus["top_states"]), 3)
        self.assertIn("scale_members", consensus["evidence"])
        analysis, meta = chain_selection.analyze_clip_signals(
            {"clip_id": "clip_a", "fps": resolution["analysis_fps"]},
            resolution["signals"],
            self.action("pushup"),
            prior_signal=None,
            phase_policy="t1_6_bilateral_mean",
            chain_config=self.chain_config,
            phase_resolution=resolution,
            phase_state_override=consensus,
        )
        self.assertEqual(analysis["status"], "ok")
        self.assertEqual(
            analysis["diagnostics"]["chain_source"],
            "phase_state_multiscale_consensus",
        )
        self.assertIn("event_consensus_evidence", analysis["diagnostics"])
        self.assertEqual(meta["phase_state_source"], "multiscale_consensus")

    def test_analysis_arbitration_preserves_set_boundaries(self) -> None:
        def fake_analysis(set_count: int, reps_per_set: int) -> dict:
            return {
                "sets": [
                    {
                        "reps": [{}] * reps_per_set,
                        "chain_confidence": 0.8,
                        "is_default": index == 0,
                    }
                    for index in range(set_count)
                ],
                "diagnostics": {"selected_top_events": []},
            }

        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [{"anchor_frame": 100}, {"anchor_frame": 200}],
            "evidence": {},
        }
        single = fake_analysis(2, 3)
        consensus = fake_analysis(1, 7)
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            single,
        )

    def test_analysis_arbitration_accepts_material_consensus_gain(self) -> None:
        def fake_analysis(
            set_count: int,
            reps_per_set: int,
            event_frames: list[int],
        ) -> dict:
            return {
                "sets": [
                    {
                        "reps": [{}] * reps_per_set,
                        "chain_confidence": 0.8,
                        "is_default": index == 0,
                    }
                    for index in range(set_count)
                ],
                "diagnostics": {
                    "selected_top_events": [
                        {"frame": frame, "graph_frame": frame}
                        for frame in event_frames
                    ]
                },
            }

        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [{"anchor_frame": 100}, {"anchor_frame": 200}],
            "evidence": {},
        }
        single = fake_analysis(1, 3, [100, 200, 300])
        consensus = fake_analysis(
            1, 6, [100, 200, 300, 400, 500, 600]
        )
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_rejects_disjoint_material_consensus_gain(
        self,
    ) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (100, 200, 300)
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}, {}],
                    "chain_confidence": 0.88,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (400, 500, 600, 700, 800)
                ]
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": frame, "kind": "interior", "evidence": {}}
                for frame in (100, 200, 300, 400, 500, 600, 700, 800)
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            single,
        )

    def test_analysis_arbitration_accepts_gain_within_single_event_span(
        self,
    ) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (250, 300, 400)
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}, {}, {}],
                    "chain_confidence": 0.88,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (150, 250, 350, 450, 550, 650)
                ]
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": frame, "kind": "interior", "evidence": {}}
                for frame in (150, 250, 350, 450, 550, 650)
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_accepts_single_endpoint_rep_gain(self) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": 100, "graph_frame": 100},
                    {"frame": 200, "graph_frame": 200},
                    {"frame": 300, "graph_frame": 300},
                    {"frame": 400, "graph_frame": 400},
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}],
                    "chain_confidence": 0.86,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (50, 100, 200, 300, 400)
                ],
                "endpoint_closure_edges": [{"left_index": 0, "right_index": 1}],
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": 50, "kind": "leading_partial", "evidence": {}},
                {"anchor_frame": 100, "kind": "interior", "evidence": {}},
                {"anchor_frame": 200, "kind": "interior", "evidence": {}},
                {"anchor_frame": 300, "kind": "interior", "evidence": {}},
                {"anchor_frame": 400, "kind": "interior", "evidence": {}},
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_accepts_local_extrema_peak_gain(self) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": 200, "graph_frame": 200},
                    {"frame": 300, "graph_frame": 300},
                    {"frame": 400, "graph_frame": 400},
                    {"frame": 500, "graph_frame": 500},
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}],
                    "chain_confidence": 0.86,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "endpoint_closure_edges": [{"left_index": 0, "right_index": 1}],
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (100, 200, 300, 400, 500)
                ],
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {
                    "anchor_frame": 100,
                    "kind": "leading_partial",
                    "evidence": {
                        "local_extrema_stable": True,
                        "scale_support": 4,
                        "multi_scale_consensus": True,
                    },
                },
                *[
                    {"anchor_frame": frame, "kind": "interior", "evidence": {}}
                    for frame in (200, 300, 400, 500)
                ],
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_rejects_unexplained_single_rep_gain(self) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": 100, "graph_frame": 100},
                    {"frame": 200, "graph_frame": 200},
                    {"frame": 300, "graph_frame": 300},
                    {"frame": 400, "graph_frame": 400},
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}],
                    "chain_confidence": 0.89,
                    "is_default": True,
                }
            ],
            "diagnostics": {"endpoint_closure_edges": []},
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": 50, "kind": "interior", "evidence": {}},
                {"anchor_frame": 100, "kind": "interior", "evidence": {}},
                {"anchor_frame": 200, "kind": "interior", "evidence": {}},
                {"anchor_frame": 300, "kind": "interior", "evidence": {}},
                {"anchor_frame": 400, "kind": "interior", "evidence": {}},
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            single,
        )

    def test_analysis_arbitration_accepts_near_boundary_recovery_gain(
        self,
    ) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (100, 200, 300, 400, 500, 600)
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}, {}, {}],
                    "chain_confidence": 0.88,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (40, 100, 200, 300, 400, 500, 600)
                ]
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {
                    "anchor_frame": 40,
                    "kind": "interior",
                    "evidence": {
                        "near_boundary_rejected_top": True,
                        "scale_support": 4,
                    },
                },
                *[
                    {"anchor_frame": frame, "kind": "interior", "evidence": {}}
                    for frame in (100, 200, 300, 400, 500, 600)
                ],
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_rejects_unexplained_endpoint_span_cleanup(
        self,
    ) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": 50, "graph_frame": 50},
                    {"frame": 150, "graph_frame": 150},
                    {"frame": 250, "graph_frame": 250},
                    {"frame": 350, "graph_frame": 350},
                    {"frame": 450, "graph_frame": 450},
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.88,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": 150, "graph_frame": 150},
                    {"frame": 250, "graph_frame": 250},
                    {"frame": 350, "graph_frame": 350},
                    {"frame": 450, "graph_frame": 450},
                ],
                "endpoint_closure_edges": [],
                "low_sampling_endpoint_pruned_events": [],
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": 150, "kind": "interior", "evidence": {}},
                {"anchor_frame": 250, "kind": "interior", "evidence": {}},
                {"anchor_frame": 350, "kind": "interior", "evidence": {}},
                {"anchor_frame": 450, "kind": "interior", "evidence": {}},
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            single,
        )

    def test_analysis_arbitration_accepts_explicit_endpoint_cleanup(
        self,
    ) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (50, 150, 250, 350, 450)
                ]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.88,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [
                    {"frame": frame, "graph_frame": frame}
                    for frame in (150, 250, 350, 450)
                ],
                "low_sampling_endpoint_pruned_events": [
                    {
                        "role": "leading",
                        "frame": 50,
                        "kind": "leading_partial",
                        "stable_fraction": 0.2,
                        "scale_support": 4,
                    }
                ],
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [
                {"anchor_frame": frame, "kind": "interior", "evidence": {}}
                for frame in (150, 250, 350, 450)
            ],
            "evidence": {},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            consensus,
        )

    def test_analysis_arbitration_rejects_unexplained_event_loss(self) -> None:
        single = {
            "sets": [
                {
                    "reps": [{}, {}, {}],
                    "chain_confidence": 0.8,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "selected_top_events": [{"frame": 100, "graph_frame": 100}]
            },
        }
        consensus = {
            "sets": [
                {
                    "reps": [{}, {}],
                    "chain_confidence": 0.9,
                    "is_default": True,
                }
            ],
            "diagnostics": {
                "endpoint_closure_edges": [{"left_index": 0, "right_index": 1}]
            },
        }
        consensus_state = {
            "status": "ok",
            "period_frames": 100.0,
            "top_states": [{"anchor_frame": 250}],
            "evidence": {"near_duplicate_top_count": 1},
        }
        self.assertIs(
            signals_v3.select_analysis_hypothesis(
                single,
                consensus,
                consensus_state,
                self.config["core"]["event_consensus"],
            ),
            single,
        )

    def test_ambiguous_resolution_short_circuits_chain(self) -> None:
        resolution = {
            "status": "ambiguous",
            "selected_phase_signal": "knee_flexion_bilateral_mean",
            "detector": {"smoothing_seconds": 0.2},
            "evidence": {"status": "ambiguous"},
        }
        analysis, _ = chain_selection.analyze_clip_signals(
            {"clip_id": "clip_a", "fps": 30.0},
            {"signal": np.zeros(60, dtype=np.float64)},
            self.action("squat"),
            prior_signal=None,
            phase_policy="t1_6_bilateral_mean",
            chain_config=self.chain_config,
            phase_resolution=resolution,
        )
        self.assertEqual(analysis["status"], "phase_ambiguous")
        self.assertEqual(analysis["sets"], [])


if __name__ == "__main__":
    unittest.main()
