from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_clip_report


def make_bundle(exercise: str = "squat") -> dict:
    reps = []
    for rep_id in range(1, 4):
        start = 300 + (rep_id - 1) * 60
        bottom = start + 30
        reps.append(
            {
                "rep_id": rep_id,
                "start_frame": start,
                "end_frame": start + 59,
                "bottom_frame": bottom,
                "start_time_s": start / 30.0,
                "end_time_s": (start + 59) / 30.0,
                "bottom_time_s": bottom / 30.0,
                "duration_s": 2.0,
                "bottom_angle_deg": 95.0,
                "confidence": 0.9,
            }
        )
    active_issue = {
        "issue_id": "squat_knee_valgus_01" if exercise == "squat" else "pushup_asymmetry_01",
        "issue_code": "squat_knee_valgus" if exercise == "squat" else "pushup_asymmetry",
        "exercise": exercise,
        "scope": "rep",
        "rep_ids": [1, 2],
        "representative_rep_id": 1,
        "frequency": "2/3",
        "frequency_ratio": 2 / 3,
        "severity": "medium",
        "confidence": 0.9,
        "anchor_frame": 330,
        "anchor_time_s": 11.0,
        "evidence_interval": [320, 340],
        "measurements": {
            "primary_metric": "knee_valgus_bottom" if exercise == "squat" else "elbow_asymmetry_bottom",
            "primary_value": 25.0,
            "unit": "deg",
            "thresholds": {"low": 12.0, "medium": 20.0, "high": 30.0, "unit": "deg"},
        },
        "visual_overlay": {
            "view": "front",
            "highlight_joints": ["left_knee", "right_knee", "left_ankle", "right_ankle"]
            if exercise == "squat"
            else ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
            "draw": ["skeleton", "angle_arc"],
            "camera_zoom": 1.2,
        },
        "text": {
            "title": "膝内扣" if exercise == "squat" else "左右发力不对称",
            "detail": "第 1–2 个 rep 出现偏差 <测试>。",
            "cue": "提示 <保持稳定>。",
        },
        "suppressed": False,
        "suppression_reason": None,
        "show_by_default": True,
    }
    suppressed_issue = {
        "issue_id": "squat_depth_insufficient_suppressed",
        "issue_code": "squat_depth_insufficient",
        "exercise": "squat",
        "scope": "not_evaluated",
        "rep_ids": [],
        "representative_rep_id": None,
        "frequency": "0/3",
        "frequency_ratio": 0.0,
        "severity": "none",
        "confidence": None,
        "anchor_frame": None,
        "anchor_time_s": None,
        "evidence_interval": None,
        "measurements": {"metric_statuses": {"knee_flexion_rom": ["disabled"]}},
        "visual_overlay": {},
        "text": {"title": "下蹲深度不足（未评估）", "detail": "视角不支持。", "cue": "补录侧视。"},
        "suppressed": True,
        "suppression_reason": "view_unsupported",
        "show_by_default": False,
    }
    return {
        "clip_id": f"synthetic_{exercise}",
        "issue_output": {
            "clip_id": f"synthetic_{exercise}",
            "issues": [active_issue, suppressed_issue],
            "summary": {"active_issue_count": 1, "suppressed_issue_count": 1},
        },
        "analysis": {
            "clip_id": f"synthetic_{exercise}",
            "exercise": exercise,
            "view": "front",
            "split": "dev",
            "fps": 30.0,
            "trim": {"start_frame": 300, "end_frame": 480},
            "quality_status": "pass",
            "quality": {
                "status": "pass",
                "warnings": [],
                "failures": [],
                "view_support": {"disabled_metrics": ["knee_flexion_rom"]},
                "checks": {"camera_motion": {"status": "not_evaluated"}},
            },
            "segmentation": {"reps": reps},
        },
        "clip": {},
        "features": [],
        "wide": {},
        "feedback_dir": Path("/tmp/feedback"),
        "analysis_dir": Path("/tmp/analysis"),
        "video_path": Path("/tmp/video.mp4"),
    }


class TimelineTests(unittest.TestCase):
    def test_clip_relative_time_uses_trim_origin(self) -> None:
        analysis = make_bundle()["analysis"]
        self.assertAlmostEqual(render_clip_report.clip_relative_s(11.0, analysis), 1.0)
        self.assertAlmostEqual(render_clip_report.clip_relative_s(9.0, analysis), 0.0)

    def test_timeline_position_is_relative_and_clamped(self) -> None:
        bundle = make_bundle()
        analysis = bundle["analysis"]
        origin = render_clip_report.clip_origin_s(analysis)
        duration = render_clip_report.clip_duration_s(analysis)
        self.assertAlmostEqual(render_clip_report.timeline_position(origin - 1.0, analysis), 0.0)
        self.assertAlmostEqual(render_clip_report.timeline_position(origin, analysis), 0.0)
        self.assertAlmostEqual(
            render_clip_report.timeline_position(origin + duration / 2.0, analysis), 50.0
        )
        self.assertAlmostEqual(render_clip_report.timeline_position(99.0, analysis), 100.0)

    def test_timeline_html_contains_rep_bottom_and_issue_bands(self) -> None:
        bundle = make_bundle()
        active = [
            issue
            for issue in bundle["issue_output"]["issues"]
            if not issue["suppressed"]
        ]
        page = render_clip_report.timeline_html(bundle, active)
        self.assertIn("rep-bar", page)
        self.assertIn("bottom-marker", page)
        self.assertIn("issue-band", page)
        self.assertIn("squat_knee_valgus_01", page)


class FrameRenderingTests(unittest.TestCase):
    def test_skeleton_and_text_overlay_change_frame(self) -> None:
        frame = np.full((640, 480, 3), 240, dtype=np.uint8)
        landmarks = []
        for index in range(33):
            x = 0.30 + (index % 6) * 0.08
            y = 0.12 + (index // 6) * 0.13
            landmarks.append({"x": x, "y": y, "z": 0.0, "visibility": 0.95})
        issue = make_bundle()["issue_output"]["issues"][0]
        output, detected = render_clip_report.render_issue_frame(
            frame, landmarks, issue, {"left_elbow_flexion": "90", "right_elbow_flexion": "80"}
        )
        self.assertTrue(detected)
        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(np.abs(output.astype(int) - frame.astype(int)).mean(), 0.5)

    def test_missing_pose_still_renders_textual_fallback(self) -> None:
        frame = np.full((320, 240, 3), 240, dtype=np.uint8)
        issue = make_bundle()["issue_output"]["issues"][0]
        output, detected = render_clip_report.render_issue_frame(frame, None, issue, {})
        self.assertFalse(detected)
        self.assertGreater(np.abs(output.astype(int) - frame.astype(int)).mean(), 0.1)

    def test_build_web_video_trims_clip_and_encodes_vp8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
            )
            self.assertTrue(writer.isOpened())
            for index in range(90):
                writer.write(np.full((240, 320, 3), index % 255, dtype=np.uint8))
            writer.release()

            bundle = make_bundle()
            bundle["video_path"] = source
            bundle["analysis"]["trim"] = {"start_frame": 10, "end_frame": 29}
            target = render_clip_report.build_web_video(bundle, root / "report")
            self.assertEqual(target.suffix, ".webm")

            cap = cv2.VideoCapture(str(target))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30.0, places=2)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 240)
            cap.release()


class TextLayoutTests(unittest.TestCase):
    def test_wrap_text_keeps_cjk_punctuation_with_previous_line(self) -> None:
        font = render_clip_report.load_font(18)
        text = "提示：保持两手位置稳定并让两侧同时推起；若差异持续，建议降低难度。"
        lines = render_clip_report.wrap_text(text, font, 220)
        punctuation = set("，。！？；：、）》】")
        self.assertEqual("".join(lines), text)
        self.assertFalse(any(line and all(char in punctuation for char in line) for line in lines))
        self.assertFalse(any(line and line[0] in punctuation for line in lines))


class HtmlTests(unittest.TestCase):
    def test_card_html_escapes_text_and_passes_relative_seek_times(self) -> None:
        issue = make_bundle()["issue_output"]["issues"][0]
        card = render_clip_report.card_html(
            issue, "annotated_frames/test.png", True, 30.0, 10.0
        )
        self.assertIn("&lt;测试&gt;", card)
        self.assertIn("selectIssue('squat_knee_valgus_01', 1.000000, 0.666667, 1.333333)", card)
        self.assertIn("annotated_frames/test.png", card)
        self.assertIn("默认展示", card)

    def test_report_contains_player_timeline_and_suppressed_items(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            page = render_clip_report.build_report_html(
                bundle,
                json.loads((ROOT / "configs" / "task1_4_feedback_templates.json").read_text()),
                {"squat_knee_valgus_01": "annotated_frames/test.png"},
                "video.mp4",
                "source.mp4",
                output_dir,
            )
        self.assertIn('<video id="player"', page)
        self.assertIn("player.currentTime", page)
        self.assertIn("seekRelative", page)
        self.assertIn("addEventListener('seeked'", page)
        self.assertIn("证据区间回放结束", page)
        self.assertIn("原始长视频（MP4V，仅下载）", page)
        self.assertIn("时间线与证据区间", page)
        self.assertIn("当前视角不支持", page)
        self.assertIn("issues.json", page)

    def test_index_links_each_report(self) -> None:
        summary = {
            "generated_at": "2026-10-25T00:00:00+00:00",
            "clip_count": 1,
            "frame_count": 2,
            "results": [
                {
                    "clip_id": "synthetic_squat",
                    "exercise": "squat",
                    "rep_count": 3,
                    "active_issue_count": 1,
                    "severity_counts": {"high": 0, "medium": 1, "low": 0},
                    "frame_count": 2,
                    "quality_status": "pass",
                }
            ],
        }
        page = render_clip_report.build_index_html(summary)
        self.assertIn('href="synthetic_squat/report.html"', page)
        self.assertIn("动作反馈报告索引", page)


if __name__ == "__main__":
    unittest.main()
