"""ActionSegmenter 单元测试（节拍切分 + 能量兜底 + snap + 副歌重复段）。"""

import unittest

import numpy as np

from app.pipeline.segmenter import ActionSegmenter
from tests.helpers import make_keypoints


class TestActionSegmenter(unittest.TestCase):
    def test_snap_to_beat(self) -> None:
        seg = ActionSegmenter()
        self.assertEqual(seg.snap_to_beat(42, [0, 10, 20, 30, 40, 50]), 40)
        self.assertEqual(seg.snap_to_beat(47, [0, 10, 20, 30, 40, 50]), 50)
        self.assertEqual(seg.snap_to_beat(5, []), 5)

    def test_compute_energy_length(self) -> None:
        kps = make_keypoints(50)
        seg = ActionSegmenter()
        energy = seg.compute_energy(kps)
        self.assertEqual(energy.shape, (50,))

    def test_segment_with_beats_is_seamless(self) -> None:
        """节拍切分：边界落在节拍帧上，且无缝覆盖全片。"""
        kps = make_keypoints(64)
        beats = [0, 8, 16, 24, 32, 40, 48, 56]
        seg = ActionSegmenter(beats_per_action=4)
        actions = seg.segment(kps, fps=30, beats=beats)

        self.assertGreaterEqual(len(actions), 1)
        # 无缝覆盖
        self.assertEqual(actions[0].start_frame, 0)
        self.assertEqual(actions[-1].end_frame, kps.total_frames - 1)
        for a, b in zip(actions, actions[1:]):
            self.assertEqual(a.end_frame, b.start_frame)
            # 内部边界吸附到节拍帧
            self.assertIn(a.end_frame, beats)

    def test_segment_fallback_without_beats(self) -> None:
        """无节拍时退化为能量峰切分，仍无缝覆盖。"""
        kps = make_keypoints(60)
        # 制造两处明显停顿（低能量）以获得多于 1 个边界
        for t in range(60):
            if 20 <= t < 25 or 45 <= t < 50:
                kps.data[t, :, 0:2] *= 0.05
        actions = ActionSegmenter().segment(kps, fps=30, beats=None)
        self.assertGreaterEqual(len(actions), 1)
        self.assertEqual(actions[0].start_frame, 0)
        self.assertEqual(actions[-1].end_frame, 59)
        for a, b in zip(actions, actions[1:]):
            self.assertEqual(a.end_frame, b.start_frame)

    def test_detect_repeated_segments(self) -> None:
        """副歌重复段检测：两段相同姿态的小节被识别，并标注 killing_point。"""
        kps = make_keypoints(30)
        # 第 0、2 小节：双手上举（姿态 A）；第 1 小节：双手下垂（姿态 B）
        for t in range(30):
            if (0 <= t < 10) or (20 <= t < 30):
                kps.data[t, 15, 1] = 90.0
                kps.data[t, 16, 1] = 90.0
            else:
                kps.data[t, 15, 1] = 330.0
                kps.data[t, 16, 1] = 330.0

        seg = ActionSegmenter(beats_per_action=1, min_action_seconds=0.2)
        beats = [0, 10, 20, 30]
        repeats = seg.detect_repeated_segments(kps, fps=30, beats=beats)
        self.assertEqual(len(repeats), 2)
        self.assertIn((0, 10), repeats)
        self.assertIn((20, 30), repeats)

        actions = seg.segment(kps, fps=30, beats=beats)
        # 第 0 与第 2 个动作落在重复段 → 有 killing_point；第 1 个动作没有
        self.assertNotEqual(actions[0].annotations.killing_point, "")
        self.assertEqual(actions[1].annotations.killing_point, "")
        self.assertNotEqual(actions[2].annotations.killing_point, "")


if __name__ == "__main__":
    unittest.main()
