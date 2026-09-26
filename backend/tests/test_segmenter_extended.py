"""ActionSegmenter 覆盖盲区补测：长小节拆分、节拍/重音提取、余弦相似度边界。"""

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from app.pipeline.segmenter import ActionSegmenter
from tests.helpers import make_keypoints


class TestActionSegmenterExtended(unittest.TestCase):
    def test_split_long_bars_inserts_interior_energy_minimum(self) -> None:
        """超长动作段应在内部能量局部极小值处补切分，并吸附到节拍帧。"""
        seg = ActionSegmenter()
        energy = np.ones(300, dtype=np.float64)
        energy[150] = -5.0  # 内部明显的「停顿/换动作」低能量点
        beats = [0, 50, 100, 150, 200, 250, 300]

        result = seg._split_long_bars([0, 300], energy, beats, max_frames=240)

        self.assertIn(150, result)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 300)
        # 拆分后每一段都不再超过 max_frames
        for a, b in zip(result, result[1:]):
            self.assertLessEqual(b - a, 240)

    def test_split_long_bars_without_interior_beat_keeps_original(self) -> None:
        """长小节内无可用节拍可吸附时，保持原边界不变（不产生非法切分）。"""
        seg = ActionSegmenter()
        energy = np.ones(300, dtype=np.float64)
        energy[150] = -5.0
        # 只有首尾两个节拍，snap 后不满足 s < snapped < e，无法补切分
        result = seg._split_long_bars([0, 300], energy, [0, 300], max_frames=240)
        self.assertEqual(result, [0, 300])

    def test_detect_beats_returns_ordered_frames(self) -> None:
        """librosa 节拍提取：整数帧号、有序、去重、落在视频范围内。"""
        seg = ActionSegmenter()
        audio_path = self._write_click_track()

        fps = 30
        beats = seg.detect_beats(audio_path, fps)

        self.assertGreaterEqual(len(beats), 2)
        self.assertEqual(beats, sorted(beats))
        self.assertEqual(len(beats), len(set(beats)))
        for b in beats:
            self.assertEqual(b, int(b))
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, round(2.0 * fps))

    def test_detect_onsets_returns_ordered_frames(self) -> None:
        """librosa onset 提取：整数帧号、有序、去重、落在视频范围内。"""
        seg = ActionSegmenter()
        audio_path = self._write_click_track()

        onsets = seg.detect_onsets(audio_path, 30)

        self.assertGreaterEqual(len(onsets), 1)
        self.assertEqual(onsets, sorted(onsets))
        self.assertEqual(len(onsets), len(set(onsets)))
        for o in onsets:
            self.assertEqual(o, int(o))
            self.assertGreaterEqual(o, 0)
            self.assertLessEqual(o, round(2.0 * 30))

    def test_cosine_similarity_zero_vector_returns_zero(self) -> None:
        """零向量签名做余弦相似度不应产生 NaN，返回 0。"""
        seg = ActionSegmenter()
        sim = seg._cosine_similarity(np.zeros(24), np.zeros(24))
        self.assertEqual(sim, 0.0)

    def test_compute_energy_single_frame_returns_zeros(self) -> None:
        """单帧序列能量数组长度为 1 且为 0（不越界）。"""
        kps = make_keypoints(1)
        energy = ActionSegmenter().compute_energy(kps)
        self.assertEqual(energy.shape, (1,))
        self.assertEqual(float(energy[0]), 0.0)

    @staticmethod
    def _write_click_track() -> str:
        """生成 2 秒 120 BPM 的合成节拍点击音轨（WAV），返回临时路径。"""
        sr = 22050
        duration = 2.0
        n = int(sr * duration)
        x = np.zeros(n, dtype=np.float32)
        for k in range(4):
            i = int(k * 0.5 * sr)
            x[i : i + 60] = 0.9
        path = os.path.join(tempfile.gettempdir(), "dance_test_click.wav")
        sf.write(path, x, sr)
        return path


if __name__ == "__main__":
    unittest.main()
