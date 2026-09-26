"""SequenceSmoother 单元测试（真实 Savitzky-Golay 平滑验证）。"""

import unittest

import numpy as np

from app.pipeline.smoother import SequenceSmoother
from tests.helpers import make_keypoints


class TestSequenceSmoother(unittest.TestCase):
    def test_smooth_reduces_noise_and_preserves_shape(self) -> None:
        total = 60
        kps = make_keypoints(total)

        # 左腕 x 坐标叠加平滑正弦 + 高斯噪声
        t = np.arange(total)
        clean = 220.0 + 30.0 * np.sin(2 * np.pi * t / 20.0)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0, 8.0, size=total)
        kps.data[:, 15, 0] = noisy.astype(np.float32)

        smoother = SequenceSmoother(window=11, polyorder=3)
        out = smoother.smooth(kps)

        # 帧数/关键点数不变（统一时间轴铁律：不丢帧）
        self.assertEqual(out.total_frames, total)
        self.assertEqual(out.data.shape, (total, 33, 4))

        smoothed = out.data[:, 15, 0].astype(np.float64)
        err_noisy = float(np.std(noisy - clean))
        err_smoothed = float(np.std(smoothed - clean))
        self.assertLess(err_smoothed, err_noisy, "平滑后应更接近真实信号")

    def test_short_sequence_unchanged(self) -> None:
        kps = make_keypoints(2)
        smoother = SequenceSmoother()
        out = smoother.smooth(kps)
        self.assertEqual(out.data.shape, (2, 33, 4))
        np.testing.assert_allclose(out.data, kps.data, rtol=1e-6, atol=1e-6)

    def test_visibility_unchanged(self) -> None:
        kps = make_keypoints(40)
        kps.data[:, 15, 3] = 0.7
        out = SequenceSmoother().smooth(kps)
        np.testing.assert_allclose(out.data[:, :, 3], kps.data[:, :, 3])


if __name__ == "__main__":
    unittest.main()
