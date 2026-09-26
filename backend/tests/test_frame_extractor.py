"""FrameExtractor 单元测试（真实合成视频抽取）。"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.frame_extractor import FrameExtractor


def _write_synthetic_video(path: Path, num_frames: int, fps: int, size=(64, 64)) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    assert writer.isOpened(), "VideoWriter 打开失败"
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestFrameExtractor(unittest.TestCase):
    def test_extract_resamples_and_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "sample.avi"
            _write_synthetic_video(video_path, num_frames=10, fps=10)

            out_dir = tmp / "frames"
            meta = FrameExtractor(target_fps=5).extract(video_path, out_dir)

            # 10 帧 @10fps = 1 秒 → 目标 5fps 应为 5 帧
            self.assertEqual(meta.fps, 5)
            self.assertEqual(meta.total_frames, 5)
            # 文件名保序连续
            names = sorted(p.name for p in out_dir.glob("f_*.jpg"))
            self.assertEqual(names, [f"f_{i:06d}.jpg" for i in range(1, 6)])


if __name__ == "__main__":
    unittest.main()
