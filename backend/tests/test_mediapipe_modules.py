"""mediapipe 相关模块（person_segmenter / pose_estimator）测试。

mediapipe 1.0+ 使用 Tasks API 且需显式提供模型文件。本测试：
- 校验模块可 import（惰性导入）；
- 用 mock 结果对象验证关键点提取 / 前景 mask 逻辑（不依赖模型推理）；
- 真实 smoke：selfie_segmenter.tflite + pose_landmarker_lite.task 加载并对真实
  人像图跑一次推理（模型文件在 backend/models/，测试图在 tests/data/pose.jpg）。
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.config import POSE_LANDMARKER_MODEL, SELFIE_SEGMENTATION_MODEL
from app.models.schemas import VideoMeta

_FIXTURE = Path(__file__).parent / "data" / "pose.jpg"


class _Landmark:
    def __init__(self, x: float, y: float, z: float, vis: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = vis


class _PoseResult:
    def __init__(self, landmarks: list[_Landmark], world: list[_Landmark] | None) -> None:
        self.pose_landmarks = [landmarks]
        self.pose_world_landmarks = [world] if world else None


class _SegmentResult:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.confidence_masks = masks


class TestMediaPipeModules(unittest.TestCase):
    def test_modules_importable(self) -> None:
        import app.pipeline.person_segmenter  # noqa: F401
        import app.pipeline.pose_estimator  # noqa: F401

    # ------------------------------------------------------------------
    # pose_estimator：mock 逻辑测试（不依赖模型推理）
    # ------------------------------------------------------------------
    def test_pose_extract_landmarks(self) -> None:
        from app.pipeline.pose_estimator import PoseEstimator

        est = PoseEstimator.__new__(PoseEstimator)  # 跳过 __init__（无需模型）
        landmarks = [_Landmark(0.5, 0.25, -0.1, 0.99) for _ in range(33)]
        world = [_Landmark(0.0, 0.0, 1.5, 1.0) for _ in range(33)]
        result = _PoseResult(landmarks, world)

        out = est._extract_landmarks(result, w=640, h=480)
        self.assertEqual(out.shape, (33, 4))
        self.assertAlmostEqual(out[0, 0], 0.5 * 640)
        self.assertAlmostEqual(out[0, 1], 0.25 * 480)
        self.assertAlmostEqual(out[0, 2], 1.5)
        self.assertAlmostEqual(out[0, 3], 0.99)

    def test_pose_extract_empty_result(self) -> None:
        from app.pipeline.pose_estimator import PoseEstimator

        est = PoseEstimator.__new__(PoseEstimator)
        out = est._extract_landmarks(_PoseResult([], None), w=640, h=480)
        self.assertEqual(out.shape, (33, 4))
        np.testing.assert_allclose(out, 0.0)

    def test_pose_estimator_model_load(self) -> None:
        """pose_landmarker_lite.task 应能成功加载（修复后的验证）。"""
        from app.pipeline.pose_estimator import PoseEstimator

        if not POSE_LANDMARKER_MODEL.exists():
            with self.assertRaises(FileNotFoundError):
                PoseEstimator()
            return
        est = PoseEstimator()
        try:
            self.assertIsNotNone(est)
        finally:
            est.close()

    def test_pose_estimator_real_inference(self) -> None:
        """对真实人像图跑一次推理，确认返回 33 个 landmarks。"""
        from app.pipeline.pose_estimator import PoseEstimator

        if not _FIXTURE.exists() or not POSE_LANDMARKER_MODEL.exists():
            self.skipTest("缺少模型文件或测试图片 fixtures")
        est = PoseEstimator()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frames = Path(tmp)
                cv2.imwrite(str(frames / "f_000001.jpg"), cv2.imread(str(_FIXTURE)))
                kps = est.estimate(
                    frames, VideoMeta(video_id="test", total_frames=1)
                )
                self.assertEqual(kps.data.shape, (1, 33, 4))
                nz = int((kps.data[0, :, 0:2] > 0).sum())
                self.assertGreater(nz, 20, "真实推理应检测到多数关键点")
        finally:
            est.close()

    # ------------------------------------------------------------------
    # person_segmenter：mock + 真实 smoke
    # ------------------------------------------------------------------
    def test_foreground_mask_threshold(self) -> None:
        from app.pipeline.person_segmenter import PersonSegmenter

        seg = PersonSegmenter.__new__(PersonSegmenter)
        seg.confidence_threshold = 0.5
        conf = np.array([[0.1, 0.9], [0.6, 0.4]], dtype=np.float32)
        mask = seg._foreground_mask(_SegmentResult([conf]))
        np.testing.assert_array_equal(mask, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    def test_person_segmenter_availability(self) -> None:
        from app.pipeline.person_segmenter import PersonSegmenter

        if not SELFIE_SEGMENTATION_MODEL.exists():
            with self.assertRaises(FileNotFoundError):
                PersonSegmenter()
            return
        seg = PersonSegmenter()
        try:
            self.assertIsNotNone(seg)
        finally:
            seg.close()

    def test_person_segmenter_real_segmentation(self) -> None:
        """对真实人像图跑一次分割，输出帧号不变且背景被处理。"""
        from app.pipeline.person_segmenter import PersonSegmenter

        if not _FIXTURE.exists() or not SELFIE_SEGMENTATION_MODEL.exists():
            self.skipTest("缺少模型文件或测试图片 fixtures")
        seg = PersonSegmenter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                frames = tmp / "frames"
                masked_dir = tmp / "masked"
                frames.mkdir()
                cv2.imwrite(str(frames / "f_000001.jpg"), cv2.imread(str(_FIXTURE)))
                n = seg.segment(frames, masked_dir)
                self.assertEqual(n, 1)
                self.assertTrue((masked_dir / "f_000001.jpg").exists())
        finally:
            seg.close()


if __name__ == "__main__":
    unittest.main()
