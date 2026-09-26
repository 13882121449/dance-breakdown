"""CalibrationService 单元测试：切分点调整 + 关键点修正 + 局部重定向。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.models.schemas import (
    ActionMetadata,
    ActionSegment,
    CalibrationData,
    KeypointCorrection,
    SegmentAdjustment,
)
from app.pipeline.retargeter import Retargeter
from app.services import storage
from app.services.calibration_service import CalibrationService
from tests.helpers import make_keypoints


def _make_metadata(video_id: str = "dance_001") -> ActionMetadata:
    return ActionMetadata(
        video_id=video_id,
        fps=30,
        duration_sec=2.0,
        total_frames=60,
        actions=[
            ActionSegment(
                action_id="act_001",
                name="动作 1",
                start_frame=0,
                end_frame=30,
                start_time_sec=0.0,
                end_time_sec=1.0,
            ),
            ActionSegment(
                action_id="act_002",
                name="动作 2",
                start_frame=30,
                end_frame=60,
                start_time_sec=1.0,
                end_time_sec=2.0,
            ),
        ],
    )


class TestCalibrationService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.metadata_dir = tmp / "metadata"
        self.keypoints_dir = tmp / "keypoints"
        self.animations_dir = tmp / "animations"
        self.service = CalibrationService(
            metadata_dir=self.metadata_dir,
            keypoints_dir=self.keypoints_dir,
            animations_dir=self.animations_dir,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed(self) -> None:
        """落盘元数据 + 关键点 + 动画（模拟管线完成态）。"""
        metadata = _make_metadata("dance_001")
        storage.save_metadata(self.metadata_dir, metadata)

        kps = make_keypoints(60, video_id="dance_001")
        storage.save_keypoints(self.keypoints_dir, kps, smoothed=True)

        clip = Retargeter().retarget(kps)
        storage.save_animation(self.animations_dir, "dance_001", clip)

    def test_get_calibration(self) -> None:
        self._seed()
        data = self.service.get_calibration("dance_001")
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["calibrated_by"], "local_user")

    def test_segment_adjustment(self) -> None:
        self._seed()
        calibration = CalibrationData(
            segment_adjustments=[
                SegmentAdjustment(action_id="act_001", start_frame=5, end_frame=25)
            ]
        )
        metadata = self.service.apply_calibration("dance_001", calibration)

        seg = metadata.actions[0]
        self.assertEqual((seg.start_frame, seg.end_frame), (5, 25))
        self.assertAlmostEqual(seg.start_time_sec, 5 / 30, places=3)
        self.assertAlmostEqual(seg.end_time_sec, 25 / 30, places=3)
        self.assertEqual(metadata.calibration.status, "calibrated")
        self.assertIsNotNone(metadata.calibration.calibrated_at)

        # 回写落盘
        reloaded = storage.load_metadata(self.metadata_dir, "dance_001")
        self.assertEqual(reloaded.actions[0].start_frame, 5)

    def test_keypoint_correction_local_retarget(self) -> None:
        self._seed()

        before_anim = storage.load_animation(self.animations_dir, "dance_001")
        root_before = list(before_anim.root_positions[10])
        wrist_before = list(before_anim.joint_rotations["L_Wrist"][10])

        calibration = CalibrationData(
            keypoint_corrections=[
                KeypointCorrection(
                    action_id="act_001",
                    frame=10,
                    joint="L_Wrist",
                    value=[1.0, 2.0, 3.0],
                )
            ]
        )
        metadata = self.service.apply_calibration("dance_001", calibration)
        self.assertEqual(metadata.calibration.status, "calibrated")

        # 关键点已被修正（L_Wrist = landmark 15）
        kps = storage.load_keypoints(self.keypoints_dir, "dance_001", smoothed=True)
        np.testing.assert_allclose(kps.data[10, 15, 0:3], [1.0, 2.0, 3.0])

        # 动画帧号 1:1 不变，根节点（Hips）不受腕部修正影响，腕部旋转改变
        after_anim = storage.load_animation(self.animations_dir, "dance_001")
        self.assertEqual(len(after_anim.root_positions), 60)
        self.assertEqual(len(after_anim.joint_rotations["L_Wrist"]), 60)
        np.testing.assert_allclose(after_anim.root_positions[10], root_before)
        self.assertFalse(np.allclose(after_anim.joint_rotations["L_Wrist"][10], wrist_before))

    def test_unknown_joint_raises(self) -> None:
        self._seed()
        calibration = CalibrationData(
            keypoint_corrections=[
                KeypointCorrection(
                    action_id="act_001", frame=5, joint="Hips", value=[0, 0, 0]
                )
            ]
        )
        with self.assertRaises(ValueError):
            self.service.apply_calibration("dance_001", calibration)

    def test_missing_metadata_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.service.get_calibration("dance_missing")


if __name__ == "__main__":
    unittest.main()
