"""CalibrationService 覆盖盲区补测：非法入参、局部重定向范围、动画缺失兜底。"""

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


class TestCalibrationServiceExtended(unittest.TestCase):
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

    def _seed(self, with_animation: bool = True) -> None:
        metadata = _make_metadata("dance_001")
        storage.save_metadata(self.metadata_dir, metadata)
        kps = make_keypoints(60, video_id="dance_001")
        storage.save_keypoints(self.keypoints_dir, kps, smoothed=True)
        if with_animation:
            clip = Retargeter().retarget(kps)
            storage.save_animation(self.animations_dir, "dance_001", clip)

    def test_segment_adjustment_invalid_range_raises(self) -> None:
        self._seed()
        calibration = CalibrationData(
            segment_adjustments=[
                SegmentAdjustment(action_id="act_001", start_frame=30, end_frame=10)
            ]
        )
        with self.assertRaises(ValueError):
            self.service.apply_calibration("dance_001", calibration)

    def test_segment_adjustment_missing_action_raises(self) -> None:
        self._seed()
        calibration = CalibrationData(
            segment_adjustments=[
                SegmentAdjustment(action_id="act_999", start_frame=0, end_frame=10)
            ]
        )
        with self.assertRaises(FileNotFoundError):
            self.service.apply_calibration("dance_001", calibration)

    def test_keypoint_correction_frame_out_of_range_raises(self) -> None:
        self._seed()
        calibration = CalibrationData(
            keypoint_corrections=[
                KeypointCorrection(
                    action_id="act_001", frame=99, joint="L_Wrist", value=[1.0, 2.0, 3.0]
                )
            ]
        )
        with self.assertRaises(ValueError):
            self.service.apply_calibration("dance_001", calibration)

    def test_keypoint_correction_value_too_short_raises(self) -> None:
        self._seed()
        calibration = CalibrationData(
            keypoint_corrections=[
                KeypointCorrection(
                    action_id="act_001", frame=10, joint="L_Wrist", value=[1.0]
                )
            ]
        )
        with self.assertRaises(ValueError):
            self.service.apply_calibration("dance_001", calibration)

    def test_local_retarget_only_affects_target_segment(self) -> None:
        """局部重定向只改动受影响动作段，其余动作段的动画帧保持不变。"""
        self._seed()

        before = storage.load_animation(self.animations_dir, "dance_001")
        # 记录 act_002（帧 30~59）在修正前的旋转
        seg2_before = {
            joint: [list(r) for r in before.joint_rotations[joint][30:60]]
            for joint in before.joints
        }

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
        self.service.apply_calibration("dance_001", calibration)

        after = storage.load_animation(self.animations_dir, "dance_001")
        # act_001 内被修正帧的腕部旋转发生变化
        self.assertFalse(
            np.allclose(after.joint_rotations["L_Wrist"][10], before.joint_rotations["L_Wrist"][10])
        )
        # act_002 的动画帧逐帧不变（未被局部重定向波及）
        for joint in before.joints:
            for local, g in enumerate(range(30, 60)):
                np.testing.assert_allclose(
                    after.joint_rotations[joint][g],
                    seg2_before[joint][local],
                    atol=1e-9,
                )

    def test_load_or_retarget_full_fallback_when_animation_missing(self) -> None:
        """动画文件缺失时，关键点修正应从关键点全量重定向生成动画。"""
        self._seed(with_animation=False)

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

        # 动画文件已被兜底生成并回写，帧号 1:1
        clip = storage.load_animation(self.animations_dir, "dance_001")
        self.assertEqual(len(clip.root_positions), 60)
        self.assertEqual(len(clip.joint_rotations["L_Wrist"]), 60)


if __name__ == "__main__":
    unittest.main()
