"""Retargeter 单元测试（方向向量法四元数正确性）。"""

import unittest

import numpy as np

from app.pipeline.retargeter import (
    MODEL_DEFINITION,
    Retargeter,
    build_joint_positions,
    quat_between,
    quat_rotate,
)
from tests.helpers import make_keypoints


class TestRetargeter(unittest.TestCase):
    def test_quat_between_recovers_target(self) -> None:
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        q = quat_between(v0, v1)
        np.testing.assert_allclose(quat_rotate(q, v0), v1, atol=1e-6)
        # 单位四元数
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)

    def test_quat_between_identity(self) -> None:
        v = np.array([0.3, -0.4, 0.5])
        q = quat_between(v, v)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_retarget_output_structure(self) -> None:
        total = 40
        kps = make_keypoints(total)
        # 手腕做周期运动，产生非平凡骨骼方向
        for t in range(total):
            kps.data[t, 15, 0] += 40.0 * np.sin(2 * np.pi * t / 20.0)
            kps.data[t, 15, 1] += 20.0 * np.cos(2 * np.pi * t / 20.0)

        retargeter = Retargeter()
        clip = retargeter.retarget(kps)

        self.assertEqual(clip.start_frame, 0)
        self.assertEqual(clip.end_frame, total - 1)
        self.assertEqual(len(clip.root_positions), total)
        self.assertEqual(set(clip.joints), set(MODEL_DEFINITION["joints"]))
        for joint in clip.joints:
            self.assertEqual(len(clip.joint_rotations[joint]), total)

    def test_retarget_per_bone_rotation(self) -> None:
        """逐骨骼校验：quat_rotate(q, rest_dir) ≈ target_dir，且 q 为单位四元数。"""
        kps = make_keypoints(10)
        for t in range(10):
            kps.data[t, 15, 0] += 50.0 * np.sin(t)
            kps.data[t, 16, 1] += 30.0 * np.cos(t)

        retargeter = Retargeter()
        clip = retargeter.retarget(kps)
        pos = build_joint_positions(kps.data[0], retargeter.depth_scale)

        for joint in retargeter.joints:
            q = np.asarray(clip.joint_rotations[joint][0], dtype=np.float64)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=5)
            parent = retargeter._parent_of.get(joint)
            if parent is None:
                np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)
                continue
            rest_dir = retargeter._rest_dirs[joint]
            target_dir = retargeter.bone_direction(pos[parent], pos[joint])
            np.testing.assert_allclose(
                quat_rotate(q, rest_dir), target_dir, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()
