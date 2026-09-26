"""Retargeter 覆盖盲区补测：反向四元数、零向量归一化、派生关节构建。"""

import unittest

import numpy as np

from app.pipeline.retargeter import (
    _normalize,
    build_joint_positions,
    quat_between,
    quat_rotate,
)
from tests.helpers import base_frame


class TestRetargeterExtended(unittest.TestCase):
    def test_quat_between_antiparallel(self) -> None:
        """反向 180°：返回绕正交轴的单位四元数，能把 v0 旋转到 -v0。"""
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([-1.0, 0.0, 0.0])
        q = quat_between(v0, v1)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)
        np.testing.assert_allclose(quat_rotate(q, v0), v1, atol=1e-6)

    def test_quat_between_antiparallel_axis_aligned(self) -> None:
        """反向向量与 x 轴共线时走正交轴兜底分支，仍能正确旋转。"""
        v0 = np.array([0.0, 0.0, 1.0])
        v1 = np.array([0.0, 0.0, -1.0])
        q = quat_between(v0, v1)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)
        np.testing.assert_allclose(quat_rotate(q, v0), v1, atol=1e-6)

    def test_normalize_zero_vector_returns_zero(self) -> None:
        """零向量归一化返回零向量而非 NaN。"""
        out = _normalize(np.zeros(3))
        np.testing.assert_allclose(out, np.zeros(3), atol=0.0)

    def test_build_joint_positions_derived_joints(self) -> None:
        """派生关节 Hips/Spine/Chest/Neck 由 landmark 中点/插值正确计算。"""
        frame = base_frame()
        pos = build_joint_positions(frame, depth_scale=200.0)

        lm = lambda i: np.array(  # noqa: E731
            [float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2]) * 200.0]
        )
        hips = (lm(23) + lm(24)) / 2.0
        chest = (lm(11) + lm(12)) / 2.0
        spine = (hips + chest) / 2.0
        neck = chest + (chest - spine) * 0.5

        np.testing.assert_allclose(pos["Hips"], hips)
        np.testing.assert_allclose(pos["Spine"], spine)
        np.testing.assert_allclose(pos["Chest"], chest)
        np.testing.assert_allclose(pos["Neck"], neck)
        # 直接映射的关节应精确等于对应 landmark
        np.testing.assert_allclose(pos["L_Wrist"], lm(15))
        np.testing.assert_allclose(pos["R_Ankle"], lm(28))
        np.testing.assert_allclose(pos["Head"], lm(0))


if __name__ == "__main__":
    unittest.main()
