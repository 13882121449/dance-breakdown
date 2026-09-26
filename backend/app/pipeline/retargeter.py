"""骨骼重定向：方向向量法（33 landmarks → 模型关节四元数）。

对每段骨骼（父关节→子关节）计算方向向量，用最短弧四元数把模型 rest 方向
旋转到目标方向；根节点（Hips）用关键点位置驱动平移。

统一时间轴铁律：重定向逐帧进行，``anim[t]`` 与 ``kps[t]``、视频第 ``t`` 帧
严格 1:1，长度等于 ``total_frames``，不做任何重采样。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.models.schemas import AnimationClip, KeypointSequence

# ---------------------------------------------------------------------------
# 模型定义（Mixamo 标准人形骨骼，17 关节）
# ---------------------------------------------------------------------------
# 说明：Hips/Spine/Chest/Neck 为「派生关节」，由关键点中点/插值计算得到，
# 无单一 landmark 对应，故不出现在 retarget_map 中。
MODEL_DEFINITION: dict[str, Any] = {
    "name": "mixamo_ybot",
    "file": "assets/models/ybot.glb",
    "joints": [
        "Hips", "Spine", "Chest", "Neck", "Head",
        "L_Shoulder", "L_Elbow", "L_Wrist",
        "R_Shoulder", "R_Elbow", "R_Wrist",
        "L_Hip", "L_Knee", "L_Ankle",
        "R_Hip", "R_Knee", "R_Ankle",
    ],
    "retarget_map": {
        "0": "Head",
        "11": "L_Shoulder", "12": "R_Shoulder",
        "13": "L_Elbow", "14": "R_Elbow",
        "15": "L_Wrist", "16": "R_Wrist",
        "23": "L_Hip", "24": "R_Hip",
        "25": "L_Knee", "26": "R_Knee",
        "27": "L_Ankle", "28": "R_Ankle",
    },
    # 父关节 → 子关节 的骨骼层级
    "bones": [
        ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"), ("Neck", "Head"),
        ("Chest", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
        ("Chest", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"),
        ("Hips", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"),
        ("Hips", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"),
    ],
    # 规范 T-pose（仅方向有意义，单位任意，供 rest_dir 计算）
    "rest_pose": {
        "Hips": (0.0, 0.0, 0.0),
        "Spine": (0.0, 0.25, 0.0),
        "Chest": (0.0, 0.45, 0.0),
        "Neck": (0.0, 0.58, 0.0),
        "Head": (0.0, 0.75, 0.0),
        "L_Shoulder": (-0.25, 0.45, 0.0),
        "L_Elbow": (-0.50, 0.45, 0.0),
        "L_Wrist": (-0.75, 0.45, 0.0),
        "R_Shoulder": (0.25, 0.45, 0.0),
        "R_Elbow": (0.50, 0.45, 0.0),
        "R_Wrist": (0.75, 0.45, 0.0),
        "L_Hip": (-0.10, 0.0, 0.0),
        "L_Knee": (-0.10, -0.45, 0.0),
        "L_Ankle": (-0.10, -0.90, 0.0),
        "R_Hip": (0.10, 0.0, 0.0),
        "R_Knee": (0.10, -0.45, 0.0),
        "R_Ankle": (0.10, -0.90, 0.0),
    },
}


# ---------------------------------------------------------------------------
# 四元数工具（模块级，便于单测）
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def quat_between(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """最短弧四元数：把单位向量 v0 旋转到 v1，返回 ``[w, x, y, z]``。"""
    v0 = _normalize(np.asarray(v0, dtype=np.float64))
    v1 = _normalize(np.asarray(v1, dtype=np.float64))
    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))

    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 同向 → 单位四元数
    if dot < -1.0 + 1e-9:
        # 反向 180°：绕任一正交轴旋转
        axis = np.cross(v0, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v0, np.array([0.0, 1.0, 0.0]))
        axis = _normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    axis = _normalize(np.cross(v0, v1))
    w = np.sqrt((1.0 + dot) / 2.0)
    sin_half = np.sqrt((1.0 - dot) / 2.0)
    return np.array([w, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 ``q=[w,x,y,z]`` 旋转向量 ``v``，返回旋转后向量。"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q[0], q[1], q[2], q[3]
    # v' = v + 2*w*(u×v) + 2*(u×(u×v))，其中 u=(x,y,z)
    u = np.array([x, y, z])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


# ---------------------------------------------------------------------------
# 关节位置构建
# ---------------------------------------------------------------------------
def build_joint_positions(
    frame: np.ndarray, depth_scale: float = 200.0
) -> dict[str, np.ndarray]:
    """由单帧 33 landmarks（``[33,4]``）构建 17 关节 3D 位置。

    x/y 为像素坐标，z 为 world 相对深度（米）；``depth_scale`` 把 z 换算到
    与像素近似的尺度，使方向向量的三维比例合理（相对精度可接受，MVP）。
    """
    def lm(i: int) -> np.ndarray:
        x, y, z = float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2])
        return np.array([x, y, z * depth_scale])

    hip_l, hip_r = lm(23), lm(24)
    sh_l, sh_r = lm(11), lm(12)
    hips = (hip_l + hip_r) / 2.0
    chest = (sh_l + sh_r) / 2.0
    spine = (hips + chest) / 2.0
    neck = chest + (chest - spine) * 0.5

    return {
        "Hips": hips,
        "Spine": spine,
        "Chest": chest,
        "Neck": neck,
        "Head": lm(0),
        "L_Shoulder": lm(11),
        "L_Elbow": lm(13),
        "L_Wrist": lm(15),
        "R_Shoulder": lm(12),
        "R_Elbow": lm(14),
        "R_Wrist": lm(16),
        "L_Hip": lm(23),
        "L_Knee": lm(25),
        "L_Ankle": lm(27),
        "R_Hip": lm(24),
        "R_Knee": lm(26),
        "R_Ankle": lm(28),
    }


class Retargeter:
    """方向向量法骨骼重定向器。"""

    def __init__(
        self,
        model_def: Optional[dict[str, Any]] = None,
        depth_scale: float = 200.0,
    ) -> None:
        self.model_def: dict[str, Any] = dict(model_def or MODEL_DEFINITION)
        self.depth_scale: float = depth_scale
        self.joints: list[str] = list(self.model_def["joints"])
        self.bones: list[tuple[str, str]] = list(self.model_def["bones"])
        self.rest_pose: dict[str, tuple[float, float, float]] = dict(
            self.model_def["rest_pose"]
        )
        # 预计算 rest 方向
        self._rest_dirs: dict[str, np.ndarray] = {
            child: _normalize(
                np.asarray(self.rest_pose[child]) - np.asarray(self.rest_pose[parent])
            )
            for parent, child in self.bones
        }
        # 子关节 → 父关节 索引
        self._parent_of: dict[str, str] = {c: p for p, c in self.bones}

    def bone_direction(self, p_parent: np.ndarray, p_child: np.ndarray) -> np.ndarray:
        """骨骼方向向量（单位化）。"""
        return _normalize(np.asarray(p_child) - np.asarray(p_parent))

    def solve_rotation(
        self, rest_dir: np.ndarray, target_dir: np.ndarray
    ) -> np.ndarray:
        """求解 rest_dir 到 target_dir 的最短弧四元数。"""
        return quat_between(rest_dir, target_dir)

    def retarget(
        self, kps: KeypointSequence, model_def: Optional[dict[str, Any]] = None
    ) -> AnimationClip:
        """全量重定向：返回覆盖所有帧的 :class:`AnimationClip`。"""
        if model_def is not None:
            self.__init__(model_def, self.depth_scale)  # type: ignore[misc]

        total = kps.total_frames
        joint_rotations: dict[str, list[list[float]]] = {j: [] for j in self.joints}
        root_positions: list[list[float]] = []

        for t in range(total):
            pos = build_joint_positions(kps.data[t], self.depth_scale)
            root_positions.append([float(v) for v in pos["Hips"]])
            for joint in self.joints:
                q = self._joint_rotation(joint, pos)
                joint_rotations[joint].append([float(v) for v in q])

        return AnimationClip(
            clip_name=f"{kps.video_id}_anim",
            start_frame=0,
            end_frame=max(0, total - 1),
            joints=self.joints,
            joint_rotations=joint_rotations,
            root_positions=root_positions,
        )

    def _joint_rotation(
        self, joint: str, pos: dict[str, np.ndarray]
    ) -> np.ndarray:
        """计算单关节局部旋转四元数（根关节 Hips 为恒等）。"""
        parent = self._parent_of.get(joint)
        if parent is None:
            return np.array([1.0, 0.0, 0.0, 0.0])  # 根关节
        target_dir = self.bone_direction(pos[parent], pos[joint])
        if np.linalg.norm(pos[joint] - pos[parent]) < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return self.solve_rotation(self._rest_dirs[joint], target_dir)
