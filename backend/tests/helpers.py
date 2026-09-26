"""测试辅助：合成关键点数据构造。"""

from __future__ import annotations

import numpy as np

from app.models.schemas import KeypointSequence

# 合成人形姿态的 2D 像素坐标（640x480 画布，仅填充关键关节）
BASE_2D: dict[int, tuple[float, float]] = {
    0: (320, 60),       # 鼻
    11: (280, 140),     # 左肩
    12: (360, 140),     # 右肩
    13: (250, 220),     # 左肘
    14: (390, 220),     # 右肘
    15: (220, 300),     # 左腕
    16: (420, 300),     # 右腕
    23: (290, 340),     # 左髋
    24: (350, 340),     # 右髋
    25: (290, 440),     # 左膝
    26: (350, 440),     # 右膝
    27: (290, 470),     # 左踝
    28: (350, 470),     # 右踝
}


def base_frame() -> np.ndarray:
    """返回一个 T-pose 的 ``[33, 4]`` 帧（z=0, vis=1）。"""
    frame = np.zeros((33, 4), dtype=np.float32)
    frame[:, 3] = 1.0
    for idx, (x, y) in BASE_2D.items():
        frame[idx, 0] = x
        frame[idx, 1] = y
        frame[idx, 2] = 0.0
    return frame


def make_keypoints(
    total_frames: int, video_id: str = "test" 
) -> KeypointSequence:
    """构造 ``total_frames`` 帧的静态 T-pose 关键点序列。"""
    data = np.stack([base_frame() for _ in range(total_frames)], axis=0)
    return KeypointSequence(
        video_id=video_id,
        total_frames=total_frames,
        num_keypoints=33,
        data=data,
    )
