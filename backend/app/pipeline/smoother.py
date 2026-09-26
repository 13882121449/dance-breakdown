"""时序平滑：Savitzky-Golay 滤波，去除关键点抖动。

只平滑空间坐标 ``(x, y, z)``，``vis`` 可见度保持原值不参与滤波。
沿时间轴（axis=0）逐关键点、逐通道滤波，帧数不变（统一时间轴铁律）。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from app.models.schemas import KeypointSequence


class SequenceSmoother:
    """关键点序列时序平滑器。"""

    def __init__(self, window: int = 7, polyorder: int = 2) -> None:
        if window % 2 == 0:
            raise ValueError(f"Savitzky-Golay 窗口长度必须为奇数，收到 {window}")
        if polyorder >= window:
            raise ValueError(f"polyorder({polyorder}) 必须小于 window({window})")
        self.window: int = window
        self.polyorder: int = polyorder

    def smooth(
        self, kps: KeypointSequence, window: int | None = None
    ) -> KeypointSequence:
        """对关键点序列做 Savitzky-Golay 平滑。

        Args:
            kps: 输入关键点序列（``[T, 33, 4]``）。
            window: 平滑窗口（奇数），缺省使用构造时传入值。
        """
        win = window or self.window
        if win % 2 == 0:
            win += 1

        data = kps.data
        t = data.shape[0]
        if t < 3:
            # 帧数过少无法做窗口滤波，原样返回
            return KeypointSequence(
                video_id=kps.video_id,
                total_frames=t,
                num_keypoints=kps.num_keypoints,
                data=data.copy(),
            )

        # 窗口不能超过帧数，且需为奇数
        win = min(win, t if t % 2 == 1 else t - 1)
        if win < 3:
            win = 3
        if win > t:
            win = t if t % 2 == 1 else t - 1

        # 空间坐标 x,y,z 平滑；vis 保留
        xyz = data[:, :, 0:3]
        vis = data[:, :, 3:4]

        smoothed = np.empty_like(data, dtype=np.float64)
        smoothed[:, :, 0:3] = savgol_filter(
            xyz, window_length=win, polyorder=self.polyorder, axis=0, mode="interp"
        )
        smoothed[:, :, 3:4] = vis

        return KeypointSequence(
            video_id=kps.video_id,
            total_frames=t,
            num_keypoints=kps.num_keypoints,
            data=smoothed.astype(np.float32),
        )
