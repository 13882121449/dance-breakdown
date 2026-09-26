"""帧抽取：OpenCV 逐帧读取视频，统一重采样到目标帧率并保序落盘。

统一时间轴铁律：
- 输出帧文件名 ``f_%06d.jpg`` 的序号即为整数帧号 ``t``（从 1 开始），
  与关键点/动画帧严格 1:1；
- 通过均匀采样把任意源帧率映射到目标帧率（默认 30fps），不丢序、不插帧。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from app.config import TARGET_FPS
from app.models.schemas import VideoMeta


class FrameExtractor:
    """从视频抽取帧序列。"""

    def __init__(self, target_fps: int = TARGET_FPS) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps 必须为正数，收到 {target_fps}")
        self.target_fps: int = target_fps

    def extract(
        self,
        video_path: Any,
        out_dir: Any,
        video_id: str = "",
        target_fps: int | None = None,
    ) -> VideoMeta:
        """抽取视频帧到 ``out_dir``，返回 :class:`VideoMeta`。

        Args:
            video_path: 视频文件路径。
            out_dir: 输出目录（帧保存到该目录下，文件名 f_000001.jpg ...）。
            video_id: 视频标识（缺省时取视频文件名 stem）。
            target_fps: 目标帧率，缺省使用构造时传入的帧率。
        """
        fps = target_fps or self.target_fps
        video_path = Path(video_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件：{video_path}")

        try:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if src_fps <= 0:
                src_fps = float(fps)
            total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_src <= 0:
                raise IOError(f"视频帧数为 0 或无法读取：{video_path}")

            duration_sec = total_src / src_fps
            # 目标帧总数 = 视频时长 × 目标帧率（四舍五入，至少 1 帧）
            total_target = max(1, int(round(duration_sec * fps)))
            # 每输出一帧对应多少个源帧（均匀采样步长）
            step = src_fps / fps

            saved = 0
            src_idx = 0
            next_target_src = 0.0
            while saved < total_target:
                ok, frame = cap.read()
                if not ok:
                    break
                # 源帧序号达到下一个采样点时才落盘，保证保序且 1:1
                if src_idx >= int(round(next_target_src)):
                    out_path = out_dir / f"f_{saved + 1:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved += 1
                    next_target_src += step
                src_idx += 1

            if saved == 0:
                raise IOError(f"未能从视频抽取任何帧：{video_path}")

            vid = video_id or video_path.stem
            return VideoMeta(
                video_id=vid,
                fps=fps,
                duration_sec=saved / fps,
                total_frames=saved,
                status="frames_extracted",
            )
        finally:
            cap.release()
