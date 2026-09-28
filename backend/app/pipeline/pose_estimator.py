"""人体姿态估计：MediaPipe PoseLandmarker（33 landmarks，含 3D world 坐标）。

输出关键点序列 ``[T, 33, 4]``，每帧每点 ``(x, y, z, vis)``：
- ``x/y``：像素坐标（归一化 landmark × 图像宽/高）；
- ``z``：MediaPipe world 相对深度（米，髋原点归一化）；
- ``vis``：可见度（0~1）。

mediapipe 1.0+ 使用 Tasks API（``mp.tasks.vision.PoseLandmarker``），模型文件
（``pose_landmarker_full.task``）需显式提供；模块惰性导入，缺依赖时仅实例化报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import POSE_LANDMARKER_MODEL
from app.models.schemas import (
    KEYPOINT_DIM,
    MEDIAPIPE_NUM_KEYPOINTS,
    KeypointSequence,
    VideoMeta,
)

try:  # pragma: no cover
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


class PoseEstimator:
    """MediaPipe Pose 姿态估计器（Tasks API，33 landmarks）。"""

    def __init__(
        self,
        model_asset_path: Optional[Any] = None,
        num_poses: int = 5,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe 未安装：请安装 mediapipe（Python 3.13 可用 1.0+ 版本），"
                "或改用 ONNX Runtime 版姿态估计模型（如 MoveNet/BlazePose）。"
            )
        model_path = self._resolve_model(model_asset_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        # 单人锁定跟踪：记录上一帧选中人物的 bbox 中心/面积，跨帧匹配避免跳变
        self._tracked_center: Optional[tuple[float, float]] = None
        self._tracked_area: float = 0.0
        self._lost_frames: int = 0

    def estimate(
        self, frames_dir: Any, video_meta: VideoMeta
    ) -> KeypointSequence:
        """对帧目录逐帧估计姿态，返回 :class:`KeypointSequence`。"""
        frames_dir = Path(frames_dir)
        frame_paths = sorted(frames_dir.glob("f_*.jpg"))

        frames: list[np.ndarray] = []
        for frame_path in frame_paths:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                # 读不到帧时补全零关键点，保证帧号连续（统一时间轴铁律）
                frames.append(
                    np.zeros((MEDIAPIPE_NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
                )
                continue
            frames.append(self._estimate_frame(bgr))

        if not frames:
            data = np.empty((0, MEDIAPIPE_NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        else:
            data = np.stack(frames, axis=0).astype(np.float32)

        return KeypointSequence(
            video_id=video_meta.video_id,
            total_frames=int(data.shape[0]),
            num_keypoints=MEDIAPIPE_NUM_KEYPOINTS,
            data=data,
        )

    def _estimate_frame(self, bgr: np.ndarray) -> np.ndarray:
        """对单帧估计 33 关键点，返回 ``[33, 4]``。"""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        return self._extract_landmarks(result, w, h)

    def _extract_landmarks(self, result: Any, w: int, h: int) -> np.ndarray:
        out = np.zeros((MEDIAPIPE_NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

        pose_landmarks = getattr(result, "pose_landmarks", None)
        world_landmarks = getattr(result, "pose_world_landmarks", None)
        if not pose_landmarks:
            self._lost_frames += 1
            if self._lost_frames > 30:
                self._tracked_center = None
                self._tracked_area = 0.0
            return out

        self._lost_frames = 0
        idx = self._select_person(pose_landmarks)
        landmarks = pose_landmarks[idx]
        world = world_landmarks[idx] if world_landmarks and idx < len(world_landmarks) else None

        for i in range(MEDIAPIPE_NUM_KEYPOINTS):
            if i >= len(landmarks):
                break
            lm = landmarks[i]
            out[i, 0] = lm.x * w
            out[i, 1] = lm.y * h
            if world is not None and i < len(world):
                out[i, 2] = world[i].z
            else:
                out[i, 2] = lm.z
            out[i, 3] = getattr(lm, "visibility", 1.0)
        return out

    def _select_person(self, pose_landmarks: list[Any]) -> int:
        """从多人中锁定「主要人物」：面积最大者，且跨帧跟踪不跳变。"""
        n = len(pose_landmarks)
        if n == 1:
            return 0

        boxes: list[tuple[float, float, float]] = []
        for lm in pose_landmarks:
            xs = [p.x for p in lm if getattr(p, "visibility", 1.0) > 0.3]
            ys = [p.y for p in lm if getattr(p, "visibility", 1.0) > 0.3]
            if not xs:
                boxes.append((0.0, 0.0, 0.0))
                continue
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            boxes.append((cx, cy, area))

        if self._tracked_center is not None:
            best, best_score = 0, float("inf")
            for i, (cx, cy, area) in enumerate(boxes):
                if area <= 0:
                    continue
                dc = (cx - self._tracked_center[0]) ** 2 + (
                    cy - self._tracked_center[1]
                ) ** 2
                da = abs(area - self._tracked_area) / max(self._tracked_area, 1e-6)
                score = dc + da
                if score < best_score:
                    best_score, best = score, i
        else:
            best = max(range(n), key=lambda i: boxes[i][2])

        cx, cy, area = boxes[best]
        self._tracked_center = (cx, cy)
        self._tracked_area = area
        return best

    def _resolve_model(self, model_asset_path: Optional[Any]) -> Path:
        path = Path(model_asset_path) if model_asset_path is not None else POSE_LANDMARKER_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"姿态估计模型文件不存在：{path}。请下载 pose_landmarker_full.task "
                "（https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/1/pose_landmarker_full.task）到该路径。"
            )
        return path

    def close(self) -> None:
        """释放底层模型资源。"""
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
