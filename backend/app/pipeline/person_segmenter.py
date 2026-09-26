"""人像分割/去背景：MediaPipe ImageSegmenter（Selfie Segmentation 模型）。

复杂背景下先对每帧做人像分割、背景置黑，显著提升后续姿态估计准确率。
分割**不改变帧号**：``frames_masked/`` 与 ``frames/`` 逐帧 1:1 保序。

mediapipe 1.0+ 已移除旧版 ``mp.solutions`` API，改用 Tasks API
（``mp.tasks.vision.ImageSegmenter``），且模型文件需显式提供（wheel 不内置）。
模块采用惰性导入：mediapipe 缺失时仅在实例化时抛出带说明的 ImportError。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import SELFIE_SEGMENTATION_MODEL

try:  # pragma: no cover - 依赖是否可用由运行环境决定
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


class PersonSegmenter:
    """MediaPipe Selfie Segmentation 人像分割器（Tasks API）。"""

    def __init__(
        self,
        model_asset_path: Optional[Any] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        """初始化分割模型。

        Args:
            model_asset_path: ``selfie_segmenter.tflite`` 模型文件路径；
                缺省使用 :data:`SELFIE_SEGMENTATION_MODEL`。
            confidence_threshold: 前景置信度阈值（>= 该值视为人像）。
        """
        if mp is None:
            raise ImportError(
                "mediapipe 未安装：请安装 mediapipe（Python 3.13 可用 1.0+ 版本），"
                "或改用 ONNX Runtime 版人像分割模型（见 architecture.md 备选方案）。"
            )
        self.confidence_threshold: float = confidence_threshold
        model_path = self._resolve_model(model_asset_path)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment(self, frames_dir: Any, out_dir: Any) -> int:
        """对 ``frames_dir`` 下所有 ``f_*.jpg`` 做人像分割，背景置黑输出到 ``out_dir``。

        Returns:
            处理帧数。
        """
        frames_dir = Path(frames_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(frames_dir.glob("f_*.jpg"))
        count = 0
        for frame_path in frame_paths:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._segmenter.segment(mp_image)
            mask = self._foreground_mask(result)
            if mask is None:
                continue
            masked = cv2.bitwise_and(bgr, bgr, mask=mask)
            cv2.imwrite(str(out_dir / frame_path.name), masked)
            count += 1
        return count

    def _foreground_mask(self, result: Any) -> Optional[np.ndarray]:
        """从分割结果提取前景二值 mask（HxW uint8）。"""
        masks = getattr(result, "confidence_masks", None)
        if not masks:
            return None
        conf = masks[0]  # 0~1，1=前景
        # mediapipe 1.0+ 的 confidence_masks 返回 mp.Image，需取底层 ndarray
        if not isinstance(conf, np.ndarray):
            conf = conf.numpy_view()
        return (conf > self.confidence_threshold).astype(np.uint8)

    def _resolve_model(self, model_asset_path: Optional[Any]) -> Path:
        path = Path(model_asset_path) if model_asset_path is not None else SELFIE_SEGMENTATION_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"人像分割模型文件不存在：{path}。请下载 selfie_segmenter.tflite "
                "（https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                "selfie_segmenter/float16/latest/selfie_segmenter.tflite）到该路径。"
            )
        return path

    def close(self) -> None:
        """释放底层模型资源。"""
        if getattr(self, "_segmenter", None) is not None:
            self._segmenter.close()
