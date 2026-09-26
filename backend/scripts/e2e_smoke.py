"""真实管线端到端冒烟：合成视频走完整 orchestrator 全链路（临时脚本）。

验证：帧号 1:1、动作段无缝覆盖、17 维注解齐全、动画帧数 == total_frames。
不依赖任何 mock，使用真实 mediapipe 模型（无姿态时回退为零关键点）。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 允许直接 `python scripts/e2e_smoke.py` 运行时正确导入 backend 下的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.pipeline.pipeline import PipelineOrchestrator
from app.models.schemas import ALL_ANNOTATION_FIELDS


def _write_synthetic_video(path: Path, num_frames: int = 30, fps: int = 30) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (128, 128),
    )
    assert writer.isOpened(), "VideoWriter 打开失败"
    for i in range(num_frames):
        # 移动方块，提供非零像素变化（真实人像检测不到，姿态回退为零）
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        x = (i * 3) % 96
        frame[16:48, x : x + 32] = (255, 255, 255)
        writer.write(frame)
    writer.release()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = tmp / "synthetic.avi"
        _write_synthetic_video(video_path)

        orchestrator = PipelineOrchestrator()
        metadata = orchestrator.process(video_path, video_id="dance_e2e")

        total = metadata.total_frames
        fps = metadata.fps
        print(f"video_id={metadata.video_id} fps={fps} total_frames={total}")
        print(f"num_actions={len(metadata.actions)}")

        # 1) 动作段无缝覆盖全片 [0, total-1]
        assert metadata.actions, "应至少产出 1 个动作段"
        assert metadata.actions[0].start_frame == 0, metadata.actions[0].start_frame
        assert metadata.actions[-1].end_frame == total - 1, metadata.actions[-1].end_frame
        for a, b in zip(metadata.actions, metadata.actions[1:]):
            assert a.end_frame == b.start_frame, (a.end_frame, b.start_frame)

        # 2) 17 维注解齐全（模板补齐后非空）
        for seg in metadata.actions:
            for field in ALL_ANNOTATION_FIELDS:
                assert getattr(seg.annotations, field) != "", f"{seg.action_id}.{field} 为空"

        # 3) 动画帧号 1:1
        from app.config import ANIMATIONS_DIR, KEYPOINTS_DIR
        from app.services import storage

        clip = storage.load_animation(ANIMATIONS_DIR, "dance_e2e")
        assert len(clip.root_positions) == total, (len(clip.root_positions), total)
        for joint in clip.joints:
            assert len(clip.joint_rotations[joint]) == total, joint

        # 4) 关键点帧号 1:1
        kps = storage.load_keypoints(KEYPOINTS_DIR, "dance_e2e", smoothed=True)
        assert kps.total_frames == total, (kps.total_frames, total)

        print("E2E_OK")


if __name__ == "__main__":
    main()
