"""端到端 API 测试：上传 → 处理 → 状态 → 动作 → 注解 → 播放 → 校准。

使用 FastAPI TestClient + 假编排器（不跑真实 mediapipe 管线），
通过 dependency_overrides 把服务注入到临时目录，验证路由契约与统一响应格式。
"""

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ActionMetadata, ActionSegment, VideoMeta
from app.pipeline.retargeter import Retargeter
from app.services import storage
from app.services.annotation_service import AnnotationService, get_annotation_service
from app.services.calibration_service import CalibrationService, get_calibration_service
from app.services.video_service import VideoService, get_video_service
from tests.helpers import make_keypoints


class _FakePipeline:
    """假管线：落盘关键点 + 动画 + 元数据（模拟完整处理结果）。"""

    def __init__(self, metadata_dir: Path, keypoints_dir: Path, animations_dir: Path) -> None:
        self.metadata_dir = metadata_dir
        self.keypoints_dir = keypoints_dir
        self.animations_dir = animations_dir

    def process(self, video_path, video_id=None, model_def=None) -> ActionMetadata:
        kps = make_keypoints(30, video_id=video_id)
        storage.save_keypoints(self.keypoints_dir, kps, smoothed=True)

        clip = Retargeter().retarget(kps)
        storage.save_animation(self.animations_dir, video_id, clip)

        metadata = ActionMetadata(
            video_id=video_id,
            fps=30,
            duration_sec=1.0,
            total_frames=30,
            actions=[
                ActionSegment(
                    action_id="act_001",
                    name="动作 1",
                    start_frame=0,
                    end_frame=30,
                    start_time_sec=0.0,
                    end_time_sec=1.0,
                )
            ],
            keypoints_file=f"keypoints/{video_id}_kps_smoothed.npy",
            animation_file=f"animations/{video_id}_anim.json",
        )
        storage.save_metadata(self.metadata_dir, metadata)
        return metadata


class TestApiEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.metadata_dir = tmp / "metadata"
        self.keypoints_dir = tmp / "keypoints"
        self.animations_dir = tmp / "animations"
        self.upload_dir = tmp / "uploads"

        self.video_service = VideoService(
            orchestrator=_FakePipeline(
                self.metadata_dir, self.keypoints_dir, self.animations_dir
            ),
            upload_dir=self.upload_dir,
            metadata_dir=self.metadata_dir,
            keypoints_dir=self.keypoints_dir,
            animations_dir=self.animations_dir,
        )
        self.annotation_service = AnnotationService(metadata_dir=self.metadata_dir)
        self.calibration_service = CalibrationService(
            metadata_dir=self.metadata_dir,
            keypoints_dir=self.keypoints_dir,
            animations_dir=self.animations_dir,
        )

        app.dependency_overrides[get_video_service] = lambda: self.video_service
        app.dependency_overrides[get_annotation_service] = lambda: self.annotation_service
        app.dependency_overrides[get_calibration_service] = lambda: self.calibration_service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.client.close()
        self._tmp.cleanup()

    def _wait_completed(self, video_id: str, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.client.get(f"/api/video/{video_id}/status")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["code"], 0)
            if body["data"]["status"] == "completed":
                return
            time.sleep(0.02)
        raise AssertionError("处理状态等待超时")

    def test_health(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"code": 0, "message": "ok", "data": {}})

    def test_full_flow(self) -> None:
        with self.client as client:
            # 1) 上传
            resp = client.post(
                "/api/video/upload",
                files={"file": ("dance.mp4", b"fake-video-content", "video/mp4")},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()["data"]
            self.assertTrue(data["video_id"].startswith("dance_"))
            self.assertEqual(data["status"], "uploaded")
            video_id = data["video_id"]

            # 2) 处理（后台线程）
            resp = client.post(f"/api/video/{video_id}/process")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["data"]["status"], "processing")

            # 3) 轮询状态 → 完成
            self._wait_completed(video_id)

            # 4) 元数据
            resp = client.get(f"/api/video/{video_id}/metadata")
            self.assertEqual(resp.json()["code"], 0)
            self.assertEqual(resp.json()["data"]["total_frames"], 30)

            # 5) 动作列表
            resp = client.get(f"/api/video/{video_id}/actions")
            actions = resp.json()["data"]
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["action_id"], "act_001")

            # 6) 更新注解
            resp = client.patch(
                f"/api/video/{video_id}/actions/act_001",
                json={"annotations": {"difficulty": "中级"}, "name": "甩耳朵舞"},
            )
            self.assertEqual(resp.json()["data"]["annotations"]["difficulty"], "中级")
            self.assertEqual(resp.json()["data"]["annotation_source"], "manual")

            # 7) 播放数据
            resp = client.get(f"/api/playback/{video_id}")
            playback = resp.json()["data"]
            self.assertEqual(playback["video_id"], video_id)
            self.assertIn("joint_rotations", playback["animation"])
            self.assertEqual(playback["keypoints"]["total_frames"], 30)

            # 8) 单帧关键点
            resp = client.get(f"/api/playback/{video_id}/keypoints/0")
            self.assertEqual(resp.json()["data"]["frame"], 0)
            self.assertEqual(len(resp.json()["data"]["keypoints"]), 33)

            # 9) 提交校准
            resp = client.post(
                f"/api/video/{video_id}/calibration",
                json={
                    "segment_adjustments": [
                        {"action_id": "act_001", "start_frame": 2, "end_frame": 28}
                    ],
                    "keypoint_corrections": [],
                },
            )
            self.assertEqual(resp.json()["code"], 0)
            self.assertEqual(resp.json()["data"]["calibration"]["status"], "calibrated")
            self.assertEqual(resp.json()["data"]["actions"][0]["start_frame"], 2)

    def test_unknown_video_returns_404_unified(self) -> None:
        with self.client as client:
            resp = client.get("/api/video/dance_missing/status")
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(resp.json()["code"], 404)
            self.assertIn("message", resp.json())


if __name__ == "__main__":
    unittest.main()
