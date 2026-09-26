"""API 层覆盖盲区补测：统一 {code,message,data} 错误码与参数校验 422。"""

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ActionMetadata, ActionSegment
from app.pipeline.retargeter import Retargeter
from app.services import storage
from app.services.annotation_service import AnnotationService, get_annotation_service
from app.services.calibration_service import CalibrationService, get_calibration_service
from app.services.video_service import VideoService, get_video_service
from tests.helpers import make_keypoints


class _FakePipeline:
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


class TestApiErrors(unittest.TestCase):
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

    def _completed_video_id(self) -> str:
        resp = self.client.post(
            "/api/video/upload",
            files={"file": ("dance.mp4", b"fake-video-content", "video/mp4")},
        )
        video_id = resp.json()["data"]["video_id"]
        self.client.post(f"/api/video/{video_id}/process")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            s = self.client.get(f"/api/video/{video_id}/status").json()["data"]["status"]
            if s == "completed":
                return video_id
            time.sleep(0.02)
        raise AssertionError("处理等待超时")

    def test_upload_invalid_extension_returns_400_unified(self) -> None:
        resp = self.client.post(
            "/api/video/upload",
            files={"file": ("dance.txt", b"not-a-video", "text/plain")},
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertEqual(body["code"], 400)
        self.assertEqual(body["data"], {})
        self.assertIn("message", body)

    def test_unknown_annotation_field_returns_400_unified(self) -> None:
        video_id = self._completed_video_id()
        resp = self.client.patch(
            f"/api/video/{video_id}/actions/act_001",
            json={"annotations": {"not_a_field": "x"}},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["code"], 400)

    def test_frame_keypoints_out_of_range_returns_400_unified(self) -> None:
        video_id = self._completed_video_id()
        resp = self.client.get(f"/api/playback/{video_id}/keypoints/999")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["code"], 400)

    def test_calibration_invalid_segment_returns_400_unified(self) -> None:
        video_id = self._completed_video_id()
        resp = self.client.post(
            f"/api/video/{video_id}/calibration",
            json={
                "segment_adjustments": [
                    {"action_id": "act_001", "start_frame": 28, "end_frame": 2}
                ],
                "keypoint_corrections": [],
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["code"], 400)

    def test_calibration_validation_error_returns_422_unified(self) -> None:
        """请求体缺少必填字段 → 参数校验失败 422，统一格式且 code=422。"""
        video_id = self._completed_video_id()
        resp = self.client.post(
            f"/api/video/{video_id}/calibration",
            json={"segment_adjustments": [{"action_id": "act_001"}]},
        )
        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertEqual(body["code"], 422)
        self.assertIn("message", body)
        self.assertIn("data", body)


if __name__ == "__main__":
    unittest.main()
