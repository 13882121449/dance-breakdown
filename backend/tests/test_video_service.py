"""VideoService 单元测试：上传、状态流转、元数据读取、进度回调。"""

import tempfile
import time
import unittest
from pathlib import Path

from app.models.schemas import ActionMetadata, ActionSegment, VideoMeta
from app.services import storage
from app.services.video_service import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
    VideoService,
)


class _FakeOrchestrator:
    """轻量假编排器：不跑真实管线，直接落盘一份元数据。"""

    def __init__(self, metadata_dir: Path, fail: bool = False) -> None:
        self.metadata_dir = metadata_dir
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def process(self, video_path, video_id=None, model_def=None) -> ActionMetadata:
        self.calls.append((str(video_path), video_id or ""))
        if self.fail:
            raise RuntimeError("fake pipeline failure")
        metadata = ActionMetadata(
            video_id=video_id or "dance_001",
            fps=30,
            total_frames=10,
            actions=[
                ActionSegment(
                    action_id="act_001",
                    name="动作 1",
                    start_frame=0,
                    end_frame=10,
                )
            ],
        )
        storage.save_metadata(self.metadata_dir, metadata)
        return metadata


def _wait_status(service: VideoService, video_id: str, target: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.get_status(video_id)["status"]
        if status == target:
            return service.get_status(video_id)
        time.sleep(0.01)
    raise AssertionError(f"等待状态 {target} 超时，当前 {service.get_status(video_id)}")


class TestVideoService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.upload_dir = tmp / "uploads"
        self.metadata_dir = tmp / "metadata"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _service(self, fail: bool = False) -> VideoService:
        return VideoService(
            orchestrator=_FakeOrchestrator(self.metadata_dir, fail=fail),
            upload_dir=self.upload_dir,
            metadata_dir=self.metadata_dir,
        )

    def test_upload_success(self) -> None:
        service = self._service()
        data = service.upload("dance.mp4", b"fake-video-bytes")
        self.assertTrue(data["video_id"].startswith("dance_"))
        self.assertEqual(data["status"], STATUS_UPLOADED)
        self.assertEqual(data["size_bytes"], len(b"fake-video-bytes"))
        # 文件确实落盘
        path = service.get_video_path(data["video_id"])
        self.assertTrue(path.exists())

    def test_upload_rejects_invalid_extension(self) -> None:
        service = self._service()
        with self.assertRaises(ValueError):
            service.upload("dance.txt", b"abc")

    def test_upload_rejects_empty(self) -> None:
        service = self._service()
        with self.assertRaises(ValueError):
            service.upload("dance.mp4", b"")

    def test_start_process_completes(self) -> None:
        service = self._service()
        data = service.upload("dance.mp4", b"fake-video-bytes")
        video_id = data["video_id"]

        result = service.start_process(video_id)
        self.assertEqual(result["status"], STATUS_PROCESSING)

        status = _wait_status(service, video_id, STATUS_COMPLETED)
        self.assertEqual(status["progress"]["stage"], "done")
        self.assertIsNone(status["error"])

        # 处理后元数据可读
        metadata = service.get_metadata(video_id)
        self.assertEqual(metadata.video_id, video_id)
        self.assertEqual(len(metadata.actions), 1)

    def test_start_process_failure(self) -> None:
        service = self._service(fail=True)
        data = service.upload("dance.mp4", b"fake-video-bytes")
        video_id = data["video_id"]

        service.start_process(video_id)
        status = _wait_status(service, video_id, STATUS_FAILED)
        self.assertIn("fake pipeline failure", status["error"])

    def test_get_status_unknown_raises(self) -> None:
        service = self._service()
        with self.assertRaises(FileNotFoundError):
            service.get_status("dance_missing")

    def test_get_status_disk_fallback_after_restart(self) -> None:
        # 模拟进程重启：内存态清空，但磁盘已有元数据 → completed
        service = self._service()
        metadata = ActionMetadata(
            video_id="dance_001",
            fps=30,
            total_frames=10,
            actions=[ActionSegment(action_id="act_001", start_frame=0, end_frame=10)],
        )
        storage.save_metadata(self.metadata_dir, metadata)

        status = service.get_status("dance_001")
        self.assertEqual(status["status"], STATUS_COMPLETED)
        self.assertEqual(status["progress"]["percent"], 100)


if __name__ == "__main__":
    unittest.main()
