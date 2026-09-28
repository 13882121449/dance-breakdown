"""拆解真实舞蹈视频：跑完整管线并输出动作列表 + 注解摘要。"""
import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.pipeline import PipelineOrchestrator
from app.models.schemas import ALL_ANNOTATION_FIELDS
from app.config import UPLOAD_DIR

SRC = r'D:\微信\xwechat_files\wxid_qmwzo4knhigj12_70da\msg\video\2026-09\cc7a951a7de524ee97690806d8ced53a.mp4'
VIDEO_ID = 'dance_real'


def on_progress(video_id, stage, percent, message):
    print(f'[进度 {percent:3d}%] {stage}: {message}', flush=True)


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f'{VIDEO_ID}.mp4'
    if not dest.exists():
        shutil.copyfile(SRC, str(dest))
        print(f'已复制视频到 {dest}', flush=True)
    else:
        print(f'视频已存在 {dest}，跳过复制', flush=True)

    t0 = time.time()
    orchestrator = PipelineOrchestrator(progress_callback=on_progress)
    metadata = orchestrator.process(dest, video_id=VIDEO_ID)

    print('=' * 64, flush=True)
    print(
        f'video_id={metadata.video_id} fps={metadata.fps} '
        f'total_frames={metadata.total_frames} 耗时={time.time() - t0:.1f}s',
        flush=True,
    )
    print(f'拆解出 {len(metadata.actions)} 个动作段：', flush=True)
    print('=' * 64, flush=True)
    for i, seg in enumerate(metadata.actions, 1):
        print(
            f'动作 {i}: [{seg.start_frame}-{seg.end_frame}] {seg.name} '
            f'({seg.start_time_sec:.1f}s - {seg.end_time_sec:.1f}s)',
            flush=True,
        )
        for field in ALL_ANNOTATION_FIELDS:
            v = getattr(seg.annotations, field)
            if v:
                print(f'    - {field}: {v}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
