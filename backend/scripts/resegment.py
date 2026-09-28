"""只重跑动作切分（复用已有关键点/动画，用 ffmpeg 重新检测节拍）。"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 让 audioread/librosa 能找到 imageio-ffmpeg 提供的 ffmpeg 二进制
import imageio_ffmpeg

_ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ['PATH'] = _ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

from app.config import KEYPOINTS_DIR, ANIMATIONS_DIR, METADATA_DIR, UPLOAD_DIR
from app.services import storage
from app.models.schemas import VideoMeta
from app.pipeline.segmenter import ActionSegmenter
from app.pipeline.metadata_builder import MetadataBuilder

VIDEO_ID = 'dance_real'


def main():
    kps = storage.load_keypoints(KEYPOINTS_DIR, VIDEO_ID, smoothed=True)
    clip = storage.load_animation(ANIMATIONS_DIR, VIDEO_ID)
    old = storage.load_metadata(METADATA_DIR, VIDEO_ID)

    video_meta = VideoMeta(
        video_id=VIDEO_ID, fps=old.fps, duration_sec=old.duration_sec,
        total_frames=old.total_frames, status='frames_extracted',
    )

    seg = ActionSegmenter()
    video_path = UPLOAD_DIR / f'{VIDEO_ID}.mp4'

    print('提取音频...', flush=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_wav = Path(tempfile.mkdtemp()) / 'audio.wav'
    subprocess.run(
        [ffmpeg_exe, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '22050', '-ac', '1', '-y', str(tmp_wav)],
        check=True, capture_output=True,
    )

    print('检测节拍/重音...', flush=True)
    beats = seg.detect_beats(tmp_wav, old.fps)
    onsets = seg.detect_onsets(tmp_wav, old.fps)
    print(f'beats={len(beats)} onsets={len(onsets)}', flush=True)

    segments = seg.segment(kps, old.fps, beats, onsets)
    print(f'重新切分: {len(segments)} 个动作段', flush=True)

    meta = MetadataBuilder().build(
        video_id=VIDEO_ID, video_meta=video_meta, segments=segments, clip=clip,
        kps_path=old.keypoints_file, anim_path=old.animation_file,
    )
    MetadataBuilder().save(meta, METADATA_DIR / f'{VIDEO_ID}.json')

    print('=' * 64, flush=True)
    print(f'共 {len(meta.actions)} 个动作段:', flush=True)
    for i, s in enumerate(meta.actions, 1):
        line = f'{i}. [{s.start_frame}-{s.end_frame}] ({s.start_time_sec:.1f}s-{s.end_time_sec:.1f}s)'
        auto = []
        for f in ['rhythm', 'beat_accent', 'killing_point']:
            v = getattr(s.annotations, f)
            if v:
                auto.append(v)
        if auto:
            line += '  |  ' + '；'.join(auto)
        print(line, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
