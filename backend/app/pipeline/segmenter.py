"""动作切分：K-pop「节拍为主 + 能量峰辅助」。

主策略（architecture.md 2.5）：
- librosa 节拍/onset 为切分锚点，动作段边界吸附到最近节拍帧；
- 关键点能量峰作为无节拍/长片段兜底与细分依据；
- 副歌重复段检测（关键点相似度）用于定位 killing point。

统一时间轴铁律：切分只产出整数帧号边界，**不改变帧号**，动作段无缝覆盖全片
（``actions[i].end_frame == actions[i+1].start_frame``）。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.models.schemas import ActionSegment, Annotations, KeypointSequence

# 动作时长约束（秒）
MIN_ACTION_SECONDS: float = 0.5
MAX_ACTION_SECONDS: float = 8.0
# 默认「一个动作 = 4 个整拍」（4/4 一小节）
BEATS_PER_ACTION: int = 4
# 副歌重复段相似度阈值（余弦相似度）
REPEAT_SIMILARITY_THRESHOLD: float = 0.9

# 用于能量/相似度计算的关节索引（手腕/脚踝权重更高）
_ENERGY_JOINTS: dict[int, float] = {
    15: 2.0,  # L_Wrist
    16: 2.0,  # R_Wrist
    27: 2.0,  # L_Ankle
    28: 2.0,  # R_Ankle
    13: 1.5,  # L_Elbow
    14: 1.5,  # R_Elbow
    25: 1.5,  # L_Knee
    26: 1.5,  # R_Knee
    11: 1.0,  # L_Shoulder
    12: 1.0,  # R_Shoulder
    23: 1.0,  # L_Hip
    24: 1.0,  # R_Hip
}
# 相似度签名的关节索引（肩/肘/腕/髋/膝/踝）
_SIGNATURE_JOINTS: list[int] = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class ActionSegmenter:
    """动作切分器（节拍为主，能量峰辅助）。"""

    def __init__(
        self,
        beats_per_action: int = BEATS_PER_ACTION,
        min_action_seconds: float = MIN_ACTION_SECONDS,
        max_action_seconds: float = MAX_ACTION_SECONDS,
    ) -> None:
        self.beats_per_action: int = beats_per_action
        self.min_action_seconds: float = min_action_seconds
        self.max_action_seconds: float = max_action_seconds

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def segment(
        self,
        kps: KeypointSequence,
        fps: int,
        beats: Optional[list[int]] = None,
        onsets: Optional[list[int]] = None,
    ) -> list[ActionSegment]:
        """对关键点序列切分，返回动作段列表（无缝覆盖全片）。

        Args:
            kps: 平滑后关键点序列（``[T, 33, 4]``）。
            fps: 帧率（用于把秒换算成帧）。
            beats: 节拍帧号列表（librosa 产出，可缺省）。
            onsets: 强拍/重音帧号列表（librosa 产出，可缺省）。
        """
        t = kps.total_frames
        if t <= 0:
            return []

        min_frames = max(1, int(round(self.min_action_seconds * fps)))
        max_frames = max(min_frames * 2, int(round(self.max_action_seconds * fps)))

        beats = self._clamp_sorted_unique(beats, t)
        onsets = self._clamp_sorted_unique(onsets, t)

        energy = self.compute_energy(kps)

        # 主策略：节拍切分；兜底：能量峰
        if len(beats) >= 2:
            boundaries = self._bar_boundaries(beats)
            boundaries = self._split_long_bars(
                boundaries, energy, beats, max_frames
            )
        else:
            boundaries = self._detect_energy_boundaries(kps, min_frames)

        # 无缝覆盖：首尾固定为 0 与 T-1，中间边界递增
        boundaries = self._finalize_boundaries(boundaries, t, min_frames)

        # 副歌重复段（killing point 定位）
        repeats = self.detect_repeated_segments(kps, fps, beats)

        segments: list[ActionSegment] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            seg = self._build_segment(i, start, end, fps, beats, onsets, repeats)
            segments.append(seg)
        return segments

    # ------------------------------------------------------------------
    # 能量与边界
    # ------------------------------------------------------------------
    def compute_energy(self, kps: KeypointSequence) -> np.ndarray:
        """逐帧关节运动能量 ``E[t] = Σ_j w_j·||p_j[t+1] - p_j[t]||``。

        返回长度 ``T`` 的一维数组（末帧能量回填为上一帧）。
        """
        data = kps.data
        t = data.shape[0]
        if t < 2:
            return np.zeros(t, dtype=np.float64)

        # 2D 像素位移（x,y），z 为米制相对深度、尺度不同，不混入能量
        pos = data[:, :, 0:2].astype(np.float64)
        diff = np.diff(pos, axis=0)  # [T-1, 33, 2]
        speed = np.linalg.norm(diff, axis=2)  # [T-1, 33]

        weights = np.ones(kps.num_keypoints, dtype=np.float64)
        for j, w in _ENERGY_JOINTS.items():
            if j < len(weights):
                weights[j] = w

        energy = np.sum(speed * weights[None, :], axis=1)  # [T-1]
        energy = np.concatenate([energy, energy[-1:]]) if t > 1 else energy
        return energy

    def snap_to_beat(self, boundary: int, beats: list[int]) -> int:
        """把候选边界吸附到最近节拍帧：``argmin_b |b - boundary|``。"""
        if not beats:
            return boundary
        beats_arr = np.asarray(beats, dtype=np.int64)
        idx = int(np.argmin(np.abs(beats_arr - boundary)))
        return int(beats_arr[idx])

    def detect_beats(self, audio_path: Any, fps: int, sr: Optional[int] = None) -> list[int]:
        """用 librosa 提取节拍时间点，换算为整数帧号。"""
        import librosa

        y, sr = librosa.load(str(audio_path), sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beat_frames, sr=sr)
        return sorted({int(round(t * fps)) for t in times if t >= 0})

    def detect_onsets(self, audio_path: Any, fps: int, sr: Optional[int] = None) -> list[int]:
        """用 librosa 提取 onset/强拍重音点，换算为整数帧号。"""
        import librosa

        y, sr = librosa.load(str(audio_path), sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        times = librosa.frames_to_time(onset_frames, sr=sr)
        return sorted({int(round(t * fps)) for t in times if t >= 0})

    # ------------------------------------------------------------------
    # 副歌重复段（killing point）
    # ------------------------------------------------------------------
    def detect_repeated_segments(
        self,
        kps: KeypointSequence,
        fps: int,
        beats: Optional[list[int]] = None,
    ) -> list[tuple[int, int]]:
        """检测副歌重复段，返回重复出现的 ``(start_frame, end_frame)`` 列表。"""
        t = kps.total_frames
        if t < 2:
            return []

        bars = self._make_bars(beats, fps, t)
        if len(bars) < 2:
            return []

        sigs = [self._bar_signature(kps, s, e) for s, e in bars]
        if len(sigs) < 2:
            return []

        best_pair: Optional[tuple[int, int]] = None
        best_sim = -1.0
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                sim = self._cosine_similarity(sigs[i], sigs[j])
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)

        if best_pair is None or best_sim < REPEAT_SIMILARITY_THRESHOLD:
            return []

        i, j = best_pair
        return [tuple(bars[i]), tuple(bars[j])]  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _clamp_sorted_unique(
        self, values: Optional[list[int]], total_frames: int
    ) -> list[int]:
        # 允许 total_frames（作为最后一小节的排他右边界），边界最终会在
        # _finalize_boundaries 中收敛到 [0, total_frames-1]。
        if not values:
            return []
        return sorted({int(v) for v in values if 0 <= int(v) <= total_frames})

    def _bar_boundaries(self, beats: list[int]) -> list[int]:
        """以每 ``beats_per_action`` 个节拍为一个动作，产出边界帧。"""
        step = max(1, self.beats_per_action)
        boundaries = beats[::step]
        if not boundaries or boundaries[0] != 0:
            boundaries = [0] + [b for b in boundaries if b > 0]
        return boundaries

    def _split_long_bars(
        self,
        boundaries: list[int],
        energy: np.ndarray,
        beats: list[int],
        max_frames: int,
    ) -> list[int]:
        """对超长动作段，在其内部能量局部极小值处补充切分（并吸附到节拍）。"""
        result: list[int] = []
        boundaries = sorted(set(boundaries))
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            result.append(s)
            if e - s > max_frames:
                extra = self._energy_minimum_in_range(energy, s, e)
                if extra is not None and s < extra < e:
                    snapped = self.snap_to_beat(extra, beats)
                    if s < snapped < e:
                        result.append(snapped)
        result.append(boundaries[-1])
        return sorted(set(result))

    def _detect_energy_boundaries(
        self, kps: KeypointSequence, min_frames: int
    ) -> list[int]:
        """能量峰兜底：找「停顿/换动作」的局部极小值作为边界。"""
        energy = self.compute_energy(kps)
        t = len(energy)
        if t < 3:
            return []

        # 移动平均平滑去噪
        win = max(3, min(7, t if t % 2 == 1 else t - 1))
        smoothed = np.convolve(energy, np.ones(win) / win, mode="same")

        mean = float(np.mean(smoothed)) + 1e-9
        boundaries: list[int] = []
        for i in range(1, t - 1):
            is_min = smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]
            is_pause = smoothed[i] < mean * 0.5
            if is_min and is_pause:
                boundaries.append(i)

        # 最小动作长度过滤
        return self._filter_spacing(boundaries, min_frames)

    def _energy_minimum_in_range(
        self, energy: np.ndarray, s: int, e: int
    ) -> Optional[int]:
        if e - s < 3:
            return None
        seg = energy[s + 1 : e]
        if seg.size == 0:
            return None
        idx = int(np.argmin(seg))
        return s + 1 + idx

    def _finalize_boundaries(
        self, boundaries: list[int], total_frames: int, min_frames: int
    ) -> list[int]:
        """保证首尾 0/T-1、递增、无缝、最小动作长度。"""
        t = total_frames
        result = [0] + sorted({b for b in boundaries if 0 < b < t - 1}) + [t - 1]
        result = sorted(set(result))
        # 合并过短段：向后合并相邻过短段
        result = self._filter_spacing(result, min_frames, keep_ends=True)
        if len(result) < 2:
            return [0, max(1, t - 1)]
        result[0] = 0
        result[-1] = t - 1
        return result

    def _filter_spacing(
        self, boundaries: list[int], min_frames: int, keep_ends: bool = False
    ) -> list[int]:
        if not boundaries:
            return []
        if keep_ends:
            # 首尾边界必须保留（保证无缝覆盖全片）
            first, last = boundaries[0], boundaries[-1]
            out: list[int] = [first]
            for b in sorted(boundaries[1:-1]):
                if b - out[-1] >= min_frames:
                    out.append(b)
            if out[-1] != last:
                out.append(last)
            return out
        out = []
        for b in sorted(boundaries):
            if not out or b - out[-1] >= min_frames:
                out.append(b)
        return out

    def _make_bars(
        self, beats: Optional[list[int]], fps: int, total_frames: int
    ) -> list[tuple[int, int]]:
        if beats and len(beats) >= 2:
            pts = sorted({int(b) for b in beats if 0 <= int(b) <= total_frames})
            # 保证最后一小节覆盖到片尾
            if pts and pts[-1] < total_frames:
                pts.append(total_frames)
            bars: list[tuple[int, int]] = []
            for i in range(len(pts) - 1):
                s, e = pts[i], pts[i + 1]
                if e - s >= 2:
                    bars.append((s, e))
            return bars
        # 无节拍：固定 2 秒窗口兜底
        window = max(2, int(round(2.0 * fps)))
        return [(i, min(i + window, total_frames)) for i in range(0, total_frames, window)]

    def _bar_signature(
        self, kps: KeypointSequence, start: int, end: int
    ) -> np.ndarray:
        """小节的姿态签名：对肩/肘/腕/髋/膝/踝取均值，做平移+尺度归一化。"""
        seg = kps.data[start:end, :, 0:2].astype(np.float64)  # [n, 33, 2]
        mean = seg.mean(axis=0)  # [33, 2]
        # 平移：以髋中点（23/24 均值）为原点
        hip = (mean[23] + mean[24]) / 2.0
        centered = mean - hip
        # 尺度：以躯干长度（肩中点 11/12 到髋中点）归一化
        shoulder = (mean[11] + mean[12]) / 2.0
        scale = np.linalg.norm(shoulder - hip) + 1e-6
        normed = centered / scale
        return normed[_SIGNATURE_JOINTS].ravel()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _build_segment(
        self,
        index: int,
        start: int,
        end: int,
        fps: int,
        beats: list[int],
        onsets: list[int],
        repeats: list[tuple[int, int]],
    ) -> ActionSegment:
        """组装一个动作段，并填充自动可检测的 K-pop 注解维度。"""
        n_beats = sum(1 for b in beats if start <= b <= end)
        n_onsets = sum(1 for o in onsets if start <= o <= end)

        ann = Annotations()
        # 节奏：本动作覆盖的节拍数
        if n_beats > 0:
            ann.rhythm = f"约 {n_beats} 拍完成"
        # 节拍卡点：动作内是否含强拍重音
        if n_onsets > 0:
            ann.beat_accent = f"动作落点卡在 {n_onsets} 个强拍重音上"
        # killing point：是否落在副歌重复段内
        for rs, re in repeats:
            if start < re and end > rs:
                ann.killing_point = "副歌反复的招牌记忆点动作"
                break

        return ActionSegment(
            action_id=f"act_{index + 1:03d}",
            name=f"动作 {index + 1}",
            start_frame=start,
            end_frame=end,
            start_time_sec=round(start / fps, 3),
            end_time_sec=round(end / fps, 3),
            annotations=ann,
            annotation_source="template",
        )
