/**
 * 前端与后端共用的数据类型定义（对齐 backend/app/models/schemas.py）。
 *
 * 统一时间轴铁律：全系统以整数帧号 currentFrame 为唯一主键，
 * 时间戳秒 = frame / fps。
 */

/** 处理状态 */
export type ProcessingStatus = 'uploaded' | 'processing' | 'completed' | 'failed'

export interface ProgressInfo {
  stage: string
  percent: number
  message: string
}

export interface VideoStatus {
  video_id: string
  status: ProcessingStatus
  progress: ProgressInfo
  error: string | null
}

export interface UploadResult {
  video_id: string
  filename: string
  status: string
  size_bytes: number
}

/** 17 维动作注解（PRD 11 维基础 + 6 维 K-pop 专属） */
export interface Annotations {
  force_point: string
  body_posture: string
  rhythm: string
  common_mistakes: string
  killing_point: string
  sharpness: string
  power_groove: string
  beat_accent: string
  weight_transfer: string
  joint_angle: string
  breathing: string
  music_beat: string
  transition: string
  safety: string
  difficulty: string
  facial_expression: string
  wave_isolation: string
  [key: string]: string
}

export interface AnimationRef {
  clip_name: string
  start_frame: number
  end_frame: number
}

export interface ActionSegment {
  action_id: string
  name: string
  start_frame: number
  end_frame: number
  start_time_sec: number
  end_time_sec: number
  annotations: Annotations
  annotation_source: string
  keypoints_ref?: string | null
  animation_ref?: AnimationRef | null
}

export interface SegmentAdjustment {
  action_id: string
  start_frame: number
  end_frame: number
}

export interface KeypointCorrection {
  action_id: string
  frame: number
  joint: string
  value: number[]
}

export interface CalibrationData {
  status: string
  segment_adjustments: SegmentAdjustment[]
  keypoint_corrections: KeypointCorrection[]
  calibrated_at?: string | null
  calibrated_by: string
}

export interface ModelSummary {
  name: string
  file: string
  joints: string[]
  retarget_map: Record<string, string>
}

export interface ActionMetadata {
  video_id: string
  fps: number
  duration_sec: number
  total_frames: number
  model: ModelSummary
  actions: ActionSegment[]
  calibration: CalibrationData
  keypoints_file: string
  animation_file: string
}

export interface AnimationClip {
  clip_name: string
  start_frame: number
  end_frame: number
  joints: string[]
  joint_rotations: Record<string, number[][]>
  root_positions: number[][]
}

export interface PlaybackData {
  video_id: string
  fps: number
  total_frames: number
  duration_sec: number
  video_url: string
  model: ModelSummary
  actions: ActionSegment[]
  calibration: CalibrationData
  animation: AnimationClip
  keypoints: {
    file: string
    num_keypoints: number
    total_frames: number
  }
}

export interface FrameKeypoints {
  video_id: string
  frame: number
  keypoints: number[][]
}

export interface AllKeypoints {
  video_id: string
  total_frames: number
  num_keypoints: number
  keypoints: number[][][]
}

// ---------------------------------------------------------------------------
// 17 维注解分组（P0 共 8 维 / P1 共 9 维），供 AnnotationEditor 渲染
// ---------------------------------------------------------------------------
export interface AnnotationField {
  key: keyof Annotations
  label: string
}

/** P0：基础 4 维 + K-pop 核心 4 维 */
export const P0_ANNOTATION_FIELDS: AnnotationField[] = [
  { key: 'force_point', label: '发力点' },
  { key: 'body_posture', label: '身体姿态' },
  { key: 'rhythm', label: '节奏' },
  { key: 'common_mistakes', label: '常见错误' },
  { key: 'killing_point', label: '记忆点动作' },
  { key: 'sharpness', label: '整齐度/锐利度' },
  { key: 'power_groove', label: '力度与爆发' },
  { key: 'beat_accent', label: '节拍卡点' },
]

/** P1：其余 9 维 */
export const P1_ANNOTATION_FIELDS: AnnotationField[] = [
  { key: 'weight_transfer', label: '重心转移' },
  { key: 'joint_angle', label: '关节角度/幅度' },
  { key: 'breathing', label: '呼吸' },
  { key: 'music_beat', label: '音乐卡点' },
  { key: 'transition', label: '动作衔接' },
  { key: 'safety', label: '安全提示' },
  { key: 'difficulty', label: '难度等级' },
  { key: 'facial_expression', label: '表情管理' },
  { key: 'wave_isolation', label: 'wave/isolation' },
]

/** 空注解模板（新增/复位时使用） */
export function emptyAnnotations(): Annotations {
  return {
    force_point: '',
    body_posture: '',
    rhythm: '',
    common_mistakes: '',
    killing_point: '',
    sharpness: '',
    power_groove: '',
    beat_accent: '',
    weight_transfer: '',
    joint_angle: '',
    breathing: '',
    music_beat: '',
    transition: '',
    safety: '',
    difficulty: '',
    facial_expression: '',
    wave_isolation: '',
  }
}
