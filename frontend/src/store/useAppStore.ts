import { create } from 'zustand'

import type {
  ActionMetadata,
  ActionSegment,
  Annotations,
  PlaybackData,
  VideoStatus,
} from '../api/types'

/**
 * 全局应用状态。
 *
 * 依据架构「统一时间轴铁律」：全系统以整数帧号 currentFrame 为唯一主键，
 * 视频播放器、3D 骨架、动作列表、注解编辑器均通过该帧号驱动对齐。
 */
export interface AppState {
  /** 当前播放帧号（唯一时间轴主键） */
  currentFrame: number
  /** 当前处理的视频 ID，null 表示尚未上传 */
  videoId: string | null
  /** 处理状态（uploaded / processing / completed / failed） */
  status: VideoStatus | null
  /** 动作元数据（处理完成后填充） */
  metadata: ActionMetadata | null
  /** 播放数据（含视频地址 + 动画） */
  playback: PlaybackData | null
  /** 动作段列表（与 metadata.actions 同步） */
  actions: ActionSegment[]
  /** 当前选中的动作 action_id */
  selectedActionId: string | null
  /** 注解草稿（按字段名保存），保存时 PATCH 回后端 */
  annotationDraft: Annotations

  setCurrentFrame: (frame: number) => void
  setVideoId: (id: string | null) => void
  setStatus: (status: VideoStatus | null) => void
  setMetadata: (metadata: ActionMetadata | null) => void
  setPlayback: (playback: PlaybackData | null) => void
  setActions: (actions: ActionSegment[]) => void
  setSelectedActionId: (id: string | null) => void
  setAnnotationDraft: (draft: Annotations) => void
  updateAction: (action: ActionSegment) => void
  reset: () => void
}

function emptyDraft(): Annotations {
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

export const useAppStore = create<AppState>((set) => ({
  currentFrame: 0,
  videoId: null,
  status: null,
  metadata: null,
  playback: null,
  actions: [],
  selectedActionId: null,
  annotationDraft: emptyDraft(),

  setCurrentFrame: (frame) => set({ currentFrame: Math.max(0, Math.round(frame)) }),
  setVideoId: (id) => set({ videoId: id }),
  setStatus: (status) => set({ status }),
  setMetadata: (metadata) => set({ metadata }),
  setPlayback: (playback) => set({ playback }),
  setActions: (actions) => set({ actions }),
  setSelectedActionId: (id) => set({ selectedActionId: id }),
  setAnnotationDraft: (draft) => set({ annotationDraft: draft }),
  updateAction: (action) =>
    set((state) => ({
      actions: state.actions.map((a) => (a.action_id === action.action_id ? action : a)),
    })),
  reset: () =>
    set({
      currentFrame: 0,
      videoId: null,
      status: null,
      metadata: null,
      playback: null,
      actions: [],
      selectedActionId: null,
      annotationDraft: emptyDraft(),
    }),
}))
