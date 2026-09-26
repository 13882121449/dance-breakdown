import axios, { type AxiosInstance, type AxiosResponse } from 'axios'

import type {
  ActionMetadata,
  ActionSegment,
  AllKeypoints,
  CalibrationData,
  FrameKeypoints,
  PlaybackData,
  ProgressInfo,
  UploadResult,
  VideoStatus,
} from './types'

/** 后端 API 基地址：优先取环境变量，默认指向本地 FastAPI */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/**
 * 全项目统一 API 响应格式：{ code, message, data }
 * code === 0 表示成功；code !== 0 表示业务错误。
 */
export interface ApiResponse<T = unknown> {
  code: number
  message: string
  data: T
}

/** 封装后的 axios 实例，所有请求统一挂载到后端基地址 */
export const client: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * 解包统一响应：code !== 0 时抛出 Error（message 为后端提示）。
 * 供所有 API 函数复用，避免重复解析。
 */
export async function unwrap<T>(promise: Promise<AxiosResponse<ApiResponse<T>>>): Promise<T> {
  const res = await promise
  if (res.data.code !== 0) {
    throw new Error(res.data.message || '请求失败')
  }
  return res.data.data
}

/** 把后端返回的相对 URL（如 /api/video/x/file）拼成绝对地址，供 <video src> 使用 */
export function absoluteUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path
  return `${API_BASE_URL.replace(/\/+$/, '')}${path.startsWith('/') ? path : `/${path}`}`
}

// ---------------------------------------------------------------------------
// API 函数（对齐 T04 契约）
// ---------------------------------------------------------------------------
export function uploadVideo(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  return unwrap(
    client.post('/api/video/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  )
}

export function processVideo(videoId: string): Promise<{ video_id: string; status: string }> {
  return unwrap(client.post(`/api/video/${videoId}/process`))
}

export function getStatus(videoId: string): Promise<VideoStatus> {
  return unwrap(client.get(`/api/video/${videoId}/status`))
}

export function getProgress(videoId: string): Promise<ProgressInfo> {
  return unwrap(client.get(`/api/video/${videoId}/progress`))
}

export function getMetadata(videoId: string): Promise<ActionMetadata> {
  return unwrap(client.get(`/api/video/${videoId}/metadata`))
}

export function getActions(videoId: string): Promise<ActionSegment[]> {
  return unwrap(client.get(`/api/video/${videoId}/actions`))
}

export function patchAction(
  videoId: string,
  actionId: string,
  body: { annotations?: Record<string, string>; name?: string },
): Promise<ActionSegment> {
  return unwrap(client.patch(`/api/video/${videoId}/actions/${actionId}`, body))
}

export function getPlayback(videoId: string): Promise<PlaybackData> {
  return unwrap(client.get(`/api/playback/${videoId}`))
}

export function getKeypoints(videoId: string, frame: number): Promise<FrameKeypoints> {
  return unwrap(client.get(`/api/playback/${videoId}/keypoints/${frame}`))
}

export function getAllKeypoints(videoId: string): Promise<AllKeypoints> {
  return unwrap(client.get(`/api/playback/${videoId}/keypoints`))
}

export function getCalibration(videoId: string): Promise<CalibrationData> {
  return unwrap(client.get(`/api/video/${videoId}/calibration`))
}

export function postCalibration(
  videoId: string,
  calibration: CalibrationData,
): Promise<ActionMetadata> {
  return unwrap(client.post(`/api/video/${videoId}/calibration`, calibration))
}

export default client
