import { useEffect, useRef, useState } from 'react'

import {
  getMetadata,
  getPlayback,
  getStatus,
  processVideo,
  uploadVideo,
} from '../api/client'
import { useAppStore } from '../store/useAppStore'

type Phase = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'

/**
 * 视频上传：上传 → 触发处理 → 进度轮询（上传中/处理中/完成/失败四态）。
 * 完成后拉取元数据与播放数据写入全局 store。
 */
export default function VideoUploader() {
  const videoId = useAppStore((s) => s.videoId)
  const setVideoId = useAppStore((s) => s.setVideoId)
  const setStatus = useAppStore((s) => s.setStatus)
  const setMetadata = useAppStore((s) => s.setMetadata)
  const setPlayback = useAppStore((s) => s.setPlayback)
  const setActions = useAppStore((s) => s.setActions)
  const setSelectedActionId = useAppStore((s) => s.setSelectedActionId)
  const reset = useAppStore((s) => s.reset)
  const status = useAppStore((s) => s.status)

  const [phase, setPhase] = useState<Phase>('idle')
  const [fileName, setFileName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const pollingRef = useRef<number | null>(null)

  // 组件卸载时清理轮询
  useEffect(() => {
    return () => {
      if (pollingRef.current !== null) {
        window.clearInterval(pollingRef.current)
      }
    }
  }, [])

  const onCompleted = async (id: string) => {
    try {
      const [metadata, playback] = await Promise.all([getMetadata(id), getPlayback(id)])
      setMetadata(metadata)
      setPlayback(playback)
      setActions(metadata.actions)
      if (metadata.actions.length > 0) {
        setSelectedActionId(metadata.actions[0].action_id)
      }
      setPhase('completed')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setPhase('failed')
    }
  }

  const startPolling = (id: string) => {
    if (pollingRef.current !== null) {
      window.clearInterval(pollingRef.current)
    }
    pollingRef.current = window.setInterval(async () => {
      try {
        const s = await getStatus(id)
        setStatus(s)
        if (s.status === 'completed') {
          if (pollingRef.current !== null) {
            window.clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          await onCompleted(id)
        } else if (s.status === 'failed') {
          if (pollingRef.current !== null) {
            window.clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          setError(s.error ?? '处理失败')
          setPhase('failed')
        }
      } catch (err) {
        if (pollingRef.current !== null) {
          window.clearInterval(pollingRef.current)
          pollingRef.current = null
        }
        setError(err instanceof Error ? err.message : String(err))
        setPhase('failed')
      }
    }, 1500)
  }

  const handleUpload = async () => {
    const input = document.getElementById('video-file-input') as HTMLInputElement | null
    const file = input?.files?.[0]
    if (!file) {
      setError('请先选择视频文件')
      return
    }

    reset()
    setError(null)
    setFileName(file.name)
    setPhase('uploading')

    try {
      const up = await uploadVideo(file)
      setVideoId(up.video_id)
      setStatus({
        video_id: up.video_id,
        status: 'uploaded',
        progress: { stage: 'uploaded', percent: 0, message: '已上传' },
        error: null,
      })

      await processVideo(up.video_id)
      setPhase('processing')
      startPolling(up.video_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setPhase('failed')
    }
  }

  const percent = status?.progress.percent ?? 0

  return (
    <div className="flex items-center gap-3">
      <input
        id="video-file-input"
        type="file"
        accept="video/mp4,video/quicktime,video/x-m4v,video/x-msvideo,video/webm,.mp4,.mov,.m4v,.avi,.webm"
        className="hidden"
        onChange={(e) => setFileName(e.target.files?.[0]?.name ?? '')}
      />

      <label
        htmlFor="video-file-input"
        className="cursor-pointer rounded border border-neutral-600 px-3 py-1.5 text-sm text-neutral-200 transition-colors hover:bg-neutral-800"
      >
        选择视频
      </label>

      <button
        type="button"
        onClick={handleUpload}
        disabled={phase === 'uploading' || phase === 'processing'}
        className="rounded bg-violet-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
      >
        {phase === 'uploading' ? '上传中…' : '上传并拆解'}
      </button>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        {phase === 'processing' && (
          <>
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-neutral-800">
              <div
                className="h-full bg-violet-500 transition-all"
                style={{ width: `${percent}%` }}
              />
            </div>
            <span className="whitespace-nowrap text-xs text-neutral-300">
              {status?.progress.message ?? '处理中'} {percent}%
            </span>
          </>
        )}
        {phase === 'completed' && (
          <span className="text-xs text-emerald-400">处理完成，可查看动作与 3D 骨架</span>
        )}
        {phase === 'failed' && error && <span className="text-xs text-red-400">{error}</span>}
        {fileName && phase !== 'processing' && (
          <span className="truncate text-xs text-neutral-500">{fileName}</span>
        )}
      </div>

      <div className="w-44 shrink-0 truncate rounded border border-neutral-800 px-2 py-1.5 text-xs text-neutral-400">
        {videoId ?? ''}
      </div>
    </div>
  )
}
