import { useEffect, useRef } from 'react'

import { absoluteUrl } from '../api/client'
import { useAppStore } from '../store/useAppStore'

/**
 * 视频播放器：`<video>` 与 currentFrame 双向同步。
 *
 * - 视频 timeupdate → setCurrentFrame（用户拖动/播放时更新帧号）
 * - currentFrame 变化（点动作/拖时间轴）→ seek 视频
 */
export default function VideoPlayer() {
  const videoId = useAppStore((s) => s.videoId)
  const playback = useAppStore((s) => s.playback)
  const currentFrame = useAppStore((s) => s.currentFrame)
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame)
  const videoRef = useRef<HTMLVideoElement>(null)

  const fps = playback?.fps ?? 30

  // store → video：外部改帧号时 seek 视频
  useEffect(() => {
    const video = videoRef.current
    if (!video || !playback) return
    const target = currentFrame / fps
    if (Math.abs(video.currentTime - target) > 0.03) {
      video.currentTime = target
    }
  }, [currentFrame, fps, playback])

  // video → store：播放/拖动时同步帧号
  const handleTimeUpdate = () => {
    const video = videoRef.current
    if (!video) return
    setCurrentFrame(Math.round(video.currentTime * fps))
  }

  if (!videoId || !playback) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-neutral-900 text-sm text-neutral-500">
        视频预览区
      </div>
    )
  }

  return (
    <div className="h-full w-full bg-black">
      <video
        ref={videoRef}
        src={absoluteUrl(playback.video_url)}
        controls
        playsInline
        onTimeUpdate={handleTimeUpdate}
        className="h-full w-full object-contain"
      />
    </div>
  )
}
