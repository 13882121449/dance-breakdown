import { lazy, Suspense } from 'react'

import ActionList from './components/ActionList'
import AnnotationEditor from './components/AnnotationEditor'
import CalibrationEditor from './components/CalibrationEditor'
import VideoPlayer from './components/VideoPlayer'
import VideoUploader from './components/VideoUploader'
import { useAppStore } from './store/useAppStore'

// 3D 骨架依赖 three.js（体积大），懒加载以加快首屏渲染
const Pose3DViewer = lazy(() => import('./components/Pose3DViewer'))

/** 帧号格式化为 mm:ss.s */
function formatTime(frame: number, fps: number): string {
  const sec = Math.max(0, frame / (fps || 30))
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

/**
 * 应用主布局：
 * 左 ActionList / 中 3D 骨架 + 视频 / 右 AnnotationEditor / 底时间轴。
 * 全系统以整数帧号 currentFrame 为唯一主键驱动三者对齐。
 */
function App() {
  const currentFrame = useAppStore((s) => s.currentFrame)
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame)
  const videoId = useAppStore((s) => s.videoId)
  const metadata = useAppStore((s) => s.metadata)

  const fps = metadata?.fps ?? 30
  const totalFrames = metadata?.total_frames ?? 0
  const maxFrame = Math.max(0, totalFrames - 1)

  return (
    <div className="flex h-screen flex-col bg-neutral-950 text-neutral-100">
      {/* 顶栏 */}
      <div className="flex items-center gap-4 border-b border-neutral-800 px-4 py-2">
        <div className="whitespace-nowrap text-lg font-bold text-white">舞蹈动作拆解</div>
        <VideoUploader />
      </div>

      {/* 主体：左列表 / 中 3D+视频 / 右注解 */}
      <div className="flex min-h-0 flex-1">
        <div className="w-72 shrink-0 border-r border-neutral-800 bg-neutral-900/40">
          <div className="border-b border-neutral-800 px-3 py-2 text-sm text-neutral-400">
            动作列表
          </div>
          <ActionList />
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 border-b border-neutral-800">
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center text-sm text-neutral-500">
                  3D 视图加载中…
                </div>
              }
            >
              <Pose3DViewer />
            </Suspense>
          </div>
          <div className="h-52 shrink-0">
            <VideoPlayer />
          </div>
        </div>

        <div className="w-96 shrink-0 border-l border-neutral-800 bg-neutral-900/40">
          <div className="flex h-full flex-col">
            <div className="min-h-0 flex-1">
              <AnnotationEditor />
            </div>
            <div className="shrink-0 border-t border-neutral-800">
              <CalibrationEditor />
            </div>
          </div>
        </div>
      </div>

      {/* 底时间轴 */}
      <div className="flex items-center gap-3 border-t border-neutral-800 px-4 py-2">
        <span className="whitespace-nowrap text-xs text-neutral-400">
          帧 {currentFrame} / {totalFrames}
        </span>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={Math.min(currentFrame, maxFrame)}
          onChange={(e) => setCurrentFrame(Number(e.target.value))}
          disabled={!videoId || maxFrame <= 0}
          className="flex-1 accent-violet-500"
        />
        <span className="whitespace-nowrap text-xs text-neutral-400">
          {formatTime(currentFrame, fps)}
        </span>
      </div>
    </div>
  )
}

export default App
