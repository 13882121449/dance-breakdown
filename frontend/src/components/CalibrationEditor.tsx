import { useEffect, useState, type ReactElement } from 'react'

import { getActions, getMetadata, postCalibration } from '../api/client'
import { useAppStore } from '../store/useAppStore'

/**
 * 校准编辑器：调整选中动作段的起止帧（切分点），提交后回写校准数据。
 * 关键点修正（3D 拖动）留待后续版本，本版仅支持切分点调整。
 */
export default function CalibrationEditor(): ReactElement {
  const videoId = useAppStore((s) => s.videoId)
  const actions = useAppStore((s) => s.actions)
  const selectedActionId = useAppStore((s) => s.selectedActionId)
  const currentFrame = useAppStore((s) => s.currentFrame)
  const metadata = useAppStore((s) => s.metadata)
  const setMetadata = useAppStore((s) => s.setMetadata)
  const setActions = useAppStore((s) => s.setActions)

  const action = actions.find((a) => a.action_id === selectedActionId) ?? null

  const [startFrame, setStartFrame] = useState(0)
  const [endFrame, setEndFrame] = useState(0)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const maxFrame = Math.max(0, (metadata?.total_frames ?? 0) - 1)

  // 选中动作或边界变化时，用当前边界初始化输入
  useEffect(() => {
    setStartFrame(action?.start_frame ?? 0)
    setEndFrame(action?.end_frame ?? 0)
    setMessage(null)
  }, [selectedActionId, action?.start_frame, action?.end_frame])

  if (!videoId || !action) {
    return (
      <div className="px-4 py-3 text-xs text-neutral-500">
        选中动作后可在此调整切分点
      </div>
    )
  }

  const handleSubmit = async () => {
    if (startFrame >= endFrame) {
      setMessage('起点必须小于终点')
      return
    }
    setSaving(true)
    setMessage(null)
    try {
      await postCalibration(videoId, {
        status: 'pending',
        segment_adjustments: [
          { action_id: action.action_id, start_frame: startFrame, end_frame: endFrame },
        ],
        keypoint_corrections: [],
        calibrated_by: 'local_user',
      })
      // 校准已回写，刷新元数据与动作列表（切分点更新）
      const [meta, acts] = await Promise.all([getMetadata(videoId), getActions(videoId)])
      setMetadata(meta)
      setActions(acts)
      setMessage('已保存校准')
    } catch (err) {
      setMessage(`保存失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="p-4">
      <div className="mb-3 text-sm font-semibold text-white">校准 · 切分点</div>

      <div className="mb-2 truncate text-xs text-neutral-400">
        动作：{action.name}
      </div>

      <div className="flex items-center gap-2">
        <label className="w-12 shrink-0 text-xs text-neutral-400">起点</label>
        <input
          type="number"
          min={0}
          max={maxFrame}
          value={startFrame}
          onChange={(e) => setStartFrame(Number(e.target.value))}
          className="w-full rounded border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm text-neutral-100 outline-none focus:border-violet-500"
        />
        <button
          type="button"
          onClick={() => setStartFrame(currentFrame)}
          className="shrink-0 rounded border border-neutral-600 px-2 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          当前帧
        </button>
      </div>

      <div className="mt-2 flex items-center gap-2">
        <label className="w-12 shrink-0 text-xs text-neutral-400">终点</label>
        <input
          type="number"
          min={0}
          max={maxFrame}
          value={endFrame}
          onChange={(e) => setEndFrame(Number(e.target.value))}
          className="w-full rounded border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm text-neutral-100 outline-none focus:border-violet-500"
        />
        <button
          type="button"
          onClick={() => setEndFrame(currentFrame)}
          className="shrink-0 rounded border border-neutral-600 px-2 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          当前帧
        </button>
      </div>

      <button
        type="button"
        onClick={handleSubmit}
        disabled={saving}
        className="mt-3 w-full rounded bg-violet-600 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
      >
        {saving ? '保存中…' : '提交校准'}
      </button>

      {message && (
        <div
          className={`mt-2 text-xs ${
            message.startsWith('保存失败') ? 'text-red-400' : 'text-emerald-400'
          }`}
        >
          {message}
        </div>
      )}
    </div>
  )
}
