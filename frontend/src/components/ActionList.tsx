import type { ReactElement } from 'react'

import { useAppStore } from '../store/useAppStore'

/** 把秒格式化为 mm:ss.s */
function formatTime(sec: number): string {
  const s = Math.max(0, sec)
  const m = Math.floor(s / 60)
  const rest = (s % 60).toFixed(1)
  return `${m}:${rest.padStart(4, '0')}`
}

/**
 * 动作列表：展示名称/起止时间/注解摘要。
 * 点击动作 → 选中并 seekTo(start_frame)；当前帧所在动作自动高亮。
 */
export default function ActionList(): ReactElement {
  const actions = useAppStore((s) => s.actions)
  const currentFrame = useAppStore((s) => s.currentFrame)
  const selectedActionId = useAppStore((s) => s.selectedActionId)
  const setSelectedActionId = useAppStore((s) => s.setSelectedActionId)
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame)

  if (actions.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-sm text-neutral-500">
        处理完成后，动作列表将在此显示
      </div>
    )
  }

  return (
    <ul className="h-full divide-y divide-neutral-800 overflow-y-auto">
      {actions.map((action) => {
        const active =
          currentFrame >= action.start_frame && currentFrame < action.end_frame
        const selected = action.action_id === selectedActionId
        const summary = [
          `${formatTime(action.start_time_sec)} - ${formatTime(action.end_time_sec)}`,
          action.annotations.difficulty,
          action.annotations.rhythm,
        ]
          .filter(Boolean)
          .join(' · ')

        return (
          <li key={action.action_id}>
            <button
              type="button"
              onClick={() => {
                setSelectedActionId(action.action_id)
                setCurrentFrame(action.start_frame)
              }}
              className={`w-full px-3 py-2 text-left transition-colors ${
                active || selected
                  ? 'bg-violet-500/20 text-white'
                  : 'text-neutral-200 hover:bg-neutral-800/60'
              }`}
            >
              <div className="text-sm font-semibold">{action.name}</div>
              <div className="mt-0.5 truncate text-xs text-neutral-400">{summary}</div>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
