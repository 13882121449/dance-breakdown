import { useEffect, useState, type ReactElement } from 'react'

import { patchAction } from '../api/client'
import {
  P0_ANNOTATION_FIELDS,
  P1_ANNOTATION_FIELDS,
  emptyAnnotations,
  type Annotations,
  type AnnotationField,
} from '../api/types'
import { useAppStore } from '../store/useAppStore'

/** 单字段输入（纯 HTML textarea + Tailwind） */
function FieldInput({
  field,
  value,
  onChange,
}: {
  field: AnnotationField
  value: string
  onChange: (value: string) => void
}): ReactElement {
  return (
    <label className="mb-3 block">
      <span className="mb-1 block text-xs text-neutral-400">{field.label}</span>
      <textarea
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        rows={2}
        className="w-full resize-y rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-violet-500"
      />
    </label>
  )
}

/** 分组标题 */
function GroupTitle({ children }: { children: string }): ReactElement {
  return <div className="mb-2 mt-4 text-sm font-semibold text-violet-300">{children}</div>
}

/**
 * 注解编辑器：展示/编辑选中动作的 17 维注解（P0 8 维 / P1 9 维），
 * PATCH 保存后 annotation_source 变为 manual。
 */
export default function AnnotationEditor(): ReactElement {
  const videoId = useAppStore((s) => s.videoId)
  const actions = useAppStore((s) => s.actions)
  const selectedActionId = useAppStore((s) => s.selectedActionId)
  const draft = useAppStore((s) => s.annotationDraft)
  const setDraft = useAppStore((s) => s.setAnnotationDraft)
  const updateAction = useAppStore((s) => s.updateAction)

  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const action = actions.find((a) => a.action_id === selectedActionId) ?? null

  // 选中动作变化时，把该动作注解载入草稿（仅在选中变化时触发）
  useEffect(() => {
    setDraft(action ? { ...action.annotations } : emptyAnnotations())
    setMessage(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedActionId])

  const setField = (key: keyof Annotations, value: string) => {
    setDraft({ ...draft, [key]: value })
  }

  const handleSave = async () => {
    if (!videoId || !action) return
    setSaving(true)
    setMessage(null)
    try {
      const updated = await patchAction(videoId, action.action_id, {
        annotations: draft,
      })
      updateAction(updated)
      setDraft({ ...updated.annotations })
      setMessage('已保存（manual）')
    } catch (err) {
      setMessage(`保存失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {action ? (
        <>
          <div className="flex items-center justify-between gap-2 px-4 pt-4">
            <div className="truncate text-base font-semibold">{action.name}</div>
            <span
              className={`shrink-0 rounded px-2 py-0.5 text-xs ${
                action.annotation_source === 'manual'
                  ? 'bg-amber-500/20 text-amber-300'
                  : 'bg-neutral-700 text-neutral-300'
              }`}
            >
              {action.annotation_source === 'manual' ? '人工' : '模板'}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <GroupTitle>P0 · 核心（8 维）</GroupTitle>
            {P0_ANNOTATION_FIELDS.map((field) => (
              <FieldInput
                key={field.key}
                field={field}
                value={draft[field.key]}
                onChange={(v) => setField(field.key, v)}
              />
            ))}

            <div className="my-2 border-t border-neutral-800" />

            <GroupTitle>P1 · 进阶（9 维）</GroupTitle>
            {P1_ANNOTATION_FIELDS.map((field) => (
              <FieldInput
                key={field.key}
                field={field}
                value={draft[field.key]}
                onChange={(v) => setField(field.key, v)}
              />
            ))}
          </div>

          <div className="border-t border-neutral-800 p-3">
            {message && (
              <div
                className={`mb-2 text-xs ${
                  message.startsWith('保存失败') ? 'text-red-400' : 'text-emerald-400'
                }`}
              >
                {message}
              </div>
            )}
            <button
              type="button"
              disabled={saving}
              onClick={handleSave}
              className="w-full rounded bg-violet-600 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? '保存中…' : '保存注解'}
            </button>
          </div>
        </>
      ) : (
        <div className="flex flex-1 items-center justify-center p-4 text-sm text-neutral-500">
          选中一个动作后，可在此查看并编辑注解
        </div>
      )}
    </div>
  )
}
