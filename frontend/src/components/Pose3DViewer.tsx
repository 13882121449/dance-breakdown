import { useEffect, useRef, useState } from 'react'

import { OrbitControls, Line } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'

import { getAllKeypoints } from '../api/client'
import { useAppStore } from '../store/useAppStore'
import {
  BONES,
  JOINTS,
  normalizeSkeleton,
  toTuple,
  type JointName,
  type Vec3,
} from '../three/skeleton'

type SkeletonPositions = Record<JointName, Vec3>

/** 骨架网格：17 关节画球 + 骨骼连线 */
function Skeleton({ positions }: { positions: SkeletonPositions }) {
  return (
    <group>
      {BONES.map(([from, to]) => (
        <Line
          key={`${from}-${to}`}
          points={[toTuple(positions[from]), toTuple(positions[to])]}
          color="#8b5cf6"
          lineWidth={2}
        />
      ))}
      {JOINTS.map((joint) => {
        const p = toTuple(positions[joint])
        const color = joint === 'Head' ? '#f472b6' : '#38bdf8'
        return (
          <mesh key={joint} position={p}>
            <sphereGeometry args={[0.045, 16, 16]} />
            <meshStandardMaterial color={color} />
          </mesh>
        )
      })}
    </group>
  )
}

/**
 * 3D 骨架视图：随 currentFrame 渲染对应帧的关键点。
 *
 * 播放前一次性拉取全量关键点到内存，播放时按帧内存取用，实现 30fps 平滑骨架
 * （避免逐帧网络请求 + 80ms 防抖导致的掉帧）。
 */
export default function Pose3DViewer() {
  const videoId = useAppStore((s) => s.videoId)
  const currentFrame = useAppStore((s) => s.currentFrame)
  const [positions, setPositions] = useState<SkeletonPositions | null>(null)
  const keypointsRef = useRef<number[][][] | null>(null)

  // videoId 变化时一次性拉取全量关键点到内存
  useEffect(() => {
    keypointsRef.current = null
    setPositions(null)
    if (!videoId) return

    let cancelled = false
    getAllKeypoints(videoId)
      .then((data) => {
        if (cancelled) return
        keypointsRef.current = data.keypoints
        const frame = data.keypoints[currentFrame]
        if (frame) setPositions(normalizeSkeleton(frame))
      })
      .catch(() => {
        // 加载失败保持空骨架，不打断其它交互
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoId])

  // 播放时直接从内存取帧，无网络开销
  useEffect(() => {
    const all = keypointsRef.current
    if (!all) return
    const frame = all[currentFrame]
    if (frame) setPositions(normalizeSkeleton(frame))
  }, [currentFrame])

  return (
    <div className="relative h-full w-full bg-neutral-900">
      <Canvas camera={{ position: [0, 0.2, 2.4], fov: 50 }} className="h-full w-full">
        <ambientLight intensity={0.7} />
        <directionalLight position={[2, 3, 2]} intensity={0.9} />
        {positions && <Skeleton positions={positions} />}
        <gridHelper args={[4, 20, '#3f3f46', '#27272a']} position={[0, -1.1, 0]} />
        <OrbitControls enableDamping target={[0, 0, 0]} />
      </Canvas>

      {!videoId && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-sm text-neutral-500">
          上传视频后在此显示 3D 骨架
        </div>
      )}
    </div>
  )
}
