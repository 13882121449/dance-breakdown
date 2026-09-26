/**
 * 骨架绘制辅助：MediaPipe 33 landmarks → 17 主关节 + 连线定义 + 坐标归一化。
 *
 * 坐标映射（与 architecture.md 一致）：
 * - x → 左右（像素）
 * - -y → 上下（MediaPipe y 向下，翻转为 three.js 的 y 向上）
 * - z → 前后（相对深度，米）
 * 归一化：以髋中点（Hips）为原点，按躯干长度缩放 x/y，z 单独缩放，
 * 使骨架适配视口且保持比例合理。
 */

export interface Vec3 {
  x: number
  y: number
  z: number
}

/** 17 个主关节（与后端 MODEL_DEFINITION 一致） */
export const JOINTS = [
  'Hips',
  'Spine',
  'Chest',
  'Neck',
  'Head',
  'L_Shoulder',
  'L_Elbow',
  'L_Wrist',
  'R_Shoulder',
  'R_Elbow',
  'R_Wrist',
  'L_Hip',
  'L_Knee',
  'L_Ankle',
  'R_Hip',
  'R_Knee',
  'R_Ankle',
] as const

export type JointName = (typeof JOINTS)[number]

/** 骨骼连线（父关节 → 子关节） */
export const BONES: Array<[JointName, JointName]> = [
  ['Hips', 'Spine'],
  ['Spine', 'Chest'],
  ['Chest', 'Neck'],
  ['Neck', 'Head'],
  ['Chest', 'L_Shoulder'],
  ['L_Shoulder', 'L_Elbow'],
  ['L_Elbow', 'L_Wrist'],
  ['Chest', 'R_Shoulder'],
  ['R_Shoulder', 'R_Elbow'],
  ['R_Elbow', 'R_Wrist'],
  ['Hips', 'L_Hip'],
  ['L_Hip', 'L_Knee'],
  ['L_Knee', 'L_Ankle'],
  ['Hips', 'R_Hip'],
  ['R_Hip', 'R_Knee'],
  ['R_Knee', 'R_Ankle'],
]

/** MediaPipe 33 landmark 中实际用到的索引 */
const LM = {
  head: 0,
  lShoulder: 11,
  rShoulder: 12,
  lElbow: 13,
  rElbow: 14,
  lWrist: 15,
  rWrist: 16,
  lHip: 23,
  rHip: 24,
  lKnee: 25,
  rKnee: 26,
  lAnkle: 27,
  rAnkle: 28,
} as const

/** z 方向（米）的缩放增益：让前后位移在骨架视图中可见 */
const Z_GAIN = 6

type RawPos = [number, number, number]

function mid(a: RawPos, b: RawPos): RawPos {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
}

/**
 * 由单帧 33 关键点（[[x,y,z,vis]×33]）构建 17 关节的原始像素坐标。
 * 派生关节（Hips/Spine/Chest/Neck）由中点/插值计算。
 */
export function buildJointPositions(frame: number[][]): Record<JointName, RawPos> {
  const lm = (i: number): RawPos => {
    const p = frame[i]
    return p ? [p[0] ?? 0, p[1] ?? 0, p[2] ?? 0] : [0, 0, 0]
  }

  const hips = mid(lm(LM.lHip), lm(LM.rHip))
  const chest = mid(lm(LM.lShoulder), lm(LM.rShoulder))
  const spine = mid(hips, chest)
  const neck: RawPos = [
    chest[0] + (chest[0] - spine[0]) * 0.5,
    chest[1] + (chest[1] - spine[1]) * 0.5,
    chest[2] + (chest[2] - spine[2]) * 0.5,
  ]

  return {
    Hips: hips,
    Spine: spine,
    Chest: chest,
    Neck: neck,
    Head: lm(LM.head),
    L_Shoulder: lm(LM.lShoulder),
    L_Elbow: lm(LM.lElbow),
    L_Wrist: lm(LM.lWrist),
    R_Shoulder: lm(LM.rShoulder),
    R_Elbow: lm(LM.rElbow),
    R_Wrist: lm(LM.rWrist),
    L_Hip: lm(LM.lHip),
    L_Knee: lm(LM.lKnee),
    L_Ankle: lm(LM.lAnkle),
    R_Hip: lm(LM.rHip),
    R_Knee: lm(LM.rKnee),
    R_Ankle: lm(LM.rAnkle),
  }
}

/**
 * 归一化：居中到 Hips、翻转 y、按躯干长度缩放，返回可渲染的关节坐标。
 */
export function normalizeSkeleton(frame: number[][]): Record<JointName, Vec3> {
  const raw = buildJointPositions(frame)
  const hips = raw.Hips
  const chest = raw.Chest
  const torso = Math.hypot(chest[0] - hips[0], chest[1] - hips[1])
  const scale = Math.max(1, torso)

  const positions = {} as Record<JointName, Vec3>
  for (const joint of JOINTS) {
    const p = raw[joint]
    positions[joint] = {
      x: (p[0] - hips[0]) / scale,
      y: -(p[1] - hips[1]) / scale,
      z: (p[2] - hips[2]) * Z_GAIN,
    }
  }
  return positions
}

/** 把 Vec3 转为 three.js 的 [x,y,z] 元组 */
export function toTuple(v: Vec3): [number, number, number] {
  return [v.x, v.y, v.z]
}
