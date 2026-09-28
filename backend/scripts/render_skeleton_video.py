"""用关键点数据渲染 2.5D 人体复刻视频（躯干/四肢分段粗细 + z 排序）。

升级自骨架版：肢体用不同粗细模拟体积，z 排序让远端肢体先画（遮挡正确）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from app.config import KEYPOINTS_DIR

LM = {
    'head': 0, 'lShoulder': 11, 'rShoulder': 12, 'lElbow': 13, 'rElbow': 14,
    'lWrist': 15, 'rWrist': 16, 'lHip': 23, 'rHip': 24, 'lKnee': 25, 'rKnee': 26,
    'lAnkle': 27, 'rAnkle': 28,
}

# (骨骼, 厚度px, 颜色BGR)。thickness 模拟肢体体积：躯干最粗、四肢次之、末端最细。
BONES = [
    ('Hips', 'Spine', 34, (205, 130, 90)),
    ('Spine', 'Chest', 32, (205, 130, 90)),
    ('Chest', 'Neck', 14, (190, 120, 85)),
    ('Neck', 'Head', 12, (190, 120, 85)),
    ('Chest', 'L_Shoulder', 16, (210, 170, 70)),
    ('L_Shoulder', 'L_Elbow', 19, (210, 170, 70)),
    ('L_Elbow', 'L_Wrist', 14, (210, 170, 70)),
    ('Chest', 'R_Shoulder', 16, (210, 170, 70)),
    ('R_Shoulder', 'R_Elbow', 19, (210, 170, 70)),
    ('R_Elbow', 'R_Wrist', 14, (210, 170, 70)),
    ('Hips', 'L_Hip', 18, (70, 170, 210)),
    ('L_Hip', 'L_Knee', 21, (70, 170, 210)),
    ('L_Knee', 'L_Ankle', 15, (70, 170, 210)),
    ('Hips', 'R_Hip', 18, (70, 170, 210)),
    ('R_Hip', 'R_Knee', 21, (70, 170, 210)),
    ('R_Knee', 'R_Ankle', 15, (70, 170, 210)),
]

HEAD_RADIUS = 22
HEAD_COLOR = (160, 170, 250)

Z_GAIN = 8.0
VIEW_YAW = 35.0


def build_joints(frame):
    def lm(i):
        p = frame[i]
        return np.array([p[0], p[1], p[2]], dtype=np.float64)

    hips = (lm(LM['lHip']) + lm(LM['rHip'])) / 2
    chest = (lm(LM['lShoulder']) + lm(LM['rShoulder'])) / 2
    spine = (hips + chest) / 2
    neck = chest + (chest - spine) * 0.5
    return {
        'Hips': hips, 'Spine': spine, 'Chest': chest, 'Neck': neck, 'Head': lm(LM['head']),
        'L_Shoulder': lm(LM['lShoulder']), 'L_Elbow': lm(LM['lElbow']), 'L_Wrist': lm(LM['lWrist']),
        'R_Shoulder': lm(LM['rShoulder']), 'R_Elbow': lm(LM['rElbow']), 'R_Wrist': lm(LM['rWrist']),
        'L_Hip': lm(LM['lHip']), 'L_Knee': lm(LM['lKnee']), 'L_Ankle': lm(LM['lAnkle']),
        'R_Hip': lm(LM['rHip']), 'R_Knee': lm(LM['rKnee']), 'R_Ankle': lm(LM['rAnkle']),
    }


def normalize(joints):
    hips = joints['Hips']
    chest = joints['Chest']
    torso = np.hypot(chest[0] - hips[0], chest[1] - hips[1])
    scale = max(1.0, torso)
    out = {}
    for name, p in joints.items():
        out[name] = np.array([
            (p[0] - hips[0]) / scale,
            -(p[1] - hips[1]) / scale,
            (p[2] - hips[2]) * Z_GAIN,
        ])
    return out


def main():
    kps = np.load(str(KEYPOINTS_DIR / 'dance_real_kps_smoothed.npy'))
    T = kps.shape[0]
    fps = 30

    W, H = 720, 960
    cx, cy = W / 2, H / 2
    s = 260.0

    yaw_rad = np.radians(VIEW_YAW)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    out_path = Path(__file__).resolve().parent.parent.parent / 'dance_real_skeleton.mp4'
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for t in range(T):
        joints = normalize(build_joints(kps[t]))
        img = np.full((H, W, 3), 18, dtype=np.uint8)

        # 投影：x 绕 y 轴旋转后做正交投影；zr 保留用于排序
        pts, depth = {}, {}
        for name, p in joints.items():
            x, y, z = p
            xr = x * cos_y + z * sin_y
            zr = -x * sin_y + z * cos_y
            pts[name] = (int(cx + xr * s), int(cy - y * s))
            depth[name] = zr

        # z 排序（painter's algorithm）：远（zr 小）的先画
        order = sorted(
            range(len(BONES)),
            key=lambda i: (depth[BONES[i][0]] + depth[BONES[i][1]]) / 2,
        )
        for i in order:
            a, b, thick, color = BONES[i]
            pa, pb = pts[a], pts[b]
            cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)
            # 圆角端点，模拟胶囊体
            r = thick // 2
            cv2.circle(img, pa, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, pb, r, color, -1, cv2.LINE_AA)

        # 手脚末端小球
        for name in ['L_Wrist', 'R_Wrist', 'L_Ankle', 'R_Ankle']:
            cv2.circle(img, pts[name], 9, (220, 200, 190), -1, cv2.LINE_AA)

        # 头
        cv2.circle(img, pts['Head'], HEAD_RADIUS, HEAD_COLOR, -1, cv2.LINE_AA)

        sec = t / fps
        cv2.putText(img, f'{int(sec // 60)}:{sec % 60:04.1f}', (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
        if t % 1000 == 0:
            print(f'render {t}/{T}', flush=True)

    writer.release()
    print(f'DONE 输出: {out_path} ({T} 帧)', flush=True)


if __name__ == '__main__':
    main()
