"""渲染 WHAM 输出的 SMPL 人体复刻视频（软件光栅化，无 pyrender/OpenGL）。

流程：加载 wham_output.pkl 的 verts + SMPL faces → 绕 y 轴 35° 侧角 → 正交投影
（720x960 竖屏）→ 面法线 Blinn-Phong 明暗 → painter's algorithm 由远到近
→ cv2.fillPoly（按亮度分桶）→ 写 mp4。

用法：
  python scripts/render_smpl_video.py --smoke    # 抽帧自验输出 PNG
  python scripts/render_smpl_video.py --all      # 全量渲染 mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CANVAS_W: int = 720
CANVAS_H: int = 960
FPS: int = 30
Y_ROTATION_DEG: float = 35.0
BG_COLOR: tuple = (24, 24, 24)
BASE_GRAY: float = 0.8
AMBIENT: float = 0.22
DIFFUSE: float = 0.78
SPECULAR: float = 0.45
SHININESS: float = 24.0
BRIGHTNESS_BUCKETS: int = 16
LIGHT_DIR = np.array([-0.45, 0.55, 0.70], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
VIEW_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
HALF = LIGHT_DIR + VIEW_DIR
HALF /= np.linalg.norm(HALF)


class SMPLRenderer:
    """WHAM 输出为相机坐标系（x-right, y-down, z-forward，米制）。
    正交投影：y 直接映射（py 向下增大），固定缩放按身高，每帧居中。"""

    def __init__(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self.verts = verts.astype(np.float32)  # (T, 6890, 3)
        self.faces = faces.astype(np.int64)    # (13776, 3)
        self.cos_t = float(np.cos(np.deg2rad(Y_ROTATION_DEG)))
        self.sin_t = float(np.sin(np.deg2rad(Y_ROTATION_DEG)))
        # 固定缩放：按 1.75m 身高占画布高 85%
        self.scale = 0.85 * CANVAS_H / 1.75

    def _rotate(self, v: np.ndarray) -> tuple:
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        rx = x * self.cos_t + z * self.sin_t
        rz = -x * self.sin_t + z * self.cos_t
        return rx, y, rz

    def render_frame(self, v: np.ndarray) -> np.ndarray:
        rx, ry, rz = self._rotate(v)
        # 每帧居中（人物始终在画面中央，大小恒定）
        self.cx = float((rx.min() + rx.max()) / 2.0)
        self.cy = float((ry.min() + ry.max()) / 2.0)
        self.center_view = np.array(
            [self.cx, self.cy, float(rz.mean())], np.float32
        )

        # y-down 相机系：直接映射（py 向下增大），头(y小)在画布上方
        px = CANVAS_W / 2.0 + (rx - self.cx) * self.scale
        py = CANVAS_H / 2.0 + (ry - self.cy) * self.scale
        px_i = np.rint(px).astype(np.int32)
        py_i = np.rint(py).astype(np.int32)

        vv = np.stack([rx, ry, rz], axis=1)  # (N,3) view 空间
        tri = self.faces
        p0 = vv[tri[:, 0]]; p1 = vv[tri[:, 1]]; p2 = vv[tri[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        flip = np.sum(normals * (centroid - self.center_view), axis=1) < 0
        normals[flip] = -normals[flip]
        norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / norm
        diffuse = np.clip(normals @ LIGHT_DIR, 0.0, 1.0)
        spec = np.clip(normals @ HALF, 0.0, 1.0) ** SHININESS
        brightness = np.clip(AMBIENT + DIFFUSE * diffuse + SPECULAR * spec, 0.0, 1.0)

        depth = (rz[tri[:, 0]] + rz[tri[:, 1]] + rz[tri[:, 2]]) / 3.0
        pts = np.stack([px_i[tri], py_i[tri]], axis=2)  # (T,3,2)

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        # 地面阴影：脚底投影椭圆
        shadow_cx = (px.min() + px.max()) / 2.0
        shadow_cy = float(py.max()) + 6.0
        shadow_rx = (px.max() - px.min()) * 0.42
        shadow_ry = max(3.0, shadow_rx * 0.13)
        cv2.ellipse(
            canvas,
            (int(shadow_cx), int(min(shadow_cy, CANVAS_H - 4))),
            (int(shadow_rx), int(shadow_ry)),
            0, 0, 360, (8, 8, 8), -1,
        )

        # painter：由远到近（相机系 z 大 = 远，先画）
        order = np.argsort(-depth, kind="stable")
        pts = pts[order]; brightness = brightness[order]

        for b in range(BRIGHTNESS_BUCKETS):
            lo = b / BRIGHTNESS_BUCKETS; hi = (b + 1) / BRIGHTNESS_BUCKETS
            mask = (brightness >= lo) & (brightness < hi)
            if not np.any(mask):
                continue
            mid = (lo + hi) / 2.0
            gray = int(round(255.0 * BASE_GRAY * mid))
            cv2.fillPoly(canvas, pts[mask].astype(np.int32), (gray, gray, gray))
        return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="D:/wham-data/output/dance/wham_output.pkl")
    ap.add_argument("--faces", default="D:/wham-data/smpl_faces.npy")
    ap.add_argument("--out", default="D:/wham-data/output/dance/wham_smpl.mp4")
    ap.add_argument("--smoke-dir", default="D:/wham-data/output/dance/smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    print("[init] 加载 wham_output.pkl ...")
    data = joblib.load(args.pkl)
    verts = data[0]["verts"]
    faces = np.load(args.faces)
    print(f"[init] verts={verts.shape} faces={faces.shape}")

    renderer = SMPLRenderer(verts, faces)
    total = verts.shape[0]

    if args.smoke:
        out_dir = Path(args.smoke_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in (total // 10, total // 2, total - 1):
            t0 = time.perf_counter()
            canvas = renderer.render_frame(verts[t])
            dt = time.perf_counter() - t0
            cv2.imwrite(str(out_dir / f"smoke_t{t}.png"), canvas)
            print(f"[smoke] t={t} 耗时 {dt:.3f}s -> smoke_t{t}.png")
        return

    if args.all or args.frames > 0:
        n = args.frames if args.frames > 0 else total
        writer = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CANVAS_W, CANVAS_H)
        )
        if not writer.isOpened():
            raise IOError(f"无法创建视频输出：{args.out}")
        t_start = time.perf_counter()
        for t in range(n):
            canvas = renderer.render_frame(verts[t])
            writer.write(canvas)
            if t % 500 == 0:
                el = time.perf_counter() - t_start
                print(f"[render] frame {t}/{n}  elapsed {el:.1f}s  {(t+1)/el:.2f} fps")
        writer.release()
        el = time.perf_counter() - t_start
        print(f"[render] 完成 {n} 帧，耗时 {el:.1f}s（{n/el:.2f} fps）-> {args.out}")


if __name__ == "__main__":
    main()
