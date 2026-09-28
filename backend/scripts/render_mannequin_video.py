"""渲染真人模型复刻视频：GLB 蒙皮 + opencv 软件光栅化。

流程：加载动画四元数 + GLB 模型 → LBS 蒙皮 → 绕 y 轴 35° 侧角 → 正交投影
（720x960 竖屏，y 向下）→ 面法线 Lambert 明暗 → painter's algorithm 由远到近
→ cv2.fillPoly（按亮度分桶）→ 写 mp4。

用法：
  python -m scripts.render_mannequin_video --smoke   # 抽帧 t=300/1500/3000 输出 PNG
  python -m scripts.render_mannequin_video --all      # 全量 5450 帧渲染 mp4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.gltf_loader import load_glb
from app.pipeline.skin_animator import SkinAnimator

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CANVAS_W: int = 720
CANVAS_H: int = 960
FPS: int = 30
Y_ROTATION_DEG: float = 35.0
BG_COLOR: tuple[int, int, int] = (24, 24, 24)
BASE_GRAY: float = 0.8
AMBIENT: float = 0.22
DIFFUSE: float = 0.78
SPECULAR: float = 0.45
SHININESS: float = 24.0
BRIGHTNESS_BUCKETS: int = 16
# 光方向（view 空间：前 +Z / 上 +Y / 左 -X），从左前上方打光
LIGHT_DIR: np.ndarray = np.array([-0.45, 0.55, 0.70], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
# 半程向量（视线 + 光），用于 Blinn-Phong 高光
VIEW_DIR: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
HALF: np.ndarray = LIGHT_DIR + VIEW_DIR
HALF /= np.linalg.norm(HALF)


def _load_animation(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    joints: list[str] = data["joints"]
    jr = data["joint_rotations"]
    quats: dict[str, np.ndarray] = {
        j: np.asarray(jr[j], dtype=np.float32) for j in joints
    }
    return joints, quats


def _load_keypoints(path: Path) -> np.ndarray:
    """加载平滑后关键点序列 ``[T, 33, 4]``。"""
    return np.load(str(path))


class Renderer:
    """单帧软件光栅化渲染器（复用投影/光照/面缓存）。"""

    def __init__(self, animator: SkinAnimator) -> None:
        self.animator = animator
        self.model = animator.model

        # 由 rest pose 计算固定视口变换（缩放/居中，避免逐帧漂移）
        rest = animator.skin_rest()
        rest_all = np.concatenate(rest, axis=0)
        self._fit_view(rest_all)

        # 预计算 y 旋转
        theta = np.deg2rad(Y_ROTATION_DEG)
        self.cos_t = float(np.cos(theta))
        self.sin_t = float(np.sin(theta))

    def _fit_view(self, verts: np.ndarray) -> None:
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        cos_t, sin_t = float(np.cos(np.deg2rad(Y_ROTATION_DEG))), float(
            np.sin(np.deg2rad(Y_ROTATION_DEG))
        )
        rx = x * cos_t + z * sin_t
        rz = -x * sin_t + z * cos_t
        ry = y
        self.cx_rot = float((rx.min() + rx.max()) / 2.0)
        self.cy_rot = float((ry.min() + ry.max()) / 2.0)
        rw = float(rx.max() - rx.min()) or 1.0
        rh = float(ry.max() - ry.min()) or 1.0
        self.scale = float(0.85 * min(CANVAS_W / rw, CANVAS_H / rh))
        # body 中心（view 空间，用于法线朝外校正）
        self.center_view = np.array([self.cx_rot, self.cy_rot, float(rz.mean())], np.float32)

    def render_frame(self, kps_frame: np.ndarray) -> np.ndarray:
        """渲染一帧，返回 BGR 画布。"""
        skinned = self.animator.skin_frame_from_kps(kps_frame)
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        all_depths: list[np.ndarray] = []
        all_brightness: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        proj_x: list[np.ndarray] = []
        proj_y: list[np.ndarray] = []

        for verts, mesh in zip(skinned, self.model.meshes):
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            rx = x * self.cos_t + z * self.sin_t
            rz = -x * self.sin_t + z * self.cos_t
            ry = y

            px = CANVAS_W / 2.0 + (rx - self.cx_rot) * self.scale
            py = CANVAS_H / 2.0 - (ry - self.cy_rot) * self.scale
            px_i = np.rint(px).astype(np.int32)
            py_i = np.rint(py).astype(np.int32)
            proj_x.append(px_i)
            proj_y.append(py_i)

            # view 空间顶点（用于法线/深度）
            vv = np.stack([rx, ry, rz], axis=1)  # (N,3)
            tri = mesh.indices  # (T,3)
            p0 = vv[tri[:, 0]]
            p1 = vv[tri[:, 1]]
            p2 = vv[tri[:, 2]]

            normals = np.cross(p1 - p0, p2 - p0)
            centroid = (p0 + p1 + p2) / 3.0
            # 法线朝外校正（对大体凸的人体近似有效）
            flip = np.sum(normals * (centroid - self.center_view), axis=1) < 0
            normals[flip] = -normals[flip]
            norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
            normals = normals / norm
            # Blinn-Phong：漫反射 + 高光 + 环境光合成 0~1 强度
            diffuse = np.clip(normals @ LIGHT_DIR, 0.0, 1.0)
            spec = np.clip(normals @ HALF, 0.0, 1.0) ** SHININESS
            brightness = np.clip(AMBIENT + DIFFUSE * diffuse + SPECULAR * spec, 0.0, 1.0)

            depth = (rz[tri[:, 0]] + rz[tri[:, 1]] + rz[tri[:, 2]]) / 3.0  # (T,)

            pts = np.stack([px_i[tri], py_i[tri]], axis=2)  # (T,3,2)

            all_depths.append(depth)
            all_brightness.append(brightness)
            all_tris.append(pts)

        # 地面阴影：人物脚底投影椭圆，增强「落地」立体感
        all_px = np.concatenate(proj_x)
        all_py = np.concatenate(proj_y)
        shadow_cx = (all_px.min() + all_px.max()) / 2.0
        shadow_cy = float(all_py.max()) + 6.0
        shadow_rx = (all_px.max() - all_px.min()) * 0.42
        shadow_ry = max(3.0, shadow_rx * 0.13)
        cv2.ellipse(
            canvas,
            (int(shadow_cx), int(min(shadow_cy, CANVAS_H - 4))),
            (int(shadow_rx), int(shadow_ry)),
            0, 0, 360, (8, 8, 8), -1,
        )

        depths = np.concatenate(all_depths)
        brightness = np.concatenate(all_brightness)
        tris = np.concatenate(all_tris, axis=0)  # (T,3,2)

        # painter：由远到近
        order = np.argsort(depths, kind="stable")
        tris = tris[order]
        brightness = brightness[order]

        # 按亮度分桶批量 fillPoly
        for b in range(BRIGHTNESS_BUCKETS):
            lo = b / BRIGHTNESS_BUCKETS
            hi = (b + 1) / BRIGHTNESS_BUCKETS
            mask = (brightness >= lo) & (brightness < hi)
            if not np.any(mask):
                continue
            mid = (lo + hi) / 2.0
            gray = int(round(255.0 * BASE_GRAY * mid))
            color = (gray, gray, gray)
            cv2.fillPoly(canvas, tris[mask].astype(np.int32), color)

        return canvas


def _render_smoke(renderer: Renderer, kps: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in (300, 1500, 3000):
        t0 = time.perf_counter()
        canvas = renderer.render_frame(kps[t])
        dt = time.perf_counter() - t0
        out = out_dir / f"smoke_t{t}.png"
        cv2.imwrite(str(out), canvas)
        print(f"[smoke] t={t} 耗时 {dt:.3f}s -> {out.name}")


def _render_all(
    renderer: Renderer,
    kps: np.ndarray,
    out_path: Path,
    total: int,
) -> None:
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        raise IOError(f"无法创建视频输出：{out_path}")

    t_start = time.perf_counter()
    for t in range(total):
        canvas = renderer.render_frame(kps[t])
        writer.write(canvas)
        if t % 500 == 0:
            el = time.perf_counter() - t_start
            fps_est = (t + 1) / el if el > 0 else 0.0
            print(f"[render] frame {t}/{total}  elapsed {el:.1f}s  {fps_est:.2f} fps")

    writer.release()
    el = time.perf_counter() - t_start
    print(f"[render] 完成：{total} 帧，总耗时 {el:.1f}s（{total / el:.2f} fps）")
    print(f"[render] 输出：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anim", default="data/animations/dance_real_anim.json")
    parser.add_argument("--glb", default="assets/models/Soldier.glb")
    parser.add_argument("--out", default="dance_real_mannequin.mp4")
    parser.add_argument("--smoke-dir", default="data/animations/smoke_preview")
    parser.add_argument("--smoke", action="store_true", help="抽帧 t=300/1500/3000 自验")
    parser.add_argument("--all", action="store_true", help="全量渲染视频")
    parser.add_argument("--frames", type=int, default=0, help="仅渲染前 N 帧（调试）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    anim_path = project_root / args.anim
    glb_path = project_root / args.glb
    out_path = project_root / args.out
    smoke_dir = project_root / args.smoke_dir

    kps = _load_keypoints(project_root / "data/keypoints/dance_real_kps_smoothed.npy")
    total = kps.shape[0]
    print(f"[init] 关键点帧数 {total}")

    t0 = time.perf_counter()
    model = load_glb(glb_path)
    animator = SkinAnimator(model)
    print(f"[init] GLB 解析 {time.perf_counter() - t0:.2f}s，mesh {len(model.meshes)} 个")

    renderer = Renderer(animator)

    # 单帧计时（首帧）
    t0 = time.perf_counter()
    _ = renderer.render_frame(kps[0])
    print(f"[init] 单帧渲染 {time.perf_counter() - t0:.3f}s")

    if args.smoke:
        _render_smoke(renderer, kps, smoke_dir)
        return

    if args.all or args.frames > 0:
        n = args.frames if args.frames > 0 else total
        _render_all(renderer, kps, out_path, n)
        return

    print("请指定 --smoke 或 --all")


if __name__ == "__main__":
    main()
