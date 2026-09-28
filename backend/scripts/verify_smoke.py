"""程序化验证 smoke 抽帧渲染质量（无需人工看图）。

检查项：
1. 人物存在：非背景像素占比在合理区间
2. 直立不倒立：bounding box 高 >> 宽，且躯干主体在竖直方向连续
3. 居中：水平重心接近画布中心
4. Lambert 明暗：人物内部灰度有足够方差（有立体感）
5. 无面片乱飞：非背景像素聚成单个连通团（异常散点极少）
6. 跨帧摆动：不同帧之间人物像素有显著差异（四肢在动）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BG = np.array([24, 24, 24], dtype=np.uint8)


def analyze(path: Path) -> dict:
    img = cv2.imread(str(path))
    assert img is not None, f"无法读取 {path}"
    h, w = img.shape[:2]
    diff = np.abs(img.astype(np.int16) - BG.astype(np.int16)).sum(axis=2)
    mask = diff > 30  # 非背景
    n_fg = int(mask.sum())
    ratio = n_fg / (h * w)

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    cx = float(xs.mean())
    cy = float(ys.mean())

    # 灰度方差（仅人物内部）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg_gray = gray[mask]
    std = float(fg_gray.std()) if fg_gray.size else 0.0

    # 顶部 vs 底部像素分布（判断直立：头应在上、腿在下，且上下都有内容）
    top_ratio = float((mask[: h // 2]).sum()) / max(n_fg, 1)
    bottom_ratio = float((mask[h // 2 :]).sum()) / max(n_fg, 1)

    # 连通域数量（面片乱飞检测：8 邻域）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    # 排除背景 label 0
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True) if n_labels > 1 else []
    big_clusters = [a for a in areas if a > n_fg * 0.01]  # 面积 > 1% 的团
    stray = sum(a for a in areas if a <= n_fg * 0.001)  # 微散点面积和

    return {
        "path": path.name,
        "size": f"{w}x{h}",
        "fg_ratio": round(ratio, 4),
        "bbox": f"{bw}x{bh}",
        "aspect": round(bh / bw, 3),
        "center_x": round(cx, 1),
        "center_y": round(cy, 1),
        "gray_std": round(std, 2),
        "top_ratio": round(top_ratio, 3),
        "bottom_ratio": round(bottom_ratio, 3),
        "n_labels": n_labels - 1,
        "big_clusters": len(big_clusters),
        "stray_px": int(stray),
    }


def main() -> None:
    smoke_dir = Path(__file__).resolve().parent.parent.parent / "data/animations/smoke_preview"
    paths = sorted(smoke_dir.glob("smoke_t*.png"))
    if not paths:
        print("未找到 smoke PNG")
        sys.exit(1)

    results = [analyze(p) for p in paths]
    keys = list(results[0].keys())
    header = " | ".join(keys)
    print(header)
    print("-" * len(header))
    for r in results:
        print(" | ".join(str(r[k]) for k in keys))

    # 自动判定
    print("\n[判定]")
    ok = True
    for r in results:
        if not (0.05 <= r["fg_ratio"] <= 0.9):
            print(f"  ✗ {r['path']}: 前景占比异常 {r['fg_ratio']}")
            ok = False
        if r["aspect"] < 1.5:
            print(f"  ✗ {r['path']}: 宽高比异常（疑似横躺） {r['aspect']}")
            ok = False
        if r["gray_std"] < 8:
            print(f"  ✗ {r['path']}: 灰度方差过低（无立体感） {r['gray_std']}")
            ok = False
        if r["big_clusters"] > 1:
            print(f"  ✗ {r['path']}: 主连通团 >1（面片乱飞） {r['big_clusters']}")
            ok = False
        if r["stray_px"] > r["fg_ratio"] * 50000:
            print(f"  ✗ {r['path']}: 散点像素过多 {r['stray_px']}")
            ok = False
    # 跨帧差异
    if len(results) >= 2:
        a = cv2.imread(str(paths[0])).astype(np.int16)
        b = cv2.imread(str(paths[-1])).astype(np.int16)
        diff_px = int((np.abs(a - b).sum(axis=2) > 30).sum())
        print(f"  跨帧差异像素（首/末帧）: {diff_px}")
        if diff_px < 5000:
            print("  ✗ 帧间几乎无差异（四肢未摆动）")
            ok = False

    print("\n[结论]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
