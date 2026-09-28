#!/usr/bin/env python3
"""Plan B：用 YOLOv8-pose 替换 ViTPose，绕开 mmcv/mmpose/mmdet 编译难题。

WHAM 的 ``lib/models/preproc/detector.py`` 硬依赖 mmpose(ViTPose) 做 2D 关键点。
mmcv 1.x 在 torch2.1+cu118 上无预编译 wheel、源码编译又常失败。本脚本把
``DetectionModel`` 重写为基于 ultralytics YOLOv8-pose（自带 COCO 17 关键点，
首次运行自动下载 ``yolov8x-pose.pt``），输出结构与原版完全一致，因此下游
FeatureExtractor / demo.py 无需任何改动。

用法（在 WSL2 终端、WHAM 仓库目录下）:
    python patch_detector_yolopose.py          # 默认 WHAM_DIR=$HOME/wham/WHAM
    WHAM_DIR=/path/to/WHAM python patch_detector_yolopose.py
    python patch_detector_yolopose.py --restore  # 还原被替换的 detector.py

注意：
  - 仅单人为目标（舞蹈场景单人，与项目一致）。
  - Plan B 不需要 vitpose-h-multi-coco.pth，也不需要 mmcv/mmpose/mmdet。
  - 效果上 WHAM 官方推荐 ViTPose；Plan B 为“先跑通”的降级方案，效果仍优于 MediaPipe。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPLACEMENT = '''"""Plan B drop-in：YOLOv8-pose 替换 ViTPose（由 patch_detector_yolopose.py 生成）。

输出结构与原版 DetectionModel 完全一致：
    tracking_results[_id] = { 'frame_id', 'bbox'(cxcys), 'keypoints'([N,17,3] COCO) }
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from ultralytics import YOLO

VIS_THRESH = 0.3
BBOX_CONF = 0.25
S_FACTOR = 1.05
MINIMUM_FRAMES = 30
MINIMUM_JOINTS = 6


class DetectionModel(object):
    def __init__(self, device):
        # YOLOv8-pose：同时输出 person bbox + COCO 17 关键点（首次运行自动下载权重）
        self.pose_model = YOLO("yolov8x-pose.pt")
        self.device = device
        self.initialize_tracking()

    def initialize_tracking(self):
        self.frame_id = 0
        self.tracking_results = {"id": [], "frame_id": [], "bbox": [], "keypoints": []}

    def xyxy_to_cxcys(self, bbox, s_factor=S_FACTOR):
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.0 * s_factor
        return np.array([cx, cy, scale])

    def track(self, img, fps, length):
        res = self.pose_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF,
            save=False, verbose=False,
        )[0]

        if res.keypoints is None or len(res.keypoints.data) == 0:
            self.frame_id += 1
            return

        kps = res.keypoints.data.cpu().numpy()      # [N, 17, 3] = (x, y, conf)
        boxes = res.boxes.xyxy.cpu().numpy()        # [N, 4]
        confs = res.boxes.conf.cpu().numpy()        # [N]

        # 单人：取置信度最高的 person
        idx = int(np.argmax(confs))
        kp = kps[idx]
        if (kp[:, -1] > VIS_THRESH).sum() < MINIMUM_JOINTS:
            self.frame_id += 1
            return

        self.tracking_results["id"].append(0)
        self.tracking_results["frame_id"].append(self.frame_id)
        self.tracking_results["bbox"].append(self.xyxy_to_cxcys(boxes[idx]))
        self.tracking_results["keypoints"].append(kp)
        self.frame_id += 1

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = np.array(self.tracking_results["keypoints"])
        mask = X[..., -1] > VIS_THRESH
        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(), kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = (bb[2] + bb[0]) / 2.0, (bb[3] + bb[1]) / 2.0
            s = max(bb[2] - bb[0], bb[3] - bb[1])
            bbox[i] = np.array([cx, cy, s * s_factor / 200.0])
        self.tracking_results["bbox"] = bbox

    def process(self, fps):
        self.compute_bboxes_from_keypoints()
        output = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.tracking_results["keypoints"])):
            output[0]["frame_id"].append(self.tracking_results["frame_id"][i])
            output[0]["bbox"].append(self.tracking_results["bbox"][i])
            output[0]["keypoints"].append(self.tracking_results["keypoints"][i])

        # 丢弃过短跟踪段（与原版一致）
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]["bbox"]) < MINIMUM_FRAMES:
                del output[_id]
                continue
            for key in ("frame_id", "bbox", "keypoints"):
                output[_id][key] = np.array(output[_id][key])
        return output
'''


def locate_wham_dir() -> Path:
    d = Path(os.environ.get("WHAM_DIR", Path.home() / "wham" / "WHAM"))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="还原被替换的 detector.py")
    args = ap.parse_args()

    wham = locate_wham_dir()
    target = wham / "lib" / "models" / "preproc" / "detector.py"
    backup = wham / "lib" / "models" / "preproc" / "detector.py.vitpose.bak"

    if args.restore:
        if backup.exists():
            shutil.copy2(backup, target)
            print(f"[restore] 已还原 {target}（来自备份）")
        else:
            print("[restore] 未找到备份，跳过")
        return

    if not target.exists():
        sys.exit(f"未找到 {target}，WHAM_DIR 是否正确？当前: {wham}")

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[patch] 已备份原 detector.py -> {backup.name}")

    target.write_text(REPLACEMENT, encoding="utf-8")
    print(f"[patch] 已写入 YOLOv8-pose 版 detector.py -> {target}")
    print("[patch] 下一步: 直接 bash run_demo.sh <视频>（无需 mmcv/ViTPose）")


if __name__ == "__main__":
    main()
