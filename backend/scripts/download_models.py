"""下载 mediapipe Tasks 模型（含重试 + 完整性校验）。

用法：
    python scripts/download_models.py [pose_lite|pose_full|all]

模型（.task 为 zip 容器，.tflite 为单文件 flatbuffer），下载后校验：
- .task：zipfile 可打开且包含至少一个 .tflite 条目；
- .tflite：以 TFL3 魔数开头。
"""
import os
import sys
import time
import zipfile

import requests

MODELS = {
    "selfie_segmenter": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "file": "selfie_segmenter.tflite",
        "kind": "tflite",
    },
    "pose_lite": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "file": "pose_landmarker_lite.task",
        "kind": "task",
    },
    "pose_full": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "file": "pose_landmarker_full.task",
        "kind": "task",
    },
    "pose_heavy": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "file": "pose_landmarker_heavy.task",
        "kind": "task",
    },
}

# 备用镜像（ghproxy 类，按顺序尝试）
MIRROR_PREFIXES = [
    "",
    "https://ghfast.top/",
    "https://mirror.ghproxy.com/",
]


def _valid_tflite(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(8)
    return len(head) >= 8 and head[4:8] == b"TFL3"


def _valid_task(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            return any(n.endswith(".tflite") for n in names)
    except Exception:
        return False


def download(model_key: str) -> bool:
    spec = MODELS[model_key]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, spec["file"])
    validator = _valid_tflite if spec["kind"] == "tflite" else _valid_task

    for prefix in MIRROR_PREFIXES:
        url = prefix + spec["url"]
        for attempt in range(2):
            try:
                print(f"[{model_key}] try {url}", flush=True)
                r = requests.get(url, timeout=300, stream=True)
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code}", flush=True)
                    break
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                size = os.path.getsize(tmp)
                ok = validator(tmp)
                print(f"    size={size} valid={ok}", flush=True)
                if ok:
                    os.replace(tmp, dest)
                    print(f"[{model_key}] OK -> {dest}", flush=True)
                    return True
                os.remove(tmp)
            except Exception as e:  # noqa: BLE001
                print(f"    err {type(e).__name__}: {e}", flush=True)
            time.sleep(2)
    print(f"[{model_key}] FAILED", flush=True)
    return False


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = {
        "pose_lite": ["pose_lite"],
        "pose_full": ["pose_full"],
        "pose_heavy": ["pose_heavy"],
        "selfie": ["selfie_segmenter"],
        "all": ["selfie_segmenter", "pose_lite", "pose_full", "pose_heavy"],
    }.get(which, [which])
    ok_all = True
    for k in keys:
        if k not in MODELS:
            print(f"unknown model key: {k}", flush=True)
            ok_all = False
            continue
        ok_all = download(k) and ok_all
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
