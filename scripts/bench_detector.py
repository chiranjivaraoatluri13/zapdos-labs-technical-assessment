"""Detector backends and sizes: latency, and how many of the reference model's detections each keeps.

Reference: SH17 YOLOv8s, PyTorch at 1280 on an available device (MPS/CUDA/CPU). Frames: 10 evenly spaced from each of the three dev
clips. A reference box counts as kept when the tested model has a box of the same class at IoU >= 0.5.
Models that haven't been exported are skipped. To export all of them:

python scripts/download_data.py                                    # yolo8s.pt, yolo8n.pt
python scripts/export_models.py --detector                         # yolo8s 736x1280 (CoreML)
python scripts/export_models.py --detector --square --imgsz 1280 960
python scripts/export_models.py --detector --weights weights/sh17/yolo8n.pt
python scripts/export_models.py --detector --onnx
python scripts/export_models.py --detector --onnx --weights weights/sh17/yolo8n.pt
python scripts/bench_detector.py                                   # writes eval/results/detector_bench.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights/sh17"
CLIPS = ["sb_crm_workshop_0109", "sb_red_lrs1_0007", "sb_tippler_lowlight_0062"]
CLASSES = {"person": (0, 0.25), "head": (12, 0.30), "helmet": (10, 0.25), "vest": (16, 0.30)}  # (id, min conf)
CONFIGS = [  # name, weights, imgsz, device
    ("v8s torch-reference 1280", W / "yolo8s.pt", 1280,
     "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")),
    ("v8s coreml-fp16 736x1280", W / "yolo8s_736x1280.mlpackage", [736, 1280], None),
    ("v8s coreml-fp16 cpu-only 736x1280", W / "yolo8s_736x1280.mlpackage", [736, 1280], "cpu_only"),
    ("v8s coreml-fp16 1280x1280", W / "yolo8s_1280.mlpackage", 1280, None),
    ("v8s coreml-fp16 960x960", W / "yolo8s_960.mlpackage", 960, None),
    ("v8n coreml-fp16 736x1280", W / "yolo8n_736x1280.mlpackage", [736, 1280], None),
    ("v8s onnxruntime-cpu 736x1280", W / "yolo8s_736x1280.onnx", [736, 1280], "cpu"),
    ("v8n onnxruntime-cpu 736x1280", W / "yolo8n_736x1280.onnx", [736, 1280], "cpu"),
    ("v8s torch-cpu 1280", W / "yolo8s.pt", 1280, "cpu"),
    ("v8s openvino-cpu 736x1280", W / "yolo8s_736x1280_openvino_model", [736, 1280], "intel:cpu"),
    ("v8s openvino-int8-cpu 736x1280", W / "yolo8s_736x1280_int8_openvino_model", [736, 1280], "intel:cpu"),
    ("v8s openvino-gpu 736x1280", W / "yolo8s_736x1280_openvino_model", [736, 1280], "intel:gpu"),
    ("v8s openvino-int8-gpu 736x1280", W / "yolo8s_736x1280_int8_openvino_model", [736, 1280], "intel:gpu"),
]


def frames():
    out = []
    for c in CLIPS:
        cap = cv2.VideoCapture(str(ROOT / f"clips/{c}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, n - 1, 10).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                out.append(f)
        cap.release()
    return out


def run(model, weights, imgs, imgsz, device):
    cpu_only = device == "cpu_only"  # CoreML restricted to the CPU, as run.py --cpu-only does
    kw = dict(imgsz=imgsz, conf=0.10, device=None if cpu_only else device,
              classes=[c for c, _ in CLASSES.values()], verbose=False)
    model.predict(imgs[0], **kw)
    if cpu_only:
        import coremltools as ct
        model.predictor.model.backend.model = ct.models.MLModel(str(weights), compute_units=ct.ComputeUnit.CPU_ONLY)
    for _ in range(3):
        model.predict(imgs[0], **kw)
    dets, ms = [], []
    for f in imgs:
        t0 = time.perf_counter()
        b = model.predict(f, **kw)[0].boxes.cpu().numpy()
        ms.append((time.perf_counter() - t0) * 1000)
        dets.append(np.hstack([b.xyxy, b.conf[:, None], b.cls[:, None]]))
    return dets, np.array(ms)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def kept(ref, test, cls, min_conf):
    hit = total = 0
    for r, t in zip(ref, test):
        r = r[(r[:, 5] == cls) & (r[:, 4] >= min_conf)]
        t = t[t[:, 5] == cls]
        total += len(r)
        if len(r) and len(t):
            hit += int((iou(r, t).max(1) >= 0.5).sum())
    return hit, total


def main():
    if not CONFIGS[0][1].exists():
        raise SystemExit(f"reference model {CONFIGS[0][1]} missing: run scripts/download_data.py")
    imgs = frames()
    if not imgs:
        raise SystemExit("no benchmark frames: download the labelled clips first")
    rows, ref = [], None
    for name, w, imgsz, device in CONFIGS:
        if w.suffix == ".mlpackage" and sys.platform != "darwin":
            continue
        if not w.exists():
            print(f"skip {name}: {w.name} not exported")
            continue
        if str(device).startswith("intel:gpu"):
            import openvino as ov
            if "GPU" not in ov.Core().available_devices:
                print(f"skip {name}: no Intel GPU")
                continue
        dets, ms = run(YOLO(str(w), task="detect"), w, imgs, imgsz, device)
        ref = ref or dets
        row = {"model": name, "mean_ms": round(float(ms.mean()), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}
        for cname, (c, mc) in CLASSES.items():
            h, n = kept(ref, dets, c, mc)
            row[cname] = f"{h}/{n}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    # the committed Apple results stay in detector_bench.json; other machines write their own file
    out = ROOT / ("eval/results/detector_bench.json" if sys.platform == "darwin" else
                  f"eval/results/detector_bench_{sys.platform}.json")
    out.write_text(json.dumps({"frames": len(imgs), "reference": CONFIGS[0][0], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
