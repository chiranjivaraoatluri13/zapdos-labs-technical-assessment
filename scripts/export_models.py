"""Export the two models for faster inference. With neither --detector nor --classifier, both are exported.

- helmet classifier -> ONNX (ONNX Runtime CPU; PyTorch eager spends most of its time on per-layer overhead
  for a 64x64 MobileNet)
- SH17 detector     -> CoreML fp16 (Apple Neural Engine on M-series Macs; the Apple analogue of TensorRT),
                       or ONNX fp32 with --onnx

python scripts/export_models.py [--detector] [--classifier]
python scripts/export_models.py --detector --square --imgsz 1280 960          # other inputs, for comparison
python scripts/export_models.py --detector [--onnx] --weights weights/sh17/yolo8n.pt
python scripts/export_models.py --detector --openvino --int8   # Intel: fp32 IR plus an INT8 copy for the CPU
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.helmet import INPUT, build_model  # noqa: E402


def export_classifier(weights):
    m = build_model()
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    m.eval()
    out = weights.with_suffix(".onnx")
    torch.onnx.export(m, torch.zeros(1, 3, INPUT, INPUT), out, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print("classifier ->", out)


def export_detector(weights, imgsz, aspect=None, fmt="coreml"):
    """Fixed-shape model, saved next to the weights as <stem>_<input>.mlpackage or .onnx. With aspect
    (w/h, e.g. 16/9) the input is a stride-aligned rectangle (1280 -> 736x1280), so a 1080p frame is
    letterboxed with 16 px of padding instead of 560."""
    from ultralytics import YOLO
    shape = [math.ceil(imgsz / aspect / 32) * 32, imgsz] if aspect else imgsz
    opts = {"coreml": {"half": True, "nms": False}, "onnx": {"opset": 17, "simplify": False},
            "openvino": {"half": False}}[fmt]  # OpenVINO picks fp16 on GPU/NPU at compile time
    path = Path(YOLO(str(weights)).export(format=fmt, imgsz=shape, **opts))
    tag = f"{shape[0]}x{shape[1]}" if aspect else str(imgsz)
    dest = weights.with_name(f"{weights.stem}_{tag}_openvino_model" if fmt == "openvino"
                             else f"{weights.stem}_{tag}{path.suffix}")
    if dest.is_dir():
        shutil.rmtree(dest)
    path.replace(dest)
    print("detector ->", dest)
    return dest


CALIB_DIR = ROOT / "data/raw/steelbench/clips"


def _camera(clip):
    """clip_CRM_Work_Shop_20251115_002718_0109 -> crm_work_shop (date, time and index stripped)."""
    return "_".join(clip.removeprefix("clip_").split("_")[:-3]).lower()


def calibration_frames(per_clip=6):
    """Frames for INT8 calibration from SteelBench sample clips whose camera is not one of the six
    evaluation cameras, so calibration never sees the footage it is scored on."""
    import cv2
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import LABELLED
    held = {_camera(c) for c in LABELLED.values()}
    clips = sorted(p for p in CALIB_DIR.glob("*.mp4") if _camera(p.stem) not in held)
    if not clips:
        raise SystemExit(f"no calibration clips in {CALIB_DIR}: run scripts/download_data.py")
    frames = []
    for p in clips:
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, n - 1, per_clip).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
    print(f"calibration: {len(frames)} frames from {len(clips)} clips on {len({_camera(p.stem) for p in clips})} "
          f"cameras outside the evaluation set")
    return frames


def quantize_openvino(fp32_dir):
    """Post-training INT8 (NNCF) of an exported OpenVINO detector -> <name>_int8_openvino_model. The
    detection head's box decoding stays in float, as in ultralytics' own INT8 export."""
    import nncf
    import openvino as ov
    from ultralytics.data.augment import LetterBox
    xml = next(fp32_dir.glob("*.xml"))
    model = ov.Core().read_model(xml)
    h, w = model.inputs[0].shape[2:]
    box = LetterBox((h, w), auto=False, stride=32)

    def prep(frame):
        img = box(image=frame)[..., ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(img)[None].astype(np.float32) / 255.0

    head = "model.22"
    ignored = nncf.IgnoredScope(patterns=[f".*{head}/.*/Add", f".*{head}/.*/Sub.*", f".*{head}/.*/Mul.*",
                                          f".*{head}/.*/Div.*", f".*{head}\\.dfl.*"], types=["Sigmoid"])
    q = nncf.quantize(model, nncf.Dataset(calibration_frames(), prep), preset=nncf.QuantizationPreset.MIXED,
                      ignored_scope=ignored)
    dest = fp32_dir.with_name(fp32_dir.name.replace("_openvino_model", "_int8_openvino_model"))
    if dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir()
    ov.save_model(q, dest / xml.name, compress_to_fp16=False)
    shutil.copy(fp32_dir / "metadata.yaml", dest / "metadata.yaml")
    print("detector ->", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--classifier", action="store_true")
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280])
    ap.add_argument("--square", action="store_true", help="square input instead of 16:9")
    ap.add_argument("--onnx", action="store_true", help="export the detector to ONNX instead of CoreML")
    ap.add_argument("--openvino", action="store_true", help="export the detector to OpenVINO IR instead of CoreML")
    ap.add_argument("--int8", action="store_true",
                    help="with --openvino: also write an INT8 copy calibrated on non-evaluation cameras (needs nncf)")
    ap.add_argument("--weights", help="detector .pt to export (default: the config's)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.classifier or not args.detector:
        export_classifier(ROOT / cfg["ppe"]["classifier"]["weights"])
    if args.detector or not args.classifier:
        weights = ROOT / (args.weights or cfg["detector"]["weights"])
        for s in args.imgsz:
            fmt = "openvino" if args.openvino else "onnx" if args.onnx else "coreml"
            dest = export_detector(weights, s, None if args.square else 16 / 9, fmt)
            if args.int8 and fmt == "openvino":
                quantize_openvino(dest)


if __name__ == "__main__":
    main()
