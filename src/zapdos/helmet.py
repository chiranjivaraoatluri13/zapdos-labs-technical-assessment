"""Second-stage helmet classifier on head crops cut from the native-resolution frame.

SteelBench heads are 17-42 px tall (median 26) at 1080p. The full-frame detector rarely fires its
helmet class at that size, so the helmet decision is made here instead: a square crop around the
head (or around where the head should be, from the person box), resized to 64x64, classified by a
small ImageNet-pretrained MobileNetV3 trained on Hard Hat Workers heads degraded to that pixel budget,
plus weakly labelled SteelBench heads from cameras outside the evaluation (scripts/build_steelbench_heads.py).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

INPUT = 64
CTX = 1.6  # crop side = CTX x head size: the head plus a little hair/shoulder context
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def square_crop(img, box, ctx=CTX, out=INPUT):
    """Square patch centred on box, side ctx * max(w, h), border-replicated, resized to out x out."""
    x1, y1, x2, y2 = box[:4]
    side = max(2, int(round(ctx * max(x2 - x1, y2 - y1))))
    patch = cv2.getRectSubPix(img, (side, side), ((x1 + x2) / 2, (y1 + y2) / 2))
    interp = cv2.INTER_AREA if side > out else cv2.INTER_LINEAR
    return cv2.resize(patch, (out, out), interpolation=interp)


def head_from_person(box):
    """Where the head should be when no head/helmet box was found: top-centre, ~16% of body height."""
    x1, y1, x2, y2 = box[:4]
    s = 0.16 * (y2 - y1)
    cx = (x1 + x2) / 2
    return (cx - s / 2, y1, cx + s / 2, y1 + s)


def build_model(pretrained=False):
    m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)  # 0 = helmet, 1 = bare head
    return m


def to_tensor(crops):
    """list/array of HxWx3 BGR uint8 crops -> normalised NCHW float tensor."""
    x = np.asarray(crops)[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2).copy())


class HelmetClassifier:
    """Uses the exported .onnx on ONNX Runtime CPU when present (0.7 ms/head on M2, 2 threads; PyTorch
    eager is ~4x slower on this tiny net and competes with the detector for the GPU)."""

    def __init__(self, weights, device="cpu", threads=2):
        weights = Path(weights)
        self.meta = json.loads(weights.with_suffix(".json").read_text())
        self.threshold = self.meta["threshold"]
        onnx_path = weights.with_suffix(".onnx")
        if onnx_path.exists():
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime-cpu"
        else:
            self.session = None
            self.device = device
            self.model = build_model()
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.model.eval().to(device)
            self.backend = f"torch-{device}"

    @torch.no_grad()
    def p_bare(self, frame, head_boxes):
        """Probability that each head is bare, one per box."""
        if not len(head_boxes):
            return np.zeros(0)
        x = to_tensor([square_crop(frame, b) for b in head_boxes])
        if self.session is not None:
            logits = self.session.run(None, {"x": x.numpy()})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            return e[:, 1] / e.sum(1)
        return torch.softmax(self.model(x.to(self.device)), 1)[:, 1].cpu().numpy()
