"""One-pass PPE detector: persons, heads, helmets and vests from a single SH17 YOLO forward pass."""
import ast
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


def available_device(want):
    """The configured torch device if this machine has it, else CUDA, else CPU (the default config
    asks for mps, which only Apple silicon has)."""
    import torch
    present = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if present.get(str(want).split(":")[0], False):
        return want
    fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device {want} not available here; using {fallback}")
    return fallback


def detector_backend(cfg):
    """Which backend PPEDetector will load for this config: onnx | openvino | coreml | torch."""
    backend = cfg.get("backend", "auto")
    if backend in ("onnx", "openvino", "torch"):
        return backend
    coreml = cfg.get("coreml")
    if coreml and sys.platform == "darwin" and Path(coreml).exists():
        return "coreml"
    return "torch"


def openvino_device(cfg):
    return str(cfg.get("openvino_device", "CPU")).upper()


def detector_model_path(cfg):
    """The model file (or export directory) the detector actually loads. `openvino` is one export
    directory, or a mapping from device (CPU/GPU/NPU) to export directory."""
    ov = cfg.get("openvino", "weights/sh17/yolo8s_736x1280_openvino_model")
    if isinstance(ov, dict):
        ov = ov[openvino_device(cfg)]
    return Path({"onnx": cfg.get("onnx", "weights/sh17/yolo8s_736x1280.onnx"),
                 "openvino": ov,
                 "coreml": cfg.get("coreml") or "",
                 "torch": cfg["weights"]}[detector_backend(cfg)])


def model_fingerprint(path):
    """sha256 of a model file, or of every file (name and bytes) in an export directory."""
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"detector model not found: {path}")
    for f in files:
        h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass
class Detections:
    persons: object      # ultralytics Boxes (numpy), passed straight to the tracker
    heads: np.ndarray    # (N, 5): x1, y1, x2, y2, conf
    helmets: np.ndarray
    vests: np.ndarray


class PPEDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.imgsz = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.cls = cfg["classes"]
        self.min_conf = cfg["min_conf"]
        coreml = cfg.get("coreml")
        units = cfg.get("coreml_units", "cpu_and_ne")
        backend = detector_backend(cfg)
        self.model_path = detector_model_path(cfg)
        # ONNX and OpenVINO stay explicit rather than changing performance when an export happens to exist.
        if backend == "onnx":
            self._load_onnx(self.model_path)
            self.model, self.device, self.backend = None, "cpu", "onnxruntime-cpu"
            names = self.names
        elif backend == "openvino":
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"{self.model_path}: export first with scripts/export_models.py --detector --openvino")
            dev = openvino_device(cfg)
            if dev == "CPU":  # loaded directly: ultralytics leaves OpenVINO at its default thread count
                self._load_openvino_cpu(self.model_path)
                self.model, self.device, names = None, "cpu", self.names
                self.backend = "openvino-cpu" + ("-int8" if "int8" in self.model_path.name else "")
            else:
                self.model, self.device, self.backend = YOLO(str(self.model_path), task="detect"), f"intel:{dev.lower()}", f"openvino-{dev.lower()}"
                names = self.model.names
        elif backend == "coreml":
            # input shape comes from the export metadata. ultralytics loads CoreML models for CPU +
            # Neural Engine; only cpu_only changes that (below), other coreml_units values are a label
            self.model, self.device, self.backend = YOLO(coreml, task="detect"), None, f"coreml-{units}"
            names = self.model.names
        else:
            self.model, self.device, self.backend = YOLO(cfg["weights"]), cfg["device"], f"torch-{cfg['device']}"
            names = self.model.names
        for kind, expected in {"person": "person", "head": "head", "helmet": "helmet", "vest": "safety-vest"}.items():
            if names[self.cls[kind]] != expected:
                raise ValueError(f"detector class {kind} does not match SH17: {names[self.cls[kind]]}")
        # first MPS/CPU call compiles kernels; keep it out of the measured latency
        self(np.zeros((1080, 1920, 3), np.uint8))
        if self.backend == "coreml-cpu_only":  # swap the loaded model for a CPU-only one (no NE, no GPU)
            import coremltools as ct
            self.model.predictor.model.backend.model = ct.models.MLModel(coreml, compute_units=ct.ComputeUnit.CPU_ONLY)
            self(np.zeros((1080, 1920, 3), np.uint8))

    def __call__(self, frame):
        if self.model is None:  # ONNX Runtime or OpenVINO on the CPU, run directly
            boxes = self._raw_boxes(frame)
        else:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                   iou=self.cfg.get("iou", .7), classes=list(self.cls.values()), verbose=False)[0]
            boxes = r.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)

        def pick(name):
            m = (cls == self.cls[name]) & (boxes.conf >= self.min_conf[name])
            return np.hstack([boxes.xyxy[m], boxes.conf[m, None]])

        return Detections(boxes[cls == self.cls["person"]], pick("head"), pick("helmet"), pick("vest"))

    def _load_onnx(self, path):
        import onnxruntime as ort
        from ultralytics.data.augment import LetterBox
        if not path.is_file():
            raise FileNotFoundError(f"{path}: export first with scripts/export_models.py --detector --onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.cfg.get("onnx_threads", 4)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"])
        self.input_shape = tuple(self.session.get_inputs()[0].shape[2:])
        if max(self.input_shape) != self.imgsz:
            raise ValueError(f"ONNX input {self.input_shape} does not match imgsz={self.imgsz}")
        self.input_name = self.session.get_inputs()[0].name
        self.letterbox = LetterBox(self.input_shape, auto=False, stride=32)
        self.infer = lambda blob: self.session.run(None, {self.input_name: blob})[0]

    def _load_openvino_cpu(self, path):
        """OpenVINO on the CPU with an explicit thread count: on hybrid Intel CPUs the runtime's default
        uses only the performance cores (Core Ultra 7 256V: 4 of 8, INT8 160 ms vs 83 ms with all 8)."""
        import openvino as ov
        import yaml
        from ultralytics.data.augment import LetterBox
        core = ov.Core()
        model = core.read_model(next(path.glob("*.xml")))
        opts = {"PERFORMANCE_HINT": "LATENCY"}
        threads = self.cfg.get("openvino_threads")
        if threads is not None:
            opts["INFERENCE_NUM_THREADS"] = str(threads or os.cpu_count())
        request = core.compile_model(model, "CPU", opts).create_infer_request()
        meta = yaml.safe_load((path / "metadata.yaml").read_text())
        self.names = meta["names"]
        self.input_shape = tuple(int(d) for d in model.inputs[0].shape[2:])
        if max(self.input_shape) != self.imgsz:
            raise ValueError(f"OpenVINO input {self.input_shape} does not match imgsz={self.imgsz}")
        self.letterbox = LetterBox(self.input_shape, auto=False, stride=32)
        self.infer = lambda blob: request.infer({0: blob})[0]

    def _raw_boxes(self, frame):
        import torch
        from ultralytics.utils import nms, ops
        img = self.letterbox(image=frame)
        blob = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.
        raw = self.infer(blob)
        det = nms.non_max_suppression(torch.from_numpy(raw), self.conf, self.cfg.get("iou", .7),
                                      list(self.cls.values()), max_det=300)[0]
        det[:, :4] = ops.scale_boxes(blob.shape[2:], det[:, :4], frame.shape)
        return Boxes(det[:, :6].numpy(), frame.shape[:2])
