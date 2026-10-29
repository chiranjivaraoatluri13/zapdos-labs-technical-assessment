"""Train the head-crop helmet classifier on Hard Hat Workers, degraded to SteelBench's pixel budget.

Hard Hat Workers heads are 2-3x larger than SteelBench's (median 64 vs 26 px), and its bare heads
are larger on average than its helmeted ones, so a model could learn "big = bare". Every training
crop is therefore shrunk to a random 22-80 px crop side (a 14-50 px head) and resized back to 64,
then blurred / JPEG-compressed / darkened / colour-shifted / face-pixelated like the target footage.

In-domain crops from scripts/build_steelbench_heads.py (--extra) are mixed in, oversampled, from
cameras outside the evaluation set. A few helmet-only cameras among them are held out to set the
operating threshold: the false-bare rate on real steel-plant helmets is what decides alert precision,
and a threshold set on Hard Hat Workers does not transfer to it.

Inputs: python scripts/download_data.py --hardhat --full-labels, then scripts/build_steelbench_heads.py.

python scripts/train_helmet.py                       # writes the .pt and its .json (threshold)
python scripts/export_models.py --classifier         # inference loads the .onnx next to the .pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.helmet import CTX, INPUT, build_model, square_crop, to_tensor  # noqa: E402

STORE, STORE_CTX = 80, 2.0  # cached crops keep extra context so training can jitter the window


def coco_heads(data, split):
    """{image file: [(bbox, bare)]} for the head boxes of one Hard Hat Workers split (>= 8 px tall)."""
    coco = json.loads((data / split / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        name = cats[a["category_id"]]
        if name in ("hardhat", "no-hardhat") and a["bbox"][3] >= 8:
            by_img.setdefault(files[a["image_id"]], []).append((a["bbox"], int(name == "no-hardhat")))
    return by_img


def load_split(data, split, cache, max_helmet_ratio):
    """Cached head crops of one split, helmets subsampled to at most max_helmet_ratio x the bare heads."""
    path = cache / f"hhw_{split}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], z["y"]
    by_img = coco_heads(data, split)
    n_bare = sum(lab for v in by_img.values() for _, lab in v)
    keep_helmet = min(1.0, max_helmet_ratio * n_bare / max(1, sum(len(v) for v in by_img.values()) - n_bare))
    rng = np.random.default_rng(0)  # its own seed: the same crops whether or not the cache exists
    xs, ys = [], []
    for file, anns in by_img.items():
        img = cv2.imread(str(data / split / file))
        for (x, y, w, h), lab in anns:
            if lab == 0 and rng.random() > keep_helmet:
                continue
            xs.append(square_crop(img, (x, y, x + w, y + h), ctx=STORE_CTX, out=STORE))
            ys.append(lab)
    x, y = np.stack(xs), np.array(ys, np.int64)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y)
    return x, y


def degrade(img, rng):
    """One training view: jittered head window -> pixel budget -> camera-like corruption -> 64x64."""
    side = STORE * CTX / STORE_CTX * rng.uniform(0.85, 1.2)
    c = STORE / 2 + rng.uniform(-0.1, 0.1, 2) * side
    win = cv2.getRectSubPix(img, (int(side), int(side)), (float(c[0]), float(c[1])))
    s = int(rng.uniform(22, 80))  # crop side on the target camera = head of ~14-50 px
    win = cv2.resize(cv2.resize(win, (s, s), interpolation=cv2.INTER_AREA), (INPUT, INPUT))
    if rng.random() < 0.5:
        win = win[:, ::-1]
    if rng.random() < 0.5:  # hue / saturation shift: yellow vs white helmets, sodium lighting
        hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
        hsv[..., 1] *= rng.uniform(0.3, 1.3)
        win = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    f = win.astype(np.float32) / 255
    if rng.random() < 0.3:  # low light: gamma + sensor noise
        f = f ** rng.uniform(1.5, 2.8) + rng.normal(0, rng.uniform(0.01, 0.05), f.shape)
    f = (f - 0.5) * rng.uniform(0.6, 1.3) + 0.5 + rng.uniform(-0.15, 0.15)  # contrast / brightness, haze
    win = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.3:
        win = cv2.GaussianBlur(win, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.15:  # SteelBench face-blur patch over the lower face
        y0, x0 = int(INPUT * rng.uniform(0.4, 0.55)), int(INPUT * rng.uniform(0.3, 0.4))
        p = win[y0:y0 + INPUT // 4, x0:x0 + INPUT // 3]
        win[y0:y0 + INPUT // 4, x0:x0 + INPUT // 3] = cv2.resize(cv2.resize(p, (3, 3)), p.shape[1::-1],
                                                                  interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.1:
        win = cv2.cvtColor(cv2.cvtColor(win, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # IR / night
    if rng.random() < 0.5:
        _, enc = cv2.imencode(".jpg", win, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 90))])
        win = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(win)


def batches(x, y, bs, rng, augment):
    order = rng.permutation(len(x)) if augment else np.arange(len(x))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        imgs = [degrade(x[j], rng) for j in idx]  # evaluation views are degraded too, from a fixed seed (predict)
        yield to_tensor(imgs), torch.from_numpy(y[idx])


@torch.no_grad()
def predict(model, x, y, dev, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)  # fixed degradations so epochs are comparable
    ps = []
    for xb, _ in batches(x, y, 512, rng, augment=False):
        ps.append(torch.softmax(model(xb.to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


def operating_point(p, y, max_fpr):
    """Lowest threshold whose false-bare rate on helmeted heads is <= max_fpr."""
    thr = float(np.quantile(p[y == 0], 1 - max_fpr))
    return thr, float((p[y == 1] >= thr).mean()), float((p[y == 0] >= thr).mean())


def evaluate(model, xva, yva, xcal, max_fpr, dev):
    """(threshold, HHW recall on bare, false-bare rate, log line). The threshold allows max_fpr false
    "bare" on the steel-plant calibration helmets when given (xcal), else on HHW valid helmets."""
    p = predict(model, xva, yva, dev)
    thr, tpr, fpr = operating_point(p, yva, max_fpr)
    acc = float(((p >= 0.5) == yva).mean())
    line = f"valid acc {acc:.3f}  HHW recall on bare {tpr:.1%} at false-bare {fpr:.1%} (thr {thr:.2f})"
    if xcal is None:
        return thr, tpr, fpr, line
    pcal = predict(model, xcal, np.zeros(len(xcal), np.int64), dev)
    thr = float(np.quantile(pcal, 1 - max_fpr))
    fpr = float((pcal >= thr).mean())
    tpr = float((p[yva == 1] >= thr).mean())
    line += f" | steel-plant thr {thr:.2f}: HHW recall {tpr:.1%}, HHW false-bare {(p[yva == 0] >= thr).mean():.1%}"
    return thr, tpr, fpr, line


def load_extra(path, repeat, calib_frac):
    """In-domain crops -> (train x, train y, calibration x). Calibration = every 1/calib_frac-th
    helmet-only camera (sorted by name), never trained on."""
    z = np.load(path)
    x, y, cam = z["x"], z["y"], z["cam"]
    bare_cams = set(cam[y == 1])
    helmet_only = sorted(set(cam) - bare_cams)
    calib = set(helmet_only[::round(1 / calib_frac)])
    is_cal = np.isin(cam, list(calib))
    tr = np.where(~is_cal)[0]
    tr = np.concatenate([np.repeat(tr[y[tr] == 0], repeat[0]), np.repeat(tr[y[tr] == 1], repeat[1])])
    return x[tr], y[tr], x[is_cal], sorted(calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/hardhat")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="weights/helmet_cls/mnv3s_64_sb2.pt", help="the model configs/rules.yaml uses")
    ap.add_argument("--extra", default="data/cache/sb_heads.npz", help="in-domain crops; '' = HHW only")
    ap.add_argument("--extra-repeat", type=int, nargs=2, default=[3, 15], metavar=("HELMET", "BARE"),
                    help="times each in-domain helmet / bare crop is repeated per epoch")
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-fpr", type=float, default=0.02)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    data, cache = ROOT / args.data, ROOT / args.cache
    t0 = time.time()
    xtr, ytr = load_split(data, "train", cache, 2.0)
    xva, yva = load_split(data, "valid", cache, 2.0)
    xcal, calib_cams = None, []
    if args.extra:
        xex, yex, xcal, calib_cams = load_extra(ROOT / args.extra, args.extra_repeat, args.calib_frac)
        xtr, ytr = np.concatenate([xtr, xex]), np.concatenate([ytr, yex])
        print(f"in-domain: {len(xex)} training views ({int(yex.sum())} bare), {len(xcal)} calibration helmets "
              f"from {calib_cams}")
    print(f"crops: train {len(xtr)} ({ytr.mean():.0%} bare), valid {len(xva)} ({yva.mean():.0%} bare) "
          f"[{time.time() - t0:.0f}s]")

    model = build_model(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(xtr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.15)
    w = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    best, out = -1.0, ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in batches(xtr, ytr, args.bs, rng, augment=True):
            loss = loss_fn(model(xb.to(args.device)), yb.to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(yb), n + len(yb)
        thr, tpr, fpr, line = evaluate(model, xva, yva, xcal, args.max_fpr, args.device)
        print(f"epoch {ep + 1}: loss {tot / n:.3f}  {line}  [{time.time() - t0:.0f}s]")
        if tpr > best:
            best = tpr
            torch.save(model.state_dict(), out)
            meta = {"input": INPUT, "ctx": CTX, "threshold": thr, "valid_recall_at_thr": tpr,
                    "false_bare_at_thr": fpr, "threshold_set_on": "steel-plant calibration helmets" if xcal
                    is not None else "HHW valid", "calibration_cameras": calib_cams,
                    "epoch": ep + 1, "train_crops": int(len(xtr)),
                    "data": "Hard Hat Workers v2 (Roboflow, via HF keremberke/hard-hat-detection)"
                            + (f" + SteelBench weak labels ({args.extra})" if args.extra else ""),
                    "base": "torchvision mobilenet_v3_small IMAGENET1K_V1"}
            out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"best: HHW recall on bare {best:.1%} at the {args.max_fpr:.0%} false-bare threshold -> {out}")


if __name__ == "__main__":
    main()
