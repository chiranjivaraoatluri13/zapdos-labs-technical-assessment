"""Fetch the public inputs.

  default        SteelBench 50-clip sample (mp4 + annotations), SH17 PPE weights, and the six labelled
                 clips linked under short names in clips/
  --full-labels  SteelBench full-release labels: 1,342 annotation JSONs + manifests (camera_id per clip)
  --hardhat      Hard Hat Workers train and valid splits (COCO export, HF mirror)
The last two are only needed to retrain the helmet classifier and can be combined.
scripts/build_steelbench_heads.py fetches the full-release frames it needs itself (frame_jobs).

Usage: python scripts/download_data.py [--weights yolo8s.pt yolo8n.pt] [--full-labels] [--hardhat]
"""
import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_API = "https://huggingface.co/api/datasets/ThinkingHub/PP/tree/main/"
HF_FILE = "https://huggingface.co/datasets/ThinkingHub/PP/resolve/main/"
SH17_RELEASE = "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/"
MANIFESTS = ("batch_manifest.csv", "camera_zones.csv", "gt_clips.json", "safety_rules.yaml")
# the six hand-labelled clips under short names in clips/: dev (labels/), then held out (labels_heldout/)
LABELLED = {
    "sb_crm_workshop_0109": "clip_CRM_Work_Shop_20251115_002718_0109",
    "sb_red_lrs1_0007": "clip_RED_LRS-1_20251111_000204_0007",
    "sb_tippler_lowlight_0062": "clip_Tippler_3&4_19.1-19.2TEND_20251205_001622_0062",
    # held out: cameras not used to develop the rules
    "sb_bf_cdi_cc2_0031": "clip_BF_CDI_Ground_CC2_20251111_000745_0031",
    "sb_rers1_0012": "clip_RERS_RERS-1_20251118_000308_0012",
    "sb_tar_lowlight_0000": "clip_TAR_Plant_DC-Floor_left_20251115_000000_0000",
}
FRAMES_PER_CLIP = 8
HARDHAT = "https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/"


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip  {dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return f"ok    {dest.name}"


def hf_ls(path):
    """Every entry under a repo folder (the tree API pages at 1000)."""
    url, items = HF_API + urllib.parse.quote(path) + "?limit=1000", []
    while url:
        with urllib.request.urlopen(url) as r:
            items += json.load(r)
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
            url = m.group(1) if m else None
    return items


def hf_url(path):
    return HF_FILE + urllib.parse.quote(path)


def sample_jobs(out_dir):
    return [(hf_url(i["path"]), out_dir / Path(i["path"]).relative_to("sample"))
            for sub in ("sample/clips", "sample/annotations") for i in hf_ls(sub)]


def full_label_jobs(out_dir):
    jobs = [(hf_url(f"manifests/{m}"), out_dir / "manifests" / m) for m in MANIFESTS]
    return jobs + [(hf_url(i["path"]), out_dir / "annotations" / Path(i["path"]).name)
                   for i in hf_ls("data/annotations")]


def frame_jobs(out_dir, clip_ids):
    return [(hf_url(f"data/frames/{c}/frame_{k}.jpg"), out_dir / "frames" / c / f"frame_{k}.jpg")
            for c in clip_ids for k in range(FRAMES_PER_CLIP)]


def fetch_all(jobs):
    done = 0
    with ThreadPoolExecutor(16) as pool:
        for msg in pool.map(lambda j: fetch(*j), jobs):
            done += 1
            if len(jobs) < 100 or done % 200 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {msg}", flush=True)


def fetch_hardhat(out):
    for split in ("train", "valid"):  # scripts/train_helmet.py trains on train, scores on valid
        z = out / f"{split}.zip"
        print(fetch(HARDHAT + f"{split}.zip", z), flush=True)
        if not (out / split / "_annotations.coco.json").exists():
            zipfile.ZipFile(z).extractall(out / split)


def link_labelled_clips():
    for short, clip in LABELLED.items():
        link = ROOT / "clips" / f"{short}.mp4"
        if not link.exists():
            source = ROOT / "data/raw/steelbench/clips" / f"{clip}.mp4"
            try:
                link.symlink_to(source)
            except OSError:  # Windows often requires elevated rights for symlinks
                shutil.copyfile(source, link)
    print(f"labelled clips linked in clips/: {', '.join(LABELLED)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=["yolo8s.pt", "yolo8n.pt"])
    ap.add_argument("--full-labels", action="store_true")
    ap.add_argument("--hardhat", action="store_true")
    args = ap.parse_args()

    if args.hardhat:
        fetch_hardhat(ROOT / "data/raw/hardhat")
    if args.full_labels:
        fetch_all(full_label_jobs(ROOT / "data/raw/steelbench_full"))
    if args.hardhat or args.full_labels:
        return
    fetch_all(sample_jobs(ROOT / "data/raw/steelbench")
              + [(SH17_RELEASE + w, ROOT / "weights/sh17" / w) for w in args.weights])
    link_labelled_clips()


if __name__ == "__main__":
    main()
