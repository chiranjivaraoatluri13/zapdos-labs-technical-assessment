"""Download a pinned demonstration clip and run the existing safety pipeline.

After the README's dependency/model setup:
  python scripts/run_sample.py single_person --cpu-only
  python scripts/run_sample.py --all --backend openvino --openvino-device GPU
  python scripts/run_sample.py --list

Polygons are illustrative. Selecting demonstrations is not an accuracy benchmark.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'samples/review_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(sample, cache):
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (sample['clip_id'] + '.mp4')
    expected = sample['sha256']
    if dest.exists():
        if digest(dest) != expected:
            raise RuntimeError(f'Cached video checksum differs: {dest}. Preserve or remove it before retrying.')
        print(f'Using verified cached video: {dest.name}', flush=True)
        return dest
    print(f'Downloading {sample["clip_id"]}', flush=True)
    with tempfile.NamedTemporaryFile(dir=cache, prefix=dest.stem + '-', suffix='.part', delete=False) as stream:
        partial = Path(stream.name)
        with urllib.request.urlopen(sample['source_url'], timeout=120) as response:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
    if digest(partial) != expected:
        raise RuntimeError(f'Download checksum differs; incomplete file retained at {partial}')
    partial.replace(dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sample', nargs='?', help='sample alias or full sample ID; use --list to see the selected videos')
    parser.add_argument('--all', action='store_true', help='run every selected demonstration sequentially')
    parser.add_argument('--list', action='store_true', help='list the selected samples without loading models')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--backend', choices=['auto', 'torch', 'onnx', 'openvino'])
    parser.add_argument('--openvino-device', choices=['CPU', 'GPU', 'NPU'])
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--process-fps', type=float, default=10)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/samples')
    parser.add_argument('--show', action='store_true', help='show a single video at its source pace for a screen recording')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    samples = manifest['samples']
    if args.list:
        for name, sample in samples.items():
            print(f'{name}: {sample["sample_id"]} ({sample["duration_s"]:g}s) - {sample["review_note"]}')
        return
    if args.all and args.sample:
        parser.error('use one sample name or --all')
    if args.sample not in samples:
        args.sample = next((name for name, sample in samples.items() if sample['sample_id'] == args.sample), args.sample)
    if not args.all and args.sample not in samples:
        parser.error('choose ' + ', '.join(samples) + ', a full sample ID, or --all')
    if args.show and args.all:
        parser.error('--show takes one sample')
    if args.process_fps <= 0:
        parser.error('--process-fps must be positive')
    out = args.out.resolve()
    chosen = list(samples.items()) if args.all else [(args.sample, samples[args.sample])]
    videos = [(name, sample, download(sample, ROOT / 'data/raw/review_videos')) for name, sample in chosen]
    if args.download_only:
        return
    # Keep all relative model/config paths consistent with the documented run.py command.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'src'))
    import run as entry
    from zapdos.pipeline import build_models, run
    from verify_run import verify
    settings = SimpleNamespace(config=str(ROOT / 'configs/rules.yaml'), backend=args.backend,
                               device=None, cpu_only=args.cpu_only, openvino_device=args.openvino_device,
                               process_fps=args.process_fps, no_idle_skip=True, log_readings=True, overwrite=False)
    cfg = entry.load_config(settings)
    models = build_models(cfg)
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown (source ZIP or git unavailable)'
    for name, sample, video in videos:
        print(f'\nRunning {name}; demonstration zone, not site policy.', flush=True)
        events, perf = run(video, ROOT / sample['zones'], out, cfg, camera_id=sample['sample_id'],
                           save_video=True, models=models, show=args.show)
        folder = out / sample['sample_id']
        checked = verify(folder)
        provenance = {'sample': name, 'source_url': sample['source_url'], 'video_sha256': sample['sha256'],
                      'dataset_revision': manifest['dataset_revision'], 'code_commit': commit,
                      'process_fps': args.process_fps, 'idle_skip': False, 'media_verification': checked,
                      'note': 'Media validity and demonstrations do not establish event accuracy.'}
        (folder / 'sample_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(f'VERIFIED {name}: {len(events)} alerts; {perf["throughput_fps"]} processed FPS; {checked}', flush=True)


if __name__ == '__main__':
    main()
