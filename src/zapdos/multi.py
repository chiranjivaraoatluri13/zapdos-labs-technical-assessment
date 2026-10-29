"""Several cameras in one process with one copy of the models.

Each camera decodes in its own thread into a short queue. One thread runs the detector on the cameras
in turn (the Neural Engine or GPU), and the calling thread runs tracking, helmet crops, rules and media
for each frame the detector hands over, so accelerator and CPU work overlap. A camera's frames stay
in order, so its alerts match a single-camera run of the same video. A failure in a decode or detect
thread is raised in the calling thread instead of leaving it waiting.

Per-camera fps in the summary is frames over the shared wall time, so it assumes videos of equal
length (the benchmark loops one clip per camera); a shorter video would look slow.
"""
import queue
import sys
import threading
import time

from .pipeline import Camera, build_models

END = object()


def _decode(cam, q):
    try:
        for item in cam.frames():
            q.put(item)
    except Exception as e:  # handed on through the detect thread
        q.put(e)
    finally:
        q.put(END)


def _detect(cams, frame_qs, out_q, detector):
    live = list(range(len(cams)))
    try:
        while live:
            for i in list(live):
                item = frame_qs[i].get()
                if isinstance(item, Exception):
                    raise item
                if item is END:
                    live.remove(i)
                    out_q.put((i, END, None, None))
                    continue
                if item[2] is None:  # a sample the camera skipped (adaptive sampling)
                    out_q.put((i, item, None, None))
                    continue
                t0 = time.perf_counter()
                det = detector(item[2])
                out_q.put((i, item, det, (time.perf_counter() - t0) * 1000))
    except Exception as e:
        out_q.put((None, e, None, None))


def run_many(sources, out_root, cfg, models=None, quiet=False, queue_frames=4):
    """sources: [(video, zones_path or None, camera_id or None)]. Returns {camera_id: (events, perf)} and a
    summary of the whole process (total throughput, slowest camera, peak memory)."""
    models = models or build_models(cfg)
    cams = [Camera(v, z, out_root, cfg, models, cid) for v, z, cid in sources]
    for cam in cams:
        cam.tag = f"{cam.camera_id} "
    frame_qs = [queue.Queue(queue_frames) for _ in cams]
    out_q = queue.Queue(queue_frames * len(cams))

    wall0 = time.perf_counter()
    threads = [threading.Thread(target=_decode, args=(c, q), daemon=True) for c, q in zip(cams, frame_qs)]
    threads.append(threading.Thread(target=_detect, args=(cams, frame_qs, out_q, models[0]), daemon=True))
    for t in threads:
        t.start()
    done = 0
    while done < len(cams):
        i, item, det, detect_ms = out_q.get()
        if isinstance(item, Exception):
            raise RuntimeError("a decode or detect thread failed") from item
        if item is END:
            done += 1
            continue
        cams[i].step(*item, det=det, detect_ms=detect_ms, quiet=quiet)
    wall = time.perf_counter() - wall0

    results = {cam.camera_id: cam.close(wall0) for cam in cams}
    frames = sum(len(cam.ms["detect"]) for cam in cams)
    # samples handled per second, skipped ones included: the camera rate each one keeps up with
    per_cam = [(len(cam.ms["detect"]) + cam.skipped) / wall for cam in cams]
    summary = {"cameras": len(cams), "process_fps": cams[0].fps, "wall_s": round(wall, 1),
               "total_fps": round(frames / wall, 1), "min_camera_fps": round(min(per_cam), 1),
               "keeps_up": min(per_cam) >= cams[0].fps, "peak_rss_gb": _peak_rss_gb()}
    return results, summary


def _peak_rss_gb():
    try:
        import resource  # not on Windows
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KB on Linux
    return round(rss * (1 if sys.platform == "darwin" else 1024) / 1e9, 2)
