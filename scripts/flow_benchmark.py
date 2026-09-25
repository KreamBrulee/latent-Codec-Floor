"""Optical-flow estimator selection against exact ground truth (P-3 B5).

    python scripts/flow_benchmark.py

DECISION RULE, fixed before any RAFT result was seen (2026-09-25):
  Choose the estimator with the lowest mean endpoint error (EPE, px) averaged
  over all probes below, each probe weighted equally. If two are within 10%
  of each other, prefer the faster one. Disqualify an estimator whose repeated
  runs on identical input differ by more than 1% of its EPE (the metric would
  not be reproducible).
  The effect of the choice on any paper number is NOT a criterion.

Probes (ground truth known exactly by construction):
  * textured pans at 0.6, 2.5, 5, 10, 20 px/frame along a 30 degree direction:
    uniform flow = -velocity (content moves opposite to the window)
  * fast_action (textured): ball pixels move by their ball's centre
    displacement; static background has zero flow. Pixels covered by a ball
    in either frame of the pair, other than the moving ball itself, are
    excluded (occlusion: flow undefined).

Writes runs/flow_benchmark_<UTC>.json.
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from codecfloor import synthetic
from codecfloor.flow import BORDER, farneback
from codecfloor.video_io import to_gray

RUNS = Path(__file__).resolve().parents[1] / "runs"
PAN_SPEEDS = [0.6, 2.5, 5.0, 10.0, 20.0]
ANGLE = math.radians(30)
T = 9  # 8 pairs per probe: enough to average, cheap enough for CPU RAFT


def pan_probe(speed: float):
    canvas = synthetic._texture_bg(1024, 1024, seed=4)
    vx, vy = speed * math.cos(ANGLE), speed * math.sin(ANGLE)
    clip = np.stack([synthetic._window(canvas, 64 + i * vx, 64 + i * vy, 256, 256) for i in range(T)])
    gt = np.broadcast_to(np.array([-vx, -vy], np.float32), (T - 1, 256, 256, 2))
    valid = np.zeros((T - 1, 256, 256), bool)
    valid[:, BORDER:-BORDER, BORDER:-BORDER] = True
    return clip, gt, valid


def ball_probe():
    clip = synthetic.fast_action(t=T)
    tracks = synthetic.ball_tracks(t=T)
    yy, xx = np.mgrid[:256, :256]
    gt = np.zeros((T - 1, 256, 256, 2), np.float32)
    valid = np.ones((T - 1, 256, 256), bool)
    for i in range(T - 1):
        masks0 = [(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r for cx, cy, r in tracks[i]]
        masks1 = [(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r for cx, cy, r in tracks[i + 1]]
        # Later balls are drawn over earlier ones; a pixel belongs to the topmost.
        owner = np.full((256, 256), -1)
        for k, m in enumerate(masks0):
            owner[m] = k
        for k, ((c0x, c0y, _), (c1x, c1y, _)) in enumerate(zip(tracks[i], tracks[i + 1])):
            gt[i][owner == k] = (c1x - c0x, c1y - c0y)
        covered1 = np.any(masks1, axis=0)
        # Background that a ball covers in the next frame: occluded.
        valid[i][(owner == -1) & covered1] = False
        # Ball pixels overlapping another ball in either frame: ambiguous.
        for k in range(len(masks0)):
            others = [m for j, m in enumerate(masks1) if j != k]
            valid[i][(owner == k) & np.any(others, axis=0)] = False
        valid[i][:BORDER] = valid[i][-BORDER:] = False
        valid[i][:, :BORDER] = valid[i][:, -BORDER:] = False
    ball = np.stack([(owner_mask(tracks[i]) >= 0) for i in range(T - 1)])
    return clip, gt, valid, ball


def owner_mask(balls):
    yy, xx = np.mgrid[:256, :256]
    owner = np.full((256, 256), -1)
    for k, (cx, cy, r) in enumerate(balls):
        owner[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = k
    return owner


# ---------------------------------------------------------------- estimators
def est_farneback(clip):
    g = (to_gray(clip) * 255).astype(np.uint8)
    return np.stack([farneback(g[i], g[i + 1]) for i in range(len(g) - 1)])


def make_raft(size: str, device: str):
    import torch
    from torchvision.models.optical_flow import (Raft_Large_Weights, Raft_Small_Weights,
                                                 raft_large, raft_small)
    weights = Raft_Large_Weights.DEFAULT if size == "large" else Raft_Small_Weights.DEFAULT
    model = (raft_large if size == "large" else raft_small)(weights=weights).eval().to(device)
    tf = weights.transforms()

    @torch.no_grad()
    def run(clip):
        x = torch.from_numpy(clip).permute(0, 3, 1, 2)
        a, b = tf(x[:-1], x[1:])
        out = model(a.to(device), b.to(device))[-1]
        return out.permute(0, 2, 3, 1).float().cpu().numpy()

    run.weights = str(weights)
    return run


def evaluate(fn, probes, repeats=2):
    res, t_total, n_pairs = {}, 0.0, 0
    drift = 0.0
    for name, (clip, gt, valid, region) in probes.items():
        flows = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            flows.append(fn(clip))
            t_total += time.perf_counter() - t0
            n_pairs += len(clip) - 1
        epe = np.linalg.norm(flows[0] - gt, axis=-1)
        entry = {"epe": float(epe[valid].mean())}
        if region is not None:
            entry["epe_ball"] = float(epe[valid & region].mean())
            entry["epe_background"] = float(epe[valid & ~region].mean())
            gmag = np.linalg.norm(gt, axis=-1)[valid & region]
            fmag = np.linalg.norm(flows[0], axis=-1)[valid & region]
            entry["ball_speed_recovered"] = float(fmag.mean() / gmag.mean())
        else:
            entry["speed_recovered"] = float(np.linalg.norm(flows[0], axis=-1)[valid].mean()
                                             / np.linalg.norm(gt, axis=-1)[valid].mean())
        d = float(np.abs(flows[0] - flows[1]).max())
        entry["repeat_max_abs_diff"] = d
        drift = max(drift, d / max(entry["epe"], 1e-9))
        res[name] = entry
    return {
        "probes": res,
        "mean_epe": float(np.mean([r["epe"] for r in res.values()])),
        "ms_per_pair": 1000 * t_total / n_pairs,
        "max_repeat_diff_over_epe": drift,
    }


def main():
    if not cv2.__version__.startswith("4."):
        sys.exit(f"refusing to run on OpenCV {cv2.__version__} (DECISIONS P-1 B7)")
    probes = {f"pan_{s:g}px": (*pan_probe(s), None) for s in PAN_SPEEDS}
    clip, gt, valid, ball = ball_probe()
    probes["balls"] = (clip, gt, valid, ball)

    estimators = {"farneback": est_farneback}
    env = {"python": platform.python_version(),
           **{p: version(p) for p in ("numpy", "opencv-python-headless")}}
    try:
        import torch
        env.update(torch=torch.__version__, torchvision=version("torchvision"),
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for size in ("small", "large"):
            estimators[f"raft_{size}_{dev}"] = make_raft(size, dev)
        if dev == "cuda":
            estimators["raft_large_cpu"] = make_raft("large", "cpu")
    except ImportError:
        print("torch not installed: benchmarking Farneback only")

    results = {}
    for name, fn in estimators.items():
        print(f"running {name} ...", flush=True)
        results[name] = evaluate(fn, probes)

    now = datetime.now(timezone.utc)
    record = {"kind": "flow_benchmark", "created_utc": now.isoformat(timespec="seconds"),
              "decision_rule": __doc__.split("Probes")[0].strip(),
              "config": {"pan_speeds": PAN_SPEEDS, "angle_deg": 30, "frames": T, "border": BORDER},
              "environment": env, "results": results}
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"flow_benchmark_{now:%Y%m%dT%H%M%SZ}.json"
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    cols = list(probes)
    print(f"\n{'estimator':18s}" + "".join(f"{c:>11s}" for c in cols) + f"{'mean EPE':>10s}{'ms/pair':>9s}{'repeat':>9s}")
    for name, r in results.items():
        print(f"{name:18s}" + "".join(f"{r['probes'][c]['epe']:11.3f}" for c in cols)
              + f"{r['mean_epe']:10.3f}{r['ms_per_pair']:9.1f}{r['max_repeat_diff_over_epe']:9.1e}")
    print("\nball probe:", {n: {k: round(v, 3) for k, v in r["probes"]["balls"].items()} for n, r in results.items()})
    print(f"-> {out.relative_to(RUNS.parent)}")


if __name__ == "__main__":
    main()
