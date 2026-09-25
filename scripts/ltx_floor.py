"""Pilot LTX-Video codec floor on the derf pilot corpus.

    python scripts/ltx_floor.py [--dtype float32|bfloat16]

For every clip in the latest runs/corpus_*.json: stratify the SOURCE clips
(p99 flow, equal thirds, P-3 B4), round-trip through LTXCodec (G-1 settings),
and measure with the same instruments as Table III: motion energy (ME), motion
retained (MR), warping error (WE), PSNR, plus per-pair motion dispersion (Q-12).
Also: a repeat round trip on one clip (G-3 determinism) and peak VRAM (G-4).

Writes runs/ltx_floor_<UTC>.json. Pilot: N is small and FVMD is not yet
integrated, so this fills the paper's §V sentence, not Table II's FVMD column.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch

from codecfloor import flow, metrics, motion
from codecfloor.ltx_codec import LTXCodec

ROOT = Path(__file__).resolve().parents[1]


def dispersion(video, flows=None):
    f = flow.video_flow(video) if flows is None else flows
    per_pair = np.linalg.norm(f, axis=-1).mean(axis=(1, 2))
    return float(per_pair.std() / per_pair.mean()) if per_pair.mean() > 0 else float("nan")


def measure(video):
    f = flow.video_flow(video)
    errs = metrics.warping_error_per_pair(video, f)
    return {"warp_error": float(errs.mean()), "flow_mag": metrics.mean_flow_magnitude(video, f),
            "raw_frame_diff": metrics.raw_frame_diff(video), "motion_cv": dispersion(video, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    if not cv2.__version__.startswith("4."):
        sys.exit("OpenCV must be 4.x (DECISIONS P-1 B7)")

    corpus_recs = sorted((ROOT / "runs").glob("corpus_*.json"))
    corpus = {}
    for p in corpus_recs:  # later records win for a repeated clip
        corpus.update(json.load(open(p))["clips"])
    clips = {n: np.load(ROOT / "data" / "corpus" / f"{n}.npy") for n in sorted(corpus)}
    for n, v in clips.items():
        import hashlib
        if hashlib.sha256(v.tobytes()).hexdigest() != corpus[n]["sha256_clip"]:
            sys.exit(f"{n}: clip on disk does not match its corpus record")
    print(f"{len(clips)} clips from {[p.name for p in corpus_recs]}", flush=True)

    # Phase A: VAE only. With RAFT also resident the 12 GB card overflowed and
    # Windows spilled 4.4 GB to shared system memory (first attempt, ~36 min
    # before it was stopped), so the two models never share the GPU.
    torch.cuda.reset_peak_memory_stats()
    codec = LTXCodec(args.revision, dtype=args.dtype)
    codec_cfg = codec.config()
    recs, secs = {}, {}
    t_start = time.perf_counter()
    for i, (n, src) in enumerate(clips.items(), 1):
        t0 = time.perf_counter()
        recs[n] = codec.round_trip(src)
        torch.cuda.synchronize()
        secs[n] = time.perf_counter() - t0
        eta = (time.perf_counter() - t_start) / i * (len(clips) - i)
        print(f"[A {i:2d}/{len(clips)}] round trip {n:28s} {secs[n]:5.1f}s  (phase A ETA {eta / 60:4.1f} min)", flush=True)
    first = next(iter(clips))
    again = codec.round_trip(clips[first])  # G-3: identical input twice
    det = {"clip": first, "identical": bool(np.array_equal(recs[first], again)),
           "max_abs_diff": int(np.abs(recs[first].astype(int) - again.astype(int)).max())}
    peak_vae = torch.cuda.max_memory_allocated() / 2 ** 20
    del codec
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"phase A done: determinism {det}, peak VRAM {peak_vae:.0f} MiB", flush=True)

    # Phase B: RAFT only.
    torch.cuda.reset_peak_memory_stats()
    profiles, per_clip = {}, {}
    t_start = time.perf_counter()
    for i, (n, src) in enumerate(clips.items(), 1):
        profiles[n] = motion.profile(src)
        eta = (time.perf_counter() - t_start) / i * (len(clips) - i)
        print(f"[B {i:2d}/{len(clips)}] profiled {n:28s} (ETA {eta / 60:4.1f} min)", flush=True)
    cut = motion.assign_strata(list(profiles.values()))
    t_start = time.perf_counter()
    for i, (n, src) in enumerate(clips.items(), 1):
        rec, dt = recs[n], secs[n]
        s, r = measure(src), measure(rec)
        per_clip[n] = {
            "stratum": profiles[n].stratum, "profile": profiles[n].as_dict(),
            "source": s, "reconstruction": r,
            "motion_retained": r["flow_mag"] / s["flow_mag"],
            "delta_warp_abs": r["warp_error"] - s["warp_error"],
            "delta_warp_rel": (r["warp_error"] - s["warp_error"]) / s["warp_error"],
            "psnr": metrics.recon_psnr(src, rec), "seconds": dt,
        }
        c = per_clip[n]
        eta = (time.perf_counter() - t_start) / i * (len(clips) - i)
        print(f"[C {i:2d}/{len(clips)}] {n:28s} {c['stratum']:12s} MR {100 * c['motion_retained']:6.1f}%  "
              f"dWE {100 * c['delta_warp_rel']:+7.1f}%  PSNR {c['psnr']:5.2f}  (ETA {eta / 60:4.1f} min)", flush=True)

    strata = {}
    for st in ("slow_pan", "fast_action", "fine_texture"):
        cs = [c for c in per_clip.values() if c["stratum"] == st]
        strata[st] = {"n": len(cs), **{k: {"mean": float(np.mean([c[k] for c in cs])),
                                            "min": float(np.min([c[k] for c in cs])),
                                            "max": float(np.max([c[k] for c in cs]))}
                                        for k in ("motion_retained", "delta_warp_rel", "delta_warp_abs", "psnr")}}

    now = datetime.now(timezone.utc)
    record = {
        "kind": "ltx_floor_pilot", "created_utc": now.isoformat(timespec="seconds"),
        "label": "LTX-Video 0.9.5 VAE round trip, pilot corpus (Xiph derf), N small; FVMD not included",
        "corpus_records": [p.name for p in corpus_recs], "codec": codec_cfg, "flow": flow.config(),
        "strata_cutpoints": cut, "determinism": det,
        "peak_vram_mib": {"vae_phase": peak_vae, "raft_phase": torch.cuda.max_memory_allocated() / 2 ** 20},
        "environment": {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
                        **{p: version(p) for p in ("numpy", "opencv-python-headless", "diffusers", "torch")}},
        "clips": per_clip, "strata": strata,
    }
    out = ROOT / "runs" / f"ltx_floor_{now:%Y%m%dT%H%M%SZ}.json"
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"\ndeterminism: {det}\npeak VRAM MiB {record['peak_vram_mib']}")
    for st, v in strata.items():
        print(f"{st:13s} n={v['n']}  MR {100 * v['motion_retained']['mean']:5.1f}%  "
              f"dWE rel {100 * v['delta_warp_rel']['mean']:+6.1f}%  PSNR {v['psnr']['mean']:5.2f}")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
