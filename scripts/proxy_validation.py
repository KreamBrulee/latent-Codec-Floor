"""Proxy-codec validation record: source of paper Table III and handover F-1/F-2.

    python scripts/proxy_validation.py

Writes runs/proxy_validation_<UTC timestamp>.json and prints the table. The
JSON is the source of truth; the paper table is transcribed from it.

Every value produced here is from a proxy codec on synthetic clips. Label any
use of it "proxy codec (temporal decimation x8), NOT the LTX-Video VAE".
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2

from codecfloor import flow, metrics, synthetic
from codecfloor.proxy_codec import ProxyCodec

FACTOR = 8
MODE = "linear"
RUNS = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    # cv2.remap (used to warp in warping error) changed in 5.0 and moved the
    # metric by up to 6% (Q-7).
    if not cv2.__version__.startswith("4."):
        sys.exit(f"refusing to run on OpenCV {cv2.__version__}; pinned to 4.x (DECISIONS P-1 B7)")

    codec = ProxyCodec(FACTOR, MODE)
    clips = {}
    for name, make in synthetic.CLIPS.items():
        src = make()
        rec = codec.round_trip(src)
        o, r = metrics.temporal_metrics(src), metrics.temporal_metrics(rec)
        clips[name] = {
            "frames": int(src.shape[0]),
            "source": o.as_dict(),
            "reconstruction": r.as_dict(),
            # Paper §V: motion energy = mean optical-flow magnitude.
            "motion_retained": r.flow_mag / o.flow_mag,
            "delta_warp_abs": r.warp_error - o.warp_error,
            "delta_warp_rel": (r.warp_error - o.warp_error) / o.warp_error,
        }

    now = datetime.now(timezone.utc)
    record = {
        "kind": "proxy_validation",
        "label": codec.config()["label"],
        "created_utc": now.isoformat(timespec="seconds"),
        "config": {**codec.config(), "flow": flow.config()},
        "environment": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("numpy", "opencv-python-headless", "scikit-image")},
        },
        "clips": clips,
    }

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"proxy_validation_{now:%Y%m%dT%H%M%SZ}.json"
    # D-5: records are immutable; a re-run is a new file.
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    print(f"{record['label']}\n{'clip':13s} {'retained':>9s} {'dwarp abs':>10s} {'dwarp rel':>10s}")
    for name, c in clips.items():
        print(f"{name:13s} {100 * c['motion_retained']:8.1f}% {c['delta_warp_abs']:+10.5f} "
              f"{100 * c['delta_warp_rel']:+9.1f}%")
    print(f"-> {out.relative_to(RUNS.parent)}")


if __name__ == "__main__":
    main()
