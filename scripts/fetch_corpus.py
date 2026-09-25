"""Pilot corpus: the first frames of Xiph "derf" HD test sequences.

    python scripts/fetch_corpus.py

Fetches only the bytes needed (HTTP range requests on raw .y4m), converts
BT.709 limited-range YUV 4:2:0 to RGB, normalises, and writes
data/corpus/<name>.npy (gitignored: the corpus stays on the desktop, §5.6)
plus runs/corpus_<UTC>.json (sources, byte ranges, SHA-256, settings).

Normalisation (DECISIONS L-1):
  * 50/60 fps sources: every 2nd frame (dropping, never interpolating), so
    all clips sit at 24-30 fps, near LTX-Video's training rate.
  * resize to 768 px wide (INTER_AREA), centre-crop height to 416:
    both divisible by 32 as LTX requires.
  * T = 33 frames (8k+1).

Licensing: derf sequences are distributed for research and standardisation
(SVT, TUM, CDVL/NTIA origins). This is a pilot corpus for a review paper;
C-1 (licensed corpus for the full study) is still open.
"""
from __future__ import annotations

import hashlib
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = "https://media.xiph.org/video/derf/y4m/"
CLIPS = sys.argv[1:] or [
    "crowd_run_1080p50", "park_joy_420_720p50", "touchdown_pass_1080p", "speed_bag_1080p",
    "ducks_take_off_420_720p50", "riverbed_1080p25", "aspen_1080p", "red_kayak_1080p",
    "blue_sky_1080p25", "sunflower_1080p25", "old_town_cross_420_720p50", "station2_1080p25",
    "pedestrian_area_1080p25", "rush_hour_1080p25", "tractor_1080p25",
]
T, W, H = 33, 768, 416
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "corpus"


def get(url: str, start: int, end: int) -> bytes:
    # curl, not urllib: Python's CA bundle lacks the current Let's Encrypt
    # chain for media.xiph.org (the certificate itself is valid; curl verifies
    # it against the Windows store). Verification stays on.
    r = subprocess.run(["curl", "-sS", "--fail", "-r", f"{start}-{end}", url],
                       capture_output=True, timeout=900)
    if r.returncode:
        raise IOError(r.stderr.decode(errors="replace").strip())
    return r.stdout


def parse_header(url: str) -> tuple[dict, int]:
    head = get(url, 0, 1023)
    line = head[:head.index(b"\n")].decode()
    tok = {t[0]: t[1:] for t in line.split()[1:]}
    fps_n, fps_d = map(int, tok["F"].split(":"))
    info = {"header": line, "width": int(tok["W"]), "height": int(tok["H"]),
            "fps": fps_n / fps_d, "colorspace": tok.get("C", "420jpeg")}
    if info["colorspace"] not in ("420", "420jpeg", "420mpeg2", "420paldv", "422"):
        raise ValueError(f"{url}: expected 8-bit 4:2:0 or 4:2:2, got {info['colorspace']}")
    return info, len(line) + 1


def yuv_to_rgb_bt709(buf: bytes, w: int, h: int, ch: int) -> np.ndarray:
    """ch = chroma plane height: h // 2 for 4:2:0, h for 4:2:2."""
    # HD content is BT.709; OpenCV's I420 conversion assumes BT.601.
    n = (w // 2) * ch
    y = np.frombuffer(buf, np.uint8, w * h).reshape(h, w).astype(np.float32)
    u = np.frombuffer(buf, np.uint8, n, w * h).reshape(ch, w // 2).astype(np.float32)
    v = np.frombuffer(buf, np.uint8, n, w * h + n).reshape(ch, w // 2).astype(np.float32)
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0
    y = (y - 16.0) * (255.0 / 219.0)
    u, v = u * (255.0 / 224.0), v * (255.0 / 224.0)
    rgb = np.stack([y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u], -1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def fetch(name: str) -> dict:
    url = BASE + name + ".y4m"
    info, off = parse_header(url)
    w, h = info["width"], info["height"]
    step = 2 if info["fps"] > 40 else 1
    n_src = (T - 1) * step + 1
    ch = h if info["colorspace"] == "422" else h // 2
    frame_bytes = 6 + w * h + 2 * (w // 2) * ch  # "FRAME\n" + Y + U + V
    raw = get(url, off, off + n_src * frame_bytes - 1)
    if len(raw) != n_src * frame_bytes:
        raise IOError(f"{name}: short read {len(raw)} of {n_src * frame_bytes}")
    frames = []
    for i in range(0, n_src, step):
        chunk = raw[i * frame_bytes:(i + 1) * frame_bytes]
        if not chunk.startswith(b"FRAME\n"):
            raise ValueError(f"{name}: frame {i} lacks a bare FRAME marker")
        rgb = yuv_to_rgb_bt709(chunk[6:], w, h, ch)
        rs = cv2.resize(rgb, (W, round(h * W / w)), interpolation=cv2.INTER_AREA)
        top = (rs.shape[0] - H) // 2
        frames.append(rs[top:top + H])
    clip = np.stack(frames)
    assert clip.shape == (T, H, W, 3), clip.shape
    np.save(DATA / f"{name}.npy", clip)
    return {**info, "url": url, "byte_range": [off, off + n_src * frame_bytes - 1],
            "sha256_raw": hashlib.sha256(raw).hexdigest(), "frame_step": step,
            "effective_fps": info["fps"] / step, "source_frames": n_src,
            "sha256_clip": hashlib.sha256(clip.tobytes()).hexdigest()}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    entries = {}
    # The server gives ~0.8 MB/s per connection; 8 clips in parallel.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch, n): n for n in CLIPS}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                e = entries[name] = fut.result()
                print(f"ok   {name:28s} {e['width']}x{e['height']} @{e['fps']:.2f} step {e['frame_step']}", flush=True)
            except Exception as exc:  # one bad source must not sink the pilot
                print(f"FAIL {name}: {exc}", file=sys.stderr, flush=True)
    entries = {n: entries[n] for n in CLIPS if n in entries}
    if not entries:
        sys.exit("no clips fetched; not writing a corpus record")
    now = datetime.now(timezone.utc)
    rec = {"kind": "corpus", "created_utc": now.isoformat(timespec="seconds"),
           "normalisation": {"frames": T, "width": W, "height": H, "fps_rule": "every 2nd frame if > 40 fps",
                             "colour": "BT.709 limited range -> RGB", "resize": "INTER_AREA to width, centre crop"},
           "licence_note": "Xiph derf research/standardisation sequences; pilot only, C-1 open",
           "clips": entries}
    out = ROOT / "runs" / f"corpus_{now:%Y%m%dT%H%M%SZ}.json"
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"{len(entries)}/{len(CLIPS)} clips -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
