"""Regenerate every figure in docs/figures/ (referenced by docs/PROJECT_GUIDE.md).

    python scripts/make_figures.py

Two kinds of figure, kept distinct:
  * RESULT figures (fig3, fig4) are drawn only from runs/*.json records. The
    records are the source of truth; a figure is a view of them (HANDOVER §5.6).
  * DIAGNOSTIC figures (fig1, fig2, fig5, fig6) illustrate mechanisms and are
    recomputed deterministically from the fixtures. Their numbers are written
    to docs/figures/figure_data.json so the guide's tables can quote them.

Everything here is the proxy codec on synthetic clips, NOT the LTX-Video VAE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from codecfloor import flow, synthetic  # noqa: E402
from codecfloor.proxy_codec import ProxyCodec  # noqa: E402
from codecfloor.video_io import to_gray  # noqa: E402

OUT = ROOT / "docs" / "figures"
RUNS = ROOT / "runs"
P4_RECORD = RUNS / "proxy_validation_20260925T083829Z.json"
P3_RECORD = RUNS / "proxy_validation_20260925T091403Z.json"
BENCH_RECORD = RUNS / "flow_benchmark_20260925T091042Z.json"

# Reference palette (dataviz skill): first three categorical slots validate
# for all pairs on the light surface; aqua is below 3:1, so every chart ships
# with a legend and a table in the guide.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fce4d8", "#f5a57f", "#eb6834", "#b8461b", "#7a2a0c"])
CLIPS = ["slow_pan", "fast_action", "fine_texture"]
NICE = {"slow_pan": "Slow pan", "fast_action": "Fast action", "fine_texture": "Fine texture"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})
DOT = dict(markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)  # surface ring


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def mags(video):
    return np.linalg.norm(flow.video_flow(video), axis=-1)


# ------------------------------------------------------------ fig1 filmstrip
def fig1_filmstrip():
    frames = [0, 2, 4, 6, 8]
    codec = ProxyCodec()
    fig, axes = plt.subplots(6, len(frames), figsize=(7.2, 9.0))
    for r, name in enumerate(CLIPS):
        src = synthetic.CLIPS[name]()
        rec = codec.round_trip(src)
        for k, (label, v) in enumerate((("source", src), ("proxy", rec))):
            for c, t in enumerate(frames):
                ax = axes[2 * r + k, c]
                ax.imshow(v[t]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                if r == 0 and k == 0:
                    ax.set_title(f"t = {t}" + (" (keyframe)" if t % 8 == 0 else ""), fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{NICE[name]}\n{label}", fontsize=9, color=INK2)
    fig.suptitle("Fixtures and the proxy round trip, frames 0-8 (keyframes 0 and 8)",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.4, w_pad=0.2)
    save(fig, "fig1_fixture_filmstrip.png")


# ------------------------------------------------------- fig2 per-frame motion
def legacy_round_trip(video, factor=8):
    """Pre-P-4 proxy: no 8k+1 check, truncating cast. Illustration only."""
    t = video.shape[0]
    kept = video[np.arange(0, t, factor)].astype(np.float32)
    out = np.empty_like(video, dtype=np.float32)
    for i in range(t):
        pos = i / factor
        lo = int(np.floor(pos)); hi = min(lo + 1, len(kept) - 1); w = pos - lo
        out[i] = (1 - w) * kept[lo] + w * kept[hi]
    return np.clip(out, 0, 255).astype(np.uint8)


def fig2_per_frame_motion(data):
    codec = ProxyCodec()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
    data["fig2"] = {}
    for ax, name in zip(axes, CLIPS):
        src = synthetic.CLIPS[name]()
        s = mags(src).mean(axis=(1, 2))
        r = mags(codec.round_trip(src)).mean(axis=(1, 2))
        legacy = mags(legacy_round_trip(src[:32])).mean(axis=(1, 2))
        data["fig2"][name] = {"source": s.tolist(), "proxy": r.tolist(), "legacy_32": legacy.tolist()}
        for kf in (8, 16, 24):
            ax.axvline(kf - 0.5, color=AXIS, lw=0.8, zorder=0)
        ax.plot(np.arange(len(s)), s, color=BLUE, label="source")
        ax.plot(np.arange(len(r)), r, color=ORANGE, label="proxy, 33 frames (current)")
        ax.plot(np.arange(len(legacy)), legacy, color=AQUA, lw=1.5, label="proxy, 32 frames (pre-P-4)")
        ax.axvspan(23.5, 31, color=AQUA, alpha=0.10, lw=0)
        ax.set_title(NICE[name]); ax.set_xlabel("frame pair t -> t+1")
        ax.set_ylim(bottom=0); ax.set_xlim(-0.5, 31.5)
    axes[0].set_ylabel("mean |flow| (px/frame, RAFT)")
    axes[0].annotate("pre-P-4: frames 25-31 frozen\n(nothing to interpolate towards)",
                     xy=(27.5, 0.015), xytext=(3, 0.12), fontsize=8, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Per-frame motion: the proxy restores motion only at keyframes (hairlines)",
                 fontsize=11, fontweight="semibold", y=1.15)
    save(fig, "fig2_per_frame_motion.png")


# --------------------------------------------------- fig3 Table III evolution
def fig3_table3_evolution():
    p4, p3 = (json.load(open(p))["clips"] for p in (P4_RECORD, P3_RECORD))
    # Draft values are transcribed from the paper as handed over (no record
    # exists; they were reproduced exactly under OpenCV 4.14 during P-1/P-4).
    draft = {"slow_pan": (71.5, 1367.0), "fast_action": (3.9, -40.4), "fine_texture": (3.9, 0.2)}
    versions = ["draft", "P-4", "P-3\n(current)"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for c, name in enumerate(CLIPS):
        ret = [draft[name][0], 100 * p4[name]["motion_retained"], 100 * p3[name]["motion_retained"]]
        rel = [draft[name][1], 100 * p4[name]["delta_warp_rel"], 100 * p3[name]["delta_warp_rel"]]
        for r, (vals, unit) in enumerate(((ret, "motion retained (%)"), (rel, "Δ warping error, relative (%)"))):
            ax = axes[r, c]
            ax.plot(range(3), vals, color=BLUE, marker="o", **DOT)
            for x, v in enumerate(vals):
                # Above a local peak, below otherwise, so labels never sit on the line.
                up = v >= np.mean([vals[j] for j in (x - 1, x + 1) if 0 <= j < 3])
                ax.annotate(f"{v:+.1f}%" if r else f"{v:.1f}%", (x, v), textcoords="offset points",
                            xytext=(0, 9 if up else -15), ha="center", fontsize=8, color=INK)
            ax.set_xticks(range(3)); ax.set_xticklabels(versions if r else [])
            ax.set_xlim(-0.4, 2.4)
            lo, hi = min(vals + [0]), max(vals + [0])
            pad = 0.25 * (hi - lo or 1)
            ax.set_ylim(lo - pad, hi + pad)
            ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
            if c == 0:
                ax.set_ylabel(unit)
            if r == 0:
                ax.set_title(NICE[name])
    fig.suptitle("Paper Table III across fixes (proxy codec, synthetic clips; each panel its own scale)",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, "fig3_table3_evolution.png")


# ------------------------------------------------------ fig4 flow benchmark
def fig4_flow_benchmark():
    res = json.load(open(BENCH_RECORD))["results"]
    est = [("farneback", "Farneback", BLUE), ("raft_small_cuda", "RAFT-small", ORANGE),
           ("raft_large_cuda", "RAFT-large (chosen)", AQUA)]
    rows = [("pan_0.6px", "pan 0.6 px/frame", "epe"), ("pan_2.5px", "pan 2.5", "epe"),
            ("pan_5px", "pan 5", "epe"), ("pan_10px", "pan 10", "epe"), ("pan_20px", "pan 20", "epe"),
            ("balls", "balls: all valid px", "epe"), ("balls", "balls: interior", "epe_ball"),
            ("balls", "balls: background", "epe_background")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for y, (probe, label, key) in enumerate(rows):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        for dy, (k, name, col) in zip((-0.18, 0, 0.18), est):
            ax.plot(res[k]["probes"][probe][key], y + dy, "o", color=col, label=name if y == 0 else None,
                    zorder=3, **DOT)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("endpoint error, px (log scale; lower is better)")
    ax.grid(axis="y", visible=False)
    ax.axhline(4.5, color=AXIS, lw=0.8)
    means = "   ".join(f"{name}: {res[k]['mean_epe']:.3f}" for k, name, _ in est)
    ax.set_title(f"Flow estimators vs exact ground truth. Mean EPE over 6 probes:  {means}", fontsize=9)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig4_flow_benchmark.png")


# --------------------------------------------------------- fig5 flow fields
def fig5_flow_fields(data):
    import flow_benchmark as fb
    clip, gt, valid, _ = fb.ball_probe()
    i = 2
    g = (to_gray(clip) * 255).astype(np.uint8)
    fn = flow.farneback(g[i], g[i + 1])
    rf = flow.video_flow(clip[i:i + 2])[0]
    gm = np.linalg.norm(gt[i], axis=-1)
    e_fn = np.linalg.norm(fn - gt[i], axis=-1)
    e_rf = np.linalg.norm(rf - gt[i], axis=-1)
    v = valid[i]
    data["fig5"] = {"pair": i, "epe_farneback": float(e_fn[v].mean()), "epe_raft_large": float(e_rf[v].mean())}
    vmax = float(np.percentile(gm[v], 99.5))
    emax = 5.0
    panels = [("frame t", clip[i], None), ("ground-truth |flow|", gm, BLUES),
              ("Farneback |flow|", np.linalg.norm(fn, axis=-1), BLUES), ("RAFT-large |flow|", np.linalg.norm(rf, axis=-1), BLUES),
              (f"Farneback error (mean {e_fn[v].mean():.2f} px)", e_fn, ORANGES),
              (f"RAFT-large error (mean {e_rf[v].mean():.2f} px)", e_rf, ORANGES)]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.9))
    ims = {}
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(img)
        else:
            m = np.ma.masked_where(~v, img)
            cm = cmap.copy(); cm.set_bad("#d8d7d0")
            ims[cmap.name] = ax.imshow(m, cmap=cm, vmin=0, vmax=vmax if cmap is BLUES else emax)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(ims["blues"], ax=axes[0, :], shrink=0.8, label="|flow|, px/frame", pad=0.02)
    fig.colorbar(ims["oranges"], ax=axes[1, :], shrink=0.8, label="endpoint error, px (clipped at 5)", pad=0.02)
    fig.suptitle(f"fast_action pair {i}->{i + 1}: Farneback smears ball motion into the static background\n"
                 "(gray = excluded: occluded or border)", fontsize=10.5, fontweight="semibold")
    save(fig, "fig5_flow_fields.png")


# ------------------------------------------ fig6 stratification vs moving area
def render_balls(n: int) -> np.ndarray:
    """fast_action with only the first n balls, rendered as synthetic.fast_action does."""
    import cv2
    h = w = 256
    bg = cv2.GaussianBlur(synthetic._texture_bg(h, w, seed=2), (31, 31), 0)
    skins = [synthetic._texture_bg(64, 64, seed=10 + k) for k in range(3)]
    yy, xx = np.mgrid[:h, :w]
    frames = []
    for balls in synthetic.ball_tracks():
        f = bg.copy()
        for k, (cx, cy, r) in enumerate(balls[:n]):
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            tex = skins[k][(yy[m] - cy) % 64, (xx[m] - cx) % 64]
            f[m] = (0.5 * tex + 0.5 * np.array(synthetic._BALL_COLOURS[k])).astype(np.uint8)
        frames.append(f)
    return np.stack(frames)


def fig6_stratification(data):
    assert np.array_equal(render_balls(3), synthetic.fast_action()), "local renderer drifted from fixture"
    ref = mags(synthetic.fine_texture()).ravel()
    ref_stats = {"mean": ref.mean(), "p95": np.percentile(ref, 95), "p99": np.percentile(ref, 99)}
    radii = [r for _, _, r in synthetic.ball_tracks()[0]]
    area, stats = [], {"mean": [], "p95": [], "p99": []}
    for n in (1, 2, 3):
        m = mags(render_balls(n)).ravel()
        area.append(100 * sum(np.pi * r * r for r in radii[:n]) / 256 ** 2)
        stats["mean"].append(m.mean()); stats["p95"].append(np.percentile(m, 95)); stats["p99"].append(np.percentile(m, 99))
    ratios = {k: [float(x / ref_stats[k]) for x in v] for k, v in stats.items()}
    data["fig6"] = {"moving_area_pct": area, "fine_texture": {k: float(v) for k, v in ref_stats.items()},
                    "fast_action_n_balls": {k: [float(x) for x in v] for k, v in stats.items()}, "ratio": ratios}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(1, color=INK2, lw=1)
    ax.text(area[0] - 0.1, 1.12, "above: fast clip correctly ranked above fine_texture", fontsize=8, color=INK2)
    for (k, col, lab) in (("mean", BLUE, "mean flow (old rule)"), ("p95", ORANGE, "p95"), ("p99", AQUA, "p99 (chosen)")):
        ax.plot(area, ratios[k], color=col, marker="o", label=lab, **DOT)
    ax.set_yscale("log"); ax.set_xlabel("moving area of the fast clip (% of frame; 1, 2, 3 balls)")
    ax.set_ylabel("statistic(fast clip) / statistic(fine_texture)")
    ax.set_xticks(area); ax.set_xticklabels([f"{a:.1f}%" for a in area]); ax.set_xlim(area[0] - 0.5, area[-1] + 0.5)
    ax.legend(loc="lower right")
    ax.set_title("Which statistic still sees small fast objects? (RAFT flow)")
    save(fig, "fig6_stratification_statistic.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"label": "proxy codec (temporal decimation x8), NOT the LTX-Video VAE", "flow": flow.config()}
    fig1_filmstrip()
    fig2_per_frame_motion(data)
    fig3_table3_evolution()
    fig4_flow_benchmark()
    fig5_flow_fields(data)
    fig6_stratification(data)
    (OUT / "figure_data.json").write_text(json.dumps(data, indent=2))
    print("wrote docs/figures/figure_data.json")


if __name__ == "__main__":
    main()
