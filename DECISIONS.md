# Decisions log

One row per decision, written when made (protocol R-7). A decision is changed by
adding a new row that supersedes the old one, never by editing it.

## Standing decisions (backfilled from HANDOVER §4 on 2026-09-23; original dates not recorded)

| ID | Decision | Rationale | Revisit if |
|---|---|---|---|
| D-1 | VAE tiling OFF for floor measurement; if unavoidable, fixed + recorded + sensitivity check at two tile configs | Tiling is taxonomy mechanism 3 (chunk-boundary flicker, WF-VAE Fig. 3b); enabling it injects the artefact being measured | Untiled impossible at chosen resolution, after reducing resolution/frame count first |
| D-2 | FVMD primary, warping error secondary diagnostic | Finding F-2: warping error fell 40% while 96% of motion was destroyed | A reference-free metric penalising both excess and deficient motion is found |
| D-3 | Domain layer is GPU-free; `Codec` interface with `ProxyCodec` / `LTXCodec` as peers | Allowed CPU validation of the harness and F-2 at zero GPU cost | Never |
| D-4 | Synthetic clips are for instrument validation only | They are not evidence about any real codec | Never |
| D-5 | Runs are sealed and immutable; re-measurement creates a new run | Reproducibility of reported numbers | Never |
| D-6 | Core package imports and runs without torch; heavy imports are function-local | Laptop demonstration (Modes A/B); D-3 made testable | Never |

## Step decisions

| Date | ID | Decision | Rejected alternatives and why | Revisit if |
|---|---|---|---|---|
| 2026-09-23 | P-1 B1 | Flat layout: `<repo>/codecfloor/` | `src/` layout: its benefit (no accidental cwd import) is covered by the editable install plus the subprocess-isolated guard | A test passes from cwd but fails installed |
| 2026-09-23 | P-1 B2 | `.venv` on CPython 3.14.3 (pyenv) | 3.12.4: no reason to prefer yet. Anaconda 3.8: end-of-life. CUDA torch cp314 win wheels verified on the pytorch index (2.14.0+cu130) | Any gpu-tier package lacks a 3.14 wheel at G-1: rebuild the venv on 3.12.4 |
| 2026-09-23 | P-1 B3 | Portability guard blocks gpu **and** viz tiers | gpu-only: `pip install -e .` (core only) is a valid install per HANDOVER §5.3 | A viz module is added: allowlist it explicitly |
| 2026-09-23 | P-1 B4 | Guard = import hook recording every attempt, in a fresh `-I` subprocess | `sys.modules` check: misses `try/except ImportError` and passes trivially where torch is absent | — |
| 2026-09-23 | P-1 B5 | `opencv-python-headless` | `opencv-python`: pulls in GUI/Qt, and no GUI call is used | An interactive viewer becomes necessary |
| 2026-09-23 | P-1 B6 | Lower bounds in pyproject; exact versions recorded below and (later) in each run manifest | Exact pins: break the one-command laptop install | Manifest schema exists: versions move there |
| 2026-09-23 | P-1 B7 | Pin `opencv-python-headless<5` | Accepting 5.0.0: `cv2.remap` changed; warping error differs by up to 6.2% (slow_pan) vs 4.14.0 while flow/SSIM/PSNR are bit-identical. The original harness was presumably validated on 4.x (unrecorded) | Deliberate migration to 5.x, with a fingerprint diff recorded as its own step |

| 2026-09-25 | P-4 B1 | `temporal_decimate_restore` raises unless T = 8k+1 | Pad with last frame: recreates the frozen tail. Extrapolate: invents motion. Truncate output: breaks paired metrics. 8k+1 is LTX's own contract (LTX-Video README; diffusers `pipeline_ltx.py`) | A codec with a different temporal contract is added (F-5) |
| 2026-09-25 | P-4 B2 | Synthetic clips default to 33 frames | 25: discards data. 41: needless. 33 keeps frames 0–31 of each generator T-independent (tested) | C-2 fixes the real-corpus frame count |
| 2026-09-25 | P-4 B3 | Proxy numbers live in `runs/proxy_validation_<UTC>.json`, written with mode `x` (never overwritten), versions included; paper tables transcribed from it | Printing only: nothing citable. Values inside tests: a test is not a record | Run manifest schema exists: migrate the record to it |
| 2026-09-25 | P-4 B4 | Motion energy = mean Farneback flow magnitude over all pixels (paper §V definition) | Not revisited in this step; the same whole-frame dilution is Q-6 | Q-6 resolved in P-3 |
| 2026-09-25 | P-4 B5 | Synthetic pans sampled at sub-pixel offsets (Lanczos-4 `warpAffine`) instead of `int()`-truncated crops. *Surfaced during implementation (R-2), not in the original step.* | Keep stepping: slow_pan had 13/32 identical consecutive frames (flow CV 0.826), fine_texture 2/3px alternation (CV 0.181); the slow_pan baseline warp error was near zero by construction, producing the paper's +1367%. New CVs 0.015 / 0.002 | — |

| 2026-09-25 | P-2 B1 | `Codec` exposes `encode`, `decode`, `config`; `round_trip` is built on them | Round-trip only: the generation arm (D-1) decodes latents that were never encoded, and paper R1 needs decode isolable | G-1 shows decode needs more than the latent: such inputs become constructor config, fixed and recorded |
| 2026-09-25 | P-2 B2 | `Codec` is an ABC | `typing.Protocol`: structural, so an incomplete codec fails only when first called | — |
| 2026-09-25 | P-2 B3 | Base `round_trip` enforces the pixel contract (same shape, uint8) | Per-codec checks: a new codec could skip them, and misaligned frames still yield metrics | — |
| 2026-09-25 | P-2 B5 | Flat modules: `codec.py`, `proxy_codec.py`, LTX as a peer module; `temporal_decimate_restore` kept as a wrapper | `codecs/` subpackage (HANDOVER §5.4 example): not needed for two classes (R-8) | A third codec is added |
| 2026-09-25 | P-2 B6 | Proxy latent = kept keyframes; T recovered as factor·(k−1)+1 | Passing T alongside: a side channel the real codec does not have | — |
| 2026-09-25 | P-2 — | `ProxyCodec` rejects unknown `mode` (previously any value other than `"nearest"` ran linear silently) | — | — |

| 2026-09-25 | P-3 B1 | fast_action balls carry rigidly attached texture (user choice) | Flat discs: Farneback recovered 35% of true displacement (textured 80%); the fixture measured the estimator, not the codec (Q-11) | — |
| 2026-09-25 | P-3 B2 | Proxy decode rounds (`rint`) instead of truncating | Truncation: −0.3 level bias; moved slow-pan Δwarp 89.5% → 93.7% under Farneback | G-1 shows the LTX postprocess truncates |
| 2026-09-25 | P-3 B3 | Nearest mode ties → later keyframe (`floor(pos+½)`) | Python `round`: half-to-even, inconsistent (frame 4 → kf 0, frame 12 → kf 16) | — |
| 2026-09-25 | P-3 B4 | Stratify on flow **p99**; **equal thirds** (top third by p99 = fast; rest split at median texture energy) (user choices) | Mean flow: diluted by static background (Q-6). p95: misses fast motion under 5% of frame (failed the 2-ball probe). 0.70 texture quantile: unjustified (R-4), left stratum sizes to chance | C-3 bootstrap shows p99 cut unstable on the real corpus |
| 2026-09-25 | P-3 B5 | Flow estimator = **RAFT-large** (torchvision `Raft_Large_Weights.C_T_SKHT_V2`), chosen by a rule fixed before the run: lowest mean EPE over 6 ground-truth probes. Record `runs/flow_benchmark_20260925T091042Z.json` | Farneback: mean EPE 0.123 vs 0.073 px; better on pure pans (≈0.02 px) but smears local motion into background (0.56 vs 0.02 px). RAFT-small: 0.275. User instruction: accuracy over portability | Real-corpus evidence at F-2 that RAFT fails on real motion Farneback handles |
| 2026-09-25 | P-3 B5a | TF32 disabled inside flow computation (restored after); `CHUNK = 8` | TF32 on: flow depended on batch size (max 0.06 px). Off: chunk-invariant to 2e-4 px, CPU/GPU agree to 3e-4 px; cost 24 → 33 ms/pair | — |
| 2026-09-25 | P-3 B5b | **D-6 narrowed:** the core package still *imports* without torch (guard unchanged), but computing any flow-based metric requires torch + torchvision (CPU build suffices) | Keeping Farneback for portability: user ranked accuracy above laptop portability | — |
| 2026-09-25 | P-3 B6 | One flow module; `BORDER = 8` justified as RAFT's 1/8-resolution feature stride | Separate literals per module (Farneback params were duplicated in metrics and motion) | Estimator changes |
| 2026-09-25 | P-3 B7 | Labelled, not changed: fixture constants arbitrary by design (D-4); `warp_error_p95` descriptive; `write_video` fps is container metadata | — | — |

| 2026-09-25 | G-1 | **LTX-Video 0.9.5 VAE** (`Lightricks/LTX-Video-0.9.5`, `vae/`), decode timestep **0**, posterior **mode**, tiling and frame-wise processing **off**, TF32 off | 0.9.0: no timestep-conditioned decoder, so not the decoder the paper describes. 0.9.1: `decoder_inject_noise=[T,T,T,F]`, stochastic decode (Q-1). Timestep > 0 or noise mixing: invokes the decoder's denoising role, measuring codec plus partial generation (C2 in the G-1 example). Posterior sample: adds encoder noise | A reviewer requires the pipeline's own decode setting for a specific checkpoint; report as a sensitivity row |
| 2026-09-25 | L-1 | Pilot corpus = 15 Xiph "derf" HD sequences, first 33 frames at 24–30 fps (50 fps sources: every 2nd frame, dropped not interpolated), BT.709 YUV→RGB, resized to width 768 and centre-cropped to 768×416 (÷32) | Waiting for C-1 (licensed corpus): the user asked for pilot numbers now. OpenCV's I420 conversion: BT.601, wrong for HD | C-1 delivers a licensed corpus: re-run, and these numbers become a pilot footnote |
| 2026-09-25 | L-2 | Fetch through `curl` (Windows cert store) | Python urllib failed TLS: its CA bundle lacks the current Let's Encrypt chain; the server certificate is valid (Aug–Nov 2026). Disabling verification: rejected | — |

**P-3 result (record `runs/proxy_validation_20260925T091403Z.json`, RAFT-large, OpenCV 4.14.0):** motion retained / Δwarp abs / Δwarp rel: slow_pan 97.7% / +0.0014 / +98.1%, fast_action 1.6% / −0.0015 / −47.2%, fine_texture 0.8% / +0.0064 / +37.9%. Supersedes the P-4 values below. RAFT measures the sources' motion correctly (slow pan 0.604 vs true 0.600 px/frame).

**P-4 result (record `runs/proxy_validation_20260925T083829Z.json`, OpenCV 4.14.0):** motion retained / Δwarp rel.: slow_pan 91.8% / +93.7%, fast_action 5.2% / −22.0%, fine_texture 5.1% / +42.8%. Tail fix alone: 91.4% / +1791%, 5.2% / −22.0%, 5.1% / +29.4%. Superseded paper values: 71.5% / +1367%, 3.9% / −40.4%, 3.9% / +0.2%.

## Environment of record

**P-1, 2026-09-23 (desktop):** CPython 3.14.3, numpy 2.5.3, opencv-python-headless 4.14.0,
scikit-image 0.26.0, scipy 1.18.1, matplotlib 3.11.2, pandas 3.0.6, pytest 9.1.1.
GPU RTX 3060 12 GB, driver 591.59. gpu tier not yet installed.
