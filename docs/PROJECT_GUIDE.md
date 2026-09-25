# Codec-Floor Attribution: Project Guide

This guide explains what the project measures, how the code is organised, what has been built so far, the problems found along the way and how each was solved, and what remains before the paper's missing values can be measured.

It sits alongside three other documents:

| Document | Role |
|---|---|
| `HANDOVER_Implementation_Protocol.md` | The working protocol (step format, rules R-1 to R-8), the roadmap, standing decisions D-1 to D-6, and the risk register |
| `DECISIONS.md` | Every design decision, with the date, the alternatives rejected, and when to revisit it |
| `runs/*.json` | Immutable measurement records; every number in the paper is transcribed from one of these |

State as of 2026-09-25: prerequisite steps P-1 to P-4 complete, 44 tests passing, next step G-1.

---

## 1. What the project is about

Latent video generators (LTX-Video, CogVideoX, …) are two-stage pipelines:

```
pixels ──► VAE encoder ──► latent ──► diffusion transformer (denoiser) ──► latent ──► VAE decoder ──► pixels
            (codec)                                                                   (codec)
```

Temporal defects such as flicker, jitter and lost motion are measured on the output pixels and usually blamed entirely on the denoiser. But the video passes through a lossy codec on the way in and on the way out. LTX-Video compresses about 1:192 (32×32 spatially, 8× temporally).

**The research question:** how much of the temporal error in generated video comes from the codec alone, and therefore how much improvement was ever available to methods that only change the denoiser?

**The codec floor** is the temporal error measured on a pure encode→decode round trip, with no generation involved. It is a lower bound on the error attributable to the codec. The project measures the floor, measures end-to-end generation error, and decomposes one against the other, stratified by motion content (slow pan / fast action / fine texture). The motion-dependent mechanisms in the paper's taxonomy predict that the codec's share differs between these strata.

The paper (`Review_Paper_IEEE.docx`) is a mechanism-level review with these measurements embedded. Its remaining blanks are the LTX-Video measurements (§8 of this guide).

---

## 2. How the work is conducted

The handover document defines a strict protocol. In practice:

- **Step format.** Every step states *why it exists*, *what it decides* (options, choice, rejected alternatives, revisit condition), *what could silently produce a wrong number*, and its *assumptions*, each tagged `[VERIFIED]`, `[INFERRED]` or `[ASSUMED]`. All of this comes before any code is written.
- **Acceptance checks must be able to fail.** Each new check was run against a deliberately broken version of the code (a *mutation test*) to prove it catches the failure it targets. §7 lists these.
- **No fabricated values (R-4).** Every constant in the code is justified by a measurement or a source, or explicitly labelled as arbitrary.
- **Records are immutable (D-5).** Measurement scripts write `runs/<kind>_<UTC timestamp>.json` in create-only mode. A re-run makes a new file, so old results are never overwritten.
- **Selection criteria are fixed before results are seen.** The flow-estimator comparison (§5.4) wrote its decision rule into the script before any result existed, so the choice could not drift toward whichever estimator flattered the hypothesis.

---

## 3. Repository map

```
latent-Codec-Floor/
├── codecfloor/                 the package (domain layer)
│   ├── video_io.py             read/write/normalise video → (T,H,W,3) uint8 RGB
│   ├── codec.py                Codec interface: encode / decode / config / round_trip
│   ├── proxy_codec.py          ProxyCodec: transparent stand-in for mechanisms M1+M4
│   ├── flow.py                 optical flow (RAFT-large): single source for all flow numbers
│   ├── metrics.py              warping error, motion energy, raw frame diff, PSNR
│   ├── motion.py               motion profiling and stratification
│   └── synthetic.py            deterministic test clips with known ground-truth motion
├── scripts/
│   ├── proxy_validation.py     produces paper Table III → runs/proxy_validation_*.json
│   └── flow_benchmark.py       estimator selection against ground truth → runs/flow_benchmark_*.json
├── tests/                      44 tests (§7)
├── runs/                       immutable measurement records
├── DECISIONS.md                decision log
├── HANDOVER_Implementation_Protocol.md
└── pyproject.toml              dependency tiers
```

### Module by module

**`video_io.py`** is the pixel contract. Everything downstream assumes `(T, H, W, 3)` uint8 RGB. `normalise()` fixes resolution and frame count, cropping in time rather than resampling so that the harness introduces no temporal interpolation of its own. It is unchanged from the handed-over code.

**`codec.py`** holds the `Codec` abstract base class, the one seam between the measurement pipeline and any model:

| Method | Purpose |
|---|---|
| `encode(video) → latent` | Pixels to latent; the latent type is implementation-defined |
| `decode(latent) → video` | Separate from encode, because the generation arm decodes latents that were never encoded, and the paper's requirement R1 needs decode isolable |
| `config() → dict` | Every setting that affects output, for the run manifest |
| `round_trip(video)` | Built on the two above; **enforces the pixel contract**: rejects any codec whose output changes the frame count, the size or the dtype |

`ProxyCodec` implements it now. `LTXCodec` will be a peer (step F-1), so the validated pipeline runs unchanged on the real model.

**`proxy_codec.py`: `ProxyCodec(factor=8, mode="linear"|"nearest")`.**
- **Encode** keeps every 8th frame, like a strided convolution.
- **Decode** restores the frames between keyframes by linear blending (or by repeating the nearest keyframe), then rounds to uint8.
- **Frame contract:** it requires `T = 8k+1` frames, LTX's own contract. Decode therefore recovers `T` from the keyframe count alone.
- **Honesty requirement:** the module states that it is *not* the LTX VAE, and every output carries the label "proxy codec (temporal decimation x8), NOT the LTX-Video VAE".

**`flow.py`** is the single source of optical flow.
- `video_flow(video)` returns `(T-1, H, W, 2)` flow using RAFT-large (torchvision), chosen by benchmark (§5.4).
- Torch is imported inside the function, so the package still imports without it.
- TF32 is disabled during the call and restored afterwards. Frame pairs are processed in chunks of 8, and results are identical for any chunk size.
- `BORDER = 8` is RAFT's feature stride: the outermost 8-pixel cell has no outward neighbours, so its flow is extrapolated and is excluded.
- `farneback()` is kept only as the benchmark baseline.

**`metrics.py`** computes reference-free temporal metrics, which the decomposition needs because generated video has no ground truth.

| Metric | What it measures | Note |
|---|---|---|
| `warp_error` | Predict each frame from the next using flow, then take the mean residual | Secondary only (D-2): it *improves* when motion is removed (finding F-2) |
| `flow_mag` | Mean flow magnitude, the paper's "motion energy" | — |
| `raw_frame_diff` | Frame difference without motion compensation | For context |
| `recon_psnr` | Paired spatial fidelity | Floor path only; never enters the decomposition |

`temporal_metrics()` computes flow once and reuses it for every metric.

**`motion.py`** profiles and stratifies clips.
- `profile()` measures DSSIM complexity, flow mean / p95 / p99, and high-frequency (Laplacian) energy.
- `strata_cutpoints()` implements the stratification rule:
  - **fast_action** is the top third by **p99 flow**;
  - the remaining clips split at the median texture energy into **fine_texture** and **slow_pan**;
  - the result is equal thirds.
- `assign_strata()` accepts frozen cut-points, so new clips can be stratified against a recorded corpus instead of recomputing (step C-3).

**`synthetic.py`** holds the test fixtures (instrument validation only, D-4). All clips are 33 frames at 256×256.
- `slow_pan`: a blurred scene panning at 0.6 px/frame, at sub-pixel precision.
- `fast_action`: three textured balls bouncing at about 10.6 px/frame. `ball_tracks()` returns their exact positions, which the renderer itself uses, so the flow ground truth cannot drift from the pixels.
- `fine_texture`: a high-frequency texture panning at (2.2, 1.3) px/frame, plus per-frame noise.

---

## 4. Environment

| | |
|---|---|
| Python | 3.14.3 in `.venv` (pyenv) |
| Core | numpy 2.5.3, opencv-python-headless **4.14** (pinned `<5`, see §5.1), scikit-image 0.26, scipy 1.18 |
| GPU tier | torch 2.14.0+cu130, torchvision 0.29.0+cu130 |
| Hardware | RTX 3060 12 GB, driver 591.59 |

```powershell
# desktop (CUDA)
python -m venv .venv
.venv\Scripts\pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -e ".[viz,dev]"

# laptop (CPU): plain PyPI torch is enough; CPU and GPU flow agree to 3e-4 px
pip install -e ".[viz,dev]" torch torchvision
```

**Portability (D-6, narrowed in P-3).** The package must *import* without torch. `tests/test_portable.py` enforces this by blocking torch and every other heavy import in a subprocess. Computing any flow-based metric does need torch and torchvision, because the flow estimator is RAFT. This was a deliberate trade: accuracy ranked above laptop portability.

---

## 5. What was done, step by step

The four prerequisite steps (P-1 to P-4) repair and harden the handed-over CPU harness before any GPU measurement. They were run in the order P-1, P-4, P-2, P-3, because P-4 fixed a value already printed in the paper.

### 5.1 P-1: package skeleton, dependency tiers, portability guard

**Starting state.** The handover described a validated `codecfloor/` package with scripts and tests. The folder actually held five loose `.py` files: no package layout, tests, scripts, `Codec` class, `pyproject.toml` or `DECISIONS.md`. `import codecfloor` failed.

**Done:**
- Moved the modules into `codecfloor/`. SHA-256 checksums confirmed every file was byte-identical after the move.
- Wrote `pyproject.toml` with the core / viz / gpu / dev tiers.
- Created the venv.
- Wrote the portability guard.
- Started `DECISIONS.md`.

**Challenges and solutions:**

| Challenge | Solution |
|---|---|
| A portability test that just checks `"torch" in sys.modules` passes trivially on a laptop without torch, and misses `try: import torch / except ImportError` | An import hook in a fresh, isolated subprocess records every *attempt* to import a blocked package. Modules are found from the filesystem, because `pkgutil` silently skips broken ones. Self-tests plant violations in a throwaway package; planting one in the real package made the test fail and name the file |
| pip installed **OpenCV 5.0.0**, a new major version | Compared every metric under 4.14 and 5.0. Flow, SSIM and PSNR were bit-identical, but **warping error differed by up to 6.2%** (`cv2.remap` changed). Pinned `<5`, and the paper's original Table III then reproduced *exactly* under 4.14 |

### 5.2 P-4: frozen tail in the proxy; Table III re-derived

**The problem.** With 32-frame clips and 8× decimation, the kept keyframes are 0, 8, 16 and 24. Frames 25–31 lie past the last keyframe, had nothing to interpolate towards, and were **held frozen**. Seven of 32 frames were static: an artefact of the proxy, not of decimation plus interpolation (M1/M4). It inflated the headline F-2 number.

**Done:**
- The proxy now requires `T = 8k+1`. This was checked against the LTX-Video README ("divisible by 8 + 1") and the diffusers formula `latent_frames = (T-1)//8 + 1`.
- Clips were lengthened to 33 frames; frames 0–31 are unchanged, which a test checks.
- `scripts/proxy_validation.py` now produces Table III as an immutable record.

**A second problem, surfaced by a new test.** The test asserted "no two consecutive source frames are identical" and failed on the *correct* code. The slow pan's position was `int(i * 0.6)`, so it **stuttered**: 13 of 32 frame pairs were identical, with 1-pixel jumps in between. Its "near-zero baseline" warping error, and therefore the paper's +1367%, was produced by the fixture. The pans now move at sub-pixel precision (Lanczos-4 warp). Speed variation fell from CV 0.826 to 0.015, and a test bounds it.

### 5.3 P-2: the `Codec` interface

**Done:** the `Codec` abstract base class and `ProxyCodec` behind it. `temporal_decimate_restore()` remains as a thin wrapper for existing callers.

**Safety net:** SHA-256 hashes of the proxy's output were taken *before* the refactor, and the refactor moved no byte. The run record was regenerated and matched in every value. Changing the interpolation weight by 0.1% made three golden tests fail. Planted broken codecs (dropping a frame, cropping, returning floats) are all rejected by `round_trip`.

**Found while reading the code:** any `mode` other than `"nearest"` silently ran linear, so a typo like `"Linear"` would pass unnoticed. It is now rejected.

### 5.4 P-3: constant audit, stratifier fix, flow-estimator selection

Evidence was measured *before* the step was written.

**1. The flow estimator could not see the fast motion.** With the true ball speed known exactly, Farneback recovered only **35%** of the balls' displacement. Flat-coloured discs have no interior texture to track (the aperture problem). With texture on the balls it recovered 80%. Everything flow-based was affected: motion energy, warping error and stratification.

**2. The stratifier was wrong.** It binned clips by *mean* flow, which the static background dilutes. On the fixtures it labelled fast_action as slow_pan and fine_texture as fast_action.

**3. Truncation bias.** `astype(uint8)` truncates instead of rounding, and that alone moved the slow-pan Table III value by 4 points.

**Decisions (the user made B1, B4 and B5):**

| Item | Resolution |
|---|---|
| Flat balls | Balls now carry rigidly attached texture. A test proves the trajectories are unchanged, and it fails when a ball is shifted by 3 px |
| Stratification | p99 flow detects fast motion covering at least 1% of the frame; p95 failed a 2-ball probe. **Equal thirds** replaced the unjustified 0.70 quantile, giving each stratum the same sample size for FVMD |
| Flow estimator | **Benchmark against exact ground truth** (below) |
| Rounding | `np.rint`, as real decoders do |
| Nearest-mode ties | Python's `round()` rounds half to even, which sent frame 4 back to keyframe 0 but frame 12 forward to 16. Ties now consistently go to the later keyframe |
| Duplicated flow settings | One flow module, one set of settings |
| Metrics docstring | It called warping error "primary", contradicting D-2. Corrected |

**The flow benchmark** (`scripts/flow_benchmark.py`, record `runs/flow_benchmark_20260925T091042Z.json`) used six probes with exact ground truth:
- textured pans at 0.6, 2.5, 5, 10 and 20 px/frame;
- the bouncing balls, with occluded pixels excluded.

The rule was fixed beforehand: lowest mean endpoint error (EPE) wins, with ties within 10% going to the faster estimator.

| Estimator | Mean EPE (px) | Balls: interior / background EPE | ms/pair |
|---|---|---|---|
| Farneback | 0.123 | 1.31 / 0.56 | 9 |
| RAFT-small | 0.275 | 2.18 / 0.14 | 13 |
| **RAFT-large** | **0.073** | **0.80 / 0.02** | 24 (33 with TF32 off) |

RAFT-large won. Farneback is slightly better on pure global pans but smears local motion into the static background, and local motion is exactly the paper's concern.

**The TF32 problem.** On the Ampere GPU, PyTorch lets cuDNN use TF32 by default, which made RAFT's output depend on batch size (up to 0.06 px). With TF32 disabled, the output is chunk-invariant to 2e-4 px, and CPU and GPU agree to 3e-4 px.

**A bug of my own, caught by a hash check.** Moving the ball physics into `ball_tracks()` changed the output. The cause was in the textured version written minutes earlier: the bounce loop `for k in (0, 1)` reused the ball index `k`, so every ball got the same texture. The flat version was confirmed byte-identical to the original fixture; the textured one now gives each ball its own texture. No recorded result ever used the buggy version.

---

## 6. Findings and current numbers

### Table III (proxy codec on synthetic clips, NOT LTX)

Record: `runs/proxy_validation_20260925T091403Z.json`.

| Clip | Motion retained | Δ warp (abs.) | Δ warp (rel.) |
|---|---|---|---|
| Slow pan | 97.7% | +0.0014 | +98.1% |
| Fast action | 1.6% | −0.0015 | −47.2% |
| Fine texture | 0.8% | +0.0064 | +37.9% |

**How it evolved:**

| Version | Fast action (retained / Δwarp rel.) | Cause of change |
|---|---|---|
| Paper as handed over | 3.9% / −40.4% | Frozen tail, stuttering pans, flat balls, Farneback, truncation |
| After P-4 | 5.2% / −22.0% | Frozen tail and pan stutter removed |
| After P-3 (current) | **1.6% / −47.2%** | Textured balls, RAFT flow, rounding |

**What the findings mean now:**
- **F-1: motion loss is stratified.** It still holds, more sharply: slow pan keeps 97.7% of its motion while fast action and fine texture keep about 1%.
- **F-2: warping error is unsafe as a primary metric.** It still holds, and more strongly. On fast action, warping error *falls* 47% while 98% of the motion is destroyed. It holds on fast action only: on fine texture, warping error correctly rises.
- **Validity check:** RAFT measures the fixtures' motion correctly. Slow pan reads 0.604 px/frame against a true 0.600; fast action reads 0.737 against an expected ≈0.75.

---

## 7. Verification: what each test protects

44 tests, about 14 s on the GPU. Where the last column names a break, that break was actually applied and the tests failed:

| File | Protects | Proven by breaking |
|---|---|---|
| `test_portable.py` | Core imports without torch (D-6); flags even `try/except` imports | Planted `import torch` → fails and names the file |
| `test_proxy_codec.py` | 8k+1 contract, no frozen frames, fixtures independent of T, uniform pan speed, textured balls on identical trajectories, consistent nearest-mode ties | Off-by-one on the last keyframe → 6 failures; old stepping (CV 0.826) → fails; balls shifted 3 px → fails |
| `test_codec.py` | Byte-exact proxy output (golden hashes); the round-trip contract; incomplete codecs cannot be instantiated | Interpolation weight ×0.999 → 3 failures |
| `test_motion.py` | Fixtures land in their own strata; equal thirds; frozen cut-points reused | — (the old rule fails the first test by construction) |
| `test_flow.py` | RAFT accuracy on a known pan; chunk invariance; TF32 flags restored; refuses sizes it would have to pad | TF32 re-enabled → fails; normalisation removed → fails |

Run everything:

```powershell
.venv\Scripts\python.exe -m pytest -q                  # 44 passed
.venv\Scripts\python.exe scripts\proxy_validation.py   # reproduces Table III
.venv\Scripts\python.exe scripts\flow_benchmark.py     # RAFT-large lowest mean EPE (~0.073)
```

---

## 8. The paper: what changed, what is pending

Every edit to `Review_Paper_IEEE.docx` is a **tracked change** authored as "Claude", with comments explaining each one, so it can be accepted or rejected in Word.

**Done:**
- Table III and the §V sentence carry the current values.
- Seven references now have authors and titles. [7] had the wrong title, and "CREPA:" was not part of [10]'s title.
- The author block lists Kadam, Shitole, Wavre, Singh and Zambare, with emails.

**Comments left for the authors:**
- The slow-pan sentence in §V ("near-zero baseline… negligible") no longer holds.
- Reference [8] (LangPrecip) is a radar-precipitation paper and only weakly supports mechanism M8. Consider a video-domain source.
- The RAFT citation now matches the method; consider naming RAFT where motion energy is defined.

**Pending: 18 LTX measurement values**

| Paper location | Needs |
|---|---|
| §V: LTX change in warping error vs motion energy | G-1 → G-4, corpus (C-1…C-3), codec wrapper (F-1). The cheapest: round trip plus flow, with no FVMD or generation |
| Table II floor column; §III.A "fast is ___× slow" | + FVMD (F-2), floor runs (F-3, F-4) |
| Table II end-to-end and codec share; abstract; §III.A %; conclusion (8 blanks) | + the generation arm (D-1), and decisions on Q-8 and Q-9 below |

---

## 9. Open questions that gate the headline numbers

| ID | Question | Why it matters |
|---|---|---|
| Q-1 | Is the LTX decoder deterministic at a fixed setting? | If not, the floor is a distribution and the statistics change (G-3) |
| Q-2 | Does untiled inference fit in 12 GB? | Tiling is itself mechanism M3 (D-1) |
| Q-8 | How is "codec share" defined? | Floor ÷ end-to-end assumes FVMD adds up across codec and denoiser; a Fréchet distance does not, and a stratum can show floor > end-to-end |
| Q-9 | Which stratum does a generated video belong to? | Stratifying by its own motion biases toward the hypothesis: a generator that under-moves lands in slow_pan |
| C-3 | Is p99 stable on real clips? | A tail statistic; bootstrap on the real corpus |

**Next step, G-1:** install diffusers and determine exactly what the LTX decode call takes and whether a deterministic, reconstruction-only setting exists. There is already one lead: the diffusers LTX pipeline defaults to `decode_timestep = 0.0` with no decode noise. G-1 must confirm what that does inside the decoder, because LTX's decoder also performs the final denoising step.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Codec floor | Temporal error of an encode→decode round trip; a lower bound on codec-attributable error |
| Proxy codec | A transparent stand-in reproducing only decimation (M1) and interpolation (M4); never presented as LTX |
| Motion energy | Mean optical-flow magnitude (paper §V) |
| Warping error | Residual after predicting a frame from its neighbour via flow; rewards smoothness, hence secondary |
| FVMD | Fréchet Video Motion Distance; the primary metric (D-2), still to be integrated (F-2) |
| EPE | Endpoint error: distance between estimated and true flow vectors, in px |
| 8k+1 | LTX's frame-count contract (9, 17, 25, 33, …) |
| Strata | slow_pan / fast_action / fine_texture: equal thirds by p99 flow, then texture energy |
| Golden hash | SHA-256 of an output taken at a verified point; any unintended change fails the test |
| Mutation test | Deliberately breaking the code to prove a test catches it |
