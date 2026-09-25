# Implementation Handover & Working Protocol
## Codec-Floor Attribution System

**Purpose of this document.** Two things. First, it carries the project state forward so any session can resume without re-deriving context. Second — and this is the part that governs everything — it defines **the contract for how implementation work is conducted**: what I must give you before any code is written, what you must verify, and what gets recorded. Read §2 before anything else. If a step is ever delivered to you in a form that violates §2, reject it and say so.

---

## 1. Where the project currently stands

### 1.1 Objective status

| # | Objective | Status | Evidence |
|---|---|---|---|
| O1 | Architecture study & mechanism taxonomy | **Complete** | Nine-mechanism taxonomy, each with a primary source; scope statement of which mechanisms the protocol can and cannot observe |
| O2 | Implement measurement protocol | **Partial** | CPU harness built and validated; GPU path blocked on G-1 (below) |
| O3 | Measure floor across strata & compression ratios | Not started | Depends on O2 |
| O4 | Measure generation error, decompose | Not started | Depends on O2, O3 |
| O5 | Quantify headroom | Not started | Depends on O4 |
| O6 | Publish protocol, floors, harness | Not started | — |

### 1.2 Code that exists and is validated

```
codecfloor/
├── codecfloor/
│   ├── video_io.py      # read/write/normalise → (T,H,W,3) uint8 RGB
│   ├── motion.py        # DSSIM complexity, flow stats, HF energy, stratification
│   ├── metrics.py       # reference-free warping error, raw frame diff, PSNR
│   ├── proxy_codec.py   # temporal decimation + interpolation (NOT the LTX VAE)
│   └── synthetic.py     # deterministic test clips: slow_pan, fast_action, fine_texture
└── scripts/
    ├── make_demo_figures.py
    ├── make_architecture_figure.py
    └── make_methodology_figure.py
```

All CPU-only. All tested. No GPU dependency anywhere in the domain layer — **this is deliberate and must be preserved** (see D-3).

> **Correction, 2026-09-23.** The tree above describes the intended state, not the working tree. Actually present: the five domain modules (byte-identical to the handed-over files), now in `codecfloor/` with `pyproject.toml`, `tests/test_portable.py` and `DECISIONS.md` (added in P-1). **Absent:** `scripts/`, any original tests, the `Codec` interface / `ProxyCodec` class (only free functions exist), `MeasurementPath` / `MetricEngine` / `ErrorDecomposer`, and `cli.py`. See Q-5. Prerequisite steps P-1 to P-4 run before G-1.

### 1.3 Findings already established

**F-1 — Motion loss is stratified.** Proxy codec, synthetic clips, motion energy retained: slow pan 71%, fast action 4%, fine texture 4%. Direction matches the taxonomy's prediction.

**F-2 — Warping error is not safe as a primary metric.** On fast action, warping error *decreased* 40% while 96% of motion energy was destroyed. Smoothness-based metrics reward over-smoothing; a video with no motion left is trivially self-consistent. Consequence: FVMD is primary, warping error is a secondary diagnostic only.

> **Corrected 2026-09-25 (P-4, then P-3).** The original figures came from 32-frame clips with a frozen tail, integer-stepped pans, flat-coloured balls, and Farneback flow, which under-read local motion. Current values, from `runs/proxy_validation_20260925T091403Z.json` (RAFT-large flow): motion retained slow pan **97.7%**, fast action **1.6%**, fine texture **0.8%**. On fast action, warping error decreased **47.2%** while **98.4%** of motion energy was removed. F-2 holds for fast action; on fine texture, warping error correctly rises (+37.9%).

> F-1 and F-2 are from a **proxy codec on synthetic clips**. They demonstrate a mechanism and validate the instrument. They are **not** measurements of the LTX floor and must never be presented as such.

---

## 2. THE WORKING PROTOCOL

This section is the reason the document exists.

### 2.1 The step format — what you receive before any code

Every implementation step is delivered in this exact structure. No code appears before sections A–D of that step.

```
STEP <id> — <short name>

A. WHY THIS STEP EXISTS
   Which objective / research question it serves. If it serves none,
   it should not be done.

B. WHAT IS BEING DECIDED
   The design decisions this step forces, each with:
     - the options considered
     - the option chosen
     - the reason the others were rejected
     - what would make us revisit the choice

C. WHAT COULD INVALIDATE THE RESULT
   The specific ways this step can silently produce a wrong number.
   Not generic risk — named, concrete failure modes.

D. ASSUMPTIONS BEING MADE
   Every assumption, tagged:
     [VERIFIED]  confirmed against source/docs/experiment — citation given
     [INFERRED]  reasoned from something verified — reasoning given
     [ASSUMED]   taken on faith — must be checked before results are used

E. THE IMPLEMENTATION
   Code. Comments explain *why*, never *what*.

F. ACCEPTANCE CHECK
   A concrete command you run and a concrete expected outcome.
   "It runs without error" is NOT an acceptance check.
   Must be able to FAIL.

G. WHAT TO RECORD
   Values, versions, settings that enter the run manifest.

H. WHAT THIS UNBLOCKS / WHAT REMAINS BLOCKED
```

### 2.2 Rules of engagement

**R-1 — One step at a time.** No step begins until the previous one's acceptance check has passed *on your machine*. Not "should pass" — passed.

**R-2 — No silent decisions.** If a step forces a choice I have not surfaced in section B, that is a defect in the step. Call it out.

**R-3 — Uncertainty is tagged, never smoothed over.** Every factual claim carries `[VERIFIED]`, `[INFERRED]`, or `[ASSUMED]`. If I state a number about LTX, diffusers, or any paper without a tag and a source, treat it as unverified and check it.

**R-4 — No fabricated values, ever.** No placeholder numbers presented as measurements, no illustrative figures without an explicit label. Applies to code defaults too: a hardcoded threshold with no justification is a fabricated value.

**R-5 — Failure modes before features.** Section C precedes section E. If I cannot name how a step could silently produce a wrong number, I do not understand the step well enough to implement it.

**R-6 — You can stop and interrogate any step.** Legitimate at any point: *"justify that choice", "what happens if that assumption is wrong", "show me the alternative we rejected", "what would this look like if it were broken?"* A step that cannot survive this is not ready.

**R-7 — Decisions get logged, not remembered.** Every section-B decision goes into `DECISIONS.md` with date, rationale, and revisit condition. Research decisions made three weeks ago are not reliably recalled — and a reviewer will ask why you chose what you chose.

**R-8 — Scope discipline.** No step adds capability that no objective requires. Speculative generality is how side projects consume a timeline.

### 2.3 What you are responsible for

The protocol only works if the verification is real:

- **Run the acceptance check yourself.** Do not accept a reported pass.
- **Check the `[ASSUMED]` tags.** These are where wrong results come from.
- **Push back when a justification is thin.** "Because it's standard" is not a justification. Ask what breaks if we do the opposite.
- **Maintain `DECISIONS.md`.** One line per decision, written when made.

### 2.4 Worked example of the format

```
STEP G-1 — Determine the LTX decoder's conditioning interface

A. WHY
   O2. The floor protocol requires a reconstruction-only decode pass.
   LTX-Video §2 states the decoder also performs the final denoising
   step, so "decode" may not be a pure operation. Every subsequent
   measurement depends on resolving this.

B. WHAT IS BEING DECIDED
   B1. Which decoder invocation constitutes the "codec-only" path.
       Options: (i) lowest available noise/timestep setting;
                (ii) bypass the denoising component if separable;
                (iii) accept that it is partly generative and document.
       Not yet decided — this step gathers the evidence to decide.
       Revisit condition: n/a, this IS the decision step.

C. WHAT COULD INVALIDATE THE RESULT
   C1. Decoder is stochastic → repeated round trips differ → the
       "floor" is a distribution, not a value, and single-run numbers
       are meaningless.
   C2. Default settings invoke generation → we measure codec+partial
       generation and call it codec. Silent, and fatal to the claim.
   C3. Setting differs between diffusers versions → results are not
       reproducible across environments.

D. ASSUMPTIONS
   [VERIFIED] LTX assigns the decoder the final denoising step
              — LTX-Video paper §2.
   [VERIFIED] Compression is 32×32×8, 128 latent channels, ~1:192
              — LTX-Video paper, abstract and §2.
   [ASSUMED]  diffusers exposes the conditioning as a callable
              argument rather than burying it — UNCHECKED, this step
              checks it.

E. IMPLEMENTATION
   Inspection only, no production code:
     - locate the LTX VAE/decoder class in the installed diffusers
     - print the decode signature and any timestep/noise parameters
     - read the forward pass for where conditioning enters

F. ACCEPTANCE CHECK
   Produce a written statement of: the exact decode call signature,
   every conditioning parameter with its default, and whether a
   deterministic setting exists.
   FAILS if the answer is "probably" anything.

G. RECORD
   diffusers version, class name, full signature, chosen setting,
   and the justification → DECISIONS.md and the run manifest schema.

H. UNBLOCKS
   G-2 (still-image round trip), and all of O3.
   BLOCKS UNTIL DONE: every floor measurement.
```

---

## 3. Implementation roadmap

Ordered by dependency. Rationale is given because the order is not arbitrary — it is sequenced so that **the cheapest steps that can kill the project come first**.

### Phase G — Gate checks (must complete before any measurement)

| Step | Name | Why it is first |
|---|---|---|
| **G-1** | Decoder conditioning interface | Blocks everything. Unresolved, every floor number is uninterpretable. |
| **G-2** | Still-image round trip | Cheapest possible test of the decode path. If a single frame does not round-trip stably, video certainly will not. |
| **G-3** | Determinism check | Run G-2 five times with identical input. If outputs differ, the floor is a distribution and the whole statistical design changes. Better to know now. |
| **G-4** | VRAM envelope & tiling decision | See D-1 — this is a research decision, not an engineering one. |

### Phase C — Corpus

| Step | Name | Why here |
|---|---|---|
| **C-1** | Source and licence a real corpus | Must precede profiling. Licensing is a hard constraint from the ethics commitment, not an afterthought. |
| **C-2** | Fix the normalisation spec | Resolution and frame count must be frozen before any measurement, or runs are incomparable. |
| **C-3** | Profile and stratify; freeze cut-points | Cut-points must be recorded; stratification computed per-run is not reproducible. |

### Phase F — Floor measurement

| Step | Name | Why here |
|---|---|---|
| **F-1** | Codec wrapper implementing the `Codec` interface | Isolates LTX behind the same interface as ProxyCodec, so the validated CPU pipeline runs unchanged. |
| **F-2** | FVMD integration | Primary metric must exist before the first real run — F-2 in §1.3 is exactly why we do not run first and choose metrics later. |
| **F-3** | Pilot floor run, N≈20, one stratum | Smallest run that can expose a broken pipeline. |
| **F-4** | Full floor run, all strata | Only after F-3 passes. |
| **F-5** | CogVideoX floor run | Compression-scaling comparison (O3). |

### Phase D — Decomposition

| Step | Name | Why last |
|---|---|---|
| **D-1** | Generation arm, base model | Requires a working metric engine and a validated floor. |
| **D-2** | LoRA fine-tune arm | Only meaningful once base generation is measured. |
| **D-3** | Decomposition and headroom | Terminal step; consumes everything above. |

---

## 4. Standing decisions already made

These are in force unless explicitly revisited. Each has a revisit condition.

**D-1 — Tiled VAE inference is a research decision, not an engineering convenience.**
Tiling is **mechanism 3 in our own taxonomy** — chunk-boundary discontinuities are a documented cause of flicker (WF-VAE Fig. 3(b)). If we enable tiling to fit 12 GB, *we introduce the artefact we are trying to measure*, and the measured floor becomes partly an artefact of our own inference configuration.
*Decision:* tiling must be OFF for floor measurement, or if unavoidable, its settings must be fixed, recorded, and reported as a stated limitation, with a sensitivity check at two tile configurations.
*Revisit if:* VRAM makes untiled inference impossible at the chosen resolution — in which case reduce resolution or frame count first, before enabling tiling.

**D-2 — FVMD primary, warping error secondary.**
Established by F-2. *Revisit if:* a reference-free metric is found that penalises both excess and deficient motion.

**D-3 — The domain layer stays GPU-free.**
`Codec` is an interface; `ProxyCodec` and `LTXCodec` are peers. This is what allowed the entire harness to be validated on CPU and F-2 to be found at zero cost. *Revisit:* never. This one is load-bearing.

**D-4 — Synthetic clips are for instrument validation only.**
They never produce reported findings. *Revisit:* never.

**D-5 — Runs are sealed and immutable.**
No measurement is appended to a sealed run. Re-measurement creates a new run. *Revisit:* never.

**D-6 — The core package must import and run without torch.** *(Narrowed 2026-09-25, DECISIONS P-3 B5b: it must still import without torch, but flow-based metrics need torch + torchvision; the CPU build suffices.)*
Development happens on the desktop; demonstration happens on the laptop. Heavy dependencies are lazily imported inside their implementations, never at core module scope, so that `pip install -e ".[viz]"` on a machine with no CUDA yields a working Mode A and Mode B (see §5). Enforced by `tests/test_portable.py`, not by convention. *Revisit:* never — this is D-3 made testable, and it is also what made finding F-2 possible.

---

## 5. Portability: development host vs demonstration host

### 5.1 The constraint

Development happens on the desktop (CUDA, model weights, heavy dependencies). Demonstration happens on the laptop — to a guide, a review panel, or a collaborator — on a machine with no GPU, no weights, and no patience for a build.

The failure mode being designed against is general and common: a project so entangled with its execution environment that **showing even a skeleton requires standing up the entire stack**. If demonstrating the work means a twenty-minute container build, the work does not get demonstrated — and a project that cannot be shown is judged on its documents alone.

### 5.2 Why this is architectural, not convenience

D-3 already requires a GPU-free domain layer. That rule is currently kept by discipline, and discipline rots: one `import torch` at the top of a core module, added during a debugging session, silently breaks laptop portability and nobody notices until the demo. This section turns D-3 into something **testable**, and the laptop requirement is what forces the test to exist.

There is a research benefit too, not only a presentational one. The same separation is what allowed the entire harness to be validated on CPU and finding F-2 to be obtained at zero GPU cost. Portability and cheap experimentation are the same property.

### 5.3 Dependency tiers

Dependencies are declared in tiers, installable independently:

| Tier | Contains | Installs with | Host |
|---|---|---|---|
| **core** | numpy, opencv, scikit-image, scipy | `pip install -e .` | Both |
| **viz** | matplotlib, pandas | `pip install -e ".[viz]"` | Both |
| **gpu** | torch, diffusers, peft, finetrainers | `pip install -e ".[gpu]"` | Desktop only |
| **dev** | pytest, ruff | `pip install -e ".[dev]"` | Both |

`core + viz` must be sufficient to run every demonstration mode in §5.5 except the full GPU run. On a clean laptop the install should be a single `pip` command and take under a minute.

### 5.4 The import rule — the thing that actually breaks

Heavy dependencies are **never imported at module top level in core packages**. GPU imports are local to the implementation that needs them:

```python
# codecfloor/codecs/ltx.py

from ..codec import Codec          # core — always safe

class LTXCodec(Codec):
    def __init__(self, model_id: str, revision: str):
        # Imported here, not at module scope, so that importing the
        # package on a machine without torch does not fail. This is
        # load-bearing for laptop portability (see D-6) and is enforced
        # by tests/test_portable.py.
        import torch
        from diffusers import AutoencoderKLLTXVideo
        ...
```

The same applies to registries: a codec registry may *name* `LTXCodec` without importing it, resolving the class lazily on first use.

### 5.5 The three demonstration modes

| Mode | What runs | What it proves | Needs | Cold start |
|---|---|---|---|---|
| **A — Artifact replay** | Reads a sealed run from the repo; regenerates every figure and table from stored measurements | The results, the reporting pipeline, reproducibility of published figures | core + viz | seconds |
| **B — Live CPU pipeline** | Full measurement pipeline end to end, using `ProxyCodec` on synthetic clips | That the *code path* works, not just the outputs — stratification, metric engine, decomposition, manifest sealing | core + viz | under a minute |
| **C — Full GPU run** | Real codec, real corpus | The actual measurement | gpu + weights + corpus | desktop only |

Mode B is the important one and it already exists in embryo. It runs the identical `MeasurementPath`, `MetricEngine` and `ErrorDecomposer` as the GPU path — only the `Codec` implementation is substituted. Demonstrating it is a genuine demonstration of the system, not a mock-up, and the substitution is exactly the design property (D-3) that makes the architecture worth talking about. When asked "can you show it running?", Mode B is the answer.

### 5.6 What crosses between machines

| Travels in git | Stays on the desktop |
|---|---|
| All source code | Model weights |
| Sealed run exports (JSON/CSV — kilobytes) | The video corpus |
| Run manifests | Raw latents and intermediate tensors |
| Figure-generation scripts | Generated video output |
| Corpus manifest: filenames, checksums, profiles, strata | |
| A handful of small sample clips for Mode B | |

Figures are **never** committed as the source of truth — the measurements are, and figures regenerate from them. This keeps the repository small and means a figure can be restyled for the paper without re-running anything.

The corpus is referenced by checksum, not carried. The manifest records what was measured; the bytes stay where the compute is.

### 5.7 Docker's role

Docker reproduces the **GPU environment** for archival and for third parties reproducing the results. It is never a prerequisite for running the project.

> **Hard rule:** `git clone` followed by `pip install -e ".[viz]"` on a clean laptop, with no Docker, no CUDA and no weights, must yield working Mode A and Mode B. If that ever stops being true, the regression is fixed before any feature work continues.

### 5.8 Acceptance check for the portability constraint

This is a permanent check, run on the laptop, not a one-time setup step:

```bash
# 1. Core imports must not require torch
python -c "import codecfloor; print('core import OK')"

# 2. Portability guard test must pass
pytest tests/test_portable.py -q

# 3. Mode A — replay a sealed run, regenerate all figures
python -m codecfloor.cli replay --run runs/<run-id>.json --out figures/

# 4. Mode B — full pipeline on CPU with the proxy codec
python -m codecfloor.cli demo --codec proxy --corpus synthetic
```

`tests/test_portable.py` asserts that importing every module under `codecfloor/` succeeds with `torch` absent from `sys.modules`, and fails loudly if any core module has acquired a heavy top-level import. Without that test the rule is a comment, and comments do not hold.

---

## 6. Open questions and risk register

| ID | Question / risk | Impact if it goes badly | Resolved by |
|---|---|---|---|
| **Q-1** | Is the LTX decoder deterministic at a fixed setting? | Floor becomes a distribution; statistical design changes | G-3 |
| **Q-2** | Can untiled inference fit in 12 GB at usable resolution? | Forces D-1 revisit, weakens the claim | G-4 |
| **Q-3** | Which real corpus, and under what licence? | Ethics commitment breach if wrong | C-1 |
| **Q-4** | How many samples does FVD need to stabilise here? | Under-powered results that look precise but are noise | F-4 |
| **R-1** | Codec floor turns out negligible | Premise weakened; becomes a negative-result paper | F-3 (early, cheap) |
| **R-2** | Floor is flat across strata | Contradicts the taxonomy prediction — reportable, but reframes the contribution | F-4 |
| **R-3** | Someone publishes the same attribution first | Contribution reduced to replication | Ongoing literature monitoring |
| **Q-5** | Original `scripts/` (figure generators) and tests are not in the working tree (found 2026-09-23). *Update 2026-09-25:* the paper (§V) defines motion energy as mean optical-flow magnitude; with that definition, paper Table III reproduces exactly on OpenCV 4.14. **But** frames 25–31 of the 32-frame clips lie past the last kept keyframe and are frozen by the proxy. Restricted to the valid range 0–24: fast-action Δwarp −40.4% → **−22.6%**, fine-texture +0.2% → **+29.3%**, slow-pan retained 71.5% → **91.8%**. The F-2 *direction* survives; the paper's numbers do not | Paper Table III, the "40.4% / 96.1%" sentence in §V, and handover F-1/F-2 figures are boundary-artefact inflated | **Resolved by P-4 (2026-09-25):** 8k+1 [VERIFIED]; fixture stepping also fixed; see DECISIONS P-4. Paper edits pending |
| **Q-6** | `assign_strata` bins on *mean* flow, which dilutes localised motion: on the synthetic fixtures it labels fast_action→slow_pan and fine_texture→fast_action (flow_p95 would order them correctly) | Real-corpus strata wrong in exactly the fast/local-motion case the taxonomy is about | **Resolved by P-3 (2026-09-25):** p99 flow, equal thirds; fixtures now land in their own strata. p99 stability still to check at C-3 |
| **Q-8** | "Codec share" (paper Table II, abstract, conclusion) is undefined. The implied floor ÷ end-to-end assumes FVMD is additive across codec and denoiser; a Fréchet distance is not, and a stratum can show floor > end-to-end | The headline percentage is arithmetic on a non-additive quantity (paper's own R2) | Decision before F-2; empirical additivity check on the proxy |
| **Q-9** | Generated video has no source clip, so its stratum is ambiguous. Stratifying by the generated video's own motion creates selection bias (a generator that under-moves lands in slow_pan) | Per-stratum end-to-end numbers biased in the direction of the hypothesis | Decision before D-1 (e.g. image+caption conditioning from the stratum's source clips) |
| **Q-10** | Paper cites RAFT [15] for optical flow; every harness number uses Farneback (OpenCV) | Reviewer-visible mismatch between stated and actual method | **Resolved by P-3 (2026-09-25):** harness now uses RAFT-large (benchmark-selected), matching the citation |
| **Q-11** | Farneback (current params) is accurate on the pans (EPE 0.03 px) but on fast_action recovers **35%** of the true ≈10.4 px/frame ball displacement: flat-coloured discs have no interior texture to track (textured discs: 80%). The fast-action source's motion energy is under-read, and part of its warping error is flow failure, not incoherence (measured 2026-09-25) | Table III fast-action row and the F-2 magnitude partly reflect the estimator, not the codec. The same limit applies to stratification and §V on real fast content | **Resolved by P-3 (2026-09-25):** textured balls; RAFT-large reads the fixture's motion correctly (0.737 vs ≈0.75 px/frame expected) |
| **Q-7** | Warping error is library-version sensitive (up to 6.2% between OpenCV 4.14 and 5.0; flow/SSIM/PSNR unaffected) | Silent cross-machine drift in a reported metric | Mitigated by pin (DECISIONS P-1 B7); library versions must enter the run manifest |

> R-1 and R-2 are **not failures**. They are outcomes. The project is designed so that either answer is publishable; what is not acceptable is not knowing.

---

## 7. Definition of done

A step is done when **all** hold:

1. Acceptance check has been run by you and passed.
2. Every `[ASSUMED]` tag in the step is either resolved to `[VERIFIED]` or explicitly carried forward into the risk register.
3. Section-B decisions are written into `DECISIONS.md`.
4. Values in section G are recorded in the manifest.
5. You can explain, without reference to the document, why the step exists and what would make its output wrong.

Condition 5 is the real one. The rest are bookkeeping.

---

## 8. Starting instruction for the next session

> Resume the Codec-Floor Attribution project. I have read the working protocol in §2 and expect every implementation step in that format — sections A–D before any code, acceptance checks that can fail, and assumptions tagged.
>
> Begin with **STEP G-1**: determine the LTX-Video decoder's conditioning interface in the installed diffusers version. Do not write production code in this step; it is an inspection step. I need to know exactly what the decode call takes, what the defaults are, and whether a deterministic reconstruction-only setting exists.
>
> Current state: O1 complete, CPU harness validated, findings F-1 and F-2 established on a proxy codec. Standing decisions D-1 through D-6 are in force, including the portability constraint in §5.
