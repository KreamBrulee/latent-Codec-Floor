# codecfloor

How much of the temporal error in latent-diffusion video comes from the VAE codec rather than the denoiser? `codecfloor` measures the **codec floor**, the temporal error of an encode→decode round trip, so it can be separated from end-to-end generation error, stratified by motion content.

**Status (2026-09-25):**
- Prerequisite steps P-1 to P-4 are complete; 44 tests pass.
- The CPU/GPU harness is validated on a proxy codec.
- Next is step G-1: inspecting the LTX-Video decoder.

## Quick start

```powershell
python -m venv .venv
# Desktop with CUDA. On a CPU-only machine, drop the --index-url.
.venv\Scripts\pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -e ".[viz,dev]"

.venv\Scripts\python.exe -m pytest -q                  # 44 passed
.venv\Scripts\python.exe scripts\proxy_validation.py   # paper Table III -> runs/
```

## Where to read

| | |
|---|---|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | Architecture, everything built so far, challenges and solutions, findings, what's next |
| [HANDOVER_Implementation_Protocol.md](HANDOVER_Implementation_Protocol.md) | Working protocol, roadmap, standing decisions, risk register |
| [DECISIONS.md](DECISIONS.md) | Every design decision, with alternatives and revisit conditions |
| `runs/` | Immutable measurement records |

All results so far come from a **proxy codec on synthetic clips, NOT the LTX-Video VAE**.
