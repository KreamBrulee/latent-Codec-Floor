"""LTX-Video VAE behind the Codec interface (roadmap F-1).

Codec-only path, fixed by G-1 (DECISIONS.md, L-series):
  * VAE from LTX-Video 0.9.5: timestep-conditioned decoder with no noise
    injection, so decode is deterministic given the timestep. (0.9.1 injects
    noise in 3 of 4 decoder blocks; 0.9.0 has no timestep conditioning.)
  * decode timestep 0: the decoder is told the latent is clean. This is the
    pipeline's own default (decode_timestep=0.0, decode_noise_scale=None ->
    no noise mixed in), i.e. reconstruction, not the final denoising step.
  * encode takes the posterior mode, not a sample: deterministic.
  * tiling and frame-wise processing OFF (D-1: tiling is mechanism M3).
  * TF32 off for the call (restored after), as for the flow estimator.

torch/diffusers are imported inside methods so the package imports without
them (D-6).
"""
from __future__ import annotations

import numpy as np

from .codec import Codec

REPO = "Lightricks/LTX-Video-0.9.5"


class LTXCodec(Codec):
    name = "ltx-video-0.9.5-vae"

    def __init__(self, revision: str, dtype: str = "float32", decode_timestep: float = 0.0,
                 device: str = "cuda"):
        import torch
        from diffusers import AutoencoderKLLTXVideo

        self.revision, self.dtype_name, self.decode_timestep = revision, dtype, decode_timestep
        self.dtype = getattr(torch, dtype)
        self.device = device
        vae = AutoencoderKLLTXVideo.from_pretrained(REPO, subfolder="vae", revision=revision,
                                                    torch_dtype=self.dtype)
        vae.disable_tiling()
        for flag in ("use_framewise_encoding", "use_framewise_decoding"):
            if getattr(vae, flag, False):
                raise RuntimeError(f"{flag} is on; it would chunk time (D-1)")
        self.vae = vae.to(device).eval()
        cfg = self.vae.config
        self.t_ratio = cfg.temporal_compression_ratio or 8
        self.s_ratio = cfg.spatial_compression_ratio or 32

    def config(self) -> dict:
        import diffusers
        import torch
        cfg = self.vae.config
        return {"codec": self.name, "repo": REPO, "revision": self.revision, "subfolder": "vae",
                "dtype": self.dtype_name, "decode_timestep": self.decode_timestep,
                "posterior": "mode", "tiling": False, "tf32": False, "device": self.device,
                "timestep_conditioning": bool(cfg.timestep_conditioning),
                "decoder_inject_noise": list(cfg.decoder_inject_noise),
                "spatial_compression": self.s_ratio, "temporal_compression": self.t_ratio,
                "latent_channels": cfg.latent_channels,
                "diffusers": diffusers.__version__, "torch": torch.__version__}

    def _no_tf32(self):
        import contextlib

        import torch

        @contextlib.contextmanager
        def ctx():
            saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
            torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.no_grad():
                    yield
            finally:
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
        return ctx()

    def encode(self, video: np.ndarray):
        import torch
        t, h, w, _ = video.shape
        if (t - 1) % self.t_ratio or h % self.s_ratio or w % self.s_ratio:
            raise ValueError(f"LTX needs T=8k+1 and H,W divisible by 32; got {video.shape}")
        x = torch.from_numpy(video).to(self.device).permute(3, 0, 1, 2)[None]
        x = (x.to(self.dtype) / 127.5) - 1.0
        with self._no_tf32():
            return self.vae.encode(x).latent_dist.mode()

    def decode(self, latent) -> np.ndarray:
        import torch
        temb = torch.tensor([self.decode_timestep], device=self.device, dtype=self.dtype)
        with self._no_tf32():
            out = self.vae.decode(latent, temb, return_dict=False)[0]
        out = ((out.float().clamp(-1, 1) + 1.0) * 127.5).round()
        return out[0].permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()
