"""P-2: the Codec interface and ProxyCodec behind it."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from codecfloor import synthetic
from codecfloor.codec import Codec
from codecfloor.proxy_codec import ProxyCodec

# SHA-256 of ProxyCodec(8, mode).round_trip(clip). First taken before the P-2
# refactor, which moved no byte (P-2 C1). Re-taken 2026-09-25 after P-3's
# intended changes: rounding (B2), tie-breaking (B3), textured balls (B1).
# numpy 2.5.3, OpenCV 4.14.0. Any other change must leave these intact.
GOLDEN = {
    ("slow_pan", "linear"): "021277f5653347f0a14cbab0668c3952c70c0e902c7ef9b714c39680cc03d431",
    ("slow_pan", "nearest"): "a2e8cb3706687b8abc4d07642475624da7a89514f44eb973a6fa34704e706690",
    ("fast_action", "linear"): "66ce66c72a7fdf877bb68d65312a767242804165b5230a4ff6f1f0596d49458f",
    ("fast_action", "nearest"): "2edaafa938ccd917ef0faa68f2878940e6080eb0a290a87357ae101ea929faac",
    ("fine_texture", "linear"): "d9ac39d4aba01c8e1fe123c246fb1e5a874396e329a6ee7d7384087ef34cd508",
    ("fine_texture", "nearest"): "da419bfcb561f2755461ff4702ed684abee6d31784bd22f6b0f576be60245c6c",
}


@pytest.mark.parametrize("name,mode", list(GOLDEN))
def test_proxy_output_is_byte_identical_to_validated_version(name, mode):
    out = ProxyCodec(8, mode).round_trip(synthetic.CLIPS[name]())
    assert hashlib.sha256(out.tobytes()).hexdigest() == GOLDEN[(name, mode)]


def test_decode_recovers_length_from_keyframes():
    v = synthetic.fast_action()
    latent = ProxyCodec().encode(v)
    assert latent.shape[0] == (len(v) - 1) // 8 + 1
    assert ProxyCodec().decode(latent).shape == v.shape


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        ProxyCodec(8, "Linear")


def test_incomplete_codec_cannot_be_instantiated():
    class Half(Codec):
        name = "half"

        def encode(self, video):
            return video

    with pytest.raises(TypeError):
        Half()


class _Broken(Codec):
    """A codec that silently alters the video; the base class must refuse it."""
    name = "broken"

    def __init__(self, damage):
        self.damage = damage

    def encode(self, video):
        return video

    def decode(self, latent):
        return self.damage(latent)

    def config(self):
        return {}


@pytest.mark.parametrize("damage", [
    lambda v: v[:-1],                                   # drops a frame
    lambda v: v[:, :-32],                               # crops rows
    lambda v: v.astype(np.float32),                     # leaks float output
])
def test_round_trip_rejects_contract_violations(damage):
    with pytest.raises(ValueError, match="round trip changed"):
        _Broken(damage).round_trip(synthetic.slow_pan())


def test_round_trip_rejects_bad_input():
    with pytest.raises(ValueError, match="expected"):
        ProxyCodec().round_trip(synthetic.slow_pan().astype(np.float32))
