"""P-4: the proxy must not freeze frames past the last keyframe.

Each test targets a named failure mode from step P-4, section C.
"""
from __future__ import annotations

import numpy as np
import pytest

from codecfloor import synthetic
from codecfloor.proxy_codec import temporal_decimate_restore


@pytest.mark.parametrize("t", [32, 34, 8, 1])
def test_rejects_lengths_outside_ltx_contract(t):
    v = np.zeros((t, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="8k\\+1"):
        temporal_decimate_restore(v, 8)


@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("name", list(synthetic.CLIPS))
def test_no_frozen_tail(name, mode):
    # C2: an off-by-one on the last keyframe would reintroduce a static run.
    v = synthetic.CLIPS[name]()
    r = temporal_decimate_restore(v, 8, mode)
    assert r.shape == v.shape
    assert np.array_equal(r[-1], v[-1])
    assert np.array_equal(r[::8], v[::8])
    if mode == "linear":
        # Every source clip moves on every frame, so any identical consecutive
        # pair in the output can only be a hold artefact.
        assert all(not np.array_equal(v[i], v[i + 1]) for i in range(len(v) - 1))
        held = [i for i in range(len(r) - 1) if np.array_equal(r[i], r[i + 1])]
        assert held == [], f"static output frames at {held}"


def test_window_at_integer_offset_is_plain_crop():
    # The sub-pixel sampler must not alter content when there is nothing to
    # interpolate; otherwise B5 changed more than the stepping.
    c = synthetic._texture_bg(128, 128, seed=7)
    assert np.array_equal(synthetic._window(c, 5, 9, 64, 48), c[9:57, 5:69])


@pytest.mark.parametrize("name", ["slow_pan", "fine_texture"])
def test_pans_move_at_uniform_speed(name):
    # P-4 B5: int-truncated offsets gave 0/1px (slow_pan) and 2/3px
    # (fine_texture) steps, a per-pair flow CV of roughly 0.8 and 0.2. The
    # bound sits well below both, so the stepping cannot come back unnoticed.
    from codecfloor.flow import farneback as _flow
    from codecfloor.video_io import to_gray
    g = (to_gray(synthetic.CLIPS[name]()) * 255).astype(np.uint8)
    speed = np.array([np.linalg.norm(_flow(g[i], g[i + 1]), axis=-1).mean()
                      for i in range(len(g) - 1)])
    assert speed.std() / speed.mean() < 0.05, speed.round(2)


@pytest.mark.parametrize("name", list(synthetic.CLIPS))
def test_33_frame_fixtures_extend_the_32_frame_ones(name):
    # C1: changing the default length must not change the shared frames, or
    # old and corrected tables are not comparable frame for frame.
    f = synthetic.CLIPS[name]
    assert np.array_equal(f(t=33)[:32], f(t=32))


def test_textured_balls_follow_identical_trajectories():
    # P-3 C1: texturing must change appearance only. Both versions share the
    # background, so each frame's ball region is where it differs from it; the
    # regions must coincide up to 1px (cv2.circle vs analytic-disc edges).
    import cv2
    flat = synthetic.fast_action(textured=False)
    tex = synthetic.fast_action()
    bg = cv2.GaussianBlur(synthetic._texture_bg(256, 256, seed=2), (31, 31), 0)
    k = np.ones((3, 3), np.uint8)
    for f, t in zip(flat, tex):
        a = np.any(f != bg, -1).astype(np.uint8)
        b = np.any(t != bg, -1).astype(np.uint8)
        assert a.sum() > 1000
        assert not np.any(a & ~cv2.dilate(b, k).astype(bool))
        assert not np.any(b & ~cv2.dilate(a, k).astype(bool))


def test_nearest_mode_ties_break_consistently():
    # P-3 B3: every half-way frame goes to the later keyframe.
    v = np.arange(33, dtype=np.uint8)[:, None, None, None].repeat(3, -1).repeat(2, 1).repeat(2, 2)
    r = temporal_decimate_restore(v, 8, "nearest")
    assert [int(r[i, 0, 0, 0]) for i in (4, 12, 20, 28)] == [8, 16, 24, 32]
