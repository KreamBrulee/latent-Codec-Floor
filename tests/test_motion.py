"""P-3: stratification (Q-6) as an instrument check on the fixtures (D-4)."""
from __future__ import annotations

import pytest

from codecfloor import motion, synthetic


@pytest.fixture(scope="module")
def profiles():
    return {name: motion.profile(make()) for name, make in synthetic.CLIPS.items()}


def test_fixtures_land_in_their_own_strata(profiles):
    # The old mean-flow rule gave fast_action->slow_pan and
    # fine_texture->fast_action on exactly these clips.
    ps = list(profiles.values())
    motion.assign_strata(ps)
    assert {n: p.stratum for n, p in profiles.items()} == {n: n for n in profiles}


def test_frozen_cutpoints_are_reused_not_recomputed(profiles):
    ps = list(profiles.values())
    cut = motion.assign_strata(ps)
    frozen = dict(cut, fast_action_flow_p99_cut=1e9)
    motion.assign_strata(ps, frozen)
    assert all(p.stratum != "fast_action" for p in ps)


def test_equal_thirds():
    mk = lambda f, h: motion.MotionProfile(dssim=0, flow_mag=0, flow_p95=0, flow_p99=f, hf_energy=h)
    ps = [mk(f, h) for f, h in zip(range(9), [5, 1, 9, 2, 8, 3, 7, 4, 6])]
    motion.assign_strata(ps)
    counts = {s: sum(p.stratum == s for p in ps) for s in ("fast_action", "fine_texture", "slow_pan")}
    assert counts == {"fast_action": 3, "fine_texture": 3, "slow_pan": 3}, counts


def test_too_few_clips_is_an_error():
    with pytest.raises(ValueError):
        motion.strata_cutpoints([])
