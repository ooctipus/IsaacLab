# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, device="cpu").app

"""Rest everything follows."""

import pytest
import torch
from isaaclab_physx.physics import PhysxCfg

from isaaclab.managers import TerminationManager, TerminationTermCfg
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


class DummyEnv:
    """Minimal mutable env stub for the termination manager tests."""

    def __init__(self, num_envs: int, device: str, sim: SimulationContext):
        self.num_envs = num_envs
        self.device = device
        self.sim = sim
        self.counter = 0  # mutable step counter used by test terms


def fail_every_5_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 5."""
    cond = env.counter > 0 and (env.counter % 5 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_10_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 10."""
    cond = env.counter > 0 and (env.counter % 10 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_3_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 3."""
    cond = env.counter > 0 and (env.counter % 3 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_selected_envs(env, term: str) -> torch.Tensor:
    """Return the environment mask selected for a term by the test."""
    return env.term_masks[term]


@pytest.fixture
def env():
    sim = SimulationContext(SimulationCfg(device="cpu", physics=PhysxCfg()))
    yield DummyEnv(num_envs=20, device="cpu", sim=sim)
    sim.clear_instance()


def test_initial_state_and_shapes(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps),
        "term_10": TerminationTermCfg(func=fail_every_10_steps),
    }
    tm = TerminationManager(cfg, env)

    # Active term names
    assert tm.active_terms == ["term_5", "term_10"]

    # Public term values have expected shapes and start as all False
    assert tm.get_term("term_5").shape == (env.num_envs,)
    assert tm.get_term("term_10").shape == (env.num_envs,)
    assert tm.dones.shape == (env.num_envs,)
    assert tm.time_outs.shape == (env.num_envs,)
    assert tm.terminated.shape == (env.num_envs,)
    assert torch.all(~tm.get_term("term_5")) and torch.all(~tm.get_term("term_10"))


def test_term_transitions_and_reset_metrics(env):
    """Reset metrics report the terms that ended the current episodes."""
    cfg = {
        "term_3": TerminationTermCfg(func=fail_every_3_steps, time_out=False),
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),
    }
    tm = TerminationManager(cfg, env)

    env.counter = 3
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(out)
    assert tm.reset() == {"Episode_Termination/term_3": 1.0, "Episode_Termination/term_5": 0.0}

    env.counter = 4
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))

    env.counter = 5
    out = tm.compute()
    assert torch.all(~tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert tm.reset() == {"Episode_Termination/term_3": 0.0, "Episode_Termination/term_5": 1.0}

    env.counter = 15
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert tm.reset() == {"Episode_Termination/term_3": 1.0, "Episode_Termination/term_5": 1.0}


def test_reset_metrics_track_each_environments_last_episode(env):
    """Reset metrics retain the last completed episode for environments not reset now."""
    env.term_masks = {
        "first": torch.arange(env.num_envs) == 0,
        "second": torch.arange(env.num_envs) == 1,
    }
    tm = TerminationManager(
        {
            "first": TerminationTermCfg(func=fail_selected_envs, params={"term": "first"}),
            "second": TerminationTermCfg(func=fail_selected_envs, params={"term": "second"}),
        },
        env,
    )
    tm.compute()
    metrics = tm.reset([0, 1])
    assert metrics["Episode_Termination/first"] == pytest.approx(1 / env.num_envs)
    assert metrics["Episode_Termination/second"] == pytest.approx(1 / env.num_envs)

    env.term_masks["first"] = torch.arange(env.num_envs) == 2
    env.term_masks["second"] = torch.zeros(env.num_envs, dtype=torch.bool)
    tm.compute()
    metrics = tm.reset([2])
    assert metrics["Episode_Termination/first"] == pytest.approx(2 / env.num_envs)
    assert metrics["Episode_Termination/second"] == pytest.approx(1 / env.num_envs)

    env.term_masks["first"][:] = False
    tm.compute()
    metrics = tm.reset([0])
    assert metrics["Episode_Termination/first"] == pytest.approx(1 / env.num_envs)
    assert metrics["Episode_Termination/second"] == pytest.approx(1 / env.num_envs)


def test_time_out_vs_terminated_split(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),  # terminated
        "term_10": TerminationTermCfg(func=fail_every_10_steps, time_out=True),  # timeout
    }
    tm = TerminationManager(cfg, env)

    # Step 5: terminated fires, not timeout
    env.counter = 5
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(~tm.time_outs)

    # Step 10: both fire; timeout and terminated both True
    env.counter = 10
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(tm.time_outs)
