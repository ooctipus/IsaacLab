# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import to_dict

pytestmark = pytest.mark.unit


class SampleCommand(CommandTerm):
    """Random command with observable scheduling order, independent of simulation."""

    def __init__(self, cfg):
        super().__init__(cfg, SimpleNamespace(num_envs=3, device="cpu"))
        self.events = []
        self._command = torch.zeros(3)

    @property
    def command(self):
        return self._command

    def _update_metrics(self):
        self.events.append("metrics")

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs)[env_ids]
        self.events.append(("sample", ids.tolist()))
        self._command[env_ids] = torch.rand(len(ids))

    def _update_command(self):
        self.events.append("update")


def test_reset_only_compute_does_not_schedule_or_draw_randomness():
    term = SampleCommand(CommandTermCfg(resampling_time_range=None))
    assert torch.isinf(term.time_left).all()
    torch.manual_seed(42)
    expected_command = torch.rand(3)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    term.reset()
    torch.testing.assert_close(term.command, expected_command, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    term.events.clear()
    # Timer values are not requests when scheduling is explicitly disabled.
    term.time_left.copy_(torch.tensor([-1.0, 0.0, 1.0]))
    term.compute(100.0)
    assert term.events == ["metrics", "update"]
    assert torch.equal(term.time_left, torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(term.command_counter, torch.ones(3, dtype=torch.long))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize("env_ids", [None, [0, 2], slice(1, 3), [], slice(0, 0)])
def test_reset_only_explicit_reset_resamples_only_selected_environments(env_ids):
    term = SampleCommand(CommandTermCfg(resampling_time_range=None))
    term.time_left.fill_(-1.0)
    term.command_counter.fill_(7)
    ids = torch.arange(3)[slice(None) if env_ids is None else env_ids]
    rng = torch.random.get_rng_state()
    assert term.reset(env_ids) == {}
    expected_counter = torch.full((3,), 7, dtype=torch.long)
    expected_counter[ids] = 1
    assert torch.equal(term.command_counter, expected_counter)
    assert torch.equal(torch.isinf(term.time_left), torch.isin(torch.arange(3), ids))
    assert term.events == ([("sample", ids.tolist())] if ids.numel() else [])
    if not ids.numel():
        assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("interval", [(0.25, 0.25), (0.25, 0.75)])
def test_timed_scheduling_preserves_timer_draws_expiry_and_order(interval):
    term = SampleCommand(CommandTermCfg(resampling_time_range=interval))
    assert torch.equal(term.time_left, torch.zeros(3))
    # The established sequence draws timers before commands, even for equal bounds.
    torch.manual_seed(7)
    expected_time = torch.empty(3).uniform_(*interval)
    expected_command = torch.rand(3)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    term.reset()
    torch.testing.assert_close(term.time_left, expected_time, rtol=0, atol=0)
    torch.testing.assert_close(term.command, expected_command, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)

    term.time_left.copy_(torch.tensor([0.1, 1.0, 0.2]))
    term.events.clear()
    expected_time = torch.empty(2).uniform_(*interval)
    expected_command = torch.rand(2)
    expected_rng_after = torch.random.get_rng_state()
    torch.random.set_rng_state(expected_rng)
    term.compute(0.2)
    assert term.events == ["metrics", ("sample", [0, 2]), "update"]
    torch.testing.assert_close(term.time_left[[0, 2]], expected_time, rtol=0, atol=0)
    torch.testing.assert_close(term.command[[0, 2]], expected_command, rtol=0, atol=0)
    assert term.time_left[1] == 0.8
    assert torch.equal(term.command_counter, torch.tensor([2, 1, 2]))
    assert torch.equal(torch.random.get_rng_state(), expected_rng_after)


def test_reset_only_configuration_serializes_and_can_resume_timed_scheduling():
    cfg = CommandTermCfg(resampling_time_range=None)
    assert to_dict(cfg)["resampling_time_range"] is None
    term = SampleCommand(cfg)
    term.reset()
    cfg.resampling_time_range = (1.0, 1.0)
    term.reset()
    assert torch.equal(term.time_left, torch.ones(3))
    term.events.clear()
    term.compute(1.0)
    assert term.events == ["metrics", ("sample", [0, 1, 2]), "update"]
