# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the live G1 checkpoint preflight path and its canonical receipt."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parent
PREFLIGHT = ROOT / "g1_checkpoint_semantic_preflight.py"


def _module():
    spec = importlib.util.spec_from_file_location("g1_checkpoint_semantic_preflight", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_rollout_checks_live_action_target_without_host_reductions() -> None:
    """The live smoke must execute fixed device reductions and mapped target checks."""
    module = _module()
    action_term = SimpleNamespace(
        cfg=SimpleNamespace(scale=5.0, clip={".*": (-5.0, 5.0)}),
        joint_ids=torch.tensor((1, 0), dtype=torch.int64),
        raw_actions=torch.zeros(2, 2),
        processed_actions=torch.zeros(2, 2),
        joint_position_target=torch.zeros(2, 2),
    )
    physical_target = torch.zeros(2, 2)
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=physical_target)))

    class Model:
        observation_schema = SimpleNamespace(assert_valid=lambda _observations: None)

        def action_sample(self, _observations, _context, *, deterministic):
            assert deterministic
            return torch.tensor(((0.1, 0.2), (0.3, 0.4)))

    class Env:
        num_envs = 2
        device = "cpu"

        def reset(self):
            return {"state": {"joint": torch.zeros(2, 1)}}

        def step(self, actions):
            action_term.raw_actions.copy_(actions)
            action_term.processed_actions.copy_(actions * action_term.cfg.scale)
            action_term.joint_position_target.copy_(action_term.processed_actions)
            physical_target[:, action_term.joint_ids] = action_term.joint_position_target
            return (
                {"state": {"joint": torch.zeros(2, 1)}},
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    result = module._short_rollout(
        env=Env(),
        evaluation_scope=lambda *_args, **_kwargs: nullcontext(),
        command=object(),
        domain_scope=object(),
        history_factory=lambda _observations: None,
        model=Model(),
        context=torch.ones(1, 2),
        robot=robot,
        action_term=action_term,
        seed=4728,
        steps=2,
    )

    assert result["finite"] is True
    assert result["done_rows"] == result["final_observation_rows"] == 0
    assert result["action_request_max_abs_error"] == 0.0
    assert result["processed_action_max_abs_error"] == 0.0
    assert result["action_target_max_abs_error"] == 0.0
