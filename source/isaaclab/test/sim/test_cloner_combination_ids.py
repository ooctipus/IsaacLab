# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-combination identity tracking without simulator dependencies."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from isaaclab.cloner import make_clone_plan


def test_make_clone_plan_stores_combination_ids(monkeypatch):
    """The clone plan identifies the effective combination selected for each environment."""

    class MultiAssetSpawnerCfg:
        pass

    class MultiUsdFileCfg:
        pass

    fake_sim = ModuleType("isaaclab.sim")
    fake_sim.MultiAssetSpawnerCfg = MultiAssetSpawnerCfg
    fake_sim.MultiUsdFileCfg = MultiUsdFileCfg
    monkeypatch.setitem(sys.modules, "isaaclab.sim", fake_sim)

    stairs = SimpleNamespace(
        prim_path="/World/envs/env_.*/Stairs",
        spawn=SimpleNamespace(spawn_path=None),
    )
    chair = SimpleNamespace(
        prim_path="/World/envs/env_.*/Chair",
        spawn=SimpleNamespace(spawn_path=None),
    )
    valid_set = torch.tensor([[0, -1], [-1, 0]], dtype=torch.long)

    def choose_rows(combinations, num_clones, device):
        del num_clones
        return combinations[torch.tensor([1, 0, 1], dtype=torch.long, device=device)]

    plan = make_clone_plan(
        cfgs=[stairs, chair],
        num_clones=3,
        env_spacing=1.0,
        device="cpu",
        clone_strategy=choose_rows,
        valid_set=valid_set,
    )

    assert plan.combination_rows.tolist() == [[-1, 0], [0, -1]]
    assert plan.combination_ids.tolist() == [0, 1, 0]
    assert torch.equal(plan.combination_rows[plan.combination_ids], valid_set[torch.tensor([1, 0, 1])])
