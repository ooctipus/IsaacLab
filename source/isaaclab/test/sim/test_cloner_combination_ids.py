# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-combination identity tracking without simulator dependencies."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from isaaclab.cloner import InclusionSet, make_clone_plan, make_valid_clone_combinations


def test_make_valid_clone_combinations_preserves_semantic_ids():
    """Weights and variants preserve the original inclusion-set ID for every valid row."""
    rows, combination_ids = make_valid_clone_combinations(
        asset_names=["stairs", "chair"],
        variant_counts=[2, 1],
        clone_combinations=[
            InclusionSet(assets=["stairs"], weight=2),
            InclusionSet(assets=["chair"], weight=1),
        ],
        return_combination_ids=True,
    )

    assert rows.tolist() == [
        [0, -1],
        [1, -1],
        [-1, 0],
        [0, -1],
        [1, -1],
        [-1, 0],
    ]
    assert combination_ids.tolist() == [0, 0, 1, 0, 0, 1]


def test_make_clone_plan_stores_combination_ids(monkeypatch):
    """The clone plan maps strategy-selected valid rows back to semantic combination IDs."""

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
    valid_set_combination_ids = torch.tensor([4, 9], dtype=torch.long)

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
        valid_set_combination_ids=valid_set_combination_ids,
    )

    assert plan.combination_ids.tolist() == [9, 4, 9]
