# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`isaaclab.cloner.UsdReplicateContext` + per-cfg ``usd_replicate``."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from dataclasses import dataclass

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner import UsdReplicateContext, get_replicate_ctx, replicate
from isaaclab.cloner.usd_replicator import usd_replicate
from isaaclab.layout import make_stage_layout
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.isaacsim_ci


@dataclass
class _SrcCfg:
    """Identity-keyed marker used as a source registry entry."""

    name: str


@pytest.fixture
def sim():
    """Provide a fresh simulation context for each test on CPU."""
    with build_simulation_context(device="cpu", dt=0.01, add_lighting=False) as sim:
        yield sim


def _identity_pose(num_envs: int) -> torch.Tensor:
    pose = torch.zeros((num_envs, 7), dtype=torch.float32)
    pose[:, 6] = 1.0
    return pose


def test_usd_replicate_single_cfg_three_envs(sim):
    """One per-env cfg replicates from src env to every other env, including self-copy."""
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(3):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")

    cfg = _SrcCfg("robot")
    layout = make_stage_layout(
        sources=[cfg],
        destinations=["/World/envs/env_{}/Robot"],
        sources_per_world=[[], [0], [0], [0]],
        env_pose=_identity_pose(3),
    )
    sim.set_stage_layout(layout)

    usd_replicate(cfg, layout, cfg_idx=0)
    replicate()

    stage = sim_utils.get_current_stage()
    for i in range(3):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/base_link").IsValid()


def test_usd_replicate_skips_shared_scope_cfg(sim):
    """A cfg that lives only in shared scope must not author any CopySpecs."""
    from unittest.mock import patch

    import isaaclab.cloner.usd_replicator as _ur

    cfg = _SrcCfg("ground")
    layout = make_stage_layout(
        sources=[cfg],
        destinations=["/World/Ground"],
        sources_per_world=[[0], [], []],
        env_pose=_identity_pose(2),
    )
    sim.set_stage_layout(layout)

    captured: list[tuple[str, str]] = []
    real_create = _ur.Sdf.CreatePrimInLayer

    def capturing_create(layer, path):
        captured.append(("create", str(path)))
        return real_create(layer, path)

    with patch.object(_ur.Sdf, "CreatePrimInLayer", capturing_create):
        usd_replicate(cfg, layout, cfg_idx=0)
        replicate()

    assert captured == []


def test_usd_replicate_self_copy_skips_copy_spec(sim):
    """Per-env replicate queues self-copy for the source env, but ``replicate()`` must skip it."""
    from unittest.mock import patch

    import isaaclab.cloner.usd_replicator as _ur

    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    cfg = _SrcCfg("robot")
    layout = make_stage_layout(
        sources=[cfg],
        destinations=["/World/envs/env_{}/Robot"],
        sources_per_world=[[], [0], [0]],
        env_pose=_identity_pose(2),
    )
    sim.set_stage_layout(layout)

    copy_calls: list[tuple[str, str]] = []
    real_copy = _ur.Sdf.CopySpec

    def capturing_copy(src_layer, src_path, dst_layer, dst_path):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy(src_layer, src_path, dst_layer, dst_path)

    with patch.object(_ur.Sdf, "CopySpec", capturing_copy):
        usd_replicate(cfg, layout, cfg_idx=0)
        replicate()

    assert all(src != dst for src, dst in copy_calls), f"Self-copy detected: {copy_calls}"
    assert any(dst == "/World/envs/env_1/Robot" for _, dst in copy_calls)


def test_usd_replicate_multiple_cfgs_share_one_context(sim):
    """Two cfgs queue into the same UsdReplicateContext; one drain authors all of them."""
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(2):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")
    sim_utils.create_prim("/World/envs/env_0/A", "Xform")
    sim_utils.create_prim("/World/envs/env_0/B", "Xform")

    cfg_a = _SrcCfg("a")
    cfg_b = _SrcCfg("b")
    layout = make_stage_layout(
        sources=[cfg_a, cfg_b],
        destinations=["/World/envs/env_{}/A", "/World/envs/env_{}/B"],
        sources_per_world=[[], [0, 1], [0, 1]],
        env_pose=_identity_pose(2),
    )
    sim.set_stage_layout(layout)

    ctx_before = get_replicate_ctx(UsdReplicateContext)
    usd_replicate(cfg_a, layout, cfg_idx=0)
    usd_replicate(cfg_b, layout, cfg_idx=1)
    ctx_after = get_replicate_ctx(UsdReplicateContext)
    assert ctx_before is ctx_after  # singleton across queue calls

    replicate()

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/A").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/B").IsValid()


def test_usd_replicate_depth_order_parent_lands_before_child(sim):
    """Independent cfgs at parent and child paths replicate in path-depth order."""
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(2):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Parent", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Parent/Child", "Xform")

    cfg_parent = _SrcCfg("parent")
    cfg_child = _SrcCfg("child")
    layout = make_stage_layout(
        sources=[cfg_parent, cfg_child],
        destinations=["/World/envs/env_{}/Parent", "/World/envs/env_{}/Parent/Child"],
        sources_per_world=[[], [0, 1], [0, 1]],
        env_pose=_identity_pose(2),
    )
    sim.set_stage_layout(layout)

    # Queue the deeper cfg first; depth sort inside replicate() must still author parent first.
    usd_replicate(cfg_child, layout, cfg_idx=1)
    usd_replicate(cfg_parent, layout, cfg_idx=0)
    replicate()

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent/Child").IsValid()


def test_usd_replicate_drain_clears_registry_for_subsequent_scene(sim):
    """``replicate()`` must clear the registry so a fresh scene re-constructs the context."""
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    cfg = _SrcCfg("robot")
    layout = make_stage_layout(
        sources=[cfg],
        destinations=["/World/envs/env_{}/Robot"],
        sources_per_world=[[], [0], [0]],
        env_pose=_identity_pose(2),
    )
    sim.set_stage_layout(layout)

    first = get_replicate_ctx(UsdReplicateContext)
    usd_replicate(cfg, layout, cfg_idx=0)
    replicate()

    # After drain a fresh ctx is built next time get_replicate_ctx is called.
    second = get_replicate_ctx(UsdReplicateContext)
    assert second is not first
    # No queued specs leaked into the new context.
    assert second._copy_specs == []  # type: ignore[attr-defined]
    second.replicate()  # drain to keep the registry clean for the next test
