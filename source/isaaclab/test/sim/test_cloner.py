# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the retained low-level USD cloning utilities."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import builtins
from unittest.mock import patch

import pytest
import torch

from pxr import Usd

import isaaclab.cloner.usd as usd_cloner
import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan, UsdReplicateContext, grid_positions
from isaaclab.sim import build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


def _plan(sources, destinations, env_ids, mask=None, positions=None):
    if mask is None:
        mask = torch.ones((len(sources), len(env_ids)), dtype=torch.bool)
    if positions is None:
        positions = torch.zeros((len(env_ids), 3))
    return ClonePlan(
        sources=tuple(sources),
        destinations=tuple(destinations),
        clone_mask=mask,
        env_ids=env_ids,
        positions=positions,
        replicate_physics=True,
    )


def _replicate(context, plan, rows):
    plan.context_rows[type(context)] = tuple(rows)
    context.replicate(plan)


@pytest.fixture
def sim():
    with build_simulation_context(device="cpu", dt=0.01, add_lighting=False) as context:
        yield context


def test_usd_replicate_applies_each_row_mask(sim):
    """Each prototype reaches exactly the environments selected by its plan row."""
    for path in ("/World/template/A", "/World/template/B", "/World/envs"):
        sim_utils.create_prim(path, "Xform")
    env_ids = torch.arange(3, dtype=torch.long)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    plan = _plan(
        ["/World/template/A", "/World/template/B"],
        ["/World/envs/env_{}/A", "/World/envs/env_{}/B"],
        env_ids,
        mask,
    )
    _replicate(UsdReplicateContext(sim.stage), plan, (0, 1))

    assert sim.stage.GetPrimAtPath("/World/envs/env_0/A").IsValid()
    assert sim.stage.GetPrimAtPath("/World/envs/env_1/B").IsValid()
    assert sim.stage.GetPrimAtPath("/World/envs/env_2/A").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/envs/env_0/B").IsValid()


def test_usd_context_preserves_asset_offset_below_positioned_env_roots(sim):
    """Environment origins move roots without overwriting a nested asset's local pose."""
    camera_offset = (0.57, -0.8, 0.5)
    env_ids = torch.arange(2, dtype=torch.long)
    positions = grid_positions(2, 3.0)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Camera", "Camera", translation=camera_offset)

    context = UsdReplicateContext(sim.stage)
    _replicate(context, _plan(["/World/envs/env_0"], ["/World/envs/env_{}"], env_ids, positions=positions), (0,))

    for env_id in env_ids.tolist():
        env_prim = sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
        assert tuple(env_prim.GetAttribute("xformOp:translate").Get()) == pytest.approx(positions[env_id].tolist())
        camera = sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Camera")
        assert tuple(camera.GetAttribute("xformOp:translate").Get()) == pytest.approx(camera_offset)


def test_usd_replicate_orders_parent_before_child(sim):
    """Out-of-order rows still author a valid parent and child at every destination."""
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")

    plan = _plan(
        ["/World/template/Parent/Child", "/World/template/Parent"],
        ["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"],
        torch.arange(2),
    )
    _replicate(UsdReplicateContext(sim.stage), plan, (0, 1))

    for env_id in range(2):
        assert sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Parent").IsValid()
        assert sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Parent/Child").IsValid()


def test_usd_replicate_skips_self_copy(sim):
    """The prototype destination is retained without passing identical paths to ``CopySpec``."""
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")
    calls = []
    copy_spec = usd_cloner.Sdf.CopySpec

    def capture(source_layer, source_path, destination_layer, destination_path, *args):
        calls.append((str(source_path), str(destination_path)))
        return copy_spec(source_layer, source_path, destination_layer, destination_path, *args)

    with patch.object(usd_cloner.Sdf, "CopySpec", capture):
        plan = _plan(["/World/envs/env_0"], ["/World/envs/env_{}"], torch.arange(2))
        _replicate(UsdReplicateContext(sim.stage), plan, (0,))

    assert calls and all(source != destination for source, destination in calls)
    assert sim.stage.GetPrimAtPath("/World/envs/env_1/Robot/base_link").IsValid()


def test_fabric_notice_suspension_noops_without_usdrt(monkeypatch):
    """Missing optional ``usdrt`` bindings do not prevent USD-only cloning."""
    from isaaclab.cloner import _fabric_notices

    class _FakeBindings:
        def validate_with(self, fabric_id: int) -> bool:
            raise AssertionError("missing usdrt should prevent fabric-id lookup")

    monkeypatch.setattr(_fabric_notices, "get_bindings", lambda: _FakeBindings())
    real_import = builtins.__import__

    def import_without_usdrt(name, *args, **kwargs):
        if name == "usdrt":
            raise ModuleNotFoundError("No module named 'usdrt'", name="usdrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_usdrt)
    with _fabric_notices.disabled_fabric_change_notifies(Usd.Stage.CreateInMemory()):
        pass


@pytest.mark.parametrize(
    ("parents", "pattern", "expected"),
    [
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta"],
            "/World/rig_0_[^/]*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor"],
        ),
        (
            ["/World/group_a/slot_0", "/World/group_b/slot_1"],
            "/World/group_[^/]*/slot_[^/]*/Sensor",
            ["/World/group_a/slot_0/Sensor", "/World/group_b/slot_1/Sensor"],
        ),
    ],
)
def test_spawn_decorator_respects_segment_wildcards(sim, parents, pattern, expected):
    """Spawner expansion matches path segments without inventing similarly prefixed parents."""
    for path in parents:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(pattern, cfg)

    assert all(sim.stage.GetPrimAtPath(path).IsValid() for path in expected)
    assert not sim.stage.GetPrimAtPath("/World/rig_00/Sensor").IsValid()
