# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for direct USD clone operations (no PhysX dependency)."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import builtins
from unittest.mock import patch

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan, UsdReplicateContext, grid_transforms, usd_replicate
from isaaclab.sim import build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as context:
        yield context


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources only to the selected environments."""
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/template/B", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    for env_id in range(3):
        sim_utils.create_prim(f"/World/envs/env_{env_id}", "Xform")

    mask = np.zeros((2, 3), dtype=np.bool_)
    mask[0, [0, 2]] = True
    mask[1, 1] = True
    usd_replicate(
        sim.stage,
        sources=("/World/template/A", "/World/template/B"),
        destinations=("/World/envs/env_{}/Object/A", "/World/envs/env_{}/Object/B"),
        env_ids=np.arange(3, dtype=np.int64),
        mask=mask,
    )

    assert sim.stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert sim.stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert sim.stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()


def test_usd_replicate_context_consumes_plan(sim):
    """The USD backend consumes the same NumPy plan as every clone backend."""
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    plan = ClonePlan(
        sources=("/World/template/A",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.asarray([[False, True]], dtype=np.bool_),
        env_ids=np.asarray([10, 20], dtype=np.int64),
        positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )

    UsdReplicateContext(sim.stage).replicate(plan)

    assert not sim.stage.GetPrimAtPath("/World/envs/env_10").IsValid()
    prim = sim.stage.GetPrimAtPath("/World/envs/env_20")
    assert prim.IsValid()
    assert tuple(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()) == (4.0, 5.0, 6.0)


def test_usd_replicate_nested_asset_preserves_local_offset_with_positions(sim):
    """Grid positions belong to environment roots, not nested assets."""
    camera_offset = (0.57, -0.8, 0.5)
    env_ids = np.arange(2, dtype=np.int64)
    positions, _ = grid_transforms(2, 3.0)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Camera", "Camera", translation=camera_offset)

    usd_replicate(
        sim.stage,
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        env_ids=env_ids,
        positions=positions,
    )
    usd_replicate(
        sim.stage,
        sources=("/World/envs/env_0/Camera",),
        destinations=("/World/envs/env_{}/Camera",),
        env_ids=env_ids,
        positions=positions,
    )

    for env_id in range(2):
        env_translate = sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}").GetAttribute("xformOp:translate").Get()
        assert tuple(env_translate) == pytest.approx(positions[env_id])
        camera_translate = (
            sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Camera").GetAttribute("xformOp:translate").Get()
        )
        assert tuple(camera_translate) == pytest.approx(camera_offset)


def test_disabled_fabric_change_notifies_noops_when_usdrt_unavailable(monkeypatch):
    """Fabric notice suspension tolerates a Kit build without ``usdrt``."""
    from isaaclab.cloner import _fabric_notices

    class _FakeBindings:
        def validate_with(self, fabric_id: int) -> bool:
            raise AssertionError("missing usdrt should prevent fabric-id lookup")

    monkeypatch.setattr(_fabric_notices, "get_bindings", lambda: _FakeBindings())
    real_import = builtins.__import__

    def _import_without_usdrt(name, *args, **kwargs):
        if name == "usdrt":
            raise ModuleNotFoundError("No module named 'usdrt'", name="usdrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_usdrt)
    with _fabric_notices.disabled_fabric_change_notifies(Usd.Stage.CreateInMemory()):
        pass


def test_usd_replicate_depth_order_parent_child(sim):
    """Parents are authored before children even when rows arrive out of order."""
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    usd_replicate(
        sim.stage,
        sources=("/World/template/Parent/Child", "/World/template/Parent"),
        destinations=("/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"),
        env_ids=np.asarray([0, 1], dtype=np.int64),
    )

    for env_id in range(2):
        assert sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Parent").IsValid()
        assert sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Parent/Child").IsValid()


def test_usd_replicate_self_copy_skips_copy_spec(sim):
    """A source environment is not copied onto itself."""
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")
    copy_calls: list[tuple[str, str]] = []
    real_copy_spec = Sdf.CopySpec

    def capturing_copy_spec(src_layer, src_path, dst_layer, dst_path, *args):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy_spec(src_layer, src_path, dst_layer, dst_path, *args)

    with patch.object(Sdf, "CopySpec", capturing_copy_spec):
        usd_replicate(
            sim.stage,
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            env_ids=np.asarray([0, 1], dtype=np.int64),
        )

    assert all(source != destination for source, destination in copy_calls)
    assert any(destination == "/World/envs/env_1" for _, destination in copy_calls)


@pytest.mark.parametrize(
    "parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr",
    [
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta", "/World/rig_0_gamma"],
            "/World/rig_0_[^/]*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor", "/World/rig_0_gamma/Sensor"],
            "/World/rig_00/Sensor",
            "/World/rig_0_[^/]*",
        ),
        (
            ["/World/group_a/slot_0", "/World/group_a/slot_1", "/World/group_b/slot_0"],
            "/World/group_[^/]*/slot_[^/]*/Sensor",
            [
                "/World/group_a/slot_0/Sensor",
                "/World/group_a/slot_1/Sensor",
                "/World/group_b/slot_0/Sensor",
            ],
            "/World/group_0/slot_0/Sensor",
            "/World/group_[^/]*/slot_[^/]*",
        ),
        (
            ["/World/template/Object"],
            "/World/template/Object/proto_.*",
            ["/World/template/Object/proto_0"],
            "/World/template/Object0/proto_0",
            "/World/template/Object",
        ),
    ],
)
def test_clone_decorator_wildcard_patterns(
    sim, parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr
):
    """The clone decorator preserves wildcard segment boundaries."""
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    for child_path in expected_child_paths:
        assert sim.stage.GetPrimAtPath(child_path).IsValid()
    assert not sim.stage.GetPrimAtPath(bad_path).IsValid()
    assert len(sim_utils.find_matching_prims(match_expr)) == len(parent_paths)
