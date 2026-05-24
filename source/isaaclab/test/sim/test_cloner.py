# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD cloner utilities (no PhysX dependency)."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan, make_clone_plan, sequential, usd_replicate
from isaaclab.cloner.cloner_utils import (
    ascend_source_prims,
    descend_source_prims,
    expand_clone_plan_paths,
    iter_clone_plan_matches,
    source_prim,
)
from isaaclab.sim import build_simulation_context
from isaaclab.utils.math import combine_frame_transforms

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources to selected envs and author translate ops from positions."""
    # Prepare sources under /World/template
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/template/B", "Xform")

    # Prepare destination env namespaces
    num_envs = 3
    env_ids = torch.arange(num_envs, dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    # Map A -> env 0 and 2; B -> env 1 only
    mask = torch.zeros((2, num_envs), dtype=torch.bool)
    mask[0, [0, 2]] = True
    mask[1, [1]] = True

    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A", "/World/template/B"],
        destinations=["/World/envs/env_{}/Object/A", "/World/envs/env_{}/Object/B"],
        env_ids=env_ids,
        mask=mask,
    )

    # Validate replication and translate op
    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()

    # Check xformOp:translate authored for env_2/A
    prim = stage.GetPrimAtPath("/World/envs/env_2/Object/A")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert any(op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops)


def test_usd_replicate_depth_order_parent_child(sim):
    """Replicate parent and child when provided out of order; parent should exist before child."""
    # Prepare sources
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/Parent", "Xform")
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")

    # Destinations (single env)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    # Provide child first, then parent; depth sort should handle this
    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/Parent/Child", "/World/template/Parent"],
        destinations=["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"],
        env_ids=env_ids,
    )

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent/Child").IsValid()


def test_usd_replicate_self_copy_skips_copy_spec(sim):
    """usd_replicate must not call Sdf.CopySpec when source and destination paths are identical.

    Sdf.CopySpec(src, src) is a no-op in the current USD version so it does not corrupt children,
    but the call is still wasteful. The guard ensures it is skipped entirely. This test mocks
    Sdf.CopySpec to verify it is called exactly once (for env_1) and never for the self case (env_0).
    """
    from unittest.mock import patch

    import isaaclab.cloner.cloner_utils as _cloner_mod

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    copy_calls: list[tuple[str, str]] = []
    real_copy_spec = _cloner_mod.Sdf.CopySpec

    def capturing_copy_spec(src_layer, src_path, dst_layer, dst_path):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy_spec(src_layer, src_path, dst_layer, dst_path)

    with patch.object(_cloner_mod.Sdf, "CopySpec", capturing_copy_spec):
        usd_replicate(
            stage,
            sources=["/World/envs/env_0"],
            destinations=["/World/envs/env_{}"],
            env_ids=torch.tensor([0, 1], dtype=torch.long),
            mask=torch.ones((1, 2), dtype=torch.bool),
        )

    # CopySpec must be called for env_1 but never for env_0 (self-copy)
    assert all(src != dst for src, dst in copy_calls), f"Self-copy detected in CopySpec calls: {copy_calls}"
    assert any(dst == "/World/envs/env_1" for _, dst in copy_calls), "CopySpec was not called for env_1"


@pytest.mark.parametrize(
    "parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr",
    [
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta", "/World/rig_0_gamma"],
            "/World/rig_0_.*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor", "/World/rig_0_gamma/Sensor"],
            "/World/rig_00/Sensor",
            "/World/rig_0_.*",
        ),
        (
            [
                "/World/group_a/slot_0",
                "/World/group_a/slot_1",
                "/World/group_b/slot_0",
                "/World/group_b/slot_1",
            ],
            "/World/group_.*/slot_.*/Sensor",
            [
                "/World/group_a/slot_0/Sensor",
                "/World/group_a/slot_1/Sensor",
                "/World/group_b/slot_0/Sensor",
                "/World/group_b/slot_1/Sensor",
            ],
            "/World/group_0/slot_0/Sensor",
            "/World/group_.*/slot_.*",
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
    """The @clone decorator handles two distinct wildcard patterns correctly.

    Case A – ``.*`` in root_path (parent is a regex): the child prim is spawned at
    ``source_prim_paths[0]`` as a prototype and then copied to every other matching
    parent via ``Sdf.CopySpec``, so **all** parents end up with the child.  The old
    ``prim_path.replace(".*", "0")`` approach created spurious intermediate prims
    that inflated ``find_matching_prims`` counts and broke tiled-camera initialization.

    Case B – ``.*`` only in asset_path (leaf): no parent regex, so
    ``source_prim_paths == [root_path]`` (one entry, no copy step).  Replacing
    ``".*"`` → ``"0"`` in the asset name gives the intended prototype name
    (e.g. ``proto_asset_0``) under the single real parent.
    """
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    stage = sim_utils.get_current_stage()

    # Every expected child path must exist
    for child_path in expected_child_paths:
        assert stage.GetPrimAtPath(child_path).IsValid(), (
            f"Prim was not spawned at '{child_path}'. The @clone decorator may have used the wrong spawn path."
        )

    # The spurious path from the old replace(".*", "0") must NOT exist
    assert not stage.GetPrimAtPath(bad_path).IsValid(), (
        f"Spurious prim found at '{bad_path}'. "
        "The @clone decorator incorrectly derived the spawn path by replacing '.*' with '0'."
    )

    # find_matching_prims must see exactly the original parents — no spurious extras
    all_matching = sim_utils.find_matching_prims(match_expr)
    assert len(all_matching) == len(parent_paths), (
        f"Expected {len(parent_paths)} matching prims, got {len(all_matching)}. "
        "Spurious parent prims were likely created by the @clone decorator."
    )


def test_make_clone_plan_returns_flat_source_rows(sim):
    """make_clone_plan exposes the flat source-to-env mask used by scene cloning."""
    sources, destinations, clone_mask = make_clone_plan(
        [["/World/envs/env_0/Object", "/World/envs/env_1/Object"]],
        ["/World/envs/env_{}/Object"],
        num_clones=4,
        clone_strategy=sequential,
        device=sim.cfg.device,
    )

    assert sources == ("/World/envs/env_0/Object", "/World/envs/env_1/Object")
    assert destinations == ("/World/envs/env_{}/Object", "/World/envs/env_{}/Object")
    assert clone_mask.shape == (2, 4)
    assert clone_mask.dtype == torch.bool
    assert torch.all(clone_mask.sum(dim=0) == 1)
    actual_source_idx = clone_mask.to(torch.int).argmax(dim=0).cpu()
    assert torch.equal(actual_source_idx, torch.tensor([0, 1, 0, 1]))


def test_iter_clone_plan_matches(sim):
    """ClonePlan entries can be matched by destination path expression."""
    sources, destinations, clone_mask = make_clone_plan(
        [["/World/envs/env_0/Object", "/World/envs/env_1/Object"]],
        ["/World/envs/env_{}/Object"],
        num_clones=4,
        clone_strategy=sequential,
        device=sim.cfg.device,
    )
    plan = ClonePlan(sources=sources, destinations=destinations, clone_mask=clone_mask)

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_0/Object/Body/Camera",
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            (0, 2),
        ),
        (
            "/World/envs/env_1/Object/Body/Camera",
            "/World/envs/env_1/Object",
            "/World/envs/env_{}/Object",
            (1, 3),
        ),
    ]

    plan = ClonePlan(
        sources=("/World/envs/env_3/Object",),
        destinations=("/World/envs/env_{}/Object",),
        clone_mask=torch.tensor([[False, False, True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_3/Object/Body/Camera",
            "/World/envs/env_3/Object",
            "/World/envs/env_{}/Object",
            (2, 3),
        )
    ]

    plan = ClonePlan(
        sources=("/World/source/Object",),
        destinations=("/World/scenes/{}/Object",),
        clone_mask=torch.tensor([[True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/scenes/.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/source/Object/Body/Camera",
            "/World/source/Object",
            "/World/scenes/{}/Object",
            (0, 1),
        )
    ]

    plan = ClonePlan(
        sources=("/World/source",),
        destinations=("/World/scenes/{}",),
        clone_mask=torch.tensor([[True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/scenes/.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/source/Object/Body/Camera",
            "/World/source",
            "/World/scenes/{}",
            (0, 1),
        )
    ]

    plan = ClonePlan(
        sources=("/World/envs/env_0", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, True], [True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    # Without nearest-template filtering, both plan entries that cover the queried path are yielded.
    assert matches == [
        (
            "/World/envs/env_0/Object/Body/Camera",
            "/World/envs/env_0",
            "/World/envs/env_{}",
            (0, 1),
        ),
        (
            "/World/envs/env_0/Object/Body/Camera",
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            (0, 1),
        ),
    ]


def _make_simple_plan(device: str, num_envs: int = 4) -> ClonePlan:
    """Construct a homogeneous ClonePlan against ``/World/envs/env_{}/Robot``."""
    return ClonePlan(
        sources=("/World/source/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool, device=device),
    )


def test_expand_clone_plan_paths_basic(sim):
    """``expand_clone_plan_paths`` returns one concrete path per env."""
    plan = _make_simple_plan(sim.cfg.device, num_envs=3)
    paths = expand_clone_plan_paths(plan, "/World/envs/env_.*/Robot/base")
    assert paths == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_1/Robot/base",
        "/World/envs/env_2/Robot/base",
    ]


def test_expand_clone_plan_paths_partial_coverage_returns_none(sim):
    """Variant clones leave envs uncovered → ``None`` entries (not an error)."""
    # Two source variants split across 4 envs.
    mask = torch.zeros((2, 4), dtype=torch.bool, device=sim.cfg.device)
    mask[0, [0, 1]] = True
    mask[1, [2]] = True  # env 3 deliberately uncovered
    plan = ClonePlan(
        sources=("/World/source/RobotA", "/World/source/RobotB"),
        destinations=("/World/envs/env_{}/RobotA", "/World/envs/env_{}/RobotB"),
        clone_mask=mask,
    )
    paths = expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotA/base")
    assert paths == [
        "/World/envs/env_0/RobotA/base",
        "/World/envs/env_1/RobotA/base",
        None,
        None,
    ]


def test_expand_clone_plan_paths_double_coverage_raises(sim):
    """Plan corruption: two entries claim the same env → raises."""
    plan = ClonePlan(
        sources=("/World/envs/env_0", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, True], [True, True]], device=sim.cfg.device),
    )
    with pytest.raises(RuntimeError, match="matched twice"):
        expand_clone_plan_paths(plan, "/World/envs/env_.*/Object/Body/Camera")


def test_source_prim_returns_resolved_prim(sim):
    """``source_prim`` resolves the source-side ``Usd.Prim`` for ``prim_expr``."""
    sim_utils.create_prim("/World/source", "Xform")
    sim_utils.create_prim("/World/source/Robot", "Xform")
    sim_utils.create_prim("/World/source/Robot/base", "Xform")

    plan = _make_simple_plan(sim.cfg.device, num_envs=2)
    prim, source_root, destination, env_ids = source_prim(plan, "/World/envs/env_.*/Robot/base")
    assert prim.GetPath().pathString == "/World/source/Robot/base"
    assert source_root == "/World/source/Robot"
    assert destination == "/World/envs/env_{}/Robot"
    assert env_ids == (0, 1)


def test_descend_source_prims_filters_descendants(sim):
    """``descend_source_prims`` walks descendants (incl. self) and applies the predicate."""
    sim_utils.create_prim("/World/source", "Xform")
    sim_utils.create_prim("/World/source/Robot", "Xform")
    sim_utils.create_prim("/World/source/Robot/base", "Xform")
    sim_utils.create_prim("/World/source/Robot/eef", "Xform")

    plan = _make_simple_plan(sim.cfg.device, num_envs=1)
    matches = descend_source_prims(plan, "/World/envs/env_.*/Robot", lambda p: p.GetPath().name == "eef")
    assert len(matches) == 1
    prim, _, _, _ = matches[0]
    assert prim.GetPath().pathString == "/World/source/Robot/eef"


def test_ascend_source_prims_filters_ancestors(sim):
    """``ascend_source_prims`` walks ancestors (incl. self) up to ``source_root``."""
    sim_utils.create_prim("/World/source", "Xform")
    sim_utils.create_prim("/World/source/Robot", "Xform")
    sim_utils.create_prim("/World/source/Robot/base", "Xform")
    sim_utils.create_prim("/World/source/Robot/base/imu", "Xform")

    plan = _make_simple_plan(sim.cfg.device, num_envs=1)
    matches = ascend_source_prims(plan, "/World/envs/env_.*/Robot/base/imu", lambda p: p.GetPath().name == "base")
    assert len(matches) == 1
    prim, _, _, _ = matches[0]
    assert prim.GetPath().pathString == "/World/source/Robot/base"


def test_ascend_source_prims_stops_at_source_root(sim):
    """Walk does not escape ``source_root`` -- returns ``[]`` when only the env scope matches."""
    sim_utils.create_prim("/World/source", "Xform")
    sim_utils.create_prim("/World/source/Robot", "Xform")
    sim_utils.create_prim("/World/source/Robot/base", "Xform")

    plan = _make_simple_plan(sim.cfg.device, num_envs=1)
    matches = ascend_source_prims(
        plan, "/World/envs/env_.*/Robot/base", lambda p: p.GetPath().pathString == "/World/source"
    )
    assert matches == []


def test_source_prim_no_match_returns_none_tuple(sim):
    """No plan entry covers the expression → ``(None, None, None, None)``."""
    plan = _make_simple_plan(sim.cfg.device, num_envs=1)
    out = source_prim(plan, "/World/other/path")
    assert out == (None, None, None, None)


def test_winA_compose_world_pose_against_env_pose(sim):
    """Win A: ``combine_frame_transforms(env_pose, rel_pose)`` matches the per-env reference."""
    device = sim.cfg.device
    env_pose = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.5, 0.0, 0.0, 0.7071068, 0.7071068],  # 90° about +Z
        ],
        device=device,
    )
    rel_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
    rel_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    world_pos, world_quat = combine_frame_transforms(
        env_pose[:, :3], env_pose[:, 3:], rel_pos.expand(3, 3), rel_quat.expand(3, 4)
    )
    expected = torch.tensor([[1.0, 0.0, 0.0], [2.0, 2.0, 3.0], [-1.0, 1.0, 0.5]], device=device)
    assert torch.allclose(world_pos, expected, atol=1e-4)
    assert torch.allclose(world_quat, env_pose[:, 3:], atol=1e-5)


def test_winB_no_destination_prims_on_stage(sim):
    """Win B: helpers resolve queries from source prims, never touching destinations."""
    sim_utils.create_prim("/World/source", "Xform")
    sim_utils.create_prim("/World/source/Robot", "Xform")
    sim_utils.create_prim("/World/source/Robot/base", "Xform")
    plan = _make_simple_plan(sim.cfg.device, num_envs=4)
    stage = sim_utils.get_current_stage()

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Robot/base"))
    assert matches == [("/World/source/Robot/base", "/World/source/Robot", "/World/envs/env_{}/Robot", (0, 1, 2, 3))]

    paths = expand_clone_plan_paths(plan, "/World/envs/env_.*/Robot/base")
    assert paths == [f"/World/envs/env_{i}/Robot/base" for i in range(4)]
    assert all(not stage.GetPrimAtPath(p).IsValid() for p in paths)

    prim, source_root, destination, env_ids = source_prim(plan, "/World/envs/env_.*/Robot/base")
    assert prim.GetPath().pathString == "/World/source/Robot/base"
    assert (source_root, destination, env_ids) == ("/World/source/Robot", "/World/envs/env_{}/Robot", (0, 1, 2, 3))


def test_winB_heterogeneous_partition(sim):
    """Heterogeneous variant clones: per-env counts differ across sources."""
    device = sim.cfg.device
    mask = torch.zeros((3, 4), dtype=torch.bool, device=device)
    mask[0, [0, 1]] = True
    mask[1, [2]] = True
    mask[2, [3]] = True
    plan = ClonePlan(
        sources=("/World/source/Robot",) * 3,
        destinations=("/World/envs/env_{}/RobotA", "/World/envs/env_{}/RobotB", "/World/envs/env_{}/RobotC"),
        clone_mask=mask,
    )
    assert expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotA/base") == [
        "/World/envs/env_0/RobotA/base",
        "/World/envs/env_1/RobotA/base",
        None,
        None,
    ]
    assert expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotB/base") == [
        None,
        None,
        "/World/envs/env_2/RobotB/base",
        None,
    ]
    assert expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotC/base") == [
        None,
        None,
        None,
        "/World/envs/env_3/RobotC/base",
    ]
