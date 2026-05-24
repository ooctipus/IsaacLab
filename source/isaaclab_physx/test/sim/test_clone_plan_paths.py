# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_physx.cloner.clone_plan_paths import (
    expand_clone_plan_child_paths,
    expand_clone_plan_path,
    expand_clone_plan_paths,
    resolve_clone_plan_source_paths,
)

from pxr import Usd, UsdPhysics

from isaaclab.cloner.clone_plan import ClonePlan


def test_expand_clone_plan_path_homogeneous_env_root():
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert expand_clone_plan_path("/World/envs/env_.*/Robot/base", plan) == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_1/Robot/base",
        "/World/envs/env_2/Robot/base",
        "/World/envs/env_3/Robot/base",
    ]


def test_expand_clone_plan_paths_are_env_major():
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    assert expand_clone_plan_paths(("/World/envs/env_.*/Robot/LF_FOOT", "/World/envs/env_.*/Robot/RF_FOOT"), plan) == [
        "/World/envs/env_0/Robot/LF_FOOT",
        "/World/envs/env_0/Robot/RF_FOOT",
        "/World/envs/env_1/Robot/LF_FOOT",
        "/World/envs/env_1/Robot/RF_FOOT",
    ]


def test_expand_clone_plan_path_heterogeneous_rows():
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool),
    )

    assert expand_clone_plan_path("/World/envs/env_.*/Object/body", plan) == [
        "/World/envs/env_0/Object/body",
        "/World/envs/env_1/Object/body",
        "/World/envs/env_2/Object/body",
        "/World/envs/env_3/Object/body",
    ]


def test_expand_clone_plan_child_paths_walks_sources_env_major():
    stage = Usd.Stage.CreateInMemory()
    for path in (
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/foot",
        "/World/envs/env_1/Robot/body",
    ):
        prim = stage.DefinePrim(path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)

    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Robot"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"),
        clone_mask=torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool),
    )

    assert expand_clone_plan_child_paths(
        lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI), clone_plan=plan, stage=stage
    ) == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/foot",
        "/World/envs/env_1/Robot/body",
        "/World/envs/env_2/Robot/base",
        "/World/envs/env_2/Robot/foot",
        "/World/envs/env_3/Robot/body",
    ]


def test_expand_clone_plan_path_with_nested_destination_postfix():
    plan = ClonePlan(
        sources=("/World/templates/RobotA",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    assert expand_clone_plan_path("/World/envs/env_.*/Robot/base", plan) == [
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_1/Robot/base",
    ]


def test_expand_clone_plan_path_outside_plan_returns_none():
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    assert expand_clone_plan_path("/World/Other/Robot", plan) is None


def test_resolve_clone_plan_source_paths_skips_inactive_rows():
    plan = ClonePlan(
        sources=("/World/templates/RobotA", "/World/templates/RobotB"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"),
        clone_mask=torch.tensor([[True, False], [False, False]], dtype=torch.bool),
    )

    assert resolve_clone_plan_source_paths("/World/envs/env_.*/Robot/base", plan) == ["/World/templates/RobotA/base"]
