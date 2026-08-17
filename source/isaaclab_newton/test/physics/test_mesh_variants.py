# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reset-time Newton mesh variants."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.envs.mdp import randomize_rigid_body_mesh
from isaaclab_newton.physics.mesh_variants import (
    _disable_mesh_variant_resources,
    _prepare_mesh_variant_resources,
    build_mesh_variant_sets,
)
from newton import Mesh, ModelBuilder

from isaaclab.managers import SceneEntityCfg


def _variant_builder(source: str, half_extent: float, body_suffix: str = "/body") -> ModelBuilder:
    vertices = np.asarray(
        [
            (-half_extent, -half_extent, -half_extent),
            (half_extent, -half_extent, -half_extent),
            (0.0, half_extent, -half_extent),
            (0.0, 0.0, half_extent),
        ],
        dtype=np.float32,
    )
    faces = np.asarray((0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3), dtype=np.int32)
    builder = ModelBuilder()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        label=source + body_suffix,
        lock_inertia=True,
    )
    builder.add_shape_mesh(body, mesh=Mesh(vertices, faces))
    joint = builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    return builder


def _clone_plan(cfg, active_mask: torch.Tensor):
    return SimpleNamespace(
        sources=("/World/envs/env_0/object", "/World/envs/env_1/object"),
        destinations=("/World/envs/env_{}/object",) * 2,
        clone_mask=active_mask,
        env_ids=torch.arange(2),
        cfg_rows={id(cfg): (0, 1)},
    )


def test_mesh_variant_resources_do_not_add_rigid_bodies() -> None:
    """Compile inactive meshes as global resources, not parked objects."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large_path = "/World/IsaacLabPrototypes/group_0/variant_1"
    large = _variant_builder(large_path, 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(small)
    body_count = builder.body_count
    plan = SimpleNamespace(
        sources=("/World/envs/env_0/object", large_path),
        clone_mask=torch.tensor(((True, True), (False, False))),
        cfg_rows={id(cfg): (0, 1)},
    )

    resources = _prepare_mesh_variant_resources(
        {"object": cfg}, builder, dict(zip(plan.sources, (small, large), strict=True)), plan
    )

    assert builder.body_count == body_count
    assert tuple(len(shapes) for shapes in resources["object"]) == (1, 1)
    model = builder.finalize(device="cpu")
    _disable_mesh_variant_resources(model, resources)
    resource_shapes = np.asarray(resources["object"]).flatten()
    np.testing.assert_array_equal(model.shape_flags.numpy()[resource_shapes], 0)


def test_build_mesh_variant_sets_keeps_source_builders() -> None:
    """Pass source builders directly to Newton and map only target shape indices."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(large)
    model = builder.finalize(device="cpu")
    plan = _clone_plan(cfg, torch.eye(2, dtype=torch.bool))

    definitions, render_sets = build_mesh_variant_sets(
        {"object": cfg},
        model,
        dict(zip(plan.sources, (small, large), strict=True)),
        plan,
    )
    (definition,) = definitions

    assert definition.variant_builders == (small, large)
    np.testing.assert_array_equal(definition.shape_indices, ((0,), (1,)))
    np.testing.assert_array_equal(definition.initial_variant_ids, (0, 1))
    assert not hasattr(definition, "body_indices")
    assert render_sets == ()

    _, render_sets = build_mesh_variant_sets(
        {"object": cfg},
        model,
        dict(zip(plan.sources, (small, large), strict=True)),
        plan,
        {},
    )
    assert render_sets == ()


def test_build_mesh_variant_sets_forwards_inertia_offset() -> None:
    """Let Newton apply inertia offsets after source validation."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.25)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(large)
    model = builder.finalize(device="cpu")
    plan = _clone_plan(cfg, torch.eye(2, dtype=torch.bool))

    (definition,), _ = build_mesh_variant_sets(
        {"object": cfg},
        model,
        dict(zip(plan.sources, (small, large), strict=True)),
        plan,
    )

    assert definition.variant_builders == (small, large)
    assert definition.inertia_diagonal_offset == 0.25
    for source in (small, large):
        np.testing.assert_allclose(np.asarray(source.body_inertia[0]).reshape(3, 3), np.eye(3))


def test_build_mesh_variant_sets_rejects_unbuilt_source() -> None:
    """Reject a candidate that was not instantiated in any initial world."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(small)
    model = builder.finalize(device="cpu")
    plan = _clone_plan(cfg, torch.tensor(((True, True), (False, False))))

    with pytest.raises(ValueError, match="every source"):
        build_mesh_variant_sets(
            {"object": cfg},
            model,
            dict(zip(plan.sources, (small, large), strict=True)),
            plan,
        )


def test_build_mesh_variant_sets_accepts_different_body_paths() -> None:
    """Resolve each initial body through its own source-relative path."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2, "/other_body")
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(large)
    model = builder.finalize(device="cpu")
    plan = _clone_plan(cfg, torch.eye(2, dtype=torch.bool))

    definitions, _ = build_mesh_variant_sets(
        {"object": cfg},
        model,
        dict(zip(plan.sources, (small, large), strict=True)),
        plan,
    )
    (definition,) = definitions

    np.testing.assert_array_equal(definition.shape_indices, ((0,), (1,)))


def test_build_mesh_variant_sets_keeps_render_shapes_outside_model() -> None:
    """Retain render prototypes per variant and map them to the live body."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(large)
    model = builder.finalize(device="cpu")
    plan = _clone_plan(cfg, torch.eye(2, dtype=torch.bool))
    sources = dict(zip(plan.sources, (small, large), strict=True))

    (definition,), (render_set,) = build_mesh_variant_sets({"object": cfg}, model, sources, plan, sources)

    np.testing.assert_array_equal(render_set.body_indices.numpy(), (0, 1))
    assert render_set.model_shape_indices == (0, 1)
    assert tuple(len(shapes) for shapes in render_set.visual_variants) == (1, 1)
    assert tuple(len(shapes) for shapes in render_set.collision_variants) == (1, 1)
    assert render_set.visual_variants[0][0].source is small.shape_source[0]
    assert render_set.collision_variants[1][0].source is definition.variant_builders[1].shape_source[0]
    assert render_set.collision_variants[1][0].scale == tuple(definition.variant_builders[1].shape_scale[0])


def test_randomize_rigid_body_mesh_samples_spawn_order() -> None:
    """Sample one valid integer variant for every selected environment."""

    class Asset:
        device = "cpu"
        num_instances = 8
        num_mesh_variants = 3
        written = None

        def write_mesh_variant_to_sim(self, variant_ids, env_ids) -> None:
            self.written = variant_ids, env_ids

    asset = Asset()
    env = SimpleNamespace(scene={"object": asset})
    env_ids = torch.tensor((1, 4, 6), dtype=torch.int32)
    randomize_rigid_body_mesh(env, env_ids, SceneEntityCfg("object"))

    variant_ids, written_env_ids = asset.written
    assert variant_ids.dtype == torch.int32
    assert variant_ids.shape == env_ids.shape
    assert torch.all((variant_ids >= 0) & (variant_ids < asset.num_mesh_variants))
    assert written_env_ids is env_ids
