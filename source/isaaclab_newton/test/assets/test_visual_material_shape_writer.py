# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scattering shared-material color writes into ``model.shape_color``.

Row resolution consumes the shape → bound-material map captured when Newton imports the source
stage, plus the published clone plan — no prim traversal or binding resolution is involved.
"""

from types import SimpleNamespace

import isaaclab_newton.assets.visual_material.shape_writer as shape_writer_module
import pytest
import torch
import warp as wp
from isaaclab_newton.assets.visual_material.shape_writer import VisualMaterialShapeWriter

from isaaclab.cloner import ClonePlan

_BUCKET_A = "/World/Materials/bucket_a"
_BUCKET_B = "/World/Materials/bucket_b"


def _expected_srgb(colors: torch.Tensor) -> torch.Tensor:
    """Reference sRGB transfer function, mirrored independently of the production kernel."""
    return torch.where(colors <= 0.0031308, 12.92 * colors, 1.055 * torch.pow(colors, 1.0 / 2.4) - 0.055)


_SHAPE_LABELS = [
    "/World/envs/env_0/cube_a/geometry/mesh",  # source, binds bucket_a
    "/World/envs/env_0/cube_b/geometry/mesh",  # source, binds bucket_b
    "/World/envs/env_1/cube_a/geometry/mesh",  # clone destination, carried back to source via the plan
    "/World/envs/env_1/cube_b/geometry/mesh",  # clone destination, carried back to source via the plan
    "/World/ground",  # unbound prim: no material row
    "not a prim path",  # procedural shape label: skipped
]

# shape → bound-material map as the Newton import walk captures it, in source space
_SOURCE_SHAPE_MATERIALS = {
    "/World/envs/env_0/cube_a/geometry/mesh": _BUCKET_A,
    "/World/envs/env_0/cube_b/geometry/mesh": _BUCKET_B,
}


def _clone_plan() -> ClonePlan:
    return ClonePlan(
        sources=("/World/envs/env_0/cube_a", "/World/envs/env_0/cube_b"),
        destinations=("/World/envs/env_{}/cube_a", "/World/envs/env_{}/cube_b"),
        clone_mask=torch.ones(2, 2, dtype=torch.bool),
    )


@pytest.fixture
def writer(monkeypatch):
    fake_sim = SimpleNamespace(get_clone_plan=lambda: _clone_plan())
    monkeypatch.setattr(shape_writer_module, "SimulationContext", SimpleNamespace(instance=lambda: fake_sim))

    model = SimpleNamespace(
        shape_label=list(_SHAPE_LABELS),
        shape_color=wp.array([wp.vec3(0.5, 0.5, 0.5)] * len(_SHAPE_LABELS), dtype=wp.vec3, device="cpu"),
        device="cpu",
    )
    return VisualMaterialShapeWriter(model, dict(_SOURCE_SHAPE_MATERIALS))


def test_rows_group_by_bound_material_across_clones(writer):
    rows = writer.rows_by_material()

    assert sorted(rows.keys()) == [_BUCKET_A, _BUCKET_B]
    assert rows[_BUCKET_A].tolist() == [0, 2]  # env_0 and the plan-resolved env_1 clone
    assert rows[_BUCKET_B].tolist() == [1, 3]
    assert writer.rows_by_material() is rows  # memoized


def test_write_scatters_srgb_colors_to_bound_shapes(writer):
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 1.0]])

    writer.write_colors((_BUCKET_A, _BUCKET_B), colors)

    shape_colors = wp.to_torch(writer._model.shape_color)
    expected = _expected_srgb(colors)
    assert torch.allclose(shape_colors[torch.tensor([0, 2])], expected[0].expand(2, 3), atol=1e-6)
    assert torch.allclose(shape_colors[torch.tensor([1, 3])], expected[1].expand(2, 3), atol=1e-6)
    # unbound / unresolvable rows keep their model color
    assert torch.allclose(shape_colors[torch.tensor([4, 5])], torch.full((2, 3), 0.5))


def test_write_reuses_one_flattened_scatter_plan(writer):
    material_paths = (_BUCKET_A, _BUCKET_B)

    writer.write_colors(material_paths, torch.rand(2, 3))
    plan = writer._plans[material_paths]
    writer.write_colors(material_paths, torch.rand(2, 3))

    # repeated fires reuse the memoized plan (one fused kernel launch per fire, no per-material loop)
    assert writer._plans[material_paths] is plan
    assert plan.dim == 4
    assert plan.shape_rows.numpy().tolist() == [0, 2, 1, 3]
    assert plan.material_index.numpy().tolist() == [0, 0, 1, 1]


def test_write_ignores_materials_bound_by_no_shape(writer):
    before = wp.to_torch(writer._model.shape_color).clone()

    writer.write_colors(("/World/Materials/unused",), torch.tensor([[1.0, 1.0, 1.0]]))

    assert torch.equal(wp.to_torch(writer._model.shape_color), before)


"""
Per-environment materials: the captured material path lives inside the replicated subtree, so the
row resolution must remap it per environment alongside the bound shape.
"""

_PER_ENV_MATERIAL = "/World/envs/env_0/Materials/style"
_PER_ENV_SHAPE_LABELS = [
    "/World/envs/env_0/cube/geometry/mesh",
    "/World/envs/env_1/cube/geometry/mesh",
    "/World/ground",
]


@pytest.fixture
def per_env_writer(monkeypatch):
    # homogeneous plan: the whole source env is one replication row, materials ride along
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    fake_sim = SimpleNamespace(get_clone_plan=lambda: plan)
    monkeypatch.setattr(shape_writer_module, "SimulationContext", SimpleNamespace(instance=lambda: fake_sim))

    model = SimpleNamespace(
        shape_label=list(_PER_ENV_SHAPE_LABELS),
        shape_color=wp.array([wp.vec3(0.5, 0.5, 0.5)] * len(_PER_ENV_SHAPE_LABELS), dtype=wp.vec3, device="cpu"),
        device="cpu",
    )
    # one source-space capture: the cube's mesh binds its own env's material clone
    return VisualMaterialShapeWriter(model, {"/World/envs/env_0/cube/geometry/mesh": _PER_ENV_MATERIAL})


@pytest.fixture
def heterogeneous_writer(monkeypatch):
    """Per-asset layout: variant A prototypes env 0, variant B prototypes env 1, per-env material."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/cube", "/World/envs/env_1/cube", _PER_ENV_MATERIAL),
        destinations=("/World/envs/env_{}/cube", "/World/envs/env_{}/cube", "/World/envs/env_{}/Materials/style"),
        clone_mask=torch.tensor([[True, False], [False, True], [True, True]]),
    )
    fake_sim = SimpleNamespace(get_clone_plan=lambda: plan)
    monkeypatch.setattr(shape_writer_module, "SimulationContext", SimpleNamespace(instance=lambda: fake_sim))

    model = SimpleNamespace(
        shape_label=["/World/envs/env_0/cube/geometry/mesh", "/World/envs/env_1/cube/geometry/mesh"],
        shape_color=wp.array([wp.vec3(0.5, 0.5, 0.5)] * 2, dtype=wp.vec3, device="cpu"),
        device="cpu",
    )
    # both variant prototypes capture the per-env material in SOURCE space (env 0), as the
    # spawners author the binding
    return VisualMaterialShapeWriter(
        model,
        {
            "/World/envs/env_0/cube/geometry/mesh": _PER_ENV_MATERIAL,
            "/World/envs/env_1/cube/geometry/mesh": _PER_ENV_MATERIAL,
        },
    )


def test_heterogeneous_prototypes_key_their_own_env_materials(heterogeneous_writer):
    """A variant prototype outside env 0 must key its shapes to ITS env's material clone."""
    rows = heterogeneous_writer.rows_by_material()

    assert rows[_PER_ENV_MATERIAL].tolist() == [0]
    assert rows["/World/envs/env_1/Materials/style"].tolist() == [1]


def test_per_env_material_rows_and_writes_address_each_env(per_env_writer):
    # one row set per environment clone of the material — NOT collapsed under the source path
    rows = per_env_writer.rows_by_material()
    assert sorted(rows.keys()) == [_PER_ENV_MATERIAL, "/World/envs/env_1/Materials/style"]
    assert rows[_PER_ENV_MATERIAL].tolist() == [0]
    assert rows["/World/envs/env_1/Materials/style"].tolist() == [1]

    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    per_env_writer.write_colors((_PER_ENV_MATERIAL, "/World/envs/env_1/Materials/style"), colors)

    shape_colors = wp.to_torch(per_env_writer._model.shape_color)
    expected = _expected_srgb(colors)
    assert torch.allclose(shape_colors[0], expected[0], atol=1e-6)
    assert torch.allclose(shape_colors[1], expected[1], atol=1e-6)
    assert torch.allclose(shape_colors[2], torch.full((3,), 0.5))
