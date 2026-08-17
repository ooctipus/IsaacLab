# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict all-shapes SDF collision provisioning."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonManager
from newton import CollisionPipeline, GeoType, Mesh, ModelBuilder


def _make_concave_l_prism() -> Mesh:
    """Create a concave mesh whose convex hull differs from its source geometry."""
    base = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    vertices = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    triangles: list[tuple[int, int, int]] = []
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 5)):
        triangles.append((0, b, a))
        triangles.append((6, 6 + a, 6 + b))
    for a in range(6):
        b = (a + 1) % 6
        triangles.append((a, b, b + 6))
        triangles.append((a, b + 6, a + 6))
    return Mesh(vertices, np.asarray(triangles, dtype=np.int32).reshape(-1), compute_inertia=False)


def _sdf_cfg(**kwargs) -> NewtonCollisionPipelineCfg.SDFAllShapesCfg:
    return NewtonCollisionPipelineCfg.SDFAllShapesCfg(
        sdf_max_resolution=32,
        sdf_narrow_band_inner=-0.05,
        sdf_narrow_band_outer=0.05,
        **kwargs,
    )


def test_all_shapes_sdf_preserves_convex_hull_before_mesh_conversion():
    """A requested convex hull is built first and remains the geometry cooked into the SDF."""
    builder = ModelBuilder()
    shape = builder.add_shape_mesh(body=-1, mesh=_make_concave_l_prism(), label="convex-first")
    builder.approximate_meshes(method="convex_hull", shape_indices=[shape], raise_on_failure=True)
    hull_source = builder.shape_source[shape]
    assert GeoType(builder.shape_type[shape]) == GeoType.CONVEX_MESH

    sdf_shapes = NewtonManager._configure_sdf_all_shapes(builder, _sdf_cfg())

    assert sdf_shapes == (shape,)
    assert GeoType(builder.shape_type[shape]) == GeoType.CONVEX_MESH
    assert builder.shape_source[shape] is hull_source
    assert builder.shape_sdf_max_resolution[shape] == 32


def test_all_shapes_sdf_converts_every_colliding_primitive_and_ignores_sites():
    """Every supported collider becomes an SDF mesh while non-colliding sites remain untouched."""
    builder = ModelBuilder()
    colliders = [
        builder.add_shape_plane(width=4.0, length=6.0, label="plane"),
        builder.add_shape_box(body=-1, hx=1.0, hy=2.0, hz=3.0, label="box"),
        builder.add_shape_sphere(body=-1, radius=0.5, label="sphere"),
        builder.add_shape_capsule(body=-1, radius=0.25, half_height=0.75, label="capsule"),
        builder.add_shape_ellipsoid(body=-1, rx=0.25, ry=0.5, rz=0.75, label="ellipsoid"),
        builder.add_shape_cylinder(body=-1, radius=0.5, half_height=1.0, label="cylinder"),
        builder.add_shape_cone(body=-1, radius=0.5, half_height=1.0, label="cone"),
    ]
    site = builder.add_shape_sphere(body=-1, radius=0.1, as_site=True, label="site")

    sdf_shapes = NewtonManager._configure_sdf_all_shapes(builder, _sdf_cfg(plane_thickness=0.2))

    assert sdf_shapes == tuple(colliders)
    assert all(GeoType(builder.shape_type[index]) == GeoType.MESH for index in colliders)
    assert all(builder.shape_source[index] is not None for index in colliders)
    assert all(builder.shape_scale[index] == wp.vec3(1.0) for index in colliders)
    assert GeoType(builder.shape_type[site]) == GeoType.SPHERE
    assert builder.shape_sdf_max_resolution[site] is None
    assert tuple(builder.shape_transform[colliders[0]].p) == pytest.approx((0.0, 0.0, -0.1))


def test_all_shapes_sdf_rejects_geometry_that_cannot_be_strictly_provisioned():
    """The strict policy fails instead of silently retaining an unsupported collision path."""
    builder = ModelBuilder()
    shape = builder.add_shape_box(body=-1, label="heightfield")
    builder.shape_type[shape] = GeoType.HFIELD

    with pytest.raises(ValueError, match="heightfield.*HFIELD"):
        NewtonManager._configure_sdf_all_shapes(builder, _sdf_cfg())


def test_all_shapes_sdf_preserves_an_infinite_analytic_plane():
    """An infinite plane remains an analytic half-space because no finite SDF can represent it."""
    builder = ModelBuilder()
    plane = builder.add_shape_plane(width=0.0, length=0.0, label="infinite-plane")

    sdf_shapes = NewtonManager._configure_sdf_all_shapes(builder, _sdf_cfg())

    assert sdf_shapes == ()
    assert GeoType(builder.shape_type[plane]) == GeoType.PLANE
    assert builder.shape_sdf_max_resolution[plane] is None


def test_all_shapes_sdf_finalization_provisions_texture_and_edges():
    """Finalized strict SDF colliders have both texture data and mesh edges required by routing."""
    builder = ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.75, hz=1.0, label="box")
    builder.add_shape_sphere(body=-1, radius=0.5, label="sphere")
    sdf_shapes = NewtonManager._configure_sdf_all_shapes(builder, _sdf_cfg())

    model = builder.finalize(device="cuda:0")

    NewtonManager._validate_sdf_all_shapes(model, sdf_shapes)
    sdf_indices = model._shape_sdf_index.numpy()[list(sdf_shapes)]
    edge_counts = model.shape_edge_range.numpy()[list(sdf_shapes), 1]
    assert np.all(sdf_indices >= 0)
    assert np.all(edge_counts > 0)


def test_all_shapes_sdf_config_is_not_forwarded_to_collision_pipeline():
    """Builder-only policy options never leak into ``CollisionPipeline.__init__``."""
    cfg = NewtonCollisionPipelineCfg(sdf_all_shapes=_sdf_cfg())

    assert "sdf_all_shapes" not in cfg.to_pipeline_args()


def test_contact_reduction_hashtable_size_factor_is_forwarded_to_collision_pipeline():
    """The independent contact-reduction capacity reaches Newton's collision pipeline."""
    cfg = NewtonCollisionPipelineCfg(contact_reduction_hashtable_size_factor=0.02)

    assert cfg.to_pipeline_args()["contact_reduction_hashtable_size_factor"] == pytest.approx(0.02)


def test_speculative_contact_config_is_converted_without_exposing_scheduler_timing():
    """The nested Newton config receives only the search cap; the manager owns its horizon."""
    cfg = NewtonCollisionPipelineCfg(
        speculative_config=NewtonCollisionPipelineCfg.SpeculativeContactCfg(max_speculative_extension=0.01)
    )

    speculative_cfg = cfg.to_pipeline_args()["speculative_config"]

    assert isinstance(speculative_cfg, CollisionPipeline.SpeculativeContactConfig)
    assert speculative_cfg.max_speculative_extension == pytest.approx(0.01)


@pytest.mark.parametrize("extension", [-0.001, np.inf, -np.inf, np.nan])
def test_speculative_contact_config_rejects_invalid_extension(extension):
    """Predictive search caps must be finite and nonnegative [m]."""
    with pytest.raises(ValueError, match="max_speculative_extension"):
        NewtonCollisionPipelineCfg.SpeculativeContactCfg(max_speculative_extension=extension)


@pytest.mark.parametrize("resolution", [0, 30, 1 << 16])
def test_all_shapes_sdf_rejects_invalid_resolution(resolution):
    """Texture dimensions obey Newton's positive, tiled, uint16-compatible limits."""
    with pytest.raises(ValueError, match="sdf_max_resolution"):
        NewtonCollisionPipelineCfg.SDFAllShapesCfg(sdf_max_resolution=resolution)
