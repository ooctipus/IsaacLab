# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared simulator-free collider geometry."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import newton
import newton.ik as ik
import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics.collider_geometry import (
    model_body_collider_support_points,
    model_body_collider_z_min,
    points_transform_xyzw,
)
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.context import IKObjectiveMeshCollisionBuildContext
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.mesh_collision import (
    IKObjectiveMeshCollision,
    collision_probes_sample,
)

_COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)


def test_points_transform_xyzw_applies_rotation_before_translation() -> None:
    """Collider points must include the complete owning-shape transform."""
    half_sqrt = np.sqrt(0.5)
    transformed = points_transform_xyzw(
        np.array(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float32),
        np.array((1.0, 2.0, 3.0), dtype=np.float32),
        np.array((0.0, 0.0, half_sqrt, half_sqrt), dtype=np.float32),
    )
    np.testing.assert_allclose(transformed, ((1.0, 3.0, 3.0), (-1.0, 2.0, 3.0)), atol=1.0e-6)


def test_box_collision_probes_use_newton_half_extents_and_shape_transform() -> None:
    """Newton box scale is already half-extents and must not be halved again."""
    half_sqrt = np.sqrt(0.5)
    builder = SimpleNamespace(
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.BOX],
        shape_scale=[(1.0, 2.0, 3.0)],
        shape_transform=[(0.5, -1.0, 2.0, 0.0, 0.0, half_sqrt, half_sqrt)],
        shape_flags=[_COLLIDE],
    )
    bodies, points, slots = collision_probes_sample(builder, (), n_samples=8)

    np.testing.assert_array_equal(bodies, np.zeros(8, dtype=np.int32))
    np.testing.assert_array_equal(slots, -np.ones(8, dtype=np.int32))
    np.testing.assert_allclose(points.min(axis=0), (-1.5, -2.0, -1.0), atol=1.0e-6)
    np.testing.assert_allclose(points.max(axis=0), (2.5, 0.0, 5.0), atol=1.0e-6)


def test_collision_probes_exempt_only_declared_contact_feature() -> None:
    """A semantic toe contact must not disable every collision probe on its foot."""
    builder = SimpleNamespace(
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.BOX],
        shape_scale=[(1.0, 0.5, 0.2)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    contact_point = np.array(((1.0, 0.5, -0.2),), dtype=np.float32)

    bodies, points, slots = collision_probes_sample(builder, (0,), n_samples=4, contact_points_body=contact_point)

    np.testing.assert_array_equal(bodies, np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(slots, (0, -1, -1, -1))
    np.testing.assert_allclose(points[0], contact_point[0], atol=1.0e-7)


def test_model_body_collider_z_min_applies_rotated_primitive_extents() -> None:
    """Lowest body-frame points use full shape rotation and Newton scale semantics."""
    half_sqrt = np.sqrt(0.5)
    builder = SimpleNamespace(
        shape_body=[0, 1, 2, 3],
        shape_source=[None, None, None, None],
        shape_type=[newton.GeoType.BOX, newton.GeoType.SPHERE, newton.GeoType.CAPSULE, newton.GeoType.CYLINDER],
        shape_scale=[(1.0, 2.0, 3.0), (0.25, 0.0, 0.0), (0.2, 0.5, 0.0), (0.3, 0.8, 0.0)],
        shape_transform=[
            (0.0, 0.0, 2.0, half_sqrt, 0.0, 0.0, half_sqrt),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.7, 0.0, half_sqrt, 0.0, half_sqrt),
            (0.0, 0.0, 1.0, 0.0, half_sqrt, 0.0, half_sqrt),
        ],
        shape_flags=[_COLLIDE] * 4,
    )

    z_min = model_body_collider_z_min(builder, (0, 1, 2, 3))

    np.testing.assert_allclose(z_min, (0.0, 0.75, 0.5, 0.7), atol=1.0e-6)


def test_support_points_select_default_world_low_surface_after_body_rotation() -> None:
    """Surface selection uses default-world height while returning body-local points."""
    half_sqrt = np.sqrt(0.5)
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.BOX],
        shape_scale=[(1.0, 0.5, 0.2)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, half_sqrt, 0.0, half_sqrt),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder, (0,), default_pose, points_per_body=2, height_band_m=0.0
    )
    world = points_transform_xyzw(points, default_pose[0, :3], default_pose[0, 3:7])

    np.testing.assert_array_equal(slots, (0, 0))
    np.testing.assert_allclose(world[:, 2], -1.0, atol=1.0e-6)
    assert np.linalg.norm(world[0, :2] - world[1, :2]) > 0.0


def test_support_points_select_one_planar_face_nearest_semantic_up() -> None:
    """Semantic support selection retains one real collider face even when it is tilted."""
    half_sqrt = np.sqrt(0.5)
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.BOX],
        shape_scale=[(1.0, 0.5, 0.2)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, half_sqrt, 0.0, 0.0, half_sqrt),), dtype=np.float32)
    support_up = np.array(((0.0, -1.0, 0.2),), dtype=np.float32)
    support_up /= np.linalg.norm(support_up, axis=1, keepdims=True)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=support_up,
    )
    world = points_transform_xyzw(points, default_pose[0, :3], default_pose[0, 3:7])
    normal = np.cross(world[1] - world[0], world[2] - world[0])
    normal /= np.linalg.norm(normal)

    np.testing.assert_array_equal(slots, (0, 0, 0))
    assert abs(float(normal @ support_up[0])) > 0.95


def test_semantic_support_points_apply_mesh_shape_scale() -> None:
    """Semantic mesh support points include Newton's per-shape mesh scale."""
    source = SimpleNamespace(
        vertices=np.array(((0.0, 0.0, -1.0), (0.0, 1.0, -1.0), (1.0, 0.0, -1.0)), dtype=np.float32),
        indices=np.array((0, 1, 2), dtype=np.int32),
    )
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[source],
        shape_type=[newton.GeoType.MESH],
        shape_scale=[(2.0, 3.0, 4.0)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )

    np.testing.assert_array_equal(slots, (0, 0, 0))
    np.testing.assert_allclose(points, ((0.0, 0.0, -4.0), (0.0, 3.0, -4.0), (2.0, 0.0, -4.0)))


def test_semantic_support_points_keep_outward_normals_for_mirrored_mesh_scale() -> None:
    """A negative-determinant Newton mesh scale does not swap its low and high faces."""
    source = SimpleNamespace(
        vertices=np.array([[x, y, z] for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)], dtype=np.float32),
        indices=np.array(
            (
                (0, 2, 3),
                (0, 3, 1),
                (4, 5, 7),
                (4, 7, 6),
                (0, 1, 5),
                (0, 5, 4),
                (2, 6, 7),
                (2, 7, 3),
                (0, 4, 6),
                (0, 6, 2),
                (1, 3, 7),
                (1, 7, 5),
            ),
            dtype=np.int32,
        ),
    )
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[source],
        shape_type=[newton.GeoType.MESH],
        shape_scale=[(-1.0, 1.0, 1.0)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )

    np.testing.assert_array_equal(slots, (0, 0, 0))
    np.testing.assert_allclose(points[:, 2], -np.ones(3, dtype=np.float32))


def test_semantic_mesh_support_height_band_filters_individual_faces() -> None:
    """Parallel mesh faces above the height band do not enter the low support patch."""
    source = SimpleNamespace(
        vertices=np.array(
            (
                (0.0, 0.0, -1.0),
                (0.0, 1.0, -1.0),
                (1.0, 0.0, -1.0),
                (10.0, 0.0, 0.0),
                (10.0, 1.0, 0.0),
                (11.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        indices=np.array(((0, 1, 2), (3, 4, 5)), dtype=np.int32),
    )
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[source],
        shape_type=[newton.GeoType.MESH],
        shape_scale=[(1.0, 1.0, 1.0)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )

    np.testing.assert_array_equal(slots, (0, 0, 0))
    np.testing.assert_allclose(points[:, 2], -np.ones(3, dtype=np.float32))


def test_semantic_cylinder_support_returns_planar_cap_triangle() -> None:
    """An aligned cylinder exposes three non-collinear points on its low planar cap."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.CYLINDER],
        shape_scale=[(1.0, 2.0, 0.0)],
        shape_transform=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        shape_flags=[_COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )
    normal = np.cross(points[1] - points[0], points[2] - points[0])

    np.testing.assert_array_equal(slots, (0, 0, 0))
    np.testing.assert_allclose(points[:, 2], -2.0)
    assert np.all(np.linalg.norm(points[:, :2], axis=1) <= 1.0 + 1.0e-6)
    assert np.linalg.norm(normal) > 1.0e-8


def test_semantic_support_points_do_not_fill_from_outside_height_band() -> None:
    """An undersized low patch stays undersized instead of borrowing high geometry."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0, 0],
        shape_source=[None, None],
        shape_type=[newton.GeoType.SPHERE, newton.GeoType.BOX],
        shape_scale=[(1.0, 0.0, 0.0), (1.0, 1.0, 0.5)],
        shape_transform=[
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0),
        ],
        shape_flags=[_COLLIDE, _COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=3,
        height_band_m=0.0,
        support_up_world=np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )

    np.testing.assert_array_equal(slots, (0,))
    np.testing.assert_allclose(points, ((0.0, 0.0, -1.0),))


def test_support_points_retain_spread_across_flat_collider_surface() -> None:
    """Multiple target points preserve a deterministic heel/toe-like footprint."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0, 0],
        shape_source=[None, None],
        shape_type=[newton.GeoType.SPHERE, newton.GeoType.SPHERE],
        shape_scale=[(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        shape_transform=[
            (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
        shape_flags=[_COLLIDE, _COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder, (0,), default_pose, points_per_body=2, height_band_m=0.0
    )

    np.testing.assert_array_equal(slots, (0, 0))
    np.testing.assert_allclose(points, ((-0.5, 0.0, -0.1), (0.5, 0.0, -0.1)), atol=1.0e-6)


def test_support_points_select_declared_world_direction_extreme() -> None:
    """One semantic contact feature selects the low point furthest along its declared direction."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0, 0],
        shape_source=[None, None],
        shape_type=[newton.GeoType.SPHERE, newton.GeoType.SPHERE],
        shape_scale=[(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        shape_transform=[
            (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
        shape_flags=[_COLLIDE, _COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    points, slots = model_body_collider_support_points(
        builder,
        (0,),
        default_pose,
        points_per_body=1,
        height_band_m=0.0,
        selection_directions_world=np.array(((1.0, 0.0, 0.0),), dtype=np.float32),
    )

    np.testing.assert_array_equal(slots, (0,))
    np.testing.assert_allclose(points, ((0.5, 0.0, -0.1),), atol=1.0e-6)


def test_collider_geometry_and_probes_ignore_visual_only_shapes() -> None:
    """Only shapes carrying Newton's collision flag contribute physical geometry."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0, 0],
        shape_source=[None, None],
        shape_type=[newton.GeoType.SPHERE, newton.GeoType.SPHERE],
        shape_scale=[(5.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
        shape_transform=[
            (100.0, 0.0, -100.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ],
        shape_flags=[0, _COLLIDE],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    z_min = model_body_collider_z_min(builder, (0,))
    points, slots = model_body_collider_support_points(
        builder, (0,), default_pose, points_per_body=1, height_band_m=0.0
    )
    probe_bodies, probe_points, probe_slots = collision_probes_sample(builder, (), n_samples=8)

    np.testing.assert_allclose(z_min, (0.75,), atol=1.0e-6)
    np.testing.assert_allclose(points, ((0.0, 0.0, 0.75),), atol=1.0e-6)
    np.testing.assert_array_equal(slots, (0,))
    np.testing.assert_array_equal(probe_bodies, np.zeros(8, dtype=np.int32))
    np.testing.assert_array_equal(probe_slots, -np.ones(8, dtype=np.int32))
    assert np.max(np.linalg.norm(probe_points - np.array((0.0, 0.0, 1.0)), axis=1)) <= 0.250001


@pytest.mark.parametrize("function", ("z_min", "support", "probes"))
def test_collider_consumers_reject_missing_shape_flags(function: str) -> None:
    """Controlled kinematics paths never guess whether a shape collides."""
    builder = SimpleNamespace(
        body_count=1,
        shape_body=[0],
        shape_source=[None],
        shape_type=[newton.GeoType.SPHERE],
        shape_scale=[(0.25, 0.0, 0.0)],
        shape_transform=[(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
    )
    default_pose = np.array(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)

    with pytest.raises(ValueError, match="shape_flags"):
        if function == "z_min":
            model_body_collider_z_min(builder, (0,))
        elif function == "support":
            model_body_collider_support_points(builder, (0,), default_pose, points_per_body=1, height_band_m=0.0)
        else:
            collision_probes_sample(builder, (), n_samples=1)


def test_mesh_collision_confidence_names_and_fallback_memory_are_exact() -> None:
    """The public float-confidence name is unique and fallback storage is one float per batch row."""
    parameters = inspect.signature(IKObjectiveMeshCollision).parameters
    assert "contact_confidence" in parameters
    assert "contact_mask" not in parameters
    assert "contact_confidence" in IKObjectiveMeshCollisionBuildContext.__dataclass_fields__
    assert "contact_mask" not in IKObjectiveMeshCollisionBuildContext.__dataclass_fields__

    obstacle_pose = torch.zeros((3, 7), dtype=torch.float32)
    obstacle_pose[:, 6] = 1.0
    kwargs = {
        "probe_offsets": np.zeros((1, 3), dtype=np.float32),
        "probe_bodies": np.zeros(1, dtype=np.int32),
        "probe_affects_dof": np.zeros((1, 0), dtype=np.uint8),
        "mesh": 0,
        "obstacle_pose": obstacle_pose,
        "weight": 1.0,
        "margin": 0.01,
    }
    fallback = IKObjectiveMeshCollision(**kwargs)
    external = IKObjectiveMeshCollision(
        **kwargs,
        probe_contact_slots=np.zeros(1, dtype=np.int32),
        contact_confidence=torch.zeros((3, 1), dtype=torch.float32),
    )
    fallback_bytes = fallback.estimate_memory(None, ik.IKJacobianType.ANALYTIC, 3, 3, 1)
    external_bytes = external.estimate_memory(None, ik.IKJacobianType.ANALYTIC, 3, 3, 1)
    assert fallback_bytes - external_bytes == 3 * torch.float32.itemsize


@pytest.mark.parametrize(("margin", "max_distance"), ((0.0, 0.25), (float("nan"), 0.25), (0.01, 0.0)))
def test_mesh_collision_objective_rejects_invalid_distance_scales(margin: float, max_distance: float) -> None:
    """Softplus and mesh-query scales must be finite and positive."""
    with pytest.raises(ValueError, match="finite and positive"):
        IKObjectiveMeshCollision(
            probe_offsets=np.zeros((1, 3), dtype=np.float32),
            probe_bodies=np.zeros(1, dtype=np.int32),
            probe_affects_dof=np.zeros((1, 0), dtype=np.uint8),
            mesh=0,
            obstacle_pose=torch.tensor(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),)),
            weight=1.0,
            margin=margin,
            max_distance=max_distance,
        )
