# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared simulator-free collider geometry."""

from __future__ import annotations

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics.collider_geometry import (
    model_body_collider_z_min,
    points_transform_xyzw,
)
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.mesh_collision import (
    IKObjectiveMeshCollision,
    collision_probes_sample,
)


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
    )
    bodies, points, slots = collision_probes_sample(builder, (), n_samples=8)

    np.testing.assert_array_equal(bodies, np.zeros(8, dtype=np.int32))
    np.testing.assert_array_equal(slots, -np.ones(8, dtype=np.int32))
    np.testing.assert_allclose(points.min(axis=0), (-1.5, -2.0, -1.0), atol=1.0e-6)
    np.testing.assert_allclose(points.max(axis=0), (2.5, 0.0, 5.0), atol=1.0e-6)


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
    )

    z_min = model_body_collider_z_min(builder, (0, 1, 2, 3))

    np.testing.assert_allclose(z_min, (0.0, 0.75, 0.5, 0.7), atol=1.0e-6)


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
