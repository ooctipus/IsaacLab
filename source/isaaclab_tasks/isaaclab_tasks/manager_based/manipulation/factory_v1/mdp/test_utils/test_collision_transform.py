# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the collision analyzer's Python-level transform chain.

The collision analyzer transforms query points from world space to obstacle-root
space via::

    query_in_root = quat_apply_inverse(obs_quat, point_world - obs_pos) / obs_scale

Then the warp kernel further transforms from root to collider-local via::

    q1 = q - coll_rel_pos
    q2 = quat_rotate_inv(coll_rel_quat, q1)
    q3 = q2 / coll_rel_scale

These tests verify the Python-level world→root transform using ``math_utils``
with analytically known expected results. All quaternions use **(x, y, z, w)**
convention consistent with IsaacLab.
"""

import math

import pytest
import torch

import isaaclab.utils.math as math_utils


def _world_to_root(point_w: torch.Tensor, obs_pos: torch.Tensor, obs_quat: torch.Tensor, obs_scale: torch.Tensor) -> torch.Tensor:
    """Replicate the collision analyzer's world→root transform (lines 153-158)."""
    return math_utils.quat_apply_inverse(obs_quat, point_w - obs_pos) / obs_scale


def _root_to_collider(point_root: torch.Tensor, coll_pos: torch.Tensor, coll_quat: torch.Tensor, coll_scale: torch.Tensor) -> torch.Tensor:
    """Replicate the warp kernel's root→collider transform (lines 88-91)."""
    q1 = point_root - coll_pos
    q2 = math_utils.quat_apply_inverse(coll_quat, q1)
    return q2 / coll_scale


def _quat_identity():
    """Identity quaternion in (x, y, z, w) format."""
    return torch.tensor([0.0, 0.0, 0.0, 1.0])


def _quat_90_around_z():
    """90-degree rotation around Z in (x, y, z, w) format."""
    angle = math.pi / 2
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def _quat_45_around_y():
    """45-degree rotation around Y in (x, y, z, w) format."""
    angle = math.pi / 4
    return torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])


class TestWorldToRoot:

    def test_identity_transform(self):
        result = _world_to_root(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    def test_pure_translation(self):
        result = _world_to_root(
            torch.tensor([[3.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    def test_pure_rotation_90_around_z(self):
        """90 deg CCW around Z: world X axis → root -Y axis."""
        result = _world_to_root(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_90_around_z().unsqueeze(0),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5, rtol=0)

    def test_pure_uniform_scale(self):
        result = _world_to_root(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[2.0, 2.0, 2.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.5, 0.0, 0.0]]), atol=1e-6, rtol=0)

    def test_pure_non_uniform_scale(self):
        result = _world_to_root(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[1.0, 2.0, 3.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, 1.0, 1.0]]), atol=1e-6, rtol=0)

    def test_rotation_then_uniform_scale(self):
        """90 deg Z + scale 2: world (2,0,0) → rotate → (0,-2,0) → /2 → (0,-1,0)."""
        result = _world_to_root(
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_90_around_z().unsqueeze(0),
            torch.tensor([[2.0, 2.0, 2.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5, rtol=0)

    def test_rotation_plus_non_uniform_scale(self):
        """90 deg Z + scale (1,2,1): world (2,0,0) → rotate → (0,-2,0) → /(1,2,1) → (0,-1,0).

        This tests whether the code correctly applies inverse-rotate THEN inverse-scale.
        The forward transform would be: scale → rotate → translate.
        The inverse is: translate^-1 → rotate^-1 → scale^-1.
        The code does: quat_apply_inverse(q, p - t) / s, which IS rotate^-1 then scale^-1. Correct.
        """
        result = _world_to_root(
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_90_around_z().unsqueeze(0),
            torch.tensor([[1.0, 2.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5, rtol=0)

    def test_translation_rotation_scale_combined(self):
        """pos=(1,2,3), rot=45 deg Y, scale=(2,1,0.5). Query at (3,2,3).

        Step by step:
        1. p - t = (2, 0, 0)
        2. rot_inv(45Y, (2,0,0)):
           cos(-45) = cos45, sin(-45) = -sin45
           Ry(-45) @ (2,0,0) = (2*cos45, 0, 2*sin45) = (sqrt(2), 0, sqrt(2))
        3. / scale = (sqrt(2)/2, 0, sqrt(2)/0.5) = (sqrt(2)/2, 0, 2*sqrt(2))
        """
        s2 = math.sqrt(2)
        result = _world_to_root(
            torch.tensor([[3.0, 2.0, 3.0]]),
            torch.tensor([[1.0, 2.0, 3.0]]),
            _quat_45_around_y().unsqueeze(0),
            torch.tensor([[2.0, 1.0, 0.5]]),
        )
        expected = torch.tensor([[s2 / 2, 0.0, 2 * s2]])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)


class TestNestedTransform:

    def test_root_plus_collider(self):
        """Root at (1,0,0), scale=(2,2,2). Collider at rel pos=(0.5,0,0), rot=90Z, scale=(1,1,1).

        Query: (3,0,0) world.
        World→root: (3-1, 0, 0) / (2,2,2) = (1, 0, 0)
        Root→collider: (1-0.5, 0, 0) = (0.5, 0, 0) → rot_inv(90Z) → (0, -0.5, 0) → /(1,1,1) → (0, -0.5, 0)
        """
        root_result = _world_to_root(
            torch.tensor([[3.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[2.0, 2.0, 2.0]]),
        )
        torch.testing.assert_close(root_result, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

        coll_result = _root_to_collider(
            root_result,
            torch.tensor([[0.5, 0.0, 0.0]]),
            _quat_90_around_z().unsqueeze(0),
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(coll_result, torch.tensor([[0.0, -0.5, 0.0]]), atol=1e-5, rtol=0)

    def test_nested_with_collider_non_uniform_scale(self):
        """Root identity. Collider at origin, no rotation, scale=(1,2,1).

        Query in root: (0, 1.5, 0)
        Collider local: (0, 1.5/2, 0) = (0, 0.75, 0)
        """
        result = _root_to_collider(
            torch.tensor([[0.0, 1.5, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_identity().unsqueeze(0),
            torch.tensor([[1.0, 2.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, 0.75, 0.0]]), atol=1e-6, rtol=0)

    def test_nested_collider_rotation_plus_non_uniform_scale(self):
        """Root identity. Collider at origin, rot=90Z, scale=(1,2,1).

        Query in root: (1.5, 0, 0)
        Step 1 (subtract pos): (1.5, 0, 0)
        Step 2 (inverse rotate 90Z): (0, -1.5, 0)
        Step 3 (divide scale (1,2,1)): (0, -0.75, 0)
        """
        result = _root_to_collider(
            torch.tensor([[1.5, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            _quat_90_around_z().unsqueeze(0),
            torch.tensor([[1.0, 2.0, 1.0]]),
        )
        torch.testing.assert_close(result, torch.tensor([[0.0, -0.75, 0.0]]), atol=1e-5, rtol=0)


class TestBatchTransform:

    def test_multiple_points_batched(self):
        """Verify the transform works correctly on a batch of points."""
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pos = torch.tensor([[0.0, 0.0, 0.0]]).expand(3, -1)
        quat = _quat_90_around_z().unsqueeze(0).expand(3, -1)
        scale = torch.tensor([[1.0, 1.0, 1.0]]).expand(3, -1)

        result = _world_to_root(points, pos, quat, scale)
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)
