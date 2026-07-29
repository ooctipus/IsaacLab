# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the @wp.struct + @wp.func composed SDF kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.core.multi_task.geom.sdf import (
    ColliderTransform,
    compose_root_mat,
    compute_world_point,
    get_signed_distance_mega,
    query_collider_sdf,
    scale_mat33_columns,
)

# ---------------------------------------------------------------------------
# Shared mesh
# ---------------------------------------------------------------------------

_CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 5, 4],
        [0, 1, 5],
        [2, 7, 6],
        [2, 3, 7],
        [0, 7, 3],
        [0, 4, 7],
        [1, 6, 5],
        [1, 2, 6],
    ],
    dtype=np.int32,
)


def _make_cube():
    return convert_to_warp_mesh(_CUBE_VERTS, _CUBE_FACES, "cpu")


# ---------------------------------------------------------------------------
# Thin wrapper kernels for testing @wp.func in isolation
# ---------------------------------------------------------------------------


@wp.kernel
def _test_scale_mat33_kernel(m: wp.array(dtype=wp.mat33), s: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.mat33)):
    out[0] = scale_mat33_columns(m[0], s[0])


@wp.kernel
def _test_compute_world_point_kernel(
    pos: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.quat),
    pt: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    out[0] = compute_world_point(pos[0], quat[0], pt[0])


@wp.kernel
def _test_compose_root_mat_kernel(
    quat: wp.array(dtype=wp.quat), scale: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.mat33)
):
    out[0] = compose_root_mat(quat[0], scale[0])


@wp.kernel
def _test_query_sdf_kernel(
    query: wp.array(dtype=wp.vec3),
    root_pos: wp.array(dtype=wp.vec3),
    root_mat: wp.array(dtype=wp.mat33),
    root_mat_inv: wp.array(dtype=wp.mat33),
    coll: wp.array(dtype=ColliderTransform),
    mesh_id: wp.array(dtype=wp.uint64),
    max_dist: float,
    out: wp.array(dtype=float),
):
    out[0] = query_collider_sdf(
        query[0], root_pos[0], root_mat[0], root_mat_inv[0], coll[0], mesh_id[0], max_dist, True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wp_vec3(x, y, z):
    return wp.array([wp.vec3(x, y, z)], dtype=wp.vec3, device="cpu")


def _wp_quat_identity():
    return wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device="cpu")


def _wp_quat_90z():
    a = math.pi / 4
    return wp.array([wp.quat(0.0, 0.0, math.sin(a), math.cos(a))], dtype=wp.quat, device="cpu")


def _wp_mat33_identity():
    return wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device="cpu")


def _make_collider(pos=(0, 0, 0), mat=None, mat_inv=None):
    ct = ColliderTransform()
    ct.pos = wp.vec3(*pos)
    ct.mat = mat if mat is not None else wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
    ct.mat_inv = mat_inv if mat_inv is not None else wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
    return wp.array([ct], dtype=ColliderTransform, device="cpu")


# ===========================================================================
# Tests
# ===========================================================================


class TestScaleMat33Columns:
    def _run(self, m_vals, s_vals):
        m = wp.array([wp.mat33(*m_vals)], dtype=wp.mat33, device="cpu")
        s = _wp_vec3(*s_vals)
        out = wp.zeros(1, dtype=wp.mat33, device="cpu")
        wp.launch(_test_scale_mat33_kernel, dim=1, inputs=[m, s], outputs=[out], device="cpu")
        return np.array(wp.to_torch(out).reshape(3, 3))

    def test_identity_scale(self):
        r = self._run([1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1])
        np.testing.assert_allclose(r, np.eye(3), atol=1e-6)

    def test_uniform_scale(self):
        r = self._run([1, 0, 0, 0, 1, 0, 0, 0, 1], [2, 2, 2])
        np.testing.assert_allclose(r, np.diag([2, 2, 2]), atol=1e-6)

    def test_non_uniform_scale(self):
        r = self._run([1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 2, 3])
        np.testing.assert_allclose(r, np.diag([1, 2, 3]), atol=1e-6)

    def test_rotation_plus_scale(self):
        """90Z rotation scaled by (1, 2, 1): column 1 (the sin/cos column) gets 2x."""
        c, s = 0.0, 1.0  # cos90=0, sin90=1 (approx)
        r = self._run([c, -s, 0, s, c, 0, 0, 0, 1], [1, 2, 1])
        expected = np.array([[0, -2, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_allclose(r, expected, atol=1e-5)


class TestComputeWorldPoint:
    def _run(self, pos, quat, pt):
        out = wp.zeros(1, dtype=wp.vec3, device="cpu")
        wp.launch(
            _test_compute_world_point_kernel,
            dim=1,
            inputs=[_wp_vec3(*pos), wp.array([quat], dtype=wp.quat, device="cpu"), _wp_vec3(*pt)],
            outputs=[out],
            device="cpu",
        )
        return np.array(wp.to_torch(out).reshape(3))

    def test_identity(self):
        r = self._run((1, 2, 3), wp.quat(0, 0, 0, 1), (0.5, 0, 0))
        np.testing.assert_allclose(r, [1.5, 2, 3], atol=1e-5)

    def test_90z_rotation(self):
        a = math.pi / 4
        q = wp.quat(0, 0, math.sin(a), math.cos(a))
        r = self._run((0, 0, 0), q, (1, 0, 0))
        np.testing.assert_allclose(r, [0, 1, 0], atol=1e-5)


class TestComposeRootMat:
    def _run(self, quat, scale):
        out = wp.zeros(1, dtype=wp.mat33, device="cpu")
        wp.launch(
            _test_compose_root_mat_kernel,
            dim=1,
            inputs=[wp.array([quat], dtype=wp.quat, device="cpu"), _wp_vec3(*scale)],
            outputs=[out],
            device="cpu",
        )
        return np.array(wp.to_torch(out).reshape(3, 3))

    def test_identity(self):
        r = self._run(wp.quat(0, 0, 0, 1), (1, 1, 1))
        np.testing.assert_allclose(r, np.eye(3), atol=1e-6)

    def test_uniform_scale(self):
        r = self._run(wp.quat(0, 0, 0, 1), (2, 2, 2))
        np.testing.assert_allclose(r, np.diag([2, 2, 2]), atol=1e-5)

    def test_90z_plus_scale(self):
        a = math.pi / 4
        r = self._run(wp.quat(0, 0, math.sin(a), math.cos(a)), (1, 2, 1))
        # R(90Z) = [[0,-1,0],[1,0,0],[0,0,1]], scaled columns by (1,2,1)
        expected = np.array([[0, -2, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_allclose(r, expected, atol=1e-5)


class TestQueryColliderSdf:
    """Test query_collider_sdf via thin wrapper kernel."""

    def _run(self, query, root_pos, root_mat, root_mat_inv, coll, mesh):
        out = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(
            _test_query_sdf_kernel,
            dim=1,
            inputs=[
                _wp_vec3(*query),
                _wp_vec3(*root_pos),
                wp.array([root_mat], dtype=wp.mat33, device="cpu"),
                wp.array([root_mat_inv], dtype=wp.mat33, device="cpu"),
                coll,
                wp.array([mesh.id], dtype=wp.uint64, device="cpu"),
                10.0,
            ],
            outputs=[out],
            device="cpu",
        )
        return wp.to_torch(out)[0].item()

    def _identity_mat(self):
        return wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)

    def test_identity_outside(self):
        mesh = _make_cube()
        r = self._run((1, 0, 0), (0, 0, 0), self._identity_mat(), self._identity_mat(), _make_collider(), mesh)
        assert r == pytest.approx(0.5, abs=1e-4)

    def test_identity_inside(self):
        mesh = _make_cube()
        r = self._run((0, 0, 0), (0, 0, 0), self._identity_mat(), self._identity_mat(), _make_collider(), mesh)
        assert r == pytest.approx(-0.5, abs=1e-4)

    def test_scaled_outside(self):
        """Root scale (2,2,2), query at (1.5,0,0) → 0.5 world outside."""
        mesh = _make_cube()
        root_mat = wp.mat33(2, 0, 0, 0, 2, 0, 0, 0, 2)
        root_mat_inv = wp.mat33(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5)
        r = self._run((1.5, 0, 0), (0, 0, 0), root_mat, root_mat_inv, _make_collider(), mesh)
        assert r == pytest.approx(0.5, abs=1e-4)

    def test_non_uniform_scale_inside(self):
        """Root scale (1,2,1), query at (0, 0.5, 0) → 0.5 world inside."""
        mesh = _make_cube()
        root_mat = wp.mat33(1, 0, 0, 0, 2, 0, 0, 0, 1)
        root_mat_inv = wp.mat33(1, 0, 0, 0, 0.5, 0, 0, 0, 1)
        r = self._run((0, 0.5, 0), (0, 0, 0), root_mat, root_mat_inv, _make_collider(), mesh)
        assert r == pytest.approx(-0.5, abs=1e-4)


class TestMegaKernel:
    """Test the full get_signed_distance_mega kernel."""

    def test_single_env_single_obstacle_identity(self):
        """1 env, 1 body, 2 points, 1 obstacle with 1 collider, all identity."""
        mesh = _make_cube()

        body_pos = wp.array([[wp.vec3(0, 0, 0)]], dtype=wp.vec3, device="cpu")
        body_quat = wp.array([[wp.quat(0, 0, 0, 1)]], dtype=wp.quat, device="cpu")
        body_ids = wp.array([0], dtype=wp.int32, device="cpu")
        local_pts = wp.array([[[wp.vec3(1, 0, 0), wp.vec3(0, 0, 0)]]], dtype=wp.vec3, device="cpu")
        env_ids = wp.array([0], dtype=wp.int32, device="cpu")

        obs_pos = wp.array([[wp.vec3(0, 0, 0)]], dtype=wp.vec3, device="cpu")
        obs_quat = wp.array([[wp.quat(0, 0, 0, 1)]], dtype=wp.quat, device="cpu")
        obs_scale = wp.array([[wp.vec3(1, 1, 1)]], dtype=wp.vec3, device="cpu")

        ct = ColliderTransform()
        ct.pos = wp.vec3(0, 0, 0)
        ct.mat = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
        ct.mat_inv = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
        colliders = wp.array([ct], dtype=ColliderTransform, device="cpu")
        handles = wp.array([mesh.id], dtype=wp.uint64, device="cpu")
        counts = wp.array([1], dtype=wp.int32, device="cpu")
        signs = wp.zeros(2, dtype=float, device="cpu")

        wp.launch(
            get_signed_distance_mega,
            dim=2,
            inputs=[
                body_pos,
                body_quat,
                body_ids,
                local_pts,
                env_ids,
                obs_pos,
                obs_quat,
                obs_scale,
                colliders,
                handles,
                counts,
                10.0,
                True,
                1,
                1,
                2,
                1,
                1,
            ],
            outputs=[signs],
            device="cpu",
        )

        result = wp.to_torch(signs)
        assert result[0].item() == pytest.approx(0.5, abs=1e-4)  # (1,0,0) → outside
        assert result[1].item() == pytest.approx(-0.5, abs=1e-4)  # (0,0,0) → inside

    def test_root_scale_world_distance(self):
        """Root scale 0.1 (mesh 10x). Query at (0.1, 0, 0) → 0.05 world outside."""
        mesh = _make_cube()

        body_pos = wp.array([[wp.vec3(0.1, 0, 0)]], dtype=wp.vec3, device="cpu")
        body_quat = wp.array([[wp.quat(0, 0, 0, 1)]], dtype=wp.quat, device="cpu")
        body_ids = wp.array([0], dtype=wp.int32, device="cpu")
        local_pts = wp.array([[[wp.vec3(0, 0, 0)]]], dtype=wp.vec3, device="cpu")
        env_ids = wp.array([0], dtype=wp.int32, device="cpu")

        obs_pos = wp.array([[wp.vec3(0, 0, 0)]], dtype=wp.vec3, device="cpu")
        obs_quat = wp.array([[wp.quat(0, 0, 0, 1)]], dtype=wp.quat, device="cpu")
        obs_scale = wp.array([[wp.vec3(0.1, 0.1, 0.1)]], dtype=wp.vec3, device="cpu")

        ct = ColliderTransform()
        ct.pos = wp.vec3(0, 0, 0)
        ct.mat = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
        ct.mat_inv = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
        colliders = wp.array([ct], dtype=ColliderTransform, device="cpu")
        handles = wp.array([mesh.id], dtype=wp.uint64, device="cpu")
        counts = wp.array([1], dtype=wp.int32, device="cpu")
        signs = wp.zeros(1, dtype=float, device="cpu")

        wp.launch(
            get_signed_distance_mega,
            dim=1,
            inputs=[
                body_pos,
                body_quat,
                body_ids,
                local_pts,
                env_ids,
                obs_pos,
                obs_quat,
                obs_scale,
                colliders,
                handles,
                counts,
                10.0,
                True,
                1,
                1,
                1,
                1,
                1,
            ],
            outputs=[signs],
            device="cpu",
        )

        assert wp.to_torch(signs)[0].item() == pytest.approx(0.05, abs=1e-4)
