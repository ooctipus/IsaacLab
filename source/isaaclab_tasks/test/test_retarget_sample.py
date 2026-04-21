# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Stage 1: sample_contacts.

Requires GPU + Warp for terrain mesh operations and morphological sampling.
"""

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from scipy.spatial import ConvexHull

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer
from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.sampling import (
    SupportPolygonSampler,
    SupportPolygonSamplerCfg,
)


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cuda:0"

# ANYmal-C nominal foot offsets (from FK at default stance)
FOOT_OFFSETS = np.array([
    [0.3, 0.1, -0.54],   # LF
    [-0.3, 0.1, -0.54],  # LH
    [0.3, -0.1, -0.54],  # RF
    [-0.3, -0.1, -0.54], # RH
], dtype=np.float32)

DEFAULT_JQ = np.zeros(19, dtype=np.float32)
DEFAULT_JQ[0:3] = [0, 0, 0.6]
DEFAULT_JQ[3:7] = [0, 0, 0, 1]
defs = [0, 0.4, -0.8, 0, -0.4, 0.8, 0, 0.4, -0.8, 0, -0.4, 0.8]
DEFAULT_JQ[7:] = defs


def _make_flat_mesh(size: float = 10.0, device: str = DEVICE) -> wp.Mesh:
    """Create a flat terrain mesh with enough triangles for morphological sampling."""
    mesh = trimesh.creation.box(extents=[size, size, 0.01])
    mesh.apply_translation([0, 0, -0.005])
    mesh = mesh.subdivide()
    return convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device=device)


def _make_stair_mesh(
    n_steps: int = 5, step_height: float = 0.2, step_depth: float = 0.3,
    width: float = 4.0, device: str = DEVICE,
) -> wp.Mesh:
    """Create a simple stair mesh."""
    meshes = []
    for i in range(n_steps):
        z = i * step_height
        x0 = i * step_depth
        x1 = x0 + step_depth
        box = trimesh.creation.box(extents=[step_depth, width, step_height + 0.01])
        box.apply_translation([x0 + step_depth / 2, 0, z / 2])
        tread = trimesh.creation.box(extents=[step_depth, width, 0.01])
        tread.apply_translation([x0 + step_depth / 2, 0, z + 0.005])
        meshes.extend([box, tread])
    mesh = trimesh.util.concatenate(meshes)
    return convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device=device)


def _make_sampler(cfg=None):
    if cfg is None:
        cfg = SupportPolygonSamplerCfg(num_candidates=500, oversample_candidates=2)
    return SupportPolygonSampler(
        cfg, foot_offsets=FOOT_OFFSETS, foot_ground_offset=0.06,
        standing_height=0.54, default_joint_q=DEFAULT_JQ,
    )


class TestSampleContactsFlat:
    """On flat terrain, all contacts should be at z~0 and mostly geometry-valid."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_flat_terrain_contacts_at_z0(self):
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, 19, 17, 4, device=DEVICE)
        sampler = _make_sampler()
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        assert n > 0, f"Expected at least some candidates on flat terrain, got 0. Rejections: {reject}"

        ct = buf.contact_targets.numpy()[: n * 4]
        z_vals = ct[:, 2]
        assert abs(z_vals.mean()) < 0.2, f"Contacts not near z=0: mean={z_vals.mean()}"


class TestSampleContactsConvexHull:
    """Every geometry-valid candidate must have 4 hull vertices."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_all_valid_have_4_hull_verts(self):
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, 19, 17, 4, device=DEVICE)
        sampler = _make_sampler()
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        if n == 0:
            pytest.skip("No valid candidates on flat mesh")

        ct = buf.contact_targets.numpy()
        checked = 0
        for i in range(n):
            pts = ct[i * 4:(i + 1) * 4, :2]
            spread = np.ptp(pts, axis=0)
            if spread.min() < 1e-6:
                continue
            hull = ConvexHull(pts)
            assert len(hull.vertices) == 4, (
                f"Candidate {i} has {len(hull.vertices)} hull vertices, expected 4"
            )
            checked += 1
        assert checked > 0, "No candidates with non-degenerate XY spread"


class TestSampleContactsRejectionStats:
    """Verify rejection stats are populated."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_rejection_dict_keys(self):
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, 19, 17, 4, device=DEVICE)
        sampler = _make_sampler()
        _, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        assert "too_few" in reject
        assert "hull<4" in reject
        assert "quality" in reject
        assert "base_height" in reject
