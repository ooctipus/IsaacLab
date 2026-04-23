# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Stage 1: sample_contacts.

Requires GPU + Warp for terrain mesh operations and morphological sampling.
"""

import builtins
import importlib

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from scipy.spatial import ConvexHull

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.buffer import RetargetBuffer
from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget.cfg import SupportPolygonSamplerCfg
from isaaclab_tasks.manager_based.locomotion.position.utils.kinematic import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.manager_based.locomotion.position.utils.sampling import SupportPolygonSampler


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


ANYMAL_USD = "/home/zhengyuz/Downloads/ANYmal-C/anymal_c.usd"
DEVICE = "cuda:0"
DEFAULT_JPOS = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
}

KIN_CFG = NewtonKinematicsCfg(
    usd_path=ANYMAL_USD,
    device=DEVICE,
    default_pos=(0, 0, 0.6),
    default_joint_pos=DEFAULT_JPOS,
)


@pytest.fixture(scope="module")
def robot():
    kin = NewtonKinematics(KIN_CFG)
    foot_names = [n for n in kin.body_names if "FOOT" in n.upper()]
    foot_ids = kin.find_body_indices(foot_names)
    return kin, foot_ids


def _make_flat_mesh(size: float = 10.0, device: str = DEVICE) -> wp.Mesh:
    """Create a flat terrain mesh with enough triangles for morphological sampling."""
    mesh = trimesh.creation.box(extents=[size, size, 0.01])
    mesh.apply_translation([0, 0, -0.005])
    mesh = mesh.subdivide()
    return convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device=device)


def _make_stair_mesh(
    n_steps: int = 5,
    step_height: float = 0.2,
    step_depth: float = 0.3,
    width: float = 4.0,
    device: str = DEVICE,
) -> wp.Mesh:
    """Create a simple stair mesh."""
    meshes = []
    for i in range(n_steps):
        z = i * step_height
        x0 = i * step_depth
        box = trimesh.creation.box(extents=[step_depth, width, step_height + 0.01])
        box.apply_translation([x0 + step_depth / 2, 0, z / 2])
        top = trimesh.creation.box(extents=[step_depth, width, 0.01])
        top.apply_translation([x0 + step_depth / 2, 0, z + 0.005])
        meshes.extend([box, top])
    mesh = trimesh.util.concatenate(meshes)
    return convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device=device)


def _make_sampler(kin, foot_ids, cfg=None):
    if cfg is None:
        cfg = SupportPolygonSamplerCfg()
    return SupportPolygonSampler(cfg, kin=kin, foot_body_ids=foot_ids)


class TestSampleContactsFlat:
    """On flat terrain, all contacts should be at z~0 and mostly geometry-valid."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_flat_terrain_contacts_at_z0(self, robot):
        kin, foot_ids = robot
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        sampler = _make_sampler(kin, foot_ids)
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        assert n > 0, f"Expected at least some candidates on flat terrain, got 0. Rejections: {reject}"

        ct = buf.contact_targets.numpy()[: n * 4]
        z_vals = ct[:, 2]
        assert abs(z_vals.mean()) < 0.2, f"Contacts not near z=0: mean={z_vals.mean()}"


class TestSampleContactsConvexHull:
    """Geometry-valid candidates form a non-degenerate polygon (>=3 hull vertices)."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_valid_have_polygon_hull(self, robot):
        kin, foot_ids = robot
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        sampler = _make_sampler(kin, foot_ids)
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        if n == 0:
            pytest.skip("No valid candidates on flat mesh")

        ct = buf.contact_targets.numpy()
        checked = 0
        for i in range(n):
            pts = ct[i * 4 : (i + 1) * 4, :2]
            spread = np.ptp(pts, axis=0)
            if spread.min() < 1e-6:
                continue
            hull = ConvexHull(pts)
            assert len(hull.vertices) >= 3, f"Candidate {i} has {len(hull.vertices)} hull vertices, expected >=3"
            checked += 1
        assert checked > 0, "No candidates with non-degenerate XY spread"


class TestSampleContactsRejectionStats:
    """Verify rejection stats are populated."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_rejection_dict_keys(self, robot):
        kin, foot_ids = robot
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(200, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        sampler = _make_sampler(kin, foot_ids)
        _, reject = sampler(wp_mesh, np.zeros(3), buf, 50)
        assert "out_of_reach" in reject
        assert "shape_infeasible" in reject


# ---------------------------------------------------------------------------
# Multi-robot reachability test
#
# Per-foot ``(r, theta, z)`` envelope is the sampler's only pre-IK physics
# filter; if it omits any axis, polygons whose feet are outside the leg's
# workspace sneak past and waste IK iterations (and on rough/stair terrain
# they dominate downstream FootPositionError rejections). This test exercises
# every registered robot on flat + moderate-stair meshes and asserts that
# every returned polygon's feet actually lie inside the measured envelope.
# ---------------------------------------------------------------------------


_MULTI_ROBOTS = ["anymal_c", "spot", "go2"]


def _resolve_usd(usd_path: str) -> str:
    """Resolve remote URLs to a local cache path so the kinematics loader can open them."""
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    status = check_file_path(usd_path)
    if status == 0:
        pytest.skip(f"USD not found: {usd_path}")
    if status == 2:
        return retrieve_file_path(usd_path, force_download=False)
    return usd_path


@pytest.fixture(scope="module")
def _robot_presets_registered():
    """Import the robot-preset subpackage once so ``RobotArticulationCfg`` is populated."""
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    importlib.import_module("isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots")


def _build_sampler_for_robot(robot_name: str):
    """Instantiate a sampler for the named robot preset (flat-terrain sizing)."""
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.robot_presets import (
        RetargetFootBodyNamesCfg,
        RobotArticulationCfg,
    )

    robot_cfg = getattr(RobotArticulationCfg, robot_name)
    usd_path = _resolve_usd(robot_cfg.spawn.usd_path)
    kin_cfg = NewtonKinematicsCfg(
        usd_path=usd_path,
        device=DEVICE,
        default_pos=(0.0, 0.0, robot_cfg.init_state.pos[2]),
        default_joint_pos=robot_cfg.init_state.joint_pos,
    )
    kin = NewtonKinematics(kin_cfg)
    foot_names = getattr(RetargetFootBodyNamesCfg, robot_name)
    foot_ids = kin.find_body_indices(foot_names)
    sampler = SupportPolygonSampler(SupportPolygonSamplerCfg(), kin=kin, foot_body_ids=foot_ids)
    return kin, foot_ids, sampler


def _assert_feet_in_reach_envelope(buf: RetargetBuffer, sampler: SupportPolygonSampler, n: int, robot_name: str):
    """At least one rotation per group has a canonical shape near the FK manifold.

    The sampler emits ``gs`` cyclic rotations per polygon and the pipeline's
    group-collapse picks the lowest-cost rotation post-IK. The shape filter
    runs on the original (r = 0) assignment of each group, so the guarantee
    is that every group contains at least one polygon for which every foot
    has *some* FK sample within :attr:`SupportPolygonSampler._fk_shape_tol`
    (per-foot marginal NN -- the continuous analog of the old per-foot
    voxel-union grid).

    ``contact_targets`` already includes :attr:`foot_ground_offset` (it is
    the foot-body target, not the raw terrain contact). Canonicalisation
    folds translation / yaw / plane-fit pitch+roll / per-foot hip azimuth
    out of the polygon, so NN in this space is a pure shape-match.
    """
    nc = buf.num_contacts
    gs = sampler.group_size
    assert n % gs == 0, f"[{robot_name}] n={n} not divisible by group_size={gs}"
    ct = buf.contact_targets_t[: n * nc].view(n, nc, 3).detach()
    query_shape = sampler._canonicalize_shape(ct)
    fk_samples = sampler._fk_shape_samples

    chunk = 64
    max_foot_nn = torch.empty(n, device=ct.device)
    for k0 in range(0, n, chunk):
        k1 = min(k0 + chunk, n)
        diff = query_shape[k0:k1].unsqueeze(1) - fk_samples.unsqueeze(0)
        foot_dist = diff.norm(dim=-1)  # [chunk, N, nc]
        max_foot_nn[k0:k1] = foot_dist.amin(dim=1).amax(dim=-1)
    shape_ok = (max_foot_nn < sampler._fk_shape_tol).view(n // gs, gs)
    group_has_valid_rotation = shape_ok.any(dim=-1)
    assert group_has_valid_rotation.all(), (
        f"[{robot_name}] {int((~group_has_valid_rotation).sum())}/{n // gs} groups had no shape-feasible rotation"
    )


class TestMultiRobotReachablePatches:
    """Sampler returns polygons whose feet actually lie in each robot's reachable envelope."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize("robot_name", _MULTI_ROBOTS)
    def test_flat_patches_reachable(self, _robot_presets_registered, robot_name):
        kin, foot_ids, sampler = _build_sampler_for_robot(robot_name)
        wp_mesh = _make_flat_mesh()
        buf = RetargetBuffer(400, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        n_desired = 100
        n, reject = sampler(wp_mesh, np.zeros(3), buf, n_desired)
        assert n >= n_desired // 2, (
            f"[{robot_name}] flat terrain yielded only {n}/{n_desired} polygons. Rejections: {reject}"
        )
        _assert_feet_in_reach_envelope(buf, sampler, n, robot_name)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize("robot_name", _MULTI_ROBOTS)
    def test_stair_patches_reachable(self, _robot_presets_registered, robot_name):
        kin, foot_ids, sampler = _build_sampler_for_robot(robot_name)
        # Modest stairs: step height picked so every quadruped in the matrix
        # can actually reach across a step boundary (ANYmal's default stance
        # sits near its legs' downward extreme, so 15 cm steps are already
        # unreachable for that robot).
        wp_mesh = _make_stair_mesh(n_steps=6, step_height=0.08, step_depth=0.35, width=4.0)
        buf = RetargetBuffer(400, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 100)
        # Stair terrain can be tight for some robots; require only that at
        # least a handful of reachable polygons were produced (not zero).
        assert n > 0, f"[{robot_name}] stair terrain produced zero polygons. Rejections: {reject}"
        _assert_feet_in_reach_envelope(buf, sampler, n, robot_name)


class TestStairSlopeDiversity:
    """Sampler preserves slope diversity on stair terrain.

    Regression guard: an earlier z-reach filter compared world-frame foot z
    against a per-foot envelope and rejected every polygon whose feet
    spanned a step boundary, so only near-flat polygons survived. The
    current shape-space NN filter canonicalises polygons into a
    centroid-origin / plane-aligned frame before matching, so sloped
    polygons (whose shape matches an FK sample under an appropriate base
    tilt) are accepted.
    """

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize("robot_name", _MULTI_ROBOTS)
    def test_polygons_span_step_boundaries(self, _robot_presets_registered, robot_name):
        kin, foot_ids, sampler = _build_sampler_for_robot(robot_name)
        # Steps sized so *every* robot in the matrix can pitch to match:
        # Go2's standing_height (~0.33 m) is the limiter, so we pick a step
        # height that still drives a meaningful slope there. Narrow treads so
        # a larger fraction of random polygons straddle a boundary.
        step_h = 0.05
        step_d = 0.22
        wp_mesh = _make_stair_mesh(n_steps=12, step_height=step_h, step_depth=step_d, width=5.0)
        buf = RetargetBuffer(800, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
        # Morphological patch sampling consumes the global torch RNG before the
        # sampler reseeds it, so pin the global state here to keep the test
        # deterministic across pytest orderings.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        n, reject = sampler(wp_mesh, np.zeros(3), buf, 400)
        assert n >= 20, f"[{robot_name}] stair terrain produced only {n} polygons. Rejections: {reject}"

        nc = buf.num_contacts
        ct = buf.contact_targets_t[: n * nc].view(n, nc, 3).detach().cpu().numpy()
        z_spread = ct[:, :, 2].max(axis=1) - ct[:, :, 2].min(axis=1)
        # Polygons straddling at least one step boundary have z_spread ~step_h.
        # Before the base-frame fix this was ~0 for every surviving polygon.
        frac_sloped = float((z_spread > 0.5 * step_h).mean())
        assert frac_sloped >= 0.10, (
            f"[{robot_name}] only {frac_sloped * 100:.1f}% of polygons have "
            f"z_spread > {0.5 * step_h:.3f} m (p50={np.median(z_spread):.3f}, "
            f"p95={np.quantile(z_spread, 0.95):.3f}). Regression: base-frame "
            f"z-reach filter is rejecting sloped polygons."
        )
