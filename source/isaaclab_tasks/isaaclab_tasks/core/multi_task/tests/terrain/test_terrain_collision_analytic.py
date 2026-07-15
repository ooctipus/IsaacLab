# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the analytic Jacobian of :class:`IKObjectiveMeshCollision`.

The softplus-smoothed collision residual has a closed-form Jacobian that
composes the signed-distance gradient with Newton's spatial motion
subspace. We validate it three ways:

1. Central finite differences on the base-translation and revolute
   joint coordinates (direct perturbation of ``joint_q``).
2. Parity with the existing autodiff path on every DoF column.
3. Three probe regimes -- far above terrain, at the surface,
   penetrating -- to exercise both softplus tails and both branches of
   the ``max(sign_pen, z_pen)`` subgradient.
"""

from __future__ import annotations

import newton.ik as ik
import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.core.multi_task.kinematics import (
    IKConstraintMeshClearance,
    IKObjectiveMeshCollision,
    NewtonKinematics,
    NewtonKinematicsCfg,
    collision_probes_sample,
)
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.mesh_collision import IKObjectiveMeshNonpenetration


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cuda:0"


def _make_flat_terrain_mesh(device: str):
    """A single watertight thick slab with its top face at ``z = 0``.

    Must be watertight -- ``wp.mesh_query_point``'s winding-number sign
    flips randomly across seams of non-closed meshes, which produces
    inconsistent autodiff gradients at the boundary.
    """
    slab = trimesh.creation.box(extents=(4.0, 4.0, 1.0))
    slab.apply_translation((0.0, 0.0, -0.5))
    return convert_to_warp_mesh(slab.vertices, slab.faces, device=device)


@pytest.fixture(scope="module")
def setup(canonical_topology_mjcf):
    kin = NewtonKinematics(
        NewtonKinematicsCfg(mjcf_path=str(canonical_topology_mjcf), device=DEVICE, collapse_fixed_joints=False)
    )
    foot_ids = [i for i, n in enumerate(kin.body_names) if "FOOT" in n.upper()]
    wp_mesh = _make_flat_terrain_mesh(DEVICE)
    return kin, foot_ids, wp_mesh


def _make_optimizer(
    kin: NewtonKinematics,
    wp_mesh,
    foot_ids: list[int],
    jacobian_mode: ik.IKJacobianType,
    *,
    weight: float = 3.0,
    margin: float = 0.05,
    n_samples: int = 4,
    contact_confidence: float = 1.0,
) -> tuple:
    """Build a single-problem LM optimizer with only the collision objective.

    Returns ``(impl, obj)`` where ``impl`` is the low-level
    :class:`newton.ik.IKOptimizerLM` and ``obj`` is the collision objective.
    """
    from newton.ik import IKOptimizerLM

    probe_bodies, probe_offsets, probe_slots = collision_probes_sample(kin.builder, foot_ids, n_samples)
    obstacle_pose = torch.zeros(1, 7, dtype=torch.float32, device=DEVICE)
    obstacle_pose[:, 6] = 1.0
    confidence = torch.full((1, len(foot_ids)), contact_confidence, dtype=torch.float32, device=DEVICE)
    obj = IKObjectiveMeshCollision(
        probe_offsets=probe_offsets,
        probe_bodies=probe_bodies,
        probe_affects_dof=kin.topology.body_dof_ancestry[probe_bodies],
        mesh=wp_mesh,
        obstacle_pose=obstacle_pose,
        weight=weight,
        margin=margin,
        max_distance=2.0,
        probe_contact_slots=probe_slots,
        contact_confidence=confidence,
        one_sided_up_axis=(0.0, 0.0, 1.0),
    )
    impl = IKOptimizerLM(
        model=kin.model,
        n_batch=1,
        objectives=[obj],
        jacobian_mode=jacobian_mode,
    )
    return impl, obj


def _make_clearance_optimizer(
    kin: NewtonKinematics,
    wp_mesh,
    foot_ids: list[int],
    *,
    n_samples: int = 4,
) -> tuple:
    """Build a single-problem optimizer exposing physical signed clearance."""
    from newton.ik import IKOptimizerLM

    probe_bodies, probe_offsets, _ = collision_probes_sample(kin.builder, foot_ids, n_samples)
    obstacle_pose = torch.zeros(1, 7, dtype=torch.float32, device=DEVICE)
    obstacle_pose[:, 6] = 1.0
    constraint = IKConstraintMeshClearance(
        probe_offsets=probe_offsets,
        probe_bodies=probe_bodies,
        probe_affects_dof=kin.topology.body_dof_ancestry[probe_bodies],
        mesh=wp_mesh,
        obstacle_pose=obstacle_pose,
        max_distance=2.0,
        one_sided_up_axis=(0.0, 0.0, 1.0),
    )
    optimizer = IKOptimizerLM(
        model=kin.model,
        n_batch=1,
        objectives=[constraint],
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return optimizer, constraint


def _make_nonpenetration_optimizer(
    kin: NewtonKinematics,
    wp_mesh,
    foot_ids: list[int],
    *,
    n_samples: int = 4,
    tolerance_m: float = 0.002,
    maximum_penetration_m: float = 0.0,
) -> tuple:
    """Build a single-problem optimizer with an ungated nonpenetration hinge."""
    from newton.ik import IKOptimizerLM

    probe_bodies, probe_offsets, _ = collision_probes_sample(kin.builder, foot_ids, n_samples)
    obstacle_pose = torch.zeros(1, 7, dtype=torch.float32, device=DEVICE)
    obstacle_pose[:, 6] = 1.0
    objective = IKObjectiveMeshNonpenetration(
        probe_offsets=probe_offsets,
        probe_bodies=probe_bodies,
        probe_affects_dof=kin.topology.body_dof_ancestry[probe_bodies],
        mesh=wp_mesh,
        obstacle_pose=obstacle_pose,
        tolerance_m=tolerance_m,
        maximum_penetration_m=maximum_penetration_m,
        max_distance=2.0,
        one_sided_up_axis=(0.0, 0.0, 1.0),
    )
    optimizer = IKOptimizerLM(
        model=kin.model,
        n_batch=1,
        objectives=[objective],
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return optimizer, objective


def _compute_residuals(impl, jq_np: np.ndarray) -> np.ndarray:
    """Run FK + residuals for a single problem, returning the residual row."""
    jq = wp.array(jq_np[None, :], dtype=wp.float32, device=DEVICE)
    impl._compute_residuals(jq)
    return impl.residuals.numpy()[0].copy()


def _compute_jacobian(impl, jq_np: np.ndarray) -> np.ndarray:
    """Run FK + residuals + Jacobian, returning the Jacobian slab."""
    jq = wp.array(jq_np[None, :], dtype=wp.float32, device=DEVICE)
    impl._compute_residuals(jq)
    ctx = impl._ctx_solver(jq)
    impl._jacobian_at(ctx)
    return impl.jacobian.numpy()[0].copy()


def _coord_to_dof(kin: NewtonKinematics) -> dict[int, int]:
    """Map joint_q coord index to joint_qd DoF index (revolute + base translation only)."""
    # Free root has joint_q[0:3] = translation (DoFs 0..2), joint_q[3:7] = quat (DoFs 3..5).
    # Subsequent revolute joints have coord = dof + 1 (quat adds an extra coord).
    mapping: dict[int, int] = {0: 0, 1: 1, 2: 2}  # base translation
    q_start = kin.model.joint_q_start.numpy()
    qd_start = kin.model.joint_qd_start.numpy()
    joint_type = kin.model.joint_type.numpy()
    for j in range(1, kin.model.joint_count):
        if int(joint_type[j]) != 1:
            continue
        mapping[int(q_start[j])] = int(qd_start[j])
    return mapping


def _finite_difference_jacobian(
    impl,
    jq_np: np.ndarray,
    coord_to_dof: dict[int, int],
    eps: float = 1.0e-4,
) -> tuple[np.ndarray, list[int]]:
    """Central finite differences for the subset of DoFs we can perturb directly.

    Returns ``(J_fd, tested_dofs)`` where ``J_fd[probe_idx, dof_idx]`` is the
    FD gradient and ``tested_dofs`` lists the DoF columns that were filled.
    """
    n_res = impl.n_residuals
    n_dofs = impl.n_dofs
    J_fd = np.zeros((n_res, n_dofs), dtype=np.float64)
    tested = []
    for coord_idx, dof_idx in sorted(coord_to_dof.items()):
        q_plus = jq_np.copy()
        q_minus = jq_np.copy()
        q_plus[coord_idx] += eps
        q_minus[coord_idx] -= eps
        r_plus = _compute_residuals(impl, q_plus).astype(np.float64)
        r_minus = _compute_residuals(impl, q_minus).astype(np.float64)
        J_fd[:, dof_idx] = (r_plus - r_minus) / (2.0 * eps)
        tested.append(dof_idx)
    return J_fd, tested


def _make_test_config(kin: NewtonKinematics, base_z: float) -> np.ndarray:
    jq = kin.default_joint_q.copy().astype(np.float32)
    jq[2] = base_z
    return jq


class TestTerrainCollisionAnalytic:
    """Validate the analytic Jacobian against FD and autodiff."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize(
        "regime,base_z",
        [("far_above", 2.0), ("near_surface", 0.54), ("penetrating", 0.0)],
    )
    def test_analytic_matches_autodiff(self, setup, regime, base_z):
        kin, foot_ids, wp_mesh = setup
        jq = _make_test_config(kin, base_z)

        impl_a, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)
        impl_ad, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.AUTODIFF)

        J_analytic = _compute_jacobian(impl_a, jq)
        J_autodiff = _compute_jacobian(impl_ad, jq)

        diff = np.abs(J_analytic - J_autodiff)
        max_err = float(diff.max())
        scale = float(np.abs(J_autodiff).max()) + 1e-8
        rel_err = max_err / scale
        assert max_err < 1e-3 or rel_err < 1e-3, (
            f"[{regime}] analytic vs autodiff max_abs={max_err:.3e} (rel={rel_err:.3e})"
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize(
        "regime,base_z",
        [("far_above", 2.0), ("near_surface", 0.54), ("penetrating", 0.0)],
    )
    def test_analytic_matches_fd(self, setup, regime, base_z):
        kin, foot_ids, wp_mesh = setup
        jq = _make_test_config(kin, base_z)

        impl_a, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)
        impl_fd, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.AUTODIFF)
        # ^ reuse an autodiff-mode impl for residual-only evaluations in FD

        J_analytic = _compute_jacobian(impl_a, jq)
        coord_to_dof = _coord_to_dof(kin)
        J_fd, tested = _finite_difference_jacobian(impl_fd, jq, coord_to_dof)

        # Compare only the DoFs we perturbed directly.
        J_a_subset = J_analytic[:, tested].astype(np.float64)
        J_fd_subset = J_fd[:, tested]
        diff = np.abs(J_a_subset - J_fd_subset)
        max_err = float(diff.max())
        scale = float(np.abs(J_fd_subset).max()) + 1e-8
        rel_err = max_err / scale
        # FD on softplus with margin=0.05 is noisy -- allow slightly looser
        # absolute tolerance but require tight relative error when the gradient
        # magnitude is nontrivial.
        assert max_err < 5e-3 or rel_err < 5e-3, (
            f"[{regime}] analytic vs FD max_abs={max_err:.3e} (rel={rel_err:.3e}) on {len(tested)} DoFs"
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_penetrating_has_nonzero_gradient(self, setup):
        """Sanity: the penetrating regime must produce nontrivial gradients.

        Guards against a silent bug where all three regimes report zero
        Jacobians -- which would also pass ``test_analytic_matches_fd``.
        """
        kin, foot_ids, wp_mesh = setup
        jq = _make_test_config(kin, 0.0)

        impl_a, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)
        J = _compute_jacobian(impl_a, jq)
        assert np.abs(J).max() > 1e-2, f"expected nonzero gradient on penetrating config, got {np.abs(J).max():.3e}"

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_contact_confidence_scales_residual_and_jacobian(self, setup):
        """A mapped collision row is scaled by sqrt(1-c) for c in {0, .25, 1}."""
        kin, foot_ids, wp_mesh = setup
        joint_q = _make_test_config(kin, 0.0)
        evidence = {}
        mapped = None
        for confidence in (0.0, 0.25, 1.0):
            optimizer, objective = _make_optimizer(
                kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC, contact_confidence=confidence
            )
            evidence[confidence] = (_compute_residuals(optimizer, joint_q), _compute_jacobian(optimizer, joint_q))
            mapped = objective._probe_contact_slots_np >= 0

        assert mapped is not None and mapped.any() and (~mapped).any()
        residual_zero, jacobian_zero = evidence[0.0]
        assert np.abs(residual_zero[mapped]).max() > 1.0e-4
        assert np.abs(jacobian_zero[mapped]).max() > 1.0e-2
        scale = np.sqrt(0.75)
        np.testing.assert_allclose(evidence[0.25][0][mapped], scale * residual_zero[mapped], atol=2.0e-6, rtol=2.0e-6)
        np.testing.assert_allclose(evidence[0.25][1][mapped], scale * jacobian_zero[mapped], atol=2.0e-6, rtol=2.0e-6)
        np.testing.assert_array_equal(evidence[1.0][0][mapped], 0.0)
        np.testing.assert_array_equal(evidence[1.0][1][mapped], 0.0)
        np.testing.assert_allclose(evidence[0.25][0][~mapped], residual_zero[~mapped])
        np.testing.assert_allclose(evidence[1.0][1][~mapped], jacobian_zero[~mapped])


class TestTerrainClearanceConstraint:
    """Validate the hard feature physical units and analytic derivative."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_signed_clearance_is_ungated_and_measured_in_meters(self, setup):
        kin, foot_ids, wp_mesh = setup
        optimizer, _ = _make_clearance_optimizer(kin, wp_mesh, foot_ids)
        penetrating = _make_test_config(kin, 0.50)
        near_surface = _make_test_config(kin, 0.54)
        raised = near_surface.copy()
        raised[2] += 0.1

        penetration = _compute_residuals(optimizer, penetrating)
        near_surface_clearance = _compute_residuals(optimizer, near_surface)
        raised_clearance = _compute_residuals(optimizer, raised)

        assert penetration.max() > 0.0
        np.testing.assert_allclose(raised_clearance - near_surface_clearance, -0.1, atol=2.0e-5, rtol=0.0)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_analytic_vertical_derivative_matches_finite_difference(self, setup):
        kin, foot_ids, wp_mesh = setup
        optimizer, _ = _make_clearance_optimizer(kin, wp_mesh, foot_ids)
        joint_q = _make_test_config(kin, 0.54)
        analytic = _compute_jacobian(optimizer, joint_q)[:, 2]
        epsilon = 1.0e-4
        above = joint_q.copy()
        below = joint_q.copy()
        above[2] += epsilon
        below[2] -= epsilon
        finite_difference = (_compute_residuals(optimizer, above) - _compute_residuals(optimizer, below)) / (
            2.0 * epsilon
        )

        np.testing.assert_allclose(analytic, finite_difference, atol=5.0e-3, rtol=5.0e-3)


class TestTerrainNonpenetrationObjective:
    """Validate the ungated zero-at-contact hinge and analytic derivative."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_clear_and_allowed_boundary_have_exactly_zero_rows(self, setup):
        kin, foot_ids, wp_mesh = setup
        optimizer, _ = _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids)
        clear = _make_test_config(kin, 2.0)

        np.testing.assert_array_equal(_compute_residuals(optimizer, clear), 0.0)
        np.testing.assert_array_equal(_compute_jacobian(optimizer, clear), 0.0)

        clearance_optimizer, _ = _make_clearance_optimizer(kin, wp_mesh, foot_ids)
        penetrating = _make_test_config(kin, 0.50)
        allowed_depth = float(_compute_residuals(clearance_optimizer, penetrating).max())
        assert allowed_depth > 0.0
        boundary_optimizer, _ = _make_nonpenetration_optimizer(
            kin, wp_mesh, foot_ids, maximum_penetration_m=allowed_depth
        )
        np.testing.assert_array_equal(_compute_residuals(boundary_optimizer, penetrating), 0.0)
        np.testing.assert_array_equal(_compute_jacobian(boundary_optimizer, penetrating), 0.0)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_penetration_is_normalized_by_tolerance(self, setup):
        kin, foot_ids, wp_mesh = setup
        tolerance_m = 0.002
        clearance_optimizer, _ = _make_clearance_optimizer(kin, wp_mesh, foot_ids)
        optimizer, _ = _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids, tolerance_m=tolerance_m)
        penetrating = _make_test_config(kin, 0.50)

        physical_depth = _compute_residuals(clearance_optimizer, penetrating)
        residual = _compute_residuals(optimizer, penetrating)
        expected = np.maximum(physical_depth, 0.0) / tolerance_m

        assert residual.max() > 0.0
        np.testing.assert_allclose(residual, expected, atol=2.0e-5, rtol=2.0e-6)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_analytic_vertical_derivative_matches_finite_difference_away_from_knee(self, setup):
        kin, foot_ids, wp_mesh = setup
        optimizer, _ = _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids)
        joint_q = _make_test_config(kin, 0.50)
        residual = _compute_residuals(optimizer, joint_q)
        active = residual > 1.0
        assert active.any()

        analytic = _compute_jacobian(optimizer, joint_q)[:, 2]
        epsilon = 1.0e-4
        above = joint_q.copy()
        below = joint_q.copy()
        above[2] += epsilon
        below[2] -= epsilon
        finite_difference = (_compute_residuals(optimizer, above) - _compute_residuals(optimizer, below)) / (
            2.0 * epsilon
        )

        np.testing.assert_allclose(analytic[active], finite_difference[active], atol=0.5, rtol=5.0e-3)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_active_contact_confidence_does_not_gate_nonpenetration(self, setup):
        kin, foot_ids, wp_mesh = setup
        joint_q = _make_test_config(kin, 0.0)
        soft_optimizer, soft_objective = _make_optimizer(
            kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC, contact_confidence=1.0
        )
        hinge_optimizer, _ = _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids)
        mapped = soft_objective._probe_contact_slots_np >= 0
        soft_residual = _compute_residuals(soft_optimizer, joint_q)
        hinge_residual = _compute_residuals(hinge_optimizer, joint_q)

        assert mapped.any()
        np.testing.assert_array_equal(soft_residual[mapped], 0.0)
        assert hinge_residual[mapped].max() > 0.0

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_invalid_hinge_scales_are_rejected(self, setup):
        kin, foot_ids, wp_mesh = setup
        with pytest.raises(ValueError, match="tolerance_m"):
            _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids, tolerance_m=0.0)
        with pytest.raises(ValueError, match="maximum_penetration_m"):
            _make_nonpenetration_optimizer(kin, wp_mesh, foot_ids, maximum_penetration_m=-1.0e-3)
