# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the analytic Jacobian of :class:`IKObjectiveTerrainCollision`.

The softplus-smoothed collision residual has a closed-form Jacobian that
composes the signed-distance gradient with Newton's spatial motion
subspace. We validate it with:

1. Central finite differences on the base-translation and revolute
   joint coordinates (direct perturbation of ``joint_q``).
2. Three probe regimes -- far above terrain, at the surface,
   penetrating -- to exercise both softplus tails and both branches of
   the ``max(sign_pen, z_pen)`` subgradient.

Warp mesh-query kernels do not support backward evaluation, so constructing
this objective with Newton's explicit autodiff mode must fail clearly instead
of silently producing zero Jacobians.
"""

from __future__ import annotations

import newton.ik as ik
import numpy as np
import pytest
import trimesh
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

from isaaclab_tasks.core.multi_task.kinematics import (
    IKObjectiveTerrainCollision,
    NewtonKinematics,
    NewtonKinematicsCfg,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


ANYMAL_USD = "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
DEVICE = "cuda:0"
DEFAULT_JPOS = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
}


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
def setup():
    kin = NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=ANYMAL_USD,
            device=DEVICE,
            default_pos=(0, 0, 0.6),
            default_joint_pos=DEFAULT_JPOS,
        )
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
) -> tuple:
    """Build a single-problem LM optimizer with only the collision objective.

    Returns ``(impl, obj)`` where ``impl`` is the low-level
    :class:`newton.ik.IKOptimizerLM` and ``obj`` is the collision objective.
    """
    from types import SimpleNamespace

    from newton._src.sim.ik.ik_lm_optimizer import IKOptimizerLM

    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveTerrainCollisionCfg,
    )

    cfg = IKObjectiveTerrainCollisionCfg(weight=weight, margin=margin, n_samples=n_samples)
    buffer = RetargetBuffer(1, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device=DEVICE)
    pipeline = SimpleNamespace(kin=kin, foot_body_ids=foot_ids, buffer=buffer)
    obj = IKObjectiveTerrainCollision(cfg, pipeline, wp_mesh)
    impl = IKOptimizerLM(
        model=kin.model,
        n_batch=1,
        objectives=[obj],
        jacobian_mode=jacobian_mode,
    )
    return impl, obj


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
    """Validate the analytic Jacobian against finite differences."""

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_rejects_unsupported_autodiff(self, setup):
        kin, foot_ids, wp_mesh = setup
        with pytest.raises(ValueError, match="does not support autodiff"):
            _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.AUTODIFF)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    @pytest.mark.parametrize(
        "regime,base_z",
        [("far_above", 2.0), ("near_surface", 0.3), ("penetrating", -0.2)],
    )
    def test_analytic_matches_fd(self, setup, regime, base_z):
        kin, foot_ids, wp_mesh = setup
        jq = _make_test_config(kin, base_z)

        impl_a, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)
        impl_fd, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)

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
        jq = _make_test_config(kin, -0.2)

        impl_a, _ = _make_optimizer(kin, wp_mesh, foot_ids, ik.IKJacobianType.ANALYTIC)
        J = _compute_jacobian(impl_a, jq)
        assert np.abs(J).max() > 1e-2, f"expected nonzero gradient on penetrating config, got {np.abs(J).max():.3e}"
