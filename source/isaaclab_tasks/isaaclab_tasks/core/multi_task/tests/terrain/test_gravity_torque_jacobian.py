# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the gravity_torque analytic Jacobian.

Compares the kernel-evaluated Jacobian against a finite-difference Jacobian
in joint_qd space. FD is computed by perturbing joint_q in the directions
that correspond to the joint_qd basis (linear-xyz directly; angular-xyz via
left-multiplied axis-angle quaternion exponential, which matches Newton's
free-joint integration convention).
"""

from __future__ import annotations

import newton.ik as ik
import numpy as np
import pytest
import torch
import warp as wp

ANYMAL_USD = "/home/zhengyuz/Downloads/ANYmal-C/anymal_c.usd"
DEVICE = "cuda:0"


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _perturb_joint_q(jq: torch.Tensor, dof: int, eps: float) -> torch.Tensor:
    out = jq.clone()
    if dof < 3:
        out[:, dof] += eps
        return out
    if dof < 6:
        ax_idx = dof - 3
        half = 0.5 * eps
        s = float(np.sin(half))
        c = float(np.cos(half))
        dq = torch.zeros((jq.shape[0], 4), device=jq.device, dtype=jq.dtype)
        dq[:, ax_idx] = s
        dq[:, 3] = c
        out[:, 3:7] = _quat_mul_xyzw(dq, jq[:, 3:7])
        return out
    out[:, 7 + (dof - 6)] += eps
    return out


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


@pytest.mark.skipif(not wp.is_device_available(DEVICE), reason="GPU required")
def test_gravity_torque_analytic_matches_fd():
    from isaaclab_tasks.core.multi_task.kinematics import (
        NewtonKinematics,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveGravityTorqueCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.gravity_torque import (
        IKObjectiveGravityTorque,
        _compute_subtree_com,
        _gravity_torque_residuals,
    )

    DEFAULT_JPOS = {
        ".*HAA": 0.0,
        ".*F_HFE": 0.4,
        ".*H_HFE": -0.4,
        ".*F_KFE": -0.8,
        ".*H_KFE": 0.8,
    }
    kin = NewtonKinematics(
        NewtonKinematicsCfg(usd_path=ANYMAL_USD, device=DEVICE, default_pos=(0, 0, 0.6), default_joint_pos=DEFAULT_JPOS)
    )

    class _PipelineMock:
        def __init__(self, k):
            self.kin = k

    obj = IKObjectiveGravityTorque(IKObjectiveGravityTorqueCfg(weight=0.5), _PipelineMock(kin))

    N = 4
    n_rev = obj.n_rev
    n_dofs = kin.model.joint_dof_count
    n_jq = kin.model.joint_coord_count

    torch.manual_seed(0)
    jq = torch.from_numpy(kin.default_joint_q).float().to(DEVICE).unsqueeze(0).repeat(N, 1)
    jq[:, 0:2] += torch.randn(N, 2, device=DEVICE) * 0.5
    jq[:, 7:] += torch.randn(N, n_jq - 7, device=DEVICE) * 0.3

    body_q = wp.zeros((N, kin.model.body_count), dtype=wp.transform, device=DEVICE)
    subtree_com = wp.zeros((N, n_rev), dtype=wp.vec3, device=DEVICE)
    residuals = wp.zeros((N, n_rev), dtype=wp.float32, device=DEVICE)

    joint_body = wp.array(obj._parent_bodies_np, dtype=wp.int32, device=DEVICE)
    axes_local = wp.from_numpy(obj._axes_local_np, dtype=wp.vec3, device=DEVICE)
    downstream = wp.array(obj._downstream_bodies_np, dtype=wp.int32, device=DEVICE)
    offsets = wp.array(obj._downstream_offsets_np, dtype=wp.int32, device=DEVICE)
    subtree_mass = wp.array(obj._subtree_mass_np, dtype=wp.float32, device=DEVICE)
    subtree_inv_mass = wp.array(obj._subtree_inv_mass_np, dtype=wp.float32, device=DEVICE)
    body_com = wp.from_numpy(obj._body_com_np, dtype=wp.vec3, device=DEVICE)
    body_mass = wp.array(obj._body_mass_np, dtype=wp.float32, device=DEVICE)
    g_vec = wp.vec3(*obj._gravity_np.tolist())

    def _eval(jq_t: torch.Tensor) -> np.ndarray:
        kin.eval_fk_batched(wp.from_torch(jq_t.contiguous()), body_q=body_q)
        wp.launch(
            _compute_subtree_com,
            dim=[N, n_rev],
            inputs=[body_q, body_com, body_mass, downstream, offsets, subtree_inv_mass],
            outputs=[subtree_com],
        )
        wp.launch(
            _gravity_torque_residuals,
            dim=[N, n_rev],
            inputs=[
                body_q,
                subtree_com,
                subtree_mass,
                joint_body,
                axes_local,
                g_vec,
                obj.weight,
                0,
            ],
            outputs=[residuals],
        )
        wp.synchronize()
        return wp.to_torch(residuals).detach().cpu().numpy().copy()

    eps = 1e-3
    fd_qd = np.zeros((N, n_rev, n_dofs), dtype=np.float32)
    for d in range(n_dofs):
        rp = _eval(_perturb_joint_q(jq, d, eps))
        rm = _eval(_perturb_joint_q(jq, d, -eps))
        fd_qd[:, :, d] = (rp - rm) / (2.0 * eps)

    solver = kin.create_ik_solver([obj], N, jacobian_mode=ik.IKJacobianType.MIXED)
    jq_in = wp.from_torch(jq.clone().contiguous())
    jq_out = wp.zeros_like(jq_in)
    solver.step(jq_in, jq_out, iterations=1)
    wp.synchronize()
    jac = wp.to_torch(solver.jacobian).detach().cpu().numpy().copy()
    if jac.shape[1] != n_rev:
        jac = jac[:, :n_rev, :]

    abs_err = np.abs(jac - fd_qd)
    # FD with eps=1e-3 has O(eps^2) truncation + ~1e-5 relative FP noise; 5e-3
    # is comfortably above that and well below any genuine miscompute (the
    # autodiff-mismatch we caught earlier was O(10)).
    assert abs_err.max() < 5e-3, f"analytic gravity_torque Jacobian disagrees with FD: max |Δ| = {abs_err.max():.3e}"
