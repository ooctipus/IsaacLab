# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the stability_margin analytic Jacobian.

Hinge residual ``r = w * max(0, -min_signed)`` where ``min_signed`` is the
smallest signed distance from the CoM (xy projection) to any active-edge
of the support polygon (CCW-ordered). The Jacobian is zero inside the
polygon and only the active edge contributes when outside.

Constructs a config with the CoM clearly outside the polygon by raising
the front-foot z and tilting the base back, then compares the analytic
kernel against finite-difference in joint_qd space.
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
def test_stability_margin_analytic_matches_fd():
    from isaaclab_tasks.core.multi_task.kinematics import (
        NewtonKinematics,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveStabilityMarginCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.stability_margin import (
        IKObjectiveStabilityMargin,
        _compute_joint_subtree_origin_coms,
        _stability_margin_residuals,
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

    foot_body_ids = [i for i, n in enumerate(kin.body_names) if "FOOT" in n.upper()]
    n_feet = len(foot_body_ids)
    n_dofs = kin.model.joint_dof_count
    n_jq = kin.model.joint_coord_count

    class _BufMock:
        # Per-problem is_contact, slot order matches foot_body_ids.
        def __init__(self, n: int, n_feet: int, device: str):
            self.is_contact_t = torch.ones(n * n_feet, dtype=torch.uint8, device=device)

    class _PipelineMock:
        def __init__(self, k, foot_ids: list[int], n: int):
            self.kin = k
            self.foot_body_ids = foot_ids
            self.buffer = _BufMock(n, len(foot_ids), DEVICE)

    N = 4
    pipe = _PipelineMock(kin, foot_body_ids, N)
    obj = IKObjectiveStabilityMargin(IKObjectiveStabilityMarginCfg(weight=1.0), pipe)

    # Configurations that force the CoM well outside the polygon: take
    # the URDF default and pitch the base ~30 deg nose-up. Pure base
    # translation moves the whole robot rigidly (CoM and polygon track),
    # so we need orientation change or asymmetric joint angles to break
    # the static balance.
    torch.manual_seed(0)
    jq = torch.from_numpy(kin.default_joint_q).float().to(DEVICE).unsqueeze(0).repeat(N, 1)
    pitch = 1.5 + 0.05 * torch.randn(N, device=DEVICE)
    half = 0.5 * pitch
    qy = torch.sin(half)
    qw = torch.cos(half)
    jq[:, 3] = 0.0
    jq[:, 4] = qy
    jq[:, 5] = 0.0
    jq[:, 6] = qw
    jq[:, 7:] += torch.randn(N, n_jq - 7, device=DEVICE) * 0.1

    body_q = wp.zeros((N, kin.model.body_count), dtype=wp.transform, device=DEVICE)
    residuals = wp.zeros((N, 1), dtype=wp.float32, device=DEVICE)

    # Allocate scratch matching what the objective uses internally.
    is_contact = wp.from_numpy(np.ones((N, n_feet), dtype=np.uint8), dtype=wp.uint8, device=DEVICE)
    foot_body_indices = wp.array(np.asarray(foot_body_ids, dtype=np.int32), dtype=wp.int32, device=DEVICE)
    body_mass = wp.array(obj._body_mass_np, dtype=wp.float32, device=DEVICE)
    scratch_xy = wp.zeros((N, n_feet), dtype=wp.vec2, device=DEVICE)
    scratch_slot = wp.zeros((N, n_feet), dtype=wp.int32, device=DEVICE)
    a_slot = wp.zeros(N, dtype=wp.int32, device=DEVICE)
    b_slot = wp.zeros(N, dtype=wp.int32, device=DEVICE)
    e_xy = wp.zeros(N, dtype=wp.vec2, device=DEVICE)
    p_xy = wp.zeros(N, dtype=wp.vec2, device=DEVICE)
    edge_len = wp.zeros(N, dtype=wp.float32, device=DEVICE)
    subtree_bodies = wp.array(obj._joint_subtree_bodies_np, dtype=wp.int32, device=DEVICE)
    subtree_offsets = wp.array(obj._joint_subtree_offsets_np, dtype=wp.int32, device=DEVICE)
    subtree_inv_mass = wp.array(obj._joint_subtree_inv_mass_np, dtype=wp.float32, device=DEVICE)
    subtree_com_buf = wp.zeros((N, kin.model.joint_count), dtype=wp.vec3, device=DEVICE)

    def _eval(jq_t: torch.Tensor) -> np.ndarray:
        kin.eval_fk_batched(wp.from_torch(jq_t.contiguous()), body_q=body_q)
        wp.launch(
            _compute_joint_subtree_origin_coms,
            dim=[N, kin.model.joint_count],
            inputs=[body_q, body_mass, subtree_bodies, subtree_offsets, subtree_inv_mass],
            outputs=[subtree_com_buf],
        )
        wp.launch(
            _stability_margin_residuals,
            dim=N,
            inputs=[
                body_q,
                body_mass,
                kin.model.body_count,
                foot_body_indices,
                is_contact,
                scratch_xy,
                scratch_slot,
                n_feet,
                obj._total_mass_inv,
                obj.weight,
                0,
            ],
            outputs=[residuals, a_slot, b_slot, e_xy, p_xy, edge_len],
        )
        wp.synchronize()
        return wp.to_torch(residuals).detach().cpu().numpy().copy()

    r0 = _eval(jq)
    print(f"baseline residuals: {r0[:, 0].tolist()}")
    assert np.all(r0[:, 0] > 0.05), "test config is supposed to be outside polygon"

    eps = 1e-3
    fd_qd = np.zeros((N, 1, n_dofs), dtype=np.float32)
    for d in range(n_dofs):
        rp = _eval(_perturb_joint_q(jq, d, eps))
        rm = _eval(_perturb_joint_q(jq, d, -eps))
        fd_qd[:, :, d] = (rp - rm) / (2.0 * eps)

    # Run analytic via solver (which calls compute_residuals → compute_jacobian_analytic).
    solver = kin.create_ik_solver([obj], N, jacobian_mode=ik.IKJacobianType.MIXED)
    jq_in = wp.from_torch(jq.clone().contiguous())
    jq_out = wp.zeros_like(jq_in)
    solver.step(jq_in, jq_out, iterations=1)
    wp.synchronize()
    jac = wp.to_torch(solver.jacobian).detach().cpu().numpy().copy()
    jac = jac[:, :1, :]

    abs_err = np.abs(jac - fd_qd)
    # Stability_margin's residual is piecewise-smooth (active-edge changes
    # are cusps). Tolerance 1e-2 covers FD noise + occasional epsilon-scale
    # active-edge swap in the FD perturbation; genuinely-wrong gradients
    # are typically O(1) on this objective.
    print(f"analytic vs FD: max={abs_err.max():.4e}, median={np.median(abs_err):.4e}")
    if abs_err.max() >= 1e-2:
        flat = abs_err.flatten()
        for fi in np.argsort(-flat)[:5]:
            idx = np.unravel_index(fi, abs_err.shape)
            print(f"  worst {idx}: FD={fd_qd[idx]:.4e}  analytic={jac[idx]:.4e}")
    assert abs_err.max() < 1e-2, f"analytic stability_margin Jacobian disagrees with FD: max |Δ| = {abs_err.max():.3e}"
