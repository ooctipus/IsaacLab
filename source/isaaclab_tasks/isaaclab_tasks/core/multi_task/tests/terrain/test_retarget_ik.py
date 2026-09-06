# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Stage 3: batched IK retargeting.

FK round-trip: IK result -> FK -> foot positions match targets.
Requires Newton + Warp (no IsaacSim).
"""

import importlib

import newton.ik as ik
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


DEVICE = "cuda:0"
FOOT_ERR_TOL = 0.02


def _resolve_usd(usd_path: str) -> str:
    """Resolve the declared robot asset to a local path or skip when unavailable."""
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    status = check_file_path(usd_path)
    if status == 0:
        pytest.skip(f"USD not found: {usd_path}")
    return retrieve_file_path(usd_path, force_download=False) if status == 2 else usd_path


def _make_solver(model, foot_ids, base_body_id, n_problems):
    device = str(model.device)
    contact_objs = [
        ik.IKObjectivePosition(
            link_index=cid,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device),
            weight=1.0,
        )
        for cid in foot_ids
    ]
    base_pos_obj = ik.IKObjectivePosition(
        link_index=base_body_id,
        link_offset=wp.vec3(0, 0, 0),
        target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device),
        weight=0.05,
    )
    base_rot_obj = ik.IKObjectiveRotation(
        link_index=base_body_id,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.zeros(n_problems, dtype=wp.vec4, device=device),
        weight=0.5,
    )
    jl_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=n_problems,
        objectives=[*contact_objs, base_pos_obj, base_rot_obj, jl_obj],
        optimizer=ik.IKOptimizer.LM,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return solver, contact_objs, base_pos_obj, base_rot_obj


def _solve(solver, buf, contact_objs, base_pos_obj, base_rot_obj, jcc, N):
    """Fill targets from buffer and run solver.step."""
    buf.scatter_contact_targets(contact_objs, N)
    wp.copy(base_pos_obj.target_positions, buf.base_target_pos, count=N)
    wp.copy(base_rot_obj.target_rotations, buf.base_target_rot, count=N)
    jq_in = wp.from_torch(buf.joint_q_init_t[:N].contiguous())
    jq_out = wp.from_torch(buf.joint_q_result_t[:N].contiguous())
    solver.step(jq_in, jq_out, iterations=50)
    buf.joint_q_result_t[:N] = wp.to_torch(jq_out)


def _fill_buf(buf, foot_pos, jq, N, nc):
    """Write targets into buffer via torch views."""
    td = torch.device(DEVICE)
    fp_t = torch.from_numpy(foot_pos).to(td)
    jq_t = torch.from_numpy(jq).to(td)

    for i in range(N):
        buf.contact_targets_t[i * nc : (i + 1) * nc] = fp_t if fp_t.ndim == 2 else fp_t[i]
    buf.base_target_pos_t[:N] = torch.tensor([[0, 0, 0.6]], device=td).expand(N, -1)
    buf.base_target_rot_t[:N] = torch.tensor([[0, 0, 0, 1]], device=td).expand(N, -1)
    buf.joint_q_init_t[:N] = jq_t.unsqueeze(0).expand(N, -1)
    buf._geom_valid[:N] = True


@pytest.fixture(scope="module")
def robot_setup():
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots.anymal_c")
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots.robot_presets import (
        FootBodyNamesCfg,
        RobotArticulationCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget.sampler_base import resolve_contact_body_names

    robot_cfg = RobotArticulationCfg.anymal_c
    kin = NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=_resolve_usd(robot_cfg.spawn.usd_path),
            device=DEVICE,
            default_pos=robot_cfg.init_state.pos,
            default_joint_pos=robot_cfg.init_state.joint_pos,
        )
    )
    foot_names = resolve_contact_body_names(FootBodyNamesCfg.anymal_c, kin.body_names)
    foot_ids = kin.find_body_indices(foot_names)
    foot_pos = np.array([kin.default_body_q[body_id][:3] for body_id in foot_ids])
    return kin, foot_ids, kin.default_joint_q, foot_pos


class TestIKIdentity:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_identity(self, robot_setup):
        kin, foot_ids, jq, foot_pos = robot_setup
        N, nc = 1, len(foot_ids)
        buf = RetargetBuffer(N, kin.model.joint_coord_count, kin.model.body_count, nc, DEVICE)
        buf.num_written = N
        buf.num_geometry_valid = N
        _fill_buf(buf, foot_pos, jq, N, nc)

        solver, co, bpo, bro = _make_solver(kin.model, foot_ids, 0, N)
        _solve(solver, buf, co, bpo, bro, kin.model.joint_coord_count, N)

        result = buf.joint_q_result_t[0].cpu().numpy()
        state = kin.eval_fk(wp.array(result, dtype=float, device=DEVICE))
        bq = state.body_q.numpy()
        for f_idx, fid in enumerate(foot_ids):
            err = np.linalg.norm(bq[fid][:3] - foot_pos[f_idx])
            assert err < FOOT_ERR_TOL, f"Foot {f_idx} error={err:.4f}m"


class TestIKPerturbation:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_perturbation(self, robot_setup):
        kin, foot_ids, jq, foot_pos = robot_setup
        N, nc = 1, len(foot_ids)

        perturbed = foot_pos.copy()
        perturbed[0, 2] += 0.10
        perturbed[2, 2] += 0.10
        perturbed[1, 2] -= 0.05
        perturbed[3, 2] -= 0.05

        buf = RetargetBuffer(N, kin.model.joint_coord_count, kin.model.body_count, nc, DEVICE)
        buf.num_written = N
        buf.num_geometry_valid = N
        _fill_buf(buf, perturbed, jq, N, nc)

        solver, co, bpo, bro = _make_solver(kin.model, foot_ids, 0, N)
        _solve(solver, buf, co, bpo, bro, kin.model.joint_coord_count, N)

        result = buf.joint_q_result_t[0].cpu().numpy()
        state = kin.eval_fk(wp.array(result, dtype=float, device=DEVICE))
        bq = state.body_q.numpy()
        for f_idx, fid in enumerate(foot_ids):
            err = np.linalg.norm(bq[fid][:3] - perturbed[f_idx])
            assert err < FOOT_ERR_TOL, f"Foot {f_idx} error={err:.4f}m"

        rev_diff = np.abs(result[7:] - jq[7:])
        assert rev_diff.max() > 0.01, "Joint angles did not change despite perturbed targets"


class TestIKBatched:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_batched_consistency(self, robot_setup):
        kin, foot_ids, jq, foot_pos = robot_setup
        N, nc = 5, len(foot_ids)

        buf = RetargetBuffer(N, kin.model.joint_coord_count, kin.model.body_count, nc, DEVICE)
        buf.num_written = N
        buf.num_geometry_valid = N

        td = torch.device(DEVICE)
        fp_t = torch.from_numpy(foot_pos).to(td)
        jq_t = torch.from_numpy(jq).to(td)

        for i in range(N):
            dz = 0.01 * i
            buf.contact_targets_t[i * nc : (i + 1) * nc] = fp_t + torch.tensor([0, 0, dz], device=td)
            buf.base_target_pos_t[i] = torch.tensor([0, 0, 0.6 + dz], device=td)
            buf.base_target_rot_t[i] = torch.tensor([0, 0, 0, 1], device=td)
            buf.joint_q_init_t[i] = jq_t
        buf._geom_valid[:N] = True

        solver, co, bpo, bro = _make_solver(kin.model, foot_ids, 0, N)
        _solve(solver, buf, co, bpo, bro, kin.model.joint_coord_count, N)

        for i in range(N):
            result = buf.joint_q_result_t[i].cpu().numpy()
            state = kin.eval_fk(wp.array(result, dtype=float, device=DEVICE))
            bq = state.body_q.numpy()
            for f_idx, fid in enumerate(foot_ids):
                target = foot_pos[f_idx] + [0, 0, 0.01 * i]
                err = float(np.linalg.norm(bq[fid][:3] - target))
                assert err < FOOT_ERR_TOL, f"Batch {i} foot {f_idx} error={err:.4f}m"
