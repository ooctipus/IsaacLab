# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Stage 4: validate_results with user-defined criteria."""

import newton
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer
from isaaclab_tasks.core.multi_task.terrain.retarget.pipeline import _validate_results as validate_results


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


@pytest.fixture(scope="module")
def robot_setup():
    kin = NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=ANYMAL_USD,
            device=DEVICE,
            default_pos=(0, 0, 0.6),
            default_joint_pos=DEFAULT_JPOS,
        )
    )
    foot_ids = [i for i, n in enumerate(kin.body_names) if "FOOT" in n.upper()]
    foot_pos = np.array([kin.default_body_q[fid][:3] for fid in foot_ids])
    return kin, foot_ids, kin.default_joint_q, foot_pos, kin.default_body_q


def _fill_buf(buf, jq, foot_pos, device):
    nc = len(foot_pos)
    td = torch.device(device)
    buf.joint_q_result_t[0] = torch.from_numpy(jq).to(td)
    buf.contact_targets_t[:nc] = torch.from_numpy(foot_pos).to(td)
    buf._geom_valid[0] = True


def _make_foot_err_criterion(kin, foot_ids, max_err=0.02):
    def check(buffer, N):
        nc = len(foot_ids)
        nb = kin.model.body_count
        tpl = newton.ModelBuilder()
        tpl.add_usd(kin.usd_path, collapse_fixed_joints=False)
        bldr = newton.ModelBuilder()
        for _ in range(N):
            bldr.add_world(tpl)
        fk_m = bldr.finalize(device=buffer.device)
        jq_t = buffer.joint_q_result_t[:N].contiguous().view(-1)
        fk_m.joint_q = wp.from_torch(jq_t)
        st = fk_m.state()
        newton.eval_fk(fk_m, fk_m.joint_q, wp.zeros(fk_m.joint_dof_count, dtype=float, device=buffer.device), st)
        body_q = wp.to_torch(st.body_q).view(N, nb, 7)  # type: ignore[arg-type]
        ct = buffer.contact_targets_t[: N * nc].view(N, nc, 3)
        idx = torch.tensor(foot_ids, device=buffer.device, dtype=torch.long)
        err = (body_q[:, idx, :3] - ct).norm(dim=-1).max(dim=-1).values
        return err <= max_err

    return check


def _make_joint_margin_criterion(kin, margin=0.1):
    def check(buffer, N):
        jl = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[arg-type]
        ju = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[arg-type]
        lo, hi = jl[6:], ju[6:]
        n_rev = lo.shape[0]
        jq = buffer.joint_q_result_t[:N, 7 : 7 + n_rev]
        safe_lo = lo + margin * (hi - lo)
        safe_hi = hi - margin * (hi - lo)
        violation = ((safe_lo - jq).clamp(min=0) + (jq - safe_hi).clamp(min=0)).max(dim=-1).values
        return violation <= 0

    return check


class TestValidateKnownGood:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_default_stance_passes(self, robot_setup):
        kin, foot_ids, jq, foot_pos, bq = robot_setup
        buf = RetargetBuffer(1, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), DEVICE)
        buf.num_written = 1
        buf.num_geometry_valid = 1
        _fill_buf(buf, jq, foot_pos, DEVICE)

        criteria = {
            "foot_err": _make_foot_err_criterion(kin, foot_ids),
            "joint_margin": _make_joint_margin_criterion(kin),
        }
        reject, _ = validate_results(buf, criteria)
        assert reject.get("ok", 0) == 1, f"Default stance should pass, got: {reject}"


class TestValidateJointViolation:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_joint_limit_violation(self, robot_setup):
        kin, foot_ids, jq, foot_pos, bq = robot_setup
        jq_bad = jq.copy()
        ju = kin.model.joint_limit_upper.numpy()
        jq_bad[7] = ju[6] * 0.99

        state = kin.eval_fk(wp.array(jq_bad, dtype=float, device=DEVICE))
        bq_bad = state.body_q.numpy()
        bad_foot = np.array([bq_bad[fid][:3] for fid in foot_ids])

        buf = RetargetBuffer(1, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), DEVICE)
        buf.num_written = 1
        buf.num_geometry_valid = 1
        _fill_buf(buf, jq_bad, bad_foot, DEVICE)

        criteria = {
            "foot_err": _make_foot_err_criterion(kin, foot_ids),
            "joint_margin": _make_joint_margin_criterion(kin),
        }
        reject, _ = validate_results(buf, criteria)
        assert reject.get("ok", 0) == 0, f"Near-limit joints should be rejected: {reject}"


class TestValidateFootError:
    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_foot_position_error(self, robot_setup):
        kin, foot_ids, jq, foot_pos, bq = robot_setup
        bad_foot = foot_pos.copy()
        bad_foot[0] += [0.1, 0, 0]

        buf = RetargetBuffer(1, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), DEVICE)
        buf.num_written = 1
        buf.num_geometry_valid = 1
        _fill_buf(buf, jq, bad_foot, DEVICE)

        criteria = {"foot_err": _make_foot_err_criterion(kin, foot_ids)}
        reject, _ = validate_results(buf, criteria)
        assert reject.get("foot_err", 0) > 0, f"Corrupted foot should be rejected: {reject}"
