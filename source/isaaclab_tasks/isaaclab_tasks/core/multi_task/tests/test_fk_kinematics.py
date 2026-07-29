# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton-based forward kinematics via NewtonKinematics."""

import newton
import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg


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


class TestNewtonKinematics:
    """Verify NewtonKinematics FK and joint resolution."""

    @pytest.fixture(scope="class")
    def kin(self):
        return NewtonKinematics(
            NewtonKinematicsCfg(
                usd_path=ANYMAL_USD,
                device=DEVICE,
                default_pos=(0, 0, 0.6),
                default_joint_pos=DEFAULT_JPOS,
            )
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_default_stance(self, kin):
        base_pos = kin.default_body_q[0][:3]
        assert abs(base_pos[2] - 0.6) < 0.1, f"Base z={base_pos[2]}, expected ~0.6"

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_joint_pos_dict_resolution(self, kin):
        jq = kin.default_joint_q[7:]
        assert len(jq) > 0
        assert not np.allclose(jq, 0), "Joint positions should be non-zero from dict"

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_find_joint_dof_indices(self, kin):
        haa = kin.find_joint_dof_indices(".*HAA")
        assert len(haa) == 4, f"Expected 4 HAA joints, got {len(haa)}"
        assert all(isinstance(i, int) for i in haa)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_fk_ik_roundtrip(self, kin):
        jq = wp.array(kin.default_joint_q, dtype=float, device=DEVICE)
        state = kin.eval_fk(jq)

        recovered_q = wp.zeros_like(jq)
        recovered_qd = wp.zeros(kin.model.joint_dof_count, dtype=float, device=DEVICE)
        newton.eval_ik(kin.model, state, recovered_q, recovered_qd)

        np.testing.assert_allclose(
            jq.numpy(),
            recovered_q.numpy(),
            atol=1e-4,
            err_msg="FK->IK round-trip failed",
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_fk_nonzero_changes_positions(self, kin):
        default_bq = kin.default_body_q.copy()

        jq_mod = kin.default_joint_q.copy()
        jq_mod[7:] = 0.3
        state = kin.eval_fk(wp.array(jq_mod, dtype=float, device=DEVICE))
        mod_bq = state.body_q.numpy()

        diff = np.abs(mod_bq[:, :3] - default_bq[:, :3]).max()
        assert diff > 0.01, f"Changing joints should move bodies, max diff={diff:.4f}m"
