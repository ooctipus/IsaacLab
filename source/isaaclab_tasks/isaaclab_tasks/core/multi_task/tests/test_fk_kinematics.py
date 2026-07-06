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


DEVICE = "cuda:0"
DEFAULT_JPOS = {".*_joint": 0.2}


class TestNewtonKinematics:
    """Verify NewtonKinematics FK and joint resolution."""

    @pytest.fixture(scope="class")
    @classmethod
    def kinematics(cls, canonical_topology_mjcf):
        return NewtonKinematics(
            NewtonKinematicsCfg(
                mjcf_path=str(canonical_topology_mjcf),
                device=DEVICE,
                default_pos=(0, 0, 0.6),
                default_joint_pos=DEFAULT_JPOS,
            )
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_default_stance(self, kinematics):
        base_pos = kinematics.default_body_q[0][:3]
        assert abs(base_pos[2] - 0.6) < 0.1, f"Base z={base_pos[2]}, expected ~0.6"

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_joint_pos_dict_resolution(self, kinematics):
        jq = kinematics.default_joint_q[7:]
        assert len(jq) > 0
        assert not np.allclose(jq, 0), "Joint positions should be non-zero from dict"

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_find_joint_scalar_coordinates(self, kinematics):
        coordinates, velocities, names = kinematics.find_joint_scalar_coordinates(".*_joint")
        assert len(coordinates) == len(velocities) == len(names) == 4
        assert all(isinstance(index, int) for index in coordinates + velocities)
        assert coordinates == sorted(coordinates)
        assert velocities == sorted(velocities)

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_fk_ik_roundtrip(self, kinematics):
        jq = wp.array(kinematics.default_joint_q, dtype=float, device=DEVICE)
        state = kinematics.eval_fk(jq)

        recovered_q = wp.zeros_like(jq)
        recovered_qd = wp.zeros(kinematics.model.joint_dof_count, dtype=float, device=DEVICE)
        newton.eval_ik(kinematics.model, state, recovered_q, recovered_qd)

        np.testing.assert_allclose(
            jq.numpy(),
            recovered_q.numpy(),
            atol=1e-4,
            err_msg="FK->IK round-trip failed",
        )

    @pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
    def test_fk_nonzero_changes_body_orientation(self, kinematics):
        default_bq = kinematics.default_body_q.copy()

        jq_mod = kinematics.default_joint_q.copy()
        jq_mod[7:] = 0.3
        state = kinematics.eval_fk(wp.array(jq_mod, dtype=float, device=DEVICE))
        mod_bq = state.body_q.numpy()

        diff = np.abs(mod_bq[:, 3:7] - default_bq[:, 3:7]).max()
        assert diff > 0.01, f"Changing joints should rotate bodies, max quaternion diff={diff:.4f}"
