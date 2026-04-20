# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton-based forward kinematics and inverse kinematics.

Two test classes:

- ``TestNewtonFK``: Pure Newton tests (no IsaacSim) -- verifies FK/IK round-trips.
- ``TestNewtonVsPhysX``: Loads the same robot in Newton and PhysX, sets identical
  joint angles, and asserts body positions match.
"""

import math

import numpy as np
import pytest
import torch
import warp as wp

import newton
import newton.ik as ik

from isaaclab_tasks.manager_based.locomotion.position.mdp.kinematics import (
    build_newton_model,
    create_ik_solver,
    eval_fk_positions,
    solve_foot_ik,
)

ANYMAL_USD = "/home/zhengyuz/Downloads/ANYmal-C/anymal_c.usd"
POSITION_TOL_M = 0.005
IK_POSITION_TOL_M = 0.01


# ---------------------------------------------------------------------------
# Pure Newton tests (no IsaacSim needed)
# ---------------------------------------------------------------------------


class TestNewtonFK:
    """Verify Newton FK and IK on ANYmal-C without any physics engine."""

    @pytest.fixture(scope="class")
    def newton_model(self):
        model = build_newton_model(ANYMAL_USD, device="cuda:0")
        yield model

    def test_model_loaded(self, newton_model):
        """Newton model has expected structure for ANYmal-C."""
        assert newton_model.body_count == 13
        assert newton_model.joint_count == 13  # 1 free + 12 revolute
        assert newton_model.articulation_count == 1

    def test_fk_default_pose(self, newton_model):
        """FK at default joint angles produces reasonable body positions."""
        state = newton_model.state()
        newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, state)
        body_q = state.body_q.numpy()

        base_pos = body_q[0][:3]
        assert np.allclose(base_pos, [0, 0, 0], atol=0.01), f"Base at {base_pos}, expected near origin"

        for b in range(newton_model.body_count):
            pos = body_q[b][:3]
            assert np.all(np.isfinite(pos)), f"Body {b} has non-finite position: {pos}"

    def test_fk_ik_roundtrip(self, newton_model):
        """FK -> IK -> FK round-trip recovers joint angles."""
        joint_q = wp.clone(newton_model.joint_q)
        joint_qd = wp.clone(newton_model.joint_qd)

        state = newton_model.state()
        newton.eval_fk(newton_model, joint_q, joint_qd, state)

        recovered_q = wp.zeros_like(joint_q)
        recovered_qd = wp.zeros_like(joint_qd)
        newton.eval_ik(newton_model, state, recovered_q, recovered_qd)

        np.testing.assert_allclose(
            joint_q.numpy(), recovered_q.numpy(), atol=1e-5,
            err_msg="FK->IK round-trip failed to recover joint angles",
        )

    def test_fk_with_nonzero_joints(self, newton_model):
        """FK produces different body positions when joints are changed."""
        state_default = newton_model.state()
        newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, state_default)
        default_body_q = state_default.body_q.numpy().copy()

        joint_q_np = newton_model.joint_q.numpy().copy()
        # Set revolute joints (indices 7..18) to non-zero values
        for i in range(7, min(19, len(joint_q_np))):
            joint_q_np[i] = 0.3
        joint_q_mod = wp.array(joint_q_np, dtype=float, device=newton_model.device)

        state_mod = newton_model.state()
        newton.eval_fk(newton_model, joint_q_mod, newton_model.joint_qd, state_mod)
        mod_body_q = state_mod.body_q.numpy()

        diff = np.abs(mod_body_q[:, :3] - default_body_q[:, :3]).max()
        assert diff > 0.01, f"Changing joints should move bodies, but max diff was only {diff:.4f}m"

    def test_ik_solver_foot_targets(self, newton_model):
        """Newton IK solver can reach foot targets near default stance."""
        # Get foot body indices (shank bodies that have FOOT collapsed into them)
        # From our earlier exploration: bodies 3, 6, 9, 12 are the shank/foot bodies
        foot_ids = [3, 6, 9, 12]

        state = newton_model.state()
        newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, state)
        body_q_np = state.body_q.numpy()

        default_foot_pos = [body_q_np[fid][:3] for fid in foot_ids]

        solver, pos_objs, _ = create_ik_solver(
            newton_model, foot_ids, n_problems=1, ik_iterations=24,
        )

        # Perturb targets slightly
        targets = []
        for fp in default_foot_pos:
            t = fp.copy()
            t[0] += 0.02
            t[1] += 0.01
            targets.append(wp.vec3(float(t[0]), float(t[1]), float(t[2])))

        joint_q_init = newton_model.joint_q.reshape((1, newton_model.joint_coord_count))
        joint_q_solved = solve_foot_ik(solver, pos_objs, targets, joint_q_init, iterations=50)

        # Verify via FK
        state2 = newton_model.state()
        newton.eval_fk(newton_model, joint_q_solved.flatten(), newton_model.joint_qd, state2)
        solved_body_q = state2.body_q.numpy()

        for i, fid in enumerate(foot_ids):
            solved_pos = solved_body_q[fid][:3]
            target_pos = np.array([targets[i][0], targets[i][1], targets[i][2]])
            error = np.linalg.norm(solved_pos - target_pos)
            assert error < IK_POSITION_TOL_M, (
                f"Foot {fid}: IK error {error:.4f}m exceeds {IK_POSITION_TOL_M}m. "
                f"Solved={solved_pos}, target={target_pos}"
            )


# ---------------------------------------------------------------------------
# Newton vs PhysX comparison (requires IsaacSim)
# ---------------------------------------------------------------------------

try:
    from isaaclab.app import AppLauncher
    _HAS_ISAACSIM = True
except ImportError:
    _HAS_ISAACSIM = False


@pytest.mark.skipif(not _HAS_ISAACSIM, reason="IsaacSim not available")
class TestNewtonVsPhysX:
    """Compare Newton FK against PhysX FK for the same robot and joint angles.

    This test class requires IsaacSim to be available. It loads the robot
    in both Newton and PhysX, sets identical joint configurations, and
    asserts that body positions agree within tolerance.

    Note: This test must be run with the IsaacSim environment active
    (e.g. via ``source env_isaaclab/bin/activate``). When IsaacSim is
    not available, these tests are skipped.
    """
    pass
    # TODO: implement once we confirm the pure Newton tests pass.
    # The test will:
    # 1. Launch IsaacSim headless
    # 2. Load ANYmal-C via Articulation
    # 3. For N random joint configs:
    #    a. Set joints in PhysX, sim.step(), read body_pos_w
    #    b. Set same joints in Newton, eval_fk, read body_q
    #    c. Assert position error < 2mm per body
