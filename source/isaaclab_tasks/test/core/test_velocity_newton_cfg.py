# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


def test_g1_rough_newton_has_sufficient_constraint_capacity():
    env_cfg = G1RoughEnvCfg()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300


def test_velocity_command_markers_are_scene_plan_participants():
    cfg = LocomotionVelocityRoughEnvCfg()

    assert cfg.scene.command_goal_marker is cfg.commands.base_velocity.goal_vel_visualizer_cfg
    assert cfg.scene.command_current_marker is cfg.commands.base_velocity.current_vel_visualizer_cfg
