# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks for the position-locomotion future command wiring."""

from __future__ import annotations


def test_locomotion_position_uses_future_command_and_curriculum():
    """The working position env should use the future task-table command stack."""
    from isaaclab.managers import EventTermCfg

    from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from isaaclab_tasks.core.multi_task.mdp.curriculums import success_rate_sampler
    from isaaclab_tasks.core.multi_task.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import JointWithinLimitCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg)

    assert isinstance(cfg.commands.goal_point, StateCommandCfg)
    assert cfg.scene.terrain.prim_path == "/World/ground"
    assert cfg.scene.terrain.use_terrain_origins is True
    assert cfg.scene.height_scanner.mesh_prim_paths == ["/World/ground"]
    assert not hasattr(cfg.commands.goal_point.task_table, "state_frame")
    assert cfg.curriculum.terrain_levels.func is success_rate_sampler
    assert "success_rates_bind" not in cfg.curriculum.terrain_levels.params
    assert any(
        isinstance(criteria_cfg, JointWithinLimitCfg)
        and criteria_cfg.limit_ratio == 0.9
        and criteria_cfg.name == "joint_limit"
        for criteria_cfg in cfg.commands.goal_point.task_table.pipeline_cfg.criteria
    )

    reset_terms = [
        name for name, term in vars(cfg.events).items() if isinstance(term, EventTermCfg) and term.mode == "reset"
    ]
    assert reset_terms == []


def test_locomotion_position_uses_verified_legacy_terminations():
    """Position tasks should keep the termination surface from the working runs."""
    from isaaclab.envs import mdp as base_mdp
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    from isaaclab_tasks.core.multi_task.mdp.terminations import joint_reaction_overload
    from isaaclab_tasks.core.multi_task.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.core.multi_task.terrain import mdp
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg)

    done_terms = {name: term for name, term in vars(cfg.terminations).items() if isinstance(term, DoneTerm)}

    assert list(done_terms) == [
        "time_out",
        "abnormal_robot",
        "drop",
        "base_contact",
        "joint_reaction",
        "success",
    ]
    assert getattr(cfg.terminations, "abnormal") is None
    assert not hasattr(cfg.terminations, "oob")

    assert cfg.terminations.abnormal_robot.func is mdp.abnormal_robot_state
    assert cfg.terminations.drop.func is base_mdp.root_height_below_minimum
    assert cfg.terminations.drop.params == {"minimum_height": -20.0}
    assert cfg.terminations.joint_reaction.func is joint_reaction_overload
    assert cfg.terminations.joint_reaction.params["sensor_cfg"].name == "joint_wrench"
    assert cfg.terminations.joint_reaction.params["force_ratio"] == 6.0
    assert "force_mode" not in cfg.terminations.joint_reaction.params
    assert cfg.terminations.success.func is mdp.success_terminate


def test_locomotion_position_anymal_c_command_resolves_without_robot_preset():
    """Anymal-C position tasks should not require a separate robot preset for retarget fields."""
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from isaaclab_tasks.core.multi_task.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp.commands.commands_cfg import BaseStatePayloadCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg, {"anymal_c", "newton_mjwarp", "all", "base"})

    goal_cfg = cfg.commands.goal_point
    assert isinstance(goal_cfg, StateCommandCfg)
    assert isinstance(goal_cfg.payload, BaseStatePayloadCfg)
    assert goal_cfg.task_table.pipeline_cfg.foot_body_names == ".*FOOT.*"
    assert goal_cfg.task_table.pipeline_cfg.lateral_hip_joint_pattern == ".*HAA"


def test_locomotion_position_subterrains_do_not_request_flat_patches():
    """Position terrain presets should rely on the task-table pipeline, not terrain flat patches."""
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets import SubTerrainPresetCfg

    presets = SubTerrainPresetCfg()
    preset_names = (
        "terrain_curriculum",
        "gap",
        "pit",
        "extreme_stair",
        "slope_inv",
        "square_pillar_obstacle",
        "stepping_stone",
        "stepping_stone_curriculum",
        "radiating_beam",
        "flat",
        "default",
    )
    for preset_name in preset_names:
        for terrain_name, terrain_cfg in getattr(presets, preset_name).items():
            assert not getattr(terrain_cfg, "flat_patch_sampling", None), f"{preset_name}.{terrain_name}"


def test_locomotion_position_newton_mjwarp_preset_enables_newton_actuators(monkeypatch):
    """Newton MJWarp position presets should opt into the Newton actuator fast path."""
    import sys

    from isaaclab.actuators import ActuatorNetLSTMCfg

    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(
        sys,
        "argv",
        ["test", "presets=anymal_c,newton_mjwarp,terrain_pose"],
    )

    cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")
    actuator_cfg = cfg.scene.robot.actuators["legs"]

    assert cfg.sim.use_newton_actuators is True
    assert isinstance(actuator_cfg, ActuatorNetLSTMCfg)
    assert isinstance(actuator_cfg.network_file, str)
    assert actuator_cfg.network_file
    assert actuator_cfg.network_file.endswith(".pt")


def test_locomotion_position_newton_mjwarp_keeps_actuator_choice_explicit(monkeypatch):
    """Newton MJWarp should not conflict with explicit actuator presets."""
    import sys

    from isaaclab.actuators import ImplicitActuatorCfg

    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(
        sys,
        "argv",
        ["test", "presets=anymal_c,newton_mjwarp,implicit_actuator,terrain_pose"],
    )

    cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")

    assert isinstance(cfg.scene.robot.actuators["legs"], ImplicitActuatorCfg)
    assert cfg.scene.robot.spawn.joint_drive_props is not None
    assert cfg.scene.robot.spawn.joint_drive_props.stiffness == 40.0
    assert cfg.scene.robot.spawn.joint_drive_props.damping == 5.0
