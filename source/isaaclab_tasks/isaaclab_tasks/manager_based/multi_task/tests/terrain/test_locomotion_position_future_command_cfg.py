# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks for the position-locomotion future command wiring."""

from __future__ import annotations


def test_locomotion_position_uses_future_command_and_curriculum():
    """The working position env should use the future task-table command stack."""
    from isaaclab.managers import EventTermCfg

    from isaaclab_tasks.manager_based.locomotion.position.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.manager_based.multi_task.mdp.curriculums import success_rate_sampler
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp.commands.commands_cfg import RelativeStateCommandCfg
    from isaaclab_tasks.manager_based.multi_task.terrain.retarget.criteria_cfg import JointWithinLimitCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg)

    assert isinstance(cfg.commands.goal_point, RelativeStateCommandCfg)
    assert cfg.scene.terrain.prim_path == "/World/ground"
    assert cfg.scene.terrain.use_terrain_origins is True
    assert cfg.scene.height_scanner.mesh_prim_paths == ["/World/ground"]
    assert not hasattr(cfg.commands.goal_point.task_table, "state_frame")
    assert cfg.curriculum.terrain_levels.func is success_rate_sampler
    assert "success_rates_bind" in cfg.curriculum.terrain_levels.params
    assert any(
        isinstance(criteria_cfg, JointWithinLimitCfg)
        and criteria_cfg.limit_ratio == 0.9
        and criteria_cfg.name == "joint_limit"
        for criteria_cfg in cfg.commands.goal_point.task_table.pipeline_cfg.criteria
    )

    reset_terms = [
        name
        for name, term in vars(cfg.events).items()
        if isinstance(term, EventTermCfg) and term.mode == "reset"
    ]
    assert reset_terms == []


def test_locomotion_position_anymal_c_command_resolves_without_robot_preset():
    """Anymal-C position tasks should not require a separate robot preset for retarget fields."""
    from isaaclab_tasks.manager_based.locomotion.position.config.anymal_c.anymal_c_env_cfg import (
        AnymalCLocomotionPositionCommandEnvCfg,
    )
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp.commands.commands_cfg import RelativeStateCommandCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = AnymalCLocomotionPositionCommandEnvCfg()
    resolve_presets(cfg, {"newton", "all", "base"})

    goal_cfg = cfg.commands.goal_point
    assert isinstance(goal_cfg, RelativeStateCommandCfg)
    assert isinstance(goal_cfg.payload, RelativeStateCommandCfg.BaseStatePayloadCfg)
    assert goal_cfg.task_table.pipeline_cfg.foot_body_names == ".*FOOT.*"
    assert goal_cfg.task_table.pipeline_cfg.lateral_hip_joint_pattern == ".*HAA"


def test_locomotion_position_subterrains_do_not_request_flat_patches():
    """Position terrain presets should rely on the task-table pipeline, not terrain flat patches."""
    from isaaclab_tasks.manager_based.locomotion.position.terrain_preset import SubTerrainPresetCfg

    presets = SubTerrainPresetCfg()
    preset_names = (
        "all",
        "eval",
        "gap",
        "pit",
        "extreme_stair",
        "slope_inv",
        "stepping_stone",
        "radiating_beam",
        "flat",
        "foot_sampled_commands",
    )
    for preset_name in preset_names:
        for terrain_name, terrain_cfg in getattr(presets, preset_name).items():
            assert not getattr(terrain_cfg, "flat_patch_sampling", None), f"{preset_name}.{terrain_name}"


def test_locomotion_position_flat_patch_command_preset_restores_old_stack():
    """The old flat-patch command stack should remain selectable when requested."""
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands import (
        RelativeStateCommandCfg as FlatPatchRelativeStateCommandCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp.curriculums import (
        terrain_spawn_goal_pair_success_rate_levels,
    )
    from isaaclab_tasks.manager_based.locomotion.position.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg, {"flat_patch_commands"})

    assert isinstance(cfg.commands.goal_point, FlatPatchRelativeStateCommandCfg)
    assert cfg.curriculum.terrain_levels.func is terrain_spawn_goal_pair_success_rate_levels
    assert all(
        {"spawn", "target"} <= set(getattr(terrain_cfg, "flat_patch_sampling", {}))
        for terrain_cfg in cfg.scene.terrain.terrain_generator.sub_terrains.values()
    )


def test_locomotion_position_flat_patch_command_preset_keeps_old_command_schema(monkeypatch):
    """Global command presets should stay inside the selected command stack."""
    import sys

    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands import (
        RelativeStateCommandCfg as FlatPatchRelativeStateCommandCfg,
    )
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(
        sys,
        "argv",
        ["test", "presets=base,flat_patch_commands,terrain_pose"],
    )

    cfg, _ = resolve_task_config("Isaac-Position-Anymal-C-v0", "rsl_rl_cfg_entry_point")
    command_cfg = cfg.commands.goal_point

    assert isinstance(command_cfg, FlatPatchRelativeStateCommandCfg)
    assert isinstance(command_cfg.commands["terrain_pose_cmd"], FlatPatchRelativeStateCommandCfg.TerrainCommands)
