# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ownership regressions for the Position-style motion environment root."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import fields

import pytest

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

_ROOT_BYPASS_FIELDS = (
    "source_artifact_root",
    "reference_artifact_root",
    "motion_split",
    "expert_sample_grid",
    "reference_kinematics_factory",
    "transition_state_factory",
    "applied_actions_before_timeout",
)


def test_motion_environment_root_has_exactly_framework_fields() -> None:
    """Motion-specific configuration must remain below manager-group fields."""
    motion_fields = {field.name for field in fields(MotionImitationEnvCfg)}
    framework_fields = {field.name for field in fields(ManagerBasedRLEnvCfg)}

    assert motion_fields == framework_fields


def test_legacy_motion_aggregate_and_materializer_boundaries_are_absent() -> None:
    """Robot, manager, and trajectory ownership must not hide behind aggregate factories."""
    presets = importlib.import_module("isaaclab_tasks.core.multi_task.motion.config.presets")
    robots = importlib.import_module("isaaclab_tasks.core.multi_task.motion.config.robots")
    environment = importlib.import_module("isaaclab_tasks.core.multi_task.motion.config.environment")

    for module, names in (
        (presets, ("MotionPresetCfg", "MotionPresetsCfg")),
        (robots, ("MotionRobotPreset",)),
        (environment, ("MotionEnvironmentCfg", "smpl_environment_cfg", "g1_environment_cfg")),
    ):
        assert not any(hasattr(module, name) for name in names)
    assert not hasattr(MotionImitationEnvCfg, "compose_motion")
    assert not hasattr(MotionImitationEnv, "motion_preset")
    assert not hasattr(MotionImitationEnv, "motion_robot")
    assert not hasattr(MotionImitationEnv, "motion_materializer")
    assert not hasattr(MotionImitationEnv, "motion_bank")
    assert importlib.util.find_spec("isaaclab_tasks.core.multi_task.motion.materializers") is None


@pytest.mark.parametrize(
    ("name", "source", "joint_count", "decimation"),
    (
        ("smpl_cmu", "smpl_cmu", 69, 15),
        ("g1_lafan", "g1_lafan", 29, 4),
        ("g1_cmu", "smpl_cmu", 29, 4),
    ),
)
def test_one_broadcast_name_resolves_direct_motion_environment_axes(
    name: str,
    source: str,
    joint_count: int,
    decimation: int,
) -> None:
    """One preset name must resolve the scene and table without a composition pass."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={name})

    assert not hasattr(cfg, "motion")
    assert not any(hasattr(cfg, name) for name in _ROOT_BYPASS_FIELDS)
    assert isinstance(cfg.scene.robot, ArticulationCfg)
    assert cfg.decimation == decimation
    assert cfg.commands.motion.payload.episode_length_steps > 0
    assert cfg.commands.motion.task_table.source.identifier == source
    assert callable(cfg.commands.motion.task_table.frame_builder_factory)
    assert cfg.commands.motion.task_table.task_row_mode in ("source_frames", "clip_time_ranges")
    assert cfg.commands.motion.task_table.reset_sources
    assert cfg.commands.motion.payload.step_fields == ()
    horizon = cfg.commands.motion.payload.episode_length_steps
    assert horizon > 0
    assert cfg.terminations.time_out.params == {"applied_actions_before_timeout": horizon}
    assert len(cfg.scene.robot.actuators) > 0 or joint_count == 69
