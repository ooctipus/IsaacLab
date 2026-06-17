# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: ``Isaac-Position-v0`` and ``Isaac-Factory-v0`` env cfgs construct.

Used as a fast integration safety net before / after structural
refactors of the multi_task subpackage. Mirrors what the hydra pipeline
does up to (but not including) Kit initialisation:

1. Triggers gym task registration via ``import isaaclab_tasks``.
2. Resolves ``env_cfg_entry_point`` from the gym registry.
3. Instantiates the env cfg class.
4. Round-trips the cfg through ``cfg.to_dict()`` (the same path hydra
   feeds to ``OmegaConf.create`` in :func:`register_task`). Catches
   serialisation issues that show up only when training launches --
   e.g. nested cfgs whose annotations OmegaConf rejects, or
   ``class_type`` fields not registered as ``ResolvableString``.

If any of the above breaks (broken import path, missing module after a
move, cfg construction error, OmegaConf-incompatible annotation), this
test catches it without paying the cost of the heavier
``test_env_cfg_no_forbidden_imports.py``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401 -- registers the gym tasks


def test_position_uses_global_terrain_like_old_position_task() -> None:
    """Position scene should use one global terrain, not one ground prim per env."""
    from isaaclab_tasks.core.multi_task.position_env_cfg import LocomotionPositionCommandEnvCfg

    cfg = LocomotionPositionCommandEnvCfg()

    assert cfg.scene.terrain.prim_path == "/World/ground"
    assert cfg.scene.terrain.use_terrain_origins is True
    assert cfg.scene.height_scanner.mesh_prim_paths == ["/World/ground"]
    assert cfg.scene.env_spacing == 0.0
    assert cfg.commands.goal_point.states_relative is False


def test_position_joint_reaction_uses_magnitude_force_mode() -> None:
    """Position joint-reaction termination should match the old full-force magnitude gate."""
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.termination_presets import PositionTerminationsCfg

    cfg = PositionTerminationsCfg()

    assert cfg.joint_reaction.params["force_mode"] == "magnitude"


@pytest.mark.parametrize("task_name", ["Isaac-Position-v0", "Isaac-Factory-v0"])
def test_env_cfg_constructs(task_name: str) -> None:
    """The env cfg referenced by ``env_cfg_entry_point`` imports + constructs."""
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    # Standard manager-based env cfg fields. If a structural move left a
    # field unresolved (e.g. curriculum cfg pointing at a missing module),
    # construction would have raised before this assertion.
    assert hasattr(cfg, "scene")
    assert hasattr(cfg, "actions")
    assert hasattr(cfg, "observations")
    assert hasattr(cfg, "events")


def test_factory_success_rate_callback_targets_reset_state_command() -> None:
    """Factory difficulty curriculum should bind the reset-state success rates."""
    spec = gym.spec("Isaac-Factory-v0")
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    callback = cfg.curriculum.difficulty_scheduler.params["success_rate_callback"]

    expected = "env.command_manager.get_term('reset_state').success_rates"
    assert callback == expected

    rates = torch.tensor([0.5, 1.0])
    reset_state = SimpleNamespace(success_rates=rates)
    eval_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: reset_state))
    assert eval(callback, {}, {"env": eval_env}) is rates  # noqa: S307


def test_factory_difficulty_scheduler_waits_for_accumulator_rates() -> None:
    """Initial reset may run curriculum before accumulator reset materializes rates."""
    from isaaclab_tasks.core.multi_task.factory.mdp.curriculum import DifficultyScheduler

    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=None)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    assert result is scheduler.difficulty_frac
    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 2)


def test_factory_difficulty_scheduler_averages_ready_success_rates() -> None:
    """Difficulty scheduler should average bound rate tensors internally."""
    from isaaclab_tasks.core.multi_task.factory.mdp.curriculum import DifficultyScheduler

    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=torch.ones(4))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 3)
    torch.testing.assert_close(result, torch.tensor(0.3))


def test_factory_actor_critic_preset_composes_with_value_shift_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor-critic model preset should not replace the factory PPO algorithm branch."""
    from isaaclab_tasks.core.multi_task.factory.config.agents.rsl_rl_ppo_cfg import ValueShiftAlgorithmCfg
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "presets=actor_critic,beta_value_shift",
        ],
    )

    _, agent_cfg = resolve_task_config("Isaac-Factory-v0", "rsl_rl_cfg_entry_point")

    assert isinstance(agent_cfg.algorithm, ValueShiftAlgorithmCfg)


def test_position_anymal_c_base_contact_matches_old_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anymal-C base-contact termination should monitor only the base body."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["pytest", "presets=anymal_c"])

    env_cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")

    assert env_cfg.terminations.base_contact.params["sensor_cfg"].body_names == "base"


@pytest.mark.parametrize(
    ("preset_name", "command_name"),
    [
        ("terrain_pose", "terrain_pose_cmd"),
        ("terrain_pos", "terrain_position_cmd"),
    ],
)
def test_position_terrain_command_duration_matches_old_task(
    monkeypatch: pytest.MonkeyPatch,
    preset_name: str,
    command_name: str,
) -> None:
    """Terrain command hold duration should match the old position task."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["pytest", f"presets={preset_name}"])

    env_cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")

    assert env_cfg.commands.goal_point.commands[command_name].duration == (0.05, 1.0)


def test_position_success_gate_payload_matches_old_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Position command success should keep the old settled/naturalness gates."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["pytest", "presets=anymal_c,terrain_pose"])

    env_cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")
    payload = env_cfg.commands.goal_point.payload

    assert payload.success_effort_multiplier == 0.8
    assert payload.joint_wrench_sensor_name == "joint_wrench"
    assert payload.contact_sensor_name == "contact_forces"
    assert payload.success_min_foot_weight_fraction == 0.80
    assert payload.success_body_lin_speed_thresh == 0.30
    assert payload.success_body_ang_speed_thresh == 0.30


def test_position_simba_big_actor_uses_big_model() -> None:
    """Position simba_big actor preset should use the wider SimBa actor cfg."""
    from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import PositionLocomotionPPORunnerCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = PositionLocomotionPPORunnerCfg()
    resolve_presets(cfg, selected=("simba_big",))

    assert cfg.actor.hidden_dim == 512
    assert cfg.critic.hidden_dim == 1024


def test_position_preserves_old_preset_surface_except_flat_commands() -> None:
    """Migrated position cfg should keep old selectable preset names."""
    from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import (
        PositionActorPresetCfg,
        PositionCriticPresetCfg,
        PositionObsGroupsPresetCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.curriculum_presets import CurriculumPresetCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.observation_presets import ObservationsCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.reward_presets import RewardsCfg

    def fields(obj) -> set[str]:
        return {field.name for field in dataclass_fields(obj)}

    simba_names = {"simba", "simba_big", "simba_mlp", "simba_mlp_big", "simba_cnn", "simba_cnn_big"}

    assert {"commander", "task_easing", "lstm", "flat", "encoder", *simba_names, "default"} <= fields(
        PositionActorPresetCfg()
    )
    assert {"flat", "lstm", "encoder", *simba_names, "default"} <= fields(PositionCriticPresetCfg())
    assert {"flat", "encoder", *simba_names, "default"} <= fields(PositionObsGroupsPresetCfg())
    assert {"flat", "encoder", *simba_names, "default"} <= fields(ObservationsCfg())
    assert {"rew_v1", "rew_v2", "default"} <= fields(RewardsCfg())
    assert {"foot_sampled_commands", "default"} <= fields(CurriculumPresetCfg())
    assert "flat_patch_commands" not in fields(CurriculumPresetCfg())


def test_position_old_preset_names_resolve_except_flat_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old position preset names should still resolve through Hydra."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    old_names = (
        "flat",
        "encoder",
        "simba_mlp",
        "simba_mlp_big",
        "simba_cnn",
        "simba_cnn_big",
        "rew_v1",
        "rew_v2",
        "foot_sampled_commands",
        "base",
        "base_foot",
        "terrain",
        "terrain_pos",
        "terrain_pose",
        "pose",
        "pos",
        "vel",
        "all_commands",
    )
    for name in old_names:
        monkeypatch.setattr(sys, "argv", ["pytest", f"presets=anymal_c,{name}"])
        resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")

    monkeypatch.setattr(sys, "argv", ["pytest", "presets=flat_patch_commands"])
    with pytest.raises(ValueError, match="flat_patch_commands"):
        resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")


def test_position_beta_value_shift_sampler_includes_value_shift_strategy() -> None:
    """Position beta_value_shift should combine Beta sampling with value-shift scoring."""
    from isaaclab_tasks.core.multi_task.curriculum import ValueShiftSamplingStrategyCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.curriculum_presets import PositionCurriculumSamplerCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = PositionCurriculumSamplerCfg()
    resolve_presets(cfg, selected=("beta_value_shift",))

    sampling = cfg.terrain_levels.params["sampling"]
    value_shift = [s for s in sampling.strategies if isinstance(s, ValueShiftSamplingStrategyCfg)]

    assert len(value_shift) == 1
    assert value_shift[0].state_buffer_bind == "env.command_manager.get_term('goal_point').table.task_partition"


def test_position_beta_value_shift_preset_uses_value_shift_algorithm() -> None:
    """Position beta_value_shift should select the matching ValueShiftPPO algorithm cfg."""
    from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import (
        PositionLocomotionPPORunnerCfg,
        ValueShiftAlgorithmCfg,
    )
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = PositionLocomotionPPORunnerCfg()
    resolve_presets(cfg, selected=("beta_value_shift",))

    assert isinstance(cfg.algorithm, ValueShiftAlgorithmCfg)
    assert cfg.algorithm.gamma == 0.999
    assert cfg.algorithm.share_cnn_encoders is False


@pytest.mark.parametrize("task_name", ["Isaac-Position-v0", "Isaac-Factory-v0"])
def test_env_cfg_to_dict_serialises(task_name: str) -> None:
    """``cfg.to_dict()`` produces a fully-flattened dict with no dataclass instances.

    Hydra's :func:`register_task` feeds ``cfg.to_dict()`` into
    ``OmegaConf.create``. If any nested dataclass cfg uses an
    annotation OmegaConf can't validate (e.g. ``type[X]`` without ``| str``,
    or untyped containers leaving dataclass instances exposed), the
    training launch crashes during ``register_task``. This test checks
    that the dict is OmegaConf-clean by asserting no dataclass instances
    leak through.
    """
    spec = gym.spec(task_name)
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    cfg_dict = cfg.to_dict()

    def _assert_no_dataclasses(value, path: str = "") -> None:
        if hasattr(value, "__dataclass_fields__"):
            raise AssertionError(
                f"Unflattened dataclass {type(value).__name__} at {path!r} -- "
                "this would crash OmegaConf during hydra's register_task."
            )
        if isinstance(value, dict):
            for k, v in value.items():
                _assert_no_dataclasses(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _assert_no_dataclasses(v, f"{path}[{i}]")

    _assert_no_dataclasses(cfg_dict)


@pytest.mark.parametrize(
    "module_path",
    [
        # Pure-Python leaf modules that the restructure relocated. Importing
        # them catches stale ``from .X import Y`` references where ``X`` is
        # now a sibling-package or has been renamed. Modules that pull in
        # Kit / Newton / USD must NOT be added here -- they segfault when
        # imported outside a launched Kit app.
        "isaaclab_tasks.core.multi_task.curriculum.sampling",
        "isaaclab_tasks.core.multi_task.curriculum.sampling.sampler",
        "isaaclab_tasks.core.multi_task.curriculum.sampling.sampler_cfg",
        "isaaclab_tasks.core.multi_task.curriculum.sampling.sampling_strategies",
        "isaaclab_tasks.core.multi_task.curriculum.sampling.sampling_strategies_cfg",
        "isaaclab_tasks.core.multi_task.curriculum.state_layout",
        "isaaclab_tasks.core.multi_task.curriculum.state_buffer",
        "isaaclab_tasks.core.multi_task.curriculum.success_monitor",
        "isaaclab_tasks.core.multi_task.curriculum.reset_state",
        "isaaclab_tasks.core.multi_task.terrain.terrains.patch_sampling.cfg",
        "isaaclab_tasks.core.multi_task.terrain.terrains.patch_sampling.morph",
        "isaaclab_tasks.core.multi_task.terrain.terrains.patch_sampling.rejection",
        "isaaclab_tasks.core.multi_task.terrain.retarget.cfg",
        "isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg",
        "isaaclab_tasks.core.multi_task.factory.assembly_profile",
        "isaaclab_tasks.core.multi_task.factory.assembly_profile_cfg",
    ],
)
def test_relocated_module_imports(module_path: str) -> None:
    """Modules touched by the directory restructure import without errors.

    A targeted version of "import everything" -- limited to pure-Python
    leaf modules so the test doesn't pull in Kit/Newton-bound modules
    that segfault outside a launched Kit app. Catches stale relative
    imports that the restructure left behind (e.g.
    ``from .reset_state import X`` after ``reset_state.py`` moved
    sibling).
    """
    importlib.import_module(module_path)
