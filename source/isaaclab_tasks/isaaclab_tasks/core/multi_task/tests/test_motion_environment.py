# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3E tests for online motion-history semantics."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import MujocoCollisionPropertiesCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysxContactSensorCfg

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.sim import PlaneCfg
from isaaclab.sim.schemas import CollisionBaseCfg

from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command import StateCommand
from isaaclab_tasks.core.multi_task.motion_env_cfg import (
    MotionImitationEnvCfg,
)
from isaaclab_tasks.utils import resolve_presets

_BFM_PRESETS = (
    "presets=g1,lafan,physx,timing_sim200_control50_horizon501,sampling_clip_time,"
    "evidence_physical_auxiliary,randomization_physics_observation_pose_push,"
    "helpers_discriminator_auxiliary,tracking_reset_frame,tracking_interval_9p6m,model_residual_6x1024,"
    "replay_episode_uniform_5120k,schedule_1024x1_211p2m,optimization_lr3e4_implied0_actor0p05,"
    "context_expert_half_8192,exploration_std0p05_range5,seed_4728,expert_clock_50hz"
)


def test_motion_config_resolution_does_not_import_pxr_before_simulation_app() -> None:
    """Every motion preset must resolve before the simulator runtime is imported."""
    code = textwrap.dedent(
        """
        import sys
        from dataclasses import fields

        from isaaclab.managers import ActionTermCfg, ObservationTermCfg, RewardTermCfg
        from isaaclab_tasks.utils.hydra import resolve_task_config

        expected_evidence = (
            "penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque",
            "penalty_undesired_contact", "penalty_feet_ori", "penalty_ankle_roll", "penalty_slippage",
        )
        meta_runner = (
            "helpers_discriminator,tracking_off,model_plain_2x1024,replay_transition_uniform_2m,"
            "schedule_50x10_5m,optimization_lr1e4_implied0p1_actor0p01,context_online_10k,"
            "exploration_std0p2_range1,seed_0,expert_clock_source_rows"
        )
        bfm_runner = (
            "helpers_discriminator_auxiliary,tracking_reset_frame,tracking_interval_9p6m,model_residual_6x1024,"
            "replay_episode_uniform_5120k,schedule_1024x1_211p2m,optimization_lr3e4_implied0_actor0p05,"
            "context_expert_half_8192,exploration_std0p05_range5,seed_4728,expert_clock_50hz"
        )
        g1_environment = (
            "physx,timing_sim200_control50_horizon501,sampling_clip_time,evidence_physical_auxiliary,"
            "randomization_physics_observation_pose_push"
        )
        profiles = {
            "smpl_cmu": (
                f"presets=smpl,cmu,newton_mjwarp,timing_sim450_control30_horizon300,"
                f"sampling_source_rows,{meta_runner}"
            ),
            "g1_lafan": f"presets=g1,lafan,{g1_environment},{bfm_runner}",
            "g1_cmu": f"presets=g1,cmu,{g1_environment},{bfm_runner}",
        }
        for profile, selection in profiles.items():
            sys.argv = ["test", selection]
            env_cfg, runner_cfg = resolve_task_config(
                "Isaac-Motion-Imitation-v0",
                "rsl_rl_cfg_entry_point",
            )
            expected_source = "cmu_humenv_smpl" if profile in ("smpl_cmu", "g1_cmu") else "lafan_g1_29dof"
            assert env_cfg.commands.motion.task_table.source.identifier == expected_source
            assert env_cfg.decimation == (15 if profile == "smpl_cmu" else 4)
            assert runner_cfg is not None
            action_terms = tuple(
                value
                for field in fields(env_cfg.actions)
                if isinstance(value := getattr(env_cfg.actions, field.name), ActionTermCfg)
            )
            assert len(action_terms) == 1
            assert str(action_terms[0].class_type).endswith(
                (":NativeMujocoControlAction", ":G1JointPositionAction")
            )
            reward_names = tuple(
                name for name, term in vars(env_cfg.rewards).items() if isinstance(term, RewardTermCfg)
            )
            assert reward_names == ()
            if profile == "smpl_cmu":
                assert not hasattr(env_cfg.observations, "transition")
            else:
                evidence_names = tuple(
                    name
                    for name, term in vars(env_cfg.observations.transition).items()
                    if isinstance(term, ObservationTermCfg)
                )
                assert evidence_names == expected_evidence
                assert env_cfg.observations.transition.concatenate_terms is False
        assert not any(name == "pxr" or name.startswith("pxr.") for name in sys.modules)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_motion_registry_resolution_reports_no_simulator_runtime_imports() -> None:
    """Fresh registry resolution reports every simulator module and its first importer."""
    code = textwrap.dedent(
        """
        import builtins
        import json
        import sys
        import traceback

        forbidden = ("isaacsim", "omni", "carb")
        first_importer = {}
        original_import = builtins.__import__

        def traced_import(name, *args, **kwargs):
            prefix = name.split(".", 1)[0]
            if prefix in forbidden and prefix not in first_importer:
                first_importer[prefix] = "".join(traceback.format_stack(limit=16))
            return original_import(name, *args, **kwargs)

        builtins.__import__ = traced_import
        try:
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils.hydra import resolve_presets
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            config = load_cfg_from_registry("Isaac-Motion-Imitation-v0", "env_cfg_entry_point")
            config = resolve_presets(config, frozenset())
            config.to_dict()
        finally:
            builtins.__import__ = original_import

        loaded = sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
        report = {
            "loaded": loaded,
            "first_importer": {
                prefix: first_importer.get(prefix, "<loaded outside builtins.__import__>")
                for prefix in sorted({name.split(".", 1)[0] for name in loaded})
            },
        }
        print("__MOTION_IMPORT_REPORT__" + json.dumps(report))
        """
    )
    completed = subprocess.run((sys.executable, "-c", code), capture_output=True, text=True, check=False)
    report_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("__MOTION_IMPORT_REPORT__")), None
    )
    assert completed.returncode == 0 and report_line is not None, completed.stdout + completed.stderr
    report = json.loads(report_line.removeprefix("__MOTION_IMPORT_REPORT__"))

    assert report["loaded"] == [], json.dumps(report, indent=2)


def test_motion_task_registration_uses_one_shared_environment() -> None:
    """Both native presets must enter through the same simulator and learner boundary."""
    spec = gym.spec("Isaac-Motion-Imitation-v0")

    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":MotionImitationEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":MotionForwardBackwardRunnerCfg")


def test_motion_scene_uses_a_local_native_collision_plane() -> None:
    """Ground construction must be local and preserve native SMPL floor facts."""
    env = resolve_presets(
        MotionImitationEnvCfg(),
        selected={"smpl", "cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"},
    )
    scene = env.scene

    assert isinstance(scene.ground.spawn, PlaneCfg)
    assert scene.ground.spawn.size == (200.0, 200.0)
    assert scene.ground.spawn.axis == "Z"
    assert scene.ground.init_state.pos == (0.0, 0.0, 0.0)
    assert scene.ground.spawn.collision_props is not None
    assert scene.ground.spawn.collision_props.collision_enabled
    assert scene.ground.spawn.physics_material.static_friction == 0.7
    assert scene.ground.spawn.physics_material.dynamic_friction == 0.7
    assert isinstance(scene.ground.spawn.collision_props, MujocoCollisionPropertiesCfg)
    assert scene.ground.spawn.collision_props.margin == 0.001
    assert scene.ground.spawn.collision_props.solimp == (0.99, 0.99, 0.003, 0.5, 2.0)
    assert scene.ground.spawn.collision_props.solref == (0.015, 1.0)
    assert not env.sim.physics.solver_cfg.enable_native_ccd
    assert not env.sim.physics.solver_cfg.enable_multiccd


@pytest.mark.parametrize(
    ("robot", "backend", "capabilities", "collision_type", "sensor_type", "friction"),
    (
        ("smpl", "newton_mjwarp", frozenset(), MujocoCollisionPropertiesCfg, None, 0.7),
        ("smpl", "physx", frozenset(), CollisionBaseCfg, None, 0.7),
        (
            "g1",
            "newton_mjwarp",
            frozenset(("evidence_physical_auxiliary", "randomization_physics_observation_pose_push")),
            MujocoCollisionPropertiesCfg,
            NewtonContactSensorCfg,
            1.0,
        ),
        (
            "g1",
            "physx",
            frozenset(("evidence_physical_auxiliary", "randomization_physics_observation_pose_push")),
            CollisionBaseCfg,
            PhysxContactSensorCfg,
            1.0,
        ),
    ),
)
def test_motion_ground_and_sensor_compose_robot_and_backend_axes(
    robot: str,
    backend: str,
    capabilities: frozenset[str],
    collision_type: type,
    sensor_type: type | None,
    friction: float,
) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected={robot, "cmu", backend, *capabilities})

    assert isinstance(env.scene.ground.spawn.collision_props, collision_type)
    assert env.scene.ground.spawn.physics_material.static_friction == friction
    assert env.scene.ground.spawn.physics_material.dynamic_friction == friction
    assert (env.scene.contact_forces is None) is (sensor_type is None)
    assert sensor_type is None or isinstance(env.scene.contact_forces, sensor_type)


@pytest.mark.parametrize("max_episode_length", (300, 501))
def test_motion_timeout_uses_the_environment_episode_edge(max_episode_length: int) -> None:
    """The standard timeout must truncate on the environment-owned action count."""
    env = SimpleNamespace(
        max_episode_length=max_episode_length,
        episode_length_buf=torch.tensor(
            (
                max_episode_length - 1,
                max_episode_length,
                max_episode_length + 1,
            )
        ),
    )

    torch.testing.assert_close(isaaclab_mdp.time_out(env), torch.tensor((False, True, True)))


def test_explicit_axes_select_the_g1_lafan_environment_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent axes should select the G1-LAFAN environment and learner profile."""
    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["test", _BFM_PRESETS])

    env_cfg, runner_cfg = resolve_task_config("Isaac-Motion-Imitation-v0", "rsl_rl_cfg_entry_point")

    assert env_cfg.commands.motion.task_table.source.identifier == "lafan_g1_29dof"
    assert env_cfg.decimation == 4
    assert str(env_cfg.actions.joint_position.class_type).endswith(":G1JointPositionAction")
    assert isinstance(runner_cfg, MotionForwardBackwardRunnerCfg)
    assert runner_cfg.num_envs == 1024
    assert runner_cfg.resolve_num_envs(None, env_cfg.scene.num_envs) == 1024


def _fixtures() -> Path:
    return Path(__file__).parents[6] / "scripts/reinforcement_learning/forward_backward/phase3/fixtures"


def test_motion_probe_keeps_host_operations_outside_the_timed_loop() -> None:
    """The timed loop must contain only environment steps and device-side counters."""
    source = (_fixtures().parent / "motion_environment_probe.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_steps")
    loops = sorted((node for node in ast.walk(function) if isinstance(node, ast.For)), key=lambda node: node.lineno)

    assert len(loops) == 2
    timed_loop = loops[-1]
    action_view = timed_loop.body[0]
    assert isinstance(action_view, ast.Assign)
    assert isinstance(action_view.value, ast.Subscript)
    assert isinstance(action_view.value.value, ast.Name)
    assert action_view.value.value.id == "action_program"
    calls = [node for statement in timed_loop.body for node in ast.walk(statement) if isinstance(node, ast.Call)]
    attribute_calls = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    named_calls = [node.func.id for node in calls if isinstance(node.func, ast.Name)]

    assert attribute_calls.count("step") == 1
    assert set(attribute_calls) <= {"step", "sum", "add_", "reshape", "bool"}
    assert not named_calls


def test_exact_motion_binding_uses_one_command_owned_transaction() -> None:
    """Exact payload binding must close the outgoing edge before zero-dt materialization."""
    events: list[tuple[str, float | torch.Tensor]] = []

    class Payload:
        def bind(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
            events.append(("bind", task_rows.clone()))

        def update(self, step_dt: float, command: torch.Tensor, error: torch.Tensor) -> None:
            del command, error
            events.append(("update", step_dt))

    payload = Payload()
    command = object.__new__(StateCommand)
    command._env = SimpleNamespace(common_step_counter=7, step_dt=0.02)
    command._payload = payload
    command._command = torch.empty(3, 0)
    command._err = torch.empty(3, 0)
    command._update_step = 6
    command.cmd_indices = torch.zeros(3, dtype=torch.int64)
    env_ids = torch.tensor((0, 2), dtype=torch.int64)
    task_rows = torch.tensor((4, 5), dtype=torch.int64)

    command.bind_rows(env_ids, task_rows)

    assert events[0] == ("update", 0.02)
    assert events[1][0] == "bind"
    torch.testing.assert_close(events[1][1], task_rows)
    assert events[2] == ("update", 0.0)
    assert command._update_step == 7
