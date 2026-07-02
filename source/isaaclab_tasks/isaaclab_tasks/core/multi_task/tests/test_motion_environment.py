# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3E tests for online motion-history semantics."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from isaaclab_newton.sim.schemas import MujocoCollisionPropertiesCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import PlaneCfg

from isaaclab_tasks.core.multi_task.motion.mdp import AppliedTransitionHistory, AppliedTransitionHistoryLayout
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionStatePayload
from isaaclab_tasks.core.multi_task.motion.mdp.runtime import motion_time_out
from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv, _observation_manager_supports_selected_reset
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg, MotionSceneCfg
from isaaclab_tasks.utils import resolve_presets


def test_motion_config_resolution_does_not_import_pxr_before_simulation_app() -> None:
    """Every motion preset must resolve before the simulator runtime is imported."""
    code = textwrap.dedent(
        """
        import sys

        from isaaclab_tasks.utils.hydra import resolve_task_config

        for name in ("smpl_cmu", "g1_lafan", "g1_cmu"):
            sys.argv = ["test", f"presets={name}"]
            env_cfg, runner_cfg = resolve_task_config(
                "Isaac-Motion-Imitation-v0",
                "rsl_rl_cfg_entry_point",
            )
            expected_source = "smpl_cmu" if name in ("smpl_cmu", "g1_cmu") else "g1_lafan"
            assert env_cfg.commands.motion.task_table.source.identifier == expected_source
            assert env_cfg.decimation == (15 if name == "smpl_cmu" else 4)
            assert runner_cfg is not None
            assert str(env_cfg.actions.joint_position.class_type).endswith(
                (":MotionMujocoControlAction", ":MotionJointPositionAction")
            )
            assert str(env_cfg.rewards.environment.func).endswith(":motion_transition_reward")
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


def test_motion_task_registration_uses_one_shared_environment() -> None:
    """Both native presets must enter through the same simulator and learner boundary."""
    spec = gym.spec("Isaac-Motion-Imitation-v0")

    assert spec.entry_point == "isaaclab_tasks.core.multi_task.motion_env:MotionImitationEnv"
    assert spec.kwargs["env_cfg_entry_point"].endswith(":MotionImitationEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":MotionForwardBackwardRunnerPresetsCfg")


def test_motion_runtime_factory_rejects_the_narrow_reset_only_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment boundary must require every operation and tensor it consumes."""

    class ResetOnlyRuntime:
        def reset(self, env_ids: torch.Tensor) -> None:
            del env_ids

    payload = object.__new__(MotionStatePayload)
    payload.auxiliary_evidence_names = ()
    payload._transition_state = None
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            motion=SimpleNamespace(
                payload=SimpleNamespace(transition_state_factory=lambda _env, _payload: ResetOnlyRuntime())
            )
        )
    )
    env.command_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(payload=payload) if name == "motion" else None
    )
    monkeypatch.setattr(ManagerBasedRLEnv, "load_managers", lambda _env: None)

    with pytest.raises(TypeError, match=r"capture_current\(\)"):
        MotionImitationEnv.load_managers(env)


def test_motion_scene_uses_a_local_native_collision_plane() -> None:
    """Ground construction must be local and preserve native SMPL floor facts."""
    scene = resolve_presets(MotionSceneCfg(num_envs=1, env_spacing=3.0), selected={"smpl_cmu"})
    env = resolve_presets(MotionImitationEnvCfg(), selected={"smpl_cmu"})

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


@pytest.mark.parametrize("applied_actions_before_timeout", (300, 501))
def test_motion_timeout_uses_the_source_applied_action_edge(applied_actions_before_timeout: int) -> None:
    """SMPL and G1 must truncate on their observed native applied-action count."""
    env = SimpleNamespace(
        episode_length_buf=torch.tensor(
            (
                applied_actions_before_timeout - 1,
                applied_actions_before_timeout,
                applied_actions_before_timeout + 1,
            )
        ),
    )

    torch.testing.assert_close(motion_time_out(env, applied_actions_before_timeout), torch.tensor((False, True, True)))


def test_g1_lafan_preset_selects_the_paired_environment_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """One preset token should select both G1-LAFAN environment and learner profiles."""
    from isaaclab_tasks.core.multi_task.motion.config.agents import G1LafanForwardBackwardRunnerCfg
    from isaaclab_tasks.utils.hydra import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["test", "presets=g1_lafan"])

    env_cfg, runner_cfg = resolve_task_config("Isaac-Motion-Imitation-v0", "rsl_rl_cfg_entry_point")

    assert env_cfg.commands.motion.task_table.source.identifier == "g1_lafan"
    assert env_cfg.decimation == 4
    assert str(env_cfg.actions.joint_position.class_type).endswith(":MotionJointPositionAction")
    assert isinstance(runner_cfg, G1LafanForwardBackwardRunnerCfg)
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


@pytest.mark.parametrize(
    ("compute_final_obs", "include_final_obs", "expected"),
    (
        (True, True, torch.tensor((False, True))),
        (True, False, torch.tensor((False, False))),
        (False, True, torch.tensor((False, False))),
    ),
)
def test_motion_environment_marks_only_materialized_final_observations_valid(
    monkeypatch: pytest.MonkeyPatch,
    compute_final_obs: bool,
    include_final_obs: bool,
    expected: torch.Tensor,
) -> None:
    """Validity must not claim an absent capture or a stale disabled-capture value."""
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(compute_final_obs=compute_final_obs)
    env.obs_buf = {"policy": torch.zeros(2, 1)}
    env.episode_length_buf = torch.zeros(2, dtype=torch.int64)
    env._transition_episode_steps = torch.empty(2, dtype=torch.int64)
    env._final_observation_valid = torch.zeros(2, dtype=torch.bool)
    env._motion_runtime = SimpleNamespace(
        capture_current=lambda observations: None,
        action_applied=torch.ones(2, dtype=torch.bool),
        auxiliary_evidence=torch.empty(2, 0),
    )

    def _base_step(_env, _action):
        extras = {"final_obs": {"policy": torch.ones(2, 1)}} if include_final_obs else {}
        return (
            env.obs_buf,
            torch.zeros(2),
            torch.tensor((False, True)),
            torch.zeros(2, dtype=torch.bool),
            extras,
        )

    monkeypatch.setattr(ManagerBasedRLEnv, "step", _base_step)
    pointer = env._final_observation_valid.data_ptr()
    *_, extras = MotionImitationEnv.step(env, torch.zeros(2, 1))

    torch.testing.assert_close(extras["final_obs_valid"], expected)
    assert extras["final_obs_valid"].data_ptr() == pointer


def test_exact_motion_reset_binds_after_normal_manager_reset_and_clears_pending_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation resets must retain normal manager ordering and one-shot exact binding."""
    events: list[object] = []
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._evaluation_clip_indices = None
    env._evaluation_all_env_ids = torch.arange(2, dtype=torch.int64)
    env._evaluation_task_rows = torch.empty(2, dtype=torch.int64)
    payload = object.__new__(MotionStatePayload)

    def bind_clip_start(env_ids: torch.Tensor, clip_indices: torch.Tensor) -> None:
        events.append(("bind", env_ids.clone(), clip_indices.clone()))

    payload.bind_clip_start = bind_clip_start
    command = SimpleNamespace(
        payload=payload,
        table=SimpleNamespace(clip_start_rows=torch.tensor((1, 0), dtype=torch.int64)),
        cmd_indices=torch.zeros(2, dtype=torch.int64),
    )
    env.command_manager = SimpleNamespace(
        get_term=lambda name: command if name == "motion" else None,
    )

    def base_reset_idx(_env, env_ids) -> None:
        events.append(("base", torch.as_tensor(env_ids).clone()))

    def base_reset(_env, *, env_ids):
        _env._reset_idx(env_ids)
        return {"state": torch.zeros(2, 1)}, {}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", base_reset_idx)
    monkeypatch.setattr(ManagerBasedRLEnv, "reset", base_reset)

    result = MotionImitationEnv.reset_motion_clips(env, torch.tensor((1, 0), dtype=torch.int64))

    assert result[0]["state"].shape == (2, 1)
    assert [event[0] for event in events] == ["base", "bind"]


def test_exact_motion_reset_binds_only_selected_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluation refill must reset and rebind only the lanes receiving new clips."""
    events: list[object] = []
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=4)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._evaluation_clip_indices = None
    env._evaluation_task_rows = torch.empty(4, dtype=torch.int64)
    payload = object.__new__(MotionStatePayload)

    def bind_clip_start(env_ids: torch.Tensor, clip_indices: torch.Tensor) -> None:
        events.append(("bind", env_ids.clone(), clip_indices.clone()))

    payload.bind_clip_start = bind_clip_start
    command = SimpleNamespace(
        payload=payload,
        table=SimpleNamespace(clip_start_rows=torch.tensor((1, 2, 0), dtype=torch.int64)),
        cmd_indices=torch.zeros(4, dtype=torch.int64),
    )
    env.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)

    def base_reset_idx(_env, env_ids) -> None:
        events.append(("base", torch.as_tensor(env_ids).clone()))

    def base_reset(_env, *, env_ids):
        _env._reset_idx(env_ids)
        return {"state": torch.zeros(4, 1)}, {}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", base_reset_idx)
    monkeypatch.setattr(ManagerBasedRLEnv, "reset", base_reset)
    env_ids = torch.tensor((1, 3), dtype=torch.int64)
    clip_indices = torch.tensor((2, 0), dtype=torch.int64)

    result = MotionImitationEnv.reset_motion_clips(env, clip_indices, env_ids=env_ids)

    assert result[0]["state"].shape == (4, 1)
    assert [event[0] for event in events] == ["base", "bind"]
    torch.testing.assert_close(events[0][1], env_ids)
    torch.testing.assert_close(events[1][1], env_ids)
    torch.testing.assert_close(events[1][2], clip_indices)
    torch.testing.assert_close(command.cmd_indices, torch.tensor((0, 0, 0, 1), dtype=torch.int64))
    assert env._evaluation_clip_indices is None

    with pytest.raises(RuntimeError, match="strictly increasing"):
        MotionImitationEnv.reset_motion_clips(
            env,
            clip_indices,
            env_ids=torch.tensor((3, 1), dtype=torch.int64),
        )
    with pytest.raises(RuntimeError, match="must be in range"):
        MotionImitationEnv.reset_motion_clips(
            env,
            clip_indices,
            env_ids=torch.tensor((1, 4), dtype=torch.int64),
        )


def test_exact_motion_reset_uses_direct_clip_start_rows() -> None:
    """Evaluation refill must not allocate a task-row-by-clip broadcast matrix."""
    source = inspect.getsource(MotionImitationEnv._bind_motion_clip_starts)

    assert "clip_start_rows" in source
    assert "torch.index_select" in source
    assert "[:, None]" not in source


def test_selected_motion_reset_requires_stateless_manager_features() -> None:
    """Selected refill is available only when the observation manager owns no hidden state."""
    term = SimpleNamespace(history_length=0, noise=None, modifiers=None)
    manager = SimpleNamespace(
        _group_obs_class_term_cfgs={"policy": []},
        _group_obs_class_instances=[],
        _group_obs_term_cfgs={"policy": [term]},
    )

    assert _observation_manager_supports_selected_reset(manager)
    term.noise = object()
    assert not _observation_manager_supports_selected_reset(manager)
    term.noise = None
    term.history_length = 1
    assert not _observation_manager_supports_selected_reset(manager)
    term.history_length = 0
    term.modifiers = [object()]
    assert not _observation_manager_supports_selected_reset(manager)
    term.modifiers = None
    manager._group_obs_class_term_cfgs["policy"] = [object()]
    assert not _observation_manager_supports_selected_reset(manager)


def test_selected_motion_reset_preserves_active_lanes_without_training_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed evaluation must reset selected lanes without mutating another active trajectory."""
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(
        num_envs=3,
        reset=lambda env_ids: None,
        write_data_to_sim=lambda: None,
    )
    env.sim = SimpleNamespace(
        device=torch.device("cpu"),
        forward=lambda: None,
    )
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(num_rerenders_on_reset=0)
    env.extras = {}
    env.obs_buf = {"policy": torch.tensor(((1.0,), (2.0,), (3.0,)))}
    env._evaluation_selected = torch.zeros(3, dtype=torch.bool)
    env._evaluation_task_rows = torch.empty(3, dtype=torch.int64)
    env._selected_reset_observations_are_stateless = True
    env.episode_length_buf = torch.tensor((7, 8, 9), dtype=torch.int64)
    reset_calls: list[tuple[str, torch.Tensor]] = []

    def compute_observations(*, update_history: bool) -> dict[str, torch.Tensor]:
        assert update_history is False
        return {"policy": torch.tensor(((10.0,), (20.0,), (30.0,)))}

    env.observation_manager = SimpleNamespace(
        reset=lambda env_ids: reset_calls.append(("observation", env_ids.clone())),
        compute=compute_observations,
    )
    env.action_manager = SimpleNamespace(
        reset=lambda env_ids: reset_calls.append(("action", env_ids.clone())),
    )
    env._bind_motion_clip_starts = lambda env_ids, clip_indices: reset_calls.append(
        ("bind", torch.stack((env_ids, clip_indices)))
    )
    monkeypatch.setattr(
        ManagerBasedRLEnv,
        "reset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training reset must not run")),
    )

    observations, _extras = MotionImitationEnv.reset_motion_clips_selected(
        env,
        torch.tensor((4,), dtype=torch.int64),
        env_ids=torch.tensor((1,), dtype=torch.int64),
    )

    torch.testing.assert_close(observations["policy"], torch.tensor(((1.0,), (20.0,), (3.0,))))
    torch.testing.assert_close(env.episode_length_buf, torch.tensor((7, 0, 9)))
    assert [name for name, _value in reset_calls] == ["observation", "action", "bind"]


def test_selected_motion_reset_rejects_stateful_observations_before_mutation() -> None:
    """History or corruption must fail before any selected environment state is reset."""
    env = object.__new__(MotionImitationEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._evaluation_task_rows = torch.empty(2, dtype=torch.int64)
    env._selected_reset_observations_are_stateless = False
    reset_called = False

    def reset_scene(_env_ids: torch.Tensor) -> None:
        nonlocal reset_called
        reset_called = True

    env.scene = SimpleNamespace(num_envs=2, reset=reset_scene)

    with pytest.raises(RuntimeError, match="history-free, corruption-free stateless"):
        MotionImitationEnv.reset_motion_clips_selected(
            env,
            torch.tensor((0,), dtype=torch.int64),
            env_ids=torch.tensor((1,), dtype=torch.int64),
        )

    assert reset_called is False


def _copy_g1_history_source(
    fields: dict[str, torch.Tensor],
    trace: np.lib.npyio.NpzFile,
    prefix: str,
    step: int,
    action_step: int,
) -> None:
    """Copy one native reached row into the fixed caller-owned source buffers."""
    state = torch.from_numpy(trace[f"{prefix}_state"][step])
    fields["processed_action"].copy_(torch.from_numpy(trace["processed_action"][action_step]))
    fields["base_angular_velocity"].copy_(state[:, 61:64])
    fields["joint_position"].copy_(state[:, :29])
    fields["joint_velocity"].copy_(state[:, 29:58])
    fields["projected_gravity"].copy_(state[:, 58:61])


def test_applied_transition_history_matches_g1_before_boundary_and_excludes_reset_seed() -> None:
    """Field order and pre-append timing match native G1 while the known seed bug stays removed."""
    layout = AppliedTransitionHistoryLayout(
        fields=(
            ("processed_action", 29),
            ("base_angular_velocity", 3),
            ("joint_position", 29),
            ("joint_velocity", 29),
            ("projected_gravity", 3),
        ),
        length=4,
    )
    value = torch.zeros(2, layout.width)
    fields = {name: torch.empty(2, width) for name, width in layout.fields}
    applied = torch.ones(2, dtype=torch.bool)
    history = AppliedTransitionHistory(layout, value, fields=fields, applied=applied)

    assert history.value is value
    with np.load(_fixtures() / "g1_lafan_same_step_trace_v1.npz", allow_pickle=False) as trace:
        torch.testing.assert_close(history.value, torch.from_numpy(trace["current_history_actor"][0]))

        _copy_g1_history_source(fields, trace, "returned", 0, 0)
        history.append()
        # History is observed before appending the current reached state. It
        # therefore appears one physical edge later, exactly as in BFM.
        torch.testing.assert_close(history.value, torch.from_numpy(trace["returned_history_actor"][1]))

        _copy_g1_history_source(fields, trace, "returned", 1, 1)
        history.append()
        torch.testing.assert_close(history.value, torch.from_numpy(trace["final_history_actor"][2]))

        history.reset(torch.arange(2))
        assert not history.value.any()
        # Native BFM inserts its internal autoreset seed after returning row 2,
        # which leaks into row 3. The corrected common history deliberately does
        # not reproduce that bug.
        assert torch.from_numpy(trace["returned_history_actor"][3]).any()
        assert not history.value.any()


@pytest.mark.parametrize("num_envs", (1, 16, 1024))
def test_applied_transition_history_is_fixed_shape_and_clone_invariant(num_envs: int) -> None:
    """The same in-place tensor law must hold at single, small-vector, and native scale."""
    layout = AppliedTransitionHistoryLayout(fields=(("x", 2),), length=3)
    value = torch.zeros(num_envs, layout.width)
    fields = {"x": torch.empty(num_envs, 2)}
    applied = torch.empty(num_envs, dtype=torch.bool)
    history = AppliedTransitionHistory(layout, value, fields=fields, applied=applied)
    storage_pointer = value.untyped_storage().data_ptr()

    row = torch.arange(num_envs, dtype=torch.float32)[:, None].repeat(1, 2)
    fields["x"].copy_(row)
    applied.fill_(True)
    history.append()
    expected = torch.cat((row, row.new_zeros(num_envs, 4)), dim=-1)
    torch.testing.assert_close(history.value, expected, rtol=0.0, atol=0.0)

    next_row = row + 1000.0
    fields["x"].copy_(next_row)
    applied.copy_(torch.arange(num_envs) % 2 == 0)
    history.append()
    updated = torch.arange(0, num_envs, 2)
    expected[updated, 2:4] = row[updated]
    expected[updated, :2] = next_row[updated]
    torch.testing.assert_close(history.value, expected, rtol=0.0, atol=0.0)

    reset = torch.arange(0, num_envs, 2)
    history.reset(reset)
    expected[reset] = 0.0
    torch.testing.assert_close(history.value, expected, rtol=0.0, atol=0.0)
    assert history.value is value
    assert history.value.untyped_storage().data_ptr() == storage_pointer


def test_applied_transition_history_validates_fixed_buffers_at_construction() -> None:
    """Bad shapes and dtypes must fail at binding rather than inside the hot append path."""
    layout = AppliedTransitionHistoryLayout(fields=(("x", 2),), length=3)
    value = torch.zeros(4, layout.width)
    fields = {"x": torch.zeros(4, 2)}
    applied = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="value must have shape"):
        AppliedTransitionHistory(layout, torch.zeros(4, layout.width + 1), fields=fields, applied=applied)
    with pytest.raises(ValueError, match="History field 'x'"):
        AppliedTransitionHistory(layout, value, fields={"x": torch.zeros(4, 3)}, applied=applied)
    with pytest.raises(ValueError, match="must not alias"):
        alias = value.view(-1)[:8].view(4, 2)
        AppliedTransitionHistory(layout, value, fields={"x": alias}, applied=applied)
    with pytest.raises(ValueError, match="applied must be boolean"):
        AppliedTransitionHistory(layout, value, fields=fields, applied=torch.zeros(4))
