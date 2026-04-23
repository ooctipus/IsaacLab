# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static verification that the new :class:`MultiTaskEnvCfg` constructs cleanly.

Covers only the "the cfg loads" layer — no sim required. Catches regressions in:

- Import graph (env cfg module + transitive imports).
- Configclass instantiation + ``__post_init__`` arithmetic.
- Gym registration landing with the expected entry point.
- Every :class:`DoneTerm` declared with ``time_out=False`` (finite-horizon
  framing — a regression on this flag silently breaks value bootstrap semantics).

Live-sim validation (actual Articulation reads, rollouts, rewards) is the job of
``utils/tools/smoke_multi_task_command.py``, which requires Isaac Sim.
"""

from __future__ import annotations

import gymnasium as gym


def test_gym_task_registers():
    """``Isaac-Position-MultiTask-v0`` is registered with the expected entry point."""
    import isaaclab_tasks  # noqa: F401 — triggers registration

    assert "Isaac-Position-MultiTask-v0" in gym.envs.registry
    spec = gym.envs.registry["Isaac-Position-MultiTask-v0"]
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg:MultiTaskEnvCfg"
    )


def test_env_cfg_imports_and_constructs():
    """Importing and instantiating :class:`MultiTaskEnvCfg` does not raise."""
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    # Sanity: key fields landed on the instance after ``__post_init__``.
    assert cfg.decimation == 4
    assert cfg.episode_length_s == 4.0
    assert cfg.sim.dt == 0.005
    assert cfg.sim.render_interval == cfg.decimation


def test_every_termination_is_not_time_out():
    """Regression gate: every ``DoneTerm`` in this env cfg has ``time_out=False``.

    Stage-3 decision: finite-horizon framing, rsl_rl bootstrap must not fire. If a
    future change flips any of these to ``time_out=True`` (or adds a new DoneTerm
    with the default flag), this test fails loudly.
    """
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    # Collect every DoneTerm attribute on the terminations cfg.
    terminations = cfg.terminations
    done_terms = {name: term for name, term in vars(terminations).items() if isinstance(term, DoneTerm)}
    assert done_terms, "Env cfg has no DoneTerms — sanity failure."
    leaks = [name for name, term in done_terms.items() if term.time_out]
    assert not leaks, f"Termination(s) with time_out=True (bootstrap would fire): {leaks}"


def test_commands_cfg_is_multitask():
    """The ``goal_point`` command is a :class:`MultiTaskCfg` (not the legacy one)."""
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import MultiTaskCfg
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    assert isinstance(cfg.commands.goal_point, MultiTaskCfg)


def test_all_tasks_are_present():
    """The env exposes every registered task name, including compound tasks."""
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    task_names = set(cfg.commands.goal_point.tasks.keys())
    expected = {
        "velocity",
        "position",
        "reach_point_in_air",
        "pose",
        "two_feet_stand",
        "reach_while_tripod",
        "reach_at_target_speed",
    }
    assert task_names == expected, f"task key mismatch: got {task_names}, expected {expected}"


def test_reach_while_tripod_structure():
    """``reach_while_tripod`` is one base-pos instant + four foot-z instants.

    Exercises the multi-entity path in the spec builder (feet are distinct entities
    from the base) and multi-instant gating in the composer.
    """
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import MultiTaskCfg
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    subtasks = cfg.commands.goal_point.tasks["reach_while_tripod"]
    # 1 base pos + 4 foot z = 5 instants.
    assert len(subtasks) == 5
    assert all(isinstance(st, MultiTaskCfg.InstantaneousTaskCfg) for st in subtasks)

    # Three feet target ground (z_max ≤ 0.1); one foot targets lifted (z_min ≥ 0.1).
    foot_subtasks = subtasks[1:]
    ground_count = sum(1 for st in foot_subtasks if max(st.sampler.maximum) < 0.1)
    lifted_count = sum(1 for st in foot_subtasks if min(st.sampler.minimum) > 0.1)
    assert ground_count == 3, f"expected 3 feet on ground, got {ground_count}"
    assert lifted_count == 1, f"expected 1 foot lifted, got {lifted_count}"


def test_reach_at_target_speed_structure():
    """``reach_at_target_speed`` is one base-pos instant + one base-speed tracking."""
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import MultiTaskCfg
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    subtasks = cfg.commands.goal_point.tasks["reach_at_target_speed"]
    assert len(subtasks) == 2
    # One instant (base pos), one tracking (base speed).
    types = [type(st).__name__ for st in subtasks]
    assert "InstantaneousTaskCfg" in types
    assert "TrackingTaskCfg" in types

    # Speed sampler range must be ∈ [0.2, 1.5] m/s.
    tracking = next(st for st in subtasks if isinstance(st, MultiTaskCfg.TrackingTaskCfg))
    assert tracking.sampler.minimum == [0.2]
    assert tracking.sampler.maximum == [1.5]


def test_task_type_composition_is_correct():
    """Each task's subtask list uses the expected instant/tracking types."""
    from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import MultiTaskCfg
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    tasks = cfg.commands.goal_point.tasks

    # velocity = two tracking subtasks (lin_vel, ang_vel).
    assert len(tasks["velocity"]) == 2
    assert all(isinstance(st, MultiTaskCfg.TrackingTaskCfg) for st in tasks["velocity"])

    # position / reach_point_in_air = one instant body-pos subtask each.
    for name in ("position", "reach_point_in_air"):
        assert len(tasks[name]) == 1
        assert isinstance(tasks[name][0], MultiTaskCfg.InstantaneousTaskCfg)

    # pose = body-pos + body-quat, both instant.
    assert len(tasks["pose"]) == 2
    assert all(isinstance(st, MultiTaskCfg.InstantaneousTaskCfg) for st in tasks["pose"])

    # two_feet_stand = one body-quat instant subtask.
    assert len(tasks["two_feet_stand"]) == 1
    assert isinstance(tasks["two_feet_stand"][0], MultiTaskCfg.InstantaneousTaskCfg)


def test_reach_point_in_air_z_target_is_elevated():
    """Regression guard: the air-reach sampler must have z > standing height.

    If someone copy-pastes the ground-level position sampler into this task, this
    test fails loudly.
    """
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    air = cfg.commands.goal_point.tasks["reach_point_in_air"][0]
    # sampler min/max for z is the 3rd component.
    assert air.sampler.minimum[2] >= 1.0
    assert air.sampler.maximum[2] >= air.sampler.minimum[2]


def test_two_feet_stand_targets_large_pitch():
    """The two-feet-stand quat sampler must target near-vertical pitch."""
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    stand = cfg.commands.goal_point.tasks["two_feet_stand"][0]
    # Sampler's minimum/maximum encode (roll, pitch, yaw) Euler mins/maxs.
    pitch_min, pitch_max = stand.sampler.minimum[1], stand.sampler.maximum[1]
    assert min(abs(pitch_min), abs(pitch_max)) > 1.0, (
        f"two_feet_stand pitch target should be ≈ ±π/2, got [{pitch_min}, {pitch_max}]"
    )


def test_pose_quat_sampler_has_out_dim_override():
    """The quat sampler in the pose task must declare ``out_dim=4``.

    Without that override, ``target_dim_max`` derived from the 3-Euler param count
    would be 3, which can't hold a 4-vec quaternion — the command term would raise
    at ``_compute_state_delta_error`` when dim_x_cur=4.
    """
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    quat_subtask = cfg.commands.goal_point.tasks["pose"][1]
    assert quat_subtask.sampler.out_dim == 4


def test_pose_quat_sampler_param_length_matches_four_pairs():
    """``get_kernel_input`` must emit 8 floats (4 interleaved min/range pairs) for pose quat.

    Guards the tight coupling between ``out_dim`` and the padded param tensor length —
    regression here would quietly produce a ``target_dim_max < 4`` and crash at runtime.
    """
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import MultiTaskEnvCfg

    cfg = MultiTaskEnvCfg()
    quat_subtask = cfg.commands.goal_point.tasks["pose"][1]
    params = quat_subtask.sampler.get_kernel_input(device="cpu")
    assert params.numel() == 8  # 4 (min, range) pairs


def test_reward_term_points_at_task_reward():
    """The task reward term resolves to the command term's ``task_reward``."""
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import (
        MultiTaskEnvCfg,
        command_task_reward,
    )

    cfg = MultiTaskEnvCfg()
    assert cfg.rewards.task.func is command_task_reward
    assert cfg.rewards.task.params == {"command_name": "goal_point"}


def test_success_done_term_points_at_task_done():
    """The success termination resolves to the command term's ``task_done``."""
    from isaaclab_tasks.manager_based.locomotion.position.multi_task_env_cfg import (
        MultiTaskEnvCfg,
        command_task_done,
    )

    cfg = MultiTaskEnvCfg()
    assert cfg.terminations.success.func is command_task_done
    assert cfg.terminations.success.params == {"command_name": "goal_point"}
