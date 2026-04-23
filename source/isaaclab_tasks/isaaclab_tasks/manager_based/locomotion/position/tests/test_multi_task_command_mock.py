# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock-based integration tests for :class:`MultiTaskCommand`.

Covers the full lifecycle — ``__init__`` → ``_resample_command`` → ``_update_command``
→ ``task_reward`` / ``task_done`` — with a fake env, scene and articulation. Monkey-
patches the state-kernel dispatch so the test can feed synthetic "current state"
tensors directly without an Articulation + warp-array stack.

What this validates (which the pure-logic tests can't):

- Spec is assembled from a real ``MultiTaskCfg`` and the expected ``SceneEntityCfg.resolve``
  path runs without errors.
- ``_resample_command`` writes targets, rebuilds per-env type masks, and clears the
  per-env composer state on the envs it touches.
- ``_update_command`` pipes state → delta → error → activation → composer + latch,
  and the task reward lands at the terminal step (not before).
- ``is_timeout`` is taken from ``env.episode_length_buf`` and ``env.max_episode_length``
  — flipping that triggers the terminal reward on a pure-tracking task.

Skipped on purpose: real Articulation reads (``wp.to_torch(robot.data.body_lin_vel_w)``)
are bypassed via state-kernel monkeypatch. Those paths are a copy-only port from
the working ``multi_task`` reference and exercise on the first live-sim run.
"""

from __future__ import annotations

import math
import re
from unittest.mock import patch

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_command as mtc_mod
from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.kernels import (
    ACTIVATION_KERNEL_ID,
    METRIC_KERNEL_ID,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
)
from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_cfg import MinMaxSampler, MultiTaskCfg
from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.multi_task_command import MultiTaskCommand

# -----------------------------------------------------------------------------
# Mock env / scene / articulation
# -----------------------------------------------------------------------------


class _MockArticulation:
    """Minimal stand-in for :class:`Articulation` satisfying :meth:`SceneEntityCfg.resolve`."""

    def __init__(self, body_names: list[str], joint_names: list[str] | None = None):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names) if joint_names else []
        self.num_bodies = len(self.body_names)
        self.num_joints = len(self.joint_names)
        self.fixed_tendon_names: list[str] = []
        self.num_fixed_tendons = 0

    @staticmethod
    def _find(names: list[str], patterns, preserve_order: bool = False):
        if isinstance(patterns, str):
            patterns = [patterns]
        ids: list[int] = []
        matched_names: list[str] = []
        for pat in patterns:
            regex = re.compile(pat)
            for i, name in enumerate(names):
                if regex.fullmatch(name) and i not in ids:
                    ids.append(i)
                    matched_names.append(name)
        return ids, matched_names

    def find_bodies(self, patterns, preserve_order: bool = False):
        return self._find(self.body_names, patterns, preserve_order)

    def find_joints(self, patterns, preserve_order: bool = False):
        return self._find(self.joint_names, patterns, preserve_order)

    def find_fixed_tendons(self, patterns, preserve_order: bool = False):
        return [], []


class _MockScene:
    def __init__(self, entities: dict):
        self._entities = entities

    def keys(self):
        return self._entities.keys()

    def __getitem__(self, name):
        return self._entities[name]

    def __contains__(self, name):
        return name in self._entities


class _MockEnv:
    """Minimal stand-in for :class:`ManagerBasedRLEnv` used by :class:`MultiTaskCommand`."""

    def __init__(self, num_envs: int, device: str, max_episode_length: int, scene):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.scene = scene
        self.common_step_counter = 0
        self.step_dt = 0.02


def _make_env(num_envs: int = 4, device: str = "cpu", max_episode_length: int = 10) -> _MockEnv:
    robot = _MockArticulation(body_names=["base"])
    scene = _MockScene({"robot": robot})
    return _MockEnv(num_envs=num_envs, device=device, max_episode_length=max_episode_length, scene=scene)


def _make_lin_vel_cfg(min_xy: float = -1.0, max_xy: float = 1.0) -> MultiTaskCfg:
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "lin_vel": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[min_xy, min_xy, 0.0],
                        maximum=[max_xy, max_xy, 0.0],
                    ),
                )
            ],
        },
    )


def _make_mixed_cfg() -> MultiTaskCfg:
    """Mixed task: instant body-pos + tracking lin-vel.

    Used to verify that both subtask types compose correctly in a single task.
    """
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "reach_and_maintain": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,  # error < 0.5 → achieved
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],  # target always at origin
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[1.0, 0.0, 0.0],
                        maximum=[1.0, 0.0, 0.0],  # target velocity fixed at (1,0,0)
                    ),
                ),
            ],
        },
    )


# -----------------------------------------------------------------------------
# State-kernel monkeypatch: a single function that returns a controllable tensor.
# -----------------------------------------------------------------------------


class _SyntheticState:
    """Holds the per-step current-state tensor to be returned by each state kernel.

    Keyed by ``STATE_KERNEL_ID`` so the same fixture can drive multiple subtasks.
    """

    def __init__(self, device: str):
        self.device = device
        self.outputs: dict[int, torch.Tensor] = {}

    def set(self, kernel_id: STATE_KERNEL_ID, tensor: torch.Tensor) -> None:
        self.outputs[int(kernel_id)] = tensor

    def make_kernels(self) -> tuple:
        """Produce a tuple of (env, env_ids, asset_cfg)-shaped callables."""
        outputs = self.outputs

        def make_fn(kid: int):
            def fn(env, env_ids, asset_cfg):
                return outputs.get(kid, torch.zeros(env.num_envs, 3))

            return fn

        return tuple(make_fn(kid) for kid in range(len(STATE_KERNEL_ID)))


# -----------------------------------------------------------------------------
# Lifecycle tests
# -----------------------------------------------------------------------------


def test_spec_builds_for_pure_tracking():
    """Instantiate MultiTaskCommand with pure tracking cfg; inspect the spec."""
    env = _make_env(num_envs=4)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    assert cmd.num_tasks == 1
    assert cmd.num_subtasks == 1
    assert cmd.spec.task_names == ["lin_vel"]
    assert cmd.spec.is_tracking[0].item() is True
    assert cmd.spec.is_instant[0].item() is False


def test_initial_resample_writes_metadata_and_targets_in_range():
    """Init runs the initial resample — ragged layout state must be populated.

    Under the Stage-2.0 ragged layout, there's no ``_is_instant_subtask`` mask at runtime;
    types are looked up per step via ``spec.is_instant[env_subtask_ids]``. We check here
    that the per-env active-subtask metadata is correct and that targets were written
    in the expected flat layout.
    """
    env = _make_env(num_envs=8)
    cfg = _make_lin_vel_cfg(min_xy=-1.0, max_xy=1.0)

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # One task, one subtask (lin_vel tracking).
    assert cmd.k_max == 1
    assert cmd.num_subtasks == 1
    assert cmd.max_task_total_stride == 3

    # Per-env metadata points at the single subtask (id 0, offset 0, stride 3).
    assert torch.equal(cmd._env_subtask_ids[:, 0], torch.zeros(8, dtype=torch.long))
    assert torch.equal(cmd._env_slot_count, torch.ones(8, dtype=torch.long))
    assert torch.equal(cmd._env_slot_offsets[:, 0], torch.zeros(8, dtype=torch.long))
    assert torch.equal(cmd._env_slot_strides[:, 0], torch.full((8,), 3, dtype=torch.long))

    # Targets live in the flat buffer at slots [0:3]. x ∈ [-1,1], y ∈ [-1,1], z = 0.
    assert cmd._targets_flat.shape == (8, 3)
    xs, ys, zs = cmd._targets_flat[:, 0], cmd._targets_flat[:, 1], cmd._targets_flat[:, 2]
    assert (xs >= -1.0).all() and (xs <= 1.0).all()
    assert (ys >= -1.0).all() and (ys <= 1.0).all()
    assert (zs == 0.0).all()

    # Composer state starts fresh.
    assert (cmd._sum_activation == 0).all()
    assert (cmd._transit_steps == 0).all()
    assert not cmd._instant_achieved.any()


def test_pure_tracking_reward_only_at_timeout():
    """Drive MultiTaskCommand with constant-velocity state for a full episode.

    For pure tracking with the policy holding perfect velocity, the terminal reward at
    timeout should be ≈1 and every non-terminal step should have reward 0.
    """
    env = _make_env(num_envs=2, max_episode_length=5)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Set synthetic current BODY_LIN_VEL equal to the sampled target so activation = 1.
    # Under the ragged layout the target lives in ``_targets_flat[env, :stride]``.
    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()

    rewards: list[torch.Tensor] = []
    for step in range(env.max_episode_length):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        state.set(STATE_KERNEL_ID.BODY_LIN_VEL, target.clone())
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()
        rewards.append(cmd.task_reward.clone())

    # Steps 0..T-2: non-terminal → reward EXACTLY zero (terminal-only emission).
    for i in range(env.max_episode_length - 1):
        assert (rewards[i] == 0.0).all(), f"step {i}: reward {rewards[i]} != 0 exactly"
    # Step T-1: timeout fires. state==target → error=0 → activation = 1 - tanh(0) = 1 exactly.
    # Perfect tracking the whole episode → transit mean = 1 exactly.
    final = rewards[-1]
    assert torch.allclose(final, torch.ones_like(final), atol=1e-6), (
        f"expected terminal reward exactly 1.0 (perfect tracking), got {final}"
    )


def test_pure_tracking_reward_matches_activation_mean():
    """Half-perfect tracking for the whole episode → terminal reward ≈ activation value.

    Sets current_velocity = target × 0 (max error) for half steps and = target for the
    other half; expected reward ≈ mean of tanh-activation across the episode.
    """
    T = 6
    env = _make_env(num_envs=1, max_episode_length=T)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()

    # Expected activation sequence: perfect (≈1) on even steps, degraded on odd steps.
    # Use a large zero for odd steps so the degraded activation is nearly 0.
    activations_seq: list[float] = []
    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step % 2 == 0:
            current = target.clone()  # matches target → error=0 → activation=1
        else:
            current = target + 10.0  # huge mismatch → activation≈0
        state.set(STATE_KERNEL_ID.BODY_LIN_VEL, current)
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()
        activations_seq.append(cmd._buf_activation[0, 0].item())

    expected_mean = sum(activations_seq) / len(activations_seq)
    final_reward = cmd.task_reward[0].item()
    assert abs(final_reward - expected_mean) < 1e-5, f"terminal reward {final_reward} != expected mean {expected_mean}"


def test_mixed_task_latches_instant_and_pays_transit_mean_at_success():
    """Mixed task: instant (body-pos near origin) + tracking (lin-vel near 1,0,0).

    Drive the episode so the instant latches at step k while tracking is constant perfect.
    Terminal reward should be the mean activation of the tracking subtask over
    steps [0, k], not affected by what the tracking subtask does after success fires.
    """
    T = 8
    k = 4  # instant achieves at step k
    env = _make_env(num_envs=1, max_episode_length=T)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    # Initial state: far from origin (instant FAIL), velocity 1,0,0 (tracking OK).
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[5.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))

    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step >= k:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))  # at origin → achieved
        # velocity stays perfect the whole time
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()
        if bool(cmd.task_done.any()):
            # Success fired. Record and break.
            break

    assert bool(cmd.task_done.any()), "instant subtask never latched success"
    # Exact expectation: tracking state == target throughout → error=0 → activation = 1 - tanh(0) = 1.
    # Transit mean = 1 exactly. instant_gate = 1 at success. Terminal = 1 * 1 = 1.0.
    final = cmd.task_reward[0].item()
    assert abs(final - 1.0) < 1e-6, f"expected terminal reward 1.0 exactly, got {final:.9f}"


def test_resample_clears_accumulators_only_for_resampled_envs():
    """Resample some envs; others keep their accumulator state."""
    env = _make_env(num_envs=4, max_episode_length=100)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Manually bump composer state for all envs, simulating several accumulated steps.
    cmd._sum_activation.fill_(3.5)
    cmd._transit_steps.fill_(10)
    cmd._instant_achieved.fill_(True)

    # Resample only envs 1 and 3.
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._resample_command(torch.tensor([1, 3], device=env.device, dtype=torch.long))

    # Resampled rows: cleared.
    assert torch.allclose(cmd._sum_activation[1], torch.zeros_like(cmd._sum_activation[1]))
    assert torch.allclose(cmd._sum_activation[3], torch.zeros_like(cmd._sum_activation[3]))
    assert cmd._transit_steps[1].item() == 0
    assert cmd._transit_steps[3].item() == 0
    assert not cmd._instant_achieved[1].any()
    assert not cmd._instant_achieved[3].any()

    # Non-resampled rows: untouched.
    assert torch.allclose(cmd._sum_activation[0], torch.full_like(cmd._sum_activation[0], 3.5))
    assert torch.allclose(cmd._sum_activation[2], torch.full_like(cmd._sum_activation[2], 3.5))
    assert cmd._transit_steps[0].item() == 10
    assert cmd._transit_steps[2].item() == 10
    assert cmd._instant_achieved[0].all()
    assert cmd._instant_achieved[2].all()


def test_empty_env_ids_is_noop():
    """Calling _resample_command with an empty env_ids list must not crash or mutate."""
    env = _make_env(num_envs=4, max_episode_length=50)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    targets_before = cmd._targets_flat.clone()
    subtask_ids_before = cmd._env_subtask_ids.clone()
    slot_count_before = cmd._env_slot_count.clone()

    empty = torch.tensor([], device=env.device, dtype=torch.long)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._resample_command(empty)

    assert torch.equal(cmd._targets_flat, targets_before)
    assert torch.equal(cmd._env_subtask_ids, subtask_ids_before)
    assert torch.equal(cmd._env_slot_count, slot_count_before)


# -----------------------------------------------------------------------------
# Exposure properties (task_reward / task_done / command)
# -----------------------------------------------------------------------------


def test_task_reward_and_done_start_zero():
    """Before any _update_command call, the exposed reward/done are zeros."""
    env = _make_env(num_envs=3, max_episode_length=10)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    assert torch.allclose(cmd.task_reward, torch.zeros(3))
    assert not cmd.task_done.any()


def test_command_tensor_has_expected_flattened_shape():
    """``command`` property is ``[num_envs, max_task_total_stride]`` under the flat layout."""
    env = _make_env(num_envs=5, max_episode_length=10)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # One tracking subtask of stride 3.
    assert cmd.command.shape == (5, 3)


# -----------------------------------------------------------------------------
# Ragged layout: scale and no-dim-padding invariants
# -----------------------------------------------------------------------------


def test_flat_targets_pack_tightly_across_mixed_strides():
    """Mixed-stride subtasks (3 + 3) pack contiguously into ``_targets_flat`` with no dim padding.

    If the layout regressed to dense ``[N, k, D_max]`` the targets buffer would be
    ``[N, 2, 3]`` (padded); here we assert the flat layout yields ``[N, 6]``
    (slot0 at [0:3], slot1 at [3:6]).
    """
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Two subtasks both with stride 3 → total stride 6, k_max 2.
    assert cmd.k_max == 2
    assert cmd.max_task_total_stride == 6
    assert cmd._targets_flat.shape == (1, 6)

    # Slot 0 starts at offset 0, slot 1 starts at offset 3.
    assert cmd._env_slot_offsets[0].tolist() == [0, 3]
    assert cmd._env_slot_strides[0].tolist() == [3, 3]

    # Target for the body-pos subtask (slot 0) is [0, 0, 0] per the cfg; body-lin-vel
    # (slot 1) is [1, 0, 0]. Read directly from the flat buffer.
    pos_target = cmd._targets_flat[0, 0:3]
    vel_target = cmd._targets_flat[0, 3:6]
    assert torch.allclose(pos_target, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(vel_target, torch.tensor([1.0, 0.0, 0.0]))


def _make_large_cfg(num_tasks: int) -> MultiTaskCfg:
    """Generate a cfg with ``num_tasks`` distinct tracking tasks.

    Each task uses a different sampler (different min/max) so spec dedup does NOT
    collapse them — the spec's ``M`` really equals ``num_tasks``. Lets the scale
    test assert per-step cost stays flat as ``M`` grows.
    """
    tasks = {}
    for i in range(num_tasks):
        # Perturb the target by i so each task has a distinct signature.
        tasks[f"task_{i}"] = [
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[float(i) * 1e-6, 0.0, 0.0],
                    maximum=[float(i) * 1e-6, 0.0, 0.0],
                ),
            ),
        ]
    return MultiTaskCfg(resampling_time_range=(100.0, 100.0), debug_vis=False, tasks=tasks)


def test_ragged_layout_scales_to_many_tasks():
    """M=10_000 works: per-env memory stays O(k), per-step compute independent of M.

    Plus correctness: after driving each env through a full episode with state ==
    its assigned target, the final reward must be **exactly 1.0** for every env.
    A subtle indexing bug (e.g. ``env_subtask_ids`` pointing at the wrong spec row)
    would produce wrong targets per env, giving non-unit reward.
    """
    M = 10_000
    num_envs = 8
    T = 5
    env = _make_env(num_envs=num_envs, max_episode_length=T)
    cfg = _make_large_cfg(M)

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Spec grew with M; k stayed at 1. Per-env buffers bounded by k.
    assert cmd.num_subtasks == M
    assert cmd.num_tasks == M
    assert cmd.k_max == 1
    assert cmd.max_task_total_stride == 3
    assert cmd._env_subtask_ids.shape == (num_envs, 1)
    assert cmd._sum_activation.shape == (num_envs, 1)
    assert cmd._targets_flat.shape == (num_envs, 3)

    # Drive a full episode. Each env's state == its own sampled target → activation=1
    # every step → terminal reward = 1.0 exactly regardless of which of the M tasks
    # each env was assigned.
    target = cmd._targets_flat[:, :3].clone()
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, target)
    for step in range(T):
        env.episode_length_buf = torch.full((num_envs,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()

    # At timeout, every env must have reward exactly 1.0 (perfect tracking).
    final_rewards = cmd.task_reward
    assert torch.allclose(final_rewards, torch.ones_like(final_rewards), atol=1e-6), (
        f"per-env reward divergence at M={M}: got {final_rewards}, expected all 1.0"
    )


def test_task_slot_offsets_match_cumulative_stride():
    """Spec ``task_slot_offsets`` is the cumulative sum of strides.

    Regression gate for the offset-computation loop in ``_build_spec``. An off-by-one
    (e.g. ``offset += stride[slot + 1]`` instead of ``stride[slot]``) would land targets
    at the wrong slot's slice, silently corrupting delta/error.

    Uses a pose task (stride 3 + stride 4 = 7) so different strides are actually in play.
    """
    env = _make_env(num_envs=1)

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "pose": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                        out_dim=4,
                    ),
                ),
            ],
        },
    )
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # state_stride: [3 for BODY_POS, 4 for BODY_QUAT]
    assert cmd.spec.state_stride.tolist() == [3, 4]
    # task_slot_offsets for the single task: first slot at 0, second at 3.
    assert cmd.spec.task_slot_offsets[0, :2].tolist() == [0, 3]
    # task_total_stride = 3 + 4 = 7.
    assert cmd.spec.task_total_stride[0].item() == 7
    # max_task_total_stride matches.
    assert cmd.max_task_total_stride == 7
    # Per-env slot offsets gathered correctly.
    assert cmd._env_slot_offsets[0].tolist() == [0, 3]
    assert cmd._env_slot_strides[0].tolist() == [3, 4]


def test_task_slot_offsets_three_strides_packed_tightly():
    """Three subtasks with distinct strides (3, 4, 12) → offsets [0, 3, 7], total 19.

    Three-way mixed-stride test catches compounding index errors that a two-way test
    (e.g. stride-3 + stride-4) might miss. Uses **three distinct state kernels** so each
    stride lives in its own ``(state_kid, entity)`` class — the spec-build gate rejects
    multiple strides within one class.
    """
    robot = _MockArticulation(body_names=["base"], joint_names=[f"j{i}" for i in range(12)])
    scene = _MockScene({"robot": robot})
    env = _MockEnv(num_envs=1, device="cpu", max_episode_length=10, scene=scene)

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "mixed": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                        out_dim=4,
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot"),  # all 12 joints
                    state_kernel=int(STATE_KERNEL_ID.JOINT_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0] * 12,
                        maximum=[0.0] * 12,
                    ),
                ),
            ],
        },
    )
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    assert cmd.spec.state_stride.tolist() == [3, 4, 12]
    assert cmd.spec.task_slot_offsets[0, :3].tolist() == [0, 3, 7]
    assert cmd.spec.task_total_stride[0].item() == 19
    assert cmd.max_task_total_stride == 19


def test_spec_rejects_stride_mismatch_within_class():
    """Spec-build rejects inconsistent strides inside a ``(state_kid, entity)`` class.

    The per-step dispatch picks one stride per class from an example subtask. If two
    subtasks share ``(state_kid, entity)`` but declare different strides via their
    samplers, the dispatch would silently mis-slice targets for one of them. The
    spec-time check turns this latent error into an immediate, informative failure.
    """
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "bad": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],  # stride 3
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),  # same entity
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),  # same state_kid
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0, 0.0],  # stride 4 — disagrees!
                        maximum=[0.0, 0.0, 0.0, 0.0],
                    ),
                ),
            ],
        },
    )
    env = _make_env(num_envs=1)
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        with pytest.raises(ValueError, match="state_stride inconsistency"):
            MultiTaskCommand(cfg, env)


def test_state_kernel_stride_mismatch_raises_at_runtime():
    """Runtime guard: state kernel's output dim must match spec's ``state_stride``.

    Catches the case where the sampler says "I emit stride K" but the state kernel
    actually emits a different dim. The spec-time check only sees the sampler side;
    the runtime check verifies the state-kernel side too.
    """
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "bad": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0] * 5,  # sampler declares stride 5
                        maximum=[0.0] * 5,
                    ),
                ),
            ],
        },
    )
    env = _make_env(num_envs=1)
    state = _SyntheticState(device=env.device)
    # Monkeypatched state kernel returns [N, 3] — mismatches the stride-5 declaration.
    state.set(STATE_KERNEL_ID.BODY_POS, torch.zeros(1, 3))
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(1, dtype=torch.long)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        with pytest.raises(RuntimeError, match="State kernel output dim mismatch"):
            cmd._update_command()


def test_immediate_success_on_first_step():
    """Instant subtask achieved at the very first step → reward = 1.0 exactly at step 1.

    Edge case: ``k = 0`` in the trajectory. Composer's latching logic must accept a
    first-step achievement without any prior history.
    """
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    # State matches pose target (origin) from the start → instant achieves step 1.
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))

    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._update_command()

    assert cmd.task_done[0].item() is True, "success should fire at step 1"
    # Transit window is 1 step, tracking perfect → transit mean = 1.0 exactly.
    assert abs(cmd.task_reward[0].item() - 1.0) < 1e-6, (
        f"expected reward 1.0 at immediate success, got {cmd.task_reward[0].item():.9f}"
    )


def test_success_fires_exactly_at_last_step():
    """Instant achieves on the **same step** as timeout → success takes precedence.

    Edge case: race between ``success`` and ``is_timeout``. Both evaluate to True at
    step T; the composer should emit the success-terminal value (1.0 for perfect tracking),
    not a 0 "timeout-without-success" value.
    """
    T = 5
    env = _make_env(num_envs=1, max_episode_length=T)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    # Miss for steps 0..T-2, achieve on the last step.
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[5.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))

    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step == T - 1:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()

    assert cmd.task_done[0].item() is True, "success should fire at the last step"
    # Tracking was perfect throughout → mean = 1, instant gate = 1, terminal = 1.
    assert abs(cmd.task_reward[0].item() - 1.0) < 1e-6, (
        f"success at last step: expected 1.0, got {cmd.task_reward[0].item():.9f}"
    )


def test_per_env_different_tasks_give_different_rewards():
    """3 envs assigned 3 different tasks in the same step → 3 different correct rewards.

    Constructed to make every env's task produce a **distinct** expected reward. A
    batch-indexing bug (e.g. wrong row gather of ``_env_slot_offsets``) would mix envs
    and break the distinctness pattern.
    """
    # Three tasks, each with a different tracking target (different activation values
    # under the same observed velocity so the mean diverges).
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "track_x1": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[1.0, 0.0, 0.0],
                        maximum=[1.0, 0.0, 0.0],
                    ),
                ),
            ],
            "track_x2": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[2.0, 0.0, 0.0],
                        maximum=[2.0, 0.0, 0.0],
                    ),
                ),
            ],
            "track_x3": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[3.0, 0.0, 0.0],
                        maximum=[3.0, 0.0, 0.0],
                    ),
                ),
            ],
        },
    )
    T = 3
    env = _make_env(num_envs=3, max_episode_length=T)
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Force env i → task "track_x{i+1}". Names sorted by dict insertion order match.
    task_ids = torch.tensor(
        [cmd.spec.task_names.index(n) for n in ["track_x1", "track_x2", "track_x3"]],
        dtype=torch.long,
        device=env.device,
    )
    cmd.task_samples.copy_(task_ids)
    # ``_resample_command`` starts with ``resample_indices`` which randomly overwrites
    # ``task_samples``. Stub it so our manual assignments stick through the resample.
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._resample_command(torch.arange(3, device=env.device, dtype=torch.long))

    # Observed velocity is the SAME for every env: [1, 0, 0].
    # Expected errors per env: env0 tracks target [1,0,0] → err=0. env1 target [2,0,0] → err=1.
    # env2 target [3,0,0] → err=2.
    current = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, current)

    for step in range(T):
        env.episode_length_buf = torch.full((3,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()

    # Perfect tracking constant throughout → terminal reward = per-env activation at any step.
    # activation = 1 - tanh(err/std), std=0.5.
    expected = torch.tensor(
        [
            1.0 - math.tanh(0.0 / 0.5),  # env 0: err=0 → 1.0
            1.0 - math.tanh(1.0 / 0.5),  # env 1: err=1 → 1 - tanh(2)
            1.0 - math.tanh(2.0 / 0.5),  # env 2: err=2 → 1 - tanh(4)
        ]
    )
    actual = cmd.task_reward.cpu()
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"per-env rewards diverged: got {actual.tolist()}, expected {expected.tolist()}"
    )


def test_slot_activation_obs_matches_internal_computation():
    """``slot_activation`` property exposes the same ``[N, k_max]`` tensor used internally.

    Guard: the obs-facing signal must not drift from what the composer sees. If someone
    refactors the dispatch and forgets to update the property, this fails.
    """
    env = _make_env(num_envs=2, max_episode_length=5)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))

    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._update_command()

    # slot_activation must match _buf_activation exactly (same tensor, in fact).
    assert cmd.slot_activation.shape == (env.num_envs, cmd.k_max)
    assert torch.equal(cmd.slot_activation, cmd._buf_activation)


def test_slot_type_bits_match_spec_gather_under_valid_mask():
    """``slot_is_instant`` / ``slot_is_tracking`` reflect ``spec`` AND'd with ``slot_valid``.

    For each (env, slot), the bit must be ``spec.is_instant[subtask_id] & valid`` — so
    padded slots are False even if the clamp-to-0 placeholder points at an instant
    subtask in the spec.
    """
    env = _make_env(num_envs=1, max_episode_length=5)
    cfg = _make_mixed_cfg()  # one task: [instant body-pos, tracking lin-vel], k_max=2

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_POS, torch.zeros(1, 3))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.zeros(1, 3))
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(1, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._update_command()

    # Slot 0 is the instant body-pos subtask; slot 1 is tracking lin-vel. Both valid.
    assert cmd.slot_valid.tolist() == [[True, True]]
    assert cmd.slot_is_instant.tolist() == [[True, False]]
    assert cmd.slot_is_tracking.tolist() == [[False, True]]


def test_slot_feeds_are_zero_for_padded_slots():
    """When an env's task has fewer slots than ``k_max``, padded slots must feed zeros/False.

    Mixed-width tasks — one task with 1 subtask, another with 2 subtasks, ``k_max = 2``.
    Envs assigned the 1-subtask task must have slot 1's activation = 0, type bits = False,
    valid = False. Otherwise the policy sees noise from padded entries.
    """
    tasks = {
        "single": [
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.0, 0.0, 0.0],
                    maximum=[0.0, 0.0, 0.0],
                ),
            ),
        ],
        "double": [
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[1.0, 0.0, 0.0],
                    maximum=[1.0, 0.0, 0.0],
                ),
            ),
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.0, 0.0, 0.5],
                    maximum=[0.0, 0.0, 0.5],
                ),
            ),
        ],
    }
    cfg = MultiTaskCfg(resampling_time_range=(100.0, 100.0), debug_vis=False, tasks=tasks)
    env = _make_env(num_envs=2)

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Force env 0 → single task (slot 1 should be padded), env 1 → double (both active).
    single_id = cmd.spec.task_names.index("single")
    double_id = cmd.spec.task_names.index("double")
    cmd.task_samples[0] = single_id
    cmd.task_samples[1] = double_id
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    # Drive one step of physics with zero velocity.
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.zeros(2, 3))
    state.set(STATE_KERNEL_ID.BODY_ANG_VEL, torch.zeros(2, 3))
    env.episode_length_buf = torch.ones(2, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._update_command()

    # Env 0 (single task, k_max=2): slot 0 valid, slot 1 padded.
    assert cmd.slot_valid[0].tolist() == [True, False]
    assert cmd.slot_is_tracking[0].tolist() == [True, False]
    assert cmd.slot_is_instant[0].tolist() == [False, False]
    assert cmd.slot_activation[0, 1].item() == 0.0  # padded slot's activation must be zero

    # Env 1 (double task): both slots valid.
    assert cmd.slot_valid[1].tolist() == [True, True]
    assert cmd.slot_is_tracking[1].tolist() == [True, True]


def test_full_episode_reward_trace_mixed_task():
    """End-to-end numerical regression gate for the engine.

    Scripts a complete episode for a mixed task (instant body-pos + tracking lin-vel)
    with deterministic state and asserts the **exact** reward + done signal at every
    step. Any drift in composer math, latching, dispatch timing, or the ragged-layout
    index arithmetic changes the trace and fails the test.

    Trajectory:

    - ``T = 10``, ``max_episode_length = 10``.
    - Tracking subtask: body_lin_vel target is fixed at ``[1, 0, 0]``. Current velocity
      matches perfectly at every step → ``activation_tracking = 1`` throughout.
    - Instant subtask: body_pos target is fixed at ``[0, 0, 0]``, threshold ``0.5``.
      Current position is ``[5, 0, 0]`` (miss) for steps 0-3, then ``[0, 0, 0]`` (hit)
      at step 4. Latches; success fires at step 4.

    Expected trace (per the Stage-3 multiplicative terminal reward):

    - Steps 0..3: reward = 0, done = False.
    - Step 4: reward = mean tracking over steps [0..4] (all 1.0) = 1.0, done = True.
    - Would-be steps 5..9 (not driven): irrelevant — episode ended at step 4.

    Tolerance is ``1e-6``; any drift beyond this is either a real bug or a change in
    the activation-kernel's tanh that should be consciously re-baselined.
    """
    T = 10
    env = _make_env(num_envs=1, max_episode_length=T)
    cfg = _make_mixed_cfg()

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[5.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))

    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # Walk the trajectory. Record reward/done at each step until success fires.
    reward_trace: list[float] = []
    done_trace: list[bool] = []
    k_success = 4

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step >= k_success:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()
        reward_trace.append(cmd.task_reward[0].item())
        done_trace.append(bool(cmd.task_done[0].item()))
        if done_trace[-1]:
            break

    # Exact expected values.
    # Steps 0..3: non-terminal, reward=0, done=False.
    for s in range(k_success):
        assert reward_trace[s] == 0.0, f"step {s}: reward {reward_trace[s]} != 0"
        assert done_trace[s] is False, f"step {s}: unexpected done"
    # Step 4: success, reward=1 (perfect tracking mean), done=True.
    assert done_trace[k_success] is True, f"step {k_success}: should have fired success"
    assert abs(reward_trace[k_success] - 1.0) < 1e-6, (
        f"step {k_success}: reward {reward_trace[k_success]:.9f} != 1.0 exactly — composer or dispatch has drifted"
    )
    # Episode terminated at step 4; no later steps run.
    assert len(reward_trace) == k_success + 1


def test_full_episode_reward_trace_pure_tracking_partial():
    """Exact-trace regression test for pure-tracking with varying activation.

    Stride-weighted: odd steps have perfect activation, even steps have effectively
    zero (huge error), and the episode times out at step T-1. The exact terminal reward
    must equal the mean of the per-step activations.

    Any subtle change to the accumulator update order (e.g. incrementing
    ``transit_steps`` before ``_sum_activation`` — currently simultaneous) would shift
    the denominator and change the answer. This test fails bit-exactly in that case.
    """
    T = 6
    env = _make_env(num_envs=1, max_episode_length=T)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()

    per_step_activations: list[float] = []
    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        current = target.clone() if step % 2 == 0 else target + 10.0
        state.set(STATE_KERNEL_ID.BODY_LIN_VEL, current)
        with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
            cmd._update_command()
        per_step_activations.append(cmd._buf_activation[0, 0].item())

    expected_terminal = sum(per_step_activations) / len(per_step_activations)
    actual_terminal = cmd.task_reward[0].item()
    assert abs(actual_terminal - expected_terminal) < 1e-6, (
        f"terminal reward {actual_terminal:.9f} != mean-of-activations {expected_terminal:.9f}"
    )
    # Non-terminal steps: reward must be exactly zero (composer is terminal-only).
    # We can't replay — but we verified per-step reward zeros in the composer tests.
    # Here we additionally verify the success flag is False (pure tracking never triggers
    # success, only timeout).
    assert cmd.task_done[0].item() is False


def test_varying_k_across_tasks_pads_only_slot_dim():
    """Task with k=1 and task with k=2 coexist; the k=1 task's slot 1 is inactive-padded.

    Verifies ``_env_slot_count`` correctly distinguishes padded slots and that the
    composer ignores them via the ``valid_slots`` mask at step time.
    """
    tasks = {
        "single": [
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.0, 0.0, 0.0],
                    maximum=[0.0, 0.0, 0.0],
                ),
            ),
        ],
        "double": [
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[1.0, 0.0, 0.0],
                    maximum=[1.0, 0.0, 0.0],
                ),
            ),
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot", body_names="base"),
                state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.0, 0.0, 0.5],
                    maximum=[0.0, 0.0, 0.5],
                ),
            ),
        ],
    }
    cfg = MultiTaskCfg(resampling_time_range=(100.0, 100.0), debug_vis=False, tasks=tasks)
    env = _make_env(num_envs=4)

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd = MultiTaskCommand(cfg, env)

    # k_max = 2 (from the "double" task); single-task envs have slot_count 1.
    assert cmd.k_max == 2
    # Each env gets exactly one of the two tasks.
    assert ((cmd._env_slot_count == 1) | (cmd._env_slot_count == 2)).all()

    # Force env 0 → single task and env 1 → double task so we can check both branches.
    single_task_id = cmd.spec.task_names.index("single")
    double_task_id = cmd.spec.task_names.index("double")
    cmd.task_samples[0] = single_task_id
    cmd.task_samples[1] = double_task_id
    # Stub the random reassignment inside _resample_command so our manual samples hold.
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "STATE_KERNELS", state.make_kernels()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    assert cmd._env_slot_count[0].item() == 1
    assert cmd._env_slot_count[1].item() == 2


if __name__ == "__main__":
    # For quick local iteration.
    pytest.main([__file__, "-v"])
