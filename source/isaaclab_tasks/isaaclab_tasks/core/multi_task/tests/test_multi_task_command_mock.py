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

import isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_command as mtc_mod
from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.kernel_ids import (
    ACTIVATION_KERNEL_ID,
    METRIC_KERNEL_ID,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
)
from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_cfg import (
    MinMaxSampler,
    MultiTaskCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.multi_task_command import MultiTaskCommand

# -----------------------------------------------------------------------------
# Mock env / scene / articulation
# -----------------------------------------------------------------------------


class _MockArticulationData:
    """Just enough of :class:`ArticulationData` for the rotation helper.

    :meth:`MultiTaskCommand._rotate_canonical_slots_to_body_frame` reads
    ``root_quat_w`` to rotate POS / LIN_VEL / ANG_VEL canonical slots into
    base frame. With identity quats the rotation is a no-op — existing
    mock-based tests remain frame-agnostic and their world-frame assertions
    still hold. The helper's ``_as_torch`` accepts torch tensors directly so
    we skip the Warp round-trip (avoids requiring ``wp.init()`` in mocks).
    """

    def __init__(self, num_envs: int, device: str):
        # Identity quat in (x, y, z, w) — matches articulation convention.
        q = torch.zeros(num_envs, 4, device=device)
        q[:, 3] = 1.0
        self.root_quat_w = q


class _MockArticulation:
    """Minimal stand-in for :class:`Articulation` satisfying :meth:`SceneEntityCfg.resolve`."""

    def __init__(
        self,
        body_names: list[str],
        joint_names: list[str] | None = None,
        num_envs: int = 8,
        device: str = "cpu",
    ):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names) if joint_names else []
        self.num_bodies = len(self.body_names)
        self.num_joints = len(self.joint_names)
        self.fixed_tendon_names: list[str] = []
        self.num_fixed_tendons = 0
        self.data = _MockArticulationData(num_envs, device)

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
    def __init__(self, entities: dict, num_envs: int = 8, device: str = "cpu"):
        self._entities = entities
        # Zero env_origins so the real ``index_body_env_local_pos`` post-processing
        # is a no-op against mocked body-position buffers.
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        # Mirror the real ``InteractiveScene`` API: ``scene.sensors[name]`` resolves
        # contact sensors. The mock articulation doubles as both since tests only
        # exercise the resolve path (not real sensor data reads).
        self.sensors = entities
        # Refresh each articulation's ``data.root_quat_w`` to match this scene's
        # num_envs — call sites pass ``num_envs`` to the scene but not to the
        # articulation (which defaults to 8). Identity quats, so rotation is a
        # no-op; this just keeps tensor shapes consistent.
        for ent in entities.values():
            if hasattr(ent, "data"):
                ent.data = _MockArticulationData(num_envs, device)

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
        self.sim = _MockSimulation()


class _MockVisualizationRegistry:
    """Minimal visualization callback registry required by :class:`CommandTerm`."""

    def add_debug_vis_callback(self, _term):
        return object()

    def clear_debug_vis_callback(self, _term) -> None:
        pass


class _MockSimulation:
    """Minimal simulation surface required by :class:`CommandTerm`."""

    def __init__(self):
        self.vis_marker_registry = _MockVisualizationRegistry()


def _make_env(num_envs: int = 4, device: str = "cpu", max_episode_length: int = 10) -> _MockEnv:
    robot = _MockArticulation(body_names=["base"])
    scene = _MockScene({"robot": robot}, num_envs=num_envs, device=device)
    return _MockEnv(num_envs=num_envs, device=device, max_episode_length=max_episode_length, scene=scene)


def _make_lin_vel_cfg(min_xy: float = -1.0, max_xy: float = 1.0) -> MultiTaskCfg:
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        # ``quality_easing = 1.0`` reproduces the legacy "reward = mean
        # tracking activation" semantics that these unit tests assert. The
        # production default is 0.5 (sqrt softening); set explicitly here so
        # the closed-form expected values in each test stay valid regardless
        # of the production default.
        quality_easing=1.0,
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
        # See ``_make_lin_vel_cfg`` — pin to legacy mean semantics for
        # closed-form test arithmetic.
        quality_easing=1.0,
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
    """Holds the per-step synthetic tensor keyed by ``STATE_KERNEL_ID``.

    The real prepare-phase reader returns a full per-asset buffer (e.g.
    ``body_pos_w`` shape ``[N, num_bodies, 3]``). The mock maps each set kernel
    to its registered :class:`BUFFER_KIND` and returns the synthetic tensor as
    the pre-prepared raw buffer, so the downstream ``index_fn`` + ``compute_fn``
    run against it exactly as they would in production.
    """

    def __init__(self, device: str):
        self.device = device
        self.outputs: dict[int, torch.Tensor] = {}

    def set(self, kernel_id: STATE_KERNEL_ID, tensor: torch.Tensor) -> None:
        self.outputs[int(kernel_id)] = tensor

    def make_readers(self) -> tuple:
        """Replacement for ``BUFFER_KIND_READERS``.

        Resolves each set state kernel to its buffer kind at mock-build time so
        a single synthetic tensor per kernel answers the prepared read, then the
        real indexer + compute handle slicing and math.
        """
        from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.impl.kernels_torch import (
            STATE_KERNEL_BUFFER_KIND,
        )
        from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.kernel_ids import BUFFER_KIND

        # Fold per-kernel outputs into per-buffer outputs (same-buffer kernels
        # collapse to whichever kid was set — tests never collide on a buffer
        # shared between two active kernels).
        buffer_outputs: dict[int, torch.Tensor] = {}
        for kid, tensor in self.outputs.items():
            bk = int(STATE_KERNEL_BUFFER_KIND[kid])
            buffer_outputs[bk] = tensor

        def make_fn(buffer_kind: int):
            def fn(env, asset_name):
                out = buffer_outputs.get(buffer_kind)
                if out is None:
                    return torch.zeros(env.num_envs, 1, 3, device=env.device)
                if out.ndim == 2:
                    return out.unsqueeze(1)
                return out

            return fn

        return tuple(make_fn(bk) for bk in range(len(BUFFER_KIND)))


# -----------------------------------------------------------------------------
# Lifecycle tests
# -----------------------------------------------------------------------------


def test_spec_builds_for_pure_tracking():
    """Instantiate MultiTaskCommand with pure tracking cfg; inspect the spec."""
    env = _make_env(num_envs=4)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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

    For pure tracking with the policy holding perfect velocity, the terminal reward
    should be ≈1 at the anticipated-timeout steps and 0 before. The composer's
    ``is_timeout`` fires at ``episode_length_buf >= max_episode_length - 1`` (one
    step early) so the reward manager — which reads one step before
    ``_update_command`` runs — sees the latched terminal value at the outer
    timeout step. That means both steps ``T-2`` and ``T-1`` carry the terminal
    reward; only ``0..T-3`` are non-terminal.
    """
    env = _make_env(num_envs=2, max_episode_length=5)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Set synthetic current BODY_LIN_VEL equal to the sampled target so activation = 1.
    # Under the ragged layout the target lives in ``_targets_flat[env, :stride]``.
    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()

    rewards: list[torch.Tensor] = []
    for step in range(env.max_episode_length):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        state.set(STATE_KERNEL_ID.BODY_LIN_VEL, target.clone())
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
            cmd._update_command()
        rewards.append(cmd.task_reward.clone())

    # Steps 0..T-3: pre-anticipation window — reward EXACTLY zero.
    for i in range(env.max_episode_length - 2):
        assert (rewards[i] == 0.0).all(), f"step {i}: reward {rewards[i]} != 0 exactly"
    # Steps T-2 and T-1: ``is_timeout`` fires (``buf >= max - 1``). state==target →
    # activation = 1 → transit mean = 1 → terminal reward = 1.
    for i in [env.max_episode_length - 2, env.max_episode_length - 1]:
        assert torch.allclose(rewards[i], torch.ones_like(rewards[i]), atol=1e-6), (
            f"step {i}: expected terminal reward 1.0 (perfect tracking), got {rewards[i]}"
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step >= k:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))  # at origin → achieved
        # velocity stays perfect the whole time
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Manually bump composer state for all envs, simulating several accumulated steps.
    cmd._sum_activation.fill_(3.5)
    cmd._transit_steps.fill_(10)
    cmd._instant_achieved.fill_(True)

    # Resample only envs 1 and 3.
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    targets_before = cmd._targets_flat.clone()
    subtask_ids_before = cmd._env_subtask_ids.clone()
    slot_count_before = cmd._env_slot_count.clone()

    empty = torch.tensor([], device=env.device, dtype=torch.long)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    assert torch.allclose(cmd.task_reward, torch.zeros(3))
    assert not cmd.task_done.any()


def test_command_tensor_has_canonical_shape():
    """``command_reach`` / ``command_track`` are split by subtask type.

    The lin_vel cfg has one tracking subtask → track tensor gets 3 channels,
    reach tensor is empty (kept as stride-1 placeholder). Legacy ``command``
    is the concatenation.
    """
    env = _make_env(num_envs=5, max_episode_length=10)
    cfg = _make_lin_vel_cfg()

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    assert cmd.spec.reach_canonical_width == 0
    assert cmd.spec.track_canonical_width == 3
    assert cmd.command_reach.shape == (5, 1)  # stride-1 placeholder for empty layout
    assert cmd.command_track.shape == (5, 3)


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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    scene = _MockScene({"robot": robot}, num_envs=1)
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    assert cmd.spec.state_stride.tolist() == [3, 4, 12]
    assert cmd.spec.task_slot_offsets[0, :3].tolist() == [0, 3, 7]
    assert cmd.spec.task_total_stride[0].item() == 19
    assert cmd.max_task_total_stride == 19


def test_read_groups_fuse_same_asset_same_kernel_subtasks():
    """Subtasks sharing ``(state_kid, asset.name)`` coalesce into a single read group.

    Four single-body subtasks on the same asset with the same state kernel should
    produce exactly one read group — so the runtime hits the state kernel once per
    step instead of four times. Regression guard against someone reverting the
    fusion to per-entity dispatch.
    """
    robot = _MockArticulation(body_names=["b0", "b1", "b2", "b3", "b4"])
    scene = _MockScene({"robot": robot}, num_envs=1)
    env = _MockEnv(num_envs=1, device="cpu", max_episode_length=5, scene=scene)

    def _pos_subtask(body_name: str) -> MultiTaskCfg.InstantaneousTaskCfg:
        return MultiTaskCfg.InstantaneousTaskCfg(
            asset_cfg=SceneEntityCfg("robot", body_names=body_name),
            state_kernel=int(STATE_KERNEL_ID.BODY_POS),
            metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
            activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
            activation_kernel_param=0.5,
            sampler=MinMaxSampler(
                kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                minimum=[0.0, 0.0, 0.0],
                maximum=[0.0, 0.0, 0.0],
            ),
        )

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={"multi_body_pos": [_pos_subtask(b) for b in ("b1", "b2", "b3", "b4")]},
    )
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # All four subtasks collapse into a single read group.
    assert cmd.num_subtasks == 4
    assert set(cmd.spec.read_group_id.tolist()) == {0}
    assert len(cmd.spec.read_group_member_sids) == 1
    assert len(cmd.spec.read_group_member_sids[0]) == 4
    assert len(cmd.spec.read_group_member_asset_cfgs[0]) == 4
    # Member indices span 0..3, in first-seen order.
    assert sorted(cmd.spec.subtask_member_index.tolist()) == [0, 1, 2, 3]


def test_read_groups_split_across_different_kernels():
    """Same asset, different state kernels → distinct read groups.

    Ensures we don't accidentally merge a POS subtask with a QUAT subtask just
    because they share an asset — the kernel output shapes differ.
    """
    env = _make_env(num_envs=1, max_episode_length=5)
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Two distinct read groups — one per kernel — same asset.
    assert set(cmd.spec.read_group_id.tolist()) == {0, 1}
    assert len(cmd.spec.read_group_member_sids) == 2


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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        with pytest.raises(ValueError, match="state_stride inconsistency"):
            MultiTaskCommand(cfg, env)


def test_state_kernel_stride_mismatch_raises_at_runtime():
    """Runtime guard: state kernel's output dim must match spec's ``state_stride``.

    The canonical-layout build-time check catches sampler-vs-kernel mismatches where
    the sampler declares a bogus stride. The runtime check covers the complementary
    case where the sampler + canonical agree but the state kernel itself (at run
    time) emits a different dim — e.g. a misconfigured asset_cfg picking 5 bodies
    instead of 1, or a kernel implementation bug.
    """
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "bad": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],  # stride 3 — matches BODY_LIN_VEL
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        },
    )
    env = _make_env(num_envs=1)
    state = _SyntheticState(device=env.device)
    # Spec build passes (stride 3). At runtime, monkeypatch the reader to return
    # a bogus raw buffer with 5 bodies — the subsequent compute flattens to
    # stride 15, which must be caught by the dispatch's stride check.
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.zeros(1, 5, 3))  # reader lies: 5 bodies instead of 1
    env.episode_length_buf = torch.ones(1, dtype=torch.long)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step == T - 1:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
        quality_easing=1.0,  # legacy mean semantics for closed-form arithmetic
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.arange(3, device=env.device, dtype=torch.long))

    # Observed velocity is the SAME for every env: [1, 0, 0].
    # Expected errors per env: env0 tracks target [1,0,0] → err=0. env1 target [2,0,0] → err=1.
    # env2 target [3,0,0] → err=2.
    current = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, current)

    for step in range(T):
        env.episode_length_buf = torch.full((3,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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


def test_canonical_delta_writes_into_entity_block_pos_slice():
    """Instant subtask writes to the reach tensor only; track tensor stays empty.

    Single entity (base), one instant BODY_POS subtask →
    ``reach_canonical_width == 3`` and ``track_canonical_width == 0``.
    Target at origin, state at ``[2, 0, 0]`` → delta = ``[-2, 0, 0]`` in the
    reach tensor, track tensor is all zero.
    """
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "pos_only": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],  # target always at origin
                    ),
                ),
            ],
        },
    )
    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[2.0, 0.0, 0.0]]))
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Instant subtask → reach tensor gets all 3 channels, track is empty.
    assert cmd.spec.reach_canonical_width == 3
    assert cmd.spec.track_canonical_width == 0
    assert cmd.spec.canonical_offset.tolist() == [0]
    assert cmd.spec.canonical_stride.tolist() == [3]

    env.episode_length_buf = torch.ones(1, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._update_command()

    assert torch.allclose(cmd.command_reach[0], torch.tensor([-2.0, 0.0, 0.0]), atol=1e-6)


def test_canonical_delta_tight_layout_for_base_and_foot():
    """Two instant subtasks on different entities → both write to the reach tensor.

    Base uses BODY_POS (stride 3). Foot uses BODY_POS_Z (stride 1). Both
    instant → reach tensor width = 3 + 1 = 4. Track tensor is empty. Base
    block in reach occupies ``[0:3]``, foot block ``[3:4]``.
    """
    robot = _MockArticulation(body_names=["base", "foot"])
    scene = _MockScene({"robot": robot}, num_envs=1)
    env = _MockEnv(num_envs=1, device="cpu", max_episode_length=10, scene=scene)

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "base_pos_and_foot_z": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[1.0, 2.0, 3.0],
                        maximum=[1.0, 2.0, 3.0],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="foot"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.2],
                        maximum=[0.2],
                    ),
                ),
            ],
        },
    )

    state = _SyntheticState(device=env.device)
    # BODY_POS and BODY_POS_Z share the BODY_POS_W buffer — one coherent
    # per-asset tensor of shape [N=1, num_bodies=2, 3] covers both subtasks.
    # Base (body 0) current = [0, 0, 0]; foot (body 1) current = [0, 0, 0.05].
    # Targets: base → [1, 2, 3] (delta = [1, 2, 3]); foot z → 0.2 (delta = 0.15).
    state.set(
        STATE_KERNEL_ID.BODY_POS,
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]]),
    )

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Both instant → reach tensor gets 3 (base POS) + 1 (foot POS_Z) = 4 channels;
    # track tensor is empty.
    assert cmd.spec.reach_canonical_width == 4
    assert cmd.spec.track_canonical_width == 0
    assert cmd.spec.canonical_offset.tolist() == [0, 3]
    assert cmd.spec.canonical_stride.tolist() == [3, 1]

    env.episode_length_buf = torch.ones(1, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._update_command()

    assert torch.allclose(cmd.command_reach[0, 0:3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
    assert abs(cmd.command_reach[0, 3].item() - 0.15) < 1e-6


def test_canonical_delta_pos_and_pos_z_get_disjoint_channels():
    """POS and POS_Z on the same entity + same type get disjoint channels (no z-slot aliasing).

    Block width = 3 (POS) + 1 (POS_Z) = 4. BODY_POS writes xyz into [0:3],
    BODY_POS_Z writes z into channel 3. No overwrite — reach-standing + reach-
    crouch-z can coexist.
    """
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "pos_and_pos_z": [
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
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.5],
                        maximum=[0.5],
                    ),
                ),
            ],
        },
    )
    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Disjoint channels — POS at [0:3], POS_Z at channel 3.
    assert cmd.spec.reach_canonical_width == 4
    assert cmd.spec.track_canonical_width == 0
    offsets = cmd.spec.canonical_offset.tolist()
    strides = cmd.spec.canonical_stride.tolist()
    # Either (POS first, POS_Z second) or vice versa depending on dedup order.
    assert sorted(offsets) == [0, 3]
    assert sorted(strides) == [1, 3]


def test_reach_and_track_deltas_are_routed_to_separate_tensors():
    """Mixed reach + track subtasks on same entity write to disjoint tensors.

    One instant BODY_POS_Z (reach) + one tracking BODY_POS_Z (track), both on
    the same entity. The reach-POS_Z delta lands in ``command_reach``; the
    track-POS_Z delta lands in ``command_track``. Neither overwrites the other
    — that's the whole point of the split.
    """
    env = _make_env(num_envs=1, max_episode_length=5)
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "reach_and_track_z": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.7],  # reach z = 0.7
                        maximum=[0.7],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.2],  # crouch z = 0.2
                        maximum=[0.2],
                    ),
                ),
            ],
        },
    )
    state = _SyntheticState(device=env.device)
    # Current base pos so that POS_Z source returns z = 0.4 for both subtasks.
    state.set(STATE_KERNEL_ID.BODY_POS_Z, torch.tensor([[0.0, 0.0, 0.4]]))
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Reach tensor gets 1 channel (the instant POS_Z); track tensor gets 1 channel
    # (the tracking POS_Z). No aliasing.
    assert cmd.spec.reach_canonical_width == 1
    assert cmd.spec.track_canonical_width == 1

    env.episode_length_buf = torch.ones(1, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._update_command()

    # Reach delta = 0.7 - 0.4 = +0.3, track delta = 0.2 - 0.4 = -0.2. Both visible.
    assert abs(cmd.command_reach[0, 0].item() - 0.3) < 1e-6
    assert abs(cmd.command_track[0, 0].item() - (-0.2)) < 1e-6


def test_progress_is_mean_of_active_slot_activations():
    """``progress`` equals ``mean(activation)`` over active slots.

    Single-env mixed task (2 slots, both valid). With one slot's activation = 1.0 and
    the other's = 0.0, progress must be exactly 0.5 — regression guard against
    slipping from "mean over active" to "sum" or "min".
    """
    env = _make_env(num_envs=1, max_episode_length=5)
    cfg = _make_mixed_cfg()  # [instant body-pos, tracking lin-vel], k_max=2

    state = _SyntheticState(device=env.device)
    # Instant body-pos: target is origin, state at origin → error=0 → activation=1 (LESS kernel).
    # Tracking lin-vel: target [1,0,0], state [1e9, 0, 0] → error huge → activation ≈ 0.
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1e9, 0.0, 0.0]]))

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    env.episode_length_buf = torch.ones(1, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._update_command()

    assert cmd.progress.shape == (1,)
    # Activation 0 (instant) ≈ 1.0, activation 1 (tracking) ≈ 0 → mean ≈ 0.5.
    assert abs(cmd.progress[0].item() - 0.5) < 1e-4


def test_progress_ignores_padded_slots():
    """Envs whose task has fewer slots than ``k_max`` average only over their active slots.

    Mixed-width tasks: "single" (1 subtask, always activation=1) vs "double" (2 subtasks,
    both activation=1). Both envs should report progress = 1.0 — if progress divided by
    ``k_max`` instead of ``slot_count``, the single-task env would land at 0.5.
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
                    minimum=[0.0, 0.0, 0.0],
                    maximum=[0.0, 0.0, 0.0],
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
                    minimum=[0.0, 0.0, 0.0],
                    maximum=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    }
    cfg = MultiTaskCfg(resampling_time_range=(100.0, 100.0), debug_vis=False, tasks=tasks)
    env = _make_env(num_envs=2)

    state = _SyntheticState(device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Env 0 → "single", env 1 → "double"; both expected progress = 1.0.
    single_id = cmd.spec.task_names.index("single")
    double_id = cmd.spec.task_names.index("double")
    cmd.task_samples[0] = single_id
    cmd.task_samples[1] = double_id
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    # Perfect tracking everywhere → activation = 1 on every active slot.
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.zeros(2, 3))
    state.set(STATE_KERNEL_ID.BODY_ANG_VEL, torch.zeros(2, 3))
    env.episode_length_buf = torch.ones(2, dtype=torch.long, device=env.device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._update_command()

    # Both envs average to 1.0 — padded slot does NOT dilute the single-task env.
    assert abs(cmd.progress[0].item() - 1.0) < 1e-6
    assert abs(cmd.progress[1].item() - 1.0) < 1e-6


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

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Walk the trajectory. Record reward/done at each step until success fires.
    reward_trace: list[float] = []
    done_trace: list[bool] = []
    k_success = 4

    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        if step >= k_success:
            state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()

    per_step_activations: list[float] = []
    for step in range(T):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        current = target.clone() if step % 2 == 0 else target + 10.0
        state.set(STATE_KERNEL_ID.BODY_LIN_VEL, current)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
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
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    assert cmd._env_slot_count[0].item() == 1
    assert cmd._env_slot_count[1].item() == 2


def test_command_active_mask_disambiguates_cross_task_channels():
    """The per-channel active mask flags "live vs inactive" per env, not per channel.

    Two envs, two tasks, one shared channel and one unique channel:

    - Task A = ``only_pos``: single instant BODY_POS on base → populates the
      reach-layout pos channel (3 floats).
    - Task B = ``pos_plus_quat``: instant BODY_POS on base + instant BODY_QUAT
      on base → populates both pos AND quat channels.

    Env 0 gets task A, env 1 gets task B. After resample, the pos channels
    must be ``1.0`` for BOTH envs (shared), but the quat channels must be
    ``1.0`` only for env 1 (the task-B env). This catches regressions where
    the mask either (a) isn't refreshed on reset, (b) fails to clear prior
    values, or (c) applies task B's channels to env 0.
    """
    env = _make_env(num_envs=2)
    base = SceneEntityCfg("robot", body_names="base")
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "only_pos": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
            "pos_plus_quat": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
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
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_QUAT, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Pin env 0 → task A, env 1 → task B and force a refresh.
    task_a = cmd.spec.task_names.index("only_pos")
    task_b = cmd.spec.task_names.index("pos_plus_quat")
    cmd.task_samples[0] = task_a
    cmd.task_samples[1] = task_b
    cmd.resample_indices = lambda env_ids: None  # pin the manual samples
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    # Layout: reach_canonical_width = 3 (pos) + 4 (quat) = 7; track_w = 0.
    assert cmd.spec.reach_canonical_width == 7
    assert cmd.spec.track_canonical_width == 0

    mask = cmd.command_active  # [2, 7]
    assert mask.shape == (2, 7)

    # Pos channels [0, 3) should be 1.0 for BOTH envs (shared across tasks).
    assert torch.allclose(mask[:, 0:3], torch.ones((2, 3), device=env.device))
    # Quat channels [3, 7) should be 1.0 only for env 1, 0.0 for env 0.
    assert torch.allclose(mask[0, 3:7], torch.zeros(4, device=env.device))
    assert torch.allclose(mask[1, 3:7], torch.ones(4, device=env.device))


def test_command_active_clears_on_task_switch():
    """Stale-channel regression: switching an env's task from B→A must clear
    the channels that were live under B but are inactive under A.

    If ``_command_active[env_ids] = spec.task_active_mask[task_idx]`` were a
    bitwise OR instead of an assignment, this test would fail.
    """
    env = _make_env(num_envs=1)
    base = SceneEntityCfg("robot", body_names="base")
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "big": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                        out_dim=4,
                    ),
                ),
            ],
            "small": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        },
    )

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_POS, torch.tensor([[0.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.BODY_QUAT, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    big = cmd.spec.task_names.index("big")
    small = cmd.spec.task_names.index("small")

    # Start on "big" — quat channels live.
    cmd.task_samples[0] = big
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0], device=env.device, dtype=torch.long))
    assert torch.allclose(cmd.command_active[0, 3:7], torch.ones(4, device=env.device))

    # Switch to "small" — quat channels must go back to 0.
    cmd.task_samples[0] = small
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0], device=env.device, dtype=torch.long))
    assert torch.allclose(cmd.command_active[0, 0:3], torch.ones(3, device=env.device))
    assert torch.allclose(cmd.command_active[0, 3:7], torch.zeros(4, device=env.device))


def test_command_active_mask_ignores_joint_kernel_subtasks():
    """Joint-kernel subtasks (``canonical_offset = -1``) contribute zero 1.0's to the mask.

    Joint kernels have no canonical projection — they read/write joint-indexed
    state rather than a canonical per-body block, so the reach/track obs layouts
    allocate no channels for them. The spec builder guards this with
    ``if canon_off < 0: continue``. If someone later "fixes" the guard with
    ``clamp_min(0)``, channel 0 of every joint-kernel task's mask lights up
    spuriously. This test catches that regression.

    Layout with a joint-only task A and a body-pos task B:
      - reach_canonical_width = 3 (pos channels, from task B only)
      - track_canonical_width = 0
      - mask width = 3
      - Env 0 (task A, joint-only) → mask row all 0.0
      - Env 1 (task B, body-pos)   → mask row all 1.0
    """
    robot = _MockArticulation(body_names=["base"], joint_names=[f"j{i}" for i in range(6)])
    scene = _MockScene({"robot": robot}, num_envs=2)
    env = _MockEnv(num_envs=2, device="cpu", max_episode_length=10, scene=scene)

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "joint_only": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot"),  # all 6 joints
                    state_kernel=int(STATE_KERNEL_ID.JOINT_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0] * 6,
                        maximum=[0.0] * 6,
                    ),
                ),
            ],
            "body_pos_only": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=SceneEntityCfg("robot", body_names="base"),
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.1,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        },
    )

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.JOINT_POS, torch.zeros(2, 6))
    state.set(STATE_KERNEL_ID.BODY_POS, torch.zeros(2, 3))

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Layout: joint kernel contributes no canonical channels; only body-pos does.
    assert cmd.spec.reach_canonical_width == 3
    assert cmd.spec.track_canonical_width == 0

    # The joint-kernel subtask must have canonical_offset = -1 (no projection).
    joint_task = cmd.spec.task_names.index("joint_only")
    joint_sid = int(cmd.spec.task_subtask_ids[joint_task, 0].item())
    assert int(cmd.spec.canonical_offset[joint_sid].item()) == -1

    # Pin env 0 → joint-only, env 1 → body-pos.
    cmd.task_samples[0] = joint_task
    cmd.task_samples[1] = cmd.spec.task_names.index("body_pos_only")
    cmd.resample_indices = lambda env_ids: None
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd._resample_command(torch.tensor([0, 1], device=env.device, dtype=torch.long))

    mask = cmd.command_active  # [2, 3]
    assert mask.shape == (2, 3)
    # Joint-only env: no canonical channels should be flagged active.
    assert torch.allclose(mask[0], torch.zeros(3, device=env.device))
    # Body-pos env: pos channels all active.
    assert torch.allclose(mask[1], torch.ones(3, device=env.device))


def test_safety_subtask_discounts_terminal_reward_multiplicatively():
    """End-to-end: a TrackingTaskCfg(expose_in_obs=False) attached to a tracking task discounts G.

    Construct a task with one tracking subtask (BODY_LIN_VEL, target=fixed) and
    one safety subtask whose violation is held constant by the synthetic state.
    Drive a full episode; assert that:

      reward_terminal = tracking_mean × safety_mean

    where each is the mean over the transit window of activation values.
    """
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        quality_easing=1.0,  # bare product so terminal == tracking_mean × safety_mean
        tasks={
            "track_with_safety": [
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
                # Safety: minimize "joint position" (proxy for "violation").
                # Soft-safety = TrackingTaskCfg with expose_in_obs=False; with
                # synthetic state held at a known value and target=0, error is
                # constant; activation = 1 - tanh(error/scale) is constant.
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=SceneEntityCfg("robot"),  # all joints
                    state_kernel=int(STATE_KERNEL_ID.JOINT_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=1.0,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0],
                        maximum=[0.0],
                    ),
                    expose_in_obs=False,
                ),
            ],
        },
    )

    # Need joints for JOINT_POS kernel; rebuild env with a multi-joint robot.
    robot = _MockArticulation(body_names=["base"], joint_names=[f"j{i}" for i in range(1)])
    scene = _MockScene({"robot": robot}, num_envs=1)
    env = _MockEnv(num_envs=1, device="cpu", max_episode_length=10, scene=scene)

    # Per-step state: lin_vel exactly at target → tracking activation = 1.
    # Joint pos at value v (target 0) → safety violation = v, activation = 1 - tanh(v).
    v = 0.5  # safety violation magnitude
    expected_safety_act = 1.0 - math.tanh(v / 1.0)

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.JOINT_POS, torch.tensor([[v]]))

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    rewards: list[float] = []
    for step in range(env.max_episode_length):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
            cmd._update_command()
        rewards.append(float(cmd.task_reward.item()))

    # Pre-anticipation steps (0..T-3) should be 0; T-2 and T-1 latch terminal.
    for step in range(env.max_episode_length - 2):
        assert rewards[step] == 0.0, f"step {step}: pre-terminal reward {rewards[step]} ≠ 0"

    # Terminal value = tracking_mean (=1.0) · safety_mean (=expected_safety_act).
    expected_terminal = 1.0 * expected_safety_act
    for step in (env.max_episode_length - 2, env.max_episode_length - 1):
        assert abs(rewards[step] - expected_terminal) < 1e-5, (
            f"step {step}: terminal reward {rewards[step]:.5f} ≠ expected {expected_terminal:.5f} "
            f"(tracking · safety = 1.0 · {expected_safety_act:.5f})"
        )


def test_safety_subtask_with_perfect_safety_recovers_pure_tracking_reward():
    """Sanity: violation = 0 → safety_factor = 1 → terminal = tracking_mean (no discount)."""
    robot = _MockArticulation(body_names=["base"], joint_names=["j0"])
    scene = _MockScene({"robot": robot}, num_envs=1)
    env = _MockEnv(num_envs=1, device="cpu", max_episode_length=10, scene=scene)

    cfg = MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        tasks={
            "track_safe": [
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
                    asset_cfg=SceneEntityCfg("robot"),
                    state_kernel=int(STATE_KERNEL_ID.JOINT_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=1.0,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0],
                        maximum=[0.0],
                    ),
                    expose_in_obs=False,
                ),
            ],
        },
    )

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))
    state.set(STATE_KERNEL_ID.JOINT_POS, torch.tensor([[0.0]]))  # zero violation

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    for step in range(env.max_episode_length):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
            cmd._update_command()

    # Terminal latched at step T-2, persists at T-1. Tracking exact = 1, safety
    # exact = 1 - tanh(0) = 1. Product = 1.
    assert abs(cmd.task_reward.item() - 1.0) < 1e-5, (
        f"perfect-safety pure-tracking should yield reward 1.0, got {cmd.task_reward.item():.5f}"
    )


def test_safety_subtask_no_safety_in_cfg_yields_unchanged_terminal():
    """A cfg without any expose_in_obs=False subtask → quality_factor reflects only
    ordinary tracking → identical to the legacy non-safety path."""
    env = _make_env(num_envs=1, max_episode_length=10)
    cfg = _make_lin_vel_cfg()  # tracking-only, no safety

    state = _SyntheticState(device=env.device)
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, torch.tensor([[1.0, 0.0, 0.0]]))
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
        cmd = MultiTaskCommand(cfg, env)

    # Set vel exactly at the (post-resample) target so activation = 1 every step.
    stride = int(cmd._env_slot_strides[0, 0].item())
    target = cmd._targets_flat[:, :stride].clone()
    state.set(STATE_KERNEL_ID.BODY_LIN_VEL, target.clone())

    for step in range(env.max_episode_length):
        env.episode_length_buf = torch.full((env.num_envs,), step + 1, dtype=torch.long, device=env.device)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", state.make_readers()):
            cmd._update_command()

    assert torch.allclose(cmd.task_reward, torch.ones_like(cmd.task_reward), atol=1e-6), (
        "no-safety cfg must produce vacuous safety_factor=1; terminal = tracking_mean = 1.0"
    )


if __name__ == "__main__":
    # For quick local iteration.
    pytest.main([__file__, "-v"])
