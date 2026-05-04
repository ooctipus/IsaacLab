# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Byte-identity check: Warp mega-kernel vs PyTorch reference.

The Warp path (``cfg.use_warp_dispatch=True``) collapses ~440 per-step kernel
launches into a single ``wp.launch``. This test runs the production cfg under
both paths with the same synthetic state and asserts elementwise equality of
the four output tensors (``_buf_error``, ``_buf_activation``, ``_command_reach``,
``_command_track``).

Skipped when CUDA is unavailable (Warp kernel only compiles for CUDA here).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.multi_task.mdp.commands import multi_task_command as mtc_mod
from isaaclab_tasks.manager_based.multi_task.mdp.commands.kernels_torch import (
    ACTIVATION_KERNEL_ID,
    BUFFER_KIND,
    METRIC_KERNEL_ID,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
)
from isaaclab_tasks.manager_based.multi_task.mdp.commands.multi_task_cfg import (
    MinMaxSampler,
    MultiTaskCfg,
)
from isaaclab_tasks.manager_based.multi_task.mdp.commands.multi_task_command import MultiTaskCommand

_NEED_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp mega-kernel requires CUDA")

# ---------------------------------------------------------------------------
# Mocks — same shapes as test_multi_task_command_mock.py.
# ---------------------------------------------------------------------------

_ANYMAL_BODY_NAMES = [
    "base",
    "LF_HIP",
    "RF_HIP",
    "LH_HIP",
    "RH_HIP",
    "LF_THIGH",
    "RF_THIGH",
    "LH_THIGH",
    "RH_THIGH",
    "LF_SHANK",
    "RF_SHANK",
    "LH_SHANK",
    "RH_SHANK",
    "LF_FOOT",
    "RF_FOOT",
    "LH_FOOT",
    "RH_FOOT",
]


class _MockArticulation:
    def __init__(self, body_names):
        self.body_names = list(body_names)
        self.joint_names: list[str] = []
        self.num_bodies = len(self.body_names)
        self.num_joints = 0
        self.fixed_tendon_names: list[str] = []
        self.num_fixed_tendons = 0

    @staticmethod
    def _find(names, patterns, preserve_order=False):
        if isinstance(patterns, str):
            patterns = [patterns]
        ids, matched = [], []
        for pat in patterns:
            rx = re.compile(pat)
            for i, n in enumerate(names):
                if rx.fullmatch(n) and i not in ids:
                    ids.append(i)
                    matched.append(n)
        return ids, matched

    def find_bodies(self, patterns, preserve_order=False):
        return self._find(self.body_names, patterns, preserve_order)

    def find_joints(self, patterns, preserve_order=False):
        return [], []

    def find_fixed_tendons(self, patterns, preserve_order=False):
        return [], []


class _MockArticulationData:
    """Identity ``root_quat_w`` for the in-dispatch rotation helper."""

    def __init__(self, num_envs, device):
        q = torch.zeros(num_envs, 4, device=device)
        q[:, 3] = 1.0
        self.root_quat_w = q  # torch; helper's _as_torch passes it through


class _MockScene:
    def __init__(self, entities, num_envs, device):
        self._entities = entities
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        # Attach identity ``data.root_quat_w`` to each mocked articulation so
        # the dispatch rotation helper is a no-op (equivalence with Warp is
        # preserved — both paths rotate by identity).
        for ent in entities.values():
            ent.data = _MockArticulationData(num_envs, device)
        self.sensors = entities

    def keys(self):
        return self._entities.keys()

    def __getitem__(self, name):
        return self._entities[name]

    def __contains__(self, name):
        return name in self._entities


class _MockEnv:
    def __init__(self, num_envs, device, max_episode_length, scene):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.scene = scene
        self.common_step_counter = 0
        self.step_dt = 0.02


def _make_env(num_envs: int, device: str):
    robot = _MockArticulation(_ANYMAL_BODY_NAMES)
    contact_forces = _MockArticulation(_ANYMAL_BODY_NAMES)
    scene = _MockScene({"robot": robot, "contact_forces": contact_forces}, num_envs=num_envs, device=device)
    return _MockEnv(num_envs=num_envs, device=device, max_episode_length=200, scene=scene)


def _make_readers(num_envs: int, device: str, seed: int) -> tuple:
    """Fixed-seed random synthetic per-kind source tensors."""
    g = torch.Generator(device=device).manual_seed(seed)
    nb = len(_ANYMAL_BODY_NAMES)

    body_pos = torch.randn((num_envs, nb, 3), generator=g, device=device)
    # Unit quats so metric_quaternion is well-defined.
    quat = torch.randn((num_envs, nb, 4), generator=g, device=device)
    body_quat = torch.nn.functional.normalize(quat, dim=-1)
    body_lin = torch.randn((num_envs, nb, 3), generator=g, device=device)
    body_ang = torch.randn((num_envs, nb, 3), generator=g, device=device)
    # Contact forces: mix of above and below the 1 N² threshold so the
    # count kernels see both branches.
    contact = torch.randn((num_envs, nb, 3), generator=g, device=device) * 0.8

    by_kind = {
        int(BUFFER_KIND.JOINT_POS): torch.zeros((num_envs, 1), device=device),
        int(BUFFER_KIND.JOINT_VEL): torch.zeros((num_envs, 1), device=device),
        int(BUFFER_KIND.BODY_POS_W): body_pos,
        int(BUFFER_KIND.BODY_QUAT_W): body_quat,
        int(BUFFER_KIND.BODY_LIN_VEL_W): body_lin,
        int(BUFFER_KIND.BODY_ANG_VEL_W): body_ang,
        int(BUFFER_KIND.CONTACT_NET_FORCES_W): contact,
    }

    def make_reader(kind: int):
        def reader(env, asset_name):
            return by_kind[kind]

        return reader

    return tuple(make_reader(int(k)) for k in BUFFER_KIND)


# ---------------------------------------------------------------------------
# CFG — touches every supported Warp kernel branch.
# ---------------------------------------------------------------------------


def _mixed_cfg(use_warp: bool) -> MultiTaskCfg:
    """Mini cfg that exercises every Warp branch we support.

    Tasks:
      - ``vel``: tracking LIN_VEL + ANG_VEL (geometric / tanh, stride 3).
      - ``pose``: instant BODY_POS + BODY_QUAT (pos/quat / tanh, stride 3/4).
      - ``tripod``: instant BODY_POS + instant CONTACT_COUNT (geom/less, K=4).
      - ``gait``: tracking CONTACT_COUNT_DIFF (geom/greater, K=4).
      - ``safety``: tracking ANG_VEL with GAUSSIAN activation (target=0,
        ``expose_in_obs=False``) — exercises the GAUSSIAN activation branch
        in both PyTorch and Warp paths.
    """
    base = SceneEntityCfg("robot", body_names="base")
    feet = SceneEntityCfg("contact_forces", body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        use_warp_dispatch=use_warp,
        tasks={
            "vel": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[-1.0, -1.0, 0.0],
                        maximum=[1.0, 1.0, 0.0],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, -0.5],
                        maximum=[0.0, 0.0, 0.5],
                    ),
                ),
            ],
            "pose": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.4,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[-1.0, -1.0, 0.4],
                        maximum=[1.0, 1.0, 0.6],
                    ),
                ),
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                        minimum=[-0.2, -0.2, -0.3],
                        maximum=[0.2, 0.2, 0.3],
                        out_dim=4,
                    ),
                ),
            ],
            "tripod": [
                MultiTaskCfg.InstantaneousTaskCfg(
                    asset_cfg=feet,
                    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT_COUNT),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[3.0],
                        maximum=[3.0],
                    ),
                ),
            ],
            "gait": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=feet,
                    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.GREATER),
                    activation_kernel_param=1.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0],
                        maximum=[0.0],
                    ),
                ),
            ],
            # Soft-safety task pairing an existing state kernel with the
            # GAUSSIAN activation branch — the math has no other production
            # state kernel today (mech-power needs a torque buffer that this
            # synthetic harness doesn't provide), so reuse ANG_VEL with
            # target=0 so |delta| is just the angular velocity magnitude.
            "safety_gaussian": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=base,
                    state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.GAUSSIAN),
                    activation_kernel_param=2.0,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, 0.0],
                        maximum=[0.0, 0.0, 0.0],
                    ),
                    expose_in_obs=False,
                ),
            ],
            # Per-foot contact kernel — variable K=4 canonical layout. Target
            # is ``[1, 0, 1, 0]`` (one specific per-foot pattern). Exercises
            # the ``_state_body_contact`` branch + the variable-K
            # canonical layout path introduced when the stride-1 hardcode
            # was removed.
            "per_foot": [
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=feet,
                    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT),
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[1.0, 0.0, 1.0, 0.0],
                        maximum=[1.0, 0.0, 1.0, 0.0],
                    ),
                ),
            ],
        },
    )


# ---------------------------------------------------------------------------
# The test — builds two commands side-by-side and compares outputs.
# ---------------------------------------------------------------------------


@_NEED_CUDA
def test_warp_matches_pytorch_outputs():
    device = "cuda:0"
    num_envs = 32
    readers = _make_readers(num_envs, device, seed=1234)

    def build_and_step(use_warp: bool):
        env = _make_env(num_envs=num_envs, device=device)
        with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
            torch.manual_seed(7)  # same task samples both sides
            cmd = MultiTaskCommand(_mixed_cfg(use_warp=use_warp), env)
            # Drive one step — targets are pinned by the seed, so both paths
            # start with the same `_targets_flat` and `_env_subtask_ids`.
            cmd._update_command()
        return cmd

    cmd_py = build_and_step(use_warp=False)
    cmd_wp = build_and_step(use_warp=True)

    # Sanity: both paths sampled the same tasks / targets.
    assert torch.equal(cmd_py._env_subtask_ids, cmd_wp._env_subtask_ids)
    assert torch.equal(cmd_py._env_slot_count, cmd_wp._env_slot_count)
    assert torch.allclose(cmd_py._targets_flat, cmd_wp._targets_flat)

    atol = 1e-5
    rtol = 1e-5
    assert torch.allclose(cmd_py._buf_error, cmd_wp._buf_error, atol=atol, rtol=rtol), (
        f"error mismatch: max |Δ| = {(cmd_py._buf_error - cmd_wp._buf_error).abs().max().item():.3e}"
    )
    assert torch.allclose(cmd_py._buf_activation, cmd_wp._buf_activation, atol=atol, rtol=rtol), (
        f"activation mismatch: max |Δ| = {(cmd_py._buf_activation - cmd_wp._buf_activation).abs().max().item():.3e}"
    )
    assert torch.allclose(cmd_py._command_reach, cmd_wp._command_reach, atol=atol, rtol=rtol), (
        f"reach mismatch: max |Δ| = {(cmd_py._command_reach - cmd_wp._command_reach).abs().max().item():.3e}"
    )
    assert torch.allclose(cmd_py._command_track, cmd_wp._command_track, atol=atol, rtol=rtol), (
        f"track mismatch: max |Δ| = {(cmd_py._command_track - cmd_wp._command_track).abs().max().item():.3e}"
    )

    # Composer outputs — task reward, success, progress, and in-place state.
    assert torch.allclose(cmd_py._task_reward, cmd_wp._task_reward, atol=atol, rtol=rtol), (
        f"task_reward mismatch: max |Δ| = {(cmd_py._task_reward - cmd_wp._task_reward).abs().max().item():.3e}"
    )
    assert torch.equal(cmd_py._task_done_success, cmd_wp._task_done_success), "task_done_success mismatch"
    assert torch.allclose(cmd_py._progress, cmd_wp._progress, atol=atol, rtol=rtol), (
        f"progress mismatch: max |Δ| = {(cmd_py._progress - cmd_wp._progress).abs().max().item():.3e}"
    )
    assert torch.allclose(cmd_py._sum_activation, cmd_wp._sum_activation, atol=atol, rtol=rtol), (
        f"sum_activation mismatch: max |Δ| = {(cmd_py._sum_activation - cmd_wp._sum_activation).abs().max().item():.3e}"
    )
    assert torch.equal(cmd_py._transit_steps, cmd_wp._transit_steps)
    assert torch.equal(cmd_py._instant_achieved, cmd_wp._instant_achieved)
