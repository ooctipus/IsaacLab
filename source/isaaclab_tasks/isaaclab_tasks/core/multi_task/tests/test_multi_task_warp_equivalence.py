# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Byte-identity check: Warp command backends vs PyTorch reference.

The Warp paths collapse many PyTorch launches into compact Warp backends. This
test runs the production cfg under each selectable backend with the same
synthetic state and asserts elementwise equality of the output tensors.

Scene data is provided through ``ProxyArray``s on a fake articulation —
both the Torch path (``wp.to_torch(art.data.body_pos_w)``) and the Warp
path (``art.data.body_pos_w.warp``) read from the same underlying memory,
so a single seeded data source feeds both byte-identically.

Skipped when CUDA is unavailable (Warp kernel only compiles for CUDA here).
"""

from __future__ import annotations

import re

import pytest
import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import (
    ACTIVATION_KERNEL_ID,
    METRIC_KERNEL_ID,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import (
    MinMaxSampler,
    MultiTaskCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.schedules import (
    SCHEDULE_DIRECT_QUAT_DELTA,
    SCHEDULE_DIRECT_SCALAR_DELTA,
    SCHEDULE_DIRECT_VEC3_DELTA,
    SCHEDULE_SCALAR_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
)
from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command import MultiTaskCommand

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
_ANYMAL_JOINT_NAMES = [
    "LF_HAA",
    "RF_HAA",
    "LH_HAA",
    "RH_HAA",
    "LF_HFE",
    "RF_HFE",
    "LH_HFE",
    "RH_HFE",
    "LF_KFE",
    "RF_KFE",
    "LH_KFE",
    "RH_KFE",
]


class _MockArticulation:
    def __init__(self, body_names, joint_names=None):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names) if joint_names else []
        self.num_bodies = len(self.body_names)
        self.num_joints = len(self.joint_names)
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
        return self._find(self.joint_names, patterns, preserve_order)

    def find_fixed_tendons(self, patterns, preserve_order=False):
        return [], []


def _proxy(torch_tensor: torch.Tensor, dtype) -> ProxyArray:
    """Wrap a torch tensor as a ProxyArray with the given Warp dtype.

    The Warp array aliases the torch tensor's memory; the ProxyArray exposes
    both ``.warp`` (for the Warp backend's typed indexing) and ``.torch`` (for
    the Torch reference path's ``wp.to_torch(art.data.field)`` reader).
    """
    if dtype is wp.float32:
        shape = tuple(torch_tensor.shape)
    elif dtype is wp.vec3:
        # torch shape (..., 3) — Warp folds the trailing 3 into the dtype.
        shape = tuple(torch_tensor.shape[:-1])
    elif dtype is wp.quat:
        shape = tuple(torch_tensor.shape[:-1])
    else:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    return ProxyArray(
        wp.array(
            ptr=torch_tensor.data_ptr(),
            dtype=dtype,
            shape=shape,
            device=str(torch_tensor.device),
        )
    )


class _MockArticulationData:
    """Mock ``Articulation.data`` exposing both ``.warp`` and ``.torch`` accessors.

    Production scene data is exposed via :class:`ProxyArray`; the mock matches
    that interface so both the Torch path (``wp.to_torch``) and the Warp path
    (``.warp``) can read from one seeded source.
    """

    def __init__(self, num_envs: int, num_bodies: int, num_joints: int, device: str, seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        # Identity quat for the rotation helper (kept zero-rotation so the
        # body-frame conversion is a no-op and the byte-identity comparison
        # holds across backends that handle rotation differently). ProxyArray
        # so backends can pull ``.warp`` directly like a real articulation.
        self._root_quat_w_torch = torch.zeros((num_envs, 4), device=device).contiguous()
        self._root_quat_w_torch[:, 3] = 1.0
        self.root_quat_w = _proxy(self._root_quat_w_torch, wp.quat)

        # Per-kind scene data backing the ProxyArray views below. Kept as
        # attributes so the underlying torch tensors stay alive for the
        # lifetime of the wp.array aliases inside each ProxyArray.
        self._body_pos_w_torch = torch.randn((num_envs, num_bodies, 3), generator=g, device=device).contiguous()
        quat_raw = torch.randn((num_envs, num_bodies, 4), generator=g, device=device)
        self._body_quat_w_torch = torch.nn.functional.normalize(quat_raw, dim=-1).contiguous()
        self._body_lin_vel_w_torch = torch.randn((num_envs, num_bodies, 3), generator=g, device=device).contiguous()
        self._body_ang_vel_w_torch = torch.randn((num_envs, num_bodies, 3), generator=g, device=device).contiguous()
        # Joint quantities (zero for joint_pos/joint_vel — only the Warp
        # kernels that gather these need a backing tensor, and the test cfg
        # doesn't exercise non-zero joint state).
        self._joint_pos_torch = torch.zeros((num_envs, num_joints), device=device).contiguous()
        self._joint_vel_torch = torch.zeros((num_envs, num_joints), device=device).contiguous()
        # applied_torque feeds the joint_mech_power kernel; non-zero so the
        # Warp/Torch paths see a non-trivial ``|τ · q̇|`` value (where both
        # are zero here, the product is zero — still byte-identical).
        self._applied_torque_torch = torch.randn((num_envs, num_joints), generator=g, device=device).contiguous()

        # ProxyArray views (both paths read through these).
        self.body_pos_w = _proxy(self._body_pos_w_torch, wp.vec3)
        self.body_quat_w = _proxy(self._body_quat_w_torch, wp.quat)
        self.body_lin_vel_w = _proxy(self._body_lin_vel_w_torch, wp.vec3)
        self.body_ang_vel_w = _proxy(self._body_ang_vel_w_torch, wp.vec3)
        self.joint_pos = _proxy(self._joint_pos_torch, wp.float32)
        self.joint_vel = _proxy(self._joint_vel_torch, wp.float32)
        self.applied_torque = _proxy(self._applied_torque_torch, wp.float32)


class _MockContactSensorData:
    """Mock contact sensor data exposing ``net_forces_w`` as a ProxyArray."""

    def __init__(self, num_envs: int, num_bodies: int, device: str, seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        # Contact forces straddle the 1 N² threshold so the count kernels
        # see both branches.
        self._net_forces_w_torch = (
            torch.randn((num_envs, num_bodies, 3), generator=g, device=device) * 0.8
        ).contiguous()
        self.net_forces_w = _proxy(self._net_forces_w_torch, wp.vec3)


class _MockScene:
    def __init__(self, entities, num_envs, device, seed):
        self._entities = entities
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        # Each articulation gets a seeded data block so all kinds (body_pos_w,
        # body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_pos, joint_vel,
        # applied_torque) have ProxyArray views over deterministic tensors.
        for i, (name, ent) in enumerate(entities.items()):
            if name == "contact_forces":
                ent.data = _MockContactSensorData(num_envs, ent.num_bodies, device, seed=seed + 100 + i)
            else:
                ent.data = _MockArticulationData(num_envs, ent.num_bodies, ent.num_joints, device, seed=seed + i)
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


def _make_env(num_envs: int, device: str, seed: int = 0):
    """Build a mock env whose scene exposes seeded ProxyArray data for every kind."""
    robot = _MockArticulation(_ANYMAL_BODY_NAMES, _ANYMAL_JOINT_NAMES)
    contact_forces = _MockArticulation(_ANYMAL_BODY_NAMES)
    scene = _MockScene(
        {"robot": robot, "contact_forces": contact_forces},
        num_envs=num_envs,
        device=device,
        seed=seed,
    )
    return _MockEnv(num_envs=num_envs, device=device, max_episode_length=200, scene=scene)


# ---------------------------------------------------------------------------
# CFG — touches every supported Warp kernel branch.
# ---------------------------------------------------------------------------


def _mixed_cfg(dispatch_backend: str) -> MultiTaskCfg:
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
        dispatch_backend=dispatch_backend,
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


def _shared_contact_cfg(dispatch_backend: str) -> MultiTaskCfg:
    """Cfg with three contact consumers sharing the same force predicate."""
    feet = SceneEntityCfg("contact_forces", body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        dispatch_backend=dispatch_backend,
        tasks={
            "shared_contact": [
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
            ]
        },
    )


def _shared_direct_cfg(dispatch_backend: str) -> MultiTaskCfg:
    """Cfg with several target-specific consumers sharing current-state producers."""
    base = SceneEntityCfg("robot", body_names="base")
    subtasks = []
    for i in range(4):
        subtasks.append(
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=base,
                state_kernel=int(STATE_KERNEL_ID.BODY_POS),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3 + 0.1 * i,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[-1.0 + 0.25 * i, -0.8 + 0.25 * i, 0.4 + 0.05 * i],
                    maximum=[-0.8 + 0.25 * i, -0.6 + 0.25 * i, 0.5 + 0.05 * i],
                ),
                expose_in_obs=False,
            )
        )
    for i in range(4):
        subtasks.append(
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=base,
                state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.2 + 0.1 * i,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.45 + 0.05 * i],
                    maximum=[0.5 + 0.05 * i],
                ),
                expose_in_obs=False,
            )
        )
    for i in range(4):
        subtasks.append(
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=base,
                state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.3 + 0.1 * i,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                    minimum=[-0.2 + 0.02 * i, -0.2 + 0.02 * i, -0.3 + 0.03 * i],
                    maximum=[0.2 - 0.02 * i, 0.2 - 0.02 * i, 0.3 - 0.03 * i],
                    out_dim=4,
                ),
                expose_in_obs=False,
            )
        )
    for i in range(16):
        subtasks.append(
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=SceneEntityCfg("robot"),
                state_kernel=int(STATE_KERNEL_ID.JOINT_MECH_POWER),
                metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                activation_kernel=int(ACTIVATION_KERNEL_ID.GAUSSIAN),
                activation_kernel_param=1200.0 + 50.0 * i,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                    minimum=[0.0],
                    maximum=[0.0],
                ),
                expose_in_obs=False,
            )
        )
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        dispatch_backend=dispatch_backend,
        tasks={"shared_direct": subtasks},
    )


def _unsorted_schedule_cfg(dispatch_backend: str) -> MultiTaskCfg:
    """Cfg whose authored slot order is intentionally not schedule-sorted."""
    base = SceneEntityCfg("robot", body_names="base")
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        dispatch_backend=dispatch_backend,
        tasks={
            "quat_then_pos": [
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
            ]
        },
    )


# ---------------------------------------------------------------------------
# The test — builds two commands side-by-side and compares outputs.
# ---------------------------------------------------------------------------


@_NEED_CUDA
def test_warp_matches_pytorch_outputs():
    device = "cuda:0"
    num_envs = 32

    def build_and_step(dispatch_backend: str):
        # Same seed → identical scene data across backends, so torch vs warp
        # outputs differ only by the dispatch path itself.
        env = _make_env(num_envs=num_envs, device=device, seed=1234)
        torch.manual_seed(7)  # same task samples both sides
        cmd = MultiTaskCommand(_mixed_cfg(dispatch_backend=dispatch_backend), env)
        # Drive one step — targets are pinned by the seed, so both paths
        # start with the same `_targets_flat` and `_env_subtask_ids`.
        cmd._update_command()
        return cmd

    cmd_py = build_and_step(dispatch_backend="torch")
    for backend in (
        "mega_kernel",
        "schedule_ordered_mega",
        "packed_scatter",
        "primitive_queue_local",
        "primitive_graph_local",
    ):
        cmd_wp = build_and_step(dispatch_backend=backend)

        # Sanity: both paths sampled the same tasks / targets.
        assert torch.equal(cmd_py._env_subtask_ids, cmd_wp._env_subtask_ids), backend
        assert torch.equal(cmd_py._env_slot_count, cmd_wp._env_slot_count), backend
        assert torch.allclose(cmd_py._targets_flat, cmd_wp._targets_flat), backend

        atol = 1e-5
        rtol = 1e-5
        assert torch.allclose(cmd_py._buf_error, cmd_wp._buf_error, atol=atol, rtol=rtol), (
            f"{backend} error mismatch: max |Δ| = {(cmd_py._buf_error - cmd_wp._buf_error).abs().max().item():.3e}"
        )
        assert torch.allclose(cmd_py._buf_activation, cmd_wp._buf_activation, atol=atol, rtol=rtol), (
            f"{backend} activation mismatch: max |Δ| = "
            f"{(cmd_py._buf_activation - cmd_wp._buf_activation).abs().max().item():.3e}"
        )
        assert torch.allclose(cmd_py._command_reach, cmd_wp._command_reach, atol=atol, rtol=rtol), (
            f"{backend} reach mismatch: max |Δ| = "
            f"{(cmd_py._command_reach - cmd_wp._command_reach).abs().max().item():.3e}"
        )
        assert torch.allclose(cmd_py._command_track, cmd_wp._command_track, atol=atol, rtol=rtol), (
            f"{backend} track mismatch: max |Δ| = "
            f"{(cmd_py._command_track - cmd_wp._command_track).abs().max().item():.3e}"
        )

        # Composer outputs — task reward, success, progress, and in-place state.
        assert torch.allclose(cmd_py._task_reward, cmd_wp._task_reward, atol=atol, rtol=rtol), (
            f"{backend} task_reward mismatch: max |Δ| = "
            f"{(cmd_py._task_reward - cmd_wp._task_reward).abs().max().item():.3e}"
        )
        assert torch.equal(cmd_py._task_done_success, cmd_wp._task_done_success), backend
        assert torch.allclose(cmd_py._progress, cmd_wp._progress, atol=atol, rtol=rtol), (
            f"{backend} progress mismatch: max |Δ| = {(cmd_py._progress - cmd_wp._progress).abs().max().item():.3e}"
        )
        assert torch.allclose(cmd_py._sum_activation, cmd_wp._sum_activation, atol=atol, rtol=rtol), (
            f"{backend} sum_activation mismatch: max |Δ| = "
            f"{(cmd_py._sum_activation - cmd_wp._sum_activation).abs().max().item():.3e}"
        )
        assert torch.equal(cmd_py._transit_steps, cmd_wp._transit_steps), backend
        assert torch.equal(cmd_py._instant_achieved, cmd_wp._instant_achieved), backend


@_NEED_CUDA
@pytest.mark.parametrize("dispatch_backend", ["primitive_queue_local", "primitive_graph_local"])
def test_local_primitive_dispatch_compose_graph_capture_smoke(dispatch_backend):
    device = "cuda:0"
    num_envs = 32
    env = _make_env(num_envs=num_envs, device=device, seed=4321)
    torch.manual_seed(17)
    cmd = MultiTaskCommand(_mixed_cfg(dispatch_backend=dispatch_backend), env)
    torch.lt(cmd._slot_arange, cmd._env_slot_count.unsqueeze(1), out=cmd._slot_valid)
    cmd._buf_error.zero_()
    cmd._buf_activation.zero_()
    cmd._backend.dispatch(cmd)
    assert cmd._backend._dispatch_graph is not None
    cmd._backend.compose(cmd)
    torch.cuda.synchronize()

    with wp.ScopedCapture(device=device) as capture:
        cmd._backend.dispatch(cmd)
        cmd._backend.compose(cmd)
    wp.capture_launch(capture.graph)
    torch.cuda.synchronize()

    assert torch.isfinite(cmd.task_reward).all()
    assert torch.isfinite(cmd.progress).all()


@_NEED_CUDA
def test_primitive_graph_local_shares_contact_predicate_nodes():
    device = "cuda:0"
    num_envs = 16

    def build_and_step(dispatch_backend: str):
        env = _make_env(num_envs=num_envs, device=device, seed=5678)
        torch.manual_seed(23)
        cmd = MultiTaskCommand(_shared_contact_cfg(dispatch_backend=dispatch_backend), env)
        cmd._update_command()
        return cmd

    cmd_ref = build_and_step("torch")
    cmd_graph = build_and_step("primitive_graph_local")

    plan = cmd_graph._backend.plan
    contact_work = sum(
        plan.schedule_counts_py[schedule_id]
        for schedule_id in (
            SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
            SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
            SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
        )
    )
    assert contact_work == num_envs * 3
    assert plan.contact_signature_count == 1

    atol = 1e-5
    rtol = 1e-5
    assert torch.allclose(cmd_ref._buf_error, cmd_graph._buf_error, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref._buf_activation, cmd_graph._buf_activation, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.command_reach, cmd_graph.command_reach, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.command_track, cmd_graph.command_track, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.task_reward, cmd_graph.task_reward, atol=atol, rtol=rtol)
    assert torch.equal(cmd_ref.task_done, cmd_graph.task_done)


@_NEED_CUDA
def test_primitive_graph_local_materializes_shared_signature_nodes():
    device = "cuda:0"
    num_envs = 16

    def build_and_step(dispatch_backend: str):
        env = _make_env(num_envs=num_envs, device=device, seed=6789)
        torch.manual_seed(29)
        cmd = MultiTaskCommand(_shared_direct_cfg(dispatch_backend=dispatch_backend), env)
        cmd._update_command()
        return cmd

    cmd_ref = build_and_step("torch")
    cmd_graph = build_and_step("primitive_graph_local")

    plan = cmd_graph._backend.plan
    assert plan.schedule_counts_py[SCHEDULE_DIRECT_VEC3_DELTA] == num_envs * 4
    assert plan.schedule_counts_py[SCHEDULE_DIRECT_SCALAR_DELTA] == num_envs * 4
    assert plan.schedule_counts_py[SCHEDULE_DIRECT_QUAT_DELTA] == num_envs * 4
    assert plan.schedule_counts_py[SCHEDULE_SCALAR_SUM_DELTA] == num_envs * 16
    assert plan.vec3_signature_count == 1
    assert plan.scalar_signature_count == 1
    assert plan.quat_signature_count == 1
    assert plan.scalar_sum_signature_count == 1

    atol = 1e-5
    rtol = 1e-5
    assert torch.allclose(cmd_ref._buf_error, cmd_graph._buf_error, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref._buf_activation, cmd_graph._buf_activation, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.task_reward, cmd_graph.task_reward, atol=atol, rtol=rtol)
    assert torch.equal(cmd_ref.task_done, cmd_graph.task_done)


@_NEED_CUDA
def test_schedule_ordered_mega_reorders_slots_without_changing_public_outputs():
    device = "cuda:0"
    num_envs = 16

    def build_and_step(dispatch_backend: str):
        env = _make_env(num_envs=num_envs, device=device, seed=2345)
        torch.manual_seed(11)
        cmd = MultiTaskCommand(_unsorted_schedule_cfg(dispatch_backend=dispatch_backend), env)
        cmd._update_command()
        return cmd

    cmd_ref = build_and_step("torch")
    cmd_ordered = build_and_step("schedule_ordered_mega")

    ref_state_order = cmd_ref.spec.state_kernel_id[cmd_ref._env_subtask_ids[0].long()].tolist()
    ordered_state_order = cmd_ordered.spec.state_kernel_id[cmd_ordered._env_subtask_ids[0].long()].tolist()
    assert ref_state_order == [int(STATE_KERNEL_ID.BODY_QUAT), int(STATE_KERNEL_ID.BODY_POS)]
    assert ordered_state_order == [int(STATE_KERNEL_ID.BODY_POS), int(STATE_KERNEL_ID.BODY_QUAT)]

    atol = 1e-5
    rtol = 1e-5
    assert torch.allclose(cmd_ref._targets_flat, cmd_ordered._targets_flat)
    assert torch.allclose(cmd_ref.command_reach, cmd_ordered.command_reach, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.command_track, cmd_ordered.command_track, atol=atol, rtol=rtol)
    assert torch.allclose(cmd_ref.task_reward, cmd_ordered.task_reward, atol=atol, rtol=rtol)
    assert torch.equal(cmd_ref.task_done, cmd_ordered.task_done)
    assert torch.allclose(cmd_ref.progress, cmd_ordered.progress, atol=atol, rtol=rtol)


def _interleaved_producer_cfg(dispatch_backend: str) -> MultiTaskCfg:
    """Cfg whose authored slot order interleaves two vec3 producers.

    Four subtasks all in the vec3 schedule kind: ``LIN_VEL, ANG_VEL, LIN_VEL,
    ANG_VEL``. LIN_VEL pair shares one gather signature (producer 0); ANG_VEL
    pair shares another (producer 1). primitive_graph_local's secondary sort
    regroups by producer within the schedule region, turning ``[A, B, A, B]``
    into ``[A, A, B, B]``. The torch reference keeps the authored order.
    """
    base = SceneEntityCfg("robot", body_names="base")
    lin_kwargs = dict(
        asset_cfg=base,
        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
    )
    ang_kwargs = dict(
        asset_cfg=base,
        state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
    )
    return MultiTaskCfg(
        resampling_time_range=(100.0, 100.0),
        debug_vis=False,
        dispatch_backend=dispatch_backend,
        tasks={
            "interleaved_producers": [
                MultiTaskCfg.TrackingTaskCfg(
                    **lin_kwargs,
                    activation_kernel_param=0.3,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[-1.0, -1.0, 0.0],
                        maximum=[1.0, 1.0, 0.0],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    **ang_kwargs,
                    activation_kernel_param=0.5,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, -0.5],
                        maximum=[0.0, 0.0, 0.5],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    **lin_kwargs,
                    activation_kernel_param=0.4,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[-0.5, -0.5, 0.0],
                        maximum=[0.5, 0.5, 0.0],
                    ),
                ),
                MultiTaskCfg.TrackingTaskCfg(
                    **ang_kwargs,
                    activation_kernel_param=0.6,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[0.0, 0.0, -0.3],
                        maximum=[0.0, 0.0, 0.3],
                    ),
                ),
            ]
        },
    )


@_NEED_CUDA
def test_primitive_graph_local_applies_producer_coherent_slot_order():
    """The secondary producer-coherent sort regroups slots within each schedule.

    The interleaved cfg has subtask order ``[LIN_VEL, ANG_VEL, LIN_VEL, ANG_VEL]``
    (producers ``[0, 1, 0, 1]`` within the vec3 kind). primitive_graph_local
    applies the composite ``(schedule_id, producer_within_kind)`` sort, which
    on this cfg degenerates to a pure producer-id sort because all four
    subtasks share schedule_id = SCHEDULE_DIRECT_VEC3_DELTA. The expected
    post-sort order is ``[0, 2, 1, 3]`` — producers ``[0, 0, 1, 1]`` — which
    is what a warp of consumer threads sees as broadcast-friendly.

    NOTE on why this test asserts slot order directly rather than comparing
    ``cmd_ref.command_track`` against ``cmd_graph.command_track`` (as the
    sibling :func:`test_schedule_ordered_mega_reorders_slots_without_changing_public_outputs`
    does): this cfg has two distinct LIN_VEL subtasks (0 and 2) and two
    distinct ANG_VEL subtasks (1 and 3) all bound to ``robot.base``, so
    subtasks 0 and 2 share canonical position ``[0:3]`` and subtasks 1
    and 3 share ``[3:6]``. The torch reference writes through advanced
    indexing (``self._command_track[rows, cols] = delta``) which PyTorch
    documents as having **implementation-defined behavior for overlapping
    (row, col) writes** — "what 'last' means is not guaranteed". The
    'winning' write varies based on memory layout / prior CUDA state, so
    cross-backend aggregate comparison is unreliable for this cfg by
    design. Asserting the slot reorder (the actual Phase 3 mechanism)
    tests what we care about and is robust.

    This non-determinism doesn't surface in production: real cfgs don't
    pair multiple tracking subtasks against the same (asset, state-kernel,
    body) with different targets, so canonical-overlap doesn't occur.
    Public-output invariance under producer-coherent reorder for
    overlap-free cfgs is exercised in :func:`test_warp_matches_pytorch_outputs`
    (which uses ``_mixed_cfg``, every subtask in a distinct canonical
    position).
    """
    device = "cuda:0"
    num_envs = 16
    env = _make_env(num_envs=num_envs, device=device, seed=4567)
    torch.manual_seed(31)
    cmd = MultiTaskCommand(_interleaved_producer_cfg(dispatch_backend="primitive_graph_local"), env)
    cmd._update_command()

    # Every env in this single-task cfg gets the same slot assignment, so
    # checking env 0 is sufficient to verify the reorder applied.
    observed = cmd._env_subtask_ids[0].tolist()
    assert observed == [0, 2, 1, 3], f"producer-coherent order not applied: expected [0, 2, 1, 3], got {observed}"
    # Each producer is shared by 2 consumers in this cfg.
    assert cmd._backend.plan.vec3_nodes.csr_graph.num_producers == 2
    assert cmd._backend.plan.vec3_nodes.csr_graph.num_active_consumers == 4
    assert cmd._backend.plan.vec3_nodes.csr_graph.fanout_histogram == {2: 2}
