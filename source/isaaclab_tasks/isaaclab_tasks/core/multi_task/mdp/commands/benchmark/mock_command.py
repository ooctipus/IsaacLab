# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock ``MultiTaskCommand`` construction for command-level benchmarks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    import warp as wp

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


def _make_proxy(torch_tensor: torch.Tensor, wp_dtype) -> wp.array:
    """Wrap a contiguous torch tensor as a ProxyArray with the given Warp dtype."""
    import warp as wp

    from isaaclab.utils.warp import ProxyArray

    if wp_dtype is wp.float32:
        shape = tuple(torch_tensor.shape)
    elif wp_dtype is wp.vec3 or wp_dtype is wp.quat:
        shape = tuple(torch_tensor.shape[:-1])
    else:
        raise ValueError(f"Unsupported wp_dtype {wp_dtype!r}")
    return ProxyArray(
        wp.array(
            ptr=torch_tensor.data_ptr(),
            dtype=wp_dtype,
            shape=shape,
            device=str(torch_tensor.device),
        )
    )


class _MockArticulationData:
    """Mock ``.data`` exposing ProxyArrays for every kind read by the Warp path.

    Both the Torch path (``data.body_pos_w.torch``) and the Warp path
    (``data.body_pos_w.warp``) read from one seeded source.
    """

    def __init__(self, num_envs: int, num_bodies: int, num_joints: int, device: str):
        import warp as wp

        # Identity quat for rotation helpers; ProxyArray so production-style
        # ``.warp`` / ``.torch`` accessors work just like real articulations.
        self._root_quat_w_torch = torch.zeros(num_envs, 4, device=device).contiguous()
        self._root_quat_w_torch[:, 3] = 1.0
        self.root_quat_w = _make_proxy(self._root_quat_w_torch, wp.quat)

        # Backing torch tensors (kept alive so the wp.array aliases stay valid).
        self._body_pos_w_torch = torch.randn(num_envs, num_bodies, 3, device=device).contiguous()
        quat_raw = torch.randn(num_envs, num_bodies, 4, device=device)
        self._body_quat_w_torch = torch.nn.functional.normalize(quat_raw, dim=-1).contiguous()
        self._body_lin_vel_w_torch = torch.randn(num_envs, num_bodies, 3, device=device).contiguous()
        self._body_ang_vel_w_torch = torch.randn(num_envs, num_bodies, 3, device=device).contiguous()
        self._joint_pos_torch = torch.randn(num_envs, num_joints, device=device).contiguous()
        self._joint_vel_torch = torch.randn(num_envs, num_joints, device=device).contiguous()
        self._applied_torque_torch = torch.randn(num_envs, num_joints, device=device).contiguous()

        self.body_pos_w = _make_proxy(self._body_pos_w_torch, wp.vec3)
        self.body_quat_w = _make_proxy(self._body_quat_w_torch, wp.quat)
        self.body_lin_vel_w = _make_proxy(self._body_lin_vel_w_torch, wp.vec3)
        self.body_ang_vel_w = _make_proxy(self._body_ang_vel_w_torch, wp.vec3)
        self.joint_pos = _make_proxy(self._joint_pos_torch, wp.float32)
        self.joint_vel = _make_proxy(self._joint_vel_torch, wp.float32)
        self.applied_torque = _make_proxy(self._applied_torque_torch, wp.float32)


class _MockContactSensorData:
    """Mock contact sensor data: ProxyArray for ``net_forces_w``."""

    def __init__(self, num_envs: int, num_bodies: int, device: str):
        import warp as wp

        self._net_forces_w_torch = (torch.randn(num_envs, num_bodies, 3, device=device).abs() * 5.0).contiguous()
        self.net_forces_w = _make_proxy(self._net_forces_w_torch, wp.vec3)


class MockArticulation:
    """Stand-in for :class:`Articulation` satisfying ``SceneEntityCfg.resolve``."""

    def __init__(
        self,
        body_names: list[str],
        joint_names: list[str] | None = None,
        num_envs: int = 1,
        device: str = "cpu",
    ):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names) if joint_names else []
        self.num_bodies = len(self.body_names)
        self.num_joints = len(self.joint_names)
        self.fixed_tendon_names: list[str] = []
        self.num_fixed_tendons = 0
        # ProxyArray-backed mock data — works for both Torch and Warp paths.
        if self.num_joints > 0:
            self.data = _MockArticulationData(num_envs, self.num_bodies, self.num_joints, device)
        else:
            # Contact-sensor-style entity (no joints).
            self.data = _MockContactSensorData(num_envs, self.num_bodies, device)

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


class MockScene:
    """Small scene object exposing the fields used by ``MultiTaskCommand``."""

    def __init__(self, entities: dict, num_envs: int, device: str):
        self._entities = entities
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        self.sensors = entities

    def keys(self):
        return self._entities.keys()

    def __getitem__(self, name):
        return self._entities[name]

    def __contains__(self, name):
        return name in self._entities


class MockEnv:
    """Minimal env shell for command benchmark execution."""

    def __init__(self, num_envs: int, device: str, max_episode_length: int, scene: MockScene):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.scene = scene
        self.common_step_counter = 0
        self.step_dt = 0.02


def build_mock_synthetic_readers(
    num_envs: int,
    device: str,
    body_names: list[str] | None = None,
    joint_names: list[str] | None = None,
) -> tuple:
    """Build fixed synthetic readers for each command buffer kind."""
    from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import BUFFER_KIND

    nb = len(body_names if body_names is not None else _ANYMAL_BODY_NAMES)
    nj = len(joint_names if joint_names is not None else _ANYMAL_JOINT_NAMES)
    by_kind = {
        int(BUFFER_KIND.JOINT_POS): torch.randn(num_envs, nj, device=device),
        int(BUFFER_KIND.JOINT_VEL): torch.randn(num_envs, nj, device=device),
        int(BUFFER_KIND.BODY_POS_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.BODY_QUAT_W): torch.nn.functional.normalize(
            torch.randn(num_envs, nb, 4, device=device), dim=-1
        ),
        int(BUFFER_KIND.BODY_LIN_VEL_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.BODY_ANG_VEL_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.CONTACT_NET_FORCES_W): torch.randn(num_envs, nb, 3, device=device).abs() * 5.0,
        int(BUFFER_KIND.JOINT_MECH_POWER_ABS): torch.randn(num_envs, nj, device=device).abs(),
    }

    def make_reader(kind: int):
        tensor = by_kind[kind]

        def reader(env, asset_name):
            return tensor

        return reader

    return tuple(make_reader(int(k)) for k in BUFFER_KIND)


def _build_shared_direct_tasks():
    """Build a high-fanout public-command workload over existing state kernels."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import MinMaxSampler, MultiTaskCfg

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
    return {"shared_direct": subtasks}


def _build_future_synthetic_tasks(*, interleave: bool = False):
    """Build a wide public-command workload over production state kernels."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import (
        ACTIVATION_KERNEL_ID,
        METRIC_KERNEL_ID,
        SAMPLER_KERNEL_ID,
        STATE_KERNEL_ID,
    )
    from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import MinMaxSampler, MultiTaskCfg

    base = SceneEntityCfg("robot", body_names="base")
    robot = SceneEntityCfg("robot")
    feet = SceneEntityCfg(
        "contact_forces",
        body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    )
    groups = []

    def make_tracking(asset_cfg, state_kernel: int, minimum: list[float], maximum: list[float], count: int) -> list:
        subtasks = []
        for i in range(count):
            shift = 0.01 * i
            subtasks.append(
                MultiTaskCfg.TrackingTaskCfg(
                    asset_cfg=asset_cfg,
                    state_kernel=state_kernel,
                    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
                    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                    activation_kernel_param=0.4 + 0.01 * i,
                    sampler=MinMaxSampler(
                        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
                        minimum=[v + shift for v in minimum],
                        maximum=[v + shift for v in maximum],
                    ),
                    expose_in_obs=False,
                )
            )
        return subtasks

    groups.append(make_tracking(base, int(STATE_KERNEL_ID.BODY_POS), [-1.0, -1.0, 0.4], [1.0, 1.0, 0.7], 32))
    groups.append(make_tracking(base, int(STATE_KERNEL_ID.BODY_LIN_VEL), [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 32))
    groups.append(make_tracking(base, int(STATE_KERNEL_ID.BODY_ANG_VEL), [0.0, 0.0, -1.5], [0.0, 0.0, 1.5], 32))
    groups.append(make_tracking(base, int(STATE_KERNEL_ID.BODY_POS_Z), [0.4], [0.7], 32))
    groups.append(
        make_tracking(feet, int(STATE_KERNEL_ID.BODY_CONTACT), [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 32)
    )
    groups.append(make_tracking(feet, int(STATE_KERNEL_ID.BODY_CONTACT_COUNT), [0.0], [0.0], 32))
    groups.append(make_tracking(feet, int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF), [0.0], [0.0], 32))
    groups.append(make_tracking(robot, int(STATE_KERNEL_ID.JOINT_MECH_POWER), [0.0], [0.0], 64))

    quat_subtasks = []
    for i in range(32):
        shift = 0.002 * i
        quat_subtasks.append(
            MultiTaskCfg.TrackingTaskCfg(
                asset_cfg=base,
                state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
                metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
                activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
                activation_kernel_param=0.4 + 0.01 * i,
                sampler=MinMaxSampler(
                    kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
                    minimum=[-0.2 + shift, -0.2 + shift, -0.3 + shift],
                    maximum=[0.2 - shift, 0.2 - shift, 0.3 - shift],
                    out_dim=4,
                ),
                expose_in_obs=False,
            )
        )
    groups.append(quat_subtasks)

    if interleave:
        subtasks = []
        for i in range(max(len(group) for group in groups)):
            for group in groups:
                if i < len(group):
                    subtasks.append(group[i])
    else:
        subtasks = [subtask for group in groups for subtask in group]

    return {"future_synthetic": subtasks}


def build_mock_command(num_envs: int, device: str, dispatch_backend: str, preset: str | None = None):
    """Construct a real command term against a mocked env and synthetic readers."""
    from isaaclab_tasks.core.multi_task.mdp.commands import multi_task_command as mtc_mod
    from isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command import MultiTaskCommand
    from isaaclab_tasks.core.multi_task.multi_task_env_cfg import MultiTaskEnvCfg
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.multitask_presets import MultiTaskTasksPresetCfg
    from isaaclab_tasks.utils import resolve_presets

    body_names = _ANYMAL_BODY_NAMES
    joint_names = _ANYMAL_JOINT_NAMES
    if preset in ("future_synthetic_heavy", "future_synthetic_heavy_interleaved"):
        joint_names = [f"J{i}" for i in range(128)]

    robot = MockArticulation(body_names=body_names, joint_names=joint_names, num_envs=num_envs, device=device)
    contact_forces = MockArticulation(body_names=body_names, num_envs=num_envs, device=device)
    scene = MockScene({"robot": robot, "contact_forces": contact_forces}, num_envs=num_envs, device=device)
    env = MockEnv(num_envs=num_envs, device=device, max_episode_length=200, scene=scene)

    env_cfg = MultiTaskEnvCfg()
    resolve_presets(env_cfg)
    cfg = env_cfg.commands.goal_point
    cfg.debug_vis = False
    cfg.dispatch_backend = dispatch_backend
    if preset is not None:
        if preset == "shared_direct":
            cfg.tasks = _build_shared_direct_tasks()
        elif preset == "future_synthetic":
            cfg.tasks = _build_future_synthetic_tasks()
        elif preset == "future_synthetic_interleaved":
            cfg.tasks = _build_future_synthetic_tasks(interleave=True)
        elif preset == "future_synthetic_heavy":
            cfg.tasks = _build_future_synthetic_tasks()
        elif preset == "future_synthetic_heavy_interleaved":
            cfg.tasks = _build_future_synthetic_tasks(interleave=True)
        else:
            cfg.tasks = getattr(MultiTaskTasksPresetCfg(), preset)
    readers = build_mock_synthetic_readers(num_envs, device, body_names=body_names, joint_names=joint_names)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        cmd = MultiTaskCommand(cfg, env)
    return cmd, env, readers, mtc_mod
