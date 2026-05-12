# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Meta-World+ V2 observation assembly.

The 39-d observation layout matches the paper (App. A.4):

    [0:3]   end-effector xyz
    [3]     gripper opening, normalized to ``[0, 1]``
    [4:7]   object 1 xyz
    [7:11]  object 1 quaternion
    [11:14] object 2 xyz (zero-padded if absent)
    [14:18] object 2 quaternion (zero-padded if absent)
    [18:36] frame stack — previous step's ``[0:18]``
    [36:39] goal xyz

Implemented as a single stateful term so the frame stack uses the *current*
step's ``[0:18]`` directly (no concatenation-order mismatch with IsaacLab's
oldest-first :class:`~isaaclab.utils.buffers.CircularBuffer`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


_GRIPPER_NORM = 0.1
"""Distance [m] mapped to gripper-open ``= 1.0`` (matches Meta-World's clip)."""


class MetaworldObservation(ManagerTermBase):
    """Full 39-d Meta-World+ V2 observation as a stateful term.

    Maintains a per-env buffer of the previous step's ``[0:18]`` slice. On
    reset the buffer is set to the current step's ``[0:18]`` so the very
    first observation is ``[current, current, goal]``, matching Meta-World's
    behaviour where ``_prev_obs`` is initialised from the post-reset state.

    Args:
        cfg: Observation term cfg. ``params`` must include:
            * ``frame_transformer_cfg``: SceneEntity holding the FrameTransformer
              that tracks ``(leftpad, rightpad)`` (in that order). TCP is the
              midpoint.
            * ``object1_cfg``: SceneEntity for object 1 (rigid object).
            * ``object2_cfg``: SceneEntity for object 2 or ``None`` for tasks
              with a single object (reach/push/pick-place).
            * ``goal_command_name``: Command term name exposing the goal
              ``[:, 0:3]`` in the robot base frame.
            * ``robot_cfg``: SceneEntity for the Sawyer articulation.
        env: Active environment.
    """

    cfg: ObservationTermCfg

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        params = cfg.params
        self._frame_transformer_cfg: SceneEntityCfg = params["frame_transformer_cfg"]
        self._object1_cfg: SceneEntityCfg = params["object1_cfg"]
        self._object2_cfg: SceneEntityCfg | None = params.get("object2_cfg", None)
        self._goal_command_name: str = params["goal_command_name"]
        self._robot_cfg: SceneEntityCfg = params["robot_cfg"]

        device = env.device
        num_envs = env.num_envs
        self._prev_18: torch.Tensor = torch.zeros((num_envs, 18), device=device)
        self._prev_initialized: torch.Tensor = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Mark the previous-obs cache as stale for the given envs.

        On the next ``__call__`` for these envs, ``prev_18`` is overwritten
        with the current ``[0:18]`` so the first post-reset observation is
        ``[current, current, goal]``.
        """
        # Replace via torch.where to avoid indexed inplace updates that
        # break under torch.inference_mode (the buffer can become an
        # inference tensor after the first step).
        if env_ids is None:
            self._prev_initialized = torch.zeros_like(self._prev_initialized)
            return
        ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)
        )
        mask = torch.zeros_like(self._prev_initialized)
        mask[ids] = True
        self._prev_initialized = self._prev_initialized & ~mask

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
        object1_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        object2_cfg: SceneEntityCfg | None = None,
        goal_command_name: str = "ee_pose",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del frame_transformer_cfg, object1_cfg, object2_cfg, goal_command_name, robot_cfg  # cached on self
        current_18 = self._compute_current_18(env)

        # Branch on init mask via torch.where so we don't do indexed
        # inplace updates on an inference-mode tensor.
        init_mask = self._prev_initialized.unsqueeze(-1)
        prev_18 = torch.where(init_mask, self._prev_18, current_18)

        # Update caches by reassignment (not indexed mutation).
        self._prev_18 = current_18.detach()
        self._prev_initialized = torch.ones_like(self._prev_initialized)

        goal_3 = env.command_manager.get_command(self._goal_command_name)[:, 0:3]
        return torch.cat([current_18, prev_18, goal_3], dim=-1)

    def _compute_current_18(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Build the 18-d ``[ee_xyz, gripper_open, obj1_pose, obj2_pose]`` slice.

        Positions are reported in the *env-local frame* (current world position
        minus :attr:`env.scene.env_origins`). For a fixed-base arm at the env
        origin this matches Meta-World's world-frame semantics.
        """
        env_origins = env.scene.env_origins  # (B, 3) torch tensor

        ft: FrameTransformer = env.scene[self._frame_transformer_cfg.name]
        # Source frame is the (fixed) robot base, so target_pos_source is already env-local.
        target_pos_e = ft.data.target_pos_source.torch  # (B, 2, 3): (leftpad, rightpad)
        tcp_e = 0.5 * (target_pos_e[:, 0] + target_pos_e[:, 1])

        gripper_dist = torch.linalg.norm(target_pos_e[:, 0] - target_pos_e[:, 1], dim=-1, keepdim=True)
        gripper_open = torch.clamp(gripper_dist / _GRIPPER_NORM, 0.0, 1.0)

        obj1: RigidObject = env.scene[self._object1_cfg.name]
        obj1_pos_e = obj1.data.root_pos_w.torch - env_origins
        obj1_quat = obj1.data.root_quat_w.torch  # (B, 4) — IsaacLab uses (w, x, y, z)

        if self._object2_cfg is None:
            obj2_pos_e = torch.zeros_like(obj1_pos_e)
            obj2_quat = torch.zeros_like(obj1_quat)
        else:
            obj2: RigidObject = env.scene[self._object2_cfg.name]
            obj2_pos_e = obj2.data.root_pos_w.torch - env_origins
            obj2_quat = obj2.data.root_quat_w.torch

        return torch.cat([tcp_e, gripper_open, obj1_pos_e, obj1_quat, obj2_pos_e, obj2_quat], dim=-1)
