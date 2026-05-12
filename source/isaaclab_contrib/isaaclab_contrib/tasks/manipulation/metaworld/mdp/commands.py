# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Meta-World+ paired (object, goal) command term.

Meta-World samples the cube position and goal position together at reset and
rejects pairs whose ``xy`` projection is closer than 15 cm. The same command
term is shared across reach, push, and pick-place — the only thing that
varies per task is the sampling rectangles.

The term also exposes the post-reset object position, the post-reset TCP
center, and the per-pad init positions as tensors that the reward functions
read. This keeps the "init state cache" alongside the goal it goes with so
there's a single source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


# ────────────────────────────────────────────────────────────────────────────
# Cfg
# ────────────────────────────────────────────────────────────────────────────


@configclass
class MetaworldPairedCommandCfg(CommandTermCfg):
    """Cfg for a paired ``(obj_init_xyz, goal_xyz)`` sampler with rejection.

    All ``low/high`` ranges are in the **env-local frame** (i.e. the robot
    base frame). The minimum xy-separation matches Meta-World's MJCF reset
    code (15 cm).
    """

    class_type: type[CommandTerm] | str = (
        "isaaclab_contrib.tasks.manipulation.metaworld.mdp.commands:MetaworldPairedCommand"
    )

    object_name: str = MISSING
    """Scene name of the rigid object to write the sampled init pose to."""

    frame_transformer_name: str = "tcp_frame"
    """Scene name of the FrameTransformer that exposes ``(leftpad, rightpad)``
    so the paired command can cache TCP / pad init positions for the rewards."""

    obj_low: tuple[float, float, float] = MISSING
    obj_high: tuple[float, float, float] = MISSING
    goal_low: tuple[float, float, float] = MISSING
    goal_high: tuple[float, float, float] = MISSING

    min_xy_separation: float = 0.15
    """Reject ``(obj, goal)`` pairs whose ``xy`` distance is below this."""

    max_resample_iters: int = 10
    """Cap on rejection-resample iterations (prevents pathological loops at
    extreme range overlap)."""

    hand_init_pos_e: tuple[float, float, float] = (-0.067, 0.571, 0.132)
    """Hardcoded post-reset TCP position [m] in env-local frame.

    The FrameTransformer exposes the *previous-frame* pad positions during
    :meth:`reset`, before PhysX has had a chance to settle the joints onto
    their new targets. Reading from it would yield the spawn pose, which is
    metres off — and the resulting ``init_tcp_e`` would corrupt every reward
    that uses it as a tolerance margin (push/pick-place caging xz-margin in
    particular). We side-step the staleness by hardcoding the value: the
    Sawyer's joint defaults are deterministic, so the realised post-settle
    TCP is the same every reset (within the small ``reset_joints_by_offset``
    noise, which is below the millimetre-level tolerance noise floor)."""

    init_left_pad_offset_e: tuple[float, float, float] = (-0.070, 0.605, 0.131)
    """Leftpad fingertip at default joint pose (env-local), measured from the
    FrameTransformer with its ``-0.045 m`` z offset applied — i.e. the actual
    point the reward measures."""

    init_right_pad_offset_e: tuple[float, float, float] = (-0.063, 0.538, 0.133)
    """Rightpad fingertip at default joint pose (env-local). Same convention."""


# ────────────────────────────────────────────────────────────────────────────
# Term
# ────────────────────────────────────────────────────────────────────────────


class MetaworldPairedCommand(CommandTerm):
    """Per-env (obj_init, goal) sampler with rejection on xy separation.

    Buffers exposed for reward terms to read (all in **env-local** frame):

    * :attr:`command` — goal xyz, shape ``(num_envs, 3)`` (this is what
      ``env.command_manager.get_command(name)[:, 0:3]`` returns).
    * :attr:`obj_init_pos_e` — object xyz at reset, shape ``(num_envs, 3)``.
    * :attr:`init_tcp_e` — TCP center at reset, shape ``(num_envs, 3)``.
    * :attr:`init_left_pad_e`, :attr:`init_right_pad_e` — per-pad COMs at
      reset (used by pick-place's task-local caging variant).
    """

    cfg: MetaworldPairedCommandCfg

    def __init__(self, cfg: MetaworldPairedCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        device = env.device
        n = env.num_envs
        self._command_buf = torch.zeros((n, 3), device=device)
        self.obj_init_pos_e = torch.zeros((n, 3), device=device)
        self.init_tcp_e = torch.zeros((n, 3), device=device)
        self.init_left_pad_e = torch.zeros((n, 3), device=device)
        self.init_right_pad_e = torch.zeros((n, 3), device=device)

        self._obj_low = torch.tensor(cfg.obj_low, device=device)
        self._obj_high = torch.tensor(cfg.obj_high, device=device)
        self._goal_low = torch.tensor(cfg.goal_low, device=device)
        self._goal_high = torch.tensor(cfg.goal_high, device=device)

        self._object: RigidObject = env.scene[cfg.object_name]
        self._frame_transformer: FrameTransformer = env.scene[cfg.frame_transformer_name]

    # ── CommandTerm API ─────────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        """Goal xyz in env-local frame, shape ``(num_envs, 3)``."""
        return self._command_buf

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample (obj_xyz, goal_xyz) for the given envs with rejection on xy."""
        ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)
        )
        if ids.numel() == 0:
            return

        n = ids.numel()
        device = self._env.device
        obj = self._uniform(n, self._obj_low, self._obj_high, device)
        goal = self._uniform(n, self._goal_low, self._goal_high, device)

        # Vectorised rejection: find pairs with xy-dist below threshold and
        # resample only those rows. A handful of iterations is enough at
        # batch sizes of thousands.
        for _ in range(self.cfg.max_resample_iters):
            xy_dist = torch.linalg.norm(obj[:, :2] - goal[:, :2], dim=-1)
            bad = xy_dist < self.cfg.min_xy_separation
            if not bad.any():
                break
            n_bad = int(bad.sum().item())
            obj_re = self._uniform(n_bad, self._obj_low, self._obj_high, device)
            goal_re = self._uniform(n_bad, self._goal_low, self._goal_high, device)
            obj[bad] = obj_re
            goal[bad] = goal_re

        # Write goal command + obj init buffers.
        self._command_buf[ids] = goal
        self.obj_init_pos_e[ids] = obj

        # Apply the obj pose in world frame (env_origin + env-local).
        env_origins = self._env.scene.env_origins[ids]
        obj_pos_w = env_origins + obj
        # Default orientation = identity (Meta-World's reset_model sets only xyz;
        # cube quat stays at default).
        quat = torch.zeros((n, 4), device=device)
        quat[:, 0] = 1.0  # w=1 in (w,x,y,z)
        # Stack to (n, 7): pos (3) + quat (4)
        pose = torch.cat([obj_pos_w, quat], dim=-1)
        self._object.write_root_pose_to_sim(pose, env_ids=ids)
        # Zero linvel/angvel.
        self._object.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=ids)

    def _update_command(self) -> None:
        """No per-step command update (Meta-World keeps the goal fixed across
        the episode)."""
        return

    def _update_metrics(self) -> None:
        return

    # ── Reset hook to cache init TCP/pad positions ─────────────────────────

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Sample a new (obj, goal) pair and cache TCP/pad init positions.

        We need to call ``super().reset()`` *first* so the resample fires and
        writes the obj pose. Then we read the post-reset pad world positions
        from the FrameTransformer and store them env-local for the rewards.
        """
        info = super().reset(env_ids)

        if env_ids is None:
            ids = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids
        else:
            ids = torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)

        # Pad/TCP positions are CAPTURED FROM CFG CONSTANTS rather than read
        # from the FrameTransformer: at reset() time the transformer's
        # ``target_pos_w`` is the *previous*-frame value (pre-settle spawn
        # pose), not the post-settle TCP — using it would corrupt every
        # tolerance margin that depends on init-TCP. The Sawyer's joint
        # defaults are deterministic, so the realised post-settle TCP is the
        # same every reset (within reset noise).
        device = self._env.device
        hand_init = torch.tensor(self.cfg.hand_init_pos_e, device=device).expand(self._env.num_envs, 3)
        left_init = torch.tensor(self.cfg.init_left_pad_offset_e, device=device).expand(self._env.num_envs, 3)
        right_init = torch.tensor(self.cfg.init_right_pad_offset_e, device=device).expand(self._env.num_envs, 3)
        self.init_tcp_e[ids] = hand_init[ids]
        self.init_left_pad_e[ids] = left_init[ids]
        self.init_right_pad_e[ids] = right_init[ids]

        return info

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _uniform(n: int, low: torch.Tensor, high: torch.Tensor, device: torch.device) -> torch.Tensor:
        return low + (high - low) * torch.rand((n, low.numel()), device=device)
