# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Meta-World+ gripper action term.

Meta-World drives the parallel gripper from a single scalar ``action[-1] ∈ [-1, 1]``
sent through MuJoCo position actuators on ``r_close`` (range ``[0, 0.04]``) and
``l_close`` (range ``[-0.03, 0]``):

    r_close target = action[-1]      (clipped to [0, 0.04])
    l_close target = -action[-1]     (clipped to [-0.03, 0])

Effect: ``action[-1] = +1`` drives both pads toward the centerline (CLOSE)
because ``r_close → +0.04`` slides rightclaw ``+Y`` and ``l_close → -0.03``
slides leftclaw ``-Y``, both reducing the gap. ``action[-1] = -1`` opens.
The joint range itself does the clamping.

We replicate this exactly. There is no stock IsaacLab term for an asymmetric
sign-flipped scalar-to-pair mapping, so this is a thin custom term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import (
    ActionTermCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


@configclass
class MetaworldGripperActionCfg(ActionTermCfg):
    """Cfg for the Meta-World scalar parallel gripper action.

    The articulation is expected to have two prismatic joints named
    :attr:`right_joint_name` and :attr:`left_joint_name` whose limits define
    the open/close extents (Meta-World ships ``r_close=[0, 0.04]`` and
    ``l_close=[-0.03, 0]``).
    """

    class_type: type[ActionTerm] | str = (
        "isaaclab_contrib.tasks.manipulation.metaworld.mdp.actions:MetaworldGripperAction"
    )

    asset_name: str = "robot"
    """Name of the articulation in the scene."""

    right_joint_name: str = "r_close"
    """Joint that takes the action value as its position target."""

    left_joint_name: str = "l_close"
    """Joint that takes ``-action`` as its position target."""


class MetaworldGripperAction(ActionTerm):
    """1-d scalar gripper action mapping Meta-World's mocap actuator semantics."""

    cfg: MetaworldGripperActionCfg
    _asset: Articulation

    def __init__(self, cfg: MetaworldGripperActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        joint_names = list(self._asset.joint_names)
        try:
            self._r_idx = joint_names.index(cfg.right_joint_name)
            self._l_idx = joint_names.index(cfg.left_joint_name)
        except ValueError as err:
            raise ValueError(
                f"MetaworldGripperAction: articulation {cfg.asset_name!r} is missing one of"
                f" ({cfg.right_joint_name!r}, {cfg.left_joint_name!r}). Joints: {joint_names}"
            ) from err

        self._raw_actions = torch.zeros((env.num_envs, 1), device=env.device)
        self._joint_targets = torch.zeros((env.num_envs, 2), device=env.device)
        self._joint_idx = torch.tensor([self._r_idx, self._l_idx], device=env.device, dtype=torch.long)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        # MW's ``np.clip(action, -1, 1)`` happens before the gripper drives
        # the prismatic joints. Without it, a policy emitting ``action[-1]=+3``
        # would saturate ``r_close`` to its joint limit instantly (and break
        # the parity with MW's clipped action range, which the reward's
        # ``gripper_open`` term assumes).
        actions = actions.clamp(min=-1.0, max=1.0)
        self._raw_actions[:] = actions
        a = actions.squeeze(-1)
        # Joint range clamps the rest — match Meta-World's MuJoCo semantics.
        self._joint_targets[:, 0] = a  # r_close  ←  +action
        self._joint_targets[:, 1] = -a  # l_close  ←  -action

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._joint_targets, joint_ids=self._joint_idx)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._joint_targets[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._joint_targets[env_ids] = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Arm action: DiffIK with Meta-World workspace clamp
# ────────────────────────────────────────────────────────────────────────────


@configclass
class MetaworldArmActionCfg(DifferentialInverseKinematicsActionCfg):
    """DiffIK arm action with Meta-World's workspace clamp on the EE target.

    Meta-World's :func:`set_xyz_action` adds the action delta to the mocap
    position, then ``np.clip``s the result to ``mocap_low``/``mocap_high``.
    Stock :class:`DifferentialInverseKinematicsActionCfg` doesn't have this
    clamp, so the runtime term subclasses :class:`DifferentialInverseKinematicsAction`
    and clips the controller's ``ee_pos_des`` after the relative-mode add.

    The ``class_type`` is a string so the runtime parent-class import (which
    pulls in pxr / Warp) is deferred until after :class:`AppLauncher` boots.
    """

    class_type: type[ActionTerm] | str = (
        "isaaclab_contrib.tasks.manipulation.metaworld.mdp.arm_action_impl:MetaworldArmAction"
    )

    # NB: the *class-level* ``mocap_low/high`` defaults in
    # :class:`SawyerMocapBase` (``(-0.2, 0.5, 0.06)``/``(0.2, 0.7, 0.6)``) are
    # **not** what's used at runtime. ``SawyerXYZEnv.__init__`` overrides
    # ``self.mocap_low/high`` with each task's ``hand_low/high``. For all of
    # reach/push/pick-place those values are ``(-0.5, 0.40, 0.05)`` /
    # ``(0.5, 1.0, 0.5)``. We use those — anything tighter makes goals beyond
    # ``y=0.7`` physically unreachable.
    workspace_low: tuple[float, float, float] = (-0.5, 0.40, 0.05)
    """Lower workspace corner [m] in env-local frame (Meta-World ``hand_low``)."""

    workspace_high: tuple[float, float, float] = (0.5, 1.0, 0.5)
    """Upper workspace corner [m] in env-local frame (Meta-World ``hand_high``)."""
