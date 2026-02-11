# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime factory for Per-Env action classes. All PerEnv* actions share the same pattern:
- Inherit from PerEnvMixin + base action class
- Override __init__ to allocate buffers with len(self._assigned_envs) instead of num_envs
- Override reset() to filter env_ids through _filter_env_ids before calling super().reset()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import torch

from isaaclab.managers.action_manager import ActionTerm

from .per_env_mixin import PerEnvMixin

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

BaseT = TypeVar("BaseT", bound=ActionTerm)
CfgT = TypeVar("CfgT")


def make_per_env_action(
    base_action_class: type[BaseT],
    cfg_class: type[CfgT],
    class_name: str | None = None,
) -> type:
    """Build a Per-Env action class at runtime from a base action class and its config type.

    The returned class:
    - Inherits from PerEnvMixin and base_action_class (MRO: PerEnvMixin, base_action_class, ...)
    - Allocates _raw_actions and _processed_actions with shape (len(_assigned_envs), action_dim)
    - Implements reset() to filter env_ids to assigned envs before calling super().reset()

    Args:
        base_action_class: The Isaac Lab action class (e.g. JointPositionAction).
        cfg_class: The config dataclass for that action (e.g. JointPositionActionCfg).
        class_name: Optional name for the new class (default: "PerEnv" + base_action_class.__name__).

    Returns:
        A new class that can be used as class_type in action configs.
    """
    name = class_name or f"PerEnv{base_action_class.__name__}"

    class PerEnvAction(PerEnvMixin, base_action_class):
        __doc__ = (
            f"Per-env variant of {base_action_class.__name__} for multi-task environments. Supports assigned_envs."
        )

        cfg: CfgT

        def __init__(self, cfg: CfgT, env: ManagerBasedEnv) -> None:
            super().__init__(cfg, env)
            n_envs = self._raw_actions.shape[0]
            env_idx = torch.tensor(self._assigned_envs, device=self.device)
            # Slice any tensor attribute whose dim 0 is num_envs to assigned_envs only
            for attr in list(vars(self).keys()):
                val = getattr(self, attr, None)
                if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == n_envs:
                    setattr(self, attr, val[env_idx].clone())
            self._raw_actions = torch.zeros((len(self._assigned_envs), self.action_dim), device=self.device)
            self._processed_actions = torch.zeros_like(self._raw_actions)
            # Wrap controllers so their buffers are sliced and reset/reset_idx filter env_ids
            from .per_env_controller_factory import (
                PerEnvControllerWrapper,
                make_per_env_controller_wrapper,
            )

            for ctrl_attr in ("_ik_controller", "_osc"):
                ctrl = getattr(self, ctrl_attr, None)
                if ctrl is not None and not isinstance(ctrl, PerEnvControllerWrapper):
                    setattr(
                        self,
                        ctrl_attr,
                        make_per_env_controller_wrapper(self._assigned_envs, ctrl),
                    )

        def reset(self, env_ids: Sequence[int] | None = None) -> None:
            resolved_env_ids = self._filter_env_ids(env_ids)
            if len(resolved_env_ids) == 0:
                return
            super().reset(resolved_env_ids)

    PerEnvAction.__name__ = name
    PerEnvAction.__qualname__ = name
    return PerEnvAction


# ---------------------------------------------------------------------------
# Concrete Per-Env action classes (no need to define each by hand)
# ---------------------------------------------------------------------------

from isaaclab.envs.mdp import (
    AbsBinaryJointPositionAction,
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionAction,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
    OperationalSpaceControllerActionCfg,
    SurfaceGripperBinaryActionCfg,
)
from isaaclab.envs.mdp.actions.surface_gripper_actions import SurfaceGripperBinaryAction
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)

from .per_env_assets import PerEnvSurfaceGripper


class PerEnvSurfaceGripperBinaryAction(PerEnvMixin, SurfaceGripperBinaryAction):
    """Per-env variant of SurfaceGripperBinaryAction; _asset is a PerEnvSurfaceGripper."""

    _asset: PerEnvSurfaceGripper

    def __init__(self, cfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        n_envs = self._raw_actions.shape[0]
        env_idx = torch.tensor(self._assigned_envs, device=self.device)
        for attr in list(vars(self).keys()):
            val = getattr(self, attr, None)
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == n_envs:
                setattr(self, attr, val[env_idx].clone())
        self._raw_actions = torch.zeros((len(self._assigned_envs), self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        resolved_env_ids = self._filter_env_ids(env_ids)
        if len(resolved_env_ids) == 0:
            return
        super().reset(resolved_env_ids)


PerEnvJointPositionAction = make_per_env_action(JointPositionAction, JointPositionActionCfg)
PerEnvBinaryJointPositionAction = make_per_env_action(BinaryJointPositionAction, BinaryJointPositionActionCfg)
PerEnvAbsBinaryJointPositionAction = make_per_env_action(AbsBinaryJointPositionAction, AbsBinaryJointPositionActionCfg)
PerEnvDifferentialInverseKinematicsAction = make_per_env_action(
    DifferentialInverseKinematicsAction, DifferentialInverseKinematicsActionCfg
)
PerEnvOperationalSpaceControllerAction = make_per_env_action(
    OperationalSpaceControllerAction, OperationalSpaceControllerActionCfg
)

# Registry: action config type -> PerEnv action class (for setup_group_actions etc.)
_CFG_TO_PER_ENV_CLASS: dict[type, type] = {
    JointPositionActionCfg: PerEnvJointPositionAction,
    BinaryJointPositionActionCfg: PerEnvBinaryJointPositionAction,
    AbsBinaryJointPositionActionCfg: PerEnvAbsBinaryJointPositionAction,
    DifferentialInverseKinematicsActionCfg: PerEnvDifferentialInverseKinematicsAction,
    OperationalSpaceControllerActionCfg: PerEnvOperationalSpaceControllerAction,
    SurfaceGripperBinaryActionCfg: PerEnvSurfaceGripperBinaryAction,
}


def get_per_env_action_class(cfg) -> type | None:
    """Return the PerEnv action class for the given action config, or None if not registered.

    Use this in setup_group_actions to resolve class_type from task action configs
    without hardcoding per-action mappings.
    """
    for cfg_cls, per_env_cls in _CFG_TO_PER_ENV_CLASS.items():
        if isinstance(cfg, cfg_cls):
            return per_env_cls
    return None
