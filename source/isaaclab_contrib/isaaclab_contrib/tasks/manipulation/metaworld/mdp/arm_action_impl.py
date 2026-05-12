# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime implementation of the Meta-World workspace-clamped DiffIK arm action.

Kept in a separate submodule from :mod:`actions` because importing
:class:`isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`
transitively pulls in pxr/Warp modules. Loading those before
:class:`isaaclab.app.AppLauncher` boots breaks SimulationApp startup. The
action manager resolves :attr:`MetaworldArmActionCfg.class_type` (a string) to
this class lazily *after* the simulator is up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions import MetaworldArmActionCfg


class MetaworldArmAction(DifferentialInverseKinematicsAction):
    """3-d pos-delta arm action with Meta-World's workspace clamp.

    Wraps :class:`DifferentialInverseKinematicsAction` and applies an
    env-local ``ee_pos_des`` clamp after the controller integrates the delta,
    matching MW's :func:`SawyerMocapBase.set_xyz_action`.
    """

    def __init__(self, cfg: MetaworldArmActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        device = env.device
        self._ws_low = torch.tensor(cfg.workspace_low, device=device)
        self._ws_high = torch.tensor(cfg.workspace_high, device=device)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Replicate MW's ``np.clip(action, -1, 1)`` *before* the 0.01 scaling
        # in :func:`SawyerMocapBase.set_xyz_action`. Without this, an
        # unclipped policy emitting e.g. ``[2.3, -1.7, 0.4]`` makes per-step
        # IK deltas of 2.3 cm in x — much larger than MW's max 1 cm — so the
        # workspace clamp (which only acts on ``ee_pos_des``) doesn't bring
        # parity with MW.
        actions = actions.clamp(min=-1.0, max=1.0)
        super().process_actions(actions)
        # ``ee_pos_des`` is set by the controller in the articulation's ROOT
        # frame (Sawyer is fixed-base, so root frame == env-local frame).
        # MW's ``mocap_low``/``mocap_high`` are workspace bounds in the same
        # local frame, so we clamp ``ee_pos_des`` directly without any
        # ``env_origins`` correction.
        self._ik_controller.ee_pos_des[:] = torch.clamp(self._ik_controller.ee_pos_des, self._ws_low, self._ws_high)
