# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms with temporal-response randomization for cross-engine/sim-to-real robustness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import TemporalDRRelativeJointPositionActionCfg


class TemporalDRRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint-position action with per-episode latency and bandwidth randomization.

    Every reset draws, per environment, an action delay (whole control steps) and a first-order
    low-pass coefficient. Together they randomize the *temporal shape* of the closed-loop response,
    which parameter randomization (gains, masses, friction) leaves untouched. Policies trained
    without this overfit the training engine's transient texture and transfer at 0% to another
    engine (or the real robot) even when every physical parameter matches.
    """

    cfg: TemporalDRRelativeJointPositionActionCfg

    def __init__(self, cfg: TemporalDRRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._max_delay = int(cfg.delay_steps[1])
        n, d = self.num_envs, self.action_dim
        self._ring = torch.zeros(self._max_delay + 1, n, d, device=self.device)
        self._head = 0
        self._delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self._beta = torch.ones(n, 1, device=self.device)
        self._ema = torch.zeros(n, d, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        # push newest action into the ring buffer and pick each env's delayed action
        self._head = (self._head + 1) % (self._max_delay + 1)
        self._ring[self._head] = actions
        idx = (self._head - self._delay) % (self._max_delay + 1)
        delayed = self._ring[idx, torch.arange(self.num_envs, device=self.device)]
        # per-env first-order low-pass (beta = 1 -> passthrough)
        self._ema = self._beta * delayed + (1.0 - self._beta) * self._ema
        super().process_actions(self._ema)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        lo, hi = self.cfg.delay_steps
        self._delay[ids] = torch.randint(lo, hi + 1, (n,), device=self.device)
        blo, bhi = self.cfg.ema_beta_range
        self._beta[ids] = torch.empty(n, 1, device=self.device).uniform_(blo, bhi)
        self._ema[ids] = 0.0
        self._ring[:, ids] = 0.0