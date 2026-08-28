# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment for configurable NIST board assembly."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .factory_env_cfg import FactoryBoardEnvCfg
from .mdp import board_reset


class FactoryBoardEnv(ManagerBasedRLEnv):
    """Rebuild the board config after Hydra resolves presets and overrides."""

    cfg: FactoryBoardEnvCfg

    def __init__(self, cfg: FactoryBoardEnvCfg, render_mode: str | None = None, **kwargs):
        cfg = cfg.replace(scene=cfg.scene.copy(), events=cfg.events.copy())
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        reset = self.event_manager.get_term_cfg("reset_board").func
        if not isinstance(reset, board_reset):
            raise TypeError("FactoryBoardEnv requires the resolved board reset term.")
        self.state_curriculum = reset
