# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.cloner.cloner_cfg import expand_env_regex_ns
from isaaclab.sim.simulation_context import SimulationContext

if TYPE_CHECKING:
    from pxr import Usd

    from .asset_base_cfg import AssetBaseCfg


class Asset:
    """A plan-owned asset without a runtime simulation view."""

    cfg: AssetBaseCfg
    """Configuration used to author the asset."""

    prim: Usd.Prim | None
    """The exact prim returned by the spawner, or ``None`` when the asset has no spawner."""

    def __init__(self, cfg: AssetBaseCfg):
        """Author an asset from its configuration.

        Args:
            cfg: Configuration for the asset.

        Raises:
            RuntimeError: If there is no active simulation or clone plan, or the plan does not cover the asset.
        """
        cfg.validate()
        self.cfg = cfg.copy()
        self.prim = None

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError(f"Asset at {self.cfg.prim_path!r} requires an active SimulationContext.")
        self.stage: Usd.Stage = sim.stage
        plan = sim.get_clone_plan()
        if plan is None:
            raise RuntimeError(f"Asset at {self.cfg.prim_path!r} requires an active clone plan.")
        self.cfg.prim_path = expand_env_regex_ns(self.cfg.prim_path, plan.env_template)
        if cloner.query.path_to_source(plan, self.cfg.prim_path) is None:
            raise RuntimeError(f"Asset at {self.cfg.prim_path!r} is not covered by the active clone plan.")

        if self.cfg.spawn is not None:
            source_paths = cloner.query.cfg_source_paths(plan, cfg)
            spawn_path = (
                source_paths
                if isinstance(self.cfg.spawn, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg))
                else next(path for path in source_paths if path is not None)
            )
            self.prim = self.cfg.spawn.func(
                spawn_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
