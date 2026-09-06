# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from ...kinematics import NewtonKinematicsBuildCfg
from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...utils.symmetry import AssetSymmetryCfg
from ..retarget.cfg import FactoryFamilyCfg, FactoryGeometryCfg

if TYPE_CHECKING:
    # type-only: the warp-heavy payload class parameterizes its ``class_type`` for
    # IDE autocomplete WITHOUT pulling the implementation at config import time --
    # the resolvable string is the runtime value, resolved lazily at instantiation.
    from .reset_state_command_payloads import FactoryAssemblyPayload


@configclass
class FactoryAssemblyAssetCommandCfg:
    """Command defined by the relative state between fixed and held assets."""

    orientation_threshold: float = 0.025
    """Allowed roll/pitch alignment error for success [rad]."""

    position_threshold: float = 0.0025
    """Allowed relative asset position error for success [m]."""

    duration: tuple[float, float] = (0.0, 1.0)
    """Required hold duration range for success [s]."""


@configclass
class FactoryResetStateTableCfg(StateCommandCfg.TaskTableCfg):
    """Reset-state table generation settings + builder selection."""

    class_type: Callable | str = "{DIR}.reset_state_task_table:build_factory_reset_state_task_table"
    """Builder that fills the :class:`~.reset_state_task_table.FactoryResetStateTaskTable`
    from command configuration, resolved scene configuration, and a device."""

    kinematics: NewtonKinematicsBuildCfg = NewtonKinematicsBuildCfg()
    """Shared Newton parse choices for the scene-declared robot articulation."""

    geometry: FactoryGeometryCfg = MISSING
    """Shared robot, board, held-object, and obstacle geometry."""

    families: tuple[FactoryFamilyCfg, ...] = MISSING
    """Placement families with exact fractions, candidate budgets, and visible stages."""

    rows_per_board: int = 8
    """Average reset-state rows kept PER board configuration; the total table size
    is derived as ``rows_per_board x geometry.board.num_boards`` (the
    locomotion ``pool_spacing`` idea: declare density, the size emerges from the
    world library)."""

    targets_per_board: int = 4
    """Goal states selected PER board configuration, as a spatially-spread SUBSET
    of that board's stored rows (locomotion's ``num_targets_per_cell``: targets
    are existing states, so this must be <= :attr:`rows_per_board`). Every spawn
    in a board is paired with its board's full goal set."""

    allowed_tag_pairs: list[tuple[str, str]] | None = None
    """Optional restriction on which task slots survive, by the placement-tag
    *names* of each ``(spawn, target)`` pair. ``None`` (default) keeps every spawn
    x target pair, so training is unchanged. When set, only slots whose
    ``(spawn_tag_name, target_tag_name)`` is in the list survive -- e.g.
    ``[("near_seated", "in_air"), ("in_air", "near_seated")]`` evaluates only the
    seated<->in-air transitions. Tag names are the placement sampler's: the
    assembly bands (``near_seated``, ``mid_insertion``, ``above_tip``) plus
    ``on_table`` and ``in_air``. Unknown names or a set that matches zero slots
    raise at table build."""

    state_table_fps_features: Callable | None = None
    """Feature extractor for state-table compaction and sampler layout."""


@configclass
class FactoryAssemblyPayloadCfg(StateCommandCfg.PayloadCfg):
    """Configuration for :class:`FactoryAssemblyPayload`."""

    class_type: type[FactoryAssemblyPayload] | str = "{DIR}.reset_state_command_payloads:FactoryAssemblyPayload"

    held_asset_cfg: SceneEntityCfg = MISSING
    """Held asset used for assembly progress computation."""

    fixed_asset_cfg: SceneEntityCfg = MISSING
    """Fixed asset used for assembly progress computation."""

    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Robot whose base frame expresses the command's position delta."""

    symmetry: AssetSymmetryCfg = AssetSymmetryCfg()
    """Held-asset symmetry, consumed by the :class:`~...utils.symmetry.Symmetry`.
    Default is no symmetry (identity). Use :class:`~...utils.symmetry.AssetSymmetryCfg`
    with explicit :class:`~...utils.symmetry.AxisSymmetryCfg` or
    :class:`~...utils.symmetry.SemanticSymmetryCfg` elements. Set per-variant via
    :class:`~..factory_presets.HeldAssetSymmetryCfg`."""

    held_asset_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/factory_reset_state/held_asset_target_frame"
    )
    """Debug marker for the held asset's target root-frame pose."""

    held_asset_visualizer_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
