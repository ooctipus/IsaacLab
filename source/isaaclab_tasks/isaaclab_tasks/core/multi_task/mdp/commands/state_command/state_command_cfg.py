# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration for :class:`~.state_command.StateCommand`.

The command is domain-agnostic: it owns per-env lifecycle and delegates every
domain-specific concern to a ``task_table`` builder and a ``payload`` worker
(see :class:`~.state_command.StateCommand`). The env instantiates
:class:`StateCommandCfg` directly; concrete domains (factory assembly, legged
locomotion) only supply a ``task_table`` cfg and a ``payload`` cfg, each
subclassing the nested :class:`StateCommandCfg.TaskTableCfg` /
:class:`StateCommandCfg.PayloadCfg` base (which carry the resolvable
``class_type``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass

from .task_table_view import TaskTableView


@configclass
class StateCommandCfg(CommandTermCfg):
    """Configuration for the unified state command.

    Used directly: the env sets :attr:`task_table` + :attr:`payload` to concrete
    domain cfgs (subclasses of the nested bases below). The table is built first,
    then the payload given the built table.
    """

    @configclass
    class TaskTableCfg:
        """Base immutable task-table builder configuration."""

        @configclass
        class GenerateTermCfg:
            """One declaration-order candidate or target generation term."""

            class_type: Callable | str = MISSING

        @configclass
        class ObjectiveCfg:
            """One numerical term in a family's flat solve tuple."""

            class_type: Callable | str = MISSING

        @configclass
        class SolveCfg:
            """One optional solve over a flat declared objective tuple."""

            class_type: Callable | str = MISSING
            objectives: tuple[StateCommandCfg.TaskTableCfg.ObjectiveCfg, ...] = ()
            convergence_tolerance: float | None = 1.0e-6
            convergence_check_interval: int = 1

        @configclass
        class CriterionCfg:
            """One ordered post-generation acceptance criterion.

            The callable receives ``(cfg, candidates, active_rows)`` and returns
            one boolean for each active original candidate row. Criteria run in
            declaration order, so later terms inspect only earlier survivors.
            """

            class_type: Callable | str = MISSING

        @configclass
        class SelectionCfg:
            """One accepted-candidate thinning or ordering policy."""

            class_type: Callable | str = MISSING

        @configclass
        class FamilyCfg:
            """Visible generate, optional-solve, accept, and optional-select stages."""

            name: str = MISSING
            generate: tuple[StateCommandCfg.TaskTableCfg.GenerateTermCfg, ...] = ()
            solve: StateCommandCfg.TaskTableCfg.SolveCfg | None = None
            criteria: tuple[StateCommandCfg.TaskTableCfg.CriterionCfg, ...] = ()
            selection: StateCommandCfg.TaskTableCfg.SelectionCfg | None = MISSING

        class_type: Callable | str = MISSING
        """Pure builder invoked with the command cfg, resolved scene cfg, and device.

        The result exposes ``num_tasks``; its remaining typed data is consumed
        only by the matching payload, which owns ``sample_rows(count)``.
        """

        def build(self, command_cfg: StateCommandCfg, scene_cfg: object, device: str) -> Any:
            """Build one immutable table without reading a live environment."""
            return self.class_type(command_cfg, scene_cfg, device)

        def build_inspection_view(
            self,
            command_cfg: StateCommandCfg,
            scene_cfg: object,
            device: str,
            *,
            sequence_limit: int,
        ) -> TaskTableView:
            """Build the simulator-free view consumed by the shared inspector.

            Runtime tables expose their retained states by default. Domain tables
            may override this method when inspection intentionally retains more
            construction evidence than the runtime table.
            """
            del sequence_limit
            return self.build(command_cfg, scene_cfg, device).view

        seed: int = 0
        """Independent seed used by every stochastic table-construction stage."""

        families: tuple[StateCommandCfg.TaskTableCfg.FamilyCfg, ...] = ()
        """Visible generate, solve, accept, and select policies in declaration order."""

    @configclass
    class PayloadCfg:
        """Base payload worker cfg; subclass per domain."""

        class_type: type | str = MISSING
        """Payload class invoked as class_type(cfg, env, table).

        The payload binds selected rows, owns reset writes and domain frames,
        and updates the command and error tensors.
        """

    class_type: type | str = "{DIR}.state_command:StateCommand"

    reset_assets: tuple[str, ...] = MISSING
    """Ordered scene assets represented by every physical task-table state."""

    task_table: TaskTableCfg = MISSING
    """Task-table builder cfg (a :class:`TaskTableCfg` subclass)."""

    payload: PayloadCfg = MISSING
    """Payload worker cfg (a :class:`PayloadCfg` subclass)."""

    commands: dict[str, Any] = {}
    """Domain command variants (semantics interpreted by the payload)."""

    randomize_command_indices: bool = True
    """Whether command resampling samples new table rows. When ``False`` the row
    selector is driven entirely by an external curriculum binding."""

    states_relative: bool = False
    """Whether the domain payload interprets stored positions as env-local.

    The matching payload owns origin resolution for reset and target data;
    StateCommand does not interpret this field.
    """
