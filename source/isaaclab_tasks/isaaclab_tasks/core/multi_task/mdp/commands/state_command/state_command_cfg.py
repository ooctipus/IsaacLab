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

        class_type: Callable | str = MISSING
        """Builder invoked as class_type(cfg, env).

        The result exposes ``num_tasks``; its remaining typed data is consumed
        only by the matching payload, which owns ``sample_rows(count)``.
        """

    @configclass
    class PayloadCfg:
        """Base payload worker cfg; subclass per domain."""

        class_type: type | str = MISSING
        """Payload class invoked as class_type(cfg, env, table).

        The payload binds selected rows, owns reset writes and domain frames,
        and updates the command and error tensors.
        """

    class_type: type | str = "{DIR}.state_command:StateCommand"

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
