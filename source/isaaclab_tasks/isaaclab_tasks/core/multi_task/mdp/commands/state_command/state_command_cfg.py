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
        """Base task-table builder cfg; subclass per domain."""

        class_type: Callable | str = MISSING
        """Builder (or resolvable string) invoked as ``class_type(cfg, env)``;
        must yield a table exposing ``num_tasks`` and ``gather(task_rows)``."""

    @configclass
    class PayloadCfg:
        """Base payload worker cfg; subclass per domain."""

        class_type: type | str = MISSING
        """Payload worker class (or resolvable string) invoked as
        ``class_type(cfg, env, table)``; must implement the payload protocol the
        command delegates to."""

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
    """Whether stored table states are expressed in each env's local frame.

    When ``True`` the command writes spawn states with
    :func:`~...curriculum.set_reset_state` ``is_relative=True`` (per-asset
    ``env_origins`` added) and lifts the payload target by ``env_origins``. When
    ``False`` the states are already world-frame (e.g. a single shared terrain)
    and no origin offset is applied. This replaces runtime terrain-replication
    sniffing: the deploying scene declares its frame explicitly."""
