from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .command import AssemblyTaskCommand
if TYPE_CHECKING:
    from .data_cfg import TaskDataCfg

@configclass
class AssemblyTaskCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = AssemblyTaskCommand

    tasks: list[str] = MISSING  # type: ignore

    reset_terms_when_resample: dict[str, EventTerm] = {}

    alignment_metric_cfg: TaskDataCfg = MISSING  # type: ignore
