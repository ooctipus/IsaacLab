# Copyright (c) 2022-2024, The Octi Lab and  Isaac Lab Project Developers.
# All rights reserved.

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..assembly_data import Offset
from .commands import TaskCommand, TaskDependentCommand


@configclass
class TaskDependentCommandCfg(CommandTermCfg):
    class_type: type = TaskDependentCommand

    reset_terms_when_resample: dict[str, EventTerm] = {}


@configclass
class TaskCommandCfg(TaskDependentCommandCfg):
    class_type: type = TaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    success_threshold: float = MISSING

    held_asset_cfg: SceneEntityCfg = MISSING

    fixed_asset_cfg: SceneEntityCfg = MISSING

    held_asset_offset: Offset = Offset()

    fixed_asset_offset: Offset = Offset()
