# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for environment managers.

The managers are used to handle various aspects of the environment such as randomization events, curriculum,
and observations. Each manager implements a specific functionality for the environment. The managers are
designed to be modular and can be easily extended to support new functionality.
"""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    # Term cfg classes — pure dataclasses, no heavy imports
    ("manager_term_cfg", [
        "ActionTermCfg",
        "CommandTermCfg",
        "CurriculumTermCfg",
        "EventTermCfg",
        "ManagerTermBaseCfg",
        "ObservationGroupCfg",
        "ObservationTermCfg",
        "RecorderTermCfg",
        "RewardTermCfg",
        "TerminationTermCfg",
    ]),
    # Manager implementations — deferred
    ("manager_base", ["ManagerBase", "ManagerTermBase"]),
    ("action_manager", ["ActionManager", "ActionTerm"]),
    ("command_manager", ["CommandManager", "CommandTerm"]),
    ("curriculum_manager", "CurriculumManager"),
    ("event_manager", "EventManager"),
    ("observation_manager", "ObservationManager"),
    ("recorder_manager", ["DatasetExportMode", "RecorderManager", "RecorderManagerBaseCfg", "RecorderTerm"]),
    ("reward_manager", "RewardManager"),
    ("scene_entity_cfg", "SceneEntityCfg"),
    ("termination_manager", "TerminationManager"),
)
