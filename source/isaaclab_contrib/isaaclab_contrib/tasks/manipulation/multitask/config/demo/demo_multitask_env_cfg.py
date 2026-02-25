# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_contrib.tasks.manipulation.multitask.multitask_env_cfg import (
    MultiRobotMultiTaskEnvCfg,
    SingleRobotMultiTaskEnvCfg,
)
from isaaclab_contrib.tasks.manipulation.multitask.multitask_utils import MultiTaskRegistryConfig


@configclass
class FrankaMultiTaskEnvCfg(SingleRobotMultiTaskEnvCfg):
    """Example single robot multi-task config using Stack/Reach/Lift/Cabinet Franka tasks."""

    def __post_init__(self):
        self.tasks = MultiTaskRegistryConfig(
            task_names_by_group=[
                "Isaac-Stack-Cube-Franka-v0",
                "Isaac-Reach-Franka-v0",
                "Isaac-Lift-Cube-Franka-v0",
                "Isaac-Open-Drawer-Franka-v0",
            ],
            group_size=10,
            device=self.sim.device,
        )

        super().__post_init__()


@configclass
class MultiRobotMultiTaskManipulationEnvCfg(MultiRobotMultiTaskEnvCfg):
    """Example multi-task config using Stack/Reach/Lift/Cabinet Franka tasks."""

    def __post_init__(self):
        self.tasks = MultiTaskRegistryConfig(
            task_names_by_group=[
                "Isaac-Stack-Cube-Franka-IK-Rel-v0",
                "Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0",
                "Isaac-Lift-Cube-Franka-IK-Rel-v0",
                "Isaac-Open-Drawer-Franka-IK-Rel-v0",
            ],
            group_size=10,
            device=self.sim.device,
        )
        super().__post_init__()
