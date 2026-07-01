# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Position-style manager environment for unified motion imitation."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .motion.config.environment import (
    MotionActionsCfg,
    MotionActionsPresetsCfg,
    MotionCommandsCfg,
    MotionCommandsPresetsCfg,
    MotionContactSensorCfg,
    MotionCurriculumCfg,
    MotionCurriculumPresetsCfg,
    MotionEventsCfg,
    MotionEventsPresetsCfg,
    MotionGroundCfg,
    MotionObservationsPresetsCfg,
    MotionRewardsCfg,
    MotionTerminationsCfg,
    MotionTerminationsPresetsCfg,
)
from .motion.config.presets import (
    MotionControlDecimationCfg,
    MotionEpisodeLengthSecondsCfg,
)
from .motion.config.robots import RobotArticulationCfg
from .motion.config.simulations import MotionSimulationPresetsCfg


@configclass
class MotionSceneCfg(InteractiveSceneCfg):
    """Flat motion scene whose physical entities are direct preset axes."""

    ground: AssetBaseCfg = MotionGroundCfg()  # type: ignore[assignment]
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = RobotArticulationCfg()  # type: ignore[assignment]
    contact_forces: SensorBaseCfg | None = MotionContactSensorCfg()  # type: ignore[assignment]


@configclass
class MotionImitationEnvCfg(ManagerBasedRLEnvCfg):
    """Shared environment whose components resolve from one broadcast preset name."""

    scene: MotionSceneCfg = MotionSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    sim: SimulationCfg = MotionSimulationPresetsCfg()  # type: ignore[assignment]
    actions: MotionActionsCfg = MotionActionsPresetsCfg()  # type: ignore[assignment]
    observations = MotionObservationsPresetsCfg()
    commands: MotionCommandsCfg = MotionCommandsPresetsCfg()  # type: ignore[assignment]
    events: MotionEventsCfg = MotionEventsPresetsCfg()  # type: ignore[assignment]
    rewards: MotionRewardsCfg = MotionRewardsCfg()
    terminations: MotionTerminationsCfg = MotionTerminationsPresetsCfg()  # type: ignore[assignment]
    curriculum: MotionCurriculumCfg = MotionCurriculumPresetsCfg()  # type: ignore[assignment]
    decimation: int = MotionControlDecimationCfg()  # type: ignore[assignment]
    episode_length_s: float = MotionEpisodeLengthSecondsCfg()  # type: ignore[assignment]
    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 3.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self) -> None:
        """Require Gymnasium Same-Step final observations for exact bootstrap."""
        self.compute_final_obs = True


__all__ = ["MotionImitationEnvCfg", "MotionSceneCfg"]
