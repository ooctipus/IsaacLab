# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-C robot preset. Activate with ``presets=anymal_c``."""

from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.utils import preset

import isaaclab_assets.robots.anymal as anymal

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetLateralHipJointPatternCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

# TorchScript ANYdrive LSTM checkpoint used by Newton-native actuator authoring.
# This Newton revision loads neural actuators through the PyTorch checkpoint
# loader; the authoring step adds metadata that matches Isaac Lab's LSTM input
# convention.
ANYDRIVE_3_LSTM_JIT_PATH = str(Path(__file__).parent / "assets" / "anydrive_3_lstm.pt")

_ANYMAL_C_CFG: ArticulationCfg = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_ANYMAL_C_CFG.spawn.usd_path = (  # type: ignore[attr-defined]
    "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
)
_ANYMAL_C_CFG.spawn.joint_drive_props = preset(
    implicit_actuator=sim_utils.JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=40.0,
        damping=5.0,
        max_force=120.0,
        max_joint_velocity=7.5,
    ),
    default=None,
    lstm_actuator=None,
)

ANYDRIVE_3_LSTM_ACTUATOR_CFG = anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG.replace(
    network_file=preset(
        default=anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG.network_file,
        newton_mjwarp=ANYDRIVE_3_LSTM_JIT_PATH,
    )
)
ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit_sim=80.0,
    velocity_limit_sim=7.5,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
    armature={".*": 0.15},
)
_ANYMAL_C_CFG.actuators["legs"] = preset(
    implicit_actuator=ANYDRIVE_3_SIMPLE_ACTUATOR_CFG,
    default=ANYDRIVE_3_LSTM_ACTUATOR_CFG,
    lstm_actuator=ANYDRIVE_3_LSTM_ACTUATOR_CFG,
)

RobotArticulationCfg.anymal_c = _ANYMAL_C_CFG
HeightScannerPrimPathCfg.anymal_c = "{ENV_REGEX_NS}/Robot/base"
BaseBodyNameCfg.anymal_c = "base"
NonFootContactBodyNamesCfg.anymal_c = "base"
FootBodyNamesCfg.anymal_c = ".*FOOT.*"
AsyncFootPairsCfg.anymal_c = (
    ("LF_FOOT", "RF_FOOT"),
    ("RH_FOOT", "LH_FOOT"),
    ("LF_FOOT", "LH_FOOT"),
    ("RF_FOOT", "RH_FOOT"),
)
SyncFootPairsCfg.anymal_c = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))
ExperimentNameCfg.anymal_c = "anymal_c_position_command"


# ---------------------------------------------------------------------------
# Retarget validation criteria for ANYmal-C
# ---------------------------------------------------------------------------

ANYMAL_C_LATERAL_HIP_PATTERN = ".*HAA"
"""Regex matching ANYmal-C lateral hip joint names."""

RetargetLateralHipJointPatternCfg.anymal_c = ANYMAL_C_LATERAL_HIP_PATTERN
