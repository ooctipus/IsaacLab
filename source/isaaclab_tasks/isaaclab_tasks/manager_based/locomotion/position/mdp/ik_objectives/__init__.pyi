# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Re-export from kinematics.py for backward compatibility.
# Each objective will be split into its own module in a future cleanup.
from ..kinematics import IKObjectiveFootSpread as IKObjectiveFootSpread
from ..kinematics import IKObjectiveGravityTorque as IKObjectiveGravityTorque
from ..kinematics import IKObjectiveJointDefault as IKObjectiveJointDefault
from ..kinematics import IKObjectiveStabilityMargin as IKObjectiveStabilityMargin
from ..kinematics import IKObjectiveTerrainCollision as IKObjectiveTerrainCollision
from ..kinematics import IKObjectiveTerrainContact as IKObjectiveTerrainContact
