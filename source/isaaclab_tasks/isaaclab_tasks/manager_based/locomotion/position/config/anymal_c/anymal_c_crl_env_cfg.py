# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-C variant of the sparse-reward CRL position-task environment.

Mirrors :mod:`anymal_c_env_cfg` but inherits from the CRL (sparse-reward) base class.
"""

from isaaclab.utils.configclass import configclass

from ... import position_crl_env_cfg
from .anymal_c_env_cfg import AnymalCEnvMixin


@configclass
class AnymalCLocomotionPositionCRLEnvCfg(AnymalCEnvMixin, position_crl_env_cfg.LocomotionPositionCRLEnvCfg):
    """ANYmal-C + sparse-reward CRL position task.

    Uses :class:`AnymalCEnvMixin` to configure the robot asset and contact filters,
    and :class:`position_crl_env_cfg.LocomotionPositionCRLEnvCfg` for the stripped-down
    rewards/terminations/curriculum appropriate for CRL.
    """

    pass
