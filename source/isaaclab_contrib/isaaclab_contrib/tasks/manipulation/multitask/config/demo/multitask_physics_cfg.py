# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-backend physics preset for the multitask demos."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class MultitaskPhysicsCfg(PresetCfg):
    """Physics preset selecting between PhysX and Newton backends."""

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=60,
            nconmax=80,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
