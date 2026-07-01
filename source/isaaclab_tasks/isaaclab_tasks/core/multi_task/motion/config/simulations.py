# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct simulator preset axis for motion reproduction profiles."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .presets import G1_LAFAN_PROFILE_CFG, SMPL_CMU_PROFILE_CFG

SMPL_CMU_SIMULATION_CFG = SimulationCfg(
    dt=SMPL_CMU_PROFILE_CFG.timing.physics_dt,
    render_interval=SMPL_CMU_PROFILE_CFG.timing.control_decimation,
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            use_mujoco_contacts=True,
            enable_multiccd=False,
            enable_native_ccd=False,
            tolerance=1.0e-8,
        ),
        num_substeps=1,
    ),
    use_newton_actuators=False,
)
"""Native MuJoCo Warp simulator for SMPL-CMU reproduction."""


G1_SIMULATION_CFG = SimulationCfg(
    dt=G1_LAFAN_PROFILE_CFG.timing.physics_dt,
    render_interval=G1_LAFAN_PROFILE_CFG.timing.control_decimation,
    physics=PhysxCfg(
        solver_type=1,
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_patch_count=5 * 2**20,
        gpu_found_lost_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**25,
    ),
)
"""Released PhysX simulator shared by G1 source compositions."""


@configclass
class MotionSimulationPresetsCfg(PresetCfg):
    """Full SimulationCfg selected by the shared motion preset name."""

    default: SimulationCfg = SMPL_CMU_SIMULATION_CFG
    smpl_cmu: SimulationCfg = SMPL_CMU_SIMULATION_CFG
    g1_lafan: SimulationCfg = G1_SIMULATION_CFG
    g1_cmu: SimulationCfg = G1_SIMULATION_CFG


__all__ = [
    "G1_SIMULATION_CFG",
    "SMPL_CMU_SIMULATION_CFG",
    "MotionSimulationPresetsCfg",
]
