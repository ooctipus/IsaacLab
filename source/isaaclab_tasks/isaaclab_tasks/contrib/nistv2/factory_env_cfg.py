# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL environment for assembling the complete NIST board."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.factory_env_cfg import (
    NEWTON_SOLVER_RESET_REWARD,
    NEWTON_SOLVER_RESET_TERMINATION,
    FactoryEnvCfg,
)
from isaaclab_tasks.contrib.nist.factory_presets import JointEffortNamesCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .board_layout import NUM_ASSEMBLIES
from .factory_scene_cfg import FactoryBoardSceneCfg
from .reset_env_cfg import FactoryBoardEventCfg


@configclass
class FactoryBoardObservationsCfg:
    """NIST policy state with all assembly-pair poses and no point cloud."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        prev_action = ObsTerm(func=mdp.last_action)
        assembly_frames_in_robot_root_frame = ObsTerm(func=mdp.assembly_frames_in_robot_root_frame)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class FactoryBoardRewardsCfg:
    """Penalize control effort and reward complete-board assembly."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    success_reward = RewTerm(func=mdp.assembly_success_reward, weight=100.0)
    solver_reset_reward = NEWTON_SOLVER_RESET_REWARD


@configclass
class FactoryBoardTerminationsCfg:
    """Terminate on complete-board success or invalid board state."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    oob = DoneTerm(func=mdp.any_held_asset_out_of_bound)
    progress_context = DoneTerm(func=mdp.assembly_progress_context)
    abnormal = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
    )
    success = DoneTerm(func=mdp.success_termination)
    solver_reset_required = NEWTON_SOLVER_RESET_TERMINATION


@configclass
class FactoryBoardCurriculumCfg:
    """Record terminal outcomes before reset replaces episode metadata."""

    metrics = CurrTerm(func=mdp.BoardMetrics)


@configclass
class FactoryBoardPhysicsCfg(PresetCfg):
    """Newton MJWarp preset sized for the complete board."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=3200,
            nconmax=2400,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            ls_parallel=False,
            use_mujoco_contacts=False,
            enable_sleeping=True,
            nvmax=129,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            include_static_kinematic_pairs=False,
            max_triangle_pairs=90_000_000,
            contact_reduction_hashtable_size_factor=0.04,
            rigid_contact_max=5_000_000,
            speculative_config=NewtonCollisionPipelineCfg.SpeculativeContactCfg(max_speculative_extension=0.01),
            sdf_all_shapes=NewtonCollisionPipelineCfg.SDFAllShapesCfg(
                sdf_max_resolution=256,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
            ),
        ),
        num_substeps=8,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default = newton_mjwarp


@configclass
class FactoryBoardEnvCfg(FactoryEnvCfg):
    """Factory task with every NIST assembly present in each environment."""

    scene: FactoryBoardSceneCfg = FactoryBoardSceneCfg(num_envs=1024)
    observations: FactoryBoardObservationsCfg = FactoryBoardObservationsCfg()
    events: FactoryBoardEventCfg = FactoryBoardEventCfg()
    curriculum: FactoryBoardCurriculumCfg = FactoryBoardCurriculumCfg()
    terminations: FactoryBoardTerminationsCfg = FactoryBoardTerminationsCfg()
    rewards: FactoryBoardRewardsCfg = FactoryBoardRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 14.0 * NUM_ASSEMBLIES
        self.sim.render_interval = self.decimation
        self.sim.physics = FactoryBoardPhysicsCfg()

    def play_mode(self) -> None:
        """Use a small reset bank for interactive evaluation."""
        self.scene.num_envs = 16
        reset = self.events.reset_board.params
        reset["state_table_size"] = 512
        reset["fallen_state_table_size"] = 512
