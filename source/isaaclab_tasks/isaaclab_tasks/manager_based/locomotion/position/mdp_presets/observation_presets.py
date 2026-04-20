# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from .. import mdp


@configclass
class PositionObservationsCfg:
    """Observations for the MDP (encoder variant: ``height_scan`` in its own 1D group).

    Separates the flat ``height_scan`` into a dedicated group so it can be routed through a
    per-group MLP encoder (e.g. :class:`rsl_rl.models.MLPEncoderModel`) before being fused
    with the proprioceptive ``policy`` group at the main MLP head.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TaskCfg(ObsGroup):
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()
    task: TaskCfg = TaskCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class CRLObservationsCfg:
    """CRL observations: current_state + policy + target_state.

    ``current_state`` and ``target_state`` are 12D vectors
    (pos, rot, lin_vel, ang_vel) from the command term's ``cmd_buf``.
    Position is env-local (world minus env origin); orientation and
    velocities are world-frame.  HER relabels ``target_state`` with
    achieved ``current_state`` from future timesteps.
    """

    @configclass
    class CurrentStateCfg(ObsGroup):
        """12D current world state [m, rad, m/s, rad/s]."""

        state = ObsTerm(func=mdp.command_current_state, params={"command_name": "goal_point"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations (same as PPO preset)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TargetStateCfg(ObsGroup):
        """12D target world state [m, rad, m/s, rad/s]."""

        state = ObsTerm(func=mdp.command_target_state, params={"command_name": "goal_point"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    current_state: CurrentStateCfg = CurrentStateCfg()
    policy: PolicyCfg = PolicyCfg()
    target_state: TargetStateCfg = TargetStateCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class AdvancedSkillsObservationsCfg:
    """Observations for the advanced skills MDP."""
    pass
    # TODO(Mateo)


@configclass
class ObservationsCfg(PresetCfg):
    position = PositionObservationsCfg()
    crl = CRLObservationsCfg()
    advanced_skills = AdvancedSkillsObservationsCfg()
    default = position
