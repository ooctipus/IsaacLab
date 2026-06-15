# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from .. import mdp


@configclass
class PositionObservationsCfg:
    """Observations for the MDP (encoder variant: ``height_scan`` in its own 1D group).

    Separates the flat ``height_scan`` into a dedicated group so it can be routed through a
    per-group MLP encoder before being fused with the proprioceptive ``policy``
    group at the main MLP head.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Body-frame gravity vector with magnitude preserved [m/s^2]. Pairs with
        # ``proj_gravity`` (unit direction) so the policy can also condition on
        # ``||g||`` under per-env gravity randomization.
        gravity_b = ObsTerm(func=mdp.gravity_b, noise=Unoise(n_min=-0.5, n_max=0.5))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TaskCfg(ObsGroup):
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.vision_obs,
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

    ``current_state`` and ``target_state`` are ``12 + 3 * num_feet`` vectors
    composed of root pose/vel (pos, rot, lin_vel, ang_vel) plus per-foot
    positions, all sourced from the command term. Position and foot
    positions are env-local (world minus env origin); orientation and
    velocities are world-frame. The foot section matches the success
    criterion so HER relabels ``target_state`` with achieved ``current_state``
    from future timesteps without drifting from the reward signal.
    """

    @configclass
    class CurrentStateCfg(ObsGroup):
        """Current state: root pose/vel + foot positions [m, rad, m/s, rad/s]."""

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
        """Target state: root pose/vel + foot positions [m, rad, m/s, rad/s]."""

        state = ObsTerm(func=mdp.command_target_state, params={"command_name": "goal_point"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        # Same CNN-shaped ``(B, 1, H, W)`` height scan as the PPO preset; the
        # CRL actor and critic flatten + run a configurable per-group MLP
        # encoder before the residual MLP via ``encoder_cfg`` on the runner
        # config (see :class:`PositionLocomotionCRLRunnerCfg`), so the two
        # policies consume the same observation term.
        height_scan = ObsTerm(
            func=mdp.vision_obs,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

    current_state: CurrentStateCfg = CurrentStateCfg()
    policy: PolicyCfg = PolicyCfg()
    target_state: TargetStateCfg = TargetStateCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class ObservationsCfg(PresetCfg):
    position = PositionObservationsCfg()
    crl = CRLObservationsCfg()
    default = position
