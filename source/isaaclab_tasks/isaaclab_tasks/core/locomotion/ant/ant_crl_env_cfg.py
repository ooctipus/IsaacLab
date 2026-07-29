# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned CRL variant of the Ant environment.

Strips all dense rewards from :class:`AntEnvCfg` and adds goal-conditioned
observations (``achieved_goal`` + ``task``) for Contrastive RL with HER.

The observation layout (groups sorted alphabetically, matching CRL convention):

- ``achieved_goal`` (2): current root XY position in env-local frame [m].
- ``policy`` (~60): full proprioceptive state (same as base Ant).
- ``task`` (2): commanded goal XY position in env-local frame [m].

HER's ``goal_start_idx:goal_end_idx`` resolves to the ``achieved_goal`` group
(index 0:2 in the flat vector), which is in the same coordinate space as the
``task`` goal — enabling valid relabeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .ant_env_cfg import AntEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Simple XY goal command
# ---------------------------------------------------------------------------


def _root_xyz_env_local(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root XYZ position in env-local frame [m], shape ``[num_envs, 3]``."""
    import warp as wp

    asset = env.scene[asset_cfg.name]
    pos_w = wp.to_torch(asset.data.root_pos_w)
    return (pos_w - env.scene.env_origins)[:, :3]


class XYZGoalCommand(CommandTerm):
    """Samples a random XY goal + target Z height."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.goal_xyz = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The commanded goal XYZ, shape ``[num_envs, 3]``."""
        return self.goal_xyz

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        n = env_ids.numel()
        angle = torch.empty(n, device=self.device).uniform_(0, 2 * 3.14159265)
        dist = self.cfg.goal_dist
        self.goal_xyz[env_ids, 0] = dist * torch.cos(angle)
        self.goal_xyz[env_ids, 1] = dist * torch.sin(angle)
        self.goal_xyz[env_ids, 2] = self.cfg.target_z

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class XYZGoalCommandCfg(CommandTermCfg):
    """Configuration for random XY goal + target Z height."""

    class_type: type = XYZGoalCommand

    asset_name: str = "robot"
    """Name of the robot asset."""

    goal_dist: float = 10.0
    """Fixed distance from origin for goal placement [m]."""

    target_z: float = 0.55
    """Target root height [m]. Normal upright ant is ~0.55."""


# ---------------------------------------------------------------------------
# CRL env config
# ---------------------------------------------------------------------------


@configclass
class CRLRewardsCfg:
    """Empty rewards — CRL is self-supervised."""

    pass


@configclass
class CRLTerminationsCfg:
    """Terminate on timeout, falling, or flipping over."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 2.0})


def _qpos_no_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Generalized positions: root_xy(2) + root_z(1) + root_quat_xyzw(4) + joint_pos(8) = 15."""
    import warp as wp

    asset = env.scene[asset_cfg.name]
    root_pos = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    root_quat = wp.to_torch(asset.data.root_quat_w)
    joint_pos = wp.to_torch(asset.data.joint_pos)
    return torch.cat([root_pos[:, :2], root_pos[:, 2:3], root_quat, joint_pos], dim=-1)


def _qvel_no_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Generalized velocities: root_lin_vel(3) + root_ang_vel(3) + joint_vel(8) = 14."""
    import warp as wp

    asset = env.scene[asset_cfg.name]
    root_vel = wp.to_torch(asset.data.root_vel_w)  # [N, 6] lin(3) + ang(3)
    joint_vel = wp.to_torch(asset.data.joint_vel)  # [N, 8]
    return torch.cat([root_vel, joint_vel], dim=-1)


@configclass
class CRLObservationsCfg:
    """CRL observations with 3D goal (XY position + Z height).

    - ``current_state(3)``: root XYZ position [m]. Used by HER for relabeling.
    - ``policy(29)``: qpos(15) + qvel(14).
    - ``target_state(3)``: commanded goal (goal_x, goal_y, target_z).
    """

    @configclass
    class CurrentStateCfg(ObsGroup):
        """Root XYZ position [m], shape ``[num_envs, 3]``."""

        root_xyz = ObsTerm(func=_root_xyz_env_local)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """qpos(15) + qvel(14) = 29 dims."""

        qpos = ObsTerm(func=_qpos_no_target)
        qvel = ObsTerm(func=_qvel_no_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TargetStateCfg(ObsGroup):
        goal_xyz = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_xyz"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    current_state: CurrentStateCfg = CurrentStateCfg()
    policy: PolicyCfg = PolicyCfg()
    target_state: TargetStateCfg = TargetStateCfg()


@configclass
class CRLCommandsCfg:
    """Random XY goal + target Z height."""

    goal_xyz = XYZGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        goal_dist=10.0,
        target_z=0.55,
    )


@configclass
class CRLActionsCfg:
    """Action scale tuned to match Brax's trajectory diversity (her_dist ~1.0)."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


@configclass
class AntCRLEnvCfg(AntEnvCfg):
    """Goal-conditioned CRL variant of the Ant environment.

    Inherits scene, physics, events from :class:`AntEnvCfg`.
    Replaces rewards (empty), observations (3-group CRL layout), commands (random XY goal),
    terminations (safety only), and action scale (matched to Brax gear=150).

    Timing is overridden to match Brax Spring backend:
    Brax uses dt=0.005 × n_frames=10 = 0.05s/step, 1000 steps = 50s episode.
    IsaacLab uses dt=1/120 ≈ 0.00833s, so decimation=6 gives 0.05s/step,
    and episode_length_s=50 gives 1000 steps — matching Brax exactly.
    """

    observations: CRLObservationsCfg = CRLObservationsCfg()  # type: ignore
    actions: CRLActionsCfg = CRLActionsCfg()
    rewards: CRLRewardsCfg = CRLRewardsCfg()
    terminations: CRLTerminationsCfg = CRLTerminationsCfg()
    commands: CRLCommandsCfg = CRLCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match Brax Spring: dt=0.005, n_frames=10 → 0.05s/step, 1000 steps = 50s.
        self.sim.dt = 0.005
        self.decimation = 10
        self.episode_length_s = 50.0
