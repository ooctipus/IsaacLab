# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Production env cfg for the new :class:`MultiTaskCommand`.

Flat-terrain multi-task env for Anymal-C. Five concurrent tasks sampled per episode:

- ``velocity`` — track body linear + angular velocity (two tracking subtasks).
- ``position`` — reach a body-position target near standing height (one instant subtask).
- ``reach_point_in_air`` — reach an elevated body-position target (one instant subtask).
- ``pose`` — reach a body-position AND a body-orientation target (two instant subtasks).
- ``two_feet_stand`` — tip the base to near-vertical pitch (one instant subtask).

Every termination uses ``time_out=False`` (finite-horizon framing — no rsl_rl
bootstrap). Task reward is the composer's terminal ``task_reward`` ∈ ``[0, 1]``,
plus light safety penalties. Observations follow the existing position pattern
(proprio + generated-commands).

Deferred to later stages: retarget pipeline integration (``terrain_*`` tasks),
robot-preset substitution, body-frame state kernels for policy observations.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_assets.robots.anymal as anymal

from . import mdp
from .mdp.commands.kernels import ACTIVATION_KERNEL_ID, METRIC_KERNEL_ID, SAMPLER_KERNEL_ID, STATE_KERNEL_ID
from .mdp.commands.multi_task_cfg import MinMaxSampler, MultiTaskCfg


def command_task_reward(env, command_name: str = "goal_point"):
    """Expose :attr:`MultiTaskCommand.task_reward` (terminal-only) as a reward term."""
    return env.command_manager.get_term(command_name).task_reward


def command_task_done(env, command_name: str = "goal_point"):
    """Expose :attr:`MultiTaskCommand.task_done` as a DoneTerm (success-terminate flag)."""
    return env.command_manager.get_term(command_name).task_done


# ---------------------------------------------------------------------------
# Per-slot observation feeds — activation-as-obs + structural bits.
#
# These replace the "raw parameter" anti-pattern (passing ``std`` / ``threshold``
# directly to the policy). The activation value is the engine's task-normalized
# "how close am I?" signal in ``[0, 1]``; it's the same scale across every
# activation kind the framework supports. Structural bits (``is_instant`` /
# ``is_tracking`` / ``valid``) tell the policy how to use that signal —
# "achieve and relax" vs "hold every step" vs "ignore this slot entirely."
# ---------------------------------------------------------------------------


def command_slot_activation(env, command_name: str = "goal_point"):
    """Per-slot activation score ∈ [0, 1], shape ``[num_envs, k_max]``."""
    return env.command_manager.get_term(command_name).slot_activation


def command_slot_is_instant(env, command_name: str = "goal_point"):
    """Per-slot "must-achieve" flag (float cast of bool), shape ``[num_envs, k_max]``."""
    return env.command_manager.get_term(command_name).slot_is_instant.float()


def command_slot_is_tracking(env, command_name: str = "goal_point"):
    """Per-slot "must-maintain" flag (float cast of bool), shape ``[num_envs, k_max]``."""
    return env.command_manager.get_term(command_name).slot_is_tracking.float()


def command_slot_valid(env, command_name: str = "goal_point"):
    """Per-slot active-in-task flag (float cast of bool), shape ``[num_envs, k_max]``."""
    return env.command_manager.get_term(command_name).slot_valid.float()


@configclass
class MultiTaskSceneCfg(InteractiveSceneCfg):
    """Flat plane + Anymal-C + dome light + contact sensor."""

    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.75, 0.75, 0.75)),
    )
    robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class MultiTaskActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


# --- Sampler/subtask helpers ---
# Standing height for Anymal-C, used as the nominal base z for position/pose targets.
_STANDING_Z = (0.4, 0.7)

_BASE_ENTITY = SceneEntityCfg("robot", body_names="base")


def _lin_vel_tracking() -> MultiTaskCfg.TrackingTaskCfg:
    return MultiTaskCfg.TrackingTaskCfg(
        asset_cfg=_BASE_ENTITY,
        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
        activation_kernel_param=0.3,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
            minimum=[-1.0, -1.0, 0.0],
            maximum=[1.0, 1.0, 0.0],
        ),
    )


def _ang_vel_tracking() -> MultiTaskCfg.TrackingTaskCfg:
    return MultiTaskCfg.TrackingTaskCfg(
        asset_cfg=_BASE_ENTITY,
        state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
        activation_kernel_param=0.3,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
            minimum=[0.0, 0.0, -1.5],
            maximum=[0.0, 0.0, 1.5],
        ),
    )


def _body_pos_instant(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    threshold: float,
) -> MultiTaskCfg.InstantaneousTaskCfg:
    return MultiTaskCfg.InstantaneousTaskCfg(
        asset_cfg=_BASE_ENTITY,
        state_kernel=int(STATE_KERNEL_ID.BODY_POS),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
        activation_kernel_param=threshold,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
            minimum=[x_range[0], y_range[0], z_range[0]],
            maximum=[x_range[1], y_range[1], z_range[1]],
        ),
    )


def _body_quat_instant(
    roll_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_range: tuple[float, float],
    threshold: float,
) -> MultiTaskCfg.InstantaneousTaskCfg:
    return MultiTaskCfg.InstantaneousTaskCfg(
        asset_cfg=_BASE_ENTITY,
        state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
        metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
        activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
        activation_kernel_param=threshold,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
            minimum=[roll_range[0], pitch_range[0], yaw_range[0]],
            maximum=[roll_range[1], pitch_range[1], yaw_range[1]],
            out_dim=4,  # quaternion is 4-D; out_dim pads the param tensor accordingly.
        ),
    )


def _foot_z_instant(
    foot_name: str,
    z_range: tuple[float, float],
    threshold: float,
) -> MultiTaskCfg.InstantaneousTaskCfg:
    """Foot height target (z only) — used for tripod-stand and ground-contact tasks."""
    return MultiTaskCfg.InstantaneousTaskCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=foot_name),
        state_kernel=int(STATE_KERNEL_ID.BODY_POS_Z),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
        activation_kernel_param=threshold,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
            minimum=[z_range[0]],
            maximum=[z_range[1]],
        ),
    )


def _base_speed_tracking(speed_range: tuple[float, float]) -> MultiTaskCfg.TrackingTaskCfg:
    """Scalar base-speed tracking — target ``||v||`` without pinning direction."""
    return MultiTaskCfg.TrackingTaskCfg(
        asset_cfg=_BASE_ENTITY,
        state_kernel=int(STATE_KERNEL_ID.BODY_LIN_SPEED),
        metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
        activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
        activation_kernel_param=0.3,
        sampler=MinMaxSampler(
            kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
            minimum=[speed_range[0]],
            maximum=[speed_range[1]],
        ),
    )


@configclass
class MultiTaskCommandsCfg:
    """Five task kinds for flat-terrain training.

    - ``velocity``: track a body linear- AND angular-velocity target (two tracking subtasks).
    - ``position``: reach a point on/near the ground (one instant subtask).
    - ``reach_point_in_air``: reach an elevated point (one instant subtask, z range lifted).
    - ``pose``: reach a position AND an orientation (two instant subtasks).
    - ``two_feet_stand``: tip the base to a near-vertical orientation (one instant subtask).

    The command term samples uniformly across the five tasks at reset. The composer
    produces a multiplicative terminal reward ``G ∈ [0, 1]`` per env per episode.
    """

    goal_point = MultiTaskCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        tasks={
            "velocity": [_lin_vel_tracking(), _ang_vel_tracking()],
            "position": [
                _body_pos_instant(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), z_range=_STANDING_Z, threshold=0.2)
            ],
            "reach_point_in_air": [
                _body_pos_instant(x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), z_range=(1.0, 1.5), threshold=0.25),
            ],
            "pose": [
                _body_pos_instant(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), z_range=_STANDING_Z, threshold=0.2),
                _body_quat_instant(
                    roll_range=(0.0, 0.0),
                    pitch_range=(0.0, 0.0),
                    yaw_range=(-3.14, 3.14),
                    threshold=0.3,
                ),
            ],
            "two_feet_stand": [
                _body_quat_instant(
                    # Pitch ≈ -π/2 tips Anymal-C back onto its rear feet under the
                    # x-forward, z-up convention (axis-angle about +y is negative pitch).
                    roll_range=(0.0, 0.0),
                    pitch_range=(-1.7, -1.3),
                    yaw_range=(-3.14, 3.14),
                    threshold=0.3,
                ),
            ],
            # --------------------------------------------------------------
            # Compound tasks that exercise the composer's AND-gate and the
            # transit-scoped tracking path.
            # --------------------------------------------------------------
            "reach_while_tripod": [
                # Base position target — reach a point on flat ground.
                _body_pos_instant(
                    x_range=(-2.0, 2.0),
                    y_range=(-2.0, 2.0),
                    z_range=_STANDING_Z,
                    threshold=0.3,
                ),
                # Three feet on the ground (z ≈ 0) and the fourth (RH) lifted.
                # Foot z only — xy is unconstrained, letting the policy keep the feet
                # at their natural stance relative to the base.
                _foot_z_instant("LF_FOOT", z_range=(0.0, 0.04), threshold=0.08),
                _foot_z_instant("RF_FOOT", z_range=(0.0, 0.04), threshold=0.08),
                _foot_z_instant("LH_FOOT", z_range=(0.0, 0.04), threshold=0.08),
                _foot_z_instant("RH_FOOT", z_range=(0.15, 0.25), threshold=0.08),
            ],
            "reach_at_target_speed": [
                # Base position instant — "get to this point".
                _body_pos_instant(
                    x_range=(-3.0, 3.0),
                    y_range=(-3.0, 3.0),
                    z_range=_STANDING_Z,
                    threshold=0.25,
                ),
                # Speed-magnitude tracking during transit — "at roughly this pace".
                # The direction is free; geometry of goal-vs-current dictates heading.
                _base_speed_tracking(speed_range=(0.2, 1.5)),
            ],
        },
    )


@configclass
class MultiTaskObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception — base-frame base IMU-ish signals + joint state + last action.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

        # Task signal — uniform across tasks.
        # ``goal_delta`` gives direction (what to aim for). The per-slot activation +
        # structural bits tell the policy how to use that direction: how close it is
        # to "good enough," whether this slot is a one-time reach or sustained hold,
        # and whether the slot is even active in this env's assigned task.
        # Crucially: no activation-kernel parameters (``std`` / ``threshold``) leak
        # into the observation — the activation value itself is the task-normalized
        # "how am I doing" signal, independent of the reward-shape's internals.
        goal_delta = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})
        goal_activation = ObsTerm(func=command_slot_activation, params={"command_name": "goal_point"})
        goal_is_instant = ObsTerm(func=command_slot_is_instant, params={"command_name": "goal_point"})
        goal_is_tracking = ObsTerm(func=command_slot_is_tracking, params={"command_name": "goal_point"})
        goal_slot_valid = ObsTerm(func=command_slot_valid, params={"command_name": "goal_point"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class MultiTaskEventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (-0.5, 0.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class MultiTaskRewardsCfg:
    # Task reward: terminal G ∈ [0, 1] from the composer.
    task = RewTerm(func=command_task_reward, weight=1.0, params={"command_name": "goal_point"})
    # Light safety penalties. Weights intentionally modest — the task reward should dominate.
    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.0001)
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*FOOT).*$"), "threshold": 1.0},
    )


@configclass
class MultiTaskTerminationsCfg:
    """Every termination flagged ``time_out=False`` per the finite-horizon framing.

    ``success`` fires on all-instants-achieved. For pure-tracking tasks (no instants)
    this is trivially False, so these envs only terminate at ``time_out`` or ``drop``.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=False)
    success = DoneTerm(func=command_task_done, time_out=False, params={"command_name": "goal_point"})
    drop = DoneTerm(func=mdp.root_height_below_minimum, time_out=False, params={"minimum_height": -2.0})


@configclass
class MultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: MultiTaskSceneCfg = MultiTaskSceneCfg(num_envs=4096, env_spacing=2.5)
    sim: SimulationCfg = SimulationCfg()
    observations: MultiTaskObservationsCfg = MultiTaskObservationsCfg()
    actions: MultiTaskActionsCfg = MultiTaskActionsCfg()
    commands: MultiTaskCommandsCfg = MultiTaskCommandsCfg()
    rewards: MultiTaskRewardsCfg = MultiTaskRewardsCfg()
    terminations: MultiTaskTerminationsCfg = MultiTaskTerminationsCfg()
    events: MultiTaskEventsCfg = MultiTaskEventsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(3.0, 3.0, 2.0), origin_type="asset_body", asset_name="robot", body_name="base")

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
