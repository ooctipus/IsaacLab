# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Production env cfg for the new :class:`MultiTaskCommand`.

Flat-terrain multi-task env for Anymal-C. Eight concurrent tasks sampled per episode:

- ``velocity`` — track body linear + angular velocity (two tracking subtasks).
- ``position`` — reach a body-position target near standing height.
- ``reach_point_in_air`` — reach an elevated body-position target.
- ``pose`` — reach a body-position AND a body-orientation target.
- ``two_feet_stand`` — tip the base to near-vertical pitch.
- ``tripod_walk`` — reach a base target while exactly 3 feet contact ground.
- ``run`` — bound gait: front pair planted + hind pair airborne OR the mirror.
- ``trot`` — trot gait: one diagonal planted + other diagonal airborne OR the mirror.

Every termination uses ``time_out=False`` (finite-horizon framing — no rsl_rl
bootstrap). Task reward is the composer's terminal ``task_reward`` ∈ ``[0, 1]``,
plus light safety penalties.

Policy observation uses three task signals: a canonical state delta split into
reach (instant) + track (tracking) tensors (per-entity ``pos / quat / lin_vel /
ang_vel`` block vectors where each active subtask's ``target - current`` writes
into its sub-slice), a per-channel active mask that disambiguates inactive
channels from active-at-target ones, and a scalar progress ∈ ``[0, 1]``. No type
bits, no id one-hots, no activation-kernel parameters leak into the obs — slot
identity is encoded positionally.
"""

from __future__ import annotations

# from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

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
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import MultiTaskCfg
from isaaclab_tasks.utils import PresetCfg

import isaaclab_assets.robots.anymal as anymal

from .terrain import mdp
from .terrain.mdp.commands.minimal_velocity_command import MinimalVelocityCommandCfg
from .terrain.mdp_presets.multitask_presets import MultiTaskTasksPresetCfg


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
    contact_forces = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class MultiTaskActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class MultiTaskCommandsCfg:
    """Command specifications.

    :attr:`goal_point` is a :class:`PresetCfg` — by default it resolves to the
    multi-task composer, but ``presets=minimal_velocity`` swaps in
    :class:`MinimalVelocityCommand` for debugging. All downstream ObsTerms /
    RewTerms / DoneTerms read common properties (``.command``, ``.task_reward``,
    ``.task_done``, ``.command_reach``, ``.command_track``, ``.command_active``,
    ``.progress``) so the swap is transparent to the rest of the env.
    """

    @configclass
    class CommandTermPresetCfg(PresetCfg):
        """Swap the ``goal_point`` command term implementation.

        ``default`` = :class:`MultiTaskCfg` — the full multi-task composer.
        ``minimal_velocity`` = :class:`MinimalVelocityCommandCfg` — a tiny
        drop-in control for debugging the composer. Mirrors the composer's
        sparse-terminal reward density exactly (``G = mean_t A_t``, emitted on
        timeout only) but replaces the spec/dispatch/kernel machinery with
        ~40 lines of direct PyTorch. If this trains and the composer on the
        same preset doesn't, the bug is inside :class:`MultiTaskCommand`.
        """

        default = MultiTaskCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            dispatch_backend="mega_kernel",
            tasks=MultiTaskTasksPresetCfg(),  # type: ignore[arg-type]
        )
        minimal_velocity = MinimalVelocityCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
            asset_cfg=SceneEntityCfg("robot"),
        )

    goal_point: CommandTermPresetCfg = CommandTermPresetCfg()  # type: ignore[assignment]


@configclass
class MultiTaskObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception — base-frame IMU-ish signals + joint state + last action.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TaskCfg(ObsGroup):
        # ``goal_reach_delta`` carries ``target - current`` for every ACTIVE
        # instant subtask (positionally encoded by the (entity, kernel) it
        # addresses); ``goal_track_delta`` does the same for tracking subtasks.
        # Splitting them eliminates the need for a per-slot "instant vs tracking"
        # flag in the obs — the semantic lives in which tensor the channel comes
        # from. ``goal_active`` is the per-channel active mask paired with
        # ``cat([goal_reach_delta, goal_track_delta], dim=-1)``: column ``i`` is
        # ``1`` iff column ``i`` of the concatenated delta is populated by a live
        # subtask of this env's task, else ``0``. It disambiguates "channel is
        # inactive for this task" from "channel is active and currently at
        # target" — both are zero in the delta, but only the former is zero in
        # the mask. ``goal_progress`` is a scalar ∈ ``[0, 1]`` encoding overall
        # closeness to this task's success band, independent of the reward-shape
        # kernel parameters.
        goal_reach_delta = ObsTerm(func=mdp.command_reach, params={"command_name": "goal_point"})
        goal_track_delta = ObsTerm(func=mdp.command_track, params={"command_name": "goal_point"})
        goal_active = ObsTerm(func=mdp.command_active, params={"command_name": "goal_point"})
        goal_progress = ObsTerm(func=mdp.command_progress, params={"command_name": "goal_point"})

    policy: PolicyCfg = PolicyCfg()
    task: TaskCfg = TaskCfg()


@configclass
class MultiTaskEventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (-0.5, 0.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class MultiTaskRewardsCfg:
    """Two reward terms only — soft-safety constraints (undesired-contact and
    mechanical-power) are folded into the composer as multiplicative
    :class:`~isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg.MultiTaskCfg.TrackingTaskCfg`
    subtasks declared with ``expose_in_obs=False``, rather than per-step shaping.

    Why no per-step ``undesired_contact`` / ``mech_work`` here: per-step
    shaping breaks the bootstrap contract (V accumulates expected future
    safety penalties → distorts ``γ·V(s_T)`` at reach-truncate) and can
    invert success preference when the penalty exceeds the gain. The
    multiplicative form ``G · ∏ safety_factor_k`` discounts the terminal
    smoothly while keeping ``G ∈ [0, 1]`` — see the design rationale in
    :func:`~isaaclab_tasks.core.multi_task.mdp.commands.reward_composer.multiplicative_terminal_reward`.
    The two soft-safety subtasks (``UNDESIRED_CONTACT_SAFETY``,
    ``MECH_POWER_SAFETY`` in :mod:`.terrain.tasks_cfg`) are attached to
    every task via :func:`.terrain.mdp_presets.multitask_presets._with_safety`.

    Remaining terms:

    - :attr:`task` — composer's terminal ``G ∈ [0, 1]``. Reward-manager
      ``dt`` scaling is cancelled in :meth:`__post_init__`, so
      ``weight=1.0`` means "max 1.0 reward delivered per episode."
    - :attr:`early_termination` — penalty for failure-mode terminations
      (``drop``, ``base_contact``). ``is_terminated_term`` filters out
      ``time_out=True`` events; we list keys explicitly so ``success`` and
      pure-tracking timeout don't count.
    """

    task = RewTerm(func=mdp.command_task_reward, weight=1.0, params={"command_name": "goal_point"})
    early_termination = RewTerm(
        func=mdp.is_terminated_term, weight=-1.0, params={"term_keys": ["drop", "base_contact"]}
    )


@configclass
class MultiTaskTerminationsCfg:
    """Timeout flag is per-task-type — see the two ``time_out_*`` terms.

    ``success`` fires on all-instants-achieved (real termination, no bootstrap).
    ``time_out_reach_truncate`` / ``time_out_track_terminate`` split the
    max-episode-length event by task type:

    - Reach / mixed tasks (≥1 instant subtask): timeout is a **truncation**
      (``time_out=True``). rsl_rl bootstraps ``γ·V(s_T)``, propagating value
      through partial progress even when reach didn't complete.
    - Pure-tracking tasks (no instant subtask): timeout is the **natural end**
      (``time_out=False``). Composer's ``G = transit_mean`` is the complete
      episodic return; bootstrap would double-count.

    Failure terminations (``drop``, ``base_contact``) stay ``time_out=False``
    — real terminations regardless of task type.
    """

    time_out_reach_truncate = DoneTerm(
        func=mdp.time_out_reach_truncate, time_out=True, params={"command_name": "goal_point"}
    )
    time_out_track_terminate = DoneTerm(
        func=mdp.time_out_track_terminate, time_out=False, params={"command_name": "goal_point"}
    )
    success = DoneTerm(func=mdp.command_task_done, time_out=False, params={"command_name": "goal_point"})
    drop = DoneTerm(func=mdp.root_height_below_minimum, time_out=False, params={"minimum_height": -2.0})
    # Chassis-on-ground is a failure: without this, the policy can learn to
    # rest the base on the terrain and "track" a stationary goal trivially.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        time_out=False,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


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
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # ``reward_manager.compute`` scales each term by step_dt. The composer's
        # terminal reward ∈ [0, 1] is an atomic event, not a rate — cancel the
        # dt-scaling so the delivered reward matches the composer's contract
        # multiplied by the user-set ``weight`` (which stays meaningful as
        # "how much the task reward dominates the per-episode return").
        self.rewards.task.weight *= 1.0 / (self.sim.dt * self.decimation)
        self.rewards.early_termination.weight *= 1.0 / (self.sim.dt * self.decimation)
