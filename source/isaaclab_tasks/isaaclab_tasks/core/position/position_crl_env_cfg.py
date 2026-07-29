# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse-reward position-task environment for Contrastive RL (CRL).

This variant subclasses :class:`LocomotionPositionCommandEnvCfg` and strips all dense
reward shaping. CRL is fully self-supervised and learns from reached states via
Hindsight Experience Replay (HER); the per-step reward is unused. Only termination
signals (for episode boundaries) and the goal-point command (the conditioning signal)
are retained.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
)
from isaaclab_tasks.core.multi_task.mdp.curriculums import success_rate_sampler
from isaaclab_tasks.core.multi_task.terrain.viz.sampler_images import log_spawn_goal_sampler_images

from . import mdp
from .position_env_cfg import LocomotionPositionCommandEnvCfg, ObservationsEncoderCfg, ObservationsPresetCfg


@configclass
class RewardsCfg:
    """Empty rewards container.

    CRL does not consume environment rewards; the critic is trained via InfoNCE on
    goal-conditioned trajectories (see Eysenbach 2022, Bortkiewicz 2025). We keep
    an empty class to preserve the config shape expected by :class:`ManagerBasedRLEnvCfg`.
    """

    pass


@configclass
class TerminationsCfg:
    """Termination signals retained for CRL.

    - ``time_out``: truncates episodes; bounds replay-buffer trajectory length.
    - ``drop`` / ``base_contact``: safety terminations (robot fell / non-foot contact).
      Needed so HER does not relabel goals across catastrophic failures.
    - ``success``: goal-reach termination; provides a natural episode-ending signal that
      marks the commanded goal as "achieved."

    Note: ``abnormal_robot`` is intentionally omitted — it fires for numerical/physics
    anomalies which are rare and would pollute HER buffers with short garbage episodes.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    drop = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -20})

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*foot).*$"), "threshold": 1.0},
    )

    success = DoneTerm(func=mdp.success_terminate, time_out=True)


@configclass
class ObservationsCRLCfg(ObservationsEncoderCfg):
    """CRL-specialised observation layout with an absolute-pose goal slice.

    Diff from :class:`ObservationsEncoderCfg`:

    - ``task`` group emits a 3-D *absolute* target position
      (:func:`mdp.target_pos_env`, env-local frame) instead of the relative-state
      vector from :class:`RelativeStateCommand`. This gives Contrastive RL a
      goal-space matching a slice of the state.
    - New ``achieved_goal`` group emits the 3-D current robot root position in the
      same env-local frame (:func:`mdp.achieved_pos_env`). HER's relabel
      ``goal ← future_state[goal_start_idx:goal_end_idx]`` becomes correct by
      construction when ``goal_start_idx:goal_end_idx`` points at this group.
    - ``policy`` / ``height_scan`` groups are inherited unchanged.
    """

    @configclass
    class TaskCfg(ObsGroup):
        target_pos_env = ObsTerm(func=mdp.target_pos_env, params={"command_name": "goal_point"})

    @configclass
    class AchievedGoalCfg(ObsGroup):
        achieved_pos_env = ObsTerm(func=mdp.achieved_pos_env, params={"command_name": "goal_point"})

    task: TaskCfg = TaskCfg()
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()


@configclass
class ObservationsCRLPresetCfg(ObservationsPresetCfg):
    """Selectable observation layouts including the CRL variant.

    - ``default`` / ``crl``: the CRL layout (absolute-pose goal + achieved-pose slice).
    - Legacy ``flat`` / ``encoder`` / ``simba`` / ``simba_big`` presets remain
      available for baseline PPO comparisons against the same sparse env.
    """

    crl: ObservationsCRLCfg = ObservationsCRLCfg()
    default: ObservationsCRLCfg = crl


@configclass
class CurriculumCfg:
    """Curriculum retained for CRL.

    - ``terrain_levels`` is kept because it is driven by the ``success`` termination
      signal (not by a reward) and therefore still works in a sparse-reward setting.
    - ``remove_explore_reward`` is dropped: there is no ``explore`` reward term to skip.
    """

    terrain_levels = CurrTerm(
        func=success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": SamplerCfg(
                strategies=[
                    BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates")
                ],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=20),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class LocomotionPositionCRLEnvCfg(LocomotionPositionCommandEnvCfg):
    """Fully sparse-reward variant of the position task for CRL.

    Keeps all env dynamics, observations, commands, events, scene setup, and physics
    identical to :class:`LocomotionPositionCommandEnvCfg`. The only differences are:

    - ``rewards``: emptied (no dense shaping).
    - ``terminations``: only safety + time_out + success terms retained.
    - ``curriculum``: ``remove_explore_reward`` dropped.
    """

    observations: ObservationsCRLPresetCfg = ObservationsCRLPresetCfg()  # type: ignore
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
