# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for unified forward-backward motion learning."""

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from ....rl.rsl_rl import (
    RslRlForwardBackwardAlgorithmCfg,
    RslRlForwardBackwardExpertCfg,
    RslRlForwardBackwardModelCfg,
    RslRlForwardBackwardReplayCfg,
    RslRlForwardBackwardRunnerCfg,
    RslRlForwardBackwardValueHelperCfg,
)


@configclass
class MotionObservationGroupsPresetCfg(PresetCfg):
    """Observation routes selected by robot geometry."""

    default = {
        "actor": ["policy"],
        "forward": ["policy"],
        "backward": ["policy"],
        "discriminator": ["policy"],
        "critic_discriminator": ["policy"],
    }
    g1 = {
        "actor": [
            "joint_position",
            "joint_velocity",
            "projected_gravity",
            "base_angular_velocity",
            "last_action",
            "history_actor",
        ],
        "forward": [
            "joint_position",
            "joint_velocity",
            "projected_gravity",
            "base_angular_velocity",
            "privileged_state",
            "last_action",
            "history_actor",
        ],
        "backward": [
            "joint_position",
            "joint_velocity",
            "projected_gravity",
            "base_angular_velocity",
            "privileged_state",
        ],
        "discriminator": [
            "joint_position",
            "joint_velocity",
            "projected_gravity",
            "base_angular_velocity",
            "privileged_state",
        ],
        "critic_discriminator": [
            "joint_position",
            "joint_velocity",
            "projected_gravity",
            "base_angular_velocity",
            "privileged_state",
            "last_action",
            "history_actor",
        ],
    }


@configclass
class MotionActorNetworkPresetCfg(PresetCfg):
    """Actor networks selected by the FB algorithm profile."""

    default = RslRlForwardBackwardModelCfg.NetworkCfg(
        hidden_dim=1024,
        hidden_layers=2,
        embedding_layers=2,
        residual=False,
    )
    model_plain_2x1024 = default
    model_residual_6x1024 = RslRlForwardBackwardModelCfg.NetworkCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=2,
        residual=True,
    )


@configclass
class MotionForwardNetworkPresetCfg(PresetCfg):
    """Forward networks selected by the FB algorithm profile."""

    default = RslRlForwardBackwardModelCfg.NetworkCfg(
        hidden_dim=1024,
        hidden_layers=2,
        embedding_layers=2,
        residual=False,
    )
    model_plain_2x1024 = default
    model_residual_6x1024 = RslRlForwardBackwardModelCfg.NetworkCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=6,
        residual=True,
    )


@configclass
class MotionDistributionPresetCfg(PresetCfg):
    """Actor distributions selected by the exploration profile."""

    default = RslRlForwardBackwardModelCfg.DistributionCfg(init_std=0.2)
    exploration_std0p2_range1 = default
    exploration_std0p05_range5 = RslRlForwardBackwardModelCfg.DistributionCfg(init_std=0.05)


@configclass
class MotionNormalizationGroupsPresetCfg(PresetCfg):
    """Shared normalization geometry selected by robot."""

    default: tuple[RslRlForwardBackwardModelCfg.NormalizationGroupCfg, ...] = ()
    g1 = (
        RslRlForwardBackwardModelCfg.NormalizationGroupCfg(
            name="state",
            fields=("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"),
        ),
    )


@configclass
class MotionForwardBackwardModelCfg(RslRlForwardBackwardModelCfg):
    """FB model with independently selected topology and robot normalization."""

    actor_cfg = MotionActorNetworkPresetCfg()
    forward_cfg = MotionForwardNetworkPresetCfg()
    distribution_cfg = MotionDistributionPresetCfg()
    normalization_groups = MotionNormalizationGroupsPresetCfg()


@configclass
class MotionHistoryLayoutPresetCfg(PresetCfg):
    """Replay history selected by robot observation semantics."""

    default: RslRlForwardBackwardReplayCfg.HistoryLayoutCfg | None = None
    g1 = RslRlForwardBackwardReplayCfg.HistoryLayoutCfg(
        history_field="history_actor",
        history_length=4,
        include_seed_observations=False,
        sources=[
            RslRlForwardBackwardReplayCfg.HistoryLayoutCfg.SourceCfg(observation_name="last_action"),
            RslRlForwardBackwardReplayCfg.HistoryLayoutCfg.SourceCfg(observation_name="base_angular_velocity"),
            RslRlForwardBackwardReplayCfg.HistoryLayoutCfg.SourceCfg(observation_name="joint_position"),
            RslRlForwardBackwardReplayCfg.HistoryLayoutCfg.SourceCfg(observation_name="joint_velocity"),
            RslRlForwardBackwardReplayCfg.HistoryLayoutCfg.SourceCfg(observation_name="projected_gravity"),
        ],
    )


@configclass
class MotionForwardBackwardReplayCfg(RslRlForwardBackwardReplayCfg):
    """Replay storage selected by storage, helper, and robot axes."""

    capacity_transitions = preset(
        default=2_000_000, replay_transition_uniform_2m=2_000_000, replay_episode_uniform_5120k=5_120_000
    )
    terminal_capacity_per_env = preset(default=384, replay_transition_uniform_2m=384, replay_episode_uniform_5120k=17)
    sampling = preset(
        default="transition_uniform",
        replay_transition_uniform_2m="transition_uniform",
        replay_episode_uniform_5120k="episode_uniform",
    )
    history_layout = MotionHistoryLayoutPresetCfg()


@configclass
class MotionForwardBackwardExpertCfg(RslRlForwardBackwardExpertCfg):
    """Expert sequence source projected through the selected robot geometry."""

    provider = "isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert:forward_backward_expert_buffer"
    source_bind = "env.unwrapped.command_manager.get_term('motion').table"
    priorities_bind = "env.unwrapped.command_manager.get_term('motion').payload.sampler.clip_priorities"
    sampling_mode = preset(
        default="source_rows", expert_clock_source_rows="source_rows", expert_clock_50hz="uniform_before_source_end"
    )
    sampling_step_seconds = preset(default=None, expert_clock_source_rows=None, expert_clock_50hz=0.02)
    target_projection = preset(
        default="isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_expert_target",
        g1="isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_expert_target",
    )
    target_projection_binds = preset(
        default=("env.unwrapped.scene['robot']",),
        g1=("env.unwrapped.scene['robot']", "env.unwrapped.action_manager.get_term('joint_position')"),
    )
    window_lengths = preset(default=(8,), context_online_10k=(8,), context_expert_half_8192=(8, 257))


_DISCRIMINATOR_HELPER = RslRlForwardBackwardValueHelperCfg(
    name="discriminator",
    route="critic_discriminator",
    terms=(
        RslRlForwardBackwardValueHelperCfg.TermCfg(
            name="discriminator",
            coefficient=1.0,
            source="recomputed",
            timing="state",
            context_dependent=True,
            sign=1,
        ),
    ),
    pessimism=0.5,
    actor_coefficient=preset(
        default=0.01, optimization_lr1e4_implied0p1_actor0p01=0.01, optimization_lr3e4_implied0_actor0p05=0.05
    ),
    target_tau=0.005,
)

_AUXILIARY_HELPER = RslRlForwardBackwardValueHelperCfg(
    name="auxiliary",
    route="critic_discriminator",
    terms=tuple(
        RslRlForwardBackwardValueHelperCfg.TermCfg(
            name=name,
            coefficient=coefficient,
            source="stored_evidence",
            timing="transition",
            context_dependent=False,
            sign=-1,
        )
        for name, coefficient in (
            ("penalty_torques", 0.0),
            ("penalty_action_rate", 0.1),
            ("limits_dof_pos", 10.0),
            ("limits_torque", 0.0),
            ("penalty_undesired_contact", 1.0),
            ("penalty_feet_ori", 0.4),
            ("penalty_ankle_roll", 4.0),
            ("penalty_slippage", 2.0),
        )
    ),
    reward_composition="scalar",
    pessimism=0.5,
    actor_coefficient=0.02,
    normalize_rewards=True,
    reward_normalization_decay=0.99,
    reward_normalization_epsilon=1.0e-8,
    target_tau=0.005,
)


@configclass
class MotionValueHelpersPresetCfg(PresetCfg):
    """Complete value algebras selected by the helper axis."""

    default = [_DISCRIMINATOR_HELPER]
    helpers_discriminator = default
    helpers_discriminator_auxiliary = [_DISCRIMINATOR_HELPER, _AUXILIARY_HELPER]


@configclass
class MotionForwardBackwardAlgorithmCfg(RslRlForwardBackwardAlgorithmCfg):
    """FB optimization with independent schedule, context, and exploration axes."""

    batch_size = 1024
    expert_sequence_length = 8
    gamma = 0.98
    learning_rate = preset(
        default=1.0e-4, optimization_lr1e4_implied0p1_actor0p01=1.0e-4, optimization_lr3e4_implied0_actor0p05=3.0e-4
    )
    backward_learning_rate = 1.0e-5
    discriminator_learning_rate = 1.0e-5
    optimizer = "adam"
    weight_decay = 0.0
    discriminator_weight_decay = 0.0
    fb_pessimism = 0.0
    actor_pessimism = 0.5
    orthogonality_coefficient = 100.0
    implied_value_coefficient = preset(
        default=0.1, optimization_lr1e4_implied0p1_actor0p01=0.1, optimization_lr3e4_implied0_actor0p05=0.0
    )
    implied_reward_ridge = 0.0
    discriminator_gradient_penalty_coefficient = 10.0
    context_goal_fraction = 0.2
    context_expert_fraction = 0.6
    relabel_fraction = 0.8
    context_buffer_capacity = preset(default=10_000, context_online_10k=10_000, context_expert_half_8192=8192)
    fb_target_tau = 0.01
    scale_actor_helpers = True
    max_grad_norm = None
    random_action_transitions = preset(default=50_000, schedule_50x10_5m=50_000, schedule_1024x1_211p2m=10_240)
    random_action_range = preset(
        default=(-1.0, 1.0), exploration_std0p2_range1=(-1.0, 1.0), exploration_std0p05_range5=(-5.0, 5.0)
    )
    rollout_context_refresh_steps = preset(default=150, context_online_10k=150, context_expert_half_8192=100)
    rollout_expert_fraction = preset(default=0.0, context_online_10k=0.0, context_expert_half_8192=0.5)
    rollout_expert_steps = 250
    rollout_expert_context_steps = 8


@configclass
class MotionTrackingCurriculumCfg:
    """Periodic sequence-tracking evaluation and sampling curriculum."""

    @configclass
    class ProjectionCfg:
        """One expert-target to reached-observation metric projection."""

        metric_name: str = MISSING
        target_name: str = MISSING
        observation_name: str = MISSING
        projection: str | None = None
        assignment_metric: str = "uniform_assignment"

    interval_transitions: int = MISSING
    command_bind: str = MISSING
    projections: tuple[ProjectionCfg, ...] = MISSING
    context_window_length: int = MISSING
    include_reset_frame: bool = MISSING
    allow_horizon_truncation: bool = MISSING
    shuffle_assignments: bool = MISSING
    priority_metric_name: str = MISSING
    priority_metric_minimum: float = MISSING
    priority_metric_maximum: float = MISSING
    priority_exponent_scale: float = MISSING
    priority_exponent_base: float = MISSING
    reset_source_name: str = MISSING
    evaluation_seed: int = 0


@configclass
class MotionTrackingProjectionsPresetCfg(PresetCfg):
    """Tracking metric geometry selected by robot."""

    default = (
        MotionTrackingCurriculumCfg.ProjectionCfg(
            metric_name="emd",
            target_name="policy",
            observation_name="policy",
            projection="isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_humenv_tracking_pose",
        ),
    )
    g1 = (
        MotionTrackingCurriculumCfg.ProjectionCfg(
            metric_name="emd",
            target_name="joint_position",
            observation_name="joint_position_unnoised",
        ),
        MotionTrackingCurriculumCfg.ProjectionCfg(
            metric_name="obs_state_emd",
            target_name="joint_position",
            observation_name="joint_position",
            projection="isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_observation_state_pose",
        ),
    )


@configclass
class MotionTrackingCurriculumPresetCfg(PresetCfg):
    """Optional tracking curricula selected independently from robot geometry."""

    default: MotionTrackingCurriculumCfg | None = None
    tracking_off = default
    tracking_source_edge = MotionTrackingCurriculumCfg(
        interval_transitions=preset(
            default=5_000_000, tracking_interval_5m=5_000_000, tracking_interval_9p6m=9_600_000
        ),
        command_bind="env.unwrapped.command_manager.get_term('motion')",
        projections=MotionTrackingProjectionsPresetCfg(),  # type: ignore[arg-type]
        context_window_length=8,
        include_reset_frame=False,
        allow_horizon_truncation=False,
        shuffle_assignments=False,
        priority_metric_name="emd",
        priority_metric_minimum=0.5,
        priority_metric_maximum=2.0,
        priority_exponent_scale=2.0,
        priority_exponent_base=2.0,
        reset_source_name="reference",
    )
    tracking_reset_frame = MotionTrackingCurriculumCfg(
        interval_transitions=preset(
            default=5_000_000, tracking_interval_5m=5_000_000, tracking_interval_9p6m=9_600_000
        ),
        command_bind="env.unwrapped.command_manager.get_term('motion')",
        projections=MotionTrackingProjectionsPresetCfg(),  # type: ignore[arg-type]
        context_window_length=1,
        include_reset_frame=True,
        allow_horizon_truncation=True,
        shuffle_assignments=True,
        priority_metric_name="emd",
        priority_metric_minimum=0.5,
        priority_metric_maximum=2.0,
        priority_exponent_scale=2.0,
        priority_exponent_base=2.0,
        reset_source_name="reference",
    )


@configclass
class MotionForwardBackwardRunnerCfg(RslRlForwardBackwardRunnerCfg):
    """Compose one motion FB learner from independent semantic axes."""

    class_type = preset(
        default="rsl_rl.runners:OffPolicyRunner",
        tracking_source_edge="isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking:ForwardBackwardTrackingRunner",
        tracking_reset_frame="isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking:ForwardBackwardTrackingRunner",
    )
    class_name = preset(
        default="OffPolicyRunner",
        tracking_source_edge="ForwardBackwardTrackingRunner",
        tracking_reset_frame="ForwardBackwardTrackingRunner",
    )
    seed = preset(default=0, seed_0=0, seed_4728=4728)
    num_envs = preset(default=50, schedule_50x10_5m=50, schedule_1024x1_211p2m=1024)
    num_steps_per_env = preset(default=10, schedule_50x10_5m=10, schedule_1024x1_211p2m=1)
    num_updates_per_iteration = preset(default=50, schedule_50x10_5m=50, schedule_1024x1_211p2m=16)
    max_iterations = preset(default=10_000, schedule_50x10_5m=10_000, schedule_1024x1_211p2m=206_250)
    save_interval = preset(default=1_000, schedule_50x10_5m=1_000, schedule_1024x1_211p2m=9_375)
    init_at_random_ep_len = False
    experiment_name = "motion_forward_backward"
    obs_groups = MotionObservationGroupsPresetCfg()
    model = MotionForwardBackwardModelCfg()
    replay = MotionForwardBackwardReplayCfg()
    expert = MotionForwardBackwardExpertCfg()
    algorithm = MotionForwardBackwardAlgorithmCfg()
    value_helpers = MotionValueHelpersPresetCfg()
    tracking_curriculum: MotionTrackingCurriculumCfg | None = MotionTrackingCurriculumPresetCfg()  # type: ignore[assignment]
