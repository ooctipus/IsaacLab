# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed semantic axes for unified forward-backward motion learning."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from ....rl.rsl_rl import RslRlForwardBackwardRunnerCfg


@configclass
class MotionForwardBackwardRunnerCfg(RslRlForwardBackwardRunnerCfg):
    """Compose FB learning from independent robot, source, and learner choices."""

    @configclass
    class ModelTopologyCfg(PresetCfg):
        """Actor and forward-map topology selected as one model policy."""

        default = RslRlForwardBackwardRunnerCfg.ModelTopologyCfg(
            actor=RslRlForwardBackwardRunnerCfg.NetworkCfg(
                hidden_dim=1024,
                hidden_layers=2,
                embedding_layers=2,
                residual=False,
            ),
            forward=RslRlForwardBackwardRunnerCfg.NetworkCfg(
                hidden_dim=1024,
                hidden_layers=2,
                embedding_layers=2,
                residual=False,
            ),
        )
        model_plain_2x1024 = default
        model_residual_6x1024 = RslRlForwardBackwardRunnerCfg.ModelTopologyCfg(
            actor=RslRlForwardBackwardRunnerCfg.NetworkCfg(
                hidden_dim=1024,
                hidden_layers=6,
                embedding_layers=2,
                residual=True,
            ),
            forward=RslRlForwardBackwardRunnerCfg.NetworkCfg(
                hidden_dim=1024,
                hidden_layers=6,
                embedding_layers=6,
                residual=True,
            ),
        )

    @configclass
    class ScheduleCfg(PresetCfg):
        """Collection, update, and checkpoint cadence selected as one schedule."""

        default = RslRlForwardBackwardRunnerCfg.ScheduleCfg(
            num_envs=50,
            num_steps_per_env=10,
            num_updates_per_iteration=50,
            random_action_steps=50_000,
            max_iterations=10_000,
            save_interval=1_000,
        )
        schedule_50x10_5m = default
        schedule_1024x1_211p2m = RslRlForwardBackwardRunnerCfg.ScheduleCfg(
            num_envs=1024,
            num_steps_per_env=1,
            num_updates_per_iteration=16,
            random_action_steps=10_240,
            max_iterations=206_250,
            save_interval=9_375,
        )

    @configclass
    class OptimizationCfg(PresetCfg):
        """Representation and value-helper optimization selected as one policy."""

        default = RslRlForwardBackwardRunnerCfg.OptimizationCfg(
            learning_rate=1.0e-4,
            implied_value_coefficient=0.1,
        )
        optimization_lr1e4_implied0p1_actor0p01 = default
        optimization_lr3e4_implied0_actor0p05 = RslRlForwardBackwardRunnerCfg.OptimizationCfg(
            learning_rate=3.0e-4,
            implied_value_coefficient=0.0,
        )

    @configclass
    class ObservationRoutesCfg(PresetCfg):
        """Observation geometry and route names selected only by robot."""

        default: RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg = (
            RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg(
                actor=["policy"],
                forward=["policy"],
                backward=["policy"],
                discriminator=["policy"],
                critic_discriminator=["policy"],
            )
        )
        g1 = RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg(
            actor=[
                "joint_position",
                "joint_velocity",
                "projected_gravity",
                "base_angular_velocity",
                "last_action",
                "history_actor",
            ],
            forward=[
                "joint_position",
                "joint_velocity",
                "projected_gravity",
                "base_angular_velocity",
                "privileged_state",
                "last_action",
                "history_actor",
            ],
            backward=[
                "joint_position",
                "joint_velocity",
                "projected_gravity",
                "base_angular_velocity",
                "privileged_state",
            ],
            discriminator=[
                "joint_position",
                "joint_velocity",
                "projected_gravity",
                "base_angular_velocity",
                "privileged_state",
            ],
            critic_discriminator=[
                "joint_position",
                "joint_velocity",
                "projected_gravity",
                "base_angular_velocity",
                "privileged_state",
                "last_action",
                "history_actor",
            ],
        )

    @configclass
    class NormalizationGroupsCfg(PresetCfg):
        """Normalization geometry selected only by robot."""

        default: tuple[RslRlForwardBackwardRunnerCfg.ModelCfg.NormalizationGroupCfg, ...] = ()
        g1 = (
            RslRlForwardBackwardRunnerCfg.ModelCfg.NormalizationGroupCfg(
                name="state",
                fields=("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"),
            ),
        )

    @configclass
    class ExpertClockCfg(PresetCfg):
        """Source-time sampling semantics selected as one expert clock."""

        default = RslRlForwardBackwardRunnerCfg.ExpertClockCfg(
            sampling_mode="source_rows",
            sampling_step_seconds=None,
        )
        expert_clock_source_rows = default
        expert_clock_50hz = RslRlForwardBackwardRunnerCfg.ExpertClockCfg(
            sampling_mode="uniform_before_source_end",
            sampling_step_seconds=0.02,
        )

    @configclass
    class ExplorationCfg(PresetCfg):
        """Actor distribution and random warm-up range selected as one policy."""

        default = RslRlForwardBackwardRunnerCfg.ExplorationCfg(
            distribution=RslRlForwardBackwardRunnerCfg.DistributionCfg(init_std=0.2),
            random_action_range=(-1.0, 1.0),
        )
        exploration_std0p2_range1 = default
        exploration_std0p05_range5 = RslRlForwardBackwardRunnerCfg.ExplorationCfg(
            distribution=RslRlForwardBackwardRunnerCfg.DistributionCfg(init_std=0.05),
            random_action_range=(-5.0, 5.0),
        )

    @configclass
    class ValueHelpersCfg(PresetCfg):
        """Complete ordered value algebras selected only by the helper axis."""

        _DISCRIMINATOR = RslRlForwardBackwardRunnerCfg.ValueHelperCfg(
            name="discriminator",
            route="critic_discriminator",
            terms=(
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
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
                default=0.01,
                optimization_lr1e4_implied0p1_actor0p01=0.01,
                optimization_lr3e4_implied0_actor0p05=0.05,
            ),
            target_tau=0.005,
        )
        _AUXILIARY = RslRlForwardBackwardRunnerCfg.ValueHelperCfg(
            name="auxiliary",
            route="critic_discriminator",
            terms=(
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_torques",
                    coefficient=0.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_action_rate",
                    coefficient=0.1,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="limits_dof_pos",
                    coefficient=10.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="limits_torque",
                    coefficient=0.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_undesired_contact",
                    coefficient=1.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_feet_ori",
                    coefficient=0.4,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_ankle_roll",
                    coefficient=4.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
                RslRlForwardBackwardRunnerCfg.ValueTermCfg(
                    name="penalty_slippage",
                    coefficient=2.0,
                    source="stored_evidence",
                    timing="transition",
                    context_dependent=False,
                    sign=-1,
                ),
            ),
            reward_composition="scalar",
            pessimism=0.5,
            actor_coefficient=0.02,
            normalize_rewards=True,
            reward_normalization_decay=0.99,
            reward_normalization_epsilon=1.0e-8,
            target_tau=0.005,
        )

        default: list[RslRlForwardBackwardRunnerCfg.ValueHelperCfg] = [_DISCRIMINATOR]
        helpers_discriminator = default
        helpers_discriminator_auxiliary = [_DISCRIMINATOR, _AUXILIARY]

    @configclass
    class ReplayCfg(RslRlForwardBackwardRunnerCfg.ReplayCfg):
        """Typed replay contract composed from independent storage, helper, robot, and seed axes."""

        @configclass
        class PolicyCfg(PresetCfg):
            """Capacity and sampling semantics selected as one replay policy."""

            default = RslRlForwardBackwardRunnerCfg.ReplayPolicyCfg(
                capacity_transitions=2_000_000,
                terminal_capacity_per_env=384,
                sampling="transition_uniform",
            )
            replay_transition_uniform_2m = default
            replay_episode_uniform_5120k = RslRlForwardBackwardRunnerCfg.ReplayPolicyCfg(
                capacity_transitions=5_120_000,
                terminal_capacity_per_env=17,
                sampling="episode_uniform",
            )

        @configclass
        class HistoryLayoutCfg(PresetCfg):
            """Learner-owned policy history selected only by robot."""

            default: RslRlForwardBackwardRunnerCfg.HistoryLayoutCfg | None = None
            g1 = RslRlForwardBackwardRunnerCfg.HistoryLayoutCfg(
                history_field="history_actor",
                history_length=4,
                include_seed_observations=False,
                sources=[
                    RslRlForwardBackwardRunnerCfg.HistorySourceCfg(observation_name="last_action"),
                    RslRlForwardBackwardRunnerCfg.HistorySourceCfg(observation_name="base_angular_velocity"),
                    RslRlForwardBackwardRunnerCfg.HistorySourceCfg(observation_name="joint_position"),
                    RslRlForwardBackwardRunnerCfg.HistorySourceCfg(observation_name="joint_velocity"),
                    RslRlForwardBackwardRunnerCfg.HistorySourceCfg(observation_name="projected_gravity"),
                ],
            )

        policy: RslRlForwardBackwardRunnerCfg.ReplayPolicyCfg = PolicyCfg()  # type: ignore[assignment]
        autoreset_mode: Literal["disabled", "same_step", "next_step"] = "same_step"
        history_layout: RslRlForwardBackwardRunnerCfg.HistoryLayoutCfg | None = HistoryLayoutCfg()  # type: ignore[assignment]

    @configclass
    class ContextPolicyCfg(PresetCfg):
        """Expert windows and online relabeling selected as one context policy."""

        default = RslRlForwardBackwardRunnerCfg.ContextPolicyCfg(
            expert_window_lengths=(8,),
            buffer_capacity=10_000,
            refresh_steps=150,
            rollout_expert_fraction=0.0,
        )
        context_online_10k = default
        context_expert_half_8192 = RslRlForwardBackwardRunnerCfg.ContextPolicyCfg(
            expert_window_lengths=(8, 257),
            buffer_capacity=8192,
            refresh_steps=100,
            rollout_expert_fraction=0.5,
        )

    @configclass
    class LifecycleCfg(PresetCfg):
        """Optional generic tracking lifecycle selected only by tracking policy."""

        @configclass
        class ProjectionsCfg(PresetCfg):
            """Tracking metric geometry selected only by robot."""

            default: tuple[RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg.ProjectionCfg, ...] = (
                RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg.ProjectionCfg(
                    metric_name="emd",
                    target_name="policy",
                    observation_name="policy",
                    projection=(
                        "isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_humenv_tracking_pose"
                    ),
                ),
            )
            g1 = (
                RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg.ProjectionCfg(
                    metric_name="emd",
                    target_name="joint_position",
                    observation_name="joint_position_unnoised",
                ),
                RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg.ProjectionCfg(
                    metric_name="obs_state_emd",
                    target_name="joint_position",
                    observation_name="joint_position",
                    projection=(
                        "isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_observation_state_pose"
                    ),
                ),
            )

        default: RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg | None = None
        tracking_off = default
        tracking_source_edge = RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg(
            class_name=(
                "isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking:ForwardBackwardTrackingLifecycle"
            ),
            transition_interval=preset(
                default=5_000_000,
                tracking_interval_5m=5_000_000,
                tracking_interval_9p6m=9_600_000,
            ),
            command_bind="env.unwrapped.command_manager.get_term('motion')",
            sequence_ids_bind="command.table.clip_ids",
            sequence_start_rows_bind="command.table.clip_start_rows",
            sampling_priorities_bind="command.payload.sampler.clip_priorities",
            evaluation_scope_bind="command.payload.sampler.reset_sampling_scope",
            projections=ProjectionsCfg(),  # type: ignore[arg-type]
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
            evaluation_seed=0,
        )
        tracking_reset_frame = RslRlForwardBackwardRunnerCfg.TrackingLifecycleCfg(
            class_name=(
                "isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking:ForwardBackwardTrackingLifecycle"
            ),
            transition_interval=preset(
                default=5_000_000,
                tracking_interval_5m=5_000_000,
                tracking_interval_9p6m=9_600_000,
            ),
            command_bind="env.unwrapped.command_manager.get_term('motion')",
            sequence_ids_bind="command.table.clip_ids",
            sequence_start_rows_bind="command.table.clip_start_rows",
            sampling_priorities_bind="command.payload.sampler.clip_priorities",
            evaluation_scope_bind="command.payload.sampler.reset_sampling_scope",
            projections=ProjectionsCfg(),  # type: ignore[arg-type]
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
            evaluation_seed=0,
        )

    seed: int = preset(default=0, seed_0=0, seed_4728=4728)  # type: ignore[assignment]
    schedule: RslRlForwardBackwardRunnerCfg.ScheduleCfg = ScheduleCfg()  # type: ignore[assignment]
    context_policy: RslRlForwardBackwardRunnerCfg.ContextPolicyCfg = ContextPolicyCfg()  # type: ignore[assignment]
    exploration: RslRlForwardBackwardRunnerCfg.ExplorationCfg = ExplorationCfg()  # type: ignore[assignment]
    init_at_random_ep_len = False
    experiment_name = "motion_forward_backward"
    obs_groups: RslRlForwardBackwardRunnerCfg.ObservationRoutesCfg = ObservationRoutesCfg()  # type: ignore[assignment]
    model: RslRlForwardBackwardRunnerCfg.ModelCfg = RslRlForwardBackwardRunnerCfg.ModelCfg(
        topology=ModelTopologyCfg(),  # type: ignore[arg-type]
        normalization_groups=NormalizationGroupsCfg(),  # type: ignore[arg-type]
    )
    replay: RslRlForwardBackwardRunnerCfg.ReplayCfg = ReplayCfg()  # type: ignore[assignment]
    value_helpers: list[RslRlForwardBackwardRunnerCfg.ValueHelperCfg] = ValueHelpersCfg()  # type: ignore[assignment]
    expert: RslRlForwardBackwardRunnerCfg.SequenceExpertCfg = RslRlForwardBackwardRunnerCfg.SequenceExpertCfg(
        provider="isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert:forward_backward_expert_buffer",
        source_bind="env.unwrapped.command_manager.get_term('motion').table",
        clock=ExpertClockCfg(),  # type: ignore[arg-type]
        target_projection=preset(
            default="isaaclab_tasks.core.multi_task.motion.robots.smpl.observations:smpl_expert_target",
            g1="isaaclab_tasks.core.multi_task.motion.robots.g1.observations:g1_bfm_expert_target",
        ),
        target_projection_binds=preset(
            default=("env.unwrapped.scene['robot']",),
            g1=("env.unwrapped.scene['robot']", "env.unwrapped.action_manager.get_term('joint_position')"),
        ),
    )
    algorithm: RslRlForwardBackwardRunnerCfg.AlgorithmCfg = RslRlForwardBackwardRunnerCfg.AlgorithmCfg(
        batch_size=1024,
        expert_sequence_length=8,
        gamma=0.98,
        optimization=OptimizationCfg(),  # type: ignore[arg-type]
        backward_learning_rate=1.0e-5,
        discriminator_learning_rate=1.0e-5,
        optimizer="adam",
        weight_decay=0.0,
        discriminator_weight_decay=0.0,
        fb_pessimism=0.0,
        actor_pessimism=0.5,
        orthogonality_coefficient=100.0,
        implied_reward_ridge=0.0,
        discriminator_gradient_penalty_coefficient=10.0,
        context_goal_fraction=0.2,
        context_expert_fraction=0.6,
        relabel_fraction=0.8,
        fb_target_tau=0.01,
        scale_actor_helpers=True,
        max_grad_norm=None,
        rollout_expert_steps=250,
        rollout_expert_context_steps=8,
    )
    torch_compile_mode = None
    lifecycle_extension: RslRlForwardBackwardRunnerCfg.LifecycleCfg | None = LifecycleCfg()  # type: ignore[assignment]
