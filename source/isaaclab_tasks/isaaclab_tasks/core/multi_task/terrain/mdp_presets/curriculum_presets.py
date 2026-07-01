# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    ObservationCache,
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
    ValueShiftSamplingStrategyCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

from .. import mdp
from ..viz.sampler_images import log_spawn_goal_sampler_images


@configclass
class PositionCurriculumSamplerCfg:
    terrain_levels = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": preset(
                default=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")
                    ],
                    eps=1e-4,
                ),
                uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
                beta=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")
                    ],
                    eps=1e-4,
                ),
                value_shift=SamplerCfg(
                    strategies=[
                        ValueShiftSamplingStrategyCfg(
                            weight=0.5,
                            obs_cache_bind="materialize_state_command_observations(env, 'goal_point')",
                        )
                    ],
                    eps=1e-4,
                ),
                beta_value_shift=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates"),
                        ValueShiftSamplingStrategyCfg(
                            weight=1.0,
                            obs_cache_bind="materialize_state_command_observations(env, 'goal_point')",
                        ),
                    ],
                    eps=1e-4,
                ),
                frontier=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates"),
                        FrontierSamplingStrategyCfg(
                            k=8, dilation_steps=2, weight=0.5, success_rate_bind="success_rates"
                        ),
                    ],
                    eps=1e-4,
                ),
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class CRLSamplerCfg:
    """Terrain curriculum without reward-dependent terms."""

    terrain_levels = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
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
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class PositionSuccessorCurriculumCfg(PositionCurriculumSamplerCfg):
    """Position curriculum with a cold, expression-bound goal observation cache."""

    goal_observations = CurrTerm(
        func=ObservationCache,
        params={
            "observations_bind": "materialize_state_command_target_observations(env, 'goal_point')",
        },
    )


@configclass
class CurriculumPresetCfg(PresetCfg):
    foot_sampled_commands = PositionCurriculumSamplerCfg()
    position = foot_sampled_commands
    crl = CRLSamplerCfg()
    successor = PositionSuccessorCurriculumCfg()
    default = foot_sampled_commands
