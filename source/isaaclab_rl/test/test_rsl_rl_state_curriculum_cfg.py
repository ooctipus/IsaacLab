# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RSL-RL state-curriculum configuration."""

from isaaclab_rl.rsl_rl import (
    RslRlPpoAlgorithmCfg,
    RslRlStateCurriculumCfg,
    RslRlSuccessEstimatorCfg,
    RslRlValueShiftCfg,
)


def test_state_curriculum_components_are_independently_optional():
    """Each learner signal should be enabled without requiring the other one."""
    assert RslRlStateCurriculumCfg().to_dict() == {
        "value_shift_cfg": None,
        "success_estimator_cfg": None,
    }
    assert RslRlStateCurriculumCfg(value_shift_cfg=RslRlValueShiftCfg()).success_estimator_cfg is None
    assert RslRlStateCurriculumCfg(success_estimator_cfg=RslRlSuccessEstimatorCfg()).value_shift_cfg is None


def test_state_curriculum_defaults_match_the_learner_contract():
    """Serialized field names and defaults should be accepted directly by RSL-RL."""
    cfg = RslRlStateCurriculumCfg(
        value_shift_cfg=RslRlValueShiftCfg(),
        success_estimator_cfg=RslRlSuccessEstimatorCfg(),
    )

    assert cfg.to_dict() == {
        "value_shift_cfg": {"momentum": 0.0, "evaluation_batch_size": 16384},
        "success_estimator_cfg": {
            "hidden_dims": [256, 256],
            "activation": "elu",
            "learning_rate": 1.0e-4,
            "optimizer": "adam",
            "max_grad_norm": 1.0,
        },
    }


def test_ppo_forwards_the_state_curriculum_configuration():
    """PPO serialization should preserve the nested learner configuration."""
    state_curriculum_cfg = RslRlStateCurriculumCfg(value_shift_cfg=RslRlValueShiftCfg())
    algorithm_cfg = RslRlPpoAlgorithmCfg(state_curriculum_cfg=state_curriculum_cfg)

    assert algorithm_cfg.to_dict()["state_curriculum_cfg"] == state_curriculum_cfg.to_dict()
