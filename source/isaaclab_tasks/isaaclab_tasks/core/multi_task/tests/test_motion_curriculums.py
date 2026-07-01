# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for motion-imitation curriculum terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.core.multi_task.motion.mdp.curriculums import MotionPenaltyScaleCurriculum


def _curriculum(episode_lengths: torch.Tensor) -> tuple[SimpleNamespace, MotionPenaltyScaleCurriculum]:
    env = SimpleNamespace(
        device=str(episode_lengths.device),
        num_envs=episode_lengths.numel(),
        episode_length_buf=episode_lengths,
    )
    cfg = CurriculumTermCfg(func=MotionPenaltyScaleCurriculum)
    return env, MotionPenaltyScaleCurriculum(cfg, env)


def test_penalty_curriculum_reads_completed_lengths_before_they_are_cleared() -> None:
    """A reset update must consume terminal lengths rather than post-reset zeros."""
    env, curriculum = _curriculum(torch.full((10_000,), 43, dtype=torch.long))

    state = curriculum(env, slice(None))
    env.episode_length_buf.zero_()

    assert curriculum.average_episode_length == pytest.approx(43.0)
    assert curriculum.scale == pytest.approx(0.1 * (1.0 + 1.0e-5))
    assert state == {
        "penalty_scale": curriculum.scale,
        "average_episode_length": curriculum.average_episode_length,
    }


@pytest.mark.parametrize(
    ("episode_length", "scale_factor"),
    (
        (39, 1.0 - 1.0e-5),
        (40, 1.0),
        (42, 1.0),
        (43, 1.0 + 1.0e-5),
    ),
)
def test_penalty_curriculum_uses_strict_released_thresholds(episode_length: int, scale_factor: float) -> None:
    """Only averages strictly outside the 40--42 deadband change the scale."""
    env, curriculum = _curriculum(torch.full((10_000,), episode_length, dtype=torch.long))

    curriculum(env, slice(None))

    assert curriculum.average_episode_length == pytest.approx(float(episode_length))
    assert curriculum.scale == pytest.approx(0.1 * scale_factor)


def test_penalty_curriculum_weights_only_the_reset_batch() -> None:
    """Partial resets must contribute their exact fraction of the 10,000-sample budget."""
    episode_lengths = torch.full((10_000,), 50, dtype=torch.long)
    env, curriculum = _curriculum(episode_lengths)
    curriculum(env, slice(None))

    env.episode_length_buf[2] = 10
    env.episode_length_buf[7] = 30
    curriculum(env, torch.tensor((2, 7)))

    expected = 50.0 * (1.0 - 2.0 / 10_000.0) + 20.0 * (2.0 / 10_000.0)
    assert curriculum.average_episode_length == pytest.approx(expected)


def test_penalty_curriculum_reset_preserves_state_and_properties_are_read_only() -> None:
    """Manager reset must preserve learned state, which callers can only read."""
    env, curriculum = _curriculum(torch.full((10_000,), 43, dtype=torch.long))
    curriculum(env, slice(None))
    before = (curriculum.scale, curriculum.average_episode_length)

    curriculum.reset(env_ids=torch.tensor((1, 3)))

    assert (curriculum.scale, curriculum.average_episode_length) == before
    with pytest.raises(AttributeError):
        curriculum.scale = 0.5
    with pytest.raises(AttributeError):
        curriculum.average_episode_length = 0.0


def test_penalty_curriculum_clamps_scale_to_unit_interval() -> None:
    """Multiplicative updates must not move the scale outside the released bounds."""
    high_env, high = _curriculum(torch.full((10_000,), 43, dtype=torch.long))
    high._scale = 1.0
    high(high_env, slice(None))

    low_env, low = _curriculum(torch.full((10_000,), 39, dtype=torch.long))
    low._scale = 0.0
    low(low_env, slice(None))

    assert high.scale == 1.0
    assert low.scale == 0.0
