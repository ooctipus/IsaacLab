# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the shared episode-length scale curriculum."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.core.multi_task.mdp import EpisodeLengthScaleCurriculum

_PARAMS = {
    "sample_budget": 10_000,
    "initial_scale": 0.1,
    "scale_rate": 1.0e-5,
    "lower_threshold": 40.0,
    "upper_threshold": 42.0,
    "minimum_scale": 0.0,
    "maximum_scale": 1.0,
}


def _curriculum(episode_lengths: torch.Tensor) -> tuple[SimpleNamespace, EpisodeLengthScaleCurriculum]:
    env = SimpleNamespace(
        device=str(episode_lengths.device),
        num_envs=episode_lengths.numel(),
        episode_length_buf=episode_lengths,
    )
    cfg = CurriculumTermCfg(func=EpisodeLengthScaleCurriculum, params=_PARAMS)
    return env, EpisodeLengthScaleCurriculum(cfg, env)


def _update(
    env: SimpleNamespace,
    curriculum: EpisodeLengthScaleCurriculum,
    env_ids: slice | torch.Tensor,
) -> dict[str, torch.Tensor]:
    return curriculum(env, env_ids, **_PARAMS)


def test_penalty_curriculum_reads_completed_lengths_before_they_are_cleared() -> None:
    """A reset update must consume terminal lengths rather than post-reset zeros."""
    env, curriculum = _curriculum(torch.full((10_000,), 43, dtype=torch.long))

    state = _update(env, curriculum, slice(None))
    env.episode_length_buf.zero_()

    torch.testing.assert_close(curriculum.average_episode_length, torch.tensor(43.0))
    torch.testing.assert_close(curriculum.scale, torch.tensor(0.1 * (1.0 + 1.0e-5)))
    assert state == {
        "scale": curriculum.scale,
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
def test_penalty_curriculum_uses_strict_bfm_thresholds(episode_length: int, scale_factor: float) -> None:
    """Only averages strictly outside the 40--42 deadband change the scale."""
    env, curriculum = _curriculum(torch.full((10_000,), episode_length, dtype=torch.long))

    _update(env, curriculum, slice(None))

    torch.testing.assert_close(curriculum.average_episode_length, torch.tensor(float(episode_length)))
    torch.testing.assert_close(curriculum.scale, torch.tensor(0.1 * scale_factor))


def test_penalty_curriculum_weights_only_the_reset_batch() -> None:
    """Partial resets contribute their exact fraction of the configured sample budget."""
    episode_lengths = torch.full((10_000,), 50, dtype=torch.long)
    env, curriculum = _curriculum(episode_lengths)
    _update(env, curriculum, slice(None))

    env.episode_length_buf[2] = 10
    env.episode_length_buf[7] = 30
    _update(env, curriculum, torch.tensor((2, 7)))

    expected = 50.0 * (1.0 - 2.0 / 10_000.0) + 20.0 * (2.0 / 10_000.0)
    torch.testing.assert_close(curriculum.average_episode_length, torch.tensor(expected))


def test_penalty_curriculum_reset_preserves_device_state_and_storage() -> None:
    """Manager reset leaves the bound device tensors unchanged and address-stable."""
    env, curriculum = _curriculum(torch.full((10_000,), 43, dtype=torch.long))
    _update(env, curriculum, slice(None))
    before = (curriculum.scale.clone(), curriculum.average_episode_length.clone())
    pointers = (curriculum.scale.data_ptr(), curriculum.average_episode_length.data_ptr())

    curriculum.reset(env_ids=torch.tensor((1, 3)))

    torch.testing.assert_close(curriculum.scale, before[0])
    torch.testing.assert_close(curriculum.average_episode_length, before[1])
    assert (curriculum.scale.data_ptr(), curriculum.average_episode_length.data_ptr()) == pointers


def test_penalty_curriculum_clamps_scale_to_configured_interval() -> None:
    """Multiplicative updates cannot move the scale outside configured bounds."""
    high_env, high = _curriculum(torch.full((10_000,), 43, dtype=torch.long))
    high.scale.fill_(1.0)
    _update(high_env, high, slice(None))

    low_env, low = _curriculum(torch.full((10_000,), 39, dtype=torch.long))
    low.scale.zero_()
    _update(low_env, low, slice(None))

    torch.testing.assert_close(high.scale, torch.tensor(1.0))
    torch.testing.assert_close(low.scale, torch.tensor(0.0))
