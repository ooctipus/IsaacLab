# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Factory video playback configurations."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANT_NAMES
from isaaclab_tasks.contrib.nist.factory_video_env import FactoryVariantVideoEnv
from isaaclab_tasks.contrib.nist.factory_video_env_cfg import FactoryVariantVideoEnvCfg, FactoryVideoEnvCfg
from isaaclab_tasks.contrib.nist.utils.variant_event_combinators import variant_reset_accumulator
from isaaclab_tasks.utils.hydra import resolve_presets


def test_factory_video_tasks_use_dedicated_environment_configs() -> None:
    for task_id, env_entry_point, cfg_entry_point in (
        (
            "IsaacContrib-Factory-Video-Franka",
            "isaaclab.envs:ManagerBasedRLEnv",
            "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVideoEnvCfg",
        ),
        (
            "IsaacContrib-Factory-Variant-Video-Franka",
            "isaaclab_tasks.contrib.nist.factory_video_env:FactoryVariantVideoEnv",
            "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVariantVideoEnvCfg",
        ),
    ):
        spec = gym.spec(task_id)
        assert spec.entry_point == env_entry_point
        assert spec.kwargs["env_cfg_entry_point"] == cfg_entry_point


@pytest.mark.parametrize(
    ("cfg", "presets"),
    [
        (FactoryVideoEnvCfg(), ("nut_thread_m16", "newton_mjwarp")),
        (FactoryVariantVideoEnvCfg(), ("newton_mjwarp",)),
    ],
)
def test_factory_video_cfg_records_wide_studio_view(cfg, presets) -> None:
    cfg = resolve_presets(cfg, selected=presets)
    cfg.play_mode()

    visualizer = cfg.sim.visualizer_cfgs[0]
    assert visualizer.visualizer_type == "newton_rtx"
    assert visualizer.camera_target_prim_path.endswith("/Robot")
    assert visualizer.camera_target_env_index == 0
    assert visualizer.visible_env_indices == [0]
    assert cfg.sim.render_interval == 1
    assert [recorder.source for recorder in cfg.video_recorders] == ["visualizer:newton_rtx"]
    assert cfg.video_recorders[0].capture_on_render


def test_factory_variant_video_records_one_successful_episode_per_asset() -> None:
    cfg = resolve_presets(FactoryVariantVideoEnvCfg(), selected=("newton_mjwarp",))
    cfg.play_mode()
    episode_steps = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

    assert cfg.events.reset_strategies.func == (
        "isaaclab_tasks.contrib.nist.factory_video_env:SuccessSequencedVariantReset"
    )
    assert cfg.video_recorders[0].video_length == 10 * len(ASSEMBLY_VARIANT_NAMES) * episode_steps


def test_factory_variant_video_advances_only_after_visible_success(monkeypatch: pytest.MonkeyPatch) -> None:
    env = FactoryVariantVideoEnv.__new__(FactoryVariantVideoEnv)
    env._is_closed = True
    sequence = SimpleNamespace(precollecting_phase=False, variant_index=0, _num_variants=3)
    env._video_sequence = sequence
    env._video_complete = False
    env._video_attempt_start_step = 0
    env._video_segments = []
    sequence.precollecting_phase = False
    success = torch.tensor([False])
    env.common_step_counter = 10
    env.termination_manager = SimpleNamespace(get_term=lambda _: success)
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", lambda *args, **kwargs: None)

    env._reset_idx(torch.tensor([0]))
    assert sequence.variant_index == 0
    assert env._video_attempt_start_step == 10

    for step in (20, 30, 40):
        env.common_step_counter = step
        success[0] = True
        env._reset_idx(torch.tensor([0]))

    assert env._video_segments == [(0, 10, 20), (1, 20, 30), (2, 30, 40)]
    assert env._video_complete


def test_factory_variant_video_sampler_keeps_state_weights_within_the_scheduled_asset() -> None:
    accumulator = variant_reset_accumulator.__new__(variant_reset_accumulator)
    accumulator.state_cell_indices = torch.arange(6)
    accumulator._num_variants = 3
    accumulator.cell_probabilities = torch.empty((2, 3))
    base_probabilities = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 12.0])
    base_probabilities /= base_probabilities.sum()
    sampled: dict[str, torch.Tensor] = {}

    def sample(probabilities: torch.Tensor, count: int) -> torch.Tensor:
        sampled["probabilities"] = probabilities.clone()
        return torch.multinomial(probabilities, count, replacement=True)

    accumulator._sampler = SimpleNamespace(probabilities=lambda: base_probabilities.clone(), sample=sample)
    probabilities, slots = accumulator._sample_variant(16, variant_id=1)

    assert torch.all(accumulator.state_cell_indices[slots].remainder(3) == 1)
    assert torch.count_nonzero(probabilities).item() == 2
    torch.testing.assert_close(probabilities[1] / probabilities[4], torch.tensor(0.25))
    torch.testing.assert_close(probabilities, sampled["probabilities"])
