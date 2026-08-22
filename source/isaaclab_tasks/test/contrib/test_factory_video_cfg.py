# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Factory video playback configurations."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.nist.factory_video_env_cfg import FactoryVariantVideoEnvCfg, FactoryVideoEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_factory_video_tasks_use_dedicated_environment_configs() -> None:
    for task_id, entry_point in (
        ("IsaacContrib-Factory-Video-Franka", "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVideoEnvCfg"),
        (
            "IsaacContrib-Factory-Variant-Video-Franka",
            "isaaclab_tasks.contrib.nist.factory_video_env_cfg:FactoryVariantVideoEnvCfg",
        ),
    ):
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] == entry_point


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
