# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Factory video playback configurations."""

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import pytest
import torch

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
    ("cfg", "presets", "backend"),
    [
        (FactoryVideoEnvCfg(), ("nut_thread_m16", "newton_mjwarp"), "newton_rtx"),
        (FactoryVariantVideoEnvCfg(), ("newton_mjwarp",), "newton_gl"),
    ],
)
def test_factory_video_cfg_records_wide_studio_view(cfg, presets, backend) -> None:
    cfg = resolve_presets(cfg, selected=presets)
    cfg.play_mode()

    visualizer = cfg.sim.visualizer_cfgs[0]
    assert visualizer.backend.visualizer_type == backend
    if backend == "newton_rtx":
        assert visualizer.cameras[0].prim_path == cfg.scene.video_camera.prim_path
        assert cfg.scene.video_camera.prim_path.endswith("/Robot/FactoryVideoCamera")
        assert cfg.scene.video_camera.renderer_cfg == visualizer.backend.renderer_cfg
        assert cfg.scene.video_camera.offset.pos == (3.0, 0.0, 0.4)
        assert cfg.scene.video_camera.spawn.focal_length == 42.0
    else:
        assert visualizer.cameras[0].focal_length == 42.0
        assert cfg.events.video_camera.mode == "startup"
        visualizer.cameras[0].eye = (1.0, 2.0, 3.0)
        visualizer.cameras[0].lookat = (4.0, 5.0, 6.0)
        runtime = SimpleNamespace(cfg=visualizer, set_camera_view=Mock())
        position = SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 30.0]]))
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=position))},
            sim=SimpleNamespace(visualizers=[runtime]),
        )
        cfg.events.video_camera.func(env, None, **cfg.events.video_camera.params)
        runtime.set_camera_view.assert_called_once_with((11.0, 22.0, 33.0), (14.0, 25.0, 36.0))
    assert visualizer.window.size == (1280, 720)
    assert visualizer.visible_env_indices == [0]
    assert cfg.sim.render_interval == 1
    assert [recorder.source for recorder in cfg.video_recorders] == [f"visualizer:{backend}"]
    assert cfg.video_recorders[0].capture_on_render
