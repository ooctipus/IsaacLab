# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visual-material manager terms."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import VisualMaterialCfg
from isaaclab.envs.mdp.visual_events import randomize_visual_material, randomize_visual_shape
from isaaclab.managers import EventTermCfg, SceneEntityCfg


class _Scene:
    num_envs = 4

    def __init__(self, materials):
        self.materials = dict(zip(("body", "legs"), materials, strict=True))

    def __getitem__(self, name):
        return self.materials[name]


def test_all_environment_slice_samples_one_gpu_row_per_material_and_environment() -> None:
    writes = []
    materials = [
        SimpleNamespace(
            cfg=VisualMaterialCfg(prim_path=f"/World/envs/env_0/Materials/{name}", spawn=sim_utils.PreviewSurfaceCfg()),
            is_per_env=True,
            channels=("color",),
        )
        for name in ("body", "legs")
    ]
    materials[0].write_channels = lambda materials, channels, env_ids: writes.append((materials, channels, env_ids))
    env = SimpleNamespace(
        scene=_Scene(materials),
        device="cpu",
        num_envs=4,
    )
    cfg = EventTermCfg(
        func=randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("body"), SceneEntityCfg("legs")],
            "channels": {"color": ((0.25, 0.5, 0.75), (0.25, 0.5, 0.75))},
        },
    )

    term = randomize_visual_material(cfg, env)
    term(env, slice(None), **cfg.params)

    written_materials, channels, env_ids = writes[0]
    assert written_materials == materials
    assert env_ids is None
    assert channels["color"].shape == (2, 4, 3)
    torch.testing.assert_close(channels["color"], torch.tensor([0.25, 0.5, 0.75]).expand(2, 4, 3))


def test_shape_backend_follows_only_active_render_consumers() -> None:
    cases = (
        (("newton_gl",), (), "physx", "newton"),
        ((), ("newton_warp",), "physx", "newton"),
    )
    for visualizers, renderers, physics, expected in cases:
        sim = SimpleNamespace(
            physics_manager=physics,
            visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type=name)) for name in visualizers],
            _renderer_entries=[SimpleNamespace(cfg=SimpleNamespace(renderer_type=name)) for name in renderers],
        )
        assert randomize_visual_shape._get_backend(None, SimpleNamespace(sim=sim)) == expected

    for visualizers, renderer_types in ((("kit",), ()), (("newton_gl",), ("isaac_rtx",))):
        sim = SimpleNamespace(
            visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type=name)) for name in visualizers],
            _renderer_entries=[SimpleNamespace(cfg=SimpleNamespace(renderer_type=name)) for name in renderer_types],
        )
        with pytest.raises(NotImplementedError, match="no per-shape visual storage"):
            randomize_visual_shape._get_backend(None, SimpleNamespace(sim=sim))

    selector = inspect.getsource(randomize_visual_shape._get_backend)
    assert "physics_manager" not in selector
    assert "FactoryBase._get_backend" not in selector
    assert "resolve_visualizer_types" not in selector
    assert "render_context" not in selector

    unsupported_env = SimpleNamespace(sim=SimpleNamespace(visualizers=[], _renderer_entries=[]))
    with pytest.raises(NotImplementedError, match="no per-shape visual storage"):
        randomize_visual_shape(None, unsupported_env)
