# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the Newton per-shape visual randomization term.

The term writes below the material binding (``model.shape_color``), so shapes sharing one
material diverge — the capability no material write can express and only Newton represents.
"""

from types import SimpleNamespace

import isaaclab_newton.envs.mdp.events as events_module
import pytest
import torch
import warp as wp
from isaaclab_newton.envs.mdp.events import randomize_visual_shape

from isaaclab.managers import SceneEntityCfg

_LABELS = [
    "/World/envs/env_0/Robot/base/visuals/mesh",  # not in body_names: untouched
    "/World/envs/env_0/Robot/LF_FOOT/visuals/mesh",
    "/World/envs/env_1/Robot/base/visuals/mesh",
    "/World/envs/env_1/Robot/LF_FOOT/visuals/mesh",
    "/World/ground",  # not under the asset: untouched
]


def _expected_srgb(colors: torch.Tensor) -> torch.Tensor:
    return torch.where(colors <= 0.0031308, 12.92 * colors, 1.055 * torch.pow(colors, 1.0 / 2.4) - 0.055)


class _Scene:
    def __init__(self, entries):
        self._entries = entries
        self.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
        self.cloner_cfg = SimpleNamespace(clone_regex="/World/envs/env_.*")

    def __getitem__(self, name):
        return self._entries[name]


@pytest.fixture
def term_env(monkeypatch):
    model = SimpleNamespace(
        shape_label=list(_LABELS),
        shape_color=wp.array([wp.vec3(0.5, 0.5, 0.5)] * len(_LABELS), dtype=wp.vec3, device="cpu"),
        device="cpu",
    )
    monkeypatch.setattr(events_module.NewtonManager, "get_model", classmethod(lambda cls: model))
    robot = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"))
    env = SimpleNamespace(scene=_Scene({"robot": robot}))
    cfg = SimpleNamespace(
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LF_FOOT"]),
            # degenerate range: deterministic red, so the scatter targets are assertable
            "channels": {"color": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))},
        }
    )
    term = randomize_visual_shape(cfg, env)
    return term, env, cfg, model


def test_shapes_diverge_below_the_material_binding(term_env):
    """Only the named body shapes recolor; the base (same asset, same materials) is untouched."""
    term, env, cfg, model = term_env

    term(env, None, **cfg.params)

    shape_colors = wp.to_torch(model.shape_color)
    red = _expected_srgb(torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(shape_colors[torch.tensor([1, 3])], red.expand(2, 3), atol=1e-6)
    # sibling shapes of the same asset (and unrelated prims) keep their model color
    assert torch.allclose(shape_colors[torch.tensor([0, 2, 4])], torch.full((3, 3), 0.5))


def test_env_ids_narrow_the_write(term_env):
    term, env, cfg, model = term_env

    term(env, torch.tensor([1]), **cfg.params)

    shape_colors = wp.to_torch(model.shape_color)
    red = _expected_srgb(torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(shape_colors[3], red, atol=1e-6)  # env 1 foot recolored
    assert torch.allclose(shape_colors[torch.tensor([0, 1, 2, 4])], torch.full((4, 3), 0.5))
