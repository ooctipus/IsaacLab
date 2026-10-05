# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms with explicit control targets require no scene asset."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import validate

pytestmark = pytest.mark.unit


class TargetAction(ActionTerm):
    """Write processed actions into an explicitly owned control buffer."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 2), device=env.device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._processed.copy_(2 * actions)

    def apply_actions(self):
        self._env.control.copy_(self._processed)


def test_explicit_targets_process_and_apply_without_a_scene():
    """Require no scene access when the term explicitly opts out of asset lookup."""
    env = SimpleNamespace(
        num_envs=3, device="cpu", control=torch.zeros(3, 2), sim=SimpleNamespace(is_playing=lambda: True)
    )
    cfg = ActionTermCfg(class_type=TargetAction, asset_name=None)
    validate(cfg)
    manager = ActionManager(SimpleNamespace(targets=cfg), env)
    actions = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    manager.process_action(actions)
    assert not env.control.any()
    manager.apply_action()
    torch.testing.assert_close(env.control, 2 * actions)
    assert manager.total_action_dim == 2
    assert manager.get_term("targets")._asset is None
    assert not hasattr(env, "scene")


def test_named_asset_lookup_still_requires_the_configured_asset():
    """Keep named asset resolution strict while allowing explicit target bindings."""
    asset = object()
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": asset})
    term = TargetAction(ActionTermCfg(class_type=TargetAction, asset_name="robot"), env)
    assert term._asset is asset
    with pytest.raises(KeyError, match="missing"):
        TargetAction(ActionTermCfg(class_type=TargetAction, asset_name="missing"), env)
    with pytest.raises(TypeError, match="asset_name"):
        validate(ActionTermCfg(class_type=TargetAction))
