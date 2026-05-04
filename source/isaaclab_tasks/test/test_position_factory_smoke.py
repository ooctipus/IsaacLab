# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: ``Isaac-Position-v0`` and ``Isaac-Factory-v0`` env cfgs construct.

Used as a fast integration safety net before / after structural
refactors of the multi_task subpackage. Mirrors what the hydra pipeline
does up to (but not including) Kit initialisation:

1. Triggers gym task registration via ``import isaaclab_tasks``.
2. Resolves ``env_cfg_entry_point`` from the gym registry.
3. Instantiates the env cfg class.

If any of the above breaks (broken import path, missing module after a
move, cfg construction error), this test catches it without paying the
cost of the heavier ``test_env_cfg_no_forbidden_imports.py`` (which
spawns a subprocess and runs the full hydra preset resolver).
"""

from __future__ import annotations

import importlib

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401 -- registers the gym tasks


@pytest.mark.parametrize("task_name", ["Isaac-Position-v0", "Isaac-Factory-v0"])
def test_env_cfg_constructs(task_name: str) -> None:
    """The env cfg referenced by ``env_cfg_entry_point`` imports + constructs."""
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    # Standard manager-based env cfg fields. If a structural move left a
    # field unresolved (e.g. curriculum cfg pointing at a missing module),
    # construction would have raised before this assertion.
    assert hasattr(cfg, "scene")
    assert hasattr(cfg, "actions")
    assert hasattr(cfg, "observations")
    assert hasattr(cfg, "events")
