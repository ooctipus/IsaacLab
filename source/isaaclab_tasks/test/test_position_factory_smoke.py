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
4. Round-trips the cfg through ``cfg.to_dict()`` (the same path hydra
   feeds to ``OmegaConf.create`` in :func:`register_task`). Catches
   serialisation issues that show up only when training launches --
   e.g. nested cfgs whose annotations OmegaConf rejects, or
   ``class_type`` fields not registered as ``ResolvableString``.

If any of the above breaks (broken import path, missing module after a
move, cfg construction error, OmegaConf-incompatible annotation), this
test catches it without paying the cost of the heavier
``test_env_cfg_no_forbidden_imports.py``.
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


@pytest.mark.parametrize("task_name", ["Isaac-Position-v0", "Isaac-Factory-v0"])
def test_env_cfg_to_dict_serialises(task_name: str) -> None:
    """``cfg.to_dict()`` produces a fully-flattened dict with no dataclass instances.

    Hydra's :func:`register_task` feeds ``cfg.to_dict()`` into
    ``OmegaConf.create``. If any nested dataclass cfg uses an
    annotation OmegaConf can't validate (e.g. ``type[X]`` without ``| str``,
    or untyped containers leaving dataclass instances exposed), the
    training launch crashes during ``register_task``. This test checks
    that the dict is OmegaConf-clean by asserting no dataclass instances
    leak through.
    """
    spec = gym.spec(task_name)
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    cfg_dict = cfg.to_dict()

    def _assert_no_dataclasses(value, path: str = "") -> None:
        if hasattr(value, "__dataclass_fields__"):
            raise AssertionError(
                f"Unflattened dataclass {type(value).__name__} at {path!r} -- "
                "this would crash OmegaConf during hydra's register_task."
            )
        if isinstance(value, dict):
            for k, v in value.items():
                _assert_no_dataclasses(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _assert_no_dataclasses(v, f"{path}[{i}]")

    _assert_no_dataclasses(cfg_dict)


@pytest.mark.parametrize(
    "module_path",
    [
        # Pure-Python leaf modules that the restructure relocated. Importing
        # them catches stale ``from .X import Y`` references where ``X`` is
        # now a sibling-package or has been renamed. Modules that pull in
        # Kit / Newton / USD must NOT be added here -- they segfault when
        # imported outside a launched Kit app.
        "isaaclab_tasks.manager_based.multi_task.curriculum.sampling",
        "isaaclab_tasks.manager_based.multi_task.curriculum.sampling.sampler",
        "isaaclab_tasks.manager_based.multi_task.curriculum.sampling.sampler_cfg",
        "isaaclab_tasks.manager_based.multi_task.curriculum.sampling.sampling_strategies",
        "isaaclab_tasks.manager_based.multi_task.curriculum.sampling.sampling_strategies_cfg",
        "isaaclab_tasks.manager_based.multi_task.curriculum.state_layout",
        "isaaclab_tasks.manager_based.multi_task.curriculum.state_buffer",
        "isaaclab_tasks.manager_based.multi_task.curriculum.success_monitor",
        "isaaclab_tasks.manager_based.multi_task.curriculum.reset_state",
        "isaaclab_tasks.manager_based.multi_task.terrain.terrains.patch_sampling.cfg",
        "isaaclab_tasks.manager_based.multi_task.terrain.terrains.patch_sampling.morph",
        "isaaclab_tasks.manager_based.multi_task.terrain.terrains.patch_sampling.rejection",
        "isaaclab_tasks.manager_based.multi_task.terrain.retarget.cfg",
        "isaaclab_tasks.manager_based.multi_task.terrain.retarget.criteria_cfg",
        "isaaclab_tasks.manager_based.multi_task.factory.assembly_profile",
        "isaaclab_tasks.manager_based.multi_task.factory.assembly_profile_cfg",
    ],
)
def test_relocated_module_imports(module_path: str) -> None:
    """Modules touched by the directory restructure import without errors.

    A targeted version of "import everything" -- limited to pure-Python
    leaf modules so the test doesn't pull in Kit/Newton-bound modules
    that segfault outside a launched Kit app. Catches stale relative
    imports that the restructure left behind (e.g.
    ``from .reset_state import X`` after ``reset_state.py`` moved
    sibling).
    """
    importlib.import_module(module_path)
