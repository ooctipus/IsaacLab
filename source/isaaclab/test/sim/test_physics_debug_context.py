# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend-neutral physics diagnostic context providers."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from isaaclab.envs import manager_based_env as manager_based_env_module
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.physics import PhysicsManager


@pytest.fixture(autouse=True)
def _clear_debug_context_providers():
    PhysicsManager.clear_debug_context_providers()
    yield
    PhysicsManager.clear_debug_context_providers()


def test_debug_context_provider_resolves_current_value():
    """Providers are resolved lazily so replacing a workflow buffer stays visible."""
    state = {"value": [1, 2]}
    PhysicsManager.set_debug_context_provider("episode_length", lambda: state["value"])

    assert PhysicsManager.get_debug_context() == {"episode_length": [1, 2]}

    state["value"] = [3, 4]
    assert PhysicsManager.get_debug_context() == {"episode_length": [3, 4]}


@pytest.mark.parametrize("name", ["", "EpisodeLength", "episode-length", "1_episode"])
def test_debug_context_provider_rejects_invalid_names(name):
    """Context names have one stable lower-snake-case archive spelling."""
    with pytest.raises(ValueError, match="lower snake case"):
        PhysicsManager.set_debug_context_provider(name, lambda: 1)


def test_debug_context_provider_rejects_duplicates_unless_explicitly_replaced():
    """A duplicate registration cannot silently replace diagnostic data."""
    PhysicsManager.set_debug_context_provider("step_count", lambda: 1)

    with pytest.raises(ValueError, match="already registered"):
        PhysicsManager.set_debug_context_provider("step_count", lambda: 2)

    PhysicsManager.set_debug_context_provider("step_count", lambda: 2, replace=True)
    assert PhysicsManager.get_debug_context() == {"step_count": 2}


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (lambda: None, "returned None"),
        (lambda: (_ for _ in ()).throw(RuntimeError("provider broke")), "provider broke"),
    ],
)
def test_debug_context_provider_failures_propagate(provider, message):
    """Broken providers fail with their exact registered name and cause."""
    PhysicsManager.set_debug_context_provider("workflow_value", provider)

    with pytest.raises(RuntimeError, match=rf"workflow_value.*{message}"):
        PhysicsManager.get_debug_context()


def test_debug_context_provider_remove_is_strict():
    """Removing an unknown provider is an actionable error."""
    with pytest.raises(KeyError, match="not registered"):
        PhysicsManager.remove_debug_context_provider("missing")

    PhysicsManager.set_debug_context_provider("present", lambda: 1)
    PhysicsManager.remove_debug_context_provider("present")
    assert PhysicsManager.get_debug_context() == {}


@pytest.mark.parametrize("failure_phase", ["register", "init"])
def test_manager_env_init_failure_restores_exact_context_registry(monkeypatch, failure_phase: str):
    """Extension-mode failures restore additions and replacements without touching prior providers."""

    def original_existing():
        return "original"

    def original_unrelated():
        return "unrelated"

    PhysicsManager.set_debug_context_provider("existing", original_existing)
    PhysicsManager.set_debug_context_provider("unrelated", original_unrelated)
    registry_before = dict(PhysicsManager._debug_context_providers)
    existing_sim = SimpleNamespace(physics_manager=PhysicsManager)

    class _Cfg:
        seed = None

        def validate(self) -> None:
            pass

    class _FailingEnv(ManagerBasedEnv):
        def _register_physics_debug_context(self) -> None:
            PhysicsManager.set_debug_context_provider("existing", lambda: "replacement", replace=True)
            PhysicsManager.set_debug_context_provider("new_value", lambda: "new")
            if failure_phase == "register":
                raise RuntimeError("registration failed")

        def _init_sim(self) -> None:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", True, raising=False)
    monkeypatch.setattr(
        manager_based_env_module.SimulationContext,
        "instance",
        staticmethod(lambda: existing_sim),
    )
    monkeypatch.setattr(manager_based_env_module, "resolve_cfg_presets", lambda cfg: None)

    with pytest.raises(RuntimeError, match="failed"):
        _FailingEnv(_Cfg())

    assert PhysicsManager._debug_context_providers == registry_before
    assert PhysicsManager._debug_context_providers["existing"] is original_existing
    assert PhysicsManager._debug_context_providers["unrelated"] is original_unrelated
    assert "new_value" not in PhysicsManager._debug_context_providers


@pytest.mark.parametrize(
    ("name", "error_type"),
    [
        (None, TypeError),
        ("EpisodeLength", ValueError),
        ("episode-length", ValueError),
    ],
)
def test_debug_context_provider_remove_validates_name(name, error_type):
    """Removal applies the same stable name contract as registration."""
    with pytest.raises(error_type):
        PhysicsManager.remove_debug_context_provider(name)
