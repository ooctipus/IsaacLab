# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic context belongs to the active physics manager instance."""

from types import SimpleNamespace

import pytest

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.physics import PhysicsManager


class _Manager(PhysicsManager):
    def reset(self, soft=False):
        pass

    def forward(self):
        pass

    def get_scene_data_backend(self):
        return None

    def step(self):
        pass


@pytest.fixture
def manager():
    return _Manager(SimpleNamespace())


def test_context_is_lazy_and_owned_by_one_manager(manager):
    state = {"value": [1, 2]}
    manager.set_debug_context_provider("episode_length", lambda: state["value"])
    assert manager.get_debug_context() == {"episode_length": [1, 2]}
    state["value"] = [3, 4]
    assert manager.get_debug_context() == {"episode_length": [3, 4]}
    assert _Manager(SimpleNamespace()).get_debug_context() == {}


@pytest.mark.parametrize("name", ["", "EpisodeLength", "episode-length", "1_episode"])
def test_context_rejects_invalid_names(manager, name):
    with pytest.raises(ValueError, match="lower snake case"):
        manager.set_debug_context_provider(name, lambda: 1)


def test_context_requires_explicit_replacement(manager):
    manager.set_debug_context_provider("step_count", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        manager.set_debug_context_provider("step_count", lambda: 2)
    manager.set_debug_context_provider("step_count", lambda: 2, replace=True)
    assert manager.get_debug_context() == {"step_count": 2}


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (lambda: None, "returned None"),
        (lambda: (_ for _ in ()).throw(RuntimeError("provider broke")), "provider broke"),
    ],
)
def test_context_provider_errors_propagate(manager, provider, message):
    manager.set_debug_context_provider("workflow_value", provider)
    with pytest.raises(RuntimeError, match=rf"workflow_value.*{message}"):
        manager.get_debug_context()


def test_context_remove_and_close_are_strict(manager):
    with pytest.raises(KeyError, match="not registered"):
        manager.remove_debug_context_provider("missing")
    manager.set_debug_context_provider("present", lambda: 1)
    manager.remove_debug_context_provider("present")
    assert manager.get_debug_context() == {}
    manager.set_debug_context_provider("present", lambda: 2)
    manager.close()
    assert manager.get_debug_context() == {}


@pytest.mark.parametrize("init_fails", [False, True])
def test_rl_context_registers_before_solver_init_and_rolls_back_on_failure(monkeypatch, manager, init_fails):
    manager.set_debug_context_provider("existing", lambda: "original")
    env = object.__new__(ManagerBasedRLEnv)
    env.sim = SimpleNamespace(physics_manager=manager)
    env.episode_length_buf = [1, 2]
    env.common_step_counter = 3
    seen = []

    def init_base(self):
        seen.append(manager.get_debug_context())
        if init_fails:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(ManagerBasedEnv, "_init_sim", init_base)
    if init_fails:
        with pytest.raises(RuntimeError, match="initialization failed"):
            env._init_sim()
    else:
        env._init_sim()

    assert seen == [{"common_step_counter": 3, "episode_length": [1, 2], "existing": "original"}]
    expected = {"existing": "original"} if init_fails else seen[0]
    assert manager.get_debug_context() == expected
    assert "_register_physics_debug_context" not in vars(ManagerBasedEnv)
