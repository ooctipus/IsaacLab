# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the octibenchmark nvtx_hooks module.

Uses mock objects to verify that NVTX hooks are correctly installed
without requiring a real IsaacLab environment or GPU.

Run with::

    python -m pytest scripts/octibenchmark/test_nvtx_hooks.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class _MockManager:
    """Minimal mock for an IsaacLab manager (reward, termination, etc.)."""

    def __init__(self):
        self._call_log = []

    def compute(self, **kwargs):
        self._call_log.append("compute")
        return 0.0

    def apply(self, **kwargs):
        self._call_log.append("apply")

    def process_action(self, action):
        self._call_log.append("process_action")

    def apply_action(self):
        self._call_log.append("apply_action")

    def reset(self, env_ids=None):
        self._call_log.append("reset")

    def record_pre_step(self):
        self._call_log.append("record_pre_step")

    def record_post_step(self):
        self._call_log.append("record_post_step")


class _MockSim:
    def step(self, render=False):
        pass

    def render(self):
        pass


class _MockScene:
    def write_data_to_sim(self):
        pass

    def update(self, dt=0.0):
        pass


class _MockDirectEnv:
    """Minimal mock for a DirectRLEnv."""

    def __init__(self):
        self.sim = _MockSim()
        self.scene = _MockScene()
        self._step_count = 0

    def step(self, actions):
        self._step_count += 1
        return {}, 0, False, False, {}

    def _reset_idx(self, env_ids):
        pass

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_rewards(self):
        return 0.0

    def _get_dones(self):
        return False, False

    def _get_observations(self):
        return {}


class _MockManagerBasedEnv:
    """Minimal mock for a ManagerBasedRLEnv."""

    def __init__(self):
        self.sim = _MockSim()
        self.scene = _MockScene()
        self.action_manager = _MockManager()
        self.termination_manager = _MockManager()
        self.reward_manager = _MockManager()
        self.observation_manager = _MockManager()
        self.command_manager = _MockManager()
        self.event_manager = _MockManager()
        self.recorder_manager = _MockManager()
        self.curriculum_manager = _MockManager()
        self._step_count = 0

    def step(self, actions):
        self._step_count += 1
        return {}, 0, False, False, {}

    def _reset_idx(self, env_ids):
        pass


@patch("octibenchmark.nvtx_hooks.nvtx")
class TestInstallNvtxHooksDirectEnv:
    """Tests for installing NVTX hooks on a DirectRLEnv-like object."""

    def test_wraps_step(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockDirectEnv()
        original_step = env.step
        install_nvtx_hooks(env)
        # step should be wrapped
        assert env.step is not original_step

    def test_wraps_direct_methods(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockDirectEnv()
        install_nvtx_hooks(env)
        # Call step and verify NVTX range_push/pop are called
        env.step(None)
        push_calls = [c[0][0] for c in mock_nvtx.range_push.call_args_list]
        assert "env.step" in push_calls

    def test_direct_env_methods_wrapped(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockDirectEnv()
        install_nvtx_hooks(env)
        # Call direct env methods
        env._pre_physics_step(None)
        env._get_rewards()
        env._get_dones()
        env._get_observations()
        push_calls = [c[0][0] for c in mock_nvtx.range_push.call_args_list]
        assert "direct.pre_physics_step" in push_calls
        assert "direct.get_rewards" in push_calls
        assert "direct.get_dones" in push_calls
        assert "direct.get_observations" in push_calls

    def test_sim_scene_wrapped(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockDirectEnv()
        install_nvtx_hooks(env)
        env.sim.step()
        env.sim.render()
        env.scene.write_data_to_sim()
        env.scene.update()
        push_calls = [c[0][0] for c in mock_nvtx.range_push.call_args_list]
        assert "sim.step" in push_calls
        assert "sim.render" in push_calls
        assert "scene.write_data_to_sim" in push_calls
        assert "scene.update" in push_calls


@patch("octibenchmark.nvtx_hooks.nvtx")
class TestInstallNvtxHooksManagerEnv:
    """Tests for installing NVTX hooks on a ManagerBasedRLEnv-like object."""

    def test_wraps_managers(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockManagerBasedEnv()
        install_nvtx_hooks(env)
        # Call manager methods and verify NVTX labels
        env.reward_manager.compute()
        env.termination_manager.compute()
        env.observation_manager.compute()
        env.command_manager.compute()
        push_calls = [c[0][0] for c in mock_nvtx.range_push.call_args_list]
        assert "reward.compute" in push_calls
        assert "termination.compute" in push_calls
        assert "observation.compute" in push_calls
        assert "command.compute" in push_calls

    def test_does_not_wrap_direct_on_manager_env(self, mock_nvtx):
        """Manager-based envs should NOT get direct.* hooks."""
        from octibenchmark.nvtx_hooks import install_nvtx_hooks

        env = _MockManagerBasedEnv()
        install_nvtx_hooks(env)
        # Manager env has action_manager so direct hooks should be skipped
        assert not hasattr(env, "_pre_physics_step")


@patch("octibenchmark.nvtx_hooks.nvtx")
class TestNvtxCallableProxy:
    """Tests for the _NvtxCallableProxy helper."""

    def test_proxy_preserves_return_value(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import _NvtxCallableProxy

        def my_func(x):
            return x * 2

        proxy = _NvtxCallableProxy("test_label", my_func)
        assert proxy(5) == 10

    def test_proxy_emits_nvtx(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import _NvtxCallableProxy

        proxy = _NvtxCallableProxy("my_label", lambda: None)
        proxy()
        mock_nvtx.range_push.assert_called_with("my_label")
        mock_nvtx.range_pop.assert_called()

    def test_proxy_forwards_attributes(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import _NvtxCallableProxy

        class MyCallable:
            some_attr = 42

            def __call__(self):
                pass

            def reset(self):
                return "reset_called"

        orig = MyCallable()
        proxy = _NvtxCallableProxy("label", orig)
        assert proxy.some_attr == 42
        assert proxy.reset() == "reset_called"

    def test_proxy_pops_on_exception(self, mock_nvtx):
        from octibenchmark.nvtx_hooks import _NvtxCallableProxy

        def bad_func():
            raise ValueError("test error")

        proxy = _NvtxCallableProxy("err_label", bad_func)
        with pytest.raises(ValueError, match="test error"):
            proxy()
        # range_pop should still be called (finally block)
        mock_nvtx.range_pop.assert_called()
