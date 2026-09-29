# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact native keyboard populations at the real shared-MDP episode boundary."""

import ast
import gc
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import newton
import numpy as np
import pytest
import torch
import warp as wp


def test_population_task_ownership_boundaries():
    import isaaclab_tasks.contrib.keyboard as keyboard

    directory = Path(keyboard.__file__).parent
    root = ast.parse((directory / "so101_population_env.py").read_text())
    env = next(node for node in root.body if isinstance(node, ast.ClassDef))
    assert [ast.unparse(base) for base in env.bases] == ["gym.Env"]
    # The composition root requests owner-provided successor previews; it never copies their state schema.
    for node in ast.walk(root):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "_group_obs_term_history_buffer",
                "_group_obs_term_delay_buffer",
                "_prev_pressed",
                "_just_reset",
            }
    for filename in ("so101_population_env.py", "keyboard_populations.py", "newton_selection.py"):
        tree = ast.parse((directory / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "gc" not in {alias.name for alias in node.names}
                assert not isinstance(node, ast.ImportFrom) or node.module != "gc"
            if isinstance(node, ast.ImportFrom):
                assert not {alias.name for alias in node.names} & {
                    "NewtonManager",
                    "ManagerBasedRLEnv",
                    "ArticulationView",
                }
            if isinstance(node, ast.Assign):
                assert not any(isinstance(target, ast.Attribute) and target.attr == "scene" for target in node.targets)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"ModelBuilder", "register_custom_attributes"}
    for path in (directory / "mdp").rglob("*.py"):
        assert "NewtonManager" not in path.read_text()
    for forbidden in ("population_mdp", "population_selection.py", "population_scene.py", "population_asset.py"):
        assert not (directory / forbidden).exists()


@pytest.mark.parametrize("fail_stop,fail_clear", [(False, False), (True, False), (False, True), (True, True)])
def test_close_releases_native_bindings_held_by_runtime_configuration(fail_stop, fail_clear):
    from isaaclab_tasks.contrib.keyboard.keyboard_populations import KeyboardPopulations
    from isaaclab_tasks.contrib.keyboard.newton_selection import (
        BODY,
        NewtonSelectionGroup,
        NewtonSelections,
        NewtonSelectorCfg,
    )
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    source = NewtonSelections(builder.finalize("cpu"))
    selector = NewtonSelectorCfg(BODY, path=".*", count_per_world=1)
    source.resolve(selector)
    source_ref, source_model_ref = weakref.ref(source), weakref.ref(source.model)
    model = source.model.replicate(1)
    state = model.state()
    model_ref, state_ref = weakref.ref(model), weakref.ref(state)
    owner = NewtonSelections(model, state=state, source=source)
    binding = NewtonSelectionGroup(selector, ((owner.resolve(selector), torch.tensor([0])),), 1)
    env = SO101KeyboardPopulationEnv.__new__(SO101KeyboardPopulationEnv)
    env._is_closed = False
    calls = []
    stop_error, clear_error = RuntimeError("stop callback failed"), RuntimeError("clear callback failed")

    def stop():
        calls.append("stop")
        if fail_stop:
            raise stop_error

    def clear():
        calls.append("clear")
        if fail_clear:
            raise clear_error

    env.sim = SimpleNamespace(stop=stop, clear_instance=clear)
    env.cfg = SimpleNamespace(commands=SimpleNamespace(body=binding))
    env.obs_buf, env.native_contacts = {}, {}
    bank = KeyboardPopulations.__new__(KeyboardPopulations)
    bank._owners, bank._sources, bank._bindings = [owner], [source], {"body": binding}
    env.keyboard_variants = bank
    del model, state, owner, source, binding
    assert model_ref() is not None and state_ref() is not None

    enabled = gc.isenabled()
    gc.disable()
    try:
        if fail_stop or fail_clear:
            with pytest.raises(RuntimeError) as error:
                env.close()
            assert error.value is (clear_error if fail_clear else stop_error)
            if fail_stop and fail_clear:
                assert error.value.__context__ is stop_error
        else:
            env.close()
        assert env._is_closed
        env.close()  # Closing is idempotent without retaining a runtime configuration.
        assert calls == ["stop", "clear"]
        assert model_ref() is None and state_ref() is None
        assert source_ref() is None and source_model_ref() is None
    finally:
        if enabled:
            gc.enable()
    assert isinstance(selector, NewtonSelectorCfg)  # Declarative caller configuration is untouched.


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_root_construction_uses_close_and_preserves_startup_error(cleanup_fails):
    from isaaclab_tasks.contrib.keyboard import so101_population_env as module
    from isaaclab_tasks.contrib.keyboard.keyboard_populations import KeyboardPopulations
    from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, NewtonSelections, NewtonSelectorCfg
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"  # Only the constructor's hardware boundary is replaced; native resources below are CPU.
    startup_error, cleanup_error = MemoryError("manager construction"), RuntimeError("simulation cleanup")
    references, calls = [], []

    def create_bank(env):
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body(label="/body")
        builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
        builder.end_world()
        owner = NewtonSelections(builder.finalize("cpu"))
        binding = owner.resolve(NewtonSelectorCfg(BODY, ".*", count_per_world=1))
        references.extend((weakref.ref(owner), weakref.ref(owner.model)))
        bank = KeyboardPopulations.__new__(KeyboardPopulations)
        bank._owners, bank._sources, bank._bindings = [owner], [], {"body": binding}
        env.cfg.runtime_binding = binding
        return bank

    def clear():
        calls.append("clear")
        if cleanup_fails:
            raise cleanup_error

    sim = SimpleNamespace(stop=lambda: calls.append("stop"), clear_instance=clear, reset=lambda: None)
    env = module.SO101KeyboardPopulationEnv.__new__(module.SO101KeyboardPopulationEnv)
    zeros, arange = torch.zeros, torch.arange
    enabled = gc.isenabled()
    gc.disable()
    try:
        with (
            patch.object(module, "SimulationContext", return_value=sim),
            patch.object(module, "KeyboardPopulations", new=create_bank),
            patch.object(module, "EventManager", side_effect=startup_error),
            patch.object(module.SO101KeyboardPopulationEnv, "seed", return_value=42),
            patch.object(torch, "zeros", new=lambda *a, **kw: zeros(*a, **{**kw, "device": "cpu"})),
            patch.object(torch, "arange", new=lambda *a, **kw: arange(*a, **{**kw, "device": "cpu"})),
            patch.object(torch.cuda, "current_stream", return_value=None),
            patch.object(wp, "stream_from_torch", return_value=None),
            patch.object(wp, "ScopedStream", new=lambda _: nullcontext()),
            pytest.raises(MemoryError) as error,
        ):
            module.SO101KeyboardPopulationEnv.__init__(env, cfg)
        assert error.value is startup_error
        assert startup_error.__cause__ is (cleanup_error if cleanup_fails else None)
        assert calls == ["stop", "clear"]
        assert env._is_closed and not hasattr(env, "cfg") and not hasattr(env, "keyboard_variants")
        env.close()
        startup_error.__traceback__ = cleanup_error.__traceback__ = None
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def _cadence_time_out(env):
    return (
        ((env.all_env_ids == 0) & (env.episode_length_buf >= 2))
        | ((env.all_env_ids == 1) & (env.episode_length_buf >= 4))
        | _cadence_terminated(env)
    )


def _cadence_terminated(env):
    return (env.all_env_ids == 3) & (env.episode_length_buf >= 6)


def test_default_cadence_applies_pending_requests_without_natural_termination():
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    horizon = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    interval = cfg.redistribution_interval
    assert (horizon, interval) == (150, 128)
    # An odd first timeout remains odd across 150-step episodes and never meets a 128-step boundary.
    natural_ends = range(1, 1 + horizon * interval, horizon)
    assert not any(step % interval == 0 for step in natural_ends)
    assert cfg.redistribution_mode == "truncate_pending"
    assert cfg.compute_final_obs
    for request_step in range(interval):
        next_boundary = (request_step // interval + 1) * interval
        assert 0 < next_boundary - request_step <= interval


@pytest.mark.parametrize("setting,value", [("compute_final_obs", False), ("is_finite_horizon", True)])
def test_administrative_redistribution_requires_timeout_bootstrapping(setting, value):
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    setattr(cfg, setting, value)
    with pytest.raises(ValueError, match=setting):
        SO101KeyboardPopulationEnv(cfg)


@pytest.mark.parametrize("interval", [True, False, 0, -1, 128.0, 128.5, "128", None])
def test_redistribution_interval_requires_a_positive_integer_before_native_allocation(interval):
    from isaaclab_tasks.contrib.keyboard import so101_population_env as module
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.redistribution_interval = interval
    with (
        patch.object(module, "SimulationContext", side_effect=AssertionError("native allocation")),
        patch.object(torch, "zeros", side_effect=AssertionError("tensor allocation")),
        pytest.raises(ValueError, match="positive integer"),
    ):
        module.SO101KeyboardPopulationEnv(cfg)


@pytest.mark.parametrize("mode", ["episode_boundary", "truncate_pending"])
def test_redistribution_cadence_preserves_explicit_episode_boundaries(mode):
    if not wp.is_cuda_available():
        pytest.skip("MJWarp requires CUDA")
    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 4
    cfg.keyboard_variants = cfg.keyboard_variants[:2]
    cfg.seed = 42
    cfg.commands.typing.reset.buffer_size = 8
    cfg.sim.physics.prototype_physics.load_visual_shapes = False
    cfg.redistribution_interval = 4
    cfg.redistribution_mode = mode
    cfg.compute_final_obs = True
    cfg.terminations.time_out.func = _cadence_time_out
    cfg.terminations.abnormal_robot = cfg.terminations.excessive_contact = None
    cfg.terminations.success = TerminationTermCfg(func=_cadence_terminated)
    cfg.rewards.early_termination = None
    with launch_simulation(cfg, {"headless": True}), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        env = gym.make("IsaacContrib-Keyboard-SO101-Populations", cfg=cfg)
        try:
            task = env.unwrapped
            env.reset()
            bank = task.keyboard_variants
            original = bank.variant_ids.clone()
            bank.request(torch.tensor([2], device=task.device), torch.tensor([1], device=task.device))
            actions = torch.zeros((4, 6), device=task.device)
            for index in range(3):
                _, _, terminated, truncated, _ = env.step(actions)
                assert not terminated.any()
                assert truncated.tolist() == ([True, False, False, False] if index == 1 else [False] * 4)
                assert not task.episode_interrupted.any()
                torch.testing.assert_close(bank.variant_ids, original)
                assert bank.redistribution_count == 0

            snapshots = []
            compute = task.observation_manager.compute
            preview = task.observation_manager.preview

            def observe(*args, **kwargs):
                result = compute(*args, **kwargs)
                snapshots.append((bank.variant_ids.clone(), {name: value.clone() for name, value in result.items()}))
                return result

            def observe_preview():
                result = preview()
                snapshots.append((bank.variant_ids.clone(), {name: value.clone() for name, value in result.items()}))
                return result

            with (
                patch.object(task.observation_manager, "compute", side_effect=observe),
                patch.object(task.observation_manager, "preview", side_effect=observe_preview),
            ):
                obs, _, terminated, truncated, extras = env.step(actions)
            assert not terminated.any()
            assert truncated.tolist() == [True, True, mode == "truncate_pending", False]
            assert task.episode_interrupted.tolist() == [False, False, mode == "truncate_pending", False]
            assert bank.redistribution_count == 1
            assert bank.variant_ids.tolist() == [1, 0, int(mode == "truncate_pending"), 1]
            assert task.episode_length_buf.tolist() == [0, 0, 0 if mode == "truncate_pending" else 4, 4]
            torch.testing.assert_close(snapshots[0][0], original)
            torch.testing.assert_close(snapshots[-1][0], bank.variant_ids)
            for name in obs:
                torch.testing.assert_close(extras["final_obs"][name], snapshots[0][1][name])
                torch.testing.assert_close(obs[name], snapshots[-1][1][name])
            # No stale terminal observation may accompany the next continuing transition.
            _, _, terminated, truncated, extras = env.step(actions)
            assert not terminated.any() and not truncated.any()
            assert "final_obs" not in extras
            _, _, terminated, truncated, _ = env.step(actions)
            assert terminated.tolist() == [False, False, False, True]
            assert not truncated[3]  # A real termination takes precedence over a coincident timeout.
        finally:
            env.close()


@pytest.mark.parametrize("use_graph", [False, True])
def test_exact_population_reset_resize_and_survivor_continuation(use_graph):
    if not wp.is_cuda_available():
        pytest.skip("MJWarp requires CUDA")
    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 4
    cfg.keyboard_variants = cfg.keyboard_variants[:2]
    cfg.seed = 42
    cfg.commands.typing.reset.buffer_size = 8
    cfg.sim.physics.prototype_physics.use_cuda_graph = use_graph
    cfg.sim.physics.prototype_physics.load_visual_shapes = False
    cfg.redistribution_interval = 2
    with launch_simulation(cfg, {"headless": True}), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        env = gym.make("IsaacContrib-Keyboard-SO101-Populations", cfg=cfg)
        try:
            task = env.unwrapped
            assert not isinstance(task, ManagerBasedRLEnv)
            assert not hasattr(task, "scene")
            obs, _ = env.reset()
            assert {name: value.shape for name, value in obs.items()} == {
                "policy": (4, 1080),
                "proprio": (4, 18),
                "perception": (4, 324),
            }
            bank = task.keyboard_variants
            assert bank.backend.counts == (2, 2)
            assert [population.model.joint_dof_count for population in bank.backend.populations] == [228, 24]
            action = torch.full((4, 6), 0.05, device=task.device)
            for _ in range(2):
                obs, reward, *_ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())

            command = task.command_manager.get_term("typing")
            keep = torch.tensor([1, 2, 3], device=task.device)
            q_before = command.cfg.reset_coords.read_state("joint_q")[keep].clone()
            qd_before = command.cfg.reset_dofs.read_state("joint_qd")[keep].clone()
            body_before = command.cfg.key_bodies.read_state("body_q")[keep].clone()
            ids = torch.tensor([0], device=task.device)
            pending_before = bank.variant_ids.clone()
            identities = tuple(bank.backend.populations)
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
                bank.request(ids, torch.ones_like(ids))
                assert bank.backend.populations == identities
                torch.testing.assert_close(bank.variant_ids, pending_before)
                with patch.object(newton.ModelBuilder, "finalize", side_effect=AssertionError("rebuild from builder")):
                    bank.redistribute(ids)
                assert bank.backend.counts == (1, 3)
                torch.testing.assert_close(
                    command.cfg.reset_coords.read_state("joint_q")[keep], q_before, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    command.cfg.reset_dofs.read_state("joint_qd")[keep], qd_before, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    command.cfg.key_bodies.read_state("body_q")[keep], body_before, rtol=0, atol=0
                )
                task._reset_idx(ids)
            assert command.cfg.keys.dense_active().sum(dim=1).tolist() == [6, 6, 108, 6]
            assert not task._dirty

            # Swapping actors across prototypes with unchanged counts preserves
            # every solver/graph and all survivor row indices.
            unchanged = bank.backend.populations
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
                swap_ids = torch.tensor([1, 2], device=task.device)
                bank.apply(swap_ids, torch.tensor([0, 1], device=task.device))
                assert all(a is b for a, b in zip(unchanged, bank.backend.populations))
                task._reset_idx(swap_ids)
            for _ in range(3):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                torch.testing.assert_close(task.reset_buf, terminated | truncated)
            for population in bank.backend.populations:
                assert np.isfinite(population.state_0.joint_q.numpy()).all()
                np.testing.assert_array_equal(population.solver.mjw_data.overflow.numpy(), 0)
        finally:
            env.close()
