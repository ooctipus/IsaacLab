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

from isaaclab.utils import clone

from isaaclab_tasks.contrib.keyboard.selection_paths import NewtonSelectorCfg, resolve_selection


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
    for filename in (
        "so101_population_env.py",
        "keyboard_populations.py",
        "keyboard_worlds.py",
        "newton_selection.py",
        "mujoco_selection.py",
    ):
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
    for forbidden in (
        "population_mdp",
        "population_selection.py",
        "population_scene.py",
        "population_asset.py",
        "so101_worlds_env.py",
        "worlds_mdp",
        "native_state.py",
        "native_reset_adapter.py",
    ):
        assert not (directory / forbidden).exists()
    import isaaclab_newton.physics.worlds as worlds_backend

    for path in (*directory.rglob("*.py"), Path(worlds_backend.__file__)):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Attribute):
                continue
            if node.value.attr == "directory":
                assert node.attr not in {"data", "batch", "transaction", "compaction"}
            if node.value.attr == "runtime":
                assert node.attr != "backing"
    # Prepared native physics has one owner; task modules borrow it through the backend.
    for filename in ("keyboard_worlds.py", "mujoco_selection.py"):
        source = (directory / filename).read_text()
        assert "newton_worlds_lab" not in source
        assert "newton.worlds" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "gpu_components":
                assert {alias.name for alias in node.names} <= {"directory"}
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in {
                    "MuJoCoWorlds",
                    "WorldDirectory",
                    "InstanceDirectory",
                    "FieldStorage",
                    "GraphUpdateTable",
                    "DeviceGraph",
                    "state",
                    "control",
                }
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "instance_directory":
                        assert name in {"allocate_commands", "allocate_results", "location", "handle_at"}


@pytest.mark.parametrize("fail_stop,fail_clear", [(False, False), (True, False), (False, True), (True, True)])
def test_close_releases_native_bindings_held_by_runtime_configuration(fail_stop, fail_clear):
    from isaaclab_tasks.contrib.keyboard.keyboard_populations import KeyboardPopulations
    from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonSelectionGroup, NewtonSelections
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    source = NewtonSelections(builder.finalize("cpu"))
    selector = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, path=".*", count_per_world=1)
    resolve_selection(source, selector)
    source_ref, source_model_ref = weakref.ref(source), weakref.ref(source.model)
    model = source.model.replicate(1)
    state = model.state()
    model_ref, state_ref = weakref.ref(model), weakref.ref(state)
    owner = NewtonSelections(model, state=state, source=source)
    binding = NewtonSelectionGroup(selector.index_domain, ((resolve_selection(owner, selector), torch.tensor([0])),), 1)
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
    from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonSelections
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Populations", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 1
    cfg.events = {}  # EventManager construction is the failure boundary, before real selector binding.
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
        binding = resolve_selection(
            owner, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*", count_per_world=1)
        )
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
            patch.object(module, "warp_on_torch_stream", new=lambda _: nullcontext()),
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


def test_failed_manager_reset_prevents_continuing_with_partial_episode_state():
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    env = SO101KeyboardPopulationEnv.__new__(SO101KeyboardPopulationEnv)
    env._is_closed, env._population_bindings_valid = False, True
    env.cfg = SimpleNamespace(
        keyboard_variants=(),
        sim=SimpleNamespace(physics=None),
        commands=SimpleNamespace(typing=SimpleNamespace(reset=SimpleNamespace(replay_only=True))),
    )
    env.extras = {}
    empty = SimpleNamespace(compute=lambda **kwargs: None, reset=lambda ids: {})
    env.curriculum_manager = env.observation_manager = env.action_manager = env.reward_manager = empty
    env.event_manager = SimpleNamespace(available_modes=[], reset=lambda ids: {})
    env.termination_manager = empty
    error = ValueError("reset payload preparation failed")

    def fail(ids):
        env.extras["partial_command"] = True
        raise error

    env.command_manager = SimpleNamespace(reset=fail)
    with pytest.raises(ValueError) as raised:
        env._reset_idx(torch.tensor([0]))
    assert raised.value is error
    assert env.extras["partial_command"]
    assert not env._population_bindings_valid
    with pytest.raises(RuntimeError, match="failed"):
        env._check_active()


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


@pytest.mark.parametrize("replay_only", [False, True])
def test_native_task_configuration_copy_preserves_reset_mode(replay_only):
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101-Worlds", "", overrides=["physics=newton_mjwarp"])
    cfg.commands.typing.reset.replay_only = replay_only
    cfg.commands.typing.reset.bank_path = "/test/explicit-bank.pt"
    cfg.commands.typing.reset.bank_variant = 0
    copied = clone(cfg)
    assert copied.commands.typing.reset.replay_only is replay_only
    assert copied.commands.typing.reset.bank_path == "/test/explicit-bank.pt"
    assert copied.commands.typing.reset.bank_variant == 0


def test_native_reset_preserves_event_rng_without_editing_the_ending_world():
    from isaaclab_newton.physics.worlds import NewtonWorldsCfg

    from isaaclab_tasks.contrib.keyboard.mdp.reset import sample_root_poses
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    ids = torch.tensor([1, 3])
    pose_range, velocity_range = {"z": (0.01, 0.05), "roll": (0.0, 0.75)}, {"x": (0.0, 0.0)}
    defaults = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    task = SO101KeyboardPopulationEnv.__new__(SO101KeyboardPopulationEnv)
    task.cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NewtonWorldsCfg()),
        keyboard_variants=(object(),),
        commands=SimpleNamespace(
            typing=SimpleNamespace(reset=SimpleNamespace(replay_only=False), reset_roots=SimpleNamespace(width=1))
        ),
        events=SimpleNamespace(
            reset_keyboard=SimpleNamespace(
                mode="reset", params={"pose_range": pose_range, "velocity_range": velocity_range}
            )
        ),
    )
    task.keyboard_variants = SimpleNamespace(
        reset_defaults=defaults, staged_variant_ids=lambda selected: torch.zeros_like(selected)
    )
    task.episode_length_buf = torch.ones(4, dtype=torch.long)
    task.extras = {}
    calls = []
    for name in ("observation", "action", "reward", "curriculum", "command", "event", "termination"):
        setattr(task, name + "_manager", SimpleNamespace(reset=lambda selected, name=name: calls.append(name) or {}))
    task.command_manager.get_term = lambda name: SimpleNamespace(cfg=task.cfg.commands.typing)
    task.curriculum_manager.compute = lambda **kwargs: None
    task.event_manager.available_modes = ("reset",)
    task.event_manager.apply = lambda **kwargs: pytest.fail(
        "An ending native lifetime was edited before payload publication"
    )
    task.forward = lambda: None
    torch.manual_seed(173)
    sample_root_poses(defaults.expand(2, -1).reshape(2, 1, 7), pose_range, velocity_range)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(173)
    task._reset_idx(ids)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
    assert calls == ["observation", "action", "reward", "curriculum", "command", "event", "termination"]
    assert task.episode_length_buf.tolist() == [1, 0, 1, 0]


def test_deferred_root_reset_preserves_sampling_without_redundant_physical_writes():
    from isaaclab_tasks.contrib.keyboard.mdp import reset as module

    task = SimpleNamespace(
        all_env_ids=torch.arange(4),
        cfg=SimpleNamespace(commands=SimpleNamespace(typing=SimpleNamespace(reset=SimpleNamespace(ik=object())))),
    )
    defaults = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]).expand(4, 2, 7)
    roots = SimpleNamespace(read_model=lambda attribute: defaults)
    selected = torch.tensor([0, 3])
    params = {"pose_range": {"z": (0.01, 0.05), "roll": (-0.2, 0.4)}, "velocity_range": {"z": (0.0, 0.0)}}
    with patch.object(module, "write_fixed_root_poses") as write:
        torch.manual_seed(179)
        module.reset_root_state_uniform(task, selected, roots, **params)
        expected_rng = torch.get_rng_state()
        assert write.call_count == 1
        write.reset_mock()
        torch.manual_seed(179)
        module.reset_root_state_uniform(task, selected, roots, **params, defer_to_typing=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)
        write.assert_not_called()
        task.cfg.commands.typing.reset.ik = None
        with pytest.raises(ValueError, match="require typing reset"):
            module.reset_root_state_uniform(task, selected, roots, **params, defer_to_typing=True)


@pytest.mark.parametrize("count,enabled,status", [(0, 1, 0), (1, 1, 0), (37, 1, 0), (37, 0, 0), (37, 1, 1)])
def test_snapshot_columns_complete_before_lifetime_acknowledgement(count, enabled, status):
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import _acknowledge_snapshot, _initialize_snapshot

    capacity, stride = 48, 32
    requests = np.arange(capacity, dtype=np.int32)[::-1].copy()
    destinations = np.roll(np.arange(capacity, dtype=np.int32), 7)
    qcols, qdcols, roots = np.array([3, 0, 5], np.int32), np.array([4, 1], np.int32), np.array([1, 0], np.int32)
    payload = np.arange(capacity * 32, dtype=np.float32).reshape(capacity, 32) / 16
    references = np.array([0.25, -0.5, 0.75], np.float32)
    q = wp.full((capacity, 3), -77.0, dtype=float, device="cpu")
    qd = wp.full((capacity, 2), -77.0, dtype=float, device="cpu")
    pos = wp.full((capacity, 2), wp.vec3(-77.0), device="cpu")
    quat = wp.full((capacity, 2), wp.quat(-77.0, -77.0, -77.0, -77.0), device="cpu")
    request_ids = wp.array(requests, device="cpu")
    extent, outcome = wp.full(1, count, dtype=int, device="cpu"), wp.full(1, status, dtype=int, device="cpu")
    ack = wp.zeros(capacity, dtype=wp.uint64, device="cpu")
    wp.launch(
        _initialize_snapshot,
        (stride, 3),
        inputs=[
            request_ids,
            wp.array(destinations, device="cpu"),
            extent,
            outcome,
            stride,
            wp.full(1, enabled, dtype=int, device="cpu"),
            wp.array(payload, device="cpu"),
            wp.array(qcols, device="cpu"),
            wp.array(qdcols, device="cpu"),
            wp.array(roots, device="cpu"),
            wp.array(references, device="cpu"),
            14,
            20,
            q,
            qd,
            pos,
            quat,
        ],
        device="cpu",
    )
    assert not ack.numpy().any()
    changed = destinations[:count] if enabled and status == 0 else np.empty(0, dtype=int)
    source = requests[: len(changed)]
    expected_q, expected_qd = np.full((capacity, 3), -77.0, np.float32), np.full((capacity, 2), -77.0, np.float32)
    expected_pos, expected_quat = (
        np.full((capacity, 2, 3), -77.0, np.float32),
        np.full((capacity, 2, 4), -77.0, np.float32),
    )
    expected_q[changed] = payload[source[:, None], 14 + qcols] + references
    expected_qd[changed] = payload[source[:, None], 20 + qdcols]
    for root, source_root in enumerate(roots):
        expected_pos[changed, root] = payload[source[:, None], 7 * source_root + np.arange(3)]
        expected_quat[changed, root] = payload[source[:, None], 7 * source_root + np.array([6, 3, 4, 5])]
    for actual, expected in ((q, expected_q), (qd, expected_qd), (pos, expected_pos), (quat, expected_quat)):
        np.testing.assert_array_equal(actual.numpy(), expected)
    wp.launch(
        _acknowledge_snapshot,
        capacity,
        inputs=[request_ids, extent, outcome, wp.full(1, 9, dtype=wp.uint64, device="cpu"), ack],
        device="cpu",
    )
    expected_ack = np.zeros(capacity, np.uint64)
    if status == 0:
        expected_ack[requests[:count]] = 9
    np.testing.assert_array_equal(ack.numpy(), expected_ack)


@pytest.mark.parametrize("batch_error", [0, 12])
def test_native_batch_failure_prevents_task_handle_publication(batch_error):
    from gpu_components import directory as instance_directory

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import _publish_handles

    directory = instance_directory.allocate((2,), id_capacity=1, command_capacity=1, device="cpu")
    results = instance_directory.allocate_results(1, device="cpu")
    results.instance_id.fill_(0)
    results.generation.fill_(2)
    directory.batch_result.status.fill_(batch_error)
    actors = wp.zeros(1, dtype=int, device="cpu")
    requested = wp.zeros(1, dtype=wp.int64, device="cpu")
    handles = wp.full(1, -1, dtype=int, device="cpu")
    generations = wp.zeros(1, dtype=wp.uint64, device="cpu")
    inverse = wp.full(1, -1, dtype=int, device="cpu")
    variants = wp.full(1, -1, dtype=wp.int64, device="cpu")
    failed = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(
        _publish_handles,
        1,
        [results, directory.batch_result.status, actors, requested, handles, generations, inverse, variants, failed],
        device="cpu",
    )
    assert handles.numpy().tolist() == ([-1] if batch_error else [0])
    assert generations.numpy().tolist() == ([0] if batch_error else [2])
    assert failed.numpy().tolist() == ([1] if batch_error else [0])


@pytest.mark.parametrize("failure", [None, MemoryError, RuntimeError])
def test_native_reset_retries_optional_headroom_once_before_publication(failure):
    from gpu_components import directory as instance_directory
    from gpu_components.directory_data import InstanceOperation

    from isaaclab_tasks.contrib.keyboard import keyboard_worlds
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds

    directory = instance_directory.allocate((4, 4), id_capacity=2, command_capacity=2, device="cpu")
    instance_directory.publish_admissible_slots(directory, (4, 0))
    bank = object.__new__(KeyboardWorlds)
    bank.commands = instance_directory.allocate_commands(2, device="cpu")
    bank.results = instance_directory.allocate_results(2, device="cpu")
    bank.commands.operation.fill_(int(InstanceOperation.CREATE))
    bank.commands.count.fill_(2)
    bank.commands.sequence.fill_(1)
    instance_directory.begin(directory, bank.commands)
    instance_directory.admit(directory, bank.commands)
    directory.transaction.initialized_sequence.fill_(1)
    instance_directory.publish(directory, bank.commands, bank.results)
    handles = bank.results.instance_id.numpy().copy()
    bank.world_id_by_env = wp.array(handles, dtype=int, device="cpu")
    bank.world_generation_by_env = wp.ones(2, dtype=wp.uint64, device="cpu")
    bank.env_index_by_world_id = wp.array(np.argsort(handles), dtype=int, device="cpu")
    bank.variant_ids = torch.zeros(2, dtype=torch.long)
    bank._request_env_indices, bank._failed = wp.empty(2, dtype=int, device="cpu"), wp.zeros(1, dtype=int, device="cpu")
    bank._request_variants, bank._demand = torch.empty(2, dtype=torch.long), wp.empty((2, 2), dtype=int, device="cpu")
    bank.layouts = (None, None)
    bank.env = SimpleNamespace(
        device="cpu", cfg=SimpleNamespace(worlds_spare_memory_budget_bytes=0), _population_bindings_valid=True
    )
    groups = tuple(SimpleNamespace(ready=n, world_capacity=4) for n in (4, 0))
    calls = []

    def grow(runtime, counts, *, streams):
        calls.append(("grow", counts))
        raise MemoryError("optional headroom exhausted budget")

    def resize(runtime, counts, *, streams, spare_bytes):
        calls.append(("resize", counts))
        assert spare_bytes == 0
        if failure is not None:
            raise failure("required backing unavailable or quarantined")
        instance_directory.publish_admissible_slots(directory, counts)
        for group, count in zip(groups, counts, strict=True):
            group.ready = count

    def publish():
        calls.append(("publish",))
        instance_directory.begin(directory, bank.commands)
        instance_directory.admit(directory, bank.commands)
        directory.transaction.initialized_sequence.fill_(2)
        instance_directory.publish(directory, bank.commands, bank.results)

    runtime = SimpleNamespace(
        directory=directory.data,
        batch_result=directory.batch_result,
        populations=groups,
    )
    bank.backend = SimpleNamespace(runtime=runtime, forward=publish)
    try:
        with (
            patch.object(wp, "get_stream", return_value=None),
            patch.object(keyboard_worlds, "mujoco_world_population_ready_capacity", new=lambda group: group.ready),
            patch.object(keyboard_worlds, "mujoco_worlds_grow_backing", new=grow),
            patch.object(keyboard_worlds, "mujoco_worlds_resize_backing", new=resize),
        ):
            if failure is None:
                bank._submit(torch.arange(2), torch.ones(2, dtype=torch.long))
            else:
                with pytest.raises(failure, match="required backing unavailable"):
                    bank._submit(torch.arange(2), torch.ones(2, dtype=torch.long))
        assert calls[:2] == [("grow", (4, 4)), ("resize", (2, 2))]
        if failure is None:
            assert calls[2:] == [("publish",), ("resize", (0, 2))]
            assert bank.variant_ids.tolist() == [1, 1]
            assert bank.world_generation_by_env.numpy().tolist() == [2, 2]
            assert bank.env._population_bindings_valid
        else:
            assert len(calls) == 2
            assert bank.variant_ids.tolist() == [0, 0]
            assert bank.world_id_by_env.numpy().tolist() == handles.tolist()
            assert bank.world_generation_by_env.numpy().tolist() == [1, 1]
            assert directory.data.prototype.numpy()[handles].tolist() == [0, 0]
            assert not bank.env._population_bindings_valid
    finally:
        instance_directory.close(directory, streams=())


def test_partial_native_publication_keeps_actual_successes_and_stops_task():
    from gpu_components import directory as instance_directory
    from gpu_components.directory_data import InstanceOperation

    from isaaclab_tasks.contrib.keyboard import keyboard_worlds
    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds
    from isaaclab_tasks.contrib.keyboard.so101_population_env import SO101KeyboardPopulationEnv

    directory = instance_directory.allocate((4, 4), id_capacity=2, command_capacity=2, device="cpu")
    instance_directory.publish_admissible_slots(directory, (4, 4))
    bank = object.__new__(KeyboardWorlds)
    bank.commands, bank.results = (
        instance_directory.allocate_commands(2, device="cpu"),
        instance_directory.allocate_results(2, device="cpu"),
    )
    bank.commands.operation.fill_(int(InstanceOperation.CREATE))
    bank.commands.count.fill_(2)
    bank.commands.sequence.fill_(1)
    instance_directory.begin(directory, bank.commands)
    instance_directory.admit(directory, bank.commands)
    directory.transaction.initialized_sequence.fill_(1)
    instance_directory.publish(directory, bank.commands, bank.results)
    handles = bank.results.instance_id.numpy().copy()
    bank.world_id_by_env = wp.array(handles, dtype=int, device="cpu")
    bank.world_generation_by_env = wp.ones(2, dtype=wp.uint64, device="cpu")
    bank.env_index_by_world_id = wp.array(np.argsort(handles), dtype=int, device="cpu")
    bank.variant_ids, bank._staged_variant_ids = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
    bank.desired_variant_ids = torch.ones(2, dtype=torch.long)
    bank._payload, bank._payload_enabled = torch.empty((2, 3)), wp.ones(1, dtype=int, device="cpu")
    bank._request_env_indices, bank._failed = wp.empty(2, dtype=int, device="cpu"), wp.zeros(1, dtype=int, device="cpu")
    bank._request_variants, bank._demand = torch.empty(2, dtype=torch.long), wp.empty((2, 2), dtype=int, device="cpu")
    bank.layouts, bank.reset_publication_count = (None, None), 0
    bank.env = object.__new__(SO101KeyboardPopulationEnv)
    bank.env.device, bank.env.num_envs = "cpu", 2
    bank.env._is_closed, bank.env._population_bindings_valid = False, True

    def publish_partial():
        instance_directory.begin(directory, bank.commands)
        instance_directory.admit(directory, bank.commands)
        # First request deliberately lacks initialization; the second succeeds.
        directory.transaction.initialized_sequence.assign(np.array([0, 2], dtype=np.uint64))
        instance_directory.publish(directory, bank.commands, bank.results)

    runtime = SimpleNamespace(
        directory=directory.data,
        batch_result=directory.batch_result,
        populations=tuple(SimpleNamespace(ready=4, world_capacity=4) for _ in range(2)),
    )
    bank.backend = SimpleNamespace(runtime=runtime, forward=publish_partial)
    ids, variants, snapshot = torch.arange(2), torch.ones(2, dtype=torch.long), torch.ones((2, 3))
    try:
        # CPU directory/payload exercise; no backing service or CUDA stream operation is needed.
        with (
            patch.object(wp, "get_stream", return_value=None),
            patch.object(keyboard_worlds, "mujoco_world_population_ready_capacity", new=lambda group: group.ready),
            pytest.raises(RuntimeError, match="publication failed"),
        ):
            bank.reset_from_snapshot(ids, variants, snapshot)
        assert bank.variant_ids.tolist() == [0, 1]
        assert bank.world_generation_by_env.numpy().tolist() == [1, 2]
        assert directory.data.prototype.numpy()[handles].tolist() == [0, 1]
        assert bank._staged_variant_ids.tolist() == [0, 0]
        assert bank.desired_variant_ids.tolist() == [1, 1]
        assert bank.reset_publication_count == 0
        assert not bank.env._population_bindings_valid
        for operation in (
            lambda: bank.request_variants(ids, variants),
            lambda: bank.stage_variant_changes(ids),
            lambda: bank.reset_from_snapshot(ids, variants, snapshot),
        ):
            with pytest.raises(RuntimeError, match="close this environment"):
                operation()
    finally:
        instance_directory.close(directory, streams=())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_native_reset_payload_preserves_old_lifetime(invalid):
    from gpu_components import directory as instance_directory
    from gpu_components.directory_data import InstanceOperation, InstanceStatus

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import _validate_snapshot

    directory = instance_directory.allocate((4,), id_capacity=2, command_capacity=2, device="cpu")
    instance_directory.publish_admissible_slots(directory, (4,))
    commands, results = (
        instance_directory.allocate_commands(2, device="cpu"),
        instance_directory.allocate_results(2, device="cpu"),
    )
    commands.operation.fill_(int(InstanceOperation.CREATE))
    commands.count.fill_(2)
    commands.sequence.fill_(1)
    instance_directory.begin(directory, commands)
    instance_directory.admit(directory, commands)
    directory.transaction.initialized_sequence.fill_(1)
    instance_directory.publish(directory, commands, results)
    original_ids = results.instance_id.numpy().copy()
    original_generations = results.generation.numpy().copy()
    original_slots = directory.data.slot.numpy().copy()
    commands.instance_id.assign(original_ids)
    commands.generation.assign(original_generations)
    commands.operation.fill_(int(InstanceOperation.REPLACE))
    commands.sequence.fill_(2)
    instance_directory.begin(directory, commands)
    payload = wp.array(np.array([[0, invalid, 1], [0, 0, 1]], dtype=np.float32), device="cpu")
    enabled = wp.ones(1, dtype=int, device="cpu")
    wp.launch(
        _validate_snapshot,
        payload.shape,
        [commands, directory.transaction.status, directory.batch_result.consumed, enabled, payload],
        device="cpu",
    )
    instance_directory.admit(directory, commands)
    directory.transaction.initialized_sequence.fill_(2)
    instance_directory.publish(directory, commands, results)
    assert results.status.numpy().tolist() == [int(InstanceStatus.INVALID), int(InstanceStatus.OK)]
    first = int(original_ids[0])
    assert directory.data.slot.numpy()[first] == original_slots[first]
    assert directory.data.generation.numpy()[first] == original_generations[0]
    assert directory.data.generation.numpy()[int(original_ids[1])] == original_generations[1] + 1
    assert (
        directory.batch_result.advance_allowed.numpy()[0] == 0
    )  # No physical advancement after a mandatory failed reset.


def test_native_overflow_keeps_capacity_failures_distinct_from_iteration_limits():
    import mujoco_warp as mjw

    from isaaclab_tasks.contrib.keyboard.keyboard_worlds import KeyboardWorlds, _record_overflow

    limits = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    values = wp.array(
        np.array([int(mjw.OverflowType.ITERATIONS), int(mjw.OverflowType.LS_ITERATIONS)]), dtype=int, device="cpu"
    )
    sticky = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(_record_overflow, 2, [values, 0, sticky], device="cpu")
    assert sticky.numpy()[0] == limits
    assert sticky.numpy()[0] & KeyboardWorlds.capacity_overflow_mask == 0
    values.assign(np.array([int(mjw.OverflowType.NARROWPHASE), 0], dtype=np.int32))
    wp.launch(_record_overflow, 2, [values, 0, sticky], device="cpu")
    assert sticky.numpy()[0] == limits | int(mjw.OverflowType.NARROWPHASE)
    assert sticky.numpy()[0] & KeyboardWorlds.capacity_overflow_mask == int(mjw.OverflowType.NARROWPHASE)


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
            bank.request_variants(torch.tensor([2], device=task.device), torch.tensor([1], device=task.device))
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
                bank.request_variants(ids, torch.ones_like(ids))
                assert bank.backend.populations == identities
                torch.testing.assert_close(bank.variant_ids, pending_before)
                with patch.object(newton.ModelBuilder, "finalize", side_effect=AssertionError("rebuild from builder")):
                    bank.apply_pending_variants(ids)
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
