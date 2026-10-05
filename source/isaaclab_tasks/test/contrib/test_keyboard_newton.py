# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selector contracts and the view-free 108-key task's reset/control boundary."""

import ast
import copy
import gc
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import newton
import numpy as np
import pytest
import torch
import warp as wp
from gpu_components.directory_data import InstanceDirectoryData

from isaaclab.utils import replace
from isaaclab.utils.warp.utils import warp_on_torch_stream

from isaaclab_tasks.contrib.keyboard.mujoco_selection import (
    EnvWorldBindings,
    MuJoCoScalarField,
    MuJoCoSelection,
    MuJoCoSelections,
    _ScalarSource,
    _WorldReadiness,
)
from isaaclab_tasks.contrib.keyboard.mujoco_selection import (
    scalar_field_active as native_field_active,
)
from isaaclab_tasks.contrib.keyboard.mujoco_selection import (
    scalar_field_read as native_field_read,
)
from isaaclab_tasks.contrib.keyboard.mujoco_selection import (
    scalar_field_write as native_field_write,
)
from isaaclab_tasks.contrib.keyboard.newton_selection import (
    NewtonSelectionGroup,
    NewtonSelections,
    pose_field_active,
    pose_field_read,
)
from isaaclab_tasks.contrib.keyboard.selection_paths import NewtonSelectorCfg, resolve_selection


@wp.kernel
def _probe_pose_field(
    field: Any, worlds: wp.array[int], slots: wp.array[int], active: wp.array[bool], values: wp.array[wp.transform]
):
    i = wp.tid()
    active[i] = pose_field_active(field, worlds[i], slots[i])
    values[i] = pose_field_read(field, worlds[i], slots[i])


@wp.kernel
def _stream_increment(source: wp.array[float], destination: wp.array[float]):
    destination[wp.tid()] = source[wp.tid()] + 1.0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("operation", ["reset", "reset_to", "step", "forward", "reset_keyboard"])
def test_keyboard_root_orders_torch_and_warp_producers(device, operation):
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.contrib.keyboard.so101_env import SO101KeyboardEnv

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    source, destination = (torch.zeros(4, device=device) for _ in range(2))
    source_wp, destination_wp = wp.from_torch(source), wp.from_torch(destination)
    observed = wp.empty_like(destination_wp)
    result = []

    def numerical_stage(*args, **kwargs):
        wp.launch(_stream_increment, 4, inputs=[source_wp], outputs=[destination_wp], device=device)
        result.append(destination * 2.0)

    env = SO101KeyboardEnv.__new__(SO101KeyboardEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(device=device, forward=numerical_stage if operation == "forward" else lambda: None)
    env.keyboard_variants = object()
    env._reset_idx = numerical_stage
    wp.synchronize_device(device)
    prior = wp.ScopedStream(wp.Stream(device), sync_exit=True) if device.startswith("cuda") else nullcontext()
    producer = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.startswith("cuda") else nullcontext()
    parent_operation = operation if operation in {"reset", "reset_to", "step"} else "step"
    with prior, producer, patch.object(ManagerBasedRLEnv, parent_operation, numerical_stage):
        if device.startswith("cuda"):
            torch.cuda._sleep(20_000_000)
        source.fill_(41.0)
        if operation == "reset":
            env.reset()
        elif operation == "reset_to":
            env.reset_to({})
        elif operation == "step":
            env.step(None)
        elif operation == "forward":
            env.forward()
        else:
            env.reset_keyboard(torch.arange(4, device=device), torch.zeros(4, device=device, dtype=torch.long))
        # This consumer uses the restored Warp stream, not the caller's Torch stream.
        wp.copy(observed, destination_wp)
    wp.synchronize_device(device)
    torch.testing.assert_close(wp.to_torch(observed), torch.full_like(destination, 42.0))
    assert len(result) == 1
    torch.testing.assert_close(result[0], torch.full_like(destination, 84.0))


def test_keyboard_observations_preserve_padding_participation_and_frame_math():
    from isaaclab.utils.math import subtract_frame_transforms

    from isaaclab_tasks.contrib.keyboard.mdp.observations import (
        _relative_key_positions,
        target_keys_onehot,
        typed_keys_onehot,
    )

    wp.init()
    generator = torch.Generator().manual_seed(351)
    worlds, keys, width = 11, 18, 5
    active = torch.rand(worlds, keys, generator=generator) > 0.3
    active[0] = False
    slots = torch.randint(-1, keys, (worlds, width), generator=generator)
    command = SimpleNamespace(
        target=slots, typed=slots.flip(1), key_joints=SimpleNamespace(dense_active=lambda: active)
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    for operation, tokens in ((target_keys_onehot, command.target), (typed_keys_onehot, command.typed)):
        expected = torch.zeros(worlds, width, keys)
        for world in range(worlds):
            for letter, key in enumerate(tokens[world].tolist()):
                if key >= 0 and active[world, key]:
                    expected[world, letter, key] = 1
        torch.testing.assert_close(operation(env, "typing"), expected.flatten(1), rtol=0, atol=0)

    roots = torch.randn(worlds, 1, 7, generator=generator)
    roots[0, :, 3:] = 0
    poses = torch.randn(worlds, keys, 7, generator=generator)
    root_active = torch.rand(worlds, 1, generator=generator) > 0.2
    expected, _ = subtract_frame_transforms(roots[..., :3], roots[..., 3:], poses[..., :3])
    expected = torch.where((active & root_active)[..., None], expected, 0)
    output = torch.full((worlds, keys, 3), float("nan"))

    def pose_field(values, membership):
        from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonPoseField, _PoseSource

        field, source = NewtonPoseField(), _PoseSource()
        source.values = wp.from_torch(values.reshape(-1, 7), dtype=wp.transform)
        field.sources = wp.array([source], dtype=_PoseSource, device="cpu")
        field.sources._values = source.values
        field.source_ids = wp.zeros(worlds, dtype=int, device="cpu")
        field.ids = wp.from_torch(
            torch.arange(values.shape[0] * values.shape[1], dtype=torch.int32).reshape(values.shape[:2])
        )
        field.active = wp.from_torch(membership)
        return field

    wp.launch(
        _relative_key_positions,
        (worlds, keys),
        inputs=[
            pose_field(roots, root_active),
            pose_field(poses, active),
        ],
        outputs=[wp.from_torch(output, dtype=wp.vec3)],
        device="cpu",
    )
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)
    probe_active, probe_values = wp.empty(5, dtype=bool, device="cpu"), wp.empty(5, dtype=wp.transform, device="cpu")
    wp.launch(
        _probe_pose_field,
        5,
        [
            pose_field(poses, active),
            wp.array([-1, worlds, 1, 1, 1], dtype=int, device="cpu"),
            wp.array([0, 0, -1, keys, 0], dtype=int, device="cpu"),
        ],
        [probe_active, probe_values],
        device="cpu",
    )
    np.testing.assert_array_equal(probe_active.numpy(), [False, False, False, False, bool(active[1, 0])])
    np.testing.assert_array_equal(probe_values.numpy()[:4], 0)
    np.testing.assert_array_equal(probe_values.numpy()[4], poses[1, 0].numpy() if active[1, 0] else np.zeros(7))


@pytest.fixture(params=["cpu", "cuda:0"])
def selections(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    builder = newton.ModelBuilder()
    for world in range(2):
        builder.begin_world()
        root = builder.add_link(label=f"/scene/{world}/base", mass=1.0)
        joints = [builder.add_joint_free(child=root, label=f"/scene/{world}/free")]
        for index in range(world + 1):
            child = builder.add_link(label=f"/scene/{world}/tip{index}", mass=1.0)
            joints.append(builder.add_joint_revolute(parent=root, child=child, label=f"/scene/{world}/hinge{index}"))
        builder.add_articulation(joints)
        builder.end_world()
    return NewtonSelections(builder.finalize(request.param))


def test_frequency_expansion_and_validation(selections):
    q = resolve_selection(
        selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, ".*/free", count_per_world=7)
    )
    qd = resolve_selection(
        selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_DOF, ".*/free", count_per_world=6)
    )
    assert q.world_selection_counts == (7, 7) and qd.world_selection_counts == (6, 6)
    assert copy.deepcopy(q) is q
    with pytest.raises(ValueError, match="matched no"):
        resolve_selection(selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/missing"))
    with pytest.raises(ValueError, match="Expected 1"):
        resolve_selection(selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*", count_per_world=1))
    with pytest.raises(ValueError, match="Unknown Newton index domain"):
        resolve_selection(selections, NewtonSelectorCfg("joint", ".*"))


def test_numeric_binding_composes_path_ids_without_retaining_names(selections):
    """Keep name resolution outside integer relations, including the empty selection."""
    from isaaclab_tasks.contrib.keyboard.selection_paths import query_selection_indices

    cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, (".*/tip.*", ".*"))
    ids = query_selection_indices(selections.model, cfg)
    selected = selections.bind(newton.Model.AttributeFrequency.BODY, ids)
    assert selected is resolve_selection(selections, cfg)
    assert selected.ids.numpy().tolist() == [1, 0, 3, 4, 2]
    assert not hasattr(selected, "path") and not hasattr(selected, "count_per_world")
    empty = selections.bind(newton.Model.AttributeFrequency.BODY, [])
    assert empty.world_selection_counts == (0, 0) and empty.width == 0
    assert empty.dense_active().shape == (2, 0)
    np.testing.assert_array_equal(empty.world_start.numpy(), [0, 0, 0])
    for index_domain in (newton.Model.AttributeFrequency.JOINT_COORD, newton.Model.AttributeFrequency.JOINT_DOF):
        assert selections.bind(index_domain, []).joint_types().shape == (2, 0)
    for invalid in ([True], [0, True], [0.5], ["0"], [0, 0], [-1], [selections.model.body_count]):
        with pytest.raises(ValueError):
            selections.bind(newton.Model.AttributeFrequency.BODY, invalid)


def test_numeric_selection_owners_have_no_symbolic_resolution_or_private_runtime_access():
    """Guard the name-to-ID boundary and the supported native prototype interface."""
    import isaaclab_tasks.contrib.keyboard as keyboard

    directory = Path(keyboard.__file__).parent
    assert not (directory / "native_selection.py").exists()
    assert set(EnvWorldBindings.vars) == {
        "world_id_by_env",
        "world_generation_by_env",
        "env_participating",
        "directory",
        "world_readiness_by_prototype",
    }
    assert "env_world_bindings" in MuJoCoScalarField.vars and "placement" not in MuJoCoScalarField.vars
    forbidden = {
        "NativePlacement",
        "NativeSelection",
        "NativeSelections",
        "NativePrototypeMapping",
        "NewtonMuJoCoMapping",
        "static_counts",
    }
    for path in directory.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Name)):
                assert getattr(node, "name", getattr(node, "id", "")) not in forbidden
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden
    for name in ("newton_selection.py", "mujoco_selection.py"):
        tree = ast.parse((directory / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in {"BODY", "JOINT_COORD", "JOINT_DOF"}
            if isinstance(node, ast.Import):
                assert all(alias.name != "re" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert all(
                    alias.name not in {"NewtonSelectorCfg", "query_selection_indices", "resolve_selection"}
                    for alias in node.names
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name != "resolve"
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"path", "count_per_world", "policy_width"}
    for name in ("mujoco_selection.py", "keyboard_worlds.py"):
        tree = ast.parse((directory / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mujoco_warp._src")
            if isinstance(node, ast.Attribute):
                assert not node.attr.startswith("mjc_")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "group":
                assert node.attr not in {"rows", "contacts", "ccd", "updates", "observe_launch", "_owner"}


def test_numeric_selection_domains_use_the_model_frequency_identity():
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    owner = NewtonSelections(builder.finalize("cpu"))
    selection = owner.bind(newton.Model.AttributeFrequency.BODY, [0])
    assert selection.index_domain is newton.Model.AttributeFrequency.BODY
    for domain in ("body", "BODY", int(newton.Model.AttributeFrequency.BODY), True):
        with pytest.raises(ValueError, match="index domain"):
            owner.bind(domain, [0])


def test_retired_selection_owner_preserves_existing_bindings_without_a_cache_cycle():
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    model = builder.finalize("cpu")
    owner = NewtonSelections(model, state=model.state(), control=model.control())
    cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*", count_per_world=1)
    selected = resolve_selection(owner, cfg)
    references = [weakref.ref(value) for value in (owner, model, owner.state, owner.control)]
    del model
    enabled = gc.isenabled()
    gc.disable()
    try:
        owner.retire()
        owner.retire()
        for selector in (cfg, replace(cfg, path="/body")):
            with pytest.raises(RuntimeError, match="retired"):
                resolve_selection(owner, selector)
        del owner
        assert all(reference() is not None for reference in references)
        assert selected.read_state("body_q").shape == (1, 1, 7)
        selected.refresh()
        del selected
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def test_ragged_membership_empty_world_and_reactivation(selections):
    bodies = resolve_selection(selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, (".*/tip.*", ".*")))
    assert bodies.world_selection_counts == (2, 3)  # overlap was deduplicated
    assert bodies.ids.numpy().tolist() == [1, 0, 3, 4, 2]  # pattern order, then model order
    with pytest.raises(ValueError, match="equal static counts"):
        bodies.dense_ids()
    pointers = (bodies.freq_ids.ptr, bodies.world_start.ptr)
    wp.to_torch(selections.body_active)[0] = False
    wp.to_torch(selections.world_active)[1] = False
    selections.refresh()
    np.testing.assert_array_equal(bodies.world_start.numpy(), [0, 1, 1])
    assert bodies.freq_ids.numpy()[0] == 1
    assert bodies.env_ids.numpy()[0] == 0
    assert bodies.slot_ids.numpy()[0] == 0
    selections.body_active.fill_(True)
    selections.world_active.fill_(True)
    selections.refresh()
    np.testing.assert_array_equal(bodies.world_start.numpy(), [0, 2, 5])
    assert (bodies.freq_ids.ptr, bodies.world_start.ptr) == pointers


def test_dense_mask_clears_values_without_caching_state(selections):
    bodies = resolve_selection(
        selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/base", count_per_world=1)
    )
    values = wp.ones(selections.model.body_count, dtype=wp.float32, device=selections.model.device)
    wp.to_torch(selections.body_active)[0] = False
    selections.refresh()
    np.testing.assert_array_equal(bodies.dense(values).cpu().numpy(), [[0.0], [1.0]])
    values.fill_(3.0)
    np.testing.assert_array_equal(bodies.dense(values).cpu().numpy(), [[0.0], [3.0]])


def test_membership_refresh_is_capture_safe(selections):
    if not selections.model.device.is_cuda:
        pytest.skip("CUDA graph test")
    selected = resolve_selection(selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*"))
    with wp.ScopedCapture(device=selections.model.device) as capture:
        selections.refresh()
    selections.world_active.fill_(False)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(selected.world_start.numpy(), [0, 0, 0])
    selections.world_active.fill_(True)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(selected.world_start.numpy(), [0, 2, 5])


def test_task_architecture_has_no_articulation_views():
    import isaaclab_tasks.contrib.keyboard as keyboard

    directory = Path(keyboard.__file__).parent
    for path in directory.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("newton.selection"), path
                if "mdp" in path.parts:
                    assert "NewtonManager" not in {alias.name for alias in node.names}, path
                assert not {alias.name for alias in node.names} & {
                    "Articulation",
                    "ArticulationView",
                    "ArticulationCfg",
                }, path
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"root_view", "_root_view", "_inst_idx", "_body_col_idx"}, path
    assert not (directory / "newton_view.py").exists()
    assert not (directory / "selection_manager.py").exists()
    assert not (directory / "articulation_adapter.py").exists()
    # Fixed-root writes stay with reset ownership; no second writer or persistent dispatch cache.
    owners = [path for path in directory.rglob("*.py") if "def _write_fixed_root_poses(" in path.read_text()]
    assert owners == [directory / "mdp" / "reset.py"]
    reset_source = owners[0].read_text()
    assert 'module="unique", module_options={"fuse_fp": False}' in reset_source
    assert "set_module_options" not in reset_source and "combine_frame_transforms" not in reset_source
    # Configuration descriptors have one owner; do not recreate a second bank of generated metadata.
    pool = ast.parse((directory / "keyboards" / "keyboard_pool.py").read_text())
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(pool))
    assert not any(isinstance(node, ast.Name) and node.id == "TYPING_KEYBOARD_POOL" for node in ast.walk(pool))
    tree = ast.parse((directory / "keyboard_variants.py").read_text())
    reset = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "apply")
    for node in ast.walk(reset):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in {"finalize", "add_usd", "ModelBuilder", "generate_keyboard", "spawn_keyboard"}


def test_task_graph_explicitly_retains_contact_borrowers_without_task_root():
    import isaaclab_tasks.contrib.keyboard.keyboard_worlds as module

    tree = ast.parse(Path(module.__file__).read_text())
    prepare = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "prepare"
    )
    retain = next(keyword.value for keyword in prepare.keywords if keyword.arg == "retain")
    attributes = {node.attr for node in ast.walk(retain) if isinstance(node, ast.Attribute)}
    assert {"_contact_selection", "env_index_by_world_id"} <= attributes
    assert not any(isinstance(value, ast.Name) and value.id == "self" for value in retain.elts)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("keyboard_mode, roots", [("partitioned_108", 19), ("single_108", 2)])
def test_108_key_task_without_views_and_partial_reset(use_graph, keyboard_mode, roots):
    if not wp.is_cuda_available():
        pytest.skip("MJWarp requires CUDA")
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from newton.selection import ArticulationView

    from isaaclab.app import launch_simulation
    from isaaclab.assets import AssetBaseCfg

    from isaaclab_tasks.contrib.keyboard.mdp.reset import capture_reset_state, restore_reset_state
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config(
        "IsaacContrib-Keyboard-SO101", "", overrides=["physics=newton_mjwarp", f"presets={keyboard_mode}"]
    )
    cfg.scene.num_envs = 2
    assert cfg.keyboard_variants == ()
    cfg.seed = 42
    cfg.commands.typing.reset.buffer_size = 8
    cfg.sim.physics.use_cuda_graph = use_graph
    assert type(cfg.scene.robot) is AssetBaseCfg and type(cfg.scene.keyboard) is AssetBaseCfg
    with (
        launch_simulation(cfg, {"headless": True}),
        patch.object(ArticulationView, "__init__", side_effect=AssertionError("Unexpected Newton view")),
    ):
        env = gym.make("IsaacContrib-Keyboard-SO101", cfg=cfg)
        try:
            task = env.unwrapped
            obs, _ = env.reset()
            assert {key: value.shape[1] for key, value in obs.items()} == {
                "policy": 1080,
                "proprio": 18,
                "perception": 324,
            }
            assert task.scene.articulations == {}
            assert NewtonManager.get_model().articulation_count == 2 * roots
            assert isinstance(cfg.commands.typing.keys, NewtonSelectorCfg)  # caller config stays declarative
            assert isinstance(task.cfg.commands.typing.keys, NewtonSelectorCfg)
            assert not isinstance(task.command_manager.get_term("typing").cfg.keys, NewtonSelectorCfg)
            for _ in range(4):
                obs, reward, *_ = env.step(torch.full((2, 6), 0.05, device=task.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            command = task.command_manager.get_term("typing")
            c = command.cfg
            ids = torch.tensor([0], device=task.device)
            snapshot = capture_reset_state(task, ids, c.reset_roots, c.reset_coords, c.reset_dofs)
            state, model = NewtonManager.get_state_0(), NewtonManager.get_model()
            q_before = state.joint_q.numpy().copy()
            body_before = state.body_q.numpy().copy()
            restore_reset_state(task, snapshot, ids, c.reset_roots, c.reset_coords, c.reset_dofs)
            task.sim.forward()
            np.testing.assert_array_equal(state.joint_q.numpy(), q_before)
            other_bodies = model.body_world.numpy() == 1
            np.testing.assert_array_equal(state.body_q.numpy()[other_bodies], body_before[other_bodies])
            # Membership affects sampling and observations even though physics sleep is unchanged.
            key_ids = c.key_bodies.dense_ids()
            wp.to_torch(task.selections.body_active)[key_ids[0]] = False
            wp.to_torch(task.selections.body_active)[key_ids[0, 2]] = True
            wp.to_torch(task.selections.world_active)[1] = False
            task.selections.refresh()
            command._resample_normal(torch.arange(2, device=task.device))
            wp.synchronize()
            assert torch.all(command.target[0][command.target[0] >= 0] == 2)
            assert command.target_len[1] == 0
            assert command.typed_len[0] != command.target_len[0] or torch.any(command.typed[0] != command.target[0])
            perception = task.observation_manager.compute()["perception"].reshape(2, 108, 3)
            assert torch.count_nonzero(perception[1]) == 0
            assert torch.count_nonzero(perception[0, torch.arange(108, device=task.device) != 2]) == 0
            # Reset work must preserve excluded worlds and avoid commands requiring an excluded backspace.
            other_coords = model.joint_q.numpy().shape[0] // 2
            excluded_q = NewtonManager.get_state_0().joint_q.numpy()[other_coords:].copy()
            command._solve_reset_pose(torch.arange(2, device=task.device))
            unchanged = np.array_equal(NewtonManager.get_state_0().joint_q.numpy()[other_coords:], excluded_q)
            _, _, target_len, typed_len, prefix_len = command._oversample(64)
            reachable = ((typed_len[::2] == prefix_len[::2]) & (typed_len[::2] < target_len[::2])).all()
            assert unchanged and reachable, f"Excluded world unchanged: {unchanged}; targets reachable: {reachable}"
        finally:
            env.close()


def test_selected_tip_jacobian_matches_finite_difference(selections):
    from isaaclab_tasks.contrib.keyboard.mdp.reset import KeyboardResetIKCfg, tip_jacobian

    model = selections.model
    state = model.state()
    ik = KeyboardResetIKCfg(
        joints=resolve_selection(
            selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, ".*/hinge0", count_per_world=1)
        ),
        dofs=resolve_selection(
            selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_DOF, ".*/hinge0", count_per_world=1)
        ),
        body=resolve_selection(
            selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/tip0", count_per_world=1)
        ),
        tip_offset=(0.3, 0.2, -0.1),
    )
    selections.state = state
    ids = ik.joints.dense_ids()
    wp.to_torch(state.joint_q)[ids] = 0.4
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    output = wp.zeros((2, 6, 1), dtype=wp.float32, device=model.device)
    analytic = tip_jacobian(ik, output).cpu().numpy().copy()
    samples = []
    epsilon = 0.001
    for delta in (-epsilon, epsilon):
        wp.to_torch(state.joint_q)[ids] = 0.4 + delta
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        poses = ik.body.dense(state.body_q).cpu().numpy()[:, 0]
        samples.append(np.array([wp.transform_point(wp.transform(*pose), wp.vec3(*ik.tip_offset)) for pose in poses]))
    derivative = (samples[1] - samples[0]) / (2 * epsilon)
    np.testing.assert_allclose(analytic[:, :3, 0], derivative, atol=3e-5)


@pytest.mark.parametrize("device,nested_task", [("cpu", False), ("cuda:0", False), ("cuda:0", True)])
def test_fixed_root_writer_masks_frames_and_stream_order(device, nested_task):
    from types import SimpleNamespace

    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_tasks.contrib.keyboard.mdp.reset import write_fixed_root_poses

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    builder = newton.ModelBuilder()
    for world in range(3):
        builder.begin_world()
        for column in range(2):
            body = builder.add_link(label=f"/{world}/root{column}", mass=1.0)
            child = wp.transform((0.02, -0.03, 0.04), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3))
            builder.add_articulation([builder.add_joint_fixed(-1, body, child_xform=child)])
        builder.end_world()
    model = builder.finalize(device)
    owner = NewtonSelections(model)
    owner.world_ids = torch.tensor([2, 0, 1], device=device)
    roots = resolve_selection(
        owner, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/root.*", count_per_world=2)
    )
    roots.dense_active()[2, 1] = False
    ids = torch.tensor([1, 2], device=device)
    storage = torch.zeros((2, 2, 14), device=device)
    poses = storage[..., ::2]  # Borrowed strided input, including q/-q rotations.
    poses[..., :3] = torch.tensor([0.1, -0.2, 0.3], device=device)
    poses[..., 3:] = torch.tensor([0.0, 0.0, 0.2955202, 0.9553365], device=device)
    poses[1, :, 3:] *= -1
    joint_ids = owner.root_joint_ids[roots.dense_ids()]
    initial = wp.to_torch(model.joint_X_p).clone()
    requested = torch.full((3,), -1, dtype=torch.long, device=device)
    requested[ids] = torch.arange(len(ids), device=device)
    selected = roots.dense_active() & (requested[owner.world_ids] >= 0)[:, None]
    local = poses[requested[owner.world_ids].clamp(min=0)]
    child = wp.to_torch(model.joint_X_c)[joint_ids]
    position, rotation = combine_frame_transforms(local[..., :3], local[..., 3:], child[..., :3], child[..., 3:])
    expected = initial.clone()
    expected[joint_ids[selected]] = torch.cat((position, rotation), dim=-1)[selected]
    calls = []
    env = SimpleNamespace(
        num_envs=3,
        device=device,
        notify_model_changed=lambda flags, rows, *, root_poses_only: calls.append((flags, rows, root_poses_only)),
        invalidate_fk=lambda rows: calls.append(rows),
    )
    observed = wp.empty_like(model.joint_X_p)
    wp.copy(observed, model.joint_X_p)
    rng = torch.get_rng_state().clone()
    wp.synchronize_device(device)
    producer = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.startswith("cuda") else nullcontext()
    with producer:
        outer = (
            wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))) if nested_task else nullcontext()
        )
        with outer:
            if device.startswith("cuda"):
                torch.cuda._sleep(2_000_000)
            delayed = torch.empty_like(storage)[..., ::2]
            delayed.copy_(poses)
            if nested_task:
                with warp_on_torch_stream(device):
                    write_fixed_root_poses(env, roots, ids, delayed)
                # Reproduce _demand/_failed readback after returning from a nested task scope.
                torch.cuda._sleep(200_000_000)
                wp.copy(observed, model.joint_X_p)
                readback = observed.numpy().copy()
            else:
                write_fixed_root_poses(env, roots, ids, delayed)
                # The consumer uses the restored prior Warp stream without an outer shared-stream scope.
                wp.copy(observed, model.joint_X_p)
    wp.synchronize_device(device)
    actual = wp.to_torch(observed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    if nested_task:
        np.testing.assert_allclose(readback, expected.cpu().numpy(), rtol=0, atol=2e-6)
    torch.testing.assert_close(actual[joint_ids[~selected]], initial[joint_ids[~selected]], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert calls[0][0] == newton.ModelFlags.JOINT_PROPERTIES and calls[0][1] is ids and calls[1] is ids
    assert calls[0][2] is True
    calls.clear()
    write_fixed_root_poses(env, roots, ids[:0], poses[:0])
    assert calls == []


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("grouped", [False, True])
def test_reset_kinematics_updates_only_requested_logical_worlds(device, grouped):
    from isaaclab_tasks.contrib.keyboard.mdp.reset import prepare_reset_kinematics

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    stream = (
        wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))
        if device.startswith("cuda")
        else nullcontext()
    )
    with stream:
        actors = ([4, 0], [1, 3, 2]) if grouped else ([0, 1],)
        cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/root", count_per_world=2)
        parts = []
        for logical_worlds in actors:
            builder = newton.ModelBuilder()
            global_body = builder.add_link(label="global", mass=1.0)
            builder.add_articulation([builder.add_joint_fixed(parent=-1, child=global_body)])
            for world in range(len(logical_worlds)):
                builder.begin_world()
                for articulation in ("robot", "keyboard"):
                    root = builder.add_link(label=f"/{world}/{articulation}/root", mass=1.0)
                    tip = builder.add_link(label=f"/{world}/{articulation}/tip", mass=1.0)
                    fixed = builder.add_joint_fixed(parent=-1, child=root)
                    hinge = builder.add_joint_revolute(parent=root, child=tip)
                    builder.add_articulation([fixed, hinge])
                builder.end_world()
            model = builder.finalize(device)
            state = model.state()
            # Unselected and global bodies must retain these non-FK values exactly.
            wp.to_torch(state.body_q)[:, 0] = torch.arange(model.body_count, device=device) + 100
            wp.to_torch(state.body_qd).fill_(7)
            owner = NewtonSelections(model, state=state)
            parts.append((resolve_selection(owner, cfg), torch.tensor(logical_worlds, device=device)))
        roots = (
            NewtonSelectionGroup(cfg.index_domain, parts, 5, policy_width=cfg.policy_width) if grouped else parts[0][0]
        )
        ids = torch.tensor([0, 3] if grouped else [1], device=device)
        kinematics = prepare_reset_kinematics(roots, ids)
        for angle in (0.25, 0.75):
            for (part, worlds), (model, state, mask) in zip(parts, kinematics, strict=True):
                before_q, before_qd = state.body_q.numpy().copy(), state.body_qd.numpy().copy()
                state.joint_q.fill_(angle)
                state.joint_qd.fill_(0.1)
                expected = model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, expected)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state, mask=mask)
                body_world = model.body_world.numpy()
                selected_worlds = torch.isin(worlds, ids).cpu().numpy()
                selected = (body_world >= 0) & selected_worlds[body_world.clip(min=0)]
                np.testing.assert_array_equal(state.body_q.numpy()[selected], expected.body_q.numpy()[selected])
                np.testing.assert_array_equal(state.body_qd.numpy()[selected], expected.body_qd.numpy()[selected])
                np.testing.assert_array_equal(state.body_q.numpy()[~selected], before_q[~selected])
                np.testing.assert_array_equal(state.body_qd.numpy()[~selected], before_qd[~selected])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("reverse_body_groups", [False, True])
def test_reset_ik_uses_current_bindings_without_retaining_a_graph(device, reverse_body_groups):
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.keyboard.mdp.commands.typing_commands import LetterTypingCommand
    from isaaclab_tasks.contrib.keyboard.mdp.reset import KeyboardResetIKCfg

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")

    def owner(count):
        builder = newton.ModelBuilder()
        for world in range(count):
            builder.begin_world()
            root = builder.add_link(label=f"/{world}/root", mass=1.0)
            tip = builder.add_link(label=f"/{world}/tip", mass=1.0)
            fixed = builder.add_joint_fixed(parent=-1, child=root)
            slider = builder.add_joint_prismatic(parent=root, child=tip, axis=(1, 0, 0), label=f"/{world}/slider")
            builder.add_articulation([fixed, slider])
            builder.end_world()
        result = NewtonSelections(builder.finalize(device))
        result.state = result.model.state()
        result.state.joint_q.fill_(-0.3)
        newton.eval_fk(result.model, result.state.joint_q, result.state.joint_qd, result.state)
        return result

    stream = (
        wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))
        if device.startswith("cuda")
        else nullcontext()
    )
    with stream:
        owners, actors = (owner(2), owner(1)), ([2, 0], [1])
        configs = (
            NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/root", count_per_world=1),
            NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/tip", count_per_world=1),
            NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, ".*/slider", count_per_world=1),
            NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_DOF, ".*/slider", count_per_world=1),
        )
        groups = [
            NewtonSelectionGroup(
                cfg.index_domain,
                [(resolve_selection(part, cfg), torch.tensor(ids, device=device)) for part, ids in zip(owners, actors)],
                3,
                policy_width=cfg.policy_width,
            )
            for cfg in configs
        ]
        if reverse_body_groups:
            groups[1].rebind(tuple(reversed(groups[1].native_bindings)))
        roots, bodies, coords, dofs = groups
        ik = KeyboardResetIKCfg(joints=coords, dofs=dofs, body=bodies)
        command = object.__new__(LetterTypingCommand)
        command._env = SimpleNamespace(
            num_envs=3, device=device, keyboard_variants=None, invalidate_fk=lambda _: None, forward=lambda: None
        )
        command.cfg = SimpleNamespace(
            robot_joints=coords,
            robot_dofs=dofs,
            reset_roots=roots,
            reset=SimpleNamespace(ik=ik, pre_solve_reset=None, ik_seed_joint_noise=0.0),
        )
        command._reset_ik, command._ik_iters = ik, (2, 2)
        command.target_len = torch.ones(3, dtype=torch.long, device=device)
        command._default_robot_q = torch.zeros((3, 1), device=device)
        command._ik_jacobian = wp.zeros((3, 6, 1), dtype=wp.float32, device=device)
        command._ik_limits = torch.tensor([-2.0, 2.0], device=device).expand(3, 1, 2)
        command._ik_offset, command._ik_hover = torch.zeros((3, 3), device=device), torch.zeros(3, device=device)
        target = torch.zeros((3, 3), device=device)
        target[:, 0] = torch.tensor([0.1, 0.2, 0.3], device=device)
        command.target_key_pos_w = lambda: target
        command._approach_target_quat = lambda quat: quat
        expected = torch.clamp(target[:, :1] / (1 + 0.05**2), -0.2, 0.2)
        expected += torch.clamp((target[:, :1] - expected) / (1 + 0.05**2), -0.2, 0.2)
        for selected in (torch.tensor([2], device=device), torch.arange(3, device=device)):
            old_q, old_bodies = coords.read_state("joint_q").clone(), bodies.read_state("body_q").clone()
            with patch.object(wp, "ScopedCapture", wraps=wp.ScopedCapture) as capture:
                command._solve_reset_pose(selected)
                assert capture.call_count == int(device.startswith("cuda"))
            q = coords.read_state("joint_q")
            torch.testing.assert_close(q[selected], expected[selected], atol=1e-7, rtol=0)
            untouched = torch.tensor(
                [i for i in range(3) if i not in selected.tolist()], device=device, dtype=torch.long
            )
            torch.testing.assert_close(q[untouched], old_q[untouched], atol=0, rtol=0)
            torch.testing.assert_close(bodies.read_state("body_q")[untouched], old_bodies[untouched], atol=0, rtol=0)
            assert not any(isinstance(value, wp.Graph) for value in vars(command).values())
        retired = [part.state.joint_q.numpy().copy() for part in owners]
        replacement = owner(3)
        for cfg, group in zip(configs, groups, strict=True):
            group.rebind([(resolve_selection(replacement, cfg), torch.tensor([1, 2, 0], device=device))])
        command._solve_reset_pose(torch.arange(3, device=device))
        torch.testing.assert_close(coords.read_state("joint_q"), expected, atol=1e-7, rtol=0)
        for part, old in zip(owners, retired, strict=True):
            np.testing.assert_array_equal(part.state.joint_q.numpy(), old)


def test_velocity_reduction_uses_compact_ragged_dofs(selections):
    from types import SimpleNamespace

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_tasks.contrib.keyboard.mdp.terminations import joint_vel_out_of_limit

    model = selections.model
    state = model.state()
    joints = resolve_selection(selections, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_DOF, ".*/hinge.*"))
    model.joint_velocity_limit.fill_(2.0)
    wp.to_torch(state.joint_qd)[joints.ids.numpy()[-2:]] = -3.0
    selections.state = state
    env = SimpleNamespace(num_envs=2, device=str(model.device))
    term = joint_vel_out_of_limit(TerminationTermCfg(func=joint_vel_out_of_limit, params={"joints": joints}), env)
    np.testing.assert_array_equal(term(env, joints).cpu().numpy(), [False, True])
    wp.to_torch(selections.world_active)[1] = False
    selections.refresh()
    np.testing.assert_array_equal(term(env, joints).cpu().numpy(), [False, False])


def test_bound_selection_does_not_serialize_symbolic_configuration(selections):
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.utils import class_to_dict

    from isaaclab_tasks.contrib.keyboard.mdp.observations import joint_pos

    cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, ".*/free", count_per_world=7)
    term = ObservationTermCfg(func=joint_pos, params={"joints": resolve_selection(selections, cfg)})
    assert class_to_dict(term)["params"]["joints"] == {}
    assert class_to_dict(cfg)["path"] == ".*/free"
    assert not hasattr(term.params["joints"], "path")


def test_keyboard_generator_supports_every_multiple_of_six():
    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_geometry import generate_keyboard
    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_pool import TYPING_KEYBOARD_VARIANTS

    layouts = [generate_keyboard(cfg) for cfg in TYPING_KEYBOARD_VARIANTS]
    assert {layout.active_key_count for layout in layouts} == set(range(6, 109, 6))
    for layout in layouts:
        assert layout.slot_count == 108 and layout.partition_count == 18 and layout.partition_dof == 6
        assert any(key.label.lower() in ("backspace", "bksp") for key in layout.active_keys)
        assert all(sum(key.active for key in layout.keys[p : p + 6]) in (0, 6) for p in range(0, 108, 6))
        assert not layout.warnings


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_grouped_selection_routes_exact_native_populations(device):
    """Different native widths gather/scatter by actor identity, without a synthetic state."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    sources, bindings, states = [], [], []
    actors = [torch.tensor([4, 0, 2], device=device), torch.tensor([1, 3], device=device)]
    for width, worlds in ((1, 3), (2, 2)):
        builder = newton.ModelBuilder()
        builder.begin_world()
        root = builder.add_link(label="/Robot/root", mass=1.0)
        joints = [builder.add_joint_fixed(parent=-1, child=root)]
        for column in range(width):
            body = builder.add_link(label=f"/Robot/link{column}", mass=1.0)
            joints.append(builder.add_joint_revolute(parent=root, child=body, label=f"/Robot/hinge{column}"))
        builder.add_articulation(joints)
        builder.end_world()
        source = NewtonSelections(builder.finalize(device))
        sources.append(source)
        model = source.model.replicate(worlds)
        state, control = model.state(), model.control()
        # A hot native clone must never read back model topology to bind task selections.
        with patch.object(wp.array, "numpy", side_effect=AssertionError("clone binding readback")):
            owner = NewtonSelections(model, state=state, control=control, source=source)
        bindings.append(owner)
        states.append(state)
    cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, "/Robot/hinge.*", policy_width=2)
    parts = [
        (resolve_selection(owner, cfg), world_id_by_env)
        for owner, world_id_by_env in zip(bindings, actors, strict=True)
    ]
    selected = NewtonSelectionGroup(cfg.index_domain, parts, 5, policy_width=cfg.policy_width)
    selected.write_state(
        "joint_q", torch.tensor([[1.0, 9.0], [2.0, 12.0], [3.0, 9.0], [4.0, 14.0], [5.0, 9.0]], device=device)
    )
    expected = torch.tensor([[1.0, 0.0], [2.0, 12.0], [3.0, 0.0], [4.0, 14.0], [5.0, 0.0]], device=device)
    torch.testing.assert_close(selected.read_state("joint_q"), expected)
    torch.testing.assert_close(wp.to_torch(states[0].joint_q), torch.tensor([5.0, 1.0, 3.0], device=device))
    selected.write_state(
        "joint_q", torch.tensor([[31.0, 32.0], [11.0, 99.0]], device=device), torch.tensor([3, 0], device=device)
    )
    expected[3] = torch.tensor([31.0, 32.0], device=device)
    expected[0, 0] = 11.0
    torch.testing.assert_close(selected.read_state("joint_q"), expected)
    selected.write_control("joint_target_q", expected)
    for part, world_id_by_env in parts:
        torch.testing.assert_close(
            wp.to_torch(part.owner.control.joint_target_q), expected[world_id_by_env, : part.width].flatten()
        )
    bodies_cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, "/Robot/.*", policy_width=3)
    bodies = NewtonSelectionGroup(
        bodies_cfg.index_domain,
        [(resolve_selection(owner, bodies_cfg), rows) for owner, rows in zip(bindings, actors, strict=True)],
        5,
        policy_width=bodies_cfg.policy_width,
    )
    poses = bodies.read_state("body_q")
    assert poses.shape == (5, 3, 7)
    assert not poses[[0, 2, 4], 2].any()
    from isaaclab_tasks.contrib.keyboard.mdp.observations import key_positions_b

    root_cfg = NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, "/Robot/root")
    root_parts = [(resolve_selection(owner, root_cfg), rows) for owner, rows in zip(bindings, actors, strict=True)]
    roots = NewtonSelectionGroup(newton.Model.AttributeFrequency.BODY, root_parts, 5)
    for state in states:
        wp.to_torch(state.body_q)[:, 0] = torch.arange(len(state.body_q), device=device)
    for selected_bodies, selected_roots in (
        (bodies, roots),
        *((part, root) for (part, _), (root, _) in zip(bodies.native_bindings, roots.native_bindings, strict=True)),
    ):
        expected_positions = (
            selected_bodies.read_state("body_q")[..., :3] - selected_roots.read_state("body_q")[..., :3]
        )
        expected_positions = torch.where(selected_bodies.dense_active()[..., None], expected_positions, 0)
        if selected_bodies is bodies:
            group_positions = expected_positions.flatten(1)
        torch.testing.assert_close(
            key_positions_b(SimpleNamespace(device=device), selected_bodies, selected_roots),
            expected_positions.flatten(1),
        )
        assert selected_bodies.dense_shape == expected_positions.shape[:2]
        with pytest.raises(ValueError, match="body selection"):
            selected.pose_field("state", "body_q")
        with pytest.raises(TypeError, match="transform"):
            selected_bodies.pose_field("model", "body_mass")
    # Reacquisition observes a new logical placement; no dense pose is cached.
    bodies.rebind(tuple(reversed(bodies.native_bindings)))
    roots.rebind(tuple(reversed(roots.native_bindings)))
    torch.testing.assert_close(key_positions_b(SimpleNamespace(device=device), bodies, roots), group_positions)
    from isaaclab.utils import class_to_dict

    assert class_to_dict(selected) == {}
    assert class_to_dict(cfg)["path"] == "/Robot/hinge.*"


@pytest.mark.parametrize("top_scale", [0.45, 0.9, 1.0])
def test_tapered_keycap_visual_faces_point_outward(top_scale):
    """A closed visual cap must survive ordinary back-face culling on every side."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_usd import _define_tapered_keycap

    stage = Usd.Stage.CreateInMemory()
    size = (0.017, 0.023, 0.009)
    prim = _define_tapered_keycap(stage, "/cap", size, top_scale, (0.7, 0.7, 0.7))
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    quads = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    a, b, c = points[triangles].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    assert np.all(np.einsum("ij,ij->i", normals, (a + b + c) / 3.0) > 0)
    signed_volume = np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0
    expected_volume = np.prod(size) * (1.0 + top_scale + top_scale**2) / 3.0
    assert signed_volume == pytest.approx(expected_volume, rel=2e-7)
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)


@pytest.mark.parametrize("partition_mode", ["single", "fixed_dof"])
def test_keyboard_has_no_internal_contacts(partition_mode):
    """Partition boundaries must not enable collisions between keys and their own case."""
    from newton.solvers import SolverMuJoCo

    from pxr import Usd, UsdGeom

    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_pool import TYPING_KEYBOARD_VARIANTS
    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_usd import spawn_keyboard

    cfg = replace(TYPING_KEYBOARD_VARIANTS[0], partition_mode=partition_mode)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    spawn_keyboard("/Keyboard", cfg, stage=stage)
    builder = newton.ModelBuilder(up_axis="Z")
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage, root_path="/Keyboard", load_visual_shapes=False)
    # An external probe overlaps the case; only keyboard-internal pairs should be excluded.
    pose = wp.transform_multiply(builder.body_q[builder.shape_body[0]], builder.shape_transform[0])
    probe = builder.add_link(xform=pose, mass=1.0, label="probe")
    joint = builder.add_joint_free(probe)
    builder.add_articulation([joint])
    builder.add_shape_sphere(probe, radius=0.01, label="probe_collision")
    solver = SolverMuJoCo(builder.finalize("cpu"), use_mujoco_cpu=True, use_mujoco_contacts=True)
    assert solver.mj_data.ncon > 0
    for contact in solver.mj_data.contact:
        assert any("probe_collision" in solver.mj_model.geom(int(geom)).name for geom in contact.geom)


@pytest.mark.parametrize("source_case", ["none", "shared", "variant_change", "destination_change"])
def test_keyboard_host_sources_skip_only_proven_identical_publication(monkeypatch, source_case):
    import isaaclab_tasks.contrib.keyboard.keyboard_variants as module

    class Resource:
        def __eq__(self, other):
            raise AssertionError("Resource equality cannot prove allocation identity")

    reference = None if source_case == "none" else Resource()
    other = Resource()
    registered = [reference, other if source_case == "variant_change" else reference]
    initial = [reference, other if source_case == "destination_change" else reference]
    suffix = "parts/part_000/base_link"
    shape_suffix, joint_suffix = suffix + "/shape", suffix + "/joint"

    def array(values):
        return wp.array(values, dtype=wp.int32, device="cpu")

    model = newton.Model("cpu")
    model.__dict__.update(
        body_label=[f"/world{i}/Keyboard/{suffix}" for i in range(2)],
        shape_label=[f"/world{i}/Keyboard/{shape_suffix}" for i in range(2)],
        joint_label=[f"/world{i}/Keyboard/{joint_suffix}" for i in range(2)],
        body_world=array([0, 1]),
        shape_world=array([0, 1]),
        shape_body=array([0, 1]),
        shape_type=array([int(newton.GeoType.BOX)] * 2),
        shape_flags=array([0, 0]),
        shape_source=initial.copy(),
        use_coord_layout_targets=True,
    )
    sources = [
        SimpleNamespace(
            body_label=[f"/Keyboard/{suffix}"],
            shape_label=[f"/Keyboard/{shape_suffix}"],
            joint_label=[f"/Keyboard/{joint_suffix}"],
            shape_type=array([int(newton.GeoType.BOX)]),
            shape_flags=array([0]),
            shape_source=[resource],
            joint_qd_start=array([0]),
        )
        for resource in registered
    ]
    remaining = iter(sources)
    builder = SimpleNamespace(
        begin_world=lambda: None,
        end_world=lambda: None,
        add_usd=lambda *a, **k: None,
        finalize=lambda device: next(remaining),
    )
    selection = SimpleNamespace(dense_ids=lambda: torch.tensor([[0], [1]]), joint_ids=array([0, 1]))
    layout = SimpleNamespace(
        partition_mode="fixed_dof",
        partition_dof=6,
        partition_count=18,
        active_key_count=6,
        keys=[SimpleNamespace(label="backspace", slot=0)],
    )
    layout.active_keys = layout.keys
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        selections=SimpleNamespace(
            body_active=wp.ones(2, dtype=wp.bool, device="cpu"),
            world_active=wp.ones(2, dtype=wp.bool, device="cpu"),
            refresh=lambda: None,
        ),
        sim=SimpleNamespace(physics_manager=SimpleNamespace(create_builder=lambda **kwargs: builder)),
        cfg=SimpleNamespace(
            cache_keyboard_constants=False,
            sim=SimpleNamespace(physics=SimpleNamespace(load_visual_shapes=True)),
            commands=SimpleNamespace(typing=SimpleNamespace(keys=selection, key_dofs=selection)),
        ),
        _property_world_mask=wp.zeros(3, dtype=wp.bool, device="cpu"),
    )
    state = SimpleNamespace(joint_q=wp.zeros(2, device="cpu"), joint_qd=wp.zeros(2, device="cpu"))
    control = SimpleNamespace(
        **{name: wp.zeros(2, device="cpu") for name in ("joint_target_q", "joint_target_qd", "joint_f")}
    )
    monkeypatch.setattr(module, "generate_keyboard", lambda cfg: layout)
    monkeypatch.setattr(module, "spawn_keyboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "replace_newton_builder_shape_colors", lambda *args: None)
    for name, value in (("get_model", model), ("get_state_0", state), ("get_control", control)):
        monkeypatch.setattr(module.NewtonManager, name, lambda value=value: value)
    for name in ("notify_model_changed", "set_body_sleep_policy", "invalidate_fk"):
        monkeypatch.setattr(module.NewtonManager, name, lambda *args, **kwargs: None)
    bank = module.KeyboardVariants(env, (SimpleNamespace(uniform_key_shapes=True),) * 2)
    if source_case in ("none", "shared"):

        def forbid_readback(self):
            raise AssertionError("Unchanged host descriptors must not read back reset indices")

        monkeypatch.setattr(torch.Tensor, "tolist", forbid_readback)
    bank.apply(torch.tensor([1]), torch.tensor([1]))
    assert model.shape_source[0] is initial[0]
    assert model.shape_source[1] is registered[1]
    bank.apply(torch.tensor([1]), torch.tensor([0]))
    assert model.shape_source[0] is initial[0]
    assert model.shape_source[1] is registered[0]


@pytest.mark.parametrize("use_graph", [False, True])
def test_keyboard_variant_reset_restores_geometry_inertia_and_sleep(use_graph):
    if not wp.is_cuda_available():
        pytest.skip("MJWarp requires CUDA")
    import gymnasium as gym
    import mujoco
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.sim.physics.use_cuda_graph = use_graph
    cfg.sim.physics.load_visual_shapes = True
    cfg.keyboard_variants = (cfg.keyboard_variants[0], cfg.keyboard_variants[1], cfg.keyboard_variants[-1])
    cfg.commands.typing.reset.buffer_size = 9
    with launch_simulation(cfg, {"headless": True}):
        env = gym.make("IsaacContrib-Keyboard-SO101", cfg=cfg)
        try:
            task = env.unwrapped
            env.reset()
            task.reset_keyboard([0, 1], [0, 2])
            bank = task.keyboard_variants
            model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
            assert model.articulation_count == 2 * 19  # robot + 18 keyboard partitions
            command = task.command_manager.get_term("typing")
            bodies = bank.body_ids[0].cpu().numpy()
            shapes = bank.shape_ids[0]
            key_bodies = command.key_bodies.dense_ids()[0].cpu().numpy()
            q_ids = command.key_joints.dense_ids()[0].cpu().numpy()
            qd_ids = command.cfg.key_dofs.dense_ids()[0].cpu().numpy()
            fields = ("body_mass", "body_com", "body_inertia", "body_inv_mass", "body_inv_inertia")
            shape_fields = (
                "shape_scale",
                "shape_transform",
                "shape_source_ptr",
                "shape_collision_radius",
                "shape_flags",
            )
            original = {name: getattr(model, name).numpy()[bodies].copy() for name in fields}
            original.update({name: getattr(model, name).numpy()[shapes].copy() for name in shape_fields})
            pointers = (state.joint_q.ptr, model.body_inertia.ptr, model.shape_source_ptr.ptr)
            other_q = command.cfg.reset_coords.dense_ids()[1].cpu().numpy()
            other_bodies = np.flatnonzero(model.body_world.numpy() == 1)
            neighbor = {
                name: getattr(state, name).numpy()[ids].copy()
                for name, ids in (("joint_q", other_q), ("body_q", other_bodies))
            }
            # A previous variant's COM must never survive the reset, even when the new COM is zero.
            wp.to_torch(model.body_com)[key_bodies[0]] = torch.tensor((0.1, -0.2, 0.3), device=task.device)
            task.reset_keyboard([0], [1])
            np.testing.assert_array_equal(state.joint_q.numpy()[other_q], neighbor["joint_q"])
            np.testing.assert_array_equal(state.body_q.numpy()[other_bodies], neighbor["body_q"])
            assert command.key_joints.dense_active().sum(dim=1).tolist() == [6, 108]
            assert not np.array_equal(model.body_inertia.numpy()[bodies], original["body_inertia"])
            assert not np.array_equal(model.shape_source_ptr.numpy()[shapes], original["shape_source_ptr"])
            np.testing.assert_allclose(model.body_com.numpy()[key_bodies[0]], [0, 0, 0], atol=1e-12)
            solver = NewtonManager._solver
            body_map = solver.mjc_body_to_newton.numpy()[0]
            mj_body = int(np.flatnonzero(body_map == key_bodies[0])[0])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, solver.mjw_model.body_iquat.numpy()[0, mj_body].astype(float))
            rotation = rotation.reshape(3, 3)
            tensor = rotation @ np.diag(solver.mjw_model.body_inertia.numpy()[0, mj_body]) @ rotation.T
            np.testing.assert_allclose(tensor, model.body_inertia.numpy()[key_bodies[0]], atol=1e-10, rtol=1e-5)
            np.testing.assert_allclose(
                solver.mjw_model.body_mass.numpy()[0, mj_body], model.body_mass.numpy()[key_bodies[0]]
            )
            disabled_tree_ids = solver.mj_model.body_treeid[np.isin(body_map, key_bodies[6:])]
            assert np.all(disabled_tree_ids >= 0)
            np.testing.assert_array_equal(solver.mjw_model.tree_sleep_policy.numpy()[0, disabled_tree_ids], 6)
            # Force on an excluded key cannot move it, including after graph replay.
            wp.to_torch(NewtonManager.get_control().joint_f)[qd_ids[6:]] = 100.0
            for _ in range(4):
                obs, reward, *_ = env.step(torch.zeros_like(task.action_manager.action))
                assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            np.testing.assert_array_equal(state.joint_q.numpy()[q_ids[6:]], 0)
            np.testing.assert_array_equal(state.joint_qd.numpy()[qd_ids[6:]], 0)
            assert torch.count_nonzero(obs["perception"].reshape(2, 108, 3)[0, 6:]) == 0
            contacts = NewtonManager.get_contacts()
            n = int(contacts.rigid_contact_count.numpy()[0])
            disabled_shapes = set(shapes[model.shape_flags.numpy()[shapes] == 0])
            assert not disabled_shapes.intersection(contacts.rigid_contact_shape0.numpy()[:n])
            assert not disabled_shapes.intersection(contacts.rigid_contact_shape1.numpy()[:n])
            for _ in range(2):
                task.reset_keyboard([0], [0])
                for name in fields:
                    np.testing.assert_array_equal(getattr(model, name).numpy()[bodies], original[name])
                for name in shape_fields:
                    np.testing.assert_array_equal(getattr(model, name).numpy()[shapes], original[name])
                assert command.key_joints.dense_active()[0].all()
                assert np.all(solver.mjw_model.tree_sleep_policy.numpy()[0, disabled_tree_ids] != 6)
                task.reset_keyboard([0], [1])
            task.reset_keyboard([0], [2])  # Different meshes and inertia with the same 108-key capacity.
            assert command.key_joints.dense_active()[0].sum() == 108
            assert not np.array_equal(model.shape_source_ptr.numpy()[shapes], original["shape_source_ptr"])
            assert not np.array_equal(model.body_inertia.numpy()[bodies], original["body_inertia"])
            assert (state.joint_q.ptr, model.body_inertia.ptr, model.shape_source_ptr.ptr) == pointers
            sources = command._sample_sources(torch.arange(2, device=task.device))
            for world, snapshot in enumerate(sources.tolist()):
                if snapshot >= 0:
                    assert command._buf_variant[snapshot] == bank.variant_ids[world]
            with pytest.raises(ValueError, match="outside"):
                task.reset_keyboard([0], [len(bank.layouts)])
            # Ordinary episode resets must choose a different registered keyboard for every reset world.
            for _ in range(8):
                previous = bank.variant_ids.clone()
                env.reset()
                assert torch.all(bank.variant_ids != previous)
            previous = bank.variant_ids.clone()
            task._reset_idx([0])
            assert bank.variant_ids[0] != previous[0] and bank.variant_ids[1] == previous[1]
            wp.to_torch(task.selections.world_active)[0] = False
            task.reset_keyboard([0], [0])
            assert not command.key_joints.dense_active()[0].any()
            np.testing.assert_array_equal(model.shape_flags.numpy()[shapes], 0)
        finally:
            env.close()


@pytest.mark.parametrize("retirement", ["selection", "runtime", "quarantine"])
def test_native_fields_follow_handles_strides_generations_and_reference_coordinates(retirement):
    from dataclasses import replace
    from types import SimpleNamespace

    from newton._src.solvers.mujoco.worlds import _MuJoCoWorldPopulation
    from newton.solvers import MuJoCoModelMapping, MuJoCoWorldPopulation, MuJoCoWorlds

    from isaaclab_tasks.contrib.keyboard.mujoco_selection import MuJoCoSelections, validate_native_mapping

    metadata, mappings, groups, buffers = [], [], [], []
    for prototype, count in enumerate((1, 2)):
        builder = newton.ModelBuilder()
        builder.begin_world()
        joints = []
        for index in range(count):
            body = builder.add_link(label=f"/body{index}", mass=1.0)
            joints.append(builder.add_joint_prismatic(parent=-1, child=body, label=f"/joint{index}"))
        builder.add_articulation(joints)
        builder.end_world()
        model = builder.finalize("cpu")
        model.mujoco = SimpleNamespace(dof_ref=wp.array([0.25] * count, dtype=float, device="cpu"))
        owner = NewtonSelections(model)
        metadata.append(owner)
        reverse = list(reversed(range(count)))
        native_model = SimpleNamespace(nbody=count + 1, nq=count, nv=count)
        mapping = MuJoCoModelMapping(
            model,
            native_model,
            np.array([reverse], dtype=np.int32),
            np.array([reverse], dtype=np.int32),
            np.array([[-1, *reverse]], dtype=np.int32),
            np.empty((1, 0), dtype=np.int32),
            np.full((1, count), 0.25, dtype=np.float32),
            np.array([*reverse, *[-1] * count], dtype=np.int32),
            np.array([*[-1] * count, *reverse], dtype=np.int32),
            np.full(count * 2, -1, dtype=np.int32),
            np.full(count * 2, -1, dtype=np.int32),
            np.full(count * 2, -1, dtype=np.int32),
            np.array([[*[0.25] * count, *[0.0] * count]], dtype=np.float32),
        )
        mappings.append(mapping)
        empty = np.empty(0, dtype=np.int32)
        no_actuators = replace(
            mapping,
            newton_target_by_position_actuator=empty,
            newton_dof_by_velocity_actuator=empty,
            newton_control_by_direct_actuator=empty,
            newton_joint_by_ball_actuator=empty,
            axis_by_actuator=empty,
            position_references=np.empty((1, 0), dtype=np.float32),
        )
        validate_native_mapping(owner, no_actuators)
        with pytest.raises(ValueError, match="valid authored joints"):
            validate_native_mapping(owner, replace(no_actuators, newton_joint_by_mujoco_mocap=np.array([[-1]])))
        with pytest.raises(ValueError, match="unique actuator"):
            validate_native_mapping(
                owner,
                replace(
                    mapping,
                    newton_target_by_position_actuator=np.zeros(count * 2, dtype=np.int32),
                    newton_dof_by_velocity_actuator=np.full(count * 2, -1, dtype=np.int32),
                ),
            )
        physical = wp.array(np.full((2, 8), -999, np.float32), device="cpu")
        values = wp.array(ptr=physical.ptr, shape=(2, count), strides=physical.strides, dtype=float, device="cpu")
        values.assign(np.arange(2 * count, dtype=np.float32).reshape(2, count) + 10 * (prototype + 1) + 0.25)
        buffers.append(physical)
        data = SimpleNamespace(
            qpos=values,
            qvel=wp.full((2, count), 3.0, device="cpu"),
            ctrl=wp.full((2, count * 2), -7.0, device="cpu"),
            qfrc_applied=wp.full((2, count), -8.0, device="cpu"),
            xpos=wp.full((2, count + 1), wp.vec3(1, 2, 3), dtype=wp.vec3, device="cpu"),
            xquat=wp.full((2, count + 1), wp.quat(1, 0, 0, 0), dtype=wp.quat, device="cpu"),
        )
        live, ready, contact_ready, ccd_ready = (wp.array([count], dtype=int, device="cpu") for count in (2, 2, 3, 1))
        owner = _MuJoCoWorldPopulation(
            native_model,
            prototype_index=prototype,
            data=data,
            world_storage=SimpleNamespace(capacity=2, protected_count=live, ready_count=ready),
            contact_storage=SimpleNamespace(capacity=4, ready_count=contact_ready),
            contact_count=contact_ready,
            ccd_count=ccd_ready,
            count_parameters=tuple(wp.CountParameter(n) for n in (2, 4, 1)),
        )
        owner.view = MuJoCoWorldPopulation(
            prototype, native_model, data, 2, 4, 1, live, ready, contact_ready, ccd_ready, weakref.ref(owner)
        )
        groups.append(owner.view)
        buffers.append(owner)
    directory = InstanceDirectoryData()
    for name, array in dict(
        prototype=wp.array([1, -1, 0], dtype=int, device="cpu"),
        slot=wp.array([1, -1, 0], dtype=int, device="cpu"),
        generation=wp.array([7, 0, 5], dtype=wp.uint64, device="cpu"),
        slot_starts=wp.array([0, 4, 8], dtype=int, device="cpu"),
        slot_id=wp.array([2, -1, -1, -1, -1, 0, -1, -1], dtype=int, device="cpu"),
        free_slot_count=wp.zeros(2, dtype=int, device="cpu"),  # No free slots is not an unreadable population.
    ).items():
        setattr(directory, name, array)
    actors = wp.array([2, 0, 1], dtype=int, device="cpu")
    generations = wp.array([5, 7, 0], dtype=wp.uint64, device="cpu")
    # Use the production public lifetime guard without allocating a GPU physics population.
    runtime = MuJoCoWorlds(
        device=wp.get_device("cpu"),
        populations=tuple(groups),
        directory=directory,
        _directory=SimpleNamespace(data=directory),
        _populations=buffers[1::2],
    )
    selections = MuJoCoSelections(
        tuple(metadata),
        tuple(mappings),
        runtime,
        actors,
        generations,
        num_envs=3,
        device="cpu",
    )
    # Nested device descriptors cannot safely embed pointers from another device.
    for owner in (runtime, *(item.model for item in metadata)):
        with (
            patch.object(owner, "device", object()),
            patch.object(wp, "ones", side_effect=AssertionError("Foreign owner reached device allocation")),
            patch.object(wp, "array", side_effect=AssertionError("Foreign owner reached device allocation")),
            pytest.raises(ValueError, match="same device"),
        ):
            MuJoCoSelections(
                tuple(metadata), selections.mappings, runtime, actors, generations, num_envs=3, device="cpu"
            )
    swapped = copy.copy(groups[0])
    swapped.model = copy.copy(groups[0].model)  # Same shape/topology metadata, different prepared owner.
    with pytest.raises(ValueError, match="changed|exact runtime population model"):
        MuJoCoSelections(
            tuple(metadata),
            selections.mappings,
            replace(runtime, populations=(swapped, groups[1])),
            actors,
            generations,
            num_envs=3,
            device="cpu",
        )
    selected = selections.bind(newton.Model.AttributeFrequency.JOINT_COORD, ((0,), (0, 1)), policy_width=2)
    # New consumers must reject replaced borrowed descriptors before allocating their field tables.
    with (
        patch.object(groups[0], "data", copy.copy(groups[0].data)),
        patch.object(wp, "array", side_effect=AssertionError("Invalid descriptors reached field allocation")),
        pytest.raises(ValueError, match="descriptors changed"),
    ):
        selected.scalar_field("state", "joint_q")
    assert copy.deepcopy(selected) is selected
    from isaaclab.utils import class_to_dict

    assert class_to_dict(selected) == {}
    assert selected.prototype_selection_counts == (1, 2) and selected.width == 2
    assert not hasattr(selected, "path") and not hasattr(selected, "joint_ids")
    np.testing.assert_array_equal(selected.active_counts().numpy(), [1, 2, 0])
    np.testing.assert_allclose(selected.read_state("joint_q").numpy(), [[10, 0], [23, 22], [0, 0]])
    np.testing.assert_array_equal(selected.dense_active().numpy(), [[True, False], [True, True], [False, False]])
    selected.write_state("joint_q", torch.tensor([[40.0, 41.0]]), torch.tensor([1]))
    np.testing.assert_allclose(groups[1].data.qpos.numpy()[1], [41.25, 40.25])
    selected.write_control("joint_target_q", torch.tensor([[5.0, 6.0]]), torch.tensor([1]))
    np.testing.assert_allclose(groups[1].data.ctrl.numpy()[1], [6.25, 5.25, -7, -7])
    dofs = selections.bind(newton.Model.AttributeFrequency.JOINT_DOF, ((0,), (0, 1)), policy_width=2)
    for invalid in (
        lambda: dofs.read_model("body_mass"),
        lambda: selected.read_model("joint_target_ke"),
        lambda: selected.read_state("joint_qd"),
        lambda: dofs.write_control("joint_target_q", torch.zeros((3, 2))),
        lambda: selected.write_control("joint_f", torch.zeros((3, 2))),
    ):
        with patch.object(wp, "launch", side_effect=AssertionError("Mismatched field was launched")):
            with pytest.raises(ValueError, match="index domain"):
                invalid()
    dofs.write_control("joint_target_qd", torch.tensor([[7.0, 8.0]]), torch.tensor([1]))
    dofs.write_control("joint_f", torch.tensor([[9.0, 10.0]]), torch.tensor([1]))
    np.testing.assert_allclose(groups[1].data.ctrl.numpy()[1], [6.25, 5.25, 8, 7])
    np.testing.assert_allclose(groups[1].data.qfrc_applied.numpy()[1], [10, 9])
    from isaaclab_tasks.contrib.keyboard.mdp.actions import _relative_joint_targets

    for owner in metadata:
        owner.model.joint_target_ke.fill_(20.0)
        owner.model.joint_target_kd.fill_(2.0)
        owner.model.joint_effort_limit.fill_(7.0)
    effort = wp.empty((3, 2), dtype=float, device="cpu")
    wp.launch(
        _relative_joint_targets,
        (3, 2),
        [
            selected.scalar_field("state", "joint_q"),
            dofs.scalar_field("state", "joint_qd"),
            dofs.scalar_field("model", "joint_target_ke"),
            dofs.scalar_field("model", "joint_target_kd"),
            dofs.scalar_field("model", "joint_effort_limit"),
            selected.scalar_field("control", "joint_target_q"),
            dofs.scalar_field("control", "joint_target_qd"),
            dofs.scalar_field("control", "joint_f"),
            wp.full((3, 2), 0.125, dtype=float, device="cpu"),
        ],
        [effort],
        device="cpu",
    )
    np.testing.assert_array_equal(effort.numpy(), [[-3.5, 0], [-3.5, -3.5], [0, 0]])
    np.testing.assert_array_equal(groups[1].data.ctrl.numpy()[1], [41.375, 40.375, 0, 0])
    np.testing.assert_array_equal(groups[1].data.qfrc_applied.numpy()[1], 0)
    bodies = selections.bind(newton.Model.AttributeFrequency.BODY, ((0,), (0, 1)), policy_width=2)
    with (
        patch.object(groups[0], "data", copy.copy(groups[0].data)),
        patch.object(wp, "array", side_effect=AssertionError("Invalid descriptors reached pose allocation")),
        pytest.raises(ValueError, match="descriptors changed"),
    ):
        bodies.pose_field("state", "body_q")
    assert bodies.dense_shape == (3, 2)
    np.testing.assert_allclose(bodies.read_state("body_q").numpy()[1], [[1, 2, 3, 0, 0, 0, 1]] * 2)
    np.testing.assert_allclose(bodies.read_model("body_q").numpy()[1], [[0, 0, 0, 0, 0, 0, 1]] * 2)
    from isaaclab_tasks.contrib.keyboard.mdp.observations import key_positions_b

    roots = selections.bind(newton.Model.AttributeFrequency.BODY, ((0,), (0,)), policy_width=1)
    wp.to_torch(groups[1].data.xpos)[1, 1, 0] = 1.25
    relative = key_positions_b(SimpleNamespace(device="cpu"), bodies, roots)
    expected_relative = torch.zeros(3, 6)
    expected_relative[1, 3] = 0.25
    torch.testing.assert_close(relative, expected_relative)
    probe_active, probe_values = wp.empty(5, dtype=bool, device="cpu"), wp.empty(5, dtype=wp.transform, device="cpu")
    wp.launch(
        _probe_pose_field,
        5,
        [
            bodies.pose_field("state", "body_q"),
            wp.array([-1, 3, 1, 1, 1], dtype=int, device="cpu"),
            wp.array([0, 0, -1, 2, 0], dtype=int, device="cpu"),
        ],
        [probe_active, probe_values],
        device="cpu",
    )
    np.testing.assert_array_equal(probe_active.numpy(), [False, False, False, False, True])
    np.testing.assert_array_equal(probe_values.numpy()[:4], 0)
    np.testing.assert_array_equal(probe_values.numpy()[4], [1, 2, 3, 0, 0, 0, 1])
    from mujoco_warp import ConeType, vec5

    for group, count in zip(groups, (1, 2)):
        group.model.opt = SimpleNamespace(cone=ConeType.PYRAMIDAL)
        group.model.geom_bodyid = wp.array([1, count], dtype=int, device="cpu")
    contact = SimpleNamespace(
        worldid=wp.array([1, 1, 1, -1], dtype=int, device="cpu"),
        geom=wp.array([[0, 1]] * 4, dtype=wp.vec2i, device="cpu"),
        frame=wp.array([np.eye(3), -np.eye(3), np.roll(np.eye(3), 1, axis=0), np.eye(3)], dtype=wp.mat33, device="cpu"),
        friction=wp.full(4, vec5(1.0), dtype=vec5, device="cpu"),
        dim=wp.full(4, 1, dtype=int, device="cpu"),
        efc_address=wp.array([[0], [1], [2], [3]], dtype=int, device="cpu"),
        adhesion=wp.zeros(4, dtype=float, device="cpu"),
    )
    groups[1].data.contact, groups[1].data.njmax = contact, 4
    groups[1].data.nacon = wp.array([4], dtype=int, device="cpu")
    groups[1].data.efc = SimpleNamespace(
        force=wp.array([[0.0, 0.0, 0.0, 0.0], [3.0, 5.0, 2.0, 99.0]], dtype=float, device="cpu")
    )
    with (
        patch.object(groups[0], "data", copy.copy(groups[0].data)),
        patch.object(wp, "zeros", side_effect=AssertionError("Invalid descriptors reached contact allocation")),
        pytest.raises(ValueError, match="descriptors changed"),
    ):
        bodies.prepare_contact_forces()
    bodies.prepare_contact_forces()
    inverse = wp.array([1, -1, 0], dtype=int, device="cpu")
    with pytest.raises(ValueError, match="exact runtime"):
        bodies.record_contact_forces(copy.copy(groups[1]), inverse)
    bodies.record_contact_forces(groups[1], inverse)
    # Source-body0 is nativebody2: normal vectors sum before the norm; friction is excluded.
    np.testing.assert_allclose(bodies.selected_net_normal_forces().numpy()[1], [[-2, 0, 2], [2, 0, -2]])
    # Prepared fields retain their descriptors. Reuse checks lifetime without scanning every prototype again.
    with patch(
        "isaaclab_tasks.contrib.keyboard.mujoco_selection.mujoco_world_population_validate",
        side_effect=AssertionError("Prepared selection repeated population preparation validation"),
    ):
        np.testing.assert_allclose(selected.read_state("joint_q").numpy(), [[10, 0], [40, 41], [0, 0]])
        np.testing.assert_array_equal(selected.dense_active().numpy(), [[True, False], [True, True], [False, False]])
        np.testing.assert_array_equal(
            selected.joint_types().numpy(), [[newton.JointType.PRISMATIC, 0], [newton.JointType.PRISMATIC] * 2, [0, 0]]
        )
        torch.testing.assert_close(key_positions_b(SimpleNamespace(device="cpu"), bodies, roots), expected_relative)
        bodies.prepare_contact_forces()
        np.testing.assert_allclose(bodies.selected_net_normal_forces().numpy()[1], [[-2, 0, 2], [2, 0, -2]])
    generations.assign(np.array([5, 8, 0], np.uint64))
    directory.generation.assign(np.array([8, 0, 5], np.uint64))
    np.testing.assert_array_equal(bodies.selected_net_normal_forces().numpy(), 0)
    directory.generation.assign(np.array([7, 0, 5], np.uint64))
    generations.assign(np.array([5, 6, 0], np.uint64))
    np.testing.assert_array_equal(selected.read_state("joint_q", fill=-1).numpy(), [[10, -1], [-1, -1], [-1, -1]])
    generations.assign(np.array([5, 7, 0], np.uint64))
    groups[1].world_storage_ready_count.fill_(1)
    assert not selected.dense_active()[1].any()
    torch.testing.assert_close(key_positions_b(SimpleNamespace(device="cpu"), bodies, roots), torch.zeros(3, 6))
    for physical, count in zip(buffers[::2], (1, 2)):
        np.testing.assert_array_equal(physical.numpy()[:, count:], -999)
    # A reset-only publication switches env0 and compacts env1 to another row.
    # Raw old contacts still mention row1, now env0; selected forces must not consume them again.
    directory.prototype.assign(np.array([1, -1, 1], dtype=np.int32))
    directory.slot.assign(np.array([0, -1, 1], dtype=np.int32))
    directory.slot_id.assign(np.array([-1, -1, -1, -1, 0, 2, -1, -1], dtype=np.int32))
    directory.generation.assign(np.array([7, 0, 6], dtype=np.uint64))
    generations.assign(np.array([6, 7, 0], dtype=np.uint64))
    groups[1].world_storage_ready_count.fill_(2)
    visible = bodies.selected_net_normal_forces().numpy()
    np.testing.assert_array_equal(visible[0], 0)
    np.testing.assert_allclose(visible[1], [[-2, 0, 2], [2, 0, -2]])
    values, env_ids = torch.zeros((1, 2)), torch.tensor([0])
    if retirement == "selection":
        selections.retire()
        selections.retire()
    elif retirement == "runtime":
        runtime._closed = True
    else:
        runtime._service_failed = True
    operations = (
        lambda: selections.bind(newton.Model.AttributeFrequency.JOINT_COORD, ((0,), (0, 1)), policy_width=2),
        selected.dense_active,
        selected.active_counts,
        selected.joint_types,
        lambda: selected.scalar_field("state", "joint_q"),
        lambda: selected.read_state("joint_q"),
        lambda: dofs.read_model("joint_target_ke"),
        lambda: bodies.read_state("body_q"),
        lambda: bodies.read_model("body_q"),
        lambda: bodies.pose_field("state", "body_q"),
        lambda: bodies.pose_field("model", "body_q"),
        lambda: selected.write_state("joint_q", values, env_ids),
        lambda: dofs.write_control("joint_f", values, env_ids),
        bodies.prepare_contact_forces,
        lambda: bodies.record_contact_forces(groups[1], inverse),
        bodies.selected_net_normal_forces,
    )
    with (
        patch.object(wp, "launch", side_effect=AssertionError("retired selection launched work")),
        patch.object(wp, "empty", side_effect=AssertionError("retired selection allocated work")),
        patch.object(torch, "_assert_async", side_effect=AssertionError("retired selection enqueued validation")),
    ):
        for operation in operations:
            with pytest.raises(RuntimeError, match="retired|closed|backing service failed"):
                operation()


def test_native_field_consumes_real_directory_local_slots_after_reset_and_compaction():
    from gpu_components import directory as instance_directory
    from gpu_components.directory_data import InstanceOperation

    from isaaclab_tasks.contrib.keyboard.mujoco_selection import (
        EnvWorldBindings,
        MuJoCoScalarField,
        _gather_scalars,
        _ScalarSource,
        _WorldReadiness,
    )

    directory = instance_directory.allocate((2, 3), id_capacity=2, command_capacity=2, device="cpu")
    instance_directory.publish_admissible_slots(directory, (2, 3))
    commands, results = (
        instance_directory.allocate_commands(2, device="cpu"),
        instance_directory.allocate_results(2, device="cpu"),
    )
    commands.operation.fill_(int(InstanceOperation.CREATE))
    commands.prototype.assign(np.array([0, 1], np.int32))
    commands.count.fill_(2)
    commands.sequence.fill_(1)
    instance_directory.begin(directory, commands)
    instance_directory.admit(directory, commands)
    directory.transaction.initialized_sequence.fill_(1)  # This scalar test domain has preinitialized every row below.
    instance_directory.publish(directory, commands, results)
    np.testing.assert_array_equal(results.status.numpy(), 0)
    world_id_by_env = results.instance_id.numpy()[::-1].copy()
    generations = results.generation.numpy()[::-1].copy()
    d = directory.data
    placement = EnvWorldBindings()
    placement.world_id_by_env = wp.array(world_id_by_env, dtype=int, device="cpu")
    placement.world_generation_by_env = wp.array(generations, dtype=wp.uint64, device="cpu")
    placement.env_participating = wp.ones(2, dtype=bool, device="cpu")
    placement.directory = d
    descriptors, capacities, arrays = [], [], []
    for prototype, rows in enumerate((2, 3)):
        source = _ScalarSource()
        source.values = wp.array((10 * (prototype + 1) + np.arange(rows, dtype=np.float32))[:, None], device="cpu")
        arrays.append(source.values)
        descriptors.append(source)
        capacity = _WorldReadiness()
        capacity.ready_world_count = wp.array([rows], dtype=int, device="cpu")
        capacities.append(capacity)
    placement.world_readiness_by_prototype = wp.array(capacities, dtype=_WorldReadiness, device="cpu")
    field = MuJoCoScalarField()
    field.env_world_bindings = placement
    field.sources = wp.array(descriptors, dtype=_ScalarSource, device="cpu")
    field.columns, field.offsets = (
        wp.zeros((2, 1), dtype=int, device="cpu"),
        wp.zeros((2, 1), dtype=float, device="cpu"),
    )
    field.element_participating = wp.ones((2, 1), dtype=bool, device="cpu")
    output = wp.empty((2, 1), dtype=float, device="cpu")
    wp.launch(_gather_scalars, (2, 1), [field, -1.0], [output], device="cpu")
    expected = np.array(
        [
            arrays[int(d.prototype.numpy()[identity])].numpy()[int(d.slot.numpy()[identity])]
            for identity in world_id_by_env
        ]
    )
    np.testing.assert_array_equal(output.numpy(), expected)
    assert d.slot_starts.numpy()[1] == 2 and 0 <= d.slot.numpy()[world_id_by_env[0]] < 3
    commands.count.fill_(1)
    commands.operation.assign(np.array([int(InstanceOperation.REPLACE), 0], np.int32))
    commands.instance_id.assign(np.array([world_id_by_env[1], -1], np.int32))
    commands.generation.assign(np.array([generations[1], 0], np.uint64))
    commands.prototype.fill_(1)
    commands.sequence.fill_(2)
    instance_directory.begin(directory, commands)
    instance_directory.admit(directory, commands)
    reset_value = arrays[1].numpy()[int(directory.transaction.destination_slot.numpy()[0])].copy()
    directory.transaction.initialized_sequence.fill_(2)
    instance_directory.publish(directory, commands, results)
    wp.launch(_gather_scalars, (2, 1), [field, -1.0], [output], device="cpu")
    np.testing.assert_array_equal(output.numpy(), [expected[0], [-1]])
    generations[1] = results.generation.numpy()[0]
    placement.world_generation_by_env.assign(generations)
    instance_directory.plan_compaction(directory)
    # Explicit test-domain relocation, with the directory's actual local source/destination rows.
    starts = d.slot_starts.numpy()
    for prototype, count in enumerate(directory.compaction.count.numpy()):
        src = directory.compaction.source_slots.numpy()[starts[prototype] : starts[prototype] + count]
        dst = directory.compaction.destination_slots.numpy()[starts[prototype] : starts[prototype] + count]
        if count:
            values = arrays[prototype].numpy()
            values[dst] = values[src]
            arrays[prototype].assign(values)
    directory.compaction.copied_count.assign(directory.compaction.count.numpy())
    instance_directory.publish_compaction(directory)
    wp.launch(_gather_scalars, (2, 1), [field, -1.0], [output], device="cpu")
    np.testing.assert_array_equal(output.numpy(), [expected[0], reset_value])
    instance_directory.close(directory, streams=())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compact_reset_chain_matches_newton_and_preserves_live_state(device):
    from isaaclab_tasks.contrib.keyboard.mdp.reset import ResetKinematics

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    builder = newton.ModelBuilder()
    builder.begin_world()
    bodies = [builder.add_link(mass=1.0) for _ in range(5)]
    joints = [builder.add_joint_fixed(parent=-1, child=bodies[0])]
    joints.append(
        builder.add_joint_revolute(
            parent=bodies[0],
            child=bodies[1],
            axis=(0, 0, 1),
            parent_xform=wp.transform((0.2, 0, 0), wp.quat_identity()),
        )
    )
    joints.append(
        builder.add_joint_fixed(
            parent=bodies[1], child=bodies[2], parent_xform=wp.transform((0.1, 0.2, 0), wp.quat_identity())
        )
    )
    joints.append(
        builder.add_joint_prismatic(
            parent=bodies[2],
            child=bodies[3],
            axis=(1, 0, 0),
            parent_xform=wp.transform((0.1, 0, 0.1), wp.quat_identity()),
            child_xform=wp.transform((0.02, 0, 0), wp.quat_identity()),
        )
    )
    joints.append(builder.add_joint_revolute(parent=bodies[1], child=bodies[4], axis=(0, 1, 0)))
    builder.add_articulation(joints)
    builder.end_world()
    model = builder.finalize(device)
    state = model.state()
    before_q, before_pose = state.joint_q.numpy().copy(), state.body_q.numpy().copy()
    workspace = ResetKinematics(model, [0, 1, 2], [1, 0], bodies[3], 3, (0.03, -0.02, 0.04))
    workspace.validate_model(model, [0, 1, 2], [1, 0], bodies[3])
    assert not any(isinstance(value, (newton.Model, newton.State)) for value in vars(workspace).values())
    q = torch.tensor([[0.4, 0.1, -0.3], [-0.5, 0.2, 0.9], [0.1, -0.2, 0.3]], device=device)
    root = torch.tensor([[0.4, -0.1, 0.2, 0, 0, 0, 1]] * 3, device=device)
    context = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))) if q.is_cuda else nullcontext()
    with context:
        pose, jacobian = workspace.evaluate(q, root)
        actual_pose, actual_jacobian = pose.cpu().numpy().copy(), jacobian.cpu().numpy().copy()
        np.testing.assert_array_equal(state.joint_q.numpy(), before_q)
        np.testing.assert_array_equal(state.body_q.numpy(), before_pose)
        for row in range(3):
            xp = model.joint_X_p.numpy()
            xp[0] = root[row].cpu().numpy()
            model.joint_X_p.assign(xp)
            state.joint_q.assign(q[row].cpu().numpy())
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            np.testing.assert_allclose(actual_pose[row], state.body_q.numpy()[bodies[3]], atol=1e-7, rtol=0)
            for column, coordinate in enumerate((1, 0)):
                tips = []
                for sign in (-1, 1):
                    shifted = q[row].cpu().numpy().copy()
                    shifted[coordinate] += sign * 1e-2
                    state.joint_q.assign(shifted)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    transform = wp.transform(*state.body_q.numpy()[bodies[3]])
                    tips.append(np.array(wp.transform_point(transform, wp.vec3(0.03, -0.02, 0.04))))
                np.testing.assert_allclose(
                    actual_jacobian[row, :3, column], (tips[1] - tips[0]) / 0.02, atol=3e-5, rtol=0
                )
        empty_pose, empty_jac = workspace.evaluate(q[:0], root[:0])
        assert empty_pose.shape == (0, 7) and empty_jac.shape == (0, 6, 2)
    with pytest.raises(ValueError, match="ancestors"):
        ResetKinematics(model, [0, 1, 2], [2], bodies[3], 3, (0, 0, 0))
    with pytest.raises(ValueError, match="layout"):
        workspace.evaluate(q.repeat(2, 1), root.repeat(2, 1))
    with pytest.raises(ValueError, match="identical"):
        workspace.validate_model(model, [0, 1, 2], [0, 1], bodies[3])


def test_root_pose_sampling_preserves_original_draws_and_inputs():
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

    from isaaclab_tasks.contrib.keyboard.mdp.reset import sample_root_poses

    poses = torch.tensor([[[0.2, 0.0, 0.1, 0, 0, 0, 1]] * 2] * 3)
    original = poses.clone()
    ranges = {"z": (0.015, 0.05), "roll": (0.0, 0.75), "yaw": (-0.1, 0.1)}
    torch.manual_seed(513)
    result = sample_root_poses(poses, ranges, {"x": (0.0, 0.0)})
    final_rng = torch.random.get_rng_state()
    torch.manual_seed(513)
    bounds = torch.tensor([ranges.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z", "roll", "pitch", "yaw")])
    sampled = sample_uniform(bounds[:, 0], bounds[:, 1], (3, 6), "cpu")
    expected = original.clone()
    expected[..., :3] += sampled[:, None, :3]
    delta = quat_from_euler_xyz(sampled[:, 3], sampled[:, 4], sampled[:, 5])
    expected[..., 3:] = quat_mul(expected[..., 3:], delta[:, None].expand_as(expected[..., 3:]))
    sample_uniform(0.0, 0.0, (3, 6), "cpu")
    assert torch.equal(final_rng, torch.random.get_rng_state())
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(poses, original, atol=0, rtol=0)


@wp.kernel
def _native_bounds_probe(
    field: MuJoCoScalarField,
    worlds: wp.array[int],
    slots: wp.array[int],
    active: wp.array[bool],
    values: wp.array[float],
):
    i = wp.tid()
    active[i] = native_field_active(field, worlds[i], slots[i])
    values[i] = native_field_read(field, worlds[i], slots[i])
    native_field_write(field, worlds[i], slots[i], 99.0)


def _native_scalar_selection(device="cpu"):
    """A real two-actor descriptor, without a physics engine or retained dense mirror."""
    from newton.solvers import MuJoCoWorlds

    placement = EnvWorldBindings()
    placement.world_id_by_env = wp.array([0, 1], dtype=int, device=device)
    placement.world_generation_by_env = wp.ones(2, dtype=wp.uint64, device=device)
    placement.env_participating = wp.ones(2, dtype=bool, device=device)
    placement.directory = InstanceDirectoryData()
    placement.directory.prototype = wp.zeros(2, dtype=int, device=device)
    placement.directory.slot = wp.array([0, 1], dtype=int, device=device)
    placement.directory.generation = wp.ones(2, dtype=wp.uint64, device=device)
    placement.directory.slot_starts = wp.array([0, 2], dtype=int, device=device)
    placement.directory.slot_id = wp.array([0, 1], dtype=int, device=device)
    capacity = _WorldReadiness()
    capacity.ready_world_count = wp.array([2], dtype=int, device=device)
    placement.world_readiness_by_prototype = wp.array([capacity], dtype=_WorldReadiness, device=device)
    source = _ScalarSource()
    source.values = wp.array([[3.0], [4.0]], dtype=float, device=device)
    source.broadcast_rows = 0
    field = MuJoCoScalarField()
    field.env_world_bindings = placement
    field.sources = wp.array([source], dtype=_ScalarSource, device=device)
    field.columns = wp.zeros((1, 1), dtype=int, device=device)
    field.offsets = wp.zeros((1, 1), dtype=float, device=device)
    field.element_participating = wp.ones((2, 1), dtype=bool, device=device)
    selected = MuJoCoSelection.__new__(MuJoCoSelection)
    selected.owner = object.__new__(MuJoCoSelections)
    selected.owner.device, selected.owner.num_envs = wp.get_device(device), 2
    selected.owner._env_indices = placement.world_id_by_env
    selected.owner._retired = False
    selected.owner.runtime = MuJoCoWorlds(
        device=wp.get_device(device),
        directory=placement.directory,
        _directory=SimpleNamespace(data=placement.directory),
    )
    selected.width = 1
    selected._fields = {("state", "joint_q"): field, ("control", "joint_f"): field}
    return selected, field, source, capacity


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_scalar_bad_env_index_and_slot_are_inert(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    selected, field, source, capacity = _native_scalar_selection(device)
    worlds = wp.array([-1, 2, 2**31 - 1, 0, 1, 1], dtype=int, device=device)
    slots = wp.array([0, 0, 0, -1, 1, 0], dtype=int, device=device)
    active = wp.empty(6, dtype=bool, device=device)
    values = wp.empty(6, dtype=float, device=device)
    wp.launch(_native_bounds_probe, 6, [field, worlds, slots], [active, values], device=device)
    np.testing.assert_array_equal(active.numpy(), [False, False, False, False, False, True])
    np.testing.assert_array_equal(values.numpy(), [0, 0, 0, 0, 0, 4])
    np.testing.assert_array_equal(source.values.numpy(), [[3], [99]])
    field.element_participating = wp.ones((2, 0), dtype=bool, device=device)
    field.columns = wp.zeros((1, 0), dtype=int, device=device)
    field.offsets = wp.zeros((1, 0), dtype=float, device=device)
    wp.launch(_native_bounds_probe, 6, [field, worlds, slots], [active, values], device=device)
    assert not active.numpy().any()
    np.testing.assert_array_equal(values.numpy(), 0)
    np.testing.assert_array_equal(source.values.numpy(), [[3], [99]])


@pytest.fixture(params=["mujoco", "newton", "newton_group"])
def writable_selection(request):
    """The same indexed-write contract holds at all three task selection boundaries."""
    if request.param == "mujoco":
        selected, field, source, capacity = _native_scalar_selection()
        yield selected, selected, source.values, source.values
        return
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(0.1))
        joint = builder.add_joint_prismatic(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize("cpu")
    owner = NewtonSelections(model, state=model.state(), control=model.control())
    owner.state.joint_q.assign(np.array([3, 4], dtype=np.float32))
    owner.control.joint_f.assign(np.array([3, 4], dtype=np.float32))
    coords, dofs = (
        owner.bind(newton.Model.AttributeFrequency.JOINT_COORD, [0, 1]),
        owner.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0, 1]),
    )
    if request.param == "newton_group":
        coords = NewtonSelectionGroup(newton.Model.AttributeFrequency.JOINT_COORD, ((coords, torch.arange(2)),), 2)
        dofs = NewtonSelectionGroup(newton.Model.AttributeFrequency.JOINT_DOF, ((dofs, torch.arange(2)),), 2)
    yield coords, dofs, owner.state.joint_q, owner.control.joint_f


@pytest.mark.parametrize("method,attribute", [("write_state", "joint_q"), ("write_control", "joint_f")])
@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([-1]),
        torch.tensor([2]),
        torch.tensor([2**32]),
        torch.tensor([0.5]),
        torch.tensor([True]),
        torch.tensor([[0]]),
        torch.tensor(0),
        torch.tensor([0, 0]),
        torch.tensor([0], device="meta"),
        [0],
    ],
)
def test_write_rejects_malformed_env_indices_before_mutation(writable_selection, method, attribute, ids):
    coords, dofs, state, control = writable_selection
    selected = coords if method == "write_state" else dofs
    count = len(ids) if isinstance(ids, torch.Tensor) and ids.ndim == 1 else 1
    with patch.object(wp, "launch", side_effect=AssertionError("Unsafe write was launched")):
        with pytest.raises((ValueError, RuntimeError)):
            getattr(selected, method)(attribute, torch.zeros((count, 1)), ids)
    np.testing.assert_array_equal(state.numpy().reshape(-1), [3, 4])
    np.testing.assert_array_equal(control.numpy().reshape(-1), [3, 4])


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1,)),
        torch.zeros((1, 1, 1)),
        torch.zeros((2, 1)),
        torch.zeros((1, 2)),
        torch.zeros((1, 1), device="meta"),
        [[0.0]],
    ],
)
def test_write_rejects_malformed_values_before_mutation(writable_selection, values):
    selected, _, state, control = writable_selection
    with patch.object(wp, "launch", side_effect=AssertionError("Unsafe write was launched")):
        with pytest.raises(ValueError):
            selected.write_state("joint_q", values, torch.tensor([0]))
    np.testing.assert_array_equal(state.numpy().reshape(-1), [3, 4])
    np.testing.assert_array_equal(control.numpy().reshape(-1), [3, 4])


def test_write_empty_and_valid_subset_leave_other_envs_unchanged(writable_selection):
    selected, dofs, state, control = writable_selection
    with patch.object(wp, "launch", side_effect=AssertionError("Empty write was launched")):
        selected.write_state("joint_q", torch.empty((0, 1)), torch.empty(0, dtype=torch.long))
    selected.write_state("joint_q", torch.tensor([[7.0]]), torch.tensor([1]))
    np.testing.assert_array_equal(state.numpy().reshape(-1), [3, 7])
    values = torch.tensor([[8.0], [9.0]])
    with (
        patch.object(torch, "_assert_async", side_effect=AssertionError("Full-domain writes need no index scan")),
        patch.object(torch, "zeros", side_effect=AssertionError("Full-domain writes need no index scratch")),
    ):
        dofs.write_control("joint_f", values)
    np.testing.assert_array_equal(control.numpy().reshape(-1), [8, 9])


@pytest.mark.parametrize("joint_kind", ["free", "ball"])
@pytest.mark.parametrize("coord_targets", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_field_schema_keeps_coordinate_dof_and_body_domains_distinct(monkeypatch, joint_kind, coord_targets, grouped):
    """Equal integer indices never make different entity domains interchangeable."""
    monkeypatch.setattr(newton, "use_coord_layout_targets", coord_targets)
    builder = newton.ModelBuilder()
    builder.begin_world()
    root = builder.add_link(mass=3.0, inertia=wp.mat33(0.1))
    child = builder.add_link(mass=5.0, inertia=wp.mat33(0.1))
    root_joint = (
        builder.add_joint_free(child=root) if joint_kind == "free" else builder.add_joint_ball(parent=-1, child=root)
    )
    hinge = builder.add_joint_revolute(parent=root, child=child)
    builder.add_articulation([root_joint, hinge])
    builder.end_world()
    model = builder.finalize("cpu")
    owner = NewtonSelections(model, state=model.state(), control=model.control())
    # The model snapshots this schema choice; a later process default is irrelevant.
    monkeypatch.setattr(newton, "use_coord_layout_targets", not coord_targets)
    index = 6 if joint_kind == "free" else 3  # Root quaternion scalar coordinate / hinge DOF.
    coords, dofs, first_dof = (
        owner.bind(newton.Model.AttributeFrequency.JOINT_COORD, [index]),
        owner.bind(newton.Model.AttributeFrequency.JOINT_DOF, [index]),
        owner.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0]),
    )
    if grouped:
        coords, dofs, first_dof = (
            NewtonSelectionGroup(part.index_domain, ((part, torch.tensor([0])),), 1)
            for part in (coords, dofs, first_dof)
        )
    value = torch.tensor([[0.375]])
    old_q, old_qd, old_f = (
        owner.state.joint_q.numpy().copy(),
        owner.state.joint_qd.numpy().copy(),
        owner.control.joint_f.numpy().copy(),
    )
    for invalid in (
        lambda: first_dof.read_model("body_mass"),
        lambda: first_dof.scalar_field("model", "body_mass"),
        lambda: coords.read_state("joint_qd"),
        lambda: dofs.read_state("joint_q"),
        lambda: coords.scalar_field("state", "joint_qd"),
        lambda: dofs.scalar_field("state", "joint_q"),
        lambda: coords.write_state("joint_qd", value),
        lambda: coords.write_state("joint_qd", torch.empty((0, 1)), torch.empty(0, dtype=torch.long)),
        lambda: dofs.write_state("joint_q", value),
        lambda: coords.write_control("joint_f", value),
    ):
        with pytest.raises(ValueError, match="index domain"):
            invalid()
    np.testing.assert_array_equal(owner.state.joint_q.numpy(), old_q)
    np.testing.assert_array_equal(owner.state.joint_qd.numpy(), old_qd)
    np.testing.assert_array_equal(owner.control.joint_f.numpy(), old_f)
    coords.write_state("joint_q", value)
    dofs.write_state("joint_qd", value * 2)
    dofs.write_control("joint_f", value * 3)
    torch.testing.assert_close(coords.read_state("joint_q"), value)
    torch.testing.assert_close(dofs.read_state("joint_qd"), value * 2)
    assert owner.control.joint_f.numpy()[index] == value.item() * 3
    targets, wrong_targets = (coords, dofs) if coord_targets else (dofs, coords)
    old_targets = owner.control.joint_target_q.numpy().copy()
    for invalid in (
        lambda: wrong_targets.scalar_field("control", "joint_target_q"),
        lambda: wrong_targets.write_control("joint_target_q", value),
    ):
        with pytest.raises(ValueError, match="index domain"):
            invalid()
    np.testing.assert_array_equal(owner.control.joint_target_q.numpy(), old_targets)
    targets.scalar_field("control", "joint_target_q")
    targets.write_control("joint_target_q", value)
    assert owner.control.joint_target_q.numpy()[index] == value.item()
    dofs.scalar_field("control", "joint_target_qd")
    dofs.write_control("joint_target_qd", value * 4)
    assert owner.control.joint_target_qd.numpy()[index] == value.item() * 4


def test_selection_contracts_check_joint_identity_world_domain_and_scalar_topology():
    """Equal lengths never establish ordered joint correspondence or shared ownership."""
    from isaaclab_tasks.contrib.keyboard.selection_contracts import require_scalar_joint_pair

    builder = newton.ModelBuilder()
    builder.begin_world()
    joints = []
    for index in range(2):
        body = builder.add_link(mass=1.0)
        joints.append(builder.add_joint_prismatic(parent=-1, child=body))
    builder.add_articulation(joints)
    builder.end_world()
    owner = NewtonSelections(builder.finalize("cpu"))
    coords, dofs = (
        owner.bind(newton.Model.AttributeFrequency.JOINT_COORD, [0, 1]),
        owner.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0, 1]),
    )
    require_scalar_joint_pair(coords, dofs)
    with pytest.raises(ValueError, match="same ordered scalar joints"):
        require_scalar_joint_pair(coords, owner.bind(newton.Model.AttributeFrequency.JOINT_DOF, [1, 0]))
    other = NewtonSelections(builder.finalize("cpu"))
    with pytest.raises(ValueError, match="same world domain"):
        require_scalar_joint_pair(coords, other.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0, 1]))
    grouped_coords = NewtonSelectionGroup(
        newton.Model.AttributeFrequency.JOINT_COORD, ((coords, torch.tensor([0])),), 1
    )
    grouped_dofs = NewtonSelectionGroup(newton.Model.AttributeFrequency.JOINT_DOF, ((dofs, torch.tensor([0])),), 1)
    require_scalar_joint_pair(grouped_coords, grouped_dofs)
    assert not hasattr(grouped_coords, "joint_ids")
    with pytest.raises(ValueError, match="same world domain"):
        require_scalar_joint_pair(coords, grouped_dofs)
    with pytest.raises(ValueError, match="same world domain"):
        require_scalar_joint_pair(
            grouped_coords,
            NewtonSelectionGroup(
                newton.Model.AttributeFrequency.JOINT_DOF,
                ((other.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0, 1]), torch.tensor([0])),),
                1,
            ),
        )


def test_prototype_cardinality_is_not_the_live_environment_axis():
    """An empty invalid prototype must fail even when every live world looks valid."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.keyboard.selection_contracts import require_count_per_world

    parts = []
    for count in (1, 2):
        builder = newton.ModelBuilder()
        builder.begin_world()
        for _ in range(count):
            builder.add_body(mass=1.0)
        builder.end_world()
        owner = NewtonSelections(builder.finalize("cpu"))
        parts.append(owner.bind(newton.Model.AttributeFrequency.BODY, list(range(count))))
    selection = MuJoCoSelection(
        SimpleNamespace(device=wp.get_device("cpu"), num_envs=3), newton.Model.AttributeFrequency.BODY, parts
    )
    assert selection.prototype_selection_counts == (1, 2)
    assert selection.element_participating.shape == (3, 2)
    assert not hasattr(selection, "world_selection_counts")
    assert not hasattr(parts[0], "prototype_selection_counts")
    with pytest.raises(ValueError, match="exactly 1"):
        require_count_per_world(selection, 1)
    require_count_per_world(parts[0], 1)
    with pytest.raises(ValueError, match="exactly 1"):
        require_count_per_world(parts[1], 1)


def test_partial_free_joint_is_not_a_scalar_pair(selections):
    from isaaclab_tasks.contrib.keyboard.selection_contracts import require_scalar_joint_pair

    coords = selections.bind(newton.Model.AttributeFrequency.JOINT_COORD, [0, 8])
    dofs = selections.bind(newton.Model.AttributeFrequency.JOINT_DOF, [0, 7])
    assert coords.world_selection_counts == dofs.world_selection_counts == (1, 1)
    np.testing.assert_array_equal(coords.joint_ids.numpy(), dofs.joint_ids.numpy())
    with pytest.raises(ValueError, match="components of multi-coordinate"):
        require_scalar_joint_pair(coords, dofs)


def test_group_world_domain_validates_before_rebind_and_ignores_group_enumeration_order():
    from isaaclab_tasks.contrib.keyboard.selection_contracts import require_same_world_domain

    parts = []
    for _ in range(2):
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_body(mass=1.0)
        builder.end_world()
        parts.append(NewtonSelections(builder.finalize("cpu")).bind(newton.Model.AttributeFrequency.BODY, [0]))
    first = ((parts[0], torch.tensor([1])), (parts[1], torch.tensor([0])))
    group = NewtonSelectionGroup(newton.Model.AttributeFrequency.BODY, first, 2)
    reverse = NewtonSelectionGroup(newton.Model.AttributeFrequency.BODY, tuple(reversed(first)), 2)
    require_same_world_domain(group, reverse)
    for invalid in (
        ((parts[0], torch.tensor([0])), (parts[1], torch.tensor([0]))),
        ((parts[0], torch.tensor([0])), (parts[1], torch.tensor([2]))),
    ):
        with pytest.raises(ValueError, match="disjoint, complete"):
            group.rebind(invalid)
        require_same_world_domain(group, reverse)
    with pytest.raises(ValueError, match="alias one physical model world"):
        group.rebind(((parts[0], torch.tensor([0])), (parts[0], torch.tensor([1]))))
    require_same_world_domain(group, reverse)
    swapped = NewtonSelectionGroup(
        newton.Model.AttributeFrequency.BODY, ((parts[0], torch.tensor([0])), (parts[1], torch.tensor([1]))), 2
    )
    with pytest.raises(ValueError, match="same world domain"):
        require_same_world_domain(group, swapped)
