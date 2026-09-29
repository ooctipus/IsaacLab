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
from unittest.mock import patch

import newton
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.keyboard.newton_selection import (
    BODY,
    JOINT_COORD,
    JOINT_DOF,
    NewtonSelectionGroup,
    NewtonSelections,
    NewtonSelectorCfg,
)


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
    q = selections.resolve(NewtonSelectorCfg(JOINT_COORD, ".*/free", count_per_world=7))
    qd = selections.resolve(NewtonSelectorCfg(JOINT_DOF, ".*/free", count_per_world=6))
    assert q.counts == (7, 7) and qd.counts == (6, 6)
    assert copy.deepcopy(q) is q
    with pytest.raises(ValueError, match="matched no"):
        selections.resolve(NewtonSelectorCfg(BODY, ".*/missing"))
    with pytest.raises(ValueError, match="Expected 1"):
        selections.resolve(NewtonSelectorCfg(BODY, ".*", count_per_world=1))
    with pytest.raises(ValueError, match="Unknown Newton frequency"):
        selections.resolve(NewtonSelectorCfg("joint", ".*"))


def test_retired_selection_owner_preserves_existing_bindings_without_a_cache_cycle():
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="/body", mass=1.0)
    builder.end_world()
    model = builder.finalize("cpu")
    owner = NewtonSelections(model, state=model.state(), control=model.control())
    cfg = NewtonSelectorCfg(BODY, ".*", count_per_world=1)
    selected = owner.resolve(cfg)
    references = [weakref.ref(value) for value in (owner, model, owner.state, owner.control)]
    del model
    enabled = gc.isenabled()
    gc.disable()
    try:
        owner.retire()
        owner.retire()
        for selector in (cfg, cfg.replace(path="/body")):
            with pytest.raises(RuntimeError, match="retired"):
                owner.resolve(selector)
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
    bodies = selections.resolve(NewtonSelectorCfg(BODY, (".*/tip.*", ".*")))
    assert bodies.counts == (2, 3)  # overlap was deduplicated
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
    bodies = selections.resolve(NewtonSelectorCfg(BODY, ".*/base", count_per_world=1))
    values = wp.ones(selections.model.body_count, dtype=wp.float32, device=selections.model.device)
    wp.to_torch(selections.body_active)[0] = False
    selections.refresh()
    np.testing.assert_array_equal(bodies.dense(values).cpu().numpy(), [[0.0], [1.0]])
    values.fill_(3.0)
    np.testing.assert_array_equal(bodies.dense(values).cpu().numpy(), [[0.0], [3.0]])


def test_membership_refresh_is_capture_safe(selections):
    if not selections.model.device.is_cuda:
        pytest.skip("CUDA graph test")
    selected = selections.resolve(NewtonSelectorCfg(BODY, ".*"))
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
            for _ in range(4):
                obs, reward, *_ = env.step(torch.full((2, 6), 0.05, device=task.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            command = task.command_manager.get_term("typing")
            c = command.cfg
            ids = torch.tensor([0], device=task.device)
            snapshot = capture_reset_state(task, ids, c.reset_roots, c.reset_coords, c.reset_dofs)
            state, model = NewtonManager.get_state(), NewtonManager.get_model()
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
            excluded_q = NewtonManager.get_state().joint_q.numpy()[other_coords:].copy()
            command._solve_reset_pose(torch.arange(2, device=task.device))
            unchanged = np.array_equal(NewtonManager.get_state().joint_q.numpy()[other_coords:], excluded_q)
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
        joints=selections.resolve(NewtonSelectorCfg(JOINT_COORD, ".*/hinge0", count_per_world=1)),
        dofs=selections.resolve(NewtonSelectorCfg(JOINT_DOF, ".*/hinge0", count_per_world=1)),
        body=selections.resolve(NewtonSelectorCfg(BODY, ".*/tip0", count_per_world=1)),
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fixed_root_writer_masks_frames_and_stream_order(device):
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
    roots = owner.resolve(NewtonSelectorCfg(BODY, ".*/root.*", count_per_world=2))
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
        notify_model_changed=lambda flags, rows: calls.append((flags, rows)),
        invalidate_fk=lambda rows: calls.append(rows),
    )
    observed = wp.empty_like(model.joint_X_p)
    rng = torch.get_rng_state().clone()
    wp.synchronize_device(device)
    producer = torch.cuda.stream(torch.cuda.Stream(device=device)) if device.startswith("cuda") else nullcontext()
    with producer:
        if device.startswith("cuda"):
            torch.cuda._sleep(2_000_000)
        delayed = torch.empty_like(storage)[..., ::2]
        delayed.copy_(poses)
        write_fixed_root_poses(env, roots, ids, delayed)
        # The consumer uses the restored prior Warp stream, without an outer shared-stream scope.
        wp.copy(observed, model.joint_X_p)
    wp.synchronize_device(device)
    actual = wp.to_torch(observed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    torch.testing.assert_close(actual[joint_ids[~selected]], initial[joint_ids[~selected]], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert calls[0][0] == newton.ModelFlags.JOINT_PROPERTIES and calls[0][1] is ids and calls[1] is ids
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
        cfg = NewtonSelectorCfg(BODY, ".*/root", count_per_world=2)
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
            parts.append((owner.resolve(cfg), torch.tensor(logical_worlds, device=device)))
        roots = NewtonSelectionGroup(cfg, parts, 5) if grouped else parts[0][0]
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
def test_reset_ik_uses_current_bindings_without_retaining_a_graph(device):
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
            NewtonSelectorCfg(BODY, ".*/root", count_per_world=1),
            NewtonSelectorCfg(BODY, ".*/tip", count_per_world=1),
            NewtonSelectorCfg(JOINT_COORD, ".*/slider", count_per_world=1),
            NewtonSelectorCfg(JOINT_DOF, ".*/slider", count_per_world=1),
        )
        groups = [
            NewtonSelectionGroup(
                cfg, [(part.resolve(cfg), torch.tensor(ids, device=device)) for part, ids in zip(owners, actors)], 3
            )
            for cfg in configs
        ]
        roots, bodies, coords, dofs = groups
        ik = KeyboardResetIKCfg(joints=coords, dofs=dofs, body=bodies)
        command = object.__new__(LetterTypingCommand)
        command._env = SimpleNamespace(num_envs=3, device=device, invalidate_fk=lambda _: None, forward=lambda: None)
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
            group.rebind([(replacement.resolve(cfg), torch.tensor([1, 2, 0], device=device))])
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
    joints = selections.resolve(NewtonSelectorCfg(JOINT_DOF, ".*/hinge.*"))
    model.joint_velocity_limit.fill_(2.0)
    wp.to_torch(state.joint_qd)[joints.ids.numpy()[-2:]] = -3.0
    selections.state = state
    env = SimpleNamespace(num_envs=2, device=str(model.device))
    term = joint_vel_out_of_limit(TerminationTermCfg(func=joint_vel_out_of_limit, params={"joints": joints}), env)
    np.testing.assert_array_equal(term(env, joints).cpu().numpy(), [False, True])
    wp.to_torch(selections.world_active)[1] = False
    selections.refresh()
    np.testing.assert_array_equal(term(env, joints).cpu().numpy(), [False, False])


def test_manager_serialization_keeps_declarative_selectors(selections):
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.utils import class_to_dict

    from isaaclab_tasks.contrib.keyboard.mdp.observations import joint_pos

    cfg = NewtonSelectorCfg(JOINT_COORD, ".*/free", count_per_world=7)
    term = ObservationTermCfg(func=joint_pos, params={"joints": selections.resolve(cfg)})
    assert class_to_dict(term)["params"]["joints"] == class_to_dict(cfg)


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
    cfg = NewtonSelectorCfg(JOINT_COORD, "/Robot/hinge.*", dense_width=2)
    parts = [(owner.resolve(cfg), actor_ids) for owner, actor_ids in zip(bindings, actors, strict=True)]
    selected = NewtonSelectionGroup(cfg, parts, 5)
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
    for part, actor_ids in parts:
        torch.testing.assert_close(
            wp.to_torch(part.owner.control.joint_target_q), expected[actor_ids, : part.width].flatten()
        )
    bodies_cfg = NewtonSelectorCfg(BODY, "/Robot/.*", dense_width=3)
    bodies = NewtonSelectionGroup(
        bodies_cfg, [(owner.resolve(bodies_cfg), rows) for owner, rows in zip(bindings, actors, strict=True)], 5
    )
    poses = bodies.read_state("body_q")
    assert poses.shape == (5, 3, 7)
    assert not poses[[0, 2, 4], 2].any()
    from isaaclab.utils import class_to_dict

    assert class_to_dict(selected) == class_to_dict(cfg)


@pytest.mark.parametrize("partition_mode", ["single", "fixed_dof"])
def test_keyboard_has_no_internal_contacts(partition_mode):
    """Partition boundaries must not enable collisions between keys and their own case."""
    from newton.solvers import SolverMuJoCo

    from pxr import Usd, UsdGeom

    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_pool import TYPING_KEYBOARD_VARIANTS
    from isaaclab_tasks.contrib.keyboard.keyboards.keyboard_usd import spawn_keyboard

    cfg = TYPING_KEYBOARD_VARIANTS[0].replace(partition_mode=partition_mode)
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
            model, state = NewtonManager.get_model(), NewtonManager.get_state()
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
