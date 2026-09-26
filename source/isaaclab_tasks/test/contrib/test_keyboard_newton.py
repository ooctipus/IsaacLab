# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selector contracts and the view-free 108-key task's reset/control boundary."""

import ast
import copy
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
def test_108_key_task_without_views_and_partial_reset(use_graph):
    if not wp.is_cuda_available():
        pytest.skip("MJWarp requires CUDA")
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from newton.selection import ArticulationView

    from isaaclab.app import launch_simulation
    from isaaclab.assets import AssetBaseCfg

    from isaaclab_tasks.contrib.keyboard.mdp.reset import capture_reset_state, restore_reset_state
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("IsaacContrib-Keyboard-SO101", "", overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 2
    cfg.keyboard_variants = ()
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


def test_selected_tip_jacobian_matches_finite_difference(selections, monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_tasks.contrib.keyboard.mdp.reset import KeyboardResetIKCfg, tip_jacobian

    model = selections.model
    state = model.state()
    ik = KeyboardResetIKCfg(
        joints=selections.resolve(NewtonSelectorCfg(JOINT_COORD, ".*/hinge0", count_per_world=1)),
        dofs=selections.resolve(NewtonSelectorCfg(JOINT_DOF, ".*/hinge0", count_per_world=1)),
        body=selections.resolve(NewtonSelectorCfg(BODY, ".*/tip0", count_per_world=1)),
        tip_offset=(0.3, 0.2, -0.1),
    )
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: state))
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


def test_velocity_reduction_uses_compact_ragged_dofs(selections, monkeypatch):
    from types import SimpleNamespace

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_tasks.contrib.keyboard.mdp.terminations import joint_vel_out_of_limit

    model = selections.model
    state = model.state()
    joints = selections.resolve(NewtonSelectorCfg(JOINT_DOF, ".*/hinge.*"))
    model.joint_velocity_limit.fill_(2.0)
    wp.to_torch(state.joint_qd)[joints.ids.numpy()[-2:]] = -3.0
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: state))
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
