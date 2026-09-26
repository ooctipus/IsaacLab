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
