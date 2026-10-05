# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native PD contracts across independently owned, masked populations."""

from types import SimpleNamespace

import newton
import pytest
import torch
import warp as wp

from isaaclab.utils import class_to_dict

from isaaclab_tasks.contrib.keyboard.mdp.actions import (
    NewtonRelativeJointPositionAction,
    NewtonRelativeJointPositionActionCfg,
)
from isaaclab_tasks.contrib.keyboard.newton_selection import NewtonSelectionGroup, NewtonSelections
from isaaclab_tasks.contrib.keyboard.selection_paths import NewtonSelectorCfg, resolve_selection


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def _owner(device, worlds=1, width=3):
    builder = newton.ModelBuilder()
    for world in range(worlds):
        builder.begin_world()
        root = builder.add_link(label=f"/{world}/root", mass=1.0)
        joints = [builder.add_joint_free(child=root, label=f"/{world}/free")]
        for slot in range(width):
            body = builder.add_link(label=f"/{world}/link{slot}", mass=1.0)
            joints.append(builder.add_joint_revolute(parent=root, child=body, label=f"/{world}/hinge{slot}"))
        builder.add_articulation(joints)
        builder.end_world()
    model = builder.finalize(device)
    return NewtonSelections(model, state=model.state(), control=model.control())


def _check_action(action):
    """Compare native controls and effort against the original tensor equations."""
    joints, dofs = action.joints, action.dofs
    q, qd = joints.read_state("joint_q"), dofs.read_state("joint_qd")
    target = torch.where(dofs.dense_active(), q + action.processed_actions, q)
    ke, kd = dofs.read_model("joint_target_ke"), dofs.read_model("joint_target_kd")
    limit = dofs.read_model("joint_effort_limit")
    expected_effort = (ke * (target - q) - kd * qd).clamp(-limit, limit)
    expected_effort = torch.where(dofs.dense_active(), expected_effort, 0.0)
    expected_controls = []
    for selection, attribute, values in (
        (action._targets, "joint_target_q", target),
        (dofs, "joint_target_qd", torch.zeros_like(qd)),
        (dofs, "joint_f", torch.zeros_like(qd)),
    ):
        for part, worlds in selection.native_bindings:
            actual = wp.to_torch(getattr(part.owner.control, attribute))
            expected = actual.clone()
            ids, active = part.dense_ids(), selection.dense_active()[worlds, : part.width]
            expected[ids] = torch.where(active, values[worlds, : part.width], expected[ids])
            expected_controls.append((actual, expected))
    action.apply_actions()
    for actual, expected in expected_controls:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(action.applied_effort, expected_effort, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("coord_layout", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_native_relative_pd_preserves_rounding_masks_and_target_layout(device, monkeypatch, coord_layout, grouped):
    monkeypatch.setattr(newton, "use_coord_layout_targets", coord_layout)
    owners = [_owner(device, 2, 3), _owner(device, 1, 2)] if grouped else [_owner(device, 3, 3)]
    actors = [torch.tensor([2, 0], device=device), torch.tensor([1], device=device)]
    selected = []
    for index_domain in (newton.Model.AttributeFrequency.JOINT_COORD, newton.Model.AttributeFrequency.JOINT_DOF):
        cfg = NewtonSelectorCfg(index_domain, ".*/hinge.*", policy_width=3)
        parts = [resolve_selection(owner, cfg) for owner in owners]
        selected.append(
            NewtonSelectionGroup(cfg.index_domain, list(zip(parts, actors, strict=True)), 3) if grouped else parts[0]
        )
    joints, dofs = selected
    action = NewtonRelativeJointPositionAction(
        NewtonRelativeJointPositionActionCfg(asset_name=None, joints=joints, dofs=dofs, scale=0.02),
        SimpleNamespace(num_envs=3, device=device),
    )
    # An unselected free joint makes coordinate and DOF offsets differ.
    assert joints.native_bindings[0][0].ids.numpy()[0] != dofs.native_bindings[0][0].ids.numpy()[0]
    for owner in owners:
        owner.control.joint_target_q.fill_(17.0)
        owner.control.joint_target_qd.fill_(19.0)
        owner.control.joint_f.fill_(23.0)
    joints.write_state("joint_q", torch.tensor([[2**24, 0.12345679, -0.4]] * 3, device=device))
    dofs.write_state("joint_qd", torch.tensor([[0.0, 0.23456789, -2.0]] * 3, device=device))
    for part, _ in dofs.native_bindings:
        ids = part.dense_ids()
        wp.to_torch(part.owner.model.joint_target_ke)[ids] = torch.tensor([5.0, 7.1234567, 11.0], device=device)[
            : part.width
        ]
        wp.to_torch(part.owner.model.joint_target_kd)[ids] = torch.tensor([0.5, 2.2345679, 3.0], device=device)[
            : part.width
        ]
        wp.to_torch(part.owner.model.joint_effort_limit)[ids] = torch.tensor([100.0, 0.75, 1.0], device=device)[
            : part.width
        ]
    first = joints.native_bindings[0][0]
    wp.to_torch(first.owner.body_active)[wp.to_torch(first.body_ids)[-1]] = False
    wp.to_torch(owners[-1].world_active)[-1] = False
    for owner in owners:
        owner.refresh()
    if grouped:
        joints.refresh()
        dofs.refresh()
    action.process_actions(torch.tensor([[25.0, 3.2345679, -100.0]] * 3, device=device))
    for frame in range(4):
        _check_action(action)
        # Two native substeps advance state between successive physics-frame applies.
        for _ in range(2):
            for owner in owners:
                wp.to_torch(owner.state.joint_q).add_(0.0625)
        if frame == 0:
            assert action.applied_effort[2 if grouped else 0, 0] == 0.0  # q + 0.5 rounded back to q
            for owner in owners:
                owner.body_active.fill_(True)
                owner.world_active.fill_(True)
                owner.refresh()
            if grouped:
                joints.refresh()
                dofs.refresh()
    for part, _ in dofs.native_bindings:
        wp.to_torch(part.owner.model.joint_effort_limit)[part.dense_ids()[:, 0]] = float("nan")
    _check_action(action)
    # Each field owns its mask; masked coordinate reads do not disable an active DOF write.
    joints.dense_active()[0, 1] = False
    _check_action(action)
    action.reset(torch.tensor([0], device=device))
    assert not action.raw_actions[0].any() and not action.processed_actions[0].any()
    assert not action.applied_effort[0].any()
    assert class_to_dict(joints) == {}


def test_borrowed_fields_follow_group_rebinding_and_mask_refresh(device):
    owners = [_owner(device), _owner(device)]
    actors = [torch.tensor([1], device=device), torch.tensor([0], device=device)]
    groups = []
    for index_domain in (newton.Model.AttributeFrequency.JOINT_COORD, newton.Model.AttributeFrequency.JOINT_DOF):
        cfg = NewtonSelectorCfg(index_domain, ".*/hinge.*", count_per_world=3)
        groups.append(
            NewtonSelectionGroup(
                cfg.index_domain, [(resolve_selection(o, cfg), ids) for o, ids in zip(owners, actors)], 2
            )
        )
    joints, dofs = groups
    action = NewtonRelativeJointPositionAction(
        NewtonRelativeJointPositionActionCfg(asset_name=None, joints=joints, dofs=dofs),
        SimpleNamespace(num_envs=2, device=device),
    )
    action.process_actions(torch.ones((2, 3), device=device))
    _check_action(action)
    borrowed = joints.scalar_field("state", "joint_q")
    assert joints.scalar_field("state", "joint_q") is borrowed
    control_field = action._targets.scalar_field("control", "joint_target_q")
    reordered = list(reversed(actors))
    for group in groups:
        group.rebind([(part, ids) for (part, _), ids in zip(group.native_bindings, reordered, strict=True)])
    moved = joints.scalar_field("state", "joint_q")
    assert moved is not borrowed and moved.sources.ptr == borrowed.sources.ptr
    assert action._targets.scalar_field("control", "joint_target_q").sources.ptr == control_field.sources.ptr
    _check_action(action)

    # Leaf identities can stay fixed while their explicit state/control binding changes.
    owners[0].state = owners[0].model.state()
    owners[0].state.joint_q.fill_(0.2)
    for group in groups:
        group.rebind(group.native_bindings)
    new_state_field = joints.scalar_field("state", "joint_q")
    assert new_state_field.sources.ptr != moved.sources.ptr
    assert action._targets.scalar_field("control", "joint_target_q").sources.ptr == control_field.sources.ptr
    _check_action(action)
    owners[0].control = owners[0].model.control()
    owners[0].control.joint_target_q.fill_(17.0)
    for group in groups:
        group.rebind(group.native_bindings)
    assert joints.scalar_field("state", "joint_q").sources.ptr == new_state_field.sources.ptr
    assert action._targets.scalar_field("control", "joint_target_q").sources.ptr != control_field.sources.ptr
    _check_action(action)
    retired_control = owners[0].control.joint_target_q.numpy().copy()
    replacement = _owner(device, 2)
    for group in groups:
        cfg = NewtonSelectorCfg(group.index_domain, ".*/hinge.*", count_per_world=3)
        group.rebind([(resolve_selection(replacement, cfg), torch.tensor([0, 1], device=device))])
    assert joints.scalar_field("state", "joint_q") is not borrowed
    joints.write_state("joint_q", torch.full((2, 3), 0.3, device=device))
    replacement.world_active.fill_(False)
    replacement.refresh()
    for group in groups:
        group.refresh()
    _check_action(action)
    replacement.world_active.fill_(True)
    replacement.refresh()
    for group in groups:
        group.refresh()
    _check_action(action)
    assert (owners[0].control.joint_target_q.numpy() == retired_control).all()
    bodies = resolve_selection(replacement, NewtonSelectorCfg(newton.Model.AttributeFrequency.BODY, ".*/root"))
    with pytest.raises(TypeError, match="float32"):
        bodies.scalar_field("state", "body_q")
    with pytest.raises(ValueError, match="index domain"):
        joints.scalar_field("state", "body_q")
    for selection in (bodies, joints):
        with pytest.raises(ValueError, match="field source"):
            selection.scalar_field("other", "joint_q")


def test_captured_action_reads_updated_native_state_and_episode_masks(device):
    if device == "cpu":
        pytest.skip("CUDA graph test")
    owner = _owner(device, 2)
    joints = resolve_selection(owner, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_COORD, ".*/hinge.*"))
    dofs = resolve_selection(owner, NewtonSelectorCfg(newton.Model.AttributeFrequency.JOINT_DOF, ".*/hinge.*"))
    action = NewtonRelativeJointPositionAction(
        NewtonRelativeJointPositionActionCfg(asset_name=None, joints=joints, dofs=dofs),
        SimpleNamespace(num_envs=2, device=device),
    )
    action.apply_actions()  # Create borrowed descriptors before capturing their consumer.
    with wp.ScopedCapture(device=device) as capture:
        action.apply_actions()
    for frame in range(4):
        owner.state.joint_q.fill_(0.3 * frame)
        action.process_actions(torch.full((2, 3), frame + 1.0, device=device))
        wp.to_torch(owner.world_active)[0] = frame % 2 == 0
        owner.refresh()
        owner.control.joint_target_q.fill_(17.0)
        _check_action(action)
        expected = wp.to_torch(owner.control.joint_target_q).clone()
        owner.control.joint_target_q.fill_(17.0)
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(wp.to_torch(owner.control.joint_target_q), expected, rtol=0, atol=0)
