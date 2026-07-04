# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact reward-equation tests for unified motion learning."""

from types import SimpleNamespace

import torch

from isaaclab.managers import RewardTermCfg

from isaaclab_tasks.core.multi_task import mdp as multi_task_mdp
from isaaclab_tasks.core.multi_task.motion.robots.g1 import actions as g1_actions
from isaaclab_tasks.core.multi_task.motion.robots.g1.actions import G1JointPositionAction


def _view(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def _scene_env() -> SimpleNamespace:
    num_envs = 2
    joint_position = torch.tensor(((1.2, -1.1, 0.4), (0.95, -0.95, 0.0)))
    joint_limits = torch.tensor(((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0))).expand(num_envs, -1, -1).clone()
    body_quaternion = torch.zeros(num_envs, 2, 4)
    body_quaternion[..., 3] = 1.0
    body_quaternion[0, 0] = torch.tensor((0.0, 0.0, 2.0**-0.5, 2.0**-0.5))
    root_quaternion = torch.zeros(num_envs, 4)
    root_quaternion[..., 3] = 1.0
    body_velocity = torch.tensor(
        (
            ((3.0, 4.0, 0.0), (0.0, 0.0, 7.0)),
            ((1.0, 2.0, 2.0), (0.0, 0.0, 0.0)),
        )
    )
    force = torch.tensor(
        (
            ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0)),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=_view(joint_position),
            joint_pos_limits=_view(joint_limits),
            body_quat_w=_view(body_quaternion),
            body_link_quat_w=_view(body_quaternion),
            body_com_lin_vel_w=_view(body_velocity),
            root_quat_w=_view(root_quaternion),
            GRAVITY_VEC_W=_view(torch.tensor((0.0, 0.0, -9.81)).expand(num_envs, -1).clone()),
        )
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=_view(force)))

    class _Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name: str):
            return robot if name == "robot" else self.sensors[name]

    return SimpleNamespace(num_envs=num_envs, device="cpu", scene=_Scene())


def test_g1_controller_evidence_functions_are_action_owned() -> None:
    """Controller evidence lives beside the action state that defines its equations."""
    for func in (
        g1_actions.controller_torques_l2,
        g1_actions.controller_action_rate_l2,
        g1_actions.controller_torque_limits,
    ):
        assert func.__module__ == g1_actions.__name__


def test_g1_controller_evidence_reads_action_term_owned_edge_state() -> None:
    """Controller evidence uses retained torque and the exact processed-action edge."""
    action = object.__new__(G1JointPositionAction)
    action._applied_torque = torch.tensor(((20.0, -18.0, 3.0), (19.0, -20.0, 0.0)))
    action._processed_actions = torch.tensor(((1.0, 2.0, 3.0), (0.0, -1.0, 1.0)))
    action._previous_processed_actions = torch.tensor(((0.5, 1.0, 3.5), (1.0, -1.0, -1.0)))
    action.joint_effort_limit = torch.tensor((20.0, 20.0, 10.0))
    requested_action_names: list[str] = []

    def get_term(name: str) -> G1JointPositionAction:
        requested_action_names.append(name)
        return action

    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=get_term))

    torch.testing.assert_close(
        g1_actions.controller_torques_l2(env, "joint_position"),
        action._applied_torque.square().sum(dim=-1),
    )
    torch.testing.assert_close(
        g1_actions.controller_action_rate_l2(env, "joint_position"),
        (action._processed_actions - action._previous_processed_actions).square().sum(dim=-1),
    )
    torch.testing.assert_close(
        g1_actions.controller_torque_limits(env, "joint_position", 0.95),
        torch.tensor((1.0, 1.0)),
    )

    assert requested_action_names == ["joint_position"] * 3


def test_shared_joint_rewards_match_explicit_bfm_equations() -> None:
    """Joint targets and the 0.95 hard-limit fraction remain explicit."""
    env = _scene_env()
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2])

    torch.testing.assert_close(
        multi_task_mdp.joint_position_target_l2(env, 0.0, asset_cfg),
        torch.tensor((2.81, 1.805)),
    )
    torch.testing.assert_close(
        multi_task_mdp.joint_position_limits(env, 0.95, asset_cfg),
        torch.tensor((0.4, 0.0)),
    )


def test_shared_contact_rewards_use_current_force_and_body_state() -> None:
    """Contact helpers read the current edge and preserve BFM-Zero's 3D speed math."""
    env = _scene_env()
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])

    torch.testing.assert_close(multi_task_mdp.contact_undesired(env, 1.0, sensor_cfg), torch.ones(2))
    torch.testing.assert_close(
        multi_task_mdp.body_orientation_contact(env, 1.0, sensor_cfg, asset_cfg),
        torch.zeros(2),
    )
    torch.testing.assert_close(
        multi_task_mdp.body_contact_velocity(env, 1.0, sensor_cfg, asset_cfg),
        torch.tensor((5.0, 3.0)),
    )
    torch.testing.assert_close(
        multi_task_mdp.body_heading_alignment(env, asset_cfg),
        torch.tensor((torch.pi / 2.0, 0.0)),
    )


def test_scaled_reward_binds_one_mutable_device_scale() -> None:
    """A helper scale is bound once and follows in-place curriculum updates."""
    scale = torch.tensor(0.25)
    env = SimpleNamespace(num_envs=2, device="cpu", scale=scale)

    def reward(_env, value: float) -> torch.Tensor:
        return torch.full((_env.num_envs,), value)

    cfg = RewardTermCfg(
        func=multi_task_mdp.RewardScaled,
        weight=1.0,
        params={"func": reward, "func_params": {"value": 4.0}, "scale_bind": "env.scale"},
    )
    term = multi_task_mdp.RewardScaled(cfg, env)
    torch.testing.assert_close(term(env, **cfg.params), torch.full((2,), 1.0))
    scale.fill_(0.5)
    torch.testing.assert_close(term(env, **cfg.params), torch.full((2,), 2.0))
