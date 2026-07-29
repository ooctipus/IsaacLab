# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.multi_task.mdp.rewards import contact_penalty


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


def _make_env(forces: torch.Tensor):
    contact_sensor = SimpleNamespace(
        num_sensors=forces.shape[2],
        data=SimpleNamespace(net_forces_w_history=wp.from_torch(forces)),
    )
    return SimpleNamespace(
        num_envs=forces.shape[0],
        device=str(forces.device),
        scene=SimpleNamespace(sensors={"contact_forces": contact_sensor}),
    )


def test_contact_penalty_uses_selected_sensor_bodies():
    """Direct mode penalizes only bodies listed by sensor_cfg."""
    forces = torch.zeros(2, 2, 4, 3)
    forces[0, 0, 1, 0] = 2.0
    forces[1, 0, 1, 0] = 2.0
    forces[1, 1, 3, 0] = 2.0
    env = _make_env(forces)
    contact_sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[1, 3])

    term = contact_penalty(SimpleNamespace(params={"contact_sensor_cfg": contact_sensor_cfg}), env)
    penalty = term(env, threshold=1.0, contact_sensor_cfg=contact_sensor_cfg)

    torch.testing.assert_close(penalty, torch.tensor([1, 2]))


def test_contact_penalty_can_exclude_sensor_bodies():
    """Exclude mode penalizes all contact bodies except the listed bodies."""
    forces = torch.zeros(2, 2, 4, 3)
    forces[0, 0, 0, 0] = 2.0
    forces[0, 1, 2, 0] = 2.0
    forces[1, 0, 1, 0] = 2.0
    env = _make_env(forces)
    exclude_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[1, 3])

    term = contact_penalty(
        SimpleNamespace(params={"exclude_contact_sensor_cfg": exclude_contact_sensor_cfg}),
        env,
    )
    penalty = term(env, threshold=1.0, exclude_contact_sensor_cfg=exclude_contact_sensor_cfg)

    torch.testing.assert_close(penalty, torch.tensor([2, 0]))


def test_contact_penalty_requires_one_selection_mode():
    """The reward term accepts either direct selection or exclusion, not both."""
    env = _make_env(torch.zeros(1, 1, 2, 3))
    contact_sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0])
    exclude_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[1])

    with pytest.raises(ValueError, match="exactly one"):
        contact_penalty(
            SimpleNamespace(
                params={
                    "contact_sensor_cfg": contact_sensor_cfg,
                    "exclude_contact_sensor_cfg": exclude_contact_sensor_cfg,
                }
            ),
            env,
        )
    with pytest.raises(ValueError, match="exactly one"):
        contact_penalty(SimpleNamespace(params={}), env)
