# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the Allegro reorientation tasks' reset starts the cube clear of the hand: every collision pair of every
world, after each of several full resets, is no deeper than the collision geometry's precision."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

WORLDS = 2048
RESETS = 4
PRECISION = 1.0e-4
"""Deepest overlap accepted at a reset [m]: the collision pipeline's own precision."""


def deepest_overlap(backend) -> float:
    """The deepest overlap [m] over every contact of one collision pass at the current poses."""
    backend.forward()
    contacts = backend.contacts
    backend.collision_pipeline.collide(backend.state_0, contacts, dt=getattr(backend, "collision_horizon", None))
    wp.synchronize()
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0:
        return 0.0
    shape_body = backend.model.shape_body.numpy()
    body_q = backend.state_0.body_q.numpy().astype(np.float64)

    def world_point(points, shapes):
        bodies = shape_body[shapes]
        pose = body_q[np.maximum(bodies, 0)]
        q, v = pose[:, 3:], points.astype(np.float64)
        t = 2.0 * np.cross(q[:, :3], v)
        moved = pose[:, :3] + v + q[:, 3:] * t + np.cross(q[:, :3], t)
        return np.where((bodies >= 0)[:, None], moved, v)

    a = world_point(contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_shape0.numpy()[:count])
    b = world_point(contacts.rigid_contact_point1.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    margin = contacts.rigid_contact_margin0.numpy()[:count] + contacts.rigid_contact_margin1.numpy()[:count]
    gap = ((b - a) * contacts.rigid_contact_normal.numpy()[:count]).sum(1) - margin
    return float(max(0.0, -gap.min()))


@pytest.mark.parametrize("task", ["Isaac-Reorient-Cube-Allegro", "Isaac-Reorient-Cube-Allegro-Direct"])
def test_reset_starts_the_cube_clear_of_the_hand(task):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = parse_env_cfg(task, device=device, num_envs=WORLDS, overrides=["physics=feather_pgs"])
    cfg.seed = 0
    launcher = {"device": device, "headless": True, "visualizer": None, "visualizer_explicit": True}
    with launch_simulation(cfg, launcher):
        env = gym.make(task, cfg=cfg)
        try:
            with torch.inference_mode():
                for _ in range(RESETS):
                    env.reset()
                    assert deepest_overlap(env.unwrapped.sim.backend_manager.backend) <= PRECISION
        finally:
            env.close()
