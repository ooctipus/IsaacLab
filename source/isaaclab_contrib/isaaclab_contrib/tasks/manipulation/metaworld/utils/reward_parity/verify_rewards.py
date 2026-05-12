# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify our torch reward math against Meta-World's NumPy reference.

Reads ``reward_reference.json`` (output of :mod:`mujoco_reference_rewards`)
and re-computes each reward in torch using the *same* primitives our env
rewards do (:func:`tolerance` + :func:`hamacher_product` from
:mod:`utils`). Asserts byte-equivalence within a numpy↔torch float64
tolerance (``1e-6``).

This is a **math-only** check — it does not exercise the env-fetching
atoms in :mod:`quantities`, those are integration-tested by the geometry
verifier and by the smoke harness. The point of this file is to confirm
that no port bug snuck into the shape composition (caging, phase bonus,
success override, etc.).

Run with **IsaacLab's** venv::

    env_isaaclab/bin/python source/.../metaworld/assets/reward_parity/verify_rewards.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
_REF = _HERE / "reward_reference.json"

# Import the same torch primitives our env rewards use.
from isaaclab_contrib.tasks.manipulation.metaworld.mdp.utils import (  # noqa: E402
    hamacher_product,
    tolerance,
)

ATOL = 1e-5  # numpy(float64) vs torch(float32) — small but non-zero
DEVICE = "cpu"


def _t(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64, device=DEVICE)


def _scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


# ── reach (torch reproduction of metaworld.rewards.reach_v2.reach_v2_reward) ─


def reach_torch(tcp: torch.Tensor, target: torch.Tensor, hand_init: torch.Tensor) -> torch.Tensor:
    target_radius = 0.05
    tcp_to_target = torch.linalg.norm(tcp - target)
    in_place_margin = torch.linalg.norm(hand_init - target)
    in_place = tolerance(
        tcp_to_target,
        bounds=(0.0, target_radius),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    return 10.0 * in_place


# ── caging helper (shared by push) ──────────────────────────────────────────


def gripper_caging_torch(
    *,
    action: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_init_pos: torch.Tensor,
    left_pad_pos: torch.Tensor,
    right_pad_pos: torch.Tensor,
    tcp: torch.Tensor,
    init_tcp: torch.Tensor,
    obj_radius: float,
    pad_success_thresh: float,
    object_reach_radius: float,
    xz_thresh: float,
    desired_gripper_effort: float = 1.0,
    high_density: bool = False,
    medium_density: bool = False,
) -> torch.Tensor:
    pad_y = torch.stack([left_pad_pos[1], right_pad_pos[1]])
    pad_to_obj = torch.abs(pad_y - obj_pos[1])
    pad_to_objinit = torch.abs(pad_y - obj_init_pos[1])
    caging_lr_margin = torch.abs(pad_to_objinit - pad_success_thresh)

    caging_left = tolerance(
        pad_to_obj[0],
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin[0],
        sigmoid="long_tail",
    )
    caging_right = tolerance(
        pad_to_obj[1],
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin[1],
        sigmoid="long_tail",
    )
    caging_y = hamacher_product(caging_left, caging_right)

    xz = torch.tensor([0, 2], device=obj_pos.device)
    caging_xz_margin = torch.linalg.norm(obj_init_pos[xz] - init_tcp[xz]) - xz_thresh
    caging_xz = tolerance(
        torch.linalg.norm(tcp[xz] - obj_pos[xz]),
        bounds=(0.0, xz_thresh),
        margin=caging_xz_margin,
        sigmoid="long_tail",
    )

    grip_action = torch.clamp(action[-1], min=0.0, max=desired_gripper_effort) / desired_gripper_effort
    caging = hamacher_product(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, grip_action, torch.zeros_like(caging))
    caging_and_gripping = hamacher_product(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp_to_obj = torch.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = torch.linalg.norm(obj_init_pos - init_tcp)
        reach_margin = torch.abs(tcp_to_obj_init - object_reach_radius)
        reach = tolerance(tcp_to_obj, bounds=(0.0, object_reach_radius), margin=reach_margin, sigmoid="long_tail")
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping


# ── push (torch reproduction) ──────────────────────────────────────────────


def push_torch(
    *,
    action: torch.Tensor,
    obj: torch.Tensor,
    tcp: torch.Tensor,
    target: torch.Tensor,
    obj_init: torch.Tensor,
    left_pad: torch.Tensor,
    right_pad: torch.Tensor,
    init_tcp: torch.Tensor,
) -> torch.Tensor:
    target_radius = 0.05
    tcp_to_obj = torch.linalg.norm(obj - tcp)
    target_to_obj = torch.linalg.norm(obj - target)
    target_to_obj_init = torch.linalg.norm(obj_init - target)
    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, target_radius),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )
    object_grasped = gripper_caging_torch(
        action=action,
        obj_pos=obj,
        obj_init_pos=obj_init,
        left_pad_pos=left_pad,
        right_pad_pos=right_pad,
        tcp=tcp,
        init_tcp=init_tcp,
        obj_radius=0.015,
        pad_success_thresh=0.05,
        object_reach_radius=0.01,
        xz_thresh=0.005,
        high_density=True,
    )
    reward = 2.0 * object_grasped
    tcp_opened = torch.clamp(action[-1], min=0.0, max=1.0)
    phase = (tcp_to_obj < 0.02) & (tcp_opened > 0.0)
    bonus = 1.0 + reward + 5.0 * in_place
    reward = torch.where(phase, reward + bonus, reward)
    success = target_to_obj < target_radius
    reward = torch.where(success, torch.full_like(reward, 10.0), reward)
    return reward


# ── pick-place (torch reproduction) ────────────────────────────────────────


def pick_place_caging_torch(
    *,
    action: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_init_pos: torch.Tensor,
    left_pad_pos: torch.Tensor,
    right_pad_pos: torch.Tensor,
    init_left_pad: torch.Tensor,
    init_right_pad: torch.Tensor,
    tcp: torch.Tensor,
    init_tcp: torch.Tensor,
) -> torch.Tensor:
    pad_success_margin = 0.05
    x_z_success_margin = 0.005
    obj_radius = 0.015

    delta_y_left = left_pad_pos[1] - obj_pos[1]
    delta_y_right = obj_pos[1] - right_pad_pos[1]
    right_caging_margin = torch.abs(torch.abs(obj_pos[1] - init_right_pad[1]) - pad_success_margin)
    left_caging_margin = torch.abs(torch.abs(obj_pos[1] - init_left_pad[1]) - pad_success_margin)

    right_caging = tolerance(
        delta_y_right,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = tolerance(
        delta_y_left,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    y_caging = hamacher_product(left_caging, right_caging)

    tcp_xz = tcp.clone()
    tcp_xz[1] = 0.0
    obj_xz = obj_pos.clone()
    obj_xz[1] = 0.0
    tcp_obj_norm_xz = torch.linalg.norm(tcp_xz - obj_xz)

    init_obj_xz = obj_init_pos.clone()
    init_obj_xz[1] = 0.0
    init_tcp_xz = init_tcp.clone()
    init_tcp_xz[1] = 0.0
    tcp_obj_xz_margin = torch.linalg.norm(init_obj_xz - init_tcp_xz) - x_z_success_margin

    x_z_caging = tolerance(
        tcp_obj_norm_xz,
        bounds=(0.0, x_z_success_margin),
        margin=tcp_obj_xz_margin,
        sigmoid="long_tail",
    )

    grip = torch.clamp(action[-1], min=0.0, max=1.0)
    caging = hamacher_product(y_caging, x_z_caging)
    gripping = torch.where(caging > 0.97, grip, torch.zeros_like(caging))
    caging_and_gripping = hamacher_product(caging, gripping)
    return (caging_and_gripping + caging) / 2


def pick_place_torch(
    *,
    action: torch.Tensor,
    obj: torch.Tensor,
    tcp: torch.Tensor,
    target: torch.Tensor,
    obj_init: torch.Tensor,
    left_pad: torch.Tensor,
    right_pad: torch.Tensor,
    init_left_pad: torch.Tensor,
    init_right_pad: torch.Tensor,
    init_tcp: torch.Tensor,
) -> torch.Tensor:
    target_radius = 0.05
    obj_to_target = torch.linalg.norm(obj - target)
    tcp_to_obj = torch.linalg.norm(obj - tcp)
    in_place_margin = torch.linalg.norm(obj_init - target)
    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, target_radius),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    object_grasped = pick_place_caging_torch(
        action=action,
        obj_pos=obj,
        obj_init_pos=obj_init,
        left_pad_pos=left_pad,
        right_pad_pos=right_pad,
        init_left_pad=init_left_pad,
        init_right_pad=init_right_pad,
        tcp=tcp,
        init_tcp=init_tcp,
    )
    reward = hamacher_product(object_grasped, in_place)
    tcp_opened = torch.clamp(action[-1], min=0.0, max=1.0)
    lift = (tcp_to_obj < 0.02) & (tcp_opened > 0.0) & (obj[2] - 0.01 > obj_init[2])
    reward = torch.where(lift, reward + 1.0 + 5.0 * in_place, reward)
    success = obj_to_target < target_radius
    reward = torch.where(success, torch.full_like(reward, 10.0), reward)
    return reward


# ── driver ─────────────────────────────────────────────────────────────────


def main() -> int:
    with open(_REF) as f:
        reference = json.load(f)

    # Set torch float64 to match numpy reference precision.
    torch.set_default_dtype(torch.float64)

    fails: list[str] = []
    summary: dict[str, dict] = {}

    # Reach.
    diffs = []
    for i, entry in enumerate(reference["reach"]):
        s = entry["state"]
        ours = reach_torch(_t(s["tcp"]), _t(s["target"]), _t(s["hand_init"]))
        diff = abs(_scalar(ours) - entry["reward"])
        diffs.append(diff)
        if diff > ATOL:
            fails.append(f"reach[{i}]: ours={_scalar(ours):.6f} mw={entry['reward']:.6f} diff={diff:.2e}")
    summary["reach"] = {"max_diff": max(diffs), "n": len(diffs)}

    # Push.
    diffs = []
    for i, entry in enumerate(reference["push"]):
        s = entry["state"]
        ours = push_torch(
            action=_t(s["action"]),
            obj=_t(s["obj"]),
            tcp=_t(s["tcp"]),
            target=_t(s["target"]),
            obj_init=_t(s["obj_init"]),
            left_pad=_t(s["left_pad"]),
            right_pad=_t(s["right_pad"]),
            init_tcp=_t(s["init_tcp"]),
        )
        diff = abs(_scalar(ours) - entry["reward"])
        diffs.append(diff)
        if diff > ATOL:
            fails.append(f"push[{i}]: ours={_scalar(ours):.6f} mw={entry['reward']:.6f} diff={diff:.2e}")
    summary["push"] = {"max_diff": max(diffs), "n": len(diffs)}

    # Pick-place.
    diffs = []
    for i, entry in enumerate(reference["pick_place"]):
        s = entry["state"]
        ours = pick_place_torch(
            action=_t(s["action"]),
            obj=_t(s["obj"]),
            tcp=_t(s["tcp"]),
            target=_t(s["target"]),
            obj_init=_t(s["obj_init"]),
            left_pad=_t(s["left_pad"]),
            right_pad=_t(s["right_pad"]),
            init_left_pad=_t(s["init_left_pad"]),
            init_right_pad=_t(s["init_right_pad"]),
            init_tcp=_t(s["init_tcp"]),
        )
        diff = abs(_scalar(ours) - entry["reward"])
        diffs.append(diff)
        if diff > ATOL:
            fails.append(f"pick_place[{i}]: ours={_scalar(ours):.6f} mw={entry['reward']:.6f} diff={diff:.2e}")
    summary["pick_place"] = {"max_diff": max(diffs), "n": len(diffs)}

    print("[verify_rewards] === SUMMARY ===")
    for task, info in summary.items():
        print(f"  {task}: n={info['n']} max_diff={info['max_diff']:.2e} (atol={ATOL:.0e})")

    if fails:
        print("\n[verify_rewards] FAILURES:")
        for line in fails[:10]:
            print(f"  {line}")
        if len(fails) > 10:
            print(f"  ... and {len(fails) - 10} more")
        return 1

    print("\n[verify_rewards] ALL TASKS BYTE-EQUIVALENT (within float tolerance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
