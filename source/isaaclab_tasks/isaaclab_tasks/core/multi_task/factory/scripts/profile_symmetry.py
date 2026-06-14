# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Microprofile the symmetry reducer + command-framing path.

Times, per env-batch size, the two warp launches that make up the factory
command update — the symmetry reduction (``reduce_orientation``) and the
command-framing kernel — plus the ``wp.from_torch`` wrap overhead, to locate the
bottleneck. Also verifies the symmetry reduction captures into a CUDA graph (per
AGENTS.md, warp utility launches are capture-suspect until tested). Isaacsim-free.

Run:
  S=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/profile_symmetry.py
  ./isaaclab.sh -p $S
"""

from __future__ import annotations

import time

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.utils.symmetry import AssetSymmetryCfg, AxisSymmetryCfg, Symmetry

DEVICE = "cuda:0"
SIZES = (1024, 4096, 16384, 65536)
ITERS = 500


def _rand_quats(n: int) -> torch.Tensor:
    q = torch.randn(n, 4, device=DEVICE)
    return q / q.norm(dim=-1, keepdim=True)


def _time(fn, iters: int = ITERS) -> float:
    for _ in range(5):
        fn()
    wp.synchronize_device(DEVICE)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    wp.synchronize_device(DEVICE)
    return (time.perf_counter() - t0) / iters * 1e6  # us/call


def main() -> None:
    wp.init()
    # a symmetry reducer with the factory's mix: continuous (nut) + 4-fold (peg) + identity
    symmetry = Symmetry(
        [
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0)]),
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=4)]),
            AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=1)]),
        ],
        DEVICE,
    )
    print(f"symmetry: {symmetry.num_types} types | profiling {ITERS} iters/size on {DEVICE}\n")
    print(f"{'N':>8} | {'reduce(us)':>11} | {'from_torch(us)':>15} | {'total/step(us)':>15}")
    print("-" * 60)
    for n in SIZES:
        held = _rand_quats(n)
        target = _rand_quats(n)
        type_id = torch.randint(0, 3, (n,), dtype=torch.int32, device=DEVICE)
        out_err = torch.zeros(n, device=DEVICE)
        out_near = torch.zeros(n, 4, device=DEVICE)
        held_w = wp.from_torch(held, dtype=wp.quatf)
        target_w = wp.from_torch(target, dtype=wp.quatf)
        type_w = wp.from_torch(type_id, dtype=wp.int32)
        err_w = wp.from_torch(out_err)
        near_w = wp.from_torch(out_near, dtype=wp.quatf)

        # pure reduce (no wrapping)
        t_reduce = _time(lambda: symmetry.reduce_orientation(held_w, target_w, type_w, err_w, near_w))
        # reduce with per-call wp.from_torch (the realistic per-step cost of wrapping)
        t_wrap = _time(
            lambda: symmetry.reduce_orientation(
                wp.from_torch(held, dtype=wp.quatf),
                wp.from_torch(target, dtype=wp.quatf),
                wp.from_torch(type_id, dtype=wp.int32),
                wp.from_torch(out_err),
                wp.from_torch(out_near, dtype=wp.quatf),
            )
        )
        print(f"{n:>8} | {t_reduce:>11.1f} | {t_wrap - t_reduce:>15.1f} | {t_wrap:>15.1f}")

    # graph-capture smoke (symmetry reduction must capture cleanly for the env's CUDA graph)
    print("\ngraph capture test:")
    n = 4096
    held_w = wp.from_torch(_rand_quats(n), dtype=wp.quatf)
    target_w = wp.from_torch(_rand_quats(n), dtype=wp.quatf)
    type_w = wp.from_torch(torch.randint(0, 3, (n,), dtype=torch.int32, device=DEVICE), dtype=wp.int32)
    err_w = wp.from_torch(torch.zeros(n, device=DEVICE))
    near_w = wp.from_torch(torch.zeros(n, 4, device=DEVICE), dtype=wp.quatf)
    try:
        wp.capture_begin(device=DEVICE)
        symmetry.reduce_orientation(held_w, target_w, type_w, err_w, near_w)
        graph = wp.capture_end(device=DEVICE)
        wp.capture_launch(graph)
        wp.synchronize_device(DEVICE)
        print("  PASS: reduce_orientation captured + replayed cleanly")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")


if __name__ == "__main__":
    main()
