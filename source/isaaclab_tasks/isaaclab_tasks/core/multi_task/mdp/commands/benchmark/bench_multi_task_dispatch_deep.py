# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deep dispatch timing for exact public ``MultiTaskCommand`` backends."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from collections.abc import Callable
from unittest.mock import patch

import torch
import warp as wp

if __package__:
    from .mock_command import build_mock_command
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mock_command import build_mock_command

from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_wp import (
    compute_dense_graph_producers,
    dispatch_graph_dense,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.mega_kernel.compose import compose_warp
from isaaclab_tasks.core.multi_task.mdp.commands.impl.mega_kernel.execute import dispatch_mega_warp
from isaaclab_tasks.core.multi_task.mdp.commands.impl.mega_kernel.read import fill_unified_buffer_warp
from isaaclab_tasks.core.multi_task.mdp.commands.impl.mega_kernel.rotation import (
    rotate_canonical_slots_to_body_frame_warp,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.primitive_queue_local.execute import (
    _PRIMITIVE_KERNELS,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.schedules import (
    SCHEDULE_DIRECT_QUAT_DELTA,
    SCHEDULE_DIRECT_SCALAR_DELTA,
    SCHEDULE_DIRECT_VEC3_DELTA,
    SCHEDULE_SCALAR_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
)


def _run_warmup(command, env, steps: int) -> None:
    for _ in range(steps):
        command._update_command()
        env.episode_length_buf += 1
    torch.cuda.synchronize()


def _time_cuda(fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _time_phase(totals: dict[str, float], name: str, fn: Callable[[], None]) -> None:
    totals[name] += _time_cuda(fn)


def _reset_step(command) -> None:
    torch.lt(command._slot_arange, command._env_slot_count.unsqueeze(1), out=command._slot_valid)
    command._buf_error.zero_()
    command._buf_activation.zero_()


def _launch_local_kernel(command, plan, kernel, count: int) -> None:
    wp.launch(
        kernel,
        dim=count,
        inputs=[plan.queue, plan.spec, plan.state, plan.outputs],
        device=str(command.device),
    )


def _launch_dense_graph_producers(command, plan) -> None:
    total_signature_count = (
        plan.vec3_signature_count
        + plan.scalar_signature_count
        + plan.quat_signature_count
        + plan.scalar_sum_signature_count
        + plan.contact_signature_count
    )
    if total_signature_count == 0:
        return
    wp.launch(
        compute_dense_graph_producers,
        dim=(command.num_envs, total_signature_count),
        inputs=[
            plan.vec3_nodes.nodes_view,
            plan.scalar_nodes.nodes_view,
            plan.quat_nodes.nodes_view,
            plan.scalar_sum_nodes.nodes_view,
            plan.contact_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.direct_vec3_wp,
            plan.direct_scalar_wp,
            plan.direct_quat_wp,
            plan.scalar_sum_wp,
            plan.contact_mask_wp,
            plan.vec3_signature_count,
            plan.scalar_signature_count,
            plan.quat_signature_count,
            plan.scalar_sum_signature_count,
            plan.contact_signature_count,
        ],
        device=str(command.device),
    )


def _launch_dense_graph_consumer(command, plan) -> None:
    wp.launch(
        dispatch_graph_dense,
        dim=(command.num_envs, command.k_max),
        inputs=[
            plan.env_slots,
            plan.subtask_schedule_ids_wp,
            plan.vec3_nodes.nodes_view,
            plan.scalar_nodes.nodes_view,
            plan.quat_nodes.nodes_view,
            plan.scalar_sum_nodes.nodes_view,
            plan.contact_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_vec3_wp,
            plan.direct_scalar_wp,
            plan.direct_quat_wp,
            plan.scalar_sum_wp,
            plan.contact_mask_wp,
            plan.vec3_signature_count,
            plan.scalar_signature_count,
            plan.quat_signature_count,
            plan.scalar_sum_signature_count,
            plan.contact_signature_count,
        ],
        device=str(command.device),
    )


def _time_mega(command, totals: dict[str, float]) -> None:
    plan = command._backend.plan
    if hasattr(plan, "mega"):
        plan = plan.mega
    _time_phase(totals, "read", lambda: fill_unified_buffer_warp(command, plan))
    _time_phase(totals, "execute_mega", lambda: dispatch_mega_warp(command, plan))
    _time_phase(totals, "rotate", lambda: rotate_canonical_slots_to_body_frame_warp(command, plan))
    _time_phase(totals, "compose", lambda: compose_warp(command, plan))


def _time_queue(command, totals: dict[str, float]) -> None:
    plan = command._backend.plan
    _time_phase(totals, "read", lambda: fill_unified_buffer_warp(command, plan))
    for kernel, count, schedule_id in zip(_PRIMITIVE_KERNELS, plan.schedule_counts_py, range(len(_PRIMITIVE_KERNELS))):
        if count != 0:
            _time_phase(
                totals,
                f"schedule_{schedule_id}",
                lambda k=kernel, c=count: _launch_local_kernel(command, plan, k, c),
            )
    _time_phase(totals, "rotate", lambda: rotate_canonical_slots_to_body_frame_warp(command, plan))
    _time_phase(totals, "compose", lambda: compose_warp(command, plan))


def _time_graph(command, totals: dict[str, float]) -> None:
    plan = command._backend.plan
    _time_phase(totals, "read", lambda: fill_unified_buffer_warp(command, plan))
    _time_phase(totals, "dense_graph_producers", lambda: _launch_dense_graph_producers(command, plan))
    _time_phase(totals, "dense_graph_consumer", lambda: _launch_dense_graph_consumer(command, plan))
    _time_phase(totals, "rotate", lambda: rotate_canonical_slots_to_body_frame_warp(command, plan))
    _time_phase(totals, "compose", lambda: compose_warp(command, plan))


def _print_graph_fanout(command) -> None:
    plan = command._backend.plan
    counts = plan.schedule_counts_py
    num_envs = command.num_envs
    print("# graph sharing")
    rows = (
        ("direct_vec3", counts[SCHEDULE_DIRECT_VEC3_DELTA], plan.vec3_signature_count * num_envs),
        ("direct_scalar", counts[SCHEDULE_DIRECT_SCALAR_DELTA], plan.scalar_signature_count * num_envs),
        ("direct_quat", counts[SCHEDULE_DIRECT_QUAT_DELTA], plan.quat_signature_count * num_envs),
        ("scalar_sum", counts[SCHEDULE_SCALAR_SUM_DELTA], plan.scalar_sum_signature_count * num_envs),
        (
            "contact",
            counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA]
            + counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA]
            + counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA],
            plan.contact_signature_count * num_envs,
        ),
    )
    print(f"# {'producer':<16s} {'work':>10s} {'nodes':>10s} {'fanout':>10s}")
    for name, work, nodes in rows:
        fanout = float(work) / float(nodes) if nodes else 0.0
        print(f"# {name:<16s} {work:>10d} {nodes:>10d} {fanout:>10.2f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="primitive_graph_local",
        choices=("mega_kernel", "schedule_ordered_mega", "primitive_queue_local", "primitive_graph_local"),
    )
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--preset", default="future_synthetic")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or "cuda" not in str(args.device):
        raise RuntimeError("bench_multi_task_dispatch_deep requires CUDA.")
    torch.manual_seed(0)
    command, env, readers, mtc_mod = build_mock_command(
        args.num_envs,
        args.device,
        dispatch_backend=args.backend,
        preset=args.preset,
    )
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        _run_warmup(command, env, args.warmup)
        if args.backend == "primitive_graph_local":
            _print_graph_fanout(command)
        totals: dict[str, float] = defaultdict(float)
        for _ in range(args.runs):
            _reset_step(command)
            if args.backend in ("mega_kernel", "schedule_ordered_mega"):
                _time_mega(command, totals)
            elif args.backend == "primitive_queue_local":
                _time_queue(command, totals)
            else:
                _time_graph(command, totals)
            env.episode_length_buf += 1
        print(
            "# MultiTaskCommand deep dispatch benchmark: "
            f"backend={args.backend}, num_envs={args.num_envs}, runs={args.runs}, preset={args.preset}"
        )
        print(f"{'phase':<32s} {'ms/update':>12s}")
        for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            print(f"{name:<32s} {value / args.runs:>12.4f}")
        print(f"{'total':<32s} {sum(totals.values()) / args.runs:>12.4f}")


if __name__ == "__main__":
    main()
