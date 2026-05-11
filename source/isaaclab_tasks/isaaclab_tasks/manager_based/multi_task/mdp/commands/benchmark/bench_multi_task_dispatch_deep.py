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

from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.kernels_wp import (
    compute_contact_predicate_mask,
    compute_dense_graph_producers,
    compute_direct_quat_nodes,
    compute_direct_scalar_nodes,
    compute_direct_vec3_nodes,
    compute_scalar_sum_nodes,
    dispatch_graph_contact_pair_diff,
    dispatch_graph_contact_sum,
    dispatch_graph_contact_vector,
    dispatch_graph_dense,
    dispatch_graph_direct_quat,
    dispatch_graph_direct_scalar,
    dispatch_graph_direct_vec3,
    dispatch_graph_scalar_sum,
    dispatch_primitive_local_direct_quat,
    dispatch_primitive_local_direct_scalar,
    dispatch_primitive_local_direct_vec3,
    dispatch_primitive_local_scalar_sum,
)
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.mega_kernel.compose import compose_warp
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.mega_kernel.execute import dispatch_mega_warp
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.mega_kernel.read import fill_unified_buffer_warp
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.mega_kernel.rotation import (
    rotate_canonical_slots_to_body_frame_warp,
)
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.primitive_queue_local.execute import (
    _PRIMITIVE_KERNELS,
)
from isaaclab_tasks.manager_based.multi_task.mdp.commands.impl.schedules import (
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
        inputs=[
            plan.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_graph_direct_vec3(command, plan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_vec3,
        dim=count,
        inputs=[
            plan.queue,
            plan.vec3_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_vec3_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_graph_direct_scalar(command, plan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_scalar,
        dim=count,
        inputs=[
            plan.queue,
            plan.scalar_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_scalar_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_graph_direct_quat(command, plan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_quat,
        dim=count,
        inputs=[
            plan.queue,
            plan.quat_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_quat_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_graph_scalar_sum(command, plan, count: int) -> None:
    wp.launch(
        dispatch_graph_scalar_sum,
        dim=count,
        inputs=[
            plan.queue,
            plan.scalar_sum_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.scalar_sum_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_graph_contact(command, plan, kernel, count: int) -> None:
    wp.launch(
        kernel,
        dim=count,
        inputs=[
            plan.queue,
            plan.contact_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.contact_mask_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_dense_graph(command, plan) -> None:
    wp.launch(
        dispatch_graph_dense,
        dim=(command.num_envs, command.k_max),
        inputs=[
            plan.env_slots,
            plan.subtask_schedule_ids_wp,
            plan.vec3_nodes.queue,
            plan.scalar_nodes.queue,
            plan.quat_nodes.queue,
            plan.scalar_sum_nodes.queue,
            plan.contact_nodes.queue,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_vec3_wp,
            plan.direct_scalar_wp,
            plan.direct_quat_wp,
            plan.scalar_sum_wp,
            plan.contact_mask_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
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
    counts = plan.schedule_counts_py
    _time_phase(totals, "read", lambda: fill_unified_buffer_warp(command, plan))

    if plan.use_dense_graph_consumer:
        total_signature_count = (
            plan.vec3_signature_count
            + plan.scalar_signature_count
            + plan.quat_signature_count
            + plan.scalar_sum_signature_count
            + plan.contact_signature_count
        )
        if total_signature_count != 0:
            _time_phase(
                totals,
                "dense_graph_producers",
                lambda: wp.launch(
                    compute_dense_graph_producers,
                    dim=(command.num_envs, total_signature_count),
                    inputs=[
                        plan.vec3_nodes.queue,
                        plan.scalar_nodes.queue,
                        plan.quat_nodes.queue,
                        plan.scalar_sum_nodes.queue,
                        plan.contact_nodes.queue,
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
                ),
            )
        _time_phase(totals, "dense_graph_consumer", lambda: _launch_dense_graph(command, plan))
        _time_phase(totals, "rotate", lambda: rotate_canonical_slots_to_body_frame_warp(command, plan))
        _time_phase(totals, "compose", lambda: compose_warp(command, plan))
        return

    if counts[SCHEDULE_DIRECT_VEC3_DELTA] != 0:
        if plan.use_vec3_graph:
            _time_phase(
                totals,
                "direct_vec3_producer",
                lambda: wp.launch(
                    compute_direct_vec3_nodes,
                    dim=plan.vec3_count,
                    inputs=[plan.vec3_nodes.queue, plan.spec, plan.state, plan.direct_vec3_wp],
                    device=str(command.device),
                ),
            )
            _time_phase(
                totals,
                "direct_vec3_consumer",
                lambda: _launch_graph_direct_vec3(command, plan, counts[SCHEDULE_DIRECT_VEC3_DELTA]),
            )
        else:
            _time_phase(
                totals,
                "direct_vec3_fused",
                lambda: _launch_local_kernel(
                    command, plan, dispatch_primitive_local_direct_vec3, counts[SCHEDULE_DIRECT_VEC3_DELTA]
                ),
            )
    if counts[SCHEDULE_DIRECT_SCALAR_DELTA] != 0:
        if plan.use_scalar_graph:
            _time_phase(
                totals,
                "direct_scalar_producer",
                lambda: wp.launch(
                    compute_direct_scalar_nodes,
                    dim=plan.scalar_count,
                    inputs=[plan.scalar_nodes.queue, plan.spec, plan.state, plan.direct_scalar_wp],
                    device=str(command.device),
                ),
            )
            _time_phase(
                totals,
                "direct_scalar_consumer",
                lambda: _launch_graph_direct_scalar(command, plan, counts[SCHEDULE_DIRECT_SCALAR_DELTA]),
            )
        else:
            _time_phase(
                totals,
                "direct_scalar_fused",
                lambda: _launch_local_kernel(
                    command, plan, dispatch_primitive_local_direct_scalar, counts[SCHEDULE_DIRECT_SCALAR_DELTA]
                ),
            )
    if counts[SCHEDULE_DIRECT_QUAT_DELTA] != 0:
        if plan.use_quat_graph:
            _time_phase(
                totals,
                "direct_quat_producer",
                lambda: wp.launch(
                    compute_direct_quat_nodes,
                    dim=plan.quat_count,
                    inputs=[plan.quat_nodes.queue, plan.spec, plan.state, plan.direct_quat_wp],
                    device=str(command.device),
                ),
            )
            _time_phase(
                totals,
                "direct_quat_consumer",
                lambda: _launch_graph_direct_quat(command, plan, counts[SCHEDULE_DIRECT_QUAT_DELTA]),
            )
        else:
            _time_phase(
                totals,
                "direct_quat_fused",
                lambda: _launch_local_kernel(
                    command, plan, dispatch_primitive_local_direct_quat, counts[SCHEDULE_DIRECT_QUAT_DELTA]
                ),
            )
    if counts[SCHEDULE_SCALAR_SUM_DELTA] != 0:
        if plan.use_scalar_sum_graph:
            _time_phase(
                totals,
                "scalar_sum_producer",
                lambda: wp.launch(
                    compute_scalar_sum_nodes,
                    dim=plan.scalar_sum_count,
                    inputs=[plan.scalar_sum_nodes.queue, plan.spec, plan.state, plan.scalar_sum_wp],
                    device=str(command.device),
                ),
            )
            _time_phase(
                totals,
                "scalar_sum_consumer",
                lambda: _launch_graph_scalar_sum(command, plan, counts[SCHEDULE_SCALAR_SUM_DELTA]),
            )
        else:
            _time_phase(
                totals,
                "scalar_sum_fused",
                lambda: _launch_local_kernel(
                    command, plan, dispatch_primitive_local_scalar_sum, counts[SCHEDULE_SCALAR_SUM_DELTA]
                ),
            )
    if plan.contact_count != 0:
        _time_phase(
            totals,
            "contact_producer",
            lambda: wp.launch(
                compute_contact_predicate_mask,
                dim=plan.contact_count,
                inputs=[plan.contact_nodes.queue, plan.spec, plan.state, plan.contact_mask_wp],
                device=str(command.device),
            ),
        )
    if counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA] != 0:
        _time_phase(
            totals,
            "contact_vector_consumer",
            lambda: _launch_graph_contact(
                command, plan, dispatch_graph_contact_vector, counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA]
            ),
        )
    if counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA] != 0:
        _time_phase(
            totals,
            "contact_sum_consumer",
            lambda: _launch_graph_contact(
                command, plan, dispatch_graph_contact_sum, counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA]
            ),
        )
    if counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA] != 0:
        _time_phase(
            totals,
            "contact_pair_diff_consumer",
            lambda: _launch_graph_contact(
                command, plan, dispatch_graph_contact_pair_diff, counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA]
            ),
        )
    _time_phase(totals, "rotate", lambda: rotate_canonical_slots_to_body_frame_warp(command, plan))
    _time_phase(totals, "compose", lambda: compose_warp(command, plan))


def _print_graph_fanout(command) -> None:
    plan = command._backend.plan
    counts = plan.schedule_counts_py
    print("# graph sharing")
    rows = (
        ("direct_vec3", counts[SCHEDULE_DIRECT_VEC3_DELTA], plan.vec3_count, plan.use_vec3_graph),
        ("direct_scalar", counts[SCHEDULE_DIRECT_SCALAR_DELTA], plan.scalar_count, plan.use_scalar_graph),
        ("direct_quat", counts[SCHEDULE_DIRECT_QUAT_DELTA], plan.quat_count, plan.use_quat_graph),
        ("scalar_sum", counts[SCHEDULE_SCALAR_SUM_DELTA], plan.scalar_sum_count, plan.use_scalar_sum_graph),
        (
            "contact",
            counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA]
            + counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA]
            + counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA],
            plan.contact_count,
            True,
        ),
    )
    print(f"# {'producer':<16s} {'work':>10s} {'nodes':>10s} {'fanout':>10s} {'materialized':>12s}")
    for name, work, nodes, materialized in rows:
        fanout = float(work) / float(nodes) if nodes else 0.0
        print(f"# {name:<16s} {work:>10d} {nodes:>10d} {fanout:>10.2f} {str(materialized):>12s}")


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
