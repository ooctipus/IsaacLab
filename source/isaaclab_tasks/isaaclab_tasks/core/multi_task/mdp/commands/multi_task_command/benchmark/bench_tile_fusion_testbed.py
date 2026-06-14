# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tile-fusion vs inline-mega kernel-design testbed.

A synthetic spike to evaluate whether block-local tile-based producer-consumer
fusion can match (and at higher fanout, beat) the inline mega-kernel pattern
on multi-task command dispatch shapes.

The workload mirrors our actual command primitives:
- Each subtask gathers a fixed-size block of floats from a per-env unified buffer.
- Multiple subtasks may share the same gather block (= same producer signature).
- After gather, each subtask reads a per-(env, slot) target, computes a delta,
  squared error, and writes a scalar output.

Variants compared:

A. ``inline_mega``        - One launch, each thread does gather + delta + output.
                            No shared memory. Matches today's ``dispatch_mega``.
B. ``tile_block_per_env`` - One launch, block = 1 env. Producer phase writes to
                            a shared tile keyed by signature; consumer phase
                            reads from shared. Round-robin producer scheduling.
C. ``tile_block_of_envs`` - One launch, block = M envs. Shared tile holds all
                            envs' producers in the block. Larger block fills
                            warps; each thread maps to one (env_in_block, slot).
D. ``global_materialize`` - Two launches, producers write to a global array and
                            consumers read from it. Matches today's
                            ``primitive_graph_local`` with materialization.

Run with ``./isaaclab.sh -p`` (Warp + torch must be importable):

    ./isaaclab.sh -p -m \\
      isaaclab_tasks.core.multi_task.mdp.commands.benchmark.bench_tile_fusion_testbed \\
      --num_envs 16384 --k_max 8 --gather_size 3 --runs 500

    # Sweep fanout (num_signatures from 1 to k_max):
    ./isaaclab.sh -p -m ... .bench_tile_fusion_testbed --sweep_fanout

The script also runs a quick correctness check before timing — variant outputs
must match the inline baseline elementwise. If correctness fails, perf numbers
are not reported.
"""

from __future__ import annotations

import argparse
import sys

import torch
import warp as wp

wp.init()


# Compile-time tile shape — large enough to cover any preset we care about.
# Per-block shared memory is bounded by (BLOCK_M * MAX_SIGNATURES * GATHER_SIZE * 4) bytes,
# which for the defaults below is 8 * 16 * 4 * 4 = 2 KB. RTX 5090 has ~100 KB
# per-block budget, so we have headroom.
MAX_SIGNATURES = wp.constant(64)
GATHER_SIZE = wp.constant(4)  # max floats per producer (vec4 covers vec3, scalar pad, quat)


# ---------------------------------------------------------------------------
# Variant A: inline mega — no shared memory, one thread per (env, slot).
# ---------------------------------------------------------------------------


@wp.kernel
def kernel_inline_mega(
    state: wp.array2d(dtype=float),  # [num_envs, unified_width]
    targets: wp.array2d(dtype=float),  # [num_envs, k_max]
    slot_signature: wp.array2d(dtype=int),  # [num_envs, k_max] -> sig id
    signature_gather_offset: wp.array(dtype=int),  # [num_signatures]
    output: wp.array2d(dtype=float),  # [num_envs, k_max]
    gather_size: int,
    reduction_width: int,
):
    env, slot = wp.tid()
    sig_id = slot_signature[env, slot]
    g_off = signature_gather_offset[sig_id]

    # Producer: gather + reduction (mimics joint_mech_power-style work).
    val = float(0.0)
    for i in range(gather_size):
        v = state[env, g_off + i]
        val = val + v
    # Optional extra reduction work for "heavy producer" cases.
    for k in range(reduction_width):
        val = val + state[env, g_off + (k % gather_size)]

    # Consumer: scalar delta + squared error.
    target = targets[env, slot]
    delta = target - val
    output[env, slot] = delta * delta


# ---------------------------------------------------------------------------
# Variant B: block-per-env tile fusion. 1 block = 1 env, threads = k_max.
# Producer phase: round-robin assignment of signatures to threads.
# Consumer phase: each thread reads its assigned signature's producer value.
# ---------------------------------------------------------------------------


@wp.kernel
def kernel_tile_block_per_env(
    state: wp.array2d(dtype=float),
    targets: wp.array2d(dtype=float),
    slot_signature: wp.array2d(dtype=int),
    signature_gather_offset: wp.array(dtype=int),
    output: wp.array2d(dtype=float),
    gather_size: int,
    reduction_width: int,
    num_signatures: int,
    k_max: int,
):
    """Block = 1 env. Each thread handles one consumer slot.

    Producer phase uses the all-threads-write pattern: every thread always
    reaches every ``producers[sig_id] = val`` assign even if the value would
    be unused, so the auto-sync emitted by ``tile_assign`` is convergent.
    Threads outside ``[0, num_signatures)`` write zero to a slot they don't
    own; the consumer phase only reads slots in ``[0, num_signatures)`` so
    the unused writes are harmless.
    """
    env, slot = wp.tid()  # env = block idx, slot = thread idx within block
    producers = wp.tile_zeros(shape=(MAX_SIGNATURES,), dtype=float, storage="shared")

    # Phase 1: producers — fixed-iteration loop so all threads reach the
    # same assigns. Each iteration covers k_max signatures (one per thread).
    # Compile-time iteration count = ceil(MAX_SIGNATURES / k_max), capped by
    # MAX_SIGNATURES to avoid out-of-bounds writes.
    for iter_idx in range(MAX_SIGNATURES):
        sig_id = iter_idx * k_max + slot
        val = float(0.0)
        if sig_id < num_signatures:
            g_off = signature_gather_offset[sig_id]
            for i in range(gather_size):
                v = state[env, g_off + i]
                val = val + v
            for k in range(reduction_width):
                val = val + state[env, g_off + (k % gather_size)]
        # Bounded write: every thread writes once per iteration, no divergent assign.
        if sig_id < MAX_SIGNATURES:
            producers[sig_id] = val
        if iter_idx * k_max >= num_signatures:
            break

    # Phase 2: consumer reads producer from shared tile.
    consumer_sig = slot_signature[env, slot]
    val_c = wp.tile_extract(producers, consumer_sig)
    target = targets[env, slot]
    delta = target - val_c
    output[env, slot] = delta * delta


# ---------------------------------------------------------------------------
# Variant D: global materialization (two-kernel split).
# Mimics today's primitive_graph_local materialized path.
# ---------------------------------------------------------------------------


@wp.kernel
def kernel_global_producer(
    state: wp.array2d(dtype=float),
    signature_gather_offset: wp.array(dtype=int),
    producer_values: wp.array2d(dtype=float),  # [num_envs, num_signatures]
    gather_size: int,
    reduction_width: int,
):
    env, sig = wp.tid()
    g_off = signature_gather_offset[sig]
    val = float(0.0)
    for i in range(gather_size):
        v = state[env, g_off + i]
        val = val + v
    for k in range(reduction_width):
        val = val + state[env, g_off + (k % gather_size)]
    producer_values[env, sig] = val


@wp.kernel
def kernel_global_consumer(
    targets: wp.array2d(dtype=float),
    slot_signature: wp.array2d(dtype=int),
    producer_values: wp.array2d(dtype=float),
    output: wp.array2d(dtype=float),
):
    env, slot = wp.tid()
    sig_id = slot_signature[env, slot]
    val = producer_values[env, sig_id]
    target = targets[env, slot]
    delta = target - val
    output[env, slot] = delta * delta


# ---------------------------------------------------------------------------
# Workload generation + driver.
# ---------------------------------------------------------------------------


def make_workload(num_envs: int, k_max: int, num_signatures: int, gather_size: int, device: str, seed: int = 0):
    """Allocate state, targets, and (env, slot) -> signature mapping."""
    torch.manual_seed(seed)
    unified_width = num_signatures * gather_size
    state = torch.randn((num_envs, unified_width), device=device, dtype=torch.float32)
    targets = torch.randn((num_envs, k_max), device=device, dtype=torch.float32)

    # Each signature owns a contiguous gather_size block in unified.
    signature_gather_offset = torch.arange(
        0, num_signatures * gather_size, gather_size, device=device, dtype=torch.int32
    )

    # Slot-to-signature map: each (env, slot) is assigned to a signature in
    # round-robin fashion. Fanout = ceil(k_max / num_signatures).
    slot_signature = torch.zeros((num_envs, k_max), device=device, dtype=torch.int32)
    for slot in range(k_max):
        slot_signature[:, slot] = slot % num_signatures

    output = torch.zeros((num_envs, k_max), device=device, dtype=torch.float32)
    producer_values = torch.zeros((num_envs, max(num_signatures, 1)), device=device, dtype=torch.float32)
    return {
        "state": state,
        "targets": targets,
        "signature_gather_offset": signature_gather_offset,
        "slot_signature": slot_signature,
        "output": output,
        "producer_values": producer_values,
    }


def run_inline_mega(workload, gather_size: int, reduction_width: int) -> torch.Tensor:
    """Launch the inline-mega kernel. Caller must ensure ``output`` is in a
    well-defined state — the kernel overwrites every ``(env, slot)``."""
    out = workload["output"]
    wp.launch(
        kernel_inline_mega,
        dim=(out.shape[0], out.shape[1]),
        inputs=[
            wp.from_torch(workload["state"]),
            wp.from_torch(workload["targets"]),
            wp.from_torch(workload["slot_signature"]),
            wp.from_torch(workload["signature_gather_offset"]),
            wp.from_torch(out),
            gather_size,
            reduction_width,
        ],
        device=str(out.device),
    )
    return out


def run_tile_block_per_env(
    workload, gather_size: int, reduction_width: int, num_signatures: int, k_max: int
) -> torch.Tensor:
    out = workload["output"]
    num_envs = out.shape[0]
    block_dim = max(k_max, 1)
    wp.launch_tiled(
        kernel_tile_block_per_env,
        dim=[num_envs],
        inputs=[
            wp.from_torch(workload["state"]),
            wp.from_torch(workload["targets"]),
            wp.from_torch(workload["slot_signature"]),
            wp.from_torch(workload["signature_gather_offset"]),
            wp.from_torch(out),
            gather_size,
            reduction_width,
            num_signatures,
            k_max,
        ],
        block_dim=block_dim,
        device=str(out.device),
    )
    return out


def run_global_materialize(workload, gather_size: int, reduction_width: int, num_signatures: int) -> torch.Tensor:
    out = workload["output"]
    state_wp = wp.from_torch(workload["state"])
    sig_off_wp = wp.from_torch(workload["signature_gather_offset"])
    prod_wp = wp.from_torch(workload["producer_values"])
    targets_wp = wp.from_torch(workload["targets"])
    slot_sig_wp = wp.from_torch(workload["slot_signature"])
    out_wp = wp.from_torch(out)
    device = str(out.device)
    num_envs = out.shape[0]
    k_max = out.shape[1]

    wp.launch(
        kernel_global_producer,
        dim=(num_envs, num_signatures),
        inputs=[state_wp, sig_off_wp, prod_wp, gather_size, reduction_width],
        device=device,
    )
    wp.launch(
        kernel_global_consumer,
        dim=(num_envs, k_max),
        inputs=[targets_wp, slot_sig_wp, prod_wp, out_wp],
        device=device,
    )
    return out


# ---------------------------------------------------------------------------
# Correctness check.
# ---------------------------------------------------------------------------


def check_correctness(
    num_envs: int, k_max: int, num_signatures: int, gather_size: int, reduction_width: int, device: str
) -> bool:
    """Run all variants on the same input and verify outputs match."""
    workload = make_workload(num_envs, k_max, num_signatures, gather_size, device, seed=42)
    ref = run_inline_mega(workload, gather_size, reduction_width).clone()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()

    # Tile fusion
    tile_out = run_tile_block_per_env(workload, gather_size, reduction_width, num_signatures, k_max).clone()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
    if not torch.allclose(ref, tile_out, atol=1e-4, rtol=1e-4):
        max_abs = (ref - tile_out).abs().max().item()
        print(f"[FAIL] tile_block_per_env mismatch vs inline_mega: max |Δ| = {max_abs:.3e}")
        return False

    # Global materialize
    glob_out = run_global_materialize(workload, gather_size, reduction_width, num_signatures).clone()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
    if not torch.allclose(ref, glob_out, atol=1e-4, rtol=1e-4):
        max_abs = (ref - glob_out).abs().max().item()
        print(f"[FAIL] global_materialize mismatch vs inline_mega: max |Δ| = {max_abs:.3e}")
        return False

    print(f"[OK] correctness check passed (num_signatures={num_signatures}, k_max={k_max})")
    return True


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------


def time_callable(fn, warmup: int, runs: int, device: str) -> float:
    """Time fn() under graph capture if available; return ms/run."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()
        # Capture graph for steady-state replay.
        with wp.ScopedCapture(device=device) as capture:
            fn()
        graph = capture.graph

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            wp.capture_launch(graph)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / runs

    # CPU fallback (uncaptured).
    import time

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / runs


def time_variant(
    name: str,
    workload,
    fn,
    warmup: int,
    runs: int,
    device: str,
):
    ms = time_callable(fn, warmup, runs, device)
    return name, ms


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_one_config(args, num_signatures: int):
    workload = make_workload(args.num_envs, args.k_max, num_signatures, args.gather_size, args.device, seed=0)

    def inline_fn():
        run_inline_mega(workload, args.gather_size, args.reduction_width)

    def tile_fn():
        run_tile_block_per_env(workload, args.gather_size, args.reduction_width, num_signatures, args.k_max)

    def glob_fn():
        run_global_materialize(workload, args.gather_size, args.reduction_width, num_signatures)

    rows = []
    rows.append(time_variant("inline_mega", workload, inline_fn, args.warmup, args.runs, args.device))
    rows.append(time_variant("tile_block_per_env", workload, tile_fn, args.warmup, args.runs, args.device))
    rows.append(time_variant("global_materialize", workload, glob_fn, args.warmup, args.runs, args.device))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--k_max", type=int, default=8)
    parser.add_argument("--gather_size", type=int, default=3, help="Floats per producer (1=scalar, 3=vec3, 4=quat).")
    parser.add_argument(
        "--num_signatures", type=int, default=4, help="Unique producers per env. Fanout ~= k_max / num_signatures."
    )
    parser.add_argument(
        "--reduction_width", type=int, default=0, help="Extra fake reduction work to scale producer cost."
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--sweep_fanout", action="store_true", help="Sweep num_signatures from 1 to k_max.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--skip_correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"# tile-fusion testbed: num_envs={args.num_envs}, k_max={args.k_max}, "
        f"gather_size={args.gather_size}, reduction_width={args.reduction_width}, "
        f"warmup={args.warmup}, runs={args.runs}, device={args.device}"
    )

    if not args.skip_correctness:
        sigs_to_check = [1, max(2, args.k_max // 2), args.k_max] if args.sweep_fanout else [args.num_signatures]
        for s in sigs_to_check:
            if s > args.k_max:
                continue
            ok = check_correctness(args.num_envs, args.k_max, s, args.gather_size, args.reduction_width, args.device)
            if not ok:
                sys.exit(1)
        print()

    if args.sweep_fanout:
        sig_values = [s for s in (1, 2, 4, 8, 16, 32, 64) if s <= min(args.k_max, MAX_SIGNATURES)]
    else:
        sig_values = [args.num_signatures]

    print(f"{'num_sigs':>9s} {'fanout':>8s} | {'inline_mega':>14s} {'tile_per_env':>14s} {'global_mat':>14s}")
    for s in sig_values:
        rows = run_one_config(args, s)
        d = {name: ms for name, ms in rows}
        fanout = float(args.k_max) / float(s)
        print(
            f"{s:>9d} {fanout:>8.2f} | "
            f"{d['inline_mega']:>14.4f} "
            f"{d['tile_block_per_env']:>14.4f} "
            f"{d['global_materialize']:>14.4f}"
        )


if __name__ == "__main__":
    main()
