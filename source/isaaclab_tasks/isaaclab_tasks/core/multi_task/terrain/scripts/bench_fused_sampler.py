# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone benchmark for the fused contact-sampler kernel.

Lets us measure the kernel in isolation -- no IsaacLab init, no
RetargetPipeline, just synthetic patch + FK data fed to
``run_fused_sampler``. Useful for probing the parameter sensitivities
that real training runs hide:

* ``block_dim`` sweep (occupancy vs register pressure).
* JIT compile cost vs steady-state.
* K scaling (linearity of the kernel).

Run via ``./isaaclab.sh -p`` against this file's path. Outputs a small
markdown-friendly table to stdout.

Empirical findings (RTX 5090 / sm_120 / Blackwell)
--------------------------------------------------
Forward kernel uses **154 registers per thread** (measured via
``ptxas -v`` on the cached ``.sm120.ptx``). That register count
explains every block_dim observation we see:

* Blackwell SM register file = 65,536 regs.
* At ``block_dim=512``: 512 × 154 = 78,848 > 65,536 → launch
  fails with ``CUDA error 701`` (silent: Warp prints to stderr but
  raises nothing on the Python side). Detect by timing anomaly,
  not by output checksum -- torch's CUDA pool recycles freed pages,
  so a failed-launch buffer often contains the previous valid run.
* At ``block_dim ∈ {64, 96, 128, 192}``: each converges to the
  same **384 threads/SM** (~37.5% theoretical occupancy on the
  1024-thread/SM SM), giving the flat plateau we measure
  (~40 ms ± 3% at K=2M, ~87 ms ± 2% at K=4M).
* At ``block_dim=256``: 256 × 154 = 39,424 -- only 1 block fits
  per SM (rounded down from 1.66), giving 256 threads/SM and ~25%
  occupancy. That's the 20%-slower cliff we see, every time.
* At ``block_dim=32``: hits the per-SM block-count cap and adds
  block-launch overhead, ending up ~5% slower than the plateau
  despite slightly higher arithmetic occupancy.

K scales as ``time ≈ 1.0 ms + 21.6 ms × (K/1M)`` -- i.e. ~1 ms
fixed launch + hashgrid + alloc overhead, plus ~21.6 ns/candidate
in steady state. Extrapolation to K=60M ≈ 1.3 s for the kernel
proper (sub-second after we add hash-grid build to the budget).

Practical guidance encoded in the launcher:

* Default ``block_dim=128`` is in the safe middle of the 64–192
  plateau. Sweeping isn't worth it: each fresh ``block_dim`` is a
  separate compilation hash and pays a ~1.7 s JIT cost cold.
* Don't bother with Warp's ``wp.tile_load`` / ``wp.tile_store``
  alignment knobs -- those apply to block-cooperative dense ops
  (matmul / FFT). Our irregular per-thread hashgrid lookups can't
  benefit.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import warp as wp

# Late import after wp init so the fused module's wp decorators run cleanly.
from isaaclab_tasks.core.multi_task.terrain.retarget.fused_sampler_kernel import run_fused_sampler


def make_synthetic_inputs(*, n_patches: int, n_tpl: int, area_m: float, device: str = "cuda") -> dict:
    """Create realistic-shape synthetic inputs.

    * Patches uniformly scattered in an ``area_m × area_m`` square.
    * FK templates with foot offsets in [-0.4, 0.4] m (Anymal-scale).
    * Per-foot nominal angles at the four diagonals of a square.
    """
    torch.manual_seed(42)
    patch_pts = torch.empty((n_patches, 3), device=device, dtype=torch.float32)
    patch_pts[:, 0] = (torch.rand(n_patches, device=device) - 0.5) * area_m
    patch_pts[:, 1] = (torch.rand(n_patches, device=device) - 0.5) * area_m
    patch_pts[:, 2] = torch.randn(n_patches, device=device) * 0.05  # small terrain variation

    fk_shape_samples = (torch.rand((n_tpl, 4, 3), device=device) - 0.5) * 0.8
    fk_shape_samples[..., 2] = 0.0  # canonical foot z

    nominal_angles = torch.tensor(
        [math.pi * 0.25, math.pi * 0.75, math.pi * 1.25, math.pi * 1.75],
        device=device,
        dtype=torch.float32,
    )
    return {
        "patch_pts": patch_pts,
        "fk_shape_samples": fk_shape_samples,
        "nominal_angles": nominal_angles,
    }


def time_one(*, K: int, block_dim: int, n_warmup: int, n_runs: int, inputs: dict) -> dict:
    """Run the sampler ``n_warmup + n_runs`` times; return timing + a checksum.

    Note on silent failures: at ``block_dim`` values that exceed the SM's
    register file (e.g. 512 on Blackwell with our register-heavy kernel),
    Warp prints ``CUDA error 701`` to stderr but the wp.launch returns and
    torch.cuda.synchronize() does not raise -- the output buffer keeps the
    garbage from torch.empty. The checksum is recorded for inspection but
    is *not* a reliable failure signal: torch's CUDA pool recycles freed
    pages, so a failed-launch buffer often holds bytes from the previous
    successful run with a matching shape, producing an identical checksum.
    Detection in :func:`block_dim_sweep` therefore uses a timing-anomaly
    heuristic instead.
    """
    times_ms: list[float] = []
    last_checksum: float = float("nan")
    # Warmup: first run includes JIT compile + kernel cache priming.
    for run_idx in range(n_warmup + n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = run_fused_sampler(
            seed=42,
            K=K,
            patch_pts=inputs["patch_pts"],
            fk_shape_samples=inputs["fk_shape_samples"],
            nominal_angles=inputs["nominal_angles"],
            radius=0.15,
            query_radius=0.6,
            outward_pen=10.0,
            force_all_snap=True,
            foot_ground_offset=0.02,
            block_dim=block_dim,
        )
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if run_idx >= n_warmup:
            times_ms.append(dt_ms)
            last_checksum = float(outputs["n_found"].to(torch.int64).sum().item())
        del outputs
    times_ms.sort()
    return {
        "min": times_ms[0],
        "median": times_ms[len(times_ms) // 2],
        "mean": sum(times_ms) / len(times_ms),
        "checksum": last_checksum,
    }


def block_dim_sweep(K: int, inputs: dict, dims: list[int], n_warmup: int = 1, n_runs: int = 3) -> None:
    """Sweep ``block_dim`` and print timing comparison.

    The ``valid`` column flags rows that finished < 10% of the reference
    time -- the unambiguous tell of a silent ``CUDA error 701`` launch
    failure (typically when the per-block register budget exceeds the SM
    register file, e.g. ``block_dim=512`` here at ~154 reg/thread).
    """
    print(f"\n## block_dim sweep at K = {K:,}")
    print("| block_dim | min (ms) | median (ms) | mean (ms) | rel | valid |")
    print("|---|---|---|---|---|---|")
    baseline_min: float | None = None
    for bd in dims:
        result = time_one(K=K, block_dim=bd, n_warmup=n_warmup, n_runs=n_runs, inputs=inputs)
        if baseline_min is None:
            baseline_min = result["min"]
            rel = "1.00×"
            valid = "ref"
        else:
            rel = f"{result['min'] / baseline_min:.2f}×"
            valid = "FAIL" if result["min"] < 0.1 * baseline_min else "ok"
        print(
            f"| {bd:>5d}     | {result['min']:>7.1f}  | {result['median']:>10.1f}  |"
            f" {result['mean']:>7.1f}  | {rel} | {valid} |"
        )


def k_scale_sweep(K_list: list[int], block_dim: int, inputs: dict, n_runs: int = 3) -> None:
    """Run at several K values to check linearity of the kernel."""
    print(f"\n## K scaling at block_dim = {block_dim}")
    print("| K | min (ms) | ms/M | rel | ")
    print("|---|---|---|---|")
    baseline_per_m = None
    for K in K_list:
        result = time_one(K=K, block_dim=block_dim, n_warmup=1, n_runs=n_runs, inputs=inputs)
        per_m = result["min"] / (K / 1_000_000)
        if baseline_per_m is None:
            baseline_per_m = per_m
            rel = "1.00×"
        else:
            rel = f"{per_m / baseline_per_m:.2f}×"
        print(f"| {K:>10,} | {result['min']:>7.1f}  | {per_m:>5.1f}  | {rel} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=2_000_000, help="Candidate count for the block_dim sweep.")
    parser.add_argument("--n_patches", type=int, default=200_000)
    parser.add_argument("--n_tpl", type=int, default=1024)
    parser.add_argument("--area_m", type=float, default=60.0)
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument(
        "--block_dims",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96, 128, 192, 256, 384, 512],
        help="Block sizes to sweep.",
    )
    parser.add_argument(
        "--k_scale",
        type=int,
        nargs="*",
        default=[200_000, 1_000_000, 4_000_000],
        help="K values for the linearity check.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    # Standalone scripts (no IsaacLab init) need to bring up the Warp runtime
    # explicitly before any wp.from_torch / wp.launch call.
    wp.init()

    print(
        f"# Fused-sampler benchmark\n"
        f"warp={wp.__version__}, torch={torch.__version__}, "
        f"cuda={torch.cuda.get_device_name(0)}, "
        f"n_patches={args.n_patches}, n_tpl={args.n_tpl}, area={args.area_m}m"
    )
    inputs = make_synthetic_inputs(n_patches=args.n_patches, n_tpl=args.n_tpl, area_m=args.area_m)

    block_dim_sweep(args.K, inputs, args.block_dims, n_warmup=args.n_warmup, n_runs=args.n_runs)

    if args.k_scale:
        # Use the best block_dim from the sweep -- preempt the sweep result here
        # by hard-coding 128 since that's what we're committing as the default.
        k_scale_sweep(args.k_scale, block_dim=128, inputs=inputs, n_runs=args.n_runs)


if __name__ == "__main__":
    main()
