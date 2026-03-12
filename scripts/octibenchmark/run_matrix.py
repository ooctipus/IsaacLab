# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic entry point for running benchmark matrices.

Discovers example modules from :mod:`octibenchmark.examples` and runs
the matrices they define. Each example module must expose an
``ALL_MATRICES`` dict mapping names to :class:`BenchmarkMatrix` instances.

Usage — list available examples::

    python scripts/octibenchmark/run_matrix.py --list

Usage — dry run all Shadow Hand matrices::

    python scripts/octibenchmark/run_matrix.py \\
        --example shadow_hand_matrix --dry_run

Usage — run specific matrices with kernel patterns::

    python scripts/octibenchmark/run_matrix.py \\
        --example shadow_hand_matrix \\
        --matrices shadow_vision_nonrl shadow_state_nonrl \\
        --output_dir /tmp/shadow_bench \\
        --kernel_patterns "index" "mask" "copy"
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import octibenchmark.examples as _examples_pkg


def _discover_examples() -> dict[str, dict]:
    """Discover all example modules that define ALL_MATRICES."""
    examples = {}
    pkg_path = os.path.dirname(_examples_pkg.__file__)
    for importer, name, ispkg in pkgutil.iter_modules([pkg_path]):
        if name.startswith("_"):
            continue
        mod = importlib.import_module(f"octibenchmark.examples.{name}")
        matrices = getattr(mod, "ALL_MATRICES", None)
        if matrices:
            examples[name] = matrices
    return examples


def main():
    examples = _discover_examples()

    parser = argparse.ArgumentParser(
        description="Run benchmark matrices from example modules.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and their matrices, then exit.",
    )
    parser.add_argument(
        "--example",
        type=str,
        default=None,
        choices=list(examples.keys()) if examples else None,
        help="Which example module to use.",
    )
    parser.add_argument(
        "--matrices",
        nargs="+",
        default=None,
        help="Which matrices to run from the example. Default: all.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for nsys files.")
    parser.add_argument("--wandb_project", type=str, default="isaaclab-benchmarks")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--kernel_patterns", nargs="*", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom tag for wandb (e.g. GPU name: 'RTX4090', 'GB200'). "
        "Appears in wandb group/run names to distinguish identical "
        "experiments run on different hardware.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print run summary without executing.")
    parser.add_argument("--verbose", action="store_true", help="Show full nsys commands in dry run.")
    args = parser.parse_args()

    if args.list:
        print("Available examples:\n")
        for name, matrices in sorted(examples.items()):
            print(f"  {name}")
            total = 0
            for mname, matrix in matrices.items():
                n = len(matrix.runs())
                total += n
                print(f"    {mname:<40} {n:>3} runs")
            print(f"    {'─' * 40} ───────")
            print(f"    {'total':<40} {total:>3} runs")
            print()
        return

    if args.example is None:
        parser.error("--example is required (use --list to see available examples)")

    all_matrices = examples[args.example]
    matrix_names = args.matrices or list(all_matrices.keys())

    for name in matrix_names:
        if name not in all_matrices:
            parser.error(f"Unknown matrix {name!r}. Available: {list(all_matrices.keys())}")

    # Create one shared output directory for all matrices
    output_dir = args.output_dir
    if output_dir is None and not args.dry_run:
        output_dir = tempfile.mkdtemp(prefix="octibench_")
        print(f"[run_matrix] Output directory: {output_dir}")

    total_runs = 0
    for name in matrix_names:
        matrix = all_matrices[name]
        print(f"\n{'#' * 60}")
        print(f"# Matrix: {name}")
        print(f"{'#' * 60}")

        matrix.execute(
            output_dir=output_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            kernel_patterns=args.kernel_patterns,
            no_wandb=args.no_wandb,
            tag=args.tag,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_runs += len(matrix.runs())

    print(f"\n[run_matrix] Total runs across selected matrices: {total_runs}")
    if output_dir and not args.dry_run:
        print(f"[run_matrix] All results in: {output_dir}")


if __name__ == "__main__":
    main()
