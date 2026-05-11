# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump ``CSRGraph`` fan-out statistics for each benchmark preset.

Drives the Phase 2.5 decision gate for the primitive-graph-local backend
refactor: is fan-out high enough that producer-coherent reordering
(Phase 3) is the high-leverage bet? Is it skewed enough that merge-based
producer load balancing (Phase 4) is worth pursuing?

Prints one row per (preset, producer kind) tuple. Five kinds:

* ``vec3`` — direct vec3 fields (body_lin_vel_w, body_ang_vel_w, body_pos_w, ...)
* ``scalar`` — direct scalar fields (joint_pos, joint_vel, body_pos_z)
* ``quat`` — direct quat fields (body_quat_w)
* ``scalar_sum`` — sum-reduction primitives (|joint_mech_power|)
* ``contact`` — contact-threshold primitives

Columns: ``producers`` (unique signatures), ``edges`` (active consumers),
``mean_fo`` and ``max_fo`` (fan-out stats), ``skew`` = ``max_fo / mean_fo``
(load-balancing pressure), and the full ``fanout_histogram`` keyed by
consumer-count.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from unittest.mock import patch

import torch

if __package__:
    from .mock_command import build_mock_command
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mock_command import build_mock_command


_KIND_FIELDS = ("vec3_nodes", "scalar_nodes", "quat_nodes", "scalar_sum_nodes", "contact_nodes")
_KIND_LABELS = ("vec3", "scalar", "quat", "scalar_sum", "contact")
_PRESETS: tuple[str | None, ...] = (None, "shared_direct", "future_synthetic", "future_synthetic_heavy")


def _dump_preset(preset_name: str | None, num_envs: int, device: str) -> None:
    label = preset_name if preset_name is not None else "default"
    torch.manual_seed(0)
    command, env, readers, mtc_mod = build_mock_command(
        num_envs, device, dispatch_backend="primitive_graph_local", preset=preset_name
    )
    del env  # unused — graph stats don't depend on the runtime env
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        plan = command._backend.plan  # type: ignore[attr-defined]
        print(f"\n# preset = {label} (num_envs = {num_envs})")
        header = (
            f"{'kind':<12s} {'producers':>10s} {'edges':>8s} {'mean_fo':>10s} {'max_fo':>8s} {'skew':>8s}  histogram"
        )
        print(header)
        print("-" * len(header))
        for kind_field, kind_label in zip(_KIND_FIELDS, _KIND_LABELS, strict=True):
            table = getattr(plan, kind_field)
            g = table.csr_graph
            if g.num_producers == 0:
                print(f"{kind_label:<12s} {0:>10d} {0:>8d} {'-':>10s} {'-':>8s} {'-':>8s}  {{}}")
                continue
            skew = g.max_fanout / g.mean_fanout
            print(
                f"{kind_label:<12s} {g.num_producers:>10d} {g.num_active_consumers:>8d} "
                f"{g.mean_fanout:>10.2f} {g.max_fanout:>8d} {skew:>8.2f}  {g.fanout_histogram}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=16384, help="Parallel envs in the mock command.")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for the mock command's tensors.",
    )
    args = parser.parse_args()
    for preset in _PRESETS:
        _dump_preset(preset, args.num_envs, args.device)


if __name__ == "__main__":
    main()
