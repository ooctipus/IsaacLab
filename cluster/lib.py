#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cluster job submission library.

Handles argument parsing, preset brace expansion, Cartesian sweep products,
and osmo job launching. Used by submit.sh as the main logic engine.

Usage from shell:
    python3 cluster/lib.py submit <script> [args...]
    python3 cluster/lib.py pbt <script> [args...]
    python3 cluster/lib.py cancel <prefix-N> <count>
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime

SPEC = "docker/cluster/multi_node.yaml"

POOL_TO_PLATFORM = {
    # H100
    "groot-h100-ci-02": "dgx-h100",
    "isaac-dev-h100-01": "dgx-h100",
    # L40
    "groot-l40-01": "ovx-l40",
    "groot-l40-02": "ovx-l40",
    "groot-l40-03": "ovx-l40",
    "groot-l40-04": "ovx-l40",
    "groot-l40-ci-02": "ovx-l40",
    "groot-l40-ci-03": "ovx-l40",
    "isaac-dev-l40-03": "ovx-l40",
    # L40S
    "groot-l40s-01": "ovx-l40s",
    "groot-l40s-03": "ovx-l40s",
    "isaac-dev-l40s-04": "ovx-l40s",
    "isaac-dex-l40s-02": "ovx-l40s",
    "isaac-dex-l40s-03": "ovx-l40s",
    "isaac-dex-l40s-04": "ovx-l40s",
    # GB200
    "groot-gb200-02": "gb200",
    # AGX Orin
    "isaac-hil": "agx-orin-jp6",
    "isaac-nightly": "agx-orin-jp6",
    "isaac-sqa": "agx-orin-jp6-realsense",
}

CLUSTER_DEFAULTS = {
    "image": "factory",
    "num_gpu": "1",
    "num_cpu": "2",
    "num_node": "1",
    "memory": "64",
    "storage": "64",
    "platform": "dgx-h100",
    "dataset": "isaac-lab-ppo-model",
    "master_port": "29400",
}

CLUSTER_KEY_ORDER = [
    "image",
    "num_gpu",
    "num_cpu",
    "memory",
    "platform",
    "dataset",
    "num_node",
    "storage",
    "master_port",
]

BOOL_FLAGS = {"--video", "--enable_cameras"}


def _is_truthy(val: str) -> bool:
    return val.lower() in ("1", "true", "yes", "on")


def _is_falsy(val: str) -> bool:
    return val.lower() in ("0", "false", "no", "off")


def _is_bracketed(val: str) -> bool:
    """True if val is wrapped in [], (), or {} -- not a sweep."""
    if len(val) < 2:
        return False
    return (val[0], val[-1]) in (("[", "]"), ("(", ")"), ("{", "}"))


def expand_preset_braces(val: str) -> list[str]:
    """Expand brace groups in a preset value into Cartesian product.

    Examples:
        "{a,b},{x,y}" -> ["a,x", "a,y", "b,x", "b,y"]
        "{a,b},fixed"  -> ["a,fixed", "b,fixed"]
        "plain"        -> ["plain"]
    """
    if "{" not in val:
        return [val]

    segments: list[tuple[str, list[str]]] = []
    remaining = val
    while "{" in remaining and "}" in remaining:
        before = remaining[: remaining.index("{")]
        rest = remaining[remaining.index("{") + 1 :]
        inner = rest[: rest.index("}")]
        remaining = rest[rest.index("}") + 1 :]
        if before:
            segments.append(("fixed", [before]))
        segments.append(("sweep", inner.split(",")))
    if remaining:
        segments.append(("fixed", [remaining]))

    combos = [""]
    for kind, values in segments:
        if kind == "fixed":
            combos = [c + values[0] for c in combos]
        else:
            combos = [c + v for c in combos for v in values]
    return combos


@dataclass
class ParsedArgs:
    """Result of parsing command-line arguments."""

    cluster: dict[str, str] = field(default_factory=lambda: dict(CLUSTER_DEFAULTS))
    pool: str = ""
    fixed: dict[str, str] = field(default_factory=dict)
    sweep: dict[str, list[str]] = field(default_factory=dict)
    cli_flags: list[str] = field(default_factory=list)
    hydra_dels: list[str] = field(default_factory=list)
    run_name: str = ""


_CLUSTER_KEYS = set(CLUSTER_DEFAULTS) | {"pool", "run_name"}


def _try_cluster_key(arg: str, next_arg: str | None, p: ParsedArgs) -> int:
    """Try to consume arg as a cluster/special key. Returns args consumed (0, 1, or 2)."""
    if "=" in arg:
        key, val = arg.split("=", 1)
        key_bare = key.lstrip("-")
    elif arg.lstrip("-") in _CLUSTER_KEYS and next_arg is not None:
        key_bare = arg.lstrip("-")
        val = next_arg
        return _set_cluster_key(key_bare, val, p) * 2
    else:
        return 0

    if key_bare not in _CLUSTER_KEYS:
        return 0
    _set_cluster_key(key_bare, val, p)
    return 1


def _set_cluster_key(key_bare: str, val: str, p: ParsedArgs):
    if key_bare == "pool":
        p.pool = val
    elif key_bare == "run_name":
        p.run_name = val
    elif key_bare in CLUSTER_DEFAULTS:
        p.cluster[key_bare] = val
    return True


def parse_args(raw_args: list[str]) -> ParsedArgs:
    """Parse arguments into cluster settings, fixed args, sweep dims, and flags."""
    p = ParsedArgs()
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        next_arg = raw_args[i + 1] if i + 1 < len(raw_args) else None

        # Bool flags: --video, --enable_cameras
        bare_flag = arg.split("=")[0] if "=" in arg else arg
        if bare_flag in BOOL_FLAGS:
            if "=" in arg:
                val = arg.split("=", 1)[1]
                if _is_truthy(val):
                    p.cli_flags.append(bare_flag)
                elif not _is_falsy(val):
                    print(f"Invalid boolean for {bare_flag}: {val}", file=sys.stderr)
                    sys.exit(2)
            else:
                p.cli_flags.append(arg)
            i += 1
            continue

        # Hydra delete operator
        if arg.startswith("~"):
            p.hydra_dels.append(arg)
            i += 1
            continue

        # Cluster/special keys: pool, platform, num_gpu, etc.
        # Works with key=val, --key=val, or --key val
        consumed = _try_cluster_key(arg, next_arg, p)
        if consumed:
            i += consumed
            continue

        # Bare flags (--distributed, etc.)
        if arg.startswith("--") and "=" not in arg:
            p.cli_flags.append(arg)
            i += 1
            continue

        # key=val (app args: fixed, sweep, or presets)
        if "=" in arg:
            key, val = arg.split("=", 1)
            key_bare = key.lstrip("-")

            if key_bare == "presets" and "{" in val:
                p.sweep[key] = expand_preset_braces(val)
            elif "," in val and not _is_bracketed(val) and key_bare != "presets":
                p.sweep[key] = val.split(",")
            else:
                p.fixed[key] = val
            i += 1
            continue

        print(f"Invalid argument: {arg}", file=sys.stderr)
        sys.exit(1)

    # Resolve pool -> platform
    if p.pool:
        if p.pool not in POOL_TO_PLATFORM:
            print(f"Error: unknown pool '{p.pool}'.", file=sys.stderr)
            sys.exit(1)
        p.cluster["platform"] = POOL_TO_PLATFORM[p.pool]
    elif not p.pool:
        p.pool = "isaac-dev-h100-01"

    return p


def build_cluster_str(cluster: dict[str, str]) -> str:
    return " ".join(f"{k}={cluster[k]}" for k in CLUSTER_KEY_ORDER)


def build_combos(sweep: dict[str, list[str]]) -> list[dict[str, str]]:
    """Cartesian product of all sweep dimensions."""
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def derive_run_name(base: str, combo: dict[str, str]) -> str:
    """Build run name from user base + sweep values."""
    suffix = ",".join(combo.values()) if combo else ""
    parts = [p for p in (base, suffix) if p]
    return ",".join(parts)


def build_fixed_str(fixed: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in fixed.items())


def build_cmd(
    script: str,
    pool: str,
    cluster_str: str,
    fixed_str: str,
    combo: dict[str, str],
    run_name: str,
    hydra_dels: list[str],
    cli_flags: list[str],
) -> list[str]:
    """Build the osmo command as a list of args (one element per shell token)."""
    combo_str = " ".join(f"{k}={v}" for k, v in combo.items())
    args_parts = [fixed_str, combo_str]
    if run_name:
        args_parts.append(f"--run_name={run_name}")
    args_parts.extend(hydra_dels)
    args_parts.extend(cli_flags)
    all_args = " ".join(p for p in args_parts if p)

    return (
        [
            "osmo",
            "workflow",
            "submit",
            SPEC,
            "--pool",
            pool,
            "--set",
            f"script={script}",
        ]
        + cluster_str.split()
        + [f"args={all_args}"]
    )


def launch_job(
    script: str,
    pool: str,
    cluster_str: str,
    fixed_str: str,
    combo: dict[str, str],
    run_name: str,
    hydra_dels: list[str],
    cli_flags: list[str],
    dry_run: bool = False,
):
    cmd = build_cmd(script, pool, cluster_str, fixed_str, combo, run_name, hydra_dels, cli_flags)
    print(f"LAUNCHING: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=False)


def do_cancel(prefix_pattern: str, count: int):
    prefix, start = prefix_pattern.rsplit("-", 1)
    start = int(start)
    for i in range(start, start + count):
        subprocess.run(["osmo", "workflow", "cancel", f"{prefix}-{i}"], check=False)


def do_pbt(script: str, p: ParsedArgs, dry_run: bool = False):
    if "num_populations" not in p.fixed:
        print("Missing required: num_populations", file=sys.stderr)
        sys.exit(1)
    num_pop = int(p.fixed.pop("num_populations"))
    cluster_str = build_cluster_str(p.cluster)
    dt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for idx in range(num_pop):
        pbt_args = [
            "agent.pbt.enabled=True",
            f"agent.pbt.num_policies={num_pop}",
            f"agent.pbt.policy_idx={idx}",
            f"agent.pbt.workspace={num_pop}agents_{dt}",
            "agent.pbt.directory=/mnt/amlfs-05/shared/workspace/octi",
            f"--wandb-name={idx}_{num_pop}",
            "--seed=-1",
        ]
        for k, v in p.fixed.items():
            pbt_args.append(f"{k}={v}")

        all_args = " ".join(pbt_args + p.hydra_dels + p.cli_flags)
        args_quoted = f"args={all_args}"
        cmd = (
            [
                "osmo",
                "workflow",
                "submit",
                SPEC,
                "--pool",
                p.pool,
                "--set",
                f"script={script}",
            ]
            + cluster_str.split()
            + [args_quoted]
        )
        print(f"+ {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=False)


def do_submit(script: str, p: ParsedArgs, dry_run: bool = False):
    cluster_str = build_cluster_str(p.cluster)
    fixed_str = build_fixed_str(p.fixed)
    combos = build_combos(p.sweep)

    for combo in combos:
        run_name = derive_run_name(p.run_name, combo)
        launch_job(script, p.pool, cluster_str, fixed_str, combo, run_name, p.hydra_dels, p.cli_flags, dry_run=dry_run)


def main():
    if len(sys.argv) < 2:
        print("Usage: lib.py {submit|pbt|cancel} <script> [args...]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    dry_run = os.environ.get("DRY_RUN", "") == "1"

    if mode == "cancel":
        if len(sys.argv) != 4:
            print("Usage: lib.py cancel <prefix-N> <count>", file=sys.stderr)
            sys.exit(1)
        do_cancel(sys.argv[2], int(sys.argv[3]))
    elif mode in ("submit", "pbt"):
        if len(sys.argv) < 3:
            print(f"Usage: lib.py {mode} <script> [args...]", file=sys.stderr)
            sys.exit(1)
        script = sys.argv[2]
        if not os.path.isfile(script):
            print(f"Script not found: {script}", file=sys.stderr)
            sys.exit(1)
        p = parse_args(sys.argv[3:])
        if not os.path.isfile(SPEC):
            print(f"Spec file not found: {SPEC}", file=sys.stderr)
            sys.exit(1)
        if mode == "pbt":
            do_pbt(script, p, dry_run=dry_run)
        else:
            do_submit(script, p, dry_run=dry_run)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
