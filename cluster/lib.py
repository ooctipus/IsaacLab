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
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime

SPEC = "docker/cluster/multi_node.yaml"

POOL_TO_PLATFORM = {
    # H100
    "groot-h100-01": "dgx-h100",
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
    "isaac-lab-l30s-03": "ovx-l40s",
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
AUTO_RESOURCE_KEYS = {"num_cpu", "memory", "storage"}


@dataclass(frozen=True)
class PoolNodeResources:
    """Free and allocatable resources for one pool node."""

    hostname: str
    available_gpu: int
    available_cpu: int
    available_memory: int
    available_storage: int
    allocatable_gpu: int
    allocatable_cpu: int
    allocatable_memory: int
    allocatable_storage: int


@dataclass
class ResourcePlan:
    """Resource planning diagnostics."""

    source: str
    changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


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
    cluster_overrides: set[str] = field(default_factory=set)
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
        p.cluster_overrides.add(key_bare)
    return True


def _cluster_int(p: ParsedArgs, key: str) -> int:
    try:
        return int(p.cluster[key])
    except ValueError:
        print(f"Error: {key} must be an integer, got '{p.cluster[key]}'.", file=sys.stderr)
        sys.exit(2)


def _positive_cluster_int(p: ParsedArgs, key: str) -> int:
    value = _cluster_int(p, key)
    if value < 1:
        print(f"Error: {key} must be positive, got '{value}'.", file=sys.stderr)
        sys.exit(2)
    return value


def _parse_int_quantity(value: object) -> int:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("m"):
            return int(value[:-1]) // 1000
    return int(value)


def _memory_to_gib(value: object) -> int:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("Ki"):
            return int(value[:-2]) // (1024 * 1024)
    return int(value)


def _storage_to_gib(value: object) -> int:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("Ki"):
            return int(value[:-2]) // (1024 * 1024)
        value_int = int(value)
    else:
        value_int = int(value)
    if value_int > 1024 * 1024:
        return value_int // (1024**3)
    return value_int


def _platform_resources(raw: dict, pool: str, platform: str, field: str) -> dict | None:
    return raw.get(field, {}).get(pool, {}).get(platform)


def _node_resources_from_osmo(raw: dict, pool: str, platform: str) -> PoolNodeResources | None:
    available = _platform_resources(raw, pool, platform, "platform_available_fields")
    allocatable = _platform_resources(raw, pool, platform, "platform_workflow_allocatable_fields")
    if allocatable is None:
        allocatable = _platform_resources(raw, pool, platform, "platform_allocatable_fields")
    if available is None or allocatable is None:
        return None
    return PoolNodeResources(
        hostname=str(raw.get("hostname", "unknown")),
        available_gpu=_parse_int_quantity(available.get("gpu", 0)),
        available_cpu=_parse_int_quantity(available.get("cpu", 0)),
        available_memory=_memory_to_gib(available.get("memory", 0)),
        available_storage=_storage_to_gib(available.get("storage", 0)),
        allocatable_gpu=_parse_int_quantity(allocatable.get("gpu", 0)),
        allocatable_cpu=_parse_int_quantity(allocatable.get("cpu", 0)),
        allocatable_memory=_memory_to_gib(allocatable.get("memory", 0)),
        allocatable_storage=_storage_to_gib(allocatable.get("storage", 0)),
    )


def _load_pool_node_resources(pool: str, platform: str) -> tuple[list[PoolNodeResources], str | None]:
    cmd = ["osmo", "resource", "list", "--pool", pool, "--mode", "free", "--format-type", "json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return [], f"could not query OSMO resources: {exc}"
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()
        detail = reason[-1] if reason else f"exit code {result.returncode}"
        return [], f"could not query OSMO resources: {detail}"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], f"could not parse OSMO resource JSON: {exc}"

    nodes = []
    for item in data.get("resources", []):
        node = _node_resources_from_osmo(item, pool, platform)
        if node is not None:
            nodes.append(node)
    if not nodes:
        return [], f"OSMO returned no nodes for pool={pool} platform={platform}"
    return nodes, None


def _default_auto_resources(num_gpu: int) -> dict[str, int]:
    cpu = 16 if num_gpu == 1 else num_gpu * 30
    memory = 64 if num_gpu == 1 else num_gpu * 128
    return {"num_cpu": cpu, "memory": memory, "storage": 128}


def _round_resource(key: str, value: int, num_gpu: int) -> int:
    if key == "num_cpu":
        step = max(num_gpu, 1)
    else:
        step = 16
    return max(value // step * step, 1)


def _nth_largest(values: list[int], n: int) -> int:
    return sorted(values, reverse=True)[n - 1]


def _resource_value(node: PoolNodeResources, key: str, *, available: bool) -> int:
    prefix = "available" if available else "allocatable"
    if key == "num_cpu":
        return getattr(node, f"{prefix}_cpu")
    return getattr(node, f"{prefix}_{key}")


def _proportional_resource_cap(node: PoolNodeResources, key: str, num_gpu: int, *, available: bool) -> int:
    if node.allocatable_gpu < 1:
        return 0
    allocatable = _resource_value(node, key, available=False)
    proportional = allocatable * num_gpu // node.allocatable_gpu
    if available:
        return min(_resource_value(node, key, available=True), proportional)
    return proportional


def _node_fits_resources(
    node: PoolNodeResources, num_gpu: int, num_cpu: int, memory: int, storage: int, *, available: bool
) -> bool:
    gpu = node.available_gpu if available else node.allocatable_gpu
    return (
        gpu >= num_gpu
        and _proportional_resource_cap(node, "num_cpu", num_gpu, available=available) >= num_cpu
        and _proportional_resource_cap(node, "memory", num_gpu, available=available) >= memory
        and _proportional_resource_cap(node, "storage", num_gpu, available=available) >= storage
    )


def _nodes_that_fit(
    nodes: list[PoolNodeResources], num_gpu: int, num_cpu: int, memory: int, storage: int, *, available: bool
) -> list[PoolNodeResources]:
    return [
        node for node in nodes if _node_fits_resources(node, num_gpu, num_cpu, memory, storage, available=available)
    ]


def apply_auto_resources(p: ParsedArgs, nodes: list[PoolNodeResources] | None = None) -> ResourcePlan:
    """Fill unspecified CPU, memory, and storage from pool resource data."""
    num_gpu = _positive_cluster_int(p, "num_gpu")
    num_node = _positive_cluster_int(p, "num_node")
    defaults = _default_auto_resources(num_gpu)
    plan = ResourcePlan(source="heuristic")

    if nodes is None:
        nodes, warning = _load_pool_node_resources(p.pool, p.cluster["platform"])
        if warning is not None:
            plan.warnings.append(warning)

    if nodes:
        capacity_candidates = [node for node in nodes if node.allocatable_gpu >= num_gpu]
        if len(capacity_candidates) < num_node:
            max_gpu = max((node.allocatable_gpu for node in nodes), default=0)
            print(
                f"Error: pool '{p.pool}' has only {len(capacity_candidates)} node(s) that can provide "
                f"num_gpu={num_gpu}; requested num_node={num_node}. Max GPU per node is {max_gpu}.",
                file=sys.stderr,
            )
            sys.exit(1)

        available_candidates = [node for node in nodes if node.available_gpu >= num_gpu]
        if len(available_candidates) >= num_node:
            basis = available_candidates
            plan.source = "osmo-free"
        else:
            # Size against capacity so the block message below reports the real request.
            basis = capacity_candidates
            plan.source = "osmo-capacity"

        for key in AUTO_RESOURCE_KEYS:
            if key in p.cluster_overrides:
                continue
            values = [
                _proportional_resource_cap(node, key, num_gpu, available=plan.source == "osmo-free") for node in basis
            ]
            fit = _nth_largest(values, num_node)
            value = min(defaults[key], _round_resource(key, fit, num_gpu))
            old = p.cluster[key]
            p.cluster[key] = str(value)
            if p.cluster[key] != old:
                plan.changes[key] = (old, p.cluster[key])

        final_cpu = _cluster_int(p, "num_cpu")
        final_memory = _cluster_int(p, "memory")
        final_storage = _cluster_int(p, "storage")
        capacity_fit = _nodes_that_fit(nodes, num_gpu, final_cpu, final_memory, final_storage, available=False)
        if len(capacity_fit) < num_node:
            print(
                f"Error: requested resources fit only {len(capacity_fit)} node(s) in pool '{p.pool}', "
                f"but num_node={num_node}. Request per node is gpu={num_gpu}, cpu={final_cpu}, "
                f"memory={final_memory}Gi, storage={final_storage}Gi.",
                file=sys.stderr,
            )
            sys.exit(1)
        available_fit = _nodes_that_fit(nodes, num_gpu, final_cpu, final_memory, final_storage, available=True)
        if len(available_fit) < num_node:
            free_gpus = sorted((node.available_gpu for node in nodes), reverse=True)
            max_free = free_gpus[0] if free_gpus else 0
            total_free = sum(free_gpus)
            full_node_gpu = max((node.allocatable_gpu for node in nodes), default=0)
            whole_free = sum(1 for node in nodes if full_node_gpu and node.available_gpu >= full_node_gpu)
            print(f"Error: pool '{p.pool}' is full for num_gpu={num_gpu} num_node={num_node}. Not submitting.", file=sys.stderr)
            if total_free <= 0:
                print("  Free now: 0 GPU free in this pool -- wait, or try another pool.", file=sys.stderr)
            elif max_free < num_gpu:
                print(
                    f"  Free now: best node {max_free}/{full_node_gpu} GPU, {whole_free} whole node(s) free, "
                    f"{total_free} GPU total. Highest that fits now: num_gpu={max_free} num_node=1.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  Free now: {max_free} GPU free on the best node, but cpu/memory/storage did not fit -- "
                    f"lower num_cpu/memory/storage or wait.",
                    file=sys.stderr,
                )
            sys.exit(1)
    else:
        for key, value in defaults.items():
            if key in p.cluster_overrides:
                continue
            old = p.cluster[key]
            p.cluster[key] = str(max(_cluster_int(p, key), value))
            if p.cluster[key] != old:
                plan.changes[key] = (old, p.cluster[key])

    return plan


def report_resource_plan(p: ParsedArgs, plan: ResourcePlan) -> None:
    if plan.changes:
        changes = ", ".join(f"{key}={new}" for key, (_, new) in sorted(plan.changes.items()))
        print(
            f"Auto resources for pool={p.pool} num_node={p.cluster['num_node']} num_gpu={p.cluster['num_gpu']} "
            f"({plan.source}): {changes}",
            file=sys.stderr,
        )
    for warning in plan.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def _set_derived_cluster_defaults(p: ParsedArgs) -> None:
    """Set resource defaults that depend on other cluster resources."""
    num_gpu = _cluster_int(p, "num_gpu")
    if num_gpu > 1 and "num_cpu" not in p.cluster_overrides:
        p.cluster["num_cpu"] = str(max(_cluster_int(p, "num_cpu"), num_gpu * 8))


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
    _set_derived_cluster_defaults(p)

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
    plan = apply_auto_resources(p)
    report_resource_plan(p, plan)
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
    plan = apply_auto_resources(p)
    report_resource_plan(p, plan)
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
