# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.cloner.clone_plan import ClonePlan


def get_active_clone_plan() -> ClonePlan | None:
    """Return the simulation clone plan, if one is active."""
    sim = sim_utils.SimulationContext.instance()
    return sim.get_clone_plan() if sim is not None else None


def expand_clone_plan_path(path_expr: str, clone_plan: ClonePlan | None = None) -> list[str] | None:
    """Expand an env-scoped path expression to exact clone paths.

    Args:
        path_expr: Path expression containing the clone-plan environment token.
        clone_plan: Clone plan to use. Defaults to the active simulation clone plan.

    Returns:
        Exact cloned paths in environment-major order, or ``None`` if the expression
        does not refer to the clone plan.
    """
    return expand_clone_plan_paths((path_expr,), clone_plan=clone_plan)


def expand_clone_plan_paths(path_exprs: Sequence[str], clone_plan: ClonePlan | None = None) -> list[str] | None:
    """Expand env-scoped path expressions to exact clone paths.

    For multiple expressions, paths are emitted in environment-major order so
    tensor views keep the same flattened ``[env, body]`` layout as wildcard
    discovery.

    Args:
        path_exprs: Path expressions containing the clone-plan environment token.
        clone_plan: Clone plan to use. Defaults to the active simulation clone plan.

    Returns:
        Exact cloned paths, or ``None`` if any expression is outside the clone plan.
    """
    clone_plan = get_active_clone_plan() if clone_plan is None else clone_plan
    if clone_plan is None:
        return None

    matches_by_expr: list[list[tuple[int, str]]] = []
    for path_expr in path_exprs:
        matches: list[tuple[int, str]] = []
        for row, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations)):
            suffix = _clone_suffix_from_path_expr(path_expr, source, destination)
            if suffix is not None:
                matches.append((row, suffix))
        if not matches:
            return None
        matches_by_expr.append(matches)

    expanded_paths: list[str] = []
    num_envs = int(clone_plan.clone_mask.shape[1])
    clone_mask = clone_plan.clone_mask.detach().cpu()
    for env_id in range(num_envs):
        for matches in matches_by_expr:
            for row, suffix in matches:
                if bool(clone_mask[row, env_id]):
                    expanded_paths.append(clone_plan.destinations[row].format(env_id) + suffix)
                    break
    return expanded_paths


def expand_clone_plan_child_paths(
    predicate: Callable[[Usd.Prim], bool],
    clone_plan: ClonePlan | None = None,
    stage: Usd.Stage | None = None,
) -> list[str] | None:
    """Expand clone-plan source child prims selected by a predicate to exact clone paths.

    Args:
        predicate: Predicate selecting authored source prims under each clone-plan source root.
        clone_plan: Clone plan to use. Defaults to the active simulation clone plan.
        stage: USD stage containing the authored clone-plan source prims.

    Returns:
        Exact cloned paths in environment-major order, or ``None`` if no clone plan is active.
    """
    clone_plan = get_active_clone_plan() if clone_plan is None else clone_plan
    if clone_plan is None:
        return None

    source_suffixes_by_row: list[list[str]] = []
    clone_mask = clone_plan.clone_mask.detach().cpu()
    for row, source in enumerate(clone_plan.sources):
        if not bool(clone_mask[row].any()):
            source_suffixes_by_row.append([])
            continue
        source_prims = sim_utils.get_all_matching_child_prims(source, predicate=predicate, stage=stage)
        suffixes = []
        for prim in source_prims:
            prim_path = prim.GetPath().pathString
            suffixes.append("" if prim_path == source else prim_path[len(source) :])
        source_suffixes_by_row.append(suffixes)

    expanded_paths: list[str] = []
    num_envs = int(clone_plan.clone_mask.shape[1])
    for env_id in range(num_envs):
        for row, suffixes in enumerate(source_suffixes_by_row):
            if bool(clone_mask[row, env_id]):
                destination = clone_plan.destinations[row].format(env_id)
                expanded_paths.extend(destination + suffix for suffix in suffixes)
    return expanded_paths


def resolve_clone_plan_source_paths(path_expr: str, clone_plan: ClonePlan | None = None) -> list[str] | None:
    """Resolve an env-scoped path expression to representative source paths.

    Args:
        path_expr: Path expression containing the clone-plan environment token.
        clone_plan: Clone plan to use. Defaults to the active simulation clone plan.

    Returns:
        Exact source paths for active clone rows, or ``None`` if the expression is outside
        the clone plan.
    """
    clone_plan = get_active_clone_plan() if clone_plan is None else clone_plan
    if clone_plan is None:
        return None

    clone_mask = clone_plan.clone_mask.detach().cpu()
    source_paths = []
    seen: set[str] = set()
    for row, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations)):
        if not bool(clone_mask[row].any()):
            continue
        suffix = _clone_suffix_from_path_expr(path_expr, source, destination)
        if suffix is not None:
            source_path = source + suffix
            if source_path not in seen:
                source_paths.append(source_path)
                seen.add(source_path)
    return source_paths or None


def resolve_clone_plan_source_path(path_expr: str, clone_plan: ClonePlan | None = None) -> str | None:
    """Resolve an env-scoped path expression to the representative source path.

    Args:
        path_expr: Path expression containing the clone-plan environment token.
        clone_plan: Clone plan to use. Defaults to the active simulation clone plan.

    Returns:
        Exact source path, or ``None`` if the expression is outside the clone plan.
    """
    source_paths = resolve_clone_plan_source_paths(path_expr, clone_plan)
    return source_paths[0] if source_paths else None


def _clone_suffix_from_path_expr(path_expr: str, source: str, destination: str) -> str | None:
    """Return the suffix under a clone-plan row matched by ``path_expr``."""
    if path_expr == source:
        return ""
    if path_expr.startswith(source + "/"):
        return path_expr[len(source) :]

    prefix, separator, postfix = destination.partition("{}")
    if not separator:
        if path_expr == destination:
            return ""
        if path_expr.startswith(destination + "/"):
            return path_expr[len(destination) :]
        return None

    if not path_expr.startswith(prefix):
        return None
    remainder = path_expr[len(prefix) :]

    if postfix:
        postfix_index = remainder.find(postfix)
        if postfix_index <= 0:
            return None
        env_token = remainder[:postfix_index]
        suffix = remainder[postfix_index + len(postfix) :]
        if suffix and not suffix.startswith("/"):
            return None
    else:
        slash_index = remainder.find("/")
        if slash_index < 0:
            env_token = remainder
            suffix = ""
        else:
            env_token = remainder[:slash_index]
            suffix = remainder[slash_index:]

    if not env_token or "/" in env_token:
        return None
    return suffix
