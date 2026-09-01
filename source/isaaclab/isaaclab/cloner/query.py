# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queries over the prototype/clone relation a :class:`~isaaclab.cloner.ClonePlan` describes.

Each plan row pairs a prototype path that exists once on the stage with a destination
template and the environments it populates, so the relation is partial in both directions: a
prototype reaches only the environments its row covers, and an environment holds only the
assets whose rows cover it. :func:`path_to_source` resolves one prototype for a clone-side
path, while :func:`iter_sources` yields every populated variant behind that path.

Environment ids are not mask columns. Column ``j`` stands for
:attr:`~isaaclab.cloner.ClonePlan.env_ids`\\ ``[j]``, which is the number
:class:`~isaaclab.cloner.ReplicateSession` formats into the template. These functions take
and return environment ids throughout.

Rows sharing one source or destination root are variants of the same asset, and the environment
picks between them. ``test/cloner/test_clone_plan_algebra.py`` pins that down.

The path primitives are aliased ``pth`` because ``path`` is a parameter name here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import torch

from . import path as pth

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _is_shared_destination(destination: str) -> bool:
    """Whether a destination names one prim shared by every environment."""
    return "{}" not in destination


def _populated_clone_rows(plan: ClonePlan) -> list[int]:
    """Return populated rows that copy a prototype into environments, in row order."""
    return [
        row
        for row, destination in enumerate(plan.destinations)
        if not _is_shared_destination(destination) and bool(plan.clone_mask[row].any())
    ]


def _clone_mapping(
    plan: ClonePlan, rows: Sequence[int], *, whole_env: bool
) -> tuple[tuple[str, ...], tuple[str, ...], torch.Tensor]:
    """Project routed plan rows into the mapping consumed by one clone context."""
    rows = tuple(rows)
    mapping = plan.clone_mask[list(rows)]
    populated_rows = tuple(_populated_clone_rows(plan))
    if whole_env and rows == populated_rows and rows and bool(mapping.all()):
        source_env = plan.env_template.format(int(plan.env_ids[0]))
        if all(
            pth.under(plan.sources[row], source_env)
            and pth.under(plan.destinations[row].format(int(plan.env_ids[0])), source_env)
            for row in rows
        ):
            return (source_env,), (plan.env_template,), mapping[:1]
    return (
        tuple(plan.sources[row] for row in rows),
        tuple(plan.destinations[row] for row in rows),
        mapping,
    )


def _row_populated(plan: ClonePlan, row: int) -> bool:
    """Whether a row names a global asset or reaches at least one environment."""
    return _is_shared_destination(plan.destinations[row]) or bool(plan.clone_mask[row].any())


def _cfg_source_paths(plan: ClonePlan, cfg: object) -> tuple[str | None, ...]:
    """Return the exact plan source paths covering a cfg.

    The result follows the covering row order. An inactive variant retains its slot as ``None``;
    a cfg nested below another asset gets the same child suffix below every parent prototype.

    Args:
        plan: Active clone plan.
        cfg: The same configuration instance used to build ``plan``.

    Returns:
        Exact prototype paths, with ``None`` for inactive rows.

    Raises:
        ValueError: If ``cfg`` owns no row or its path does not match that row in ``plan``.
    """
    rows = plan.cfg_rows.get(id(cfg))
    if rows is None:
        raise ValueError(f"Cfg {type(cfg).__name__} owns no row in the active clone plan.")
    path_expr = cfg.prim_path
    paths: list[str | None] = []
    for row in rows:
        if not _row_populated(plan, row):
            paths.append(None)
            continue
        source, destination = plan.sources[row], plan.destinations[row]
        resolved_expr = path_expr
        if "{ENV_REGEX_NS}" in resolved_expr and not _is_shared_destination(destination):
            prefix, _ = pth.split(destination)
            resolved_expr = resolved_expr.replace("{ENV_REGEX_NS}", f"{prefix}[^/]+")
        if _is_shared_destination(destination):
            suffix = pth.relative_to(resolved_expr, destination)
        else:
            matched = pth.match(resolved_expr, destination)
            suffix = None if matched is None else matched.suffix
        if suffix is None:
            raise ValueError(f"Cfg path {path_expr!r} does not match clone-plan destination {destination!r}.")
        paths.append((source.rstrip("/") + suffix) or "/")
    return tuple(paths)


def _row_env_ids(plan: ClonePlan, row: int) -> tuple[int, ...]:
    """Env ids populated by a row, or every env for a shared global row."""
    if _is_shared_destination(plan.destinations[row]):
        return tuple(plan.env_ids.tolist())
    columns = plan.clone_mask[row].nonzero(as_tuple=False).flatten()
    columns = columns.to(plan.env_ids.device)
    return tuple(plan.env_ids[columns].tolist())


def _column_for_env_id(plan: ClonePlan, env_id: int) -> int | None:
    """Mask column standing for ``env_id``, or ``None`` when the plan does not target it.

    Guards against out-of-range and negative ids, which plain indexing would raise on or
    silently wrap around.
    """
    columns = (plan.env_ids == env_id).nonzero(as_tuple=False).flatten().tolist()
    return int(columns[0]) if columns else None


def _clone_rows(plan: ClonePlan, path_expr: str, *, populated_only: bool) -> list[tuple[str, pth.TemplateMatch, int]]:
    """Collect ``(template, match, row)`` for destination templates owning ``path_expr``."""
    candidates: list[tuple[str, pth.TemplateMatch, int]] = []
    for row, template in enumerate(plan.destinations):
        if populated_only and not _row_populated(plan, row):
            continue
        if _is_shared_destination(template):
            suffix = pth.relative_to(path_expr, template)
            matched = None if suffix is None else pth.TemplateMatch("", suffix)
        else:
            matched = pth.match(path_expr, template)
        if matched is None:
            continue
        candidates.append((template, matched, row))
    return candidates


def _owning_template(plan: ClonePlan, path_expr: str) -> tuple[str, list[int], pth.TemplateMatch] | None:
    """Resolve the single destination template owning ``path_expr``.

    Returns:
        ``(template, rows, match)`` where ``rows`` are all rows sharing the winning template,
        in row order, or ``None`` when no template owns ``path_expr``.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct templates.
    """
    candidates = _clone_rows(plan, path_expr, populated_only=False)
    if not candidates:
        return None
    owning_templates = {template for template, _, _ in candidates}
    if len(owning_templates) > 1:
        raise ValueError(f"path_expr {path_expr!r}: matches multiple destination templates {sorted(owning_templates)}.")
    template, matched, _ = candidates[0]
    return template, [row for _, _, row in candidates], matched


def path_to_source(plan: ClonePlan, path_expr: str, env_id: int | None = None) -> tuple[str, str, str] | None:
    """Resolve a clone-side expression to the prototype it was cloned from.

    A *concrete* clone path names its environment in the template's clone slot, and that
    environment selects which variant to report. A *wildcard* expression
    (``.../env_[^/]+/...``) names no environment and stands for all of them, so it resolves to
    the first populated variant unless ``env_id`` says which one to take.

    Args:
        plan: Active clone plan.
        path_expr: Clone-side path expression (e.g. a sensor's ``prim_path``, with a segment
            wildcard in the env slot) or a concrete clone path.
        env_id: Environment whose variant to resolve. Defaults to the one ``path_expr`` names
            when it is concrete, and to no particular environment otherwise.

    Returns:
        A ``(source_path, destination_expr, asset_suffix)`` tuple, where ``destination_expr``
        spells the clone slot ``[^/]+`` so it reads as a path expression like every other one,
        and ``asset_suffix`` is the part of ``path_expr`` below the owning template. ``None``
        when ``path_expr`` matches no row or no matching row populates the requested
        environment.

        Partial-env coverage is supported: when the matching rows cover only a subset of envs
        (an asset present in some envs but not others, as in heterogeneous scenes), the
        returned expression resolves to just those envs.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally near templates.
    """
    owner = _owning_template(plan, path_expr)
    if owner is None:
        return None
    template, rows, matched = owner
    if env_id is None and matched.instance.isdigit():
        env_id = int(matched.instance)
    # Resolution must walk a prototype that exists on stage, so rows populating no env at all
    # are skipped rather than reported.
    if env_id is None:
        rows = [row for row in rows if _row_populated(plan, row)]
    else:
        column = _column_for_env_id(plan, env_id)
        if column is None:
            return None
        rows = [
            row for row in rows if _is_shared_destination(plan.destinations[row]) or bool(plan.clone_mask[row][column])
        ]
    if not rows:
        return None
    return plan.sources[rows[0]], template.format("[^/]+"), matched.suffix


def iter_sources(plan: ClonePlan, path_expr: str) -> Iterator[tuple[str, str, str, tuple[int, ...]]]:
    """Yield every populated plan row whose destination owns a path expression.

    Where :func:`path_to_source` names one variant, this yields them all, for callers that
    must visit each prototype behind a destination template (loading one mesh per variant).

    Example:
        For a row with prototype root ``"/World/source/Robot"``, destination template
        ``"/World/scenes/{}/Robot"`` and env ids ``(0, 2)``, querying
        ``"/World/scenes/[^/]+/Robot/base"`` yields ``("/World/source/Robot",
        "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 2))``.

    Args:
        plan: Clone plan to query.
        path_expr: Clone-side prim path or path expression.

    Yields:
        ``(source_root, destination_template, source_path, env_ids)`` per matching row, in row
        order. Rows populating no env are skipped.
    """
    for template, matched, row in _clone_rows(plan, path_expr, populated_only=True):
        template_norm = template.rstrip("/") or "/"
        source_root = plan.sources[row].rstrip("/") or "/"
        source_path = source_root + matched.suffix if source_root != "/" else matched.suffix or "/"
        yield source_root, template_norm, source_path, _row_env_ids(plan, row)
