# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preparation-only path queries; runtime selections contain numeric relations only."""

from __future__ import annotations

import re
from dataclasses import MISSING, fields, is_dataclass
from typing import Literal

import numpy as np
from newton import Model

from isaaclab.utils import configclass

from .newton_selection import BODY, JOINT_COORD, JOINT_DOF, NewtonSelection, NewtonSelections


@configclass
class NewtonSelectorCfg:
    """Match full model labels, preserving pattern order and model order within each world.

    Overlapping patterns select an entity only once. Global entities (world -1)
    are excluded. ``count_per_world`` validates static topology, before masking.
    Joint patterns expand into all coordinates or DOFs of each matched joint.
    """

    frequency: Literal["body", "joint_coord", "joint_dof"] = MISSING
    path: str | tuple[str, ...] | list[str] = MISSING
    count_per_world: int | None = None
    dense_width: int | None = None
    """Explicit policy width for a group of differently sized native selections."""


def bind_selectors(value, resolve):
    """Resolve declarative task selections recursively before constructing manager terms."""
    if isinstance(value, NewtonSelectorCfg):
        return resolve(value)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = bind_selectors(item, resolve)
    elif isinstance(value, (tuple, list)):
        return type(value)(bind_selectors(item, resolve) for item in value)
    elif is_dataclass(value):
        for field in fields(value):
            setattr(value, field.name, bind_selectors(getattr(value, field.name), resolve))
    return value


def selector_key(cfg: NewtonSelectorCfg) -> tuple:
    """Return the immutable preparation cache key for a symbolic query."""
    patterns = (cfg.path,) if isinstance(cfg.path, str) else tuple(cfg.path)
    return cfg.frequency, patterns, cfg.count_per_world, cfg.dense_width


def query_selection_ids(model: Model, cfg: NewtonSelectorCfg) -> np.ndarray:
    """Resolve ordered patterns to unique frequency IDs, excluding global entities.

    Occurrences in different bodies or joints remain distinct even when labels
    match. Repeated pattern matches of the same entity are selected once.
    """
    if cfg.frequency not in (BODY, JOINT_COORD, JOINT_DOF):
        raise ValueError(f"Unknown Newton frequency: {cfg.frequency!r}")
    labels = model.body_label if cfg.frequency == BODY else model.joint_label
    worlds = (model.body_world if cfg.frequency == BODY else model.joint_world).numpy()
    starts = (
        None
        if cfg.frequency == BODY
        else (model.joint_q_start if cfg.frequency == JOINT_COORD else model.joint_qd_start).numpy()
    )
    rows = [[] for _ in range(model.world_count)]
    seen = set()
    for pattern in selector_key(cfg)[1]:
        regex = re.compile(pattern)
        matched = False
        for entity, (label, world) in enumerate(zip(labels, worlds, strict=True)):
            if world < 0 or not regex.fullmatch(label):
                continue
            matched = True
            if entity not in seen:
                seen.add(entity)
                rows[world].extend([entity] if starts is None else range(int(starts[entity]), int(starts[entity + 1])))
        if not matched:
            raise ValueError(f"Selector {pattern!r} matched no {cfg.frequency} entities.")
    if cfg.count_per_world is not None and any(len(row) != cfg.count_per_world for row in rows):
        raise ValueError(
            f"Expected {cfg.count_per_world} {cfg.frequency} entries per world; got {list(map(len, rows))}."
        )
    return np.asarray([index for row in rows for index in row], dtype=np.int32)


def resolve_selection(owner: NewtonSelections, cfg: NewtonSelectorCfg) -> NewtonSelection:
    """Bind a symbolic query through the numeric Newton selection API."""
    model = owner.model if owner.source is None else owner.source.model
    ids = query_selection_ids(model, cfg)
    if owner.source is not None:
        stride = {BODY: model.body_count, JOINT_COORD: model.joint_coord_count, JOINT_DOF: model.joint_dof_count}[
            cfg.frequency
        ]
        ids = (ids[None, :] + np.arange(owner.model.world_count, dtype=np.int32)[:, None] * stride).reshape(-1)
    return owner.bind(cfg.frequency, ids)
