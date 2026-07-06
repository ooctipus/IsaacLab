# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declaration-order task-family execution for pure task-table builders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.utils.string import string_to_callable

if TYPE_CHECKING:
    from .state_command_cfg import StateCommandCfg


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskTableRng:
    """Independent NumPy, Torch, and Warp seed state owned by one table build."""

    seed: int
    numpy: np.random.Generator
    torch: torch.Generator
    _warp_offset: int = field(default=0, init=False, repr=False)

    def next_warp_seed(self) -> int:
        """Return a deterministic stage seed for a Warp kernel."""
        seed = (self.seed + self._warp_offset) & 0x7FFFFFFF
        self._warp_offset += 1
        return seed


@dataclass(frozen=True, slots=True)
class TaskFamilyExecution:
    """Candidate data and cached accept/select results from one family."""

    candidates: Any
    criterion_masks: tuple[torch.Tensor, ...]
    accepted_mask: torch.Tensor | None
    selected_indices: torch.Tensor


def make_task_table_rng(seed: int, device: str | torch.device) -> TaskTableRng:
    """Create independent random state for one deterministic table build.

    Args:
        seed: Non-negative table seed.
        device: Device used by Torch sampling.

    Returns:
        Independent random state for NumPy, Torch, and Warp stages.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Task-table seed must be a non-negative integer.")
    torch_rng = torch.Generator(device=torch.device(device))
    torch_rng.manual_seed(seed)
    return TaskTableRng(seed=seed, numpy=np.random.default_rng(seed), torch=torch_rng)


def execute_task_family(
    family: StateCommandCfg.TaskTableCfg.FamilyCfg,
    initial: Any,
    target_count: int | None,
    rng: TaskTableRng,
) -> TaskFamilyExecution:
    """Run generate, optional solve, criteria, and selection exactly once.

    Stage callables receive their own config first. Generate callables then
    receive ``(candidates, rng)``, the solver receives ``candidates``, criteria
    receive ``candidates``, and selection receives
    ``(candidates, accepted_mask, target_count, rng)``. Candidate storage stays
    domain-owned; this function owns only the visible stage order.

    Args:
        family: Family stage configuration.
        initial: Domain-owned global data or initial candidates.
        target_count: Domain-owned numeric selection request, or ``None`` for no numeric request.
        rng: Table-owned random state.

    Returns:
        Cached stage outputs and selected candidate indices.
    """
    candidates = initial
    for generate in family.generate:
        candidates = _callable(generate.class_type, "generate")(generate, candidates, rng)
    if family.solve is not None:
        candidates = _callable(family.solve.class_type, "solve")(family.solve, candidates)

    masks = tuple(_callable(criterion.class_type, "criterion")(criterion, candidates) for criterion in family.criteria)
    accepted = _intersect_criteria(masks)
    selected = _callable(family.selection.class_type, "selection")(
        family.selection,
        candidates,
        accepted,
        target_count,
        rng,
    )
    _validate_selection(selected, accepted)
    if _LOGGER.level != logging.NOTSET and _LOGGER.isEnabledFor(logging.INFO):
        _log_family_summary(family, candidates, masks, accepted, selected)
    return TaskFamilyExecution(candidates, masks, accepted, selected)


def _callable(value: object, stage: str):
    if callable(value):
        return value
    if isinstance(value, str):
        return string_to_callable(value)
    raise TypeError(f"Task-family {stage} class_type must resolve to a callable.")


def _intersect_criteria(masks: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    if not masks:
        return None
    first = masks[0]
    if first.dtype is not torch.bool or first.ndim != 1:
        raise ValueError("Task-family criteria must return one-dimensional boolean tensors.")
    accepted = first.clone()
    for mask in masks[1:]:
        if mask.dtype is not torch.bool or mask.shape != first.shape or mask.device != first.device:
            raise ValueError("Task-family criterion masks must share shape, dtype, and device.")
        accepted.logical_and_(mask)
    return accepted


def _validate_selection(selected: torch.Tensor, accepted: torch.Tensor | None) -> None:
    if selected.dtype is not torch.int64 or selected.ndim != 1:
        raise ValueError("Task-family selection must return a one-dimensional int64 tensor.")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("Task-family selection indices must be distinct.")
    if accepted is None:
        return
    if selected.device != accepted.device:
        raise ValueError("Task-family selection and criterion masks must share a device.")
    if selected.numel():
        if bool(torch.any((selected < 0) | (selected >= accepted.shape[0]))):
            raise ValueError("Task-family selection contains an out-of-range candidate index.")
        if not bool(torch.all(accepted[selected])):
            raise ValueError("Task-family selection may contain only accepted candidates.")


def _log_family_summary(
    family: StateCommandCfg.TaskTableCfg.FamilyCfg,
    candidates: Any,
    masks: tuple[torch.Tensor, ...],
    accepted: torch.Tensor | None,
    selected: torch.Tensor,
) -> None:
    """Log one explicitly requested construction summary for a task family."""
    if accepted is None:
        generated_count = _candidate_count(candidates, selected)
        accepted_count = generated_count
        failure_counts: tuple[int, ...] = ()
    else:
        generated_count = accepted.numel()
        counts = torch.stack((accepted.sum(), *((~mask).sum() for mask in masks))).detach().cpu().tolist()
        accepted_count = int(counts[0])
        failure_counts = tuple(int(value) for value in counts[1:])
    failures = ", ".join(
        f"{_criterion_name(criterion)}={count}"
        for criterion, count in zip(family.criteria, failure_counts, strict=True)
    )
    _LOGGER.info(
        "Task family %s: generated=%d accepted=%d selected=%d failures=[%s]",
        family.name,
        generated_count,
        accepted_count,
        selected.numel(),
        failures,
    )


def _candidate_count(candidates: Any, selected: torch.Tensor) -> int:
    """Return a readable generated count for a criterion-free inspected family."""
    if isinstance(candidates, torch.Tensor):
        return candidates.shape[0]
    try:
        return len(candidates)
    except TypeError:
        return selected.numel()


def _criterion_name(criterion: StateCommandCfg.TaskTableCfg.CriterionCfg) -> str:
    """Return one concise configured criterion label without resolving imports."""
    for attribute in ("name", "objective"):
        value = getattr(criterion, attribute, None)
        if isinstance(value, str) and value:
            return value
    class_type = criterion.class_type
    if isinstance(class_type, str):
        return class_type.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    name = getattr(class_type, "__name__", None)
    return name if isinstance(name, str) and name else type(criterion).__name__
