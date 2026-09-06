# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declaration-order task-family execution for pure task-table builders."""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.utils.string import string_to_callable

if TYPE_CHECKING:
    from .state_command_cfg import StateCommandCfg


_LOGGER = logging.getLogger(__name__)

_STAGE_DETAILS: contextvars.ContextVar[dict[str, object] | None] = contextvars.ContextVar(
    "task_family_stage_details", default=None
)


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
class TaskFamilyStageReport:
    """Row flow, wall time, and domain details of one executed family stage."""

    kind: str
    """Stage kind: ``generate``, ``solve``, ``criterion``, or ``selection``."""
    name: str
    """Configured stage name or the resolved stage callable name."""
    rows_in: int
    """Candidate rows entering the stage (active rows for criteria)."""
    rows_out: int
    """Candidate rows leaving the stage (survivors for criteria, picks for selection)."""
    seconds: float
    """Wall time including device synchronization [s]."""
    details: dict[str, object] = field(default_factory=dict)
    """Domain facts recorded through :func:`record_stage_details` while the stage ran."""

    def format(self, name_width: int = 0) -> str:
        """Return one aligned text line for this stage."""
        text = (
            f"{self.kind:<9} {self.name:<{name_width}} rows {self.rows_in:>9} -> {self.rows_out:<9} "
            f"{self.seconds:9.3f} s"
        )
        if self.details:
            text += "  " + " ".join(f"{key}={value}" for key, value in self.details.items())
        return text


@dataclass(frozen=True, slots=True)
class TaskFamilyReport:
    """Per-stage construction evidence assembled by :func:`execute_task_family`."""

    name: str
    """Family name."""
    stages: tuple[TaskFamilyStageReport, ...]
    """Executed stages in declaration order."""
    generated: int
    """Rows entering the criterion cascade."""
    accepted: int
    """Rows accepted by every criterion."""
    selected: int | None
    """Rows picked by selection, or ``None`` when the family has no selection stage."""

    @property
    def seconds(self) -> float:
        """Total wall time of all stages [s]."""
        return sum(stage.seconds for stage in self.stages)

    def format(self) -> str:
        """Return a multi-line summary with one aligned line per stage."""
        selected = "n/a" if self.selected is None else str(self.selected)
        lines = [
            f"Task family {self.name}: generated={self.generated} accepted={self.accepted} selected={selected} "
            f"seconds={self.seconds:.3f}"
        ]
        name_width = max((len(stage.name) for stage in self.stages), default=0)
        lines.extend("  " + stage.format(name_width) for stage in self.stages)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TaskFamilyExecution:
    """Candidate data and cached acceptance plus optional selection from one family."""

    candidates: Any
    criterion_masks: tuple[torch.Tensor, ...]
    accepted_mask: torch.Tensor | None
    selected_indices: torch.Tensor | None
    report: TaskFamilyReport
    """Stage-by-stage row counts, timings, and domain details."""


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


def record_stage_details(**details: object) -> None:
    """Attach domain facts to the task-family stage that is currently executing.

    Stage callables report facts the shared runner cannot observe, such as IK
    batch counts or sampler rejection tallies. The facts appear in the stage's
    :class:`TaskFamilyStageReport`. Calls outside a running stage are ignored
    so stage functions remain usable on their own.
    """
    current = _STAGE_DETAILS.get()
    if current is not None:
        current.update(details)


def execute_task_family(
    family: StateCommandCfg.TaskTableCfg.FamilyCfg,
    initial: Any,
    target_count: int | None,
    rng: TaskTableRng,
) -> TaskFamilyExecution:
    """Run generate, optional solve, criteria, and selection exactly once.

    Stage callables receive their own config first. Generate callables then
    receive ``(candidates, rng)``, the solver receives ``candidates``, criteria
    receive ``(candidates, active_rows)``, and selection receives
    ``(candidates, accepted_mask, target_count, rng)``. Candidate storage stays
    domain-owned; this function owns only the visible stage order. Stored gate
    masks are neutral for rows rejected earlier, so diagnostics attribute each
    rejected row to its first failing declared criterion. Every stage is timed
    and its row flow recorded in the returned :class:`TaskFamilyReport`.

    Args:
        family: Family stage configuration.
        initial: Domain-owned global data or initial candidates.
        target_count: Domain-owned numeric selection request, or ``None`` for no numeric request.
        rng: Table-owned random state.

    Returns:
        Cached stage outputs, selected candidate indices, and the stage report.
    """
    stages: list[TaskFamilyStageReport] = []
    candidates = initial
    for generate in family.generate:
        candidates = _run_stage(
            stages,
            "generate",
            _stage_name(generate),
            _row_count(candidates),
            lambda generate=generate: _callable(generate.class_type, "generate")(generate, candidates, rng),
            _row_count,
        )
    if family.solve is not None:
        candidates = _run_stage(
            stages,
            "solve",
            _stage_name(family.solve),
            _row_count(candidates),
            lambda: _callable(family.solve.class_type, "solve")(family.solve, candidates),
            _row_count,
        )

    masks_list: list[torch.Tensor] = []
    accepted = None
    if family.criteria:
        active_rows = torch.arange(candidates.num_rows, dtype=torch.int64, device=candidates.device)
        for criterion in family.criteria:
            criterion_fn = _callable(criterion.class_type, "criterion")
            gate_mask = torch.ones(candidates.num_rows, dtype=torch.bool, device=candidates.device)
            rows_in = active_rows.numel()

            def run_criterion(criterion=criterion, criterion_fn=criterion_fn, gate_mask=gate_mask):
                nonlocal active_rows
                if active_rows.numel():
                    local_mask = criterion_fn(criterion, candidates, active_rows)
                    _validate_criterion_mask(local_mask, active_rows)
                    gate_mask[active_rows] = local_mask
                    active_rows = active_rows[local_mask]
                return active_rows

            _run_stage(
                stages,
                "criterion",
                _stage_name(criterion),
                rows_in,
                run_criterion,
                lambda rows: rows.numel(),
                candidates,
                details=lambda rows, rows_in=rows_in: {"rejected": rows_in - rows.numel()},
            )
            masks_list.append(gate_mask)
        accepted = torch.zeros(candidates.num_rows, dtype=torch.bool, device=candidates.device)
        accepted[active_rows] = True
    masks = tuple(masks_list)
    selected = None
    if family.selection is None:
        if target_count is not None:
            raise ValueError("Task families without selection do not accept a numeric target count.")
    else:
        selection = family.selection
        selected = _run_stage(
            stages,
            "selection",
            _stage_name(selection),
            _row_count(candidates) if accepted is None else int(accepted.sum()),
            lambda: _callable(selection.class_type, "selection")(selection, candidates, accepted, target_count, rng),
            lambda picks: picks.numel(),
            candidates,
        )
        _validate_selection(selected, accepted)
    generated = _row_count(candidates)
    report = TaskFamilyReport(
        name=family.name,
        stages=tuple(stages),
        generated=generated,
        accepted=generated if accepted is None else int(accepted.sum()),
        selected=None if selected is None else int(selected.numel()),
    )
    if _LOGGER.level != logging.NOTSET and _LOGGER.isEnabledFor(logging.INFO):
        _log_family_report(report)
    return TaskFamilyExecution(candidates, masks, accepted, selected, report)


def _run_stage(
    stages: list[TaskFamilyStageReport],
    kind: str,
    name: str,
    rows_in: int,
    call: Callable[[], Any],
    rows_out: Callable[[Any], int],
    sync_target: Any = None,
    *,
    details: Callable[[Any], dict[str, object]] | None = None,
) -> Any:
    """Run one stage under a synchronized wall clock and append its report."""
    recorded: dict[str, object] = {}
    _synchronize(sync_target)
    started = time.perf_counter()
    token = _STAGE_DETAILS.set(recorded)
    try:
        result = call()
    finally:
        _STAGE_DETAILS.reset(token)
    _synchronize(result if sync_target is None else sync_target)
    seconds = time.perf_counter() - started
    if details is not None:
        recorded = {**details(result), **recorded}
    stages.append(TaskFamilyStageReport(kind, name, int(rows_in), int(rows_out(result)), seconds, recorded))
    return result


def _synchronize(value: Any) -> None:
    """Wait for outstanding CUDA work on the device carrying ``value``, if any."""
    device = getattr(value, "device", None)
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _row_count(value: Any) -> int:
    """Return ``num_rows`` of a candidate container, or zero for table-global inputs."""
    rows = getattr(value, "num_rows", None)
    return 0 if rows is None else int(rows)


def _callable(value: object, stage: str):
    if callable(value):
        return value
    if isinstance(value, str):
        return string_to_callable(value)
    raise TypeError(f"Task-family {stage} class_type must resolve to a callable.")


def _validate_criterion_mask(mask: torch.Tensor, active_rows: torch.Tensor) -> None:
    if mask.dtype is not torch.bool or mask.shape != active_rows.shape or mask.device != active_rows.device:
        raise ValueError("Task-family criteria must return one boolean per active original row on the same device.")


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


def _log_family_report(report: TaskFamilyReport) -> None:
    """Log one explicitly requested construction report for a task family."""
    _LOGGER.info("%s", report.format())


def _stage_name(stage_cfg: object) -> str:
    """Return one concise configured stage label without resolving imports."""
    for attribute in ("name", "objective"):
        value = getattr(stage_cfg, attribute, None)
        if isinstance(value, str) and value:
            return value
    class_type = getattr(stage_cfg, "class_type", None)
    if isinstance(class_type, str):
        return class_type.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    name = getattr(class_type, "__name__", None)
    return name if isinstance(name, str) and name else type(stage_cfg).__name__
