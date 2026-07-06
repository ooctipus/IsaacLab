# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declaration-order task-family execution."""

import logging
from dataclasses import dataclass, fields

import numpy as np
import pytest
import torch

from isaaclab.utils.string import ResolvableString

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import (
    StateCommandCfg,
    execute_task_family,
    make_task_table_rng,
    task_family,
)


@dataclass(frozen=True)
class _Candidates:
    values: torch.Tensor

    @property
    def num_rows(self) -> int:
        return self.values.shape[0]

    @property
    def device(self) -> torch.device:
        return self.values.device


def _family(events: list[str], *, solve: bool = True, criteria: bool = True):
    def generate_first(_cfg, values, rng):
        events.append("generate_first")
        return _Candidates(values.values + torch.rand(values.values.shape, generator=rng.torch))

    def generate_second(_cfg, values, rng):
        events.append("generate_second")
        offset = float(rng.numpy.uniform(1.0, 1.0))
        assert rng.next_warp_seed() == 17
        return _Candidates(values.values + offset)

    def solve_values(_cfg, values):
        events.append("solve")
        return _Candidates(values.values * 2.0)

    def positive(_cfg, values, rows):
        events.append("criterion_positive")
        return values.values[rows] > 0.0

    def below(_cfg, values, rows):
        events.append("criterion_below")
        return values.values[rows] < 3.0

    def select(_cfg, _values, accepted, target_count, _rng):
        events.append("selection")
        indices = torch.arange(4) if accepted is None else torch.nonzero(accepted).squeeze(-1)
        return indices if target_count is None else indices[:target_count]

    table_cfg = StateCommandCfg.TaskTableCfg
    return table_cfg.FamilyCfg(
        name="fixture",
        generate=(
            table_cfg.GenerateTermCfg(class_type=generate_first),
            table_cfg.GenerateTermCfg(class_type=generate_second),
        ),
        solve=table_cfg.SolveCfg(class_type=solve_values) if solve else None,
        criteria=(
            table_cfg.CriterionCfg(class_type=positive),
            table_cfg.CriterionCfg(class_type=below),
        )
        if criteria
        else (),
        selection=table_cfg.SelectionCfg(class_type=select),
    )


def test_task_family_runs_visible_stages_once_in_declaration_order() -> None:
    events: list[str] = []
    execution = execute_task_family(_family(events), _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))

    assert events == [
        "generate_first",
        "generate_second",
        "solve",
        "criterion_positive",
        "criterion_below",
        "selection",
    ]
    assert len(execution.criterion_masks) == 2
    torch.testing.assert_close(execution.accepted_mask, execution.criterion_masks[0] & execution.criterion_masks[1])
    assert execution.selected_indices.shape[0] <= 2


def test_task_family_criteria_receive_only_declaration_order_survivors() -> None:
    """Later criteria receive the accumulated original-row indices and preserve exact selection."""
    seen: list[torch.Tensor] = []
    table_cfg = StateCommandCfg.TaskTableCfg

    def first(_cfg, values, rows):
        seen.append(rows.clone())
        return values.values[rows] != 1

    def second(_cfg, values, rows):
        seen.append(rows.clone())
        return values.values[rows] > 1

    def select(_cfg, _values, accepted, _target_count, _rng):
        return accepted.nonzero(as_tuple=False).squeeze(-1)

    family = table_cfg.FamilyCfg(
        name="incremental",
        generate=(),
        solve=None,
        criteria=(
            table_cfg.CriterionCfg(class_type=first),
            table_cfg.CriterionCfg(class_type=second),
        ),
        selection=table_cfg.SelectionCfg(class_type=select),
    )
    execution = execute_task_family(family, _Candidates(torch.arange(4)), None, make_task_table_rng(17, "cpu"))

    torch.testing.assert_close(seen[0], torch.tensor((0, 1, 2, 3)))
    torch.testing.assert_close(seen[1], torch.tensor((0, 2, 3)))
    torch.testing.assert_close(execution.criterion_masks[0], torch.tensor((True, False, True, True)))
    torch.testing.assert_close(execution.criterion_masks[1], torch.tensor((False, True, True, True)))
    torch.testing.assert_close(execution.accepted_mask, torch.tensor((False, False, True, True)))
    torch.testing.assert_close(execution.selected_indices, torch.tensor((2, 3)))


def test_task_family_skips_later_criterion_work_after_every_row_is_rejected() -> None:
    """Resolved later gates stay neutral and do no domain work after the survivor set empties."""
    calls: list[str] = []
    table_cfg = StateCommandCfg.TaskTableCfg

    def reject_all(_cfg, _candidates, rows):
        calls.append("reject_all")
        return torch.zeros(rows.shape, dtype=torch.bool, device=rows.device)

    def must_not_run(_cfg, _candidates, _rows):
        calls.append("must_not_run")
        pytest.fail("A criterion ran without active rows.")

    def select(_cfg, _candidates, accepted, _target_count, _rng):
        return accepted.nonzero(as_tuple=False).squeeze(-1)

    family = table_cfg.FamilyCfg(
        name="empty",
        criteria=(
            table_cfg.CriterionCfg(class_type=reject_all),
            table_cfg.CriterionCfg(class_type=must_not_run),
        ),
        selection=table_cfg.SelectionCfg(class_type=select),
    )
    execution = execute_task_family(family, _Candidates(torch.arange(3)), None, make_task_table_rng(17, "cpu"))

    assert calls == ["reject_all"]
    torch.testing.assert_close(execution.criterion_masks[0], torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(execution.criterion_masks[1], torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(execution.accepted_mask, torch.zeros(3, dtype=torch.bool))
    assert not execution.selected_indices.numel()


@pytest.mark.parametrize(
    "bad_mask",
    (
        torch.ones(3),
        torch.ones(2, dtype=torch.bool),
        torch.ones((3, 1), dtype=torch.bool),
    ),
)
def test_task_family_rejects_invalid_active_criterion_masks(bad_mask: torch.Tensor) -> None:
    """Each criterion returns exactly one same-device boolean per active original row."""
    table_cfg = StateCommandCfg.TaskTableCfg

    def invalid(_cfg, _candidates, _rows):
        return bad_mask

    family = table_cfg.FamilyCfg(
        name="invalid",
        criteria=(table_cfg.CriterionCfg(class_type=invalid),),
        selection=table_cfg.SelectionCfg(
            class_type=lambda _cfg, _candidates, accepted, _target_count, _rng: accepted.nonzero(
                as_tuple=False
            ).squeeze(-1)
        ),
    )
    with pytest.raises(ValueError, match="one boolean per active original row"):
        execute_task_family(family, _Candidates(torch.arange(3)), None, make_task_table_rng(17, "cpu"))


def test_task_family_supports_no_solve_and_no_criteria() -> None:
    events: list[str] = []
    execution = execute_task_family(
        _family(events, solve=False, criteria=False), _Candidates(torch.zeros(4)), None, make_task_table_rng(17, "cpu")
    )

    assert events == ["generate_first", "generate_second", "selection"]
    assert execution.accepted_mask is None
    torch.testing.assert_close(execution.selected_indices, torch.arange(4))


def test_task_table_rng_is_deterministic_without_consuming_global_state() -> None:
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    left = execute_task_family(_family([]), _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))
    right = execute_task_family(_family([]), _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))

    torch.testing.assert_close(left.candidates.values, right.candidates.values)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]


def test_task_family_preserves_resolvable_string_callable_cache() -> None:
    """Repeated stages invoke the lazy callable object instead of resolving its string each time."""
    value = ResolvableString("builtins:len")

    assert task_family._callable(value, "fixture") is value
    assert value((1, 2, 3)) == 3
    assert value._resolved_callable is len


def test_task_family_rejects_selection_of_failed_candidate() -> None:
    family = _family([])
    family.criteria = (
        StateCommandCfg.TaskTableCfg.CriterionCfg(
            class_type=lambda _cfg, _values, rows: torch.zeros(rows.shape, dtype=torch.bool)
        ),
    )
    family.selection.class_type = lambda _cfg, _values, _accepted, _target, _rng: torch.tensor((0,))
    with pytest.raises(ValueError, match="only accepted"):
        execute_task_family(family, _Candidates(torch.zeros(4)), 1, make_task_table_rng(17, "cpu"))


def test_task_family_propagates_domain_selection_count_errors() -> None:
    family = _family([], criteria=False)

    def reject_underfill(_cfg, _values, _accepted, _target, _rng):
        raise RuntimeError("domain selection quota was not met")

    family.selection.class_type = reject_underfill

    with pytest.raises(RuntimeError, match="domain selection quota"):
        execute_task_family(family, _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))


def test_task_family_rejects_duplicate_selected_indices() -> None:
    family = _family([], criteria=False)
    family.selection.class_type = lambda _cfg, _values, _accepted, _target, _rng: torch.tensor((0, 0))

    with pytest.raises(ValueError, match="distinct"):
        execute_task_family(family, _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))


def test_task_family_inherited_root_info_skips_all_diagnostic_work(monkeypatch, caplog) -> None:
    """Normal construction must not inherit a root INFO diagnostic opt-in."""
    task_family._LOGGER.setLevel(logging.NOTSET)
    caplog.set_level(logging.INFO)

    def fail(*_args, **_kwargs):
        pytest.fail("Module-NOTSET family execution entered diagnostic work.")

    monkeypatch.setattr(task_family, "_log_family_summary", fail)
    execute_task_family(_family([]), _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))


def test_task_family_info_log_reports_named_counts(caplog) -> None:
    """Explicit INFO inspection reports one concise named family breakdown."""
    caplog.set_level(logging.INFO, logger=task_family.__name__)
    execution = execute_task_family(_family([]), _Candidates(torch.zeros(4)), 2, make_task_table_rng(17, "cpu"))
    assert execution.accepted_mask is not None
    positive_failures = int((~execution.criterion_masks[0]).sum())
    below_failures = int((~execution.criterion_masks[1]).sum())
    accepted = int(execution.accepted_mask.sum())

    assert caplog.messages[-1] == (
        f"Task family fixture: generated=4 accepted={accepted} selected={execution.selected_indices.numel()} "
        f"failures=[positive={positive_failures}, below={below_failures}]"
    )


def test_task_family_schema_has_no_rejected_relation_or_objective_set_layers() -> None:
    names = {field.name for field in fields(StateCommandCfg.TaskTableCfg.FamilyCfg)}
    assert names == {"name", "generate", "solve", "criteria", "selection"}
    solve_names = {field.name for field in fields(StateCommandCfg.TaskTableCfg.SolveCfg)}
    assert solve_names == {
        "class_type",
        "objectives",
        "max_iterations",
        "convergence_tolerance",
        "convergence_check_interval",
    }
