# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward shape archetypes for Meta-World+ tasks — task-agnostic primitives.

Four archetypes cover *every* MW task. Adding a new task = picking the
archetype that fits and writing a ``RewardTermCfg`` with task-specific
parameters. **No task-named functions in this module** — anything called
``drawer_open_v2`` or similar belongs in the env-cfg, expressed as one
of these archetypes.

#. :func:`tolerance_shape` — ``scale * tolerance(distance)`` with a margin.
   Reach, button-press, dial-turn, peg-insert, drawer-open's two terms.
#. :func:`hamacher_shape` — ``scale * H(tol_a, tol_b)`` with optional
   gripper modulator on ``tol_a``, optional phase bonus, optional
   success override. Window-open/close, door-open, faucet-open,
   drawer-close, button-press.
#. :func:`caging_times_in_place_shape` — ``H(caging_atom, tolerance)``
   with phase bonus + override. Pick-place, peg-insert (uses the
   ``gripper_caging`` / ``pick_place_caging`` atoms specifically).
#. :func:`linear_combo_shape` — weighted sum of atoms plus phase bonus
   and success override. Push (caging + phase bonus), sum-of-tolerances
   (drawer-open).

Each shape takes a typed cfg listing the atom callables (from
:mod:`quantities`) and the per-task constants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from .utils import hamacher_product, tolerance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# An "atom" is any callable ``env -> Tensor[(B,)]``. The :mod:`quantities`
# module supplies the standard set; tasks can pass any custom callable too.
Atom = Callable[..., torch.Tensor]


def _eval_atom(atom_or_value: Atom | float, env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """Resolve a margin / quantity that may be either an atom function or a
    plain float (interpreted as a constant for all envs)."""
    if callable(atom_or_value):
        return atom_or_value(env, **kwargs)
    return torch.full((env.num_envs,), float(atom_or_value), device=env.device, dtype=torch.float32)


# ── Phase bonus + success override (composable across shapes) ───────────────


@configclass
class PhaseBonusCfg:
    """Conditional bonus added to the base reward when all triggers fire.

    The bonus is::

        bonus = bonus_offset + bonus_in_place_mult * in_place + bonus_self_mult * base

    All three coefficients default to 0 — set whichever your task uses. The
    Meta-World V2 bonuses fall into two patterns:

    * pick-place: ``offset=1.0, in_place_mult=5.0, self_mult=0.0``
    * push:       ``offset=1.0, in_place_mult=5.0, self_mult=1.0``
      (the ``self_mult=1`` reproduces the verbatim ``reward += 1 + reward + 5*in_place``)
    """

    triggers: list[TriggerCfg] = field(default_factory=list)
    """Conditions that must *all* be true for the bonus to apply."""

    offset: float = 0.0
    in_place_mult: float = 0.0
    self_mult: float = 0.0


@configclass
class TriggerCfg:
    """A scalar comparison trigger ``atom OP threshold``."""

    atom: Atom = None  # type: ignore[assignment]
    op: str = "<"  # one of "<", "<=", ">", ">=", "=="
    threshold: float = 0.0
    atom_kwargs: dict = field(default_factory=dict)


@configclass
class SuccessOverrideCfg:
    """Hard reward override when the success criterion is met."""

    quantity: Atom = None  # type: ignore[assignment]
    threshold: float = 0.05
    op: str = "<"  # success when ``quantity OP threshold``
    value: float = 10.0
    atom_kwargs: dict = field(default_factory=dict)


def _eval_trigger(trigger: TriggerCfg, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a bool tensor ``(B,)``."""
    val = trigger.atom(env, **trigger.atom_kwargs)
    if trigger.op == "<":
        return val < trigger.threshold
    if trigger.op == "<=":
        return val <= trigger.threshold
    if trigger.op == ">":
        return val > trigger.threshold
    if trigger.op == ">=":
        return val >= trigger.threshold
    if trigger.op == "==":
        return val == trigger.threshold
    raise ValueError(f"Unknown op {trigger.op!r}")


def _apply_phase_bonus(
    reward: torch.Tensor,
    in_place: torch.Tensor,
    phase: PhaseBonusCfg,
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Add the phase bonus where all triggers fire."""
    if not phase.triggers:
        mask = torch.ones_like(reward, dtype=torch.bool)
    else:
        mask = _eval_trigger(phase.triggers[0], env)
        for t in phase.triggers[1:]:
            mask = mask & _eval_trigger(t, env)
    bonus = phase.offset + phase.in_place_mult * in_place + phase.self_mult * reward
    return torch.where(mask, reward + bonus, reward)


def _apply_success_override(
    reward: torch.Tensor,
    override: SuccessOverrideCfg,
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    val = override.quantity(env, **override.atom_kwargs)
    if override.op == "<":
        mask = val < override.threshold
    elif override.op == "<=":
        mask = val <= override.threshold
    elif override.op == ">":
        mask = val > override.threshold
    elif override.op == ">=":
        mask = val >= override.threshold
    else:
        raise ValueError(f"Unknown op {override.op!r}")
    return torch.where(mask, torch.full_like(reward, override.value), reward)


# ── Shape A: pure tolerance ─────────────────────────────────────────────────


@configclass
class ToleranceShapeCfg:
    """Pure tolerance reward — Meta-World's reach-style template.

    ``reward = scale * tolerance(distance, bounds=(0, success_radius), margin)``
    """

    distance: Atom = None  # type: ignore[assignment]
    distance_kwargs: dict = field(default_factory=dict)
    margin: Atom | float = 0.1
    """Either an atom (margin varies per env) or a float constant."""
    margin_kwargs: dict = field(default_factory=dict)
    success_radius: float = 0.05
    sigmoid: str = "long_tail"
    scale: float = 10.0


def tolerance_shape(env: ManagerBasedRLEnv, *, cfg: ToleranceShapeCfg) -> torch.Tensor:
    """Compute the Shape-A reward."""
    distance = cfg.distance(env, **cfg.distance_kwargs)
    margin = _eval_atom(cfg.margin, env, **cfg.margin_kwargs)
    in_place = tolerance(
        distance,
        bounds=(0.0, cfg.success_radius),
        margin=margin,
        sigmoid=cfg.sigmoid,
    )
    return cfg.scale * in_place


# ── Shape B: caging × in-place + lift bonus + success override ──────────────


@configclass
class CagingTimesInPlaceShapeCfg:
    """Pick-place-style template: ``hamacher(caging, in_place)`` + bonus + override."""

    caging: Atom = None  # type: ignore[assignment]
    caging_kwargs: dict = field(default_factory=dict)
    distance: Atom = None  # type: ignore[assignment]
    distance_kwargs: dict = field(default_factory=dict)
    margin: Atom | float = 0.1
    margin_kwargs: dict = field(default_factory=dict)
    success_radius: float = 0.05
    sigmoid: str = "long_tail"
    phase: PhaseBonusCfg | None = None
    success_override: SuccessOverrideCfg | None = None


def caging_times_in_place_shape(env: ManagerBasedRLEnv, *, cfg: CagingTimesInPlaceShapeCfg) -> torch.Tensor:
    distance = cfg.distance(env, **cfg.distance_kwargs)
    margin = _eval_atom(cfg.margin, env, **cfg.margin_kwargs)
    in_place = tolerance(
        distance,
        bounds=(0.0, cfg.success_radius),
        margin=margin,
        sigmoid=cfg.sigmoid,
    )
    caging = cfg.caging(env, **cfg.caging_kwargs)
    reward = hamacher_product(caging, in_place)

    if cfg.phase is not None:
        reward = _apply_phase_bonus(reward, in_place, cfg.phase, env)
    if cfg.success_override is not None:
        reward = _apply_success_override(reward, cfg.success_override, env)
    return reward


# ── Shape C: weighted sum of caging + phase bonus + success override ───────


@configclass
class WeightedAtomCfg:
    """One term in a :class:`LinearComboShapeCfg` weighted sum.

    Wrapped as a configclass so OmegaConf / Hydra can introspect the dict —
    a plain ``tuple[float, Callable, dict]`` chokes on the function ref
    when configclass instances are converted to OmegaConf nodes.
    """

    weight: float = 1.0
    atom: Atom = None  # type: ignore[assignment]
    atom_kwargs: dict = field(default_factory=dict)


@configclass
class LinearComboShapeCfg:
    """Push-style template: ``Σ wᵢ · termᵢ`` + bonus + override."""

    terms: list[WeightedAtomCfg] = field(default_factory=list)
    in_place_distance: Atom = None  # type: ignore[assignment]
    in_place_distance_kwargs: dict = field(default_factory=dict)
    in_place_margin: Atom | float = 0.1
    in_place_margin_kwargs: dict = field(default_factory=dict)
    in_place_success_radius: float = 0.05
    sigmoid: str = "long_tail"
    phase: PhaseBonusCfg | None = None
    success_override: SuccessOverrideCfg | None = None


def linear_combo_shape(env: ManagerBasedRLEnv, *, cfg: LinearComboShapeCfg) -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device)
    for term in cfg.terms:
        reward = reward + term.weight * term.atom(env, **term.atom_kwargs)

    in_place = tolerance(
        cfg.in_place_distance(env, **cfg.in_place_distance_kwargs),
        bounds=(0.0, cfg.in_place_success_radius),
        margin=_eval_atom(cfg.in_place_margin, env, **cfg.in_place_margin_kwargs),
        sigmoid=cfg.sigmoid,
    )

    if cfg.phase is not None:
        reward = _apply_phase_bonus(reward, in_place, cfg.phase, env)
    if cfg.success_override is not None:
        reward = _apply_success_override(reward, cfg.success_override, env)
    return reward


# ── Shape D: Hamacher of two tolerances + optional modulator + override ────


@configclass
class HamacherShapeCfg:
    """Hamacher product of two shaped tolerances.

    ``reward = scale * H(α · tol(d_a, b_a, m_a, σ_a), tol(d_b, b_b, m_b, σ_b))``

    where ``α`` is an optional scalar modulator atom (e.g.
    ``gripper_close_action`` for drawer-close, ``1 - gripper_open`` for
    button-press's ``tcp_closed`` term). Use :class:`ToleranceShapeCfg` to
    describe each term's shaping. Captures every MW task whose reward is
    Hamacher of two shaped distances:

    * window-open/close: ``H(tcp_to_handle, |handle_x − target_x|)``
    * door-open, faucet-open: same shape, different bounds
    * drawer-close: ``H(grip · tcp_to_handle, handle_to_target)``
    * peg-insert: ``H(caging_atom, scaled in_place)``
    * button-press's first term: ``H(tcp_closed, near_button)``
    """

    term_a: ToleranceShapeCfg = None  # type: ignore[assignment]
    term_b: ToleranceShapeCfg = None  # type: ignore[assignment]
    a_modulator: Atom | None = None
    """Optional ``(B,) → (B,)`` atom multiplied into ``term_a`` before the
    Hamacher (e.g. ``gripper_close_action`` for drawer-close)."""
    a_modulator_kwargs: dict = field(default_factory=dict)
    scale: float = 10.0
    phase: PhaseBonusCfg | None = None
    success_override: SuccessOverrideCfg | None = None


def _eval_tolerance(env: ManagerBasedRLEnv, t: ToleranceShapeCfg) -> torch.Tensor:
    """Evaluate a ``ToleranceShapeCfg`` as a raw shaped value in ``[0, 1]``
    (skipping the per-cfg ``scale``)."""
    distance = t.distance(env, **t.distance_kwargs)
    margin = _eval_atom(t.margin, env, **t.margin_kwargs)
    return tolerance(
        distance,
        bounds=(0.0, t.success_radius),
        margin=margin,
        sigmoid=t.sigmoid,
    )


def hamacher_shape(env: ManagerBasedRLEnv, *, cfg: HamacherShapeCfg) -> torch.Tensor:
    """Compute the Hamacher-shape reward."""
    a = _eval_tolerance(env, cfg.term_a)
    if cfg.a_modulator is not None:
        a = a * cfg.a_modulator(env, **cfg.a_modulator_kwargs).clamp(0.0, 1.0)
    b = _eval_tolerance(env, cfg.term_b)
    reward = cfg.scale * hamacher_product(a, b)

    if cfg.phase is not None:
        reward = _apply_phase_bonus(reward, b, cfg.phase, env)
    if cfg.success_override is not None:
        reward = _apply_success_override(reward, cfg.success_override, env)
    return reward
