# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic retarget validation criteria.

These criteria work with any articulated robot without requiring
robot-specific constants. For robot-specific criteria (e.g. lateral hip
joint limits), see the per-robot preset modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from ...kinematics import NewtonKinematics
    from .buffer import RetargetBuffer
    from .cfg import SamplerCfg
    from .criteria_cfg import (
        CollisionCheckCfg,
        JointWithinLimitCfg,
        LateralHipLimitCfg,
        SolverCostOutlierCfg,
    )


@dataclass(frozen=True, slots=True)
class RetargetCriterionContext:
    """Explicit immutable dependencies shared by Position criteria."""

    kinematics: NewtonKinematics
    contact_body_ids: tuple[int, ...]
    collision_mesh: wp.Mesh
    sampler_cfg: SamplerCfg
    solver_costs: torch.Tensor


def _evaluate(criterion_type, cfg, candidates) -> torch.Tensor:
    """Evaluate one criterion against solved family candidates."""
    if candidates.solver_costs is None:
        raise RuntimeError("Retarget criteria require solved candidates.")
    context = RetargetCriterionContext(
        kinematics=candidates.kinematics,
        contact_body_ids=candidates.contact_body_ids,
        collision_mesh=candidates.collision_mesh,
        sampler_cfg=candidates.sampler_cfg,
        solver_costs=candidates.solver_costs,
    )
    criterion = criterion_type(cfg, context)
    return criterion(candidates.buffer, candidates.buffer.num_geometry_valid)


def evaluate_foot_position_error(cfg, candidates) -> torch.Tensor:
    if candidates.foot_position_error is None:
        raise RuntimeError("Foot-position criterion requires the cached final FK measure.")
    per_foot = candidates.foot_position_error
    count, contact_count = per_foot.shape
    is_contact = candidates.buffer.is_contact_t[: count * contact_count].view(count, contact_count)
    if cfg.aggregate == "sum":
        error = torch.where(is_contact, per_foot, torch.zeros_like(per_foot)).sum(dim=-1)
    elif cfg.aggregate == "max":
        error = torch.where(is_contact, per_foot, torch.full_like(per_foot, float("-inf"))).max(dim=-1).values
    else:
        raise ValueError(f"FootPositionError.aggregate must be 'max' or 'sum', got {cfg.aggregate!r}")
    return error <= cfg.max_err


def evaluate_lateral_hip_limit(cfg, candidates) -> torch.Tensor:
    return _evaluate(LateralHipLimit, cfg, candidates)


def evaluate_joint_within_limit(cfg, candidates) -> torch.Tensor:
    return _evaluate(JointWithinLimit, cfg, candidates)


def evaluate_support_polygon_stability(cfg, candidates) -> torch.Tensor:
    if candidates.stability_margin is None or candidates.active_contact_count is None:
        raise RuntimeError("Stability criterion requires the cached final objective measure.")
    if cfg.minimum_contacts < 3:
        raise ValueError("Support-polygon stability requires at least three active contacts.")
    return (candidates.active_contact_count >= cfg.minimum_contacts) & (
        candidates.stability_margin >= cfg.minimum_margin
    )


def evaluate_solver_cost_outlier(cfg, candidates) -> torch.Tensor:
    return _evaluate(SolverCostOutlier, cfg, candidates)


def evaluate_collision_check(cfg, candidates) -> torch.Tensor:
    return _evaluate(CollisionCheck, cfg, candidates)


class LateralHipLimit:
    """Criterion: lateral hip joint angles must not exceed ``max_angle`` [rad].

    Resolves absolute joint-coordinate indices at construction time from
    ``cfg.joint_pattern``.

    Args:
        cfg: :class:`~.criteria_cfg.LateralHipLimitCfg` with
            ``joint_pattern`` and ``max_angle`` fields.
        context: Explicit kinematics used for joint-name resolution.
    """

    def __init__(self, cfg: LateralHipLimitCfg, context: RetargetCriterionContext) -> None:
        pattern = cfg.joint_pattern
        if pattern is None:
            raise ValueError("LateralHipLimit requires an explicit joint_pattern.")
        self.max_angle = cfg.max_angle
        self._joint_indices = context.kinematics.find_joint_scalar_coordinates(pattern)[0]

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        if not self._joint_indices:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        joint_indices = torch.tensor(self._joint_indices, device=buffer.device, dtype=torch.long)
        return buffer.joint_q_result_t[:N, joint_indices].abs().max(dim=-1).values <= self.max_angle


class JointWithinLimit:
    """Criterion: all non-root joints must stay inside a scaled retarget joint interval.

    The checked interval is the same effective interval used by the FK
    sampler: Newton joint limits intersected with
    ``default_joint_q ± sampler.fk_joint_range`` and any
    ``sampler.fk_joint_range_overrides``. The final interval is then
    shrunk around its center by ``limit_ratio``.

    Args:
        cfg: :class:`~.criteria_cfg.JointWithinLimitCfg` with
            ``limit_ratio``.
        context: Explicit kinematics and sampler joint-range policy.
    """

    def __init__(self, cfg: JointWithinLimitCfg, context: RetargetCriterionContext) -> None:
        if not 0.0 < cfg.limit_ratio <= 1.0:
            raise ValueError(f"JointWithinLimit.limit_ratio must be in (0, 1], got {cfg.limit_ratio}.")
        kin = context.kinematics
        jl = torch.tensor(kin.topology.joint_limit_lower, device=kin.device)
        ju = torch.tensor(kin.topology.joint_limit_upper, device=kin.device)
        default_q = torch.from_numpy(kin.default_joint_q).float().to(kin.device)
        coordinates, velocities, _ = kin.find_joint_scalar_coordinates(".*")
        if not coordinates:
            raise ValueError("JointWithinLimit requires at least one scalar joint.")
        self._coordinate_indices = torch.tensor(coordinates, device=kin.device, dtype=torch.long)
        coordinate_slots = {coordinate: slot for slot, coordinate in enumerate(coordinates)}
        scalar_default = default_q[self._coordinate_indices]
        joint_range = torch.full((len(coordinates),), float(context.sampler_cfg.fk_joint_range), device=kin.device)
        for pattern, clamp in context.sampler_cfg.fk_joint_range_overrides.items():
            for coordinate in kin.find_joint_scalar_coordinates(pattern)[0]:
                joint_range[coordinate_slots[coordinate]] = float(clamp)
        velocity_indices = torch.tensor(velocities, device=kin.device, dtype=torch.long)
        lo = torch.maximum(jl[velocity_indices], scalar_default - joint_range)
        hi = torch.minimum(ju[velocity_indices], scalar_default + joint_range)
        half_margin = 0.5 * (1.0 - cfg.limit_ratio)
        span = hi - lo
        self._safe_lo = lo + half_margin * span
        self._safe_hi = hi - half_margin * span

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        jq = buffer.joint_q_result_t[:N, self._coordinate_indices]
        return ((jq >= self._safe_lo) & (jq <= self._safe_hi)).all(dim=-1)


@dataclass
class BaseZError:
    """Criterion: base z must not deviate more than ``max_err`` from target [m].

    Args:
        max_err: Maximum absolute base z deviation [m].
    """

    max_err: float = 0.3

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        base_z = buffer.joint_q_result_t[:N, 2]
        target_z = buffer.base_target_pos_t[:N, 2]
        return (base_z - target_z).abs() <= self.max_err


class SolverCostOutlier:
    """Criterion: reject candidates whose final solver cost is an outlier.

    Residual IK divergence -- after all geometric constraints pass, the
    solver may still settle on a poor local minimum. Reject candidates
    whose cost exceeds ``threshold_multiplier * median(costs)``.

    Args:
        cfg: :class:`~.criteria_cfg.SolverCostOutlierCfg` with
            ``threshold_multiplier``.
        context: Explicit final solver costs.
    """

    def __init__(self, cfg: SolverCostOutlierCfg, context: RetargetCriterionContext) -> None:
        self.solver_costs = context.solver_costs
        self.threshold_multiplier = cfg.threshold_multiplier

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        costs = self.solver_costs[:N]
        median = costs.median()
        return costs < median * self.threshold_multiplier


class CollisionCheck:
    """Criterion: reject candidates where body probes penetrate terrain.

    Reads ``body_q`` from the buffer (populated by the pipeline after
    IK), transforms collision probe offsets into world space via a Warp
    kernel, and queries the terrain mesh for penetration. No separate
    FK pass -- uses the body transforms already stored in the buffer.

    Args:
        cfg: :class:`~.criteria_cfg.CollisionCheckCfg` with ``n_samples``
            and ``max_pen``.
        context: Explicit kinematics, terrain mesh, and foot identities.
    """

    def __init__(self, cfg: CollisionCheckCfg, context: RetargetCriterionContext) -> None:
        from ...kinematics.ik_objectives.mesh_collision import collision_probes_sample  # noqa: PLC0415

        self.kin = context.kinematics
        self.wp_mesh = context.collision_mesh
        self.max_pen = cfg.max_pen
        bodies, offsets, slots = collision_probes_sample(self.kin.builder, context.contact_body_ids, cfg.n_samples)
        self._n_probes = len(bodies)
        self._n_feet = len(context.contact_body_ids)
        self._probe_body_np = bodies
        self._probe_offset_np = offsets
        self._probe_foot_slot_np = slots
        self._wp_bufs: dict[str, tuple[wp.array, wp.array, wp.array]] = {}

    def _get_wp_bufs(self, device: str) -> tuple[wp.array, wp.array, wp.array]:
        """Lazily allocate device buffers for probe data."""
        if device not in self._wp_bufs:
            self._wp_bufs[device] = (
                wp.array(self._probe_body_np, dtype=wp.int32, device=device),
                wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=device),
                wp.array(self._probe_foot_slot_np, dtype=wp.int32, device=device),
            )
        return self._wp_bufs[device]

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nb = self.kin.model.body_count
        body_q_wp = wp.from_torch(
            buffer.body_q_t[: N * nb].contiguous(),
            dtype=wp.transformf,
        )
        probe_body, probe_offset, probe_foot_slot = self._get_wp_bufs(buffer.device)
        # Slot-ordered per-problem contact flag; contact feet shouldn't be
        # flagged for "penetrating" terrain since they're meant to touch.
        is_contact_u8 = buffer.is_contact_t[: N * self._n_feet].view(N, self._n_feet).to(torch.uint8).contiguous()
        is_contact_wp = wp.from_torch(is_contact_u8, dtype=wp.uint8)

        pen_out = torch.zeros(N * self._n_probes, device=buffer.device, dtype=torch.float32)
        wp_pen = wp.from_torch(pen_out)
        wp.launch(
            _collision_check_kernel,
            dim=[N, self._n_probes],
            inputs=[
                body_q_wp,
                nb,
                self.wp_mesh.id,
                probe_body,
                probe_offset,
                probe_foot_slot,
                is_contact_wp,
                2.0,
                self._n_probes,
            ],
            outputs=[wp_pen],
            device=buffer.device,
        )
        wp.synchronize()
        return pen_out.view(N, self._n_probes).max(dim=-1).values <= self.max_pen


@wp.kernel
def _collision_check_kernel(
    body_q: wp.array1d(dtype=wp.transformf),
    n_bodies: int,
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    probe_foot_slot: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    max_dist: float,
    n_probes: int,
    penetration: wp.array1d(dtype=wp.float32),
):
    """Penetration depth for each (candidate, probe) pair.

    Combines two signals to handle both open and closed meshes:

    - ``-sign * dist``: correct for watertight meshes where
      ``mesh_query_point`` can determine inside/outside.
    - ``surface_z - probe_z``: correct for open heightfield terrains
      where ``sign`` is always +1.

    The max of both is used so penetration is detected either way.

    Probes on a foot body (``probe_foot_slot >= 0``) are zeroed when that
    foot is in contact: a contact foot is meant to touch terrain, so the
    criterion should not flag it as penetrating.
    """
    row, probe_idx = wp.tid()
    out_idx = row * n_probes + probe_idx
    slot = probe_foot_slot[probe_idx]
    if slot >= 0 and is_contact[row, slot] != wp.uint8(0):
        penetration[out_idx] = 0.0
        return
    tf = body_q[row * n_bodies + probe_body[probe_idx]]
    world_pos = wp.transform_point(tf, probe_offset[probe_idx])
    query = wp.mesh_query_point(mesh_id, world_pos, max_dist)
    if query.result:
        surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(world_pos - surface_pt)
        sign_pen = -query.sign * dist
        z_pen = surface_pt[2] - world_pos[2]
        penetration[out_idx] = wp.max(sign_pen, z_pen)
    else:
        penetration[out_idx] = 0.0
