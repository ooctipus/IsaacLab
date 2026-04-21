# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pipeline orchestrator for geometry-constrained articulation retargeting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp
from pytorch3d.ops import sample_farthest_points

from .buffer import RetargetBuffer
from .cfg import RetargetPipelineCfg, SamplerBaseCfg

from ...utils.kinematic import NewtonKinematics

CriterionFn = Callable[[RetargetBuffer, int], torch.Tensor]
"""Signature for a validation criterion.

Args:
    buffer: The retarget buffer with IK results populated.
    n_active: Number of geometry-valid candidates (first ``n_active`` rows).

Returns:
    Boolean tensor of shape ``[n_active]`` -- ``True`` = passes this criterion.
"""


class SamplerBase(ABC):
    """Abstract base for pipeline sampling strategies.

    Constructed from a cfg, a :class:`NewtonKinematics` instance, and
    the foot body indices.  Subclasses derive any robot geometry they
    need from ``kin`` and ``foot_body_ids`` instead of receiving it
    as explicit constructor arguments.
    """

    def __init__(self, cfg: SamplerBaseCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        self.cfg = cfg
        self.kin = kin
        self.foot_body_ids = foot_body_ids

    @property
    def group_size(self) -> int:
        """Number of variant candidates per source polygon.

        When greater than 1, the pipeline collapses each group to the
        single best candidate (lowest IK cost) after solving.
        """
        return 1

    @abstractmethod
    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
    ) -> tuple[int, dict[str, int]]:
        """Sample keypoints on geometry and write results to *buffer*.

        Args:
            wp_mesh: Terrain warp mesh.
            origin: Terrain origin offset ``[3]``.
            buffer: Pre-allocated retarget buffer (written in-place).
            n_desired: Number of valid candidates to aim for.

        Returns:
            ``(num_written, rejection_stats)`` where *rejection_stats*
            maps reason strings to counts.
        """
        ...


def _validate_results(
    buffer: RetargetBuffer,
    criteria: dict[str, CriterionFn],
) -> dict[str, int]:
    """Run user-defined acceptance criteria and update ``buffer.ik_valid``.

    Each criterion is a callable ``(buffer, N) -> bool[N]``.  Criteria
    are evaluated in insertion order; the rejection breakdown reports
    the *first* failing criterion per candidate (waterfall).

    Args:
        buffer: Retarget buffer with ``joint_q_result`` populated.
        criteria: Ordered mapping from criterion name to callable.

    Returns:
        Rejection breakdown mapping each criterion name to the number
        of candidates it rejected (first-failure attribution), plus
        an ``"ok"`` key for candidates that passed everything.
    """
    N = buffer.num_geometry_valid
    if N == 0:
        return {}

    device = buffer.device
    masks: dict[str, torch.Tensor] = {}
    for name, fn in criteria.items():
        masks[name] = fn(buffer, N)

    reject: dict[str, int] = {}
    passed = torch.ones(N, device=device, dtype=torch.bool)
    for name, mask in masks.items():
        failed_here = passed & ~mask
        reject[name] = int(failed_here.sum())
        passed = passed & mask

    reject["ok"] = int(passed.sum())

    buffer._ik_valid[:N] = passed
    buffer.num_ik_valid = reject["ok"]
    buffer.num_final_valid = reject["ok"]

    return reject


class RetargetPipeline:
    """Orchestrates the staged retargeting pipeline.

    Constructed from a single :class:`RetargetPipelineCfg` that nests
    the kinematics, sampler, and foot specification.  The pipeline
    builds the :class:`NewtonKinematics`, sampler, and standard IK
    objectives internally.

    Args:
        cfg: Full pipeline configuration.
    """

    def __init__(self, cfg: RetargetPipelineCfg):
        self.cfg = cfg

        self.kin = NewtonKinematics(cfg.kin)
        self.foot_body_ids = self.kin.find_body_indices(cfg.foot_body_names)

        self.sampler = cfg.sampler.class_type(cfg.sampler, self.kin, self.foot_body_ids)

        self.buffer = RetargetBuffer(
            max_candidates=cfg.max_candidates,
            joint_coord_count=self.kin.model.joint_coord_count,
            num_bodies=self.kin.model.body_count,
            num_contacts=len(self.foot_body_ids),
            device=self.kin.device,
        )
        self._fk_model: newton.Model | None = None
        self._fk_capacity: int = 0

    def _build_objectives(
        self, N: int, wp_mesh: wp.Mesh,
    ) -> tuple[list, list[ik.IKObjectivePosition], ik.IKObjectivePosition, ik.IKObjectiveRotation]:
        """Build standard + extra IK objectives for ``N`` problems."""
        device = self.kin.device

        contact_objs = [
            ik.IKObjectivePosition(
                link_index=fid, link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(N, dtype=wp.vec3, device=device), weight=1.0,
            )
            for fid in self.foot_body_ids
        ]
        base_pos_obj = ik.IKObjectivePosition(
            link_index=0, link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(N, dtype=wp.vec3, device=device), weight=0.05,
        )
        base_rot_obj = ik.IKObjectiveRotation(
            link_index=0, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(N, dtype=wp.vec4, device=device), weight=0.5,
        )
        jl_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.kin.model.joint_limit_lower,
            joint_limit_upper=self.kin.model.joint_limit_upper, weight=10.0,
        )

        all_objs = [*contact_objs, base_pos_obj, base_rot_obj, jl_obj]

        if self.cfg.extra_objectives_factory is not None:
            extras = self.cfg.extra_objectives_factory(
                self.kin, self.foot_body_ids, N, self.sampler, wp_mesh,
            )
            all_objs.extend(extras)

        return all_objs, contact_objs, base_pos_obj, base_rot_obj

    def run(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        n_desired: int,
        criteria: dict[str, CriterionFn] | None = None,
    ) -> RetargetBuffer:
        """Run the full pipeline.

        Args:
            wp_mesh: Terrain warp mesh.
            origin: Terrain origin ``[3]``.
            n_desired: Number of valid results desired.
            criteria: Ordered dict of ``{name: fn(buffer, N) -> bool[N]}``
                acceptance criteria.  If ``None``, all IK results are accepted.

        Returns:
            The buffer with ``num_selected`` results indexed by ``selected``.
        """
        self.buffer.reset()

        n_written, self._reject_geo = self.sampler(
            wp_mesh, origin, self.buffer, n_desired,
        )

        if n_written == 0:
            return self.buffer

        N = self.buffer.num_geometry_valid
        all_objs, contact_objs, base_pos_obj, base_rot_obj = self._build_objectives(N, wp_mesh)
        has_autodiff = any(not obj.supports_analytic() for obj in all_objs)
        jac_mode = ik.IKJacobianType.MIXED if has_autodiff else ik.IKJacobianType.ANALYTIC
        solver = self.kin.create_ik_solver(all_objs, N, jacobian_mode=jac_mode)

        if contact_objs:
            self.buffer.scatter_contact_targets(contact_objs, N)
        wp.copy(base_pos_obj.target_positions, self.buffer.base_target_pos, count=N)
        wp.copy(base_rot_obj.target_rotations, self.buffer.base_target_rot, count=N)

        jq_in = wp.from_torch(self.buffer.joint_q_init_t[:N].contiguous())
        jq_out = wp.from_torch(self.buffer.joint_q_result_t[:N].contiguous())

        max_iters = self.cfg.ik_iterations
        threshold = self.cfg.ik_convergence_threshold
        batch_size = max(1, min(10, max_iters))
        prev_cost = float("inf")
        total_iters = 0
        for _ in range(0, max_iters, batch_size):
            iters = min(batch_size, max_iters - total_iters)
            solver.step(jq_in, jq_out, iterations=iters)
            total_iters += iters
            cur_cost = float(wp.to_torch(solver.costs)[:N].mean())
            if abs(prev_cost - cur_cost) < threshold:
                break
            prev_cost = cur_cost
            jq_in = jq_out

        self.buffer.joint_q_result_t[:N] = wp.to_torch(jq_out)
        self._solver_costs = wp.to_torch(solver.costs)[:N].clone()

        self._eval_fk_batched(self.buffer.joint_q_result_t[:N], N)
        self._ik_iterations_used = total_iters

        gs = self.sampler.group_size
        if gs > 1 and N >= gs:
            costs_t = wp.to_torch(solver.costs)[:N]
            n_groups = N // gs
            cost_groups = costs_t[:n_groups * gs].view(n_groups, gs)
            best_in_group = cost_groups.argmin(dim=1)
            keep_idx = best_in_group + torch.arange(n_groups, device=costs_t.device) * gs

            nc = self.buffer.num_contacts
            self.buffer.joint_q_result_t[:n_groups] = self.buffer.joint_q_result_t[keep_idx]
            self.buffer.joint_q_init_t[:n_groups] = self.buffer.joint_q_init_t[keep_idx]
            self.buffer.contact_targets_t[:n_groups * nc] = (
                self.buffer.contact_targets_t.view(-1, nc, 3)[keep_idx].view(-1, 3)
            )
            self.buffer.base_target_pos_t[:n_groups] = self.buffer.base_target_pos_t[keep_idx]
            self.buffer.base_target_rot_t[:n_groups] = self.buffer.base_target_rot_t[keep_idx]
            self.buffer._geom_valid[:n_groups] = True
            self.buffer._geom_valid[n_groups:N] = False

            N = n_groups
            self.buffer.num_written = N
            self.buffer.num_geometry_valid = N

        if criteria:
            self._reject_val = _validate_results(self.buffer, criteria)
        else:
            self.buffer._ik_valid[:] = self.buffer._geom_valid
            self.buffer.num_ik_valid = self.buffer.num_geometry_valid
            self.buffer.num_final_valid = self.buffer.num_geometry_valid
            self._reject_val = {"ok": self.buffer.num_geometry_valid}

        valid_t = self.buffer._ik_valid[: self.buffer.num_written]
        valid_indices = valid_t.nonzero(as_tuple=False).squeeze(-1)
        n_valid = valid_indices.shape[0]
        n_select = min(n_desired, n_valid)

        if n_select > 0:
            base_xyz = self.buffer.joint_q_result_t[valid_indices, 0:3]
            _, local_idx = sample_farthest_points(base_xyz.unsqueeze(0), K=n_select)
            selected = valid_indices[local_idx.squeeze(0)]
            self.buffer._selected[:n_select] = selected.to(torch.int32)

        self.buffer.num_selected = n_select
        return self.buffer

    def _eval_fk_batched(self, jq_flat: torch.Tensor, N: int) -> None:
        """Run FK on solved joint coordinates and store body_q in the buffer.

        Args:
            jq_flat: Solved joint coordinates ``[N, joint_coord_count]``.
            N: Number of candidates.
        """
        if self._fk_model is None or self._fk_capacity < N:
            self._fk_model = self.kin.build_batched_model(N)
            self._fk_capacity = N

        jq_wp = wp.from_torch(jq_flat.contiguous().view(-1))
        self._fk_model.joint_q = jq_wp
        st = self._fk_model.state()
        newton.eval_fk(
            self._fk_model, jq_wp,
            wp.zeros(self._fk_model.joint_dof_count, dtype=float, device=self.kin.device), st,
        )
        bq_torch = wp.to_torch(st.body_q)
        nb = self.kin.model.body_count
        self.buffer.body_q_t[:N * nb] = bq_torch[:N * nb]

    @property
    def rejection_summary(self) -> str:
        """Human-readable summary of the last run."""
        buf = self.buffer
        lines = [
            f"Pipeline summary:",
            f"  Candidates written: {buf.num_written}",
            f"  Geometry valid:     {buf.num_geometry_valid}",
            f"  IK valid:           {buf.num_ik_valid}",
            f"  Final valid:        {buf.num_final_valid}",
            f"  Selected:           {buf.num_selected}",
        ]
        return "\n".join(lines)
