# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pipeline orchestrator for geometry-constrained articulation retargeting."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import newton.ik as ik
import numpy as np
import torch
import warp as wp
from pytorch3d.ops import sample_farthest_points

from .buffer import RetargetBuffer
from .cfg import RetargetPipelineCfg
from .sample import sample_contacts
from .validate import CriterionFn, validate_results

if TYPE_CHECKING:
    from ..kinematics import NewtonKinematics

ObjectivesFactory = Callable[
    [int],
    tuple[list, list[ik.IKObjectivePosition], ik.IKObjectivePosition, ik.IKObjectiveRotation],
]
"""Callable ``(n_problems) -> (all_objectives, contact_objectives,
base_pos_objective, base_rot_objective)``.

``all_objectives`` is the full list passed to :meth:`NewtonKinematics.create_ik_solver`.
The contact/base objectives are returned separately so the pipeline can fill targets."""


class RetargetPipeline:
    """Orchestrates the staged retargeting pipeline.

    The pipeline:
    1. Samples contact points on geometry.
    2. Builds IK objectives via a user-provided factory, creates
       solver via :meth:`NewtonKinematics.create_ik_solver`.
    3. Validates results via user-provided criteria.
    4. Selects final results (FPS for spatial uniformity).
    """

    def __init__(
        self,
        kin: NewtonKinematics,
        objectives_factory: ObjectivesFactory,
        cfg: RetargetPipelineCfg,
        contact_body_ids: list[int],
        foot_offsets: np.ndarray,
        foot_ground_offset: float,
        standing_height: float,
        default_joint_q: np.ndarray,
    ):
        """Initialize the pipeline.

        Args:
            kin: Newton kinematics (owns the model).
            objectives_factory: Callable ``(n_problems) -> (all_objs,
                contact_objs, base_pos_obj, base_rot_obj)``.
            cfg: Pipeline configuration (sampling params, max_candidates).
            contact_body_ids: Newton body indices for contact bodies.
            foot_offsets: Nominal foot positions relative to base ``[num_contacts, 3]``.
            foot_ground_offset: Height of foot body frame above ground contact [m].
            standing_height: Base height above foot body centroid at default stance [m].
            default_joint_q: Default joint coordinates ``[joint_coord_count]``.
        """
        self.cfg = cfg
        self.kin = kin
        self.objectives_factory = objectives_factory
        self.contact_body_ids = contact_body_ids
        self.foot_offsets = foot_offsets
        self.foot_ground_offset = foot_ground_offset
        self.standing_height = standing_height
        self.default_joint_q = default_joint_q

        self.buffer = RetargetBuffer(
            max_candidates=cfg.max_candidates,
            joint_coord_count=kin.model.joint_coord_count,
            num_bodies=kin.model.body_count,
            num_contacts=len(contact_body_ids),
            device=kin.device,
        )

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

        n_written, self._reject_geo = sample_contacts(
            wp_mesh=wp_mesh,
            origin=origin,
            buffer=self.buffer,
            cfg=self.cfg.sampling,
            foot_offsets=self.foot_offsets,
            foot_ground_offset=self.foot_ground_offset,
            standing_height=self.standing_height,
            default_joint_q=self.default_joint_q,
            n_desired=n_desired,
        )

        if n_written == 0:
            return self.buffer

        # Build objectives and solver
        N = self.buffer.num_geometry_valid
        all_objs, contact_objs, base_pos_obj, base_rot_obj = self.objectives_factory(N)
        solver = self.kin.create_ik_solver(all_objs, N)

        # Fill targets from buffer
        self.buffer.scatter_contact_targets(contact_objs, N)
        wp.copy(base_pos_obj.target_positions, self.buffer.base_target_pos, count=N)
        wp.copy(base_rot_obj.target_rotations, self.buffer.base_target_rot, count=N)

        # Solve
        jq_in = wp.from_torch(self.buffer.joint_q_init_t[:N].contiguous())
        jq_out = wp.from_torch(self.buffer.joint_q_result_t[:N].contiguous())
        solver.step(jq_in, jq_out, iterations=self.cfg.ik.iterations)
        self.buffer.joint_q_result_t[:N] = wp.to_torch(jq_out)

        # Validate
        if criteria:
            self._reject_val = validate_results(self.buffer, criteria)
        else:
            self.buffer._ik_valid[:] = self.buffer._geom_valid
            self.buffer.num_ik_valid = self.buffer.num_geometry_valid
            self.buffer.num_final_valid = self.buffer.num_geometry_valid
            self._reject_val = {"ok": self.buffer.num_geometry_valid}

        # FPS selection
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
