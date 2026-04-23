# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pipeline orchestrator for geometry-constrained articulation retargeting."""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager

import newton.ik as ik
import numpy as np
import torch
import warp as wp

from ...terrains.utils.grid_downsample import grid_bucket_downsample
from ...utils.kinematic import NewtonKinematics
from .buffer import RetargetBuffer
from .cfg import RetargetPipelineCfg, SamplerBaseCfg

CriterionFn = Callable[[RetargetBuffer, int], torch.Tensor]
"""Signature for a validation criterion.

Args:
    buffer: The retarget buffer with IK results populated.
    n_active: Number of geometry-valid candidates (first ``n_active`` rows).

Returns:
    Boolean tensor of shape ``[n_active]`` -- ``True`` = passes this criterion.
"""


_TIMING_LABELS: dict[str, str] = {
    # Pipeline-level phases (keys set by :meth:`RetargetPipeline._time`).
    "sampler": "sampler (polygon candidate gen)",
    "ik_build": "IK build (solver + objectives)",
    "ik_solve": "IK solve (batched)",
    "fk_eval": "FK eval (post-solve)",
    "criteria": "criteria (acceptance checks)",
    "final_fps": "final FPS (spatial thin → target)",
    # Sub-phases of :class:`SupportPolygonSampler`. Keys unmatched here
    # fall back to their last dotted segment, so custom samplers still
    # render readably even without an entry.
    "sampler.morph": "morph-patch pool",
    "sampler.morph.setup": "allocate warp arrays",
    "sampler.morph.rasterize": "rasterize terrain",
    "sampler.morph.validity": "patch flatness check",
    "sampler.morph.candidates": "foot-reach candidate patches",
    "sampler.morph.fps": "FPS-thin morph pool",
    "sampler.neighbors": "assemble foot neighborhoods",
    "sampler.polygon_build": "stitch feet → polygons",
    "sampler.sampler_fps": "FPS-thin polygon pool",
    "sampler.prepare_ik": "build IK seeds",
}
"""Human-readable labels for timing keys in :attr:`RetargetPipeline._timings`."""


@dataclasses.dataclass(frozen=True)
class SamplerSizing:
    """Back-derived stage sizes for a :class:`SamplerBase` implementation.

    Returned by :meth:`SamplerBase.sizing` so the pipeline can size the
    shared :class:`RetargetBuffer` before calling the sampler.
    """

    n_final: int
    """Final placements returned to the caller."""

    oversample_candidates: int
    """Polygons sent to IK per final placement (``n_polygons_to_ik / n_final``)."""

    max_neighborhoods: int
    """Neighborhoods (e.g. 4-foot polygon centers) the sampler will assemble."""

    n_morph_patches: int
    """Morph-patch target for terrain contact sampling."""

    max_polygons: int
    """Upper bound on polygon candidates the sampler will emit.

    Sizes the :class:`RetargetBuffer`. The sampler must guarantee that it
    never writes more than this many rows into the buffer.
    """


def compute_sampler_sizing(
    n_final: int,
    *,
    final_fps_oversample: float = 1.5,
    criteria_yield: float = 0.5,
    polygon_fps_oversample: float = 2.0,
    polygon_assembly_yield: float = 0.8,
    patches_per_polygon: int = 4,
    morph_patch_oversample: float = 4.0,
) -> SamplerSizing:
    """Back-derive sampler stage sizes from the target final placement count.

    Walks the pipeline backwards, applying an oversample multiplier at
    every downsampling stage and an expected yield rate at every filter
    stage. Produces a :class:`SamplerSizing` that scales with ``n_final``
    rather than hitting any fixed cap.

    The cascade (post → pre):

    1. ``n_final`` placements after final FPS.
    2. ``n_final × final_fps_oversample`` after criteria validation
       (FPS needs a bigger pool to achieve good spatial spread).
    3. ``(2) / criteria_yield`` polygons entering criteria
       (accounts for cost/base-z/collision/HAA rejection).
    4. ``(3) × polygon_fps_oversample`` polygons entering polygon-FPS
       (grid-bucket FPS in the sampler needs a bigger pool too).
    5. ``(4) / polygon_assembly_yield`` neighborhoods -- the polygon-
       level yield captures the fraction passing base-height
       feasibility. Samplers that expand each polygon into cyclic
       rotation variants (``group_size > 1``) scale ``max_polygons``
       by ``group_size`` themselves so the buffer can hold every
       variant pre-collapse.
    6. ``n_final × patches_per_polygon × morph_patch_oversample`` morph-
       patches -- decoupled from K. The sampler reuses morph patches
       across centers (a single patch can serve many neighborhoods via
       different center/yaw combinations), so morph sizing only needs
       to cover the terrain densely enough for each foot-sector query
       to return at least one patch. Scaling with ``n_final`` (desired
       final count, a proxy for terrain area) is sufficient.

    Args:
        n_final: Target final placement count (e.g. ``area / spacing**2``).
        final_fps_oversample: Headroom multiplier into the final FPS.
        criteria_yield: Expected fraction of IK solves surviving criteria.
        polygon_fps_oversample: Headroom multiplier into the polygon FPS.
        polygon_assembly_yield: Expected fraction of neighborhoods whose
            assembled polygon passes base-height feasibility.
        patches_per_polygon: Foot contacts per polygon.
        morph_patch_oversample: Morph-patches per foot slot — ensures the
            neighborhood's in-range ball has several valid patches.

    Returns:
        :class:`SamplerSizing` with the stage sizes.
    """
    n_post_criteria = int(np.ceil(n_final * final_fps_oversample))
    n_polygons_pre_criteria = int(np.ceil(n_post_criteria / criteria_yield))
    n_polygons_pre_fps = int(np.ceil(n_polygons_pre_criteria * polygon_fps_oversample))
    K = int(np.ceil(n_polygons_pre_fps / polygon_assembly_yield))
    n_morph_patches = int(np.ceil(n_final * patches_per_polygon * morph_patch_oversample))
    oversample_candidates = max(1, int(np.ceil(n_polygons_pre_criteria / max(n_final, 1))))
    return SamplerSizing(
        n_final=n_final,
        oversample_candidates=oversample_candidates,
        max_neighborhoods=K,
        n_morph_patches=n_morph_patches,
        max_polygons=K,
    )


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
        self.sub_timings: dict[str, float] = {}
        """Per-subphase wall-time breakdown from the last :meth:`__call__`.

        Keys are sub-phase names (dots indicate nesting); values are seconds.
        The pipeline merges this into its own timings table under the
        ``sampler.`` prefix.
        """
        self.init_info: str | None = None
        """One-line summary of any one-time init work (e.g. reachability FK).

        Surfaced in :attr:`RetargetPipeline.rejection_summary`. Sampler
        subclasses may set this during ``__init__`` to report what was
        precomputed.
        """

    @property
    def group_size(self) -> int:
        """Number of variant candidates per source polygon.

        When greater than 1, the pipeline collapses each group to the
        single best candidate (lowest IK cost) after solving.
        """
        return 1

    @abstractmethod
    def sizing(self, n_desired: int) -> SamplerSizing:
        """Back-derive stage sizes from a target final-robot count.

        The pipeline calls this before each :meth:`run` to size the shared
        :class:`RetargetBuffer` — the sampler guarantees it will emit at
        most :attr:`SamplerSizing.max_polygons` candidates into the buffer.
        """
        ...

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
    group_size: int = 1,
) -> tuple[dict[str, int], torch.Tensor]:
    """Run user-defined acceptance criteria with waterfall attribution.

    Each criterion is a callable ``(buffer, N) -> bool[N]``. Criteria
    are evaluated in insertion order. Attribution semantics depend on
    ``group_size``:

    - ``group_size == 1``: per-candidate waterfall — each candidate is
      attributed to its first failing criterion.
    - ``group_size > 1``: group-level waterfall — candidates are
      grouped in contiguous blocks of ``group_size`` siblings (e.g.
      cyclic rotation variants). A group is attributed to the first
      criterion that eliminates its last surviving sibling. Groups
      whose ``N`` is not a multiple of ``group_size`` fall back to the
      per-candidate waterfall.

    Side-effect free — the caller is responsible for updating
    ``buffer._ik_valid`` / ``num_ik_valid`` / ``num_final_valid`` from
    the returned mask (which is per-candidate regardless of grouping).

    Args:
        buffer: Retarget buffer with ``joint_q_result`` populated and
            FK already evaluated.
        criteria: Ordered mapping from criterion name to callable.
        group_size: Candidates per source group (sampler's
            :attr:`~.SamplerBase.group_size`).

    Returns:
        ``(reject, cum_pass)`` where ``reject`` maps criterion name to
        the number of rejections (candidates or groups, per the rule
        above) plus an ``"ok"`` key for survivors, and ``cum_pass`` is
        the per-candidate ``bool[N]`` mask of all-criteria-passing rows.
    """
    N = buffer.num_geometry_valid
    device = buffer.device
    if N == 0:
        return {}, torch.zeros(0, device=device, dtype=torch.bool)

    masks: dict[str, torch.Tensor] = {}
    for name, fn in criteria.items():
        masks[name] = fn(buffer, N)

    cum_pass = torch.ones(N, device=device, dtype=torch.bool)
    reject: dict[str, int] = {}

    if group_size > 1 and N % group_size == 0:
        n_groups = N // group_size
        prev_survivor = torch.ones(n_groups, device=device, dtype=torch.bool)
        for name, mask in masks.items():
            cum_pass = cum_pass & mask
            survivor = cum_pass.view(n_groups, group_size).any(dim=1)
            killed_here = prev_survivor & ~survivor
            reject[name] = int(killed_here.sum())
            prev_survivor = survivor
        reject["ok"] = int(prev_survivor.sum())
    else:
        for name, mask in masks.items():
            failed_here = cum_pass & ~mask
            reject[name] = int(failed_here.sum())
            cum_pass = cum_pass & mask
        reject["ok"] = int(cum_pass.sum())

    return reject, cum_pass


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

        # Buffer is allocated lazily on the first ``run()`` call -- sized
        # from the sampler's yield-rate sizing for that call's ``n_desired``.
        # Subsequent calls reuse the buffer if it still fits; otherwise it
        # is reallocated to the larger sizing.
        self.buffer: RetargetBuffer | None = None
        self._fk_joint_qd: wp.array | None = None
        self._fk_body_qd: wp.array | None = None
        self._timings: dict[str, float] = {}
        self._reject_geo: dict[str, int] = {}
        self._reject_val: dict[str, int] = {}
        self._ik_iterations_used: int = 0
        self._solver_costs: torch.Tensor | None = None
        self._n_ik_problems: int = 0
        """Pre-group-collapse IK problem count from the last run.

        When ``sampler.group_size > 1``, the pipeline solves IK for
        ``group_size × n_polygons`` problems, then keeps only the best
        cost per group. This field preserves the pre-collapse count for
        the rejection summary; ``buffer.num_written`` is overwritten to
        the post-collapse count for downstream consumers.
        """
        self._n_desired: int = 0
        """Target placement count from the last :meth:`run`. Used by
        :attr:`rejection_summary` to explain the sizing cascade."""
        self._sizing: SamplerSizing | None = None
        """Back-cascaded sampler sizing from the last :meth:`run`. Used
        by :attr:`rejection_summary` to report the polygon budget."""
        self._cyclic_all_pass: int = 0
        """Groups where *every* cyclic sibling passed criteria (``gs > 1``).

        For these the rotation variants add no feasibility; collapse
        picks the cheapest of :attr:`SamplerBase.group_size` solutions
        purely as a cost improvement.
        """
        self._cyclic_salvaged: int = 0
        """Groups where only a subset of cyclic siblings passed criteria.

        These groups survived *because* cyclic search found a passing
        rotation when other rotations failed; a ``gs = 1`` run would
        have had to get lucky with its single seed to keep them.
        """

    def _ensure_buffer(self, n_desired: int) -> None:
        """Allocate (or grow) the retarget buffer for a given ``n_desired``."""
        needed = self.sampler.sizing(n_desired).max_polygons
        if self.buffer is not None and self.buffer.max_candidates >= needed:
            return
        self.buffer = RetargetBuffer(
            max_candidates=needed,
            joint_coord_count=self.kin.model.joint_coord_count,
            num_bodies=self.kin.model.body_count,
            num_contacts=len(self.foot_body_ids),
            device=self.kin.device,
        )

    @contextmanager
    def _time(self, name: str):
        """Record wall time for a pipeline phase, with CUDA sync on enter/exit."""
        if self.kin.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.kin.device.startswith("cuda"):
                torch.cuda.synchronize()
            self._timings[name] = self._timings.get(name, 0.0) + (time.perf_counter() - t0)

    def _build_objectives(
        self,
        N: int,
        wp_mesh: wp.Mesh,
    ) -> tuple[list, list[ik.IKObjectivePosition], ik.IKObjectivePosition, ik.IKObjectiveRotation]:
        """Build standard + extra IK objectives for ``N`` problems."""
        device = self.kin.device

        contact_objs = [
            ik.IKObjectivePosition(
                link_index=fid,
                link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
                weight=1.0,
            )
            for fid in self.foot_body_ids
        ]
        base_pos_obj = ik.IKObjectivePosition(
            link_index=0,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
            weight=0.05,
        )
        base_rot_obj = ik.IKObjectiveRotation(
            link_index=0,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(N, dtype=wp.vec4, device=device),
            weight=0.5,
        )
        jl_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.kin.model.joint_limit_lower,
            joint_limit_upper=self.kin.model.joint_limit_upper,
            weight=10.0,
        )

        all_objs = [*contact_objs, base_pos_obj, base_rot_obj, jl_obj]

        for obj_cfg in self.cfg.extra_objectives:
            all_objs.append(obj_cfg.build(self, wp_mesh))

        return all_objs, contact_objs, base_pos_obj, base_rot_obj

    def _build_criteria(self, wp_mesh: wp.Mesh) -> dict[str, CriterionFn]:
        """Build the acceptance-criteria dict from ``cfg.criteria``.

        Called once per ``run()`` so criteria can close over the current
        ``wp_mesh`` and the pipeline's per-run state (e.g. solver costs).
        Preserves list order in the resulting dict so rejection buckets
        attribute failures to the first criterion a candidate violates.
        """
        return {crit_cfg.name: crit_cfg.build(self, wp_mesh) for crit_cfg in self.cfg.criteria}

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
                acceptance criteria. ``None`` (default) builds criteria
                from :attr:`RetargetPipelineCfg.criteria`; pass an empty
                dict ``{}`` to skip acceptance checks entirely.

        Returns:
            The buffer with ``num_selected`` results indexed by ``selected``.
        """
        if criteria is None:
            criteria = self._build_criteria(wp_mesh)
        self._ensure_buffer(n_desired)
        self.buffer.reset()
        self._timings.clear()
        self._reject_val = {}
        self._ik_iterations_used = 0
        self._n_ik_problems = 0
        self._cyclic_all_pass = 0
        self._cyclic_salvaged = 0
        self._n_desired = n_desired
        self._sizing = self.sampler.sizing(n_desired)

        with self._time("sampler"):
            n_written, self._reject_geo = self.sampler(
                wp_mesh,
                origin,
                self.buffer,
                n_desired,
            )

        for k, v in self.sampler.sub_timings.items():
            # Dots in the key indicate nested phases; indent proportionally.
            depth = k.count(".")
            self._timings[f"{'  ' * (depth + 1)}sampler.{k}"] = v

        if n_written == 0:
            return self.buffer

        N = self.buffer.num_geometry_valid
        self._n_ik_problems = N
        with self._time("ik_build"):
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

        with self._time("ik_solve"):
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

        with self._time("fk_eval"):
            self._eval_fk_batched(N)
        self._ik_iterations_used = total_iters

        # Apply criteria to the pre-collapse candidate set so group-select
        # can prefer criteria-passing siblings over merely low-cost ones.
        # When ``gs > 1`` the rejection waterfall is group-level — a group
        # is only attributed to a criterion once *all* its siblings fail.
        gs = self.sampler.group_size
        with self._time("criteria"):
            if criteria:
                self._reject_val, cum_pass = _validate_results(self.buffer, criteria, group_size=gs)
            else:
                cum_pass = torch.ones(N, device=self.buffer.device, dtype=torch.bool)
                self._reject_val = {"ok": N}

        if gs > 1 and gs <= N and N % gs == 0:
            # Group-collapse: keep the min-cost *passing* sibling per group.
            # Non-passing siblings are masked to +inf so argmin prefers any
            # criteria-valid variant; groups with zero passing siblings still
            # produce an index but land with ``ik_valid = False`` and drop at
            # final-FPS.
            n_groups = N // gs
            costs_t = self._solver_costs[:N]
            masked_costs = torch.where(cum_pass, costs_t, torch.full_like(costs_t, float("inf")))
            cost_groups = masked_costs.view(n_groups, gs)
            best_in_group = cost_groups.argmin(dim=1)
            keep_idx = best_in_group + torch.arange(n_groups, device=costs_t.device) * gs
            pass_groups = cum_pass.view(n_groups, gs)
            group_survivor = pass_groups.any(dim=1)
            # Split surviving groups by how much cyclic feasibility bought
            # us: ``all_pass`` = every sibling passed (cyclic only helped
            # cost), ``salvaged`` = only some passed (cyclic was essential
            # for feasibility). The rest are already accounted for in the
            # criteria waterfall.
            group_all_pass = pass_groups.all(dim=1)
            self._cyclic_all_pass = int(group_all_pass.sum())
            self._cyclic_salvaged = int((group_survivor & ~group_all_pass).sum())

            nc = self.buffer.num_contacts
            nb = self.buffer.num_bodies
            self.buffer.joint_q_result_t[:n_groups] = self.buffer.joint_q_result_t[keep_idx]
            self.buffer.joint_q_init_t[:n_groups] = self.buffer.joint_q_init_t[keep_idx]
            self.buffer.contact_targets_t[: n_groups * nc] = self.buffer.contact_targets_t.view(-1, nc, 3)[
                keep_idx
            ].view(-1, 3)
            self.buffer.base_target_pos_t[:n_groups] = self.buffer.base_target_pos_t[keep_idx]
            self.buffer.base_target_rot_t[:n_groups] = self.buffer.base_target_rot_t[keep_idx]
            # body_q was FK'd from the pre-collapse joint_q layout; reshuffle
            # it alongside joint_q_result so downstream consumers reading
            # ``body_q[i]`` see the FK of the actual per-group winner.
            self.buffer.body_q_t[: n_groups * nb] = self.buffer.body_q_t.view(-1, nb, 7)[keep_idx].view(-1, 7)
            self.buffer._geom_valid[:n_groups] = True
            self.buffer._geom_valid[n_groups:N] = False
            self.buffer._ik_valid[:n_groups] = group_survivor
            self.buffer._ik_valid[n_groups:] = False

            N = n_groups
            self.buffer.num_written = N
            self.buffer.num_geometry_valid = N
            self.buffer.num_ik_valid = int(group_survivor.sum())
            self.buffer.num_final_valid = self.buffer.num_ik_valid
        else:
            self.buffer._ik_valid[:N] = cum_pass
            self.buffer.num_ik_valid = int(cum_pass.sum())
            self.buffer.num_final_valid = self.buffer.num_ik_valid

        with self._time("final_fps"):
            valid_t = self.buffer._ik_valid[: self.buffer.num_written]
            valid_indices = valid_t.nonzero(as_tuple=False).squeeze(-1)
            n_valid = valid_indices.shape[0]
            n_select = min(n_desired, n_valid)

            if n_select > 0:
                base_xyz = self.buffer.joint_q_result_t[valid_indices, 0:3]
                local_idx = grid_bucket_downsample(base_xyz, n_select)
                selected = valid_indices[local_idx]
                self.buffer._selected[:n_select] = selected.to(torch.int32)

            self.buffer.num_selected = n_select
        return self.buffer

    def _eval_fk_batched(self, N: int) -> None:
        """Run batched FK on solved joint coordinates and store ``body_q`` in the buffer.

        Allocates (or reallocates, when the buffer has grown) ``joint_qd`` and
        ``body_qd`` scratches at ``max_candidates`` and reuses the buffer's
        ``body_q`` slab as the zero-copy output. Delegates the actual kernel
        launch to :meth:`NewtonKinematics.eval_fk_batched`.

        Args:
            N: Number of candidates to evaluate (first ``N`` rows of the buffer).
        """
        max_cand = self.buffer.max_candidates
        nb = self.kin.model.body_count

        if self._fk_joint_qd is None or self._fk_joint_qd.shape[0] < max_cand:
            self._fk_joint_qd = wp.zeros(
                (max_cand, self.kin.model.joint_dof_count), dtype=wp.float32, device=self.kin.device
            )
            self._fk_body_qd = wp.zeros((max_cand, nb), dtype=wp.spatial_vectorf, device=self.kin.device)

        # Zero-copy view into the pre-allocated body_q slab as [max_cand, nb] transformf.
        bq_wp = wp.from_torch(self.buffer.body_q_t.view(max_cand, nb, 7), dtype=wp.transformf)

        self.kin.eval_fk_batched(
            self.buffer.joint_q_result[:N],
            self._fk_joint_qd[:N],
            bq_wp[:N],
            self._fk_body_qd[:N],
        )

    @property
    def rejection_summary(self) -> str:
        """Human-readable summary of the last pipeline run.

        Each stage reports its *unit* (polygons, IK candidates, groups,
        or placements) so the funnel from one count to the next is
        unambiguous. When ``sampler.group_size > 1``, criteria
        attribution is **group-level**: a group is counted as rejected
        only when all ``group_size`` rotation variants fail.

        Stages (conditionally emitted):

        1. **Sampler** — morph-patch pool → per-foot reachability
           sampling (annulus ∩ sector) → polygon-FPS. Unit: polygons.
        2. **Cyclic expansion** (``gs > 1``) — each polygon spawns
           ``gs`` rotation variants. Polygons → IK candidates.
        3. **IK solve** — per-candidate IK with the contact + base
           pose + joint-limit objectives.
        4. **Criteria** — acceptance predicates applied pre-collapse
           (per-candidate when ``gs = 1``; group-level when ``gs > 1``).
        5. **Group-select** (``gs > 1``) — keep the min-cost sibling
           among criteria-passing siblings per group; drop groups
           with no passing sibling.
        6. **Final FPS** — farthest-point downsample to ``n_desired``.
        """
        buf = self.buffer
        reject_geo = self._reject_geo
        reject_val = self._reject_val
        gs = self.sampler.group_size
        n_polys_valid = self._n_ik_problems // gs if gs > 0 else self._n_ik_problems
        n_polys_rejected = sum(reject_geo.get(k, 0) for k in ("out_of_reach", "shape_infeasible"))
        n_polys_attempted = n_polys_valid + n_polys_rejected

        def _render_table(headers: list[str], aligns: list[str], sections: list[list[list[str]]]) -> list[str]:
            """Render a Unicode box table with section separators."""
            all_rows = [headers] + [r for sec in sections for r in sec]
            cols = len(headers)
            widths = [max(len(row[i]) for row in all_rows) for i in range(cols)]

            def border(L: str, M: str, R: str) -> str:
                return L + M.join("─" * (w + 2) for w in widths) + R

            def fmt_row(cells: list[str]) -> str:
                parts = []
                for i, c in enumerate(cells):
                    if aligns[i] == "r":
                        parts.append(" " + c.rjust(widths[i]) + " ")
                    else:
                        parts.append(" " + c.ljust(widths[i]) + " ")
                return "│" + "│".join(parts) + "│"

            out = [border("┌", "┬", "┐"), fmt_row(headers), border("├", "┼", "┤")]
            for i, sec in enumerate(sections):
                if i > 0:
                    out.append(border("├", "┼", "┤"))
                for row in sec:
                    out.append(fmt_row(row))
            out.append(border("└", "┴", "┘"))
            return out

        def fmt(n: int) -> str:
            return f"{n:,}"

        header = "Retarget pipeline"
        if gs > 1:
            header += f"  ·  cyclic×{gs}"
        lines = [header]
        if self.sampler.init_info:
            lines.append(f"  init: {self.sampler.init_info}")

        # Sizing explainer: back-cascade from ``n_desired`` yields the
        # pre-cyclic polygon budget ``max_neighborhoods``. The buffer is
        # allocated for ``max_polygons = max_neighborhoods × group_size``
        # so it can hold every cyclic rotation variant pre-collapse. The
        # Sampler row below reports pre-cyclic polygons, so we show the
        # pre-cyclic budget there; the buffer figure is a parenthetical.
        polygon_budget = self._sizing.max_neighborhoods if self._sizing is not None else 0
        buffer_budget = self._sizing.max_polygons if self._sizing is not None else 0
        if self._n_desired > 0 and polygon_budget > 0:
            over = polygon_budget / self._n_desired
            line = (
                f"  sizing: target={self._n_desired} → budget={fmt(polygon_budget)} polygons"
                f" ({over:.0f}× oversample for FPS diversity)"
            )
            if gs > 1 and buffer_budget > polygon_budget:
                line += f"; buffer={fmt(buffer_budget)} rows (×{gs} cyclic)"
            lines.append(line)
            # Cascade breakdown is cfg-specific; emit it if the sampler
            # carries a ``SamplerSizingCfg`` at ``cfg.sizing``.
            sz_cfg = getattr(getattr(self.sampler, "cfg", None), "sizing", None)
            if sz_cfg is not None and all(
                hasattr(sz_cfg, k)
                for k in (
                    "final_fps_oversample",
                    "criteria_yield",
                    "polygon_fps_oversample",
                    "polygon_assembly_yield",
                )
            ):
                lines.append(
                    f"    cascade: {self._n_desired}"
                    f" ×{sz_cfg.final_fps_oversample:g}(final FPS)"
                    f" ÷{sz_cfg.criteria_yield:g}(criteria yield)"
                    f" ×{sz_cfg.polygon_fps_oversample:g}(polygon FPS)"
                    f" ÷{sz_cfg.polygon_assembly_yield:g}(geometry yield)"
                    f" = {fmt(polygon_budget)}"
                )
        lines.append("")

        # --- Main pipeline table: funnel with units per stage ---
        sections: list[list[list[str]]] = []

        # Sampler (polygon funnel).  Rejection rows render as "−N" so the
        # reader sees they subtract from ``attempted`` down to ``valid``.
        # Header compares ``attempted`` against the pre-cyclic polygon
        # budget (unit-match: this row counts pre-cyclic polygons). The
        # ``valid`` row appends the geometric yield rate.
        attempted_cell = (
            f"{fmt(n_polys_attempted)} / {fmt(polygon_budget)} budget" if polygon_budget > 0 else fmt(n_polys_attempted)
        )
        sampler_rows = [["Sampler", "polygons", attempted_cell]]
        for reason in ("out_of_reach", "shape_infeasible"):
            if reason in reject_geo:
                sampler_rows.append([f"  ─ {reason}", "", f"−{fmt(reject_geo[reason])}"])
        geo_yield = 100.0 * n_polys_valid / n_polys_attempted if n_polys_attempted > 0 else 0.0
        sampler_rows.append(["  valid", "polygons", f"{fmt(n_polys_valid)} ({geo_yield:.1f}%)"])
        sections.append(sampler_rows)

        # Cyclic expansion (unit switches polygons → candidates).
        if gs > 1:
            sections.append(
                [
                    [
                        f"Cyclic expansion (×{gs})",
                        "candidates",
                        f"{fmt(n_polys_valid)} → {fmt(self._n_ik_problems)}",
                    ]
                ]
            )

        # IK solve.
        sections.append([[f"IK solve ({self._ik_iterations_used} iters)", "candidates", fmt(self._n_ik_problems)]])

        # Criteria.  Waterfall unit = polygon groups when gs > 1, else candidates.
        if reject_val and any(k != "ok" for k in reject_val):
            crit_unit = "polygon groups" if gs > 1 else "candidates"
            crit_total_in = n_polys_valid if gs > 1 else self._n_ik_problems
            crit_header = "Criteria (group waterfall)" if gs > 1 else "Criteria (per-candidate)"
            crit_rows = [[crit_header, crit_unit, fmt(crit_total_in)]]
            for name, count in reject_val.items():
                if name == "ok":
                    continue
                crit_rows.append([f"  ─ {name}", "", f"−{fmt(count)}"])
            ok = reject_val.get("ok", 0)
            pct = 100.0 * ok / crit_total_in if crit_total_in > 0 else 0.0
            crit_rows.append(["  survivors", crit_unit, f"{fmt(ok)} ({pct:.1f}%)"])
            sections.append(crit_rows)
        else:
            sections.append([["Criteria (none applied)", "—", fmt(buf.num_ik_valid)]])

        # Group-select.  Two sub-rows break the placements by whether
        # the rotation variants actually salvaged feasibility (only some
        # rotations passed) or merely improved cost (all rotations passed).
        if gs > 1:
            group_rows = [["Group-select (min-cost passing)", "placements", fmt(buf.num_ik_valid)]]
            if buf.num_ik_valid > 0:
                group_rows.append([f"  ─ all {gs} rotations passed (cost win only)", "", fmt(self._cyclic_all_pass)])
                group_rows.append(["  ─ cyclic-salvaged (subset passed)", "", fmt(self._cyclic_salvaged)])
            sections.append(group_rows)

        # Final FPS downsample to target.
        sections.append([["Final FPS", "placements", f"{fmt(buf.num_ik_valid)} → {fmt(buf.num_selected)}"]])

        lines.extend(_render_table(["Stage", "Unit", "Count"], ["l", "l", "r"], sections))

        # --- Timings table ---
        if self._timings:
            # Sub-entries stored with a leading-space prefix so iteration
            # order keeps them grouped under their parent; exclude those
            # from the total to avoid double-counting.
            total = sum(dt for name, dt in self._timings.items() if not name.startswith(" "))
            timing_rows: list[list[str]] = []
            for name, dt in self._timings.items():
                stripped = name.lstrip(" ")
                leading = name[: len(name) - len(stripped)]
                # Prefer a human-readable label for this phase; otherwise
                # fall back to the last dotted segment (the indentation
                # already conveys the hierarchy, so prefixes are redundant).
                display = _TIMING_LABELS.get(stripped, stripped.rsplit(".", 1)[-1])
                label = leading + display
                pct = 100.0 * dt / total if total > 0 else 0.0
                timing_rows.append([label, f"{dt:.3f}s", f"{pct:.1f}%"])
            lines.append("")
            lines.append(
                f"Timings  (total {total:.3f}s across tracked phases;"
                " excludes one-time init, JIT/CUDA-graph compile, and host/device sync overhead)"
            )
            lines.extend(_render_table(["Phase", "Time", "%"], ["l", "r", "r"], [timing_rows]))

        return "\n".join(lines)
