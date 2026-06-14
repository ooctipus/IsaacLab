# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline factory Newton-IK reset-state pipeline.

Orchestrates the nut-first / sub-world-batched table build, the offline analog
of the terrain ``RetargetPipeline``:

    placements (W sub-worlds)  x  antipodal pairs (G per world)  ->  W*G problems
        -> one batched Newton IK over PER-FINGERTIP position targets
           (fingers pinned to half the pair separation -> mimic-consistent,
           approach direction left free and emerges from the solve)
        -> FK -> fingertip reachability + collision criteria (tag-gated)
        -> the surviving (arm config, nut pose, tag, aperture, family) rows.

The nut is never in the kinematic chain: its pose is sampler data, so the
accepted rows carry the COMMANDED placement exactly, and a nut without a grasp
(reach-toward states) is representable in the same batch. This replaces
``FactoryResetStateCommand._precollect_state_table``'s sim-in-the-loop fill
(``CollisionAnalyzer`` + ``RigidObjectHasher`` + in-sim DLS IK).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton.ik as ik
import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils

from ...utils.grid_downsample import grid_bucket_downsample
from .criteria import (
    edges_vs_posed_mesh_hit,
    points_min_sd,
    points_vs_body_meshes_min_sd,
    posed_collision_min_sd,
    posed_edges_vs_body_meshes_hit,
    posed_points,
    self_collision_min_sd,
)
from .model import FactoryIKModel
from .objectives import FactoryCollisionObjective, JointPinObjective
from .samplers import GraspPairSampler, NutPlacementSampler

if TYPE_CHECKING:
    from .cfg import FactoryIKPipelineCfg


@dataclass
class FactoryIKResult:
    """Accepted reset-state rows from :meth:`FactoryIKPipeline.build_table`.

    All poses are in the Franka base frame (the model origin); the production
    wiring offsets them into each env's frame when assembling reset-state rows.

    Attributes:
        joint_q: Solved joint coordinates [K, nq] -- arm coords 0..6 are
            ``panda_joint1..7`` [rad] in order, finger coords 7..8 are the two
            prismatic fingers [m], pinned to ``aperture / 2`` (mimic-consistent).
        nut_pose: Held-asset world pose [K, 7] (pos [m] + quat xyzw) -- the
            commanded placement, exact by construction.
        board_pose: Nistboard world pose [K, 7] (pos [m] + quat xyzw), sampled per
            sub-world (position + tilt).
        bolt_pose: Fixed-asset world pose [K, 7] (pos [m] + quat xyzw) -- rides the
            board at its keypoint offset.
        board_index: Board-configuration index [K] into the build's fixed library
            (the analog of locomotion's terrain ``tile_index``); spawn x target
            pairing is only valid WITHIN a configuration.
        pad_targets: World fingertip contact targets [K, 2, 3] [m], ordered
            (+jaw-y pad, -jaw-y pad) -- for visualization/debugging.
        aperture: Per-row gripper opening [K] [m] (``2 x`` finger coordinate,
            including the pad-penetration relief).
        tag: Placement tag index [K] into :attr:`tag_names`.
        tag_names: Tag name per index (assembly bands + ``on_table`` + ``in_air``).
        family: Grasp-family index [K] into :attr:`family_names` (surface
            region/mode combination of the contact pair).
        family_names: Grasp-family name per index.
    """

    joint_q: torch.Tensor
    nut_pose: torch.Tensor
    board_pose: torch.Tensor
    bolt_pose: torch.Tensor
    board_index: torch.Tensor
    pad_targets: torch.Tensor
    aperture: torch.Tensor
    tag: torch.Tensor
    tag_names: list[str]
    family: torch.Tensor
    family_names: list[str]


class FactoryIKPipeline:
    """Builds an offline factory reset-state table via batched fingertip-target Newton IK."""

    def __init__(self, cfg: FactoryIKPipelineCfg):
        self.cfg = cfg
        self.device = cfg.device
        torch.manual_seed(cfg.seed)
        self.model = FactoryIKModel(cfg)
        self.placement_sampler = NutPlacementSampler(self.model, cfg)
        self.grasp_sampler = GraspPairSampler(self.model, cfg)
        # reach rows reuse the placement tag indexing shifted by the base count
        # (the reaching_in_air slot is never produced -- in_air requires a grasp).
        self.tag_names = self.placement_sampler.tag_names + [f"reaching_{n}" for n in self.placement_sampler.tag_names]
        self.board_library: tuple[torch.Tensor, torch.Tensor] | None = None
        """Board/bolt configuration library ``(board_pose[B, 7], bolt_pose[B, 7])`` of the
        last :meth:`build_balanced_table` call; rows' ``board_index`` maps into it."""
        self._n_bands = len(cfg.placement.assembly_bands)
        m = self.model
        from .cfg import CollisionAvoidanceCfg, find_criterion

        _avoid = find_criterion(cfg.robot.solve.objectives, CollisionAvoidanceCfg)
        self._grip_probes_wp = wp.array(m.gripper_probes, dtype=wp.vec3, device=self.device)
        self._grip_probe_bodies_wp = wp.array(m.gripper_probe_bodies, dtype=wp.int32, device=self.device)
        # finger-only probe subset for the aperture-relief pass (pad penetration);
        # the complementary hand-only subset drives the nut-avoidance objective
        # (pads must touch the nut, the palm must not).
        fing = np.isin(m.gripper_probe_bodies, m.pad_bodies)
        self._fing_probes_wp = wp.array(m.gripper_probes[fing], dtype=wp.vec3, device=self.device)
        self._fing_probe_bodies_wp = wp.array(m.gripper_probe_bodies[fing], dtype=wp.int32, device=self.device)
        self._hand_probes_np = m.gripper_probes[~fing]
        self._hand_probe_bodies_np = m.gripper_probe_bodies[~fing]

        # sparse probe subsets for the avoidance OBJECTIVES (strided, preserving
        # per-body coverage since the sets are built grouped by body); the criteria
        # keep the full density and remain the correctness guarantee
        def _stride(arr_off: np.ndarray, arr_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_keep = min(_avoid.n_samples if _avoid is not None else 0, len(arr_off))
            sel = np.linspace(0, len(arr_off) - 1, n_keep).astype(np.int64)
            return arr_off[sel], arr_body[sel]

        self._obj_robot_probes_np, self._obj_robot_bodies_np = _stride(m.robot_probes, m.robot_probe_bodies)
        self._obj_hand_probes_np, self._obj_hand_bodies_np = _stride(self._hand_probes_np, self._hand_probe_bodies_np)
        # gripper collider-mesh targets for the symmetric nut-points-vs-gripper
        # queries: the pad-body subset feeds the relief pass, the full set the gate
        pad = np.isin(m.gripper_target_bodies, m.pad_bodies)
        self._pad_target_body_wp = wp.array(m.gripper_target_bodies[pad], dtype=wp.int32, device=self.device)
        self._pad_target_mesh_wp = wp.array(m.gripper_target_meshes[pad], dtype=wp.uint64, device=self.device)
        self._pad_target_tf_wp = wp.from_numpy(m.gripper_target_tf[pad], dtype=wp.transformf, device=self.device)
        self._grip_target_body_wp = wp.array(m.gripper_target_bodies, dtype=wp.int32, device=self.device)
        self._grip_target_mesh_wp = wp.array(m.gripper_target_meshes, dtype=wp.uint64, device=self.device)
        self._grip_target_tf_wp = wp.from_numpy(m.gripper_target_tf, dtype=wp.transformf, device=self.device)
        # all-link probe set (base excluded) for the robot-vs-static-obstacle criteria
        self._robot_probes_wp = wp.array(m.robot_probes, dtype=wp.vec3, device=self.device)
        self._robot_probe_bodies_wp = wp.array(m.robot_probe_bodies, dtype=wp.int32, device=self.device)
        # collider edges for the exact surface-crossing tests (thin-obstacle safe)
        self._robot_edge_p0_wp = wp.array(m.robot_edge_p0, dtype=wp.vec3, device=self.device)
        self._robot_edge_p1_wp = wp.array(m.robot_edge_p1, dtype=wp.vec3, device=self.device)
        self._robot_edge_bodies_wp = wp.array(m.robot_edge_bodies, dtype=wp.int32, device=self.device)
        self._board_edge_p0_wp = wp.array(m.board_edge_p0, dtype=wp.vec3, device=self.device)
        self._board_edge_p1_wp = wp.array(m.board_edge_p1, dtype=wp.vec3, device=self.device)
        self._held_probes_t = torch.tensor(m.held_probes, device=self.device)
        # Self-collision (robot link vs robot link).
        self._self_probes_wp = wp.array(m.self_probes, dtype=wp.vec3, device=self.device)
        self._self_probe_body_wp = wp.array(m.self_probe_bodies, dtype=wp.int32, device=self.device)
        self._self_target_body_wp = wp.array(m.self_target_bodies, dtype=wp.int32, device=self.device)
        self._self_target_mesh_wp = wp.array(m.self_target_meshes, dtype=wp.uint64, device=self.device)
        self._self_target_tf_wp = wp.from_numpy(m.self_target_tf, dtype=wp.transformf, device=self.device)
        self._self_adj_wp = wp.array(m.self_adjacency.flatten(), dtype=wp.uint8, device=self.device)
        self._n_bodies = m.body_count
        from .cfg import JointWithinLimitCfg

        crit_jlim = find_criterion(cfg.robot.criteria, JointWithinLimitCfg)
        self._joint_safe: tuple[torch.Tensor, torch.Tensor] | None = None
        if crit_jlim is not None:
            # arm coords inside the Newton joint limits shrunk by limit_ratio.
            # NOT intersected with the FK seed range (locomotion does; their solve
            # stays near its templates, factory wrist motion legitimately leaves
            # the seed neighborhood -- the intersection rejected 60% good rows)
            lo = wp.to_torch(m.model.joint_limit_lower)[m.arm_coords]
            hi = wp.to_torch(m.model.joint_limit_upper)[m.arm_coords]
            half_margin = 0.5 * (1.0 - crit_jlim.limit_ratio)
            span = hi - lo
            self._joint_safe = (lo + half_margin * span, hi - half_margin * span)
        from .cfg import CollisionCheckCfg

        crit_col = find_criterion(cfg.robot.criteria, CollisionCheckCfg)
        self._crit_robot = self._crit_grip = self._crit_held = self._crit_self = crit_col
        self._reject: dict[str, int] = {}
        self._reach_reject: dict[str, int] = {}
        self._relief_stats: dict[str, float] = {}
        self._balanced_stats: dict | None = None
        self._ik_iters_used = 0
        self._n_worlds = 0
        self._n_grasps = 0
        self._n_seeds = 1
        self._build_time: float = 0.0
        self._timings: dict[str, float] = {}

    @contextmanager
    def _timed(self, name: str):
        """Accumulate wall time (device-synchronized) for one build stage."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._timings[name] = self._timings.get(name, 0.0) + time.perf_counter() - t0

    def _joints_within_limit(self, joint_q: torch.Tensor) -> torch.Tensor:
        """Per-row arm-coords-inside-safe-interval mask (all-true when disabled)."""
        if self._joint_safe is None:
            return torch.ones(joint_q.shape[0], dtype=torch.bool, device=self.device)
        q = joint_q[:, self.model.arm_coords]
        return ((q >= self._joint_safe[0]) & (q <= self._joint_safe[1])).all(dim=-1)

    def _solve_ik(
        self,
        t_plus: torch.Tensor,
        t_minus: torch.Tensor,
        seed_arm: torch.Tensor,
        nut_pose: torch.Tensor,
        board_pose: torch.Tensor,
        bolt_pose: torch.Tensor,
        with_collision: bool = True,
        iterations: int | None = None,
    ) -> torch.Tensor:
        """Batched fingertip-target IK seeded from the pair-geometry templates."""
        from .cfg import (
            CollisionAvoidanceCfg,
            FingerPinObjectiveCfg,
            JointDefaultObjectiveCfg,
            JointLimitObjectiveCfg,
            find_criterion,
        )

        m, cfg = self.model, self.cfg
        solve_objs = cfg.robot.solve.objectives
        n = t_plus.shape[0]
        objs = [
            ik.IKObjectivePosition(
                link_index=m.pad_bodies[0],
                link_offset=wp.vec3(*m.pad_offsets[0].tolist()),
                target_positions=wp.from_torch(t_plus, dtype=wp.vec3),
                weight=1.0,
            ),
            ik.IKObjectivePosition(
                link_index=m.pad_bodies[1],
                link_offset=wp.vec3(*m.pad_offsets[1].tolist()),
                target_positions=wp.from_torch(t_minus, dtype=wp.vec3),
                weight=1.0,
            ),
            ik.IKObjectiveJointLimit(
                joint_limit_lower=m.model.joint_limit_lower,
                joint_limit_upper=m.model.joint_limit_upper,
                weight=find_criterion(solve_objs, JointLimitObjectiveCfg).weight,
            ),
            # fingers are data, not unknowns: pin both coords to half the pair
            # separation, enforcing the gripper mimic constraint structurally.
            JointPinObjective(
                coords=np.array(m.finger_coords),
                dofs=np.array(m.finger_dofs),
                targets=(0.5 * (t_plus - t_minus).norm(dim=-1)).unsqueeze(-1).expand(-1, len(m.finger_coords)),
                weight=find_criterion(solve_objs, FingerPinObjectiveCfg).weight,
            ),
        ]
        joint_default = find_criterion(solve_objs, JointDefaultObjectiveCfg)
        if joint_default is not None:
            # arm-stance pull: the same pin kernel as the fingers, constant targets
            # (locomotion's IKObjectiveJointDefault, arm coords only)
            objs.append(
                JointPinObjective(
                    coords=np.array(m.arm_coords),
                    dofs=np.array(m.arm_dofs),
                    targets=m.arm_stance.unsqueeze(0).expand(n, -1),
                    weight=joint_default.weight,
                )
            )
        avoidance = find_criterion(solve_objs, CollisionAvoidanceCfg)
        if avoidance is not None and with_collision:
            # robot (all links except the base) avoids the static obstacles and the
            # per-problem-posed board + bolt; the hand avoids the posed nut (pads
            # exempt -- they grasp it).
            ident = torch.zeros(n, 7, device=self.device)
            ident[:, 6] = 1.0
            avoid = [(mesh.id, ident) for mesh in self.model.static_obstacles.values()]
            avoid += [(m.board_mesh.id, board_pose), (m.fixed_mesh.id, bolt_pose)]
            for mesh_id, pose in avoid:
                objs.append(
                    FactoryCollisionObjective(
                        self._obj_robot_probes_np,
                        self._obj_robot_bodies_np,
                        mesh_id,
                        pose,
                        avoidance.weight,
                        avoidance.margin,
                        avoidance.max_dist,
                    )
                )
            objs.append(
                FactoryCollisionObjective(
                    self._obj_hand_probes_np,
                    self._obj_hand_bodies_np,
                    m.held_mesh.id,
                    nut_pose,
                    avoidance.weight,
                    avoidance.margin,
                    avoidance.max_dist,
                )
            )
        solver = ik.IKSolver(
            model=m.model,
            n_problems=n,
            objectives=objs,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        jq = torch.zeros(n, m.nq, device=self.device)
        jq[:, m.arm_coords] = seed_arm
        jq[:, m.finger_coords] = (0.5 * (t_plus - t_minus).norm(dim=-1)).unsqueeze(-1)
        jin, jout = wp.from_torch(jq.contiguous()), wp.from_torch(jq.clone().contiguous())
        prev = float("inf")
        self._ik_iters_used = 0
        for _ in range(0, iterations if iterations is not None else cfg.robot.solve.iterations, 10):
            solver.step(jin, jout, iterations=10)
            self._ik_iters_used += 10
            cost = float(wp.to_torch(solver.costs)[:n].mean())
            if abs(prev - cost) < cfg.robot.solve.convergence_threshold:
                break
            prev = cost
            jin = jout
        return wp.to_torch(jout)

    def _pads_world(self, body_q: torch.Tensor) -> torch.Tensor:
        """World pad contact points from FK, shape ``[N, 2, 3]``."""
        m, n = self.model, body_q.shape[0]
        return torch.stack(
            [
                math_utils.quat_apply(body_q[:, fb, 3:7], m.pad_offsets[k].expand(n, 3)) + body_q[:, fb, :3]
                for k, fb in enumerate(m.pad_bodies)
            ],
            dim=1,
        )

    def _robot_clear(
        self, body_q: torch.Tensor, board_pose: torch.Tensor, bolt_pose: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Robot (all links except the base) vs static obstacles + the posed group.

        Point queries handle clearance/containment; edge raycasts handle surface
        CROSSINGS exactly -- point probes alone miss the ~4 mm board slicing between
        them. Returns ``(all_ok[N], per-obstacle ok masks)``.
        """
        m, n = self.model, body_q.shape[0]
        ident = torch.zeros(n, 7, device=self.device)
        ident[:, 6] = 1.0
        ok_all = torch.ones(n, dtype=torch.bool, device=self.device)
        fail = {}
        obstacles = [(name, mesh.id, ident) for name, mesh in m.static_obstacles.items()]
        obstacles += [("board", m.board_mesh.id, board_pose), ("fixed_asset", m.fixed_mesh.id, bolt_pose)]
        for name, mesh_id, pose in obstacles:
            sd = posed_collision_min_sd(
                body_q,
                self._robot_probe_bodies_wp,
                self._robot_probes_wp,
                mesh_id,
                pose,
                self._crit_robot.query_radius,
                self.device,
            )
            cross = edges_vs_posed_mesh_hit(
                body_q,
                self._robot_edge_bodies_wp,
                self._robot_edge_p0_wp,
                self._robot_edge_p1_wp,
                mesh_id,
                pose,
                self.device,
            )
            ok = (sd >= -self._crit_robot.max_pen) & ~cross
            fail[name] = ok
            ok_all &= ok
        # reverse crossing direction for the thin board: its edges vs the link faces
        board_cross = posed_edges_vs_body_meshes_hit(
            self._board_edge_p0_wp,
            self._board_edge_p1_wp,
            board_pose,
            body_q,
            self._self_target_body_wp,
            self._self_target_mesh_wp,
            self._self_target_tf_wp,
            self.device,
        )
        fail["board"] = fail["board"] & ~board_cross
        return ok_all & ~board_cross, fail

    def _grip_nut_sd(
        self, body_q: torch.Tensor, nut_pose: torch.Tensor, nut_pts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric gripper<->held clearance: (gripper probes vs nut, nut probes vs gripper)."""
        m = self.model
        fwd = posed_collision_min_sd(
            body_q,
            self._grip_probe_bodies_wp,
            self._grip_probes_wp,
            m.held_mesh.id,
            nut_pose,
            self._crit_grip.query_radius,
            self.device,
        )
        rev = points_vs_body_meshes_min_sd(
            nut_pts,
            body_q,
            self._grip_target_body_wp,
            self._grip_target_mesh_wp,
            self._grip_target_tf_wp,
            self._crit_grip.query_radius,
            self.device,
        )
        return fwd, rev

    def _nut_obstacle_ok(
        self, nut_pts: torch.Tensor, tag: torch.Tensor, board_pose: torch.Tensor, bolt_pose: torch.Tensor
    ) -> torch.Tensor:
        """Held-asset points vs static obstacles + the posed group, with the
        intended-contact exemptions (bolt on assembly bands, board for on_table)."""
        m, n = self.model, nut_pts.shape[0]
        nut_ok = torch.ones(n, dtype=torch.bool, device=self.device)
        for mesh in m.static_obstacles.values():
            sd = points_min_sd(nut_pts, mesh.id, self._crit_held.query_radius, self.device)
            nut_ok &= sd >= -self._crit_held.max_pen
        p_count = nut_pts.shape[1]
        for mesh_id, pose, exempt in (
            (m.fixed_mesh.id, bolt_pose, tag < self._n_bands),
            (m.board_mesh.id, board_pose, tag == self._n_bands),
        ):
            local = math_utils.quat_apply_inverse(
                pose[:, 3:7].unsqueeze(1).expand(-1, p_count, 4).reshape(-1, 4),
                (nut_pts - pose[:, :3].unsqueeze(1)).reshape(-1, 3),
            ).view(n, p_count, 3)
            sd = points_min_sd(local, mesh_id, self._crit_held.query_radius, self.device)
            nut_ok &= (sd >= -self._crit_held.max_pen) | exempt
        return nut_ok

    def _self_clear(self, body_q: torch.Tensor) -> torch.Tensor:
        """Robot link-vs-link clearance mask [N] (adjacency-gated)."""
        sd = self_collision_min_sd(
            body_q,
            self._self_probe_body_wp,
            self._self_probes_wp,
            self._self_target_body_wp,
            self._self_target_mesh_wp,
            self._self_target_tf_wp,
            self._self_adj_wp,
            self._n_bodies,
            self._crit_self.query_radius,
            self.device,
        )
        return sd >= -self._crit_self.self_max_pen

    def build_table(
        self,
        num_placements: int | None = None,
        grasps_per_placement: int | None = None,
        board_library: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> FactoryIKResult:
        """Sample, solve, filter, and return the accepted reset-state rows.

        Args:
            num_placements: Explicit TOTAL nut placements; defaults to
                :attr:`PlacementSamplingCfg.placements_per_board` x the board
                library size.
            grasps_per_placement: Antipodal pairs per sub-world; defaults to
                :attr:`GraspSamplingCfg.grasps_per_placement`.

        Returns:
            The accepted (reachable + collision-clear) rows as a
            :class:`FactoryIKResult`. See :attr:`reject_stats` for the funnel counts.
        """
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cfg = self.cfg
        if num_placements is not None:
            w = num_placements
        else:
            n_lib = board_library[0].shape[0] if board_library is not None else cfg.board.num_boards
            w = cfg.placement.placements_per_board * n_lib
        g = grasps_per_placement if grasps_per_placement is not None else cfg.placement.grasp.grasps_per_placement

        self._timings = {}
        with self._timed("sample placements"):
            nut_pose, world_tag, board_pose, bolt_pose, board_index = self.placement_sampler.sample(
                w, board_library=board_library
            )
        self._n_worlds, self._n_grasps, self._n_seeds = w, g, self.cfg.placement.grasp.ik_seeds_per_grasp
        with self._timed("sample grasps + seed"):
            t_plus, t_minus, seed_arm, world_idx, _pair_sep, family = self.grasp_sampler.sample(nut_pose, g)
        tag = world_tag[world_idx]
        cand_nut_pose = nut_pose[world_idx]
        cand_board_pose = board_pose[world_idx].contiguous()
        cand_bolt_pose = bolt_pose[world_idx].contiguous()
        cand_board_index = board_index[world_idx]

        # two-phase solve: collision steering never improves REACH, so phase 1
        # solves the cheap reach-only layout for everyone, the unreachable are
        # culled, and only survivors pay for the avoidance-refine (warm-seeded).
        with self._timed("IK solve (reach)"):
            joint_q = self._solve_ik(
                t_plus, t_minus, seed_arm, cand_nut_pose, cand_board_pose, cand_bolt_pose, with_collision=False
            )
        n = joint_q.shape[0]
        m = self.model
        body_q = m.eval_fk(joint_q)
        pads = self._pads_world(body_q)
        reach = ((pads[:, 0] - t_plus).norm(dim=-1) < cfg.robot.solve.pos_tol) & (
            (pads[:, 1] - t_minus).norm(dim=-1) < cfg.robot.solve.pos_tol
        )
        from .cfg import CollisionAvoidanceCfg, find_criterion

        if find_criterion(cfg.robot.solve.objectives, CollisionAvoidanceCfg) is not None:
            alive = reach.nonzero(as_tuple=False).squeeze(-1)
            if alive.numel() > 0:
                with self._timed("IK solve (avoidance refine)"):
                    refined = self._solve_ik(
                        t_plus[alive].contiguous(),
                        t_minus[alive].contiguous(),
                        joint_q[alive][:, m.arm_coords].contiguous(),
                        cand_nut_pose[alive].contiguous(),
                        cand_board_pose[alive].contiguous(),
                        cand_bolt_pose[alive].contiguous(),
                        iterations=cfg.robot.solve.refine_iterations,
                    )
                joint_q[alive] = refined
        with self._timed("FK"):
            body_q = m.eval_fk(joint_q)

        # fingertip reachability: both pads on their contact points (post-refine,
        # pre-relief FK -- the refine can trade a little reach for clearance)
        pads = self._pads_world(body_q)
        reach = ((pads[:, 0] - t_plus).norm(dim=-1) < cfg.robot.solve.pos_tol) & (
            (pads[:, 1] - t_minus).norm(dim=-1) < cfg.robot.solve.pos_tol
        )

        # held-asset surface probes in world (the nut is not a model body)
        nut_pts = posed_points(self._held_probes_t, cand_nut_pose)

        # aperture relief: the pinned aperture assumes point contact at the pad
        # center, so on non-flat contact the pad corners dig into the nut. Measure
        # penetration from BOTH sides (finger probes into the nut, nut probes into
        # the pad colliders -- point-vs-mesh is one-directional), widen both fingers
        # by the worst depth (the pad normal IS the jaw axis, so the relief is 1:1),
        # rejecting rows that run out of finger travel.
        with self._timed("aperture relief queries"):
            fing_sd = posed_collision_min_sd(
                body_q,
                self._fing_probe_bodies_wp,
                self._fing_probes_wp,
                m.held_mesh.id,
                cand_nut_pose,
                self._crit_grip.query_radius,
                self.device,
            )
            nut_pad_sd = points_vs_body_meshes_min_sd(
                nut_pts,
                body_q,
                self._pad_target_body_wp,
                self._pad_target_mesh_wp,
                self._pad_target_tf_wp,
                self._crit_grip.query_radius,
                self.device,
            )
        relief = torch.maximum((-fing_sd).clamp(min=0.0), (-nut_pad_sd).clamp(min=0.0))
        # relief direction depends on the grasp mode: pinch pads clear the surface by
        # OPENING, expansion pads (inside the bore) by CLOSING -- opening would press
        # them harder into the walls.
        expand = family % 2 == 1
        relief = torch.where(expand, -relief, relief)
        finger_hi = float(wp.to_torch(m.model.joint_limit_upper)[m.finger_dofs[0]])
        finger_lo = float(wp.to_torch(m.model.joint_limit_lower)[m.finger_dofs[0]])
        new_finger = joint_q[:, m.finger_coords[0]] + relief
        relief_ok = (new_finger <= finger_hi) & (new_finger >= finger_lo)
        joint_q[:, m.finger_coords] += relief.unsqueeze(-1)
        self._relief_stats = {
            "relieved": int((relief.abs() > 1e-6).sum()),
            "max_mm": float(relief.abs().max()) * 1e3 if n > 0 else 0.0,
        }
        body_q = m.eval_fk(joint_q)

        with self._timed("criteria: robot vs obstacles"):
            grip_ok, grip_fail = self._robot_clear(body_q, cand_board_pose, cand_bolt_pose)

        # gripper vs the held asset (per-problem posed), post-relief and from BOTH
        # sides: pads should now kiss the surface, so only discretization slack remains
        with self._timed("criteria: gripper vs nut"):
            grip_nut_sd, nut_grip_sd = self._grip_nut_sd(body_q, cand_nut_pose, nut_pts)
        grip_nut_ok = relief_ok & (grip_nut_sd >= -self._crit_grip.max_pen) & (nut_grip_sd >= -self._crit_grip.max_pen)

        # nut vs the static obstacles and the posed assembly group (world points;
        # the nut is not a model body). Intended-contact exemptions: the nut
        # touches the bolt on the assembly bands (owned by the assembly profile)
        # and rests on the board for on_table.
        with self._timed("criteria: nut vs obstacles"):
            nut_ok = self._nut_obstacle_ok(nut_pts, tag, cand_board_pose, cand_bolt_pose)

        # self-collision: reject configs whose links fold into one another
        with self._timed("criteria: self-collision"):
            self_ok = self._self_clear(body_q)

        jl_ok = self._joints_within_limit(joint_q)
        valid = reach & jl_ok & grip_ok & grip_nut_ok & nut_ok & self_ok
        cum = reach
        self._reject = {"unreachable": int((~reach).sum())}
        self._reject["joint_limit"] = int((cum & ~jl_ok).sum())
        cum = cum & jl_ok
        for name, ok in grip_fail.items():
            self._reject[f"robot_vs_{name}"] = int((cum & ~ok).sum())
            cum = cum & ok
        self._reject["gripper_vs_nut"] = int((cum & ~grip_nut_ok).sum())
        cum = cum & grip_nut_ok
        self._reject["nut_collision"] = int((cum & ~nut_ok).sum())
        cum = cum & nut_ok
        self._reject["self_collision"] = int((cum & ~self_ok).sum())
        self._reject["ok"] = int(valid.sum())

        idx = valid.nonzero(as_tuple=False).squeeze(-1)

        # ---- reach/standoff rows: back accepted grasps off along their achieved
        # approach axis (EE-z) and re-solve seeded from the grasp solution. Tag-
        # gated: a floating nut with a non-grasping gripper is not a physical
        # state, so in_air parents produce no reach rows. Reach rows must CLEAR
        # the nut (no contact intended) -- no relief, clearance from both sides.
        out = {
            "joint_q": [joint_q[idx]],
            "nut": [cand_nut_pose[idx]],
            "board": [cand_board_pose[idx]],
            "bolt": [cand_bolt_pose[idx]],
            "pads": [torch.stack([t_plus[idx], t_minus[idx]], dim=1)],
            "aperture": [2.0 * joint_q[idx, m.finger_coords[0]]],
            "tag": [tag[idx]],
            "family": [family[idx]],
            "board_id": [cand_board_index[idx]],
        }
        self._reach_reject = {}
        n_base = len(self.placement_sampler.tag_names)
        src = idx[tag[idx] != self._n_bands + 1]  # exclude in_air parents
        if cfg.robot.reach is not None and cfg.robot.reach.per_grasp > 0 and src.numel() > 0:
            src = src.repeat_interleave(cfg.robot.reach.per_grasp)
            nr = src.shape[0]
            ee_z = math_utils.quat_apply(
                body_q[src, m.ee_body, 3:7], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(nr, 3)
            )
            standoff = torch.empty(nr, 1, device=self.device).uniform_(*cfg.robot.reach.standoff_range)
            rt_plus = (t_plus[src] - standoff * ee_z).contiguous()
            rt_minus = (t_minus[src] - standoff * ee_z).contiguous()
            r_nut = cand_nut_pose[src].contiguous()
            r_board = cand_board_pose[src].contiguous()
            r_bolt = cand_bolt_pose[src].contiguous()
            with self._timed("reach: IK solve"):
                r_jq = self._solve_ik(
                    rt_plus, rt_minus, joint_q[src][:, m.arm_coords].contiguous(), r_nut, r_board, r_bolt
                )
            r_bq = m.eval_fk(r_jq)
            r_pads = self._pads_world(r_bq)
            r_reach = ((r_pads[:, 0] - rt_plus).norm(dim=-1) < cfg.robot.solve.pos_tol) & (
                (r_pads[:, 1] - rt_minus).norm(dim=-1) < cfg.robot.solve.pos_tol
            )
            with self._timed("reach: criteria"):
                r_robot_ok, _ = self._robot_clear(r_bq, r_board, r_bolt)
                fwd, rev = self._grip_nut_sd(r_bq, r_nut, nut_pts[src])
                r_clear = (fwd >= cfg.robot.reach.clearance) & (rev >= cfg.robot.reach.clearance)
                r_self = self._self_clear(r_bq)
            r_jl = self._joints_within_limit(r_jq)
            r_valid = r_reach & r_jl & r_robot_ok & r_clear & r_self
            self._reach_reject = {
                "unreachable": int((~r_reach).sum()),
                "joint_limit": int((r_reach & ~r_jl).sum()),
                "robot_collision": int((r_reach & ~r_robot_ok).sum()),
                "nut_contact": int((r_reach & r_robot_ok & ~r_clear).sum()),
                "self_collision": int((r_reach & r_robot_ok & r_clear & ~r_self).sum()),
                "ok": int(r_valid.sum()),
            }
            ridx = r_valid.nonzero(as_tuple=False).squeeze(-1)
            out["joint_q"].append(r_jq[ridx])
            out["nut"].append(r_nut[ridx])
            out["board"].append(r_board[ridx])
            out["bolt"].append(r_bolt[ridx])
            out["pads"].append(torch.stack([rt_plus[ridx], rt_minus[ridx]], dim=1))
            out["aperture"].append(2.0 * r_jq[ridx, m.finger_coords[0]])
            out["tag"].append(n_base + tag[src][ridx])
            out["family"].append(family[src][ridx])
            out["board_id"].append(cand_board_index[src][ridx])

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._build_time = time.perf_counter() - t0
        return FactoryIKResult(
            joint_q=torch.cat(out["joint_q"]).contiguous(),
            nut_pose=torch.cat(out["nut"]).contiguous(),
            board_pose=torch.cat(out["board"]).contiguous(),
            bolt_pose=torch.cat(out["bolt"]).contiguous(),
            pad_targets=torch.cat(out["pads"]).contiguous(),
            aperture=torch.cat(out["aperture"]).contiguous(),
            tag=torch.cat(out["tag"]).contiguous(),
            tag_names=self.tag_names,
            family=torch.cat(out["family"]).contiguous(),
            family_names=self.grasp_sampler.family_names,
            board_index=torch.cat(out["board_id"]).contiguous(),
        )

    def build_balanced_table(self, table_size: int) -> FactoryIKResult:
        """One-shot oversample -> reject -> downsample at the BOARD level.

        Samples ``library_oversample x num_boards`` candidate configurations and
        runs ONE build round across all of them; a candidate's feasibility is
        proven by its own survivor count (>= rows_per_board qualifies), the
        library keeps ``num_boards`` qualified candidates by pose-space FPS
        (spread over position/tilt, NOT the easiest), and each kept board
        FPS-downsamples its survivors to ``rows_per_board`` rows in
        (nut-rel-bolt position, approach direction, tag) space -- the terrain
        per-cell selection, so every configuration carries a diverse state mix.

        Args:
            table_size: Target total rows; ``table_size / num_boards`` rows are
                kept per board, and every kept board is full by construction.

        Raises:
            RuntimeError: If fewer than ``num_boards`` candidates qualify -- raise
                :attr:`PlacementSamplingCfg.placements_per_board` (more supply
                per candidate), :attr:`BoardLibraryCfg.library_oversample` (more
                candidates), or relax :attr:`BoardLibraryCfg.pose_range`.
        """
        cfg = self.cfg
        rows_per_board = max(1, table_size // cfg.board.num_boards)
        n_cand = max(cfg.board.num_boards, int(round(cfg.board.num_boards * cfg.board.library_oversample)))

        # candidate configuration library: geometrically clear + pose-spread; the
        # build round below is the feasibility test. Kept on the pipeline after
        # selection -- rows' board_index maps into it.
        board_library = self.placement_sampler._sample_board(n_cand)
        r = self.build_table(board_library=board_library)
        if self.device.startswith("cuda"):
            # release the torch cache: inside a running sim the torch and warp
            # allocators compete for the same device, and torch hoarding freed
            # solver blocks starves warp into OOM.
            torch.cuda.empty_cache()

        counts = torch.bincount(r.board_index, minlength=n_cand)
        qualified = (counts >= rows_per_board).nonzero(as_tuple=False).squeeze(-1)
        if qualified.numel() < cfg.board.num_boards:
            raise RuntimeError(
                f"only {int(qualified.numel())} of {n_cand} candidate boards supplied >="
                f" {rows_per_board} rows (need {cfg.board.num_boards}) -- raise"
                " placement.placements_per_board or board.library_oversample, or relax board.pose_range"
            )

        # keep num_boards spread over pose space (the same (pos, 0.1*rpy) feature
        # the candidate sampler spreads in) -- selecting by supply would bias the
        # library toward easy mild poses and lose the hard-but-feasible worlds
        q_pose = board_library[0][qualified]
        rpy = torch.stack(math_utils.euler_xyz_from_quat(q_pose[:, 3:7]), dim=-1)
        sel = qualified[grid_bucket_downsample(torch.cat([q_pose[:, :3], 0.1 * rpy], dim=-1), cfg.board.num_boards)]

        remap = torch.full((n_cand,), -1, dtype=torch.long, device=self.device)
        remap[sel] = torch.arange(sel.numel(), device=self.device)
        self.board_library = (board_library[0][sel].contiguous(), board_library[1][sel].contiguous())
        row_board = remap[r.board_index]
        rows = (row_board >= 0).nonzero(as_tuple=False).squeeze(-1)

        joint_q = r.joint_q[rows]
        nut_pose = r.nut_pose[rows]
        board_pose = r.board_pose[rows]
        bolt_pose = r.bolt_pose[rows]
        pad_targets = r.pad_targets[rows]
        aperture = r.aperture[rows]
        tag = r.tag[rows]
        family = r.family[rows]
        board_index = row_board[rows]

        # final diversity selection PER BOARD (terrain per-cell selection): each
        # board FPS-downsamples its own survivors to rows_per_board in (nut
        # position in the bolt frame, approach direction, tag) space. The tag
        # one-hot keeps state KINDS apart -- a reach row shares its source grasp's
        # nut pose and would otherwise dedupe against it -- so within a board the
        # picks spread across tags first, then within tags.
        m = self.model
        sel = cfg.row_selection
        body_q = m.eval_fk(joint_q)
        ee_z = math_utils.quat_apply(
            body_q[:, m.ee_body, 3:7], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(joint_q.shape[0], 3)
        )
        rel = math_utils.quat_apply_inverse(bolt_pose[:, 3:7], nut_pose[:, :3] - bolt_pose[:, :3])
        tag_hot = sel.tag_weight * torch.nn.functional.one_hot(tag, num_classes=len(self.tag_names)).float()
        feats = torch.cat([sel.nut_weight * rel, sel.approach_weight * ee_z, tag_hot], dim=-1)
        keep = []
        for b in range(cfg.board.num_boards):
            bi = (board_index == b).nonzero(as_tuple=False).squeeze(-1)
            if bi.numel() > rows_per_board:
                bi = bi[grid_bucket_downsample(feats[bi], rows_per_board)]
            keep.append(bi)
        keep = torch.cat(keep)

        kept_tag = tag[keep]
        # per-cell content: how many DISTINCT nut placements each board carries
        # (rows from the same placement -- different grasps/arm branches -- share
        # their nut pose, so row count alone overstates state variety)
        kept_board = board_index[keep]
        kept_nut = nut_pose[keep]
        nut_counts = []
        for b in range(cfg.board.num_boards):
            bi = (kept_board == b).nonzero(as_tuple=False).squeeze(-1)
            if bi.numel():
                nut_counts.append(int(torch.unique((kept_nut[bi] * 1e4).round(), dim=0).shape[0]))
        nut_counts_t = torch.tensor(nut_counts, dtype=torch.float)
        self._balanced_stats = {
            "candidates": n_cand,
            "qualified": int(qualified.numel()),
            "kept_boards": cfg.board.num_boards,
            "accumulated": int(r.tag.shape[0]),
            "kept": int(keep.shape[0]),
            "per_tag": {self.tag_names[t]: int((kept_tag == t).sum()) for t in torch.unique(kept_tag).tolist()},
            "nut_per_board": (
                int(nut_counts_t.min()),
                float(nut_counts_t.median()),
                int(nut_counts_t.max()),
            ),
        }
        return FactoryIKResult(
            joint_q=joint_q[keep].contiguous(),
            nut_pose=nut_pose[keep].contiguous(),
            board_pose=board_pose[keep].contiguous(),
            bolt_pose=bolt_pose[keep].contiguous(),
            pad_targets=pad_targets[keep].contiguous(),
            aperture=aperture[keep].contiguous(),
            tag=tag[keep].contiguous(),
            tag_names=self.tag_names,
            family=family[keep].contiguous(),
            family_names=self.grasp_sampler.family_names,
            board_index=board_index[keep].contiguous(),
        )

    @property
    def reject_stats(self) -> dict[str, int]:
        """Per-bucket candidate counts from the last :meth:`build_table` call.

        Waterfall buckets (``unreachable``, ``gripper_vs_fixed_asset``,
        ``gripper_vs_table``, ``gripper_vs_nut``, ``nut_collision``,
        ``self_collision``, ``ok``) partition the attempted candidates.
        """
        return dict(self._reject)

    @property
    def rejection_summary(self) -> str:
        """Stage-funnel table of the last build (terrain ``rejection_summary`` style).

        Sections: grasp sampler (surface -> antipodal pairs -> retained + FK seed
        library), placement sampler (board poses -> sub-worlds -> candidates), IK
        solve, aperture relief, the criteria waterfall, reach rows, and -- after
        :meth:`build_balanced_table` -- the per-tag quota fill + final FPS.
        """

        def fmt(n: int) -> str:
            return f"{n:,}"

        sections: list[list[list[str]]] = []
        gs = getattr(self.grasp_sampler, "stats", {})
        ps = getattr(self.placement_sampler, "board_stats", {})

        sampler_rows = [["Grasp sampler", "points", fmt(gs.get("surface", 0))]]
        n_pairs = gs.get("pinch", 0) + gs.get("expand", 0)
        sampler_rows.append(["  antipodal pairs", "pairs", fmt(n_pairs)])
        sampler_rows.append(["  ─ pinch", "", fmt(gs.get("pinch", 0))])
        sampler_rows.append(["  ─ expand", "", fmt(gs.get("expand", 0))])
        sampler_rows.append(["  retained (FPS)", "pairs", fmt(gs.get("retained", 0))])
        sampler_rows.append(
            [
                "  FK seed library",
                "templates",
                f"{fmt(gs.get('templates', 0))} / {fmt(gs.get('fk_samples', 0))} FK samples",
            ]
        )
        sections.append(sampler_rows)

        board_attempted = ps.get("attempted", 0)
        board_clear = ps.get("clear", 0)
        clear_pct = 100.0 * board_clear / board_attempted if board_attempted else 0.0
        place_rows = [
            [
                "Placement sampler",
                "board poses",
                f"{fmt(board_clear)} clear ({clear_pct:.0f}% of {fmt(board_attempted)})",
            ]
        ]
        place_rows.append(
            [
                "  sub-worlds × grasps × seeds",
                "candidates",
                f"{fmt(self._n_worlds)} × {self._n_grasps} × {self._n_seeds} ="
                f" {fmt(self._n_worlds * self._n_grasps * self._n_seeds)}",
            ]
        )
        sections.append(place_rows)

        sections.append(
            [
                [
                    f"IK solve ({self._ik_iters_used} iters)",
                    "candidates",
                    fmt(self._n_worlds * self._n_grasps * self._n_seeds),
                ]
            ]
        )
        if self._relief_stats:
            relief_cell = (
                f"{fmt(int(self._relief_stats['relieved']))} relieved (max {self._relief_stats['max_mm']:.1f} mm)"
            )
            sections.append([["Aperture relief", "candidates", relief_cell]])

        r = dict(self._reject)
        ok = r.pop("ok", 0)
        attempted = ok + sum(r.values())
        crit_rows = [["Criteria (waterfall)", "candidates", fmt(attempted)]]
        for name, count in r.items():
            crit_rows.append([f"  ─ {name}", "", f"−{fmt(count)}"])
        pct = 100.0 * ok / attempted if attempted else 0.0
        crit_rows.append(["  grasp rows", "rows", f"{fmt(ok)} ({pct:.1f}%)"])
        sections.append(crit_rows)

        if self._reach_reject:
            rr = dict(self._reach_reject)
            r_ok = rr.pop("ok", 0)
            r_attempted = r_ok + sum(rr.values())
            reach_rows = [["Reach rows (standoff re-solve)", "candidates", fmt(r_attempted)]]
            for name, count in rr.items():
                if count:
                    reach_rows.append([f"  ─ {name}", "", f"−{fmt(count)}"])
            reach_rows.append(["  reach rows", "rows", fmt(r_ok)])
            sections.append(reach_rows)

        if self._balanced_stats is not None:
            b = self._balanced_stats
            bal_rows = [
                [
                    "Board library (one round proves feasibility)",
                    "boards",
                    f"{fmt(b['candidates'])} candidates -> {fmt(b['qualified'])} qualified ->"
                    f" {fmt(b['kept_boards'])} kept (pose FPS)",
                ]
            ]
            bal_rows.append(["  survivor pool", "rows", fmt(b["accumulated"])])
            bal_rows.append(["  final table (per-board FPS)", "rows", fmt(b["kept"])])
            lo, med, hi = b["nut_per_board"]
            bal_rows.append(
                ["  unique nut placements / board", "placements", f"min {lo} / median {med:.0f} / max {hi}"]
            )
            for name, count in b["per_tag"].items():
                bal_rows.append([f"  ─ {name}", "", fmt(count)])
            sections.append(bal_rows)

        header = ["Stage", "unit", "count"]
        aligns = ["l", "l", "r"]
        all_rows = [header] + [row for sec in sections for row in sec]
        widths = [max(len(row[i]) for row in all_rows) for i in range(3)]

        def border(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * (w + 2) for w in widths) + right

        def fmt_row(cells: list[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                text = cell.rjust(widths[i]) if aligns[i] == "r" else cell.ljust(widths[i])
                parts.append(f" {text} ")
            return "│" + "│".join(parts) + "│"

        if self._timings:
            total = sum(self._timings.values())
            timing_rows = [["Timing (last round)", "seconds", f"{total:.2f} measured"]]
            for name, t in sorted(self._timings.items(), key=lambda kv: -kv[1]):
                timing_rows.append([f"  ─ {name}", "", f"{t:.2f}"])
            sections.append(timing_rows)

        lines = [
            f"Factory IK pipeline ({'balanced' if self._balanced_stats else 'single'} build, last round shown,"
            f" {self._build_time:.2f}s/round)",
            border("┌", "┬", "┐"),
            fmt_row(header),
            border("├", "┼", "┤"),
        ]
        for i, sec in enumerate(sections):
            if i > 0:
                lines.append(border("├", "┼", "┤"))
            lines.extend(fmt_row(row) for row in sec)
        lines.append(border("└", "┴", "┘"))
        return "\n".join(lines)
