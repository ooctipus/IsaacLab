# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build Factory reset-state rows from declared task families.

Each family generates held/fixed poses and grasp evidence, solves one flat IK
objective tuple, applies independent criteria, and selects an exact board quota.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton.ik as ik
import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.string import string_to_callable

from ...kinematics import IKExecutionStatistics, NewtonKinematics, NewtonKinematicsBuildCfg, execute_ik_batches
from ...kinematics.ik_objectives.cfg import (
    IKObjectiveJointPinCfg,
    IKObjectivePositionCfg,
)
from ...kinematics.ik_objectives.context import (
    IKJointPinObjectiveBuildContext,
    IKObjectiveBuild,
    IKObjectiveBuildContext,
    IKPositionObjectiveBuildContext,
)
from ...mdp.commands.state_command.task_family import TaskTableRng, execute_task_family
from ...utils.grid_downsample import grid_bucket_downsample
from .criteria import (
    edges_vs_posed_mesh_hit,
    measure_grasp_targets,
    posed_collision_min_sd,
    posed_edges_vs_body_meshes_hit,
    posed_points_min_sd,
    posed_points_vs_body_meshes_min_sd,
    posed_points_vs_posed_mesh_min_sd,
    self_collision_min_sd,
)
from .model import FactoryGeometry, factory_default_joint_q, factory_eval_fk
from .samplers import GraspPairSampler, HeldAssetPlacementSampler

if TYPE_CHECKING:
    from .cfg import FactoryFamilyCfg, FactoryGeometryCfg


@dataclass
class FactoryIKResult:
    """Accepted reset-state rows from the declared Factory task families.

    All poses are in the Franka base frame (the model origin); the production
    wiring offsets them into each env's frame when assembling reset-state rows.

    Attributes:
        joint_q: Solved joint coordinates [K, nq] -- arm coords 0..6 are
            ``panda_joint1..7`` [rad] in order, finger coords 7..8 are the two
            prismatic fingers [m], pinned to ``aperture / 2`` (mimic-consistent).
        held_pose: Held-asset world pose [K, 7] (pos [m] + quat xyzw) -- the
            commanded placement, exact by construction.
        board_pose: Nistboard world pose [K, 7] (pos [m] + quat xyzw), sampled per
            sub-world (position + tilt).
        board_asset_poses: Board-attached entity poses keyed by reset-asset name.
        board_index: Board-configuration index [K] into the build's fixed library
            (the analog of locomotion's terrain ``tile_index``); spawn x target
            pairing is only valid WITHIN a configuration.
        pad_targets: World fingertip contact targets [K, 2, 3] [m], ordered
            (+jaw-y pad, -jaw-y pad) -- for visualization/debugging.
        aperture: Per-row gripper opening [K] [m] (``2 x`` finger coordinate).
        tag: Placement tag index [K] into :attr:`tag_names`.
        tag_names: Tag name per index (assembly bands + ``on_table`` + ``in_air``).
        family: Grasp-family index [K] into :attr:`family_names` (surface
            region/mode combination of the contact pair).
        family_names: Grasp-family name per index.
    """

    joint_q: torch.Tensor
    held_pose: torch.Tensor
    board_pose: torch.Tensor
    board_asset_poses: dict[str, torch.Tensor]
    board_index: torch.Tensor
    pad_targets: torch.Tensor
    aperture: torch.Tensor
    tag: torch.Tensor
    tag_names: list[str]
    family: torch.Tensor
    family_names: list[str]
    task_family: torch.Tensor
    task_family_names: tuple[str, ...]
    quality_names: tuple[str, ...]
    quality: torch.Tensor
    is_grasped: torch.Tensor


@dataclass
class FactoryFamilyCandidates:
    """Domain-owned tensors flowing through one declared Factory family."""

    kinematics: NewtonKinematics
    geometry: FactoryGeometry
    placement_sampler: HeldAssetPlacementSampler
    grasp_samplers: dict[str, GraspPairSampler]
    tag_names: list[str]
    tag_indices: dict[str, int]
    board_library: tuple[torch.Tensor, dict[str, torch.Tensor]]
    num_placements: int
    held_pose: torch.Tensor | None = None
    tag: torch.Tensor | None = None
    board_pose: torch.Tensor | None = None
    board_asset_poses: dict[str, torch.Tensor] | None = None
    board_index: torch.Tensor | None = None
    grasp_sampler: GraspPairSampler | None = None
    t_plus: torch.Tensor | None = None
    t_minus: torch.Tensor | None = None
    seed_arm: torch.Tensor | None = None
    grasp_family: torch.Tensor | None = None
    finger_targets: torch.Tensor | None = None
    joint_q: torch.Tensor | None = None
    ee_approach: torch.Tensor | None = None
    target_error_m: torch.Tensor | None = None
    aperture: torch.Tensor | None = None
    is_grasped: torch.Tensor | None = None
    allow_fixed_contact: torch.Tensor | None = None
    allow_board_contact: torch.Tensor | None = None
    target_is_grasped: torch.Tensor | None = None
    gripper_clearance: float = 0.0
    solve_statistics: IKExecutionStatistics | None = None


@dataclass(slots=True)
class _FactoryIKWorkspace:
    """One capacity-sized solver and binding workspace reused by every batch."""

    solver: ik.IKSolver
    target_builds: tuple[IKObjectiveBuild, ...]
    finger_targets: torch.Tensor
    joint_q: torch.Tensor
    capacity: int


@dataclass(slots=True)
class _FactoryFKWorkspace:
    """One solver-capacity FK workspace reused by measurements and criteria."""

    joint_qd: wp.array
    body_q: wp.array
    body_qd: wp.array
    capacity: int


@dataclass(slots=True)
class _FactorySelectedFamily:
    """Only selected rows and board counts retained after one family finishes."""

    joint_q: torch.Tensor
    held_pose: torch.Tensor
    board_pose: torch.Tensor
    board_asset_poses: dict[str, torch.Tensor]
    pad_targets: torch.Tensor
    aperture: torch.Tensor
    tag: torch.Tensor
    grasp_family: torch.Tensor
    board_index: torch.Tensor
    target_error_m: torch.Tensor
    is_grasped: torch.Tensor
    board_counts: torch.Tensor
    grasp_family_names: list[str]
    generated: int
    accepted: int


def _factory_compact_selected(execution, candidate_board_count: int) -> _FactorySelectedFamily:
    """Gather selected rows so raw solve, FK, criterion, and selection storage can die."""
    candidates = execution.candidates
    selected = execution.selected_indices
    selected_boards = candidates.board_index[selected]
    generated = candidates.joint_q.shape[0]
    accepted = generated if execution.accepted_mask is None else int(execution.accepted_mask.sum())
    return _FactorySelectedFamily(
        joint_q=candidates.joint_q[selected].contiguous(),
        held_pose=candidates.held_pose[selected].contiguous(),
        board_pose=candidates.board_pose[selected].contiguous(),
        board_asset_poses={name: poses[selected].contiguous() for name, poses in candidates.board_asset_poses.items()},
        pad_targets=torch.stack((candidates.t_plus[selected], candidates.t_minus[selected]), dim=1).contiguous(),
        aperture=candidates.aperture[selected].contiguous(),
        tag=candidates.tag[selected].contiguous(),
        grasp_family=candidates.grasp_family[selected].contiguous(),
        board_index=selected_boards.contiguous(),
        target_error_m=candidates.target_error_m[selected].contiguous(),
        is_grasped=candidates.is_grasped[selected].contiguous(),
        board_counts=torch.bincount(selected_boards, minlength=candidate_board_count),
        grasp_family_names=list(candidates.grasp_sampler.family_names),
        generated=generated,
        accepted=accepted,
    )


class FactoryTaskTableBuilder:
    """Own the shared build resources while executing declared Factory families."""

    def __init__(
        self,
        kinematics_cfg: NewtonKinematicsBuildCfg,
        cfg: FactoryGeometryCfg,
        scene_cfg,
        device: str,
        families: tuple[FactoryFamilyCfg, ...],
        rng: TaskTableRng,
    ) -> None:
        from .cfg import CollisionCheckCfg

        self.cfg = cfg
        self.device = device
        collision_cfgs = tuple(
            criterion
            for family in families
            for criterion in family.criteria
            if isinstance(criterion, CollisionCheckCfg)
        )
        if len(collision_cfgs) != len(families):
            raise ValueError("Every Factory family must declare exactly one CollisionCheckCfg criterion.")
        robot_cfg = getattr(scene_cfg, cfg.robot.asset_cfg.name)
        self.kinematics = NewtonKinematics.from_articulation(kinematics_cfg, robot_cfg, device)
        self.geometry = FactoryGeometry(
            self.kinematics,
            cfg,
            scene_cfg,
            max(value.n_samples for value in collision_cfgs),
            rng.numpy,
        )
        self.grasp_samplers: dict[str, GraspPairSampler] = {}
        self.tag_names: list[str] = []
        self._tag_index: dict[str, int] = {}
        m = self.geometry
        m.pad_bodies_wp = wp.array(m.pad_bodies, dtype=wp.int32, device=self.device)
        m.pad_offsets_wp = wp.from_torch(m.pad_offsets, dtype=wp.vec3)
        m.grip_probes_wp = wp.array(m.gripper_probes, dtype=wp.vec3, device=self.device)
        m.grip_probe_bodies_wp = wp.array(m.gripper_probe_bodies, dtype=wp.int32, device=self.device)
        # Full gripper collider targets certify the independent symmetric contact gate.
        m.grip_target_body_wp = wp.array(m.gripper_target_bodies, dtype=wp.int32, device=self.device)
        m.grip_target_mesh_wp = wp.array(m.gripper_target_meshes, dtype=wp.uint64, device=self.device)
        m.grip_target_tf_wp = wp.from_numpy(m.gripper_target_tf, dtype=wp.transformf, device=self.device)
        # all-link probe set (base excluded) for the robot-vs-static-obstacle criteria
        m.robot_probes_wp = wp.array(m.robot_probes, dtype=wp.vec3, device=self.device)
        m.robot_probe_bodies_wp = wp.array(m.robot_probe_bodies, dtype=wp.int32, device=self.device)
        # collider edges for the exact surface-crossing tests (thin-obstacle safe)
        m.robot_edge_p0_wp = wp.array(m.robot_edge_p0, dtype=wp.vec3, device=self.device)
        m.robot_edge_p1_wp = wp.array(m.robot_edge_p1, dtype=wp.vec3, device=self.device)
        m.robot_edge_bodies_wp = wp.array(m.robot_edge_bodies, dtype=wp.int32, device=self.device)
        m.board_edge_p0_wp = wp.array(m.board_edge_p0, dtype=wp.vec3, device=self.device)
        m.board_edge_p1_wp = wp.array(m.board_edge_p1, dtype=wp.vec3, device=self.device)
        m.held_probes_t = torch.tensor(m.held_probes, device=self.device)
        m.robot_full_probes_wp = wp.array(m.robot_full_probes, dtype=wp.vec3, device=self.device)
        m.robot_full_probe_bodies_wp = wp.array(m.robot_full_probe_bodies, dtype=wp.int32, device=self.device)
        m.robot_full_edge_p0_wp = wp.array(m.robot_full_edge_p0, dtype=wp.vec3, device=self.device)
        m.robot_full_edge_p1_wp = wp.array(m.robot_full_edge_p1, dtype=wp.vec3, device=self.device)
        m.robot_full_edge_bodies_wp = wp.array(m.robot_full_edge_bodies, dtype=wp.int32, device=self.device)
        m.robot_target_body_wp = wp.array(m.robot_target_bodies, dtype=wp.int32, device=self.device)
        m.robot_target_mesh_wp = wp.array(m.robot_target_meshes, dtype=wp.uint64, device=self.device)
        m.robot_target_tf_wp = wp.from_numpy(m.robot_target_tf, dtype=wp.transformf, device=self.device)
        m.self_adjacency_wp = {}
        default_joint_q = factory_default_joint_q(m).unsqueeze(0).contiguous()
        default_robot_body_q = factory_eval_fk(self.kinematics, default_joint_q).contiguous()
        self.placement_sampler = HeldAssetPlacementSampler(m, cfg, rng.torch, default_robot_body_q)

    def build_family_table(
        self, rows_per_board: int, families: tuple[FactoryFamilyCfg, ...], rng: TaskTableRng
    ) -> FactoryIKResult:
        """Build exact family quotas per board and retain only fully qualified boards."""
        from ..mdp.reset_state_task_table import factory_family_quotas
        from .cfg import FactoryGraspTargetGenerateCfg, FactoryRobotSeedGenerateCfg

        quotas = factory_family_quotas(rows_per_board, families)
        board_count = self.cfg.board.num_boards
        candidate_board_count = max(board_count, round(board_count * self.cfg.board.library_oversample))
        board_library = self.placement_sampler._sample_board(candidate_board_count)
        selected_families: list[_FactorySelectedFamily | None] = []
        for family, quota in zip(families, quotas, strict=True):
            if quota == 0:
                selected_families.append(None)
                continue
            grasp_cfg = next(term for term in family.generate if isinstance(term, FactoryGraspTargetGenerateCfg))
            seed_cfg = next(term for term in family.generate if isinstance(term, FactoryRobotSeedGenerateCfg))
            raw_per_board = math.ceil(quota * family.candidate_oversample)
            placements_per_board = max(
                1,
                math.ceil(raw_per_board / (grasp_cfg.grasps_per_placement * seed_cfg.ik_seeds_per_grasp)),
            )
            initial = FactoryFamilyCandidates(
                kinematics=self.kinematics,
                geometry=self.geometry,
                placement_sampler=self.placement_sampler,
                grasp_samplers=self.grasp_samplers,
                tag_names=self.tag_names,
                tag_indices=self._tag_index,
                board_library=board_library,
                num_placements=placements_per_board * candidate_board_count,
            )
            execution = execute_task_family(family, initial, quota, rng)
            selected_families.append(_factory_compact_selected(execution, candidate_board_count))
            del execution, initial

        qualified = torch.ones(candidate_board_count, dtype=torch.bool, device=self.device)
        selected_counts: list[torch.Tensor | None] = []
        for selected_family, quota in zip(selected_families, quotas, strict=True):
            counts = None if selected_family is None else selected_family.board_counts
            selected_counts.append(counts)
            if counts is not None:
                qualified &= counts == quota
        qualified_indices = qualified.nonzero(as_tuple=False).squeeze(-1)
        if qualified_indices.numel() < board_count:
            misses = []
            for family, quota, counts in zip(families, quotas, selected_counts, strict=True):
                if counts is None:
                    continue
                missing = (counts != quota).nonzero(as_tuple=False).squeeze(-1)
                if missing.numel():
                    evidence = ", ".join(f"board {int(board)}={int(counts[board])}/{quota}" for board in missing)
                    misses.append(f"{family.name}: {evidence}")
            raise RuntimeError(
                f"only {qualified_indices.numel()} of {candidate_board_count} Factory boards satisfy every family "
                f"quota (need {board_count}; {'; '.join(misses)})"
            )
        candidate_pose = board_library[0][qualified_indices]
        rotation = torch.stack(math_utils.euler_xyz_from_quat(candidate_pose[:, 3:7]), dim=-1)
        kept_boards = qualified_indices[
            grid_bucket_downsample(
                torch.cat((candidate_pose[:, :3], 0.1 * rotation), dim=-1),
                board_count,
                generator=rng.torch,
            )
        ]
        board_remap = torch.full((candidate_board_count,), -1, dtype=torch.long, device=self.device)
        board_remap[kept_boards] = torch.arange(board_count, device=self.device)
        field_names = (
            "joint_q",
            "held_pose",
            "board_pose",
            "pad_targets",
            "aperture",
            "tag",
            "family",
            "board_index",
            "task_family",
            "quality",
            "is_grasped",
        )
        fields = {name: [] for name in field_names}
        board_asset_fields = {name: [] for name in self.cfg.board.fixed_asset_map}
        quality_names = (
            "task_family_id",
            "board_id",
            "tag_id",
            "family_quota",
            "family_generated",
            "family_accepted",
            "family_yield",
            "family_headroom",
            "pad_error_max_m",
        )
        family_names = tuple(family.name for family in families)
        grasp_family_names: list[str] | None = None
        for family_index, (family, selected_family, quota) in enumerate(
            zip(families, selected_families, quotas, strict=True)
        ):
            if selected_family is None:
                continue
            remapped_board = board_remap[selected_family.board_index]
            selected = (remapped_board >= 0).nonzero(as_tuple=False).squeeze(-1)
            remapped_board = remapped_board[selected]
            expected = quota * board_count
            if selected.numel() != expected:
                raise RuntimeError(
                    f"Factory family {family.name!r} retained {selected.numel()} rows; expected {expected}."
                )
            fields["joint_q"].append(selected_family.joint_q[selected])
            fields["held_pose"].append(selected_family.held_pose[selected])
            fields["board_pose"].append(selected_family.board_pose[selected])
            for name in board_asset_fields:
                board_asset_fields[name].append(selected_family.board_asset_poses[name][selected])
            fields["pad_targets"].append(selected_family.pad_targets[selected])
            fields["aperture"].append(selected_family.aperture[selected])
            fields["tag"].append(selected_family.tag[selected])
            fields["family"].append(selected_family.grasp_family[selected])
            fields["board_index"].append(remapped_board)
            fields["task_family"].append(torch.full((expected,), family_index, device=self.device, dtype=torch.long))
            fields["is_grasped"].append(selected_family.is_grasped[selected])
            family_yield = selected_family.accepted / selected_family.generated if selected_family.generated else 0.0
            family_headroom = selected_family.accepted / max(quota * candidate_board_count, 1)
            quality = (
                torch.tensor(
                    (
                        float(family_index),
                        0.0,
                        0.0,
                        float(quota),
                        float(selected_family.generated),
                        float(selected_family.accepted),
                        family_yield,
                        family_headroom,
                        0.0,
                    ),
                    device=self.device,
                )
                .expand(expected, -1)
                .clone()
            )
            quality[:, 1] = remapped_board
            quality[:, 2] = selected_family.tag[selected]
            quality[:, -1] = selected_family.target_error_m[selected]
            fields["quality"].append(quality)
            grasp_family_names = selected_family.grasp_family_names

        return FactoryIKResult(
            joint_q=torch.cat(fields["joint_q"]).contiguous(),
            held_pose=torch.cat(fields["held_pose"]).contiguous(),
            board_pose=torch.cat(fields["board_pose"]).contiguous(),
            board_asset_poses={name: torch.cat(values).contiguous() for name, values in board_asset_fields.items()},
            pad_targets=torch.cat(fields["pad_targets"]).contiguous(),
            aperture=torch.cat(fields["aperture"]).contiguous(),
            tag=torch.cat(fields["tag"]).contiguous(),
            tag_names=self.tag_names,
            family=torch.cat(fields["family"]).contiguous(),
            family_names=grasp_family_names or [],
            board_index=torch.cat(fields["board_index"]).contiguous(),
            task_family=torch.cat(fields["task_family"]).contiguous(),
            task_family_names=family_names,
            quality_names=quality_names,
            quality=torch.cat(fields["quality"]).contiguous(),
            is_grasped=torch.cat(fields["is_grasped"]).contiguous(),
        )


def _factory_pads_world(geometry: FactoryGeometry, body_q: torch.Tensor) -> torch.Tensor:
    """Return world contact-pad points [m], shape [row_count, 2, 3]."""
    count = body_q.shape[0]
    return torch.stack(
        [
            math_utils.quat_apply(body_q[:, body, 3:7], geometry.pad_offsets[index].expand(count, 3))
            + body_q[:, body, :3]
            for index, body in enumerate(geometry.pad_bodies)
        ],
        dim=1,
    )


def _factory_for_each_fk(candidates: FactoryFamilyCandidates, consume) -> None:
    """Evaluate and consume body poses one solver-capacity interval at a time."""
    if candidates.solve_statistics is None or candidates.joint_q is None:
        raise RuntimeError("Factory FK requires solved coordinates and execution statistics.")
    kinematics = candidates.kinematics
    capacity = min(candidates.solve_statistics.batch_capacity, candidates.joint_q.shape[0])
    device = candidates.geometry.device
    workspace = _FactoryFKWorkspace(
        joint_qd=wp.zeros((capacity, kinematics.model.joint_dof_count), dtype=wp.float32, device=device),
        body_q=wp.empty((capacity, kinematics.model.body_count), dtype=wp.transformf, device=device),
        body_qd=wp.empty((capacity, kinematics.model.body_count), dtype=wp.spatial_vectorf, device=device),
        capacity=capacity,
    )
    for start in range(0, candidates.joint_q.shape[0], capacity):
        stop = min(start + capacity, candidates.joint_q.shape[0])
        active_count = stop - start
        kinematics.eval_fk_batched(
            wp.from_torch(candidates.joint_q[start:stop]),
            workspace.joint_qd[:active_count],
            workspace.body_q[:active_count],
            workspace.body_qd[:active_count],
        )
        body_q = wp.to_torch(workspace.body_q)[:active_count].view(active_count, kinematics.model.body_count, 7)
        consume(start, stop, body_q)


def _factory_joints_within_limit(geometry: FactoryGeometry, joint_q: torch.Tensor, limit_ratio: float) -> torch.Tensor:
    """Return rows whose arm coordinates stay inside the declared safe interval."""
    if not 0.0 < limit_ratio <= 1.0:
        raise ValueError("Factory joint-limit ratio must be in (0, 1].")
    lower = torch.tensor(geometry.kinematics.topology.joint_limit_lower, device=geometry.device)[geometry.arm_dofs]
    upper = torch.tensor(geometry.kinematics.topology.joint_limit_upper, device=geometry.device)[geometry.arm_dofs]
    margin = 0.5 * (1.0 - limit_ratio) * (upper - lower)
    arm_q = joint_q[:, geometry.arm_coords]
    return ((arm_q >= lower + margin) & (arm_q <= upper - margin)).all(dim=-1)


def _factory_build_objectives(
    candidates: FactoryFamilyCandidates,
    objective_cfgs,
    batch_size: int,
    finger_targets: torch.Tensor,
) -> tuple[list[object], tuple[IKObjectiveBuild, ...]]:
    """Build declared objectives against capacity-sized mutable target buffers."""
    geometry = candidates.geometry
    base_context = IKObjectiveBuildContext(
        kinematics=candidates.kinematics,
        asset_name=geometry.cfg.robot.asset_cfg.name,
        batch_size=batch_size,
    )
    builds = []
    for objective_cfg in objective_cfgs:
        if isinstance(objective_cfg, IKObjectivePositionCfg):
            declared_bodies = objective_cfg.current.bodies
            declared_bodies = (declared_bodies,) if isinstance(declared_bodies, str) else tuple(declared_bodies)
            pad_names = tuple(candidates.kinematics.body_names[index] for index in geometry.pad_bodies)
            if declared_bodies != pad_names:
                raise ValueError(
                    f"Factory point targets declare bodies {declared_bodies!r}; expected ordered pads {pad_names!r}."
                )
            context = IKPositionObjectiveBuildContext(
                kinematics=candidates.kinematics,
                asset_name=base_context.asset_name,
                batch_size=batch_size,
                body_offsets=geometry.pad_offsets.detach().cpu().numpy(),
            )
        elif isinstance(objective_cfg, IKObjectiveJointPinCfg):
            context = IKJointPinObjectiveBuildContext(
                kinematics=candidates.kinematics,
                asset_name=base_context.asset_name,
                batch_size=batch_size,
                coordinate_indices=np.asarray(geometry.finger_coords),
                dof_indices=np.asarray(geometry.finger_dofs),
                targets=finger_targets,
            )
        else:
            context = base_context
        builder = objective_cfg.class_type
        builder = builder if callable(builder) else string_to_callable(builder)
        build = builder(objective_cfg, context)
        if not isinstance(build, IKObjectiveBuild):
            raise TypeError(f"IK objective builder returned {type(build).__name__}, expected IKObjectiveBuild.")
        if build.target_bind not in (None, "generated.grasp_points"):
            raise ValueError(f"Unknown Factory objective target binding {build.target_bind!r}.")
        builds.append(build)
    return [objective for build in builds for objective in build.objectives], tuple(builds)


def _factory_bind_grasp_targets(
    builds: tuple[IKObjectiveBuild, ...], candidates: FactoryFamilyCandidates, start: int, stop: int
) -> None:
    """Bind one source interval directly without materializing stacked grasp targets."""
    active_count = stop - start
    sources = (candidates.t_plus[start:stop], candidates.t_minus[start:stop])
    for build in builds:
        if build.target_bind is None:
            continue
        if len(build.objectives) != len(sources):
            raise ValueError("Factory grasp targets require one position objective per fingertip.")
        for objective, source in zip(build.objectives, sources, strict=True):
            wp.copy(objective.target_positions, wp.from_torch(source, dtype=wp.vec3), count=active_count)


def _factory_solve_targets(
    candidates: FactoryFamilyCandidates,
    solve_cfg,
    seed_arm: torch.Tensor,
) -> torch.Tensor:
    """Solve generated rows through one memory-bounded reusable workspace."""
    geometry = candidates.geometry
    row_count = candidates.t_plus.shape[0]
    finger_target_count = (
        len(geometry.finger_coords)
        if any(isinstance(objective, IKObjectiveJointPinCfg) for objective in solve_cfg.objectives)
        else 0
    )
    representative_finger_targets = torch.empty((1, finger_target_count), device=geometry.device)
    representative_objectives, _ = _factory_build_objectives(
        candidates, solve_cfg.objectives, 1, representative_finger_targets
    )
    jacobian_mode = (
        ik.IKJacobianType.MIXED
        if any(not objective.supports_analytic() for objective in representative_objectives)
        else ik.IKJacobianType.ANALYTIC
    )
    float_bytes = wp.types.type_size_in_bytes(wp.float32)

    def estimate_memory(batch_size: int) -> int:
        solver_bytes = ik.IKSolver.estimate_memory(
            candidates.kinematics.model,
            batch_size,
            representative_objectives,
            jacobian_mode=jacobian_mode,
        ).total_bytes
        return solver_bytes + batch_size * geometry.nq * float_bytes

    def build_batch(batch_size: int) -> _FactoryIKWorkspace:
        finger_targets = torch.empty((batch_size, finger_target_count), device=geometry.device)
        objectives, builds = _factory_build_objectives(candidates, solve_cfg.objectives, batch_size, finger_targets)
        return _FactoryIKWorkspace(
            solver=candidates.kinematics.create_ik_solver(objectives, batch_size, jacobian_mode=jacobian_mode),
            target_builds=builds,
            finger_targets=finger_targets,
            joint_q=torch.empty((batch_size, geometry.nq), device=geometry.device),
            capacity=batch_size,
        )

    joint_q = torch.empty((row_count, geometry.nq), device=geometry.device)

    def solve_batch(workspace, start, stop, max_iterations, tolerance, check_interval):
        active_count = stop - start
        _factory_bind_grasp_targets(workspace.target_builds, candidates, start, stop)
        if finger_target_count:
            workspace.finger_targets[:active_count].copy_(candidates.finger_targets[start:stop])
        active_joint_q = workspace.joint_q[:active_count]
        active_joint_q.zero_()
        active_joint_q[:, geometry.arm_coords] = seed_arm[start:stop]
        if finger_target_count:
            active_joint_q[:, geometry.finger_coords] = candidates.finger_targets[start:stop]
        result = workspace.solver.solve(
            wp.from_torch(workspace.joint_q),
            wp.from_torch(workspace.joint_q),
            max_iterations=max_iterations,
            active_problem_count=active_count,
            convergence_tolerance=tolerance,
            convergence_check_interval=check_interval,
        )
        joint_q[start:stop].copy_(active_joint_q)
        return result

    candidates.solve_statistics = execute_ik_batches(
        problem_count=row_count,
        device=geometry.device,
        estimate_memory=estimate_memory,
        build_batch=build_batch,
        solve_batch=solve_batch,
        max_iterations=solve_cfg.max_iterations,
        convergence_tolerance=solve_cfg.convergence_tolerance,
        convergence_check_interval=solve_cfg.convergence_check_interval,
    )
    del representative_finger_targets
    return joint_q


def _factory_solve_family(candidates: FactoryFamilyCandidates, solve_cfg) -> FactoryFamilyCandidates:
    """Solve and measure generated rows without retaining full-family body poses."""
    geometry = candidates.geometry
    candidates.joint_q = _factory_solve_targets(candidates, solve_cfg, candidates.seed_arm)
    count = candidates.joint_q.shape[0]
    candidates.target_error_m = torch.empty(count, device=geometry.device)
    candidates.ee_approach = torch.empty((count, 3), device=geometry.device)

    def measure(start: int, stop: int, body_q: torch.Tensor) -> None:
        measure_grasp_targets(
            body_q,
            geometry.pad_bodies_wp,
            geometry.pad_offsets_wp,
            candidates.t_plus[start:stop],
            candidates.t_minus[start:stop],
            geometry.ee_body,
            candidates.target_error_m[start:stop],
            candidates.ee_approach[start:stop],
            geometry.device,
        )

    _factory_for_each_fk(candidates, measure)
    candidates.aperture = (2.0 * candidates.joint_q[:, geometry.finger_coords[0]]).contiguous()
    candidates.is_grasped = candidates.target_is_grasped.contiguous()
    return candidates


def _factory_robot_clear(
    geometry: FactoryGeometry,
    body_q: torch.Tensor,
    board_pose: torch.Tensor,
    board_asset_poses: dict[str, torch.Tensor],
    criterion,
) -> torch.Tensor:
    """Return robot-vs-obstacle clearance including exact surface crossings."""
    row_count = body_q.shape[0]
    identity = torch.zeros(row_count, 7, device=geometry.device)
    identity[:, 6] = 1.0
    clear = torch.ones(row_count, dtype=torch.bool, device=geometry.device)
    obstacles = [(mesh.id, identity) for mesh in geometry.static_obstacles.values()]
    obstacles.append((geometry.board_mesh.id, board_pose))
    obstacles.extend(
        (geometry.board_asset_meshes[name].id, board_asset_poses[name]) for name in geometry.board_asset_meshes
    )
    for mesh_id, pose in obstacles:
        signed_distance = posed_collision_min_sd(
            body_q,
            geometry.robot_probe_bodies_wp,
            geometry.robot_probes_wp,
            mesh_id,
            pose,
            criterion.query_radius,
            geometry.device,
        )
        crossing = edges_vs_posed_mesh_hit(
            body_q,
            geometry.robot_edge_bodies_wp,
            geometry.robot_edge_p0_wp,
            geometry.robot_edge_p1_wp,
            mesh_id,
            pose,
            geometry.device,
        )
        clear &= (signed_distance >= -criterion.max_pen) & ~crossing
    board_crossing = posed_edges_vs_body_meshes_hit(
        geometry.board_edge_p0_wp,
        geometry.board_edge_p1_wp,
        board_pose,
        body_q,
        geometry.robot_target_body_wp,
        geometry.robot_target_mesh_wp,
        geometry.robot_target_tf_wp,
        geometry.device,
    )
    return clear & ~board_crossing


def _factory_gripper_held_distance(
    geometry: FactoryGeometry,
    body_q: torch.Tensor,
    held_pose: torch.Tensor,
    criterion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric gripper-to-held and held-to-gripper distances [m]."""
    forward = posed_collision_min_sd(
        body_q,
        geometry.grip_probe_bodies_wp,
        geometry.grip_probes_wp,
        geometry.held_mesh.id,
        held_pose,
        criterion.query_radius,
        geometry.device,
    )
    reverse = posed_points_vs_body_meshes_min_sd(
        geometry.held_probes_t,
        held_pose,
        body_q,
        geometry.grip_target_body_wp,
        geometry.grip_target_mesh_wp,
        geometry.grip_target_tf_wp,
        criterion.query_radius,
        geometry.device,
    )
    return forward, reverse


def _factory_held_obstacle_clear(
    geometry: FactoryGeometry,
    held_pose: torch.Tensor,
    board_pose: torch.Tensor,
    board_asset_poses: dict[str, torch.Tensor],
    allow_fixed_contact: torch.Tensor,
    allow_board_contact: torch.Tensor,
    criterion,
) -> torch.Tensor:
    """Return held-object clearance with explicit assembly/support exemptions."""
    row_count = held_pose.shape[0]
    clear = torch.ones(row_count, dtype=torch.bool, device=geometry.device)
    for mesh in geometry.static_obstacles.values():
        signed_distance = posed_points_min_sd(
            geometry.held_probes_t, held_pose, mesh.id, criterion.query_radius, geometry.device
        )
        clear &= signed_distance >= -criterion.max_pen
    posed_obstacles = [(geometry.board_mesh.id, board_pose, allow_board_contact)]
    for name, mesh in geometry.board_asset_meshes.items():
        allow_contact = (
            allow_fixed_contact
            if name == geometry.cfg.board.fixed_asset_cfg.name
            else torch.zeros_like(allow_fixed_contact)
        )
        posed_obstacles.append((mesh.id, board_asset_poses[name], allow_contact))
    for mesh_id, pose, exempt in posed_obstacles:
        signed_distance = posed_points_vs_posed_mesh_min_sd(
            geometry.held_probes_t,
            held_pose,
            mesh_id,
            pose,
            criterion.query_radius,
            geometry.device,
        )
        clear &= (signed_distance >= -criterion.max_pen) | exempt
    return clear


def _factory_self_clear(geometry: FactoryGeometry, body_q: torch.Tensor, criterion) -> torch.Tensor:
    """Return robot link self-clearance under criterion-local adjacency."""
    adjacency = geometry.self_adjacency_wp.get(criterion.adjacency_hops)
    if adjacency is None:
        adjacency = wp.array(
            geometry.self_adjacency(criterion.adjacency_hops).flatten(),
            dtype=wp.uint8,
            device=geometry.device,
        )
        geometry.self_adjacency_wp[criterion.adjacency_hops] = adjacency
    signed_distance = self_collision_min_sd(
        body_q,
        geometry.robot_full_probe_bodies_wp,
        geometry.robot_full_probes_wp,
        geometry.robot_target_body_wp,
        geometry.robot_target_mesh_wp,
        geometry.robot_target_tf_wp,
        adjacency,
        geometry.body_count,
        criterion.query_radius,
        geometry.device,
    )
    return signed_distance >= -criterion.self_max_pen


def _factory_tag_index(candidates: FactoryFamilyCandidates, name: str) -> int:
    """Register one generator-declared semantic tag."""
    if not name:
        raise ValueError("Factory generator tags must be nonempty.")
    index = candidates.tag_indices.get(name)
    if index is None:
        index = len(candidates.tag_names)
        candidates.tag_indices[name] = index
        candidates.tag_names.append(name)
    return index


def factory_generate_assembly_pose(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Generate assembly-path held-object poses for one family."""
    del rng
    values = candidates.placement_sampler.sample_assembly(cfg, candidates.num_placements, candidates.board_library)
    (
        candidates.held_pose,
        candidates.tag,
        candidates.board_pose,
        candidates.board_asset_poses,
        candidates.board_index,
    ) = values
    tag_map = torch.tensor(
        [_factory_tag_index(candidates, name) for name in cfg.assembly_bands],
        device=candidates.geometry.device,
        dtype=torch.long,
    )
    candidates.tag = tag_map[candidates.tag]
    candidates.allow_fixed_contact = torch.ones_like(candidates.tag, dtype=torch.bool)
    candidates.allow_board_contact = torch.zeros_like(candidates.tag, dtype=torch.bool)
    return candidates


def factory_generate_support_pose(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Generate board-supported held-object poses for one family."""
    del rng
    values = candidates.placement_sampler.sample_support(
        cfg,
        candidates.num_placements,
        candidates.board_library,
        _factory_tag_index(candidates, cfg.tag),
    )
    (
        candidates.held_pose,
        candidates.tag,
        candidates.board_pose,
        candidates.board_asset_poses,
        candidates.board_index,
    ) = values
    candidates.allow_fixed_contact = torch.zeros_like(candidates.tag, dtype=torch.bool)
    candidates.allow_board_contact = torch.ones_like(candidates.tag, dtype=torch.bool)
    return candidates


def factory_generate_free_pose(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Generate free-space held-object poses for one family."""
    del rng
    values = candidates.placement_sampler.sample_free(
        cfg,
        candidates.num_placements,
        candidates.board_library,
        _factory_tag_index(candidates, cfg.tag),
    )
    (
        candidates.held_pose,
        candidates.tag,
        candidates.board_pose,
        candidates.board_asset_poses,
        candidates.board_index,
    ) = values
    candidates.allow_fixed_contact = torch.zeros_like(candidates.tag, dtype=torch.bool)
    candidates.allow_board_contact = torch.zeros_like(candidates.tag, dtype=torch.bool)
    return candidates


def factory_generate_grasp_targets(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Generate antipodal contact targets and expand placement-owned fields."""
    if candidates.held_pose is None:
        raise RuntimeError("Factory grasp generation requires a held-object pose generator first.")
    key = repr(cfg.sampling.to_dict())
    sampler = candidates.grasp_samplers.get(key)
    if sampler is None:
        sampler = GraspPairSampler(candidates.geometry, cfg.sampling, rng.torch)
        candidates.grasp_samplers[key] = sampler
    t_plus, t_minus, source, grasp_family = sampler.sample_targets(candidates.held_pose, cfg.grasps_per_placement)
    candidates.grasp_sampler = sampler
    candidates.t_plus = t_plus
    candidates.t_minus = t_minus
    candidates.grasp_family = grasp_family
    candidates.target_is_grasped = torch.ones(t_plus.shape[0], dtype=torch.bool, device=t_plus.device)
    finger_target = 0.5 * (t_plus - t_minus).norm(dim=-1)
    candidates.finger_targets = finger_target[:, None].expand(-1, len(candidates.geometry.finger_coords)).contiguous()
    candidates.board_asset_poses = {
        name: poses[source].contiguous() for name, poses in candidates.board_asset_poses.items()
    }
    for name in (
        "held_pose",
        "tag",
        "board_pose",
        "board_index",
        "allow_fixed_contact",
        "allow_board_contact",
    ):
        setattr(candidates, name, getattr(candidates, name)[source].contiguous())
    return candidates


def factory_generate_robot_seeds(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Expand generated targets across their nearest independent robot seeds."""
    del rng
    if candidates.grasp_sampler is None or candidates.t_plus is None or candidates.t_minus is None:
        raise RuntimeError("Factory robot-seed generation requires grasp targets first.")
    t_plus, t_minus, seed_arm, source = candidates.grasp_sampler.seed_targets(
        candidates.t_plus, candidates.t_minus, cfg.ik_seeds_per_grasp
    )
    candidates.t_plus = t_plus
    candidates.t_minus = t_minus
    candidates.seed_arm = seed_arm
    candidates.board_asset_poses = {
        name: poses[source].contiguous() for name, poses in candidates.board_asset_poses.items()
    }
    for name in (
        "held_pose",
        "tag",
        "board_pose",
        "board_index",
        "grasp_family",
        "finger_targets",
        "allow_fixed_contact",
        "allow_board_contact",
        "target_is_grasped",
    ):
        setattr(candidates, name, getattr(candidates, name)[source].contiguous())
    return candidates


def factory_generate_approach_targets(cfg, candidates: FactoryFamilyCandidates, rng: TaskTableRng):
    """Offset generated grasp targets along each seed end-effector approach axis."""
    if candidates.seed_arm is None or candidates.t_plus is None or candidates.t_minus is None:
        raise RuntimeError("Factory approach generation requires robot seeds first.")
    geometry = candidates.geometry
    count = candidates.seed_arm.shape[0]
    joint_q = factory_default_joint_q(geometry).expand(count, -1).clone()
    joint_q[:, geometry.arm_coords] = candidates.seed_arm
    body_q = factory_eval_fk(candidates.kinematics, joint_q)
    approach = math_utils.quat_apply(
        body_q[:, geometry.ee_body, 3:7],
        torch.tensor([0.0, 0.0, 1.0], device=geometry.device).expand(count, 3),
    )
    standoff = torch.empty(count, 1, device=geometry.device).uniform_(*cfg.standoff_range, generator=rng.torch)
    candidates.t_plus = (candidates.t_plus - standoff * approach).contiguous()
    candidates.t_minus = (candidates.t_minus - standoff * approach).contiguous()
    finger_target = 0.5 * (candidates.t_plus - candidates.t_minus).norm(dim=-1)
    candidates.finger_targets = finger_target[:, None].expand(-1, len(geometry.finger_coords)).contiguous()
    candidates.target_is_grasped.zero_()
    candidates.gripper_clearance = cfg.clearance
    return candidates


def factory_solve_ik(cfg, candidates: FactoryFamilyCandidates) -> FactoryFamilyCandidates:
    """Apply the declared batched Newton IK solve."""
    count = candidates.t_plus.shape[0]
    candidates = _factory_solve_family(candidates, cfg)
    if candidates.joint_q.shape[0] != count:
        raise RuntimeError("Factory solve must preserve candidate count; criteria own acceptance.")
    return candidates


def factory_target_error_criterion(cfg, candidates: FactoryFamilyCandidates) -> torch.Tensor:
    """Accept rows whose cached fingertip target error meets the declared bound."""
    if cfg.max_error_m <= 0.0:
        raise ValueError("Factory target-error bound must be positive.")
    return candidates.target_error_m <= cfg.max_error_m


def factory_held_pose_bounds_criterion(cfg, candidates: FactoryFamilyCandidates) -> torch.Tensor:
    """Accept rows whose generated held pose lies inside declared axis bounds."""
    axes = ("x", "y", "z")
    lower = torch.tensor(
        [cfg.bounds.get(axis, (-torch.inf, torch.inf))[0] for axis in axes], device=candidates.held_pose.device
    )
    upper = torch.tensor(
        [cfg.bounds.get(axis, (-torch.inf, torch.inf))[1] for axis in axes], device=candidates.held_pose.device
    )
    return ((candidates.held_pose[:, :3] >= lower) & (candidates.held_pose[:, :3] <= upper)).all(dim=-1)


def factory_joint_limit_criterion(cfg, candidates: FactoryFamilyCandidates) -> torch.Tensor:
    """Accept rows whose solved arm remains inside the declared joint interval."""
    return _factory_joints_within_limit(candidates.geometry, candidates.joint_q, cfg.limit_ratio)


def factory_collision_criterion(cfg, candidates: FactoryFamilyCandidates) -> torch.Tensor:
    """Certify collisions one solver-capacity FK interval at a time."""
    geometry = candidates.geometry
    accepted = torch.empty(candidates.joint_q.shape[0], dtype=torch.bool, device=geometry.device)

    def certify(start: int, stop: int, body_q: torch.Tensor) -> None:
        board_asset_poses = {name: poses[start:stop] for name, poses in candidates.board_asset_poses.items()}
        robot_ok = _factory_robot_clear(geometry, body_q, candidates.board_pose[start:stop], board_asset_poses, cfg)
        forward, reverse = _factory_gripper_held_distance(geometry, body_q, candidates.held_pose[start:stop], cfg)
        grasp_ok = (forward >= -cfg.max_pen) & (reverse >= -cfg.max_pen)
        reach_ok = (forward >= candidates.gripper_clearance) & (reverse >= candidates.gripper_clearance)
        gripper_ok = torch.where(candidates.is_grasped[start:stop], grasp_ok, reach_ok)
        held_ok = _factory_held_obstacle_clear(
            geometry,
            candidates.held_pose[start:stop],
            candidates.board_pose[start:stop],
            board_asset_poses,
            candidates.allow_fixed_contact[start:stop],
            candidates.allow_board_contact[start:stop],
            cfg,
        )
        accepted[start:stop].copy_(robot_ok & gripper_ok & held_ok & _factory_self_clear(geometry, body_q, cfg))

    _factory_for_each_fk(candidates, certify)
    return accepted


def factory_fps_selection(
    cfg,
    candidates: FactoryFamilyCandidates,
    accepted: torch.Tensor | None,
    target_count: int | None,
    rng: TaskTableRng,
) -> torch.Tensor:
    """Select the requested accepted quota on each independently feasible board."""
    geometry = candidates.geometry
    count = candidates.joint_q.shape[0]
    accepted = torch.ones(count, dtype=torch.bool, device=candidates.geometry.device) if accepted is None else accepted
    if target_count is None:
        return accepted.nonzero(as_tuple=False).squeeze(-1)
    if target_count < 1:
        return torch.empty(0, dtype=torch.long, device=candidates.geometry.device)
    if cfg.position_frame == "fixed_asset":
        fixed_pose = candidates.board_asset_poses[geometry.cfg.board.fixed_asset_cfg.name]
        position = math_utils.quat_apply_inverse(fixed_pose[:, 3:7], candidates.held_pose[:, :3] - fixed_pose[:, :3])
    elif cfg.position_frame == "world":
        position = candidates.held_pose[:, :3]
    else:
        raise ValueError(f"Unknown Factory FPS position frame {cfg.position_frame!r}.")
    position = position[:, cfg.position_axes]
    approach = candidates.ee_approach
    tag = torch.nn.functional.one_hot(candidates.tag, num_classes=len(candidates.tag_names)).float()
    features = torch.cat((cfg.position_weight * position, cfg.approach_weight * approach, cfg.tag_weight * tag), dim=-1)
    selected = []
    for board_index in range(candidates.board_library[0].shape[0]):
        rows = (accepted & (candidates.board_index == board_index)).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() < target_count:
            continue
        selected.append(rows[grid_bucket_downsample(features[rows], target_count, generator=rng.torch)])
    if not selected:
        return torch.empty(0, dtype=torch.long, device=candidates.geometry.device)
    return torch.cat(selected).contiguous()
