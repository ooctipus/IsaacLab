# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build Position task rows from the declared terrain-stance family.

Pure function module -- no classes, no state. Called once during command
term initialization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp

from isaaclab.utils.string import string_to_callable
from isaaclab.utils.warp import convert_to_warp_mesh

from ....kinematics import IKExecutionStatistics, NewtonKinematics, execute_ik_batches
from ....kinematics.ik_objectives.cfg import IKObjectiveMeshCollisionCfg
from ....kinematics.ik_objectives.context import (
    IKContactObjectiveBuildContext,
    IKObjectiveBuild,
    IKObjectiveMeshCollisionBuildContext,
)
from ....kinematics.ik_objectives.mesh_collision import collision_probes_sample
from ....kinematics.ik_objectives.stability_margin import stability_margin_measure
from ....mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTablePointEvidence,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
)
from ....mdp.commands.state_command.task_family import execute_task_family, make_task_table_rng
from ....utils.grid_downsample import extract_features, grid_bucket_downsample
from ...retarget.buffer import RetargetBuffer
from ...retarget.sampler_base import resolve_contact_body_names

if TYPE_CHECKING:
    import trimesh

    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveSceneCfg

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from .commands_cfg import Commands, TerrainCommands


_COMMAND_PARAM_NAMES = (
    "pos_x",
    "pos_y",
    "pos_z",
    "roll",
    "pitch",
    "yaw",
    "lin_vel_x",
    "lin_vel_y",
    "lin_vel_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "duration",
)


@wp.kernel
def _position_contact_measures(
    body_q: wp.array2d(dtype=wp.transform),
    contact_body_ids: wp.array1d(dtype=wp.int32),
    contact_targets: wp.array1d(dtype=wp.vec3),
    is_contact: wp.array1d(dtype=wp.bool),
    contact_count: int,
    active_contact_count: wp.array1d(dtype=wp.int32),
    foot_position_error: wp.array2d(dtype=wp.float32),
):
    """Measure active contacts and per-foot target error without temporary tensors."""
    row, contact = wp.tid()
    position = wp.transform_get_translation(body_q[row, contact_body_ids[contact]])
    foot_position_error[row, contact] = wp.length(position - contact_targets[row * contact_count + contact])
    if contact == 0:
        active = int(0)
        for slot in range(contact_count):
            if is_contact[row * contact_count + slot]:
                active = active + 1
        active_contact_count[row] = active


@dataclass(frozen=True, slots=True)
class RelativeStateTaskTable:
    """Index-based task table for the locomotion :class:`~...mdp.commands.StateCommand`."""

    num_tasks: int
    spawn_index: torch.Tensor
    """Index into :attr:`states` for each task's spawn point."""
    target_index: torch.Tensor
    """Index into :attr:`states` for each task's target point."""
    tile_index: torch.Tensor
    """Terrain tile (``row * num_cols + col``) each task belongs to ``[num_tasks]``."""
    params: torch.Tensor
    """Per-task sampled parameters ``[num_tasks, 13]``:
    ``[0:3]`` pos offset, ``[3:6]`` rot, ``[6:9]`` lin_vel,
    ``[9:12]`` ang_vel, ``[12]`` hold_time."""
    task_mask: torch.Tensor
    """Active command mask ``[num_tasks, 12 + num_joints]``."""
    payload_flags: torch.Tensor
    """Opaque payload flags ``[num_tasks, num_payload_flags]``."""
    offsets: torch.Tensor
    """CSR offsets ``[num_cmd_types + 1]`` into the task table."""
    task_partition: torch.Tensor
    """Command type id for each task row ``[num_tasks]``."""
    kind: torch.Tensor
    """Command type tag ``[num_cmd_types]``: 0=pos, 1=pose, 2=vel."""
    states: ResetStateBank
    """Canonical one-entity reset-state bank."""
    view: TaskTableView
    """Two-frame task sequences and exact retained Newton mechanics."""

    kinematics: NewtonKinematics
    """Exact Newton mechanics retained for target-state FK."""
    contact_body_names: tuple[str, ...]
    """Contact body names resolved in Newton order."""
    contact_body_ids: tuple[int, ...]
    """Contact body ids in Newton order."""

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample task rows uniformly on the table device."""
        return torch.randint(0, self.num_tasks, (count,), device=self.spawn_index.device)

    def gather(self, task_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spawn and target state-row indices for selected tasks.

        Args:
            task_rows: Task-table row indices ``[n]``.

        Returns:
            Spawn and target indices into :attr:`states`, each ``[n]``.
        """
        return self.spawn_index[task_rows], self.target_index[task_rows]


@dataclass(frozen=True, slots=True)
class PositionTerrainStanceInput:
    """Table-global inputs consumed by the terrain-stance generator."""

    kinematics: NewtonKinematics
    asset_name: str
    warp_mesh: object
    origin: np.ndarray
    sampling_x_range: tuple[float, float]
    sampling_y_range: tuple[float, float]
    target_count: int


@dataclass(slots=True)
class PositionTerrainStanceCandidates:
    """Concrete generated and solved facts passed between Position stages."""

    buffer: RetargetBuffer
    kinematics: NewtonKinematics
    asset_name: str
    contact_body_names: tuple[str, ...]
    contact_body_ids: tuple[int, ...]
    sampler_cfg: object
    collision_mesh: wp.Mesh
    solver_costs: torch.Tensor | None = None
    stability_margin: torch.Tensor | None = None
    active_contact_count: torch.Tensor | None = None
    foot_position_error: torch.Tensor | None = None
    solve_statistics: IKExecutionStatistics | None = None

    @property
    def num_rows(self) -> int:
        """Number of geometry-valid rows entering the criterion cascade."""
        return self.buffer.num_geometry_valid

    @property
    def device(self) -> str:
        """Device carrying candidate rows."""
        return self.buffer.device


@dataclass(slots=True)
class _PositionIKWorkspace:
    """One capacity-sized Position objective and solver workspace."""

    solver: ik.IKSolver
    targets: dict[str, list[ik.IKObjective]]
    contact_mask: torch.Tensor
    contact_confidence: torch.Tensor
    joint_q_tail: torch.Tensor | None
    capacity: int


def generate_position_terrain_stance(cfg, initial: PositionTerrainStanceInput, rng) -> PositionTerrainStanceCandidates:
    """Generate one terrain-stance candidate batch without solving or filtering."""
    sampler_cfg = _sampler_with_inner_sampling_bounds(
        cfg.sampler,
        initial.sampling_x_range,
        initial.sampling_y_range,
        override=True,
    )
    kinematics = initial.kinematics
    contact_body_names = tuple(resolve_contact_body_names(cfg.foot_body_names, kinematics.body_names))
    contact_body_ids = tuple(kinematics.find_body_indices(contact_body_names))
    sampler = sampler_cfg.class_type(sampler_cfg, kinematics, contact_body_ids, rng.torch)
    sizing = sampler.sizing(initial.target_count)
    buffer = RetargetBuffer(
        max_candidates=sizing.ik_capacity,
        joint_coord_count=kinematics.model.joint_coord_count,
        num_bodies=kinematics.model.body_count,
        num_contacts=len(contact_body_ids),
        device=kinematics.device,
    )
    output = sampler(
        initial.warp_mesh,
        initial.origin,
        buffer,
        initial.target_count,
        seed=rng.next_warp_seed(),
    )
    if output.is_contact is not None and output.num_written > 0:
        count = output.num_written * buffer.num_contacts
        buffer.is_contact_t[:count].copy_(output.is_contact.reshape(-1)[:count])
    return PositionTerrainStanceCandidates(
        buffer=buffer,
        kinematics=kinematics,
        asset_name=initial.asset_name,
        contact_body_names=contact_body_names,
        contact_body_ids=contact_body_ids,
        sampler_cfg=sampler_cfg,
        collision_mesh=initial.warp_mesh,
    )


def _build_ik_objectives(
    candidates,
    objective_cfgs,
    batch_size: int,
    contact_mask: torch.Tensor,
    contact_confidence: torch.Tensor,
    obstacle_pose: torch.Tensor,
    collision_probes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
):
    """Build one flat objective tuple and its generated-target routes."""
    contact_context = IKContactObjectiveBuildContext(
        kinematics=candidates.kinematics,
        asset_name=candidates.asset_name,
        batch_size=batch_size,
        contact_body_ids=candidates.contact_body_ids,
        contact_mask=contact_mask,
    )
    objectives: list[ik.IKObjective] = []
    targets: dict[str, list[ik.IKObjective]] = {}
    for objective_cfg in objective_cfgs:
        context = contact_context
        if isinstance(objective_cfg, IKObjectiveMeshCollisionCfg):
            probe_bodies, probe_offsets, probe_slots = collision_probes[objective_cfg.n_samples]
            context = IKObjectiveMeshCollisionBuildContext(
                kinematics=candidates.kinematics,
                asset_name=candidates.asset_name,
                batch_size=batch_size,
                collision_mesh=candidates.collision_mesh,
                obstacle_pose=obstacle_pose,
                probe_offsets=probe_offsets,
                probe_bodies=probe_bodies,
                probe_contact_slots=probe_slots,
                contact_confidence=contact_confidence,
            )
        built = objective_cfg.class_type(objective_cfg, context)
        if not isinstance(built, IKObjectiveBuild):
            raise TypeError(f"Objective builder returned {type(built).__name__}, expected IKObjectiveBuild.")
        objectives.extend(built.objectives)
        if built.target_bind is not None:
            targets.setdefault(built.target_bind, []).extend(built.objectives)
    return objectives, targets


def solve_position_terrain_stance(cfg, candidates: PositionTerrainStanceCandidates) -> PositionTerrainStanceCandidates:
    """Solve generated terrain candidates with one memory-sized reusable workspace."""
    buffer = candidates.buffer
    count = buffer.num_geometry_valid
    if count == 0:
        candidates.solver_costs = torch.empty(0, dtype=torch.float32, device=buffer.device)
        candidates.stability_margin = torch.empty(0, dtype=torch.float32, device=buffer.device)
        candidates.active_contact_count = torch.empty(0, dtype=torch.int32, device=buffer.device)
        candidates.foot_position_error = torch.empty(
            (0, len(candidates.contact_body_ids)), dtype=torch.float32, device=buffer.device
        )
        return candidates

    model = candidates.kinematics.model
    contact_count = len(candidates.contact_body_ids)
    collision_probes = {
        objective_cfg.n_samples: collision_probes_sample(
            candidates.kinematics.builder,
            candidates.contact_body_ids,
            objective_cfg.n_samples,
        )
        for objective_cfg in cfg.objectives
        if isinstance(objective_cfg, IKObjectiveMeshCollisionCfg)
    }

    def build_objectives(batch_size: int):
        contact_mask = torch.empty((batch_size, contact_count), dtype=torch.uint8, device=buffer.device)
        contact_confidence = torch.empty((batch_size, contact_count), dtype=torch.float32, device=buffer.device)
        obstacle_pose = torch.zeros((batch_size, 7), dtype=torch.float32, device=buffer.device)
        obstacle_pose[:, 6] = 1.0
        objectives, targets = _build_ik_objectives(
            candidates,
            cfg.objectives,
            batch_size,
            contact_mask,
            contact_confidence,
            obstacle_pose,
            collision_probes,
        )
        return contact_mask, contact_confidence, objectives, targets

    (
        representative_contact_mask,
        representative_contact_confidence,
        representative_objectives,
        representative_targets,
    ) = build_objectives(1)
    del representative_targets
    jacobian_mode = (
        ik.IKJacobianType.MIXED
        if any(not objective.supports_analytic() for objective in representative_objectives)
        else ik.IKJacobianType.ANALYTIC
    )
    coordinate_count = model.joint_coord_count
    dof_count = model.joint_dof_count
    body_count = model.body_count
    uint8_bytes = wp.types.type_size_in_bytes(wp.uint8)
    float_bytes = wp.types.type_size_in_bytes(wp.float32)
    spatial_bytes = wp.types.type_size_in_bytes(wp.spatial_vector)

    def estimate_memory(batch_size: int) -> int:
        solver_bytes = ik.IKSolver.estimate_memory(
            model, batch_size, representative_objectives, jacobian_mode=jacobian_mode
        ).total_bytes
        binding_bytes = batch_size * (
            contact_count * (uint8_bytes + float_bytes) + 7 * float_bytes + coordinate_count * float_bytes
        )
        fk_scratch_bytes = batch_size * (dof_count * float_bytes + body_count * spatial_bytes)
        return max(solver_bytes + binding_bytes, fk_scratch_bytes)

    def build_batch(batch_size: int) -> _PositionIKWorkspace:
        contact_mask, contact_confidence, objectives, targets = build_objectives(batch_size)
        contact_targets = targets.get("generated.foot_targets", [])
        base_position_targets = targets.get("generated.base_position", [])
        base_rotation_targets = targets.get("generated.base_rotation", [])
        if len(contact_targets) != contact_count:
            raise ValueError("Position IK requires one generated foot target per contact body.")
        if len(base_position_targets) != 1 or len(base_rotation_targets) != 1:
            raise ValueError("Position IK requires one base-position and one base-rotation target.")
        solver = candidates.kinematics.create_ik_solver(objectives, batch_size, jacobian_mode=jacobian_mode)
        return _PositionIKWorkspace(
            solver=solver,
            targets=targets,
            contact_mask=contact_mask,
            contact_confidence=contact_confidence,
            joint_q_tail=(
                None
                if count % batch_size == 0
                else torch.empty((batch_size, coordinate_count), dtype=torch.float32, device=buffer.device)
            ),
            capacity=batch_size,
        )

    solver_costs = torch.empty(count, dtype=torch.float32, device=buffer.device)
    candidates.stability_margin = torch.empty(count, dtype=torch.float32, device=buffer.device)
    candidates.active_contact_count = torch.empty(count, dtype=torch.int32, device=buffer.device)
    candidates.foot_position_error = torch.empty((count, contact_count), dtype=torch.float32, device=buffer.device)

    def solve_batch(workspace, start, stop, max_iterations, tolerance, check_interval):
        active_count = stop - start
        workspace.contact_mask[:active_count].copy_(
            buffer.is_contact_t[start * contact_count : stop * contact_count].view(active_count, contact_count)
        )
        workspace.contact_confidence[:active_count].copy_(workspace.contact_mask[:active_count])
        contact_targets = workspace.targets["generated.foot_targets"]
        buffer.scatter_contact_targets(contact_targets, active_count, src_offset=start)
        wp.copy(
            workspace.targets["generated.base_position"][0].target_positions,
            buffer.base_target_pos,
            src_offset=start,
            count=active_count,
        )
        wp.copy(
            workspace.targets["generated.base_rotation"][0].target_rotations,
            buffer.base_target_rot,
            src_offset=start,
            count=active_count,
        )
        if active_count == workspace.capacity:
            joint_q_in = wp.from_torch(buffer.joint_q_init_t[start:stop])
            joint_q_out = wp.from_torch(buffer.joint_q_result_t[start:stop])
        else:
            if workspace.joint_q_tail is None:
                raise RuntimeError("Position IK tail workspace is missing.")
            workspace.joint_q_tail[:active_count].copy_(buffer.joint_q_init_t[start:stop])
            joint_q_in = wp.from_torch(workspace.joint_q_tail)
            joint_q_out = joint_q_in
        result = workspace.solver.solve(
            joint_q_in,
            joint_q_out,
            max_iterations=max_iterations,
            active_problem_count=active_count,
            convergence_tolerance=tolerance,
            convergence_check_interval=check_interval,
        )
        if active_count != workspace.capacity:
            buffer.joint_q_result_t[start:stop].copy_(workspace.joint_q_tail[:active_count])
        solver_costs[start:stop].copy_(wp.to_torch(workspace.solver.costs)[:active_count])
        return result

    candidates.solve_statistics = execute_ik_batches(
        problem_count=count,
        device=buffer.device,
        estimate_memory=estimate_memory,
        build_batch=build_batch,
        solve_batch=solve_batch,
        max_iterations=cfg.max_iterations,
        convergence_tolerance=cfg.convergence_tolerance,
        convergence_check_interval=cfg.convergence_check_interval,
    )
    del representative_contact_mask, representative_contact_confidence

    batch_size = candidates.solve_statistics.batch_capacity
    joint_qd = wp.zeros((batch_size, dof_count), dtype=wp.float32, device=buffer.device)
    body_qd = wp.zeros((batch_size, body_count), dtype=wp.spatial_vectorf, device=buffer.device)
    body_q = wp.from_torch(buffer.body_q_t.view(buffer.max_candidates, body_count, 7), dtype=wp.transformf)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        active_count = stop - start
        candidates.kinematics.eval_fk_batched(
            buffer.joint_q_result[start:stop],
            joint_qd[:active_count],
            body_q[start:stop],
            body_qd[:active_count],
        )

    candidates.solver_costs = solver_costs
    body_q_t = buffer.body_q_t[: count * body_count].view(count, body_count, 7)
    is_contact = buffer.is_contact_t[: count * contact_count].view(count, contact_count)
    candidates.stability_margin = stability_margin_measure(
        candidates.kinematics,
        body_q_t,
        candidates.contact_body_ids,
        is_contact,
        batch_capacity=batch_size,
        output=candidates.stability_margin,
    )
    contact_body_ids = wp.array(candidates.contact_body_ids, dtype=wp.int32, device=buffer.device)
    wp.launch(
        _position_contact_measures,
        dim=(count, contact_count),
        inputs=[
            body_q[:count],
            contact_body_ids,
            buffer.contact_targets,
            wp.from_torch(buffer.is_contact_t, dtype=wp.bool),
            contact_count,
        ],
        outputs=[
            wp.from_torch(candidates.active_contact_count),
            wp.from_torch(candidates.foot_position_error),
        ],
        device=buffer.device,
    )
    return candidates


def select_position_terrain_stance(cfg, candidates, accepted, target_count, rng) -> torch.Tensor:
    """FPS-thin accepted candidates without generation, repair, or solving."""
    if target_count is None:
        raise ValueError("Position terrain selection requires a table-resolved target count.")
    buffer = candidates.buffer
    count = buffer.num_geometry_valid
    if accepted is None:
        accepted = torch.ones(count, device=buffer.device, dtype=torch.bool)
    selected = accepted.nonzero(as_tuple=False).squeeze(-1)
    if selected.numel() < target_count:
        raise RuntimeError(
            f"Position terrain family requested exactly {target_count} states but only {selected.numel()} passed."
        )
    states = buffer.joint_q_result_t[selected]
    features = extract_features(states, cfg.features)
    local_rows = grid_bucket_downsample(features, target_count, generator=rng.torch)
    return selected[local_rows].contiguous()


def build_relative_state_task_table(
    command_cfg: StateCommandCfg,
    scene_cfg: InteractiveSceneCfg,
    device: str,
) -> RelativeStateTaskTable:
    """Build a Position task table from resolved declarations only."""
    table_cfg = command_cfg.task_table
    if len(command_cfg.reset_assets) != 1:
        raise ValueError("Position requires exactly one reset articulation.")
    asset_name = command_cfg.reset_assets[0]
    articulation_cfg = getattr(scene_cfg, asset_name, None)
    if articulation_cfg is None:
        raise ValueError(f"Position scene does not declare articulation {asset_name!r}.")

    terrain_cfg = scene_cfg.terrain
    if terrain_cfg.terrain_type != "generator" or terrain_cfg.terrain_generator is None:
        raise ValueError("RelativeStateCommand requires scene.terrain.terrain_type='generator'.")
    terrain_generator_cfg = copy.deepcopy(terrain_cfg.terrain_generator)
    if terrain_generator_cfg.seed is None:
        raise ValueError("Position terrain generation requires an explicit non-None seed.")
    if table_cfg.seed != terrain_generator_cfg.seed:
        raise ValueError("Position task-table and runtime terrain seeds must match.")
    rng = make_task_table_rng(table_cfg.seed, device)
    generator_type = terrain_generator_cfg.class_type
    if isinstance(generator_type, str):
        generator_type = string_to_callable(generator_type)
    terrain = generator_type(terrain_generator_cfg, device=device, rng=rng.numpy)
    if terrain_cfg.use_terrain_origins:
        terrain_origins = torch.as_tensor(terrain.terrain_origins, dtype=torch.float32, device=device)
    else:
        terrain_origins = _synthesize_terrain_origins(
            num_rows=int(terrain_generator_cfg.num_rows),
            num_cols=int(terrain_generator_cfg.num_cols),
            cell_size=terrain_generator_cfg.size,
            device=device,
        )

    return build_task_table(
        terrain_mesh=terrain.terrain_mesh,
        terrain_origins=terrain_origins,
        cell_size=terrain_generator_cfg.size,
        table_cfg=table_cfg,
        articulation_cfg=articulation_cfg,
        asset_name=asset_name,
        commands=command_cfg.commands,
        device=device,
        rng=rng,
    )


def _terrain_grid_bounds(
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return inner terrain-grid XY bounds [m]."""
    half_x = 0.5 * cell_size[0]
    half_y = 0.5 * cell_size[1]
    x_range = (
        float(terrain_origins[..., 0].amin().item() - half_x),
        float(terrain_origins[..., 0].amax().item() + half_x),
    )
    y_range = (
        float(terrain_origins[..., 1].amin().item() - half_y),
        float(terrain_origins[..., 1].amax().item() + half_y),
    )
    return x_range, y_range


def _synthesize_terrain_origins(
    num_rows: int,
    num_cols: int,
    cell_size: tuple[float, float],
    device: str,
) -> torch.Tensor:
    """Return terrain-generator tile origins when importer env origins are disabled."""
    origins = torch.zeros(num_rows, num_cols, 3, device=device)
    rows = (torch.arange(num_rows, device=device, dtype=torch.float32) + 0.5) * cell_size[0]
    cols = (torch.arange(num_cols, device=device, dtype=torch.float32) + 0.5) * cell_size[1]
    rows -= cell_size[0] * num_rows * 0.5
    cols -= cell_size[1] * num_cols * 0.5
    origins[..., 0], origins[..., 1] = torch.meshgrid(rows, cols, indexing="ij")
    return origins


def _state_count_from_spacing(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    spacing: float,
    area_divisor: float,
) -> int:
    """Derive terrain-state count from spacing and sampling area."""
    if spacing <= 0.0:
        raise ValueError(f"pool_spacing must be positive, got {spacing}.")
    if area_divisor <= 0.0:
        raise ValueError(f"pool_spacing_area_divisor must be positive, got {area_divisor}.")
    area = max((x_range[1] - x_range[0]) * (y_range[1] - y_range[0]), 0.0)
    return max(1, int(area / (area_divisor * spacing**2)))


def _centered_sampling_bounds(
    grid_x_range: tuple[float, float],
    grid_y_range: tuple[float, float],
    sampling_size: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return full-grid or centered clipped sampling bounds [m]."""
    if sampling_size is None:
        return grid_x_range, grid_y_range

    size_x, size_y = sampling_size
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"pool_sampling_size must be positive, got {sampling_size}.")

    center_x = 0.5 * (grid_x_range[0] + grid_x_range[1])
    center_y = 0.5 * (grid_y_range[0] + grid_y_range[1])
    half_x = 0.5 * size_x
    half_y = 0.5 * size_y
    x_range = (max(grid_x_range[0], center_x - half_x), min(grid_x_range[1], center_x + half_x))
    y_range = (max(grid_y_range[0], center_y - half_y), min(grid_y_range[1], center_y + half_y))
    return x_range, y_range


def _sampler_with_inner_sampling_bounds(
    sampler_cfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    override: bool = False,
) -> object:
    """Return a copied sampler bounded to the requested terrain range."""
    patch_cfg = getattr(sampler_cfg, "patch", None)
    if patch_cfg is None:
        return sampler_cfg

    patch_updates = {}
    if override or patch_cfg.x_range is None:
        patch_updates["x_range"] = x_range
    if override or patch_cfg.y_range is None:
        patch_updates["y_range"] = y_range
    if not patch_updates:
        return sampler_cfg

    return sampler_cfg.replace(patch=patch_cfg.replace(**patch_updates))


def _newton_state_joints(kinematics: NewtonKinematics) -> tuple[list[int], list[str]]:
    """Return scalar articulation coordinates and names after the free root."""
    if kinematics.n_root_coords != 7:
        raise ValueError("Position terrain stances require free-root Newton mechanics.")
    coordinates, _, names = kinematics.find_joint_scalar_coordinates(".*")
    expected = list(range(kinematics.n_root_coords, int(kinematics.model.joint_coord_count)))
    if coordinates != expected:
        raise ValueError("Position runtime articulation state requires every non-root Newton coordinate to be scalar.")
    return coordinates, names


def build_task_table(
    terrain_mesh: trimesh.Trimesh,
    terrain_origins: torch.Tensor,
    cell_size: tuple[float, float],
    table_cfg,
    articulation_cfg: ArticulationCfg,
    asset_name: str,
    commands: dict[str, Commands | TerrainCommands],
    device: str,
    rng,
) -> RelativeStateTaskTable:
    """Generate terrain stances and compose spawn-target task rows.

    Args:
        terrain_mesh: Combined terrain triangle mesh [m].
        terrain_origins: Per-cell origins ``[num_rows, num_cols, 3]``.
        cell_size: ``(width, height)`` of each terrain cell [m].
        table_cfg: Visible Position mechanics, family, density, and pairing configuration.
        articulation_cfg: Declared robot articulation configuration.
        asset_name: Scene name of the robot articulation.
        commands: Command type dict from the command cfg.
        device: Torch/Warp device string.
        rng: Table-owned random state already consumed by terrain generation.

    Returns:
        Typed Position task table with canonical reset states and inspection view.
    """
    from .commands_cfg import PoseCommands, PositionCommands, TerrainCommands, VelocityCommands

    pool_spacing = table_cfg.pool_spacing
    pool_spacing_area_divisor = table_cfg.pool_spacing_area_divisor
    pool_sampling_size = table_cfg.pool_sampling_size
    pairing = table_cfg.pairing
    exclude_self_pairs = pairing.exclude_self
    max_spawns_per_cell = pairing.max_spawns_per_cell
    num_targets_per_cell = pairing.num_targets_per_cell
    num_rows, num_cols = terrain_origins.shape[0], terrain_origins.shape[1]
    num_subterrains = num_rows * num_cols
    cell_size_t = torch.tensor(cell_size, device=device)
    grid_x_range, grid_y_range = _terrain_grid_bounds(terrain_origins, cell_size)
    sampling_x_range, sampling_y_range = _centered_sampling_bounds(
        grid_x_range,
        grid_y_range,
        pool_sampling_size,
    )

    # --- Step 1: Generate one accepted terrain-stance family. ---
    total_states = _state_count_from_spacing(
        sampling_x_range,
        sampling_y_range,
        pool_spacing,
        pool_spacing_area_divisor,
    )
    kinematics = NewtonKinematics.from_articulation(table_cfg.kinematics, articulation_cfg, device)
    wp_mesh = convert_to_warp_mesh(terrain_mesh.vertices, terrain_mesh.faces, device=device)

    if len(table_cfg.families) != 1 or table_cfg.families[0].name != "terrain_stance":
        raise ValueError("Position requires exactly one 'terrain_stance' family.")
    family = table_cfg.families[0]
    execution = execute_task_family(
        family,
        PositionTerrainStanceInput(
            kinematics=kinematics,
            asset_name=asset_name,
            warp_mesh=wp_mesh,
            origin=np.zeros(3, dtype=np.float32),
            sampling_x_range=sampling_x_range,
            sampling_y_range=sampling_y_range,
            target_count=total_states,
        ),
        total_states,
        rng,
    )
    candidates = execution.candidates
    buffer = candidates.buffer
    survivors_idx = execution.selected_indices
    if survivors_idx is None:
        raise TypeError("Position task families require an explicit selection stage.")
    target_count = int(survivors_idx.numel())
    joint_pose = buffer.joint_q_result_t[survivors_idx].clone()
    contact_targets = buffer.contact_targets_t.view(buffer.max_candidates, buffer.num_contacts, 3)[
        survivors_idx
    ].clone()
    contact_valid = buffer.is_contact_t.view(buffer.max_candidates, buffer.num_contacts)[survivors_idx].clone()
    if target_count == 0:
        rejected = {
            criterion.name: int((~mask).sum()) for criterion, mask in zip(family.criteria, execution.criterion_masks)
        }
        raise RuntimeError(f"Position terrain family produced no selected states; rejected={rejected}.")

    joint_coordinates, joint_names = _newton_state_joints(candidates.kinematics)
    states = joint_pose
    num_joints = len(joint_names)
    if states.shape[1] != 7 + num_joints:
        raise RuntimeError(
            f"Retargeted state width {states.shape[1]} does not match {num_joints} declared robot joints."
        )

    # --- Step 2: Bin states by terrain cell (CSR, no padding) ---
    # Drop states that fall outside the sub-terrain grid (e.g. the flat border
    # around the terrain). The IK sampler feeds on the full mesh and has no
    # notion of sub-terrain boundaries, so a raw ``.clamp`` would silently
    # push border states into edge cells and break geometric isolation.
    grid_origin = terrain_origins[0, 0, :2].to(device) - cell_size_t * 0.5

    base_xy = states[:, :2]
    cell_xy = (base_xy - grid_origin.unsqueeze(0)) / cell_size_t.unsqueeze(0)
    row_idx = torch.floor(cell_xy[:, 0]).long()
    col_idx = torch.floor(cell_xy[:, 1]).long()
    in_grid = (row_idx >= 0) & (row_idx < num_rows) & (col_idx >= 0) & (col_idx < num_cols)
    kept_state_idx = in_grid.nonzero(as_tuple=False).squeeze(-1)
    flat_cell_kept = row_idx[in_grid] * num_cols + col_idx[in_grid]

    # CSR layout: cell_values[cell_offsets[c]:cell_offsets[c+1]] -> canonical state rows in cell c.
    sort_order = flat_cell_kept.argsort()
    cell_values = kept_state_idx[sort_order]

    counts_per_cell = torch.bincount(flat_cell_kept, minlength=num_subterrains)
    cell_offsets = torch.zeros(num_subterrains + 1, device=device, dtype=torch.long)
    cell_offsets[1:] = counts_per_cell.cumsum(0)

    # --- Step 3: Per-cell spawn × target pairing (shared across command types) ---
    # For each cell c with n_c states, the spawn set is either the full cell
    # or a downsampled subset of size ``min(max_spawns_per_cell, n_c)``. The
    # target set is independently either the full target-candidate pool or a
    # downsampled subset of size ``min(num_targets_per_cell, n_candidates)``.
    # Cell layout is built once and reused for every command type.
    pair_spawn_parts: list[torch.Tensor] = []
    pair_target_parts: list[torch.Tensor] = []
    pair_tile_parts: list[torch.Tensor] = []
    offsets_cpu = cell_offsets.cpu().tolist()
    spawn_xy = states[:, :2]
    for cell in range(num_subterrains):
        start = offsets_cpu[cell]
        end = offsets_cpu[cell + 1]
        n_c = end - start
        if n_c == 0:
            continue
        if exclude_self_pairs and n_c < 2:
            # Need at least two distinct states to form a non-self pair.
            continue
        ids = cell_values[start:end]
        if max_spawns_per_cell == 0:
            spawn_ids_in_cell = ids
            target_candidate_ids = ids
        else:
            if max_spawns_per_cell < 1:
                raise ValueError(f"max_spawns_per_cell must be >= 1 or 0 for unlimited, got {max_spawns_per_cell}.")
            n_spawns = min(int(max_spawns_per_cell), n_c)
            local_idx = grid_bucket_downsample(spawn_xy[ids], n_spawns, generator=rng.torch)
            spawn_ids_in_cell = ids[local_idx]
            spawn_mask = torch.zeros(n_c, device=device, dtype=torch.bool)
            spawn_mask[local_idx] = True
            target_candidate_ids = ids[~spawn_mask]
            if target_candidate_ids.numel() == 0:
                target_candidate_ids = ids
        if num_targets_per_cell <= 0:
            target_ids_in_cell = target_candidate_ids
        else:
            n_targets = min(int(num_targets_per_cell), int(target_candidate_ids.shape[0]))
            local_idx = grid_bucket_downsample(spawn_xy[target_candidate_ids], n_targets, generator=rng.torch)
            target_ids_in_cell = target_candidate_ids[local_idx]
        n_t = int(target_ids_in_cell.shape[0])
        spawn_ids = spawn_ids_in_cell.repeat_interleave(n_t)
        target_ids = target_ids_in_cell.repeat(int(spawn_ids_in_cell.shape[0]))
        if exclude_self_pairs:
            keep = spawn_ids != target_ids
            spawn_ids = spawn_ids[keep]
            target_ids = target_ids[keep]
        pair_count = int(spawn_ids.shape[0])
        if pair_count == 0:
            continue
        pair_spawn_parts.append(spawn_ids)
        pair_target_parts.append(target_ids)
        pair_tile_parts.append(torch.full((pair_count,), cell, device=device, dtype=torch.long))

    if pair_spawn_parts:
        pair_spawn = torch.cat(pair_spawn_parts)
        pair_target = torch.cat(pair_target_parts)
        pair_tile = torch.cat(pair_tile_parts)
    else:
        pair_spawn = torch.zeros(0, device=device, dtype=torch.long)
        pair_target = torch.zeros(0, device=device, dtype=torch.long)
        pair_tile = torch.zeros(0, device=device, dtype=torch.long)
    num_pairs_per_type = int(pair_spawn.shape[0])
    if num_pairs_per_type == 0:
        raise RuntimeError("No terrain cells contained valid retargeted states; cannot build a task table.")
    used_state_ids, inverse = torch.unique(torch.cat([pair_spawn, pair_target]), sorted=True, return_inverse=True)
    pair_spawn = inverse[:num_pairs_per_type]
    pair_target = inverse[num_pairs_per_type:]
    states = states.index_select(0, used_state_ids)
    contact_targets = contact_targets.index_select(0, used_state_ids).contiguous()
    contact_valid = contact_valid.index_select(0, used_state_ids).contiguous()

    # --- Step 4: Replicate pair layout per command type; sample per-type params ---
    ranges = torch.zeros((len(commands), 13, 2), device=device)
    mask = torch.zeros((len(commands), 12), device=device, dtype=torch.bool)
    kind = torch.zeros(len(commands), dtype=torch.int32, device=device)

    spawn_indices_list = []
    target_indices_list = []
    tile_indices_list = []
    params_list = []
    mask_list = []
    payload_flags_list = []
    row_counts = []

    for cmd_id, val in enumerate(commands.values()):
        is_terrain_command = isinstance(val, TerrainCommands)
        if is_terrain_command:
            if val.match_base_pos:
                mask[cmd_id, :3] = True
            if val.match_base_rot:
                mask[cmd_id, 3:6] = True
            ranges[cmd_id, 12, 0] = val.duration[0]
            ranges[cmd_id, 12, 1] = val.duration[1]
            kind[cmd_id] = 1 if val.match_base_rot else 0
        else:
            for data_id, name in enumerate(_COMMAND_PARAM_NAMES):
                data = getattr(val, name)
                if data is not None and isinstance(data, tuple):
                    if data_id < 12:
                        mask[cmd_id, data_id] = True
                    ranges[cmd_id, data_id, 0] = data[0]
                    ranges[cmd_id, data_id, 1] = data[1]
            if isinstance(val, PositionCommands):
                kind[cmd_id] = 0
            elif isinstance(val, PoseCommands):
                kind[cmd_id] = 1
            elif isinstance(val, VelocityCommands):
                kind[cmd_id] = 2

        range_min = ranges[cmd_id, :, 0].view(1, 13)
        range_span = ranges[cmd_id, :, 1] - ranges[cmd_id, :, 0]
        task_params = (
            torch.rand(num_pairs_per_type, 13, device=device, generator=rng.torch) * range_span.view(1, 13) + range_min
        )

        full_mask = torch.zeros(num_pairs_per_type, 12 + num_joints, device=device, dtype=torch.bool)
        full_mask[:, :12] = mask[cmd_id].view(1, 12)
        if is_terrain_command:
            full_mask[:, 12:] = True

        spawn_indices_list.append(pair_spawn)
        target_indices_list.append(pair_target)
        tile_indices_list.append(pair_tile)
        params_list.append(task_params)
        mask_list.append(full_mask)
        payload_flags = torch.zeros(num_pairs_per_type, 1, device=device, dtype=torch.bool)
        payload_flags[:, 0] = is_terrain_command
        payload_flags_list.append(payload_flags)
        row_counts.append(num_pairs_per_type)

    all_spawn = torch.cat(spawn_indices_list, dim=0)
    all_target = torch.cat(target_indices_list, dim=0)
    all_tile = torch.cat(tile_indices_list, dim=0)
    all_params = torch.cat(params_list, dim=0)
    all_masks = torch.cat(mask_list, dim=0)
    all_payload_flags = torch.cat(payload_flags_list, dim=0)

    counts_t = torch.tensor(row_counts, device=device, dtype=torch.long)
    offsets = torch.zeros(len(commands) + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts_t, dim=0)

    reset_states = ResetStateBank(
        layout=ResetStateLayout(
            names=(asset_name,),
            kinds=("articulation",),
            joint_names=(tuple(joint_names),),
            joint_offsets=(0, num_joints),
        ),
        root_pose=states[:, :7].unsqueeze(1).contiguous(),
        root_velocity=torch.zeros(states.shape[0], 1, 6, dtype=torch.float32, device=device),
        joint_position=states[:, joint_coordinates].contiguous(),
        joint_velocity=torch.zeros(states.shape[0], num_joints, dtype=torch.float32, device=device),
    )
    task_count = int(all_spawn.shape[0])
    sequence_offsets = torch.arange(task_count + 1, dtype=torch.int64, device=device).mul_(2)
    sequence_state_indices = torch.stack((all_spawn, all_target), dim=-1).reshape(-1).contiguous()
    joint_q_default = (
        torch.from_numpy(candidates.kinematics.default_joint_q).to(device=device, dtype=torch.float32).contiguous()
    )
    terrain_builder = newton.ModelBuilder()
    terrain_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            terrain_mesh.vertices,
            terrain_mesh.faces.reshape(-1),
            compute_inertia=False,
        ),
    )
    if len(candidates.kinematics.builder.joint_q) != joint_q_default.numel():
        raise RuntimeError("Retained Position robot changed its generalized-coordinate layout.")
    view = TaskTableView(
        sequences=TaskTableSequenceIndex(offsets=sequence_offsets, state_indices=sequence_state_indices),
        state_bank=reset_states,
        kinematic_view=TaskTableKinematicView(
            model_builder_state=candidates.kinematics.builder,
            model_builder_shared=terrain_builder,
            world_spacing=(0.0, 0.0, 0.0),
            joint_q_default=joint_q_default,
            root_entity_names=(asset_name,),
            root_state_indices=torch.zeros(1, dtype=torch.int64, device=device),
            root_q_indices=torch.arange(7, dtype=torch.int64, device=device).view(1, 7),
            joint_coordinate_names=tuple((asset_name, name) for name in joint_names),
            joint_state_indices=torch.arange(num_joints, dtype=torch.int64, device=device),
            joint_q_indices=torch.tensor(joint_coordinates, dtype=torch.int64, device=device),
        ),
        points=(
            TaskTablePointEvidence(
                name="contact_targets",
                points=contact_targets,
                valid=contact_valid,
                color=(0.15, 0.75, 1.0),
                radius=0.025,
            ),
        ),
        quality=TaskTableQuality(
            names=("terrain_cell",),
            values=all_tile.to(torch.float32).unsqueeze(-1).contiguous(),
            scope="sequence",
        ),
    )
    task_partition = torch.bucketize(torch.arange(task_count, device=device), offsets[1:-1], right=True)
    return RelativeStateTaskTable(
        num_tasks=task_count,
        spawn_index=all_spawn,
        target_index=all_target,
        tile_index=all_tile,
        params=all_params,
        task_mask=all_masks,
        payload_flags=all_payload_flags,
        offsets=offsets,
        task_partition=task_partition,
        kind=kind,
        states=reset_states,
        view=view,
        kinematics=candidates.kinematics,
        contact_body_names=candidates.contact_body_names,
        contact_body_ids=candidates.contact_body_ids,
    )
