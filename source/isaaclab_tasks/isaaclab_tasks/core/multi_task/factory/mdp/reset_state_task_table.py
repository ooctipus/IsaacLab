# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the simulator-free Factory reset-state task table."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch

from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from isaaclab_tasks.core.multi_task.mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTablePointEvidence,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
    make_task_table_rng,
)
from isaaclab_tasks.core.multi_task.utils.grid_downsample import extract_features, grid_bucket_downsample

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

    from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg


@dataclass(frozen=True, slots=True)
class FactoryResetStateTaskTable:
    """Canonical Factory states and paired spawn/target endpoints."""

    states: ResetStateBank
    """Immutable entity-major reset-state bank."""

    view: TaskTableView
    """Exact endpoint sequences and retained Newton mechanics."""

    state_tag_indices: torch.Tensor
    """Placement-strategy tag per state, shape [num_states]."""

    state_board_indices: torch.Tensor
    """Board-configuration index per state, shape [num_states]."""

    state_tag_names: tuple[str, ...]
    """Placement-strategy names indexed by state_tag_indices."""

    state_family_indices: torch.Tensor
    """Declared task-family index per state, shape [num_states]."""

    state_family_names: tuple[str, ...]
    """Declared task-family names indexed by state_family_indices."""

    state_coords: torch.Tensor
    """Curriculum feature coordinates per state, shape [num_states, feature_dim]."""

    spawn_index: torch.Tensor
    """Spawn-state index per task, shape [num_tasks]."""

    target_index: torch.Tensor
    """Target-state index per task, shape [num_tasks]."""

    @property
    def num_states(self) -> int:
        """Number of canonical states."""
        return self.states.row_count

    @property
    def num_tasks(self) -> int:
        """Number of paired spawn/target tasks."""
        return int(self.spawn_index.shape[0])

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample task indices uniformly on the table device."""
        return torch.randint(0, self.num_tasks, (count,), device=self.spawn_index.device)

    def gather(self, task_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return canonical spawn and target state indices for task rows."""
        return self.spawn_index[task_rows], self.target_index[task_rows]


def factory_family_quotas(rows_per_board: int, families) -> tuple[int, ...]:
    """Allocate exact per-board family quotas by stable largest remainder."""
    if type(rows_per_board) is not int or rows_per_board < 1:
        raise ValueError("Factory rows_per_board must be a positive integer.")
    if not families:
        raise ValueError("Factory requires at least one placement family.")
    names = tuple(family.name for family in families)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Factory family names must be nonempty and unique.")
    fractions = tuple(float(family.fraction) for family in families)
    if any(not math.isfinite(value) or value < 0.0 for value in fractions):
        raise ValueError("Factory family fractions must be finite and non-negative.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"Factory family fractions must sum to one, got {sum(fractions)}.")
    if any(
        not math.isfinite(float(family.candidate_oversample)) or float(family.candidate_oversample) < 1.0
        for family in families
    ):
        raise ValueError("Factory family candidate_oversample values must be finite and at least one.")

    ideal = tuple(rows_per_board * fraction for fraction in fractions)
    quotas = [math.floor(value) for value in ideal]
    remaining = rows_per_board - sum(quotas)
    order = sorted(range(len(families)), key=lambda index: (-(ideal[index] - quotas[index]), index))
    for index in order[:remaining]:
        quotas[index] += 1
    return tuple(quotas)


def build_factory_reset_state_task_table(
    command_cfg: StateCommandCfg,
    scene_cfg: InteractiveSceneCfg,
    device: str,
) -> FactoryResetStateTaskTable:
    """Build one Factory table from declared command and scene configuration.

    Args:
        command_cfg: Factory state-command configuration.
        scene_cfg: Resolved scene configuration containing every reset asset.
        device: Torch and Newton construction device.

    Returns:
        Canonical states and paired task endpoints.
    """
    table_cfg = command_cfg.task_table
    reset_assets = tuple(command_cfg.reset_assets)
    if not reset_assets:
        raise ValueError("Factory reset_assets must declare at least one entity.")

    geometry_cfg = table_cfg.geometry
    _validate_reset_asset_owners(reset_assets, geometry_cfg)
    rng = make_task_table_rng(int(table_cfg.seed), str(device))
    (
        state,
        state_tag_indices,
        state_board_indices,
        state_tag_names,
        state_family_indices,
        state_family_names,
        pad_targets,
        quality,
        builder,
    ) = _precollect_from_builder(table_cfg, geometry_cfg, scene_cfg, str(device), reset_assets, rng)
    feature_rows = _feature_rows(state)
    state_coords = extract_features(feature_rows, table_cfg.state_table_fps_features).contiguous()
    spawn_index, target_index = _pair_within_boards(
        table_cfg,
        state_coords,
        state_tag_indices,
        state_board_indices,
        state_tag_names,
        rng.torch,
    )
    view = _task_table_view(builder, state, spawn_index, target_index, pad_targets, quality)
    return FactoryResetStateTaskTable(
        states=state,
        view=view,
        state_tag_indices=state_tag_indices,
        state_board_indices=state_board_indices,
        state_tag_names=state_tag_names,
        state_family_indices=state_family_indices,
        state_family_names=state_family_names,
        state_coords=state_coords,
        spawn_index=spawn_index,
        target_index=target_index,
    )


def _validate_reset_asset_owners(reset_assets: tuple[str, ...], geometry_cfg) -> None:
    """Require one declared owner for every Factory reset entity."""
    if len(reset_assets) != len(set(reset_assets)):
        raise ValueError(f"Factory reset_assets contains duplicate names: {reset_assets}.")
    board_cfg = geometry_cfg.board
    roles = (
        geometry_cfg.robot.asset_cfg.name,
        board_cfg.board_asset_cfg.name,
        geometry_cfg.held_asset_cfg.name,
        *tuple(board_cfg.fixed_asset_map),
    )
    if len(roles) != len(set(roles)):
        raise ValueError(f"Factory robot, board, held, and board-attached entity owners must be unique: {roles}.")
    fixed_name = board_cfg.fixed_asset_cfg.name
    if fixed_name not in board_cfg.fixed_asset_map:
        raise ValueError(f"Factory primary fixed asset {fixed_name!r} is absent from fixed_asset_map.")
    missing = sorted(set(roles) - set(reset_assets))
    extra = sorted(set(reset_assets) - set(roles))
    if missing or extra:
        raise ValueError(
            "Factory reset_assets must exactly contain robot, board, held, and every board-attached entity; "
            f"missing={missing}, extra={extra}."
        )


def _precollect_from_builder(table_cfg, geometry_cfg, scene_cfg, device: str, reset_assets: tuple[str, ...], rng):
    """Run the simulator-free builder and encode geometric survivors canonically."""
    from ..retarget.task_table_builder import FactoryTaskTableBuilder

    builder = FactoryTaskTableBuilder(
        table_cfg.kinematics,
        geometry_cfg,
        scene_cfg,
        device,
        rng,
    )
    result = builder.build_family_table(int(table_cfg.rows_per_board), tuple(table_cfg.families), rng)
    geometry = builder.geometry

    coordinate_indices, _, coordinate_names = geometry.kinematics.find_joint_scalar_coordinates(".*")
    coordinate_names = tuple(coordinate_names)
    coordinate_indices = tuple(coordinate_indices)
    robot_name = geometry_cfg.robot.asset_cfg.name
    layout = _reset_state_layout(scene_cfg, reset_assets, robot_name, coordinate_names)

    state = _build_reset_state_bank(
        geometry_cfg,
        scene_cfg,
        layout,
        result,
        coordinate_indices,
    )
    state_tag_indices = result.tag.contiguous()
    state_board_indices = result.board_index.contiguous()
    state_tag_names = tuple(result.tag_names)
    state_family_indices = result.task_family.contiguous()
    state_family_names = result.task_family_names
    pad_targets = result.pad_targets.contiguous()
    quality = TaskTableQuality(result.quality_names, result.quality.contiguous(), scope="state")
    return (
        state,
        state_tag_indices,
        state_board_indices,
        state_tag_names,
        state_family_indices,
        state_family_names,
        pad_targets,
        quality,
        builder,
    )


def _reset_state_layout(
    scene_cfg,
    reset_assets: tuple[str, ...],
    robot_name: str,
    robot_joint_names: tuple[str, ...],
) -> ResetStateLayout:
    """Resolve canonical entity and joint axes in declared reset order."""
    kinds: list[str] = []
    joint_names: list[tuple[str, ...]] = []
    offsets = [0]
    for name in reset_assets:
        if not hasattr(scene_cfg, name):
            raise ValueError(f"Factory reset asset {name!r} is absent from the declared scene config.")
        asset_cfg = getattr(scene_cfg, name)
        if isinstance(asset_cfg, ArticulationCfg):
            if name != robot_name:
                raise ValueError(
                    f"Factory reset articulation {name!r} has no declared mechanics; only {robot_name!r} is resolved."
                )
            kind = "articulation"
            names = robot_joint_names
        elif isinstance(asset_cfg, RigidObjectCfg):
            kind = "rigid_object"
            names = ()
        else:
            raise TypeError(
                f"Factory reset asset {name!r} must be ArticulationCfg or RigidObjectCfg, "
                f"got {type(asset_cfg).__name__}."
            )
        kinds.append(kind)
        joint_names.append(names)
        offsets.append(offsets[-1] + len(names))
    return ResetStateLayout(reset_assets, tuple(kinds), tuple(joint_names), tuple(offsets))


def _build_reset_state_bank(
    geometry_cfg,
    scene_cfg,
    layout: ResetStateLayout,
    result,
    coordinate_indices: tuple[int, ...],
) -> ResetStateBank:
    """Encode accepted builder rows into one entity-major state bank."""
    row_count = result.joint_q.shape[0]
    device = result.joint_q.device
    root_pose = torch.empty((row_count, layout.entity_count, 7), dtype=torch.float32, device=device)
    root_velocity = torch.empty((row_count, layout.entity_count, 6), dtype=torch.float32, device=device)
    joint_position = torch.empty((row_count, layout.joint_offsets[-1]), dtype=torch.float32, device=device)
    joint_velocity = torch.zeros_like(joint_position)

    pose_by_asset = {
        geometry_cfg.held_asset_cfg.name: result.held_pose,
        geometry_cfg.board.board_asset_cfg.name: result.board_pose,
    }
    pose_by_asset.update(result.board_asset_poses)
    robot_name = geometry_cfg.robot.asset_cfg.name

    for entity_index, (name, kind) in enumerate(zip(layout.names, layout.kinds, strict=True)):
        asset_cfg = getattr(scene_cfg, name)
        pose = pose_by_asset.get(name)
        if pose is None:
            if name != robot_name:
                raise ValueError(f"Factory reset asset {name!r} has no generated pose owner.")
            pose = torch.tensor((*asset_cfg.init_state.pos, *asset_cfg.init_state.rot), device=device).expand(
                row_count, -1
            )
        root_pose[:, entity_index].copy_(pose)
        root_velocity[:, entity_index, :3] = torch.tensor(asset_cfg.init_state.lin_vel, device=device)
        root_velocity[:, entity_index, 3:] = torch.tensor(asset_cfg.init_state.ang_vel, device=device)

        if kind != "articulation":
            continue
        joint_slice = layout.joint_slice(name)
        if name == robot_name:
            joint_position[:, joint_slice].copy_(result.joint_q[:, list(coordinate_indices)])

    return ResetStateBank(
        layout=layout,
        root_pose=root_pose.contiguous(),
        root_velocity=root_velocity.contiguous(),
        joint_position=joint_position.contiguous(),
        joint_velocity=joint_velocity.contiguous(),
    )


def _task_table_view(
    table_builder,
    states: ResetStateBank,
    spawn_index: torch.Tensor,
    target_index: torch.Tensor,
    pad_targets: torch.Tensor,
    quality: TaskTableQuality,
) -> TaskTableView:
    """Compose exact endpoint sequences and retained production geometry."""
    geometry_model = table_builder.geometry
    builder = newton.ModelBuilder()
    for name, (vertices, faces) in geometry_model.obstacle_geom.items():
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False),
            label=name,
        )
    builder.add_builder(geometry_model.builder)
    if len(builder.joint_q) != geometry_model.nq:
        raise RuntimeError("Retained Factory robot builder changed generalized-coordinate layout.")

    geometry = {
        table_builder.cfg.board.board_asset_cfg.name: (geometry_model.board_verts, geometry_model.board_faces),
        table_builder.cfg.held_asset_cfg.name: (geometry_model.held_verts, geometry_model.held_faces),
    }
    geometry.update(geometry_model.board_asset_geom)
    root_entity_names: list[str] = []
    root_state_indices: list[int] = []
    root_q_indices: list[list[int]] = []
    for entity_index, (name, kind) in enumerate(zip(states.layout.names, states.layout.kinds, strict=True)):
        if kind != "rigid_object":
            continue
        if name not in geometry:
            raise ValueError(f"Factory view has no retained production geometry for reset entity {name!r}.")
        vertices, faces = geometry[name]
        q_start = len(builder.joint_q)
        body = builder.add_body(label=name)
        builder.add_shape_mesh(
            body=body,
            mesh=newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False),
            label=name,
        )
        root_entity_names.append(name)
        root_state_indices.append(entity_index)
        root_q_indices.append(list(range(q_start, q_start + 7)))

    robot_name = table_builder.cfg.robot.asset_cfg.name
    coordinate_indices, _, coordinate_names = geometry_model.kinematics.find_joint_scalar_coordinates(".*")
    joint_slice = states.layout.joint_slice(robot_name)
    joint_state_indices = torch.arange(joint_slice.start, joint_slice.stop, device=states.root_pose.device)
    if joint_state_indices.numel() != len(coordinate_names):
        raise ValueError("Factory robot table joints and retained Newton coordinates differ.")

    device = states.root_pose.device
    kinematic_view = TaskTableKinematicView(
        model_builder_state=builder,
        joint_q_default=torch.tensor(builder.joint_q, dtype=torch.float32, device=device),
        root_entity_names=tuple(root_entity_names),
        root_state_indices=torch.tensor(root_state_indices, dtype=torch.int64, device=device),
        root_q_indices=torch.tensor(root_q_indices, dtype=torch.int64, device=device).reshape(-1, 7),
        joint_coordinate_names=tuple((robot_name, name) for name in coordinate_names),
        joint_state_indices=joint_state_indices,
        joint_q_indices=torch.tensor(coordinate_indices, dtype=torch.int64, device=device),
    )
    sequence_count = spawn_index.shape[0]
    offsets = torch.arange(0, 2 * (sequence_count + 1), 2, dtype=torch.int64, device=device)
    state_indices = torch.stack((spawn_index, target_index), dim=1).reshape(-1).contiguous()
    return TaskTableView(
        sequences=TaskTableSequenceIndex(offsets=offsets, state_indices=state_indices),
        state_bank=states,
        kinematic_view=kinematic_view,
        points=(TaskTablePointEvidence("grasp_pad_targets", pad_targets),),
        quality=quality,
    )


def _feature_rows(state: ResetStateBank) -> torch.Tensor:
    """Materialize existing curriculum feature input without storing it."""
    chunks: list[torch.Tensor] = []
    for entity_index, name in enumerate(state.layout.names):
        chunks.extend((state.root_pose[:, entity_index], state.root_velocity[:, entity_index]))
        joint_slice = state.layout.joint_slice(name)
        if joint_slice.start != joint_slice.stop:
            chunks.extend((state.joint_position[:, joint_slice], state.joint_velocity[:, joint_slice]))
    return torch.cat(chunks, dim=-1)


def _pair_within_boards(
    table_cfg,
    state_coords: torch.Tensor,
    state_tag_indices: torch.Tensor,
    state_board_indices: torch.Tensor,
    state_tag_names: tuple[str, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair spawn and target states only within each board configuration."""
    num_states = int(state_coords.shape[0])
    state_ids = torch.arange(num_states, device=state_coords.device, dtype=torch.long)
    targets_per_board = int(table_cfg.targets_per_board)
    if targets_per_board <= 0 or targets_per_board > int(table_cfg.rows_per_board):
        raise ValueError(
            f"targets_per_board={targets_per_board} must be in [1, rows_per_board={int(table_cfg.rows_per_board)}]."
        )
    spawn_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for board in torch.unique(state_board_indices):
        board_ids = state_ids[state_board_indices == board]
        count = min(targets_per_board, int(board_ids.shape[0]))
        local_target = grid_bucket_downsample(state_coords[board_ids], count, generator=generator).sort().values
        board_targets = board_ids[local_target]
        spawn_chunks.append(board_ids.repeat_interleave(int(board_targets.shape[0])))
        target_chunks.append(board_targets.repeat(int(board_ids.shape[0])))
    spawn_index = torch.cat(spawn_chunks)
    target_index = torch.cat(target_chunks)
    spawn_index, target_index = _filter_allowed_tag_pairs(
        table_cfg, spawn_index, target_index, state_tag_indices, state_tag_names
    )
    return spawn_index, target_index


def _filter_allowed_tag_pairs(table_cfg, spawn_index, target_index, state_tag_indices, state_tag_names):
    """Keep only task endpoints whose placement-tag pair is allowed."""
    allowed = table_cfg.allowed_tag_pairs
    if not allowed:
        return spawn_index, target_index
    name_to_id = {name: index for index, name in enumerate(state_tag_names)}
    unknown = sorted({name for pair in allowed for name in pair} - set(name_to_id))
    if unknown:
        raise ValueError(
            f"allowed_tag_pairs references unknown placement tags {unknown}; available: {state_tag_names}."
        )
    pair_ids = {(name_to_id[spawn], name_to_id[target]) for spawn, target in allowed}
    spawn_tags = state_tag_indices[spawn_index]
    target_tags = state_tag_indices[target_index]
    keep = torch.zeros(spawn_index.shape[0], dtype=torch.bool, device=spawn_index.device)
    for spawn_id, target_id in pair_ids:
        keep |= (spawn_tags == spawn_id) & (target_tags == target_id)
    if not bool(keep.any()):
        present = sorted(state_tag_names[index] for index in torch.unique(state_tag_indices).tolist())
        raise ValueError(f"allowed_tag_pairs={allowed} matched 0 task slots; tags present: {present}.")
    return spawn_index[keep], target_index[keep]
