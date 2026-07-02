# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure exact MotionTaskTable storage and lookup paths on one canonical source split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from motion_environment_identity import motion_composition_runtime_dependencies

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    _SIMULATOR_JOINT_NAMES as G1_LIVE_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_BODY_NAMES,
    G1_BEHAVIOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.smpl import (
    _SMPL_SIMULATOR_BODY_NAMES as SMPL_LIVE_BODY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.smpl import (
    _SMPL_SIMULATOR_JOINT_NAMES as SMPL_LIVE_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionTaskTable, MotionTaskTableCfg
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_state_payload import _MotionReferenceResolver
from isaaclab_tasks.core.multi_task.motion.trajectory.g1 import G1LafanFrameBuilder
from isaaclab_tasks.core.multi_task.motion.trajectory.smpl import SmplHumEnvFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH

_COMMON_CONSTRUCTION_MODULES = (
    "isaaclab.sim.schemas.schemas_cfg",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics",
    "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics_cfg",
    "isaaclab_tasks.core.multi_task.motion.config.source_skeletons",
    "isaaclab_tasks.core.multi_task.motion.config.sources",
    "isaaclab_tasks.core.multi_task.motion.data._identity",
    "isaaclab_tasks.core.multi_task.motion.data.clip_index",
    "isaaclab_tasks.core.multi_task.motion.data.importers._hashing",
    "isaaclab_tasks.core.multi_task.motion.data.sample_grid",
    "isaaclab_tasks.core.multi_task.motion.data.skeleton",
    "isaaclab_tasks.core.multi_task.motion.frames",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_state_payload",
    "isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table",
)
_PROFILE_CONSTRUCTION_MODULES = {
    "g1_lafan": (
        "isaaclab_tasks.core.multi_task.motion.config.robots.g1",
        "isaaclab_tasks.core.multi_task.motion.data.importers.bfm_g1_joblib",
        "isaaclab_tasks.core.multi_task.motion.trajectory.g1",
    ),
    "smpl_cmu": (
        "isaaclab_assets.robots.smpl.smpl_constants",
        "isaaclab_tasks.core.multi_task.motion.config.robots.smpl",
        "isaaclab_newton.sim.schemas.schemas_cfg",
        "isaaclab_tasks.core.multi_task.motion.data.importers.humenv_hdf5",
        "isaaclab_tasks.core.multi_task.motion.trajectory.smpl",
    ),
}

_RESET_FIELDS = (
    "root_position",
    "root_rotation",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_position",
    "joint_velocity",
)

_G1_BODY_BY_JOINT = dict(zip(G1_BEHAVIOR_JOINT_NAMES, G1_BEHAVIOR_BODY_NAMES[1:], strict=True))
G1_LIVE_BODY_NAMES = (
    G1_BEHAVIOR_BODY_NAMES[0],
    *(_G1_BODY_BY_JOINT[joint_name] for joint_name in G1_LIVE_JOINT_NAMES),
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module_sha256(module_name: str) -> str:
    """Hash one exact Python source module in the measured construction path."""
    module = importlib.import_module(module_name)
    path = Path(inspect.getsourcefile(module) or "")
    if not path.is_file():
        raise RuntimeError(f"Cannot locate construction source module {module_name!r}.")
    return _sha256(path)


def _symbol_sha256(value: object) -> str:
    """Hash one exact Python symbol instead of its unrelated module neighbors."""
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _callable_name(value: object) -> str:
    """Return one stable module-qualified callable name."""
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(f"Table construction callable lacks a stable Python identity: {value!r}.")
    return f"{module}:{name}"


def _resolved_construction_contract(
    preset: str,
    cfg: MotionImitationEnvCfg,
    table_cfg: MotionTaskTableCfg,
    split,
) -> dict[str, object]:
    """Project the exact resolved inputs that determine one table measurement."""
    source = table_cfg.source
    skeleton = source.build_skeleton()
    grid = table_cfg.expert_sample_grid
    return {
        "preset": preset,
        "control_dt_seconds": float(cfg.sim.dt * cfg.decimation),
        "source": {
            "identifier": source.identifier,
            "format": source.format,
            "semantic_level": source.semantic_level,
            "source_fps": source.source_fps,
            "license": source.license,
            "open_source": _callable_name(source.open_source),
            "skeleton_factory": _callable_name(source.skeleton_factory),
            "skeleton_identity_sha256": skeleton.identity_sha256,
            "split": {
                "name": split.name,
                "artifact": split.artifact,
                "artifact_sha256": split.artifact_sha256,
                "source_content_sha256": split.source_content_sha256,
                "clip_count": split.clip_count,
                "frame_count": split.frame_count,
            },
        },
        "table": {
            "frame_builder_factory": _callable_name(table_cfg.frame_builder_factory),
            "reference_kinematics_factory": _callable_name(table_cfg.reference_kinematics_factory),
            "expert_sample_grid": {
                "mode": grid.mode.value,
                "step_seconds": grid.step_seconds,
            },
            "task_row_mode": table_cfg.task_row_mode,
            "task_sampling_law": table_cfg.task_sampling_law,
            "reset_sources": [list(value) for value in table_cfg.reset_sources],
        },
    }


def _construction_code_identity(
    preset: str,
    source_importer_type: type,
    frame_builder_type: type,
    reference_artifact_root: Path | None,
    resolved_contract: dict[str, object],
) -> dict[str, object]:
    """Close one table measurement over exact code and reference-model bytes."""
    modules = {
        *_COMMON_CONSTRUCTION_MODULES,
        *_PROFILE_CONSTRUCTION_MODULES[preset],
        source_importer_type.__module__,
        frame_builder_type.__module__,
    }
    python_sources = {name: _module_sha256(name) for name in sorted(modules)}
    python_sources["benchmark_motion_task_table"] = _sha256(Path(__file__).resolve())
    python_symbols = {
        "isaaclab_tasks.core.multi_task.motion.mdp.commands.MotionTaskTableCfg": _symbol_sha256(MotionTaskTableCfg)
    }
    if preset == "g1_lafan":
        if reference_artifact_root is None:
            raise ValueError("g1_lafan requires --reference_artifact_root.")
        reference_path = reference_artifact_root.expanduser().resolve() / "humanoidverse/data/robots/g1/g1_29dof.xml"
    else:
        reference_path = Path(SMPL_HUMENV_MJCF_PATH).resolve()
    reference_assets = {f"reference/{reference_path.name}": _sha256(reference_path)}
    identity = {
        "python_sources": python_sources,
        "python_symbols": python_symbols,
        "reference_assets": reference_assets,
        "resolved_construction": resolved_contract,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "bundle_sha256": hashlib.sha256(canonical).hexdigest()}


def _rss_bytes() -> int:
    """Return current resident host memory [bytes] on Linux."""
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    """Return peak resident host memory [bytes]."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rates(
    operation: Callable[[], None],
    *,
    units_per_call: int,
    unit: str,
    warmup: int,
    iterations: int,
    samples: int,
    device: torch.device,
) -> dict[str, object]:
    """Measure repeated eager operations and retain sample-level uncertainty."""
    for _ in range(warmup):
        operation()
    _synchronize(device)

    rates = []
    seconds = []
    for _ in range(samples):
        gc.collect()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        _synchronize(device)
        elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        rates.append(units_per_call * iterations / elapsed)
    ordered = sorted(rates)
    return {
        "unit": unit,
        "units_per_call": units_per_call,
        "warmup_calls": warmup,
        "iterations_per_sample": iterations,
        "sample_count": samples,
        "sample_seconds": seconds,
        "sample_rates": rates,
        "minimum_rate": ordered[0],
        "median_rate": statistics.median(ordered),
        "maximum_rate": ordered[-1],
    }


def _frame_builder(
    preset: str,
    source_cfg,
    reference_artifact_root: Path | None,
    device: torch.device,
):
    """Construct the exact source-to-live-order builder without a simulator."""
    if preset == "g1_lafan":
        if reference_artifact_root is None:
            raise ValueError("g1_lafan requires --reference_artifact_root.")
        path = reference_artifact_root.expanduser().resolve() / "humanoidverse/data/robots/g1/g1_29dof.xml"
        reference = NewtonKinematics(
            NewtonKinematicsCfg(
                usd_path=None,
                mjcf_path=str(path),
                device=str(device),
                collapse_fixed_joints=False,
            )
        )
        return G1LafanFrameBuilder(
            source_skeleton=source_cfg.build_skeleton(),
            reference_kinematics=reference,
            live_joint_names=G1_LIVE_JOINT_NAMES,
            live_body_names=G1_LIVE_BODY_NAMES,
        )

    reference = NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=None,
            mjcf_path=str(SMPL_HUMENV_MJCF_PATH),
            device=str(device),
            collapse_fixed_joints=False,
        )
    )
    return SmplHumEnvFrameBuilder(
        source_skeleton=source_cfg.build_skeleton(),
        reference_kinematics=reference,
        live_joint_names=SMPL_LIVE_JOINT_NAMES,
        live_body_names=SMPL_LIVE_BODY_NAMES,
    )


def _storage_pareto(preset: str, frames: int, table: MotionTaskTable) -> list[dict[str, object]]:
    """Return exact byte consequences of reviewed representation choices."""
    if preset == "smpl_cmu":
        tiers = (
            ("reset_state_only", 151, True, False, "derived_if_stored_alone"),
            ("expert_observation_only", 358, False, True, "derived_if_stored_alone"),
            ("production_state_and_observation", 509, True, True, "materialized"),
        )
    else:
        tiers = (
            ("reset_state_only", 71, True, False, "derived_if_stored_alone"),
            ("production_shared_root_reference", 461, True, True, "materialized"),
            ("expert_projection_if_stored_alone", 527, False, True, "derived_if_stored_alone"),
            ("production_plus_duplicate_expert_projection", 988, True, True, "rejected_duplicate_corpus"),
        )

    result = []
    for name, width, can_reset, direct_expert, status in tiers:
        dense_bytes = frames * width * torch.tensor([], dtype=torch.float32).element_size()
        if status == "materialized" and dense_bytes != table.frames.memory_bytes:
            raise RuntimeError(f"The production {preset} tier differs from MotionTaskTable trajectory bytes.")
        result.append(
            {
                "name": name,
                "status": status,
                "float32_scalars_per_frame": width,
                "dense_frame_bytes": dense_bytes,
                "can_reset_simulator": can_reset,
                "direct_expert_rows": direct_expert,
            }
        )
    return result


def _canonical_batch(table: MotionTaskTable, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic valid clip, local-frame, and time rows."""
    rows = torch.arange(batch_size, dtype=torch.int64, device=table.device)
    clip_indices = torch.remainder(rows, len(table.clip_index.clips))
    frame_counts = table.frame_counts.index_select(0, clip_indices)
    local_frames = torch.remainder(rows * 7919, frame_counts)
    fractions = torch.remainder(rows * 104729, 1_000_003).to(torch.float32) / 1_000_003.0
    last_time = (frame_counts - 1).to(torch.float32) / table.source_fps.index_select(0, clip_indices)
    return clip_indices, local_frames, fractions * last_time


def _parity(
    table: MotionTaskTable,
    resolver: _MotionReferenceResolver,
    clip_indices: torch.Tensor,
    time_seconds: torch.Tensor,
    field_names: tuple[str, ...],
) -> dict[str, object]:
    """Compare fixed runtime lookup against the allocating table oracle."""
    resolver.resolve(field_names)
    oracle = table.reference_view(clip_indices, time_seconds)
    maximum_absolute_error = {}
    for name in field_names:
        expected = oracle.field(name)
        actual = resolver.reference[name]
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        maximum_absolute_error[name] = float(torch.max(torch.abs(actual - expected)))
    return {
        "passed": True,
        "rtol": 2.0e-5,
        "atol": 2.0e-6,
        "maximum_absolute_error_by_field": maximum_absolute_error,
    }


def _measure(args: argparse.Namespace) -> dict[str, object]:
    """Construct one exact task table and benchmark every reviewed lookup boundary."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={args.preset})
    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if args.motion_split == "train" else source_cfg.evaluation
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(args.threads)

    baseline_rss = _rss_bytes()
    source_path = args.source_artifact_root.expanduser().resolve() / split.artifact
    source = source_cfg.open_split(args.source_artifact_root, split)
    source_importer_type = type(source)
    after_open_rss = _rss_bytes()
    index = source.inspect()
    after_inspect_rss = _rss_bytes()
    if (
        index.source_content_sha256 != split.source_content_sha256
        or len(index.clips) != split.clip_count
        or index.total_frames != split.frame_count
    ):
        raise RuntimeError("The opened source differs from the selected canonical split.")

    builder = _frame_builder(args.preset, source_cfg, args.reference_artifact_root, device)
    started = time.perf_counter()
    table = MotionTaskTable.build(
        source,
        index,
        builder,
        table_cfg.task_row_mode,
        table_cfg.reset_sources,
        table_cfg.expert_sample_grid,
        seed=0,
        device=device,
    )
    _synchronize(device)
    construction_seconds = time.perf_counter() - started
    remaining_clips = getattr(source, "remaining_clips", 0)
    remaining_frames = getattr(source, "remaining_frames", 0)
    close = getattr(source, "close", None)
    if close is not None:
        close()
    del source
    gc.collect()
    after_build_rss = _rss_bytes()

    field_names = table.frames.available_fields
    stored_field_names = table.frames.stored_fields
    logical_row_width = sum(math.prod(table.field(name).shape[1:]) for name in field_names)
    clip_indices, local_frames, time_seconds = _canonical_batch(table, args.batch_size)
    last_time = (table.frame_counts.index_select(0, clip_indices) - 1).to(
        torch.float32
    ) / table.source_fps.index_select(0, clip_indices)
    control_dt = float(cfg.sim.dt * cfg.decimation)
    reached_time = torch.minimum(time_seconds + control_dt, last_time)
    time_program = tuple(
        torch.minimum(time_seconds + offset * control_dt, last_time) for offset in range(args.time_program_length)
    )

    resolver_time = time_seconds.clone()
    resolver = _MotionReferenceResolver(table, clip_indices, resolver_time, field_names)
    resolver.bind(field_names)
    parity = _parity(table, resolver, clip_indices, resolver_time, field_names)
    resolver_pointers = {name: value.data_ptr() for name, value in resolver.reference.items()}
    storages: dict[tuple[str, int | None, int], int] = {}
    for name in stored_field_names:
        value = table.field(name)
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr())
        size = storage.nbytes()
        previous = storages.setdefault(key, size)
        if previous != size:
            raise RuntimeError("Logical trajectory fields disagree on one physical storage size.")
    trajectory_bytes = sum(storages.values())
    denominator = index.total_frames * torch.tensor([], dtype=torch.float32).element_size()
    if trajectory_bytes % denominator:
        raise RuntimeError("Physical trajectory storage is not an integral float32 width per source frame.")
    physical_row_width = trajectory_bytes // denominator
    root_reference_aliases = table.frames.body_position is not None and all(
        table.field(root_name).untyped_storage().data_ptr() == table.field(body_name).untyped_storage().data_ptr()
        for root_name, body_name in zip(
            _RESET_FIELDS[:4],
            ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity"),
            strict=True,
        )
    )
    exact_capacity = all(table.field(name).shape[0] == index.total_frames for name in stored_field_names)
    storage_contract_passed = exact_capacity and table.frames.memory_bytes == trajectory_bytes
    storage_contract_passed &= root_reference_aliases if args.preset == "g1_lafan" else not root_reference_aliases
    if not storage_contract_passed:
        raise RuntimeError("MotionTaskTable physical storage differs from its preset contract.")

    reset_rows = table.clip_offsets.index_select(0, clip_indices) + local_frames
    reset_sources = table_cfg.reset_sources
    rebound = MotionTaskTable.from_storage(
        table.clip_index,
        table.frames,
        table.joint_names,
        table.reference_frame_names,
        table.frame_builder_version,
        table.frame_builder_identity_sha256,
        table.task_row_mode,
        reset_sources,
        table.expert_sample_grid,
        seed=table.seed,
    )
    if rebound.frames is not table.frames or rebound.cache_identity != table.cache_identity:
        raise RuntimeError("Exact-capacity table binding changed trajectory storage or identity.")
    del rebound

    sink: object = None
    program_index = 0

    def field_lookup() -> None:
        nonlocal sink, program_index
        sink = table.field(field_names[program_index % len(field_names)])
        program_index += 1

    def reset_gather() -> None:
        nonlocal sink
        sink = tuple(torch.index_select(table.field(name), 0, reset_rows) for name in _RESET_FIELDS)

    def reference_oracle() -> None:
        nonlocal sink, program_index
        selected_time = time_program[program_index % len(time_program)]
        program_index += 1
        view = table.reference_view(clip_indices, selected_time)
        sink = tuple(view.field(name) for name in field_names)

    def fixed_reference() -> None:
        nonlocal program_index
        resolver_time.copy_(time_program[program_index % len(time_program)])
        program_index += 1
        resolver.resolve(field_names)

    def current_reached_reference() -> None:
        nonlocal sink
        current = table.reference_view(clip_indices, time_seconds)
        reached = table.reference_view(clip_indices, reached_time)
        sink = tuple((current.field(name), reached.field(name)) for name in field_names)

    def bind_storage() -> None:
        nonlocal sink
        sink = MotionTaskTable.from_storage(
            table.clip_index,
            table.frames,
            table.joint_names,
            table.reference_frame_names,
            table.frame_builder_version,
            table.frame_builder_identity_sha256,
            table.task_row_mode,
            reset_sources,
            table.expert_sample_grid,
            seed=table.seed,
        )

    throughput = {
        "named_field_lookup": _rates(
            field_lookup,
            units_per_call=1,
            unit="calls_per_second",
            warmup=args.warmup,
            iterations=args.field_iterations,
            samples=args.samples,
            device=device,
        ),
        "reset_state_gather": _rates(
            reset_gather,
            units_per_call=args.batch_size,
            unit="rows_per_second",
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            device=device,
        ),
        "allocating_reference_oracle": _rates(
            reference_oracle,
            units_per_call=args.batch_size,
            unit="rows_per_second",
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            device=device,
        ),
        "fixed_runtime_reference": _rates(
            fixed_reference,
            units_per_call=args.batch_size,
            unit="rows_per_second",
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            device=device,
        ),
        "current_reached_reference": _rates(
            current_reached_reference,
            units_per_call=2 * args.batch_size,
            unit="transition_rows_per_second",
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            device=device,
        ),
        "bind_exact_capacity_table": _rates(
            bind_storage,
            units_per_call=1,
            unit="binds_per_second",
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            device=device,
        ),
    }
    del sink
    pointer_stable = resolver_pointers == {name: value.data_ptr() for name, value in resolver.reference.items()}
    if not pointer_stable:
        raise RuntimeError("The fixed runtime resolver replaced output storage during lookup.")

    runtime_dependencies = motion_composition_runtime_dependencies(args.preset)
    runtime_dependencies_sha256 = hashlib.sha256(
        json.dumps(runtime_dependencies, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_bytes = table.memory_bytes - trajectory_bytes
    task_sampling_law = table_cfg.task_sampling_law
    if task_sampling_law != table.task_sampling_law:
        raise RuntimeError("Resolved table sampling law differs from the runtime table property.")
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {
        "schema": "forward_backward_phase3c_motion_task_table_lookup_v4",
        "code_identity": _construction_code_identity(
            args.preset,
            source_importer_type,
            type(builder),
            args.reference_artifact_root,
            _resolved_construction_contract(args.preset, cfg, table_cfg, split),
        ),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dependencies": runtime_dependencies,
            "dependencies_sha256": runtime_dependencies_sha256,
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
        },
        "source": {
            "preset": args.preset,
            "split": split.name,
            "artifact": split.artifact,
            "artifact_sha256": _sha256(source_path),
            "source_content_sha256": index.source_content_sha256,
            "clips": len(index.clips),
            "frames": index.total_frames,
            "remaining_clips_after_build": remaining_clips,
            "remaining_frames_after_build": remaining_frames,
        },
        "task_table": {
            "identity_sha256": table.cache_identity,
            "frame_builder_identity_sha256": table.frame_builder_identity_sha256,
            "frame_builder_version": table.frame_builder_version,
            "task_row_mode": table.task_row_mode,
            "task_sampling_law": task_sampling_law,
            "reset_source_names": table.reset_source_names,
            "reset_source_probabilities": table.reset_source_probabilities.cpu().tolist(),
            "field_names": field_names,
            "stored_field_names": stored_field_names,
            "field_shapes": {name: list(table.field(name).shape[1:]) for name in field_names},
            "reset_fields": _RESET_FIELDS,
            "logical_row_width": logical_row_width,
            "physical_row_width": physical_row_width,
            "trajectory_bytes": trajectory_bytes,
            "compact_metadata_bytes": metadata_bytes,
            "resident_bytes": table.memory_bytes,
            "root_reference_aliases": root_reference_aliases,
            "unique_physical_storage_count": len(storages),
            "storage_contract_passed": storage_contract_passed,
            "storage_pareto": _storage_pareto(args.preset, index.total_frames, table),
        },
        "construction": {
            "seconds": construction_seconds,
            "frames_per_second": index.total_frames / construction_seconds,
            "host_rss_bytes": {
                "baseline": baseline_rss,
                "after_source_open": after_open_rss,
                "after_source_inspect": after_inspect_rss,
                "after_table_build_and_source_release": after_build_rss,
                "process_peak": _peak_rss_bytes(),
            },
            "cuda_memory": cuda_memory,
        },
        "parameters": {
            "batch_size": args.batch_size,
            "control_dt_seconds": control_dt,
            "time_program_length": args.time_program_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "field_iterations": args.field_iterations,
            "samples": args.samples,
        },
        "correctness": {
            "fixed_reference_matches_allocating_oracle": parity,
            "fixed_reference_output_pointers_stable": pointer_stable,
        },
        "throughput": throughput,
    }


def main() -> None:
    """Parse one benchmark request and atomically persist the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smpl_cmu", "g1_lafan"), required=True)
    parser.add_argument("--motion_split", choices=("train", "evaluation"), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--time_program_length", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--field_iterations", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "threads",
        "batch_size",
        "time_program_length",
        "warmup",
        "iterations",
        "field_iterations",
        "samples",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")

    record = _measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
