# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a production task table with Newton FK and a Newton viewer, without simulation.

Usage::

    SCRIPT=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/scripts/inspect_task_table.py
    uv run python $SCRIPT --task Isaac-Position-v0 --command goal_point presets=anymal_c,newton_mjwarp
    uv run python $SCRIPT --task Isaac-Position-v0 --command goal_point --visualizer newton_gl presets=anymal_c
    uv run python $SCRIPT --task Isaac-Position-v0 --command goal_point --sequences 64 presets=anymal_c
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import logging
import math
import sys
import time
from collections.abc import Iterator, Sequence

_SEQUENCE_LIMIT = 16
_STATIC_REFRESH_SECONDS = 0.05
_TASK_FAMILY_LOGGER = "isaaclab_tasks.core.multi_task.mdp.commands.state_command.task_family"
_VISER_INSTALL_GUIDANCE = "Install Viser with: ./isaaclab.sh -i 'visualizer[viser]'"
_VISUALIZERS: dict[str, tuple[str, str | None, str | None]] = {
    "viser": ("ViewerViser", "viser", _VISER_INSTALL_GUIDANCE),
    "newton_gl": ("ViewerGL", None, None),
}
"""Selectable viewer backends: Newton viewer class, optional dependency module, and install guidance."""


def _resolve_viewer_type(name: str) -> type:
    """Return the Newton viewer class for one ``--visualizer`` selection."""
    class_name, optional_module, guidance = _VISUALIZERS[name]
    if optional_module is not None:
        try:
            importlib.import_module(optional_module)
        except ModuleNotFoundError as error:
            if error.name != optional_module:
                raise
            raise SystemExit(guidance) from error
    viewer_module = importlib.import_module("newton.viewer")
    return getattr(viewer_module, class_name)


def _timeline_events(
    frame_counts: Sequence[int], frame_dt: Sequence[float]
) -> Iterator[tuple[float, tuple[tuple[int, int], ...]]]:
    """Merge stored-frame clocks into sparse sequence/frame updates."""
    if len(frame_counts) != len(frame_dt) or not frame_counts:
        raise ValueError("Timeline frame counts and sample periods must be matching nonempty sequences.")
    if any(count < 1 for count in frame_counts) or any(step <= 0.0 for step in frame_dt):
        raise ValueError("Timeline sequences require frames and positive sample periods [s].")

    yield 0.0, tuple((sequence_index, 0) for sequence_index in range(len(frame_counts)))
    pending = [
        (step, sequence_index, 1)
        for sequence_index, (count, step) in enumerate(zip(frame_counts, frame_dt, strict=True))
        if count > 1
    ]
    heapq.heapify(pending)
    while pending:
        event_time = pending[0][0]
        updates = []
        while pending and math.isclose(pending[0][0], event_time, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _, sequence_index, frame_index = heapq.heappop(pending)
            updates.append((sequence_index, frame_index))
            next_frame = frame_index + 1
            if next_frame < frame_counts[sequence_index]:
                heapq.heappush(pending, (next_frame * frame_dt[sequence_index], sequence_index, next_frame))
        yield event_time, tuple(updates)


def _spread_sequence_indices(sequence_count: int, requested: int, device):
    """Pick up to ``requested`` sequences evenly spaced over the table.

    Domain tables often order sequences by region (Position groups pairs per
    terrain cell), so the first rows would all show one region. Even spacing
    represents every region in proportion while staying deterministic.
    """
    import torch  # noqa: PLC0415

    if sequence_count <= 0:
        raise ValueError("The task table contains no sequences.")
    if requested <= 0:
        raise ValueError("The number of inspected sequences must be positive.")
    count = min(requested, sequence_count)
    return torch.linspace(0.0, float(sequence_count - 1), count, device=device).round().to(torch.int64)


def _repeat_kinematic_model(view, world_count: int):
    """Add shared geometry once, repeat state geometry, and allocate one FK batch."""
    import newton  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    kinematics = view.kinematic_view
    builder = newton.ModelBuilder()
    if kinematics.model_builder_shared is not None:
        builder.add_builder(kinematics.model_builder_shared)
    for _ in range(world_count):
        builder.add_world(kinematics.model_builder_state)
    model = builder.finalize(device=str(kinematics.joint_q_default.device))
    coordinate_count = kinematics.joint_q_default.numel()
    if model.joint_coord_count != world_count * coordinate_count:
        raise ValueError("Shared and repeated Newton model coordinates do not match the retained q mapping.")
    joint_q = torch.empty(
        (world_count, coordinate_count),
        dtype=kinematics.joint_q_default.dtype,
        device=kinematics.joint_q_default.device,
    )
    return (
        model,
        model.state(),
        joint_q,
        wp.from_torch(joint_q.reshape(-1)),
        wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device),
    )


def _vec3_array(points):
    """Return a Warp ``vec3`` view of float32 ``[count, 3]`` Torch positions, as Newton viewers expect."""
    import warp as wp  # noqa: PLC0415

    return wp.from_torch(points.reshape(-1, 3).contiguous(), dtype=wp.vec3)


def _color_array(color: tuple[float, float, float], count: int, device):
    """Return one shared RGB color per primitive; windowed viewers require per-primitive arrays."""
    import warp as wp  # noqa: PLC0415

    return wp.full(count, wp.vec3(*color), dtype=wp.vec3, device=wp.device_from_torch(device))


def _log_evidence(viewer, view, state_rows, world_offsets, sequence_quality, *, log_constant_quality=True) -> None:
    """Log table-owned geometry and quality without reconstructing domain facts."""
    for item in view.points:
        points = item.points.expand(state_rows.numel(), -1, -1) if item.scope == "global" else item.points[state_rows]
        valid = None
        if item.valid is not None:
            valid = item.valid.expand(state_rows.numel(), -1) if item.scope == "global" else item.valid[state_rows]
        points = points + world_offsets[:, None]
        points = points.reshape(-1, 3) if valid is None else points[valid]
        viewer.log_points(
            f"evidence/{item.name}",
            _vec3_array(points),
            radii=item.radius,
            colors=_color_array(item.color, points.shape[0], points.device),
        )

    for item in view.lines:
        endpoints = (
            item.endpoints.expand(state_rows.numel(), -1, -1, -1)
            if item.scope == "global"
            else item.endpoints[state_rows]
        )
        valid = None
        if item.valid is not None:
            valid = item.valid.expand(state_rows.numel(), -1) if item.scope == "global" else item.valid[state_rows]
        endpoints = endpoints + world_offsets[:, None, None]
        endpoints = endpoints.reshape(-1, 2, 3) if valid is None else endpoints[valid]
        viewer.log_lines(
            f"evidence/{item.name}",
            _vec3_array(endpoints[:, 0]),
            _vec3_array(endpoints[:, 1]),
            colors=_color_array(item.color, endpoints.shape[0], endpoints.device),
            width=item.width,
        )

    quality = view.quality
    if quality is None:
        return
    if quality.scope != "state" and not log_constant_quality:
        return
    if quality.scope == "global":
        values = quality.values[:1]
    elif quality.scope == "sequence":
        values = sequence_quality
    else:
        values = quality.values[state_rows]
    for column, name in enumerate(quality.names):
        if quality.scope == "global":
            viewer.log_scalar(f"quality/{name}", values[0, column])
        else:
            for world_index in range(values.shape[0]):
                viewer.log_scalar(f"quality/{name}/world_{world_index}", values[world_index, column])


def _inspect_static(view, viewer_type: type, selected) -> None:
    """Show the two declared frames of each selected static sequence.

    Args:
        view: Task-table view to display.
        viewer_type: Newton viewer class to instantiate.
        selected: Sequence indices to display, shape [sequence_count], int64.
    """
    import newton  # noqa: PLC0415
    import torch  # noqa: PLC0415

    sequence_count = int(selected.numel())
    counts = view.sequences.offsets[selected + 1] - view.sequences.offsets[selected]
    if not bool(torch.all(counts == 2)):
        raise ValueError("Static task-table sequences must contain exactly two frames.")
    sequence_indices = selected.repeat_interleave(2)
    frame_indices = torch.tensor((0, 1), dtype=torch.int64, device=sequence_indices.device).repeat(sequence_count)
    state_rows = view.sequences.state_rows(sequence_indices, frame_indices)
    sequence_quality = (
        view.quality.values[sequence_indices] if view.quality is not None and view.quality.scope == "sequence" else None
    )
    model, state, joint_q, joint_q_warp, joint_qd_zero = _repeat_kinematic_model(view, 2 * sequence_count)
    viewer = viewer_type()
    try:
        viewer.set_model(model)
        viewer.set_world_offsets(view.kinematic_view.world_spacing)
        world_offsets = torch.as_tensor(viewer.world_offsets.numpy(), dtype=torch.float32, device=joint_q.device)
        view.kinematic_view.joint_q_into(view.state_bank, state_rows, joint_q)
        newton.eval_fk(model, joint_q_warp, joint_qd_zero, state)
        # Windowed viewers only process input while frames are submitted, so the
        # fixed scene is re-submitted at a low rate until the viewer closes.
        frame_time = 0.0
        first_frame = True
        while True:
            viewer.begin_frame(frame_time)
            viewer.log_state(state)
            _log_evidence(viewer, view, state_rows, world_offsets, sequence_quality, log_constant_quality=first_frame)
            viewer.end_frame()
            first_frame = False
            if not viewer.is_running():
                break
            time.sleep(_STATIC_REFRESH_SECONDS)
            frame_time += _STATIC_REFRESH_SECONDS
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def _inspect_timed(view, viewer_type: type, selected) -> None:
    """Loop exact stored frames on their table-declared physical clocks.

    Args:
        view: Task-table view to display.
        viewer_type: Newton viewer class to instantiate.
        selected: Sequence indices to display, shape [sequence_count], int64.
    """
    import newton  # noqa: PLC0415
    import torch  # noqa: PLC0415

    sequence_count = int(selected.numel())
    starts = view.sequences.offsets[selected]
    ends = view.sequences.offsets[selected + 1]
    frame_counts = tuple(int(value) for value in (ends - starts).detach().cpu())
    frame_dt = tuple(float(value) for value in view.sequences.frame_dt[selected].detach().cpu())
    sequence_starts = tuple(int(value) for value in starts.detach().cpu())
    device = view.sequences.offsets.device
    flat_indices = starts.clone()
    state_rows = torch.empty(sequence_count, dtype=torch.int64, device=device)
    sequence_quality = (
        view.quality.values[selected] if view.quality is not None and view.quality.scope == "sequence" else None
    )
    model, state, joint_q, joint_q_warp, joint_qd_zero = _repeat_kinematic_model(view, sequence_count)
    viewer = viewer_type()
    try:
        viewer.set_model(model)
        viewer.set_world_offsets(view.kinematic_view.world_spacing)
        world_offsets = torch.as_tensor(viewer.world_offsets.numpy(), dtype=torch.float32, device=joint_q.device)
        cycle_duration = max(count * step for count, step in zip(frame_counts, frame_dt, strict=True))
        cycle = 0
        wall_start = time.monotonic()
        while viewer.is_running():
            for event_index, (event_time, updates) in enumerate(_timeline_events(frame_counts, frame_dt)):
                absolute_time = cycle * cycle_duration + event_time
                delay = absolute_time - (time.monotonic() - wall_start)
                if delay > 0.0:
                    time.sleep(delay)
                if not viewer.is_running():
                    break
                for sequence_index, frame_index in updates:
                    flat_indices[sequence_index] = sequence_starts[sequence_index] + frame_index
                if view.sequences.state_indices is None:
                    state_rows.copy_(flat_indices)
                else:
                    torch.index_select(view.sequences.state_indices, 0, flat_indices, out=state_rows)
                view.kinematic_view.joint_q_into(view.state_bank, state_rows, joint_q)
                newton.eval_fk(model, joint_q_warp, joint_qd_zero, state)
                viewer.begin_frame(absolute_time)
                viewer.log_state(state)
                _log_evidence(
                    viewer,
                    view,
                    state_rows,
                    world_offsets,
                    sequence_quality,
                    log_constant_quality=cycle == 0 and event_index == 0,
                )
                viewer.end_frame()
            cycle += 1
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main(argv: list[str] | None = None) -> None:
    """Resolve one task table and inspect its first static tuples or timed clips."""
    from isaaclab_tasks.utils import setup_preset_cli  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered task whose production task table is inspected.")
    parser.add_argument("--command", required=True, help="State command that owns the production task table.")
    parser.add_argument(
        "--sequences",
        type=int,
        default=_SEQUENCE_LIMIT,
        help="Number of sequences to display, spread evenly across the table.",
    )
    parser.add_argument(
        "--visualizer",
        choices=tuple(_VISUALIZERS),
        default="viser",
        help="Newton viewer backend used to display the table.",
    )
    args, hydra_args = setup_preset_cli(parser, argv)
    viewer_type = _resolve_viewer_type(args.visualizer)

    import isaaclab_tasks  # noqa: F401, PLC0415
    from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: PLC0415

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *hydra_args]
        env_cfg, _ = resolve_task_config(args.task, "")
    finally:
        sys.argv = original_argv

    try:
        command_cfg = getattr(env_cfg.commands, args.command)
    except AttributeError as error:
        raise ValueError(f"Task has no command named {args.command!r}.") from error
    family_logger = logging.getLogger(_TASK_FAMILY_LOGGER)
    previous_level = family_logger.level
    previous_disabled = family_logger.disabled
    previous_propagate = family_logger.propagate
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    family_logger.addHandler(handler)
    family_logger.setLevel(logging.INFO)
    family_logger.disabled = False
    family_logger.propagate = False
    started = time.perf_counter()
    try:
        view = command_cfg.task_table.build_inspection_view(
            command_cfg, env_cfg.scene, env_cfg.sim.device, sequence_limit=args.sequences
        )
    finally:
        family_logger.removeHandler(handler)
        handler.close()
        family_logger.setLevel(previous_level)
        family_logger.disabled = previous_disabled
        family_logger.propagate = previous_propagate
    build_seconds = time.perf_counter() - started
    print(
        f"Task table built: seconds={build_seconds:.3f} states={view.state_bank.row_count} "
        f"sequences={view.sequences.sequence_count} frames={view.sequences.frame_count}"
    )
    selected = _spread_sequence_indices(view.sequences.sequence_count, args.sequences, view.sequences.offsets.device)
    print(f"Inspecting {selected.numel()} of {view.sequences.sequence_count} sequences spread evenly across the table.")
    if view.sequences.is_timed:
        _inspect_timed(view, viewer_type, selected)
    else:
        _inspect_static(view, viewer_type, selected)


if __name__ == "__main__":
    main()
