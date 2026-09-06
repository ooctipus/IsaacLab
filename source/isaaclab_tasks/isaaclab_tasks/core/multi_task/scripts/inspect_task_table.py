# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a production task table with Newton FK and Viser, without simulation."""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import sys
import time
from collections.abc import Iterator, Sequence

_SEQUENCE_LIMIT = 16
_TASK_FAMILY_LOGGER = "isaaclab_tasks.core.multi_task.mdp.commands.state_command.task_family"
_VISER_INSTALL_GUIDANCE = "Install Viser with: ./isaaclab.sh -i 'visualizer[viser]'"


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


def _log_evidence(viewer, view, state_rows, world_offsets, sequence_quality, *, log_constant_quality=True) -> None:
    """Log table-owned geometry and quality without reconstructing domain facts."""
    import warp as wp  # noqa: PLC0415

    for item in view.points:
        points = item.points.expand(state_rows.numel(), -1, -1) if item.scope == "global" else item.points[state_rows]
        valid = None
        if item.valid is not None:
            valid = item.valid.expand(state_rows.numel(), -1) if item.scope == "global" else item.valid[state_rows]
        points = points + world_offsets[:, None]
        viewer.log_points(
            f"evidence/{item.name}",
            wp.from_torch((points.reshape(-1, 3) if valid is None else points[valid]).contiguous()),
            radii=item.radius,
            colors=item.color,
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
            wp.from_torch(endpoints[:, 0].contiguous()),
            wp.from_torch(endpoints[:, 1].contiguous()),
            colors=item.color,
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


def _inspect_static(view, viewer_type: type, sequence_count: int) -> None:
    """Show the two declared frames of each selected static sequence."""
    import newton  # noqa: PLC0415
    import torch  # noqa: PLC0415

    counts = view.sequences.offsets[1 : sequence_count + 1] - view.sequences.offsets[:sequence_count]
    if not bool(torch.all(counts == 2)):
        raise ValueError("Static task-table sequences must contain exactly two frames.")
    sequence_indices = torch.arange(
        sequence_count, dtype=torch.int64, device=view.sequences.offsets.device
    ).repeat_interleave(2)
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
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        _log_evidence(viewer, view, state_rows, world_offsets, sequence_quality)
        viewer.end_frame()
        while viewer.is_running():
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def _inspect_timed(view, viewer_type: type, sequence_count: int) -> None:
    """Loop exact stored frames on their table-declared physical clocks."""
    import newton  # noqa: PLC0415
    import torch  # noqa: PLC0415

    offsets = view.sequences.offsets[: sequence_count + 1].detach().cpu()
    frame_counts = tuple(int(value) for value in offsets[1:] - offsets[:-1])
    frame_dt = tuple(float(value) for value in view.sequences.frame_dt[:sequence_count].detach().cpu())
    sequence_starts = tuple(int(value) for value in offsets[:-1])
    device = view.sequences.offsets.device
    flat_indices = view.sequences.offsets[:sequence_count].clone()
    state_rows = torch.empty(sequence_count, dtype=torch.int64, device=device)
    sequence_quality = (
        view.quality.values[:sequence_count] if view.quality is not None and view.quality.scope == "sequence" else None
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
    args, hydra_args = setup_preset_cli(parser, argv)

    try:
        import viser  # noqa: F401, PLC0415
    except ModuleNotFoundError as error:
        if error.name != "viser":
            raise
        raise SystemExit(_VISER_INSTALL_GUIDANCE) from error
    from newton.viewer import ViewerViser  # noqa: PLC0415

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
            command_cfg, env_cfg.scene, env_cfg.sim.device, sequence_limit=_SEQUENCE_LIMIT
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
    sequence_count = min(_SEQUENCE_LIMIT, view.sequences.sequence_count)
    if sequence_count == 0:
        raise ValueError("The task table contains no sequences.")
    if view.sequences.is_timed:
        _inspect_timed(view, ViewerViser, sequence_count)
    else:
        _inspect_static(view, ViewerViser, sequence_count)


if __name__ == "__main__":
    main()
