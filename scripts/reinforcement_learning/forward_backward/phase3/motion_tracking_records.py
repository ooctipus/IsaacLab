# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline serialization of motion-tracking evaluation results."""

import torch

from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import ForwardBackwardTrackingEvaluation


def motion_tracking_metrics_to_dict(
    evaluation: ForwardBackwardTrackingEvaluation,
) -> dict[str, dict[str, float | int]]:
    """Materialize final per-clip scalar rows on the host."""
    metric_names = tuple(evaluation.metric_values)
    columns = [
        *(evaluation.metric_values[name] for name in metric_names),
        evaluation.source_frame_counts.to(torch.float64),
        evaluation.evaluated_frame_counts.to(torch.float64),
        evaluation.coverage_fraction,
    ]
    rows = torch.stack(columns, dim=-1).cpu()
    metric_count = len(metric_names)
    metrics = {
        clip_id: {
            **{name: float(row[index]) for index, name in enumerate(metric_names)},
            "num_frames": int(row[metric_count + 1]),
            "source_num_frames": int(row[metric_count]),
            "evaluated_num_frames": int(row[metric_count + 1]),
            "coverage_fraction": float(row[metric_count + 2]),
        }
        for clip_id, row in zip(evaluation.sequence_ids, rows, strict=True)
    }
    return metrics
