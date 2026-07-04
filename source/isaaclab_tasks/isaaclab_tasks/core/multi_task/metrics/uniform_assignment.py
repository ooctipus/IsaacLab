# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch-owned workspace for exact GPU uniform assignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from .impl.uniform_assignment_warp import (
    UNIFORM_ASSIGNMENT_BLOCK_DIM,
    uniform_assignment_cost,
    uniform_assignment_cost_scalar,
    uniform_assignment_pairwise_distance,
    uniform_assignment_prepare_bucket,
    uniform_assignment_prepare_flat_bucket,
)


class UniformAssignmentWorkspace:
    """Power-of-two-grouped dense GPU buckets for exact variable-length assignment."""

    _MINIMUM_FRAME_CAPACITY = 32
    _SCALAR_FRAME_EXTENT_MAX = 2 * UNIFORM_ASSIGNMENT_BLOCK_DIM
    _MAXIMUM_BUCKET_COUNT = 8

    @dataclass(slots=True)
    class Bucket:
        """Fixed tensors for one power-of-two length group."""

        frame_group_bound: int
        frame_extent: int
        row_indices: torch.Tensor
        lengths: torch.Tensor
        observed: torch.Tensor
        target: torch.Tensor
        cost: torch.Tensor
        potential_rows: torch.Tensor
        potential_columns: torch.Tensor
        matching: torch.Tensor
        previous: torch.Tensor
        minimum: torch.Tensor
        used: torch.Tensor
        output: torch.Tensor

    def __init__(
        self,
        lengths: tuple[int, ...],
        device: torch.device,
        feature_width: int = 29,
    ) -> None:
        if (
            not isinstance(lengths, tuple)
            or not lengths
            or any(type(value) is not int or value < 1 for value in lengths)
            or not isinstance(device, torch.device)
            or device.type != "cuda"
            or feature_width < 1
        ):
            raise ValueError("GPU EMD requires immutable positive frame lengths, a CUDA device, and a feature width.")

        grouped_rows: dict[int, list[int]] = {}
        for row, length in enumerate(lengths):
            frame_group_bound = max(self._MINIMUM_FRAME_CAPACITY, 1 << (length - 1).bit_length())
            grouped_rows.setdefault(frame_group_bound, []).append(row)
        if len(grouped_rows) > self._MAXIMUM_BUCKET_COUNT:
            raise ValueError("GPU EMD supports at most eight nonempty power-of-two frame buckets.")

        wp.init()
        self.capacity = len(lengths)
        self.max_frames = max(lengths)
        self.device = device
        self.feature_width = feature_width
        buckets = []
        for frame_group_bound, rows in sorted(grouped_rows.items()):
            frame_extent = max(lengths[row] for row in rows)
            row_indices = torch.tensor(rows, dtype=torch.int64, device=self.device)
            bucket_lengths = torch.tensor(
                [lengths[row] for row in rows],
                dtype=torch.int64,
                device=self.device,
            )
            bucket_size = len(rows)
            observed = torch.empty(
                bucket_size,
                frame_extent,
                feature_width,
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.empty_like(observed)
            cost = torch.empty(
                bucket_size,
                frame_extent,
                frame_extent,
                dtype=torch.float32,
                device=self.device,
            )
            scratch_shape = (bucket_size, frame_extent + 1)
            potential_rows = torch.empty(scratch_shape, dtype=torch.float64, device=self.device)
            potential_columns = torch.empty_like(potential_rows)
            matching = torch.empty(scratch_shape, dtype=torch.int32, device=self.device)
            previous = torch.empty_like(matching)
            minimum = torch.empty_like(potential_rows)
            used = torch.empty_like(matching)
            buckets.append(
                self.Bucket(
                    frame_group_bound=frame_group_bound,
                    frame_extent=frame_extent,
                    row_indices=row_indices,
                    lengths=bucket_lengths,
                    observed=observed,
                    target=target,
                    cost=cost,
                    potential_rows=potential_rows,
                    potential_columns=potential_columns,
                    matching=matching,
                    previous=previous,
                    minimum=minimum,
                    used=used,
                    output=torch.empty(bucket_size, dtype=torch.float64, device=self.device),
                )
            )
        self._buckets = tuple(buckets)

    def compute(
        self,
        observed: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write exact transport costs through the immutable fixed bucket layout."""
        batch_size = observed.shape[0]
        if (
            observed.ndim != 3
            or target.shape != observed.shape
            or observed.shape[:2] != (self.capacity, self.max_frames)
            or observed.dtype is not torch.float32
            or target.dtype is not torch.float32
            or observed.device != self.device
            or target.device != self.device
            or not observed.is_contiguous()
            or not target.is_contiguous()
            or output.shape != (self.capacity,)
            or output.dtype is not torch.float64
            or output.device != self.device
            or observed.shape[2] < 1
            or observed.shape[2] > self.feature_width
            or batch_size != self.capacity
        ):
            raise ValueError("GPU EMD inputs do not match the fixed workspace contract.")
        feature_width = observed.shape[2]
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        for bucket in self._buckets:
            bucket_size = bucket.row_indices.shape[0]
            wp.launch(
                uniform_assignment_prepare_bucket,
                dim=(bucket_size, bucket.frame_extent),
                inputs=[
                    wp.from_torch(observed),
                    wp.from_torch(target),
                    wp.from_torch(bucket.row_indices),
                    self.max_frames,
                    feature_width,
                    self.feature_width,
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                ],
                stream=stream,
            )
            wp.launch(
                uniform_assignment_pairwise_distance,
                dim=(bucket_size, bucket.frame_extent, bucket.frame_extent),
                inputs=[
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                    feature_width,
                    wp.from_torch(bucket.cost),
                ],
                stream=stream,
            )
            assignment_inputs = [
                wp.from_torch(bucket.cost),
                wp.from_torch(bucket.lengths),
                wp.from_torch(bucket.potential_rows),
                wp.from_torch(bucket.potential_columns),
                wp.from_torch(bucket.matching),
                wp.from_torch(bucket.previous),
                wp.from_torch(bucket.minimum),
                wp.from_torch(bucket.used),
                wp.from_torch(bucket.output),
            ]
            if bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX:
                wp.launch(
                    uniform_assignment_cost_scalar,
                    dim=bucket_size,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            else:
                wp.launch_tiled(
                    uniform_assignment_cost,
                    dim=[bucket_size],
                    block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            output.index_copy_(0, bucket.row_indices, bucket.output)

    def compute_flat(
        self,
        observed: torch.Tensor,
        observed_starts: torch.Tensor,
        target: torch.Tensor,
        target_starts: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write exact costs from compact flat clip rows without dense trace padding."""
        if (
            observed.ndim != 2
            or target.ndim != 2
            or observed.shape[1] != target.shape[1]
            or observed.dtype is not torch.float32
            or target.dtype is not torch.float32
            or observed.device != self.device
            or target.device != self.device
            or not observed.is_contiguous()
            or observed.shape[1] < 1
            or observed.shape[1] > self.feature_width
            or observed_starts.shape != (self.capacity,)
            or target_starts.shape != (self.capacity,)
            or observed_starts.dtype is not torch.int64
            or target_starts.dtype is not torch.int64
            or observed_starts.device != self.device
            or target_starts.device != self.device
            or not observed_starts.is_contiguous()
            or not target_starts.is_contiguous()
            or output.shape != (self.capacity,)
            or output.dtype is not torch.float64
            or output.device != self.device
        ):
            raise ValueError("Flat GPU EMD inputs do not match the fixed workspace contract.")
        feature_width = observed.shape[1]
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        for bucket in self._buckets:
            bucket_size = bucket.row_indices.shape[0]
            wp.launch(
                uniform_assignment_prepare_flat_bucket,
                dim=(bucket_size, bucket.frame_extent),
                inputs=[
                    wp.from_torch(observed),
                    wp.from_torch(target),
                    wp.from_torch(observed_starts),
                    wp.from_torch(target_starts),
                    wp.from_torch(bucket.row_indices),
                    wp.from_torch(bucket.lengths),
                    feature_width,
                    self.feature_width,
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                ],
                stream=stream,
            )
            wp.launch(
                uniform_assignment_pairwise_distance,
                dim=(bucket_size, bucket.frame_extent, bucket.frame_extent),
                inputs=[
                    wp.from_torch(bucket.observed),
                    wp.from_torch(bucket.target),
                    feature_width,
                    wp.from_torch(bucket.cost),
                ],
                stream=stream,
            )
            assignment_inputs = [
                wp.from_torch(bucket.cost),
                wp.from_torch(bucket.lengths),
                wp.from_torch(bucket.potential_rows),
                wp.from_torch(bucket.potential_columns),
                wp.from_torch(bucket.matching),
                wp.from_torch(bucket.previous),
                wp.from_torch(bucket.minimum),
                wp.from_torch(bucket.used),
                wp.from_torch(bucket.output),
            ]
            if bucket.frame_extent <= self._SCALAR_FRAME_EXTENT_MAX:
                wp.launch(
                    uniform_assignment_cost_scalar,
                    dim=bucket_size,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            else:
                wp.launch_tiled(
                    uniform_assignment_cost,
                    dim=[bucket_size],
                    block_dim=UNIFORM_ASSIGNMENT_BLOCK_DIM,
                    inputs=assignment_inputs,
                    stream=stream,
                )
            output.index_copy_(0, bucket.row_indices, bucket.output)
