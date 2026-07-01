# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-place applied-transition history over caller-owned storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AppliedTransitionHistoryLayout:
    """Fixed field-major history layout.

    Attributes:
        fields: Ordered (name, width) pairs. Each field occupies all of its
            time lags before the next field begins.
        length: Number of reached transition values retained per field.
    """

    fields: tuple[tuple[str, int], ...]
    length: int

    def __post_init__(self) -> None:
        """Validate field names, widths, and history length."""
        names = tuple(name for name, _width in self.fields)
        if not names or len(names) != len(set(names)):
            raise ValueError("History fields must be nonempty and unique.")
        if any(not name or width < 1 for name, width in self.fields):
            raise ValueError("History field names and widths must be nonempty and positive.")
        if self.length < 1:
            raise ValueError("History length must be positive.")

    @property
    def width(self) -> int:
        """Total flattened history width."""
        return self.length * sum(width for _name, width in self.fields)


class AppliedTransitionHistory:
    """Update one caller-owned field-major history buffer in place.

    Reset observations are seeds rather than reached states, so reset clears
    their rows and no append occurs. Source tensors and the applied mask are
    bound once at construction. Callers update those fixed tensors and then
    call append; the hot call performs no validation or allocation.

    The flat storage orders all newest-first lags for one field before the next
    field. The value property returns that exact caller-owned tensor without
    materializing another representation.
    """

    def __init__(
        self,
        layout: AppliedTransitionHistoryLayout,
        value: torch.Tensor,
        *,
        fields: Mapping[str, torch.Tensor],
        applied: torch.Tensor,
    ) -> None:
        """Bind fixed history storage, source fields, and the applied-row mask.

        Args:
            layout: Field order, widths, and history length.
            value: Contiguous history storage with shape [num_envs, layout.width].
            fields: One contiguous [num_envs, width] source tensor per field.
            applied: Contiguous boolean applied-row mask with shape [num_envs].
        """
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != layout.width:
            raise ValueError(f"value must have shape [num_envs, {layout.width}] with num_envs positive.")
        if not value.is_floating_point() or not value.is_contiguous() or value.requires_grad:
            raise ValueError("value must be a contiguous, non-gradient floating-point tensor.")

        field_names = tuple(name for name, _width in layout.fields)
        if tuple(fields) != field_names:
            raise ValueError(f"History fields must be exactly {field_names} in that order.")

        num_envs = value.shape[0]
        sources: list[torch.Tensor] = []
        field_views: list[torch.Tensor] = []
        offset = 0
        for name, width in layout.fields:
            source = fields[name]
            if (
                source.shape != (num_envs, width)
                or source.device != value.device
                or source.dtype != value.dtype
                or not source.is_contiguous()
                or source.requires_grad
            ):
                raise ValueError(
                    f"History field {name!r} must be contiguous and non-gradient with shape {(num_envs, width)}, "
                    f"dtype {value.dtype}, and device {value.device}."
                )
            if source.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                raise ValueError(f"History field {name!r} must not alias the history value storage.")
            end = offset + layout.length * width
            sources.append(source)
            field_views.append(value[:, offset:end].view(num_envs, layout.length, width))
            offset = end

        if (
            applied.shape != (num_envs,)
            or applied.dtype != torch.bool
            or applied.device != value.device
            or not applied.is_contiguous()
        ):
            raise ValueError(f"applied must be boolean, contiguous, on {value.device}, and have shape {(num_envs,)}.")

        self.layout = layout
        self._value = value
        self._sources = tuple(sources)
        self._field_views = tuple(field_views)
        self._applied = applied.view(num_envs, 1)

    @property
    def value(self) -> torch.Tensor:
        """Caller-owned field-major, newest-first history storage."""
        return self._value

    def append(self) -> None:
        """Append the bound reached facts on rows backed by an applied action."""
        for source, history in zip(self._sources, self._field_views, strict=True):
            for lag in range(self.layout.length - 1, 0, -1):
                torch.where(self._applied, history[:, lag - 1], history[:, lag], out=history[:, lag])
            torch.where(self._applied, source, history[:, 0], out=history[:, 0])

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear rows so reset seeds cannot become transition history.

        Args:
            env_ids: Environment row indices, a basic slice, or None for all rows.
        """
        if env_ids is None:
            self._value.zero_()
        elif isinstance(env_ids, slice):
            self._value[env_ids].zero_()
        else:
            self._value.index_fill_(0, env_ids, 0.0)
