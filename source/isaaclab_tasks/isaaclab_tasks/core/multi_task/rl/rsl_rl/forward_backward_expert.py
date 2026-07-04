# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an immutable forward-backward expert corpus from a bound sequence source."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.modules.reward_channels import get_forward_backward_schema_hash
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from rsl_rl.utils import resolve_callable

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


@runtime_checkable
class ForwardBackwardSequenceSource(Protocol):
    """Sequence storage capable of producing one clip-safe sampled view."""

    device: torch.device

    def sample(
        self,
        mode: Literal["source_rows", "uniform_before_source_end"],
        step_seconds: float | None,
    ) -> ForwardBackwardSampledSequence:
        """Return a sampled sequence view."""


@runtime_checkable
class ForwardBackwardSampledSequence(Protocol):
    """Projected-corpus input exposed by a domain-owned sequence source."""

    source: ForwardBackwardSequenceSource
    device: torch.device
    clip_ids: tuple[str, ...]
    clip_offsets: tuple[int, ...]
    dataset_id: str
    data_hash: str

    def field(self, name: str) -> torch.Tensor:
        """Return one named sampled field."""


def _offsets_hash(offsets: tuple[int, ...]) -> str:
    """Hash clip boundaries without copying a device tensor to the host."""
    digest = hashlib.sha256()
    for value in offsets:
        digest.update(value.to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _source(
    env: RslRlVecEnvWrapper,
    expression: str,
    device: str | torch.device,
) -> ForwardBackwardSequenceSource:
    """Resolve one configured sequence source at the public connector boundary."""
    if not isinstance(expression, str) or not expression:
        raise TypeError("source_bind must be a nonempty expression.")
    source = eval(expression, {}, {"env": env})  # noqa: S307
    if not isinstance(source, ForwardBackwardSequenceSource):
        raise TypeError("source_bind must resolve to a ForwardBackwardSequenceSource.")
    if source.device != torch.device(device):
        raise ValueError(f"Sequence source is on {source.device}, but the learner is on {torch.device(device)}.")
    return source


def _projected_fields(
    projection: Callable[..., object],
    owners: tuple[object, ...],
    sampled: ForwardBackwardSampledSequence,
) -> tuple[Mapping[str, torch.Tensor], object]:
    """Apply one pure domain projection to sampled source fields."""
    result = projection(*owners, sampled.source, sampled.field)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Expert target projection must return (fields, identity).")
    fields, identity = result
    if not isinstance(fields, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in fields.items()
    ):
        raise TypeError("Expert target fields must map names to tensors.")
    return fields, identity


def forward_backward_expert_buffer(
    env: RslRlVecEnvWrapper,
    observation_schema: ForwardBackwardObservationSchema,
    device: str,
    *,
    source_bind: str,
    sampling_mode: Literal["source_rows", "uniform_before_source_end"],
    sampling_step_seconds: float | None,
    target_projection: str,
    target_projection_binds: tuple[str, ...],
    window_lengths: tuple[int, ...],
    seed: int = 0,
) -> ForwardBackwardExpertBuffer:
    """Project a bound sequence source onto the learner's backward route.

    Args:
        env: RSL-RL vector-environment wrapper.
        observation_schema: Learner observation fields and routes.
        device: Learner tensor device.
        source_bind: Expression resolving the domain-owned sequence source.
        sampling_mode: Relation between stored source rows and expert samples.
        sampling_step_seconds: Uniform sample period [s], or None for source rows.
        target_projection: Qualified pure projection callable selected by domain config.
        target_projection_binds: Expressions resolving the projection's explicitly owned inputs.
        window_lengths: Expert edge-window lengths available to the learner.
        seed: Expert sampler seed.

    Returns:
        Immutable clip-safe expert buffer on the declared sampling clock.
    """
    source = _source(env, source_bind, device)
    sampled = source.sample(sampling_mode, sampling_step_seconds)
    if not isinstance(sampled, ForwardBackwardSampledSequence):
        raise TypeError("Sequence source sample() must return a ForwardBackwardSampledSequence.")
    if sampled.source is not source or sampled.device != source.device:
        raise ValueError("Sampled sequence must retain its source and device.")
    if len(sampled.clip_offsets) != len(sampled.clip_ids) + 1:
        raise ValueError("Sampled clip offsets and stable identifiers must align.")
    if sampled.clip_offsets[0] != 0 or any(
        end <= start for start, end in zip(sampled.clip_offsets[:-1], sampled.clip_offsets[1:], strict=True)
    ):
        raise ValueError("Every sampled clip must occupy one nonempty contiguous range.")

    projection = resolve_callable(target_projection)
    owners = tuple(eval(expression, {}, {"env": env}) for expression in target_projection_binds)  # noqa: S307
    target_fields, projection_identity = _projected_fields(projection, owners, sampled)
    backward_route = tuple(observation_schema.route("backward"))
    if set(target_fields) != set(backward_route):
        raise ValueError("Expert target fields must match the declared backward route exactly.")
    widths = dict(observation_schema.field_widths)
    frame_count = sampled.clip_offsets[-1]
    for name in backward_route:
        value = target_fields[name]
        if value.ndim != 2 or value.shape != (frame_count, widths[name]):
            raise ValueError(
                f"Expert target {name!r} must have shape {(frame_count, widths[name])}, got {tuple(value.shape)}."
            )
    frames = (
        target_fields[backward_route[0]]
        if len(backward_route) == 1
        else torch.cat(tuple(target_fields[name] for name in backward_route), dim=-1)
    )
    if frames.device != sampled.device or frames.dtype is not torch.float32 or frames.requires_grad:
        raise ValueError("Expert projection must be detached float32 on the sampled-sequence device.")

    data_hash = get_forward_backward_schema_hash(
        {
            "source": sampled.data_hash,
            "projection": projection_identity,
        }
    )
    clip_offsets = torch.tensor(sampled.clip_offsets, dtype=torch.int64, device=sampled.device)
    priorities = torch.ones(len(sampled.clip_ids), dtype=torch.float32, device=sampled.device)
    schema = ForwardBackwardExpertSchema(
        dataset_id=sampled.dataset_id,
        data_hash=data_hash,
        feature_schema_hash=observation_schema.schema_hash,
        clip_offsets_hash=_offsets_hash(sampled.clip_offsets),
        expert_feature_width=observation_schema.route_width("backward"),
        num_frames=frames.shape[0],
        num_clips=len(sampled.clip_ids),
        window_lengths=window_lengths,
    )
    return ForwardBackwardExpertBuffer(
        frames,
        clip_offsets,
        priorities,
        schema,
        seed=seed,
        clip_ids=sampled.clip_ids,
        clip_length_values=tuple(
            end - start for start, end in zip(sampled.clip_offsets[:-1], sampled.clip_offsets[1:], strict=True)
        ),
    )
