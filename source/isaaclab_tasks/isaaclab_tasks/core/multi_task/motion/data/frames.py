# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-oriented motion frame tensors shared across construction and command storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import torch

Interpolation = Literal["linear", "slerp"]
if TYPE_CHECKING:
    from ...kinematics import KinematicTree, NewtonKinematics
    from ..retarget import MotionSemanticTargets
    from .clip_index import MotionClipIndex
    from .skeleton import MotionSkeleton


_FRAME_INTERPOLATION: dict[str, Interpolation] = {
    "root_position": "linear",
    "root_rotation": "slerp",
    "root_linear_velocity": "linear",
    "root_angular_velocity": "linear",
    "joint_position": "linear",
    "joint_velocity": "linear",
    "body_position": "linear",
    "body_rotation": "slerp",
    "body_linear_velocity": "linear",
    "body_angular_velocity": "linear",
}


class MotionFrameSource(Protocol):
    """Structural robot-oriented frame source consumed by expert projections."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered simulator joint names."""

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered physical and derived reference-frame names."""

    @property
    def device(self) -> torch.device:
        """Device shared by every frame tensor."""

    def field(self, name: str) -> torch.Tensor:
        """Return one named robot-oriented frame tensor."""


@runtime_checkable
class MotionFrameBuilder(Protocol):
    """Robot-owned exact and semantic construction stages."""

    source_skeleton: MotionSkeleton
    exact_coordinates: bool

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered simulator joint names."""

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered physical and derived reference-frame names."""

    @property
    def semantic_reference_kinematics(self) -> NewtonKinematics:
        """Exact target-robot mechanics used by semantic IK."""

    @property
    def semantic_target_tree(self) -> KinematicTree:
        """Grouped target-robot topology used to seed semantic IK."""

    @property
    def version(self) -> str:
        """Declared frame-construction math version."""

    @property
    def construction_identity_sha256(self) -> str:
        """Complete frame-construction policy identity."""

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity robot-frame storage on the requested device."""

    def build_exact_coordinates(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor | None,
        source_fps: float,
    ) -> MotionFrames:
        """Materialize and certify one exact free-root coordinate clip."""

    def generate_semantic_targets(
        self,
        root_position: torch.Tensor,
        local_rotation_xyzw: torch.Tensor,
    ) -> MotionSemanticTargets:
        """Generate concrete target-robot semantics."""

    def build_semantic_corpus(
        self,
        joint_q: torch.Tensor,
        clip_index: MotionClipIndex,
    ) -> MotionFrames:
        """Materialize one compact solved corpus with segment-correct derivatives."""


@dataclass(frozen=True, slots=True)
class MotionFrames:
    """Concrete trajectory columns sharing one frame axis and device.

    Every present tensor is contiguous, detached float32, and read-only
    after construction by contract. Joint columns are in live simulator
    order. Positions and velocities are world-frame SI values, and rotations
    are xyzw quaternions. Root columns are required simulator-reset facts;
    body columns are optional reference/evidence frames ordered by the
    table's ``reference_frame_names`` and may append non-rigid derived
    frames. Derived observations are deliberately absent: every consumer
    projects them from these robot-space physical fields.
    """

    root_position: torch.Tensor | None = None
    """Root-link position [m], shape [frame_count, 3], float."""

    root_rotation: torch.Tensor | None = None
    """Root-link xyzw orientation, shape [frame_count, 4], float."""

    root_linear_velocity: torch.Tensor | None = None
    """Root-link linear velocity [m/s], shape [frame_count, 3], float."""

    root_angular_velocity: torch.Tensor | None = None
    """Root-link angular velocity [rad/s], shape [frame_count, 3], float."""

    joint_position: torch.Tensor | None = None
    """Simulator-ordered joint positions [rad], shape [frame_count, joint_count], float."""

    joint_velocity: torch.Tensor | None = None
    """Simulator-ordered joint velocities [rad/s], shape [frame_count, joint_count], float."""

    body_position: torch.Tensor | None = None
    """Reference-frame positions [m], shape [frame_count, reference_frame_count, 3], float."""

    body_rotation: torch.Tensor | None = None
    """Reference-frame xyzw orientations, shape [frame_count, reference_frame_count, 4], float."""

    body_linear_velocity: torch.Tensor | None = None
    """Reference-frame linear velocities [m/s], shape [frame_count, reference_frame_count, 3], float."""

    body_angular_velocity: torch.Tensor | None = None
    """Reference-frame angular velocities [rad/s], shape [frame_count, reference_frame_count, 3], float."""

    _NAMES: ClassVar[tuple[str, ...]] = tuple(_FRAME_INTERPOLATION)
    _ROOT_FIELDS: ClassVar[tuple[str, ...]] = (
        "root_position",
        "root_rotation",
        "root_linear_velocity",
        "root_angular_velocity",
    )

    def __post_init__(self) -> None:
        """Validate fixed trajectory semantics and a shared frame axis."""
        values = {name: getattr(self, name) for name in self._NAMES}
        present = {name: value for name, value in values.items() if value is not None}
        if not present:
            raise ValueError("Motion trajectory frames must contain at least one column.")
        first = next(iter(present.values()))
        frame_count = first.shape[0] if first.ndim > 0 else -1
        for name, value in present.items():
            if (
                value.ndim < 2
                or value.shape[0] != frame_count
                or value.dtype is not torch.float32
                or value.device != first.device
                or not value.is_contiguous()
                or value.requires_grad
            ):
                raise ValueError(
                    f"Trajectory column {name!r} must be contiguous detached float32 with "
                    f"frame axis {frame_count} on {first.device}."
                )

        self._validate_group(values, ("joint_position", "joint_velocity"), required=True)
        self._validate_group(
            values,
            ("root_position", "root_rotation", "root_linear_velocity", "root_angular_velocity"),
        )
        self._validate_group(
            values,
            ("body_position", "body_rotation", "body_linear_velocity", "body_angular_velocity"),
        )
        root_stored = values["root_position"] is not None
        body_stored = values["body_position"] is not None
        if root_stored == body_stored:
            raise ValueError("Trajectory frames require exactly one root owner: explicit root or body row zero.")

        expected_tail = {
            "root_position": (3,),
            "root_rotation": (4,),
            "root_linear_velocity": (3,),
            "root_angular_velocity": (3,),
        }
        for name, shape in expected_tail.items():
            value = values[name]
            if value is not None and value.shape[1:] != shape:
                raise ValueError(f"Trajectory column {name!r} must end in {shape}.")

        joint_position = values["joint_position"]
        joint_velocity = values["joint_velocity"]
        assert joint_position is not None and joint_velocity is not None
        if joint_position.ndim != 2 or joint_velocity.shape != joint_position.shape:
            raise ValueError("Joint position and velocity must share shape [frame_count, joint_count].")

        body_position = values["body_position"]
        if body_position is not None:
            body_rotation = values["body_rotation"]
            body_linear_velocity = values["body_linear_velocity"]
            body_angular_velocity = values["body_angular_velocity"]
            assert body_rotation is not None
            assert body_linear_velocity is not None
            assert body_angular_velocity is not None
            body_shape = body_position.shape[:2]
            if (
                body_position.shape != (*body_shape, 3)
                or body_rotation.shape != (*body_shape, 4)
                or body_linear_velocity.shape != (*body_shape, 3)
                or body_angular_velocity.shape != (*body_shape, 3)
            ):
                raise ValueError("Reference-frame columns must share [frame_count, reference_frame_count, ...].")

    def validate_values(self) -> None:
        """Reject nonfinite trajectory values and nonunit stored quaternions.

        This validation runs only while a table is constructed. Runtime
        reference lookup therefore retains its allocation-free hot path.
        """
        for name in self.stored_fields:
            value = self.field(name)
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Trajectory column {name!r} must contain only finite values.")
        for name in ("root_rotation", "body_rotation"):
            value = getattr(self, name)
            if value is None:
                continue
            norms = torch.linalg.vector_norm(value, dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1.0e-5, atol=1.0e-5):
                raise ValueError(f"Trajectory column {name!r} must contain unit quaternions.")

    @staticmethod
    def _validate_group(
        values: dict[str, torch.Tensor | None], names: tuple[str, ...], *, required: bool = False
    ) -> None:
        present = tuple(values[name] is not None for name in names)
        if required and not all(present):
            raise ValueError(f"Trajectory columns {names} are required together.")
        if any(present) and not all(present):
            raise ValueError(f"Trajectory columns {names} must be all present or all absent.")

    @property
    def stored_fields(self) -> tuple[str, ...]:
        """Names of concrete columns present in this trajectory."""
        return tuple(name for name in self._NAMES if getattr(self, name) is not None)

    @property
    def available_fields(self) -> tuple[str, ...]:
        """Names of logical columns available to runtime consumers."""
        return tuple(name for name in self._NAMES if getattr(self, name) is not None or name in self._ROOT_FIELDS)

    @property
    def root_storage(self) -> Literal["explicit", "body_row_zero"]:
        """Physical owner of the logical root columns."""
        return "explicit" if self.root_position is not None else "body_row_zero"

    @property
    def frame_count(self) -> int:
        """Number of stored trajectory frames."""
        return self.field(self.stored_fields[0]).shape[0]

    @property
    def device(self) -> torch.device:
        """Shared tensor device."""
        return self.field(self.stored_fields[0]).device

    def field(self, name: str) -> torch.Tensor:
        """Return one stored column or a logical root view over body row zero."""
        if name not in self._NAMES:
            raise KeyError(f"Unknown motion trajectory field: {name!r}.")
        value = getattr(self, name)
        if value is None and name in self._ROOT_FIELDS:
            body_value = getattr(self, name.replace("root_", "body_", 1))
            if body_value is not None:
                return body_value[:, 0]
        if value is None:
            raise KeyError(f"Motion trajectory field {name!r} is absent in this composition.")
        return value

    def _copy_clip_(self, start: int, end: int, source: MotionFrames) -> None:
        """Copy one built clip into its exact destination range."""
        if source.stored_fields != self.stored_fields:
            raise ValueError(
                "Built clip columns differ from the allocated table columns: "
                f"expected {self.stored_fields}, got {source.stored_fields}."
            )
        if source.frame_count != end - start or source.device != self.device:
            raise ValueError("Built clip frame count or device differs from its destination range.")
        for name in self.stored_fields:
            destination = self.field(name)[start:end]
            value = source.field(name)
            if value.shape[1:] != destination.shape[1:]:
                raise ValueError(
                    f"Built {name!r} shape {tuple(value.shape)} differs from destination {tuple(destination.shape)}."
                )
            destination.copy_(value)

    def interpolation(self, name: str) -> Interpolation:
        """Return the fixed temporal rule for one concrete trajectory field."""
        self.field(name)
        return _FRAME_INTERPOLATION[name]
