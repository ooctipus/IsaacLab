# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed source-skeleton provenance for trajectory construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..identity import canonical_sha256, validate_nonempty, validate_sha256


@dataclass(frozen=True, slots=True)
class MotionSkeleton:
    """Immutable kinematic definition of a declared motion source.

    Attributes:
        identifier: Stable human-readable skeleton identifier.
        content_sha256: SHA-256 of the source skeleton artifact.
        body_names: Bodies in topological order.
        parent_indices: Parent body per body, with -1 for the root.
        rest_translation_m: Parent-relative rest translations [m].
        rest_rotation_wxyz: Parent-relative unit quaternions in wxyz order.
        joint_names: Joint coordinates in source order.
        joint_child_body_indices: Child body moved by each joint coordinate.
        joint_axes: Unitless hinge axes, or ``None`` for unrestricted rotation.
        landmarks: Semantic source landmarks used by cross-skeleton reconstruction.
        root_translation_frame: Frame of source root translations.
        root_rotation_convention: Source root-rotation representation.
        position_unit: Source position unit. Trajectory builders emit SI units.
        angle_unit: Source angle unit. Trajectory builders emit radians.
        identity_sha256: Deterministic hash of every interpreting field.
    """

    @dataclass(frozen=True, slots=True)
    class Landmark:
        """One semantic role and its source position/orientation bodies."""

        name: str
        position_body_name: str
        rotation_body_name: str

        def __post_init__(self) -> None:
            """Reject unnamed semantic roles or body bindings."""
            validate_nonempty("landmark name", self.name)
            validate_nonempty("landmark position_body_name", self.position_body_name)
            validate_nonempty("landmark rotation_body_name", self.rotation_body_name)

    identifier: str
    content_sha256: str
    body_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    rest_translation_m: tuple[tuple[float, float, float], ...]
    rest_rotation_wxyz: tuple[tuple[float, float, float, float], ...]
    joint_names: tuple[str, ...]
    joint_child_body_indices: tuple[int, ...]
    joint_axes: tuple[tuple[float, float, float] | None, ...]
    root_translation_frame: str
    root_rotation_convention: str
    landmarks: tuple[Landmark, ...] = ()
    position_unit: str = "m"
    angle_unit: str = "rad"
    identity_sha256: str = field(init=False)
    coordinate_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate topology and freeze a canonical identity."""
        validate_nonempty("identifier", self.identifier)
        validate_sha256("content_sha256", self.content_sha256)
        validate_nonempty("root_translation_frame", self.root_translation_frame)
        validate_nonempty("root_rotation_convention", self.root_rotation_convention)
        validate_nonempty("position_unit", self.position_unit)
        validate_nonempty("angle_unit", self.angle_unit)
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("body_names must be nonempty and unique.")
        if len({landmark.name for landmark in self.landmarks}) != len(self.landmarks):
            raise ValueError("Motion-skeleton landmark names must be unique.")
        for landmark in self.landmarks:
            if landmark.position_body_name not in self.body_names or landmark.rotation_body_name not in self.body_names:
                raise ValueError("Motion-skeleton landmark bodies must exist in body_names.")
        if len(self.parent_indices) != len(self.body_names):
            raise ValueError("parent_indices must contain one entry per body.")
        if self.parent_indices[0] != -1:
            raise ValueError("The first body must be the root with parent index -1.")
        for body_index, parent_index in enumerate(self.parent_indices[1:], start=1):
            if parent_index < 0 or parent_index >= body_index:
                raise ValueError("parent_indices must define a topologically ordered tree.")
        if len(self.rest_translation_m) != len(self.body_names):
            raise ValueError("rest_translation_m must contain one xyz vector per body.")
        if any(len(value) != 3 for value in self.rest_translation_m):
            raise ValueError("Every rest translation must contain xyz.")
        if any(not math.isfinite(component) for value in self.rest_translation_m for component in value):
            raise ValueError("Source-skeleton rest translations must be finite.")
        if len(self.rest_rotation_wxyz) != len(self.body_names):
            raise ValueError("rest_rotation_wxyz must contain one quaternion per body.")
        if any(len(value) != 4 for value in self.rest_rotation_wxyz):
            raise ValueError("Every rest rotation must contain wxyz.")
        if any(
            any(not math.isfinite(component) for component in value)
            or not math.isclose(sum(component * component for component in value), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
            for value in self.rest_rotation_wxyz
        ):
            raise ValueError("Source-skeleton rest rotations must be finite unit quaternions.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be nonempty and unique.")
        if len(self.joint_child_body_indices) != len(self.joint_names) or any(
            body_index < 1 or body_index >= len(self.body_names) for body_index in self.joint_child_body_indices
        ):
            raise ValueError("joint_child_body_indices must identify one non-root body per joint.")
        if len(self.joint_axes) != len(self.joint_names) or any(
            value is not None and len(value) != 3 for value in self.joint_axes
        ):
            raise ValueError("joint_axes must contain one xyz hinge axis or None per joint.")
        if any(
            value is not None
            and (
                any(not math.isfinite(component) for component in value)
                or not math.isclose(
                    sum(component * component for component in value), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5
                )
            )
            for value in self.joint_axes
        ):
            raise ValueError("Source-skeleton hinge axes must be finite unit vectors.")

        coordinates = {
            "body_names": self.body_names,
            "parent_indices": self.parent_indices,
            "rest_translation_m": self.rest_translation_m,
            "rest_rotation_wxyz": self.rest_rotation_wxyz,
            "joint_child_body_indices": self.joint_child_body_indices,
            "joint_names": self.joint_names,
            "joint_axes": self.joint_axes,
            "root_translation_frame": self.root_translation_frame,
            "root_rotation_convention": self.root_rotation_convention,
            "position_unit": self.position_unit,
            "angle_unit": self.angle_unit,
        }
        coordinate_identity = canonical_sha256(coordinates)
        identity = canonical_sha256(
            {
                "identifier": self.identifier,
                "content_sha256": self.content_sha256,
                "coordinates": coordinates,
                "landmarks": tuple(
                    {
                        "name": landmark.name,
                        "position_body_name": landmark.position_body_name,
                        "rotation_body_name": landmark.rotation_body_name,
                    }
                    for landmark in self.landmarks
                ),
            }
        )
        object.__setattr__(self, "coordinate_identity_sha256", coordinate_identity)
        object.__setattr__(self, "identity_sha256", identity)

    @property
    def num_bodies(self) -> int:
        """Number of source bodies."""
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        """Number of source joints."""
        return len(self.joint_names)
