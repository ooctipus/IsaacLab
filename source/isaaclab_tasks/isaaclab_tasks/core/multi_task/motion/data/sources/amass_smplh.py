# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode native AMASS SMPL-H clips and shape-specific source mechanics."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import quat_from_rotation_vector

from ....kinematics import fit_ordered_hinge_coordinates, time_unwrap_angles
from ...identity import canonical_sha256, validate_nonempty, validate_sha256
from ..clip_index import MotionClipIndex
from ..skeleton import MotionSkeleton
from ..smpl import (
    SMPL_BODY_NAMES,
    SMPL_COMPATIBLE_POSE_PROFILE_SHA256,
    SMPL_LBS_FORMAT,
    SMPL_PARENT_INDICES,
    SmplLbsModel,
    load_smpl_lbs_model,
)
from ..source import MotionSourceCfg

_AMASS_FIELDS = ("trans", "gender", "mocap_framerate", "betas", "dmpls", "poses")
_AMASS_BODY_COUNT = 52
_AMASS_LANDMARK_BODIES = (
    ("pelvis", "Pelvis"),
    ("left_hip", "L_Hip"),
    ("left_knee", "L_Knee"),
    ("left_ankle", "L_Ankle"),
    ("left_toe", "L_Toe"),
    ("right_hip", "R_Hip"),
    ("right_knee", "R_Knee"),
    ("right_ankle", "R_Ankle"),
    ("right_toe", "R_Toe"),
    ("torso", "Torso"),
    ("spine", "Spine"),
    ("chest", "Chest"),
    ("neck", "Neck"),
    ("left_thorax", "L_Thorax"),
    ("right_thorax", "R_Thorax"),
    ("head", "Head"),
    ("left_shoulder", "L_Shoulder"),
    ("left_elbow", "L_Elbow"),
    ("left_wrist", "L_Wrist"),
    ("right_shoulder", "R_Shoulder"),
    ("right_elbow", "R_Elbow"),
    ("right_wrist", "R_Wrist"),
)


@dataclass(frozen=True, slots=True)
class AmassSmplhClip:
    """One native AMASS clip before resampling, grounding, or body-model projection.

    Attributes:
        root_translation_m: World root translation [m], shape ``[frame_count, 3]``.
        local_axis_angle_rad: World-root then parent-local SMPL-H rotation vectors
            [rad], shape ``[frame_count, 52, 3]``.
        betas: Native AMASS body-shape coefficients.
        gender: Body-model gender key declared by AMASS.
        source_fps: Native sample rate [Hz].
        pose_body_indices: Raw SMPL-H pose row used by each declared source body.
    """

    root_translation_m: np.ndarray
    local_axis_angle_rad: np.ndarray
    betas: np.ndarray
    gender: str
    source_fps: float
    pose_body_indices: tuple[int, ...] = tuple(range(_AMASS_BODY_COUNT))

    def __post_init__(self) -> None:
        """Validate the immutable native-array boundary."""
        frame_count = self.root_translation_m.shape[0] if self.root_translation_m.ndim == 2 else 0
        if frame_count < 1 or self.root_translation_m.shape != (frame_count, 3):
            raise ValueError("AMASS root translation must have shape [frame_count, 3].")
        if self.local_axis_angle_rad.shape != (frame_count, _AMASS_BODY_COUNT, 3):
            raise ValueError("AMASS SMPL-H pose must have shape [frame_count, 52, 3].")
        if self.root_translation_m.dtype != np.float64 or self.local_axis_angle_rad.dtype != np.float64:
            raise ValueError("Native AMASS translation and pose must retain float64 storage.")
        if self.betas.dtype != np.float64 or self.betas.shape != (16,):
            raise ValueError("Native AMASS betas must have float64 shape [16].")
        if not all(
            value.flags.c_contiguous for value in (self.root_translation_m, self.local_axis_angle_rad, self.betas)
        ):
            raise ValueError("Native AMASS arrays must be C-contiguous.")
        if not all(
            np.isfinite(value).all() for value in (self.root_translation_m, self.local_axis_angle_rad, self.betas)
        ):
            raise ValueError("Native AMASS arrays must contain only finite values.")
        if self.gender not in ("female", "male", "neutral"):
            raise ValueError("AMASS gender must be female, male, or neutral.")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("AMASS source_fps must be finite and positive [Hz].")
        if (
            not self.pose_body_indices
            or self.pose_body_indices[0] != 0
            or len(set(self.pose_body_indices)) != len(self.pose_body_indices)
            or any(
                type(index) is not int or index < 0 or index >= _AMASS_BODY_COUNT for index in self.pose_body_indices
            )
        ):
            raise ValueError("AMASS pose_body_indices must uniquely map each body to a valid raw pose row.")
        for value in (self.root_translation_m, self.local_axis_angle_rad, self.betas):
            value.setflags(write=False)

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""
        return self.root_translation_m.shape[0]

    def local_pose(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world-root and parent-local SMPL-H rotations on the requested device."""
        _validate_smplh_skeleton(source_skeleton, self.pose_body_indices)
        translation = torch.as_tensor(self.root_translation_m.copy(), dtype=torch.float32, device=device)
        raw_axis_angle = torch.as_tensor(self.local_axis_angle_rad.copy(), dtype=torch.float32, device=device)
        axis_angle = raw_axis_angle[:, self.pose_body_indices]
        return translation, quat_from_rotation_vector(axis_angle)

    def free_root_coordinates(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, None]:
        """Express the native pose in the skeleton's ordered XYZ hinge coordinates."""
        _validate_smplh_skeleton(source_skeleton, self.pose_body_indices)
        translation, rotation = self.local_pose(source_skeleton, device=device)
        coordinate, _ = fit_ordered_hinge_coordinates(
            rotation[:, 1:], rotation.new_tensor(source_skeleton.joint_axes[:3])
        )
        coordinate = time_unwrap_angles(coordinate.reshape(self.frame_count, -1))
        return torch.cat((translation, rotation[:, 0], coordinate), dim=-1), None


@dataclass(frozen=True, slots=True)
class AmassClipRow:
    """One native AMASS file with immutable metadata needed before motion decoding.

    Attributes:
        relative_path: Source-root-relative native file path.
        source_sha256: SHA-256 of the complete native file bytes.
        gender: SMPL body-model gender declared by the native file.
        source_frame_count: Number of native motion frames.
        source_fps: Native sample rate [Hz].
        betas: Sixteen dimensionless AMASS body-shape coefficients.
    """

    relative_path: str
    source_sha256: str
    gender: str
    source_frame_count: int
    source_fps: float
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require one concrete source-root-relative NPZ and exact shape metadata."""
        relative_path = Path(self.relative_path)
        if (
            not self.relative_path
            or "\\" in self.relative_path
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".npz"
            or not relative_path.name.endswith("_poses.npz")
            or relative_path.as_posix() != self.relative_path
        ):
            raise ValueError("AMASS clip rows require one normalized source-root-relative *_poses.npz path.")
        validate_sha256("AMASS clip row source_sha256", self.source_sha256)
        if self.gender not in ("female", "male", "neutral"):
            raise ValueError("AMASS clip rows require a female, male, or neutral body model.")
        if type(self.source_frame_count) is not int or self.source_frame_count < 1:
            raise ValueError("AMASS clip rows require a positive source frame count.")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("AMASS clip rows require a finite positive source sample rate [Hz].")
        if len(self.betas) != 16 or any(not math.isfinite(value) for value in self.betas):
            raise ValueError("AMASS clip rows require 16 finite body-shape coefficients.")

    @classmethod
    def from_file(cls, relative_path: str, source_path: str | Path) -> AmassClipRow:
        """Read one native file once and freeze the metadata needed by runtime inspection.

        Args:
            relative_path: Source-root-relative native file path.
            source_path: Local native file used during offline preparation.

        Returns:
            Concrete row containing the source identity, shape, and timing facts.
        """
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_bytes = source_path.read_bytes()
        clip = _decode_amass_smplh_bytes(source_bytes)
        return cls(
            relative_path=relative_path,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            gender=clip.gender,
            source_frame_count=clip.frame_count,
            source_fps=clip.source_fps,
            betas=tuple(float(value) for value in clip.betas),
        )


_AMASS_CLIP_ROW_FIELDS = (
    "relative_path",
    "source_sha256",
    "gender",
    "source_frame_count",
    "source_fps",
    *(f"beta_{index}" for index in range(16)),
)


def read_amass_clip_rows(path: str | Path) -> tuple[AmassClipRow, ...]:
    """Read deterministic concrete AMASS clip rows from one UTF-8 CSV file.

    Args:
        path: Concrete clip-row CSV artifact.

    Returns:
        Rows in their declared scientific order.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _AMASS_CLIP_ROW_FIELDS:
            raise ValueError(f"AMASS clip-row columns must equal {_AMASS_CLIP_ROW_FIELDS}.")
        try:
            rows = tuple(
                AmassClipRow(
                    relative_path=row["relative_path"],
                    source_sha256=row["source_sha256"],
                    gender=row["gender"],
                    source_frame_count=int(row["source_frame_count"]),
                    source_fps=float(row["source_fps"]),
                    betas=tuple(float(row[f"beta_{index}"]) for index in range(16)),
                )
                for row in reader
            )
        except (TypeError, ValueError) as error:
            raise ValueError("AMASS clip rows contain invalid numeric fields.") from error
    if not rows or len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("AMASS clip rows must be nonempty and have unique relative paths.")
    return rows


def write_amass_clip_rows(path: str | Path, rows: tuple[AmassClipRow, ...]) -> None:
    """Write deterministic concrete AMASS clip rows with Unix line endings.

    Args:
        path: Destination CSV artifact.
        rows: Rows in scientific output order.
    """
    if not rows or len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("AMASS clip rows must be nonempty and have unique relative paths.")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_AMASS_CLIP_ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {
                "relative_path": row.relative_path,
                "source_sha256": row.source_sha256,
                "gender": row.gender,
                "source_frame_count": row.source_frame_count,
                "source_fps": format(row.source_fps, ".17g"),
            }
            values.update({f"beta_{index}": format(value, ".17g") for index, value in enumerate(row.betas)})
            writer.writerow(values)


def amass_clip_rows_content_sha256(rows: tuple[AmassClipRow, ...]) -> str:
    """Return the scientific identity of the ordered AMASS source files.

    Args:
        rows: Concrete rows in scientific order.

    Returns:
        SHA-256 over every source-relative path and native file-content hash.
    """
    if not rows:
        raise ValueError("AMASS source identity requires at least one concrete clip row.")
    return canonical_sha256(tuple((row.relative_path, row.source_sha256) for row in rows))


def _decode_amass_smplh_bytes(source_bytes: bytes) -> AmassSmplhClip:
    """Decode native AMASS SMPL-H bytes without transforming their motion.

    Args:
        source_bytes: Complete native AMASS motion archive bytes.

    Returns:
        Strictly validated native clip.
    """
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
        if tuple(archive.files) != _AMASS_FIELDS:
            raise ValueError(f"AMASS fields differ from the native ordered contract: {archive.files}.")
        translation = np.ascontiguousarray(archive["trans"])
        pose = np.ascontiguousarray(archive["poses"])
        betas = np.ascontiguousarray(archive["betas"])
        dmpls = archive["dmpls"]
        gender = archive["gender"]
        source_fps = archive["mocap_framerate"]

    frame_count = translation.shape[0] if translation.ndim == 2 else 0
    if pose.shape != (frame_count, _AMASS_BODY_COUNT * 3):
        raise ValueError("AMASS poses must have shape [frame_count, 156].")
    if not isinstance(dmpls, np.ndarray) or dmpls.dtype != np.float64 or dmpls.shape != (frame_count, 8):
        raise ValueError("AMASS DMPL coefficients must have float64 shape [frame_count, 8].")
    if not np.isfinite(dmpls).all():
        raise ValueError("AMASS DMPL coefficients must contain only finite values.")
    if not isinstance(gender, np.ndarray) or gender.ndim != 0 or gender.dtype.kind != "U":
        raise ValueError("AMASS gender must be a scalar Unicode array.")
    if (
        not isinstance(source_fps, np.ndarray)
        or source_fps.ndim != 0
        or not np.issubdtype(source_fps.dtype, np.floating)
    ):
        raise ValueError("AMASS mocap_framerate must be a scalar floating-point array [Hz].")
    return AmassSmplhClip(
        root_translation_m=translation,
        local_axis_angle_rad=pose.reshape(frame_count, _AMASS_BODY_COUNT, 3),
        betas=betas,
        gender=str(gender.item()),
        source_fps=float(source_fps.item()),
    )


def load_amass_smplh_clip(path: str | Path) -> AmassSmplhClip:
    """Load one native AMASS SMPL-H NPZ file without transforming its motion.

    Args:
        path: Native AMASS motion archive.

    Returns:
        Strictly validated native clip.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return _decode_amass_smplh_bytes(source_path.read_bytes())


def _load_verified_amass_smplh_clip(path: Path, expected_sha256: str) -> AmassSmplhClip:
    """Read, verify, and decode one selected native file exactly once."""
    if not path.is_file():
        raise FileNotFoundError(path)
    source_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"AMASS source file hash differs for {path}: expected {expected_sha256}, got {actual_sha256}.")
    return _decode_amass_smplh_bytes(source_bytes)


def _validate_smplh_skeleton(skeleton: MotionSkeleton, pose_body_indices: tuple[int, ...]) -> None:
    """Require one ordered XYZ hinge chain for every declared source body."""
    expected_children = tuple(body for body in range(1, len(pose_body_indices)) for _ in range(3))
    xyz_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if (
        skeleton.num_bodies != len(pose_body_indices)
        or skeleton.joint_child_body_indices != expected_children
        or skeleton.joint_axes != xyz_axes * (len(pose_body_indices) - 1)
        or skeleton.root_rotation_convention != "xyzw"
    ):
        raise ValueError("Raw AMASS requires the mechanics view's declared bodies and ordered XYZ hinges.")


@dataclass(frozen=True, slots=True)
class AmassSmplhMechanics:
    """One explicit shape-linear SMPL-H mechanics view.

    A full mechanics object maps all 52 raw pose rows. A reduced object must
    identify itself as a ``body_core`` view and carries its exact pose-row map;
    raw clips still retain all 52 source rows.

    Attributes:
        identifier: Concrete mechanics/view name.
        content_sha256: SHA-256 of the licensed model-derived mechanics artifact.
        body_names: Bodies in topological order.
        parent_indices: Parent body per body, with ``-1`` for the root.
        pose_body_indices: Unique raw SMPL-H pose row per body.
        rest_joint_position_m: Zero-shape joint positions [m], shape ``[body_count, 3]``.
        rest_joint_shape_basis_m: Linear joint-position basis [m], shape
            ``[body_count, 3, beta_count]``.
    """

    identifier: str
    content_sha256: str
    body_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    pose_body_indices: tuple[int, ...]
    rest_joint_position_m: np.ndarray
    rest_joint_shape_basis_m: np.ndarray

    def __post_init__(self) -> None:
        """Validate one concrete mechanics artifact without a body-model runtime dependency."""
        validate_nonempty("AMASS mechanics identifier", self.identifier)
        validate_sha256("AMASS mechanics content_sha256", self.content_sha256)
        body_count = len(self.body_names)
        if body_count < 2 or len(set(self.body_names)) != body_count:
            raise ValueError("AMASS mechanics body names must be nonempty and unique.")
        if len(self.parent_indices) != body_count or self.parent_indices[0] != -1:
            raise ValueError("AMASS mechanics must declare one topological root.")
        if any(parent < 0 or parent >= body for body, parent in enumerate(self.parent_indices[1:], start=1)):
            raise ValueError("AMASS mechanics parents must precede their children.")
        if (
            len(self.pose_body_indices) != body_count
            or self.pose_body_indices[0] != 0
            or len(set(self.pose_body_indices)) != body_count
            or any(
                type(index) is not int or index < 0 or index >= _AMASS_BODY_COUNT for index in self.pose_body_indices
            )
        ):
            raise ValueError("AMASS mechanics pose rows must uniquely map every body to raw row 0..51.")
        full_view = body_count == _AMASS_BODY_COUNT and self.pose_body_indices == tuple(range(_AMASS_BODY_COUNT))
        if not full_view and "body_core" not in self.identifier:
            raise ValueError("Reduced AMASS mechanics must identify itself explicitly as a body_core view.")
        beta_count = self.rest_joint_shape_basis_m.shape[-1] if self.rest_joint_shape_basis_m.ndim == 3 else 0
        if (
            self.rest_joint_position_m.dtype != np.float64
            or self.rest_joint_position_m.shape != (body_count, 3)
            or self.rest_joint_shape_basis_m.dtype != np.float64
            or self.rest_joint_shape_basis_m.shape != (body_count, 3, beta_count)
            or beta_count < 1
            or beta_count > 16
        ):
            raise ValueError("AMASS mechanics rest positions/basis have invalid float64 shapes.")
        if not self.rest_joint_position_m.flags.c_contiguous or not self.rest_joint_shape_basis_m.flags.c_contiguous:
            raise ValueError("AMASS mechanics arrays must be C-contiguous.")
        if not np.isfinite(self.rest_joint_position_m).all() or not np.isfinite(self.rest_joint_shape_basis_m).all():
            raise ValueError("AMASS mechanics arrays must contain only finite values.")
        self.rest_joint_position_m.setflags(write=False)
        self.rest_joint_shape_basis_m.setflags(write=False)

    @property
    def beta_count(self) -> int:
        """Number of shape coefficients represented by this mechanics artifact."""
        return self.rest_joint_shape_basis_m.shape[-1]

    def skeleton(self, betas: np.ndarray) -> MotionSkeleton:
        """Materialize one source skeleton for exact clip shape coefficients."""
        if betas.dtype != np.float64 or betas.shape != (16,) or not np.isfinite(betas).all():
            raise ValueError("AMASS skeleton construction requires native finite betas[16].")
        shape = self.rest_joint_position_m + np.einsum(
            "bck,k->bc", self.rest_joint_shape_basis_m, betas[: self.beta_count], optimize=True
        )
        rest_translation = np.zeros_like(shape)
        for body, parent in enumerate(self.parent_indices[1:], start=1):
            rest_translation[body] = shape[body] - shape[parent]
        shape_identity = canonical_sha256(
            {
                "mechanics": self.content_sha256,
                "betas": tuple(float(value) for value in betas[: self.beta_count]),
            }
        )
        xyz_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        body_names = set(self.body_names)
        landmarks = tuple(
            MotionSkeleton.Landmark(role, body_name, body_name)
            for role, body_name in _AMASS_LANDMARK_BODIES
            if body_name in body_names
        )
        return MotionSkeleton(
            identifier=f"{self.identifier}_{shape_identity[:16]}",
            content_sha256=shape_identity,
            body_names=self.body_names,
            parent_indices=self.parent_indices,
            rest_translation_m=tuple(tuple(float(component) for component in row) for row in rest_translation),
            rest_rotation_wxyz=((1.0, 0.0, 0.0, 0.0),) * len(self.body_names),
            joint_names=tuple(f"{body_name}_{axis}" for body_name in self.body_names[1:] for axis in "xyz"),
            joint_child_body_indices=tuple(body for body in range(1, len(self.body_names)) for _ in range(3)),
            joint_axes=xyz_axes * (len(self.body_names) - 1),
            landmarks=landmarks,
            root_translation_frame="world",
            root_rotation_convention="xyzw",
            landmark_rotation_policy="calibrated_body",
        )


def smpl_body_core_mechanics(model: SmplLbsModel) -> AmassSmplhMechanics:
    """Derive the 22-body AMASS source view from one compact SMPL model.

    Args:
        model: Compact SMPL mechanics carrying the dense shape model.

    Returns:
        Shape-linear body-core mechanics backed by the same model digest.
    """
    body_count = 22
    zero_betas = torch.zeros((1, 10), dtype=torch.float32, device=model.device)
    rest_joints = model.shaped_joints(zero_betas)[0, :body_count]
    shape_basis = torch.einsum(
        "jv,vck->jck",
        model.joint_regressor[:body_count],
        model.shape_blend_directions_m,
    )
    return AmassSmplhMechanics(
        identifier=f"smpl_{model.gender}_body_core_22",
        content_sha256=canonical_sha256(
            {
                "format": SMPL_LBS_FORMAT,
                "source_sha256": model.source_sha256,
                "view": "body_core_22",
            }
        ),
        body_names=SMPL_BODY_NAMES[:body_count],
        parent_indices=SMPL_PARENT_INDICES[:body_count],
        pose_body_indices=tuple(range(body_count)),
        rest_joint_position_m=np.ascontiguousarray(rest_joints.detach().cpu().double().numpy()),
        rest_joint_shape_basis_m=np.ascontiguousarray(shape_basis.detach().cpu().double().numpy()),
    )


class AmassSmplhClips:
    """Ordered native AMASS files with source-owned, shape-specific mechanics."""

    compatible_pose_profile_sha256 = SMPL_COMPATIBLE_POSE_PROFILE_SHA256

    __slots__ = (
        "_data_root",
        "_index",
        "_mechanics",
        "_model_cache",
        "_models",
        "_rows",
        "_skeleton_genders",
        "_skeletons",
        "_used_rows",
    )

    def __init__(
        self,
        data_root: str | Path,
        rows: tuple[AmassClipRow, ...],
        *,
        models_by_gender: Mapping[str, SmplLbsModel],
    ) -> None:
        """Bind concrete clip rows and the three compact licensed SMPL mechanics.

        Args:
            data_root: Directory containing the row-declared native AMASS files.
            rows: Ordered concrete clip rows carrying immutable source metadata.
            models_by_gender: Female, male, and neutral compact SMPL models.
        """
        self._data_root = Path(data_root)
        if not self._data_root.is_dir():
            raise FileNotFoundError(self._data_root)
        self._rows = rows
        if not self._rows or len({row.relative_path for row in self._rows}) != len(self._rows):
            raise ValueError("AMASS requires nonempty concrete rows with unique relative paths.")

        self._models = dict(models_by_gender)
        if set(self._models) != {"female", "male", "neutral"} or any(
            not isinstance(model, SmplLbsModel) or model.gender != gender for gender, model in self._models.items()
        ):
            raise ValueError("AMASS requires matching female, male, and neutral compact SMPL models.")
        if any(model.device.type != "cpu" for model in self._models.values()):
            raise ValueError("AMASS source models must enter through one CPU artifact boundary.")
        self._mechanics = {gender: smpl_body_core_mechanics(model) for gender, model in self._models.items()}
        self._model_cache: dict[tuple[str, str], SmplLbsModel] = {
            (gender, "cpu"): model for gender, model in self._models.items()
        }
        self._skeletons: tuple[MotionSkeleton, ...] = ()
        self._skeleton_genders: dict[str, str] = {}
        self._index: MotionClipIndex | None = None
        self._used_rows: set[int] = set()

    def inspect(self) -> MotionClipIndex:
        """Group shape-specific mechanics using only concrete row metadata."""
        if self._index is not None:
            return self._index
        skeletons = []
        skeleton_ids: dict[str, int] = {}
        skeletons_by_shape: dict[tuple[str, tuple[float, ...]], MotionSkeleton] = {}
        clips = []
        for row in self._rows:
            mechanics = self._mechanics.get(row.gender)
            if mechanics is None:
                raise ValueError(f"AMASS clip {row.relative_path!r} has no declared {row.gender!r} mechanics.")
            shape_key = (row.gender, row.betas[: mechanics.beta_count])
            skeleton = skeletons_by_shape.get(shape_key)
            if skeleton is None:
                skeleton = mechanics.skeleton(np.asarray(row.betas, dtype=np.float64))
                skeletons_by_shape[shape_key] = skeleton
            skeleton_id = skeleton_ids.get(skeleton.identity_sha256)
            if skeleton_id is None:
                skeleton_id = len(skeletons)
                skeleton_ids[skeleton.identity_sha256] = skeleton_id
                skeletons.append(skeleton)
                self._skeleton_genders[skeleton.identity_sha256] = row.gender
            elif self._skeleton_genders[skeleton.identity_sha256] != row.gender:
                raise ValueError("One AMASS source skeleton identity resolved to multiple genders.")
            clips.append(
                MotionClipIndex.Clip(
                    clip_id=row.relative_path,
                    frame_count=row.source_frame_count,
                    source_fps=row.source_fps,
                    content_sha256=row.source_sha256,
                    skeleton_id=skeleton_id,
                )
            )
        self._skeletons = tuple(skeletons)
        self._index = MotionClipIndex(
            source_content_sha256=amass_clip_rows_content_sha256(self._rows),
            skeleton_identity_sha256s=tuple(skeleton.identity_sha256 for skeleton in self._skeletons),
            clips=tuple(clips),
        )
        return self._index

    def skeleton(self, skeleton_id: int) -> MotionSkeleton:
        """Return one exact body-shape source skeleton."""
        self.inspect()
        if type(skeleton_id) is not int or skeleton_id < 0 or skeleton_id >= len(self._skeletons):
            raise ValueError(f"Unknown skeleton id: {skeleton_id!r}.")
        return self._skeletons[skeleton_id]

    def smpl_subject_model(
        self,
        skeleton_identity_sha256: str,
        device: str | torch.device,
    ) -> SmplLbsModel:
        """Return cached subject-shape mechanics for one source skeleton."""
        self.inspect()
        gender = self._skeleton_genders.get(skeleton_identity_sha256)
        if gender is None:
            raise ValueError("Unknown AMASS source-skeleton identity.")
        device_key = str(torch.device(device))
        key = (gender, device_key)
        model = self._model_cache.get(key)
        if model is None:
            model = self._models[gender].to(device)
            self._model_cache[key] = model
        return model

    def clips(self, clip_indices: tuple[int, ...]) -> Iterator[tuple[int, AmassSmplhClip]]:
        """Yield selected raw clips with their explicit mechanics view."""
        index = self.inspect()
        if any(
            type(clip_index) is not int or clip_index < 0 or clip_index >= len(index.clips)
            for clip_index in clip_indices
        ):
            raise IndexError("AMASS clip index is out of range.")
        if len(set(clip_indices)) != len(clip_indices):
            raise ValueError("AMASS clip indices must be unique within one stream request.")
        if set(clip_indices).intersection(self._used_rows):
            raise ValueError("AMASS clip rows may be decoded only once.")
        for clip_index in clip_indices:
            row = self._rows[clip_index]
            clip = _load_verified_amass_smplh_clip(self._data_root / row.relative_path, row.source_sha256)
            if (
                clip.gender != row.gender
                or clip.frame_count != row.source_frame_count
                or clip.source_fps != row.source_fps
                or not np.array_equal(clip.betas, np.asarray(row.betas, dtype=np.float64))
            ):
                raise ValueError(f"AMASS source metadata changed for {row.relative_path!r}.")
            mechanics = self._mechanics[row.gender]
            skeleton = mechanics.skeleton(clip.betas)
            expected = index.skeleton_identity_sha256s[index.clips[clip_index].skeleton_id]
            if skeleton.identity_sha256 != expected:
                raise ValueError("AMASS source mechanics changed after inspection.")
            self._used_rows.add(clip_index)
            yield clip_index, replace(clip, pose_body_indices=mechanics.pose_body_indices)

    def close(self) -> None:
        """Complete the source lifetime; no native file is held open between calls."""
        self._used_rows.update(range(len(self._rows)))


def open_amass_smplh_source(
    path: Path,
    source_root: Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
    verified_artifact_sha256: str,
) -> AmassSmplhClips:
    """Open raw AMASS clips from one verified concrete-row artifact."""
    if verified_artifact_sha256 != split.artifact_sha256:
        raise ValueError("Raw AMASS split digest was not verified by the source boundary.")
    if source.source_fps is not None:
        raise ValueError("Raw AMASS clips declare their native mocap frame rate.")
    if source.clip_directory is None:
        raise ValueError("Raw AMASS requires an explicit source-root-relative clip_directory.")

    rows = read_amass_clip_rows(path)
    data_root = source_root / source.clip_directory
    source_content_sha256 = amass_clip_rows_content_sha256(rows)
    if source_content_sha256 != split.source_content_sha256:
        raise ValueError(
            f"Raw AMASS source content hash differs for {split.name}: "
            f"expected {split.source_content_sha256}, got {source_content_sha256}."
        )

    dependency_paths = source.resolve_dependencies(source_root)
    expected_dependencies = {"smpl_female", "smpl_male", "smpl_neutral"}
    if set(dependency_paths) != expected_dependencies:
        raise ValueError(f"Raw AMASS requires exactly these compact dependencies: {sorted(expected_dependencies)}.")
    dependencies = {dependency.name: dependency for dependency in source.dependencies}
    models = {
        gender: load_smpl_lbs_model(
            dependency_paths[f"smpl_{gender}"],
            artifact_sha256=dependencies[f"smpl_{gender}"].artifact_sha256,
        )
        for gender in ("female", "male", "neutral")
    }
    clips = AmassSmplhClips(
        data_root,
        rows,
        models_by_gender=models,
    )
    index = clips.inspect()
    if len(index.clips) != split.clip_count or index.total_frames != split.frame_count:
        raise ValueError(
            f"Raw AMASS split counts differ for {split.name}: "
            f"expected {split.clip_count}/{split.frame_count}, "
            f"got {len(index.clips)}/{index.total_frames}."
        )
    return clips
