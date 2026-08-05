# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode retarget-pipeline ``motion_retarget_named_inspection_v5`` dump payloads.

bfm-converter-20260805: the training campaign's data seam. Each split is an
immutable frozen index artifact (``*.split_index_v1.json``) listing accepted
per-clip ``.pt`` payloads by content hash; payload bytes are re-verified at
decode. Clips ride the EXACT coordinate route: the source skeletons are the
packaged exact-route contracts reused verbatim, so the resolved coordinate
identity equals the route profile constants by construction, and
:meth:`RetargetDumpV5Clip.local_pose` refuses semantic decoding so a silent
trajectory re-solve is structurally impossible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from ...identity import file_sha256, validate_sha256
from ..clip_index import MotionClipIndex
from ..skeleton import MotionSkeleton
from ..source import MotionSourceCfg
from .cmu_humenv_smpl_coordinates import HUMENV_SMPL_COORDINATE_PROFILE_SHA256, cmu_humenv_smpl_skeleton
from .lafan_g1_29dof_coordinates import lafan_g1_29dof_skeleton

INDEX_SCHEMA = "bfm_retarget_dump_split_index_v1"
PAYLOAD_SCHEMA = "motion_retarget_named_inspection_v5"
TARGET_FPS = 30.0
_STRIDE_RATIO_TOLERANCE = 1.0e-4
"""Relative bound on ``source_fps / (TARGET_FPS * stride) - 1``.

Dump payloads store ``frame_dt_s`` as float32 (e.g. LAFAN 0.033333 -> 30.0003 Hz),
so the analytic route's 1e-6 integer-multiple law cannot be met bitwise; the
quantization error class is <= 1e-5.
"""

_ROUTES = ("cmu_smpl", "lafan_g1")


def _route_skeleton(route: str) -> MotionSkeleton:
    """Return the exact-route source skeleton contract for one dump route."""
    if route == "lafan_g1":
        skeleton = lafan_g1_29dof_skeleton()
        # Lazy import: robots.* consumes motion.data; only the constant is needed here.
        from ...robots.g1.reference import G1_EXACT_COORDINATE_PROFILE_SHA256

        expected = G1_EXACT_COORDINATE_PROFILE_SHA256
    elif route == "cmu_smpl":
        skeleton = cmu_humenv_smpl_skeleton()
        expected = HUMENV_SMPL_COORDINATE_PROFILE_SHA256
    else:
        raise ValueError(f"Unsupported retarget dump route: {route!r}.")
    if skeleton.coordinate_identity_sha256 != expected:
        raise ValueError(
            f"Retarget dump {route!r} skeleton no longer matches the exact-route coordinate profile: "
            f"expected {expected}, got {skeleton.coordinate_identity_sha256}. "
            "The v5-dump source must resolve the EXACT projection; refusing a silent trajectory re-solve."
        )
    return skeleton


def resolve_stride(frame_dt_s: float, route: str) -> int:
    """Return the integer decimation stride from the native clock to ``TARGET_FPS``."""
    if not math.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
        raise ValueError("Retarget dump frame_dt_s must be finite and positive [s].")
    source_fps = 1.0 / frame_dt_s
    stride = round(source_fps / TARGET_FPS)
    if stride < 1 or abs(source_fps / (TARGET_FPS * stride) - 1.0) > _STRIDE_RATIO_TOLERANCE:
        raise ValueError(
            f"Retarget dump clock {source_fps:.6f} Hz is not an integer multiple of {TARGET_FPS} Hz "
            f"within {_STRIDE_RATIO_TOLERANCE} ({route!r})."
        )
    return stride


@dataclass(frozen=True, slots=True)
class RetargetDumpV5Clip:
    """One decoded, target-ordered, target-clock retarget dump clip."""

    joint_q: torch.Tensor
    """Free-root generalized positions (world xyz [m], xyzw quaternion, scalar hinges [rad])."""

    route: str
    """Producing route, ``cmu_smpl`` or ``lafan_g1``."""

    def __post_init__(self) -> None:
        """Validate the decoded exact-coordinate boundary once."""
        if self.route not in _ROUTES:
            raise ValueError(f"Unsupported retarget dump route: {self.route!r}.")
        if (
            self.joint_q.ndim != 2
            or self.joint_q.shape[0] < 3
            or self.joint_q.dtype is not torch.float32
            or not self.joint_q.is_contiguous()
        ):
            raise ValueError("Retarget dump clips require contiguous float32 [frame_count>=3, 7+n] positions.")
        if not bool(torch.isfinite(self.joint_q).all()):
            raise ValueError("Retarget dump generalized positions must contain only finite values.")

    @property
    def source_fps(self) -> float:
        """Converter-owned output sample rate [Hz]."""
        return TARGET_FPS

    @property
    def frame_count(self) -> int:
        """Number of target-clock frames."""
        return int(self.joint_q.shape[0])

    def free_root_coordinates(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode exact free-root coordinates; SMPL derives stored-edge velocities."""
        if self.joint_q.shape[1] != 7 + source_skeleton.num_joints:
            raise ValueError("Retarget dump coordinates differ from the declared source skeleton width.")
        joint_q = self.joint_q.to(device)
        if self.route == "lafan_g1":
            return joint_q, None
        return joint_q, _smpl_stored_edge_velocity(joint_q, source_skeleton)

    def local_pose(
        self,
        source_skeleton: MotionSkeleton,
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refuse semantic decoding: v5 dump clips are exact-route only (risk-4 guard)."""
        raise RuntimeError(
            "Retarget dump v5 clips must ride the EXACT coordinate route; a semantic local_pose request "
            "means the builder fell back to trajectory re-solving, which would measure the wrong solver."
        )


def _smpl_stored_edge_velocity(joint_q: torch.Tensor, source_skeleton: MotionSkeleton) -> torch.Tensor:
    """Derive the tree's stored-edge SMPL velocity policy at the converter clock.

    Mirrors the SMPL target's derived-velocity path (root gradient, root-local
    angular velocity, ``smpl_joint_velocity_stored_warp`` joint rates) for one
    clip, because the exact route requires native generalized velocities.
    """
    import warp as wp
    from isaaclab.utils.math import quat_apply_inverse

    # Lazy imports keep motion.data free of import-time robots/kinematics dependencies.
    from ....kinematics import time_gradient_segmented, time_quaternion_angular_velocity_segmented
    from ...robots.smpl.reference_warp import smpl_joint_velocity_stored_warp

    frame_count = joint_q.shape[0]
    joint_count = source_skeleton.num_bodies - 1
    if any(axis is None for axis in source_skeleton.joint_axes) or source_skeleton.num_joints != 3 * joint_count:
        raise ValueError("SMPL stored-edge velocities require one ordered three-hinge chain per non-root body.")
    offsets = torch.tensor((0, frame_count), dtype=torch.int64, device=joint_q.device)
    step_seconds = torch.tensor((1.0 / TARGET_FPS,), dtype=torch.float32, device=joint_q.device)
    root_linear = time_gradient_segmented(joint_q[:, :3], offsets, step_seconds)
    angular_world = time_quaternion_angular_velocity_segmented(joint_q[:, 3:7], offsets, step_seconds)
    angular_local = quat_apply_inverse(joint_q[:, 3:7], angular_world)
    axes = torch.tensor(source_skeleton.joint_axes, dtype=torch.float32, device=joint_q.device)
    axes = axes.view(joint_count, 3, 3).contiguous()
    joint_velocity = torch.empty((frame_count, 3 * joint_count), dtype=torch.float32, device=joint_q.device)
    wp.launch(
        smpl_joint_velocity_stored_warp,
        dim=(frame_count, joint_count),
        inputs=[
            wp.from_torch(joint_q.contiguous()),
            wp.from_torch(axes),
            wp.from_torch(offsets),
            wp.from_torch(step_seconds),
            1,
            frame_count,
            joint_count,
        ],
        outputs=[wp.from_torch(joint_velocity)],
        device=str(joint_q.device),
    )
    return torch.cat((root_linear, angular_local, joint_velocity), dim=-1).contiguous()


def _validated_index(payload: object) -> dict[str, object]:
    """Validate the frozen split-index envelope."""
    if not isinstance(payload, dict) or payload.get("schema") != INDEX_SCHEMA:
        raise ValueError("Retarget dump split index does not use the frozen v1 schema.")
    route = payload.get("route")
    if route not in _ROUTES:
        raise ValueError(f"Retarget dump split index declares an unsupported route: {route!r}.")
    if payload.get("payload_schema") != PAYLOAD_SCHEMA:
        raise ValueError("Retarget dump split index does not bind v5 inspection payloads.")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Retarget dump split index must list at least one accepted clip.")
    clip_ids = [clip.get("clip_id") for clip in clips]
    if clip_ids != sorted(clip_ids) or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Retarget dump split index clips must be unique and sorted by clip_id.")
    validate_sha256("split index manifest_sha256", payload.get("manifest_sha256"))
    return payload


class RetargetDumpV5Clips:
    """Frozen-index retarget dump split decoded one payload file at a time."""

    __slots__ = ("_entries", "_index", "_route", "_skeleton", "_source_root", "_source_sha256")

    def __init__(self, index_payload: dict[str, object], source_root: Path, source_sha256: str) -> None:
        self._route: str = index_payload["route"]  # type: ignore[assignment]
        self._entries: list[dict[str, object]] | None = list(index_payload["clips"])  # type: ignore[arg-type]
        self._source_root = source_root
        self._source_sha256 = source_sha256
        self._skeleton = _route_skeleton(self._route)
        self._index: MotionClipIndex | None = None

    def _require_open(self) -> list[dict[str, object]]:
        if self._entries is None:
            raise RuntimeError("Retarget dump v5 source is closed.")
        return self._entries

    def inspect(self) -> MotionClipIndex:
        """Return the frozen accepted-clip metadata at the converter clock."""
        if self._index is not None:
            return self._index
        descriptors = []
        for entry in self._require_open():
            stride = resolve_stride(float(entry["frame_dt_s"]), self._route)
            if stride != int(entry["stride"]):
                raise ValueError(f"Retarget dump stride drifted for {entry['clip_id']!r}.")
            output_frames = len(range(0, int(entry["frame_count"]), stride))
            if output_frames != int(entry["output_frame_count"]) or output_frames < 3:
                raise ValueError(f"Retarget dump output clock drifted for {entry['clip_id']!r}.")
            descriptors.append(
                MotionClipIndex.Clip(
                    clip_id=str(entry["clip_id"]),
                    frame_count=output_frames,
                    source_fps=TARGET_FPS,
                    content_sha256=str(entry["payload_sha256"]),
                    skeleton_id=0,
                )
            )
        self._index = MotionClipIndex(
            source_content_sha256=self._source_sha256,
            skeleton_identity_sha256s=(self._skeleton.identity_sha256,),
            clips=tuple(descriptors),
        )
        return self._index

    def skeleton(self, skeleton_id: int) -> MotionSkeleton:
        """Return the reused packaged exact-route coordinate contract."""
        if type(skeleton_id) is not int or skeleton_id != 0:
            raise ValueError(f"Unknown skeleton id: {skeleton_id!r}.")
        return self._skeleton

    def _decode(self, entry: dict[str, object]) -> RetargetDumpV5Clip:
        """Verify and decode one payload file into target order and clock."""
        clip_id = str(entry["clip_id"])
        path = self._source_root / str(entry["file"])
        if not path.is_file():
            raise FileNotFoundError(f"Retarget dump payload does not exist: {path}")
        actual = file_sha256(path)
        if actual != entry["payload_sha256"]:
            raise ValueError(f"Retarget dump payload hash differs for {clip_id!r}: got {actual}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        joint_q = payload["joint_q"]
        names = tuple(name for _, name in payload["joint_coordinate_names"])
        q_rows = payload["q_rows"].tolist()
        if (
            payload.get("schema") != PAYLOAD_SCHEMA
            or tuple(payload["clip_ids"]) != (str(entry["payload_clip_id"]),)
            or int(payload["frame_count"]) != int(entry["frame_count"])
            or not bool(payload["accepted"][0])
            or joint_q.shape != (int(entry["frame_count"]), 7 + len(names))
            or len(q_rows) != len(names)
        ):
            raise ValueError(f"Retarget dump payload {clip_id!r} differs from its frozen index entry.")
        if self._route == "cmu_smpl":
            from ...robots.smpl.articulation import smpl_live_joint_mujoco_names

            names = smpl_live_joint_mujoco_names(names)
        column_by_name = {name: q_rows[position] for position, name in enumerate(names)}
        if set(column_by_name) != set(self._skeleton.joint_names):
            raise ValueError(f"Retarget dump payload {clip_id!r} joint names differ from the route contract.")
        columns = torch.tensor(
            [column_by_name[name] for name in self._skeleton.joint_names], dtype=torch.int64
        )
        ordered = torch.cat((joint_q[:, :7], joint_q.index_select(1, columns)), dim=-1)
        stride = int(entry["stride"])
        return RetargetDumpV5Clip(joint_q=ordered[::stride].contiguous(), route=self._route)

    def clips(self, clip_indices: tuple[int, ...]) -> Iterator[tuple[int, RetargetDumpV5Clip]]:
        """Yield selected decoded clips as source-index and clip pairs."""
        index = self.inspect()
        entries = self._require_open()
        for clip_index in clip_indices:
            if type(clip_index) is not int or clip_index < 0 or clip_index >= len(index.clips):
                raise IndexError(f"Clip index is out of range: {clip_index!r}.")
            yield clip_index, self._decode(entries[clip_index])

    def close(self) -> None:
        """Release the retained index entries."""
        self._entries = None


def _split(name: str, artifact: str, sha256: str, clip_count: int, frame_count: int) -> MotionSourceCfg.SplitCfg:
    """Pin one frozen split index (the index file is its own content identity)."""
    return MotionSourceCfg.SplitCfg(
        name=name,
        artifact=artifact,
        artifact_sha256=sha256,
        source_content_sha256=sha256,
        clip_count=clip_count,
        frame_count=frame_count,
    )


def open_retarget_dump_v5_source(
    path: Path,
    source_root: Path,
    split: MotionSourceCfg.SplitCfg,
    source: MotionSourceCfg,
    verified_artifact_sha256: str,
) -> RetargetDumpV5Clips:
    """Open one frozen retarget dump split index."""
    if not path.is_relative_to(source_root):
        raise ValueError("Retarget dump split index must reside below its explicit source root.")
    if verified_artifact_sha256 != split.artifact_sha256 or verified_artifact_sha256 != split.source_content_sha256:
        raise ValueError("Retarget dump split index digest was not verified by the source boundary.")
    if source.clip_directory is not None or source.source_fps is not None:
        raise ValueError("Retarget dump sources declare per-clip clocks and index-listed payload files.")
    index_payload = _validated_index(json.loads(path.read_text()))
    clips = index_payload["clips"]
    total_output_frames = sum(int(entry["output_frame_count"]) for entry in clips)  # type: ignore[union-attr]
    if len(clips) != split.clip_count or total_output_frames != split.frame_count:  # type: ignore[arg-type]
        raise ValueError(
            f"Retarget dump split counts differ from the registration pins: "
            f"clips={len(clips)}, frames={total_output_frames}."  # type: ignore[arg-type]
        )
    return RetargetDumpV5Clips(index_payload, source_root, verified_artifact_sha256)


# BFM campaign 2026-08-05 (bfm-converter-20260805) our-data registrations. Counts pin the
# fail-closed ACCEPTED subsets from the campaign dump MANIFEST (sha256 f162299a1aa1...baf3)
# at the converter-owned 30 Hz clock; the decoder owns its pins so the composition root
# stays a readable alias site.
CMU_RETARGET_DUMP_SOURCE = MotionSourceCfg(
    identifier="cmu_smpl_retarget_dump_v5",
    open_source=open_retarget_dump_v5_source,
    format="split_index_v1_over_motion_retarget_named_inspection_v5_pt_payloads",
    semantic_level="smpl_robot_generalized_positions",
    decoder_version="retarget_dump_v5_exact_v1",
    source_fps=None,
    license="amass_cmu_and_smpl_registered_source_required",
    clip_directory=None,
    train=_split(
        "train",
        "cmu_smpl_train.split_index_v1.json",
        "aad4bf468baa3d4dc92944454d182114ef9cb08ce3c9751d0785ded7777efdde",
        1_553,
        682_501,
    ),
    evaluation=_split(
        "test",
        "cmu_smpl_test.split_index_v1.json",
        "cd481d988e768ffb9be8b712601e8ea10a807079ee700e92edf6a44e331222a6",
        168,
        76_250,
    ),
)
LAFAN_RETARGET_DUMP_SOURCE = MotionSourceCfg(
    identifier="lafan_g1_retarget_dump_v5",
    open_source=open_retarget_dump_v5_source,
    format="split_index_v1_over_motion_retarget_named_inspection_v5_pt_payloads",
    semantic_level="robot_pose_g1_not_canonical_lafan",
    decoder_version="retarget_dump_v5_exact_v1",
    source_fps=None,
    license="ubisoft_laforge_lafan1_research_dataset",
    clip_directory=None,
    train=_split(
        "train",
        "lafan_g1_train.split_index_v1.json",
        "87d881b51a624f070009999b81719cb7ccd38215c76a2e4edd48064f95c212da",
        843,
        252_900,
    ),
    evaluation=_split(
        "evaluation",
        "lafan_g1_eval.split_index_v1.json",
        "57a5198e6a3fa3d84a62a379f2f3794376e9b552df81b3285e3d8d057dcb1ce0",
        33,
        223_301,
    ),
)
