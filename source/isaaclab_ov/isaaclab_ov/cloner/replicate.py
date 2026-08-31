# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX replication hook for IsaacLab's cloning pipeline.

Called from the scene cloning path in place of immediate PhysX or Newton
replication.  Unlike those replicators, ovphysx.PhysX does not exist yet at
this point in the scene setup — it is created lazily on the first
:meth:`~isaaclab_ov.physics.OvPhysxManager.reset` call.

This function records an active clone recipe on :class:`OvPhysxManager`.  When
:meth:`~isaaclab_ov.physics.OvPhysxManager._warmup_and_load` eventually
creates the ``PhysX`` instance, env-0-only loads replay each recipe via
``physx.clone(source, targets, transforms)`` after loading. Full-stage loads instead
materialize or overlay every recipe in serialized USDA before attaching OVStage.
Recipes remain active for the current simulation context so a forced re-warmup
rebuilds the same topology without modifying the live USD stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab import cloner

from isaaclab_ov._clone import CloneTransform

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def _select_env_ids(env_ids: torch.Tensor, mapping: torch.Tensor, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    row_mask = mapping[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


def _matrix_to_clone_transform(matrix: Gf.Matrix4d) -> CloneTransform:
    """Convert a USD pose matrix to an OvPhysX xyzw clone transform."""
    matrix = matrix.RemoveScaleShear()
    position = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quaternion.GetReal()),
    )


def _pose_tensor_rows(tensor: torch.Tensor | None, name: str, component_count: int) -> list[list[float]] | None:
    """Validate and copy an optional per-environment pose tensor to CPU rows."""
    if tensor is None:
        return None
    if tensor.ndim != 2 or tensor.shape[1] != component_count:
        raise ValueError(f"{name} must have shape [num_envs, {component_count}], got {list(tensor.shape)}.")
    return tensor.detach().cpu().tolist()


def _validate_pose_rows(name: str, rows: list[list[float]] | None, env_ids: Sequence[int]) -> None:
    """Validate that optional pose rows contain every selected environment."""
    if rows is None:
        return
    for env_id in env_ids:
        if env_id < 0 or env_id >= len(rows):
            raise ValueError(f"{name} does not contain selected environment id {env_id}; it has {len(rows)} rows.")


class OvReplicateContext:
    """Apply one clone-plan mapping to an OvPhysX simulation."""

    replicate_priority = 0
    clones_whole_env = True

    def __init__(self, sim_context: SimulationContext):
        """Initialize the context.

        Args:
            sim_context: Simulation context that owns this clone backend.
        """
        self._sim = sim_context
        self.stage = sim_context.stage
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

    def replicate(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Publish clone operations from the current flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mapping: Bool/int mask selecting envs per source.
            positions: Optional per-environment world positions [m], shape
                ``[num_envs, 3]``.
            quaternions: Optional per-environment orientations in xyzw order,
                shape ``[num_envs, 4]``.

        Raises:
            ValueError: If a provided pose tensor is malformed or lacks a selected
                environment, or if an active source or source anchor prim is invalid.
        """
        positions_list = _pose_tensor_rows(positions, "positions", 3)
        quaternions_list = _pose_tensor_rows(quaternions, "quaternions", 4)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for i, src in enumerate(sources):
            active_env_ids = [int(env_id) for env_id in _select_env_ids(env_ids, mapping, i).tolist()]
            if not active_env_ids:
                continue
            _validate_pose_rows("positions", positions_list, active_env_ids)
            _validate_pose_rows("quaternions", quaternions_list, active_env_ids)

            self_env_id: int | None = None
            matched = cloner.path.match(src, destinations[i])
            if matched is not None and matched.instance.isdigit():
                self_env_id = int(matched.instance)

            source_prim = self.stage.GetPrimAtPath(src)
            if not source_prim.IsValid():
                raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {src}")
            source_world = xform_cache.GetLocalToWorldTransform(source_prim).RemoveScaleShear()
            if self_env_id is None:
                source_anchor_world = Gf.Matrix4d(1.0)
            else:
                prefix, _ = cloner.path.split(destinations[i])
                source_anchor_path = f"{prefix}{self_env_id}"
                source_anchor = self.stage.GetPrimAtPath(source_anchor_path)
                if not source_anchor.IsValid():
                    raise ValueError(
                        f"OvPhysX clone source anchor prim is not valid on the stage: {source_anchor_path}"
                    )
                source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
            source_relative = source_world * source_anchor_world.GetInverse()

            targets: list[str] = []
            target_transforms: list[CloneTransform] = []
            for env_id in active_env_ids:
                if env_id == self_env_id:
                    continue
                targets.append(destinations[i].format(env_id))

                target_env_world = Gf.Matrix4d(1.0)
                if positions_list is not None:
                    pos = positions_list[env_id]
                    target_env_world.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                if quaternions_list is not None:
                    quat = quaternions_list[env_id]
                    target_env_world.SetRotateOnly(
                        Gf.Quatd(
                            float(quat[3]),
                            Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2])),
                        )
                    )
                target_transforms.append(_matrix_to_clone_transform(source_relative * target_env_world))

            if targets:
                self._sim.physics_manager._register_clone_transforms(src, targets, target_transforms)
