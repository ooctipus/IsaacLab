# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime writer for canonical reset-state banks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .reset_state_bank import ResetStateBank

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _runtime_asset_types() -> tuple[type[Articulation], type[RigidObject]]:
    """Import concrete runtime asset types only when a live scene is bound."""
    from isaaclab.assets import Articulation, RigidObject

    return Articulation, RigidObject


class ResetStateWriter:
    """Bind a canonical state-bank layout to runtime scene entities once.

    Args:
        env: Runtime manager-based environment owning the scene entities.
        states: Canonical simulator-free reset states.
        reset_assets: Exact entity names in canonical table order.
        states_relative: Whether stored root positions are relative to environment origins.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        states: ResetStateBank,
        reset_assets: tuple[str, ...],
        states_relative: bool,
    ) -> None:
        if reset_assets != states.layout.names:
            raise ValueError(
                f"Reset assets must exactly match the canonical state layout: {reset_assets} != {states.layout.names}."
            )
        if states.root_pose.device != torch.device(env.device):
            raise ValueError(
                f"Reset-state bank and runtime environment must share one device: "
                f"{states.root_pose.device} != {env.device}."
            )
        self._env = env
        self._states = states
        self._entities: tuple[Articulation | RigidObject, ...] = self._bind_entities()
        capacity = env.num_envs
        device = states.root_pose.device
        max_joint_count = max(
            stop - start for start, stop in zip(states.layout.joint_offsets[:-1], states.layout.joint_offsets[1:])
        )
        self._root_pose = torch.empty(capacity, 7, dtype=torch.float32, device=device)
        self._root_velocity = torch.empty(capacity, 6, dtype=torch.float32, device=device)
        self._joint_position = torch.empty(capacity, max_joint_count, dtype=torch.float32, device=device)
        self._joint_velocity = torch.empty_like(self._joint_position)
        self._origins = torch.empty(capacity, 3, dtype=torch.float32, device=device) if states_relative else None

    def _bind_entities(self) -> tuple[Articulation | RigidObject, ...]:
        articulation_type, rigid_object_type = _runtime_asset_types()
        entities: list[Articulation | RigidObject] = []
        layout = self._states.layout
        for name, kind, joint_names in zip(layout.names, layout.kinds, layout.joint_names, strict=True):
            entity = self._env.scene[name]
            entity_type = articulation_type if kind == "articulation" else rigid_object_type
            if not isinstance(entity, entity_type):
                raise TypeError(f"Reset entity {name!r} must be {entity_type.__name__}, got {type(entity).__name__}.")
            if kind == "articulation" and tuple(entity.joint_names) != joint_names:
                raise ValueError(
                    f"Runtime articulation {name!r} joint order differs from the table: "
                    f"{tuple(entity.joint_names)} != {joint_names}."
                )
            entities.append(entity)
        return tuple(entities)

    def write(self, env_ids: torch.Tensor, state_rows: torch.Tensor) -> None:
        """Write selected state rows into matching runtime environments.

        Args:
            env_ids: Runtime environment indices, shape ``[count]``.
            state_rows: State-bank row indices paired with :paramref:`env_ids`, shape ``[count]``.
        """
        states = self._states
        count = env_ids.shape[0]
        root_pose = self._root_pose[:count]
        root_velocity = self._root_velocity[:count]
        origins = self._origins[:count] if self._origins is not None else None
        if origins is not None:
            torch.index_select(self._env.scene.env_origins, 0, env_ids, out=origins)
        for entity_index, (name, kind, entity) in enumerate(
            zip(states.layout.names, states.layout.kinds, self._entities, strict=True)
        ):
            torch.index_select(states.root_pose[:, entity_index], 0, state_rows, out=root_pose)
            if origins is not None:
                root_pose[:, :3].add_(origins)
            torch.index_select(states.root_velocity[:, entity_index], 0, state_rows, out=root_velocity)
            entity.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids, skip_forward=True)
            entity.write_root_velocity_to_sim_index(
                root_velocity=root_velocity,
                env_ids=env_ids,
                skip_forward=kind == "articulation",
            )
            if kind == "articulation":
                joint_slice = states.layout.joint_slice(name)
                joint_count = joint_slice.stop - joint_slice.start
                joint_position = self._joint_position[:count, :joint_count]
                joint_velocity = self._joint_velocity[:count, :joint_count]
                torch.index_select(states.joint_position[:, joint_slice], 0, state_rows, out=joint_position)
                torch.index_select(states.joint_velocity[:, joint_slice], 0, state_rows, out=joint_velocity)
                entity.write_joint_position_to_sim_index(
                    position=joint_position,
                    env_ids=env_ids,
                    skip_forward=True,
                )
                entity.write_joint_velocity_to_sim_index(
                    velocity=joint_velocity,
                    env_ids=env_ids,
                )
