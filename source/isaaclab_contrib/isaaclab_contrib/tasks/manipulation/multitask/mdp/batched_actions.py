# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group-aware batched action terms for heterogeneous environments.

Each action term class iterates ``robot_meta`` to discover robot groups and managing per-group
controllers/buffers internally.

``robot_meta`` is keyed by **clone-group name** (not asset name).

Supported action types:

* :class:`BatchedDiffIKAction` — Differential IK (multi-group, shared columns)
* :class:`BatchedBinaryGripperAction` — Binary open/close gripper (multi-group, shared columns)
* :class:`BatchedRelJointPosAction` — Relative joint position delta (single-group per term)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase

from .batched_actions_cfg import (
    BatchedBinaryGripperActionCfg,
    BatchedDiffIKActionCfg,
    BatchedRelJointPosActionCfg,
    DiffIKGroupCfg,
    GripperGroupCfg,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import GroupView


# ============================================================
# Differential IK
# ============================================================


class _IKGroup:
    """Per-group bookkeeping for :class:`BatchedDiffIKAction`."""

    __slots__ = (
        "key",
        "asset",
        "gv",
        "controller",
        "joint_ids",
        "body_idx",
        "jacobi_body_idx",
        "jacobi_joint_ids",
        "scale",
        "raw_actions",
        "offset_pos",
        "offset_rot",
        "offset_skew",
        "offset_rot_mat",
    )

    def __init__(
        self,
        key: str,
        asset: Articulation,
        gv: GroupView,
        gcfg: DiffIKGroupCfg,
        controller: DifferentialIKController,
        joint_ids: list[int],
        body_idx: int,
        scale: torch.Tensor,
        device: str,
    ):
        self.key = key
        self.asset = asset
        self.gv = gv
        self.controller = controller

        self.joint_ids: slice | list[int] = slice(None) if len(joint_ids) == asset.num_joints else joint_ids

        self.body_idx = body_idx
        if asset.is_fixed_base:
            self.jacobi_body_idx = body_idx - 1
            self.jacobi_joint_ids = joint_ids
        else:
            self.jacobi_body_idx = body_idx
            self.jacobi_joint_ids = [i + 6 for i in joint_ids]

        n = gv.count
        self.scale = scale
        self.raw_actions = torch.zeros(n, controller.action_dim, device=device)

        if gcfg.body_offset is not None:
            self.offset_pos = torch.tensor(gcfg.body_offset.pos, device=device).expand(n, -1).clone()
            self.offset_rot = torch.tensor(gcfg.body_offset.rot, device=device).expand(n, -1).clone()
            self.offset_skew = -math_utils.skew_symmetric_matrix(self.offset_pos)
            self.offset_rot_mat = math_utils.matrix_from_quat(self.offset_rot)
        else:
            self.offset_pos = self.offset_rot = None
            self.offset_skew = self.offset_rot_mat = None


class BatchedDiffIKAction(ActionTerm):
    """Batched differential IK action for multi-robot environments.

    Iterates ``groups`` (keyed by clone-group name) to discover robot
    groups and maintains per-group :class:`DifferentialIKController`
    instances.  Asset references and IK body are resolved from the
    corresponding ``robot_meta`` entries.

    All groups share the same action columns since their env rows
    are disjoint.  ``action_dim`` equals the IK controller's
    action dimension (e.g. 6 for ``"pose"`` mode).
    """

    cfg: BatchedDiffIKActionCfg

    def __init__(self, cfg: BatchedDiffIKActionCfg, env: ManagerBasedEnv):
        ManagerTermBase.__init__(self, cfg, env)
        self._asset = None
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None

        layout = env.scene.layout
        robot_meta: dict = cfg.robot_meta or {}
        device = env.device

        self._groups: list[_IKGroup] = []

        for group_key, gcfg in cfg.groups.items():
            if not isinstance(gcfg, DiffIKGroupCfg):
                raise TypeError(f"Expected DiffIKGroupCfg for group '{group_key}', got {type(gcfg).__name__}")
            meta = robot_meta[group_key]
            meta.asset_cfg.resolve(env.scene)
            asset: Articulation = env.scene[meta.asset_cfg.name]
            gv = layout[group_key, meta.asset_cfg.name]

            joint_ids, _ = asset.find_joints(gcfg.joint_names)
            body_ids, _ = asset.find_bodies(meta.asset_cfg.body_names[0])
            body_idx = body_ids[0]

            controller = DifferentialIKController(
                cfg=cfg.controller,
                num_envs=gv.count,
                device=device,
            )

            scale_val = gcfg.scale if gcfg.scale is not None else cfg.scale
            n = gv.count
            dim = controller.action_dim
            scale = torch.tensor(scale_val, device=device).expand(n, dim).clone()

            self._groups.append(_IKGroup(group_key, asset, gv, gcfg, controller, joint_ids, body_idx, scale, device))

        if not self._groups:
            raise ValueError("BatchedDiffIKAction: no groups configured in cfg.groups.")

        self._action_dim = self._groups[0].controller.action_dim
        self._raw = torch.zeros(env.num_envs, self._action_dim, device=device)
        self._processed = torch.zeros_like(self._raw)

    # -- ActionTerm interface --

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @torch.inference_mode()
    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        for g in self._groups:
            group_actions = actions[g.gv.write, : g.controller.action_dim]
            g.raw_actions[:] = group_actions
            processed = group_actions * g.scale
            ee_pos, ee_quat = _ik_frame_pose(g)
            g.controller.set_command(processed, ee_pos, ee_quat)

    @torch.inference_mode()
    def apply_actions(self):
        for g in self._groups:
            ee_pos, ee_quat, jacobian = _ik_apply_data(g)
            joint_pos = wp.to_torch(g.asset.data.joint_pos)[g.gv.read][:, g.joint_ids]
            joint_pos_des = g.controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
            g.asset.set_joint_position_target_index(
                target=joint_pos_des,
                joint_ids=g.joint_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw[:] = 0.0
            for g in self._groups:
                g.raw_actions[:] = 0.0
        else:
            self._raw[env_ids] = 0.0
            ids_t = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self._env.device,
            )
            for g in self._groups:
                local, _ = g.gv.filter(ids_t)
                if local.numel() > 0:
                    g.raw_actions[local] = 0.0


def _ik_frame_pose(g: _IKGroup) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute EE pose in root frame for one group."""
    r = g.gv.read
    ee_pos_w = wp.to_torch(g.asset.data.body_pos_w)[r, g.body_idx]
    ee_quat_w = wp.to_torch(g.asset.data.body_quat_w)[r, g.body_idx]
    root_pos_w = wp.to_torch(g.asset.data.root_pos_w)[r]
    root_quat_w = wp.to_torch(g.asset.data.root_quat_w)[r]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    if g.offset_pos is not None:
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b,
            ee_quat_b,
            g.offset_pos,
            g.offset_rot,
        )
    return ee_pos_b, ee_quat_b


def _ik_apply_data(g: _IKGroup) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute EE pose and body-frame Jacobian in one pass.

    Combines the work of ``_ik_frame_pose`` and the former Jacobian helper
    so that ``root_quat_w`` is fetched only once, and precomputed offset
    matrices (``offset_skew``, ``offset_rot_mat``) are reused.

    Returns:
        Tuple of (ee_pos_b, ee_quat_b, jacobian).
    """
    r = g.gv.read
    ee_pos_w = wp.to_torch(g.asset.data.body_pos_w)[r, g.body_idx]
    ee_quat_w = wp.to_torch(g.asset.data.body_quat_w)[r, g.body_idx]
    root_pos_w = wp.to_torch(g.asset.data.root_pos_w)[r]
    root_quat_w = wp.to_torch(g.asset.data.root_quat_w)[r]

    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )

    jacobian = wp.to_torch(g.asset.root_view.get_jacobians())[r, g.jacobi_body_idx, :, g.jacobi_joint_ids]
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

    if g.offset_pos is not None:
        assert g.offset_skew is not None and g.offset_rot_mat is not None
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b,
            ee_quat_b,
            g.offset_pos,
            g.offset_rot,
        )
        jacobian[:, 0:3, :] += torch.bmm(g.offset_skew, jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(g.offset_rot_mat, jacobian[:, 3:, :])

    return ee_pos_b, ee_quat_b, jacobian


# ============================================================
# Binary gripper
# ============================================================


class _GripperGroup:
    """Per-group bookkeeping for :class:`BatchedBinaryGripperAction`."""

    __slots__ = (
        "key",
        "asset",
        "gv",
        "joint_ids",
        "open_cmd",
        "close_cmd",
        "raw_actions",
        "processed",
    )

    def __init__(
        self,
        key: str,
        asset: Articulation,
        gv: GroupView,
        gcfg: GripperGroupCfg,
        device: str,
    ):
        self.key = key
        self.asset = asset
        self.gv = gv
        n = gv.count

        joint_ids, joint_names = asset.find_joints(gcfg.joint_names)
        self.joint_ids = joint_ids
        num_j = len(joint_ids)

        self.open_cmd = torch.zeros(num_j, device=device)
        idx, _, vals = string_utils.resolve_matching_names_values(
            gcfg.open_command_expr,
            joint_names,
        )
        self.open_cmd[idx] = torch.tensor(vals, device=device)

        self.close_cmd = torch.zeros(num_j, device=device)
        idx, _, vals = string_utils.resolve_matching_names_values(
            gcfg.close_command_expr,
            joint_names,
        )
        self.close_cmd[idx] = torch.tensor(vals, device=device)

        self.raw_actions = torch.zeros(n, 1, device=device)
        self.processed = torch.zeros(n, num_j, device=device)


class BatchedBinaryGripperAction(ActionTerm):
    """Batched binary gripper action for multi-robot environments.

    Iterates ``groups`` to discover which robot groups have grippers.
    All groups share a single binary action column (dim=1).
    """

    cfg: BatchedBinaryGripperActionCfg

    def __init__(self, cfg: BatchedBinaryGripperActionCfg, env: ManagerBasedEnv):
        ManagerTermBase.__init__(self, cfg, env)
        self._asset = None
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None

        layout = env.scene.layout
        robot_meta: dict = cfg.robot_meta or {}
        device = env.device

        self._groups: list[_GripperGroup] = []

        for group_key, gcfg in cfg.groups.items():
            if not isinstance(gcfg, GripperGroupCfg):
                raise TypeError(f"Expected GripperGroupCfg for group '{group_key}', got {type(gcfg).__name__}")
            meta = robot_meta[group_key]
            meta.asset_cfg.resolve(env.scene)
            asset: Articulation = env.scene[meta.asset_cfg.name]
            gv = layout[group_key, meta.asset_cfg.name]
            self._groups.append(_GripperGroup(group_key, asset, gv, gcfg, device))

        if not self._groups:
            raise ValueError("BatchedBinaryGripperAction: no groups configured.")

        self._raw = torch.zeros(env.num_envs, 1, device=device)
        self._processed = torch.zeros_like(self._raw)

    # -- ActionTerm interface --

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @torch.inference_mode()
    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        for g in self._groups:
            group_actions = actions[g.gv.write, :1]
            g.raw_actions[:] = group_actions
            binary_mask = group_actions.squeeze(-1) < 0
            g.processed[binary_mask] = g.close_cmd
            g.processed[~binary_mask] = g.open_cmd

    @torch.inference_mode()
    def apply_actions(self):
        for g in self._groups:
            g.asset.set_joint_position_target_index(
                target=g.processed,
                joint_ids=g.joint_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw[:] = 0.0
            for g in self._groups:
                g.raw_actions[:] = 0.0
        else:
            self._raw[env_ids] = 0.0
            ids_t = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self._env.device,
            )
            for g in self._groups:
                local, _ = g.gv.filter(ids_t)
                if local.numel() > 0:
                    g.raw_actions[local] = 0.0


# ============================================================
# Relative joint position (one group per term)
# ============================================================


class BatchedRelJointPosAction(ActionTerm):
    """Relative joint position action for one robot group.

    Each group registers its own term so that different robots get
    independent action columns.  The term only reads/writes the env
    rows belonging to its group via :class:`GroupView`; other rows
    are zero-padded by the policy.
    """

    cfg: BatchedRelJointPosActionCfg

    def __init__(self, cfg: BatchedRelJointPosActionCfg, env: ManagerBasedEnv):
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None

        layout = env.scene.layout
        robot_meta: dict = cfg.robot_meta or {}
        device = env.device

        meta = robot_meta[cfg.group_name]
        meta.asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[meta.asset_cfg.name]
        self._gv = layout[cfg.group_name, meta.asset_cfg.name]
        self._group_name = cfg.group_name

        joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(joint_ids)
        self._joint_ids: slice | list[int] = slice(None) if self._num_joints == self._asset.num_joints else joint_ids
        self._scale = float(cfg.scale)

        self._group_raw = torch.zeros(
            self._gv.count,
            self._num_joints,
            device=device,
        )
        self._raw = torch.zeros(
            env.num_envs,
            self._num_joints,
            device=device,
        )
        self._processed = torch.zeros_like(self._raw)

    # -- ActionTerm interface --

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @torch.inference_mode()
    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._group_raw[:] = actions[self._gv.write, : self._num_joints]

    @torch.inference_mode()
    def apply_actions(self):
        current = wp.to_torch(self._asset.data.joint_pos)[self._gv.read][:, self._joint_ids]
        target = current + self._group_raw * self._scale
        self._asset.set_joint_position_target_index(
            target=target,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw[:] = 0.0
            self._group_raw[:] = 0.0
        else:
            self._raw[env_ids] = 0.0
            ids_t = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self._env.device,
            )
            local, _ = self._gv.filter(ids_t)
            if local.numel() > 0:
                self._group_raw[local] = 0.0
