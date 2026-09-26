# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered, topology-compatible keyboards applied at episode reset."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner.newton_clone_utils import replace_newton_builder_shape_colors
from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ModelFlags, ShapeFlags
from newton.solvers import SolverMuJoCo

from pxr import Usd, UsdGeom

from .keyboards.keyboard_gen_cfg import KeyboardSpawnerCfg
from .keyboards.keyboard_geometry import generate_keyboard
from .keyboards.keyboard_usd import spawn_keyboard


@wp.kernel
def _copy_variant_property(
    worlds: wp.array[int],
    variants: wp.array[int],
    ids: wp.array2d[int],
    values: wp.array2d[Any],
    output: wp.array[Any],
):
    row, column = wp.tid()
    output[ids[worlds[row], column]] = values[variants[row], column]


class KeyboardVariants:
    """Own immutable variant resources and one variant ID per world; never own live physics state.

    Variants have 18 fixed-root partitions of six scalar sliders. Collision shapes
    are boxes; cap and label meshes are pre-registered visual resources. The solver's
    existing property notification synchronizes joint, shape, and inertial changes.
    """

    def __init__(self, env, configs: tuple[KeyboardSpawnerCfg, ...]):
        if not configs:
            raise ValueError("At least one keyboard variant is required.")
        self.env = env
        self.layouts = tuple(generate_keyboard(cfg) for cfg in configs)
        for cfg, layout in zip(configs, self.layouts):
            if (
                not cfg.uniform_key_shapes
                or layout.partition_mode != "fixed_dof"
                or layout.partition_dof != 6
                or layout.partition_count != 18
                or layout.active_key_count % 6
            ):
                raise ValueError("Keyboard variants require 18 six-DOF slots and active counts divisible by six.")
        self.variant_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.counts = torch.tensor([k.active_key_count for k in self.layouts], device=env.device)
        self.backspaces = torch.tensor(
            [
                next(k.slot for k in layout.active_keys if k.label.lower() in ("backspace", "bksp"))
                for layout in self.layouts
            ],
            device=env.device,
        )
        self.labels = tuple(tuple(key.label for key in layout.keys) for layout in self.layouts)
        model = NewtonManager.get_model()
        # Finalized templates retain mesh/BVH resources on the simulation device for the bank's lifetime.
        self._models = []
        for cfg in configs:
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, "Z")
            spawn_keyboard("/Keyboard", cfg, stage=stage)
            builder = env.sim.physics_manager.create_builder(up_axis="Z")
            builder.begin_world()
            builder.add_usd(stage, root_path="/Keyboard", hide_collision_shapes=True)
            replace_newton_builder_shape_colors(builder, stage)
            builder.end_world()
            self._models.append(builder.finalize(env.device))

        def binding(labels, worlds):
            rows = [
                [i for i, (label, world) in enumerate(zip(labels, worlds)) if world == e and "/Keyboard/" in label]
                for e in range(env.num_envs)
            ]
            suffixes = [labels[i].split("/Keyboard/", 1)[1] for i in rows[0]]
            for row in rows:
                if [labels[i].split("/Keyboard/", 1)[1] for i in row] != suffixes:
                    raise ValueError("Keyboard slots must have identical ordered shape/body labels in every world.")
            return np.asarray(rows, dtype=np.int32), suffixes

        body_ids, body_names = binding(model.body_label, model.body_world.numpy())
        shape_ids, shape_names = binding(model.shape_label, model.shape_world.numpy())
        self.body_ids = torch.as_tensor(body_ids, device=env.device)
        self.shape_ids = shape_ids
        self._properties = []
        self._sources = []
        source_bodies, source_shapes = [], []
        for source in self._models:
            body_map = {label.split("/Keyboard/", 1)[1]: i for i, label in enumerate(source.body_label)}
            shape_map = {label.split("/Keyboard/", 1)[1]: i for i, label in enumerate(source.shape_label)}
            source_bodies.append(np.array([body_map[name] for name in body_names]))
            source_shapes.append(np.array([shape_map[name] for name in shape_names]))
            self._sources.append([source.shape_source[i] for i in source_shapes[-1]])
            types, flags = source.shape_type.numpy(), source.shape_flags.numpy()
            if not np.array_equal(types[source_shapes[-1]], model.shape_type.numpy()[shape_ids[0]]):
                raise ValueError("Keyboard variants must preserve each registered shape's geometry type.")
            if any(flags[i] & int(ShapeFlags.COLLIDE_SHAPES) and types[i] != GeoType.BOX for i in source_shapes[-1]):
                raise ValueError("Registered keyboards currently require box collision geometry.")

        def register(names, destination, indices):
            ids = wp.array(destination, dtype=wp.int32, device=model.device)
            for name in names:
                output = getattr(model, name)
                if output is None:
                    continue
                data = np.stack([getattr(src, name).numpy()[idx] for src, idx in zip(self._models, indices)])
                values = wp.array(data, dtype=output.dtype, device=model.device)
                self._properties.append((ids, values, output))

        register(
            ("body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia"), body_ids, source_bodies
        )
        register(
            (
                "shape_transform",
                "shape_scale",
                "shape_source_ptr",
                "shape_flags",
                "shape_color",
                "shape_opacity",
                "shape_collision_radius",
                "shape_collision_aabb_lower",
                "shape_collision_aabb_upper",
                "shape_margin",
                "shape_gap",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu",
                "shape_material_mu_torsional",
                "shape_material_mu_rolling",
            ),
            shape_ids,
            source_shapes,
        )
        command = env.cfg.commands.typing
        self.keys, self.dofs = command.keys, command.key_dofs
        qd_ids = self.dofs.dense_ids().cpu().numpy()
        joint_ids = self.dofs.joint_ids.numpy().reshape(qd_ids.shape)
        joint_names = [model.joint_label[i].split("/Keyboard/", 1)[1] for i in joint_ids[0]]
        source_joints, source_dofs = [], []
        for source in self._models:
            mapping = {label.split("/Keyboard/", 1)[1]: i for i, label in enumerate(source.joint_label)}
            joints = np.array([mapping[name] for name in joint_names])
            source_joints.append(joints)
            source_dofs.append(source.joint_qd_start.numpy()[joints])
        register(("joint_X_p", "joint_X_c"), joint_ids, source_joints)
        register(
            (
                "joint_axis",
                "joint_target_ke",
                "joint_target_kd",
                "joint_damping",
                "joint_armature",
                "joint_effort_limit",
                "joint_velocity_limit",
                "joint_friction",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_limit_ke",
                "joint_limit_kd",
            ),
            qd_ids,
            source_dofs,
        )
        # Slot ownership is stable even when a partition is disabled.
        self._body_partition = torch.tensor(
            [int(name.split("/")[1].split("_")[1]) for name in body_names], device=env.device
        )
        body_columns = {body: i for i, body in enumerate(body_ids[0])}
        self._shape_body_column = torch.tensor(
            [body_columns[int(model.shape_body.numpy()[i])] for i in shape_ids[0]], device=env.device
        )
        self._shape_ids_t = torch.as_tensor(shape_ids, device=env.device)

    def apply(self, env_ids: torch.Tensor, variant_ids: torch.Tensor) -> None:
        """Install registered meshes, physical properties and participation before reset state writes.

        Both arguments are one-dimensional int64 tensors on the task device. This is
        an episode-boundary operation outside graph capture; model storage stays fixed.
        """
        if env_ids.shape != variant_ids.shape or env_ids.ndim != 1:
            raise ValueError("env_ids and variant_ids must be equally sized vectors.")
        if not len(env_ids):
            return
        if torch.any(variant_ids < 0) or torch.any(variant_ids >= len(self.layouts)):
            raise ValueError("Keyboard variant index is outside the registered bank.")
        if torch.any(env_ids < 0) or torch.any(env_ids >= self.env.num_envs) or len(env_ids.unique()) != len(env_ids):
            raise ValueError("Reset worlds must be unique valid environment indices.")
        model = NewtonManager.get_model()
        worlds = wp.from_torch(env_ids.to(torch.int32))
        variants = wp.from_torch(variant_ids.to(torch.int32))
        for ids, values, output in self._properties:
            wp.launch(
                _copy_variant_property,
                dim=(len(env_ids), ids.shape[1]),
                inputs=[worlds, variants, ids, values],
                outputs=[output],
                device=model.device,
            )
        # Host mesh descriptors are rendering resources; physics uses the matching device pointers above.
        for world, variant in zip(env_ids.tolist(), variant_ids.tolist()):
            for shape, source in zip(self.shape_ids[world], self._sources[variant]):
                model.shape_source[shape] = source
        self.variant_ids[env_ids] = variant_ids
        active = self._body_partition[None, :] < (self.counts[variant_ids, None] // 6)
        wp.to_torch(self.env.selections.body_active)[self.body_ids[env_ids]] = active
        participating = active & wp.to_torch(self.env.selections.world_active)[env_ids, None]
        shape_ids = self._shape_ids_t[env_ids]
        flags = wp.to_torch(model.shape_flags)
        flags[shape_ids] = torch.where(participating[:, self._shape_body_column], flags[shape_ids], 0)
        self.env.selections.refresh()
        state, control = NewtonManager.get_state(), NewtonManager.get_control()
        wp.to_torch(state.joint_q)[self.keys.dense_ids()[env_ids]] = 0.0
        wp.to_torch(state.joint_qd)[self.dofs.dense_ids()[env_ids]] = 0.0
        target_ids = self.keys.dense_ids() if model.use_coord_layout_targets else self.dofs.dense_ids()
        wp.to_torch(control.joint_target_q)[target_ids[env_ids]] = 0.0
        wp.to_torch(control.joint_target_qd)[self.dofs.dense_ids()[env_ids]] = 0.0
        wp.to_torch(control.joint_f)[self.dofs.dense_ids()[env_ids]] = 0.0
        self.env._property_world_mask.zero_()
        wp.to_torch(self.env._property_world_mask)[env_ids] = True
        NewtonManager.notify_model_changed(
            ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES,
            world_mask=self.env._property_world_mask,
        )
        selected = self.body_ids[env_ids]
        for mask, policy in (
            (participating, SolverMuJoCo.SleepPolicy.ALLOWED),
            (~participating, SolverMuJoCo.SleepPolicy.ALWAYS),
        ):
            ids = selected[mask].to(torch.int32).contiguous()
            if ids.numel():
                NewtonManager.set_body_sleep_policy(wp.from_torch(ids), policy)
        NewtonManager.invalidate_fk(env_ids=worlds)
