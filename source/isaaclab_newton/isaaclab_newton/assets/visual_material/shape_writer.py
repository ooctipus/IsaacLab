# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visual-material shape addressing and writes.

Newton consumers render from model state, not from the USD stage: the tiled-camera sensor and
the Newton viewer both read ``model.shape_color`` (the viewer re-syncs it every frame). The
Newton :class:`~isaaclab_newton.assets.visual_material.VisualMaterial` therefore writes colors
here directly, and texture listeners resolve their shape rows through the same plan. The scatter
emulates the material indirection the Newton model lacks for colors (compare
``shape_texture_ids`` for textures); if the model ever gains a per-shape material table, this
module disappears.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from isaaclab import cloner
from isaaclab.sim import SimulationContext


@wp.func
def _linear_channel_to_srgb(x: float) -> float:
    """Encode one linear channel with the sRGB transfer function (``model.shape_color`` stores sRGB)."""
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055


@wp.kernel
def scatter_srgb_shape_colors(
    colors: wp.array(dtype=wp.vec3),
    material_index: wp.array(dtype=wp.int32),
    shape_rows: wp.array(dtype=wp.int32),
    shape_color: wp.array(dtype=wp.vec3),
):
    """Gather one linear bucket color per slot, sRGB-encode it, and scatter into ``shape_color``."""
    tid = wp.tid()
    color = colors[material_index[tid]]
    shape_color[shape_rows[tid]] = wp.vec3(
        _linear_channel_to_srgb(color[0]),
        _linear_channel_to_srgb(color[1]),
        _linear_channel_to_srgb(color[2]),
    )


@dataclass
class _VisualMaterialWritePlan:
    """Precomputed single-kernel scatter plan for one notified visual-material tuple."""

    dim: int
    """Number of shape rows the scatter writes."""
    shape_rows: wp.array(dtype=wp.int32)
    """Target rows in ``model.shape_color``, flattened over the notified materials."""
    material_index: wp.array(dtype=wp.int32)
    """Index of the notified material owning each row, aligned with :attr:`shape_rows`."""
    staging_colors: wp.array(dtype=wp.vec3)
    """Persistent device buffer the incoming linear RGB colors are copied into."""
    staging_colors_torch: torch.Tensor
    """Zero-copy Torch view of :attr:`staging_colors` used for the per-fire copy."""


class VisualMaterialShapeWriter:
    """Scatters bucket-material color writes into one Newton model's ``shape_color``.

    All indexing is memoized: the material → shape-row map is resolved once from the authored
    bindings (spawn-time state this write path never mutates), and the flattened scatter plan —
    including a persistent staging buffer — is cached per notified material tuple (event terms
    fire with a stable list). Each fire is therefore one copy of the tiny (K, 3) colors into the
    staging buffer and one fused gather/sRGB-encode/scatter kernel launch, with no intermediate
    allocations.

    Shapes are matched through ``model.shape_label`` prim paths against the shape → bound-material
    map captured when Newton imported the source stage (see
    :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_builder_shape_colors`), carried
    into each cloned environment through the published clone plan — no stage queries at runtime.
    """

    def __init__(self, model, source_shape_materials: dict[str, str]):
        self._model = model
        self._source_shape_materials = source_shape_materials
        self._rows: dict[str, torch.Tensor] | None = None
        self._plans: dict[tuple[str, ...], _VisualMaterialWritePlan] = {}

    def write_colors(self, material_paths: tuple[str, ...], colors: torch.Tensor) -> None:
        """Scatter one linear RGB color per notified material into ``model.shape_color``.

        Color is the only channel the Newton model stores a per-shape value for today; scalar
        response channels (roughness, metallic, ...) become writable when the model gains
        per-material tables (``shape_material_id`` + ``material_<channel>``).

        Args:
            material_paths: Material prim paths, one per row of ``colors``.
            colors: One linear RGB color per material, shape (len(material_paths), 3), float.
        """
        plan = self.plan(material_paths)
        if plan.dim == 0:
            return
        plan.staging_colors_torch.copy_(colors.to(dtype=torch.float32))
        wp.launch(
            scatter_srgb_shape_colors,
            dim=plan.dim,
            inputs=[plan.staging_colors, plan.material_index, plan.shape_rows, self._model.shape_color],
            device=self._model.device,
        )

    def plan(self, material_paths: tuple[str, ...]) -> _VisualMaterialWritePlan:
        """Return the flattened scatter plan for one notified material tuple (memoized)."""
        plan = self._plans.get(material_paths)
        if plan is None:
            rows_by_material = self.rows_by_material()
            device = wp.device_to_torch(self._model.device)
            empty = torch.empty(0, dtype=torch.long, device=device)
            row_groups = [rows_by_material.get(path, empty) for path in material_paths]
            counts = torch.tensor([rows.numel() for rows in row_groups], dtype=torch.long, device=device)
            shape_rows = torch.cat(row_groups).to(dtype=torch.int32)
            material_index = torch.repeat_interleave(torch.arange(len(material_paths), device=device), counts).to(
                dtype=torch.int32
            )
            staging_colors = wp.zeros(len(material_paths), dtype=wp.vec3, device=self._model.device)
            plan = _VisualMaterialWritePlan(
                dim=int(shape_rows.numel()),
                shape_rows=wp.from_torch(shape_rows.contiguous(), dtype=wp.int32),
                material_index=wp.from_torch(material_index.contiguous(), dtype=wp.int32),
                staging_colors=staging_colors,
                staging_colors_torch=wp.to_torch(staging_colors),
            )
            self._plans[material_paths] = plan
        return plan

    def rows_by_material(self) -> dict[str, torch.Tensor]:
        """Group ``model.shape_label`` rows by their bound material, from the import capture (memoized).

        The shape → bound-material map is captured in source space when Newton imports the stage
        (USD binding precedence is resolved there, on real prims), so this never queries the
        stage. Each model label is carried back to its source: labels under a cloned subtree are
        stripped to the source builder's label through the published clone plan, and the looked-up
        material follows the shape into its environment via
        :func:`~isaaclab.cloner.query.path_to_clone` — a material replicated by its own plan row
        resolves to that environment's clone, a bucket material passes through unchanged.
        """
        if self._rows is not None:
            return self._rows

        clone_plan = SimulationContext.instance().get_clone_plan()

        # clone-root → (source root, env id), for inverting the per-env label rename
        clone_roots: dict[str, tuple[str, int]] = {}
        if clone_plan is not None:
            for plan_row, (source, template) in enumerate(
                zip(clone_plan.sources, clone_plan.destinations, strict=True)
            ):
                if "{}" not in template:
                    continue
                for column in torch.nonzero(clone_plan.clone_mask[plan_row]).flatten().tolist():
                    # Mask columns are positions within the plan, not env ids; replication formats
                    # destinations with the env id each column stands for.
                    env_id = column if clone_plan.env_ids is None else int(clone_plan.env_ids[column])
                    clone_roots[template.format(env_id)] = (source, env_id)

        def bound_material(label: str) -> str | None:
            root = label
            while root and root not in clone_roots:
                root = root.rpartition("/")[0]
            if not root:
                # not under any cloned subtree: a global shape, captured verbatim
                return self._source_shape_materials.get(label)
            source, env_id = clone_roots[root]
            material = self._source_shape_materials.get(cloner.path.rebase(label, root, source))
            if material is None:
                return None
            return cloner.query.path_to_clone(clone_plan, material, env_id) or material

        rows_by_material: dict[str, list[int]] = {}
        for row, label in enumerate(self._model.shape_label):
            material = bound_material(label)
            if material is not None:
                rows_by_material.setdefault(material, []).append(row)

        device = wp.device_to_torch(self._model.device)
        self._rows = {
            material_path: torch.tensor(rows, dtype=torch.long, device=device)
            for material_path, rows in rows_by_material.items()
        }
        return self._rows
