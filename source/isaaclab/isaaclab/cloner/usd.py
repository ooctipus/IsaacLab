# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ._fabric_notices import disabled_fabric_change_notifies
from .path import split


def _select_env_ids(env_ids: torch.Tensor, mask: torch.Tensor | None, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    if mask is None:
        return env_ids
    row_mask = mask if mask.dim() == 1 else mask[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


class UsdReplicateContext:
    """Apply one clone-plan mapping to a USD stage."""

    # USD destinations must exist before native physics contexts consume them.
    replicate_priority = -100
    clones_whole_env = True

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to author replicated prim specs into.
        """
        self.stage = stage

    def replicate(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Apply replication rows from the current flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mask: Optional per-source or shared mask.
            positions: Optional per-environment world positions [m]. Authored only for
                instance-root destination templates (for example, ``.../env_{}``).
            quaternions: Optional per-environment orientations in xyzw order. Authored only
                for instance-root destination templates (for example, ``.../env_{}``).
        """
        items = [
            (source, destinations[i], _select_env_ids(env_ids, mask, i), positions, quaternions)
            for i, source in enumerate(sources)
        ]
        if not items:
            return

        # Suspend Fabric's per-Sdf.CopySpec notice listener for the duration of the copy work;
        # no-op outside a live Kit application.
        with disabled_fabric_change_notifies(self.stage):
            self._apply(items)

    def _apply(self, items: list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]) -> None:
        """Author copy specs into the stage's root layer."""
        rl = self.stage.GetRootLayer()

        def dp_depth(template: str) -> int:
            """Return destination prim path depth for stable parent-first replication."""
            dp = template.format(0)
            return Sdf.Path(dp).pathElementCount

        depth_to_items: dict[int, list[tuple[str, str, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]] = {}
        for item in items:
            depth_to_items.setdefault(dp_depth(item[1]), []).append(item)

        for depth in sorted(depth_to_items.keys()):
            with Sdf.ChangeBlock():
                for src, tmpl, target_envs, positions, quaternions in depth_to_items[depth]:
                    _, clone_suffix = split(tmpl)
                    is_instance_root = clone_suffix == ""

                    for wid in target_envs.tolist():
                        wid = int(wid)
                        dp = tmpl.format(wid)
                        Sdf.CreatePrimInLayer(rl, dp)
                        # ``CreatePrimInLayer`` authors missing intermediate ancestors (e.g. the
                        # ``Groceries`` scope in ``env_{}/Groceries/Object``) as ``over`` specs. A
                        # ``def`` copied below an ``over`` ancestor never composes as defined, so
                        # Hydra skips it and its references stay unexpanded. Promote such ancestors
                        # to ``def``; for ancestors already defined elsewhere this is a no-op.
                        ancestor = Sdf.Path(dp).GetParentPath()
                        while ancestor != Sdf.Path.absoluteRootPath:
                            ancestor_spec = rl.GetPrimAtPath(ancestor)
                            if ancestor_spec is None or ancestor_spec.specifier != Sdf.SpecifierOver:
                                break
                            ancestor_spec.specifier = Sdf.SpecifierDef
                            ancestor = ancestor.GetParentPath()
                        if src != dp:
                            Sdf.CopySpec(rl, Sdf.Path(src), rl, Sdf.Path(dp))

                        # Author positions/quaternions for instance roots only.
                        if is_instance_root and (positions is not None or quaternions is not None):
                            ps = rl.GetPrimAtPath(dp)
                            op_names = []
                            if positions is not None:
                                p = positions[wid]
                                t_attr = ps.GetAttributeAtPath(dp + ".xformOp:translate")
                                if t_attr is None:
                                    t_attr = Sdf.AttributeSpec(ps, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                                t_attr.default = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                                op_names.append("xformOp:translate")
                            if quaternions is not None:
                                q = quaternions[wid]
                                o_attr = ps.GetAttributeAtPath(dp + ".xformOp:orient")
                                if o_attr is None:
                                    o_attr = Sdf.AttributeSpec(ps, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
                                o_attr.default = Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                                op_names.append("xformOp:orient")
                            if op_names:
                                op_order = ps.GetAttributeAtPath(dp + ".xformOpOrder") or Sdf.AttributeSpec(
                                    ps, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                                )
                                op_order.default = Vt.TokenArray(op_names)
