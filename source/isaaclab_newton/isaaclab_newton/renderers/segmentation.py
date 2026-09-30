# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segmentation-mapping helpers for the Newton Warp renderer.

Newton's ray tracer emits a single per-pixel *shape index* (``shape_index_image``), i.e. the global
index of the model shape hit by each ray. Isaac Lab's camera contract instead exposes two
segmentation outputs — ``semantic_segmentation`` and ``instance_segmentation`` — each with its own
id space and an accompanying ``idToLabels`` / ``idToSemantics`` mapping
(see :class:`~isaaclab.sensors.camera.CameraData`).

This module reconstructs those outputs on the host from the Newton model's per-shape prim paths
(``model.shape_label``) and cfg-declared semantic tags retained by the scene's clone plan, then remaps
the shape-index image into each requested output with a Warp kernel. Colors match the Isaac RTX /
OVRTX palette so colorized outputs are visually consistent across renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import warp as wp

# Colorization (host ``random_color_from_id`` / ``pack_rgba``) and the reserved BACKGROUND / UNLABELLED
# ids are shared with the RTX and OVRTX renderers to keep colorized segmentation visually consistent.
from isaaclab import cloner
from isaaclab.renderers.segmentation_colors import BACKGROUND_ID, UNLABELLED_ID, pack_rgba, random_color_from_id
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    import newton

    from isaaclab.cloner import ClonePlan

_UNLABELLED_COLOR: int = 0xFF000000
"""Packed RGBA color for UNLABELLED pixels: ``(0, 0, 0, 255)`` opaque black."""

# Newton ray-tracer sentinel values. These mirror the constants in
# ``newton._src.sensors.warp_raytrace.raytrace`` (``NO_HIT_SHAPE_ID``, ``MAX_SHAPE_ID``), which are
# not re-exported by Newton's public API. Keep in sync if Newton changes them.
_NEWTON_NO_HIT_SHAPE_ID: int = 0xFFFFFFFF
"""Shape-index value written by the Newton ray tracer when a ray hits nothing (ray miss)."""
_NEWTON_MAX_SHAPE_ID: int = 0xFFFFFFF0
"""Guard threshold: any shape_index >= this value is a synthetic/sentinel id, not a real shape."""

_FIRST_ID: int = 2
"""First id assigned to a real semantic/instance group (0 and 1 are reserved)."""

SegId: TypeAlias = int
"""Integer segmentation id assigned to a group and written into the output image per pixel.

Distinct from Newton's ``shape_index`` (the raw per-pixel geometry index the ray tracer emits):
``shape_index`` identifies a single shape in the Newton model, while ``SegId`` identifies a semantic
or instance group that may span many shapes. The ``shape_to_id`` array bridges them:
``shape_to_id[shape_index] -> SegId``.
"""

_SegKind = Literal["semantic_segmentation", "instance_segmentation"]

SemanticType: TypeAlias = str
"""A semantic type string as authored in :class:`UsdSemantics.LabelsAPI`, e.g. ``"class"``."""

SemanticLabels: TypeAlias = list[str]
"""The list of label strings for one semantic type, e.g. ``["cartpole"]``."""

SemanticLabelString: TypeAlias = str
"""Comma-joined form of :data:`SemanticLabels` used in info dicts, e.g. ``"cartpole"``."""

SemanticRoot: TypeAlias = str
"""Exact clone-plan asset root used as the instance grouping key."""


# ------------------------------------------------------------------------------------------------
# Semantic-filter predicate — RTX parity.
#
# The Isaac RTX renderer evaluates :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter`
# inside Replicator's C++ ``SyntheticData`` instance-mapping filter (see
# :func:`isaaclab_physx.renderers.isaac_rtx_renderer._camera_semantic_filter_predicate`). The kit-less
# Newton path has no Replicator, so the three functions below re-implement the *same* predicate grammar
# on the host so :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.semantic_filter` selects
# exactly the labels the RTX renderer would. The grammar is documented on
# :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter`.
# ------------------------------------------------------------------------------------------------


def _eval_label_expr(expr: str, labels: set[str]) -> bool:
    """Evaluate one clause's label expression against a prim's label set (RTX-parity grammar).

    Grammar (matching the Isaac RTX / Replicator ``semantic_filter`` predicate): an OR of ``|`` / ``,``
    separated terms; each term is an AND of ``&`` separated factors; each factor is ``label``,
    ``!label`` (negation) or ``*`` (any).
    """
    expr = expr.strip()
    if not expr:
        return False
    for term in expr.replace(",", "|").split("|"):
        term = term.strip()
        if not term:
            continue
        if all(_eval_label_factor(factor.strip(), labels) for factor in term.split("&")):
            return True
    return False


def _eval_label_factor(factor: str, labels: set[str]) -> bool:
    """Evaluate a single filter factor: ``*`` (any), ``!label`` (absent) or ``label`` (present)."""
    if factor == "*":
        return True
    if factor.startswith("!"):
        return factor[1:].strip() not in labels
    return factor in labels


def _parse_semantic_filter(semantic_filter: str | list[str]) -> list[tuple[str, str]]:
    """Normalize a semantic filter into a list of ``(type, label_expr)`` clauses (RTX parity).

    Reproduces the normalization in
    :func:`isaaclab_physx.renderers.isaac_rtx_renderer._camera_semantic_filter_predicate` so the same
    :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter` value behaves identically on
    the Newton Warp backend: a list is treated as bare semantic types (``["class"] -> "class:*"``); a
    string is a ``;``-separated disjunction of ``type:label_expr`` clauses (e.g.
    ``"class:* ; *:shelf"``).  Within each ``;``-separated segment, commas that precede an
    ``identifier:`` token introduce additional ``(type, label_expr)`` pairs (e.g.
    ``"class:cartpole, material:wood"`` yields two independent clauses).  Commas within a
    ``label_expr`` that are not followed by a type token remain as label-level OR operators (handled
    by :func:`_eval_label_expr`).

    The split heuristic relies on the fact that label factors never contain ``:``, so any
    comma-separated token that contains ``:`` must be the start of a new ``type:label_expr`` pair
    rather than a continuation of the current label expression.
    """
    if isinstance(semantic_filter, list):
        text = "; ".join(f"{t}:*" for t in semantic_filter)
    else:
        text = semantic_filter
    clauses: list[tuple[str, str]] = []
    for segment in text.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        # Accumulate comma-separated tokens into type:label_expr clauses.  A token that contains
        # ':' opens a new clause; a token without ':' is a label-level OR term and is re-joined
        # (with its comma) onto the current clause's label expression.
        parts = segment.split(",")
        current = parts[0]
        for part in parts[1:]:
            if ":" in part:
                sem_type, _, label_expr = current.partition(":")
                clauses.append((sem_type.strip(), label_expr.strip() or "*"))
                current = part
            else:
                current = current + "," + part
        sem_type, _, label_expr = current.partition(":")
        clauses.append((sem_type.strip(), label_expr.strip() or "*"))
    return clauses


@wp.kernel(enable_backward=False)
def _remap_shape_index_to_id_kernel(
    shape_index: wp.array(dtype=wp.uint32, ndim=4),
    shape_to_id: wp.array(dtype=wp.uint32),
    shape_count: wp.int32,
    out_id: wp.array(dtype=wp.int32, ndim=4),
):
    """Write the segmentation id of each pixel's shape."""
    w, c, y, x = wp.tid()
    idx = shape_index[w, c, y, x]
    if idx == wp.uint32(wp.static(_NEWTON_NO_HIT_SHAPE_ID)):
        # Ray missed all geometry — no surface was hit.
        out_id[w, c, y, x] = wp.int32(wp.static(BACKGROUND_ID))
    elif idx >= wp.uint32(wp.static(_NEWTON_MAX_SHAPE_ID)):
        # Synthetic hit (particles, deformable mesh, etc.) — real geometry but no shape-array entry.
        out_id[w, c, y, x] = wp.int32(wp.static(UNLABELLED_ID))
    elif idx < wp.uint32(shape_count):
        # Normal geometry hit — look up the pre-built segmentation id for this shape.
        out_id[w, c, y, x] = wp.int32(shape_to_id[wp.int32(idx)])
    else:
        # Index is in the valid uint32 range but beyond shape_count — should not happen in practice.
        out_id[w, c, y, x] = wp.int32(wp.static(UNLABELLED_ID))


@wp.kernel(enable_backward=False)
def _remap_shape_index_to_color_kernel(
    shape_index: wp.array(dtype=wp.uint32, ndim=4),
    shape_to_color: wp.array(dtype=wp.uint32),
    shape_count: wp.int32,
    out_color: wp.array(dtype=wp.uint32, ndim=4),
):
    """Write the packed RGBA color of each pixel's shape."""
    w, c, y, x = wp.tid()
    idx = shape_index[w, c, y, x]
    if idx == wp.uint32(wp.static(_NEWTON_NO_HIT_SHAPE_ID)):
        # Ray missed all geometry — no surface was hit.
        out_color[w, c, y, x] = wp.uint32(0)
    elif idx >= wp.uint32(wp.static(_NEWTON_MAX_SHAPE_ID)):
        # Synthetic hit (particles, deformable mesh, etc.) — real geometry but no shape-array entry.
        out_color[w, c, y, x] = wp.uint32(wp.static(_UNLABELLED_COLOR))
    elif idx < wp.uint32(shape_count):
        # Normal geometry hit — look up the pre-built color for this shape.
        out_color[w, c, y, x] = shape_to_color[wp.int32(idx)]
    else:
        # Index is in the valid uint32 range but beyond shape_count — should not happen in practice.
        out_color[w, c, y, x] = wp.uint32(wp.static(_UNLABELLED_COLOR))


@dataclass
class NewtonSegmentationMapping:
    """Device lookup tables and host metadata for one segmentation output.

    Built once per (kind, colorize) by :meth:`NewtonSegmentationMapper.mapping` and reused across frames
    since the scene geometry and semantics are static after construction.
    """

    kind: _SegKind
    colorize: bool
    shape_count: int
    shape_to_id: wp.array  # uint32, shape [shape_count]
    shape_to_color: wp.array | None  # uint32, shape [shape_count]; None when not colorized
    info: dict[str, dict]

    def convert_shape_index_to_output(self, shape_index: wp.array, out_view: wp.array) -> None:
        """Generate the segmentation output into ``out_view`` (colorized RGBA or raw uint32 ids).

        Args:
            shape_index: Newton ``shape_index_image`` scratch, ``(world_count, 1, H, W)`` uint32.
            out_view: Destination view aliasing the camera output buffer as ``(world_count, 1, H, W)``,
                dtype uint32 (both colorized and non-colorized).
        """
        if self.shape_count == 0:
            out_view.zero_()
            return
        if self.colorize:
            wp.launch(
                _remap_shape_index_to_color_kernel,
                dim=shape_index.shape,
                inputs=[shape_index, self.shape_to_color, self.shape_count],
                outputs=[out_view],
                device=out_view.device,
            )
        else:
            wp.launch(
                _remap_shape_index_to_id_kernel,
                dim=shape_index.shape,
                inputs=[shape_index, self.shape_to_id, self.shape_count],
                outputs=[out_view],
                device=out_view.device,
            )


class NewtonSegmentationMapper:
    """Build per-shape segmentation lookup tables from a Newton model and clone plan."""

    def __init__(self, model: newton.Model, plan: ClonePlan, cfg) -> None:
        """Initialize the mapper from the Newton model, clone plan, and renderer config.

        Construction is cheap — it only captures references and snapshots ``model.shape_label``.
        Call :meth:`build_mapping` to do the actual per-shape plan resolution and id assignment.

        Args:
            model: The compiled Newton model; ``model.shape_label`` maps shape indices to USD prim paths.
            plan: Replication layout carrying cfg-declared semantic tags. A plan without tags treats
                every shape as unlabelled; labels embedded only in USD are not discovered.
            cfg: Renderer config exposing ``semantic_filter`` and ``semantic_segmentation_mapping``.

        Raises:
            ValueError: If no clone plan owns the rendered scene.
        """
        if plan is None:
            raise ValueError("Newton segmentation requires an active clone plan.")
        self._cfg = cfg
        self._shape_labels: list[str] = list(model.shape_label)
        for shape_index, prim_path in enumerate(self._shape_labels):
            if cloner.query.path_to_source(plan, prim_path) is None:
                raise ValueError(f"Newton model shape {shape_index} at {prim_path!r} is not covered by the clone plan.")
        self._shape_count = len(self._shape_labels)
        self._device = str(model.device)
        self._filter_clauses = _parse_semantic_filter(cfg.semantic_filter)
        self._semantic_rows: list[tuple[str, str, frozenset[int], dict[SemanticType, SemanticLabels]]] = []
        for row, tags in enumerate(plan.semantic_tags):
            labels: dict[SemanticType, SemanticLabels] = {}
            for semantic_type, value in tags:
                if value not in labels.setdefault(semantic_type, []):
                    labels[semantic_type].append(value)
            if not labels:
                continue
            columns = plan.clone_mask[row].nonzero(as_tuple=False).flatten().tolist()
            env_ids = columns if plan.env_ids is None else [int(plan.env_ids[column]) for column in columns]
            self._semantic_rows.append((plan.sources[row], plan.destinations[row], frozenset(env_ids), labels))
        self._matched_cache: dict[str, tuple[dict[SemanticType, SemanticLabels], SemanticRoot] | None] = {}
        self._mappings: dict[tuple[str, bool], NewtonSegmentationMapping] = {}

    def build_mapping(self, kind: _SegKind, colorize: bool) -> None:
        """Build and cache the :class:`NewtonSegmentationMapping` for ``kind`` at the requested colorization."""
        key = (kind, colorize)
        if key not in self._mappings:
            with Timer(
                f"[INFO]: Time taken for NewtonSegmentationMapper.build_mapping ({kind}, colorize={colorize})",
                f"newton_segmentation_build_mapping_{kind}_{colorize}",
                enable=True,
            ):
                self._mappings[key] = self._build_mapping(kind, colorize)

    def get_mapping(self, kind: _SegKind, colorize: bool) -> NewtonSegmentationMapping:
        """Return the pre-built :class:`NewtonSegmentationMapping` for ``kind``; must call
        :meth:`build_mapping` first."""
        return self._mappings[(kind, colorize)]

    # -- host resolution ---------------------------------------------------------------------------

    def _resolve_semantic_match(self, prim_path: str) -> tuple[dict[SemanticType, SemanticLabels], SemanticRoot] | None:
        """Return filtered labels and the nearest planned instance root owning ``prim_path``."""
        if prim_path in self._matched_cache:
            return self._matched_cache[prim_path]

        result: tuple[dict[SemanticType, SemanticLabels], SemanticRoot] | None = None
        candidates: list[tuple[int, str, dict[SemanticType, SemanticLabels]]] = []
        for source, destination, env_ids, labels in self._semantic_rows:
            if ("{}" not in destination or env_ids) and (
                suffix := cloner.path.relative_to(prim_path, source)
            ) is not None:
                candidates.append((len(suffix), source, labels))
            matched = cloner.path.match(prim_path, destination)
            if matched is None:
                continue
            if "{}" in destination:
                if not matched.instance.isdigit() or int(matched.instance) not in env_ids:
                    continue
                root = destination.format(matched.instance)
            else:
                root = destination
            candidates.append((len(matched.suffix), root, labels))
        for _suffix_length, root, labels in sorted(candidates, key=lambda candidate: candidate[0]):
            if kept := self._apply_filter(labels):
                result = kept, root
                break
        self._matched_cache[prim_path] = result
        return result

    def _apply_filter(self, labels: dict[SemanticType, SemanticLabels]) -> dict[SemanticType, SemanticLabels]:
        """Restrict ``labels`` (``{type: [labels]}``) to the types/labels passing the semantic filter.

        Applies the RTX-parity predicate parsed by :func:`_parse_semantic_filter`: a semantic type is
        kept when some filter clause matches its type (or ``*``) and its label set satisfies the
        clause's expression, mirroring the Isaac RTX / Replicator ``semantic_filter`` behavior. Returns
        an empty dict when the prim matches no clause (i.e. it is UNLABELLED for this filter).
        """
        if not self._filter_clauses:
            # Empty filter means no filtering — all labels pass through, matching RTX behavior.
            return dict(labels)
        kept: dict[SemanticType, SemanticLabels] = {}
        for sem_type, sem_labels in labels.items():
            label_set = set(sem_labels)
            for clause_type, clause_expr in self._filter_clauses:
                if clause_type not in ("*", sem_type):
                    continue
                if _eval_label_expr(clause_expr, label_set):
                    kept[sem_type] = list(sem_labels)
                    break
        return kept

    @staticmethod
    def _semantics_payload(labels: dict[SemanticType, SemanticLabels]) -> dict[SemanticType, SemanticLabelString]:
        """Collapse planned labels into the comma-joined payload used in ``idToLabels`` / ``idToSemantics``.

        Converts ``{"class": ["cartpole"]}`` → ``{"class": "cartpole"}``.
        """
        return {sem_type: ",".join(sem_labels) for sem_type, sem_labels in labels.items()}

    def _build_mapping(self, kind: _SegKind, colorize: bool) -> NewtonSegmentationMapping:
        """Build the :class:`NewtonSegmentationMapping` for ``kind``.

        Iterates over every shape in the Newton model by index, resolves its planned semantic tags,
        and assigns a :data:`SegId` by grouping shapes that belong to
        the same logical unit:

        - ``semantic_segmentation``: shapes with identical label payloads share one id (e.g. all
          ``class:cartpole`` shapes across all environments map to the same class id).
        - ``instance_segmentation``: shapes under the same tagged plan root share
          one id (e.g. pole and cart of one robot instance share that instance's id, while a second
          robot gets a distinct id).

        Shapes with no matching tagged plan row are assigned :data:`UNLABELLED_ID`; ray-miss
        pixels receive :data:`BACKGROUND_ID` at kernel dispatch time via
        :meth:`NewtonSegmentationMapping.convert_shape_index_to_output`.
        """
        shape_to_id = np.zeros(self._shape_count, dtype=np.uint32)

        # seg_id -> label/semantics metadata written into the info dict.
        id_labels: dict[SegId, object] = {}
        id_semantics: dict[SegId, dict[SemanticType, SemanticLabelString]] = {}

        # Maps a group key to its assigned seg_id so that shapes belonging to the same group
        # (same semantic class for semantic_segmentation, same tagged plan root for
        # instance_segmentation) reuse the same id rather than receiving a new one each time.
        group_key_to_id: dict[object, SegId] = {}
        next_id = _FIRST_ID

        for shape_index, prim_path in enumerate(self._shape_labels):
            match = self._resolve_semantic_match(prim_path)
            if match is None:
                shape_to_id[shape_index] = UNLABELLED_ID
                continue
            matched, semantic_root = match
            if kind == "instance_segmentation":
                # All shapes under the same concrete planned root form one instance group.
                group_key = semantic_root
                label_value = semantic_root
                semantics_value = self._semantics_payload(matched)
            else:  # semantic_segmentation
                payload = self._semantics_payload(matched)
                # All shapes with the same label payload (e.g. class:cartpole) form one semantic group.
                group_key = tuple(sorted(payload.items()))
                label_value = payload
                semantics_value = None

            seg_id = group_key_to_id.get(group_key)
            if seg_id is None:
                seg_id = next_id
                next_id += 1
                group_key_to_id[group_key] = seg_id
                id_labels[seg_id] = label_value
                if semantics_value is not None:
                    id_semantics[seg_id] = semantics_value
            shape_to_id[shape_index] = seg_id

        info = self._build_info(kind, colorize, id_labels, id_semantics)
        shape_to_color = self._build_color_palette(kind, colorize, shape_to_id, id_labels)

        return NewtonSegmentationMapping(
            kind=kind,
            colorize=colorize,
            shape_count=self._shape_count,
            shape_to_id=wp.array(shape_to_id, dtype=wp.uint32, device=self._device),
            shape_to_color=(
                wp.array(shape_to_color, dtype=wp.uint32, device=self._device) if shape_to_color is not None else None
            ),
            info=info,
        )

    # -- info + color assembly ---------------------------------------------------------------------

    def _id_to_color(self, seg_id: int, label_value: object) -> tuple[int, int, int, int]:
        """Resolve the RGBA color for a :data:`SegId`.

        Checks ``semantic_segmentation_mapping`` in the renderer config for a user-specified color
        override keyed by ``"type:label"`` (e.g. ``"class:cartpole"``). Falls back to the palette
        color derived from the id via :func:`random_color_from_id` when no override matches.
        """
        mapping = self._cfg.semantic_segmentation_mapping
        if mapping and isinstance(label_value, dict):
            for sem_type, sem_labels in label_value.items():
                for lbl in sem_labels.split(","):
                    override = mapping.get(f"{sem_type}:{lbl}")
                    if override is not None:
                        return tuple(int(component) for component in override)  # type: ignore[return-value]
        return random_color_from_id(seg_id)

    def _reserved_semantics_label(self, name: str) -> dict[SemanticType, SemanticLabelString]:
        """Return the fixed label payload for a reserved id (``BACKGROUND`` or ``UNLABELLED``)."""
        return {"class": name}

    def _build_info(
        self,
        kind: _SegKind,
        colorize: bool,
        id_labels: dict[SegId, object],
        id_semantics: dict[SegId, dict[SemanticType, SemanticLabelString]],
    ) -> dict[str, dict]:
        """Assemble the Replicator-compatible info dict containing ``idToLabels`` and, for instance
        segmentation, ``idToSemantics``.

        When colorized, info dict keys are ``(r, g, b, a)`` color tuples matching the packed RGBA
        pixel values; otherwise they are raw :data:`SegId` integers. Reserved ids
        (:data:`BACKGROUND_ID`, :data:`UNLABELLED_ID`) are always included.
        """
        # Reserved entries present for every segmentation output.
        reserved_labels: dict[SegId, object]
        reserved_semantics: dict[SegId, dict[SemanticType, SemanticLabelString]]
        if kind == "semantic_segmentation":
            reserved_labels = {
                BACKGROUND_ID: self._reserved_semantics_label("BACKGROUND"),
                UNLABELLED_ID: self._reserved_semantics_label("UNLABELLED"),
            }
            reserved_semantics = {}
        else:
            reserved_labels = {BACKGROUND_ID: "BACKGROUND", UNLABELLED_ID: "UNLABELLED"}
            reserved_semantics = {
                BACKGROUND_ID: self._reserved_semantics_label("BACKGROUND"),
                UNLABELLED_ID: self._reserved_semantics_label("UNLABELLED"),
            }

        all_labels = {**reserved_labels, **id_labels}

        def key_for(seg_id: SegId) -> SegId | tuple:
            if colorize:
                return self._id_to_color(seg_id, id_labels.get(seg_id, all_labels[seg_id]))
            return seg_id

        id_to_labels = {key_for(seg_id): value for seg_id, value in all_labels.items()}
        info: dict[str, dict] = {"idToLabels": id_to_labels}

        if kind == "instance_segmentation":
            all_semantics = {**reserved_semantics, **id_semantics}
            info["idToSemantics"] = {key_for(seg_id): value for seg_id, value in all_semantics.items()}
        return info

    def _build_color_palette(
        self,
        kind: _SegKind,
        colorize: bool,
        shape_to_id: np.ndarray,
        id_labels: dict[SegId, object],
    ) -> np.ndarray | None:
        """Build a per-shape packed-RGBA color palette for the colorize kernel.

        Returns a ``uint32`` array of length ``shape_count`` where each entry is the RGBA color for
        that shape's :data:`SegId`, honoring ``semantic_segmentation_mapping`` overrides. Returns
        ``None`` when colorization is not requested or the model has no shapes.
        """
        if not colorize or self._shape_count == 0:
            return None
        color_cache: dict[SegId, int] = {}
        shape_to_color = np.zeros(self._shape_count, dtype=np.uint32)
        for shape_index in range(self._shape_count):
            seg_id: SegId = int(shape_to_id[shape_index])
            packed = color_cache.get(seg_id)
            if packed is None:
                packed = pack_rgba(self._id_to_color(seg_id, id_labels.get(seg_id)))
                color_cache[seg_id] = packed
            shape_to_color[shape_index] = packed
        return shape_to_color
