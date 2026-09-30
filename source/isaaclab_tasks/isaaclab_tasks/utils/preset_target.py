# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed enum of typed preset categories with per-target metadata.

Each :class:`PresetTarget` member carries everything the preset CLI layer
needs to know about that category in one place:

* ``label`` -- the Hydra-style selector key (e.g. ``"physics"`` for
  ``physics=NAME``) and ``self.value``.
* ``base_classes`` -- the cfg base classes whose subclass instances belong to
  this bucket. Help-time bucketing in :mod:`isaaclab_tasks.utils.preset_cli`
  routes variants by ``isinstance`` against these. Empty for
  :attr:`PresetTarget.DOMAIN`, which is the catch-all whose membership is
  "no typed target matched".
Adding a new typed target = appending one enum member with its label, base
classes. The CLI layer needs no other wiring.
"""

from __future__ import annotations

import enum

from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


class PresetTarget(enum.Enum):
    """Typed preset categories.

    **Bucketing contract.** Help-time bucketing in
    :mod:`isaaclab_tasks.utils.preset_cli` routes each preset variant to a
    typed target through :meth:`matches`. A variant whose cfg value does not
    match any typed target falls into :attr:`DOMAIN` and shows up under the
    ``presets:`` catch-all in ``--help``.

    To opt into a typed help-text listing, a cfg class must subclass the
    corresponding physics, renderer, or visualizer base class. A variant whose
    class does *not* subclass a typed base still **resolves
    correctly at runtime** -- hydra applies the selected name across every
    matching ``PresetCfg`` field regardless of class; the typed bucketing only
    governs which header it appears under in ``--help``.

    Adding a new target = appending one enum member.
    """

    # Members. Tuple values are (label, base_classes); the
    # enum metaclass collects the whole namespace before constructing members,
    # so ``__new__`` below unpacks each tuple regardless of declaration order.
    PHYSICS = ("physics", (PhysicsCfg,))
    """Physics backends -- ``physics=NAME`` selector."""

    RENDERER = ("renderer", (RendererCfg,))
    """Camera-sensor renderers -- ``renderer=NAME`` selector."""

    VISUALIZER = ("visualizer", (VisualizerCfg,))
    """Interactive visualizers -- ``visualizer=NAME`` selector."""

    DOMAIN = ("presets",)
    """Free-form env-specific presets -- ``presets=NAME[,...]`` selector (catch-all).

    No ``base_classes`` -- any variant that does not match a typed target ends
    up here. The ``presets=`` token also acts as a
    broadcast: hydra's resolver applies a DOMAIN-bucketed name to every
    matching ``PresetCfg`` regardless of target. ``self.value`` matches the
    CLI selector key (``"presets"``) so the CLI layer can dispatch by
    enum value without a hardcoded constant.
    """

    def __new__(
        cls,
        label: str,
        base_classes: tuple[type, ...] = (),
    ):
        """Construct a member from its ``(label, base_classes)`` tuple.

        Args:
            label: Hydra-style selector key (e.g. ``"physics"`` is recognized
                as the ``physics=NAME`` token and becomes ``self.value``).
            base_classes: Cfg base classes whose instances route to this
                target via :func:`isinstance`. Defaults to ``()`` (no typed
                routing).
        Returns:
            A new enum member with ``_value_`` set to *label* and its
            ``base_classes`` attribute.
        """
        obj = object.__new__(cls)
        obj._value_ = label
        obj.base_classes = tuple(base_classes)
        return obj
