# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for environment cloning partitioning.

Built-in :class:`CloneGroup` descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==========================  =============================================
Descriptor                  Selection logic
==========================  =============================================
:class:`InclusionSet`       Explicit list of asset names
:class:`ExclusionSet`       Everything *except* listed asset names
:class:`PrefixGroup`        Assets whose name starts with a prefix
:class:`SuffixGroup`        Assets whose name ends with a suffix
:class:`PatternGroup`       Regex full-match on asset names
:class:`PredicateGroup`     Arbitrary ``Callable[[str], bool]``
:class:`UnionGroup`         Logical OR of child descriptors
:class:`IntersectionGroup`  Logical AND of child descriptors
==========================  =============================================
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.cloner.cloner_strategies import random as random_strategy
from isaaclab.utils.configclass import configclass

# ── base class ────────────────────────────────────────────────────────────────


@configclass
class CloneGroup:
    """Base class for clone group descriptors.

    Subclasses implement :meth:`resolve_assets` to decide which scene
    assets belong to the group.  The :attr:`weight` controls relative
    environment count when the total is divided across groups.

    To create a custom grouping strategy, subclass ``CloneGroup`` and
    override :meth:`resolve_assets`::

        @configclass
        class MyCustomGroup(CloneGroup):
            prefix: str = MISSING

            def resolve_assets(self, all_asset_names):
                return [a for a in all_asset_names if a.startswith(self.prefix)]
    """

    weight: int = 1
    """Relative weight for env partitioning."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        """Return the asset names that belong to this group.

        Args:
            all_asset_names: All registered asset names in the scene.

        Returns:
            Subset of ``all_asset_names`` claimed by this group.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement resolve_assets(). See CloneGroup docstring for an example."
        )


# ── built-in descriptors ─────────────────────────────────────────────────────


@configclass
class InclusionSet(CloneGroup):
    """Clone group defined by explicitly listing included assets.

    The :attr:`assets` list references scene-config attribute names
    (the same strings used as keys when accessing ``scene[name]``).

    Example::

        InclusionSet(assets=["lift_table", "lift_object"], weight=1)
    """

    assets: list[str] = MISSING
    """Asset names to include in this group."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        known = set(all_asset_names)
        return [a for a in self.assets if a in known]


@configclass
class ExclusionSet(CloneGroup):
    """Clone group that includes everything *except* the listed assets.

    Useful when a group should contain most scene assets and only a
    few should be excluded.

    Example::

        ExclusionSet(exclude=["ground_plane", "light"], weight=1)
    """

    exclude: list[str] = MISSING
    """Asset names to *exclude* from this group."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        excluded = set(self.exclude)
        return [a for a in all_asset_names if a not in excluded]


@configclass
class PrefixGroup(CloneGroup):
    """Clone group that selects assets whose name starts with a prefix.

    Example::

        PrefixGroup(prefix="lift_", weight=1)
        # matches "lift_table", "lift_object", ...
    """

    prefix: str = MISSING
    """Assets whose name starts with this string are included."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        return [a for a in all_asset_names if a.startswith(self.prefix)]


@configclass
class SuffixGroup(CloneGroup):
    """Clone group that selects assets whose name ends with a suffix.

    Example::

        SuffixGroup(suffix="_frame", weight=1)
        # matches "ee_frame", "cabinet_frame", ...
    """

    suffix: str = MISSING
    """Assets whose name ends with this string are included."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        return [a for a in all_asset_names if a.endswith(self.suffix)]


@configclass
class PatternGroup(CloneGroup):
    """Clone group that selects assets matching any of the given regex patterns.

    Each pattern is tested as a **full match** against the asset name
    (equivalent to ``re.fullmatch``).  Standard ``re`` syntax is supported.

    Example::

        PatternGroup(patterns=["lift_.*", "cabinet"], weight=1)
        # matches "lift_table", "lift_object", "cabinet"
    """

    patterns: list[str] = MISSING
    """Regex patterns (full-match) to test against asset names."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        compiled = [re.compile(p) for p in self.patterns]
        return [a for a in all_asset_names if any(r.fullmatch(a) for r in compiled)]


@configclass
class PredicateGroup(CloneGroup):
    """Clone group defined by an arbitrary callable predicate.

    The :attr:`predicate` receives each asset name and returns ``True``
    to include it.  This is the most flexible built-in descriptor.

    Example::

        PredicateGroup(
            predicate=lambda name: "sensor" not in name,
            weight=2,
        )
    """

    predicate: Callable[[str], bool] = MISSING
    """Callable that returns ``True`` for asset names to include."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        return [a for a in all_asset_names if self.predicate(a)]


@configclass
class UnionGroup(CloneGroup):
    """Clone group that takes the union of multiple child descriptors.

    An asset is included if **any** child descriptor claims it.

    Example::

        UnionGroup(
            groups=[
                PrefixGroup(prefix="lift_"),
                InclusionSet(assets=["shared_sensor"]),
            ],
            weight=1,
        )
    """

    groups: list[CloneGroup] = MISSING
    """Child descriptors whose results are merged (union)."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for g in self.groups:
            for a in g.resolve_assets(all_asset_names):
                if a not in seen:
                    seen.add(a)
                    result.append(a)
        return result


@configclass
class IntersectionGroup(CloneGroup):
    """Clone group that takes the intersection of multiple child descriptors.

    An asset is included only if **all** child descriptors claim it.

    Example::

        IntersectionGroup(
            groups=[
                PrefixGroup(prefix="lift_"),
                ExclusionSet(exclude=["lift_debug_viz"]),
            ],
            weight=1,
        )
    """

    groups: list[CloneGroup] = MISSING
    """Child descriptors whose results are intersected."""

    def resolve_assets(self, all_asset_names: list[str]) -> list[str]:
        if not self.groups:
            return []
        sets = [set(g.resolve_assets(all_asset_names)) for g in self.groups]
        common = sets[0]
        for s in sets[1:]:
            common &= s
        return [a for a in all_asset_names if a in common]


# ── top-level config ──────────────────────────────────────────────────────────


@configclass
class CloneCfg:
    """Describes how to partition environments into groups for selective cloning.

    Each entry in :attr:`clone_groups` maps a group name to a
    :class:`CloneGroup` descriptor.  Built-in descriptors include
    :class:`InclusionSet`; users can subclass :class:`CloneGroup` to
    implement arbitrary grouping logic.

    Assets **not** mentioned in any group are cloned into **all**
    environments (the default behaviour).  An asset name appearing
    in multiple groups is cloned into the **union** of those groups'
    environments.

    The :attr:`clone_strategy` controls how environments are assigned to
    groups and how prototype variants are selected:

    - ``random``: Shuffled assignment (env IDs randomly distributed)
    - ``sequential``: Contiguous blocks (env 0..n to group 0, etc.)
    - ``interleaved``: Alternating/cycling pattern (env 0 to group 0, env 1 to group 1, ...)

    Example::

        from isaaclab.cloner import random, sequential, interleaved

        CloneCfg(
            clone_strategy=random,  # shuffled env assignment
            clone_groups={
                "lift": InclusionSet(assets=["lift_table", "lift_object"], weight=1),
                "cabinet": InclusionSet(assets=["cabinet", "cabinet_frame"], weight=1),
                "reach": InclusionSet(assets=["reach_table"], weight=1),
            },
        )
    """

    clone_groups: dict[str, CloneGroup] = MISSING
    """Mapping from group name to its :class:`CloneGroup` descriptor."""

    clone_strategy: callable = random_strategy
    """Strategy for env-to-group assignment. Default is :func:`random` (shuffled)."""
