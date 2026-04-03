# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for environment cloning partitioning."""

from __future__ import annotations

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

    def disabled_asset_names(self, all_asset_names: list[str]) -> set[str]:
        """Return asset names that belong exclusively to weight-zero groups.

        Assets shared with at least one enabled group (weight > 0) are not
        returned even if they also appear in a disabled group.

        Args:
            all_asset_names: All asset names defined in the scene config.

        Returns:
            Names that should not be spawned because every group that claims
            them has ``weight == 0``.
        """
        all_claimed: set[str] = set()
        enabled_assets: set[str] = set()
        for group in self.clone_groups.values():
            assets = set(group.resolve_assets(all_asset_names))
            all_claimed |= assets
            if group.weight > 0:
                enabled_assets |= assets
        return all_claimed - enabled_assets
