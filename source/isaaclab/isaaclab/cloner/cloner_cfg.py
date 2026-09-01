# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

DEFAULT_ENV_TEMPLATE = "/World/envs/env_{}"
"""Default path template for a replicated env prim; ``{}`` marks the environment index."""


def expand_env_regex_ns(path_expr: str, env_template: str = DEFAULT_ENV_TEMPLATE) -> str:
    """Replace the ``{ENV_REGEX_NS}`` macro with the environment namespace it stands for.

    The macro spares a configuration from spelling the namespace, and with it the segment
    wildcard that names one environment. Clone planning expands it against the clone template,
    and entity constructors expand it on their private cfg copy.

    Args:
        path_expr: Prim path expression, with or without the macro.
        env_template: Environment path template whose ``{}`` marks the environment index.

    Returns:
        ``path_expr`` with the macro replaced, unchanged when it holds no macro.
    """
    # a plain replace, not str.format: the rest of the expression may hold braces of its own
    return path_expr.replace("{ENV_REGEX_NS}", env_template.format("[^/]+"))


@configclass
class InclusionSet:
    """Legal clone combination defined by explicitly listing active assets."""

    assets: list[str] = MISSING
    """Scene asset names active in this clone combination."""

    weight: int = 1
    """Relative sampling weight for this clone combination."""


@configclass
class CloneCfg:
    """Configuration for environment replication.

    The composition root includes this policy in its configuration tree, and every scene
    consumer receives the resulting published plan.
    """

    clone_combinations: list[InclusionSet] = []
    """Legal scene-asset combinations for heterogeneous clone planning.

    Each entry names the assets that are active in one legal combination.
    Assets not referenced by any entry are active in every combination. An
    empty list keeps the homogeneous/default behavior.
    """

    clone_template: str = DEFAULT_ENV_TEMPLATE
    """Path template for every replicated env prim, where ``{}`` is the environment index.

    The regex form used to expand ``{ENV_REGEX_NS}`` cfg macros is
    ``clone_template.format("[^/]+")``, which confines the slot to one path segment.
    """

    replicate_physics: bool = True
    """Whether native physics replication contexts receive the clone plan.

    Mandatory model publishers still consume the plan when this is disabled.
    """

    filter_collisions: bool = True
    """Whether PhysX collision groups isolate cloned environments."""
