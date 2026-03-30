# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group-aware action terms for multi-task / multi-robot environments.

Mirrors the :class:`~.utils.scatter_term` pattern for observations: multiple
per-asset sub-terms share a single action-column slot in the policy's output.
The manager sees one term; internally each sub-term handles its own group.

Typical usage::

    arm = GroupedActionTermCfg(terms=[
        DifferentialInverseKinematicsActionCfg(asset_name="openarm_robot", ...),
        DifferentialInverseKinematicsActionCfg(asset_name="franka_robot", ...),
    ])
    gripper = GroupedActionTermCfg(terms=[
        BinaryJointPositionActionCfg(asset_name="openarm_robot", ...),
        BinaryJointPositionActionCfg(asset_name="franka_robot", ...),
    ])
    # action_dim = 6 + 1 = 7  (not 6 + 6 + 1 + 1 = 14)
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.scene.env_layout import GroupView, filter_to_group
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class GroupedActionTermCfg(ActionTermCfg):
    """Fan-out action term: one shared action column set dispatched to N per-group sub-terms.

    Mirrors :class:`~.utils.scatter_term` for observations.  All sub-terms must
    share the same ``action_dim``; group membership is inferred from each term's
    ``asset_name`` via the scene layout—no extra ``groups`` field needed.

    Example::

        arm = GroupedActionTermCfg(terms=[
            DifferentialInverseKinematicsActionCfg(asset_name="openarm_robot", ...),
            DifferentialInverseKinematicsActionCfg(asset_name="franka_robot", ...),
        ])
    """

    class_type: type = None
    asset_name: str = ""  # overrides MISSING from base; this term has no single asset
    terms: list[ActionTermCfg] = MISSING
    """Sub-term configs, one per robot group. All must have equal ``action_dim``."""

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = GroupedActionTerm


class GroupedActionTerm(ActionTerm):
    """Dispatches shared action columns to per-group sub-terms.

    The action manager sees one term with ``action_dim`` = shared dim. Each step:

    * :meth:`process_actions` — for each sub-term, ``actions[group_global_ids]``
      is forwarded to ``sub_term.process_actions``.
    * :meth:`apply_actions` — each sub-term's :meth:`apply_actions` is called.

    Sub-terms are instantiated with a dynamically-scoped ``num_envs`` equal to
    their group size so their internal buffers are correctly sized without any
    changes to manager source code.
    """

    cfg: GroupedActionTermCfg

    def __init__(self, cfg: GroupedActionTermCfg, env: ManagerBasedEnv):
        # Bypass ActionTerm.__init__ to skip the single-asset lookup.
        ManagerTermBase.__init__(self, cfg, env)
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True
        self._debug_vis_handle = None

        layout = env.scene.layout
        self._sub_terms: list[tuple[GroupView | None, ActionTerm]] = []

        for term_cfg in cfg.terms:
            gv = layout.view_for_asset(term_cfg.asset_name)
            asset = env.scene[term_cfg.asset_name]
            group_size = asset.num_instances
            sub_term = _make_sub_term(term_cfg, env, group_size)
            self._sub_terms.append((gv, sub_term))

        dims = {t.action_dim for _, t in self._sub_terms}
        if len(dims) != 1:
            raise ValueError(f"GroupedActionTerm: all sub-terms must share action_dim, got {sorted(dims)}")

        dim = next(iter(dims))
        self._raw_buf = torch.zeros(env.num_envs, dim, device=env.device)
        self._proc_buf = torch.zeros(env.num_envs, dim, device=env.device)

        self.set_debug_vis(cfg.debug_vis)

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._sub_terms[0][1].action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Scattered raw actions [shape ``(num_envs, action_dim)``]."""
        self._raw_buf.zero_()
        for gv, term in self._sub_terms:
            if gv is not None:
                self._raw_buf[gv.global_ids] = term.raw_actions
            else:
                self._raw_buf[:] = term.raw_actions
        return self._raw_buf

    @property
    def processed_actions(self) -> torch.Tensor:
        """Scattered processed actions [shape ``(num_envs, action_dim)``]."""
        self._proc_buf.zero_()
        for gv, term in self._sub_terms:
            if gv is not None:
                self._proc_buf[gv.global_ids] = term.processed_actions
            else:
                self._proc_buf[:] = term.processed_actions
        return self._proc_buf

    def process_actions(self, actions: torch.Tensor):
        """Gather group rows from ``actions`` and forward to each sub-term.

        Args:
            actions: Full-env action tensor, shape ``(num_envs, action_dim)``.
        """
        for gv, term in self._sub_terms:
            group_actions = actions[gv.global_ids] if gv is not None else actions
            term.process_actions(group_actions)

    def apply_actions(self) -> None:
        for _, term in self._sub_terms:
            term.apply_actions()

    def reset(self, env_ids) -> None:
        if isinstance(env_ids, slice):
            # Full reset (slice(None)) — reset each sub-term completely.
            for _, term in self._sub_terms:
                term.reset(env_ids=env_ids)
            return
        for gv, term in self._sub_terms:
            if gv is None:
                term.reset(env_ids=env_ids)
            else:
                term_env_ids, matched = filter_to_group(gv.layout, env_ids)
                if matched.numel() > 0:
                    term.reset(env_ids=term_env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        for _, term in self._sub_terms:
            term.set_debug_vis(debug_vis)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _make_sub_term(term_cfg: ActionTermCfg, env: ManagerBasedEnv, group_size: int) -> ActionTerm:
    """Return an instance of ``term_cfg.class_type`` with ``num_envs`` fixed to ``group_size``.

    Creates a lightweight subclass that overrides the ``num_envs`` property so the
    sub-term allocates ``(group_size, ...)`` buffers rather than ``(num_envs, ...)``.
    No manager source code is modified.
    """
    base_cls = term_cfg.class_type
    if not isinstance(base_cls, type):
        base_cls = base_cls._resolve()
    meta = type(base_cls)
    scoped_cls = meta(base_cls.__name__, (base_cls,), {"num_envs": property(lambda self, n=group_size: n)})
    return scoped_cls(term_cfg, env)
