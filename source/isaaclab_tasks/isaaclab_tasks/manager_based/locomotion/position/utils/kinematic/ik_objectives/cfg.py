# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for custom Newton IK objectives.

Each :class:`IKObjectiveBaseCfg` subclass declares the static parameters
of an IK objective and knows how to instantiate it against the live
:class:`RetargetPipeline` (which provides the kinematics model, foot
body indices, and sampler state that the objective needs at build
time). This keeps the pipeline config declarative: the preset lists
which objectives to attach and their knobs, without a
caller-provided factory callable.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import newton.ik as ik
    import warp as wp

    from ...mdp.retarget.pipeline import RetargetPipeline


@configclass
class IKObjectiveBaseCfg:
    """Base configuration for a retarget IK objective.

    Subclasses set :attr:`class_type` to the objective implementation
    (resolvable ``"{DIR}.module:ClassName"`` string) and implement
    :meth:`build` to plumb runtime state from the pipeline into the
    objective's constructor.
    """

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Objective implementation class (resolvable string or direct type)."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> ik.IKObjective:
        """Instantiate the IK objective against a live pipeline.

        Args:
            pipeline: The initialized :class:`RetargetPipeline`
                (provides ``kin``, ``foot_body_ids``, ``sampler``).
            wp_mesh: Terrain warp mesh for the current ``run`` call.

        Returns:
            A Newton IK objective ready to append to the solver's
            objective list.
        """
        raise NotImplementedError


@configclass
class IKObjectiveTerrainCollisionCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveTerrainCollision`.

    The objective probes robot bodies against the terrain mesh and
    penalizes penetration. Foot bodies are excluded from probing since
    they are expected to contact the terrain.
    """

    class_type: type | str = "{DIR}.terrain_collision:IKObjectiveTerrainCollision"

    weight: float = 3.0
    """Residual weight [unitless]."""

    margin: float = 0.05
    """Softplus temperature [m]. Larger values soften the penalty's knee."""

    n_samples: int = 4
    """Surface probe points per body."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> ik.IKObjective:
        from .terrain_collision import IKObjectiveTerrainCollision

        return IKObjectiveTerrainCollision(
            mesh_id=wp_mesh.id,
            builder=pipeline.kin.builder,
            exclude_bodies=pipeline.foot_body_ids,
            weight=self.weight,
            margin=self.margin,
            n_samples=self.n_samples,
        )


@configclass
class IKObjectiveStabilityMarginCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveStabilityMargin`.

    Reads the CCW foot ordering from :attr:`RetargetPipeline.sampler`
    so the stability residual's signed-area computation matches the
    sampler's polygon layout.
    """

    class_type: type | str = "{DIR}.stability_margin:IKObjectiveStabilityMargin"

    weight: float = 1.0
    """Residual weight [unitless]."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> ik.IKObjective:
        from .stability_margin import IKObjectiveStabilityMargin

        foot_ids_ccw = [int(pipeline.foot_body_ids[j]) for j in pipeline.sampler._foot_ccw_order]
        return IKObjectiveStabilityMargin(
            model=pipeline.kin.model,
            foot_body_indices=foot_ids_ccw,
            weight=self.weight,
        )


@configclass
class IKObjectiveJointRegularizeCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveJointRegularize`.

    Each entry in :attr:`joint_targets` maps a joint-name regex to the
    target angle [rad] its matched DOFs are pulled toward. DOFs not
    matched by any pattern are left free. If multiple patterns match the
    same DOF, the **last** matching entry's target wins (Python dict
    insertion order). Patterns that match zero joints on the current
    robot are silently skipped -- useful for multi-robot presets where
    each robot uses a different joint-naming convention.

    An empty :attr:`joint_targets` falls back to
    :attr:`RetargetPipelineCfg.joint_regularize_targets` (typically
    resolved per robot preset). If both are empty the objective raises
    at build time -- omit it from ``extra_objectives`` to disable
    regularization entirely.

    Example::

        IKObjectiveJointRegularizeCfg(
            joint_targets={
                ".*HAA": 0.0,  # ANYmal-C HAA -> 0 rad
                ".*hip_joint": 0.0,  # go2/b2 HAA   -> 0 rad
                ".*hip_x": 0.0,  # spot HAA     -> 0 rad
            },
            weight=3.0,
        )
    """

    class_type: type | str = "{DIR}.joint_regularize:IKObjectiveJointRegularize"

    joint_targets: dict[str, float] = field(default_factory=dict)
    """Mapping of joint-name regex -> target angle [rad]. Empty dict falls back to the pipeline cfg."""

    weight: float = 1.0
    """Uniform residual weight [unitless] applied to every matched DOF."""

    def build(self, pipeline: RetargetPipeline, wp_mesh: wp.Mesh) -> ik.IKObjective:
        from .joint_regularize import IKObjectiveJointRegularize

        targets_map = self.joint_targets if self.joint_targets else pipeline.cfg.joint_regularize_targets
        if not targets_map:
            raise ValueError(
                "IKObjectiveJointRegularizeCfg requires at least one entry in joint_targets "
                "or RetargetPipelineCfg.joint_regularize_targets (typically resolved per robot preset)."
            )
        dof_to_target: dict[int, float] = {}
        for pattern, target in targets_map.items():
            for idx in pipeline.kin.find_joint_dof_indices(pattern):
                dof_to_target[idx] = float(target)
        if not dof_to_target:
            raise ValueError(
                f"IKObjectiveJointRegularizeCfg: none of the patterns {list(targets_map)} matched any revolute joint."
            )
        indices = sorted(dof_to_target.keys())
        targets = [dof_to_target[i] for i in indices]
        return IKObjectiveJointRegularize(
            joint_dof_indices=indices,
            joint_dof_targets=targets,
            weight=self.weight,
        )
