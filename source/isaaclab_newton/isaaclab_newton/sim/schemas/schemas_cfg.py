# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import ClassVar, Literal

from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootBaseCfg,
    CollisionBaseCfg,
    DeformableBodyPropertiesBaseCfg,
    JointDriveBaseCfg,
    MeshCollisionBaseCfg,
    RigidBodyBaseCfg,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class NewtonRigidBodyPropertiesCfg(RigidBodyBaseCfg):
    """Newton-targeted rigid body properties.

    Base class for cfgs that author rigid-body attributes consumed by any of
    Newton's solver options (MuJoCo, XPBD, Featherstone, Semi-implicit, Kamino).
    Newton has no native ``newton:*`` rigid-body attributes today, so this class
    is currently empty — solver-specific subclasses (e.g.,
    :class:`MujocoRigidBodyPropertiesCfg`) carry the actual fields.

    The ``newton:`` namespace is reserved here so future Newton-native
    rigid-body fields can be added without an API change.

    See :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` for more information.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}


@configclass
class NewtonDeformableBodyPropertiesCfg(DeformableBodyPropertiesBaseCfg):
    """Newton-specific properties to apply to a deformable body.

    Currently empty. Backend-specific fields can be added here when Newton exposes
    a registered deformable body property schema.

    The ``newton:`` namespace is reserved here so future Newton-native
    deformable-body fields can be added without an API change.

    See :meth:`~isaaclab.sim.schemas.modify_deformable_body_properties` for more information.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}


@configclass
class MujocoRigidBodyPropertiesCfg(NewtonRigidBodyPropertiesCfg):
    """MuJoCo-solver-specific rigid body properties.

    Extends :class:`NewtonRigidBodyPropertiesCfg` with body-level gravity
    compensation, consumed only when running Newton's MuJoCo solver.

    See :meth:`~isaaclab.sim.schemas.modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "mjc"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    gravcomp: float | None = None
    """Gravity compensation scale for the body [dimensionless].

    ``0.0`` = no compensation; ``1.0`` = full compensation.
    Written to ``mjc:gravcomp`` on the rigid-body prim.
    Body-level gravcomp must be set for joint-level actuatorgravcomp to have any effect.
    """


@configclass
class NewtonJointDrivePropertiesCfg(JointDriveBaseCfg):
    """Newton-targeted joint drive properties.

    Base class for cfgs that author joint-drive attributes consumed by any of
    Newton's solver options. Newton has no native ``newton:*`` joint-drive
    attributes today, so this class is currently empty — solver-specific
    subclasses (e.g., :class:`MujocoJointDrivePropertiesCfg`) carry the actual
    fields.

    The ``newton:`` namespace is reserved here so future Newton-native
    joint-drive fields can be added without an API change.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}


@configclass
class MujocoJointDrivePropertiesCfg(NewtonJointDrivePropertiesCfg):
    """MuJoCo-solver-specific joint drive properties.

    Extends :class:`NewtonJointDrivePropertiesCfg` with joint-level gravity
    compensation routing, consumed only when running Newton's MuJoCo solver.

    See :meth:`~isaaclab.sim.schemas.modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "mjc"
    _usd_applied_schema: ClassVar[str | None] = "MjcJointAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    actuatorgravcomp: bool | None = None
    """Route gravity compensation forces through the actuator channel.

    When ``True``, compensation forces go to ``qfrc_actuator`` (subject to force limits).
    Requires body-level :attr:`MujocoRigidBodyPropertiesCfg.gravcomp`.
    Written to ``mjc:actuatorgravcomp`` via ``MjcJointAPI``.
    """


@configclass
class NewtonCollisionPropertiesCfg(CollisionBaseCfg):
    """Newton-specific collision properties.

    Extends :class:`~isaaclab.sim.schemas.CollisionBaseCfg` with Newton-native
    contact geometry attributes.

    See :meth:`~isaaclab.sim.schemas.modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    contact_margin: float | None = None
    """Outward inflation of the collision surface [m].

    Extends the effective collision surface outward. Sum of both bodies' margins is
    used for collision detection. Essential for thin shells and cloth.
    Written to ``newton:contactMargin`` via ``NewtonCollisionAPI``.
    Range: [0, inf).
    """

    contact_gap: float | None = None
    """Additional contact detection gap [m].

    AABBs are expanded by this value; contacts detected earlier to avoid tunneling.
    Written to ``newton:contactGap`` via ``NewtonCollisionAPI``.
    Set to ``-inf`` to use Newton's builder default. Range: [0, inf).
    """


@configclass
class MujocoCollisionPropertiesCfg(NewtonCollisionPropertiesCfg):
    """MuJoCo-solver-specific collision properties.

    Extends :class:`NewtonCollisionPropertiesCfg` with the raw per-geometry
    contact parameters consumed by Newton's MuJoCo solver.

    See :meth:`~isaaclab.sim.schemas.modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "mjc"
    _usd_applied_schema: ClassVar[str | None] = "MjcCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}
    _usd_field_types: ClassVar[dict[str, str]] = {"solimp": "DoubleArray", "solref": "DoubleArray"}

    margin: float | None = None
    """MuJoCo collision-detection margin [m].

    Written to ``mjc:margin`` via ``MjcCollisionAPI``. MuJoCo activates the
    constraint when distance falls below ``margin - gap``.
    """

    solimp: tuple[float, float, float, float, float] | None = None
    """MuJoCo contact-impedance parameters.

    Components are ``[0] d0 [dimensionless]``, ``[1] d_width [dimensionless]``,
    ``[2] width [m]``, ``[3] midpoint [dimensionless]``, and
    ``[4] power [dimensionless]``. Written to ``mjc:solimp`` via
    ``MjcCollisionAPI``.
    """

    solref: tuple[float, float] | None = None
    """MuJoCo contact reference parameters.

    In the positive convention, components are ``[0] time constant [s]`` and
    ``[1] damping ratio [dimensionless]``. When both values are negative,
    MuJoCo interprets their magnitudes as direct stiffness and damping. Their
    SI units depend on the constraint dimension: translational contact uses
    [N/m] and [N·s/m], while rotational contact uses [N·m/rad] and
    [N·m·s/rad]. Written without conversion to ``mjc:solref`` via
    ``MjcCollisionAPI``.
    """

    def __post_init__(self) -> None:
        """Validate MuJoCo's collision-parameter domains."""
        if self.margin is not None and (not math.isfinite(self.margin) or self.margin < 0.0):
            raise ValueError("margin must be finite and non-negative.")
        if self.solimp is not None:
            if len(self.solimp) != 5 or not all(math.isfinite(value) for value in self.solimp):
                raise ValueError("solimp must contain five finite values.")
            d0, d_width, width, midpoint, power = self.solimp
            if not (0.0 <= d0 <= 1.0 and 0.0 <= d_width <= 1.0):
                raise ValueError("solimp d0 and d_width must lie in [0, 1].")
            if width <= 0.0:
                raise ValueError("solimp width must be positive.")
            if not 0.0 <= midpoint <= 1.0:
                raise ValueError("solimp midpoint must lie in [0, 1].")
            if power < 1.0:
                raise ValueError("solimp power must be at least one.")
        if self.solref is not None:
            if len(self.solref) != 2 or not all(math.isfinite(value) for value in self.solref):
                raise ValueError("solref must contain two finite values.")
            first, second = self.solref
            if not ((first > 0.0 and second > 0.0) or (first < 0.0 and second < 0.0)):
                raise ValueError("solref values must be both positive or both negative.")


@configclass
class NewtonMeshCollisionPropertiesCfg(NewtonCollisionPropertiesCfg, MeshCollisionBaseCfg):
    """Newton-specific mesh collision properties.

    Extends :class:`NewtonCollisionPropertiesCfg` with convex-hull vertex limit.

    See :meth:`~isaaclab.sim.schemas.modify_mesh_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMeshCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    max_hull_vertices: int | None = None
    """Maximum vertices in the convex hull approximation [dimensionless].

    Only relevant when ``physics:approximation = "convexHull"``.
    Written to ``newton:maxHullVertices`` via ``NewtonMeshCollisionAPI``.
    Set to ``-1`` to use as many vertices as needed for a perfect hull.
    """


@configclass
class NewtonSDFCollisionPropertiesCfg(NewtonCollisionPropertiesCfg):
    """Newton-specific SDF and hydroelastic collision properties.

    Extends :class:`NewtonCollisionPropertiesCfg` with SDF generation and
    hydroelastic-contact attributes consumed by Newton's USD importer.

    See :meth:`~isaaclab.sim.schemas.modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonSDFCollisionAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    sdf_max_resolution: int | None = None
    """Maximum SDF grid dimension.

    Newton requires this value to be divisible by 8. If
    :attr:`sdf_target_voxel_size` is also authored, Newton uses the target voxel
    size and ignores this resolution.
    Written to ``newton:sdfMaxResolution`` via ``NewtonSDFCollisionAPI``.
    """

    sdf_narrow_band_inner: float | None = None
    """Inner narrow-band distance for SDF generation [m].

    Written to ``newton:sdfNarrowBandInner`` via ``NewtonSDFCollisionAPI``.
    """

    sdf_narrow_band_outer: float | None = None
    """Outer narrow-band distance for SDF generation [m].

    Written to ``newton:sdfNarrowBandOuter`` via ``NewtonSDFCollisionAPI``.
    """

    sdf_target_voxel_size: float | None = None
    """Target SDF voxel size [m].

    Takes precedence over :attr:`sdf_max_resolution` in Newton's USD importer.
    Written to ``newton:sdfTargetVoxelSize`` via ``NewtonSDFCollisionAPI``.
    """

    sdf_texture_format: Literal["uint8", "uint16", "float32"] | None = None
    """Subgrid texture storage format for generated SDFs.

    Written to ``newton:sdfTextureFormat`` via ``NewtonSDFCollisionAPI``.
    """

    sdf_padding: float | None = None
    """SDF AABB padding [m].

    Written to ``newton:sdfPadding`` via ``NewtonSDFCollisionAPI``.
    """

    hydroelastic_enabled: bool | None = None
    """Whether Newton should use SDF-based hydroelastic contacts for this shape.

    Both participating collision shapes must enable hydroelastic contacts for
    Newton to use this path. Written to ``newton:hydroelasticEnabled`` via
    ``NewtonSDFCollisionAPI``.
    """

    hydroelastic_stiffness: float | None = None
    """Hydroelastic contact stiffness.

    Written to ``newton:hydroelasticStiffness`` via ``NewtonSDFCollisionAPI``.
    """


@configclass
class NewtonMaterialPropertiesCfg(RigidBodyMaterialBaseCfg):
    """Newton-specific rigid body material properties.

    Extends :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`
    with Newton-native friction attributes.

    See :meth:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    torsional_friction: float | None = None
    """Torsional friction coefficient (resistance to spinning at a contact point) [dimensionless].

    Written to ``newton:torsionalFriction`` via ``NewtonMaterialAPI``.
    Range: [0, inf).
    """

    rolling_friction: float | None = None
    """Rolling friction coefficient (resistance to rolling motion) [dimensionless].

    Written to ``newton:rollingFriction`` via ``NewtonMaterialAPI``.
    Range: [0, inf).
    """


@configclass
class NewtonArticulationRootPropertiesCfg(ArticulationRootBaseCfg):
    """Newton-specific articulation root properties.

    Extends :class:`~isaaclab.sim.schemas.ArticulationRootBaseCfg` with
    Newton-native self-collision control.

    See :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonArticulationRootAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    self_collision_enabled: bool | None = None
    """Whether self-collisions between bodies in this articulation are enabled.

    Written to ``newton:selfCollisionEnabled`` via ``NewtonArticulationRootAPI``.
    Newton's resolver checks this native attribute first before falling back to
    ``physxArticulation:enabledSelfCollisions``.
    """
