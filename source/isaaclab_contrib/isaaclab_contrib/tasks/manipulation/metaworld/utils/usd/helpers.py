# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared USD-authoring helpers for hand-built Meta-World task assets.

The pattern (proven by ``mw_drawer.usda``):

* Each task USD has a root ``Xform`` with ``ArticulationRootAPI``.
* ``base_body`` is a kinematic rigid body welded to world via a fixed
  joint, with its outer geometry as Cube children.
* ``moving_body`` is a rigid body with the joint axis attached.
* A small zero-extent ``handle`` (or ``goal_marker``) rigid body is
  fixed-jointed to the moving_body so reward terms can read its world
  pose via :class:`~isaaclab.sensors.FrameTransformer`.
* The articulating joint (Prismatic or Revolute) connects base→moving.

This module exposes the helpers that author each piece. Each per-task
build script imports + composes these.
"""

from __future__ import annotations

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


def set_xform(prim, translate, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddOrientOp().Set(Gf.Quatf(*quat_wxyz))


def add_rigid_body_anchor(
    stage,
    prim_path: str,
    *,
    translate,
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    mass: float = 0.05,
):
    """Create a tiny (1 mm) rigid-body anchor — child geoms attach to it."""
    body = stage.DefinePrim(prim_path, "Xform")
    set_xform(body, translate, quat_wxyz)
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    body.GetAttribute("physics:mass").Set(mass)
    body.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(1e-6, 1e-6, 1e-6))
    return body


def add_box_geom(
    stage,
    parent_body_path: str,
    name: str,
    *,
    half_extents,
    local_pos=(0.0, 0.0, 0.0),
    rgba=(0.7, 0.55, 0.4, 1.0),
    collidable: bool = True,
):
    cube_path = f"{parent_body_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*half_extents))
    cube.CreateDisplayColorAttr([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
    if collidable:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def add_cylinder_geom(
    stage,
    parent_body_path: str,
    name: str,
    *,
    radius: float,
    height: float,
    axis: str = "Z",
    local_pos=(0.0, 0.0, 0.0),
    rgba=(0.5, 0.5, 0.5, 1.0),
    collidable: bool = True,
):
    cyl_path = f"{parent_body_path}/{name}"
    cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetAxisAttr().Set(axis)
    xf = UsdGeom.Xformable(cyl)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
    if collidable:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl.GetPrim()


def add_fixed_joint_to_world(stage, prim_path, *, body1_path, world_pos, world_quat=(1.0, 0.0, 0.0, 0.0)):
    """Weld a rigid body to the world frame at the given world pose."""
    j = UsdPhysics.FixedJoint.Define(stage, prim_path)
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_pos))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(*world_quat))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return j.GetPrim()


def add_fixed_joint(
    stage,
    prim_path,
    *,
    body0_path,
    body1_path,
    local_pos0=(0.0, 0.0, 0.0),
    local_pos1=(0.0, 0.0, 0.0),
    local_rot0=(1.0, 0.0, 0.0, 0.0),
    local_rot1=(1.0, 0.0, 0.0, 0.0),
):
    j = UsdPhysics.FixedJoint.Define(stage, prim_path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1))
    return j.GetPrim()


def add_prismatic_joint(
    stage,
    prim_path,
    *,
    body0_path,
    body1_path,
    local_pos0,
    local_pos1,
    axis: str,
    lower: float,
    upper: float,
    damping: float = 0.0,
    stiffness: float = 0.0,
):
    j = UsdPhysics.PrismaticJoint.Define(stage, prim_path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    j.CreateAxisAttr().Set(axis)
    j.CreateLowerLimitAttr().Set(lower)
    j.CreateUpperLimitAttr().Set(upper)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateMaxForceAttr().Set(200.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateDampingAttr().Set(damping)
    drive.CreateStiffnessAttr().Set(stiffness)
    PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "linear")
    PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
    return j.GetPrim()


def add_revolute_joint(
    stage,
    prim_path,
    *,
    body0_path,
    body1_path,
    local_pos0,
    local_pos1,
    axis: str,
    lower_deg: float,
    upper_deg: float,
    damping: float = 0.0,
    stiffness: float = 0.0,
):
    j = UsdPhysics.RevoluteJoint.Define(stage, prim_path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    j.CreateAxisAttr().Set(axis)
    j.CreateLowerLimitAttr().Set(lower_deg)
    j.CreateUpperLimitAttr().Set(upper_deg)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateMaxForceAttr().Set(200.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateDampingAttr().Set(damping)
    drive.CreateStiffnessAttr().Set(stiffness)
    PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "angular")
    PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
    return j.GetPrim()


def add_handle_marker(stage, prim_path: str, *, world_pos):
    """A zero-extent rigid-body marker for FrameTransformer to read."""
    marker = stage.DefinePrim(prim_path, "Xform")
    set_xform(marker, world_pos)
    UsdPhysics.RigidBodyAPI.Apply(marker)
    UsdPhysics.MassAPI.Apply(marker)
    marker.GetAttribute("physics:mass").Set(0.0001)
    return marker


def stage_init(out_path):
    """Create a fresh USD stage with z-up axis + meters."""
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage
