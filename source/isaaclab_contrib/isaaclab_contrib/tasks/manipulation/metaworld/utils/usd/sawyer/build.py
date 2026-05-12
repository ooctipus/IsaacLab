# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build ``sawyer_with_gripper.usd`` by grafting Meta-World's parallel-jaw
gripper onto Nucleus's instanceable Sawyer arm.

Why graft instead of converting the full MJCF: Meta-World's MJCF→USD output
nests bodies in a tree (``/.../base/right_arm_base_link/right_l0/right_l1/...``)
which PhysX's replicated-physics fast-path doesn't initialise correctly under
env cloning. The Nucleus Sawyer USD has bodies as flat siblings under
``/sawyer/`` and is known to clone cleanly. So we keep the Nucleus arm
verbatim, then author 5 small bodies + 5 joints for the Meta-World gripper
as flat siblings — same structural pattern as the rest of the rig.

Geometry (all boxes, no meshes):

* ``hand``      — rail box ``0.005×0.055×0.005`` m, fixed-jointed to ``right_hand``
                  with offset ``(0, 0, 0.12)`` m and Meta-World's ``hand`` quat.
* ``rightclaw`` — box ``0.045×0.003×0.015`` m, prismatic-jointed to ``hand``
                  at ``(0, -0.05, 0)``, axis ``Y``, range ``[0, 0.04]``.
* ``leftclaw``  — same box, prismatic-jointed to ``hand`` at ``(0, +0.05, 0)``,
                  axis ``Y``, range ``[-0.03, 0]``.
* ``rightpad``  — same box, fixed-jointed to ``rightclaw`` at ``(0, +0.003, 0)``.
* ``leftpad``   — same box, fixed-jointed to ``leftclaw`` at ``(0, -0.003, 0)``.

Action semantics from Meta-World ``do_simulation([+a, -a], ...)``:
``action[-1] = +1`` drives ``r_close → 0.04`` and ``l_close → -0.03``, which
moves both pads toward the centerline (``r_close`` adds ``+Y`` to rightclaw,
``l_close`` adds ``+Y * (-0.03) = -0.03Y`` to leftclaw — both toward ``Y=0``).

Usage::

    ./isaaclab.sh -p source/.../metaworld/assets/sawyer/sawyer_with_gripper.py
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# This script lives at utils/usd/sawyer/; the USD goes to assets/sawyer/usd/.
_DEFAULT_OUT = Path(__file__).resolve().parents[3] / "assets" / "sawyer" / "usd" / "sawyer_with_gripper.usda"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default=str(_DEFAULT_OUT))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

NUCLEUS_SAWYER_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/RethinkRobotics/Sawyer/sawyer_instanceable.usd"

# ── geometry constants (verbatim from Meta-World MJCF) ────────────────────────
HAND_OFFSET_FROM_RIGHT_HAND_POS = (0.0, 0.0, 0.12)
# MJCF quat (-1, 0, 1, 0) normalised → -π/2 about +Y
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ = (-_INV_SQRT2, 0.0, _INV_SQRT2, 0.0)

RAIL_HALF_EXTENTS = (0.005, 0.055, 0.005)
PAD_HALF_EXTENTS = (0.045, 0.003, 0.015)

R_CLOSE_LIMITS = (0.0, 0.04)
L_CLOSE_LIMITS = (-0.03, 0.0)

RIGHT_CLAW_LOCAL_Y = -0.05
LEFT_CLAW_LOCAL_Y = +0.05
RIGHTPAD_LOCAL_Y = +0.003  # rightpad attaches to rightclaw at +Y from claw
LEFTPAD_LOCAL_Y = -0.003  # leftpad attaches to leftclaw at -Y

GRIPPER_BODY_MASS = 0.05  # rail
PAD_MASS = 0.1  # rightpad/leftpad — Meta-World uses mass=1 for rightpad; we
# use 0.1 so the gripper isn't tip-heavy. Reward terms only
# read pad poses, not dynamics.
ARMATURE = 0.001


# ── USD authoring helpers ─────────────────────────────────────────────────────


def _set_xform(
    prim: Usd.Prim,
    translate: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Author a single ``translate`` + ``orient`` xform pair on ``prim``."""
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t_op = xformable.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*translate))
    r_op = xformable.AddOrientOp()
    r_op.Set(Gf.Quatf(*quat_wxyz))


def _add_rigid_body_box(
    stage: Usd.Stage,
    prim_path: str,
    *,
    half_extents: tuple[float, float, float],
    translate: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mass: float,
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    collidable: bool = True,
    with_geom: bool = True,
) -> Usd.Prim:
    """Create an Xform with a PhysicsRigidBodyAPI + PhysicsMassAPI, optionally
    plus a Cube child as the visual/collision geom.

    Args:
        with_geom: If ``False``, skip the cube child — the body is a pure
            kinematic frame with mass+inertia but no visual or collision.
            Used for the ``hand`` body (Meta-World's rail is dropped here for
            visual cleanliness; the body itself still anchors the gripper
            joints).
        collidable: When ``with_geom=True``, controls whether
            :class:`PhysicsCollisionAPI` is applied to the cube.
    """
    body = stage.DefinePrim(prim_path, "Xform")
    _set_xform(body, translate, quat_wxyz)

    # rigid body schemas
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    body.GetAttribute("physics:mass").Set(mass)
    # Cheap diagonal inertia ~ box principal inertia, scaled
    diag = (
        (mass / 12.0) * (half_extents[1] ** 2 + half_extents[2] ** 2),
        (mass / 12.0) * (half_extents[0] ** 2 + half_extents[2] ** 2),
        (mass / 12.0) * (half_extents[0] ** 2 + half_extents[1] ** 2),
    )
    body.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(*diag))

    if not with_geom:
        return body

    # cube child for visual + (optional) collision
    cube_path = f"{prim_path}/box"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    # Cube's `size` is the side length; scale via xformOp:scale so we get
    # arbitrary box dimensions without writing custom geometry.
    cube.GetSizeAttr().Set(2.0)  # unit cube ±1; scale handles the rest
    cube_xformable = UsdGeom.Xformable(cube)
    cube_xformable.ClearXformOpOrder()
    cube_xformable.AddScaleOp().Set(Gf.Vec3f(*half_extents))
    cube.CreateDisplayColorAttr([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
    if collidable:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return body


def _add_fixed_joint(
    stage: Usd.Stage,
    prim_path: str,
    *,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot0_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot1_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> Usd.Prim:
    joint = UsdPhysics.FixedJoint.Define(stage, prim_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(*local_rot0_wxyz))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(*local_rot1_wxyz))
    return joint.GetPrim()


def _add_prismatic_joint(
    stage: Usd.Stage,
    prim_path: str,
    *,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    axis: str,
    lower: float,
    upper: float,
    drive_stiffness: float = 0.0,
    drive_damping: float = 0.0,
) -> Usd.Prim:
    joint = UsdPhysics.PrismaticJoint.Define(stage, prim_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(lower)
    joint.CreateUpperLimitAttr().Set(upper)
    # Apply position drive so PhysX actuates against the target.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateMaxForceAttr().Set(200.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateDampingAttr().Set(drive_damping)
    drive.CreateStiffnessAttr().Set(drive_stiffness)
    # Joint state API so IsaacLab can read joint pos/vel.
    UsdPhysics.PhysicsJointStateAPI = getattr(UsdPhysics, "PhysicsJointStateAPI", None)
    PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    if joint.GetPrim().HasAttribute("physxJoint:armature"):
        joint.GetPrim().GetAttribute("physxJoint:armature").Set(ARMATURE)
    return joint.GetPrim()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Add the Sawyer arm by referencing the Nucleus instanceable USD into /sawyer.
    sawyer = stage.DefinePrim("/sawyer", "Xform")
    sawyer.GetReferences().AddReference(NUCLEUS_SAWYER_USD)
    stage.SetDefaultPrim(sawyer)
    print("[graft] referenced Nucleus arm at /sawyer")

    # Articulation root sits on /sawyer (the Nucleus USD applies it on its
    # default-prim, and our reference inherits it).

    # ── hand (rail) ────────────────────────────────────────────────────────
    # Initial world position is approximate — the fixed joint will pull it
    # into place at sim init. We place near right_hand's expected pose so the
    # initial transient is small.
    _add_rigid_body_box(
        stage,
        "/sawyer/hand",
        half_extents=RAIL_HALF_EXTENTS,
        translate=(0.0, 0.0, 0.12),  # placeholder — refined by joint at sim time
        quat_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        mass=GRIPPER_BODY_MASS,
        # ``hand`` is a kinematic frame anchoring the gripper joints; we drop
        # its visual geom (Meta-World's rail) entirely. The cube would have
        # to be ``contype=0 conaffinity=0`` per the MJCF default class anyway,
        # and visually it just makes the gripper look detached from the wrist.
        with_geom=False,
    )
    # right_hand → hand fixed joint (joint sits inside body1 = hand).
    _add_fixed_joint(
        stage,
        "/sawyer/hand/right_hand_to_hand",
        body0_path="/sawyer/right_hand",
        body1_path="/sawyer/hand",
        local_pos0=HAND_OFFSET_FROM_RIGHT_HAND_POS,
        local_rot0_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        local_pos1=(0.0, 0.0, 0.0),
        local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    # ── rightclaw + r_close prismatic ──────────────────────────────────────
    # Initial xform Z = 0.075 instead of 0.12: the right_hand → hand joint
    # pulls hand from initial z=0.12 down to hand_z = right_hand_z - 0.12
    # (≈ 0.207 in our setup), but the chain hand → claw → pad apparently
    # doesn't propagate that translation correctly under PhysX flat-sibling
    # articulation parsing — pads end up ~4.5 cm above hand at runtime
    # instead of co-located (which is what MW's MJCF tree gives). Authoring
    # the claws/pads at z=0.075 (= 0.12 - 0.045) places them at the right
    # final world Z so they end up co-located with hand, matching MW MJCF.
    _add_rigid_body_box(
        stage,
        "/sawyer/rightclaw",
        half_extents=PAD_HALF_EXTENTS,
        translate=(0.0, RIGHT_CLAW_LOCAL_Y, 0.12),
        quat_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        mass=GRIPPER_BODY_MASS,
        rgba=(1.0, 1.0, 1.0, 1.0),
    )
    _add_prismatic_joint(
        stage,
        "/sawyer/rightclaw/r_close",
        body0_path="/sawyer/hand",
        body1_path="/sawyer/rightclaw",
        local_pos0=(0.0, RIGHT_CLAW_LOCAL_Y, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Y",
        lower=R_CLOSE_LIMITS[0],
        upper=R_CLOSE_LIMITS[1],
        drive_stiffness=400.0,
        drive_damping=10.0,
    )

    # ── rightpad fixed-jointed to rightclaw ────────────────────────────────
    # See rightclaw note above for the 0.12 → 0.075 z-shift rationale.
    _add_rigid_body_box(
        stage,
        "/sawyer/rightpad",
        half_extents=PAD_HALF_EXTENTS,
        translate=(0.0, RIGHT_CLAW_LOCAL_Y + RIGHTPAD_LOCAL_Y, 0.12),
        quat_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        mass=PAD_MASS,
        rgba=(1.0, 1.0, 1.0, 1.0),
    )
    _add_fixed_joint(
        stage,
        "/sawyer/rightpad/rightpad_to_rightclaw",
        body0_path="/sawyer/rightclaw",
        body1_path="/sawyer/rightpad",
        # ``+0.045`` in claw-local X compensates the 4.5 cm pad-vs-hand
        # z-offset PhysX introduces. Claw body world rotation maps
        # local +X → world -Z (post-rotation chain), so +0.045 in local X
        # shifts pad world Z down by 4.5 cm.
        local_pos0=(0.045, RIGHTPAD_LOCAL_Y, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )

    # ── leftclaw + l_close prismatic ───────────────────────────────────────
    _add_rigid_body_box(
        stage,
        "/sawyer/leftclaw",
        half_extents=PAD_HALF_EXTENTS,
        translate=(0.0, LEFT_CLAW_LOCAL_Y, 0.12),
        quat_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        mass=GRIPPER_BODY_MASS,
        rgba=(0.0, 1.0, 1.0, 1.0),
    )
    _add_prismatic_joint(
        stage,
        "/sawyer/leftclaw/l_close",
        body0_path="/sawyer/hand",
        body1_path="/sawyer/leftclaw",
        local_pos0=(0.0, LEFT_CLAW_LOCAL_Y, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Y",
        lower=L_CLOSE_LIMITS[0],
        upper=L_CLOSE_LIMITS[1],
        drive_stiffness=400.0,
        drive_damping=10.0,
    )

    # ── leftpad fixed-jointed to leftclaw ──────────────────────────────────
    _add_rigid_body_box(
        stage,
        "/sawyer/leftpad",
        half_extents=PAD_HALF_EXTENTS,
        translate=(0.0, LEFT_CLAW_LOCAL_Y + LEFTPAD_LOCAL_Y, 0.12),
        quat_wxyz=HAND_OFFSET_FROM_RIGHT_HAND_QUAT_WXYZ,
        mass=PAD_MASS,
        rgba=(0.0, 1.0, 1.0, 1.0),
    )
    _add_fixed_joint(
        stage,
        "/sawyer/leftpad/leftpad_to_leftclaw",
        body0_path="/sawyer/leftclaw",
        body1_path="/sawyer/leftpad",
        local_pos0=(0.045, LEFTPAD_LOCAL_Y, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )

    stage.GetRootLayer().Save()
    print(f"[graft] wrote {out_path} ({os.path.getsize(out_path)} bytes)")


main()
simulation_app.close()
