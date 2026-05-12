# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author ``mw_drawer.usda`` — a flat-sibling articulated drawer that
mimics Meta-World's MJCF drawer geometry exactly.

Why hand-author instead of using Sektion: the Sektion cabinet's body
walls block the Sawyer's gripper approach paths because Sektion is a
full kitchen unit (~80 cm tall). MW's drawer is a small box (17 cm tall)
with a U-shaped handle the gripper can hook through from above. We
replicate MW's exact geometry as flat-sibling rigid bodies + a single
prismatic joint, mirroring the approach used for ``sawyer_with_gripper``.

MJCF reference: ``metaworld/assets/objects/assets/drawer.xml``.

Geometry (all boxes, masses approximate MW's MJCF values):

* ``drawercase``: cabinet body composed of 5 collision boxes
    - left wall   (-0.11, 0,    0)     half-extents (0.008, 0.1,   0.084)
    - right wall  (+0.11, 0,    0)     same
    - back wall   (0,    +0.092, -0.008) (0.102, 0.008, 0.076)
    - bottom      (0,    -0.008, -0.07)  (0.102, 0.092, 0.014)
    - top         (0,     0,    +0.076) (0.102, 0.1,   0.008)
  positioned at world (0, 0.9, 0.084) — matches MW.

* ``drawer_link``: slidable drawer composed of 5 walls + 3 handle capsules
  (we use boxes for capsules — collision behaviour is similar enough).
  Positioned at offset (0, -0.01, 0.006) from drawercase, with a prismatic
  joint along world Y, range ``[-0.16, 0]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# This script lives at utils/usd/; the drawer USD goes to assets/drawer/usd/.
_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "assets" / "drawer" / "usd" / "mw_drawer.usda"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default=str(_DEFAULT_OUT))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# ── MW MJCF geometry constants ──────────────────────────────────────────────

DRAWER_BASE_POS = (0.0, 0.9, 0.084)  # MW: cabinet body @ world (0, 0.9, 0); drawercase_link offset (0, 0, 0.084).
DRAWER_LINK_OFFSET = (0.0, -0.01, 0.006)  # drawer_link offset from drawercase

# drawercase walls (half-extents, position relative to drawercase)
CASE_WALLS = [
    {"pos": (-0.11, 0.0, 0.0), "half": (0.008, 0.1, 0.084), "mass": 0.05},  # left
    {"pos": (+0.11, 0.0, 0.0), "half": (0.008, 0.1, 0.084), "mass": 0.05},  # right
    {"pos": (0.0, +0.092, -0.008), "half": (0.102, 0.008, 0.076), "mass": 0.05},  # back
    {"pos": (0.0, -0.008, -0.07), "half": (0.102, 0.092, 0.014), "mass": 0.05},  # bottom
    {"pos": (0.0, 0.0, +0.076), "half": (0.102, 0.1, 0.008), "mass": 0.05},  # top
]

# drawer_link walls (half-extents, position relative to drawer_link)
DRAWER_WALLS = [
    {"pos": (0.0, -0.082, 0.008), "half": (0.1, 0.008, 0.052), "mass": 0.04},  # front (y=-0.082)
    {"pos": (0.0, +0.082, 0.008), "half": (0.1, 0.008, 0.052), "mass": 0.04},  # back
    {"pos": (-0.092, 0.0, 0.008), "half": (0.008, 0.074, 0.052), "mass": 0.04},  # left
    {"pos": (+0.092, 0.0, 0.008), "half": (0.008, 0.074, 0.052), "mass": 0.04},  # right
    {"pos": (0.0, 0.0, -0.052), "half": (0.1, 0.09, 0.008), "mass": 0.04},  # bottom
]

# Handle capsules (approximated as boxes for collision simplicity).
# MW uses 3 capsules forming a U: two vertical posts at x=±0.05,
# one horizontal bar at y=-0.15. We approximate each as a thin box.
HANDLE_BARS = [
    {"pos": (-0.05, -0.12, 0.0), "half": (0.009, 0.03, 0.009), "mass": 0.02},  # left post
    {"pos": (+0.05, -0.12, 0.0), "half": (0.009, 0.03, 0.009), "mass": 0.02},  # right post
    {"pos": (0.0, -0.15, 0.0), "half": (0.05, 0.009, 0.009), "mass": 0.02},  # cross bar
]

DRAWER_JOINT_RANGE = (-0.16, 0.0)
DRAWER_JOINT_DAMPING = 2.0  # MW MJCF damping="2"


# ── USD authoring helpers (identical pattern to sawyer_with_gripper.py) ─────


def _set_xform(prim, translate, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddOrientOp().Set(Gf.Quatf(*quat_wxyz))


def _add_rigid_body_box(
    stage,
    prim_path: str,
    *,
    half_extents,
    translate,
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    mass: float,
    rgba=(0.7, 0.55, 0.4, 1.0),
    collidable=True,
):
    body = stage.DefinePrim(prim_path, "Xform")
    _set_xform(body, translate, quat_wxyz)

    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    body.GetAttribute("physics:mass").Set(mass)
    diag = (
        (mass / 12.0) * (half_extents[1] ** 2 + half_extents[2] ** 2),
        (mass / 12.0) * (half_extents[0] ** 2 + half_extents[2] ** 2),
        (mass / 12.0) * (half_extents[0] ** 2 + half_extents[1] ** 2),
    )
    body.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(*diag))

    cube_path = f"{prim_path}/box"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(2.0)
    cube_xformable = UsdGeom.Xformable(cube)
    cube_xformable.ClearXformOpOrder()
    cube_xformable.AddScaleOp().Set(Gf.Vec3f(*half_extents))
    cube.CreateDisplayColorAttr([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
    if collidable:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return body


def _add_static_geom_to_body(
    stage,
    parent_body_path: str,
    name: str,
    *,
    half_extents,
    local_pos,
    rgba=(0.7, 0.55, 0.4, 1.0),
):
    """Attach a Cube as a child geom (collision + visual) of an existing body."""
    cube_path = f"{parent_body_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*half_extents))
    cube.CreateDisplayColorAttr([Gf.Vec3f(rgba[0], rgba[1], rgba[2])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def _add_fixed_joint(
    stage, prim_path, *, body0_path, body1_path, local_pos0=(0.0, 0.0, 0.0), local_pos1=(0.0, 0.0, 0.0)
):
    j = UsdPhysics.FixedJoint.Define(stage, prim_path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return j.GetPrim()


def _add_prismatic_joint(
    stage, prim_path, *, body0_path, body1_path, local_pos0, local_pos1, axis, lower, upper, damping=0.0, stiffness=0.0
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


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim("/drawer", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # ── drawercase body ─────────────────────────────────────────────────────
    # The drawercase is a STATIC cabinet shell — we set it as kinematic by
    # making it the articulation's fixed base.
    _add_rigid_body_box(
        stage,
        "/drawer/drawercase",
        half_extents=(0.001, 0.001, 0.001),  # tiny placeholder body anchor
        translate=DRAWER_BASE_POS,
        mass=1.0,
        rgba=(0.6, 0.45, 0.3, 1.0),
    )
    # Attach the case walls as static collision geoms on the drawercase body.
    for i, w in enumerate(CASE_WALLS):
        _add_static_geom_to_body(
            stage,
            "/drawer/drawercase",
            f"wall_{i}",
            half_extents=w["half"],
            local_pos=w["pos"],
            rgba=(0.6, 0.45, 0.3, 1.0),
        )

    # Make the drawercase kinematic (welded to world) by adding a fixed joint
    # to a synthetic worldFrame. PhysX requires articulation roots to be
    # rooted; the easiest is to add a fixed joint between drawercase and
    # the implicit world frame.
    fixed_to_world = UsdPhysics.FixedJoint.Define(stage, "/drawer/case_to_world")
    fixed_to_world.CreateBody1Rel().SetTargets([Sdf.Path("/drawer/drawercase")])
    fixed_to_world.CreateLocalPos0Attr().Set(Gf.Vec3f(*DRAWER_BASE_POS))
    fixed_to_world.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_to_world.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    fixed_to_world.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    print("[drawer] drawercase walls attached")

    # ── drawer_link body ────────────────────────────────────────────────────
    drawer_translate = (
        DRAWER_BASE_POS[0] + DRAWER_LINK_OFFSET[0],
        DRAWER_BASE_POS[1] + DRAWER_LINK_OFFSET[1],
        DRAWER_BASE_POS[2] + DRAWER_LINK_OFFSET[2],
    )
    _add_rigid_body_box(
        stage,
        "/drawer/drawer_link",
        half_extents=(0.001, 0.001, 0.001),  # placeholder body anchor
        translate=drawer_translate,
        mass=0.04,
        rgba=(0.85, 0.85, 0.85, 1.0),
    )
    for i, w in enumerate(DRAWER_WALLS):
        _add_static_geom_to_body(
            stage,
            "/drawer/drawer_link",
            f"wall_{i}",
            half_extents=w["half"],
            local_pos=w["pos"],
            rgba=(0.85, 0.85, 0.85, 1.0),
        )

    # ── drawer handle (3 bars welded to drawer_link) ─────────────────────────
    for i, h in enumerate(HANDLE_BARS):
        _add_static_geom_to_body(
            stage,
            "/drawer/drawer_link",
            f"handle_{i}",
            half_extents=h["half"],
            local_pos=h["pos"],
            rgba=(0.95, 0.95, 0.95, 1.0),
        )
    print("[drawer] drawer_link walls + handle attached")

    # Add a *named* handle reference body (zero-extent rigid pose marker
    # IsaacLab can target with a FrameTransformer for the reward).
    handle_marker_path = "/drawer/drawer_handle"
    marker = stage.DefinePrim(handle_marker_path, "Xform")
    _set_xform(marker, drawer_translate, (1.0, 0.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(marker)
    UsdPhysics.MassAPI.Apply(marker)
    marker.GetAttribute("physics:mass").Set(0.0001)
    # Weld handle marker to drawer_link at the front face (y=-0.15).
    _add_fixed_joint(
        stage,
        "/drawer/handle_to_drawer",
        body0_path="/drawer/drawer_link",
        body1_path=handle_marker_path,
        local_pos0=(0.0, -0.15, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )

    # ── prismatic joint: drawer_link slides along Y in [-0.16, 0] ───────────
    _add_prismatic_joint(
        stage,
        "/drawer/goal_slidey",
        body0_path="/drawer/drawercase",
        body1_path="/drawer/drawer_link",
        local_pos0=DRAWER_LINK_OFFSET,
        local_pos1=(0.0, 0.0, 0.0),
        axis="Y",
        lower=DRAWER_JOINT_RANGE[0],
        upper=DRAWER_JOINT_RANGE[1],
        damping=DRAWER_JOINT_DAMPING,
        stiffness=0.0,
    )
    print("[drawer] goal_slidey prismatic joint added (Y axis, range [-0.16, 0])")

    stage.Save()
    print(f"[drawer] wrote {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
