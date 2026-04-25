# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-conforming spawn validation with batched IK and viser visualization.

Thin CLI wrapper around :class:`RetargetPipeline`.  Generates a terrain mesh
(either a full terrain grid from a :class:`SubTerrainPresetCfg` preset, or a
single sub-terrain), runs the pipeline, and visualises accepted/rejected
candidates in viser.

Invoke as ``./isaaclab.sh -p <this_file>.py`` with any of the arg combos below.
Setting ``$SCRIPT`` first keeps the commands readable::

    SCRIPT=source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/position/utils/tools/validate_spawn_points.py

    # Default: whole ``all`` terrain grid with ANYmal-C
    ./isaaclab.sh -p $SCRIPT

    # Whole terrain grid from a named ``SubTerrainPresetCfg`` preset
    ./isaaclab.sh -p $SCRIPT --terrain eval --robot go2

    # Evaluation mix with Spot, more placements
    ./isaaclab.sh -p $SCRIPT --terrain eval --robot spot --max_robots 500

    # Single sub-terrain (overrides ``--terrain``)
    ./isaaclab.sh -p $SCRIPT --sub_terrain EXTREME_STAIR --robot anymal_c --difficulty 0.9

    # Stepping-stone sub-terrain with Go2 at an easier difficulty
    ./isaaclab.sh -p $SCRIPT --sub_terrain STEPPING_STONE --robot go2 --difficulty 0.5
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse
import builtins
import dataclasses
import socket
import time

import numpy as np
import torch
import trimesh
import warp as wp

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils.warp import convert_to_warp_mesh

VISER_PORT = 8765

# Hip-abduction/adduction joint patterns per robot preset. ``None`` skips the
# HAA-limit criterion for robots where it does not apply.
_HAA_PATTERNS: dict[str, str | None] = {
    "anymal_c": ".*HAA",
    "go2": ".*hip_joint",
    "spot": ".*hip_x",
    # Bipeds have no HAA analogue (no hip-abduction joint that can splay feet
    # out laterally in the way a quadruped's HAA can), so disable the gate.
    "h1": None,
}


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------


def _register_position_task() -> None:
    """Prevent re-registration of gym envs and eagerly import preset modules."""
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    import importlib

    importlib.import_module("isaaclab_tasks.manager_based.multi_task.terrain.terrains")
    importlib.import_module("isaaclab_tasks.manager_based.multi_task.terrain.mdp_presets.robots")


def _sub_terrain_lookup() -> dict[str, SubTerrainBaseCfg]:
    """Return the uppercase ``SubTerrainBaseCfg`` constants defined in terrain_cfg.

    Imports the ``terrain_cfg`` submodule directly (rather than going through the
    ``terrains`` package, which uses :func:`lazy_export` and so does not expose
    names via :func:`dir`).
    """
    import importlib

    tcfg = importlib.import_module("isaaclab_tasks.manager_based.multi_task.terrain.terrains.terrain_cfg")
    return {
        n: getattr(tcfg, n)
        for n in sorted(dir(tcfg))
        if n.isupper() and isinstance(getattr(tcfg, n), SubTerrainBaseCfg)
    }


def _terrain_preset_lookup() -> dict[str, dict]:
    """Return all sub-terrain-dict fields on :class:`SubTerrainPresetCfg`.

    ``SubTerrainPresetCfg`` is a dataclass whose preset fields are set at
    instance level, so we enumerate via :func:`dataclasses.fields` rather than
    :func:`dir` on the class.
    """
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp_presets.terrain_presets import (
        SubTerrainPresetCfg,
    )

    inst = SubTerrainPresetCfg()
    presets: dict[str, dict] = {}
    for f in dataclasses.fields(inst):
        v = getattr(inst, f.name)
        if isinstance(v, dict):
            presets[f.name] = v
    return presets


def _sub_terrain_mesh(sub_cfg: SubTerrainBaseCfg, difficulty: float):
    """Build mesh and origin for a single sub-terrain."""
    cfg = sub_cfg.copy()
    cfg.difficulty = difficulty
    meshes, origin = cfg.function(difficulty, cfg)
    mesh = trimesh.util.concatenate(meshes)
    tf = np.eye(4)
    tf[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(tf)
    origin += tf[0:3, -1]
    return mesh, origin


def _terrain_preset_mesh(preset_dict: dict, device: str):
    """Build a full terrain-grid mesh from a :class:`SubTerrainPresetCfg` preset dict.

    Returns the mesh, the origin (always at ``[0, 0, 0]``), and the inner
    non-border sampling extents ``(x_range, y_range)`` in the mesh frame.
    """
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.terrain_generator import TerrainGenerator

    cfg = TerrainGeneratorCfg(
        size=(10.0, 10.0),
        num_rows=10,
        num_cols=20,
        border_width=20.0,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        seed=42,
        curriculum=True,
        sub_terrains=preset_dict,
    )
    gen = TerrainGenerator(cfg=cfg, device=device)
    inner_x = cfg.num_rows * cfg.size[0] / 2.0
    inner_y = cfg.num_cols * cfg.size[1] / 2.0
    return (
        gen.terrain_mesh,
        np.zeros(3, dtype=np.float32),
        (-inner_x, inner_x),
        (-inner_y, inner_y),
    )


def _resolve_foot_names(robot: str, body_names: list[str]) -> list[str]:
    """Resolve the foot body names for ``robot``.

    Prefers the explicit preset list on
    :class:`RetargetFootBodyNamesCfg` (required for robots whose foot
    bodies don't contain ``"foot"`` in the name, e.g. H1's
    ``*_ankle_link``). Falls back to a case-insensitive substring
    filter so robots without a preset still work.
    """
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp_presets.robots.robot_presets import (
        RetargetFootBodyNamesCfg,
    )

    preset_foot_names = getattr(RetargetFootBodyNamesCfg, robot, None)
    if preset_foot_names is None:
        return [n for n in body_names if "foot" in n.lower()]
    missing = [n for n in preset_foot_names if n not in body_names]
    if missing:
        raise ValueError(
            f"RetargetFootBodyNamesCfg.{robot} names not found in robot USD: missing={missing}, available={body_names}"
        )
    return list(preset_foot_names)


def _draw_active_support_overlay(viewer, buf, feet_all, passed_idx, nc, device) -> None:
    """Draw a thick polygon of only the contact feet per selected placement.

    Distinct from the green reach polygon (which shows all ``nc`` sampled
    positions). The overlay traces the actual support region used by the
    stability criterion: a triangle for tripod placements, a segment for
    biped, a quad for 4-contact.
    """
    is_contact_np = buf.is_contact_t.cpu().numpy().reshape(-1, nc)
    s_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    n_quad = n_tri = n_bi = 0
    for idx in passed_idx:
        feet = feet_all[idx]
        mask = is_contact_np[idx].astype(bool)
        active = feet[mask].copy()
        if len(active) < 2:
            continue
        active[..., 2] += 0.03
        if len(active) == 2:
            s_list.append(active[0:1])
            e_list.append(active[1:2])
            n_bi += 1
            continue
        centroid = active[..., :2].mean(axis=0, keepdims=True)
        delta = active[..., :2] - centroid
        order = np.argsort(np.arctan2(delta[..., 1], delta[..., 0]))
        ordered = active[order]
        s_list.append(ordered)
        e_list.append(np.roll(ordered, -1, axis=0))
        if len(active) == 3:
            n_tri += 1
        else:
            n_quad += 1
    if not s_list:
        return
    s_arr = np.concatenate(s_list, axis=0)
    e_arr = np.concatenate(e_list, axis=0)
    viewer.log_lines(
        "polygons_support_active",
        wp.array(s_arr.tolist(), dtype=wp.vec3, device=device),
        wp.array(e_arr.tolist(), dtype=wp.vec3, device=device),
        colors=(1.0, 0.95, 0.2),
        width=0.012,
    )
    print(f"  Active-contact support polygons (yellow, thick): quad={n_quad} tri={n_tri} biped={n_bi}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import newton
    from newton.viewer import ViewerViser

    from isaaclab_tasks.manager_based.multi_task.mdp.util.kinematics import NewtonKinematicsCfg
    from isaaclab_tasks.manager_based.multi_task.mdp.util.kinematics.ik_objectives.cfg import (
        IKObjectiveGravityTorqueCfg,
        IKObjectiveStabilityMarginCfg,
        IKObjectiveTerrainCollisionCfg,
    )
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
    )
    from isaaclab_tasks.manager_based.multi_task.terrain.mdp.retarget.cfg import (
        PatchSamplingCfg,
        SamplerCfg,
    )
    from isaaclab_tasks.manager_based.multi_task.terrain.utils.criteria_cfg import (
        CollisionCheckCfg,
        FootPositionErrorCfg,
        HaaLimitCfg,
        SolverCostOutlierCfg,
        SupportPolygonStabilityCfg,
    )

    _register_position_task()

    from isaaclab_tasks.manager_based.multi_task.terrain.mdp_presets.robots.robot_presets import (
        RetargetBasePosWeightCfg,
        RetargetBaseRotWeightCfg,
        RetargetGravityWeightCfg,
        RetargetJointRegularizeTargetsCfg,
        RobotArticulationCfg,
    )

    def _preset_for(preset_cls: type, robot: str) -> object:
        """Read a robot's field on a :class:`PresetCfg` subclass, or fall back to default."""
        return getattr(preset_cls, robot, preset_cls().default)

    sub_terrains = _sub_terrain_lookup()
    terrain_presets = _terrain_preset_lookup()

    parser = argparse.ArgumentParser(description="Terrain-conforming spawn validation.")
    parser.add_argument(
        "--terrain",
        type=str,
        default="all",
        choices=sorted(terrain_presets.keys()),
        help="Terrain preset key (builds a whole terrain grid).",
    )
    parser.add_argument(
        "--sub_terrain",
        type=str,
        default=None,
        choices=sorted(sub_terrains.keys()),
        help="Single sub-terrain name. If set, overrides ``--terrain``.",
    )
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument(
        "--robot",
        type=str,
        default="anymal_c",
        choices=sorted(_HAA_PATTERNS.keys()),
        help="Robot preset (resolves USD, base height, joints, HAA pattern).",
    )
    density_group = parser.add_mutually_exclusive_group()
    density_group.add_argument(
        "--max_robots",
        type=int,
        default=None,
        help="Number of final placements. Mutually exclusive with ``--spacing``.",
    )
    density_group.add_argument(
        "--spacing",
        type=float,
        default=None,
        help=(
            "Target minimum distance [m] between final placements. Auto-picks "
            "``max_robots = sampling_area / (3 * spacing**2)`` so the grid-bucket "
            "downsample has enough candidates to actually enforce the spacing "
            "(pure ``area/spacing**2`` is a square-grid upper bound and leaves "
            "the downsample a no-op when criteria yield <~33%% -- the empirical "
            "rate on typical sub-terrains)."
        ),
    )
    parser.add_argument(
        "--min_contacts",
        type=int,
        default=-1,
        help=(
            "Minimum contact slots a candidate must fill to be accepted (see"
            " :attr:`SamplerCfg.min_contacts`). ``-1`` (default) requires"
            " every slot to snap to a patch. A positive ``m`` enables the"
            " soft polygon path: slots without an in-reach patch are marked"
            " ``is_contact = False`` and get a template-projected target."
            " ``m = 2`` lets mixed-geometry terrain emit nc=2/3/4 stances"
            " on a single robot."
        ),
    )
    parser.add_argument(
        "--terrain_snap_distance",
        type=float,
        default=0.15,
        help=(
            "xy distance [m] within which a projected foot snaps to a"
            " morph patch (see :attr:`SamplerCfg.terrain_snap_distance`)."
            " Feet farther than this from any patch are treated as air."
            " ``0.15`` (default) matches typical morph-patch spacing."
            " Bump to ``0.25``-``0.30`` on rough terrain (CONTOUR,"
            " EXTREME_STAIR) where patches cluster on flat regions."
        ),
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Skip the viser viewer; print diagnostics and exit.",
    )
    args = parser.parse_args()
    if args.max_robots is None and args.spacing is None:
        args.max_robots = 300

    device = "cuda:0"

    # --- Robot preset ---
    robot_cfg = getattr(RobotArticulationCfg, args.robot)
    robot_usd = robot_cfg.spawn.usd_path
    base_height = robot_cfg.init_state.pos[2]
    default_jpos = robot_cfg.init_state.joint_pos
    haa_pattern = _HAA_PATTERNS[args.robot]

    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    file_status = check_file_path(robot_usd)
    if file_status == 0:
        raise FileNotFoundError(f"USD not found: {robot_usd}")
    if file_status == 2:
        robot_usd = retrieve_file_path(robot_usd, force_download=False)

    # --- Terrain ---
    sampler_x_range: tuple[float, float] | None = None
    sampler_y_range: tuple[float, float] | None = None
    if args.sub_terrain:
        print(f"Terrain : sub_terrain={args.sub_terrain} (difficulty={args.difficulty})")
        mesh, origin = _sub_terrain_mesh(sub_terrains[args.sub_terrain], args.difficulty)
    else:
        print(f"Terrain : preset={args.terrain}")
        mesh, origin, sampler_x_range, sampler_y_range = _terrain_preset_mesh(terrain_presets[args.terrain], device)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    print(f"  {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # --- Density derivation ---
    # When the user passes ``--spacing``, derive the final robot count from
    # the sampling area. All sampler stage sizes (morph-patches, 4-foot
    # neighborhoods, IK oversample, buffer capacity) are auto-derived inside
    # the sampler from ``n_desired`` via the yield-rate cascade, so we don't
    # have to touch any cfg knobs here.
    if args.spacing is not None:
        if sampler_x_range is not None and sampler_y_range is not None:
            x_lo, x_hi = sampler_x_range
            y_lo, y_hi = sampler_y_range
        else:
            v = np.asarray(mesh.vertices)
            x_lo, x_hi = float(v[:, 0].min()), float(v[:, 0].max())
            y_lo, y_hi = float(v[:, 1].min()), float(v[:, 1].max())
        area = max((x_hi - x_lo) * (y_hi - y_lo), 0.0)
        # Yield factor accounts for (a) criteria rejections (empirically ~33%
        # pass rate on typical sub-terrains) and (b) the grid-bucket
        # downsample's effective cell_side ~= sqrt(area/(1.5*k)) needing to
        # exceed ``spacing`` for the thinning to bite. Factor 1/3 gives a
        # non-cluttered result on meshes where candidates concentrate on easy
        # patches.
        args.max_robots = max(1, int(area / (3.0 * args.spacing**2)))
        print(f"  Spacing mode: area={area:.1f} m^2, spacing={args.spacing:.3f} m -> max_robots={args.max_robots}")

    # --- Robot ---
    print(f"Robot   : {args.robot} ({robot_usd})")

    kin_cfg = NewtonKinematicsCfg(
        usd_path=robot_usd,
        device=device,
        default_pos=(0.0, 0.0, base_height),
        default_joint_pos=default_jpos,
    )

    from isaaclab_tasks.manager_based.multi_task.mdp.util.kinematics import NewtonKinematics

    _tmp = NewtonKinematics(kin_cfg)
    foot_names = _resolve_foot_names(args.robot, _tmp.body_names)
    del _tmp

    # Waterfall order: hard physical constraints first (collision, haa_limit,
    # support-polygon stability) so their buckets report the true rate of
    # physical invalidity. Cost is a residual IK-divergence catch on the
    # physically valid subset. HaaLimitCfg is omitted when haa_pattern is
    # falsy (e.g. bipeds) so the criterion never reaches its validation.
    criteria_list = [
        CollisionCheckCfg(n_samples=16, max_pen=0.02),
    ]
    if haa_pattern:
        criteria_list.append(HaaLimitCfg(joint_pattern=haa_pattern, max_angle=1.05))
    criteria_list += [
        SupportPolygonStabilityCfg(),
        FootPositionErrorCfg(max_err=0.4, aggregate="sum"),
        SolverCostOutlierCfg(threshold_multiplier=3.0),
    ]

    # Per-robot joint-regularize targets pulled from the robot preset class.
    joint_regularize_targets = getattr(RetargetJointRegularizeTargetsCfg, args.robot, {})

    # Per-robot IK weights pulled from preset classes — no CLI overrides.
    # Each robot tunes its own stance-specific values alongside foot names /
    # regularize targets; ablations live in the preset module, not the tool.
    base_rot_weight = _preset_for(RetargetBaseRotWeightCfg, args.robot)
    base_pos_weight = _preset_for(RetargetBasePosWeightCfg, args.robot)
    gravity_weight = _preset_for(RetargetGravityWeightCfg, args.robot)
    print(f"IK wts  : base_rot={base_rot_weight}, base_pos={base_pos_weight}, gravity={gravity_weight}")

    patch_cfg = PatchSamplingCfg(x_range=sampler_x_range, y_range=sampler_y_range)
    # The HAA joint's mechanical limit is tighter than the URDF default
    # (which the USD loader strips to ``±1e10``). Clamp FK sampling for
    # HAA to ±0.7 rad so templates never come from a leg-across-body
    # configuration — without this, IK seeded from the template jq
    # produces visually crossed-leg poses.
    fk_overrides = {haa_pattern: 0.7} if haa_pattern else {}
    sampler_cfg = SamplerCfg(
        patch=patch_cfg,
        min_contacts=args.min_contacts,
        terrain_snap_distance=args.terrain_snap_distance,
        fk_joint_range_overrides=fk_overrides,
        outward_snap_penalty=1.0,
    )
    print(f"Sampler : min_contacts={args.min_contacts} fk_joint_range_overrides={fk_overrides}")

    pipeline_cfg = RetargetPipelineCfg(
        kin=kin_cfg,
        sampler=sampler_cfg,
        foot_body_names=foot_names,
        joint_regularize_targets=joint_regularize_targets,
        base_pos_weight=base_pos_weight,
        base_rot_weight=base_rot_weight,
        extra_objectives=[
            IKObjectiveTerrainCollisionCfg(weight=2.0, margin=0.05, n_samples=4),
            IKObjectiveStabilityMarginCfg(weight=1.0),
            *([IKObjectiveGravityTorqueCfg(weight=gravity_weight)] if gravity_weight > 0.0 else []),
            # IKObjectiveJointRegularizeCfg(weight=0.02),
        ],
        criteria=criteria_list,
    )

    t0 = time.time()
    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    foot_ids = pipeline.foot_body_ids

    geom = kin.foot_geometry(foot_ids)
    foot_offsets = geom["foot_offsets"]
    standing_height = geom["standing_height"]
    print(f"  Bodies={kin.model.body_count} Joints={kin.model.joint_count}")
    print(f"  Feet: {foot_names} -> ids {foot_ids}")
    foot_spread = np.linalg.norm(foot_offsets[:, :2].max(axis=0) - foot_offsets[:, :2].min(axis=0))
    print(f"  standing_height={standing_height:.3f}m foot_spread={foot_spread:.3f}m")

    print("\n--- Running retarget pipeline ---")

    buf = pipeline.run(wp_mesh, origin, n_desired=args.max_robots)
    t_total = time.time() - t0

    # Rejection summary already includes per-criterion counts and timings.
    print(pipeline.rejection_summary)
    print(f"  Wall time: {t_total:.2f}s")

    if buf.num_selected == 0:
        print("No valid candidates. Exiting.")
        sys.exit(1)

    # Selected placements' contact-count histogram.
    sel_idx = buf._selected[: buf.num_selected].to(torch.long)
    nc = len(foot_ids)
    is_c_sel = buf.is_contact_t.view(-1, nc)[sel_idx]
    n_active_sel = is_c_sel.to(torch.int32).sum(dim=-1).cpu().tolist()
    nc_hist = {}
    for k in n_active_sel:
        nc_hist[int(k)] = nc_hist.get(int(k), 0) + 1
    print(f"  Selected placement contact-count histogram: {dict(sorted(nc_hist.items()))}")

    if args.no_viewer:
        print("\n--no_viewer: exiting without visualization.")
        return

    # --- Visualization ---
    print("\n--- Visualization ---")
    sel = buf._selected[: buf.num_selected].cpu().numpy()
    jq_results = buf.joint_q_result_t.cpu().numpy()
    ct_np = buf.contact_targets_t.cpu().numpy()
    nc = len(foot_ids)
    cpc = kin.model.joint_coord_count

    solved_qs = [jq_results[idx] for idx in sel]
    selected_feet = [ct_np[idx * nc : (idx + 1) * nc] for idx in sel]

    vis_builder = newton.ModelBuilder()
    vis_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            vertices=mesh.vertices.flatten().tolist(),
            indices=mesh.faces.flatten().tolist(),
        ),
        scale=wp.vec3(1, 1, 1),
    )

    if solved_qs:
        template = newton.ModelBuilder()
        template.add_usd(robot_usd, collapse_fixed_joints=False)
        for _ in solved_qs:
            vis_builder.add_world(template)

    vis_model = vis_builder.finalize(device=device)
    if solved_qs:
        vis_jq = vis_model.joint_q.numpy().copy()
        for i, sq in enumerate(solved_qs):
            vis_jq[i * cpc : (i + 1) * cpc] = sq
        vis_model.joint_q = wp.array(vis_jq, dtype=float, device=device)

    vis_state = vis_model.state()
    newton.eval_fk(
        vis_model,
        vis_model.joint_q,
        wp.zeros(vis_model.joint_dof_count, dtype=float, device=device),
        vis_state,
    )

    viewer = ViewerViser(port=VISER_PORT)
    viewer.set_model(vis_model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))

    from isaaclab_tasks.manager_based.multi_task.terrain.terrains.utils.patch_sampling_cfg import (
        CircleFootprintCfg,
        MorphologicalPatchSamplingCfg,
    )

    # Conservative candidate density (~10 patches/m^2) so even low-valid-fraction
    # terrains (e.g. FLOATING_ISLAND at ~26% valid cells) comfortably satisfy
    # the request and no fallback is needed.
    if sampler_x_range is not None and sampler_y_range is not None:
        vis_area = (sampler_x_range[1] - sampler_x_range[0]) * (sampler_y_range[1] - sampler_y_range[0])
    else:
        v_xy = np.asarray(mesh.vertices)
        vis_area = float((v_xy[:, 0].max() - v_xy[:, 0].min()) * (v_xy[:, 1].max() - v_xy[:, 1].min()))
    vis_num_patches = max(100, min(10000, int(vis_area * 10)))

    fc_cfg = MorphologicalPatchSamplingCfg(
        num_patches=vis_num_patches,
        footprint=CircleFootprintCfg(radius=0.04),
        max_height_diff=0.03,
        horizontal_scale=0.03,
        oversample_ratio=3.0,
        x_range=sampler_x_range if sampler_x_range is not None else (-1e6, 1e6),
        y_range=sampler_y_range if sampler_y_range is not None else (-1e6, 1e6),
    )

    fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
    origin_t = torch.tensor(origin, dtype=torch.float, device=fp.device)
    fp[:, :3] += origin_t
    foot_pts = fp[:, :3].cpu().numpy()
    fl = foot_pts.copy()
    fl[:, 2] += 0.02
    viewer.log_points(
        "foot_candidates",
        wp.array(fl.tolist(), dtype=wp.vec3, device=device),
        radii=0.01,
        colors=(0.15, 0.5, 0.15),
    )

    if selected_feet:
        sf = np.concatenate(selected_feet)
        sf_l = sf.copy()
        if sf_l.ndim == 2:
            sf_l[:, 2] += 0.02
        viewer.log_points(
            "selected_feet",
            wp.array(sf_l.tolist(), dtype=wp.vec3, device=device),
            radii=0.03,
            colors=(0, 1, 1),
        )

    # Support-polygon edges, colored by pipeline outcome (sampler out_of_reach
    # rejections never reach this stage, so they aren't drawn):
    #   red    -- passed sampler, failed IK criteria (collision / haa_limit / stability / cost)
    #   orange -- passed criteria but dropped by final FPS downsample
    #   green  -- selected (kept)
    n_written = buf.num_written
    if n_written > 0:
        geom_valid_np = buf._geom_valid[:n_written].cpu().numpy()
        ik_valid_np = buf._ik_valid[:n_written].cpu().numpy()
        selected_mask = np.zeros(n_written, dtype=bool)
        selected_mask[sel] = True
        rejected_idx = np.nonzero(geom_valid_np & ~ik_valid_np)[0]
        filtered_idx = np.nonzero(ik_valid_np & ~selected_mask)[0]
        passed_idx = sel

        feet_all = ct_np.reshape(-1, nc, 3)  # [max_candidates, nc, 3]

        def _polygon_edges(indices: np.ndarray, lift_z: float = 0.015):
            """Order each polygon's feet by angle around its centroid and emit edges."""
            if indices.size == 0:
                return None, None
            feet = feet_all[indices].copy()  # [M, nc, 3]
            feet[..., 2] += lift_z
            centroid_xy = feet[:, :, :2].mean(axis=1, keepdims=True)
            delta = feet[:, :, :2] - centroid_xy
            angles = np.arctan2(delta[..., 1], delta[..., 0])
            order = np.argsort(angles, axis=1)
            ordered = np.take_along_axis(feet, order[:, :, None].repeat(3, axis=2), axis=1)
            starts = ordered.reshape(-1, 3)
            ends = np.roll(ordered, -1, axis=1).reshape(-1, 3)
            return starts, ends

        for indices, color, name, lift in [
            (rejected_idx, (1.0, 0.15, 0.15), "polygons_rejected", 0.01),
            (filtered_idx, (1.0, 0.55, 0.0), "polygons_bucket_filtered", 0.015),
            (passed_idx, (0.1, 0.9, 0.2), "polygons_selected", 0.02),
        ]:
            starts, ends = _polygon_edges(indices, lift_z=lift)
            if starts is None:
                continue
            viewer.log_lines(
                name,
                wp.array(starts.tolist(), dtype=wp.vec3, device=device),
                wp.array(ends.tolist(), dtype=wp.vec3, device=device),
                colors=color,
                width=0.004,
            )
        print(
            f"  Support polygons: {len(rejected_idx)} criteria-rejected (red),"
            f" {len(filtered_idx)} bucket-filtered (orange),"
            f" {len(passed_idx)} selected (green)"
        )

        # Active-support polygon overlay: only the *contact* feet per
        # selected placement. For 4-contact this overlaps green; for
        # tripod it's a triangle inside the quad; for biped it's a
        # segment. This is the polygon SupportPolygonStability tests
        # CoM against.
        _draw_active_support_overlay(viewer, buf, feet_all, passed_idx, nc, device)

    # Visualize collision probe points on ALL robots
    from isaaclab_tasks.manager_based.multi_task.mdp.util.kinematics import _build_collision_probes

    probe_bodies, probe_offsets, _probe_slots = _build_collision_probes(kin.builder, foot_ids, n_samples=16)
    if solved_qs:
        from isaaclab.utils.math import quat_apply as _qa

        all_probe_world = []
        for sq in solved_qs:
            fk_jq = wp.array(sq, dtype=float, device=device)
            fk_state = kin.eval_fk(fk_jq)
            bq_np = fk_state.body_q.numpy()
            for i in range(len(probe_bodies)):
                bid = probe_bodies[i]
                off = np.array(probe_offsets[i], dtype=np.float32)
                bq = bq_np[bid]
                pos = bq[:3]
                quat = bq[3:7]
                q_t = torch.tensor(quat, dtype=torch.float32).unsqueeze(0)
                o_t = torch.tensor(off, dtype=torch.float32).unsqueeze(0)
                rotated = _qa(q_t, o_t).squeeze(0).numpy()
                all_probe_world.append((pos + rotated).tolist())
        viewer.log_points(
            "collision_probes",
            wp.array(all_probe_world, dtype=wp.vec3, device=device),
            radii=0.005,
            colors=(1.0, 0.2, 0.2),
        )
        print(f"  Collision probes: {len(all_probe_world)} points on {len(solved_qs)} robots (red)")

    viewer.begin_frame(0.0)
    viewer.log_state(vis_state)
    viewer.end_frame()

    hn = socket.gethostname()
    print(f"\n  http://localhost:{VISER_PORT}")
    print(f"  http://{hn}:{VISER_PORT}")
    print("  Dots: Green=candidates, Cyan=selected feet, Red=collision probes")
    print("  Polygons: Red=criteria-rejected, Orange=bucket-filtered, Green=selected")
    print(f"\n  {len(solved_qs)} robots placed.")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while viewer.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    wp.init()
    main()
