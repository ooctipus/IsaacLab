# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampler metrics harness for regression tracking.

Runs a :class:`RetargetPipeline` against a fixed robot x sub-terrain grid
and emits structured metrics covering:

* **Yield rates** -- per-stage waterfall fractions (sampler, IK, criteria,
  final).
* **Shape diversity** -- how broadly the accepted polygon set covers the
  FK canonical-shape manifold.
* **Spatial diversity** -- how broadly the accepted placements cover the
  terrain (centroid xy variance, yaw entropy).
* **Cost** -- wall time.

Backs a committed JSON baseline that the regression test compares against
to catch accidental behavior drift in :class:`Sampler`.

Invoke as a CLI to regenerate the baseline JSON fixture (paths relative
to the repository root)::

    ./isaaclab.sh -p \\
        source/.../position/utils/tools/sampler_metrics.py \\
        --output source/.../position/tests/data/sampler_baseline.json

Programmatic use from a test::

    metrics = run_metrics_grid(
        robots=["anymal_c"],
        sub_terrains=["FLAT"],
        difficulties=[0.5],
        n_desired=50,
        seed=0,
    )
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SamplerMetrics:
    """Metrics captured for a single (robot, sub_terrain, difficulty) run.

    Fields:
        robot: Robot preset name.
        sub_terrain: Sub-terrain name (uppercase constant in ``terrain_cfg``).
        difficulty: Difficulty value in ``[0, 1]``.
        n_desired: Target placement count passed to the pipeline.
        n_proposed: Polygons proposed by the sampler
            (pre-rejection; sampler_yield denominator).
        n_sampler_accepted: Polygons accepted by the sampler
            (pre-IK; sampler_yield numerator). Each accepted polygon is
            one IK problem.
        n_ik_problems: IK problems actually solved; equals
            ``n_sampler_accepted``.
        n_ik_passed: Polygons whose IK solution passed all criteria.
        n_final: Final placements after spatial FPS (``<= n_desired``).
        sampler_yield: ``n_sampler_accepted / n_proposed``.
        ik_yield: ``n_ik_passed / n_sampler_accepted`` -- polygon-level
            "does this candidate survive IK + criteria?".
        criteria_yield: alias for ``ik_yield`` (same gate).
        final_yield: ``n_final / n_desired`` (ratio achieved to requested).
        shape_pairwise_mean: Mean L2 distance between centered accepted
            polygons (proxy for shape diversity; higher = more diverse).
        shape_pairwise_p50: Median pairwise L2 distance.
        shape_pairwise_p95: 95th percentile pairwise L2 distance.
        centroid_xy_var: Trace of the xy covariance of accepted-polygon
            centroids (spatial spread).
        yaw_entropy: Shannon entropy (natural log) of the accepted-polygon
            yaw distribution, binned into 16 bins over [0, 2*pi).
        sampler_wall_time_s: Wall time of the sampler phase [s].
        total_wall_time_s: Wall time of the full pipeline run [s].
        reject_geo: Geometry-stage rejection waterfall (sampler-side).
        reject_val: Validation-stage rejection waterfall (criteria-side).
        min_contacts: Sampler ``min_contacts`` knob at run time (``-1``
            means the legacy hard-polygon path).
        nc_histogram: Mapping ``active_contact_count -> count`` over the
            final selected placements, derived from ``is_contact_t``. On
            the hard-polygon path this is just ``{nc: n_final}``; on the
            soft polygon path it exposes the nc=2/3/4 mixture.
    """

    robot: str
    sub_terrain: str
    difficulty: float
    n_desired: int
    n_proposed: int
    n_sampler_accepted: int
    n_ik_problems: int
    n_ik_passed: int
    n_final: int
    sampler_yield: float
    ik_yield: float
    criteria_yield: float
    final_yield: float
    shape_pairwise_mean: float
    shape_pairwise_p50: float
    shape_pairwise_p95: float
    centroid_xy_var: float
    yaw_entropy: float
    sampler_wall_time_s: float
    total_wall_time_s: float
    reject_geo: dict[str, int]
    reject_val: dict[str, int]
    min_contacts: int = -1
    nc_histogram: dict[int, int] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _register_presets() -> None:
    import builtins
    import importlib

    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.terrains")
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots")


def _resolve_usd(usd_path: str) -> str:
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    status = check_file_path(usd_path)
    if status == 0:
        raise FileNotFoundError(f"USD not found: {usd_path}")
    if status == 2:
        return retrieve_file_path(usd_path, force_download=False)
    return usd_path


def _sub_terrain_mesh(sub_terrain_name: str, difficulty: float):
    import importlib

    import trimesh

    tcfg = importlib.import_module("isaaclab_tasks.core.multi_task.terrain.terrains.terrain_cfg")
    sub_cfg = getattr(tcfg, sub_terrain_name)
    cfg = sub_cfg.copy()
    cfg.difficulty = difficulty
    meshes, origin = cfg.function(difficulty, cfg)
    mesh = trimesh.util.concatenate(meshes)
    tf = np.eye(4)
    tf[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(tf)
    origin = np.asarray(origin, dtype=np.float32) + tf[0:3, -1].astype(np.float32)
    return mesh, origin


def _build_pipeline_cfg(robot_name: str, min_contacts: int = -1):
    """Build a minimal ``RetargetPipelineCfg`` for ``robot_name``.

    Uses the same objectives/criteria as :mod:`validate_spawn_points` so
    the metrics reflect the realistic production pipeline.

    Args:
        robot_name: Robot preset name.
        min_contacts: Sampler ``min_contacts`` knob. ``-1`` (default)
            preserves hard-polygon behavior; positive values enable the
            soft polygon path (mixed-nc placements).
    """
    from isaaclab_tasks.core.multi_task.kinematics import (
        NewtonKinematics,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveJointRegularizeCfg,
        IKObjectiveStabilityMarginCfg,
        IKObjectiveTerrainCollisionCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots.robot_presets import (
        FootBodyNamesCfg,
        RetargetJointRegularizeTargetsCfg,
        RetargetLateralHipJointPatternCfg,
        RobotArticulationCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget import (
        RetargetPipelineCfg,
        resolve_foot_body_names,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import (
        PatchSamplingCfg,
        SamplerCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import (
        CollisionCheckCfg,
        FootPositionErrorCfg,
        JointWithinLimitCfg,
        LateralHipLimitCfg,
        SolverCostOutlierCfg,
        SupportPolygonStabilityCfg,
    )

    robot_cfg = getattr(RobotArticulationCfg, robot_name)
    robot_usd = _resolve_usd(robot_cfg.spawn.usd_path)

    kin_cfg = NewtonKinematicsCfg(
        usd_path=robot_usd,
        device="cuda:0",
        default_pos=(0.0, 0.0, robot_cfg.init_state.pos[2]),
        default_joint_pos=robot_cfg.init_state.joint_pos,
    )
    _tmp = NewtonKinematics(kin_cfg)
    foot_spec = getattr(FootBodyNamesCfg, robot_name, ".*foot")
    foot_names = resolve_foot_body_names(foot_spec, _tmp.body_names)
    del _tmp

    lateral_hip_pattern = getattr(
        RetargetLateralHipJointPatternCfg, robot_name, RetargetLateralHipJointPatternCfg().default
    )
    criteria_list: list[Any] = [CollisionCheckCfg(n_samples=16, max_pen=0.02), JointWithinLimitCfg(limit_ratio=0.9)]
    if lateral_hip_pattern:
        criteria_list.append(LateralHipLimitCfg(joint_pattern=lateral_hip_pattern, max_angle=1.05))
    criteria_list += [
        SupportPolygonStabilityCfg(),
        FootPositionErrorCfg(max_err=0.25, aggregate="sum"),
        SolverCostOutlierCfg(threshold_multiplier=3.0),
    ]

    joint_regularize_targets = getattr(RetargetJointRegularizeTargetsCfg, robot_name, {})

    sampler_cfg = SamplerCfg(
        patch=PatchSamplingCfg(),
        min_contacts=min_contacts,
    )

    return RetargetPipelineCfg(
        kin=kin_cfg,
        sampler=sampler_cfg,
        foot_body_names=foot_names,
        lateral_hip_joint_pattern=lateral_hip_pattern,
        joint_regularize_targets=joint_regularize_targets,
        extra_objectives=[
            IKObjectiveTerrainCollisionCfg(weight=3.0, margin=0.05, n_samples=4),
            IKObjectiveStabilityMarginCfg(weight=1.0),
            IKObjectiveJointRegularizeCfg(weight=0.02),
        ],
        criteria=criteria_list,
    )


def _shape_pairwise_stats(
    contact_targets: torch.Tensor, n: int, nc: int, max_pairs: int = 2000
) -> tuple[float, float, float]:
    """Return (mean, p50, p95) pairwise L2 distance between centered polygons.

    Centers each polygon to its xy centroid (z preserved) so the metric
    captures *shape* variation independent of placement. Subsamples
    ``max_pairs`` pairs to keep the cost O(max_pairs * nc * 3) regardless
    of ``n``.
    """
    if n < 2:
        return 0.0, 0.0, 0.0
    ct = contact_targets[: n * nc].view(n, nc, 3).clone()
    centroid_xy = ct[:, :, :2].mean(dim=1, keepdim=True)
    ct[:, :, :2] -= centroid_xy
    # Sample pairs without replacement, with a budget cap.
    rng = torch.Generator(device=ct.device).manual_seed(0)
    target = min(max_pairs, n * (n - 1) // 2)
    # Sample with replacement and drop i==j; simpler and fine for a metric.
    i = torch.randint(0, n, (target * 2,), generator=rng, device=ct.device)
    j = torch.randint(0, n, (target * 2,), generator=rng, device=ct.device)
    mask = i != j
    i = i[mask][:target]
    j = j[mask][:target]
    diff = ct[i] - ct[j]
    dist = diff.reshape(diff.shape[0], -1).norm(dim=-1)
    return (
        float(dist.mean()),
        float(dist.quantile(0.5)),
        float(dist.quantile(0.95)),
    )


def _yaw_entropy(joint_q_result: torch.Tensor, n: int, bins: int = 16) -> float:
    """Shannon entropy of the yaw distribution (natural log)."""
    if n == 0:
        return 0.0
    # Newton floating base: joint_q[0:3] = pos, [3:7] = quat (xyzw).
    q = joint_q_result[:n, 3:7]
    # yaw from quaternion (x, y, z, w). atan2(2*(wz+xy), 1-2*(y^2+z^2)).
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    yaw = torch.where(yaw < 0, yaw + 2.0 * float(np.pi), yaw)
    hist = torch.histc(yaw, bins=bins, min=0.0, max=float(2.0 * np.pi))
    p = hist / hist.sum().clamp(min=1.0)
    # Drop zeros to avoid log(0).
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def _nc_histogram(is_contact: torch.Tensor, selected_idx: torch.Tensor, nc: int) -> dict[int, int]:
    """Per-candidate active-contact histogram over the selected placements.

    ``is_contact`` is the flat buffer view ``[max_candidates * nc]``;
    ``selected_idx`` is the subset of row indices post-criteria FPS.
    Returns ``{k: count}`` for each ``k`` in ``[1, nc]`` that occurs at
    least once, with unoccupied counts omitted.
    """
    if selected_idx.numel() == 0:
        return {}
    rows = is_contact.view(-1, nc)[selected_idx]  # [n_selected, nc]
    per = rows.to(torch.int32).sum(dim=-1).cpu().tolist()
    hist: dict[int, int] = {}
    for k in per:
        hist[int(k)] = hist.get(int(k), 0) + 1
    return hist


def _centroid_xy_var(joint_q_result: torch.Tensor, n: int) -> float:
    """Trace of the xy covariance of accepted placements (from IK result base pos)."""
    if n < 2:
        return 0.0
    xy = joint_q_result[:n, 0:2]
    mean = xy.mean(dim=0, keepdim=True)
    centered = xy - mean
    return float((centered * centered).sum() / max(n - 1, 1))


def _run_single(
    robot_name: str,
    sub_terrain: str,
    difficulty: float,
    n_desired: int,
    seed: int,
    min_contacts: int = -1,
) -> SamplerMetrics:
    """Run the pipeline once and collect metrics."""

    from isaaclab.utils.warp import convert_to_warp_mesh

    from isaaclab_tasks.core.multi_task.terrain.retarget import RetargetPipeline, apply_final_fps

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    pipeline_cfg = _build_pipeline_cfg(robot_name, min_contacts)
    pipeline = RetargetPipeline(pipeline_cfg)

    mesh, origin = _sub_terrain_mesh(sub_terrain, difficulty)
    wp_mesh = convert_to_warp_mesh(mesh.vertices.astype(np.float32), mesh.faces, device="cuda:0")

    # Re-seed immediately before run() so harness RNG noise (preset build,
    # kinematics init) cannot perturb the sampler's draws.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    t0 = time.perf_counter()
    buf = pipeline.run(wp_mesh, origin, n_desired=n_desired)
    sizing = pipeline_cfg.sampler.sizing
    apply_final_fps(
        buf,
        n_desired=n_desired,
        extractor=getattr(sizing, "fps_features", None),
        spacing=getattr(sizing, "fps_spacing", None),
    )
    total_s = time.perf_counter() - t0

    sampler_s = float(pipeline._timings.get("sampler", 0.0))
    reject_geo = dict(pipeline._reject_geo)
    reject_val = dict(pipeline._reject_val)

    # Waterfall bookkeeping at polygon granularity: each accepted polygon
    # is one IK problem, so ``n_sampler_accepted`` and ``n_ik_problems``
    # coincide.
    n_ik_problems = int(pipeline._n_ik_problems)
    n_sampler_accepted = n_ik_problems
    n_proposed = n_sampler_accepted + sum(reject_geo.values())
    n_ik_passed = int(buf.num_ik_valid) if buf is not None else 0
    n_final = int(buf.num_selected) if buf is not None else 0

    sampler_yield = n_sampler_accepted / max(n_proposed, 1)
    ik_yield = n_ik_passed / max(n_sampler_accepted, 1)
    final_yield = n_final / max(n_desired, 1)

    # Shape + spatial diversity from the buffer's post-collapse rows.
    nc = buf.num_contacts
    shape_mean, shape_p50, shape_p95 = _shape_pairwise_stats(buf.contact_targets_t, buf.num_written, nc)
    yaw_entropy = _yaw_entropy(buf.joint_q_result_t, buf.num_ik_valid)
    centroid_var = _centroid_xy_var(buf.joint_q_result_t, buf.num_ik_valid)

    selected_idx = buf._selected[: buf.num_selected].to(torch.long)
    nc_hist = _nc_histogram(buf.is_contact_t, selected_idx, nc)

    # CUDA cleanup to keep sweeps from accumulating allocations across runs.
    del pipeline
    del buf
    del wp_mesh
    del mesh
    torch.cuda.empty_cache()

    return SamplerMetrics(
        robot=robot_name,
        sub_terrain=sub_terrain,
        difficulty=difficulty,
        n_desired=n_desired,
        n_proposed=n_proposed,
        n_sampler_accepted=n_sampler_accepted,
        n_ik_problems=n_ik_problems,
        n_ik_passed=n_ik_passed,
        n_final=n_final,
        sampler_yield=sampler_yield,
        ik_yield=ik_yield,
        criteria_yield=ik_yield,
        final_yield=final_yield,
        shape_pairwise_mean=shape_mean,
        shape_pairwise_p50=shape_p50,
        shape_pairwise_p95=shape_p95,
        centroid_xy_var=centroid_var,
        yaw_entropy=yaw_entropy,
        sampler_wall_time_s=sampler_s,
        total_wall_time_s=total_s,
        reject_geo=reject_geo,
        reject_val=reject_val,
        min_contacts=min_contacts,
        nc_histogram=nc_hist,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_metrics_grid(
    robots: list[str],
    sub_terrains: list[str],
    difficulties: list[float],
    n_desired: int = 300,
    seed: int = 0,
    min_contacts: int = -1,
) -> list[SamplerMetrics]:
    """Run the sampler metrics harness across a robot x terrain x difficulty grid.

    Args:
        robots: Robot preset names (e.g. ``["anymal_c", "go2"]``).
        sub_terrains: Sub-terrain constant names (e.g. ``["FLAT",
            "EXTREME_STAIR"]``). Must match uppercase names in
            ``position/terrains/terrain_cfg.py``.
        difficulties: Difficulty values in ``[0, 1]``.
        n_desired: Target placement count per cell.
        seed: RNG seed applied before each cell runs; harness is
            deterministic when the underlying ops are.
        min_contacts: Sampler ``min_contacts`` knob. ``-1`` (default)
            preserves the hard-polygon path; positive values exercise the
            soft polygon path on every cell and populate each
            :class:`SamplerMetrics` with a mixed-nc histogram.

    Returns:
        List of :class:`SamplerMetrics`, one per grid cell, in row-major
        order (robots outermost, difficulties innermost).
    """
    _register_presets()
    results: list[SamplerMetrics] = []
    for r in robots:
        for t in sub_terrains:
            for d in difficulties:
                m = _run_single(r, t, d, n_desired, seed, min_contacts)
                results.append(m)
                hist_str = ""
                if m.nc_histogram:
                    hist_str = " nc=" + ",".join(f"{k}:{v}" for k, v in sorted(m.nc_histogram.items()))
                print(
                    f"[{r} / {t} / diff={d}] "
                    f"yields sampler={m.sampler_yield:.2f} ik={m.ik_yield:.2f} "
                    f"final={m.n_final}/{m.n_desired}  "
                    f"shape_p50={m.shape_pairwise_p50:.3f}  wall={m.total_wall_time_s:.2f}s"
                    f"{hist_str}",
                    flush=True,
                )
    return results


def dump_metrics_json(results: list[SamplerMetrics], path: Path) -> None:
    """Write metrics results as a JSON file for regression comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "results": [r.to_dict() for r in results]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_grid() -> tuple[list[str], list[str], list[float]]:
    return (
        ["anymal_c", "go2", "spot"],
        ["FLAT", "EXTREME_STAIR", "STEPPING_STONE"],
        [0.5, 0.8],
    )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sampler metrics harness.")
    default_robots, default_terrains, default_difficulties = _default_grid()
    parser.add_argument("--robots", nargs="+", default=default_robots)
    parser.add_argument("--sub_terrains", nargs="+", default=default_terrains)
    parser.add_argument("--difficulties", nargs="+", type=float, default=default_difficulties)
    parser.add_argument("--n_desired", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, dump results to this JSON path (for baseline fixture).",
    )
    parser.add_argument(
        "--min_contacts",
        type=int,
        default=-1,
        help=(
            "Sampler min_contacts knob. ``-1`` (default) preserves the"
            " legacy hard-polygon path; positive values exercise the"
            " soft polygon path and fill nc_histogram with the"
            " nc=2/3/4 mixture."
        ),
    )
    args = parser.parse_args(argv)

    results = run_metrics_grid(
        robots=args.robots,
        sub_terrains=args.sub_terrains,
        difficulties=args.difficulties,
        n_desired=args.n_desired,
        seed=args.seed,
        min_contacts=args.min_contacts,
    )
    if args.output is not None:
        dump_metrics_json(results, args.output)
        print(f"\nBaseline written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]
    raise SystemExit(_cli())
