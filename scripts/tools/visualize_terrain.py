# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize terrain meshes and sampled flat patches as interactive HTML.

No Isaac Sim / Omniverse launch required. The output is an HTML file using
trimesh's three.js viewer, ideal for inspection over SSH.

Usage::

    # Visualize the RADIATING_BEAM preset at difficulty 0.5
    ./isaaclab.sh -p scripts/tools/visualize_terrain.py --terrain RADIATING_BEAM

    # Higher difficulty, custom output path
    ./isaaclab.sh -p scripts/tools/visualize_terrain.py --terrain RADIATING_BEAM --difficulty 0.8 -o /tmp/beam.html

    # Skip flat-patch sampling (avoids warp/CUDA dependency)
    ./isaaclab.sh -p scripts/tools/visualize_terrain.py --terrain GAP --no-patches

    # List all available terrain presets
    ./isaaclab.sh -p scripts/tools/visualize_terrain.py --list

Legend (sphere colors):
    - Green  = spawn patches
    - Red    = target patches
    - Yellow = terrain origin
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


def _load_terrain_module():
    """Import the position/terrains terrain_cfg module directly.

    ``isaaclab_tasks.__init__`` eagerly walks *every* subpackage via
    ``import_packages``, which can hit unrelated import errors.  We short-
    circuit that walk by setting the guard flag, then import only the
    specific package we need.
    """
    import builtins
    import importlib

    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]

    pkg = importlib.import_module("isaaclab_tasks.manager_based.multi_task.terrain.terrains")
    # lazy_export populates sys.modules with the real submodule
    return sys.modules.get(
        "isaaclab_tasks.manager_based.multi_task.terrain.terrains.terrain_cfg",
        pkg,
    )


def _list_presets(module) -> dict[str, SubTerrainBaseCfg]:
    """Return {NAME: cfg} for every UPPER_CASE SubTerrainBaseCfg in *module*."""
    return {
        name: getattr(module, name)
        for name in sorted(dir(module))
        if name.isupper() and isinstance(getattr(module, name), SubTerrainBaseCfg)
    }


def _generate_mesh(cfg, difficulty: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Call the terrain generation function and center the resulting mesh."""
    cfg = cfg.copy()
    cfg.difficulty = difficulty
    meshes, origin = cfg.function(difficulty, cfg)
    mesh = trimesh.util.concatenate(meshes)
    transform = np.eye(4)
    transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(transform)
    origin += transform[0:3, -1]
    return mesh, origin


def _reconstruct_patch_cfg(mangled_cfg):
    """Undo the PatchSamplingCfg.__post_init__ mangling and return a clean config."""
    info = mangled_cfg.patch_radius  # dict produced by __post_init__
    cfg_class = info["cfg"]
    params = {k: v for k, v in info.items() if k not in ("func", "cfg")}
    params["patched"] = True
    return cfg_class(**params)


def _sample_patches(mesh: trimesh.Trimesh, origin: np.ndarray, flat_patch_sampling: dict) -> dict[str, np.ndarray]:
    """Sample flat patches for each key in *flat_patch_sampling*."""
    import torch
    import warp as wp

    from isaaclab.utils.warp import convert_to_warp_mesh

    wp.init()
    device = "cpu"
    if torch.cuda.is_available():
        try:
            wp_test = convert_to_warp_mesh(np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]]), device="cuda:0")
            device = "cuda:0"
            del wp_test
        except Exception:
            pass
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    results: dict[str, np.ndarray] = {}
    for name, mangled_cfg in flat_patch_sampling.items():
        clean_cfg = _reconstruct_patch_cfg(mangled_cfg)
        patches = clean_cfg.func(wp_mesh, origin, clean_cfg)
        origin_t = torch.tensor(origin, dtype=torch.float, device=patches.device)
        patches[:, :3] += origin_t
        results[name] = patches[:, :3].cpu().numpy()
    return results


def _color_mesh_by_height(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Color mesh vertices by height using a trimesh-built-in colormap."""
    heights = mesh.vertices[:, 2]
    if np.ptp(heights) < 1e-6:
        mesh.visual.vertex_colors = np.full((len(heights), 4), [172, 216, 230, 200], dtype=np.uint8)
    else:
        norm = (heights - heights.min()) / np.ptp(heights)
        norm = np.clip(norm, 0.1, 0.9)
        try:
            mesh.visual.vertex_colors = trimesh.visual.color.interpolate(norm, color_map="viridis")
        except Exception:
            mesh.visual.vertex_colors = np.full((len(heights), 4), [172, 216, 230, 200], dtype=np.uint8)
    return mesh


_PATCH_PALETTE = {
    "spawn": ([50, 205, 50, 255], 0.06),
    "target": ([220, 50, 50, 255], 0.10),
}


def _add_patch_markers(
    scene: trimesh.Scene,
    patches_dict: dict[str, np.ndarray],
    origin: np.ndarray,
    suffix: str = "",
) -> None:
    """Add colored spheres for patches and a yellow origin marker to *scene*."""
    for name, pts in patches_dict.items():
        color, radius = _PATCH_PALETTE.get(name, ([50, 50, 220, 255], 0.08))
        for i, pt in enumerate(pts):
            sphere = trimesh.creation.uv_sphere(radius=radius, count=[8, 8])
            sphere.apply_translation(pt)
            sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
            scene.add_geometry(sphere, geom_name=f"{name}_{i}{suffix}")

    origin_marker = trimesh.creation.uv_sphere(radius=0.15, count=[8, 8])
    origin_marker.apply_translation(origin)
    origin_marker.visual.vertex_colors = np.tile([255, 200, 0, 255], (len(origin_marker.vertices), 1))
    scene.add_geometry(origin_marker, geom_name=f"origin{suffix}")


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — Terrain Viewer</title>
<style>
  body {{ margin:0; background:#1a1a2e; color:#eee; font-family:system-ui,sans-serif; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; font-size:13px;
           background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:6px; }}
  #info b {{ color:#fff; }}
  .legend {{ display:flex; gap:12px; margin-top:6px; }}
  .legend span {{ display:flex; align-items:center; gap:4px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  model-viewer {{ width:100vw; height:100vh; }}
</style>
</head>
<body>
<div id="info">
  <b>{title}</b> &mdash; difficulty {difficulty_str}<br>
  mesh: {num_verts:,} verts, {num_faces:,} faces
  <div class="legend">
    <span><span class="dot" style="background:#32cd32"></span> spawn ({n_spawn})</span>
    <span><span class="dot" style="background:#dc3232"></span> target ({n_target})</span>
    <span><span class="dot" style="background:#ffc800"></span> origin</span>
  </div>
</div>
<model-viewer
  src="data:model/gltf-binary;base64,{glb_b64}"
  camera-controls touch-action="pan-y"
  auto-rotate shadow-intensity="0.4"
  style="width:100vw;height:100vh"
></model-viewer>
<script type="module"
  src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js">
</script>
</body>
</html>
"""


def _export_html(
    scene: trimesh.Scene,
    path: str,
    *,
    title: str = "terrain",
    difficulty: float | list[float] = 0.5,
    patches_dict: dict[str, np.ndarray] | None = None,
) -> None:
    """Export a trimesh Scene to a self-contained HTML file with embedded GLB."""
    glb_bytes: bytes = scene.export(file_type="glb")  # type: ignore[assignment]
    glb_b64 = base64.b64encode(glb_bytes).decode("ascii")

    patches_dict = patches_dict or {}
    total_verts = sum(len(g.vertices) for g in scene.geometry.values())
    total_faces = sum(len(g.faces) for g in scene.geometry.values() if hasattr(g, "faces"))

    if isinstance(difficulty, list):
        diff_str = ", ".join(f"{d:.2f}" for d in difficulty)
    else:
        diff_str = f"{difficulty:.2f}"

    html = _HTML_TEMPLATE.format(
        title=title,
        difficulty_str=diff_str,
        num_verts=total_verts,
        num_faces=total_faces,
        n_spawn=len(patches_dict.get("spawn", [])),
        n_target=len(patches_dict.get("target", [])),
        glb_b64=glb_b64,
    )
    Path(path).write_text(html)


def main():
    parser = argparse.ArgumentParser(description="Visualize terrain mesh + flat patches as an interactive HTML file.")
    parser.add_argument(
        "--terrain",
        type=str,
        default="RADIATING_BEAM",
        help="Name of a terrain preset (e.g. RADIATING_BEAM, GAP, PIT). Use --list to see all.",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        nargs="+",
        default=[0.5],
        help="One or more difficulty values in [0, 1].  Multiple values are placed side-by-side.",
    )
    parser.add_argument("--size", type=float, nargs=2, default=None, help="Override terrain (x, y) size in meters.")
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Skip flat-patch sampling (no warp/CUDA needed).",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--port", type=int, default=0, help="Serve on this port (0 = auto-pick).")
    parser.add_argument("--no-serve", action="store_true", help="Write file and exit without serving.")
    parser.add_argument("--list", action="store_true", help="List available terrain presets and exit.")
    args = parser.parse_args()

    tcfg = _load_terrain_module()
    presets = _list_presets(tcfg)

    if args.list:
        print("Available terrain presets:")
        for name, cfg in presets.items():
            has_patches = bool(getattr(cfg, "flat_patch_sampling", None))
            print(f"  {name:30s}  patches={'yes' if has_patches else 'no'}")
        sys.exit(0)

    if args.terrain not in presets:
        print(f"Unknown terrain '{args.terrain}'. Available: {', '.join(presets)}")
        sys.exit(1)

    cfg = presets[args.terrain]
    if args.size is not None:
        cfg = cfg.copy()
        cfg.size = tuple(args.size)

    difficulties = sorted(args.difficulty)
    print(f"Terrain   : {args.terrain}")
    print(f"Size      : {cfg.size}")
    print(f"Difficulty: {difficulties}")

    scene = trimesh.Scene()
    spacing = cfg.size[0] * 1.15
    all_patches: dict[str, np.ndarray] = {}

    for idx, diff in enumerate(difficulties):
        x_offset = (idx - (len(difficulties) - 1) / 2.0) * spacing
        mesh, origin = _generate_mesh(cfg, diff)
        print(
            f"\n[d={diff:.2f}] {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces  origin=({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"
        )

        patches_dict: dict[str, np.ndarray] = {}
        if not args.no_patches and cfg.flat_patch_sampling:
            try:
                patches_dict = _sample_patches(mesh, origin, cfg.flat_patch_sampling)
                for name, pts in patches_dict.items():
                    print(f"  {name:8s}: {len(pts)} patches")
            except Exception as exc:
                print(f"  Patch sampling failed ({exc})")

        shift = np.array([x_offset, 0.0, 0.0])
        mesh.apply_translation(shift)
        origin = origin + shift
        for name in patches_dict:
            patches_dict[name] = patches_dict[name] + shift

        colored = _color_mesh_by_height(mesh)
        scene.add_geometry(colored, geom_name=f"terrain_d{diff:.2f}")
        _add_patch_markers(scene, patches_dict, origin, suffix=f"_d{diff:.2f}")

        for name, pts in patches_dict.items():
            all_patches.setdefault(name, []).append(pts)

    merged_patches = {name: np.concatenate(arrs) for name, arrs in all_patches.items()}

    diff_str = "_".join(f"{d:.1f}" for d in difficulties)
    out_path = args.output or f"/tmp/terrain_{args.terrain.lower()}_{diff_str}.html"
    _export_html(scene, out_path, title=args.terrain, difficulty=difficulties, patches_dict=merged_patches)
    out_path = str(Path(out_path).resolve())
    print(f"\nWrote -> {out_path}")

    if args.no_serve:
        return

    import functools
    import http.server
    import socket

    out_dir = str(Path(out_path).parent)
    out_name = Path(out_path).name

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out_dir)
    srv = http.server.HTTPServer(("0.0.0.0", args.port), handler)
    port = srv.server_address[1]
    hostname = socket.gethostname()
    url = f"http://localhost:{port}/{out_name}"

    print(f"\nServing at {url}")
    print(f"  (remote: http://{hostname}:{port}/{out_name})")
    print("Press Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
