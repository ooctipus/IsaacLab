# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the Factory reset-state table: held-asset (nut) poses relative to the fixed asset (bolt).

Boots the Factory env exactly like the play script (presets resolve the same
way), which triggers the reset-state :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand`
to pre-collect and finalize its reset-state table during construction. The script then reads the
stored states straight off the command term, expresses each held-asset root
pose in the fixed-asset frame, and renders them as a viser point cloud:

* spawn states are split into one point cloud per reset-strategy tag (toggle
  each tag in the viser sidebar),
* goal/target states (the ones selected by ``targets_per_board``) are highlighted,
* the fixed asset (bolt) sits at the origin as the common reference frame.

Usage::

    SCRIPT=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/visualize_reset_states.py

    ./isaaclab.sh -p $SCRIPT --task Isaac-Factory-v0 --headless
    ./isaaclab.sh -p $SCRIPT --task Isaac-Factory-v0 --headless --arrows
    ./isaaclab.sh -p $SCRIPT --task Isaac-Factory-v0 --headless --no_viewer  # diagnostics only

Requires ``viser`` (``pip install viser``) for the viewer; ``--no_viewer``
skips it and just prints the table summary.
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

VISER_PORT = 8765

# Distinct, high-contrast palette for per-tag spawn clouds.
_TAG_PALETTE = (
    (0.20, 0.55, 1.00),
    (0.30, 0.85, 0.35),
    (1.00, 0.45, 0.20),
    (0.75, 0.40, 0.95),
    (0.20, 0.85, 0.85),
    (0.95, 0.80, 0.20),
    (0.95, 0.35, 0.65),
    (0.55, 0.75, 0.30),
)


parser = argparse.ArgumentParser(description="Visualize Factory reset states (nut relative to bolt) in viser.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments used to build the table.")
parser.add_argument("--task", type=str, default="Isaac-Factory-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--command_name", type=str, default="reset_state", help="Reset-state command term name.")
parser.add_argument("--arrows", action="store_true", help="Also draw the held-asset z-axis (insertion direction).")
parser.add_argument("--point_radius", type=float, default=0.0025, help="Spawn point radius [m].")
parser.add_argument("--no_viewer", action="store_true", help="Skip the viser viewer; print diagnostics and exit.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()


# Give external code a chance to register environments before Hydra resolves the task config.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Leave only Hydra/preset overrides in sys.argv, e.g. ``presets=...``.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


def _extract_table(env: gym.Env, command_name: str) -> dict:
    """Read the reset-state table off the command term and express the nut in the bolt frame."""
    command_term = env.unwrapped.command_manager.get_term(command_name)
    payload = command_term.payload
    table = command_term.table
    state_data = table.state_data
    held_offset = payload._root_state_offset(command_term.cfg.payload.held_asset_cfg.name)
    fixed_offset = payload._root_state_offset(command_term.cfg.payload.fixed_asset_cfg.name)

    held_pos = state_data[:, held_offset : held_offset + 3]
    held_quat = state_data[:, held_offset + 3 : held_offset + 7]
    fixed_pos = state_data[:, fixed_offset : fixed_offset + 3]
    fixed_quat = state_data[:, fixed_offset + 3 : fixed_offset + 7]
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(fixed_pos, fixed_quat, held_pos, held_quat)

    num_states = int(state_data.shape[0])
    is_target = torch.zeros(num_states, dtype=torch.bool, device=state_data.device)
    is_target[torch.unique(table.target_index)] = True

    return {
        "rel_pos": rel_pos.cpu().numpy(),
        "rel_quat": rel_quat.cpu().numpy(),
        "tags": table.state_tag_indices.cpu().numpy(),
        "tag_names": list(table.state_tag_names) if table.state_tag_names else None,
        "is_target": is_target.cpu().numpy(),
        "num_slots": int(table.spawn_index.shape[0]),
    }


def _print_summary(data: dict) -> None:
    """Print a per-tag breakdown of the reset-state table."""
    tags = data["tags"]
    tag_names = data["tag_names"]
    is_target = data["is_target"]
    print("\n--- Reset-state table ---")
    print(f"  spawn states : {len(tags)}")
    print(f"  goal states  : {int(is_target.sum())}")
    print(f"  total slots  : {data['num_slots']} (spawn x goal pairs)")
    for tag in sorted(set(int(t) for t in tags.tolist())):
        name = tag_names[tag] if tag_names and 0 <= tag < len(tag_names) else f"tag_{tag}"
        mask = tags == tag
        print(f"  - {name}: {int(mask.sum())} spawn, {int((mask & is_target).sum())} goal")


def _run_viewer(data: dict, draw_arrows: bool, point_radius: float) -> None:
    """Render the relative nut poses as a viser point cloud, highlighting goals."""
    import newton
    import warp as wp
    from newton.viewer import ViewerViser

    device = "cuda:0"
    rel_pos = data["rel_pos"]
    tags = data["tags"]
    tag_names = data["tag_names"]
    is_target = data["is_target"]

    # Bolt (fixed asset) marker at the origin — the common reference frame.
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.01, hy=0.01, hz=0.02, color=(0.55, 0.55, 0.6))
    model = builder.finalize(device=device)

    viewer = ViewerViser(port=VISER_PORT)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))

    def _points(name: str, pts: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        if pts.shape[0] == 0:
            return
        viewer.log_points(name, wp.array(pts.tolist(), dtype=wp.vec3, device=device), radii=radius, colors=color)

    # One toggleable cloud per reset-strategy tag (non-goal spawns only).
    for palette_idx, tag in enumerate(sorted(set(int(t) for t in tags.tolist()))):
        name = tag_names[tag] if tag_names and 0 <= tag < len(tag_names) else f"tag_{tag}"
        mask = (tags == tag) & (~is_target)
        color = _TAG_PALETTE[palette_idx % len(_TAG_PALETTE)]
        _points(f"spawn/{name}", rel_pos[mask], point_radius, color)

    # Goal states highlighted (bright yellow, larger).
    _points("goals", rel_pos[is_target], point_radius * 2.4, (1.0, 0.95, 0.15))
    # Bolt origin.
    _points("bolt_origin", np.zeros((1, 3), dtype=np.float32), point_radius * 3.0, (1.0, 0.2, 0.2))

    if draw_arrows:
        rel_pos_t = torch.from_numpy(rel_pos)
        rel_quat_t = torch.from_numpy(data["rel_quat"])
        z_axis = torch.tensor([0.0, 0.0, 1.0]).expand(rel_pos_t.shape[0], 3)
        ends = rel_pos_t + math_utils.quat_apply(rel_quat_t, z_axis) * 0.01
        viewer.log_arrows(
            "nut_z_axis",
            wp.array(rel_pos.tolist(), dtype=wp.vec3, device=device),
            wp.array(ends.numpy().tolist(), dtype=wp.vec3, device=device),
            colors=(0.85, 0.85, 0.85),
            width=0.0008,
        )

    viewer.begin_frame(0.0)
    viewer.log_state(model.state())
    viewer.end_frame()

    hostname = socket.gethostname()
    print(f"\n  http://localhost:{VISER_PORT}")
    print(f"  http://{hostname}:{VISER_PORT}")
    print("  Bolt = grey box at origin (red dot). Spawns colored per tag. Goals = yellow.")
    print("\nPress Ctrl+C to stop.\n")

    import time

    try:
        while viewer.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Build the env (and thus the reset-state table), extract it, then visualize."""
    data: dict | None = None
    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        agent_seed = getattr(agent_cfg, "seed", None)
        env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_seed

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        try:
            data = _extract_table(env, args_cli.command_name)
            _print_summary(data)
        finally:
            env.close()

    if args_cli.no_viewer:
        print("\n--no_viewer: exiting without visualization.")
        return
    _run_viewer(data, args_cli.arrows, args_cli.point_radius)


if __name__ == "__main__":
    main()
