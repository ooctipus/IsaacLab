# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose task scenes into one heterogeneous FeatherPGS simulation.

Each cloned world contains one task scene. The demo owns composition and a
generic world lifecycle; it does not construct task environments, MDP managers,
observations, actions, or policy groups.

World lifecycle is one fixed-shape device program. It terminates a world when a
joint velocity is invalid or exceeds its positive limit after settling, or after
250 physics steps. A reset restores snapshotted joint and rigid-body state, then
asks Newton to reconcile actuator, solver, and forward-kinematics state.

.. code-block:: bash

    uv run python scripts/demos/heterogeneous_scene.py --physics=feather_pgs --viz=newton_gl

    # Finite headless smoke run with a smaller composition.
    uv run python scripts/demos/heterogeneous_scene.py \
        --num_task 3 --num_envs 3 --num_steps 300

"""

from __future__ import annotations

import argparse
import os
import sys

# Concurrent USD spawning of heterogeneous assets is unstable in pxr's worker pool.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=64, help="Number of heterogeneous worlds.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between world origins [m].")
parser.add_argument(
    "--sim_dt",
    type=float,
    default=1.0 / 200.0,
    help="Physics timestep [s]. The composed locomotion scenes are authored for 200 Hz.",
)
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of task scenes to compose from the default order. Omit to use all.",
)
parser.add_argument(
    "--num_steps", type=int, default=None, help="Stop after this many played steps. Omit to run forever."
)
parser.add_argument(
    "--physics",
    default="feather_pgs",
    choices=["feather_pgs"],
    help="Physics backend. This demo exercises FeatherPGS heterogeneous worlds.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if any(argument == "--graph" or argument.startswith("--graph=") for argument in hydra_args):
    parser.error("heterogeneous_scene.py always requires CUDA graph capture; --graph is not supported.")
if any(argument.startswith("presets=") for argument in hydra_args):
    parser.error("heterogeneous_scene.py owns the feather_pgs preset; presets= is not supported.")
sys.argv = [sys.argv[0], *hydra_args]

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add

from isaaclab_tasks.utils import resolve_task_config

# Tasks with graphable native actuators and scene-level FeatherPGS presets. Kuka
# Allegro tasks are intentionally absent because their current tuning is task-specific.
DEFAULT_TASKS = (
    "Isaac-Cartpole",
    "Isaac-Ant",
    "Isaac-Humanoid",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Flat-H1",
    "Isaac-Lift-Franka",
    "Isaac-Reorient-Franka",
    "Isaac-Open-Drawer-Franka",
    "IsaacContrib-Open-Drawer-Franka-IK-Abs",
    "IsaacContrib-Open-Drawer-Franka-IK-Rel",
)

_MIN_EPISODE_STEPS = 20
_MAX_EPISODE_STEPS = 250


@wp.kernel(enable_backward=False)
def _advance_world_lifecycle(
    age: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    default_joint_q: wp.array(dtype=wp.float32),
    default_joint_qd: wp.array(dtype=wp.float32),
    default_body_q: wp.array(dtype=wp.transformf),
    default_body_qd: wp.array(dtype=wp.spatial_vectorf),
    joint_velocity_limit: wp.array(dtype=wp.float32),
    joint_coord_world_start: wp.array(dtype=wp.int32),
    joint_dof_world_start: wp.array(dtype=wp.int32),
    body_world_start: wp.array(dtype=wp.int32),
    min_episode_steps: int,
    max_episode_steps: int,
    reset_world: wp.array(dtype=wp.bool),
):
    """Advance and, when terminated, restore one canonical Newton world."""
    world = wp.tid()
    next_age = age[world] + 1
    terminated = next_age >= max_episode_steps

    if not terminated and next_age > min_episode_steps:
        for dof in range(joint_dof_world_start[world], joint_dof_world_start[world + 1]):
            velocity = joint_qd[dof]
            limit = joint_velocity_limit[dof]
            if not wp.isfinite(velocity) or (limit > 0.0 and wp.abs(velocity) > limit):
                terminated = True

    reset_world[world] = terminated
    if terminated:
        for coordinate in range(joint_coord_world_start[world], joint_coord_world_start[world + 1]):
            joint_q[coordinate] = default_joint_q[coordinate]
        for dof in range(joint_dof_world_start[world], joint_dof_world_start[world + 1]):
            joint_qd[dof] = default_joint_qd[dof]
        for body in range(body_world_start[world], body_world_start[world + 1]):
            body_q[body] = default_body_q[body]
            body_qd[body] = default_body_qd[body]
        age[world] = 0
        reset_count[world] = reset_count[world] + 1
    else:
        age[world] = next_age


class _WorldLifecycle:
    """Setup-baked termination and reset program over Newton's model-world domain."""

    def __init__(self, device: str):
        from isaaclab_newton.physics import NewtonManager

        self._device = device
        self._manager = NewtonManager
        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state_0()
        self._age = wp.zeros(self._model.world_count, dtype=wp.int32, device=device)
        self._reset_count = wp.zeros_like(self._age)
        self._reset_world = wp.zeros(self._model.world_count, dtype=wp.bool, device=device)
        self._default_joint_q = wp.clone(self._state.joint_q)
        self._default_joint_qd = wp.clone(self._state.joint_qd)
        self._default_body_q = wp.clone(self._state.body_q)
        self._default_body_qd = wp.clone(self._state.body_qd)

        self._launch()
        self._clear_setup_effects()
        self._graph = None
        if "cuda" in device:
            with wp.ScopedCapture(device=device) as capture:
                self._launch()
            self._graph = capture.graph
            self._clear_setup_effects()

    def _clear_setup_effects(self) -> None:
        self._age.zero_()
        self._reset_count.zero_()
        self._reset_world.zero_()

    def _launch(self) -> None:
        wp.launch(
            _advance_world_lifecycle,
            dim=self._model.world_count,
            inputs=[
                self._age,
                self._reset_count,
                self._state.joint_q,
                self._state.joint_qd,
                self._state.body_q,
                self._state.body_qd,
                self._default_joint_q,
                self._default_joint_qd,
                self._default_body_q,
                self._default_body_qd,
                self._model.joint_velocity_limit,
                self._model.joint_coord_world_start,
                self._model.joint_dof_world_start,
                self._model.body_world_start,
                _MIN_EPISODE_STEPS,
                _MAX_EPISODE_STEPS,
                self._reset_world,
            ],
            device=self._device,
        )
        self._manager.notify_world_reset(self._reset_world)

    def step(self) -> None:
        """Replay the fixed-shape lifecycle once."""
        if self._graph is None:
            self._launch()
        else:
            wp.capture_launch(self._graph)

    def total_resets(self) -> int:
        """Return the completed reset count after a finite run."""
        wp.synchronize_device(self._device)
        return int(self._reset_count.numpy().sum())


def _load_task_scenes() -> tuple[list[str], list[InteractiveSceneCfg]]:
    """Resolve each selected task with its scene-level FeatherPGS preset."""
    task_ids = list(DEFAULT_TASKS if args_cli.num_task is None else DEFAULT_TASKS[: args_cli.num_task])
    if len(task_ids) < 2:
        raise ValueError("Select at least two task scenes.")
    if args_cli.num_envs < len(task_ids):
        raise ValueError(f"num_envs ({args_cli.num_envs}) must cover every selected task ({len(task_ids)}).")
    scene_cfgs = []
    for task_id in task_ids:
        env_cfg, _ = resolve_task_config(task_id, None, overrides=["presets=feather_pgs", *hydra_args])
        scene_cfgs.append(env_cfg.scene)
    return task_ids, scene_cfgs


def _compose_scene() -> tuple[list[str], InteractiveSceneCfg]:
    """Fold task assets and attach one shared light and ground plane."""
    task_ids, task_scene_cfgs = _load_task_scenes()
    for task_scene_cfg in task_scene_cfgs:
        task_scene_cfg.env_spacing = args_cli.env_spacing

    def is_global_asset(asset: AssetBaseCfg) -> bool:
        return isinstance(asset.spawn, (sim_utils.LightCfg, sim_utils.GroundPlaneCfg))

    scene_cfg = task_scene_cfgs[0]
    for task_scene_cfg in task_scene_cfgs[1:]:
        scene_cfg = scene_add(scene_cfg, task_scene_cfg, asset_skip=is_global_asset)
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.replicate_physics = True
    return task_ids, scene_cfg


def _validate_graphable_actuators() -> None:
    """Fail when any Newton actuator cannot join the required physics graph."""
    from isaaclab_newton.physics import NewtonManager

    unsupported = [
        type(actuator.controller).__name__
        for actuator in NewtonManager.get_model().actuators
        if not actuator.is_graphable()
    ]
    if unsupported:
        controllers = ", ".join(sorted(set(unsupported)))
        raise RuntimeError(
            "The heterogeneous demo requires a fully graphed actuator-plus-solver step; "
            f"non-graphable Newton controllers are present: {controllers}."
        )


def main() -> None:
    """Compose the scene, bake its world lifecycle, and step FeatherPGS."""
    if args_cli.num_steps is not None and args_cli.num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    task_ids, scene_cfg = _compose_scene()
    print(f"\n[INFO] Composing task scenes: {task_ids}")

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        physics_cfg.use_cuda_graph = True
        physics_cfg.require_cuda_graph = True
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=args_cli.sim_dt,
                device=args_cli.device,
                physics=physics_cfg,
                use_newton_actuators=True,
            )
        )
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        _validate_graphable_actuators()
        lifecycle = _WorldLifecycle(args_cli.device)
        print(f"[INFO] Composed {len(task_ids)} task scenes into {args_cli.num_envs} worlds. Stepping physics.")

        step = 0
        while sim.is_headless_or_exist_active_visualizer() and (
            args_cli.num_steps is None or step < args_cli.num_steps
        ):
            if sim.is_playing():
                sim.step()
                lifecycle.step()
                step += 1
            else:
                sim.step()

        print(f"[INFO] Completed {step} steps and {lifecycle.total_resets()} world resets.")


if __name__ == "__main__":
    main()
