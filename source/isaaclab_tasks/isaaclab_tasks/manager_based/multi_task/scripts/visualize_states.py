# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize reset states without applying actions.

This script mirrors the task/config launch path used by the RSL-RL play
script, but it does not load a checkpoint and never calls ``env.step``.
It repeatedly resets the environment, renders the reset pose for a fixed
number of frames, and resets again. This is useful for inspecting sampled
terrain-conforming spawn states.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Visualize environment reset states without stepping actions.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_steps", type=int, default=120, help="Number of render calls after each reset.")
parser.add_argument(
    "--render_dt",
    type=float,
    default=1.0 / 60.0,
    help="Wall-clock delay between render calls [s]. Set to 0 to render as fast as possible.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()


# Call an external callback if requested. This gives external code a chance to
# register environments before Hydra resolves the task config.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Leave only Hydra/preset overrides in sys.argv, e.g. ``presets=anymal_c``.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


def _render_reset_state(env: gym.Env, num_steps: int, render_dt: float):
    """Render the current reset state for ``num_steps`` frames."""
    unwrapped = env.unwrapped
    for _ in range(num_steps):
        start_time = time.time()
        unwrapped.sim.render()
        if render_dt > 0.0:
            sleep_time = render_dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def _detach_viewport_tracking(env: gym.Env):
    """Unsubscribe the asset-tracking viewport callback before simulator teardown."""
    viewport_controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if viewport_controller is None:
        return
    handle = getattr(viewport_controller, "_viewport_camera_update_handle", None)
    if handle is not None:
        handle.unsubscribe()
        viewport_controller._viewport_camera_update_handle = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Reset and render an environment repeatedly."""
    with launch_simulation(env_cfg, args_cli):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        agent_seed = getattr(agent_cfg, "seed", None)
        env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_seed

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
        try:
            reset_id = 0
            while True:
                print(f"[INFO] Showing reset {reset_id + 1}", flush=True)
                env.reset()
                _render_reset_state(env, max(args_cli.num_steps, 1), max(args_cli.render_dt, 0.0))
                reset_id += 1
        except KeyboardInterrupt:
            pass
        finally:
            _detach_viewport_tracking(env)
            env.close()


if __name__ == "__main__":
    main()
