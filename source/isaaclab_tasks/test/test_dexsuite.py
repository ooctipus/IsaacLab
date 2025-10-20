# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""
import sys
import gymnasium as gym
import torch
import carb
import omni.usd
import pytest
import functools
from collections.abc import Callable
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from isaaclab.utils import replace_strings_with_slices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import register_task_to_hydra, resolve_hydra_group_runtime_override


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Copied from hydra.py hydra_task_config, since hydra.main requires a single point of entry,
    which will not work with multiple tests. Here, we replace hydra.main with hydra initialize
    and compose."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            # replace hydra.main with initialize and compose
            with initialize(config_path=None, version_base="1.3"):
                hydra_env_cfg = compose(config_name=task_name, overrides=sys.argv[1:], return_hydra_config=True)
                hydra_env_cfg["hydra"] = hydra_env_cfg["hydra"]["runtime"]["choices"]
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # apply group overrides to mutate cfg objects before from_dict
                resolve_hydra_group_runtime_override(env_cfg, agent_cfg, hydra_env_cfg, hydra_env_cfg["hydra"])
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


def _check_random_actions(
    task_name: str,
    env_cfg: str,
    num_envs: int,
    num_steps: int = 100,
    create_stage_in_memory: bool = False,
    disable_clone_in_fabric: bool = False,
):
    """Run random actions and check environments return valid signals.

    Args:
        task_name: Name of the environment.
        device: Device to use (e.g., 'cuda').
        num_envs: Number of environments.
        num_steps: Number of simulation steps.
        multi_agent: Whether the environment is multi-agent.
        create_stage_in_memory: Whether to create stage in memory.
        disable_clone_in_fabric: Whether to disable fabric cloning.
    """
    # create a new context stage, if stage in memory is not enabled
    if not create_stage_in_memory:
        omni.usd.get_context().new_stage()

    # reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
    try:
        # set config args
        env_cfg.sim.create_stage_in_memory = create_stage_in_memory
        if disable_clone_in_fabric:
            env_cfg.scene.clone_in_fabric = False

        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:
        env.close()
        pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    # reset environment
    obs, _ = env.reset()

    # check signal
    assert _check_valid_tensor(obs)
    action_shape = (num_envs, *env.unwrapped.single_action_space.shape)
    # simulate environment for num_steps
    with torch.inference_mode():
        for _ in range(num_steps):
            actions = torch.randn(action_shape, device=env.unwrapped.sim.device)
            # apply actions
            transition = env.step(actions)
            # check signals
            for data in transition[:-1]:  # exclude info
                assert _check_valid_tensor(data), f"Invalid data: {data}"

    # close environment
    env.close()


def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, (tuple, list)):
        return all(_check_valid_tensor(value) for value in data)
    elif isinstance(data, dict):
        return all(_check_valid_tensor(value) for value in data.values())
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")


@pytest.mark.parametrize("task_name", ["Dexsuite-Kuka-Allegro-Lift", "Dexsuite-Kuka-Allegro-Reorient"])
@pytest.mark.parametrize("object", ["default", "geometry", "cube"])
@pytest.mark.parametrize("num_envs", [32])
@pytest.mark.parametrize("num_steps", [5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dexsuite_kuka_allegro_lift_reorient(task_name: str, object: str, num_envs: int, num_steps: int, device: str):
    # build CLI overrides for Hydra
    sys.argv = [sys.argv[0]]
    sys.argv.append(f"env.scene.object={object}")
    task_name = f"{task_name}-v0"

    @hydra_task_config_test(task_name, "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = device
        _check_random_actions(task_name, env_cfg, num_envs, num_steps=num_steps)

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


@pytest.mark.parametrize("task_name", ["Dexsuite-Kuka-Allegro-Fabric-Lift"])
@pytest.mark.parametrize("object", ["default", "geometry", "cube"])
@pytest.mark.parametrize("num_envs", [32])
@pytest.mark.parametrize("num_steps", [5])
@pytest.mark.parametrize("device", ["cuda"])  # fabrics sim does not support cpu
def test_dexsuite_kuka_allegro_fabric_lift(task_name: str, object: str, num_envs: int, num_steps: int, device: str):
    # build CLI overrides for Hydra
    sys.argv = [sys.argv[0]]
    sys.argv.append(f"env.scene.object={object}")
    task_name = f"{task_name}-v0"

    @hydra_task_config_test(task_name, "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = device
        _check_random_actions(task_name, env_cfg, num_envs, num_steps=num_steps)

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


@pytest.mark.parametrize("task_name", ["Dexsuite-Kuka-Allegro-Lift-Single-Camera", "Dexsuite-Kuka-Allegro-Lift-Duo-Camera"])
@pytest.mark.parametrize("camera_size", ["64x64", "128x128", "256x256"])
@pytest.mark.parametrize("camera_type", ["raycaster", "tiled_depth", "tiled_rgb"])
@pytest.mark.parametrize("object", ["default", "geometry", "cube"])
@pytest.mark.parametrize("num_envs", [32])
@pytest.mark.parametrize("num_steps", [5])
@pytest.mark.parametrize("device", ["cuda"])  # fabrics sim does not support cpu
def test_dexsuite_kuka_allegro_lift_vision(
    task_name: str,
    camera_size: str,
    camera_type: str,
    object: str,
    num_envs: int,
    num_steps: int,
    device: str
):
    # build CLI overrides for Hydra
    sys.argv = [sys.argv[0]]
    sys.argv.append(f"env.scene={camera_size}{camera_type}")
    sys.argv.append(f"env.scene.object={object}")
    task_name = f"{task_name}-v0"

    @hydra_task_config_test(task_name, "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = device
        _check_random_actions(task_name, env_cfg, num_envs, num_steps=num_steps)

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
