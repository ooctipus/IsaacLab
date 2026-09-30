# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script uses the cloner API to check if asset has been instanced properly.

Usage with different inputs (replace `<Asset-Path>` and `<Asset-Path-Instanced>` with the path to the
original asset and the instanced asset respectively):

```bash
uv run python source/tools/check_instanceable.py <Asset-Path> -n 4096 --physics
uv run python source/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096 --physics
uv run python source/tools/check_instanceable.py <Asset-Path> -n 4096
uv run python source/tools/check_instanceable.py <Asset-Path-Instanced> -n 4096
```

Output from the above commands:

```bash
>>> Cloning time (ReplicateSession): 0.648198 seconds
>>> Setup time (sim.reset): : 5.843589 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path-Instanced> : 6.491870 seconds

>>> Cloning time (ReplicateSession): 0.693133 seconds
>>> Setup time (sim.reset): 50.860526 seconds
[#clones: 4096, physics: True] Asset: <Asset-Path> : 51.553743 seconds

>>> Cloning time (ReplicateSession) : 0.687201 seconds
>>> Setup time (sim.reset) : 6.302215 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path-Instanced> : 6.989500 seconds

>>> Cloning time (ReplicateSession) : 0.678150 seconds
>>> Setup time (sim.reset) : 52.854054 seconds
[#clones: 4096, physics: False] Asset: <Asset-Path> : 53.532287 seconds
```

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
from dataclasses import MISSING

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Utility to empirically check if asset in instanced properly.")
parser.add_argument("input", type=str, help="The path to the USD file.")
parser.add_argument("-n", "--num_clones", type=int, default=128, help="Number of clones to spawn.")
parser.add_argument("-s", "--spacing", type=float, default=1.5, help="Spacing between instances in a grid.")
parser.add_argument("-p", "--physics", action="store_true", default=False, help="Clone assets using physics cloner.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import Timer
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.configclass import configclass


@configclass
class DirectCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    asset: AssetBaseCfg = MISSING
    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg())
    num_clones: int = 128
    env_spacing: float = 1.5
    replicate_physics: bool = False


def main():
    """Measure plan-owned replication and simulation setup for one USD asset."""
    # check valid file path
    if not check_file_path(args_cli.input):
        raise ValueError(f"Invalid file path: {args_cli.input}")
    cfg = DirectCfg(
        asset=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Asset",
            spawn=sim_utils.UsdFileCfg(usd_path=args_cli.input),
        ),
        num_clones=args_cli.num_clones,
        env_spacing=args_cli.spacing,
        replicate_physics=args_cli.physics,
    )
    sim = SimulationContext(cfg.sim)

    # Fabric and PhysX GPU buffers are configured through SimulationCfg/PhysxCfg defaults.
    # enable hydra scene-graph instancing
    # this is needed to visualize the scene when fabric is enabled
    sim.set_setting("/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create a timer to measure the cloning time
    with Timer(f"[#clones: {cfg.num_clones}, physics: {cfg.replicate_physics}] Asset: {args_cli.input}"):
        with Timer(">>> Cloning time (ReplicateSession)"):
            with ReplicateSession(
                (cfg.light, cfg.asset),
                cfg.num_clones,
                cfg.env_spacing,
                replicate_physics=cfg.replicate_physics,
            ):
                for asset_cfg in (cfg.light, cfg.asset):
                    asset_cfg.class_type(asset_cfg)
        # Play the simulator
        with Timer(">>> Setup time (sim.reset)"):
            sim.reset()

    # Simulate scene (if not headless)
    if not args_cli.headless:
        with contextlib.suppress(KeyboardInterrupt):
            while sim.is_playing():
                # perform step
                sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
