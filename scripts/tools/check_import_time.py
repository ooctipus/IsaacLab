# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure import + env creation time for individual tasks.

Usage:
    python scripts/tools/check_import_time.py          # without Kit
    python scripts/tools/check_import_time.py --kit     # with Kit startup

Both modes measure registry import, config instantiation, and gym.make.
With --kit, Kit is launched first and its startup time is reported separately.
"""

import argparse
import subprocess
import sys

TASKS = [
    "Isaac-Cartpole-Direct-v0",
    "Isaac-Velocity-Flat-Anymal-C-v0",
    "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
]

_WORKER_KIT = """
import os, sys, time
import gymnasium as gym

task = sys.argv[1]

os.environ["CARB_LOG_LEVEL"] = "error"
os.environ["OMNI_LOG_LEVEL"] = "error"

import builtins
_real_print = builtins.print
builtins.print = lambda *a, **kw: None

t_kit_start = time.perf_counter()
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, num_envs=1)
simulation_app = app_launcher.app
t_kit = time.perf_counter() - t_kit_start

t0 = time.perf_counter()
import isaaclab_tasks
t_registry = time.perf_counter() - t0

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

t1 = time.perf_counter()
env_cfg = parse_env_cfg(task, num_envs=1, use_fabric=True)
t_cfg = time.perf_counter() - t1

t2 = time.perf_counter()
env = gym.make(task, cfg=env_cfg)
t_make = time.perf_counter() - t2

builtins.print = _real_print

total = t_kit + t_registry + t_cfg + t_make
sys.stderr.write(
    f"RESULT {task}  kit={t_kit:.4f}s  registry={t_registry:.4f}s"
    f"  cfg={t_cfg:.4f}s  gym.make={t_make:.4f}s  total={total:.4f}s\\n"
)
sys.stderr.flush()

env.close()
simulation_app.close()
"""

_WORKER_NO_KIT = """
import sys, time
import gymnasium as gym

task = sys.argv[1]

t0 = time.perf_counter()
import isaaclab_tasks
t_registry = time.perf_counter() - t0

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

t1 = time.perf_counter()
env_cfg = parse_env_cfg(task, num_envs=1, use_fabric=True)
t_cfg = time.perf_counter() - t1

t2 = time.perf_counter()
env = gym.make(task, cfg=env_cfg)
t_make = time.perf_counter() - t2

total = t_registry + t_cfg + t_make
sys.stderr.write(
    f"RESULT {task}  registry={t_registry:.4f}s"
    f"  cfg={t_cfg:.4f}s  gym.make={t_make:.4f}s  total={total:.4f}s\\n"
)
sys.stderr.flush()

env.close()
"""


def main():
    parser = argparse.ArgumentParser(description="Measure task import and env creation time.")
    parser.add_argument("--kit", action="store_true", help="Launch Kit (headless) and report its startup time.")
    args = parser.parse_args()

    worker = _WORKER_KIT if args.kit else _WORKER_NO_KIT

    for task in TASKS:
        proc = subprocess.run(
            [sys.executable, "-c", worker, task],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        found = False
        for line in proc.stderr.splitlines():
            if line.startswith("RESULT "):
                print(line[7:])
                found = True
        if not found:
            print(f"FAIL  {task}  (exit code {proc.returncode})")
            tail = [line for line in proc.stderr.splitlines() if line.strip()][-5:]
            for line in tail:
                print(f"      {line}")


if __name__ == "__main__":
    main()
