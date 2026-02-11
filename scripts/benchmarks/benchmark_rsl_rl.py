# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="OmniPerfKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile", "OmniPerfKPIFile"],
    help="Benchmarking backend options, defaults OmniPerfKPIFile",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="/tmp",
    help="Output folder for the benchmark metrics.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# to ensure kit args don't break the benchmark arg parsing
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


app_start_time_end = time.perf_counter_ns()

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import numpy as np
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.timer import Timer

Timer.enable_display_output = False

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

imports_time_end = time.perf_counter_ns()

try:
    # enable benchmarking extension (Isaac Sim)
    from isaacsim.core.utils.extensions import enable_extension
    import carb
    from isaacsim.benchmark.services import BaseIsaacBenchmark as _IsaacBenchmark

    _ISAACSIM_BENCHMARK_AVAILABLE = True
except ModuleNotFoundError:
    _ISAACSIM_BENCHMARK_AVAILABLE = False

if _ISAACSIM_BENCHMARK_AVAILABLE and simulation_app is not None:
    BaseIsaacBenchmark = _IsaacBenchmark
    _BENCHMARK_SERVICES_AVAILABLE = True
    enable_extension("isaacsim.benchmark.services")
    # Set the benchmark settings according to the inputs
    settings = carb.settings.get_settings()
    settings.set("/exts/isaacsim.benchmark.services/metrics/metrics_output_folder", args_cli.output_folder)
    settings.set("/exts/isaacsim.benchmark.services/metrics/randomize_filename_prefix", True)
else:
    from scripts.benchmarks.kitless_reporter import KitlessBenchmark as BaseIsaacBenchmark

    _BENCHMARK_SERVICES_AVAILABLE = False

from scripts.benchmarks.utils import (
    get_isaaclab_version,
    get_mujoco_warp_version,
    get_newton_version,
    log_app_start_time,
    log_newton_finalize_builder_time,
    log_newton_initialize_solver_time,
    log_python_imports_time,
    log_rl_policy_episode_lengths,
    log_rl_policy_rewards,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_task_start_time,
    log_total_start_time,
    parse_tf_logs,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Create the benchmark
benchmark_backend = args_cli.benchmark_backend if _BENCHMARK_SERVICES_AVAILABLE else "kitless"
benchmark_kwargs = {
    "benchmark_name": "benchmark_rsl_rl_train",
    "workflow_metadata": {
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "max_iterations", "data": args_cli.max_iterations},
            {"name": "Mujoco Warp Info", "data": get_mujoco_warp_version()},
            {"name": "Isaac Lab Info", "data": get_isaaclab_version()},
            {"name": "Newton Info", "data": get_newton_version()},
        ],
    },
    "backend_type": benchmark_backend,
}
if not _BENCHMARK_SERVICES_AVAILABLE:
    benchmark_kwargs["output_dir"] = args_cli.output_folder
benchmark = BaseIsaacBenchmark(**benchmark_kwargs)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # parse configuration
    benchmark.set_phase("loading", start_recording_frametime=False, start_recording_runtime=True)
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    world_rank = 0
    world_size = 1
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
        world_rank = app_launcher.global_rank
        world_size = int(os.getenv("WORLD_SIZE", 1))

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    task_startup_time_end = time.perf_counter_ns()

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    benchmark.set_phase("sim_runtime")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if world_rank == 0:
        benchmark.store_measurements()

        # parse tensorboard file stats
        log_data = parse_tf_logs(log_dir)

        def _get_series(keys: list[str]):
            for key in keys:
                if key in log_data:
                    return np.array(log_data[key])
            return None

        def _detect_perf_time_scale(raw_times: np.ndarray | None, total_fps: np.ndarray | None, steps_per_iter: int):
            """Detect whether perf times are in seconds or milliseconds.

            Returns a scale factor to convert raw values to milliseconds.
            """
            override = os.getenv("RSL_RL_PERF_TIME_UNIT", "").strip().lower()
            if override in {"ms", "millis", "milliseconds"}:
                return 1.0
            if override in {"s", "sec", "secs", "seconds"}:
                return 1000.0
            if raw_times is None or total_fps is None:
                return 1000.0
            raw = raw_times
            total = total_fps
            valid = (raw > 0) & np.isfinite(raw) & (total > 0) & np.isfinite(total)
            if not np.any(valid):
                return 1000.0
            with np.errstate(divide="ignore", invalid="ignore"):
                fps_from_raw = (1.0 / raw) * steps_per_iter
            median_fps = float(np.median(fps_from_raw[valid]))
            median_total = float(np.median(total[valid]))
            if median_total <= 0:
                return 1000.0
            diff_seconds = abs(median_fps - median_total)
            diff_ms = abs(median_fps / 1000.0 - median_total)
            if diff_ms < diff_seconds:
                return 1.0
            return 1000.0

        collection_time = _get_series(["Perf/collection time", "Perf/collection_time"])
        learning_time = _get_series(["Perf/learning_time", "Perf/learning time"])
        total_fps = _get_series(["Perf/total_fps", "Perf/total fps"])
        if total_fps is not None:
            total_fps = total_fps * world_size

        rl_training_times = {}
        steps_per_iter = env.unwrapped.num_envs * agent_cfg.num_steps_per_env * world_size
        time_scale = _detect_perf_time_scale(collection_time, total_fps, steps_per_iter)
        if collection_time is not None:
            collection_time_ms = collection_time * time_scale
            env_step_time_ms = collection_time_ms / steps_per_iter
            with np.errstate(divide="ignore", invalid="ignore"):
                collection_fps = (1.0 / (collection_time_ms / 1000.0)) * steps_per_iter
            rl_training_times["Environment only step time"] = env_step_time_ms.tolist()
            rl_training_times["Collection Time"] = collection_time_ms.tolist()
            rl_training_times["Collection FPS"] = collection_fps.tolist()
        if learning_time is not None:
            rl_training_times["Learning Time"] = (learning_time * time_scale).tolist()
        if total_fps is not None:
            rl_training_times["Total FPS"] = total_fps.tolist()

        steps_processed = None
        if collection_time is not None:
            steps_processed = int(steps_per_iter * len(collection_time))
        elif agent_cfg.max_iterations:
            steps_processed = int(steps_per_iter * agent_cfg.max_iterations)
        if steps_processed is not None and hasattr(benchmark, "store_metadata_item"):
            benchmark.store_metadata_item("num_frames", steps_processed)

        # log additional metrics to benchmark services
        log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
        log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
        log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)
        log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
        log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
        log_newton_finalize_builder_time(benchmark, Timer.get_timer_info("newton_finalize_builder") * 1000)
        log_newton_initialize_solver_time(benchmark, Timer.get_timer_info("newton_initialize_solver") * 1000)
        log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
        if rl_training_times:
            log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
        if "Train/mean_reward" in log_data:
            log_rl_policy_rewards(benchmark, log_data["Train/mean_reward"])
        if "Train/mean_episode_length" in log_data:
            log_rl_policy_episode_lengths(benchmark, log_data["Train/mean_episode_length"])

        benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    if simulation_app is not None:
        simulation_app.close()
