# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata as metadata
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

from common import (
    add_common_train_args,
    add_isaaclab_launcher_args,
    apply_env_overrides,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    import_local_module,
    set_hydra_args,
    validate_distributed_device,
    wrap_record_video,
)
from packaging import version

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

RSL_RL_VERSION = "5.4.1"
RL_ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS = import_local_module("isaaclab_rsl_rl_cli_args", RL_ROOT / "rsl_rl" / "cli_args.py")
_TRAINING_CALLBACK_STAGES = frozenset({"prepare", "launch", "complete", "validate"})


# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _invoke_training_callback(callback, stage: str, **values: object) -> None:
    """Invoke one optional callback at a declared training lifecycle boundary."""
    if stage not in _TRAINING_CALLBACK_STAGES:
        raise ValueError(f"Unsupported training callback stage: {stage!r}.")
    if callback is not None:
        callback(stage=stage, **values)


def _validate_checkpoint(runner, checkpoint: Path) -> None:
    """Strictly restore one checkpoint through mmap-backed CPU deserialization."""
    runner.load(str(checkpoint), strict=True, map_location="cpu", mmap=True)


def _snapshot_config_for_callback(env_cfg: object) -> object:
    """Copy declared configuration before environment constructors normalize it in place."""
    return copy.deepcopy(env_cfg)


def _check_rsl_rl_version() -> str:
    """Check that the installed RSL-RL version is supported."""
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        if platform.system() == "Windows":
            cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        else:
            cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        print(
            f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
            f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
            f"\n\n\t{' '.join(cmd)}\n"
        )
        raise SystemExit(1)
    return installed_version


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL training arguments."""
    from isaaclab.utils.string import list_intersection, string_to_callable

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback",
        default=None,
        help="Fully qualified path to an externally defined callback.",
    )
    CLI_ARGS.add_rsl_rl_args(parser)
    parser.add_argument(
        "--training_callback",
        default=None,
        help="Optional module:callable invoked at preparation, launch, completion, or checkpoint validation.",
    )
    parser.add_argument(
        "--training_callback_mode",
        choices=("train", "prepare"),
        default="train",
        help="Run training normally, or construct the environment and runner for callback preparation then exit.",
    )
    parser.add_argument(
        "--validate_checkpoint",
        type=Path,
        default=None,
        help="Strictly load one checkpoint through CPU-backed deserialization without training.",
    )
    add_isaaclab_launcher_args(parser)
    # setup_preset_cli registers preset-selection help text + runs parse_known_args
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)

    remaining_args_env_registration = None
    if args_cli.external_callback:
        external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
        remaining_args_env_registration = external_callback_function()
    if args_cli.training_callback:
        args_cli.training_callback = string_to_callable(args_cli.training_callback)

    # physics=/renderer=/presets= tokens pass through the remainder for hydra to parse later
    if args_cli.training_callback_mode == "prepare" and args_cli.training_callback is None:
        parser.error("--training_callback_mode prepare requires --training_callback.")
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent."""
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    args_cli = _parse_args(argv)
    installed_version = _check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = CLI_ARGS.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        resolved_num_envs = agent_cfg.resolve_num_envs(args_cli.num_envs, env_cfg.scene.num_envs)
        env_cfg.scene.num_envs = resolved_num_envs
        agent_cfg.num_envs = resolved_num_envs
        agent_cfg.max_iterations = agent_cfg.resolve_max_iterations(args_cli.max_iterations)

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        validate_distributed_device(args_cli)

        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))
            agent_cfg.device = env_cfg.sim.device

            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        if args_cli.workflow_id:
            log_dir += f"_{args_cli.workflow_id}"
        log_dir = os.path.join(log_root_path, log_dir)

        configure_io_descriptors(env_cfg, args_cli, logger)
        env_cfg.log_dir = log_dir
        configured_env_cfg = _snapshot_config_for_callback(env_cfg) if args_cli.training_callback is not None else None

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )

        if agent_cfg.resume or agent_cfg.get_algorithm_class_name() == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        env = wrap_record_video(env, log_dir, args_cli)

        start_time = time.time()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = agent_cfg.class_type(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

        callback_values = {
            "env_cfg": env_cfg,
            "configured_env_cfg": configured_env_cfg,
            "agent_cfg": agent_cfg,
            "env": env,
            "runner": runner,
            "log_dir": Path(log_dir),
        }
        if args_cli.training_callback_mode == "prepare":
            if args_cli.validate_checkpoint is not None:
                raise ValueError("Callback preparation cannot be combined with --validate_checkpoint.")
            try:
                _invoke_training_callback(args_cli.training_callback, "prepare", **callback_values)
            finally:
                env.close()
            return
        if args_cli.validate_checkpoint is not None:
            if agent_cfg.resume or agent_cfg.get_algorithm_class_name() == "Distillation":
                raise ValueError("--validate_checkpoint cannot be combined with resume or distillation loading.")
            checkpoint = args_cli.validate_checkpoint.expanduser().resolve()
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise ValueError(f"Validation checkpoint must be a regular non-symbolic file: {checkpoint}.")
            validation_start = time.time()
            _validate_checkpoint(runner, checkpoint)
            _invoke_training_callback(
                args_cli.training_callback,
                "validate",
                checkpoint_path=checkpoint,
                **callback_values,
            )
            print(f"Checkpoint validation time: {round(time.time() - validation_start, 2)} seconds")
            env.close()
            return

        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume or agent_cfg.get_algorithm_class_name() == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_train_configs(log_dir, env_cfg, agent_cfg)
        _invoke_training_callback(args_cli.training_callback, "launch", **callback_values)

        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
            )
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            checkpoint = Path(log_dir) / f"model_{runner.current_learning_iteration}.pt"
            _invoke_training_callback(
                args_cli.training_callback,
                "complete",
                checkpoint_path=checkpoint,
                **callback_values,
            )
            env.close()
        except KeyboardInterrupt:
            pass
