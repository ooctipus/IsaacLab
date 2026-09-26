# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments shared by the RSL-RL entrypoints."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab.utils.string import string_to_callable

from ..common import resolve_seed

if TYPE_CHECKING:
    from ...rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path, latest/best, pretrained for play, or a Weights & Biases run"
            " (https://wandb.ai/<entity>/<project>/runs/<run_id>, optionally with a '?checkpoint=<iteration>' query,"
            " or the wandb:<entity>/<project>/<run_id> shorthand)."
        ),
    )
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    wandb_source_group = arg_group.add_mutually_exclusive_group()
    wandb_source_group.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help=(
            "Run ID for Weights & Biases (wandb) to load a specific run. If not provided, will not load the run from"
            " wandb."
        ),
    )
    wandb_source_group.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=(
            "Exact display name of a Weights & Biases run to load. The name must resolve to one run in the logging"
            " project."
        ),
    )
    arg_group.add_argument(
        "--wandb_checkpoint_iteration",
        type=str,
        default=None,
        help="Select which wandb checkpoint iteration to load. If not provided, the latest checkpoint will be used.",
    )
    arg_group.add_argument(
        "--wandb_username",
        type=str,
        default=None,
        help=(
            "Username for Weights & Biases (wandb). If not provided, will use the environment variable WANDB_USERNAME."
        ),
    )


def register_external_tasks(argv: list[str]) -> list[str] | None:
    """Run the ``--external_callback`` named in *argv* and return the arguments it did not consume.

    Downstream code registers its tasks in the callback, so it has to run before the preset setup
    reads the Gym metadata of those tasks.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        The arguments left for Hydra, or None when no callback was requested.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--external_callback")
    args, _ = parser.parse_known_args(argv)
    if not args.external_callback:
        return None
    return string_to_callable(args.external_callback, separator=".")()


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Load the registered RSL-RL agent configuration of a task and apply the command-line overrides.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The updated RSL-RL agent configuration.
    """
    # the task registry is only needed by callers that bypass the Hydra task resolution
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(agent_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Override an RSL-RL agent configuration with the command-line arguments.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated RSL-RL agent configuration.
    """
    if getattr(args_cli, "seed", None) is not None:
        args_cli.seed = resolve_seed(args_cli.seed)
        agent_cfg.seed = args_cli.seed
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if args_cli.wandb_run_id is not None or args_cli.wandb_run_name is not None:
        # User wants to sync from wandb
        from isaaclab.utils.wandb import get_model_checkpoint

        checkpoint_folder = get_model_checkpoint(
            run_id=args_cli.wandb_run_id,
            run_name=args_cli.wandb_run_name,
            project=agent_cfg.wandb_project,
            checkpoint=args_cli.wandb_checkpoint_iteration,
            wandb_username=args_cli.wandb_username,
        )
        # The common checkpoint path owns resuming, including W&B downloads.
        args_cli.checkpoint = checkpoint_folder
        agent_cfg.load_checkpoint = checkpoint_folder

    return agent_cfg
