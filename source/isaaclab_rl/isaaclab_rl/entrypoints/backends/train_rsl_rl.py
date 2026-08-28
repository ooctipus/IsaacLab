# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import time
from datetime import datetime

import torch
from packaging import version

from isaaclab.app import add_launcher_args, report_activity

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    scoped_torch_backend_flags,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    validate_distributed_device,
    wrap_training_capture,
    write_run_manifest,
)

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

RSL_RL_VERSION = "5.0.1"

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _resolve_heatmap_values(heatmaps: dict) -> dict[str, torch.Tensor]:
    """Resolve local values and globally reduce ratio heatmaps."""
    values = {}
    count_data = []
    packed_parts = []
    offset = 0
    for tag, data in heatmaps.items():
        if "numerator" not in data:
            values[tag] = data["values"]
            continue

        numerator = data["numerator"].detach().to(dtype=torch.float64).clone()
        denominator = data["denominator"].detach().to(dtype=torch.float64).clone()
        packed_parts.extend((numerator.flatten(), denominator.flatten()))
        count_data.append((tag, numerator.shape, denominator.shape, offset, numerator.numel(), denominator.numel()))
        offset += numerator.numel() + denominator.numel()

    if not packed_parts:
        return values

    packed = torch.cat(packed_parts)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(packed)

    for tag, numerator_shape, denominator_shape, start, numerator_size, denominator_size in count_data:
        numerator = packed[start : start + numerator_size].view(numerator_shape)
        denominator = packed[start + numerator_size : start + numerator_size + denominator_size].view(denominator_shape)
        ratio = numerator.float() / denominator.float()
        values[tag] = torch.where(denominator > 0, ratio, torch.nan)
    return values


def _tile_heatmap(values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Pad the last dimension with missing values and reshape it into tiles."""
    if values.ndim not in (1, 2):
        raise ValueError(f"Tiled heatmap values must be one- or two-dimensional, got shape {tuple(values.shape)}.")
    rows, columns = shape
    capacity = rows * columns
    if rows < 1 or columns < 1 or values.shape[-1] > capacity:
        raise ValueError(f"Heatmap shape {tuple(values.shape)} does not fit tile shape {shape}.")
    padded = torch.full((*values.shape[:-1], capacity), torch.nan, dtype=values.dtype, device=values.device)
    padded[..., : values.shape[-1]] = values
    return padded.view(*values.shape[:-1], rows, columns)


def _render_faceted_heatmap(plt, tag: str, data: dict, values: torch.Tensor):
    """Render a stack of equally shaped heatmaps as one compact figure."""
    facets, rows, columns = values.shape
    facet_labels = data.get("facet_labels")
    if facet_labels is not None and len(facet_labels) != facets:
        raise ValueError(f"Heatmap '{tag}' has {facets} facets but {len(facet_labels)} facet labels.")

    facet_columns = min(data.get("facet_columns", facets), facets)
    if facet_columns < 1:
        raise ValueError(f"Heatmap '{tag}' must use at least one facet column.")
    facet_rows = (facets + facet_columns - 1) // facet_columns
    size = data.get("figure_size", (3.0 * facet_columns, 2.5 * facet_rows))
    figure, axes = plt.subplots(facet_rows, facet_columns, figsize=size, squeeze=False)
    color_map = plt.get_cmap(data.get("cmap", "RdYlGn")).with_extremes(bad="#d9d9d9")
    cell_labels = data.get("cell_labels")
    image = None

    for index, axes_item in enumerate(axes.flat):
        if index >= facets:
            axes_item.set_axis_off()
            continue
        image = axes_item.imshow(
            values[index].numpy(),
            cmap=color_map,
            vmin=data.get("vmin", 0.0),
            vmax=data.get("vmax", 1.0),
            aspect=data.get("aspect", "equal"),
        )
        axes_item.set_xticks([])
        axes_item.set_yticks([])
        if facet_labels is not None:
            axes_item.set_title(facet_labels[index], fontsize=data.get("facet_fontsize", 9))
        for row in range(rows):
            for column in range(columns):
                value = values[index, row, column]
                if cell_labels is not None:
                    label = cell_labels[row][column]
                elif data.get("annotate_values", True):
                    label = format(float(value), data.get("value_format", ".0%")) if value.isfinite() else "—"
                else:
                    continue
                rgba = image.cmap(image.norm(float(value))) if value.isfinite() else color_map.get_bad()
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axes_item.text(
                    column,
                    row,
                    label,
                    color="black" if luminance > 0.5 else "white",
                    ha="center",
                    va="center",
                    fontsize=data.get("cell_fontsize", 6),
                )

    figure.subplots_adjust(left=0.01, right=0.9, bottom=0.01, top=0.98, wspace=0.03, hspace=0.12)
    colorbar_axes = figure.add_axes((0.92, 0.08, 0.012, 0.82))
    colorbar = figure.colorbar(image, cax=colorbar_axes, label=data.get("color_label", "Value"))
    if data.get("colorbar_percent", False):
        from matplotlib.ticker import PercentFormatter

        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    return figure


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


class _HeatmapLogger:
    """Add environment heatmaps to an RSL-RL W&B logger."""

    _INTERVAL = 250

    def __init__(self, logger, env) -> None:
        self._logger = logger
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._logger, name)

    def log(self, *args, **kwargs) -> None:
        self._logger.log(*args, **kwargs)
        iteration = kwargs["it"] if "it" in kwargs else args[0]
        if iteration % self._INTERVAL:
            return

        heatmaps = self._env.unwrapped.extras.get("heatmap", {})
        if not heatmaps:
            return
        heatmap_values = _resolve_heatmap_values(heatmaps)
        if self._logger.writer is None:
            return

        import matplotlib.pyplot as plt
        import wandb

        images = {}
        for tag, data in heatmaps.items():
            values = heatmap_values[tag].detach().float().cpu()
            if "tile_shape" in data:
                values = _tile_heatmap(values, data["tile_shape"])
            if values.ndim == 3:
                figure = _render_faceted_heatmap(plt, tag, data, values)
                images[tag] = wandb.Image(figure)
                plt.close(figure)
                continue
            if values.ndim != 2:
                raise ValueError(f"Heatmap '{tag}' must be two- or three-dimensional, got shape {tuple(values.shape)}.")
            x_labels, y_labels = data.get("x_labels"), data.get("y_labels")
            cell_labels = data.get("cell_labels")
            color_label = data.get("color_label", "Value")
            value_format = data.get("value_format", ".0%")
            rows, columns = values.shape
            size = data.get(
                "figure_size",
                (6.0, 5.0) if cell_labels is not None else (max(8.0, 0.7 * columns), max(4.0, 0.65 * rows)),
            )
            figure, axes = plt.subplots(figsize=size)
            color_map = plt.get_cmap(data.get("cmap", "RdYlGn")).with_extremes(bad="#d9d9d9")
            default_vmax = None if cell_labels is not None else 1.0
            image = axes.imshow(
                values.numpy(),
                cmap=color_map,
                vmin=data.get("vmin", 0.0),
                vmax=data.get("vmax", default_vmax),
                aspect=data.get("aspect", "equal" if cell_labels is not None else "auto"),
            )
            if x_labels is None:
                axes.set_xticks([])
            else:
                axes.set_xticks(range(columns), x_labels, rotation=60, ha="right", rotation_mode="anchor")
                axes.set_xlabel(data.get("x_label", "Assembly asset"))
            if y_labels is None:
                axes.set_yticks([])
            else:
                axes.set_yticks(range(rows), y_labels)
                axes.set_ylabel(data.get("y_label", "Reset label"))
            if cell_labels is not None:
                axes.set_title(data.get("title", tag))
            for row in range(rows):
                for column in range(columns):
                    value = values[row, column]
                    if cell_labels is not None:
                        label = cell_labels[row][column]
                    elif data.get("annotate_values", True):
                        label = format(float(value), value_format) if value.isfinite() else "—"
                    else:
                        continue
                    rgba = image.cmap(image.norm(float(value))) if value.isfinite() else color_map.get_bad()
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    axes.text(
                        column,
                        row,
                        label,
                        color="black" if luminance > 0.5 else "white",
                        ha="center",
                        va="center",
                        fontsize=10 if cell_labels is not None else 6,
                    )
            colorbar = figure.colorbar(image, ax=axes, label=color_label)
            if data.get("colorbar_percent", False):
                from matplotlib.ticker import PercentFormatter

                colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            figure.tight_layout(pad=0.3 if cell_labels is not None else 1.08)
            images[tag] = wandb.Image(figure)
            plt.close(figure)

        wandb.log(images, step=iteration)


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
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args_cli, remaining_args = setup_preset_cli(parser, argv, agent_library="rsl_rl")
    enable_cameras_for_video(args_cli)

    remaining_args_env_registration = None
    if args_cli.external_callback:
        external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
        remaining_args_env_registration = external_callback_function()

    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent while restoring the caller's Torch backend settings."""
    args_cli = _parse_args(argv)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_deterministic=False,
        cudnn_benchmark=False,
    ):
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute RSL-RL training with parsed arguments."""
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

    installed_version = _check_rsl_rl_version()

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli=args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            agent_cfg.max_iterations = (
                args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
            )

            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

            env_cfg.seed = agent_cfg.seed
            validate_distributed_device(args_cli)

            if args_cli.distributed:
                global_rank = int(os.getenv("RANK", "0"))
                agent_cfg.device = env_cfg.sim.device

                seed = agent_cfg.seed + global_rank
                env_cfg.seed = seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"Exact experiment name requested from command line: {log_dir}")
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            if args_cli.workflow_id:
                log_dir += f"_{args_cli.workflow_id}"
            log_dir = os.path.join(log_root_path, log_dir)
            write_run_manifest(
                log_dir,
                library="rsl_rl",
                task=args_cli.task,
                metadata={"agent": args_cli.agent},
            )

            configure_io_descriptors(env_cfg, args_cli, logger)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )

            if args_cli.checkpoint in CHECKPOINT_SELECTORS:
                resume_path = resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="rsl_rl",
                    task=args_cli.task,
                    checkpoint_pattern=r"model_.*\.pt",
                    metadata={"agent": args_cli.agent},
                )
            elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
                resume_path = get_checkpoint_path(
                    os.path.dirname(args_cli.checkpoint),
                    os.path.basename(args_cli.checkpoint),
                    agent_cfg.load_checkpoint,
                )
            elif args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            elif agent_cfg.algorithm.class_name == "Distillation":
                raise ValueError("Distillation training requires --checkpoint.")

            env = wrap_training_capture(env, log_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            report_activity(None)

            report_activity("Building policy")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            report_activity(None)
            if agent_cfg.logger == "wandb":
                runner.logger = _HeatmapLogger(runner.logger, env)

            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            runner.add_git_repo_to_log(__file__)
            if args_cli.checkpoint:
                print(f"[INFO]: Loading model checkpoint from: {resume_path}")
                runner.load(resume_path)

            dump_train_configs(log_dir, env_cfg, agent_cfg)

            screen.close()
            try:
                runner.learn(
                    num_learning_iterations=agent_cfg.max_iterations,
                    init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                )
                print(f"Training time: {round(time.time() - start_time, 2)} seconds")
                env.close()
            except ValueError as exc:
                if "NaN" in str(exc):
                    env.print_nonfinite_diagnostics()
                raise
            except KeyboardInterrupt:
                pass
