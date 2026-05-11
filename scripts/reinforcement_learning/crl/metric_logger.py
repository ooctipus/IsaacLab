# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight metric logger for CRL training.

Writes:

- **JSONL** (``<log_dir>/metrics.jsonl``) — one record per logged step, streamable
  and trivially parseable with ``pandas.read_json(..., lines=True)`` or ``jq``.
- **TensorBoard** events (``<log_dir>/tb/``) — live plot in tensorboard if the
  ``tensorboardX`` package is available. Fall back gracefully if not.

Design goals:
- No wandb dependency (scaling-crl's default is wandb with offline mode; we keep
  the interface compatible but don't require a wandb account).
- Zero-config for visualization: JSONL is readable by every Python user;
  TensorBoard works out of the box if you already have the package.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from typing import Any


class MetricLogger:
    """Append-only logger for training metrics.

    Usage:

    .. code-block:: python

        logger = MetricLogger(log_dir="logs/crl/my_run", config={"depth": 4, "seed": 1000})
        for epoch in range(num_epochs):
            ...
            logger.log(step=epoch * 1000, epoch=epoch, actor_loss=al, critic_loss=cl)
        logger.close()

    The JSONL file format is one record per line:

    .. code-block:: text

        {"step": 0, "wall_time": 123.4, "actor_loss": 0.5, "critic_loss": 3.1}
        {"step": 1000, ...}
    """

    def __init__(
        self,
        log_dir: str,
        *,
        config: dict[str, Any] | None = None,
        enable_tensorboard: bool = True,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        wandb_entity: str | None = None,
        wandb_name: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_mode: str = "online",
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # JSONL sink
        self._jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._jsonl_fh = open(self._jsonl_path, "a", buffering=1)  # noqa: SIM115  # line-buffered lifetime sink

        # Config snapshot (one-shot)
        if config is not None:
            with open(os.path.join(self.log_dir, "config.json"), "w") as f:
                json.dump(_json_safe(config), f, indent=2, sort_keys=True)

        # TensorBoard sink (optional)
        self._tb_writer: Any | None = None
        if enable_tensorboard:
            try:
                from tensorboardX import SummaryWriter

                tb_dir = os.path.join(self.log_dir, "tb")
                self._tb_writer = SummaryWriter(tb_dir)
                print(f"[MetricLogger] TensorBoard events -> {tb_dir}", flush=True)
            except ImportError:
                print(
                    "[MetricLogger] tensorboardX not installed — JSONL only. Install with: pip install tensorboardX",
                    flush=True,
                )

        # Wandb sink (optional). We log per-step with the same step axis as TB so the
        # "reproduction chart" (ours vs scaling-crl native, same wandb project/group)
        # aligns on env_steps cleanly.
        self._wandb = None
        if wandb_project is not None:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=wandb_project,
                    group=wandb_group,
                    entity=wandb_entity,
                    name=wandb_name,
                    tags=wandb_tags or [],
                    config=_json_safe(config) if config is not None else None,
                    mode=wandb_mode,
                    dir=log_dir,
                    reinit=True,
                )
                run_url = getattr(wandb.run, "url", None) or "(offline)"
                print(f"[MetricLogger] wandb -> {run_url}", flush=True)
            except Exception as exc:
                # Never let a wandb failure take down training. Degrade to TB+JSONL.
                print(f"[MetricLogger] wandb disabled: {exc!r}", flush=True)
                self._wandb = None

        self._start_time = time.time()
        print(f"[MetricLogger] JSONL -> {self._jsonl_path}", flush=True)

    def log(self, step: int, **metrics: float | int) -> None:
        """Write one record to JSONL, TensorBoard, and wandb (whichever are configured).

        Args:
            step: Integer step counter (typically env_steps).
            **metrics: Scalar metrics to log. Arrays are automatically reduced
                to their mean.
        """
        record = {"step": int(step), "wall_time": time.time() - self._start_time}
        for k, v in metrics.items():
            v_scalar = _to_scalar(v)
            record[k] = v_scalar
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(k, v_scalar, step)

        self._jsonl_fh.write(json.dumps(record) + "\n")

        if self._wandb is not None:
            # Log everything in one call so wandb synchronizes the step axis
            # across metrics. ``step=env_steps`` aligns with TB.
            try:
                self._wandb.log({k: v for k, v in record.items() if k != "step"}, step=step)
            except Exception as exc:
                print(f"[MetricLogger] wandb.log failed: {exc!r}", flush=True)
                self._wandb = None

    def close(self) -> None:
        """Flush and close all sinks."""
        with suppress(Exception):
            self._jsonl_fh.close()
        if self._tb_writer is not None:
            with suppress(Exception):
                self._tb_writer.flush()
                self._tb_writer.close()
        if self._wandb is not None:
            with suppress(Exception):
                self._wandb.finish()


def _to_scalar(v: Any) -> float:
    """Reduce array-like metrics to a Python float scalar."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore
    if np is not None and hasattr(v, "shape") and getattr(v, "shape", ()):
        return float(np.asarray(v).mean())
    return float(v)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce ``obj`` into a JSON-serializable structure."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)
