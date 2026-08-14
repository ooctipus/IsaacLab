# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl

pytestmark = pytest.mark.unit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli_args_rsl_rl.add_rsl_rl_args(parser)
    return parser


def test_wandb_run_name_is_a_checkpoint_source():
    args = _parser().parse_args(["--wandb_run_name", "screwing-rb1-c08-s2-0805a"])

    assert args.wandb_run_name == "screwing-rb1-c08-s2-0805a"
    assert args.wandb_run_id is None


def test_wandb_run_name_and_id_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--wandb_run_name", "display", "--wandb_run_id", "internal"])


def test_wandb_run_name_uses_resolved_id_for_resume(monkeypatch):
    checkpoint = "/tmp/models_tmp/factory_manager3/internal42/model_4700.pt"
    calls = []

    def _get_model_checkpoint(**kwargs):
        calls.append(kwargs)
        return checkpoint

    monkeypatch.setitem(
        sys.modules, "isaaclab.utils.wandb", SimpleNamespace(get_model_checkpoint=_get_model_checkpoint)
    )
    args = _parser().parse_args(
        [
            "--logger",
            "wandb",
            "--log_project_name",
            "factory_manager3",
            "--resume",
            "--wandb_run_name",
            "screwing-rb1-c08-s2-0805a",
        ]
    )
    cfg = SimpleNamespace(
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        experiment_name="old",
        run_name="",
        run_id=None,
        logger="tensorboard",
        wandb_project="old",
        neptune_project="old",
    )

    cli_args_rsl_rl.update_rsl_rl_cfg(cfg, args)

    assert calls == [
        {
            "run_id": None,
            "run_name": "screwing-rb1-c08-s2-0805a",
            "project": "factory_manager3",
            "checkpoint": None,
            "wandb_username": None,
        }
    ]
    assert cfg.experiment_name == "/tmp/models_tmp/factory_manager3"
    assert cfg.load_run == "internal42"
    assert cfg.load_checkpoint == "model_4700.pt"
    assert cfg.run_id == "internal42"
