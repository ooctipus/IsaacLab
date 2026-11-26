# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
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


@pytest.mark.parametrize("other", ["--wandb_run_id", "--checkpoint"])
def test_wandb_run_name_and_other_sources_are_mutually_exclusive(other):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--wandb_run_name", "display", other, "internal"])


def test_wandb_run_name_uses_shared_checkpoint_policy_without_changing_output(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_4700.pt"
    checkpoint.touch()
    checkpoint = str(checkpoint)
    calls = []

    def _get_model_checkpoint(**kwargs):
        calls.append(kwargs)
        return checkpoint

    monkeypatch.setattr(cli_args_rsl_rl, "get_model_checkpoint", _get_model_checkpoint)
    args = _parser().parse_args(
        [
            "--logger",
            "wandb",
            "--log_project_name",
            "factory_manager3",
            "--wandb_run_name",
            "screwing-rb1-c08-s2-0805a",
        ]
    )
    args.device = "cpu"
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
    assert args.checkpoint == checkpoint
    assert cli_args_rsl_rl.resolve_rsl_rl_training_checkpoint(args, cfg, str(tmp_path)) == checkpoint
    assert cfg.experiment_name == "old"
    assert cfg.run_name == ""
    assert cfg.load_run == ".*"
    assert cfg.load_checkpoint == "model_.*.pt"
    assert cfg.run_id is None
    assert cfg.resume is False
