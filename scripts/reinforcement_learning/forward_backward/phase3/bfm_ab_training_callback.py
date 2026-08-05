# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bfm-ab-20260805 stage-4 A/B training health callback.

The stage-4 A/B arms run far past the horizons frozen in the v3 smoke
contract (smpl 102 iterations, g1 12 iterations), so they cannot carry the v3
receipt callback whose closed identities pin the collection cadence. This
callback records the SAME health-evidence class as the stage-3 crash-canary
callback (runner summary finiteness, actor-state finiteness, replay contract
errors and terminal overflow, device scope, terminal checkpoint hash) for
EVERY arm — smpl and g1, our-data and control — as immutable JSON records
without asserting any frozen cadence. Pre-registered in the F ledger row
bfm-ab-20260805; verification happens against these records, never inside
them.
"""

from pathlib import Path


def training_callback(
    *,
    stage: str,
    env_cfg: object,
    agent_cfg: object,
    configured_env_cfg: object,
    env: object,
    runner: object,
    log_dir: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """Record one launch or completion health snapshot for a stage-4 A/B arm."""
    from forward_backward.phase3 import motion_training_receipt as receipt

    preset = receipt._preset(configured_env_cfg)
    if receipt._preset(env_cfg) != preset:
        raise ValueError("Environment construction changed the selected motion preset.")
    if stage not in ("launch", "complete"):
        raise ValueError(f"A/B health callback only records launch/complete stages, got {stage!r}.")

    replay = runner.alg.replay
    replay_shape: dict[str, object] = {"replay_type": type(replay).__name__}
    for attribute in ("capacity_steps", "terminal_capacity_per_env", "capacity_transitions"):
        value = getattr(replay, attribute, None)
        if value is not None:
            replay_shape[attribute] = int(value)

    record: dict[str, object] = {
        "schema": "bfm_ab_20260805_health_v1",
        "stage": stage,
        "preset": preset,
        "seed": int(agent_cfg.seed),
        "collection": {
            "num_envs": int(env.num_envs),
            "steps_per_iteration": int(runner.cfg["num_steps_per_env"]),
            "max_iterations": int(agent_cfg.max_iterations),
            "random_action_transitions": int(runner.alg.random_action_transitions),
            "updates_per_group": int(runner.num_updates_per_iteration),
        },
        "replay_shape": replay_shape,
    }
    if stage == "complete":
        if checkpoint_path is None:
            raise ValueError("A/B health callback completion requires checkpoint_path.")
        record["runner"] = receipt._runner_summary(runner)
        record["learner"] = receipt._learner_snapshot(env, runner)
        record["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": receipt._file_sha256(checkpoint_path),
        }
    receipt._write_json_exclusive(log_dir / f"bfm_ab_{stage}.json", record)
