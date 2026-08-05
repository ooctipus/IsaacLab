# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bfm-smokes-20260805 g1 crash-canary callback.

The stage-3 crash canary runs the g1_lafan_retarget arm for 600 iterations to
cross the historical iter-497 ``final_obs`` crash geometry live (the only prior
full g1 run died in ``ForwardBackward.process_env_step`` ->
``_as_observations(extras["final_obs"])`` at ~iter 497/206,250; the campaign
rsl_rl pin handles Mapping payloads). The frozen v3 smoke contract pins the g1
profile to exactly 12 iterations, so the canary cannot carry the v3 receipt
callback; this callback records the SAME health evidence class (runner summary
finiteness, actor-state finiteness, replay contract errors and terminal
overflow, device scope, terminal checkpoint hash) as immutable JSON records
without asserting the frozen collection cadence. Pre-registered in the F
ledger row bfm-smokes-20260805; verification happens against these records,
never inside them.
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
    """Record one launch or completion health snapshot for the crash canary."""
    from forward_backward.phase3 import motion_training_receipt as receipt

    preset = receipt._preset(configured_env_cfg)
    if receipt._preset(env_cfg) != preset:
        raise ValueError("Environment construction changed the selected motion preset.")
    if stage not in ("launch", "complete"):
        raise ValueError(f"Crash canary only records launch/complete stages, got {stage!r}.")

    record: dict[str, object] = {
        "schema": "bfm_smokes_20260805_g1_crash_canary_v1",
        "stage": stage,
        "preset": preset,
        "collection": {
            "num_envs": int(env.num_envs),
            "steps_per_iteration": int(runner.cfg["num_steps_per_env"]),
            "max_iterations": int(agent_cfg.max_iterations),
            "random_action_transitions": int(runner.alg.random_action_transitions),
            "updates_per_group": int(runner.num_updates_per_iteration),
        },
        "replay_capacity_steps": int(runner.alg.replay.capacity_steps),
        "replay_terminal_capacity_per_env": int(runner.alg.replay.terminal_capacity_per_env),
        "historical_crash_iteration": 497,
    }
    if stage == "complete":
        if checkpoint_path is None:
            raise ValueError("Crash canary completion requires checkpoint_path.")
        record["runner"] = receipt._runner_summary(runner)
        record["learner"] = receipt._learner_snapshot(env, runner)
        record["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": receipt._file_sha256(checkpoint_path),
        }
    receipt._write_json_exclusive(log_dir / f"bfm_canary_{stage}.json", record)
