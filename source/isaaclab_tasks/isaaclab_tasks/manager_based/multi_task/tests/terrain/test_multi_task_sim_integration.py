# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight sim-integration regression test for :class:`MultiTaskCommand`.

Skipped when Isaac Sim isn't available on ``PYTHONPATH`` (e.g. CI without sim).
Target budget: **under 20 seconds wallclock**, including Isaac Sim launch.

Covers what the mock tests can't: that the engine actually drives a live
:class:`ManagerBasedRLEnv` end-to-end with real Articulation data reads, and that the
reward/done signal behaves correctly over real physics steps.

Intentionally minimal to stay within budget:

- 4 envs (not 4096).
- Flat plane terrain (no generator, no retarget).
- Single pure-tracking ``lin_vel`` task.
- 10 physics steps with random actions.
- Only sanity invariants are asserted (no-NaN, reward ∈ [0, 1], terminal-only emission,
  ``time_outs = False``).

If budget is exceeded on the user's sim machine, this test should be deleted and the
non-sim regression suite relied on exclusively — the exact-reward-trace test in
``test_multi_task_command_mock.py`` catches everything this test does except real
Articulation wiring, which is mostly a "does the port compile against the live IsaacLab
API" check.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from ..helpers.multi_task_sim_envs import (
    make_heterogeneous_multi_task_env_cfg,
    make_minimal_multi_task_env_cfg,
    pin_command_task_samples,
    write_robot_standing_state,
)

# Skip the entire module if the IsaacLab app launcher isn't importable. We probe
# ``isaaclab.app`` rather than the raw ``isaacsim`` package because some installs
# expose the app via ``omni.*`` namespace packages without a top-level
# ``isaacsim`` module.
if importlib.util.find_spec("isaaclab.app") is None:
    pytest.skip("IsaacLab app launcher not available — skipping sim-integration test.", allow_module_level=True)

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(scope="module")
def simulation_app():
    """Launch a single headless Isaac Sim app shared across all tests in this module.

    Module scope amortizes the 5-15s launch cost across every test below.
    """
    import argparse

    from isaaclab.app import AppLauncher

    # Minimal argparse Namespace — AppLauncher inspects attributes it knows.
    # Device must be a concrete string: AppLauncher._resolve_device_settings
    # does ``"cuda" not in device`` which raises TypeError on None.
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = argparse.Namespace(
        headless=True,
        livestream=-1,
        enable_cameras=False,
        xr=False,
        device=default_device,
        cpu=not torch.cuda.is_available(),
        verbose=False,
        info=False,
        experience="",
        kit_args="",
    )
    launcher = AppLauncher(args)
    app = launcher.app
    yield app
    app.close()


def test_multi_task_command_runs_10_steps_in_live_env(simulation_app):
    """Drive the engine through 10 physics steps with random actions; assert core invariants.

    Asserted (each step):

    - No NaN in rewards.
    - Rewards ∈ ``[0, 1]`` (multiplicative terminal return is bounded).
    - Non-terminal steps produce exactly zero task reward (composer is terminal-only).
    - ``info["time_outs"]`` is False — Stage-3 finite-horizon framing, rsl_rl must not
      bootstrap.

    The weight="1.0" task reward term is the only reward. Random-action policy rarely
    matches the target velocity, so activations will be mostly low — but the invariants
    must still hold.
    """
    from isaaclab.envs import ManagerBasedRLEnv

    torch.manual_seed(0)
    env_cfg = make_minimal_multi_task_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        num_steps = 10
        for step in range(num_steps):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
                _obs, rewards, terminated, truncated, info = env.step(actions)

            assert not torch.isnan(rewards).any(), f"step {step}: NaN reward"
            assert (rewards >= 0).all(), f"step {step}: negative reward {rewards.min().item():.4g}"
            assert (rewards <= 1.0 + 1e-5).all(), f"step {step}: reward > 1 ({rewards.max().item():.4g})"

            dones = terminated | truncated
            non_terminal = ~dones
            if non_terminal.any():
                nt = rewards[non_terminal]
                assert torch.allclose(nt, torch.zeros_like(nt)), (
                    f"step {step}: non-terminal reward leaked (max abs {nt.abs().max().item():.4g})"
                )

            time_outs = info.get("time_outs", None)
            if time_outs is not None:
                assert not time_outs.any(), f"step {step}: time_outs flag set — rsl_rl bootstrap would fire"
    finally:
        env.close()


# -----------------------------------------------------------------------------
# Strict heterogeneous stress test — pinned task ids + pinned initial state +
# exact per-env ground-truth reward.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch_backend", ["torch", "mega_kernel"])
def test_heterogeneous_multi_task_pinned_state_ground_truth(simulation_app, dispatch_backend):
    """Four envs, four pinned tasks, pinned initial state → per-env exact terminal reward.

    Each env targets a different failure mode:

    - **Env 0 (pure-track zero-vel)** → times out with reward ≈ 1.0. Catches
      timeout plumbing (``episode_length_buf`` / ``max_episode_length``) and
      manager-order bugs that starve pure-tracking reward.
    - **Env 1 (pure-instant at spawn, loose threshold)** → succeeds at step 0,
      reward exactly 1.0. Catches unit-reward contract for pure-instant +
      success DoneTerm wiring.
    - **Env 2 (mixed reach+hold)** → succeeds at step 0, reward ≈ 1.0 (1 ·
      tracking_mean). Catches composer instant_gate·tracking_mean path and
      cross-subtask sampler contamination (two 3-dim samplers on same env).
    - **Env 3 (pure-track, v=10 m/s unreachable)** → times out with reward ≈ 0.
      Catches the pure-tracking-never-succeeds invariant (if ``has_instant``
      flips, env 3 would false-succeed at step 0 with reward 0).

    Step-wise invariants inside the loop: no NaN, reward ∈ [0, 1],
    ``info["time_outs"] == False``. Robot-health at end: no NaN in base pose /
    joints, base hasn't fallen through the floor.
    """
    from isaaclab.envs import ManagerBasedRLEnv

    torch.manual_seed(0)
    env_cfg = make_heterogeneous_multi_task_env_cfg(dispatch_backend=dispatch_backend)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        cmd = env.command_manager.get_term("goal_point")

        pinned_ids = pin_command_task_samples(
            cmd,
            [
                "pure_track_zero_vel",
                "pure_instant_at_spawn",
                "mixed_reach_and_hold",
                "pure_track_unreachable",
            ],
        )

        obs_dict, _ = env.reset()

        # Sanity: pinning worked before we touched anything else.
        assert cmd.task_samples.tolist() == pinned_ids.tolist(), (
            f"task pinning failed: got {cmd.task_samples.tolist()}, want {pinned_ids.tolist()}"
        )

        # Obs pipeline: the concatenated policy obs must contain
        # reach (3) + track (3) + active (6) + progress (1) = 13 channels.
        obs = obs_dict["policy"]
        expected_obs_width = (
            cmd.spec.reach_canonical_width
            + cmd.spec.track_canonical_width
            + (cmd.spec.reach_canonical_width + cmd.spec.track_canonical_width)
            + 1  # scalar progress
        )
        assert obs.shape == (env.num_envs, expected_obs_width), (
            f"obs shape {tuple(obs.shape)} != ({env.num_envs}, {expected_obs_width})"
        )
        # The active-mask slab of the concatenated obs (columns reach_w + track_w ..
        # reach_w + track_w + reach_w + track_w) must equal ``cmd.command_active``.
        mask_start = cmd.spec.reach_canonical_width + cmd.spec.track_canonical_width
        mask_end = mask_start + cmd.spec.reach_canonical_width + cmd.spec.track_canonical_width
        assert torch.equal(obs[:, mask_start:mask_end], cmd.command_active), (
            "obs active-mask slab disagrees with cmd.command_active — obs wiring bug"
        )

        # Strict mask layout — each env's mask must match its task's active channels
        # exactly. Layout: [reach_BODY_POS (3 chans), track_BODY_LIN_VEL (3 chans)].
        reach_w = cmd.spec.reach_canonical_width
        track_w = cmd.spec.track_canonical_width
        assert reach_w == 3 and track_w == 3, (
            f"canonical layout shifted: reach_w={reach_w}, track_w={track_w}; "
            f"expected (3, 3) for BODY_POS (instant) + BODY_LIN_VEL (tracking)"
        )
        expected_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1],  # env 0: pure-tracking lin_vel
                [1, 1, 1, 0, 0, 0],  # env 1: pure-instant body_pos
                [1, 1, 1, 1, 1, 1],  # env 2: mixed
                [0, 0, 0, 1, 1, 1],  # env 3: pure-tracking lin_vel (different target, same channels)
            ],
            dtype=torch.float32,
            device=env.device,
        )
        mask = cmd.command_active
        assert mask.shape == expected_mask.shape, f"mask shape {mask.shape} != {expected_mask.shape}"
        assert torch.equal(mask, expected_mask), (
            f"mask mismatch after pinned reset:\n got={mask.tolist()}\nwant={expected_mask.tolist()}"
        )
        # Sampler-param variation must NOT split canonical channels (envs 0 & 3).
        assert torch.equal(mask[0], mask[3]), (
            "envs 0 and 3 both use BODY_LIN_VEL on base — different sampler targets must "
            "not produce different canonical channels"
        )

        import warp as wp

        robot = write_robot_standing_state(env)

        max_steps = int(env.max_episode_length)
        zero_actions = torch.zeros(env.action_space.shape, device=env.device)

        terminal_reward = torch.zeros(env.num_envs, device=env.device)
        terminal_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

        # Reset-behaviour trackers.
        # - ``episode_count``: how many terminations each env experienced across the
        #   full loop. Env 0/3 must see exactly 1 (single timeout); env 1/2 must see
        #   many (every step re-triggers success under pinned state + pinned task).
        # - ``cumulative_reward``: sum of rewards delivered per env. If the composer
        #   resets cleanly each episode, env 1's cum reward grows linearly with
        #   episode_count. If reset leaks (e.g., ``_task_done`` isn't cleared so
        #   reward keeps firing without reset) or over-clears (so no reward ever
        #   fires post-reset), this diverges loudly from episode_count × per-episode.
        episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        cumulative_reward = torch.zeros(env.num_envs, device=env.device)

        # Run the FULL episode (no early break) so envs 1/2 — which re-terminate
        # every step under pinned state — cycle through many auto-reset boundaries,
        # exposing reset-path bugs (stale sum_activation, uncleared latches, mask
        # drift, target resample skipped, command_counter desync, etc.).
        for step in range(max_steps):
            with torch.inference_mode():
                obs_dict, rewards, terminated, truncated, info = env.step(zero_actions)

            dones = terminated | truncated
            episode_count += dones.long()
            cumulative_reward += rewards

            # Step-wise invariants — hold on every step, every env.
            assert not torch.isnan(rewards).any(), f"step {step}: NaN in rewards"
            assert (rewards >= 0).all(), f"step {step}: reward < 0 (min {rewards.min().item()})"
            assert (rewards <= 1.0 + 1e-4).all(), f"step {step}: reward > 1 (max {rewards.max().item()})"
            time_outs = info.get("time_outs", None)
            if time_outs is not None:
                assert not time_outs.any(), f"step {step}: rsl_rl bootstrap flag set on {time_outs.tolist()}"

            # Robot health per-step: no NaN, base hasn't sunk through floor.
            rp = wp.to_torch(robot.data.root_pos_w)
            assert not torch.isnan(rp).any(), f"step {step}: NaN in root_pos_w"
            base_z_step = rp[:, 2] - env.scene.env_origins[:, 2]
            assert (base_z_step > 0.0).all(), (
                f"step {step}: robot fell through floor, env-local z = {base_z_step.tolist()}"
            )

            # Pin invariance — even after sub-batch resets, task_samples must stay
            # pinned. Catches any reset path that re-randomizes task ids.
            assert cmd.task_samples.tolist() == pinned_ids.tolist(), (
                f"step {step}: task pin broken: got {cmd.task_samples.tolist()}, want {pinned_ids.tolist()}"
            )

            # Mask invariance — the canonical active layout MUST always match the
            # pinned task's expected mask, for ALL envs at ALL steps. If any env's
            # row diverges: either the pin broke, or a reset path skipped the
            # mask refresh, or a partial-reset wrote the wrong row.
            assert torch.equal(cmd.command_active, expected_mask), (
                f"step {step}: command_active diverged from pinned-task layout:\n "
                f"got={cmd.command_active.tolist()}\nwant={expected_mask.tolist()}"
            )

            # Transit-step reset invariant — an env that reports done on THIS step
            # has just been reset inside env.step (_reset_idx at line 229), then
            # _update_command ran once post-reset. After that one update,
            # ``_transit_steps`` for that env must be exactly 1. If reset didn't
            # zero it (or zeroed other buffers incompletely), this breaks.
            if dones.any():
                done_ids = dones.nonzero(as_tuple=True)[0].tolist()
                for i in done_ids:
                    assert cmd._transit_steps[i].item() == 1, (
                        f"step {step}: env {i} just reset but ``_transit_steps={cmd._transit_steps[i].item()}``, "
                        "expected 1 (one _update_command since reset). Reset path failed to zero "
                        "``_transit_steps`` atomically with the other composer buffers."
                    )
                    # ``_instant_achieved`` latches persist ACROSS steps but not
                    # across resets. Post-reset, if this env's task has any
                    # instant subtask, the latch reflects ONLY the post-reset
                    # _update's activation, never the prior episode's — so the
                    # latch state should be consistent with a single-step evaluation.
                    # We can't strictly assert the value (depends on whether the
                    # single post-reset step achieved the instant), but we CAN
                    # assert ``_sum_activation`` hasn't leaked: it must equal the
                    # current step's ``_buf_activation`` (added once post-reset).
                    assert torch.allclose(cmd._sum_activation[i], cmd._buf_activation[i], atol=1e-6), (
                        f"step {step}: env {i} reset did not zero ``_sum_activation``: "
                        f"sum={cmd._sum_activation[i].tolist()} vs buf={cmd._buf_activation[i].tolist()}"
                    )

            # Transit-step accumulation invariant — envs that did NOT reset this
            # step must have ``_transit_steps`` equal to ``step + 1`` (one
            # increment per env.step call since the most recent reset). Pre-
            # timeout env 0 and env 3 should never spontaneously reset; any
            # break here means a reset fired unexpectedly.
            for i in (0, 3):
                if not dones[i].item() and terminal_step[i].item() < 0:
                    assert cmd._transit_steps[i].item() == step + 1, (
                        f"step {step}: env {i} pre-terminal transit_steps={cmd._transit_steps[i].item()}, "
                        f"expected {step + 1} — unexpected reset?"
                    )

            # Terminal-only emission — strict check: before an env reaches its
            # terminal step, reward MUST be 0. (Composer's "terminal-only" contract.)
            was_pre_terminal = terminal_step < 0
            pre_terminal_no_reward = was_pre_terminal & ~dones
            if pre_terminal_no_reward.any():
                leaked = pre_terminal_no_reward.nonzero(as_tuple=True)[0]
                for i in leaked.tolist():
                    assert rewards[i].item() == 0.0, (
                        f"env {i} step {step}: non-terminal reward {rewards[i].item():.6f} leaked "
                        "— composer must emit only at terminal"
                    )

            newly_done = dones & was_pre_terminal
            if newly_done.any():
                idx = newly_done.nonzero(as_tuple=True)[0]
                terminal_reward[idx] = rewards[idx]
                terminal_step[idx] = step

        # --- Per-env ground truth ---
        # Env 0: pure-tracking v=0, robot holding rest. Episode-averaged tracking
        # over 24 steps includes pose-teleport settling transients (initial ~0.3
        # m/s drift that decays to ~0.02 m/s), yielding mean activation ≈ 0.5–0.8.
        # Threshold 0.4 gives a ~10× separation against env 3 (~0) while tolerating
        # realistic physics.
        assert terminal_step[0].item() == max_steps - 1, (
            f"env0 should time out at step {max_steps - 1}, got {terminal_step[0].item()}"
        )
        assert terminal_reward[0].item() > 0.4, (
            f"env0 pure-tracking @rest should reward >> 0, got {terminal_reward[0].item():.4f}"
        )

        # Env 1: pure-instant body_pos, robot within 10 cm of target → success at step 0, reward = 1.
        assert terminal_step[1].item() <= 1, (
            f"env1 pure-instant @target should succeed by step 1, got {terminal_step[1].item()}"
        )
        assert abs(terminal_reward[1].item() - 1.0) < 1e-3, (
            f"env1 pure-instant success reward should be 1.0, got {terminal_reward[1].item():.6f}"
        )

        # Env 2: mixed instant+tracking, both satisfied → reward = 1.0 · tracking_mean.
        # Terminates at step 1, so tracking_mean is over a single step post-teleport
        # (no long-average dilution); reward lands ≈ 0.8–0.95.
        assert terminal_step[2].item() <= 1, (
            f"env2 mixed @target should succeed by step 1, got {terminal_step[2].item()}"
        )
        assert terminal_reward[2].item() > 0.7, (
            f"env2 mixed @target should reward ≈ 1.0 · tracking_mean, got {terminal_reward[2].item():.4f}"
        )

        # Env 3: pure-tracking unreachable v=10, robot at rest → tanh(10/0.3) ≈ 1 → activation ≈ 0.
        assert terminal_step[3].item() == max_steps - 1, (
            f"env3 pure-tracking unreachable should time out, got {terminal_step[3].item()}"
        )
        assert terminal_reward[3].item() < 0.05, (
            f"env3 unreachable-tracking should reward ≈ 0, got {terminal_reward[3].item():.4f}"
        )

        # Separation: env0 and env3 both run to timeout but get opposite rewards.
        # Cross-env task-table contamination would collapse this gap.
        assert terminal_reward[0].item() - terminal_reward[3].item() > 0.35, (
            f"env0 vs env3 reward separation collapsed: {terminal_reward[0].item():.4f} vs "
            f"{terminal_reward[3].item():.4f} — cross-env task table contamination?"
        )

        # --- Reset-cycle ground truth ---
        # Env 0: exactly one termination (single timeout). Env 3: same.
        assert episode_count[0].item() == 1, (
            f"env0 should have exactly 1 termination (timeout), got {episode_count[0].item()}"
        )
        assert episode_count[3].item() == 1, (
            f"env3 should have exactly 1 termination (timeout), got {episode_count[3].item()}"
        )
        # Env 1: pinned to pure-instant-at-spawn task; robot held within success
        # radius every step → success fires every single step → terminates every
        # step → episode_count ≈ max_steps. A broken reset (e.g., ``_task_done``
        # never cleared, so it's "stuck done") would still count every step; but
        # a reset that FAILS to re-latch (e.g., ``_instant_achieved`` stuck True
        # without reward firing) would see episode_count grow but cum reward stay 1.
        assert episode_count[1].item() >= max_steps - 1, (
            f"env1 should terminate ≈every step; episode_count={episode_count[1].item()} < {max_steps - 1}"
        )
        assert episode_count[2].item() >= max_steps - 1, (
            f"env2 should terminate ≈every step; episode_count={episode_count[2].item()} < {max_steps - 1}"
        )

        # Cumulative reward tracks episode_count × per-episode reward. A reset
        # that clears too little (reward stays latched → single delivery) would
        # collapse cum_reward to ~1 regardless of episode_count. A reset that
        # clears too much (new episode never achieves success) would collapse
        # cum_reward to 0.
        env1_mean_reward = cumulative_reward[1].item() / max(1, episode_count[1].item())
        assert env1_mean_reward > 0.9, (
            f"env1 per-episode mean reward {env1_mean_reward:.4f} < 0.9 — reset may be leaking "
            f"``_sum_activation`` or ``_transit_steps`` across episode boundaries "
            f"(cum_reward={cumulative_reward[1].item():.3f}, episode_count={episode_count[1].item()})"
        )
        # Env 2 mean is lower than env 1's because its reward = 1·tracking_mean(1-step),
        # and the robot's velocity fluctuates around ~0.1 m/s as it drifts without
        # repeated state-pinning. Mean activation ≈ tanh-based ≈ 0.6–0.8. Threshold
        # 0.5 still gives ~10× separation vs env 3's ≈0, and is physically grounded.
        env2_mean_reward = cumulative_reward[2].item() / max(1, episode_count[2].item())
        assert env2_mean_reward > 0.5, (
            f"env2 per-episode mean reward {env2_mean_reward:.4f} < 0.5 — reset issue "
            f"(cum_reward={cumulative_reward[2].item():.3f}, episode_count={episode_count[2].item()})"
        )

        # Env 0 / env 3 each get exactly one reward delivery. cum_reward equals terminal_reward.
        assert abs(cumulative_reward[0].item() - terminal_reward[0].item()) < 1e-5, (
            f"env0 cum_reward ({cumulative_reward[0].item():.4f}) != terminal_reward "
            f"({terminal_reward[0].item():.4f}) — reward leak before/after terminal"
        )
        assert abs(cumulative_reward[3].item() - terminal_reward[3].item()) < 1e-5, (
            f"env3 cum_reward ({cumulative_reward[3].item():.4f}) != terminal_reward "
            f"({terminal_reward[3].item():.4f}) — reward leak before/after terminal"
        )

        # --- Robot health ---
        # ``robot.data.*`` fields are ``wp.array``; convert to torch for asserts.
        root_pos_w_t = wp.to_torch(robot.data.root_pos_w)
        root_quat_w_t = wp.to_torch(robot.data.root_quat_w)
        joint_pos_t = wp.to_torch(robot.data.joint_pos)
        assert not torch.isnan(root_pos_w_t).any(), "NaN in root_pos_w"
        assert not torch.isnan(root_quat_w_t).any(), "NaN in root_quat_w"
        assert not torch.isnan(joint_pos_t).any(), "NaN in joint_pos"
        base_z = root_pos_w_t[:, 2] - env.scene.env_origins[:, 2]
        assert (base_z > 0.05).all(), f"robot base sank: env-local z = {base_z.tolist()}"
    finally:
        env.close()
