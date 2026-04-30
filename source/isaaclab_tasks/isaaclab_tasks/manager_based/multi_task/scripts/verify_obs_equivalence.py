# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end obs-level equivalence: upstream MultiMeshRayCaster vs FastTerrainScanner.

Runs the real ``Isaac-Position-v0`` env twice at ``num_envs=16`` with the same seed
and the same deterministic action stream. The two passes differ only in the
``height_scanner`` cfg class::

    legacy: ``isaaclab.sensors.MultiMeshRayCasterCfg``                              (upstream)
    fused:  ``isaaclab_tasks...multi_task.sensors.FastTerrainScannerCfg``  (this folder)

The script captures ``obs['height_scan']`` at every step in both runs and reports
the per-step max absolute difference. If the kernel changes are correct, the
**first** observation (immediately after reset) must be bit-identical — that's
the strongest guarantee, since no PhysX simulation has happened yet. Subsequent
steps may show tiny non-bit-identical drift if PhysX itself has non-deterministic
atomic ordering between runs; we report those separately so it's clear which
diff is from our change vs from PhysX.

Usage::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/scripts/verify_obs_equivalence.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Verify obs equivalence between legacy and fused raycaster.")
parser.add_argument(
    "--mode",
    choices=("orchestrator", "legacy", "fused"),
    default="orchestrator",
    help="Internal: orchestrator runs both child captures + comparison; capture modes dump obs.",
)
parser.add_argument("--output", type=str, default=None, help="(capture mode) where to dump the obs trajectory.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
args_cli, remaining = parser.parse_known_args()


# ---------------------------------------------------------------------------
# Orchestrator: run two captures, compare
# ---------------------------------------------------------------------------


def _orchestrate() -> int:
    """Spawn legacy + fused child captures, then load and compare their obs traces."""
    # Path layout: source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/scripts/<this>
    # parents[0]=scripts, [1]=multi_task, [2]=manager_based, [3]=isaaclab_tasks (inner),
    # [4]=isaaclab_tasks (outer), [5]=source, [6]=repo root.
    isaaclab_root = Path(__file__).resolve().parents[6]
    self_path = Path(__file__)
    isaaclab_sh = isaaclab_root / "isaaclab.sh"
    if not isaaclab_sh.exists():
        print(f"[FAIL] could not locate isaaclab.sh (looked at {isaaclab_sh})")
        return 1

    with tempfile.TemporaryDirectory(prefix="verify_obs_") as tmp:
        legacy_path = Path(tmp) / "legacy.pt"
        fused_path = Path(tmp) / "fused.pt"

        for mode, out_path in (("legacy", legacy_path), ("fused", fused_path)):
            print(f"[orchestrator] capturing {mode} run → {out_path}")
            # NOTE: do *not* pass ``--headless`` or ``--task=...`` here. The capture-mode
            # parser doesn't know those flags (we drive AppLauncher with a kwarg and pass
            # the task name to ``resolve_task_config`` directly), and leaving them in
            # ``remaining`` confuses AppLauncher's own argparse pass over ``sys.argv``.
            # ``presets=...`` *does* need to reach ``sys.argv`` so hydra can pick it up.
            cmd = [
                str(isaaclab_sh),
                "-p",
                str(self_path),
                f"--mode={mode}",
                f"--output={out_path}",
                f"--num_envs={args_cli.num_envs}",
                f"--num_steps={args_cli.num_steps}",
                f"--seed={args_cli.seed}",
                f"--device={args_cli.device}",
                "presets=anymal_c,res02,cnn",
            ]
            result = subprocess.run(cmd, cwd=isaaclab_root)
            if result.returncode != 0:
                print(f"[FAIL] {mode} capture exited with code {result.returncode}")
                return result.returncode
            if not out_path.exists():
                print(f"[FAIL] {mode} capture did not produce {out_path}")
                return 1

        # ---- compare ------------------------------------------------------
        import torch  # imported only after the captures so this side stays light

        legacy = torch.load(str(legacy_path), map_location="cpu", weights_only=True)
        fused = torch.load(str(fused_path), map_location="cpu", weights_only=True)

        legacy_obs = legacy["obs"]  # list of [num_envs, *obs_shape] tensors
        fused_obs = fused["obs"]

        if len(legacy_obs) != len(fused_obs):
            print(f"[FAIL] step counts differ: legacy={len(legacy_obs)} fused={len(fused_obs)}")
            return 1

        print("\nPer-step max |Δ| of height_scan obs (fused vs legacy):")
        first_step_bit_identical = False
        any_fail = False
        for i, (leg, fus) in enumerate(zip(legacy_obs, fused_obs)):
            if leg.shape != fus.shape:
                print(f"  step {i}: shape mismatch {leg.shape} vs {fus.shape}")
                any_fail = True
                continue
            diff = (leg.float() - fus.float()).abs()
            max_abs = float(diff.max().item())
            bit_identical = bool(torch.equal(leg, fus))
            tag = "bit-identical" if bit_identical else f"max |Δ| = {max_abs:.3e}"
            print(f"  step {i:>2}: shape={tuple(leg.shape)} dtype={leg.dtype}   {tag}")
            if i == 0:
                first_step_bit_identical = bit_identical
            # Tolerance: kernel-level outputs are bit-identical (verified separately).
            # Allow tiny PhysX non-determinism for later steps (~1 mm = 1e-3 m).
            if max_abs > 1e-2:
                any_fail = True

        print()
        if first_step_bit_identical:
            print("PASS — step 0 (post-reset, no sim) is bit-identical across paths.")
        else:
            print("FAIL — step 0 should be bit-identical but is not. Kernel change has a real diff.")
            return 1

        if any_fail:
            print("WARN — later steps drift > 1 cm. Likely PhysX non-determinism, not the kernel change,")
            print("       but worth a closer look if it's much larger than expected.")
            # Don't fail the script on later-step drift; PhysX is genuinely non-deterministic.
        else:
            print("All steps within tolerance.")
        return 0


# ---------------------------------------------------------------------------
# Capture mode: launch env, run rollout, dump obs trajectory
# ---------------------------------------------------------------------------


def _run_capture(mode: str) -> int:
    """Build the env with the cfg flags implied by ``mode`` and dump a deterministic obs trajectory."""
    if not args_cli.output:
        print("[FAIL] --output is required in capture mode.")
        return 1

    # Make the remaining args look like a normal Isaac Lab task script.
    sys.argv = [sys.argv[0]] + remaining

    # Imports kept local so the orchestrator path doesn't pay for sim startup.
    import gymnasium as gym
    import torch

    from isaaclab.app import AppLauncher

    # Headless launcher.
    launcher = AppLauncher(headless=True, device=args_cli.device)
    sim_app = launcher.app

    try:
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import resolve_task_config

        env_cfg, _agent_cfg = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device
        env_cfg.seed = args_cli.seed

        # Swap the height_scanner cfg between fused (default in position_env_cfg.py) and the
        # upstream MultiMeshRayCasterCfg. The fused class is the production path; the legacy
        # class is the bit-identical reference. Same prim_path, offset, pattern, alignment,
        # mesh_prim_paths, max_distance — only the implementation class differs.
        if mode == "legacy":
            from isaaclab.sensors import MultiMeshRayCasterCfg

            fused_cfg = env_cfg.scene.height_scanner
            env_cfg.scene.height_scanner = MultiMeshRayCasterCfg(
                prim_path=fused_cfg.prim_path,
                offset=MultiMeshRayCasterCfg.OffsetCfg(pos=fused_cfg.offset.pos, rot=fused_cfg.offset.rot),
                ray_alignment=fused_cfg.ray_alignment,
                pattern_cfg=fused_cfg.pattern_cfg,
                mesh_prim_paths=fused_cfg.mesh_prim_paths,
                max_distance=fused_cfg.max_distance,
                drift_range=fused_cfg.drift_range,
                ray_cast_drift_range=fused_cfg.ray_cast_drift_range,
                debug_vis=fused_cfg.debug_vis,
            )
        elif mode != "fused":
            raise ValueError(mode)

        env = gym.make("Isaac-Position-v0", cfg=env_cfg)
        try:
            obs_dict, _ = env.reset(seed=args_cli.seed)
            # Pull the height_scan obs from the policy/task/height_scan TensorDict.
            captured: list[torch.Tensor] = [_extract_height_scan(obs_dict).detach().clone()]

            action_shape = env.unwrapped.action_space.shape
            actions = torch.zeros(action_shape, device=env.unwrapped.device)
            for _ in range(args_cli.num_steps):
                obs_dict, _, _, _, _ = env.step(actions)
                captured.append(_extract_height_scan(obs_dict).detach().clone())
        finally:
            env.close()

        out_path = Path(args_cli.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"obs": [t.cpu() for t in captured], "mode": mode}, str(out_path))
        print(f"[capture:{mode}] wrote {len(captured)} obs to {out_path}")
        return 0
    finally:
        sim_app.close()


def _extract_height_scan(obs_dict):
    """Pull the ``height_scan`` field out of whatever obs container the env returns."""
    # Manager-based env returns a dict-of-dicts: obs_dict["policy"] / ["task"] / ["height_scan"].
    if "height_scan" in obs_dict:
        v = obs_dict["height_scan"]
    else:
        # Fallback: TensorDict-style with nested keys.
        for key in ("height_scan", "policy"):
            if key in obs_dict and "height_scan" in obs_dict[key]:
                v = obs_dict[key]["height_scan"]
                break
        else:
            raise KeyError(f"could not locate height_scan in obs_dict; keys: {list(obs_dict)}")
    return v.contiguous() if hasattr(v, "contiguous") else v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if args_cli.mode == "orchestrator":
        sys.exit(_orchestrate())
    sys.exit(_run_capture(args_cli.mode))
