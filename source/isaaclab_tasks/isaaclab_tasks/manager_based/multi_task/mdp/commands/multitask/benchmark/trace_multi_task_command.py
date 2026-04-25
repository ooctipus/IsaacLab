# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profile :class:`MultiTaskCommand` to surface dispatch bottlenecks.

Default mode (``--mode mock``) bypasses Isaac Sim entirely: builds the production
``MultiTaskCfg`` (8 tasks, mixed instant / tracking, multi-entity) on top of a
synthetic ``ManagerBasedRLEnv`` stub and feeds the readers pre-allocated random
tensors. The dispatch path under trace is byte-identical to production — only
the warp→torch reader handoff and PhysX step are missing, which are not what
the trace targets.

``--mode sim`` falls back to a real Isaac Sim env (``MultiTaskEnvCfg``) for
end-to-end timing including the reader path. Requires Isaac Sim to be properly
installed.

Phases recorded:

- ``MultiTaskCommand.update_command`` — full per-step path.
- ``MultiTaskCommand.dispatch`` — subclass-provided dispatch
  (``MultiTaskCommandReference._dispatch`` or
  ``MultiTaskCommandWarp._dispatch``, whichever the cfg selected).

Output:

- Console: top-N ops by self CPU + self CUDA, plus per-phase totals.
- ``--output`` directory: ``trace_<timestamp>.json`` for ``chrome://tracing``
  / `Perfetto <https://ui.perfetto.dev>`_.

Usage::

    # Mock mode (no Isaac Sim required):
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/\\
locomotion/position/mdp/commands/multitask/benchmark/trace_multi_task_command.py \\
        --num_envs 4096 --num_steps 200

    # Sim mode (real env):
    ./isaaclab.sh -p ... --mode sim --num_envs 1024 --num_steps 100
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

import torch

# Standard Anymal-C body / joint sets — matches the production scene so the
# read dispatch sees the same per-body / per-joint counts.
_ANYMAL_BODY_NAMES = [
    "base",
    "LF_HIP",
    "RF_HIP",
    "LH_HIP",
    "RH_HIP",
    "LF_THIGH",
    "RF_THIGH",
    "LH_THIGH",
    "RH_THIGH",
    "LF_SHANK",
    "RF_SHANK",
    "LH_SHANK",
    "RH_SHANK",
    "LF_FOOT",
    "RF_FOOT",
    "LH_FOOT",
    "RH_FOOT",
]
_ANYMAL_JOINT_NAMES = [
    "LF_HAA",
    "RF_HAA",
    "LH_HAA",
    "RH_HAA",
    "LF_HFE",
    "RF_HFE",
    "LH_HFE",
    "RH_HFE",
    "LF_KFE",
    "RF_KFE",
    "LH_KFE",
    "RH_KFE",
]


# ---------------------------------------------------------------------------
# Mock env (mirrors test_multi_task_command_mock.py helpers).
# ---------------------------------------------------------------------------


class _MockArticulation:
    """Stand-in for :class:`Articulation` satisfying ``SceneEntityCfg.resolve``."""

    def __init__(self, body_names: list[str], joint_names: list[str] | None = None):
        self.body_names = list(body_names)
        self.joint_names = list(joint_names) if joint_names else []
        self.num_bodies = len(self.body_names)
        self.num_joints = len(self.joint_names)
        self.fixed_tendon_names: list[str] = []
        self.num_fixed_tendons = 0

    @staticmethod
    def _find(names, patterns, preserve_order=False):
        if isinstance(patterns, str):
            patterns = [patterns]
        ids, matched = [], []
        for pat in patterns:
            rx = re.compile(pat)
            for i, n in enumerate(names):
                if rx.fullmatch(n) and i not in ids:
                    ids.append(i)
                    matched.append(n)
        return ids, matched

    def find_bodies(self, patterns, preserve_order=False):
        return self._find(self.body_names, patterns, preserve_order)

    def find_joints(self, patterns, preserve_order=False):
        return self._find(self.joint_names, patterns, preserve_order)

    def find_fixed_tendons(self, patterns, preserve_order=False):
        return [], []


class _MockScene:
    def __init__(self, entities: dict, num_envs: int, device: str):
        self._entities = entities
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        self.sensors = entities

    def keys(self):
        return self._entities.keys()

    def __getitem__(self, name):
        return self._entities[name]

    def __contains__(self, name):
        return name in self._entities


class _MockEnv:
    def __init__(self, num_envs: int, device: str, max_episode_length: int, scene):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.scene = scene
        self.common_step_counter = 0
        self.step_dt = 0.02


def _build_mock_synthetic_readers(num_envs: int, device: str) -> tuple:
    """Per-step mock readers — pre-allocated random source tensors per buffer kind.

    Real readers do a warp→torch handoff (zero-copy on GPU) plus, for body pos,
    an env-origin subtraction. Both are O(N · K) and fast; the trace's purpose
    is to expose dispatch / gather / scatter cost, not the reader. We swap them
    for constant-time tensor returns.
    """
    from isaaclab_tasks.manager_based.multi_task.mdp.commands.multitask.kernels_torch import BUFFER_KIND

    nb = len(_ANYMAL_BODY_NAMES)
    nj = len(_ANYMAL_JOINT_NAMES)
    by_kind = {
        int(BUFFER_KIND.JOINT_POS): torch.randn(num_envs, nj, device=device),
        int(BUFFER_KIND.JOINT_VEL): torch.randn(num_envs, nj, device=device),
        int(BUFFER_KIND.BODY_POS_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.BODY_QUAT_W): torch.nn.functional.normalize(
            torch.randn(num_envs, nb, 4, device=device), dim=-1
        ),
        int(BUFFER_KIND.BODY_LIN_VEL_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.BODY_ANG_VEL_W): torch.randn(num_envs, nb, 3, device=device),
        int(BUFFER_KIND.CONTACT_NET_FORCES_W): torch.randn(num_envs, nb, 3, device=device).abs() * 5.0,
    }

    def make_reader(kind: int):
        tensor = by_kind[kind]

        def reader(env, asset_name):
            return tensor

        return reader

    return tuple(make_reader(int(k)) for k in BUFFER_KIND)


def _build_mock_command(num_envs: int, device: str, use_warp: bool):
    """Construct a real :class:`MultiTaskCommand` against a mocked env + readers."""
    from unittest.mock import patch

    from isaaclab_tasks.manager_based.multi_task.mdp.commands.multitask import multi_task_command as mtc_mod
    from isaaclab_tasks.manager_based.multi_task.mdp.commands.multitask.multi_task_command import (
        MultiTaskCommand,
    )
    from isaaclab_tasks.manager_based.multi_task.multi_task_env_cfg import MultiTaskEnvCfg

    robot = _MockArticulation(body_names=_ANYMAL_BODY_NAMES, joint_names=_ANYMAL_JOINT_NAMES)
    contact_forces = _MockArticulation(body_names=_ANYMAL_BODY_NAMES)
    scene = _MockScene({"robot": robot, "contact_forces": contact_forces}, num_envs=num_envs, device=device)
    env = _MockEnv(num_envs=num_envs, device=device, max_episode_length=200, scene=scene)

    cfg = MultiTaskEnvCfg().commands.goal_point
    cfg.use_warp_dispatch = use_warp
    readers = _build_mock_synthetic_readers(num_envs, device)
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        cmd = MultiTaskCommand(cfg, env)
        # Stash the readers on the command so the per-step path uses them after
        # the patch context exits (the dispatch reads the module-level name).
        return cmd, env, readers, mtc_mod


# ---------------------------------------------------------------------------
# Profiler wiring.
# ---------------------------------------------------------------------------


def _wrap_phases(command_term):
    """Add ``record_function`` markers to the command term's hot path.

    Wraps :meth:`_update_command` and :meth:`_dispatch` on the instance
    (the subclass-specific dispatch). The trace labels stay the same
    whether the term is a reference or Warp instance — the label reflects
    the phase, not the implementation.
    """
    from torch.profiler import record_function

    orig_update = command_term._update_command
    orig_dispatch = command_term._dispatch

    def wrapped_update():
        with record_function("MultiTaskCommand.update_command"):
            return orig_update()

    def wrapped_dispatch(valid_slots):
        with record_function("MultiTaskCommand.dispatch"):
            return orig_dispatch(valid_slots)

    command_term._update_command = wrapped_update
    command_term._dispatch = wrapped_dispatch


def _print_phase_summary(prof, phase_names: list[str]) -> None:
    events = prof.key_averages()
    by_key = {ev.key: ev for ev in events}
    print("\n[PHASES] (totals over recorded steps)")
    print(f"  {'phase':<42s}  {'count':>6s}  {'cpu(ms)':>10s}  {'cuda(ms)':>10s}")
    for name in phase_names:
        ev = by_key.get(name)
        if ev is None:
            print(f"  {name:<42s}  {'—':>6s}  {'—':>10s}  {'—':>10s}")
            continue
        cpu_ms = ev.self_cpu_time_total / 1e3
        cuda_attr = "self_device_time_total" if hasattr(ev, "self_device_time_total") else "self_cuda_time_total"
        cuda_ms = getattr(ev, cuda_attr) / 1e3
        print(f"  {name:<42s}  {ev.count:>6d}  {cpu_ms:>10.2f}  {cuda_ms:>10.2f}")


def _profile_loop(cmd, env, readers, mtc_mod, args_cli) -> None:
    """Run the trace loop with the profiler attached."""
    from unittest.mock import patch

    from torch.profiler import ProfilerActivity, profile, schedule

    _wrap_phases(cmd)

    out_dir = pathlib.Path(args_cli.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = out_dir / f"trace_{timestamp}.json"

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and "cuda" in str(env.device):
        activities.append(ProfilerActivity.CUDA)

    active_steps = max(1, args_cli.num_steps - args_cli.warmup_steps)
    prof_schedule = schedule(wait=0, warmup=args_cli.warmup_steps, active=active_steps, repeat=1)

    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _step in range(args_cli.num_steps):
                cmd._update_command()
                env.episode_length_buf += 1
                prof.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    sort_cpu = "self_cpu_time_total"
    sort_cuda = "self_cuda_time_total" if torch.cuda.is_available() else sort_cpu

    print("\n" + "=" * 88)
    print(f"[TOP {args_cli.top_n} BY SELF CPU TIME]")
    print("=" * 88)
    print(prof.key_averages().table(sort_by=sort_cpu, row_limit=args_cli.top_n))

    if torch.cuda.is_available() and "cuda" in str(env.device):
        print("=" * 88)
        print(f"[TOP {args_cli.top_n} BY SELF CUDA TIME]")
        print("=" * 88)
        print(prof.key_averages().table(sort_by=sort_cuda, row_limit=args_cli.top_n))

    _print_phase_summary(
        prof,
        [
            "MultiTaskCommand.update_command",
            "MultiTaskCommand.dispatch",
        ],
    )

    print(f"\n[TRACE] Wrote Chrome trace to: {trace_path}")
    print("       Open chrome://tracing or https://ui.perfetto.dev and load this file.")


# ---------------------------------------------------------------------------
# Sim-mode glue (production env path; only built when --mode sim).
# ---------------------------------------------------------------------------


def _run_sim_mode(args_cli) -> None:
    """End-to-end profile against a real ``ManagerBasedRLEnv``. Requires Isaac Sim."""
    sys.argv = [sys.argv[0]]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_tasks.manager_based.multi_task.multi_task_env_cfg import MultiTaskEnvCfg

        env_cfg = MultiTaskEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            env.reset()
            cmd = env.command_manager.get_term("goal_point")

            class _NullReaderTuple:
                pass

            _profile_loop(cmd, env, _NullReaderTuple(), None, args_cli)
        finally:
            env.close()
    finally:
        simulation_app.close()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MultiTaskCommand dispatch.")
    parser.add_argument(
        "--mode",
        choices=("mock", "sim"),
        default="mock",
        help="mock = synthetic env (no Isaac Sim); sim = real ManagerBasedRLEnv.",
    )
    parser.add_argument("--num_envs", type=int, default=4096, help="Parallel envs.")
    parser.add_argument("--num_steps", type=int, default=200, help="Total iterations (warmup + active).")
    parser.add_argument("--warmup_steps", type=int, default=40, help="Steps to skip before recording.")
    parser.add_argument("--top_n", type=int, default=25, help="Rows in the CPU / CUDA self-time tables.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(pathlib.Path.cwd() / "multitask_trace"),
        help="Directory for the Chrome trace JSON.",
    )
    parser.add_argument(
        "--mock_device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for mock-mode tensors.",
    )
    parser.add_argument(
        "--use_warp",
        action="store_true",
        help="Profile the Warp mega-kernel path (cfg.use_warp_dispatch=True).",
    )

    # Only the sim path needs AppLauncher's CLI surface.
    if "--mode" in sys.argv and "sim" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    args_cli, _ = parser.parse_known_args()
    return args_cli


def main() -> None:
    args_cli = _parse_args()

    if args_cli.mode == "sim":
        _run_sim_mode(args_cli)
        return

    torch.manual_seed(0)
    cmd, env, readers, mtc_mod = _build_mock_command(
        args_cli.num_envs, args_cli.mock_device, use_warp=args_cli.use_warp
    )
    print(f"[TRACE] Dispatch path: {'Warp mega-kernel' if args_cli.use_warp else 'PyTorch reference'}")
    _profile_loop(cmd, env, readers, mtc_mod, args_cli)


if __name__ == "__main__":
    main()
