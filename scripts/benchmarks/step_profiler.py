# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
from prettytable import PrettyTable
from collections import defaultdict


class EnvStepProfiler:
    def __init__(self):
        self._in_step = False
        self.hist_ns = defaultdict(list)
        self._wrapped = []

    def start_step(self):
        self._in_step = True

    def end_step(self):
        self._in_step = False

    def wrap(self, obj, method_name, label=None):
        """Monkey-patch obj.method_name to record wall clock time into hist_ns[label]."""
        assert hasattr(obj, method_name), f"{obj} does not has method {method_name}"
        assert callable(getattr(obj, method_name)), f"{obj}'s method {method_name} is not callable"
        orig = getattr(obj, method_name)

        def wrapped(*a, **kw):
            t0 = time.perf_counter_ns()
            try:
                return orig(*a, **kw)
            finally:
                if self._in_step:
                    self.hist_ns[label or method_name].append(time.perf_counter_ns() - t0)

        setattr(obj, method_name, wrapped)
        self._wrapped.append((obj, method_name, orig))

    def wrap_env_step(self, env):
        """Wrap the env's step itself to measure total time and delimit a step window."""
        orig_step = env.step

        def step_wrapper(actions):
            t0 = time.perf_counter_ns()
            self.start_step()
            try:
                return orig_step(actions)
            finally:
                self.hist_ns["env.step_total"].append(time.perf_counter_ns() - t0)
                self.end_step()

        env.step = step_wrapper
        self._wrapped.append((env, "step", orig_step))

    def summary_ms_per_step(self, num_steps):
        # average ms spent per env.step (sums across multiple internal calls / decimation)
        sums_ms = {k: sum(v) / 1e6 for k, v in self.hist_ns.items()}
        avg_ms_per_step = {k: sums_ms[k] / max(num_steps, 1) for k in sums_ms}
        pct_of_total = {k: (sums_ms[k] / sums_ms["env.step_total"] * 100.0) for k in sums_ms}
        # compute "unaccounted" overhead inside step
        if "env.step_total" in sums_ms:
            unacct = sums_ms["env.step_total"] - sum(sums_ms[k] for k in sums_ms if k != "env.step_total")
            avg_ms_per_step["(unaccounted)"] = unacct / max(num_steps, 1)
            pct_of_total["(unaccounted)"] = (
                (unacct / sums_ms["env.step_total"] * 100.0) if sums_ms["env.step_total"] else 0.0
            )
        return avg_ms_per_step, pct_of_total

    def summarize(self, num_steps):
        avg_ms, pct = self.summary_ms_per_step(num_steps)
        total_ms_series = [ns / 1e6 for ns in self.hist_ns["env.step_total"]]
        return avg_ms, pct, total_ms_series

    def render_table(self, num_steps, title="env.step() breakdown"):
        avg_ms, pct, _ = self.summarize(num_steps)
        table = PrettyTable()
        table.title = title
        table.field_names = ["Section", "Avg ms/step", "% of step"]
        table.align["Section"] = "l"
        table.align["Avg ms/step"] = "r"
        table.align["% of step"] = "r"

        for name, ms in sorted(avg_ms.items(), key=lambda kv: (-kv[1], kv[0])):
            table.add_row([name, f"{ms:,.3f}", f"{pct.get(name, 0.0):,.1f}%"])
        return table.get_string()


def install_env_profiler(env):
    """Call with the *inner* env (env.unwrapped) right after gym.make(...)"""
    p = EnvStepProfiler()

    # wrap the high-level step first (RecordVideo wrappers will call down to this)
    p.wrap_env_step(env)

    # sim/scene loop pieces
    p.wrap(env.sim, "step", "sim.step")
    p.wrap(env.sim, "render", "sim.render")
    p.wrap(env.scene, "write_data_to_sim", "scene.write_data_to_sim")
    p.wrap(env.scene, "update", "scene.update")
    p.wrap(env.sim, "forward", "sim.forward")

    # managers in step()
    p.wrap(env.action_manager, "process_action", "action.process")
    p.wrap(env.action_manager, "apply_action", "action.apply")
    p.wrap(env.termination_manager, "compute", "termination.compute")
    p.wrap(env.reward_manager, "compute", "reward.compute")
    p.wrap(env.command_manager, "compute", "command.compute")
    p.wrap(env.observation_manager, "compute", "observation.compute")

    # event/recorder (optional; harmless if not present/used)
    p.wrap(env.event_manager, "apply", "event.apply")
    if hasattr(env, "recorder_manager"):
        p.wrap(env.recorder_manager, "record_pre_step", "recorder.pre_step")
        p.wrap(env.recorder_manager, "record_post_step", "recorder.post_step")
        p.wrap(env.recorder_manager, "record_pre_reset", "recorder.pre_reset")
        p.wrap(env.recorder_manager, "record_post_reset", "recorder.post_reset")

    # reset path pieces (will show up only on steps that reset)
    p.wrap(env, "_reset_idx", "env._reset_idx")

    # expose for later
    env._profiler = p
    return p
