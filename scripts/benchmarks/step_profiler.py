# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
from collections import defaultdict
from prettytable import PrettyTable


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

    def _wrap_term_callable(self, term_cfg, label):
        """Wrap term_cfg.func with a proxy that times calls and preserves attributes (e.g., reset)."""
        func = getattr(term_cfg, "func", None)
        if not callable(func):
            return
        # Avoid double wrapping
        if isinstance(func, _TimedCallableProxy):
            return
        proxy = _TimedCallableProxy(self, label, func)
        setattr(term_cfg, "func", proxy)
        self._wrapped.append((term_cfg, "func", func))

    def wrap_manager_terms(self, env):
        """Wrap per-term callables for all supported managers on the env."""
        # Reward terms
        if hasattr(env, "reward_manager") and hasattr(env.reward_manager, "_term_cfgs"):
            names = getattr(env.reward_manager, "_term_names", [])
            cfgs = getattr(env.reward_manager, "_term_cfgs", [])
            for name, cfg in zip(names, cfgs):
                self._wrap_term_callable(cfg, f"reward.term:{name}")

        # Termination terms
        if hasattr(env, "termination_manager") and hasattr(env.termination_manager, "_term_cfgs"):
            names = getattr(env.termination_manager, "_term_names", [])
            cfgs = getattr(env.termination_manager, "_term_cfgs", [])
            for name, cfg in zip(names, cfgs):
                self._wrap_term_callable(cfg, f"termination.term:{name}")

        # Event terms by mode
        if hasattr(env, "event_manager") and hasattr(env.event_manager, "_mode_term_cfgs"):
            mode_cfgs = getattr(env.event_manager, "_mode_term_cfgs", {})
            mode_names = getattr(env.event_manager, "_mode_term_names", {})
            for mode, cfg_list in mode_cfgs.items():
                names = mode_names.get(mode, [f"term_{i}" for i in range(len(cfg_list))])
                for name, cfg in zip(names, cfg_list):
                    self._wrap_term_callable(cfg, f"event.term[{mode}]:{name}")

        # Observation terms by group
        if hasattr(env, "observation_manager") and hasattr(env.observation_manager, "_group_obs_term_cfgs"):
            group_cfgs = getattr(env.observation_manager, "_group_obs_term_cfgs", {})
            group_names = getattr(env.observation_manager, "_group_obs_term_names", {})
            for group, cfg_list in group_cfgs.items():
                names = group_names.get(group, [f"term_{i}" for i in range(len(cfg_list))])
                for name, cfg in zip(names, cfg_list):
                    self._wrap_term_callable(cfg, f"observation.term[{group}]:{name}")

    def summary_ms_per_step(self, num_steps):
        """Return average ms/step per label and % of total (env.step_total)."""
        sums_ms = {k: sum(v) / 1e6 for k, v in self.hist_ns.items()}
        steps = max(num_steps, 1)
        avg_ms_per_step = {k: v / steps for k, v in sums_ms.items()}
        total = sums_ms.get("env.step_total", sum(sums_ms.values())) or 1.0
        pct_of_total = {k: sums_ms[k] / total * 100.0 for k in sums_ms}
        return avg_ms_per_step, pct_of_total

    def summarize(self, num_steps):
        avg_ms, pct = self.summary_ms_per_step(num_steps)
        total_ms_series = [ns / 1e6 for ns in self.hist_ns["env.step_total"]]
        return avg_ms, pct, total_ms_series

    def render_table(self, num_steps, title="env.step() breakdown"):
        avg_ms, pct, _ = self.summarize(num_steps)
        # We compute unaccounted once at the end; ignore any precomputed entries
        avg_ms.pop("(unaccounted)", None)
        pct.pop("(unaccounted)", None)

        # Build parent→children mapping for known manager sections
        parent_children_prefix = {
            "reward.compute": ("reward.term:",),
            "termination.compute": ("termination.term:",),
            "event.apply": ("event.term[",),
            "observation.compute": ("observation.term[",),
            "env._reset_idx": (
                "action.reset",
                "reward.reset",
                "termination.reset",
                "event.reset",
                "observation.reset",
                "command.reset",
                "curriculum.reset",
                "curriculum.compute",
                "recorder.reset",
                "scene.reset",
                "recorder.pre_reset",
                "recorder.post_reset",
            ),
        }

        children_map: dict[str, list[str]] = {p: [] for p in parent_children_prefix}
        child_keys: set[str] = set()
        for parent, prefixes in parent_children_prefix.items():
            if parent not in avg_ms:
                continue
            for k in list(avg_ms.keys()):
                if any(k.startswith(pref) for pref in prefixes):
                    children_map[parent].append(k)
                    child_keys.add(k)

        # Helper to prettify child labels
        def pretty_label(key: str) -> str:
            if key.startswith("reward.term:"):
                return key.split(":", 1)[1]
            if key.startswith("termination.term:"):
                return key.split(":", 1)[1]
            if key.startswith("event.term["):
                # event.term[mode]:name → mode.name
                try:
                    left, name = key.split(":", 1)
                    mode = left[left.find("[") + 1 : left.find("]")]
                    return f"{mode}.{name}"
                except Exception:
                    return key
            if key.startswith("observation.term["):
                # observation.term[group]:name → group.name
                try:
                    left, name = key.split(":", 1)
                    group = left[left.find("[") + 1 : left.find("]")]
                    return f"{group}.{name}"
                except Exception:
                    return key
            return key

        # Prepare table
        table = PrettyTable()
        table.title = title
        table.field_names = ["Section", "Avg ms/step", "% of step"]
        table.align["Section"] = "l"
        table.align["Avg ms/step"] = "l"
        table.align["% of step"] = "l"

        # Sort parents by avg time desc, include all entries except child-only keys
        parents_sorted = [
            k
            for k, _ in sorted(avg_ms.items(), key=lambda kv: (-kv[1], kv[0]))
            if k not in child_keys and k != "(unaccounted)"
        ]

        # Compute accounted total using only top-level parents (excluding env.step_total itself)
        accounted_ms = sum(avg_ms[p] for p in parents_sorted if p != "env.step_total")

        for parent in parents_sorted:
            # parent row
            table.add_row([parent, f"{avg_ms[parent]:,.3f}", f"{pct.get(parent, 0.0):,.1f}%"])
            # child rows (if any), sorted by their own time desc; show percent vs parent
            if parent in children_map and children_map[parent]:
                child_list_sorted = sorted(children_map[parent], key=lambda k: -avg_ms.get(k, 0.0))
                children_total_ms = 0.0
                for child in child_list_sorted:
                    child_ms = avg_ms.get(child, 0.0)
                    pct_parent = (child_ms / avg_ms[parent] * 100.0) if avg_ms[parent] else 0.0
                    children_total_ms += child_ms
                    table.add_row([
                        "  └─ " + pretty_label(child),
                        "  └─ " + f"{child_ms:,.3f}",
                        "  └─ " + f"{pct_parent:,.1f}%",
                    ])
                # Add manager overhead row = parent − sum(children)
                overhead_ms = max(0.0, avg_ms[parent] - children_total_ms)
                pct_parent_overhead = (overhead_ms / avg_ms[parent] * 100.0) if avg_ms[parent] else 0.0
                table.add_row([
                    "  └─ (manager_overhead)",
                    "  └─ " + f"{overhead_ms:,.3f}",
                    "  └─ " + f"{pct_parent_overhead:,.1f}%",
                ])

        # Add unaccounted row from top-level parents only (avoid double-counting children)
        total_ms = avg_ms.get("env.step_total", 0.0)
        if total_ms:
            un_ms = max(0.0, total_ms - accounted_ms)
            un_pct = un_ms / total_ms * 100.0
            table.add_row(["(unaccounted)", f"{un_ms:,.3f}", f"{un_pct:,.1f}%"])

        return table.get_string()


def install_env_profiler(env):
    """Install wrappers to profile env.step and key manager/scene/sim calls."""
    p = EnvStepProfiler()
    p.wrap_env_step(env)

    # Common wrappers (obj, method, label)
    wraps = [
        (env.sim, "step", "sim.step"),
        (env.sim, "render", "sim.render"),
        (env.sim, "forward", "sim.forward"),
        (env.scene, "write_data_to_sim", "scene.write_data_to_sim"),
        (env.scene, "update", "scene.update"),
        # managers in step
        (env.action_manager, "process_action", "action.process"),
        (env.action_manager, "apply_action", "action.apply"),
        (env.termination_manager, "compute", "termination.compute"),
        (env.reward_manager, "compute", "reward.compute"),
        (env.command_manager, "compute", "command.compute"),
        (env.observation_manager, "compute", "observation.compute"),
        # resets (grouped under env._reset_idx)
        (env.action_manager, "reset", "action.reset"),
        (env.reward_manager, "reset", "reward.reset"),
        (env.termination_manager, "reset", "termination.reset"),
        (env.event_manager, "reset", "event.reset"),
        (env.observation_manager, "reset", "observation.reset"),
        (env.command_manager, "reset", "command.reset"),
        (env.scene, "reset", "scene.reset"),
        # event & recorder
        (env.event_manager, "apply", "event.apply"),
    ]

    # Optional managers
    if hasattr(env, "curriculum_manager"):
        wraps += [
            (env.curriculum_manager, "reset", "curriculum.reset"),
            (env.curriculum_manager, "compute", "curriculum.compute"),
        ]
    if hasattr(env, "recorder_manager"):
        wraps += [
            (env.recorder_manager, "reset", "recorder.reset"),
            (env.recorder_manager, "record_pre_step", "recorder.pre_step"),
            (env.recorder_manager, "record_post_step", "recorder.post_step"),
            (env.recorder_manager, "record_pre_reset", "recorder.pre_reset"),
            (env.recorder_manager, "record_post_reset", "recorder.post_reset"),
            (env.recorder_manager, "record_post_physics_decimation_step", "recorder.post_decimation_step"),
        ]

    for obj, meth, label in wraps:
        if hasattr(obj, meth):
            self_label = label  # avoid late-binding
            p.wrap(obj, meth, self_label)

    # reset path container (shows on steps that reset)
    p.wrap(env, "_reset_idx", "env._reset_idx")

    env._profiler = p
    p.wrap_manager_terms(env)
    return p


class _TimedCallableProxy:
    """Callable proxy that forwards attributes to the original callable and times __call__.

    This preserves methods like `reset` on callable class instances used by managers while
    adding per-call timing under the provided label.
    """

    __slots__ = ("_prof", "_label", "_orig")

    def __init__(self, profiler: EnvStepProfiler, label: str, orig_callable):
        self._prof = profiler
        self._label = label
        self._orig = orig_callable

    def __call__(self, *a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return self._orig(*a, **kw)
        finally:
            if self._prof._in_step:
                self._prof.hist_ns[self._label].append(time.perf_counter_ns() - t0)

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def __repr__(self):
        return f"TimedCallableProxy({self._label}, {self._orig!r})"
