# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight NVTX annotation injection for IsaacLab environments.

Monkey-patches environment methods to emit NVTX push/pop ranges so that
``nsys profile`` can group CUDA kernels under human-readable labels like
``sim.step``, ``reward.compute``, ``observation.compute``, etc.

No timing code — nsys handles all measurement. The only overhead is the
NVTX push/pop calls themselves (~100 ns each).
"""

from __future__ import annotations

import torch.cuda.nvtx as nvtx


def _wrap_nvtx(obj: object, method_name: str, label: str):
    """Replace ``obj.method_name`` with a wrapper that emits an NVTX range.

    Args:
        obj: The object whose method to wrap.
        method_name: Name of the method on *obj*.
        label: NVTX range label (appears in nsys timeline / stats).
    """
    orig = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        nvtx.range_push(label)
        try:
            return orig(*args, **kwargs)
        finally:
            nvtx.range_pop()

    setattr(obj, method_name, wrapped)


def install_nvtx_hooks(env):
    """Install NVTX annotations on an unwrapped IsaacLab environment.

    Detects whether *env* is a :class:`ManagerBasedRLEnv` or
    :class:`DirectRLEnv` and wraps the appropriate methods.

    Args:
        env: The **unwrapped** environment instance (``env.unwrapped``).
    """
    # -- Top-level step -------------------------------------------------------
    _wrap_nvtx(env, "step", "env.step")

    # -- Sim / scene (common to both env types) --------------------------------
    _wrap_nvtx(env.sim, "step", "sim.step")
    _wrap_nvtx(env.sim, "render", "sim.render")
    _wrap_nvtx(env.scene, "write_data_to_sim", "scene.write_data_to_sim")
    _wrap_nvtx(env.scene, "update", "scene.update")

    # -- Reset -----------------------------------------------------------------
    _wrap_nvtx(env, "_reset_idx", "env._reset_idx")

    # -- Manager-based env specific --------------------------------------------
    if hasattr(env, "action_manager"):
        _wrap_nvtx(env.action_manager, "process_action", "action.process")
        _wrap_nvtx(env.action_manager, "apply_action", "action.apply")
    if hasattr(env, "termination_manager"):
        _wrap_nvtx(env.termination_manager, "compute", "termination.compute")
    if hasattr(env, "reward_manager"):
        _wrap_nvtx(env.reward_manager, "compute", "reward.compute")
    if hasattr(env, "observation_manager"):
        _wrap_nvtx(env.observation_manager, "compute", "observation.compute")
    if hasattr(env, "command_manager"):
        _wrap_nvtx(env.command_manager, "compute", "command.compute")
    if hasattr(env, "event_manager"):
        _wrap_nvtx(env.event_manager, "apply", "event.apply")
    if hasattr(env, "recorder_manager"):
        _wrap_nvtx(env.recorder_manager, "record_pre_step", "recorder.pre_step")
        _wrap_nvtx(env.recorder_manager, "record_post_step", "recorder.post_step")
    if hasattr(env, "curriculum_manager"):
        _wrap_nvtx(env.curriculum_manager, "compute", "curriculum.compute")

    # -- Direct env specific ---------------------------------------------------
    if hasattr(env, "_pre_physics_step") and not hasattr(env, "action_manager"):
        _wrap_nvtx(env, "_pre_physics_step", "direct.pre_physics_step")
        _wrap_nvtx(env, "_apply_action", "direct.apply_action")
        _wrap_nvtx(env, "_get_rewards", "direct.get_rewards")
        _wrap_nvtx(env, "_get_dones", "direct.get_dones")
        _wrap_nvtx(env, "_get_observations", "direct.get_observations")

    # -- Per-term NVTX for manager-based envs ----------------------------------
    _install_manager_term_hooks(env)


def install_extra_nvtx_hooks(env, hooks: list[tuple[str, str]]):
    """Install user-defined NVTX hooks on the environment.

    Allows benchmark matrices to annotate task-specific methods without
    hardcoding them in this module.

    Args:
        env: The **unwrapped** environment instance.
        hooks: List of ``(attr_path, label)`` tuples. The *attr_path* is
            resolved relative to *env* using dot-separated traversal.
            The last segment is the method name to wrap.
            For example, ``("feature_extractor.step", "vision.feature_extractor")``
            wraps ``env.feature_extractor.step``.
            A single-segment path like ``("_compute_image_observations", "vision.image_obs")``
            wraps ``env._compute_image_observations``.
    """
    for attr_path, label in hooks:
        parts = attr_path.rsplit(".", 1)
        if len(parts) == 2:
            obj_path, method_name = parts
            obj = env
            for attr in obj_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
        else:
            obj = env
            method_name = parts[0]

        if not hasattr(obj, method_name):
            continue

        _wrap_nvtx(obj, method_name, label)


def _install_manager_term_hooks(env):
    """Wrap individual manager term callables with NVTX ranges.

    This gives per-term granularity in nsys, e.g.
    ``reward.term:reaching_goal``, ``observation.term[policy]:joint_pos``.
    """
    # Reward terms
    if hasattr(env, "reward_manager"):
        _wrap_manager_terms(
            env.reward_manager,
            "_term_names",
            "_term_cfgs",
            prefix="reward.term",
        )

    # Termination terms
    if hasattr(env, "termination_manager"):
        _wrap_manager_terms(
            env.termination_manager,
            "_term_names",
            "_term_cfgs",
            prefix="termination.term",
        )

    # Observation terms (grouped)
    if hasattr(env, "observation_manager"):
        group_names = getattr(env.observation_manager, "_group_obs_term_names", {})
        group_cfgs = getattr(env.observation_manager, "_group_obs_term_cfgs", {})
        for group, cfg_list in group_cfgs.items():
            names = group_names.get(group, [f"term_{i}" for i in range(len(cfg_list))])
            for name, cfg in zip(names, cfg_list):
                _wrap_term_func(cfg, f"observation.term[{group}]:{name}")

    # Event terms (by mode)
    if hasattr(env, "event_manager"):
        mode_cfgs = getattr(env.event_manager, "_mode_term_cfgs", {})
        mode_names = getattr(env.event_manager, "_mode_term_names", {})
        for mode, cfg_list in mode_cfgs.items():
            names = mode_names.get(mode, [f"term_{i}" for i in range(len(cfg_list))])
            for name, cfg in zip(names, cfg_list):
                _wrap_term_func(cfg, f"event.term[{mode}]:{name}")


def _wrap_manager_terms(
    manager: object,
    names_attr: str,
    cfgs_attr: str,
    prefix: str,
):
    """Wrap term funcs on a flat (non-grouped) manager."""
    names = getattr(manager, names_attr, [])
    cfgs = getattr(manager, cfgs_attr, [])
    for name, cfg in zip(names, cfgs):
        _wrap_term_func(cfg, f"{prefix}:{name}")


def _wrap_term_func(term_cfg: object, label: str):
    """Wrap ``term_cfg.func`` with an NVTX range, preserving attribute access."""
    func = getattr(term_cfg, "func", None)
    if func is None or not callable(func):
        return
    # Avoid double-wrapping
    if isinstance(func, _NvtxCallableProxy):
        return
    proxy = _NvtxCallableProxy(label, func)
    setattr(term_cfg, "func", proxy)


class _NvtxCallableProxy:
    """Callable proxy that emits an NVTX range around each call.

    Forwards attribute access (e.g. ``.reset``) to the original callable
    so manager introspection still works.
    """

    __slots__ = ("_label", "_orig")

    def __init__(self, label: str, orig_callable):
        self._label = label
        self._orig = orig_callable

    def __call__(self, *args, **kwargs):
        nvtx.range_push(self._label)
        try:
            return self._orig(*args, **kwargs)
        finally:
            nvtx.range_pop()

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def __repr__(self):
        return f"NvtxCallableProxy({self._label}, {self._orig!r})"
