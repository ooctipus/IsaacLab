# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra utilities with REPLACE-only preset system.

This module bypasses Hydra's default MERGE behavior for config groups.
Instead, when a preset is selected, the entire config section is REPLACED
with the preset -- no field merging.

Presets are declared by subclassing :class:`PresetCfg` (or using the
:func:`preset` factory for scalars). The system recursively discovers all
presets and their paths automatically, including inside dict-valued fields.

Override categories (applied in order):
    1. Global presets: ``presets=inference,newton_mjwarp`` -- apply everywhere matching
    2. Typed selectors: ``physics=newton_mjwarp`` / ``renderer=NAME`` /
       ``visualizer=NAME`` -- like a global preset, but must resolve against a
       config of that type or it errors
    3. Path presets: ``env.backend=newton_mjwarp`` -- REPLACE specific section
    4. Preset-path scalars: ``env.backend.dt=0.001`` -- handled by us
    5. Global scalars: ``env.decimation=10`` -- handled by Hydra

Example usage::

    presets=newton_mjwarp env.backend.dt=0.001 env.decimation=10
"""

import ast
import functools
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab.utils.configclass import (
    _copy_config_value,
    _field_module_dir,
    _wrap_resolvable_strings,
    configclass,
)

from .preset_target import PresetTarget

_LITERAL_MAP = {"true": True, "false": False, "none": None, "null": None}
_TYPED_TARGETS = {target.value: target for target in PresetTarget if target.base_classes}


@configclass
class PresetCfg:
    """Base class for declarative preset definitions.

    Subclass this and define fields as preset options.
    The field named ``default`` holds the config instance used
    when no CLI override is given. All other fields are named
    alternative presets.

    Example::

        @configclass
        class PhysicsCfg(PresetCfg):
            default: PhysxCfg = PhysxCfg()
            newton_mjwarp: MJWarpSolverCfg = MJWarpSolverCfg()

    The preset *name* (``newton_mjwarp``) is decoupled from the config class
    (``NewtonSolverCfg``): the class describes the Newton backend, while the field
    name labels which solver variant this entry selects.

    **Class-local helpers (underscore convention).** Names prefixed with
    ``_`` and callables (nested classes, methods) are skipped by the
    resolver and are NOT registered as variants. Use this to keep shared
    helpers adjacent to the variants that need them, without polluting the
    module namespace::

        @configclass
        class MultiBackendCameraCfg(PresetCfg):
            # Class-local helper -- not a variant.
            _ROTATED_OFFSET = CameraCfg.OffsetCfg(rot=(1, 0, 0, 0), ...)

            rgb = CameraCfg(data_types=["rgb"])
            albedo = CameraCfg(data_types=["albedo"], offset=_ROTATED_OFFSET)
            default = rgb
    """

    _configclass_deferred_copy: ClassVar[bool] = True


def preset(**options) -> PresetCfg:
    """Create a :class:`PresetCfg` instance from keyword arguments.

    A convenience factory that dynamically builds a ``PresetCfg`` subclass
    with one field per keyword argument, then returns an instance of it.
    The caller **must** supply a ``default`` key.

    Example::

        armature = preset(default=0.0, newton_mjwarp=0.01)
        # Equivalent to:
        # @configclass
        # class _Preset(PresetCfg):
        #     default: float = 0.0
        #     newton_mjwarp: float = 0.01
        # armature = _Preset()

    Args:
        **options: Preset alternatives keyed by name.  Must include ``default``.

    Returns:
        A ``PresetCfg`` instance whose fields are the supplied options.

    Raises:
        ValueError: If ``default`` is not provided.
    """
    if "default" not in options:
        raise ValueError("preset() requires a 'default' keyword argument.")
    annotations = {k: type(v) if v is not None else object for k, v in options.items()}
    ns = {"__annotations__": annotations, **options}
    cls = configclass(type("_Preset", (PresetCfg,), ns))
    return cls()


def _preset_fields(preset_obj) -> dict:
    """Extract all alternatives from a :class:`PresetCfg`, class attrs over instance.

    Class-level values take priority because robot-specific modules
    (e.g. ``joint_pos_env_cfg.py``) reassign fields on the class after
    instances are already created.
    """
    cls = type(preset_obj)
    d = {}
    for fn in preset_obj.__dataclass_fields__:
        if fn.startswith("_"):
            continue
        cls_val = getattr(cls, fn, None)
        d[fn] = cls_val if cls_val is not None else getattr(preset_obj, fn)
    for attr in vars(cls):
        if attr.startswith("_") or attr in d or callable(getattr(cls, attr)):
            continue
        d[attr] = getattr(cls, attr)
    return d


def _iter_cfg_items(cfg):
    if isinstance(cfg, Mapping):
        return cfg.items()
    if isinstance(cfg, list):
        return enumerate(cfg)
    return ((n, v) for n in dir(cfg) if not n.startswith("_") for v in [getattr(cfg, n, None)] if v is not None)


def _is_walkable_cfg(cfg) -> bool:
    return hasattr(cfg, "__dataclass_fields__") or isinstance(cfg, (Mapping, list))


def _walk_cfg(cfg, path: str, on_preset: Callable) -> None:
    """Depth-first walk of a config tree, calling *on_preset(parent, key, obj, path)*
    for every :class:`PresetCfg` node.  Recurses through dataclass attrs, dicts,
    nested dicts, and lists transparently."""
    for key, val in _iter_cfg_items(cfg):
        child_path = f"{path}.{key}" if path else str(key)
        if isinstance(val, PresetCfg):
            on_preset(cfg, key, val, child_path)
        elif _is_walkable_cfg(val):
            _walk_cfg(val, child_path, on_preset)


def collect_presets(cfg, path: str = "") -> dict:
    """Recursively discover :class:`PresetCfg` nodes in the config tree.

    Walks dataclass fields and dict values at any nesting depth.

    Args:
        cfg: A configclass instance to walk.
        path: Current path prefix (used during recursion).

    Returns:
        Dict mapping dotted paths to preset dicts, e.g.:
        ``{"backend": {"default": PhysxCfg(), "newton_mjwarp": MJWarpSolverCfg()}}``
    """
    result = {}

    def _record(preset_obj, preset_path):
        fields = _preset_fields(preset_obj)
        result[preset_path] = fields
        for alt in fields.values():
            if hasattr(alt, "__dataclass_fields__"):
                result.update(collect_presets(alt, preset_path))
            elif isinstance(alt, dict):
                for v in alt.values():
                    if _is_walkable_cfg(v):
                        result.update(collect_presets(v, preset_path))
            elif isinstance(alt, list):
                for v in alt:
                    if _is_walkable_cfg(v):
                        result.update(collect_presets(v, preset_path))

    if isinstance(cfg, PresetCfg):
        _record(cfg, path)
        return result

    _walk_cfg(cfg, path, lambda _p, _k, obj, cp: _record(obj, cp))
    return result


# ============================================================================
# Preset resolution
# ============================================================================


def _pick_alternative(
    preset_obj: PresetCfg,
    selected,
    path: str = "",
    explicit_name: str | None = None,
    consumed_selected: set[str] | None = None,
    typed_hits: dict[str, set[PresetTarget]] | None = None,
):
    """Choose the best alternative from a PresetCfg.

    Priority: first match in ``selected``, then ``default`` (preferring
    class-level over instance-level).

    Raises:
        ValueError: If no matching name and no ``default`` field exists.
    """
    fields = _preset_fields(preset_obj)

    def detach(value, name: str | None = None):
        if isinstance(value, PresetCfg):
            return value
        value = _copy_config_value(value)
        return _wrap_resolvable_strings(value, module_dir=_field_module_dir(preset_obj, name))

    if explicit_name is not None:
        if explicit_name in fields:
            return detach(fields[explicit_name], explicit_name)
        literal = _parse_val(explicit_name)
        if not isinstance(literal, str) or literal != explicit_name:
            return detach(literal)
        avail = list(fields)
        raise ValueError(f"Unknown preset '{explicit_name}' for {path}. Available: {avail}.")

    match_name = None
    match_value = None
    for name in selected:
        if name not in fields or name == match_name:
            continue
        val = fields[name]
        if consumed_selected is not None:
            consumed_selected.add(name)
        if typed_hits is not None:
            # record which typed targets this name landed on
            targets = {t for t in PresetTarget if t.base_classes and isinstance(val, t.base_classes)}
            if targets:
                typed_hits.setdefault(name, set()).update(targets)
        if match_name is not None:
            if match_value is not val and match_value != val:
                raise ValueError(
                    f"Conflicting global presets: '{match_name}' and '{name}' both define preset for '{path}'"
                )
        match_name, match_value = name, val
    if match_name is not None:
        return detach(match_value, match_name)
    if "default" in fields:
        return detach(fields["default"], "default")
    raise ValueError(
        f"PresetCfg {type(preset_obj).__name__} at '{path}' has no 'default' field "
        f"and none of the selected presets {selected} match its fields {set(fields.keys())}."
    )


def _resolve_active_presets(
    cfg,
    selected=(),
    explicit: dict[str, str] | None = None,
    root_path: str = "",
    *,
    strict_explicit: bool = True,
    consumed_selected: set[str] | None = None,
    typed_hits: dict[str, set[PresetTarget]] | None = None,
    consumed_explicit: set[str] | None = None,
):
    """Resolve presets by walking only the currently active tree.

    Preset alternatives are choice nodes. Once a choice is resolved, only the
    selected replacement is queued for further traversal, so inactive sibling
    branches cannot contribute descendant presets.
    """
    explicit = explicit or {}
    consumed_explicit = consumed_explicit if consumed_explicit is not None else set()

    def resolve_chain(preset_obj: PresetCfg, path: str):
        seen: set[int] = set()
        val = preset_obj
        while isinstance(val, PresetCfg):
            if id(val) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{path}': {type(val).__name__} was already visited."
                )
            seen.add(id(val))
            val = _pick_alternative(
                val,
                selected,
                path=path,
                explicit_name=explicit.get(path),
                consumed_selected=consumed_selected,
                typed_hits=typed_hits,
            )
        return val

    if isinstance(cfg, PresetCfg):
        if root_path in explicit:
            consumed_explicit.add(root_path)
        cfg = resolve_chain(cfg, root_path or "<root>")

    queue = deque([(root_path, cfg)])
    while queue:
        path, obj = queue.popleft()
        if not _is_walkable_cfg(obj):
            continue
        for key, val in _iter_cfg_items(obj):
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(val, PresetCfg):
                if child_path in explicit:
                    consumed_explicit.add(child_path)
                resolved = resolve_chain(val, child_path or "<root>")
                if isinstance(obj, list):
                    obj[int(key)] = resolved
                elif isinstance(obj, dict):
                    obj[key] = resolved
                else:
                    setattr(obj, key, resolved)
                if _is_walkable_cfg(resolved):
                    queue.append((child_path, resolved))
            elif _is_walkable_cfg(val):
                queue.append((child_path, val))

    missing = sorted(set(explicit) - consumed_explicit)
    if strict_explicit and missing:
        raise ValueError(f"Unknown or inactive preset group(s): {', '.join(missing)}")
    return cfg


def resolve_presets(cfg, selected=()):
    """Replace every :class:`PresetCfg` in the tree with the best alternative.

    For each ``PresetCfg`` found during an active-tree breadth-first walk:

    1. Pick the first name from *selected* that exists as a field on the
       preset, otherwise fall back to ``default``.
    2. Replace the preset in its parent (dict key or dataclass attr).
    3. Continue walking the replacement (which may contain more presets).

    Args:
        cfg: A configclass, dict, or PresetCfg to resolve in-place.
        selected: Set of preset names chosen by the user (e.g. from CLI
            ``presets=peg_insert_4mm,eval``).

    Returns:
        The resolved ``cfg`` (possibly a different object if the root itself
        was a PresetCfg).
    """
    return _resolve_active_presets(cfg, selected)


def resolve_config(cfg, overrides: Sequence[str]):
    """Resolve presets and dotted scalar overrides on an arbitrary config root.

    Typed selectors such as ``physics=NAME``, ``renderer=NAME``, and
    ``visualizer=NAME`` apply only when the selected alternative subclasses the
    matching target config. ``presets=NAME[,NAME,...]`` broadcasts without a
    typing claim. Other ``key=value`` entries select a preset at that path or,
    when the path is not a preset, set its scalar value.

    Args:
        cfg: Data-only configclass, mapping, list, or :class:`PresetCfg` root.
        overrides: Hydra-style ``key=value`` config overrides.

    Returns:
        The resolved config. Resolution mutates nested config objects and may
        replace ``cfg`` itself when the root is a :class:`PresetCfg`.

    Raises:
        ValueError: If an override is not ``key=value``, a named preset is
            unknown, or a typed selector does not resolve to its target type.
    """
    selected: list[str] = []
    requested_targets: dict[PresetTarget, set[str]] = {}
    explicit: dict[str, str] = {}
    for arg in overrides:
        if "=" not in arg:
            raise ValueError(f"Config override must use key=value syntax: {arg!r}")
        key, value = arg.split("=", 1)
        token = key.lstrip("-")
        if token == PresetTarget.DOMAIN.value:
            selected.extend(name.strip() for name in value.split(",") if name.strip())
        elif token in _TYPED_TARGETS:
            for name in (item.strip() for item in value.split(",") if item.strip()):
                selected.append(name)
                requested_targets.setdefault(_TYPED_TARGETS[token], set()).add(name)
        else:
            explicit[key] = value

    consumed_selected: set[str] = set()
    typed_hits: dict[str, set[PresetTarget]] = {}
    consumed_explicit: set[str] = set()
    cfg = _resolve_active_presets(
        cfg,
        selected,
        explicit,
        consumed_selected=consumed_selected,
        typed_hits=typed_hits,
        consumed_explicit=consumed_explicit,
        strict_explicit=False,
    )

    unknown_presets = set(selected) - consumed_selected
    if unknown_presets:
        preset_catalog = collect_presets(cfg)
        name_to_paths: dict[str, list[str]] = {}
        for path, fields in preset_catalog.items():
            for name in fields:
                name_to_paths.setdefault(name, []).append(path or "<root>")
        unknown = unknown_presets - set(name_to_paths)
        if unknown:
            available = {name: paths for name, paths in name_to_paths.items() if name != "default"}
            raise ValueError(_format_unknown_presets_error(unknown, available))

    _validate_typed_presets(requested_targets, typed_hits)
    for path, value in explicit.items():
        if path not in consumed_explicit:
            _setattr(cfg, path, _parse_val(value))
    return cfg


# ============================================================================
# CLI / Hydra integration
# ============================================================================


def _run_hydra(task, env_cfg, agent_cfg, hydra_args, callback):
    """Shared Hydra entry point for :func:`resolve_task_config` and :func:`hydra_task_config`."""
    if not hydra_args:
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        callback(env_cfg, agent_cfg)
        return

    original_argv, sys.argv = sys.argv, [sys.argv[0]] + hydra_args

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        env_cfg.from_dict(hydra_cfg["env"])
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        if isinstance(agent_cfg, dict) or agent_cfg is None:
            agent_cfg = hydra_cfg["agent"]
        else:
            agent_cfg.from_dict(hydra_cfg["agent"])
        callback(env_cfg, agent_cfg)

    try:
        hydra_main()
    finally:
        sys.argv = original_argv


def resolve_task_config(
    task_name: str,
    agent_cfg_entry_point: str | None,
    play_mode: bool = False,
    overrides: Sequence[str] | None = None,
):
    """Resolve env and agent configs with Hydra overrides, presets, and scalars fully applied.

    Safe to call before Kit is launched -- callable config values are stored as
    :class:`~isaaclab.utils.string.ResolvableString` and resolved lazily on
    first use, so no implementation modules are imported eagerly.

    Args:
        task_name: Task name (e.g., "IsaacContrib-Velocity-Flat-AnymalC").
        agent_cfg_entry_point: Agent config entry point key (e.g., "rsl_rl_cfg_entry_point").
        play_mode: Whether to apply the environment's playback overrides. Defaults to False.
        overrides: Optional Hydra arguments to use instead of reading them from
            :data:`sys.argv`. This keeps programmatic task composition on the same
            path as command-line composition. Defaults to None.
    Returns:
        Tuple of (env_cfg, agent_cfg) fully resolved.
    """
    task = task_name.split(":")[-1]
    env_cfg, agent_cfg, hydra_args = register_task(
        task, agent_cfg_entry_point, play_mode=play_mode, overrides=overrides
    )
    resolved = {}
    _run_hydra(task, env_cfg, agent_cfg, hydra_args, lambda e, a: resolved.update(env_cfg=e, agent_cfg=a))
    return resolved["env_cfg"], resolved["agent_cfg"]


def hydra_task_config(task_name: str, agent_cfg_entry_point: str, play_mode: bool = False) -> Callable:
    """Decorator for Hydra config with REPLACE-only preset semantics.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka")
        agent_cfg_entry_point: Agent config entry point key
        play_mode: Whether to apply the environment's playback overrides. Defaults to False.
    Returns:
        Decorated function receiving ``(env_cfg, agent_cfg, *args, **kwargs)``
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            task = task_name.split(":")[-1]
            env_cfg, agent_cfg, hydra_args = register_task(task, agent_cfg_entry_point, play_mode=play_mode)
            _run_hydra(task, env_cfg, agent_cfg, hydra_args, lambda e, a: func(e, a, *args, **kwargs))

        return wrapper

    return decorator


def _format_unknown_presets_error(unknown: set[str], name_to_paths: dict[str, list[str]], max_paths: int = 5) -> str:
    """Build a readable error message grouping presets by identical path fingerprints."""
    fingerprint_to_names: dict[tuple[str, ...], list[str]] = {}
    for name, paths in name_to_paths.items():
        key = tuple(sorted(paths))
        fingerprint_to_names.setdefault(key, []).append(name)

    lines = [f"Unknown preset(s): {', '.join(sorted(unknown))}"]
    lines += [
        "",
        "Available presets (grouped by affected paths):",
        "",
    ]
    for paths_tuple in sorted(fingerprint_to_names, key=lambda k: fingerprint_to_names[k][0]):
        names = sorted(fingerprint_to_names[paths_tuple])
        if len(names) <= 30:
            lines.append(f"  {', '.join(names)}")
        else:
            lines.append(f"  {', '.join(names[:25])}, ... ({len(names)} total)")
        shown = list(paths_tuple[:max_paths])
        for p in shown:
            lines.append(f"    -> {p}")
        remaining = len(paths_tuple) - len(shown)
        if remaining > 0:
            lines.append(f"    ... ({remaining} more)")
        lines.append("")
    return "\n".join(lines)


def _validate_typed_presets(
    requested: dict[PresetTarget, set[str]],
    typed_hits: dict[str, set[PresetTarget]],
) -> None:
    """Check that each typed selector landed on a config of its own type.

    A typed selector (``physics=NAME`` / ``renderer=NAME`` /
    ``visualizer=NAME``) is an explicit request for a backend of that type, so
    ``NAME`` must replace at least one config of that type during resolution.
    If it only matched unrelated presets that happen to share the name (a
    scalar, a sensor variant), the backend silently stays unchanged, so raise.
    The free-form ``presets=NAME`` broadcast is intentionally *not* checked --
    there the user makes no typing claim.

    Raises:
        ValueError: If a typed selector's name never resolved against a config
            of that target's type.
    """
    missing = sorted((t.value, n) for t, ns in requested.items() for n in ns if t not in typed_hits.get(n, set()))
    if missing:
        clauses = ", ".join(f"{label}={name}" for label, name in missing)
        raise ValueError(
            f"Typed preset selector(s) {clauses} did not match any preset of that type for this config. "
            "The name only matched unrelated presets (or nothing), so the target config would stay unchanged. "
            "Use a task that declares it on the matching config, or drop the selector."
        )


def register_task(
    task_name: str,
    agent_entry: str | None,
    play_mode: bool = False,
    overrides: Sequence[str] | None = None,
) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Presets are collected from nested configclasses and stored separately -
    NOT registered as Hydra groups to avoid Hydra's merge behavior.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka").
        agent_entry: Agent config entry point key.
        play_mode: Whether to apply the environment's playback overrides. Defaults to False.
        overrides: Optional Hydra arguments to compose instead of :data:`sys.argv`.
    Returns:
        Tuple of ``(env_cfg, agent_cfg, hydra_args)`` where presets have been
        resolved and ``hydra_args`` contains the remaining non-preset Hydra
        overrides.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_entry) if agent_entry else None

    cfgs = {"env": env_cfg, "agent": agent_cfg}
    config_args: list[str] = []
    hydra_args: list[str] = []
    for arg in sys.argv[1:] if overrides is None else overrides:
        if "=" not in arg:
            hydra_args.append(arg)
            continue
        key, _value = arg.split("=", 1)
        token = key.lstrip("-")
        if (
            token == PresetTarget.DOMAIN.value
            or token in _TYPED_TARGETS
            or key.startswith(("env.", "agent."))
            and not key.endswith("+")
        ):
            config_args.append(arg)
        else:
            hydra_args.append(arg)

    cfgs = resolve_config(cfgs, config_args)
    env_cfg, agent_cfg = cfgs["env"], cfgs["agent"]
    if play_mode and hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode()

    if not hydra_args:
        return env_cfg, agent_cfg, hydra_args

    # Convert to dict for Hydra (handle gym spaces and slices)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    agent_dict = agent_cfg.to_dict() if agent_cfg is not None and hasattr(agent_cfg, "to_dict") else agent_cfg
    env_dict = env_cfg.to_dict()  # type: ignore[union-attr]
    cfg_dict = replace_slices_with_strings({"env": env_dict, "agent": agent_dict})

    # Register plain config (no groups) - Hydra only handles global scalars
    ConfigStore.instance().store(name=task_name, node=OmegaConf.create(cfg_dict))
    return env_cfg, agent_cfg, hydra_args


def parse_overrides(args: list[str], presets: dict) -> tuple:
    """Categorize command line args by type.

    Args:
        args: Command line args (without script name)
        presets: {"env": {"path": {"name": cfg}}, "agent": {...}}

    Returns:
        (global_presets, preset_sel, preset_scalar, global_scalar) where:
        - global_presets: [name, ...] - apply to all matching configs
        - preset_sel: [(section, path, name), ...] - REPLACE selections
        - preset_scalar: [(full_path, value), ...] - scalars in preset paths
        - global_scalar: [arg, ...] - pass to Hydra
    """
    preset_paths = {f"{s}.{p}" if p else s for s, v in presets.items() for p in v}
    global_presets, preset_sel, preset_scalar, global_scalar = [], [], [], []

    for arg in args:
        if "=" not in arg:
            global_scalar.append(arg)
            continue
        key, val = arg.split("=", 1)
        if key == "presets":
            global_presets.extend(v.strip() for v in val.split(",") if v.strip())
        elif key in preset_paths:
            sec, path = key.split(".", 1) if "." in key else (key, "")
            preset_sel.append((sec, path, val))
        elif any(key.startswith(pp + ".") for pp in preset_paths):
            preset_scalar.append((key, val))
        else:
            global_scalar.append(arg)

    preset_sel.sort(key=lambda x: x[1].count("."))
    return global_presets, preset_sel, preset_scalar, global_scalar


def apply_overrides(
    env_cfg,
    agent_cfg,
    hydra_cfg: dict,
    global_presets: list,
    preset_sel: list,
    preset_scalar: list,
    presets: dict,
):
    """Apply preset selections and scalar overrides with REPLACE semantics.

    Presets are resolved by walking the active tree from root to leaves. A
    nested preset is only considered after its parent branch has been selected,
    which prevents inactive sibling branches from contributing colliding
    descendant paths.

    Returns:
        (env_cfg, agent_cfg) -- possibly replaced if root-level PresetCfg was resolved.

    Raises:
        ValueError: If multiple global presets conflict on an active path, or
            an explicit preset path is not reachable in the active tree.
    """
    cfgs = {"env": env_cfg, "agent": agent_cfg}

    explicit = {f"{sec}.{path}" if path else sec: name for sec, path, name in preset_sel}
    for sec in ("env", "agent"):
        if cfgs[sec] is None:
            continue
        section_explicit = {path: name for path, name in explicit.items() if path == sec or path.startswith(sec + ".")}
        cfgs[sec] = _resolve_active_presets(cfgs[sec], global_presets, section_explicit, root_path=sec)
        hydra_cfg[sec] = (
            cfgs[sec].to_dict()
            if hasattr(cfgs[sec], "to_dict")
            else dict(cfgs[sec])
            if isinstance(cfgs[sec], Mapping)
            else cfgs[sec]
        )

    _apply_preset_scalars(cfgs, hydra_cfg, preset_scalar)
    return cfgs["env"], cfgs["agent"]


def _apply_preset_scalars(cfgs: dict, hydra_cfg: dict, preset_scalar: list) -> None:
    for full_path, val_str in preset_scalar:
        sec = full_path.split(".", 1)[0]
        if sec not in cfgs:
            continue
        path = full_path[len(sec) + 1 :]
        if cfgs[sec] is not None:
            val = _parse_val(val_str)
            _setattr(cfgs[sec], path, val)
            _setattr(hydra_cfg, full_path, val)


def _setattr(obj, path: str, val):
    """Set nested attribute/key (e.g., "actions.arm_action.scale")."""
    *parts, leaf = path.split(".")
    for p in parts:
        obj = obj[p] if isinstance(obj, Mapping) else obj[int(p)] if isinstance(obj, list) else getattr(obj, p)
    if isinstance(obj, dict):
        obj[leaf] = val
    elif isinstance(obj, list):
        obj[int(leaf)] = val
    else:
        setattr(obj, leaf, val)


def _parse_val(s: str):
    """Parse string to Python value (bool, None, int, float, or str)."""
    if s.lower() in _LITERAL_MAP:
        return _LITERAL_MAP[s.lower()]
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
