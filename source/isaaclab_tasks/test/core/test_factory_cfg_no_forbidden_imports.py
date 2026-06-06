# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that Hydra task-config resolution does not import forbidden backend modules.

This mirrors ``test_env_cfg_no_forbidden_imports.py`` but exercises the full
Hydra path used by the training/play scripts: :func:`resolve_task_config`
(presets + scalar overrides applied) rather than the raw
``load_cfg_from_registry``.  ``resolve_task_config`` runs BEFORE SimulationApp
is launched so the app launcher can inspect the physics backend; therefore
resolving env/agent configs -- including applying presets such as
``presets=nut_thread_m16,eval`` -- must not pull in any module that requires
Kit to be running first.

It is scoped to ``Isaac-Factory-Franka-JointPos-v0`` because that task drives
config almost entirely through the preset system (asset variant, assembly
profile, eval sizing, ...), so a stray backend import inside a preset branch is
easy to introduce and would only surface once Hydra resolves the active tree.

Forbidden categories
--------------------
1. **Backend / simulator runtime** (``pxr``, ``omni``, ``carb``, ``isaacsim``)
   -- require SimulationApp / Kit to be initialized first.
2. **SciPy** -- loads OpenBLAS which registers ``atfork`` handlers that crash
   Kit's internal ``fork()`` during startup.

Remediation patterns
--------------------
* Use ``lazy_loader.attach_stub`` in ``__init__.py`` files with a
  corresponding ``.pyi`` stub so that implementation modules are only
  imported when first accessed.
* Guard annotation-only imports with ``TYPE_CHECKING``.
* Store ``class_type`` / ``func`` fields as fully-qualified strings
  (e.g. ``"isaaclab.assets.articulation:Articulation"``); ``cfg.validate()``
  resolves them to callables after Kit has launched.
* Use local ``# noqa: PLC0415`` imports inside functions for Kit-dependent
  symbols that cannot be imported at module level before Kit is running.

Performance note
----------------
All preset combinations are resolved in a **single subprocess** so that
``import isaaclab_tasks`` (~1.6 s) is paid only once instead of once per test.
Results are returned as JSON and cached for the parametrized test functions.
"""

import json
import subprocess
import sys
import textwrap

import pytest

# Forbidden module prefixes -- these must NOT be imported while resolving a
# task config, because they require SimulationApp / a specific physics backend
# to be started first, or because they are heavyweight runtime libraries that
# should never be needed to build pure-data config objects.
_FORBIDDEN_PREFIXES = (
    # Backend / simulator runtime (require SimulationApp / Kit)
    "pxr",  # USD Python bindings
    "omni",  # Omniverse runtime
    "carb",  # Carbonite framework
    "isaacsim",  # Isaac Sim modules
    # SciPy loads OpenBLAS which crashes Kit's fork()
    "scipy",
)

_TASK_NAME = "Isaac-Factory-Franka-JointPos-v0"
_AGENT_CFG_ENTRY_POINT = "rsl_rl_cfg_entry_point"

# Preset selections to resolve.  ``""`` means "no presets" (defaults only);
# the others mirror real CLI invocations (``presets=nut_thread_m16,eval``).
_PRESET_COMBOS = (
    "",
    "nut_thread_m16,eval",
    "peg_insert_16mm,eval",
    "nut_thread_m16,newton_mjwarp,eval",
)

# ---------------------------------------------------------------------------
# Batch subprocess: resolve every preset combo in one Python process so we only
# pay the `import isaaclab_tasks` cost once (~1.6 s) instead of once per test.
# ---------------------------------------------------------------------------


def _build_batch_script(task_name: str, agent_entry: str, preset_combos: tuple[str, ...]) -> str:
    return textwrap.dedent(f"""\
        import sys, traceback, json

        FORBIDDEN = {list(_FORBIDDEN_PREFIXES)!r}
        task_name = {task_name!r}
        agent_entry = {agent_entry!r}
        preset_combos = {list(preset_combos)!r}

        import isaaclab_tasks  # noqa: F401 -- triggers task registration
        from isaaclab_tasks.utils.hydra import resolve_task_config

        results = {{}}
        original_argv = list(sys.argv)

        for combo in preset_combos:
            violations = {{}}
            load_error = None

            # resolve_task_config reads global presets / scalar overrides from
            # sys.argv, exactly like the train/play scripts do.
            sys.argv = [original_argv[0]] + ([f"presets={{combo}}"] if combo else [])

            _orig_import = __builtins__.__import__

            def _hook(name, *args, _violations=violations, **kw):
                top = name.split('.')[0]
                if top in FORBIDDEN and top not in _violations:
                    _violations[top] = ''.join(traceback.format_stack())
                return _orig_import(name, *args, **kw)

            __builtins__.__import__ = _hook
            try:
                env_cfg, agent_cfg = resolve_task_config(task_name, agent_entry)
            except Exception as exc:
                load_error = f"{{type(exc).__name__}}: {{exc}}"
            finally:
                __builtins__.__import__ = _orig_import
                sys.argv = list(original_argv)

            # Note: we intentionally do NOT clean up sys.modules between combos.
            # The import hook intercepts every __import__ call regardless of
            # whether the module is already cached, so it reliably catches
            # violations even after the first resolution has warmed the cache.

            results[combo] = {{
                'load_error': load_error,
                'violations': violations,
            }}

        # Use a sentinel so the parser can find the JSON even when
        # resolve_task_config prints [INFO] lines to stdout.
        print("__RESULTS__" + json.dumps(results))
    """)


@pytest.fixture(scope="session")
def factory_cfg_check_results() -> dict:
    """Resolve all preset combos in a single subprocess and return results dict."""
    script = _build_batch_script(_TASK_NAME, _AGENT_CFG_ENTRY_POINT, _PRESET_COMBOS)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Find the sentinel line (resolve_task_config emits [INFO] lines to stdout).
    json_line = None
    for line in result.stdout.splitlines():
        if line.startswith("__RESULTS__"):
            json_line = line[len("__RESULTS__") :]
            break

    if json_line is None:
        return {
            "__subprocess_crash__": (
                f"Batch subprocess did not produce results.\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        }
    try:
        return json.loads(json_line)
    except json.JSONDecodeError as exc:
        return {"__json_error__": str(exc), "__raw__": json_line[:500]}


@pytest.mark.parametrize("preset_combo", _PRESET_COMBOS, ids=lambda combo: combo or "default")
def test_resolve_task_config_does_not_import_backend_modules(preset_combo: str, factory_cfg_check_results: dict):
    """Hydra config resolution must not import forbidden runtime modules.

    Resolving ``Isaac-Factory-Franka-JointPos-v0`` with the given presets must
    not pull in backend modules (pxr, omni, carb, isaacsim) or heavyweight
    libraries (scipy), since it runs before SimulationApp is launched.

    Fix: use lazy_loader.attach_stub with .pyi stubs in __init__.py files,
    TYPE_CHECKING guards for annotation-only imports, and string references
    for class_type/func fields in cfg files.
    """
    if "__subprocess_crash__" in factory_cfg_check_results:
        pytest.fail(f"Batch check subprocess crashed:\n{factory_cfg_check_results['__subprocess_crash__']}")
    if "__json_error__" in factory_cfg_check_results:
        pytest.fail(
            f"Batch check subprocess produced unparsable output: {factory_cfg_check_results['__json_error__']}\n"
            f"raw: {factory_cfg_check_results.get('__raw__')}"
        )

    if preset_combo not in factory_cfg_check_results:
        pytest.fail(f"No result for preset combo '{preset_combo}' - batch subprocess may have crashed.")

    info = factory_cfg_check_results[preset_combo]
    load_error = info.get("load_error")
    violations = info.get("violations", {})

    messages = []
    if load_error:
        messages.append(f"ERROR: config resolution crashed: {load_error}")
    if violations:
        messages.append(f"FAIL: {len(violations)} forbidden top-level module(s) imported:")
        for mod, stack in sorted(violations.items()):
            messages.append(f"\n=== {mod} ===\n{stack}")

    selection = preset_combo or "(defaults)"
    assert not violations and not load_error, (
        f"resolve_task_config('{_TASK_NAME}', '{_AGENT_CFG_ENTRY_POINT}') with presets={selection} "
        f"imported forbidden backend modules.\n"
        f"Forbidden prefixes: {_FORBIDDEN_PREFIXES}\n"
        + "\n".join(messages)
        + "\n\nFix: use lazy_loader.attach_stub with a .pyi stub in the offending "
        "__init__.py, or move the import under TYPE_CHECKING and use a string "
        "reference for isinstance checks."
    )
