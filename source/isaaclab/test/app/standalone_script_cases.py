# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discovery and subprocess helpers for standalone script smoke tests."""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOTS = (ROOT / "scripts" / "demos", ROOT / "scripts" / "tutorials")
# ``scripts/tools`` is not a root because most of its scripts are not simulator launches. The asset
# converters are: they launch the runtime needed by their importer.
EXTRA_SCRIPTS = (
    ROOT / "scripts" / "tools" / "convert_urdf.py",
    ROOT / "scripts" / "tools" / "convert_mjcf.py",
)
VISUALIZERS = ("none", "kit", "newton_gl", "newton_rtx", "rerun", "viser")
DEFAULT_READINESS_PATTERN = r"Setup complete"
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_BATCHED_NUM_ENVS = 2

_FATAL_PATTERNS = (
    "Traceback (most recent call last):",
    "Segmentation fault",
    "Fatal Python error:",
    "CUDA error:",
    "exceeded MJWarp limit",
    "nefc overflow",
)


@dataclass(frozen=True)
class ScriptOverride:
    """Describe behavior that cannot be inferred from a standalone script."""

    args: tuple[str, ...] = ()
    readiness_pattern: str | None = None
    startup_timeout: float | None = None
    skip_reason: str | None = None
    fixed_physics_backend: str | None = None
    visualizers: tuple[str, ...] | None = None
    case_skip_reasons: dict[tuple[str, str, str], str] = field(default_factory=dict)
    required_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScriptSpec:
    """Static launch contract for one standalone script."""

    path: Path
    flags: frozenset[str]
    preset_options: dict[str, tuple[str, ...]]
    args: tuple[str, ...]
    readiness_pattern: str | None
    startup_timeout: float | None
    skip_reason: str | None
    fixed_physics_backend: str | None
    visualizers: tuple[str, ...]
    case_skip_reasons: dict[tuple[str, str, str], str]
    required_modules: tuple[str, ...]

    @property
    def relative_path(self) -> str:
        """Return the repository-relative POSIX path."""
        return self.path.relative_to(ROOT).as_posix()

    @property
    def physics_backends(self) -> tuple[str, ...]:
        """Return the script's declared physics backend selections."""
        return self.preset_options.get("physics", (self.fixed_physics_backend or "isaacsim_physx",))

    @property
    def rendering_backends(self) -> tuple[str, ...]:
        """Return the script's declared rendering backend selections."""
        return self.preset_options.get("renderer", ("default",))


@dataclass(frozen=True)
class LaunchCase:
    """One script, physics backend, renderer, and visualizer combination."""

    spec: ScriptSpec
    physics_backend: str
    renderer_backend: str
    visualizer: str

    @property
    def id(self) -> str:
        """Return a stable pytest identifier."""
        stem = self.spec.relative_path.removeprefix("scripts/").removesuffix(".py").replace("/", "-")
        return f"{stem}-{self.physics_backend}-{self.renderer_backend}-{self.visualizer}"

    @property
    def skip_reason(self) -> str | None:
        """Return the static reason this launch combination cannot run."""
        key = (self.physics_backend, self.renderer_backend, self.visualizer)
        reason = self.spec.skip_reason or self.spec.case_skip_reasons.get(key)
        selected = {self.physics_backend, self.renderer_backend, self.visualizer}
        if (
            reason is None
            and {"ovphysx", "ovrtx"}.intersection(selected)
            and {
                "kit",
                "isaacsim_physx",
                "isaacsim_rtx",
            }.intersection(selected)
        ):
            return "OvPhysX/OVRTX cannot share a process with Kit/Isaac Sim"
        return reason

    def command(self) -> list[str]:
        """Build the repository launcher command for this case."""
        command = [str(ROOT / "isaaclab.sh"), "-p", self.spec.relative_path, *self.spec.args]
        if "--num_envs" in self.spec.flags and "--num_envs" not in self.spec.args:
            command.extend(("--num_envs", str(DEFAULT_BATCHED_NUM_ENVS)))
        if self.physics_backend in self.spec.preset_options.get("physics", ()):
            command.append(f"physics={self.physics_backend}")
        if self.renderer_backend in self.spec.preset_options.get("renderer", ()):
            command.append(f"renderer={self.renderer_backend}")
        if self.visualizer == "none":
            if self.spec.preset_options:
                command.append("sim.visualizer_cfgs=[]")
        else:
            command.append(f"visualizer={self.visualizer}")
        return command


@dataclass(frozen=True)
class SmokeResult:
    """Result of supervising a standalone script."""

    ready: bool
    returncode: int | None
    output: str
    elapsed: float
    stopped_after_soak: bool
    fatal_patterns: tuple[str, ...] = ()


# newton ships the same NVIDIA Ant model the Kit importer bundles, as in test_mjcf_converter.py.
_NEWTON_MJCF = str(Path(importlib.util.find_spec("newton").origin).parent / "examples" / "assets" / "nv_ant.xml")

OVERRIDES = {
    "scripts/demos/arl_robot_1.py": ScriptOverride(readiness_pattern=r"Starting demo with Lee Position Controller"),
    "scripts/demos/h1_locomotion.py": ScriptOverride(
        skip_reason="downloads a published policy and requires interactive viewport input",
        visualizers=("kit",),
    ),
    "scripts/demos/haply_teleoperation.py": ScriptOverride(
        skip_reason="requires a physical Haply device and its WebSocket service"
    ),
    "scripts/demos/heterogeneous_scene.py": ScriptOverride(
        args=("--num_task", "2"),
        readiness_pattern=r"Composed \d+ task scenes into \d+ environments",
    ),
    "scripts/demos/mpm/newton_mpm_granular.py": ScriptOverride(
        args=("--max_steps", "20"),
        readiness_pattern=r"Newton granular MPM demo ready",
        fixed_physics_backend="newton_mpm",
    ),
    "scripts/demos/mpm/newton_mpm_twoway_coupling.py": ScriptOverride(
        args=("--max_steps", "2", "--voxel_size", "0.2"),
        readiness_pattern=r"Newton two-way MPM demo ready",
        fixed_physics_backend="newton_coupler",
        required_modules=("isaaclab_contrib",),
    ),
    "scripts/demos/mpm/snowball_smash.py": ScriptOverride(
        args=("--max_steps", "20"),
        readiness_pattern=r"Newton snowball-smash demo ready",
        fixed_physics_backend="newton_mpm",
    ),
    "scripts/demos/mpm/teapot_fill.py": ScriptOverride(
        args=("--max_steps", "20"),
        readiness_pattern=r"Newton teapot-fill MPM demo ready",
        fixed_physics_backend="newton_mpm",
    ),
    "scripts/demos/multi_asset.py": ScriptOverride(args=("--num_envs", "4")),
    "scripts/demos/newton_viewer_block_and_tackle.py": ScriptOverride(
        args=("--max_steps", "20"),
        fixed_physics_backend="newton_vbd",
        required_modules=("isaaclab_contrib",),
    ),
    "scripts/demos/newton_viewer_dominoes.py": ScriptOverride(
        args=("--max_steps", "20"),
        fixed_physics_backend="newton_xpbd",
    ),
    "scripts/demos/sensors/cameras.py": ScriptOverride(args=("--num_envs", "1"), startup_timeout=900.0),
    "scripts/demos/sensors/multi_mesh_raycaster.py": ScriptOverride(
        args=("--flat_ground",),
        startup_timeout=600.0,
    ),
    "scripts/demos/sensors/newton_raycast_heightfield.py": ScriptOverride(fixed_physics_backend="newton_mjwarp"),
    "scripts/demos/sensors/newton_raycast_moving_geometry.py": ScriptOverride(fixed_physics_backend="newton_mjwarp"),
    "scripts/demos/pick_and_place.py": ScriptOverride(
        readiness_pattern=r"Gym action space|Press the 'A' key", visualizers=("kit",)
    ),
    "scripts/demos/sensors/ppisp_camera.py": ScriptOverride(
        args=("--max_steps", "3", "--image_width", "64", "--image_height", "64", "warmup_steps=1"),
        startup_timeout=600.0,
    ),
    "scripts/demos/sensors/ppisp_camera_ovrtx.py": ScriptOverride(
        args=("--max_steps", "3", "--warmup_steps", "1"),
        fixed_physics_backend="newton_mjwarp",
        required_modules=("ovrtx",),
    ),
    # Readiness fires once conversion succeeds, so the preview runs inside the soak.
    "scripts/tools/convert_urdf.py": ScriptOverride(
        args=(
            str(ROOT / "source" / "isaaclab" / "test" / "sim" / "urdfs" / "test_merge_joints.urdf"),
            str(Path(tempfile.gettempdir()) / "isaaclab_converter_smoke" / "urdf"),
            "--merge_joints",
        ),
        readiness_pattern=r"Generated USD file:",
    ),
    "scripts/tools/convert_mjcf.py": ScriptOverride(
        args=(_NEWTON_MJCF, str(Path(tempfile.gettempdir()) / "isaaclab_converter_smoke" / "mjcf")),
        readiness_pattern=r"Generated USD file:",
    ),
    "scripts/tutorials/01_assets/run_surface_gripper.py": ScriptOverride(args=("--device", "cpu")),
    "scripts/tutorials/03_envs/create_cartpole_base_env.py": ScriptOverride(readiness_pattern=r"Resetting environment"),
    "scripts/tutorials/03_envs/create_cube_base_env.py": ScriptOverride(readiness_pattern=r"Mean position error"),
    "scripts/tutorials/03_envs/create_quadruped_base_env.py": ScriptOverride(
        readiness_pattern=r"Resetting environment"
    ),
    "scripts/tutorials/03_envs/policy_inference_in_usd.py": ScriptOverride(
        skip_reason="requires a user-supplied TorchScript checkpoint"
    ),
    "scripts/tutorials/03_envs/run_cartpole_rl_env.py": ScriptOverride(readiness_pattern=r"Resetting environment"),
    "scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py": ScriptOverride(
        readiness_pattern=r"Gym action space"
    ),
}


def discover_specs() -> list[ScriptSpec]:
    """Discover executable scripts and their declarative preset choices."""
    specs = []
    for group in (*(sorted(root.rglob("*.py")) for root in SCRIPT_ROOTS), EXTRA_SCRIPTS):
        for path in group:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            if not _has_main_guard(tree):
                continue
            relative_path = path.relative_to(ROOT).as_posix()
            override = OVERRIDES.get(relative_path, ScriptOverride())
            readiness_pattern = override.readiness_pattern
            if readiness_pattern is None and re.search(DEFAULT_READINESS_PATTERN, source):
                readiness_pattern = DEFAULT_READINESS_PATTERN
            preset_options = _preset_options(tree)
            if override.visualizers is not None:
                visualizers = override.visualizers
            else:
                visualizers = tuple(dict.fromkeys(("none", *preset_options.get("visualizer", ()))))
            specs.append(
                ScriptSpec(
                    path=path,
                    flags=_literal_cli_flags(tree),
                    preset_options=preset_options,
                    args=override.args,
                    readiness_pattern=readiness_pattern,
                    startup_timeout=override.startup_timeout,
                    skip_reason=override.skip_reason,
                    fixed_physics_backend=override.fixed_physics_backend,
                    visualizers=visualizers,
                    case_skip_reasons=override.case_skip_reasons,
                    required_modules=override.required_modules,
                )
            )
    return specs


def build_cases(specs: list[ScriptSpec]) -> list[LaunchCase]:
    """Expand specs across declared physics, renderer, and visualizer choices."""
    return [
        LaunchCase(
            spec=spec,
            physics_backend=physics_backend,
            renderer_backend=renderer_backend,
            visualizer=visualizer,
        )
        for spec in specs
        for physics_backend in spec.physics_backends
        for renderer_backend in spec.rendering_backends
        for visualizer in spec.visualizers
    ]


def select_script_scope(specs: list[ScriptSpec], scope: str) -> list[ScriptSpec]:
    """Select scripts within a repository-relative demo or tutorial directory.

    Args:
        specs: Discovered standalone script specifications.
        scope: Directory below ``scripts``, or ``"all"`` for every script.

    Returns:
        Specifications selected by the requested scope.

    Raises:
        ValueError: If a non-default scope does not select any scripts.
    """
    if scope == "all":
        return specs
    selected_specs = [spec for spec in specs if f"scripts/{scope}/" in spec.relative_path]
    if not selected_specs:
        raise ValueError(f"standalone script scope selected no scripts: {scope!r}")
    return selected_specs


def select_runtime_group(cases: list[LaunchCase], runtime_group: str) -> list[LaunchCase]:
    """Select launch cases by whether they require the Isaac Sim Kit runtime."""
    if runtime_group not in {"kit", "non-kit"}:
        raise ValueError(f"unsupported runtime group: {runtime_group!r}")
    return [
        case
        for case in cases
        if (
            case.physics_backend == "isaacsim_physx"
            or case.renderer_backend == "isaacsim_rtx"
            or case.visualizer == "kit"
        )
        == (runtime_group == "kit")
    ]


def backend_is_available(backend: str) -> bool:
    """Return whether the package implementing a selected backend is importable."""
    if backend == "default":
        return True
    if backend == "isaacsim_rtx":
        return importlib.util.find_spec("isaacsim") is not None or (ROOT / "_isaac_sim").exists()
    if backend in {"physx", "isaacsim_physx"}:
        package = "isaaclab_physx"
    elif backend == "ovphysx":
        package = "isaaclab_ov"
    else:
        package = "isaaclab_newton" if backend.startswith("newton") else f"isaaclab_{backend}"
    return importlib.util.find_spec(package) is not None


def module_is_available(module: str) -> bool:
    """Return whether an optional runtime module is importable."""
    return importlib.util.find_spec(module) is not None


def visualizer_is_available(visualizer: str) -> bool:
    """Return whether the package implementing a visualizer is importable."""
    if visualizer == "none":
        return True
    if visualizer == "kit":
        return importlib.util.find_spec("isaacsim") is not None or (ROOT / "_isaac_sim").exists()
    if importlib.util.find_spec("isaaclab_visualizers") is None:
        return False
    if visualizer == "newton_gl":
        return importlib.util.find_spec("isaaclab_newton") is not None
    if visualizer == "newton_rtx":
        return True
    return importlib.util.find_spec(visualizer) is not None


def gui_is_available() -> bool:
    """Return whether the process has access to a desktop display server."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_until_ready(
    command: list[str],
    readiness_pattern: str,
    *,
    startup_timeout: float = 180.0,
    soak_time: float = 5.0,
    screenshot_path: Path | None = None,
    screenshot_delay: float = 1.0,
) -> SmokeResult:
    """Run a script until it exits or remains healthy after becoming ready.

    Infinite demos are terminated as a process group after the readiness marker
    has been observed and the soak interval has elapsed.
    """
    start_time = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    ready_at = None
    stopped_after_soak = False
    screenshot_captured = False
    fatal_patterns = set()
    monitor_fatal_patterns = True

    def record_output(chunk: bytes) -> None:
        """Record bounded output while retaining fatal-pattern state."""
        output.extend(chunk)
        decoded_output = output.decode(errors="replace")
        if monitor_fatal_patterns:
            fatal_patterns.update(pattern for pattern in _FATAL_PATTERNS if pattern in decoded_output)
        if len(output) > MAX_OUTPUT_BYTES:
            del output[:-MAX_OUTPUT_BYTES]

    def read_available_output(timeout: float) -> bool:
        """Read one available chunk per ready stream and report whether any bytes were read."""
        read_output = False
        for key, _ in selector.select(timeout=timeout):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if chunk:
                record_output(chunk)
                read_output = True
        return read_output

    try:
        while True:
            read_available_output(timeout=0.1)
            decoded = output.decode(errors="replace")
            if ready_at is None and re.search(readiness_pattern, decoded):
                ready_at = time.monotonic()

            returncode = process.poll()
            if returncode is not None:
                break
            now = time.monotonic()
            if (
                screenshot_path is not None
                and ready_at is not None
                and not screenshot_captured
                and now - ready_at >= min(screenshot_delay, soak_time / 2.0)
            ):
                _capture_screenshot(screenshot_path)
                screenshot_captured = True
            if ready_at is not None and now - ready_at >= soak_time:
                stopped_after_soak = True
                # Classify every byte already waiting in the pipe before crossing the teardown boundary.
                while read_available_output(timeout=0.0):
                    pass
                # Preserve shutdown logs without treating errors caused by intentional teardown as runtime failures.
                monitor_fatal_patterns = False
                _terminate_process_group(process)
                returncode = process.poll()
                break
            if now - start_time >= startup_timeout:
                _terminate_process_group(process)
                returncode = process.poll()
                break
    finally:
        selector.close()
        if process.poll() is None:
            _terminate_process_group(process)
        try:
            remainder, _ = process.communicate(timeout=5)
            record_output(remainder or b"")
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            remainder, _ = process.communicate(timeout=5)
            record_output(remainder or b"")

    decoded = output.decode(errors="replace")
    if ready_at is None and re.search(readiness_pattern, decoded):
        ready_at = time.monotonic()

    return SmokeResult(
        ready=ready_at is not None,
        returncode=process.returncode,
        output=decoded,
        elapsed=time.monotonic() - start_time,
        stopped_after_soak=stopped_after_soak,
        fatal_patterns=tuple(sorted(fatal_patterns)),
    )


def assert_smoke_passed(result: SmokeResult, case: LaunchCase) -> None:
    """Assert that a supervised script reached readiness without a fatal error."""
    tail = result.output[-30000:]
    assert not result.fatal_patterns, f"{case.id} emitted fatal output {result.fatal_patterns}:\n{tail}"
    assert result.ready, f"{case.id} did not reach {case.spec.readiness_pattern!r} in {result.elapsed:.1f}s:\n{tail}"
    assert result.stopped_after_soak or result.returncode == 0, (
        f"{case.id} exited with {result.returncode} before completing the soak:\n{tail}"
    )


def _has_main_guard(tree: ast.AST) -> bool:
    """Return whether an AST contains an executable ``__main__`` guard."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        names = [child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)]
        values = [child.value for child in ast.walk(node.test) if isinstance(child, ast.Constant)]
        if "__name__" in names and "__main__" in values:
            return True
    return False


def _literal_cli_flags(tree: ast.AST) -> frozenset[str]:
    """Collect literal ``argparse.add_argument`` flags from an AST."""
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        option = node.args[0]
        if (
            not isinstance(option, ast.Constant)
            or not isinstance(option.value, str)
            or not option.value.startswith("--")
        ):
            continue
        flags.add(option.value)
    return frozenset(flags)


def _has_planned_newton_rtx_camera(tree: ast.AST) -> bool:
    """Return whether the script constructs a compatible camera inside its clone lifecycle."""

    def name(node: ast.AST) -> str | None:
        return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None

    def exact_construction(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "class_type"
            and len(node.args) == 1
            and not node.keywords
            and ast.dump(node.func.value) == ast.dump(node.args[0])
        )

    sessions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.With, ast.AsyncWith))
        and any(
            isinstance(item.context_expr, ast.Call) and name(item.context_expr.func) == "ReplicateSession"
            for item in node.items
        )
    ]
    constructions = {
        ast.unparse(call.args[0]) for session in sessions for call in ast.walk(session) if exact_construction(call)
    }
    if not constructions:
        return False

    camera_fields = set()
    shared_scene = False
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        shared_scene |= any(name(base) == "MultiBackendSceneCfg" for base in class_node.bases)
        for statement in class_node.body:
            value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
            if not isinstance(value, ast.Call) or name(value.func) != "MultiBackendCameraCfg":
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            camera_fields.update(name(target) for target in targets if name(target) is not None)
    if any(expression.rsplit(".", 1)[-1] in camera_fields for expression in constructions):
        return True

    scene_constructed = any(
        expression.rsplit(".", 1)[-1] == "scene" or expression.endswith("scene_cfg") for expression in constructions
    )
    if shared_scene and scene_constructed:
        return True

    def compatible_camera(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call) or name(node.func) != "CameraCfg":
            return False
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        path = keywords.get("prim_path")
        renderer = keywords.get("renderer_cfg")
        return (
            isinstance(path, ast.Constant)
            and path.value == "{ENV_REGEX_NS}/Camera"
            and isinstance(renderer, ast.Call)
            and name(renderer.func) == "MultiBackendRendererCfg"
        )

    custom_camera = any(
        isinstance(node, ast.keyword) and node.arg == "newton_rtx" and compatible_camera(node.value)
        for node in ast.walk(tree)
    )
    return custom_camera and scene_constructed


def _preset_options(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Collect typed alternatives from inline and named ``PresetCfg`` declarations."""

    def name(node: ast.AST) -> str | None:
        return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None

    preset_classes = {**_canonical_preset_classes(), **_declared_preset_classes(tree)}

    def alternatives(value: ast.AST) -> tuple[str, ...]:
        if not isinstance(value, ast.Call):
            return ()
        if name(value.func) == "preset":
            return tuple(keyword.arg for keyword in value.keywords if keyword.arg not in {None, "default"})
        return preset_classes.get(name(value.func), ())

    targets = {"physics": "physics", "renderer": "renderer", "visualizer_cfgs": "visualizer"}
    options: dict[str, list[str]] = {target: [] for target in targets.values()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for target, names in _canonical_cfg_options().get(name(node.func), {}).items():
                options[target].extend(names)
            pairs = ((keyword.arg, keyword.value) for keyword in node.keywords)
        elif isinstance(node, ast.Assign):
            pairs = ((name(target), node.value) for target in node.targets)
        elif isinstance(node, ast.AnnAssign):
            pairs = ((name(node.target), node.value),)
        else:
            continue
        for field_name, value in pairs:
            if target := targets.get(field_name):
                options[target].extend(alternatives(value))
    if not _has_planned_newton_rtx_camera(tree):
        options["visualizer"] = [name for name in options["visualizer"] if name != "newton_rtx"]
    return {target: tuple(dict.fromkeys(names)) for target, names in options.items() if names}


def _declared_preset_classes(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Return named ``PresetCfg`` alternatives declared in one module."""

    def name(node: ast.AST) -> str | None:
        return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None

    preset_classes = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not any(name(base) == "PresetCfg" for base in node.bases):
            continue
        fields = []
        for statement in node.body:
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
                if isinstance(statement, ast.AnnAssign)
                else []
            )
            fields.extend(
                field_name
                for target in targets
                if (field_name := name(target)) is not None
                and field_name != "default"
                and not field_name.startswith("_")
            )
        preset_classes[node.name] = tuple(dict.fromkeys(fields))
    return preset_classes


@functools.cache
def _canonical_presets_tree() -> ast.Module:
    """Parse the canonical shared preset module once."""
    path = ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "utils" / "presets.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@functools.cache
def _canonical_preset_classes() -> dict[str, tuple[str, ...]]:
    """Read shared preset alternatives from their canonical data-only module."""
    return _declared_preset_classes(_canonical_presets_tree())


@functools.cache
def _canonical_cfg_options() -> dict[str, dict[str, tuple[str, ...]]]:
    """Return typed preset fields inherited through canonical config roots."""
    targets = {"physics": "physics", "renderer": "renderer", "visualizer_cfgs": "visualizer"}
    preset_classes = _canonical_preset_classes()
    result = {}
    for node in _canonical_presets_tree().body:
        if not isinstance(node, ast.ClassDef):
            continue
        options = {}
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            target = targets.get(statement.target.id)
            value = statement.value
            if target and isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if alternatives := preset_classes.get(value.func.id):
                    options[target] = alternatives
        if options:
            result[node.name] = options
    return result


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate a script and any simulator children it spawned."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _capture_screenshot(path: Path) -> None:
    """Capture the active display to ``path`` using ImageMagick."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["import", "-window", "root", str(path)], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"failed to capture {path}: {result.stderr.strip()}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"screenshot command did not create a non-empty image at {path}")
