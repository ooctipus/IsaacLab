# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare baseline/candidate sweep launches on identical live Isaac Lab inputs.

This diagnostic pauses after warmup, clones all launch buffers with aliasing
preserved, and leaves the actual environment running the baseline implementation.
It intentionally synchronizes and is not a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Read diagnostic controls followed by run_profiled.py arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline Newton checkout root.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate Newton checkout root.")
    parser.add_argument("--profile-script", type=Path, default=Path(__file__).with_name("run_profiled.py"))
    parser.add_argument(
        "--parity-output", type=Path, required=True, help="New JSON report path; existing files are refused."
    )
    parser.add_argument("--checks-per-tier", type=int, default=16)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--kernel-cache-dir")
    parser.add_argument("profile_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.profile_args and options.profile_args[0] == "--":
        options.profile_args.pop(0)
    if options.checks_per_tier < 1 or options.check_every < 1:
        parser.error("checks-per-tier and check-every must be positive")
    if "--no-graph" not in options.profile_args:
        parser.error("Pass --no-graph after -- for per-launch diagnostics")
    if options.parity_output.exists():
        parser.error(f"Parity output already exists: {options.parity_output}")
    solver_file = Path("newton/_src/solvers/feather_pgs/solver_feather_pgs.py")
    for checkout in (options.baseline, options.candidate):
        if not (checkout / solver_file).is_file():
            parser.error(f"No FeatherPGS solver found in checkout: {checkout}")
    if not options.profile_script.is_file():
        parser.error(f"Profile script does not exist: {options.profile_script}")
    return options


def clone_arguments(arguments: list) -> tuple[list, list, int, int]:
    """Copy overlapping pointer ranges once and rebuild their typed views."""
    import warp as wp
    from warp._src.types import type_size_in_bytes

    arrays = []
    for index, value in enumerate(arguments):
        if not isinstance(value, wp.array) or value.size == 0:
            continue
        if any(stride < 0 for stride in value.strides):
            raise AssertionError("Negative-stride launch buffers need explicit support")
        span = type_size_in_bytes(value.dtype) + sum(
            (length - 1) * stride for length, stride in zip(value.shape, value.strides)
        )
        arrays.append((str(value.device), value.ptr, value.ptr + span, index, value))
    arrays.sort(key=lambda item: (item[0], item[1], item[2]))
    groups = []
    for device_name, begin, end, index, value in arrays:
        if groups and groups[-1][0] == device_name and begin < groups[-1][2]:
            groups[-1][2] = max(groups[-1][2], end)
            groups[-1][3].append((index, value))
        else:
            groups.append([device_name, begin, end, [(index, value)]])
    result = list(arguments)
    owners = []
    for device_name, begin, end, members in groups:
        byte_count = end - begin
        source = wp.array(ptr=begin, shape=(byte_count,), dtype=wp.uint8, device=device_name, copy=False)
        owner = wp.empty(byte_count, dtype=wp.uint8, device=device_name)
        wp.copy(owner, source)
        owners.append(owner)
        for index, value in members:
            cloned = wp.array(
                ptr=owner.ptr + value.ptr - begin,
                shape=value.shape,
                strides=value.strides,
                dtype=value.dtype,
                device=value.device,
                copy=False,
            )
            cloned._ref = owner
            result[index] = cloned
    for index, value in enumerate(arguments):
        if isinstance(value, wp.array) and value.size == 0:
            result[index] = wp.empty(value.shape, dtype=value.dtype, device=value.device)
    return result, owners, sum(end - begin for _, begin, end, _ in groups), len(arrays) - len(groups)


def main() -> None:
    """Install matching sweep factories and inspect real post-warmup launches."""
    options = parse_arguments()
    sys.path.insert(0, str(options.baseline.resolve()))
    import numpy as np
    import warp as wp

    wp.config.enable_backward = False
    output = options.parity_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before imports or simulation can fail.
    with output.open("x") as stream:
        stream.write("{}\n")
    if options.kernel_cache_dir:
        wp.config.kernel_cache_dir = str(Path(options.kernel_cache_dir).resolve())

    import newton._src.solvers.feather_pgs.solver_feather_pgs as baseline_module

    candidate_path = options.candidate.resolve() / "newton/_src/solvers/feather_pgs/solver_feather_pgs.py"
    module_name = "newton._src.solvers.feather_pgs._sweep_parity_candidate"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    candidate_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = candidate_module
    spec.loader.exec_module(candidate_module)

    report = {
        "baseline_file": baseline_module.__file__,
        "candidate_file": str(candidate_path),
        "baseline_sha256": hashlib.sha256(Path(baseline_module.__file__).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "environment": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(("FEATHER_PGS_", "NEWTON_NARROW_PHASE_"))
        },
        "profile_args": options.profile_args,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checks": [],
        "factories": [],
        "physics_steps": 0,
        "capture_start_physics_step": None,
        "all_bitwise_equal": True,
        "all_finite": True,
        "simulation_states": [],
    }

    def write_report() -> None:
        output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    original_factory = baseline_module._get_pgs_solve_parallel_kernel
    original_launch = wp.launch_tiled
    original_step = baseline_module.SolverFeatherPGS.step
    twins = {}
    tier_counts = {}
    tier_seen = {}
    state = {"capture": False}

    def factory(*args, **kwargs):
        reference = original_factory(*args, **kwargs)
        twin = candidate_module._get_pgs_solve_parallel_kernel(*args, **kwargs)
        reference_labels = [arg.label for arg in reference.adj.args]
        candidate_labels = [arg.label for arg in twin.adj.args]
        if reference_labels != candidate_labels:
            raise AssertionError("Baseline and candidate kernel signatures differ")
        description = {"args": args, "kwargs": kwargs, "kernel": reference.key, "block_dim": reference._fpgs_block_dim}
        twins[id(reference)] = (twin, description)
        report["factories"].append(description)
        return reference

    def stepped(self, *args, **kwargs):
        report["physics_steps"] += 1
        return original_step(self, *args, **kwargs)

    def launch(*args, **kwargs):
        kernel = args[0] if args else kwargs.get("kernel")
        entry = twins.get(id(kernel))
        if not state["capture"] or entry is None:
            return original_launch(*args, **kwargs)
        twin, description = entry
        key = description["kernel"]
        tier_seen[key] = tier_seen.get(key, 0) + 1
        if tier_counts.get(key, 0) >= options.checks_per_tier or (tier_seen[key] - 1) % options.check_every:
            return original_launch(*args, **kwargs)
        if len(args) > 1:
            raise AssertionError("Expected only kernel as positional launch_tiled argument")
        inputs = list(kwargs.get("inputs", []))
        outputs = list(kwargs.get("outputs", []))
        arguments = inputs + outputs
        labels = [argument.label for argument in kernel.adj.args]
        if len(labels) != len(arguments):
            raise AssertionError("Launch argument count does not match kernel signature")

        # Finish prior parallel-stream work before freezing all aliased inputs.
        # Both launches below now consume the exact same stable pre-launch state.
        wp.synchronize_device(kwargs["device"])
        cloned, owners, cloned_bytes, alias_count = clone_arguments(arguments)
        source_by_name = dict(zip(labels, arguments))
        clone_by_name = dict(zip(labels, cloned))
        constraint_counts = source_by_name["world_constraint_count"].numpy()
        min_rows = description["kwargs"].get("min_rows", 0)
        max_rows = description["kwargs"]["rows"]
        mf_counts = source_by_name["mf_constraint_count"].numpy()
        admitted = (constraint_counts > min_rows) & (constraint_counts <= max_rows) & (mf_counts == 0)
        if description["kwargs"].get("skip_local_internal_worlds"):
            owner_values = source_by_name["local_solve_owner"].numpy()
            admitted &= owner_values == baseline_module.PGS_LOCAL_SOLVE_OWNER_GENERAL
        queue_active = int(source_by_name["general_world_count"].numpy()[0])
        if source_by_name["use_general_world_queue"]:
            queued = source_by_name["general_worlds"].numpy()[:queue_active]
            admitted &= np.isin(np.arange(len(constraint_counts)), queued)
        before_velocity = clone_by_name["v_out"].numpy()
        before_impulses = clone_by_name["world_impulses"].numpy()

        baseline_result = original_launch(*args, **kwargs)
        candidate_kwargs = dict(kwargs)
        candidate_kwargs["inputs"] = cloned[: len(inputs)]
        candidate_kwargs["outputs"] = cloned[len(inputs) :]
        candidate_kwargs.pop("kernel", None)
        original_launch(twin, **candidate_kwargs)

        checked = {}
        values_to_check = ["v_out", "world_impulses", "world_row_type", "world_row_parent", "world_row_mu"]
        if description["kwargs"].get("inkernel_response"):
            values_to_check += ["Y_world", "world_diag"]
        if baseline_module._WR_WARM:
            values_to_check += ["ww_count", "ww_shape0", "ww_shape1", "ww_point", "ww_lam"]
        for name in values_to_check:
            before = source_by_name[name].numpy()
            after = clone_by_name[name].numpy()
            same_bytes = before.tobytes() == after.tobytes()
            entry_report = {"bitwise_equal": same_bytes, "shape": list(before.shape)}
            if np.issubdtype(before.dtype, np.floating):
                entry_report["finite"] = bool(np.isfinite(before).all() and np.isfinite(after).all())
                entry_report["max_abs_delta"] = (
                    float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64)))) if before.size else 0.0
                )
                if name in ("v_out", "world_impulses") and not entry_report["finite"]:
                    report["all_finite"] = False
            if not same_bytes:
                different = before.view(np.uint8).reshape(before.shape + (before.dtype.itemsize,)) != after.view(
                    np.uint8
                ).reshape(after.shape + (after.dtype.itemsize,))
                entry_report["changed_elements"] = int(np.count_nonzero(different.any(axis=-1)))
                report["all_bitwise_equal"] = False
            checked[name] = entry_report
        actual_velocity = source_by_name["v_out"].numpy()
        actual_impulses = source_by_name["world_impulses"].numpy()
        item = {
            "physics_step": report["physics_steps"],
            "kernel": key,
            "block_dim": description["block_dim"],
            "min_rows": min_rows,
            "max_rows": max_rows,
            "admitted_worlds": int(admitted.sum()),
            "active_constraint_max": int(constraint_counts[admitted].max()) if admitted.any() else 0,
            "active_constraint_sum": int(constraint_counts[admitted].sum()),
            "queue_worlds": queue_active if source_by_name["use_general_world_queue"] else None,
            "row_phase": int(source_by_name["row_phase"]),
            "iteration_offset": int(source_by_name["iteration_offset"]),
            "friction_start_iteration": int(source_by_name["friction_start_iteration"]),
            "cloned_bytes": cloned_bytes,
            "shared_argument_ranges": alias_count,
            "velocity_elements_updated": int(
                np.count_nonzero(before_velocity.view(np.uint32) != actual_velocity.view(np.uint32))
            ),
            "impulse_elements_updated": int(
                np.count_nonzero(before_impulses.view(np.uint32) != actual_impulses.view(np.uint32))
            ),
            "arrays": checked,
        }
        report["checks"].append(item)
        tier_counts[key] = tier_counts.get(key, 0) + 1
        print(
            "SWEEP_PARITY "
            + json.dumps(
                {
                    k: item[k]
                    for k in (
                        "physics_step",
                        "block_dim",
                        "min_rows",
                        "max_rows",
                        "admitted_worlds",
                        "active_constraint_max",
                        "row_phase",
                        "velocity_elements_updated",
                        "impulse_elements_updated",
                    )
                }
            )
            + " exact="
            + str(all(v["bitwise_equal"] for v in checked.values())),
            flush=True,
        )
        write_report()
        if not report["all_bitwise_equal"] or not report["all_finite"]:
            raise AssertionError("Sweep parity or finite-output validation failed; details saved in " + str(output))
        # Keep owner allocations alive until synchronous comparisons complete.
        del owners
        return baseline_result

    baseline_module._get_pgs_solve_parallel_kernel = factory
    baseline_module.SolverFeatherPGS.step = stepped
    wp.launch_tiled = launch
    profile = runpy.run_path(options.profile_script, run_name="_real_task_sweep_profile")
    profile_main = profile["main"]
    profile_globals = profile_main.__globals__
    original_instrument = profile_globals["_instrument"]
    original_model_meta = profile_globals["_model_meta"]

    def model_meta(physics: str) -> dict:
        """Record the actual environment state before and after checked steps."""
        meta = original_model_meta(physics)
        report["simulation_states"].append(meta)
        return meta

    def instrument(env):
        original_instrument(env)
        state["capture"] = True
        report["capture_start_physics_step"] = report["physics_steps"]
        print("SWEEP_PARITY_START " + str(report["physics_steps"]), flush=True)

    profile_globals["_instrument"] = instrument
    profile_globals["_model_meta"] = model_meta
    sys.argv = [str(options.profile_script), *options.profile_args]
    try:
        profile_main()
        active_checks = [item for item in report["checks"] if item["admitted_worlds"] > 0]
        report["summary"] = {
            "checked_launches": len(report["checks"]),
            "active_launches": len(active_checks),
            "active_world_evaluations": sum(item["admitted_worlds"] for item in active_checks),
            "active_block_dims": sorted({item["block_dim"] for item in active_checks}),
            "physics_steps_sampled": len({item["physics_step"] for item in report["checks"]}),
            "velocity_elements_updated": sum(item["velocity_elements_updated"] for item in report["checks"]),
            "impulse_elements_updated": sum(item["impulse_elements_updated"] for item in report["checks"]),
        }
        if not report["simulation_states"] or not all(
            state.get("state_finite") is True for state in report["simulation_states"]
        ):
            report["all_finite"] = False
            raise AssertionError("Actual simulation state is nonfinite or its validation is unavailable")
        if not active_checks or not any(
            item["block_dim"] > 32 and (item["velocity_elements_updated"] or item["impulse_elements_updated"])
            for item in active_checks
        ):
            raise AssertionError("Corpus did not exercise an active multiwarp parallel-sweep launch")
        print("SWEEP_PARITY_FINAL " + json.dumps(report["summary"]), flush=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        write_report()
        baseline_module._get_pgs_solve_parallel_kernel = original_factory
        baseline_module.SolverFeatherPGS.step = original_step
        wp.launch_tiled = original_launch


if __name__ == "__main__":
    main()
