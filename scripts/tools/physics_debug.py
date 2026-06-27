# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect, validate, compare, and explicitly replay strict physics debug archives."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import isaaclab_newton.physics.debug_replay as debug_replay
import numpy as np
from isaaclab_newton.physics._debug_archive import (
    ARCHIVE_STATUSES,
    DebugArchiveError,
    load_archive,
)

_CAPABILITY_STATUSES = frozenset({"complete", "partial", "unavailable", "error", "disabled"})
_CAPABILITY_FIELDS = frozenset({"stage", "status", "provider", "fields", "adapter", "reason"})


class PhysicsDebugCliError(RuntimeError):
    """Raised for an actionable physics-debug CLI error."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _parse_capabilities(manifest: Mapping[str, Any], archive_path: Path) -> tuple[debug_replay.ReplayCapability, ...]:
    metadata = manifest["metadata"]
    raw_capabilities = metadata.get("capabilities")
    if raw_capabilities is None:
        return ()
    context = f"{archive_path}::__manifest__.metadata.capabilities"
    if not isinstance(raw_capabilities, Mapping):
        raise PhysicsDebugCliError(
            f"{context}: expected an object mapping capability IDs to declarations, "
            f"found {type(raw_capabilities).__name__}"
        )

    capabilities: list[debug_replay.ReplayCapability] = []
    for capability_id in sorted(raw_capabilities):
        if not isinstance(capability_id, str) or not capability_id:
            raise PhysicsDebugCliError(f"{context}: capability IDs must be non-empty strings")
        declaration = raw_capabilities[capability_id]
        capability_context = f"{context}.{capability_id}"
        if not isinstance(declaration, Mapping):
            raise PhysicsDebugCliError(f"{capability_context}: expected a capability object")
        unknown_fields = sorted(set(declaration).difference(_CAPABILITY_FIELDS))
        if unknown_fields:
            raise PhysicsDebugCliError(
                f"{capability_context}: unknown declaration fields {unknown_fields}; "
                f"allowed fields are {sorted(_CAPABILITY_FIELDS)}"
            )

        stage = declaration.get("stage")
        if not isinstance(stage, str) or stage not in debug_replay.REPLAY_STAGES:
            raise PhysicsDebugCliError(
                f"{capability_context}.stage: expected one of {sorted(debug_replay.REPLAY_STAGES)}, found {stage!r}"
            )
        status = declaration.get("status")
        if not isinstance(status, str) or status not in _CAPABILITY_STATUSES:
            raise PhysicsDebugCliError(
                f"{capability_context}.status: expected one of {sorted(_CAPABILITY_STATUSES)}, found {status!r}"
            )

        provider = declaration.get("provider")
        if provider is not None and (not isinstance(provider, str) or not provider):
            raise PhysicsDebugCliError(
                f"{capability_context}.provider: expected a non-empty string or null, found {provider!r}"
            )
        adapter = declaration.get("adapter")
        if adapter is not None and (not isinstance(adapter, str) or not adapter):
            raise PhysicsDebugCliError(
                f"{capability_context}.adapter: expected a non-empty string or null, found {adapter!r}"
            )
        reason = declaration.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise PhysicsDebugCliError(f"{capability_context}.reason: expected a string or null, found {reason!r}")

        raw_fields = declaration.get("fields")
        fields: tuple[str, ...] | None
        if raw_fields is None:
            fields = None
        elif isinstance(raw_fields, list) and all(isinstance(field, str) and field for field in raw_fields):
            if len(raw_fields) != len(set(raw_fields)):
                raise PhysicsDebugCliError(f"{capability_context}.fields: field names must be unique")
            fields = tuple(raw_fields)
        else:
            raise PhysicsDebugCliError(
                f"{capability_context}.fields: expected a list of non-empty archive keys or null"
            )

        if status == "complete":
            missing_contract = []
            if provider is None:
                missing_contract.append(f"{capability_context}.provider")
            if not fields:
                missing_contract.append(f"{capability_context}.fields")
            if adapter is None:
                missing_contract.append(f"{capability_context}.adapter")
            if missing_contract:
                raise PhysicsDebugCliError(
                    f"{capability_context}: status='complete' requires non-empty provider, fields, and adapter; "
                    f"missing {missing_contract}"
                )
        elif not reason:
            raise PhysicsDebugCliError(
                f"{capability_context}.reason: status={status!r} requires a non-empty explanation"
            )

        capabilities.append(
            debug_replay.ReplayCapability(
                capability_id=capability_id,
                stage=stage,
                status=status,
                provider=provider,
                fields=fields,
                adapter=adapter,
                reason=reason,
            )
        )
    return tuple(capabilities)


def _inspect_payload(
    archive_path: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    capabilities: Sequence[debug_replay.ReplayCapability],
) -> dict[str, Any]:
    inventory = manifest["arrays"]
    dependencies = manifest["dependencies"]
    return {
        "ok": True,
        "archive": str(archive_path),
        "format_name": manifest["format_name"],
        "format_version": manifest["format_version"],
        "status": manifest["status"],
        "created_at": manifest["created_at"],
        "required_keys": list(manifest["required_keys"]),
        "array_count": len(arrays),
        "total_bytes": sum(array.nbytes for array in arrays.values()),
        "arrays": [
            {
                "key": key,
                "dtype": inventory[key]["dtype"],
                "shape": inventory[key]["shape"],
                "nbytes": inventory[key]["nbytes"],
                "sha256": inventory[key]["sha256"],
            }
            for key in sorted(arrays)
        ],
        "capabilities": [capability.to_json() for capability in capabilities],
        "dependencies": dependencies,
        "runtime": manifest["runtime"],
        "metadata": manifest["metadata"],
    }


def _print_inspection(payload: Mapping[str, Any]) -> None:
    print(f"Archive: {payload['archive']}")
    print(f"Format: {payload['format_name']} v{payload['format_version']}")
    print(f"Status: {payload['status']}")
    print(f"Created: {payload['created_at']}")
    print(f"Arrays: {payload['array_count']} ({payload['total_bytes']} bytes)")
    required = payload["required_keys"]
    print(f"Required keys: {', '.join(required) if required else '(none)'}")

    print("Capabilities:")
    capabilities = payload["capabilities"]
    if not capabilities:
        print("  (none declared)")
    for capability in capabilities:
        provider = capability["provider"] or "<missing>"
        adapter = capability["adapter"] or "<missing>"
        field_count = "<missing>" if capability["fields"] is None else str(len(capability["fields"]))
        print(
            f"  {capability['capability_id']}: stage={capability['stage']} "
            f"status={capability['status']} provider={provider} adapter={adapter} fields={field_count}"
        )
        if capability["reason"]:
            print(f"    reason: {capability['reason']}")

    print("Dependencies:")
    for name, dependency in sorted(payload["dependencies"].items()):
        version = dependency.get("version") or "<unavailable>"
        print(f"  {name}: status={dependency['status']} version={version}")

    print("Array inventory:")
    for array in payload["arrays"]:
        print(
            f"  {array['key']}: dtype={array['dtype']} shape={array['shape']} "
            f"bytes={array['nbytes']} sha256={array['sha256']}"
        )


def _compare_archives(
    left_arrays: Mapping[str, np.ndarray],
    left_manifest: Mapping[str, Any],
    right_arrays: Mapping[str, np.ndarray],
    right_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def mismatch(path: str, kind: str, left: Any, right: Any) -> None:
        mismatches.append({"path": path, "kind": kind, "left": left, "right": right})

    for field in ("format_name", "format_version", "status", "required_keys", "dependencies", "runtime", "metadata"):
        if left_manifest[field] != right_manifest[field]:
            mismatch(f"manifest.{field}", "schema", left_manifest[field], right_manifest[field])

    left_keys = set(left_arrays)
    right_keys = set(right_arrays)
    if left_keys != right_keys:
        mismatch("arrays.keys", "schema", sorted(left_keys), sorted(right_keys))

    left_inventory = left_manifest["arrays"]
    right_inventory = right_manifest["arrays"]
    for key in sorted(left_keys & right_keys):
        left_entry = left_inventory[key]
        right_entry = right_inventory[key]
        for field in ("dtype", "dtype_descr", "shape", "nbytes"):
            if left_entry[field] != right_entry[field]:
                mismatch(
                    f"arrays.{key}.{field}",
                    "schema",
                    left_entry[field],
                    right_entry[field],
                )
        if left_entry["sha256"] != right_entry["sha256"]:
            mismatch(
                f"arrays.{key}.sha256",
                "value",
                left_entry["sha256"],
                right_entry["sha256"],
            )
    return mismatches


def _compact(value: Any, limit: int = 240) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _print_diff(payload: Mapping[str, Any]) -> None:
    if payload["match"]:
        print(f"Archives match: {payload['left']} == {payload['right']}")
        return
    print(f"Archives differ: {payload['left']} != {payload['right']}")
    for mismatch in payload["mismatches"]:
        print(
            f"  [{mismatch['kind']}] {mismatch['path']}: "
            f"left={_compact(mismatch['left'])} right={_compact(mismatch['right'])}"
        )


def _capability_blockers(
    capability: debug_replay.ReplayCapability,
    archive_keys: Collection[str],
) -> tuple[list[str], debug_replay.ReplayAdapter | None]:
    context = f"metadata.capabilities.{capability.capability_id}"
    blockers: list[str] = []
    if capability.status != "complete":
        detail = f" ({capability.reason})" if capability.reason else ""
        blockers.append(f"{context}.status is {capability.status!r}, not 'complete'{detail}")
    if capability.provider is None:
        blockers.append(f"{context}.provider is missing")
    if capability.fields is None:
        blockers.append(f"{context}.fields is missing")
    else:
        missing_fields = sorted(set(capability.fields).difference(archive_keys))
        if missing_fields:
            blockers.append(f"{context}.fields are absent from the archive: {missing_fields}")
    if capability.adapter is None:
        blockers.append(f"{context}.adapter is missing")
        return blockers, None

    adapter = debug_replay.get_replay_adapter(capability.adapter)
    if adapter is None:
        blockers.append(
            f"{context}.adapter={capability.adapter!r} has no registered execution adapter; "
            f"registered adapters are {list(debug_replay.get_replay_adapter_ids()) or '(none)'}"
        )
        return blockers, None
    if capability.stage not in adapter.stages:
        blockers.append(
            f"adapter {adapter.adapter_id!r} does not support stage {capability.stage!r}; "
            f"supported stages are {sorted(adapter.stages)}"
        )
    if capability.provider is not None and capability.provider not in adapter.providers:
        blockers.append(
            f"adapter {adapter.adapter_id!r} does not support provider {capability.provider!r}; "
            f"supported providers are {sorted(adapter.providers)}"
        )
    missing_adapter_fields = sorted(set(adapter.required_fields).difference(archive_keys))
    if missing_adapter_fields:
        blockers.append(
            f"adapter {adapter.adapter_id!r} requires archive fields that are missing: {missing_adapter_fields}"
        )
    if capability.fields is not None:
        undeclared_adapter_fields = sorted(set(adapter.required_fields).difference(capability.fields))
        if undeclared_adapter_fields:
            blockers.append(f"{context}.fields does not declare adapter-required fields: {undeclared_adapter_fields}")
    return blockers, adapter


def _select_replay(
    requested_stage: str,
    requested_capability: str | None,
    capabilities: Sequence[debug_replay.ReplayCapability],
    archive_keys: Collection[str],
) -> tuple[debug_replay.ReplayCapability, debug_replay.ReplayAdapter]:
    if requested_capability is not None:
        matching = [capability for capability in capabilities if capability.capability_id == requested_capability]
        if not matching:
            declared_ids = [capability.capability_id for capability in capabilities]
            raise PhysicsDebugCliError(
                f"Replay capability {requested_capability!r} is not declared. "
                f"Declared capabilities: {declared_ids or '(none)'}"
            )
        if requested_stage != "auto" and matching[0].stage != requested_stage:
            raise PhysicsDebugCliError(
                f"Replay capability {requested_capability!r} declares stage {matching[0].stage!r}, "
                f"which does not match requested stage {requested_stage!r}"
            )
    else:
        matching = (
            list(capabilities)
            if requested_stage == "auto"
            else [capability for capability in capabilities if capability.stage == requested_stage]
        )
    if not matching:
        declared = sorted({capability.stage for capability in capabilities})
        raise PhysicsDebugCliError(
            f"Replay stage {requested_stage!r} has no declared capability. "
            f"Declared stages: {declared or '(none)'}. No legacy-key inference is permitted."
        )

    candidates: list[tuple[debug_replay.ReplayCapability, debug_replay.ReplayAdapter]] = []
    failures: list[str] = []
    for capability in matching:
        blockers, adapter = _capability_blockers(capability, archive_keys)
        if blockers:
            failures.extend(f"{capability.capability_id}: {blocker}" for blocker in blockers)
        elif adapter is not None:
            candidates.append((capability, adapter))

    if not candidates:
        details = "\n  - ".join(failures) if failures else "no executable capability"
        raise PhysicsDebugCliError(f"Replay stage {requested_stage!r} is unavailable:\n  - {details}")
    if len(candidates) > 1:
        candidate_names = [f"{capability.capability_id} via {adapter.adapter_id}" for capability, adapter in candidates]
        raise PhysicsDebugCliError(
            f"Replay stage {requested_stage!r} is ambiguous across {candidate_names}. "
            "Select one explicitly with --capability."
        )
    return candidates[0]


def _load_replay_adapter_modules(module_names: Sequence[str]) -> None:
    """Import explicit trusted modules that register replay adapters.

    Args:
        module_names: User-supplied dotted module names in command-line order.

    Raises:
        PhysicsDebugCliError: If names are empty, duplicated, or fail to import.
    """
    empty_indices = [index for index, name in enumerate(module_names) if not isinstance(name, str) or not name.strip()]
    if empty_indices:
        raise PhysicsDebugCliError(
            f"--adapter_module values must be non-empty dotted module names; empty values at indices {empty_indices}"
        )

    duplicate_names = sorted({name for name in module_names if module_names.count(name) > 1})
    if duplicate_names:
        raise PhysicsDebugCliError(f"--adapter_module values must be unique; duplicated modules: {duplicate_names}")

    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise PhysicsDebugCliError(
                f"Failed to import trusted replay adapter module {module_name!r}: {exc}"
            ) from exc


def _encode_json(value: object, *, indent: int | None = None) -> str:
    """Encode one value with strict JSON number and type handling."""
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=indent, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise PhysicsDebugCliError(f"CLI result is not strict JSON-compatible: {exc}") from exc


def _emit_json(payload: Mapping[str, Any], *, stream: TextIO | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    print(_encode_json(payload), file=stream)


def _run_inspect(args: argparse.Namespace) -> int:
    archive_path = Path(args.archive)
    arrays, manifest = load_archive(archive_path, allowed_statuses=ARCHIVE_STATUSES)
    capabilities = _parse_capabilities(manifest, archive_path)
    payload = _inspect_payload(archive_path, arrays, manifest, capabilities)
    if args.json:
        _emit_json(payload)
    else:
        _print_inspection(payload)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    archive_path = Path(args.archive)
    allowed_statuses = frozenset(args.allowed_status) if args.allowed_status else frozenset({"complete"})
    arrays, manifest = load_archive(
        archive_path,
        required_keys=args.required_key,
        allowed_statuses=allowed_statuses,
    )
    capabilities = _parse_capabilities(manifest, archive_path)
    payload = {
        "ok": True,
        "archive": str(archive_path),
        "status": manifest["status"],
        "array_count": len(arrays),
        "capability_count": len(capabilities),
        "message": "archive is valid",
    }
    if args.json:
        _emit_json(payload)
    else:
        print(
            f"Valid archive: {archive_path} "
            f"(status={manifest['status']}, arrays={len(arrays)}, capabilities={len(capabilities)})"
        )
    return 0


def _run_diff(args: argparse.Namespace) -> int:
    left_path = Path(args.left)
    right_path = Path(args.right)
    left_arrays, left_manifest = load_archive(left_path, allowed_statuses=ARCHIVE_STATUSES)
    right_arrays, right_manifest = load_archive(right_path, allowed_statuses=ARCHIVE_STATUSES)
    _parse_capabilities(left_manifest, left_path)
    _parse_capabilities(right_manifest, right_path)
    mismatches = _compare_archives(left_arrays, left_manifest, right_arrays, right_manifest)
    payload = {
        "ok": not mismatches,
        "match": not mismatches,
        "left": str(left_path),
        "right": str(right_path),
        "mismatches": mismatches,
    }
    if args.json:
        _emit_json(payload)
    else:
        _print_diff(payload)
    return 1 if mismatches else 0


def _run_replay(args: argparse.Namespace) -> int:
    # Only explicit CLI input controls imports. Archive metadata is never used
    # as a Python module name.
    _load_replay_adapter_modules(args.adapter_module)
    archive_path = Path(args.archive)
    allowed_statuses = frozenset(args.allowed_status) if args.allowed_status else frozenset({"complete"})
    arrays, manifest = load_archive(archive_path, allowed_statuses=allowed_statuses)
    capabilities = _parse_capabilities(manifest, archive_path)
    capability, adapter = _select_replay(args.stage, args.capability, capabilities, arrays)
    request = debug_replay.ReplayRequest(
        archive_path=archive_path,
        arrays=arrays,
        manifest=manifest,
        capability=capability,
    )
    try:
        result = adapter.callback(request)
    except Exception as exc:
        raise PhysicsDebugCliError(f"Replay adapter {adapter.adapter_id!r} failed for {archive_path}: {exc}") from exc
    if result is not None and not isinstance(result, Mapping):
        raise PhysicsDebugCliError(
            f"Replay adapter {adapter.adapter_id!r} returned {type(result).__name__}; "
            "expected a JSON-compatible mapping or None"
        )
    result_payload = dict(result) if result is not None else {}
    _encode_json(result_payload)
    payload = {
        "ok": True,
        "archive": str(archive_path),
        "stage": capability.stage,
        "capability": capability.capability_id,
        "provider": capability.provider,
        "adapter": adapter.adapter_id,
        "result": result_payload,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(
            f"Replay completed: stage={capability.stage} capability={capability.capability_id} "
            f"provider={capability.provider} adapter={adapter.adapter_id}"
        )
        if result_payload:
            print(_encode_json(result_payload, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Show a validated archive inventory.")
    inspect_parser.add_argument("archive", help="Strict physics debug archive (.npz).")
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    inspect_parser.set_defaults(handler=_run_inspect)

    validate_parser = subparsers.add_parser("validate", help="Validate archive integrity and schema.")
    validate_parser.add_argument("archive", help="Strict physics debug archive (.npz).")
    validate_parser.add_argument(
        "--required_key",
        action="append",
        default=[],
        help="Require an additional exact array key. Repeat for multiple keys.",
    )
    validate_parser.add_argument(
        "--allowed_status",
        action="append",
        choices=sorted(ARCHIVE_STATUSES),
        help="Explicitly allow an archive status. Defaults to complete only.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    validate_parser.set_defaults(handler=_run_validate)

    diff_parser = subparsers.add_parser("diff", help="Compare archive schemas and array bytes exactly.")
    diff_parser.add_argument("left", help="Left strict physics debug archive (.npz).")
    diff_parser.add_argument("right", help="Right strict physics debug archive (.npz).")
    diff_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    diff_parser.set_defaults(handler=_run_diff)

    replay_parser = subparsers.add_parser("replay", help="Run an explicitly declared replay capability.")
    replay_parser.add_argument("archive", help="Strict physics debug archive (.npz).")
    replay_parser.add_argument(
        "--stage",
        choices=("auto", *sorted(debug_replay.REPLAY_STAGES)),
        default="auto",
        help="Declared replay stage to execute. Defaults to auto when exactly one adapter is executable.",
    )
    replay_parser.add_argument(
        "--capability",
        default=None,
        help="Execute one exact declared capability ID instead of stage-level selection.",
    )
    replay_parser.add_argument(
        "--adapter_module",
        action="append",
        default=[],
        metavar="DOTTED_MODULE",
        help="Import a trusted module that registers replay adapters. Repeat for multiple modules.",
    )
    replay_parser.add_argument(
        "--allowed_status",
        action="append",
        choices=sorted(ARCHIVE_STATUSES),
        help="Explicitly allow an archive status. Defaults to complete only.",
    )
    replay_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    replay_parser.set_defaults(handler=_run_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict physics debug archive CLI.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Process exit code. Diff returns one for mismatches. Invalid archives,
        unavailable replay capabilities, and adapter failures return two.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (DebugArchiveError, PhysicsDebugCliError) as exc:
        exit_code = exc.exit_code if isinstance(exc, PhysicsDebugCliError) else 2
        if getattr(args, "json", False):
            _emit_json(
                {
                    "ok": False,
                    "command": args.command,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                stream=sys.stderr,
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return exit_code
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
