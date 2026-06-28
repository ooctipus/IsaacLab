# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the local Phase 2 repository, environment, and artifact identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECTS = Path("/home/zhengyuz/Projects")
REPOSITORIES = {
    "isaaclab": PROJECTS / "IsaacLab.wt/feature-unify",
    "rsl_rl": PROJECTS / "rsl_rl",
    "metamotivo": PROJECTS / "metamotivo",
    "humenv": PROJECTS / "humenv",
    "bfm_zero": PROJECTS / "BFM-Zero",
}
ENVIRONMENTS = {
    "isaaclab": REPOSITORIES["isaaclab"] / ".venv/bin/python",
    "metamotivo": REPOSITORIES["metamotivo"] / ".venv/bin/python",
    "bfm_zero": REPOSITORIES["bfm_zero"] / ".venv/bin/python",
}
ARTIFACTS = {
    "meta_train_split": (
        "humenv",
        "data_preparation/test_train_split/0-CMU_train_0.1.txt",
        "99929805f4ab531a89bff89837d27c403625d4b4d89d1a4d381b88825548a996",
    ),
    "meta_test_split": (
        "humenv",
        "data_preparation/test_train_split/0-CMU_test_0.1.txt",
        "c9b77782f5c35e0a33b3daa18c110856554acbebed00c4b2877b836d53f9b1b7",
    ),
    "meta_model": (
        "metamotivo",
        "examples/tmp_fbcpr/run1_cmu/checkpoint/model/model.safetensors",
        "e6832b62366c83abd9d2836750ec7aea941fe28de9daa3f91d147b3305267eff",
    ),
    "meta_optimizers": (
        "metamotivo",
        "examples/tmp_fbcpr/run1_cmu/checkpoint/optimizers.pth",
        "840a96282bd6b3d231c3a2b7b57a5f6324002ec027005ea8ad6fb26c2e934845",
    ),
    "meta_oracle_generator": (
        "metamotivo",
        "examples/phase0_oracle/generate_oracle.py",
        "29bd48e134ebee371e66c24f98f9717b6d3d1e8887dc187c0a72a4a7c975265e",
    ),
    "meta_oracle_compactor": (
        "metamotivo",
        "examples/phase0_oracle/compact_oracle.py",
        "899c3cdd0e641684e7ed0a4a2b9688e1e0432f43af472a2e8d5153958f3269b7",
    ),
    "meta_oracle_manifest": (
        "metamotivo",
        "examples/phase0_oracle/oracle.json",
        "9cbfd35e2e81fd1bd9e50dae58e448797a3fe96853d37cde236e2d888002e14a",
    ),
    "meta_oracle_tensors": (
        "metamotivo",
        "examples/phase0_oracle/oracle_semantic.safetensors",
        "c572740ee0718483b0b168e58897021d511cb88729aaec3a767ebf78dcfefe43",
    ),
    "bfm_motion_sources": (
        "bfm_zero",
        "humanoidverse/data/lafan_29dof.pkl",
        "f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
    ),
    "bfm_motion_clips": (
        "bfm_zero",
        "humanoidverse/data/lafan_29dof_10s-clipped.pkl",
        "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
    ),
    "bfm_model": (
        "bfm_zero",
        "new_model_for_training_code_inference/checkpoint/model/model.safetensors",
        "33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361",
    ),
    "bfm_oracle_generator": (
        "bfm_zero",
        "phase0_oracle/generate.py",
        "1eb18944be8240e17b6ae73a0f2f98ef49c6c383f9850b8c05917e80fca066d2",
    ),
    "bfm_oracle_manifest": (
        "bfm_zero",
        "phase0_oracle/oracle.json",
        "5adc21e6f9a659378f2bd9677feff3058220fa0eafff58c5d35e5abe917e5fd9",
    ),
    "bfm_oracle_tensors": (
        "bfm_zero",
        "phase0_oracle/oracle_tensors.safetensors",
        "8e96c6dec3ceb68bf7441bc2401120cfc49869a3b36c5726422bbd6cefd72f97",
    ),
}


def _run(command: list[str], cwd: Path | None = None, *, clean_python: bool = False) -> bytes:
    env = None
    if clean_python:
        env = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "_OLD_VIRTUAL_PATH"):
            env.pop(name, None)
    return subprocess.run(command, cwd=cwd, env=env, check=True, stdout=subprocess.PIPE).stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_snapshot(path: Path) -> dict[str, object]:
    untracked = sorted(
        value.decode()
        for value in _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], path).split(b"\0")
        if value
    )
    return {
        "path": str(path),
        "branch": _run(["git", "branch", "--show-current"], path).decode().strip(),
        "commit": _run(["git", "rev-parse", "HEAD"], path).decode().strip(),
        "status": _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], path).decode().splitlines(),
        "unstaged_diff_sha256": _sha256_bytes(_run(["git", "diff", "--binary"], path)),
        "staged_diff_sha256": _sha256_bytes(_run(["git", "diff", "--cached", "--binary"], path)),
        "untracked_path_count": len(untracked),
        "untracked_path_manifest_sha256": _sha256_bytes(b"\0".join(value.encode() for value in untracked)),
        "untracked_paths": untracked,
    }


def _environment_snapshot(python: Path) -> dict[str, object]:
    probe = """
import importlib.util, json, platform, sys
import torch
names = ('rsl_rl', 'metamotivo', 'humenv', 'humanoidverse')
print(json.dumps({
    'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'packages': {name: (spec.origin if (spec := importlib.util.find_spec(name)) else None) for name in names},
}, sort_keys=True))
"""
    runtime = json.loads(_run([str(python), "-c", probe], clean_python=True).decode())
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to persist the Phase 2 dependency freeze.")
    freeze_command = [uv, "pip", "freeze", "--python", str(python)]
    freeze = _run(freeze_command, clean_python=True)
    return {
        "runtime": runtime,
        "freeze_command": freeze_command,
        "freeze_sha256": _sha256_bytes(freeze),
        "freeze_lines": freeze.decode().splitlines(),
    }


def _artifact_snapshot(repository: str, relative_path: str, expected_sha256: str) -> dict[str, object]:
    path = REPOSITORIES[repository] / relative_path
    actual = _sha256_file(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "matches": actual == expected_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    disk_query = ["df", "-B1", str(PROJECTS)]
    bfm_curve = REPOSITORIES["bfm_zero"] / "new_model_for_training_code_inference/humanoidverse_tracking_eval.csv"
    with bfm_curve.open(newline="") as stream:
        bfm_curve_rows = sum(1 for _row in csv.reader(stream)) - 1
    result = {
        "schema": "forward_backward_phase2_preflight_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": {
            "gpu": gpu_query,
            "disk": disk_query,
            "git_unstaged": ["git", "diff", "--binary"],
            "git_staged": ["git", "diff", "--cached", "--binary"],
            "git_untracked": ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        },
        "repositories": {name: _repository_snapshot(path) for name, path in REPOSITORIES.items()},
        "environments": {name: _environment_snapshot(path) for name, path in ENVIRONMENTS.items()},
        "artifacts": {name: _artifact_snapshot(*values) for name, values in ARTIFACTS.items()},
        "cardinality": {
            "meta_train_motions": sum(
                1 for _line in (REPOSITORIES["humenv"] / ARTIFACTS["meta_train_split"][1]).open()
            ),
            "meta_test_motions": sum(1 for _line in (REPOSITORIES["humenv"] / ARTIFACTS["meta_test_split"][1]).open()),
            "bfm_released_curve_rows": bfm_curve_rows,
        },
        "system": {
            "gpu_query": _run(gpu_query).decode().splitlines(),
            "disk_query": _run(disk_query).decode().splitlines(),
        },
        "provenance_corrections": {
            "meta_oracle_compactor": "Phase 0 review omitted 'a2' from the independently verified digest.",
            "bfm_model": "Phase 0 review omitted 'd' from the digest stored in the BFM oracle manifest.",
        },
    }
    if not all(value["matches"] for value in result["artifacts"].values()):
        raise RuntimeError("At least one frozen reference artifact hash does not match.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
