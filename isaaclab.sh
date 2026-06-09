#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find python to run CLI.
if [ -n "$VIRTUAL_ENV" ]; then
    python_exe="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    python_exe="$CONDA_PREFIX/bin/python"
elif [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
    python_exe="$ISAACLAB_PATH/env_isaaclab/bin/python"
elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
    python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
else
    # Fallback to system python
    python_exe="python3"
fi

# Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"

_prepend_pythonpath() {
    local path_entry="$1"
    if [ -z "$path_entry" ] || [ ! -d "$path_entry" ]; then
        return
    fi
    case ":${PYTHONPATH:-}:" in
        *":${path_entry}:"*) ;;
        *) export PYTHONPATH="${path_entry}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
}

# If a local Isaac Sim binary is present, source its env setup so that
# PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
# activate.d hook (those don't fire reliably under e.g. `conda run`).
if [ -d "$ISAACLAB_PATH/_isaac_sim" ]; then
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    elif [ -f "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" ]; then
        export ISAAC_PATH="$ISAACLAB_PATH/_isaac_sim"
        export CARB_APP_PATH="$ISAAC_PATH/kit"
        export EXP_PATH="$ISAAC_PATH/apps"
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" >/dev/null 2>&1 || true
        # Unlike setup_conda_env.sh, setup_python_env.sh prepends Kit's
        # pip_prebundle directories to PYTHONPATH. Those ship vendored copies of
        # common libraries (e.g. an older typing_extensions lacking Sentinel)
        # that then shadow the active venv/conda environment. Put the active
        # environment's site-packages first so it always wins.
        if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
            env_site_packages="$("$python_exe" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
            if [ -n "$env_site_packages" ]; then
                export PYTHONPATH="$env_site_packages:$PYTHONPATH"
            fi
        fi
    else
        echo "[WARNING] _isaac_sim is present but _isaac_sim/setup_conda_env.sh or _isaac_sim/setup_python_env.sh is missing; Isaac Sim env vars not exported." >&2
        echo "[WARNING] Re-extract the Isaac Sim binary zip if you intend to use the bundled binary." >&2
    fi
fi

# Isaac Sim's setup scripts prepend ``pip_prebundle`` paths that can shadow the
# active environment's packages (for example typing_extensions used by W&B).
# Put the selected interpreter's site-packages back in front after sourcing
# Isaac Sim so venv/conda packages win while the prebundle remains available as
# a fallback.
site_packages_paths=$("$python_exe" - <<'PY' 2>/dev/null || true
import site
import sysconfig

paths = []
for path in site.getsitepackages():
    paths.append(path)
for key in ("purelib", "platlib"):
    path = sysconfig.get_paths().get(key)
    if path:
        paths.append(path)

seen = set()
for path in paths:
    if path not in seen:
        seen.add(path)
        print(path)
PY
)
while IFS= read -r site_packages_path; do
    _prepend_pythonpath "$site_packages_path"
done <<< "$site_packages_paths"
_prepend_pythonpath "$ISAACLAB_PATH/source/isaaclab"

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
