# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for externally contributed components for Isaac Lab.

This package provides externally contributed components for Isaac Lab, such as multirotors.
These components are not part of the core Isaac Lab framework yet, but are planned to be added
in the future. They are contributed by the community to extend the capabilities of Isaac Lab.
"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_CONTRIB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_CONTRIB_METADATA = toml.load(os.path.join(ISAACLAB_CONTRIB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_CONTRIB_METADATA["package"]["version"]

# Import all manipulation tasks to register gym environments
# This follows the pattern from isaaclab_tasks and isaaclab_contrib
from isaaclab_tasks.utils import import_packages

# Blacklist: prevent importing internal utilities and MDP modules
_BLACKLIST_PKGS = ["mixin_utils", ".mdp"]

# Import all task configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
