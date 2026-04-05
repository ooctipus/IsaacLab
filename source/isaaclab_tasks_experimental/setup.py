# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_tasks_experimental' python package."""

import os
import platform
import sys

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = ["isaaclab_tasks"]

is_linux_x86_64 = (platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"))
py = f"cp{sys.version_info.major}{sys.version_info.minor}"

wheel_by_py = {
    "cp312": "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/"
             "pytorch3d-0.7.9%2Bpt2.10.0cu128-cp312-cp312-linux_x86_64.whl",
    "cp311": "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/"
             "pytorch3d-0.7.9%2Bpt2.10.0cu128-cp311-cp311-linux_x86_64.whl",
    "cp310": "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/"
             "pytorch3d-0.7.9%2Bpt2.10.0cu128-cp310-cp310-linux_x86_64.whl",
}

if is_linux_x86_64 and py in wheel_by_py:
    INSTALL_REQUIRES.append(f"pytorch3d @ {wheel_by_py[py]}")

# Installation operation
setup(
    name="isaaclab_tasks_experimental",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=["isaaclab_tasks"],
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
