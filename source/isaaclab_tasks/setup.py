# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_tasks' python package."""

import os
import toml
import platform
import sys
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch>=2.7",
    "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    "protobuf>=4.25.8,!=5.26.0",
    # basic logger
    "tensorboard",
    # automate
    "scikit-learn",
    "numba",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

is_linux_x86_64 = (platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"))
py = f"cp{sys.version_info.major}{sys.version_info.minor}"

wheel_by_py = {
    "cp311": "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
             "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp311-cp311-linux_x86_64.whl",
    "cp310": "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
             "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp310-cp310-linux_x86_64.whl",
}

if is_linux_x86_64 and py in wheel_by_py:
    INSTALL_REQUIRES.append(f"pytorch3d @ {wheel_by_py[py]}")

# Installation operation
setup(
    name="isaaclab_tasks",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["isaaclab_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
