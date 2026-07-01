# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import sys

from isaaclab import _deprioritize_prebundle_paths


def test_environment_package_precedes_kit_bundle(monkeypatch, tmp_path):
    """Ensure environment packages win while the Kit bundle remains a fallback."""
    kit_site_packages = tmp_path / "isaac-sim" / "kit" / "python" / "lib" / "python3.12" / "site-packages"
    env_site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    package_name = "isaaclab_path_order_probe"
    for site_packages, owner in ((kit_site_packages, "kit"), (env_site_packages, "environment")):
        package = site_packages / package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'OWNER = "{owner}"\n')

    original_path = sys.path.copy()
    monkeypatch.setattr(sys, "path", [str(kit_site_packages), str(env_site_packages), *original_path])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(kit_site_packages), str(env_site_packages))))

    _deprioritize_prebundle_paths()
    imported = importlib.import_module(package_name)

    assert imported.OWNER == "environment"
    assert sys.path.index(str(env_site_packages)) < sys.path.index(str(kit_site_packages))
    assert os.environ["PYTHONPATH"].split(os.pathsep)[-1] == str(kit_site_packages)
    sys.modules.pop(package_name, None)
