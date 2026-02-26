# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper for declaring lazy imports in ``__init__.py`` files."""

from __future__ import annotations

import sys
from collections.abc import Iterable

import lazy_loader as lazy


def lazy_export(*imports: tuple[str, str | Iterable[str]], submodules: list[str] | None = None) -> None:
    """Register lazy imports for the calling package's ``__init__.py``.

    This replaces the common boilerplate::

        import lazy_loader as lazy

        __getattr__, __dir__, __all__ = lazy.attach(
            __name__,
            submod_attrs={
                "my_module": ["MyClass", "my_func"],
            },
        )

    with a more concise form::

        from isaaclab.utils.lazy_imports import lazy_export

        lazy_export(
            ("my_module", ["MyClass", "my_func"]),
        )

    Args:
        *imports: Each element is a ``(submodule, names)`` pair where *submodule*
            is the module file (without ``.py``) and *names* is either a single
            string or an iterable of strings to re-export from that submodule.
        submodules: Optional list of sub-packages to expose as direct attributes
            (e.g. ``submodules=["converters", "schemas"]``).
    """
    caller_globals = sys._getframe(1).f_globals
    package_name = caller_globals["__name__"]

    submod_attrs: dict[str, list[str]] = {}
    for submod, names in imports:
        submod_attrs[submod] = [names] if isinstance(names, str) else list(names)

    __getattr__, __dir__, __all__ = lazy.attach(package_name, submodules=submodules or [], submod_attrs=submod_attrs)

    mod = sys.modules[package_name]
    mod.__getattr__ = __getattr__
    mod.__dir__ = __dir__
    mod.__all__ = __all__
