# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for physics tests."""

import pytest


def pytest_addoption(parser):
    """Add --visualize command line option for Newton visualizer debugging."""
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable Newton visualizer for debugging physics tests",
    )


@pytest.fixture
def visualize(request):
    """Fixture to check if visualization is enabled."""
    return request.config.getoption("--visualize")
