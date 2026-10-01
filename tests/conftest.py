"""Run the offline suite without accidentally depending on a graphics stack."""

import importlib.abc
import sys

import pytest


SIMULATOR_MODULES = frozenset({"mujoco", "metaworld", "mani_skill", "sapien", "OpenGL", "glfw"})


class NoSimulatorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in SIMULATOR_MODULES:
            raise RuntimeError(f"Offline tests attempted to import simulator/graphics module: {fullname}")


def pytest_addoption(parser):
    parser.addoption(
        "--no-simulator-imports", action="store_true",
        help="Fail if offline tests import simulator or graphics libraries, including during collection.",
    )


def pytest_configure(config):
    if config.getoption("--no-simulator-imports"):
        loaded = sorted(SIMULATOR_MODULES.intersection(sys.modules))
        if loaded:
            raise pytest.UsageError(f"Simulator modules loaded before offline import guard: {loaded}")
        sys.meta_path.insert(0, NoSimulatorImports())


def pytest_unconfigure(config):
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, NoSimulatorImports)]
