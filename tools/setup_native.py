from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_SOURCES = sorted(
    str(path)
    for path in (REPO_ROOT / "game" / "src").glob("*.cpp")
    if path.name != "main.cpp"
)

ext_modules = [
    Pybind11Extension(
        "SDK.native_antwar",
        [str(REPO_ROOT / "SDK" / "native_antwar.cpp"), *GAME_SOURCES],
        include_dirs=[
            str(REPO_ROOT),
            str(REPO_ROOT / "game" / "include"),
        ],
        cxx_std=17,
    )
]

setup(
    name="agent_tradition_native",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
