"""Compatibility alias: the manuscript and Online Resource 1 refer to the package as `nwpc`.

`nwpc` and `pavecarbon` are the same code; `import nwpc.analysis` works as before.
"""
import importlib
import sys

import pavecarbon
from pavecarbon import __version__  # noqa: F401

for _m in ("fe_axisym", "pavement", "damage", "lca", "analysis", "plots"):
    sys.modules[f"nwpc.{_m}"] = importlib.import_module(f"pavecarbon.{_m}")
