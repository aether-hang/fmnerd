"""Load the DDiF-derived defaults from ``TM/hyper_params.py``.

Using an explicit module spec avoids a name collision when a method folder has
its own ``hyper_params.py`` on ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import os


def _load():
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(root, "TM", "hyper_params.py")
    spec = importlib.util.spec_from_file_location("fmnerd_ddif_hyper_params", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_method_defaults(args):
    return _load().load_default(args)
