import importlib.util
import os
import sys

_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("krea2_tsam", os.path.join(_pkg, "__init__.py"), submodule_search_locations=[_pkg])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["krea2_tsam"] = _mod
_spec.loader.exec_module(_mod)
