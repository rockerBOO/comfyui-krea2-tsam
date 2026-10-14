#!/bin/sh
# pytest would import the package __init__.py as a top-level module; run from tests/ with an isolated rootdir instead.
cd "$(dirname "$0")/tests" || exit 1
PYTHONPATH=~/sd/ComfyUI PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec uv run --no-sync --project ~/sd/ComfyUI pytest --import-mode=importlib -c /dev/null --rootdir=/tmp -q . "$@"
