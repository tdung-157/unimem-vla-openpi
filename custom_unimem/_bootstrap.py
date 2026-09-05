"""Shared bootstrap for the UniMem wrapper scripts.

Registers both robots' configs into openpi's global registry and points
``HF_LEROBOT_HOME`` at the right dataset root, then lets the stock openpi entrypoints
(``scripts/train.py``, ``scripts/serve_policy.py``, ``scripts/compute_norm_stats.py``) do
the actual work — nothing in ``src/openpi`` needs to know these configs exist.

``HF_LEROBOT_HOME`` is a single process-wide variable but the two robots keep their data
in different places, so it is chosen from the config name on the command line. An
explicitly exported ``HF_LEROBOT_HOME`` always wins (the ``run_*.sh`` scripts set it).
"""

import importlib.util
import os
import pathlib
import sys
import types

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _dataset_roots() -> dict[str, str]:
    from custom_unimem import astribot_config
    from custom_unimem import motion2_config

    return {
        astribot_config.ROBOT: astribot_config.ASTRIBOT_LEROBOT_HOME,
        motion2_config.ROBOT: motion2_config.MOTION2_LEROBOT_HOME,
    }


def _robot_from_argv(argv: list[str]) -> str | None:
    """Pick the robot whose name appears in a config name on the command line."""
    roots = _dataset_roots()
    matches = {robot for arg in argv for robot in roots if robot in arg}
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise SystemExit(f"ambiguous robot in argv (matched {sorted(matches)}); set HF_LEROBOT_HOME yourself")
    return None


def bootstrap(argv: list[str] | None = None) -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from custom_unimem import astribot_config
    from custom_unimem import motion2_config

    astribot_config.register()
    motion2_config.register()

    if "HF_LEROBOT_HOME" in os.environ:
        return
    robot = _robot_from_argv(sys.argv if argv is None else argv)
    if robot is not None:
        os.environ["HF_LEROBOT_HOME"] = _dataset_roots()[robot]


def load_repo_script(module_name: str, rel_path: str) -> types.ModuleType:
    """Import a top-level repo script (e.g. ``scripts/train.py``) as a module.

    The scripts/ directory is not a package, so this loads by path rather than by name.
    """
    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
