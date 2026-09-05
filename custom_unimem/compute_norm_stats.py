"""Compute normalization statistics for a UniMem config.

    uv run python custom_unimem/compute_norm_stats.py pi05_astribot_unimem_event \
        --max-frames 200000

Two things this adds over the stock script:

1. It accepts the config name positionally as well as via ``--config-name``.
2. After the stock script writes to ``assets/<config name>/<asset_id>``, it copies the
   result into the robot's SHARED assets directory (``assets/<robot>_unimem/<asset_id>``),
   which is where every one of that robot's configs actually reads from. Norm stats depend
   only on state/actions, so all five configs share one set and you only compute it once.

On these datasets (817k frames for MOTION2) the stock path decodes video for every sample
even though only state/actions are used, which is slow. Either pass ``--max-frames``
(100-200k randomly sampled frames give stats that are indistinguishable in practice) or
use ``compute_norm_stats_fast.py``, which reads the parquet directly.
"""

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import _bootstrap
from custom_unimem import robot_paths

_bootstrap.bootstrap()

import tyro  # noqa: E402

import openpi.training.config as _config  # noqa: E402

_compute = _bootstrap.load_repo_script("openpi_compute_norm_stats_unimem", "scripts/compute_norm_stats.py")


def refuse_to_clobber_provided(paths: "list[pathlib.Path]", force: bool) -> None:
    """Abort rather than overwrite norm stats that were supplied by hand.

    A hand-pasted norm_stats.json is not reproducible from this repo — recomputing over
    it destroys it silently, and a wrong-but-plausible set of stats is close to
    undebuggable downstream. The sidecar marks the file; --force overrides.
    """
    for directory in paths:
        source_file = directory / robot_paths.NORM_STATS_SOURCE_FILE
        if not source_file.is_file():
            continue
        if json.loads(source_file.read_text()).get("provided") and not force:
            raise SystemExit(
                f"REFUSING to overwrite hand-provided norm stats in {directory}\n"
                f'  ({source_file.name} has "provided": true)\n'
                "  Pass --force to recompute them anyway, or delete that file first."
            )


def main(config_name: str, max_frames: int | None = None, force: bool = False) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id or data_config.repo_id
    shared = getattr(config.data.assets, "assets_dir", None)
    refuse_to_clobber_provided(
        [config.assets_dirs / asset_id] + ([pathlib.Path(shared) / asset_id] if shared else []), force
    )

    _compute.main(config_name, max_frames)

    written = config.assets_dirs / asset_id / "norm_stats.json"

    # Record what the stats were computed from; preflight.py checks it against the
    # dataset it resolves, since the asset id is shared across configs and machines.
    source = written.parent / robot_paths.NORM_STATS_SOURCE_FILE
    source.write_text(
        json.dumps(
            {
                "repo_ids": list(data_config.repo_ids) if data_config.repo_ids else [data_config.repo_id],
                "frames": None if max_frames else _dataset_frames(config, data_config),
                "action_horizon": config.model.action_horizon,
                "config": config_name,
            },
            indent=2,
        )
        + "\n"
    )

    shared_dir = getattr(config.data.assets, "assets_dir", None)
    if shared_dir is None:
        return
    target_dir = pathlib.Path(shared_dir) / asset_id
    if target_dir.resolve() == written.parent.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("norm_stats.json", robot_paths.NORM_STATS_SOURCE_FILE):
        shutil.copyfile(written.parent / name, target_dir / name)
    print(f"Copied stats to the shared assets dir: {target_dir}")


def _dataset_frames(config, data_config) -> int | None:
    """Frame count of the dataset these stats cover, or None if it cannot be determined.

    Only meaningful for a full pass: with --max-frames the stats come from a random
    subset, so no frame count is recorded and preflight skips the comparison.
    """
    try:
        import openpi.training.data_loader as data_loader

        return len(data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model))
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1:2] = ["--config-name", sys.argv[1]]
    tyro.cli(main)
