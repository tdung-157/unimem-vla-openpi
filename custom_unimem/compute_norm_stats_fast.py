"""Norm stats straight from the parquet, skipping video decoding.

    uv run python custom_unimem/compute_norm_stats_fast.py pi05_astribot_unimem_event_full --verify

Only ``state`` and ``actions`` go into norm stats, but the stock pipeline pulls every
sample through the full data loader — which decodes the current frame (and, for keyframe
configs, the event frames) for all three cameras. On 817k frames that dominates the
runtime for numbers that never look at a pixel.

This reads the numeric columns out of the episode parquets, builds the action chunk the
same way ``delta_timestamps`` does, and then runs the **real** repack + data transforms
over each row with a 1x1 dummy image. Reimplementing the transforms would silently drift
whenever they change; feeding rows through the actual transform objects means the delta
conversion, the gripper mask and the 16-dim slice are exactly the ones training sees.

``--verify`` builds the real pipeline and cross-checks agreement on a sample of frames
before writing anything, so a mismatch in the chunking convention is caught rather than
silently poisoning training. Run it the first time on a new dataset shape.

Adapted from scripts/compute_norm_stats_fast.py, which hardcodes the HuggingFace cache
root and an ``actions`` column name; these datasets live under ``$HF_LEROBOT_HOME`` and
name that column ``action``.
"""

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import _bootstrap
from custom_unimem import robot_paths

_bootstrap.bootstrap()

import openpi.shared.normalize as normalize  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.transforms as transforms  # noqa: E402


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


def write_source(out: pathlib.Path, repo_ids: list[str], frames: int, horizon: int, config_name: str) -> None:
    """Record what these stats were computed from.

    The asset id is shared across every config of a robot AND across copies of the same
    dataset on different machines, which is what makes the stats portable — and also what
    would let stats from one dataset be applied silently to a different one. preflight.py
    compares this file against the dataset it resolves and refuses a mismatch.
    """
    (out / robot_paths.NORM_STATS_SOURCE_FILE).write_text(
        json.dumps(
            {"repo_ids": repo_ids, "frames": frames, "action_horizon": horizon, "config": config_name},
            indent=2,
        )
        + "\n"
    )


def dataset_root(repo_id: str) -> pathlib.Path:
    home = os.environ.get("HF_LEROBOT_HOME") or str(pathlib.Path.home() / ".cache/huggingface/lerobot")
    return pathlib.Path(home) / repo_id


def episode_files(repo_id: str) -> list[pathlib.Path]:
    root = dataset_root(repo_id)
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found under {root}")
    return files


def collect(repo_ids: list[str], horizon: int, action_key: str, transform) -> tuple[np.ndarray, np.ndarray]:
    """Read numeric columns, build the action chunk, run the real transform stack."""
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    # The repack transform looks these up by name; the pixels are discarded downstream by
    # the norm-stats keys, but the image path still has to run.
    from custom_unimem.data_configs import CAM_KEYS

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for repo_id in repo_ids:
        files = episode_files(repo_id)
        for n, path in enumerate(files, start=1):
            df = pd.read_parquet(path)
            a = np.stack(df[action_key].to_numpy()).astype(np.float32)
            rows = len(a)
            # delta_timestamps yields frames t..t+horizon-1, clamped at the episode end.
            idx = np.minimum(np.arange(rows)[:, None] + np.arange(horizon)[None, :], rows - 1)
            chunks = a[idx]
            records = df.to_dict("records")
            for i in range(rows):
                row = dict(records[i])
                row[action_key] = chunks[i]
                for cam in CAM_KEYS:
                    row[cam] = dummy
                # PromptFromLeRobotTask / InjectDefaultPrompt add this downstream of the
                # parquet; it does not affect state or actions, so any string will do.
                row.setdefault("prompt", "")
                out = transform(row)
                states.append(np.asarray(out["state"], dtype=np.float32))
                actions.append(np.asarray(out["actions"], dtype=np.float32))
            if n % 50 == 0 or n == len(files):
                print(f"  {repo_id}: {n}/{len(files)} episodes, {len(states)} frames")
    return np.stack(states), np.stack(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_name", nargs="?", default=None)
    ap.add_argument("--config-name", dest="config_name_flag", default=None)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="cross-check against the real pipeline on --verify-n samples first; "
        "exits non-zero on mismatch rather than writing bad stats",
    )
    ap.add_argument("--verify-n", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="overwrite norm stats that were supplied by hand")
    args = ap.parse_args()
    config_name = args.config_name_flag or args.config_name
    if not config_name:
        raise SystemExit("provide a config name, positionally or via --config-name")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_ids = list(data_config.repo_ids) if data_config.repo_ids else [data_config.repo_id]
    horizon = config.model.action_horizon
    (action_key,) = data_config.action_sequence_keys
    print(f"repos: {repo_ids}   action_horizon: {horizon}   action column: {action_key!r}")

    shared = getattr(config.data.assets, "assets_dir", None)
    asset = data_config.asset_id or data_config.repo_id
    refuse_to_clobber_provided(
        [config.assets_dirs / asset] + ([pathlib.Path(shared) / asset] if shared else []), args.force
    )

    chain = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    print(
        "transforms: "
        + " -> ".join(
            type(t).__name__ for t in (*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs)
        )
    )
    states, actions = collect(repo_ids, horizon, action_key, chain)
    print(f"frames: {len(states)}   state {states.shape[1:]}   actions {actions.shape[1:]}")

    if args.verify:
        import openpi.training.data_loader as data_loader

        class RemoveStrings(transforms.DataTransformFn):
            def __call__(self, x):
                return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

        print("building the real pipeline to cross-check ...")
        dataset = data_loader.create_torch_dataset(data_config, horizon, config.model)
        transformed = data_loader.TransformedDataset(
            dataset,
            [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
        )
        if len(transformed) != len(states):
            raise SystemExit(f"length mismatch: pipeline {len(transformed)} vs fast path {len(states)}")
        rng = np.random.RandomState(0)
        picks = rng.choice(len(transformed), size=min(args.verify_n, len(transformed)), replace=False)
        state_err = action_err = 0.0
        for i in picks:
            ref = transformed[int(i)]
            state_err = max(state_err, float(np.abs(np.asarray(ref["state"]) - states[i]).max()))
            action_err = max(action_err, float(np.abs(np.asarray(ref["actions"]) - actions[i]).max()))
        print(f"  max |state| diff over {len(picks)} samples: {state_err:.3e}")
        print(f"  max |actions| diff:                        {action_err:.3e}")
        if state_err > 1e-4 or action_err > 1e-4:
            raise SystemExit(
                "MISMATCH — refusing to write. The fast path does not reproduce the pipeline; "
                "fix the chunking convention before trusting this."
            )
        print("  MATCH — fast path reproduces the pipeline exactly")

    stats = {}
    for key, arr in (("state", states), ("actions", actions.reshape(-1, actions.shape[-1]))):
        running = normalize.RunningStats()
        for i in range(0, len(arr), 8192):
            running.update(arr[i : i + 8192])
        stats[key] = running.get_statistics()

    asset_id = data_config.asset_id or data_config.repo_id
    out = config.assets_dirs / asset_id
    print(f"writing stats to: {out}")
    normalize.save(out, stats)
    write_source(out, repo_ids, len(states), horizon, config_name)

    shared_dir = getattr(config.data.assets, "assets_dir", None)
    if shared_dir is not None:
        target = pathlib.Path(shared_dir) / asset_id
        if target.resolve() != pathlib.Path(out).resolve():
            target.mkdir(parents=True, exist_ok=True)
            for name in ("norm_stats.json", robot_paths.NORM_STATS_SOURCE_FILE):
                shutil.copyfile(pathlib.Path(out) / name, target / name)
            print(f"copied stats to the shared assets dir: {target}")


if __name__ == "__main__":
    main()
