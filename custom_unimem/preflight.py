"""Validate a dataset before training on it. Run this first on any new machine or copy.

Checks, in the order they would bite you:

  1. the dataset named in robot_paths.py is a real directory on this machine
  2. LeRobot v2.1 layout, 30 fps, the three expected cameras, 16-dim state/action
  3. the grippers really are at dims 14/15 rather than interleaved at 7/15 — the single
     mistake that silently drives every arm joint to twice its intended angle
  4. every subtask string in meta/tasks.jsonl maps to an event in event_vocab.py
  5. the `labels` / `phase_history` columns exist, with a sane event distribution
  6. norm stats exist for the shared asset id AND were computed from this dataset
     (needs openpi; skipped if unavailable)

Usage:
    uv run python custom_unimem/preflight.py --robot astribot
    python3 custom_unimem/preflight.py --robot astribot     # steps 1-5 need no openpi
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import event_vocab as _event_vocab
from custom_unimem import robot_paths as _robot_paths

EXPECTED_CAMERAS = (
    "observation.images.cam_head",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
EXPECTED_DIM = 16
EXPECTED_FPS = 30
GRIPPER_DIMS = (14, 15)

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


class Report:
    def __init__(self) -> None:
        self.failed = False

    def __call__(self, status: str, message: str) -> None:
        print(f"[{status}] {message}")
        if status is FAIL:
            self.failed = True


def check_layout(report: Report, dataset_dir: pathlib.Path) -> dict:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    version = str(info.get("codebase_version", "?"))
    report(OK if version.startswith("v2.1") else FAIL, f"codebase_version = {version} (openpi pins LeRobot v2.1)")
    report(
        OK if info["fps"] == EXPECTED_FPS else WARN,
        f"fps = {info['fps']}, episodes = {info['total_episodes']}, frames = {info['total_frames']}",
    )

    features = info["features"]
    for camera in EXPECTED_CAMERAS:
        if camera in features:
            report(OK, f"camera {camera} {features[camera].get('shape')}")
        else:
            report(FAIL, f"camera {camera} MISSING — the repack transform looks it up by this exact name")

    for key in ("observation.state", "action"):
        shape = features.get(key, {}).get("shape")
        report(
            OK if shape == [EXPECTED_DIM] else FAIL,
            f"{key} shape = {shape} (expected [{EXPECTED_DIM}])",
        )

    names = features.get("observation.state", {}).get("names")
    flat = names[0] if names and isinstance(names[0], list) else names
    if flat and len(flat) == EXPECTED_DIM:
        trailing = [flat[14], flat[15]]
        report(
            OK if all("grip" in n.lower() for n in trailing) else WARN,
            f"state dims 14/15 named {trailing} — should be the two grippers",
        )
    return info


def check_gripper_layout(report: Report, files: list[pathlib.Path]) -> None:
    """Distinguish gripper channels from arm joints by their statistics, not their names.

    Grippers are bimodal and span a far wider range than a radian arm joint. If dim 7
    looks more gripper-like than dim 14, the dataset uses the old interleaved layout and
    the delta-action mask in data_configs.py is wrong for it.
    """
    sample = np.concatenate(
        [
            np.stack(pq.read_table(f, columns=["observation.state"]).column("observation.state").to_pylist())
            for f in files[:20]
        ]
    )
    spans = sample.max(axis=0) - sample.min(axis=0)
    trailing = float(min(spans[d] for d in GRIPPER_DIMS))
    arm_like = float(np.median([spans[d] for d in range(14)]))
    report(
        OK if trailing > arm_like else FAIL,
        f"dims 14/15 span >= {trailing:.2f} vs median arm-joint span {arm_like:.2f} "
        f"(dim 7 spans {spans[7]:.2f}) — grippers trail the arms",
    )


def check_vocabulary(report: Report, dataset_dir: pathlib.Path, vocab: _event_vocab.EventVocab) -> None:
    tasks = [
        json.loads(line)["task"]
        for line in (dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    unmatched = []
    for task in tasks:
        event_id = vocab.event_for_task(task)
        ignored = any(
            _event_vocab.normalize_task(task).startswith(_event_vocab.normalize_task(p))
            for p in vocab.ignore_task_prefixes
        )
        if event_id is None and not ignored:
            unmatched.append(task)
        elif event_id is not None:
            report(OK, f"subtask -> event {event_id} ({vocab.phrases[event_id]!r})")
    for task in unmatched:
        report(FAIL, f"subtask matches no event and is not ignored: {task!r}")
    if unmatched:
        print("       -> run label_dataset_subtasks.py --auto-vocab and update event_vocab.py")


def check_labels(report: Report, files: list[pathlib.Path], vocab: _event_vocab.EventVocab) -> None:
    missing = [f for f in files if "labels" not in pq.ParquetFile(f).schema_arrow.names]
    if missing:
        report(FAIL, f"{len(missing)}/{len(files)} episodes have no `labels` column")
        print(f"       -> uv run python custom_unimem/label_dataset_subtasks.py --robot {vocab.name} --write-parquet")
        return

    counts: collections.Counter[int] = collections.Counter()
    sequences: collections.Counter[tuple[int, ...]] = collections.Counter()
    frames = 0
    for f in files:
        table = pq.read_table(f, columns=["labels", "phase_history"])
        labels = np.asarray(table.column("labels"))
        frames += len(labels)
        counts.update(labels.tolist())
        runs = labels[np.concatenate(([True], np.diff(labels) != 0))]
        sequences[tuple(int(x) for x in runs if x >= 0)] += 1
        if not table.column("phase_history").to_pylist()[0].startswith("History:"):
            report(FAIL, f"{f.name}: phase_history does not start with 'History:'")
            return

    labeled = sum(v for k, v in counts.items() if k >= 0)
    report(OK, f"labels + phase_history present on {len(files)}/{len(files)} episodes")
    report(
        OK if 0.02 < labeled / frames < 0.5 else WARN,
        f"{labeled}/{frames} frames inside an event window ({100 * labeled / frames:.1f}%)",
    )
    for event_id in sorted(k for k in counts if k >= 0):
        if event_id not in vocab.phrases:
            report(FAIL, f"label {event_id} has no phrase in the vocabulary")
        else:
            report(OK, f"  event {event_id} {vocab.phrases[event_id]:<45} {counts[event_id]:>7} frames")
    print("       event sequences per episode:")
    for sequence, count in sequences.most_common(5):
        print(f"         {count:>5}  {sequence}")


def check_norm_stats(report: Report, robot: str, dataset_frames: int) -> None:
    try:
        from custom_unimem import _bootstrap

        _bootstrap.bootstrap([robot])
        import openpi.training.config as _config
    except ImportError as e:
        report(WARN, f"openpi not importable ({e}); skipping the norm-stats check")
        return

    name = f"pi05_{robot}_unimem_event_full"
    config = _config.get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        report(FAIL, f"no norm stats for asset '{data_config.asset_id}'")
        print(f"       -> uv run python custom_unimem/compute_norm_stats_fast.py {name} --verify")
        return
    report(OK, f"norm stats found for asset '{data_config.asset_id}'")

    # The asset id is shared across configs AND across copies of the same dataset on
    # different machines. That portability is the point, but it also means stats computed
    # from a DIFFERENT dataset would be applied here without complaint, silently
    # normalizing every state and action wrongly. Compare against what was recorded.
    assets_dir = pathlib.Path(config.data.assets.assets_dir or config.assets_dirs)
    source_file = assets_dir / data_config.asset_id / _robot_paths.NORM_STATS_SOURCE_FILE
    if not source_file.is_file():
        report(WARN, f"no {_robot_paths.NORM_STATS_SOURCE_FILE} beside the stats — cannot verify what they came from")
        return
    source = json.loads(source_file.read_text())
    if source.get("provided"):
        # Supplied by hand rather than computed here, so there is no frame count of ours
        # to compare against. Say so plainly instead of implying a check that never ran.
        report(WARN, f"stats were supplied by hand for {source.get('repo_ids')} — not verified against this dataset")
        if source.get("cross_checked"):
            report(OK, f"  recorded cross-check: {source['cross_checked'].get('against')}")
        return
    recorded = source.get("frames")
    if recorded is None:
        report(WARN, f"stats came from a --max-frames subset of {source.get('repo_ids')}; frame count not comparable")
    elif recorded == dataset_frames:
        report(OK, f"stats were computed from {recorded} frames of {source.get('repo_ids')} — matches this dataset")
    else:
        report(FAIL, f"stats cover {recorded} frames of {source.get('repo_ids')} but this dataset has {dataset_frames}")
        print(f"       -> uv run python custom_unimem/compute_norm_stats_fast.py {name} --verify")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=sorted(_event_vocab.VOCABS))
    ap.add_argument("--dataset-root", default=None, help="override the resolved dataset directory")
    args = ap.parse_args()

    paths = _robot_paths.get(args.robot)
    vocab = _event_vocab.get_vocab(paths.vocab)
    dataset_dir = (
        pathlib.Path(args.dataset_root) if args.dataset_root else pathlib.Path(paths.lerobot_home) / paths.repo_id
    )

    report = Report()
    print(f"robot   : {paths.robot}")
    print(f"dataset : {dataset_dir}")
    print(f"prompt  : {paths.prompt!r}")
    print(f"asset   : {paths.asset_id}\n")

    if not (dataset_dir / "meta" / "info.json").is_file():
        report(FAIL, f"no LeRobot dataset at {dataset_dir}")
        print(
            "       -> edit dataset_home/dataset_repo_id in robot_paths.py, or export "
            f"{args.robot.upper()}_LEROBOT_HOME / {args.robot.upper()}_REPO_ID to use another copy"
        )
        raise SystemExit(1)

    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    report(OK if files else FAIL, f"{len(files)} episode parquet files")
    if not files:
        raise SystemExit(1)

    info = check_layout(report, dataset_dir)
    check_gripper_layout(report, files)
    check_vocabulary(report, dataset_dir, vocab)
    check_labels(report, files, vocab)
    check_norm_stats(report, args.robot, int(info["total_frames"]))

    print()
    if report.failed:
        print("PREFLIGHT FAILED — fix the items marked FAIL before training.")
        raise SystemExit(1)
    print("PREFLIGHT OK — ready to train.")


if __name__ == "__main__":
    main()
