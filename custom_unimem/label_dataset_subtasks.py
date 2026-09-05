"""Turn per-frame subtask annotations into UniMem's ``labels`` + ``phase_history`` columns.

This is NOT a labeling pass in the annotation sense — nothing here asks a human for
anything. The dataset's existing subtask annotations (``meta/lerobot_annotations.json``,
already flattened by the converter into a per-frame ``task_index``) are the input; the
output is the two derived columns that the training pipeline's ``RepackTransform`` looks
up by name. The name mirrors upstream's ``examples/*/label_dataset_*.py``, which do the
same job from raw signals instead of annotations.

Both bimanual coffee datasets already annotate every frame with the subtask being
performed (LeRobot's ``task_index``). That makes event labeling a bookkeeping job rather
than a signal-processing one: an event fires where the subtask changes, and its phrase
describes the subtask that just **completed**. Compare with
``examples/xarm/label_dataset_xarm.py``, which has to detect its events from gripper and
pose signals because that dataset has no per-frame annotation.

Two columns are written into each episode parquet:

    labels        int32   -1 on unlabeled frames, 0..N-1 inside an event window
    phase_history string  "History: none", then "History: <phrase>, <phrase>, ..." of the
                          events completed so far

An event's phrase becomes visible only once the frame is no longer labeled with that
event — i.e. once its window has passed — matching how the served policy appends to its
history only after the event head has fired.

Usage
-----
Inspect what it would do (dry run; nothing is written)::

    uv run python custom_unimem/label_dataset_subtasks.py --robot motion2

Persist the columns::

    uv run python custom_unimem/label_dataset_subtasks.py --robot motion2 --write-parquet

Generate a starting vocabulary for a dataset whose subtask strings are not yet in
``event_vocab.py`` (prints a block to paste there, then exits)::

    uv run python custom_unimem/label_dataset_subtasks.py --robot astribot --auto-vocab
"""

import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import event_vocab as _event_vocab
from custom_unimem import robot_paths as _robot_paths

LABELS_COLUMN = "labels"
HISTORY_COLUMN = "phase_history"


def _robot_defaults(robot: str) -> tuple[str, str]:
    """(dataset root, repo id) for a robot. From robot_paths, so openpi is never imported."""
    paths = _robot_paths.get(robot)
    return paths.lerobot_home, paths.repo_id


def load_tasks(dataset_dir: pathlib.Path) -> dict[int, str]:
    path = dataset_dir / "meta" / "tasks.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path} — is this a LeRobot v2.1 dataset?")
    tasks: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def episode_files(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"no episode parquet under {dataset_dir / 'data'}")
    return files


def task_runs(task_index: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of a constant task index as ``(task, start, end_inclusive)``."""
    if len(task_index) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(task_index)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries - 1, [len(task_index) - 1]))
    return [(int(task_index[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def event_runs(
    task_index: np.ndarray, tasks: dict[int, str], vocab: _event_vocab.EventVocab
) -> list[tuple[int, int, int]]:
    """Event runs as ``(event_id, start, end_inclusive)``, in episode order.

    Runs whose task carries no event are dropped, and adjacent runs that map to the SAME
    event id are merged — the annotators spelled one MOTION2 subtask four different ways,
    and two spellings landing next to each other must not become two identical events in
    a row (the README's first "helpful tip": never label the same event twice running).
    """
    merged: list[tuple[int, int, int]] = []
    for task, start, end in task_runs(task_index):
        event_id = vocab.event_for_task(tasks.get(task, ""))
        if event_id is None:
            continue
        if merged and merged[-1][0] == event_id and merged[-1][2] == start - 1:
            prev_id, prev_start, _ = merged[-1]
            merged[-1] = (prev_id, prev_start, end)
        else:
            merged.append((event_id, start, end))
    return merged


def label_episode(
    task_index: np.ndarray,
    tasks: dict[int, str],
    vocab: _event_vocab.EventVocab,
    *,
    pre: int,
    post: int,
    label_final_event: bool,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Label one episode.

    Returns ``(labels, phase_history, fired)`` where ``fired`` lists ``(event_id, frame)``
    detection points, for reporting.

    The detection point for an event is the LAST frame of the subtask it completes, and
    the window spans ``[d - pre, d + post]``. Widening the window past the transition is
    deliberate: at rollout the event head sees the aftermath of an event (the cup now
    under the faucet), not the instant of the transition, so training on a window around
    it is what makes single-frame detection learnable at all.
    """
    n = len(task_index)
    labels = np.full(n, _event_vocab.NULL_LABEL, dtype=np.int32)
    runs = event_runs(task_index, tasks, vocab)
    if runs and not label_final_event and runs[-1][2] == n - 1:
        runs = runs[:-1]

    windows: list[tuple[int, int, int]] = []  # (event_id, window start, window end incl.)
    for event_id, _, end in runs:
        lo = max(0, end - pre)
        hi = min(n - 1, end + post)
        windows.append((event_id, lo, hi))
        # A later window overwrites an earlier one where they overlap: the most recent
        # event is the one the policy should be reporting at that frame.
        labels[lo : hi + 1] = event_id

    history: list[str] = []
    completed: list[str] = []
    cursor = 0
    for event_id, _, hi in windows:
        text = vocab.format_history(completed)
        # Frames up to and including the end of this window still show the history as it
        # was before the event completed.
        history.extend([text] * (min(hi + 1, n) - cursor))
        cursor = min(hi + 1, n)
        phrase = vocab.phrases[event_id]
        # Guard against a repeat sneaking in via overlapping windows.
        if not completed or completed[-1] != phrase:
            completed.append(phrase)
    history.extend([vocab.format_history(completed)] * (n - cursor))

    fired = [(event_id, end) for event_id, _, end in runs]
    return labels, history, fired


def auto_vocab(dataset_dir: pathlib.Path, files: list[pathlib.Path], tasks: dict[int, str]) -> None:
    """Print an ``EventVocab`` skeleton built from the dataset's own subtask order."""
    order: list[int] = []
    counts: collections.Counter[int] = collections.Counter()
    for path in files:
        task_index = pq.read_table(path, columns=["task_index"]).column("task_index").to_numpy()
        for task, _, _ in task_runs(task_index):
            counts[task] += 1
            if task not in order:
                order.append(task)

    print(f"\nSubtasks in first-appearance order ({dataset_dir}):\n")
    for event_id, task in enumerate(order):
        print(f"  event {event_id}: [{counts[task]:>5} runs] task_index={task}  {tasks.get(task, '<missing>')!r}")

    print("\nPaste into custom_unimem/event_vocab.py, then EDIT the phrases so each one")
    print("reads as a COMPLETION ('grabbed the cup', not 'grab the cup'), collapse any")
    print("duplicate spellings of one subtask onto a single id, and move the")
    print("episode-level task string into ignore_task_prefixes:\n")
    print("MY_VOCAB = EventVocab(")
    print('    name="<robot>",')
    print("    phrases={")
    for event_id, task in enumerate(order):
        print(f'        {event_id}: "<completed: {tasks.get(task, "?")}>",')
    print("    },")
    print("    task_prefixes={")
    for event_id, task in enumerate(order):
        print(f'        "{tasks.get(task, "?")}": {event_id},')
    print("    },")
    print("    ignore_task_prefixes=(),")
    print(")")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=sorted(_event_vocab.VOCABS), help="which event vocabulary to use")
    ap.add_argument(
        "--dataset-root", default=None, help="override the dataset directory (default: from the robot config)"
    )
    ap.add_argument(
        "--pre", type=int, default=5, help="frames before the subtask transition to include in the event window"
    )
    ap.add_argument(
        "--post", type=int, default=25, help="frames after the transition to include (default ~1 s at 30 fps)"
    )
    ap.add_argument("--limit-episodes", type=int, default=None, help="only process the first N episodes")
    ap.add_argument("--no-final-event", action="store_true", help="skip the event for the last subtask of each episode")
    ap.add_argument("--write-parquet", action="store_true", help="actually persist the columns (default: dry run)")
    ap.add_argument(
        "--overwrite-columns", action="store_true", help="allow replacing labels/phase_history that already exist"
    )
    ap.add_argument("--auto-vocab", action="store_true", help="print an EventVocab skeleton for this dataset and exit")
    ap.add_argument("--show", type=int, default=2, help="print a frame-by-frame timeline for the first N episodes")
    args = ap.parse_args()

    home, repo_id = _robot_defaults(args.robot)
    home = os.environ.get("HF_LEROBOT_HOME", home)
    dataset_dir = pathlib.Path(args.dataset_root) if args.dataset_root else pathlib.Path(home) / repo_id
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset not found: {dataset_dir}")

    tasks = load_tasks(dataset_dir)
    files = episode_files(dataset_dir)
    if args.limit_episodes is not None:
        files = files[: args.limit_episodes]

    if args.auto_vocab:
        auto_vocab(dataset_dir, files, tasks)
        return

    vocab = _event_vocab.get_vocab(args.robot)
    print(f"dataset : {dataset_dir}")
    print(f"vocab   : {vocab.name} ({vocab.num_event_classes} events, window [-{args.pre}, +{args.post}] frames)")
    print(f"mode    : {'WRITING parquet' if args.write_parquet else 'dry run (pass --write-parquet to persist)'}\n")

    unknown = {
        task
        for task in tasks.values()
        if vocab.event_for_task(task) is None
        and not any(
            _event_vocab.normalize_task(task).startswith(_event_vocab.normalize_task(p))
            for p in vocab.ignore_task_prefixes
        )
    }
    if unknown:
        print("Subtasks matched by neither task_prefixes nor ignore_task_prefixes:")
        for task in sorted(unknown):
            print(f"  {task!r}")
        raise SystemExit(
            "Refusing to label: every subtask must be classified explicitly, or an event you "
            "care about silently becomes an unlabeled frame. Add these to event_vocab.py "
            "(or run with --auto-vocab to bootstrap a mapping)."
        )

    event_counts: collections.Counter[int] = collections.Counter()
    sequences: collections.Counter[tuple[int, ...]] = collections.Counter()
    total_frames = labeled_frames = 0
    skipped = 0

    for n, path in enumerate(files):
        # A dry run only needs task_index; reading the full table (state, action, ...)
        # for 648 episodes just to count events is wasted I/O.
        columns = pq.ParquetFile(path).schema_arrow.names
        table = pq.read_table(path) if args.write_parquet else pq.read_table(path, columns=["task_index"])
        if LABELS_COLUMN in columns and not args.overwrite_columns:
            skipped += 1
            if skipped == 1:
                print(f"{path.name} already has '{LABELS_COLUMN}'; pass --overwrite-columns to relabel. Skipping.")
            continue

        task_index = np.asarray(table.column("task_index").to_numpy(), dtype=np.int64)
        labels, history, fired = label_episode(
            task_index,
            tasks,
            vocab,
            pre=args.pre,
            post=args.post,
            label_final_event=not args.no_final_event,
        )

        total_frames += len(labels)
        labeled_frames += int((labels != _event_vocab.NULL_LABEL).sum())
        sequences[tuple(e for e, _ in fired)] += 1
        for event_id, _ in fired:
            event_counts[event_id] += 1

        if n < args.show:
            print(f"--- {path.name} ({len(labels)} frames)")
            for event_id, frame in fired:
                print(f"    frame {frame:>5}  event {event_id}  {vocab.phrases[event_id]}")
            print(f"    history at frame 0        : {history[0]}")
            print(f"    history at the last frame : {history[-1]}\n")

        if args.write_parquet:
            for column in (LABELS_COLUMN, HISTORY_COLUMN):
                if column in table.column_names:
                    table = table.drop([column])
            table = table.append_column(LABELS_COLUMN, pa.array(labels, type=pa.int32()))
            table = table.append_column(HISTORY_COLUMN, pa.array(history, type=pa.string()))
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            tmp.replace(path)

    processed = len(files) - skipped
    print(f"episodes processed : {processed} ({skipped} skipped)")
    if processed:
        print(
            f"frames             : {total_frames} ({labeled_frames} labeled, {100 * labeled_frames / max(total_frames, 1):.1f}%)"
        )
        print("events fired       :")
        for event_id in sorted(event_counts):
            print(f"    {event_id}  {vocab.phrases[event_id]:<40} {event_counts[event_id]:>6}")
        print("event sequences    :")
        for seq, count in sequences.most_common(10):
            print(f"    {count:>5}  {seq}")
    if not args.write_parquet:
        print("\n(dry run — pass --write-parquet to persist the columns)")


if __name__ == "__main__":
    main()
