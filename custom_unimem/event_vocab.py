"""Event vocabularies for the bimanual robots.

An *event* (the paper's term; the dataset column and the model attribute still say
"phase" — see the fork README's terminology note) is a semantic transition point in a
task: "pushed the coffee lever", "placed the cup down". Two things are derived from the
vocabulary and must agree exactly between training and deployment:

  * ``labels`` — the integer event id written into the LeRobot parquet by
    ``label_dataset_subtasks.py`` and consumed by ``Pi0.compute_loss_event``.
  * ``phase_history`` — the running text summary of *completed* events, rendered as
    ``"History: pushed the coffee lever, moved the cup under the milk faucet"`` (and
    ``"History: none"`` before the first one). Built by the labeling script for training
    and rebuilt live from the event head's predictions by the deploy nodes.

Both of these datasets annotate every frame with the *subtask* being performed
(``task_index``), so events come for free: an event fires where the subtask changes,
and its phrase describes the subtask that just **completed**. That is the whole reason
the annotated (``*_anno_*``) dataset is the one to train UniMem configs on.

Matching is by normalized *prefix* so that the several typo variants of the same subtask
string ("... under the milk faucet", "... under the milk fauce.", and one that repeats
itself twice) all collapse onto a single event id — see ``normalize_task``.
"""

from __future__ import annotations

import dataclasses
import re

# Hard ceiling from the model: ``Pi0`` allocates ``EVENT_LOGITS_NUM_CLASSES`` logits and
# reserves the last one for the "-1 / unlabeled" bucket, so valid ids are 0..10.
# (see src/openpi/models/pi0.py)
MAX_EVENT_ID = 10

NULL_LABEL = -1


def normalize_task(task: str) -> str:
    """Lowercase, strip punctuation/whitespace noise so typo variants compare equal."""
    t = task.strip().lower().replace("_", " ")
    t = re.sub(r"[.,;:]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclasses.dataclass(frozen=True)
class EventVocab:
    """Ordered event vocabulary for one task family.

    Attributes:
        name: identifier used on the CLI (``--robot``) and in log lines.
        phrases: event id -> completion phrase used in ``phase_history``.
        task_prefixes: normalized subtask prefix -> event id. A dataset task whose
            normalized string *starts with* the prefix maps to that event. Longest
            prefix wins, so a more specific variant can override a shorter one.
        ignore_task_prefixes: subtask prefixes that carry no event at all (e.g. the
            episode-level task string on unannotated episodes).
    """

    name: str
    phrases: dict[int, str]
    task_prefixes: dict[str, int]
    ignore_task_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bad = [i for i in self.phrases if not 0 <= i <= MAX_EVENT_ID]
        if bad:
            raise ValueError(f"{self.name}: event ids {bad} outside 0..{MAX_EVENT_ID} (model limit)")
        missing = sorted(set(self.task_prefixes.values()) - set(self.phrases))
        if missing:
            raise ValueError(f"{self.name}: task prefixes map to ids {missing} with no phrase")

    @property
    def num_event_classes(self) -> int:
        """One past the largest event id — the "no event" id used at inference."""
        return max(self.phrases) + 1

    def event_for_task(self, task: str) -> int | None:
        """Event id completed by ``task``, or None if the task carries no event."""
        norm = normalize_task(task)
        for prefix in self.ignore_task_prefixes:
            if norm.startswith(normalize_task(prefix)):
                return None
        best: tuple[int, int] | None = None  # (prefix length, event id)
        for prefix, event_id in self.task_prefixes.items():
            p = normalize_task(prefix)
            if norm.startswith(p) and (best is None or len(p) > best[0]):
                best = (len(p), event_id)
        return None if best is None else best[1]

    def format_history(self, completed: list[str]) -> str:
        """Render the running memory string exactly as training writes it."""
        return f"History: {', '.join(completed)}" if completed else "History: none"


# ---------------------------------------------------------------------------
# MOTION2 — "get coffee, get ice, serve the drink"
#
# Verified against lrb_new_format/coffee_aligned_anno_data_v21 (648 episodes): 471 run
# the subtask sequence (0, 1, 2, 3, 4, 5), 167 the same sequence with the task-7 spelling
# of subtask 2, 3 more with the task-8/9 spellings, 1 skips the milk lever, and 6 are
# unannotated (episode-level task only -> no events, "History: none" throughout).
# ---------------------------------------------------------------------------
MOTION2_COFFEE = EventVocab(
    name="motion2",
    phrases={
        0: "placed the cup under the coffee faucet",
        1: "pushed the coffee lever",
        2: "moved the cup under the milk faucet",
        3: "pressed the milk lever",
        4: "placed the cup down",
        5: "served the drink",
    },
    task_prefixes={
        "left arm pick the cup and place it under the coffee faucet": 0,
        "left arm push the coffee lever": 1,
        # Deliberately short: covers "... hold it under the milk faucet",
        # "... milk fauce.", and the variant that repeats the whole sentence.
        "left arm pick up the coffee faucet": 2,
        "right arm press the milk lever": 3,
        "left arm place the cup down": 4,
        "right arm bring the cup to the serving area": 5,
    },
    ignore_task_prefixes=("get coffee, get ice, serve the drink",),
)

# ---------------------------------------------------------------------------
# Astribot — "Make an iced coffee" (astri_186_v21_relabel)
#
# Verified over all 186 episodes: 185 run the subtask sequence (0, 1, 2, 3, 4, 5) and one
# stops after subtask 4. There is no episode-level task string in this dataset — every
# frame belongs to one of the six subtasks — so ignore_task_prefixes is empty.
#
# The prefixes are trimmed to the point where each subtask stops being ambiguous: subtasks
# 0 and 2 both start "use the left hand to grasp the cup", and 3 and 5 both name the
# emerald green dispenser, so the prefixes have to run far enough to separate them.
# Longest matching prefix wins.
# ---------------------------------------------------------------------------
ASTRIBOT_COFFEE = EventVocab(
    name="astribot",
    phrases={
        0: "placed the cup under the coffee maker",
        1: "pressed the coffee button",
        2: "moved the cup to the middle of the table",
        3: "placed the cup under the green dispenser",
        4: "pressed the dispenser lever",
        5: "placed the cup on the tray",
    },
    task_prefixes={
        "use the left hand to grasp the cup located beneath the blue cup holder stand": 0,
        "use the left hand to press the middle button": 1,
        "use the left hand to grasp the cup from beneath the coffee maker": 2,
        "use the right hand to place the cup beneath the emerald green": 3,
        "use the right hand to press the black lever": 4,
        "use the right hand to pick up the cup from beneath the emerald green": 5,
    },
)

VOCABS: dict[str, EventVocab] = {
    MOTION2_COFFEE.name: MOTION2_COFFEE,
    ASTRIBOT_COFFEE.name: ASTRIBOT_COFFEE,
}


def get_vocab(robot: str) -> EventVocab:
    try:
        return VOCABS[robot]
    except KeyError:
        raise SystemExit(f"unknown robot '{robot}'; expected one of {sorted(VOCABS)}") from None
