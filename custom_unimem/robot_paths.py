"""Dataset locations, repo ids and prompts for each robot — dependency-free.

Kept apart from ``astribot_config.py`` / ``motion2_config.py`` on purpose: the labeling
script, the preflight check and the ROS2 deploy nodes need these constants but must NOT
import openpi. The deploy nodes in particular run under the system ROS2 Python (rclpy,
cv_bridge), not the uv environment that has JAX.

Each robot names exactly one dataset. To point a run somewhere else without editing this
file — a second copy of the same data on a laptop, say — export
``<ROBOT>_LEROBOT_HOME`` and/or ``<ROBOT>_REPO_ID`` (e.g. ``ASTRIBOT_REPO_ID``).

``asset_id`` is deliberately NOT derived from the repo id: it names the norm-stats
directory, and the stats depend only on state/actions, so a stable name lets stats
computed from one copy of a dataset be reused against another. The norm-stats scripts
record what they were computed from in ``norm_stats_source.json`` next to the stats, and
``preflight.py`` refuses a mismatch — see NORM_STATS_SOURCE_FILE.
"""

import dataclasses
import os

# Written next to norm_stats.json by both norm-stats scripts; checked by preflight.py.
NORM_STATS_SOURCE_FILE = "norm_stats_source.json"


@dataclasses.dataclass(frozen=True)
class RobotPaths:
    # Short name used in config names, --robot flags and the shared assets directory.
    robot: str
    # Parent directory exported as HF_LEROBOT_HOME. LeRobot resolves a dataset to
    # <lerobot_home>/<repo_id>.
    dataset_home: str
    # Must be the ANNOTATED dataset — the one whose task_index varies within an episode —
    # since that is what the event labels are derived from.
    dataset_repo_id: str
    # Stable directory name for the shared norm stats. Independent of repo_id (see above).
    asset_id: str
    # The single fixed instruction every frame is conditioned on, at training and at
    # deployment. These must stay equal or the model sees an unfamiliar instruction.
    prompt: str
    # Name of the event vocabulary in event_vocab.py.
    vocab: str

    @property
    def lerobot_home(self) -> str:
        return os.environ.get(f"{self.robot.upper()}_LEROBOT_HOME", self.dataset_home)

    @property
    def repo_id(self) -> str:
        return os.environ.get(f"{self.robot.upper()}_REPO_ID", self.dataset_repo_id)


ASTRIBOT = RobotPaths(
    robot="astribot",
    # Training node (2x H100). Full path: /mnt/data/dungnt232_1/data/astri_coffee_making_v21_relabel
    #
    # A second copy of this data sits on the laptop at
    # /home/dungnt232/Documents/Work/data/astri_186_v21_relabel — point at it with
    #   ASTRIBOT_LEROBOT_HOME=/home/dungnt232/Documents/Work/data \
    #   ASTRIBOT_REPO_ID=astri_186_v21_relabel ...
    # rather than editing this file, so the committed config always names the machine
    # that actually trains.
    dataset_home="/mnt/data/dungnt232_1/data",
    dataset_repo_id="astri_coffee_making_v21_relabel",
    # 186 episodes / 264,874 frames at 30 fps, three 480x640 cameras. Every frame carries
    # its subtask in task_index (185 of 186 episodes run the full 0..5 sequence; one stops
    # after subtask 4), which is what the event labels are derived from.
    #
    # NOTE the laptop copy's meta/info.json names the CORRECTED state layout (grippers
    # trailing at dims 14/15) while meta/info.json.bak preserves the old, wrong
    # interleaved naming with grippers at 7/15. Do not restore the .bak — the
    # delta-action mask in data_configs.py and the joint order in the deploy node both
    # assume the corrected layout, and preflight.py verifies it from the data itself.
    asset_id="astribot_coffee_v21",
    prompt="Make an iced coffee",
    vocab="astribot",
)

MOTION2 = RobotPaths(
    robot="motion2",
    dataset_home="/home/dungnt232/Documents/Work/vla_training",
    # 648 episodes / 817,315 frames at 30 fps, every frame tagged with its subtask. The
    # non-annotated twin (lrb_new_format/coffee_aligned_0906_v21) carries a single
    # episode-level task and so cannot produce event labels; it is not used here.
    dataset_repo_id="lrb_new_format/coffee_aligned_anno_data_v21",
    asset_id="motion2_coffee_anno_v21",
    prompt="get coffee, get ice, serve the drink",
    vocab="motion2",
)

ALL: dict[str, RobotPaths] = {ASTRIBOT.robot: ASTRIBOT, MOTION2.robot: MOTION2}


def get(robot: str) -> RobotPaths:
    try:
        return ALL[robot]
    except KeyError:
        raise SystemExit(f"unknown robot '{robot}'; expected one of {sorted(ALL)}") from None
