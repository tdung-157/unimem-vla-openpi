"""UniMem train configs for the MOTION2 bimanual robot.

Dataset resolution
------------------
openpi loads a LeRobot dataset by ``repo_id``; LeRobot resolves that to
``$HF_LEROBOT_HOME/<repo_id>``. The wrapper scripts export ``HF_LEROBOT_HOME`` from
``MOTION2_LEROBOT_HOME`` below, so you do not need to set it yourself.

The dataset location, repo id and prompt live in ``robot_paths.py`` (which imports
nothing) so the labeling script and the ROS2 deploy node can share them without pulling
in openpi/JAX.
"""

from custom_unimem import event_vocab as _event_vocab
from custom_unimem import robot_configs
from custom_unimem import robot_paths
from openpi.training import config as _config

PATHS = robot_paths.MOTION2

ROBOT = PATHS.robot
MOTION2_LEROBOT_HOME = PATHS.lerobot_home
MOTION2_REPO_ID = PATHS.repo_id
MOTION2_ASSET_ID = PATHS.asset_id
MOTION2_PROMPT = PATHS.prompt

VOCAB = _event_vocab.get_vocab(PATHS.vocab)

CONFIGS: list[_config.TrainConfig] = robot_configs.build_configs(
    robot=ROBOT,
    repo_id=MOTION2_REPO_ID,
    asset_id=MOTION2_ASSET_ID,
    prompt=MOTION2_PROMPT,
    vocab=VOCAB,
)


def register() -> None:
    robot_configs.register(CONFIGS)
