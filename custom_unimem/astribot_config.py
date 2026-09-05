"""UniMem train configs for the Astribot bimanual robot.

Dataset resolution
------------------
openpi loads a LeRobot dataset by ``repo_id``; LeRobot resolves that to
``$HF_LEROBOT_HOME/<repo_id>``. The wrapper scripts export ``HF_LEROBOT_HOME`` from
``ASTRIBOT_LEROBOT_HOME`` below, so you do not need to set it yourself.

The dataset location, repo id and prompt live in ``robot_paths.py`` (which imports
nothing) so the labeling script and the ROS2 deploy node can share them without pulling
in openpi/JAX.
"""

from custom_unimem import event_vocab as _event_vocab
from custom_unimem import robot_configs
from custom_unimem import robot_paths
from openpi.training import config as _config

PATHS = robot_paths.ASTRIBOT

ROBOT = PATHS.robot
ASTRIBOT_LEROBOT_HOME = PATHS.lerobot_home
ASTRIBOT_REPO_ID = PATHS.repo_id
ASTRIBOT_ASSET_ID = PATHS.asset_id
ASTRIBOT_PROMPT = PATHS.prompt

# Six events, one per annotated subtask — see event_vocab.py for the phrases and the
# prefix matching that separates the two pairs of similarly-worded subtasks.
VOCAB = _event_vocab.get_vocab(PATHS.vocab)

CONFIGS: list[_config.TrainConfig] = robot_configs.build_configs(
    robot=ROBOT,
    repo_id=ASTRIBOT_REPO_ID,
    asset_id=ASTRIBOT_ASSET_ID,
    prompt=ASTRIBOT_PROMPT,
    vocab=VOCAB,
    # Warm-starting from the earlier non-memory Astribot full fine-tune is possible (the
    # temporal path is zero-parameter, so the checkpoint loads fully), but only if that
    # checkpoint was trained under the corrected (14, -2) delta mask. Leave on pi05_base
    # unless you have verified that.
    warm_start_params=None,
)


def register() -> None:
    robot_configs.register(CONFIGS)
