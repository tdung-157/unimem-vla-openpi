"""Train a UniMem policy on one of the bimanual robots.

    uv run python custom_unimem/train.py pi05_motion2_unimem_keyframe_full \
        --exp-name=coffee_keyframe --overwrite

Registers the custom configs, then delegates to the stock ``scripts/train.py`` — every
flag that script accepts works here unchanged (``--resume``, ``--batch-size``,
``--fsdp-devices``, ``--num-workers``, ``--no-wandb-enabled``, ...).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import _bootstrap

_bootstrap.bootstrap()

import openpi.training.config as _config  # noqa: E402

_train = _bootstrap.load_repo_script("openpi_train_unimem", "scripts/train.py")

if __name__ == "__main__":
    _train.main(_config.cli())
