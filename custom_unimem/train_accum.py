"""Train a UniMem policy with gradient accumulation.

Same configs and flags as ``custom_unimem/train.py``, but each optimizer step is built
from ``GRAD_ACCUM_STEPS`` micro-batches, so the *effective* batch is
``batch_size * GRAD_ACCUM_STEPS``. Use it when the paper's batch does not fit in memory:

    GRAD_ACCUM_STEPS=2 uv run python custom_unimem/train_accum.py \\
        pi05_astribot_unimem_keyframe --exp-name=coffee_keyframe --batch-size 22

reproduces upstream's effective batch of 44 on a GPU that can only hold 22 at once.

Costs the same total compute as the un-accumulated run and gives the same gradient, but
takes ``GRAD_ACCUM_STEPS`` forward/backward passes per optimizer step, so wall-clock per
logged step is proportionally longer. ``num_train_steps`` counts OPTIMIZER steps, so it
does not need adjusting.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import _bootstrap  # noqa: E402

_bootstrap.bootstrap()

import openpi.training.config as _config  # noqa: E402

_train = _bootstrap.load_repo_script("openpi_train_accum_unimem", "scripts/train_accum_steps.py")

if __name__ == "__main__":
    _train.main(_config.cli())
