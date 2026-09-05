"""Serve a trained UniMem policy over websockets.

    uv run python custom_unimem/serve.py --port 8000 policy:checkpoint \
        --policy.config=pi05_motion2_unimem_keyframe_full \
        --policy.dir=checkpoints/pi05_motion2_unimem_keyframe_full/coffee_keyframe/100000

``policy_config.create_trained_policy`` cross-checks the config against the checkpoint's
recorded training shape (video_encoder / num_frames / event_tracking) and refuses to
serve a mismatch, so passing the wrong ``--policy.config`` fails loudly rather than
silently feeding the model a history layout it was never fit to.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from custom_unimem import _bootstrap

_bootstrap.bootstrap()

import tyro  # noqa: E402

_serve = _bootstrap.load_repo_script("openpi_serve_policy_unimem", "scripts/serve_policy.py")

if __name__ == "__main__":
    _serve.main(tyro.cli(_serve.Args))
