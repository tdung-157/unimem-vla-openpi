"""Observation / action transforms for the 16-dim bimanual robots (Astribot, MOTION2).

Both robots expose the same interface to the policy, which is why one pair of transforms
covers them:

    state / action : 16 dims = [7 left-arm joints, 7 right-arm joints,
                                left_gripper, right_gripper]
    cameras        : cam_head (base) + cam_left_wrist + cam_right_wrist, 480x640 RGB
    control rate   : 30 fps

The grippers **trail** the arms at dims 14/15 — they are not interleaved at 7/15, even
though some ``meta/info.json`` files used to name them that way. That was verified over
all 186 episodes of ``astri_coffee_making_v21`` (dims 14/15 are bimodal 0/100, |v| > pi;
dim 7 is a plain radian arm joint) and cross-checked against per-frame ``task_index``.
Getting this wrong silently doubles every commanded arm angle, so the delta-action mask
in ``data_configs.py`` and the joint order in the deploy nodes both depend on it.

On top of the plain (non-memory) transforms these add the three UniMem wire keys:

    labels        -> per-sample event id for the auxiliary classification loss (training)
    phase_history -> running text summary of completed events (training + serving)
    event_id      -> predicted event-class probabilities returned by the model (serving)

The key name ``phase_history`` is legacy and deliberately kept: it is the LeRobot column
name, the wire key the served model expects, and part of every trained checkpoint's
prompt format. See the fork README's "note on terminology".
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Number of real state / action dimensions for these robots.
BIMANUAL_DIM = 16

# Indices of the two gripper channels within the 16-dim vector (they trail the arms).
GRIPPER_INDICES = (14, 15)


def make_bimanual_example(*, num_frames: int = 1, prompt: str = "make a drink") -> dict:
    """Random observation, for inference smoke tests. ``num_frames > 1`` mimics video."""

    def img() -> np.ndarray:
        shape = (480, 640, 3) if num_frames == 1 else (num_frames, 480, 640, 3)
        return np.random.randint(256, size=shape, dtype=np.uint8)

    return {
        "observation/state": np.random.rand(BIMANUAL_DIM),
        "observation/image": img(),
        "observation/left_wrist_image": img(),
        "observation/right_wrist_image": img(),
        "prompt": prompt,
        "phase_history": "History: none",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize to uint8 HWC (single frame) or THWC (video stack).

    LeRobot hands back float32 CHW in [0, 1]; the robot runtime sends uint8 HWC already;
    the event-keyframe / fixed-stride pipelines stack those into a leading time axis.
    All four shapes land here.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 4:
        if image.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            image = einops.rearrange(image, "t c h w -> t h w c")
        return image
    if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _mask(value: bool, ref_image: np.ndarray) -> np.ndarray:
    """Image mask matching ``ref_image``'s time layout.

    For a video stack the mask must be per-frame so that after batching its shape
    ``[B, T]`` lines up with the images' ``[B, T, H, W, C]``.
    """
    if ref_image.ndim == 4:
        return np.full(ref_image.shape[0], value)
    return np.bool_(value)


@dataclasses.dataclass(frozen=True)
class BimanualInputs(transforms.DataTransformFn):
    """Dataset / runtime observation -> model input. Used for training AND inference."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # pi0 / pi0.5 take one third-person view plus two wrist views. All three exist on
        # these robots, so every mask is True (no padding images to mask out).
        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": _mask(True, base_image),
                "left_wrist_0_rgb": _mask(True, left_wrist_image),
                "right_wrist_0_rgb": _mask(True, right_wrist_image),
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "labels" in data:
            # One scalar target per sample; shape (1,) so stacking yields [B, 1], which is
            # what ``Pi0.compute_loss_event``'s jaxtyping annotation (``Labels = *b v``) wants.
            lab = np.asarray(data["labels"], dtype=np.int32).ravel()
            if lab.size != 1:
                raise ValueError(f"Expected exactly one event label per sample, got shape {lab.shape}")
            inputs["labels"] = lab.reshape(1)

        if "phase_history" in data:
            history = data["phase_history"]
            if isinstance(history, bytes):
                history = history.decode("utf-8")
            inputs["phase_history"] = history

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualOutputs(transforms.DataTransformFn):
    """Model output -> robot action space. Inference only."""

    def __call__(self, data: dict) -> dict:
        # The model pads actions out to its internal action_dim; slice back to the real 16.
        outputs = {"actions": np.asarray(data["actions"][..., :BIMANUAL_DIM])}
        if "event_id" in data:
            # Softmax over the event head's classes; the deploy node thresholds it.
            outputs["event_id"] = np.asarray(data["event_id"])
        return outputs


@dataclasses.dataclass(frozen=True)
class MemoryDropout(transforms.DataTransformFn):
    """Replace a clip's history with copies of its current frame, with probability ``prob``.

    Only for the FIXED-STRIDE video config. The event-keyframe pipeline has its own
    equivalent knobs inside the data loader (``event_dropout_prob`` /
    ``text_dropout_prob``), so this transform is not used there.

    Why it exists: with a fixed-stride clip the policy learns to read action magnitude off
    recent inter-frame motion energy. In teleoperated data that predicts the action almost
    perfectly, but it is causally backwards, and at deploy it closes a stable loop —
    stationary history -> predict ~zero motion -> robot stays put -> history still
    stationary. Escaping needs noise to accumulate, which took minutes on the real robot.

    The data does contain the counter-example (at an episode boundary LeRobot clamps the
    negative delta indices, so the clip is ``num_frames`` copies of frame 0 paired with an
    action that genuinely moves) but at roughly one sample in 900 it is far too rare to
    outweigh a feature as dominant as motion energy. This makes that case ~``prob`` of the
    training distribution instead.

    Training only: gated on ``actions`` being present, so a served policy always sees its
    real history. The draw is shared across cameras — dropping one camera's history but
    not another's is a situation the robot never encounters. All-or-nothing by design:
    one Bernoulli draw replaces the WHOLE history, never individual frames.
    """

    prob: float = 0.5

    def __call__(self, data: dict) -> dict:
        if self.prob <= 0.0 or "actions" not in data or "image" not in data:
            return data

        # Fresh Generator per call: seeded from OS entropy, so torch DataLoader workers
        # cannot end up drawing the same sequence the way a shared global seed would.
        if np.random.default_rng().random() >= self.prob:
            return data

        images = {}
        for key, value in data["image"].items():
            value = np.asarray(value)
            # (T, H, W, C) is a clip; anything else is a single frame with no history.
            images[key] = np.repeat(value[-1:], value.shape[0], axis=0) if value.ndim == 4 else value
        return {**data, "image": images}
