"""Data pipeline configs for the 16-dim bimanual robots (Astribot, MOTION2).

Three factories, one per frame-selection strategy. They are **not** interchangeable —
each one produces a different image tensor shape and a different set of dataset columns,
and a model trained under one cannot be served under another:

    BimanualEventDataConfig          event_tracking=True,  video_encoder=False
        The current frame only. The policy learns to *detect* events and reads its past
        through the ``phase_history`` text alone. Start here.

    BimanualEventKeyframeDataConfig  event_tracking=True,  video_encoder=True
        The current frame plus the actual past event-transition frames, assembled by
        ``EventMemoryDataset``. Text memory AND visual memory. This is full UniMem, and
        the only combination where video and event tracking meet.

    BimanualVideoDataConfig          event_tracking=False, video_encoder=True
        The current frame plus a FIXED time stride of past frames. No events, no labels,
        no phase_history — a plain "does temporal context help at all" baseline. This is
        the direct translation of the older ``use_mem_short_video`` configs into this
        fork's vocabulary.

See ``Pi0Config.video_encoder``'s docstring in src/openpi/models/pi0_config.py for why
the first and third never mix.
"""

import dataclasses

from custom_unimem import bimanual_policy
from openpi.training import config as _config

# Raw LeRobot feature keys, as they appear in the parquet / info.json. These are what the
# data loader passes to ``delta_timestamps`` and what ``EventMemoryDataset`` indexes with,
# so they are the pre-repack names — not the ``observation/...`` names the policy uses.
CAM_HEAD = "observation.images.cam_head"
CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"
CAM_KEYS = (CAM_HEAD, CAM_LEFT_WRIST, CAM_RIGHT_WRIST)

# Some v2.1-converted datasets have video PTS shifted by up to one 30 fps frame period
# against the parquet timestamps, which trips LeRobot's default 1e-4 s decode tolerance.
# One frame period covers that drift and is still too small to select a wrong frame.
VIDEO_TOLERANCE_S = 0.04

# [7 left joints, 7 right joints, left_gripper, right_gripper] -> delta for the 14 joints,
# absolute for the 2 grippers. See bimanual_policy.py for why the grippers are at 14/15.
DELTA_ACTION_MASK = _config._transforms.make_bool_mask(14, -2)  # noqa: SLF001


def _repack(*, with_prompt: bool, with_events: bool) -> _config._transforms.Group:
    """Raw LeRobot columns -> the ``observation/...`` keys ``BimanualInputs`` expects.

    ``RepackTransform`` does a hard dict lookup, so only ask for keys something upstream
    actually produces: "prompt" exists only when ``prompt_from_task=True``, and
    "phase_history"/"labels" only on a dataset that has been through
    ``label_dataset_subtasks.py``.
    """
    structure = {
        "observation/image": CAM_HEAD,
        "observation/left_wrist_image": CAM_LEFT_WRIST,
        "observation/right_wrist_image": CAM_RIGHT_WRIST,
        "observation/state": "observation.state",
        "actions": "action",
    }
    if with_prompt:
        structure["prompt"] = "prompt"
    if with_events:
        structure["phase_history"] = "phase_history"
        structure["labels"] = "labels"
    return _config._transforms.Group(inputs=[_config._transforms.RepackTransform(structure)])  # noqa: SLF001


def _data_transforms(model_config, *, extra_inputs=()) -> _config._transforms.Group:
    """BimanualInputs/Outputs plus the delta-joint / absolute-gripper conversion.

    The two halves of the delta conversion must always move together: training on raw
    absolute targets while inference adds the current state back on top drives every arm
    joint to roughly twice its intended angle.
    """
    group = _config._transforms.Group(  # noqa: SLF001
        inputs=[bimanual_policy.BimanualInputs(model_type=model_config.model_type), *extra_inputs],
        outputs=[bimanual_policy.BimanualOutputs()],
    )
    return group.push(
        inputs=[_config._transforms.DeltaActions(DELTA_ACTION_MASK)],  # noqa: SLF001
        outputs=[_config._transforms.AbsoluteActions(DELTA_ACTION_MASK)],  # noqa: SLF001
    )


@dataclasses.dataclass(frozen=True)
class BimanualEventDataConfig(_config.DataConfigFactory):
    """Event tracking on the current frame only (``video_encoder=False``).

    Requires the dataset to carry ``labels`` (-1 = unlabeled, 0..N-1 = event ids) and
    ``phase_history`` (the running text summary) — run ``label_dataset_subtasks.py``
    first. NOT interchangeable with the keyframe config below: that one always emits a
    (T, C, H, W) stack, which is the wrong shape for a ``video_encoder=False`` model.
    """

    # Fixed prompt for every frame, instead of the dataset's task column. UniMem wants a
    # single stable instruction with the *memory* carrying the progress information, so
    # this is set (and ``base_config.prompt_from_task`` left False) on all our configs.
    # ``InjectDefaultPrompt`` only fills in a prompt that is absent, so a task-derived
    # prompt would silently win over this one.
    default_prompt: str | None = None

    # Probability of replacing ``phase_history`` with "History: none" for a sample. Keeps
    # the policy from becoming reliant on the text summary always being informative —
    # without it, ablations that drop the text at inference fail even where vision alone
    # should suffice. 0.0 = always use the real text.
    text_dropout_prob: float = 0.0

    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_repack(with_prompt=self.default_prompt is None, with_events=True),
            data_transforms=_data_transforms(model_config),
            model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=("action",),
            video_tolerance_s=VIDEO_TOLERANCE_S,
            text_dropout_prob=self.text_dropout_prob,
        )


@dataclasses.dataclass(frozen=True)
class BimanualEventKeyframeDataConfig(_config.DataConfigFactory):
    """Event tracking + event-triggered keyframes (``video_encoder=True``).

    "Keyframe" is about WHICH frames reach the video encoder: each sample's ``T-1``
    history frames are the actual past event-transition frames of that episode (via
    ``EventMemoryDataset``), not a fixed time window. That mirrors exactly what
    ``Policy``'s SigLIP hidden-state cache serves at inference, which is why a model
    trained this way must be deployed in keyframe mode (single frame per call +
    ``reset_cache`` per rollout + ``new_keyframe`` on event) and never by client-side
    frame stacking.
    """

    default_prompt: str | None = None

    # Probability of zeroing ALL event frames for a sample, forcing the policy to act from
    # the current frame + text alone.
    event_dropout_prob: float = 0.0
    # Probability of replacing ``phase_history`` with "History: none", drawn independently
    # of the frame dropout above so the policy sees each memory modality without the other.
    text_dropout_prob: float = 0.0
    # Frames at the start of each event window to sample the keyframe from. 1 = always the
    # first labeled frame; larger matches the spread in when the event head actually fires
    # at rollout. 30 = one second at 30 fps.
    event_frame_window: int = 1
    # Optional decision-window upsampling: boost the sampling weight of the frames just
    # after ``upsample_after_event_id``'s last labeled frame — the short stretch where the
    # policy has to commit to a memory-dependent choice and which is otherwise an
    # undertrained sliver of the episode. None = disabled.
    upsample_after_event_id: int | None = None
    upsample_window_steps: int = 0
    upsample_weight: float = 1.0
    upsample_window_offset: int = 0
    # Add Gaussian noise to ``state`` inside (a prefix of) that same window, so the policy
    # cannot use proprioceptive drift as a shortcut for a decision meant to come from memory.
    noise_window_steps: int = 0
    noise_state_in_upsample_window: bool = False

    def create(self, assets_dirs, model_config):
        num_frames: int = model_config.num_frames
        if not getattr(model_config, "video_encoder", False):
            raise ValueError(
                "BimanualEventKeyframeDataConfig requires model.video_encoder=True; "
                "use BimanualEventDataConfig for single-frame event tracking."
            )
        if num_frames < 2:
            raise ValueError(f"Keyframe training needs num_frames >= 2, got {num_frames}")

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_repack(with_prompt=self.default_prompt is None, with_events=True),
            data_transforms=_data_transforms(model_config),
            model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=("action",),
            video_tolerance_s=VIDEO_TOLERANCE_S,
            video_image_keys=CAM_KEYS,
            event_memory_training=True,
            # T slots = (T-1) past event frames + the current one.
            event_memory_size=num_frames - 1,
            event_dropout_prob=self.event_dropout_prob,
            text_dropout_prob=self.text_dropout_prob,
            event_frame_window=self.event_frame_window,
            upsample_after_event_id=self.upsample_after_event_id,
            upsample_window_steps=self.upsample_window_steps,
            upsample_weight=self.upsample_weight,
            upsample_window_offset=self.upsample_window_offset,
            noise_window_steps=self.noise_window_steps,
            noise_state_in_upsample_window=self.noise_state_in_upsample_window,
        )


@dataclasses.dataclass(frozen=True)
class BimanualVideoDataConfig(_config.DataConfigFactory):
    """Fixed-stride video, NO event tracking (the "naive video" baseline).

    Never loads ``labels``/``phase_history``. Served by having the CLIENT stack the last
    ``num_frames`` frames at the same spacing and send the whole stack every call — the
    policy server's hidden-state cache is not involved at all.
    """

    default_prompt: str | None = None
    # Seconds between consecutive history frames, ending at the current one. With
    # num_frames=6 and 1.0 s this is 5 s of lookback (the span is (T-1) x stride, since
    # num_frames counts the current frame).
    frame_stride_sec: float = 1.0
    # See MemoryDropout's docstring — counteracts the motion-energy shortcut that stalls
    # the robot at episode start. Lower it before changing anything else if a
    # history-dependent behaviour starts failing; 0.0 disables the transform.
    mem_dropout_prob: float = 0.5

    def create(self, assets_dirs, model_config):
        if not getattr(model_config, "video_encoder", False):
            raise ValueError("BimanualVideoDataConfig requires model.video_encoder=True")
        num_frames: int = model_config.num_frames
        offsets = tuple(-(num_frames - 1 - i) * self.frame_stride_sec for i in range(num_frames))

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_repack(with_prompt=self.default_prompt is None, with_events=False),
            data_transforms=_data_transforms(
                model_config,
                # Training-only (gated on "actions"), so serving is unaffected.
                extra_inputs=[bimanual_policy.MemoryDropout(prob=self.mem_dropout_prob)],
            ),
            model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=("action",),
            video_tolerance_s=VIDEO_TOLERANCE_S,
            video_image_keys=CAM_KEYS,
            video_frame_offsets=offsets,
        )
