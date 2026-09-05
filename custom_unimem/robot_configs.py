"""Shared TrainConfig builder for the bimanual UniMem runs.

Astribot and MOTION2 present the identical interface to the policy (16-dim bimanual
state/action, three 480x640 cameras, 30 fps), so both robots' config sets are generated
from one builder and differ only in the constants at the top of ``astribot_config.py`` /
``motion2_config.py``: dataset location, repo id, prompt, and the event vocabulary.

Five configs per robot. The two `_full` rows are the ones to train — full fine-tuning of
the whole backbone, `ema_decay=0.999`, no freeze filter — matching how the earlier
non-memory bimanual policies on these robots were trained.

A note on where that differs from upstream: every UniMem config in this fork (all 34 of
them — the entire libero_mem*/xarm_mem* sweep plus the four unimem_example_* templates)
is a LoRA fine-tune, so the paper's own checkpoints are LoRA. Full fine-tuning appears
there only as a comment ("drop _lora from both variants, >70GB"). On a 2x H100 node the
memory that comment warns about is available, and full FT is what these robots' earlier
policies used, so that is the default here. The `_lora` rows below are kept for a single
32 GB card (the laptop's 5090), where full FT does not fit.

In the order you should actually run them:

  ``pi05_<robot>_unimem_event_full``     event tracking only, single frame, full FT.
                                        Run this first with ``--num-train-steps=5000`` as
                                        a cheap gate: if ``event_loss`` does not fall,
                                        the labels or the event window are wrong and no
                                        amount of keyframe training will fix it.
  ``pi05_<robot>_unimem_keyframe_full``  full UniMem (text + visual event memory), full
                                        FT. The headline config.
  ``pi05_<robot>_unimem_video_full``     fixed-stride video baseline, NO events. The
                                        direct translation of the older
                                        ``use_mem_short_video`` runs into this fork.
  ``pi05_<robot>_unimem_event_lora``     } LoRA variants of the first two, for a single
  ``pi05_<robot>_unimem_keyframe_lora``  } 32 GB card. Not the intended training path.

Norm stats
----------
Every config for a robot reads its norm stats from one shared directory
(``./assets/<robot>_unimem/<asset_id>``) via ``AssetsConfig.assets_dir``, because the
stats only depend on state/actions and are therefore identical across all five. Compute
them once — ``custom_unimem/compute_norm_stats.py`` copies its output into that shared
directory no matter which config name you ran it under.
"""

import flax.nnx as nnx

from custom_unimem import data_configs
from custom_unimem import event_vocab as _event_vocab
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

# pi0/pi0.5 default action horizon; 50 steps at 30 fps = 1.67 s of open-loop execution,
# which suits these real-robot rollouts.
ACTION_HORIZON = 50

# Event-keyframe depth: the current frame plus 3 past event frames. The text history
# still carries every earlier event, so this bounds ViT cost without losing the summary.
KEYFRAME_NUM_FRAMES = 4

# Fixed-stride baseline: 6 frames at 1 Hz over a 30 fps stream = 5.0 s of lookback
# (the span is (T-1) x stride, since num_frames counts the current frame). Chosen to match
# the temporal layout of the earlier short-memory runs so the two are comparable.
VIDEO_NUM_FRAMES = 6
VIDEO_FRAME_STRIDE_SEC = 1.0

# Prompt budget. pi0.5 defaults to 200 tokens, which has to hold
# "Task: <prompt>, History: <every completed event>, State: <32 discretized numbers>".
# On the MOTION2 coffee task a full history plus the state measures ~170 tokens, leaving
# very little slack — and an overflow is a silent truncation warning, not an error, that
# would chop the state off the end of the prompt. 256 buys room for a longer vocabulary
# or wordier phrases. Raising it costs prefix attention; lower it back to 200 if you
# shorten the event phrases.
MAX_TOKEN_LEN = 256

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"

# Shared per-robot assets root (see the module docstring).
ASSETS_ROOT = "./assets"


def shared_assets_dir(robot: str) -> str:
    return f"{ASSETS_ROOT}/{robot}_unimem"


def _model(*, lora: bool, event_tracking: bool, video_encoder: bool, num_frames: int) -> pi0_config.Pi0Config:
    """One place that builds every model config, so a freeze filter can never drift from
    the model it is supposed to freeze (they must be built from identical settings —
    ``get_freeze_filter`` matches on parameter-path regexes)."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=ACTION_HORIZON,
        # MUST stay True (the pi0.5 default). The event history only reaches the model
        # through the discrete-state prompt format: PaligemmaTokenizer.tokenize() renders
        # "Task: <prompt>, <phase_history>, State: <256-bin state>;\nAction: " only on the
        # branch where a state is passed, and ModelTransformFactory passes one only when
        # discrete_state_input is True. Set it False (as the earlier non-UniMem configs on
        # these robots did) and phase_history is silently dropped — the run trains happily
        # and learns no text memory at all.
        discrete_state_input=True,
        max_token_len=MAX_TOKEN_LEN,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        event_tracking=event_tracking,
        video_encoder=video_encoder,
        num_frames=num_frames,
    )


def _schedule(num_train_steps: int) -> _optimizer.CosineDecaySchedule:
    """Constant 5e-5 after a 1k-step warmup.

    ``peak_lr == decay_lr`` makes the cosine flat; ``decay_steps`` is kept equal to
    ``num_train_steps`` so the schedule stays correct if the two are ever made to differ.
    This matches what has actually worked on these datasets. The upstream UniMem examples
    instead decay to ~2.5e-6 over 20k steps, which is the thing to try if a LoRA run
    plateaus early.
    """
    return _optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=num_train_steps,
        decay_lr=5e-5,
    )


def build_configs(
    *,
    robot: str,
    repo_id: str,
    asset_id: str,
    prompt: str,
    vocab: _event_vocab.EventVocab,
    warm_start_params: str | None = None,
) -> list[_config.TrainConfig]:
    """Build the five TrainConfigs for one robot.

    Args:
        robot: short name used in every config name (``astribot`` / ``motion2``).
        repo_id: LeRobot repo id, resolved under ``$HF_LEROBOT_HOME``. Must be the
            ANNOTATED dataset that ``label_dataset_subtasks.py`` has written ``labels``
            and ``phase_history`` into — the video baseline shares it but ignores both.
        asset_id: flat name for the shared norm-stats directory.
        prompt: the single fixed instruction every frame is conditioned on. UniMem
            deliberately does NOT feed the per-frame subtask as the prompt: the whole
            point is that progress reaches the policy through event memory, not through a
            privileged instruction that already says which step it is on. This string must
            match what the deploy node sends, or the model sees an instruction it was
            never trained with.
        vocab: event vocabulary, only used here to sanity-check the id range.
        warm_start_params: optional checkpoint ``params`` dir to start from instead of
            pi05_base. Warm-starting from a non-memory fine-tune of the same robot is
            safe for the video/keyframe configs (the temporal path adds no parameters —
            temporal attention reuses each layer's own weights) but is only valid if that
            checkpoint was trained with the SAME delta-action convention.
    """
    if vocab.num_event_classes - 1 > _event_vocab.MAX_EVENT_ID:
        raise ValueError(f"{robot}: vocabulary exceeds the model's {_event_vocab.MAX_EVENT_ID} event ids")

    assets = _config.AssetsConfig(assets_dir=shared_assets_dir(robot), asset_id=asset_id)
    loader = weight_loaders.CheckpointWeightLoader(warm_start_params or PI05_BASE_PARAMS)

    def train_config(
        *,
        suffix: str,
        model: pi0_config.Pi0Config,
        data: _config.DataConfigFactory,
        lora: bool,
        batch_size: int,
        num_train_steps: int,
        fsdp_devices: int = 1,
        save_interval: int,
        keep_period: int,
    ) -> _config.TrainConfig:
        return _config.TrainConfig(
            name=f"pi05_{robot}_unimem_{suffix}",
            model=model,
            data=data,
            weight_loader=loader,
            # LoRA freezes everything except the adapters and (when present) the event
            # head; a full fine-tune freezes nothing.
            freeze_filter=model.get_freeze_filter() if lora else nnx.Nothing,
            # Multiplier on the LR schedule for ``Pi0.phase_head`` only. The head is a
            # small MLP trained from scratch on top of a pretrained backbone, so >1.0 can
            # help it catch up; 1.0 is the safe start and what every published UniMem
            # experiment uses.
            phase_head_lr_multiplier=1.0,
            lr_schedule=_schedule(num_train_steps),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            # EMA is disabled for LoRA (the adapters are small and converge fast enough
            # that EMA mostly just delays convergence) and on for full fine-tunes.
            ema_decay=None if lora else 0.999,
            batch_size=batch_size,
            fsdp_devices=fsdp_devices,
            num_train_steps=num_train_steps,
            log_interval=100,
            save_interval=save_interval,
            keep_period=keep_period,
            # No W&B account is wired up on the training cluster; metrics go to the
            # Slurm log instead (sbatch_train.sh sets PYTHONUNBUFFERED so they appear).
            wandb_enabled=False,
        )

    base_data = _config.DataConfig(prompt_from_task=False, stop_padding=False)

    def event_data(**kwargs) -> data_configs.BimanualEventDataConfig:
        return data_configs.BimanualEventDataConfig(
            repo_id=repo_id, assets=assets, base_config=base_data, default_prompt=prompt, **kwargs
        )

    def keyframe_data(**kwargs) -> data_configs.BimanualEventKeyframeDataConfig:
        return data_configs.BimanualEventKeyframeDataConfig(
            repo_id=repo_id, assets=assets, base_config=base_data, default_prompt=prompt, **kwargs
        )

    # ---------------------------------------------------------------- event tracking only
    event_lora_model = _model(lora=True, event_tracking=True, video_encoder=False, num_frames=1)
    event_full_model = _model(lora=False, event_tracking=True, video_encoder=False, num_frames=1)

    # ---------------------------------------------------------------------- full UniMem
    kf_lora_model = _model(lora=True, event_tracking=True, video_encoder=True, num_frames=KEYFRAME_NUM_FRAMES)
    kf_full_model = _model(lora=False, event_tracking=True, video_encoder=True, num_frames=KEYFRAME_NUM_FRAMES)

    # --------------------------------------------------------- fixed-stride video baseline
    video_full_model = _model(lora=False, event_tracking=False, video_encoder=True, num_frames=VIDEO_NUM_FRAMES)

    keyframe_knobs = dict(
        # Zero out all event frames for 20% of samples, so the policy stays functional on
        # the current frame + text alone rather than depending on visual memory
        # unconditionally.
        event_dropout_prob=0.2,
        # Drawn independently of the above: the policy must also cope with keyframes but
        # no text, or the single-modality ablations fail even where one modality suffices.
        text_dropout_prob=0.2,
        # One second at 30 fps. The event head does not fire at exactly frame 0 of an
        # event at rollout, so sampling the keyframe from a window matches deployment.
        event_frame_window=30,
        # Decision-window upsampling is off: this task is a linear sequence with no
        # memory-dependent fork. Turn it on (upsample_after_event_id=<id>,
        # upsample_window_steps=<frames>, upsample_weight=<x>) for a task where a short
        # stretch right after some event decides the rest of the episode.
        upsample_after_event_id=None,
    )

    return [
        train_config(
            suffix="event_full",
            model=event_full_model,
            data=event_data(text_dropout_prob=0.2),
            lora=False,
            batch_size=32,
            fsdp_devices=2,
            num_train_steps=300_000,
            save_interval=10_000,
            keep_period=100_000,
        ),
        train_config(
            suffix="keyframe_full",
            model=kf_full_model,
            data=keyframe_data(**keyframe_knobs),
            lora=False,
            # 16 over 2 GPUs = 8 per device, same per-device image count as the LoRA row.
            # batch_size must stay divisible by the device count (train.py asserts it) and
            # fsdp_devices must divide it evenly (sharding.make_mesh asserts it).
            batch_size=16,
            fsdp_devices=2,
            num_train_steps=300_000,
            save_interval=10_000,
            keep_period=100_000,
        ),
        train_config(
            suffix="video_full",
            model=video_full_model,
            data=data_configs.BimanualVideoDataConfig(
                repo_id=repo_id,
                assets=assets,
                base_config=base_data,
                default_prompt=prompt,
                frame_stride_sec=VIDEO_FRAME_STRIDE_SEC,
                mem_dropout_prob=0.5,
            ),
            lora=False,
            # 6 frames x 3 cameras = 18 images per sample — the heaviest config here.
            # 16 over 2 GPUs = 8 samples x 18 images per device, the same per-device load
            # the earlier short-memory full fine-tune ran at (24 over 3 GPUs).
            batch_size=16,
            fsdp_devices=2,
            num_train_steps=300_000,
            save_interval=10_000,
            keep_period=100_000,
        ),
        train_config(
            suffix="event_lora",
            model=event_lora_model,
            data=event_data(text_dropout_prob=0.2),
            lora=True,
            batch_size=32,
            num_train_steps=30_000,
            save_interval=5_000,
            keep_period=10_000,
        ),
        train_config(
            suffix="keyframe_lora",
            model=kf_lora_model,
            data=keyframe_data(**keyframe_knobs),
            lora=True,
            # 4 frames x 3 cameras = 12 images per sample instead of 3, so roughly a
            # quarter of the single-frame batch fits the same GPU. Tune down on OOM.
            batch_size=8,
            num_train_steps=30_000,
            save_interval=5_000,
            keep_period=10_000,
        ),
    ]


def register(configs: list[_config.TrainConfig]) -> None:
    """Register configs into openpi's global registry (idempotent)."""
    for cfg in configs:
        if cfg.name not in _config._CONFIGS_DICT:  # noqa: SLF001
            _config._CONFIGS.append(cfg)  # noqa: SLF001
            _config._CONFIGS_DICT[cfg.name] = cfg  # noqa: SLF001
