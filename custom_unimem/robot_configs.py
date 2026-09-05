"""Shared TrainConfig builder for the bimanual UniMem runs.

Astribot and MOTION2 present the identical interface to the policy (16-dim bimanual
state/action, three 480x640 cameras, 30 fps), so both robots' config sets are generated
from one builder and differ only in the constants in ``robot_paths.py``: dataset
location, repo id, prompt, and the event vocabulary.

Five configs per robot. The first three reproduce the paper's own real-robot recipe as
closely as this robot allows; the last two are a deliberate departure.

  ``pi05_<robot>_unimem_keyframe``       PAPER. Mirrors ``xarm_mem7_coruscant``, the
                                        config behind the paper's hardware checkpoints:
                                        LoRA, num_frames=4, ema_decay=None, batch 44,
                                        20k steps, cosine 2.5e-5 -> 3.5e-6, both memory
                                        dropouts off, event_frame_window=30. The
                                        headline config.
  ``pi05_<robot>_unimem_event``         PAPER, single-frame ablation: event tracking with
                                        no video encoder (the paper's *_text_only shape).
                                        Cheapest check that the labels are learnable.
  ``pi05_<robot>_unimem_video``         PAPER, fixed-stride video baseline with NO event
                                        tracking (mirrors ``xarm_mem7_video``):
                                        num_frames=4 at a 2 s stride.
  ``pi05_<robot>_unimem_keyframe_full``  } DEPARTURE: full fine-tune + EMA instead of
  ``pi05_<robot>_unimem_event_full``     } LoRA. Not what the paper ran (every one of its
                                        34 configs is LoRA); kept for the 2x H100 node,
                                        where the memory that rules LoRA in is available.

Where the paper recipe cannot be followed exactly
-------------------------------------------------
* **Three cameras, not two.** The xArm rig has an exterior + wrist pair; these robots add
  a second wrist. At the same batch size a keyframe sample therefore carries 4 frames x 3
  cameras = 12 images rather than 8, i.e. 1.5x the ViT activations. Batch 44 is kept from
  upstream; halve it if it OOMs.
* **16-dim bimanual actions with delta joints.** ``make_bool_mask(14, -2)`` instead of the
  xArm's ``(6, -1)``, and ``BimanualInputs``/``Outputs`` instead of ``XarmInputs``.
* **The prompt reaches the model differently, but means the same thing.** Upstream sets
  ``prompt_from_task=True``, and its episodes carry one task string each, so every frame
  of an episode sees a constant instruction. Our annotated datasets put the *per-frame
  subtask* in that field, so reading it would hand the policy the very progress
  information event memory is supposed to supply. ``default_prompt`` reproduces
  upstream's actual behaviour — one fixed instruction per frame — rather than its
  mechanism.

Norm stats
----------
Every config for a robot reads one shared directory
(``./assets/<robot>_unimem/<asset_id>``) via ``AssetsConfig.assets_dir``, because the
stats depend only on state/actions and are identical across all five.
"""

import flax.nnx as nnx

from custom_unimem import data_configs
from custom_unimem import event_vocab as _event_vocab
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

# pi0/pi0.5 default action horizon; 50 steps at 30 fps = 1.67 s of open-loop execution.
ACTION_HORIZON = 50

# Event-keyframe depth: the current frame plus 3 past event frames. Upstream's real-robot
# configs (xarm_mem7/9_coruscant) use exactly this.
KEYFRAME_NUM_FRAMES = 4

# Fixed-stride baseline, matching xarm_mem7_video: 4 frames at the
# LeRobotXarmVideoDataConfig default 2 s stride = 6 s of lookback ((T-1) x stride, since
# num_frames counts the current frame).
VIDEO_NUM_FRAMES = 4
VIDEO_FRAME_STRIDE_SEC = 2.0

# Frames at the start of each event window to draw the keyframe from — xarm_mem7_coruscant's
# value (mem9 uses 25). One second at 30 fps, matching the spread in when the event head
# actually fires at rollout.
EVENT_FRAME_WINDOW = 30

# max_token_len is left at the pi0.5 default (200), as upstream does. Measured worst case
# with a full six-event history plus the 16 discretized state values: 115 tokens for
# Astribot, 112 for MOTION2 — 85 tokens of headroom.

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"

# Shared per-robot assets root (see the module docstring).
ASSETS_ROOT = "./assets"


def shared_assets_dir(robot: str) -> str:
    return f"{ASSETS_ROOT}/{robot}_unimem"


def _model(*, lora: bool, event_tracking: bool, video_encoder: bool, num_frames: int) -> pi0_config.Pi0Config:
    """One place that builds every model config, so a freeze filter can never drift from
    the model it is meant to freeze — ``get_freeze_filter`` matches on parameter-path
    regexes, so the two must be built from identical settings."""
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=ACTION_HORIZON,
        # True is the pi0.5 default and what upstream relies on; spelled out because the
        # earlier non-UniMem configs on these robots set it False, and copying that would
        # silently discard the event memory. The history only reaches the model through
        # the discrete-state prompt format: PaligemmaTokenizer.tokenize() renders
        # "Task: <prompt>, <phase_history>, State: <256-bin state>;\nAction: " only on the
        # branch where a state is passed, and ModelTransformFactory passes one only when
        # this is True. Set it False and the run trains happily with no text memory.
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        event_tracking=event_tracking,
        video_encoder=video_encoder,
        num_frames=num_frames,
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
    """Build the five TrainConfigs for one robot — see the module docstring.

    Args:
        robot: short name used in every config name (``astribot`` / ``motion2``).
        repo_id: LeRobot repo id, resolved under ``$HF_LEROBOT_HOME``. Must be the
            ANNOTATED dataset that ``label_dataset_subtasks.py`` has written ``labels``
            and ``phase_history`` into — the video baseline shares it but ignores both.
        asset_id: stable name for the shared norm-stats directory.
        prompt: the single fixed instruction every frame is conditioned on. Must match
            what the deploy node sends, or the model sees an instruction it never saw in
            training. See the module docstring for why this replaces upstream's
            ``prompt_from_task=True`` rather than contradicting it.
        vocab: event vocabulary, used here only to sanity-check the id range.
        warm_start_params: optional checkpoint ``params`` dir to start from instead of
            pi05_base. Only valid if that checkpoint used the same delta-action
            convention AND the same ``discrete_state_input``.
    """
    if vocab.num_event_classes - 1 > _event_vocab.MAX_EVENT_ID:
        raise ValueError(f"{robot}: vocabulary exceeds the model's {_event_vocab.MAX_EVENT_ID} event ids")

    assets = _config.AssetsConfig(assets_dir=shared_assets_dir(robot), asset_id=asset_id)
    loader = weight_loaders.CheckpointWeightLoader(warm_start_params or PI05_BASE_PARAMS)
    base_data = _config.DataConfig(prompt_from_task=False, stop_padding=False)

    def train_config(
        *,
        suffix: str,
        model: pi0_config.Pi0Config,
        data: _config.DataConfigFactory,
        lora: bool,
        batch_size: int,
        fsdp_devices: int = 1,
        num_train_steps: int = 20_000,
        ema_decay: float | None = None,
        save_interval: int = 1_000,
        keep_period: int = 10_000,
    ) -> _config.TrainConfig:
        return _config.TrainConfig(
            name=f"pi05_{robot}_unimem_{suffix}",
            model=model,
            data=data,
            weight_loader=loader,
            # LoRA freezes everything except the adapters and the event head; a full
            # fine-tune freezes nothing.
            freeze_filter=model.get_freeze_filter() if lora else nnx.Nothing,
            # Multiplier on the LR schedule for ``Pi0.phase_head`` only. Every published
            # UniMem experiment leaves this at 1.0.
            phase_head_lr_multiplier=1.0,
            # Upstream's real-robot schedule: the CosineDecaySchedule defaults for
            # warmup (1k steps) and peak (2.5e-5), decaying to 3.5e-6 over 20k steps.
            lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000, decay_lr=3.5e-6),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=ema_decay,
            batch_size=batch_size,
            fsdp_devices=fsdp_devices,
            num_train_steps=num_train_steps,
            log_interval=20,
            save_interval=save_interval,
            keep_period=keep_period,
            # No W&B account is wired up on the training cluster; metrics go to the log
            # instead (the launchers set PYTHONUNBUFFERED so they appear promptly).
            wandb_enabled=False,
        )

    def event_data(**kwargs) -> data_configs.BimanualEventDataConfig:
        return data_configs.BimanualEventDataConfig(
            repo_id=repo_id, assets=assets, base_config=base_data, default_prompt=prompt, **kwargs
        )

    def keyframe_data(**kwargs) -> data_configs.BimanualEventKeyframeDataConfig:
        return data_configs.BimanualEventKeyframeDataConfig(
            repo_id=repo_id, assets=assets, base_config=base_data, default_prompt=prompt, **kwargs
        )

    # Upstream's memory configs run with BOTH dropouts off; the 0.2/0.2 in the
    # unimem_example_* templates is advice the actual sweep did not take. Raise
    # text_dropout_prob toward 0.2 only if a single-modality ablation later fails.
    paper_keyframe_knobs = dict(
        event_dropout_prob=0.0,
        text_dropout_prob=0.0,
        event_frame_window=EVENT_FRAME_WINDOW,
        # Off: this task is a linear sequence with no memory-dependent fork, and upstream
        # only used it for the one xArm task that had one.
        upsample_after_event_id=None,
    )

    kf_lora_model = _model(lora=True, event_tracking=True, video_encoder=True, num_frames=KEYFRAME_NUM_FRAMES)
    kf_full_model = _model(lora=False, event_tracking=True, video_encoder=True, num_frames=KEYFRAME_NUM_FRAMES)
    event_lora_model = _model(lora=True, event_tracking=True, video_encoder=False, num_frames=1)
    event_full_model = _model(lora=False, event_tracking=True, video_encoder=False, num_frames=1)
    video_lora_model = _model(lora=True, event_tracking=False, video_encoder=True, num_frames=VIDEO_NUM_FRAMES)

    return [
        # ---------------------------------------------------------------- paper recipe
        train_config(
            suffix="keyframe",
            model=kf_lora_model,
            data=keyframe_data(**paper_keyframe_knobs),
            lora=True,
            # Upstream's batch. Their rig has two cameras and ours three, so this is 1.5x
            # their per-sample ViT load: 44 x 4 frames x 3 cameras = 528 images per step.
            # Halve it if the node OOMs — that is the first knob to touch, not the LR.
            batch_size=44,
        ),
        train_config(
            suffix="event",
            model=event_lora_model,
            data=event_data(text_dropout_prob=0.0),
            lora=True,
            batch_size=44,
        ),
        train_config(
            suffix="video",
            model=video_lora_model,
            data=data_configs.BimanualVideoDataConfig(
                repo_id=repo_id,
                assets=assets,
                base_config=base_data,
                default_prompt=prompt,
                frame_stride_sec=VIDEO_FRAME_STRIDE_SEC,
                # Not an upstream knob: it counteracts the motion-energy shortcut that
                # stalled these robots at episode start under the earlier short-memory
                # runs. 0.0 disables it, which is what upstream effectively ran.
                mem_dropout_prob=0.5,
            ),
            lora=True,
            batch_size=44,
        ),
        # ------------------------------------------------------------------- departure
        # Full fine-tune + EMA. Needs the EMA fix in training/utils.py: with
        # event_tracking=True the param tree carries the event head's PRNG key, which the
        # old whole-tree EMA update multiplied by ema_decay and crashed on.
        train_config(
            suffix="keyframe_full",
            model=kf_full_model,
            data=keyframe_data(**paper_keyframe_knobs),
            lora=False,
            # 16 over 2 GPUs = 8 samples x 12 images per device. batch_size must stay
            # divisible by the device count (train.py asserts it) and fsdp_devices must
            # divide it evenly (sharding.make_mesh asserts it).
            batch_size=16,
            fsdp_devices=2,
            ema_decay=0.999,
            num_train_steps=30_000,
            save_interval=5_000,
        ),
        train_config(
            suffix="event_full",
            model=event_full_model,
            data=event_data(text_dropout_prob=0.0),
            lora=False,
            batch_size=32,
            fsdp_devices=2,
            ema_decay=0.999,
            num_train_steps=30_000,
            save_interval=5_000,
        ),
    ]


def register(configs: list[_config.TrainConfig]) -> None:
    """Register configs into openpi's global registry (idempotent)."""
    for cfg in configs:
        if cfg.name not in _config._CONFIGS_DICT:  # noqa: SLF001
            _config._CONFIGS.append(cfg)  # noqa: SLF001
            _config._CONFIGS_DICT[cfg.name] = cfg  # noqa: SLF001
