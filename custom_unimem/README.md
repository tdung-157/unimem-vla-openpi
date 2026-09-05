# UniMem on the bimanual robots (`custom_unimem`)

Event-memory (UniMem) training, serving and ROS2 deployment for the two 16-dim bimanual
robots — **Astribot** ("Make an iced coffee") and **MOTION2** ("get coffee, get ice,
serve the drink") — carried over from the earlier `openpi` checkout and rebuilt on this
fork's event-tracking / video-encoder architecture.

Everything lives in this directory as new files. The one exception is a three-line patch
to `src/` restoring `DataConfig.video_tolerance_s` (see [Gotchas](#gotchas)).

---

## What UniMem gives these robots

Both coffee tasks are long (median ~1,100 frames, 37 s for MOTION2) and their steps look
alike from a single frame: the arm at the milk faucet holding a cup looks much the same
before and after the lever has been pressed. A frame-by-frame policy has to guess. UniMem
adds two memories on top of π₀.₅:

* **Event tracking** — an MLP head on the pooled prefix classifies *which semantic event
  just happened*, trained with an auxiliary cross-entropy loss against a per-frame
  `labels` column. At rollout its prediction is appended to a running text summary
  (`phase_history`, e.g. `"History: pushed the coffee lever, moved the cup under the milk
  faucet"`) that goes back into the prompt on the next call.
* **Event keyframes** — SigLIP ingests `num_frames` frames per camera instead of one,
  and those extra frames are the *actual past event frames*, not a fixed time window. At
  serve time the policy server keeps a rolling hidden-state cache so the robot still
  sends only the current frame.

The reason this is cheap here is that **both datasets already annotate every frame with
its subtask** (`task_index`). Where the paper's xArm work had to detect events from
gripper and pose signals, here an event is simply a subtask boundary — see
[`label_dataset_subtasks.py`](label_dataset_subtasks.py) versus
[`../examples/xarm/label_dataset_xarm.py`](../examples/xarm/label_dataset_xarm.py).

### The MOTION2 event vocabulary

Derived from the 6 annotated subtasks and verified over all 648 episodes (640 run the
canonical sequence; 6 are unannotated; 2 skip a step):

| id | completion phrase | fires at the end of subtask |
|----|-------------------|------------------------------|
| 0 | placed the cup under the coffee faucet | *Left arm pick the cup and place it under the coffee faucet* |
| 1 | pushed the coffee lever | *Left arm push the coffee lever* |
| 2 | moved the cup under the milk faucet | *Left arm pick up the coffee faucet and hold it under the milk faucet* |
| 3 | pressed the milk lever | *Right arm press the milk lever* |
| 4 | placed the cup down | *Left arm place the cup down* |
| 5 | served the drink | *Right arm bring the cup to the serving area* |

### The Astribot event vocabulary

`astri_186_v21_relabel`, 186 episodes / 264,874 frames, 6 annotated subtasks (185 episodes
run the full sequence; one stops after subtask 4):

| id | completion phrase | fires at the end of subtask |
|----|-------------------|------------------------------|
| 0 | placed the cup under the coffee maker | *left hand: grasp the cup beneath the blue cup holder stand, place it on the drip tray* |
| 1 | pressed the coffee button | *left hand: press the middle button on the control panel* |
| 2 | moved the cup to the middle of the table | *left hand: grasp the cup from beneath the coffee maker, move it to the middle of the table* |
| 3 | placed the cup under the green dispenser | *right hand: place the cup beneath the emerald green dispenser* |
| 4 | pressed the dispenser lever | *right hand: press the black lever* |
| 5 | placed the cup on the tray | *right hand: pick up the cup and place it on the clear gray tray* |

Two pairs of subtasks share a long opening ("use the left hand to grasp the cup ...",
"... the emerald green beverage dispenser"), so the prefixes in
[`event_vocab.py`](event_vocab.py) run far enough to separate them; longest match wins.

For a *different* dataset whose subtask strings are not in `event_vocab.py` yet, bootstrap
a vocabulary with `--auto-vocab` (step 1 below).

---

## Walkthrough

### 0. Environment

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Dataset paths and prompts live in [`robot_paths.py`](robot_paths.py):

| robot | dataset |
|-------|---------|
| astribot | `/mnt/data/dungnt232_1/data/astri_coffee_making_v21_relabel` (the 2x H100 node) |
| motion2 | `/home/dungnt232/Documents/Work/vla_training/lrb_new_format/coffee_aligned_anno_data_v21` |

To run against a different copy without editing the file — the laptop's
`astri_186_v21_relabel`, say — export `ASTRIBOT_LEROBOT_HOME` / `ASTRIBOT_REPO_ID` (same
pattern for `MOTION2_`), so the committed config always names the machine that trains.

`asset_id` is deliberately *not* derived from `repo_id`: norm stats depend only on
state/actions, so a stable name lets stats computed from one copy be reused against
another. Both norm-stats scripts record what they came from in `norm_stats_source.json`
beside the stats, and preflight refuses a frame-count mismatch — otherwise that same
portability would let stats from the wrong dataset be applied without complaint.

Then check the data before spending GPU hours on it:

```bash
uv run python custom_unimem/preflight.py --robot astribot
```

[`preflight.py`](preflight.py) resolves the dataset, checks the v2.1 layout / 30 fps / the
three cameras / 16-dim state, proves from the data itself that the grippers really are at
dims 14/15 (comparing per-dim ranges, not just names), maps every subtask string against
the vocabulary, reports `labels` / `phase_history` coverage, and verifies the norm stats
were computed from *this* dataset. Everything but the last check needs only numpy +
pyarrow, so it runs before `uv sync` finishes.

### 1. Derive the event columns from your subtask annotations

**This does not ask you to annotate anything.** Your `meta/lerobot_annotations.json` is
the *input*: the converter already turned its per-subtask time ranges into a per-frame
`task_index`, and this step reads that and derives the two columns the UniMem data
pipeline looks up by name — `labels` (int32, the event id inside a ~1 s window around
each subtask *boundary*, `-1` elsewhere) and `phase_history` (string, the running
`"History: …"` summary). Neither can be inferred from `task_index` at load time: the
window and the completion phrasing are policy decisions, and the model's `RepackTransform`
does a hard lookup on both names — without them training dies immediately with
`KeyError: 'labels'`. It is a deterministic two-second transform; run it once per copy of
the dataset. Existing columns are rewritten
byte-identically; the write is atomic (temp file + rename), and it refuses to touch an
already-labeled dataset unless you pass `--overwrite-columns`.

```bash
# dry run first — prints the event sequence per episode and the whole-dataset histogram
uv run python custom_unimem/label_dataset_subtasks.py --robot motion2

uv run python custom_unimem/label_dataset_subtasks.py --robot motion2 --write-parquet
```

**Both datasets are already labeled**: MOTION2 648/648 episodes (103,300 of 817,315
frames inside an event window) and Astribot 186/186 (29,909 of 264,874).

For a new dataset, bootstrap the vocabulary from its own subtask strings first:

```bash
uv run python custom_unimem/label_dataset_subtasks.py --robot astribot --auto-vocab
# paste the printed EventVocab into event_vocab.py, edit the phrases to read as
# COMPLETIONS ("grabbed the cup", not "grab the cup"), collapse duplicate spellings of
# one subtask onto a single id, then re-run with --write-parquet
```

Knobs worth knowing: `--pre/--post` set the event window (default `[-5, +25]` frames,
about one second at 30 fps, centred on the transition and deliberately extending past it
— at rollout the head sees the *aftermath* of an event, not the instant of it).
`--no-final-event` drops the last subtask's event, which is only ever labeled on the
handful of frames before the episode ends (event 5 gets ~6 frames per episode versus
~31 for the others) and whose phrase never becomes visible in any history.

### 2. Norm stats

Once per robot — all five of that robot's configs read the same shared directory.

```bash
# fast path: reads the parquet directly, no video decoding (recommended on 817k frames)
uv run python custom_unimem/compute_norm_stats_fast.py pi05_astribot_unimem_event_full --verify

# or the stock path, sampling a subset
uv run python custom_unimem/compute_norm_stats.py pi05_astribot_unimem_event_full --max-frames 200000
```

`--verify` rebuilds the real data pipeline and cross-checks a sample before writing; on
both datasets it reproduces it exactly (max |diff| = 0). Both scripts copy the result into
`assets/<robot>_unimem/<asset_id>/`, which is where the configs actually look. **Already
in place** for both robots — so skip this step unless the dataset changes.

Norm stats can also be pasted in by hand. Mark them by setting `"provided": true` in
`norm_stats_source.json` beside the file: both scripts then refuse to overwrite them
without `--force` (a hand-pasted file is not reproducible from this repo, and silently
recomputing over it is unrecoverable), and preflight reports them as unverified rather
than implying a check that never ran. The Astribot stats are currently in that state.

### 3. Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python custom_unimem/train.py \
    pi05_astribot_unimem_event_full --exp-name=coffee_event --overwrite
```

On the 2x H100 node — every `_full` config is shaped for exactly that allocation, so no
flags are needed:

```bash
# with Slurm
CONFIG=pi05_astribot_unimem_event_full FAST=1 sbatch custom_unimem/sbatch_norm_stats.sh
CONFIG=pi05_astribot_unimem_event_full sbatch custom_unimem/sbatch_train.sh \
    --exp-name=coffee_event_gate --num-train-steps=5000
CONFIG=pi05_astribot_unimem_keyframe_full sbatch custom_unimem/sbatch_train.sh --exp-name=coffee_keyframe

# without Slurm, straight on the box
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
  nohup uv run python custom_unimem/train.py pi05_astribot_unimem_keyframe_full \
    --exp-name=coffee_keyframe --num-workers 12 --fsdp-devices 2 --no-wandb-enabled \
    > logs/keyframe_full.log 2>&1 &
```

`batch_size` must stay divisible by the GPU count and `fsdp_devices` must divide it
evenly — `train.py` and `sharding.make_mesh` assert both. Per-device load of the `_full`
configs on 2 GPUs: event 16 samples x 3 images, keyframe 8 x 12, video 8 x 18 (the last
matching what the earlier short-memory full fine-tune ran at on 3 GPUs). Halve
`--batch-size` on OOM.

Full fine-tuning also carries optimizer state LoRA does not: FSDP shards parameters,
gradients, the two Adam moments and the EMA copy across both devices, which is what makes
~3B parameters fit. Keep `fsdp_devices` equal to the GPU count — `sbatch_train.sh` derives
it from the allocation and passes it on the command line, overriding the config field.

**There is no separate event-head training run.** The classifier is an auxiliary head on
the same network, optimized in the same step as the flow-matching action loss:
`Pi0.compute_loss_event` returns `action_loss + 0.1 * event_loss`, and `train.py` logs both
so you can watch them independently. One config, one run, both objectives.

The `_full` configs are the intended path — full fine-tuning of the whole backbone, no
freeze filter, `ema_decay=0.999`, as the earlier non-memory bimanual policies on these
robots were trained.

> Where that departs from upstream: **every** UniMem config in this fork is a LoRA
> fine-tune — all 34, the whole `libero_mem*`/`xarm_mem*` sweep plus the four
> `unimem_example_*` templates — so the paper's own LIBERO and xArm checkpoints are LoRA.
> Full FT appears there only as a comment ("drop `_lora` from both variants, >70GB"). On
> 2x H100 that memory is available, so full is the default here; the `_lora` rows exist
> only for a single 32 GB card.

| config | model | data | GPU shape | what it tells you |
|--------|-------|------|-----------|-------------------|
| `pi05_<robot>_unimem_event_full` | event head, single frame, full FT | `BimanualEventDataConfig` | bs 32, 2 GPUs | **Start here**, with `--num-train-steps=5000`. Are the labels learnable? Watch `event_loss` fall. |
| `pi05_<robot>_unimem_keyframe_full` | event head + keyframes (T=4), full FT | `BimanualEventKeyframeDataConfig` | bs 16, 2 GPUs | **the headline config** — text + visual event memory |
| `pi05_<robot>_unimem_video_full` | fixed-stride video (T=6 @ 1 Hz), **no events** | `BimanualVideoDataConfig` | bs 16, 2 GPUs | the "does temporal context help at all" baseline — the direct translation of the older `use_mem_short_video` runs |
| `pi05_<robot>_unimem_event_lora` | event head, single frame, LoRA | `BimanualEventDataConfig` | bs 32, 1 GPU | 32 GB fallback only |
| `pi05_<robot>_unimem_keyframe_lora` | event head + keyframes, LoRA | `BimanualEventKeyframeDataConfig` | bs 8, 1 GPU | 32 GB fallback only |

Shared settings: π₀.₅, `action_dim=32`, `action_horizon=50`, `max_token_len=256`, constant
5e-5 after a 1k-step warmup, AdamW with gradient clipping at 1.0,
`phase_head_lr_multiplier=1.0`, `ema_decay=0.999` on the full fine-tunes (off for LoRA),
W&B off, 300k steps with a checkpoint every 10k. All five start from `pi05_base`. 300k is
an upper bound, not a target — these fine-tunes converge far earlier, so watch the loss
and stop at a checkpoint.
Batch sizes assume the image counts in the table (a keyframe sample carries 4 frames × 3
cameras = 12 images against the single-frame configs' 3); halve on OOM.

Training-only regularization on the memory configs: `text_dropout_prob=0.2` (replace the
history with `"History: none"`), `event_dropout_prob=0.2` on keyframes (zero out all event
frames), and `event_frame_window=30` (draw the keyframe from the first second of an event
rather than always frame 0, matching the spread in when the head fires at rollout). The
two dropouts are drawn independently on purpose — without that, a policy that always gets
both modalities together fails the moment one is missing.

### 4. Serve

```bash
CONFIG=pi05_astribot_unimem_keyframe_full \
MODEL_DIR=checkpoints/pi05_astribot_unimem_keyframe_full/coffee_keyframe/50000 \
    ./custom_unimem/run_serve.sh
```

`create_trained_policy` compares the config against the checkpoint's recorded training
shape (`video_encoder`, `num_frames`, `event_tracking`) and refuses a mismatch, so a wrong
`CONFIG` fails loudly instead of silently serving the wrong history layout.

### 5. Deploy

```bash
# MOTION2
MEMORY_MODE=text_keyframe HOST=127.0.0.1 ./custom_unimem/run_deploy_motion2.sh

# Astribot (sim cameras; use CAMERA_MODE=real for the compressed topics)
CAMERA_MODE=sim MEMORY_MODE=text_keyframe ./custom_unimem/run_deploy_astribot.sh
```

The nodes run under the **system ROS2 Python**, not `uv` — they need `rclpy`/`cv_bridge`
(and `upper_body_msgs` / `astribot_msgs`), and the launcher installs `openpi-client` into
that interpreter if it is missing.

`MEMORY_MODE` must match the checkpoint. These are wire protocols, not preferences:

| trained config | `MEMORY_MODE` | what the client sends |
|----------------|---------------|------------------------|
| `*_keyframe_*` | `text_keyframe` | current frame only + `phase_history`; `reset_cache` per rollout, `new_keyframe` after each detected event |
| `*_keyframe_*` | `keyframe` | same, but the text pinned to `"History: none"` (visual-memory-only ablation) |
| `*_event_*` | `text` | current frame + `phase_history`; no server cache involved |
| `*_video_full` | `video` | the client stacks the last 6 frames at 30-frame spacing and sends the whole stack |
| non-UniMem | `none` | no memory keys at all |

Keyboard in the deploy terminal: **`r`** starts a new rollout (clears the event history and
resets the server's visual cache — skipping this carries stale memory into the next
episode), **`c`** advances the prompt in `tasks_*.yaml`, **`e`** prints the current history
and the event head's top-3 output. The live memory string is also published on
`/vla/event_history`.

An event is only appended when the head's softmax exceeds `EVENT_THRESHOLD` (0.8 by
default) **and** differs from the last appended event. Both guards matter: a false positive
corrupts the history for the rest of the rollout, while a missed detection only waits for
the next inference call. The event head is read once per chunk, so `ACTION_HORIZON` (25 by
default) doubles as the detection period — ~0.83 s at 30 Hz.

---

## Gotchas

* **`discrete_state_input` must stay `True`.** The event history only reaches the model
  through the pi0.5 discrete-state prompt (`"Task: …, History: …, State: …;\nAction: "`);
  the tokenizer's other branch ignores `phase_history` entirely. The earlier non-UniMem
  configs on these robots set it `False`, and copying that would train a memory policy
  with no text memory, silently and without error.
* **`max_token_len=256`**, up from the pi0.5 default of 200. A full MOTION2 history plus
  the 16 discretized state values measures ~185 tokens; overflow is a truncation *warning*,
  not an error, and it would chop the state off the end of the prompt.
* **Grippers trail the arms at dims 14/15**, not interleaved at 7/15 — hence
  `make_bool_mask(14, -2)`. Some older `meta/info.json` files name the interleaved layout;
  that naming is wrong (verified over all 186 Astribot episodes and cross-checked against
  per-frame `task_index`). Getting it wrong drives every arm joint to roughly twice its
  intended angle.
* **Event ids must be 0..10.** `Pi0` allocates 12 logits and reserves the last for the
  "unlabeled" bucket. `EventVocab` enforces this.
* **`src/` patch.** `DataConfig.video_tolerance_s` (plus its one use in
  `data_loader.create_torch_dataset`) was restored from the earlier fork. These
  v2.1-converted datasets have video PTS shifted by up to one 30 fps frame period against
  the parquet timestamps, which trips LeRobot's default 1e-4 s decode tolerance and aborts
  the load; the configs set 0.04.
* **Norm stats are shared per robot** via `AssetsConfig.assets_dir`, since they depend only
  on state/actions. Compute once with either norm-stats script — both copy into the shared
  directory regardless of which config name you ran them under.
* **Warm starts.** The temporal path adds no parameters (temporal attention reuses each
  layer's own weights), so a non-memory checkpoint of the same robot loads fully into a
  video/keyframe config. But it is only valid if that checkpoint used the same delta-action
  convention *and* the same `discrete_state_input`. The older Astribot/MOTION2 checkpoints
  do not, so all configs here start from `pi05_base`.

---

## Files

| file | purpose |
|------|---------|
| `robot_paths.py` | per-machine dataset resolution, repo ids, prompts. Dependency-free so the labeler, preflight and the ROS nodes can import it without openpi. |
| `preflight.py` | validates a dataset (layout, gripper dims, vocabulary coverage, labels, norm stats) before training. |
| `event_vocab.py` | event ids, completion phrases, subtask-prefix matching, `"History: …"` rendering. |
| `label_dataset_subtasks.py` | `task_index` transitions → `labels` + `phase_history` parquet columns (`--auto-vocab`, `--write-parquet`). |
| `bimanual_policy.py` | `BimanualInputs`/`BimanualOutputs` (16-dim, 3 cameras, video-aware) plus `MemoryDropout`. |
| `data_configs.py` | the three `DataConfigFactory`s: event / event+keyframe / fixed-stride video. |
| `robot_configs.py` | the shared `TrainConfig` builder — model shapes, LR schedule, batch sizes, dropout knobs. |
| `astribot_config.py`, `motion2_config.py` | per-robot constants + `register()`. |
| `_bootstrap.py` | registers the configs and sets `HF_LEROBOT_HOME` before delegating to the stock entrypoints. |
| `train.py`, `serve.py`, `compute_norm_stats.py`, `compute_norm_stats_fast.py` | thin wrappers around the repo scripts. |
| `unimem_client.py` | the client half of UniMem: chunked action dispensing, event detection, history text, cache reset/slide. ROS-free. |
| `deploy_common.py` | `tasks.yaml` loading and the raw-terminal key listener. ROS-free. |
| `deploy_motion2_ros2.py`, `deploy_astribot_ros2.py` | the two ROS2 nodes. |
| `run_serve.sh`, `run_deploy_*.sh`, `sbatch_*.sh` | launchers. |
| `tasks_motion2.yaml`, `tasks_astribot.yaml` | prompt lists (normally one entry — see the file). |
