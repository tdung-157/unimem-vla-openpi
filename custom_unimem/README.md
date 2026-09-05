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
the vocabulary, reports `labels` / `phase_history` coverage, verifies the norm stats were
computed from *this* dataset, and confirms the `gs://` assets training needs are already
in the local openpi cache. Nothing in it touches the network, and everything but the last
two checks needs only numpy + pyarrow, so it runs before `uv sync` finishes.

That last check earns its place: on a compute node with no internet, a missing cached
asset does not fail — `maybe_download` **hangs**, silently. Set `OPENPI_DATA_HOME` in the
shell you train from (`sbatch_*.sh` and `run_serve.sh` do it for you; a bare `uv run` does
not).

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
uv run python custom_unimem/compute_norm_stats_fast.py pi05_astribot_unimem_event --verify

# or the stock path, sampling a subset
uv run python custom_unimem/compute_norm_stats.py pi05_astribot_unimem_event --max-frames 200000
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
    pi05_astribot_unimem_event --exp-name=coffee_event --overwrite
```

On the 2x H100 node — every `_full` config is shaped for exactly that allocation, so no
flags are needed:

```bash
# with Slurm
CONFIG=pi05_astribot_unimem_event FAST=1 sbatch custom_unimem/sbatch_norm_stats.sh
CONFIG=pi05_astribot_unimem_event sbatch custom_unimem/sbatch_train.sh \
    --exp-name=coffee_event_gate --num-train-steps=5000
CONFIG=pi05_astribot_unimem_keyframe sbatch custom_unimem/sbatch_train.sh --exp-name=coffee_keyframe

# without Slurm, straight on a GPU node
export OPENPI_DATA_HOME=/mnt/data/dungnt232_1/openpi_cache   # see below
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
  nohup uv run python custom_unimem/train.py pi05_astribot_unimem_keyframe \
    --exp-name=coffee_keyframe --num-workers 12 --fsdp-devices 2 --no-wandb-enabled \
    > logs/keyframe_full.log 2>&1 &
```

**`OPENPI_DATA_HOME`.** Every `gs://` asset openpi touches — the `pi05_base` weights the
configs warm-start from, the PaliGemma tokenizer — resolves to
`$OPENPI_DATA_HOME/<bucket>/<path>` (default `~/.cache/openpi`) and is downloaded on first
use. The cluster's compute nodes have no internet, so a cold cache does not fail, it hangs.
The cache at `/mnt/data/dungnt232_1/openpi_cache` is already populated
(`openpi-assets/checkpoints/pi05_base/params`, `big_vision/paligemma_tokenizer.model`);
`sbatch_*.sh` and `run_serve.sh` export it by default, the direct commands above must set
it themselves. On the laptop leave it unset.

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

The first three configs reproduce the paper's own real-robot recipe — every hyperparameter
is `xarm_mem7_coruscant`'s, the config behind its hardware checkpoints. The last two are a
deliberate departure (full fine-tune + EMA) for the 2x H100 node.

| config | recipe | model | GPU shape | what it is for |
|--------|--------|-------|-----------|----------------|
| `pi05_<robot>_unimem_keyframe` | **paper** | LoRA, events + keyframes (T=4) | bs 44, 1 GPU, 20k | **the headline config** — mirrors `xarm_mem7_coruscant` |
| `pi05_<robot>_unimem_event` | **paper** | LoRA, events, single frame | bs 44, 1 GPU, 20k | **start here** — cheapest check that the labels are learnable |
| `pi05_<robot>_unimem_video` | **paper** | LoRA, video (T=4 @ 2 s), no events | bs 44, 1 GPU, 20k | "does temporal context help at all" baseline — mirrors `xarm_mem7_video` |
| `pi05_<robot>_unimem_keyframe_full` | departure | full FT + EMA, events + keyframes | bs 16, 2 GPUs, 30k | full-capacity version of the headline config |
| `pi05_<robot>_unimem_event_full` | departure | full FT + EMA, events, single frame | bs 32, 2 GPUs, 30k | full-capacity single-frame |

Every field of the three paper configs was diffed against `xarm_mem7_coruscant` and
matches: `gemma_2b_lora` + `gemma_300m_lora`, `action_dim=32`, `action_horizon=50`,
`max_token_len` left at the pi0.5 default of 200, `num_frames=4`, `ema_decay=None`,
`batch_size=44`, `num_train_steps=20_000`, AdamW clipped at 1.0, cosine 2.5e-5 → 3.5e-6
over 20k steps after a 1k warmup, `event_dropout_prob=0.0`, `text_dropout_prob=0.0`,
`event_frame_window=30`, `stop_padding=False`, `save_interval=1_000`, `keep_period=10_000`,
`log_interval=20`, freeze filter from the identical model config.

Three things the paper recipe cannot express on this robot, and why:

* **Three cameras, not two.** A keyframe sample carries 4 frames x 3 cameras = 12 images
  rather than the xArm's 8 — 1.5x the ViT load at the same batch. Batch 44 is kept from
  upstream anyway; halve it first if the node OOMs.
* **16-dim bimanual actions.** `make_bool_mask(14, -2)` and `BimanualInputs`/`Outputs`
  instead of the xArm's `(6, -1)` and `XarmInputs`.
* **The prompt.** Upstream sets `prompt_from_task=True` and its episodes carry one task
  string each, so every frame of an episode sees a constant instruction. Our annotated
  datasets put the *per-frame subtask* in that field, so reading it would hand the policy
  the very progress information event memory exists to supply. `default_prompt` reproduces
  upstream's actual behaviour rather than its mechanism.

Two knobs where upstream's *templates* and its *trained configs* disagree, and I followed
the trained ones: `event_dropout_prob` and `text_dropout_prob` are 0.2 in the
`unimem_example_*` templates but **0.0** in every `xarm_mem*_coruscant`. Raise
`text_dropout_prob` toward 0.2 only if a single-modality ablation later fails.

### 4. Serve

```bash
CONFIG=pi05_astribot_unimem_keyframe \
MODEL_DIR=checkpoints/pi05_astribot_unimem_keyframe/coffee_keyframe/20000 \
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
| `*_video` | `video` | the client stacks the last 4 frames at 60-frame (2 s) spacing and sends the whole stack |
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
* **EMA + event tracking crashed upstream.** `nnx.state(model)` is not all floats: the
  event head's `nnx.Dropout` puts a `key<fry>` PRNG key and a `uint32` counter in the same
  tree, and `train.py`'s EMA update multiplied *every* leaf by `ema_decay` — so any config
  with `event_tracking=True` **and** `ema_decay` set died on step 1 with
  `TypeError: multiply does not accept dtypes float32, key<fry>`. No upstream config hits
  this (all 34 are LoRA with `ema_decay=None`), but every `_full` config here does. Fixed
  by `training_utils.ema_update`, which averages float leaves and carries the rest
  through; applied at all three EMA sites (`train.py`, `train_accum_steps.py` x2).
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
