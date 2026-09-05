"""Client-side event memory: the half of UniMem that lives on the robot.

Shared by both ROS2 deploy nodes. Deliberately free of ROS and openpi imports — it needs
only numpy and ``openpi_client`` — so it runs under the system Python that has rclpy.

What it does per control tick:

  1. Hands back the next action from the current chunk, calling the policy server only
     when the chunk runs out (the same trick as ``ActionChunkBroker``, reimplemented here
     because that class slices EVERY array field by step index, which would chop the
     ``event_id`` probability vector into a single scalar).
  2. On each real inference call, reads the event head's output, thresholds it, and — if
     a NEW event fired — appends its phrase to the running ``phase_history`` string that
     goes out with the next observation.
  3. Drives the policy server's SigLIP hidden-state cache for keyframe-trained models:
     ``reset_cache`` once per rollout, ``new_keyframe`` on the call after an event fired.

Memory modes
------------
The mode MUST match how the checkpoint was trained; they are not interchangeable at
serve time (see the fork README's "Running your UniMem policy").

    ``text_keyframe``  keyframe-trained model, both memory modalities live. Send the
                       CURRENT frame only; the server keeps the visual history.
    ``keyframe``       same wire protocol, but the text is pinned to "History: none" —
                       the visual-memory-only ablation.
    ``text``           single-frame event model: text history only, no cache involved.
    ``video``          fixed-stride model: the CLIENT stacks the last ``num_frames``
                       frames at ``frame_stride_frames`` spacing and sends the whole
                       stack every call. No events, no cache.
    ``none``           plain policy, no memory keys at all.

Pinning the text to "History: none" (rather than omitting the key) in ``keyframe`` mode
is deliberate: the model was trained with ``phase_history`` always present, and dropping
the key changes the tokenized prompt itself ("Task: ..., State: ..." instead of
"Task: ..., History: none, State: ..."), which is a prompt format it never saw.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
import dataclasses
import logging

import numpy as np
from openpi_client import websocket_client_policy

# Modes that expect the model to have an event head.
EVENT_MODES = ("text_keyframe", "keyframe", "text")
# Modes served through the policy server's hidden-state cache (single frame per call).
KEYFRAME_MODES = ("text_keyframe", "keyframe")
# Modes that feed the text summary to the model.
TEXT_MODES = ("text_keyframe", "text")
ALL_MODES = ("text_keyframe", "keyframe", "text", "video", "none")

# Above this probability an event-head prediction counts as a detection. High on purpose:
# a false positive permanently corrupts the history for the rest of the rollout, while a
# missed detection only delays it until the next inference call.
DEFAULT_EVENT_CONFIDENCE_THRESHOLD = 0.8


@dataclasses.dataclass
class EventRecord:
    step: int
    event_id: int
    phrase: str
    probability: float


class UniMemPolicyClient:
    """Wraps the websocket policy with event-history and video-history bookkeeping."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        prompt: str,
        event_phrases: dict[int, str],
        memory_mode: str,
        action_horizon: int,
        num_frames: int = 1,
        frame_stride_frames: int = 0,
        event_confidence_threshold: float = DEFAULT_EVENT_CONFIDENCE_THRESHOLD,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if memory_mode not in ALL_MODES:
            raise ValueError(f"memory_mode must be one of {ALL_MODES}, got {memory_mode!r}")
        if memory_mode == "video" and (num_frames < 2 or frame_stride_frames < 1):
            raise ValueError("video mode needs num_frames >= 2 and frame_stride_frames >= 1")
        self._mode = memory_mode
        self._prompt = prompt
        self._phrases = dict(event_phrases)
        # One past the largest valid event id. The event head allocates more logits than
        # any one task uses (the tail holds the "unlabeled" bucket), so anything at or
        # beyond this index means "no event here".
        self._num_event_classes = max(self._phrases) + 1 if self._phrases else 0
        self._action_horizon = int(action_horizon)
        self._num_frames = int(num_frames)
        self._frame_stride = int(frame_stride_frames)
        self._threshold = float(event_confidence_threshold)
        self._log = log or logging.getLogger("unimem_client").info

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

        self._chunk: np.ndarray | None = None
        self._cursor = 0
        self._step = 0

        self._completed: list[str] = []
        self._events: list[EventRecord] = []
        self._last_appended_event: int | None = None
        self._pending_reset = True
        self._pending_slide = False
        self._last_probs: np.ndarray | None = None

        # Per-camera raw frame ring buffers, used only by the fixed-stride video mode.
        history_len = 1 + (self._num_frames - 1) * self._frame_stride if self._mode == "video" else 1
        self._frame_history: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_len)
        )

    # ---------------------------------------------------------------- properties
    @property
    def server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    @property
    def event_tracking(self) -> bool:
        return self._mode in EVENT_MODES

    @property
    def history_text(self) -> str:
        return f"History: {', '.join(self._completed)}" if self._completed else "History: none"

    @property
    def events(self) -> list[EventRecord]:
        return list(self._events)

    @property
    def last_event_probabilities(self) -> np.ndarray | None:
        return self._last_probs

    # ------------------------------------------------------------------- control
    def reset(self) -> None:
        """Start a new rollout: drop the chunk, the event history and the server cache.

        Skipping this between episodes carries stale visual memory across rollouts — the
        policy would start episode 2 believing it had already poured the coffee.
        """
        self._chunk = None
        self._cursor = 0
        self._step = 0
        self._completed.clear()
        self._events.clear()
        self._last_appended_event = None
        self._pending_reset = True
        self._pending_slide = False
        self._frame_history.clear()
        self._log("UniMem client reset: event history cleared, server cache reset pending")

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def step(self, *, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        """Return the action for this control tick.

        Args:
            state: the robot's current 16-dim state vector.
            images: ``{"observation/image": HWC uint8, "observation/left_wrist_image": ...,
                "observation/right_wrist_image": ...}`` — always the CURRENT frame only.
                Video mode stacks the history itself from these.
        """
        if self._mode == "video":
            for key, frame in images.items():
                self._frame_history[key].append(frame)

        if self._chunk is None or self._cursor >= len(self._chunk):
            self._infer(state, images)
            self._cursor = 0

        assert self._chunk is not None
        action = self._chunk[self._cursor]
        self._cursor += 1
        self._step += 1
        return action

    # ------------------------------------------------------------------ internals
    def _observation(self, state: np.ndarray, images: dict[str, np.ndarray]) -> dict:
        obs: dict = {"observation/state": np.asarray(state, dtype=np.float32), "prompt": self._prompt}

        if self._mode == "video":
            for key in images:
                obs[key] = self._stacked(key)
        else:
            obs.update(images)

        if self._mode in TEXT_MODES:
            obs["phase_history"] = self.history_text
        elif self._mode == "keyframe":
            # Trained with the key always present — see the module docstring.
            obs["phase_history"] = "History: none"

        if self._mode in KEYFRAME_MODES:
            if self._pending_reset:
                obs["reset_cache"] = True
                self._pending_reset = False
            if self._pending_slide:
                # The server slides its hidden-state cache (evict oldest, append the frame
                # sent with THIS call) only when asked. That is why the slide is deferred
                # to the call after the detection: the frame that gets cached is the one
                # showing the completed event, not the one mid-transition.
                obs["new_keyframe"] = True
                self._pending_slide = False
        return obs

    def _stacked(self, key: str) -> np.ndarray:
        """(T, H, W, C) stack at the trained spacing, clamped at the start of the rollout.

        Clamping repeats the oldest available frame, which is what LeRobot's negative
        delta indices do at an episode boundary — so warm-up looks to the model like the
        start of a training episode rather than something it has never seen.
        """
        frames = list(self._frame_history[key])
        if not frames:
            raise RuntimeError(f"no frames buffered for {key}")
        picks = [len(frames) - 1 - (self._num_frames - 1 - i) * self._frame_stride for i in range(self._num_frames)]
        return np.stack([frames[max(0, p)] for p in picks], axis=0)

    def _infer(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        result = self._client.infer(self._observation(state, images))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise RuntimeError(f"expected an (horizon, dim) action chunk, got shape {actions.shape}")
        self._chunk = actions[: self._action_horizon]

        if self.event_tracking and "event_id" in result:
            self._handle_event(result["event_id"])

    def _handle_event(self, event_id) -> None:
        probs = np.asarray(event_id, dtype=np.float64).reshape(-1)[: self._num_event_classes]
        self._last_probs = probs
        if probs.size == 0:
            return
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        if confidence <= self._threshold or predicted not in self._phrases:
            return
        # Never append the same event twice in a row. Returning to a cyclical subtask's
        # start requires reversing the forward action, which is itself a distinct event,
        # so a repeat is always a redundant re-detection of the event already recorded.
        if predicted == self._last_appended_event:
            return

        phrase = self._phrases[predicted]
        self._completed.append(phrase)
        self._last_appended_event = predicted
        self._events.append(EventRecord(step=self._step, event_id=predicted, phrase=phrase, probability=confidence))
        self._log(f"event {predicted} '{phrase}' (p={confidence:.3f}) -> {self.history_text}")
        if self._mode in KEYFRAME_MODES:
            self._pending_slide = True
