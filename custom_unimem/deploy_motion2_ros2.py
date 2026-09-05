#!/usr/bin/env python3
"""ROS2 deployment node for a UniMem policy on the MOTION2 bimanual robot.

    ROS2 topics (3 cameras + joint_states)  ->  observation
        ->  websocket policy server (pi0.5 + event head)  ->  16-dim action chunk
        ->  ROS2 command topics (joint_commands + gripper_commands)

The observation layout is kept bit-for-bit identical to the training data so there is no
train/inference distribution shift:

    observation.state = [14 arm joints (by name), finger_left_1_joint, finger_right_1_joint]
    action            = [14 arm joint targets,    left_gripper,        right_gripper       ]

On top of the plain deployment this node runs the CLIENT half of UniMem — the event
history and the policy server's visual-memory cache — through
``custom_unimem/unimem_client.py``. Set ``memory_mode`` to whatever the checkpoint was
trained as; the modes are not interchangeable at serve time:

    pi05_motion2_unimem_keyframe[_full]  ->  memory_mode:=text_keyframe   (default)
    pi05_motion2_unimem_event[_full]     ->  memory_mode:=text
    pi05_motion2_unimem_video            ->  memory_mode:=video
    a non-UniMem checkpoint              ->  memory_mode:=none

Prerequisites
-------------
Run this INSIDE a sourced ROS2 environment (not under ``uv``) with ``upper_body_msgs``
built, and install the pure-python client into that interpreter::

    pip install -e packages/openpi-client
    pip install opencv-python

Start the policy server first (see custom_unimem/run_serve.sh), then::

    python3 custom_unimem/deploy_motion2_ros2.py --ros-args \
        -p host:=127.0.0.1 -p port:=8000 -p memory_mode:=text_keyframe

Keyboard (when stdin is a TTY): 'r' starts a new rollout (clears the event history and
resets the server's visual cache), 'c' advances to the next prompt in tasks.yaml,
'e' prints the current event history.

Topics
------
    IN   /m2/head_camera         sensor_msgs/Image
         /m2/left_wrist_camera   sensor_msgs/Image
         /m2/right_wrist_camera  sensor_msgs/Image
         /m2/joint_states        sensor_msgs/JointState
         /vla/prompt             std_msgs/String                 (optional, live prompt)
    OUT  /m2/joint_commands      sensor_msgs/JointState          {name=ARM_JOINT_NAMES, position}
         /m2/gripper_commands    upper_body_msgs/GrippersControl {side=[left,right], cmd=[bool,bool]}
         /vla/event_history      std_msgs/String                 (the live memory string)
"""

import pathlib
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from cv_bridge import CvBridge
    import message_filters
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    # Custom robot message: upper_body_msgs/msg/GrippersControl
    #   std_msgs/Header header
    #   string[] side   (e.g. ["left", "right"])
    #   bool[]   cmd
    from upper_body_msgs.msg import GrippersControl
except ImportError as e:  # pragma: no cover - clearer error when run outside ROS2
    raise SystemExit(
        "This node requires a sourced ROS2 environment (rclpy, cv_bridge, sensor_msgs, "
        "std_msgs, message_filters) with 'upper_body_msgs' built and sourced.\n"
        f"Original import error: {e}"
    )

import cv2

from custom_unimem import deploy_common
from custom_unimem import event_vocab
from custom_unimem import robot_paths
from custom_unimem import unimem_client

# ---------------------------------------------------------------------------
# State / action layout — copied verbatim from the dataset converter so the vector built
# here matches the one the model was trained on. The two grippers TRAIL the arms.
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
# Gripper *state* is the primary finger joint of each hand (continuous position).
STATE_GRIPPER_JOINT_NAMES = ["finger_left_1_joint", "finger_right_1_joint"]
STATE_DIM = len(ARM_JOINT_NAMES) + len(STATE_GRIPPER_JOINT_NAMES)  # 16
# Model gripper output is continuous; binarize like the recorded gripper commands.
GRIPPER_THRESHOLD = 0.5

PATHS = robot_paths.MOTION2

# Fixed-stride video baseline only: must match VIDEO_NUM_FRAMES / VIDEO_FRAME_STRIDE_SEC
# in robot_configs.py — 4 frames at a 2 s stride, which on a 30 fps stream is every 60th
# frame (upstream's xarm_mem7_video layout). A mismatch silently feeds the model a clip
# with different temporal spacing than it trained on.
DEFAULT_VIDEO_NUM_FRAMES = 4
DEFAULT_VIDEO_STRIDE_FRAMES = 60
# Event-keyframe models: informational only (the server owns the cache depth), but logged
# so a config/checkpoint mismatch is visible in the startup banner.
DEFAULT_KEYFRAME_NUM_FRAMES = 4


class Motion2UniMemDeployNode(Node):
    def __init__(self) -> None:
        super().__init__("motion2_unimem_deploy")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8000)
        self.declare_parameter("prompt", PATHS.prompt)
        self.declare_parameter("tasks_file", str(pathlib.Path(__file__).resolve().parent / "tasks_motion2.yaml"))
        # Which memory protocol to speak. MUST match the trained checkpoint.
        self.declare_parameter("memory_mode", "text_keyframe")
        # Steps of each predicted chunk to execute before re-querying. This also sets how
        # often the event head is read, so it doubles as the event-detection period:
        # 25 steps at 30 Hz means a new event is noticed within ~0.83 s.
        self.declare_parameter("action_horizon", 25)
        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("event_confidence_threshold", unimem_client.DEFAULT_EVENT_CONFIDENCE_THRESHOLD)
        self.declare_parameter("video_num_frames", DEFAULT_VIDEO_NUM_FRAMES)
        self.declare_parameter("video_stride_frames", DEFAULT_VIDEO_STRIDE_FRAMES)
        self.declare_parameter("use_compressed", False)
        self.declare_parameter("sync_slop", 0.05)

        self.declare_parameter("head_image_topic", "/m2/head_camera")
        self.declare_parameter("left_wrist_image_topic", "/m2/left_wrist_camera")
        self.declare_parameter("right_wrist_image_topic", "/m2/right_wrist_camera")
        self.declare_parameter("joint_states_topic", "/m2/joint_states")
        self.declare_parameter("prompt_topic", "/vla/prompt")

        self.declare_parameter("joint_command_topic", "/m2/joint_commands")
        self.declare_parameter("gripper_command_topic", "/m2/gripper_commands")
        self.declare_parameter("event_history_topic", "/vla/event_history")
        self.declare_parameter("gripper_sides", ["left", "right"])

        p = self.get_parameter
        host = str(p("host").value)
        port = int(p("port").value)
        mode = str(p("memory_mode").value)
        control_hz = float(p("control_hz").value)
        self._use_compressed = bool(p("use_compressed").value)

        self._prompt_lock = threading.Lock()
        self._tasks = deploy_common.load_tasks(str(p("tasks_file").value), self.get_logger().warning)
        self._task_index = 0
        self._prompt = self._tasks[0] if self._tasks else str(p("prompt").value)
        self._check_prompt(self._prompt)

        vocab = event_vocab.get_vocab(PATHS.vocab)
        self.get_logger().info(f"Connecting to policy server ws://{host}:{port} (memory_mode={mode}) ...")
        self._policy = unimem_client.UniMemPolicyClient(
            host=host,
            port=port,
            prompt=self._prompt,
            event_phrases=vocab.phrases,
            memory_mode=mode,
            action_horizon=int(p("action_horizon").value),
            num_frames=int(p("video_num_frames").value),
            frame_stride_frames=int(p("video_stride_frames").value),
            event_confidence_threshold=float(p("event_confidence_threshold").value),
            log=self.get_logger().info,
        )
        self.get_logger().info(f"Server metadata: {self._policy.server_metadata}")

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, dict[str, np.ndarray]] | None = None

        img_type = CompressedImage if self._use_compressed else Image
        subs = [
            message_filters.Subscriber(
                self, img_type, p("head_image_topic").value, qos_profile=qos_profile_sensor_data
            ),
            message_filters.Subscriber(
                self, img_type, p("left_wrist_image_topic").value, qos_profile=qos_profile_sensor_data
            ),
            message_filters.Subscriber(
                self, img_type, p("right_wrist_image_topic").value, qos_profile=qos_profile_sensor_data
            ),
            message_filters.Subscriber(
                self, JointState, p("joint_states_topic").value, qos_profile=qos_profile_sensor_data
            ),
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=float(p("sync_slop").value))
        self._sync.registerCallback(self._on_synced)

        self.create_subscription(String, p("prompt_topic").value, self._on_prompt, 10)

        self._joint_pub = self.create_publisher(JointState, p("joint_command_topic").value, 10)
        self._gripper_pub = self.create_publisher(GrippersControl, p("gripper_command_topic").value, 10)
        self._history_pub = self.create_publisher(String, p("event_history_topic").value, 10)
        self._gripper_sides = [str(s) for s in p("gripper_sides").value]

        self.create_timer(1.0 / control_hz, self._control_tick)

        self._keys = deploy_common.KeyListener(
            {"r": self._reset_rollout, "c": self._advance_task, "e": self._print_events},
            self.get_logger().warning,
        )
        started = self._keys.start()
        if self._tasks and started:
            lines = "\n".join(f"    {i + 1:>2}. {t}" for i, t in enumerate(self._tasks))
            self.get_logger().info(f"Loaded {len(self._tasks)} tasks:\n{lines}")
        if started:
            self.get_logger().info("Keys: 'r' new rollout, 'c' next task, 'e' show event history.")

        # Frames the model sees per camera, for the banner only: the server owns the
        # keyframe cache depth, the client owns the fixed-stride stack, and the
        # single-frame modes use exactly one.
        if mode in unimem_client.KEYFRAME_MODES:
            depth = DEFAULT_KEYFRAME_NUM_FRAMES
        elif mode == "video":
            depth = int(p("video_num_frames").value)
        else:
            depth = 1
        self.get_logger().info(
            f"Ready. control_hz={control_hz}, action_horizon={p('action_horizon').value}, "
            f"memory_mode={mode}, frames={depth}, prompt='{self._prompt}'. "
            "Waiting for synchronized observations..."
        )

    def stop(self) -> None:
        self._keys.stop()

    def _check_prompt(self, prompt: str) -> None:
        """Warn when the served prompt is not the one the checkpoint was trained on.

        Every frame of training saw exactly one instruction (``robot_paths``' ``prompt``,
        injected by ``default_prompt`` — the same single-string-per-episode behaviour
        upstream gets from ``prompt_from_task=True`` on datasets whose task field is a
        per-session constant). Sending anything else puts the model somewhere it has never
        been, and it fails quietly rather than loudly, so say so.
        """
        if prompt != PATHS.prompt:
            self.get_logger().warning(
                f"prompt {prompt!r} differs from the trained prompt {PATHS.prompt!r}. "
                "The policy was conditioned on one fixed instruction for every frame; "
                "progress is supposed to reach it through phase_history, not the prompt."
            )

    # ------------------------------------------------------------------ keyboard
    def _reset_rollout(self) -> None:
        self._policy.reset()
        self.get_logger().info("--- new rollout ---")

    def _advance_task(self) -> None:
        with self._prompt_lock:
            if not self._tasks:
                return
            self._task_index = (self._task_index + 1) % len(self._tasks)
            self._prompt = self._tasks[self._task_index]
            index, total, prompt = self._task_index + 1, len(self._tasks), self._prompt
        self._policy.set_prompt(prompt)
        self._check_prompt(prompt)
        self.get_logger().info(f"[task {index}/{total}] prompt -> '{prompt}'")

    def _print_events(self) -> None:
        self.get_logger().info(f"event history: {self._policy.history_text}")
        probs = self._policy.last_event_probabilities
        if probs is not None:
            ranked = np.argsort(probs)[::-1][:3]
            self.get_logger().info("  last event-head output: " + ", ".join(f"{int(i)}:{probs[i]:.3f}" for i in ranked))

    # ------------------------------------------------------------------ callbacks
    def _on_prompt(self, msg: String) -> None:
        new_prompt = msg.data.strip()
        with self._prompt_lock:
            if new_prompt and new_prompt != self._prompt:
                self._prompt = new_prompt
                self._policy.set_prompt(new_prompt)
                self._check_prompt(new_prompt)
                self.get_logger().info(f"Prompt updated via topic: '{new_prompt}'")

    def _decode_image(self, msg) -> np.ndarray:
        """480x640x3 uint8 RGB, matching the training frames."""
        if self._use_compressed:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _build_state(self, js: JointState) -> np.ndarray | None:
        pos_by_name = dict(zip(js.name, js.position, strict=False))
        try:
            values = [pos_by_name[n] for n in ARM_JOINT_NAMES] + [pos_by_name[n] for n in STATE_GRIPPER_JOINT_NAMES]
        except KeyError as missing:
            self.get_logger().warning(
                f"joint_states is missing expected joint {missing}; available: {list(js.name)}",
                once=True,
            )
            return None
        return np.asarray(values, dtype=np.float32)

    def _on_synced(self, head_msg, left_msg, right_msg, joint_msg) -> None:
        state = self._build_state(joint_msg)
        if state is None:
            return
        images = {
            "observation/image": self._decode_image(head_msg),
            "observation/left_wrist_image": self._decode_image(left_msg),
            "observation/right_wrist_image": self._decode_image(right_msg),
        }
        with self._lock:
            self._latest = (state, images)

    # ------------------------------------------------------------------ control loop
    def _control_tick(self) -> None:
        with self._lock:
            latest = self._latest
        if latest is None:
            self.get_logger().warning("Waiting for synchronized camera + joint_state ...", once=True)
            return
        state, images = latest

        try:
            action = self._policy.step(state=state, images=images)
        except Exception as e:  # keep the node alive across transient server hiccups
            self.get_logger().error(f"Inference failed: {e}")
            return

        action = np.asarray(action).reshape(-1)
        if action.shape[0] < STATE_DIM:
            self.get_logger().error(f"Unexpected action dim {action.shape[0]} (< {STATE_DIM})")
            return
        if not np.all(np.isfinite(action[:STATE_DIM])):
            self.get_logger().warning("Non-finite action; skipping this tick.", once=True)
            return

        self._publish(action[: len(ARM_JOINT_NAMES)].astype(float), action[len(ARM_JOINT_NAMES) : STATE_DIM])

    def _publish(self, arm_targets: np.ndarray, grippers: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()

        joint_cmd = JointState()
        joint_cmd.header.stamp = stamp
        joint_cmd.name = list(ARM_JOINT_NAMES)
        joint_cmd.position = [float(v) for v in arm_targets]
        self._joint_pub.publish(joint_cmd)

        gripper_cmd = GrippersControl()
        gripper_cmd.header.stamp = stamp
        gripper_cmd.side = list(self._gripper_sides)
        gripper_cmd.cmd = [bool(v > GRIPPER_THRESHOLD) for v in grippers]
        self._gripper_pub.publish(gripper_cmd)

        history = String()
        history.data = self._policy.history_text
        self._history_pub.publish(history)


def main() -> None:
    rclpy.init()
    node = Motion2UniMemDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
